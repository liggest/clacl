
from typing import Callable, Iterable
from itertools import chain
from enum import Enum
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_outputs import BaseModelOutput
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

from transformers.models.wavlm.modeling_wavlm import (
    # WavLMConfig,
    WavLMEncoderLayer, 
    WavLMEncoder,
    WavLMPreTrainedModel,   
    WavLMModel, 
    WavLMForSequenceClassification
)

from clacl.model.wavlm import CustomWavLMConfig
from clacl.model.wavlm import AdaPreTrainedModelOverride, AdapterLayer, AdapterToOutputLayer
from clacl.model.cl import AdapterState
from clacl.model.cl import CLManager, CLAdapter, CLParameter, CLModule
from clacl.model.cl import not_special_task

from clacl.util import logger

class AdaptivePoolState(str, Enum):
    Missing = "missing"
    Avg = "avg"
    Max = "max"

class CLInitSpecials(str, Enum):
    Self = "Self"
    First = "First"
    Last = "Last"
    All = "All"

class LComponentState(str, Enum):
    All = "all"
    """ Adapters + LayerWeights """
    Adapters = "adapters"
    LayerWeights = "layer_weights"

class CLWavLMConfig(CustomWavLMConfig):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.e_adapter_state = AdapterState(kw.pop("e_adapter_state", AdapterState.CL))
        self.l_adapter_state = AdapterState(kw.pop("l_adapter_state", AdapterState.CL))
        self.head_state = AdapterState(kw.pop("head_state", AdapterState.CL))

        # self.layer_norm_state = AdapterState(kw.pop("layer_norm_state", AdapterState.CL))

        self.cl_e_adapter_init: list[str | CLInitSpecials] = kw.pop("cl_e_adapter_init", [CLInitSpecials.Self])
        self.cl_l_adapter_init: list[str | CLInitSpecials] = kw.pop("cl_l_adapter_init", [CLInitSpecials.Self])
        self.cl_layer_weights_init: list[str | CLInitSpecials] = kw.pop("cl_layer_weights_init", [CLInitSpecials.Self])
        self.cl_head_init: list[str | CLInitSpecials] = kw.pop("cl_head_init", [CLInitSpecials.Self])

        # self.use_e_adapter = (self.e_adapter_state != AdapterState.Missing)
        # self.use_l_adapter = (self.l_adapter_state != AdapterState.Missing) or self.use_l_adapter

        # if self.l_adapter_state == AdapterState.Missing:
        #     self.output_hidden_size = self.hidden_size  # default 768

        # self.use_cl_head = (self.head_state != AdapterState.Missing)

        # self.use_cl_init_only = False

        self.l_adapter_component: LComponentState = LComponentState(kw.pop("l_adapter_component", LComponentState.All))

        # self.layer_weights_only: bool = kw.pop("layer_weights_only", False)

        self.head_adaptive_pool: AdaptivePoolState = AdaptivePoolState(kw.pop("head_adaptive_pool", AdaptivePoolState.Missing))
        self.head_adaptive_pool_size: int = kw.pop("head_adaptive_pool_size", 0)
        if self.use_adaptive_pool:
            assert self.head_adaptive_pool_size > 0

        self.head_expanding: bool = kw.pop("head_expanding", False)
        if self.head_expanding:
            assert self.head_state == AdapterState.CL

    @property
    def l_adapter_state(self):
        return self.l_adapter_state_
    
    @l_adapter_state.setter
    def l_adapter_state(self, val):
        self.l_adapter_state_ = AdapterState(val)
        if self.l_adapter_state_ == AdapterState.Missing:
            self.output_hidden_size = self.hidden_size
            self.use_l_adapter = False

    @property
    def l_adapter_component(self):
        return self.l_adapter_component_

    @l_adapter_component.setter
    def l_adapter_component(self, val):
        self.l_adapter_component_ = LComponentState(val)
        if self.l_adapter_component_ == LComponentState.LayerWeights:
            self.output_hidden_size = self.hidden_size

    # @property
    # def layer_weights_only(self):
    #     return self.layer_weights_only_
    
    # @layer_weights_only.setter
    # def layer_weights_only(self, val):
    #     self.layer_weights_only_ = bool(val)
    #     if self.layer_weights_only_:
    #         self.output_hidden_size = self.hidden_size

    @property
    def use_adaptive_pool(self):
        return self.head_adaptive_pool != AdaptivePoolState.Missing

AdaPreTrainedModelOverride.config_class = CLWavLMConfig

cl_modules = CLManager("wavlm_cl")

def _average_weights(new_module: nn.Module, CL_module: CLAdapter, task_name: str, init_list: list[str | CLInitSpecials]):
    logger.debug(f"Try to init weights of {CL_module._group}.{CL_module._name}.{task_name} by averaging {init_list!r}")
    previous_task_names = [*CL_module.adapters]
    states_to_avg = []
    for init_name in init_list:
        if init_name == CLInitSpecials.Self:
            states_to_avg.append(new_module.state_dict())
        elif init_name == CLInitSpecials.First:
            if previous_task_names:
                states_to_avg.append(CL_module.adapters[previous_task_names[0]].state_dict())
        elif init_name == CLInitSpecials.Last:
            if previous_task_names:
                states_to_avg.append(CL_module.adapters[previous_task_names[-1]].state_dict())
        elif init_name == CLInitSpecials.All:
            states_to_avg.extend(m.state_dict() for m in CL_module.adapters.values())
        elif init_name in CL_module.adapters:
            states_to_avg.append(CL_module.adapters[init_name].state_dict())
        else:
            # logger.debug(f"{init_name!r} not found during init of {CL_module._group}.{CL_module._name}.{task_name}, skipped")
            pass

    n = len(states_to_avg)
    if n == 0:
        # logger.debug(f"{CL_module._group}.{CL_module._name}.{task_name} init without averaging")
        return new_module
    
    new_state = new_module.state_dict()
    if n == 1:
        # logger.debug(f"{CL_module._group}.{CL_module._name}.{task_name} init by overriding")
        override_state = states_to_avg[0]
        for key in new_state:
            new_state[key] = override_state[key].detach().cpu().clone()
    else:
        # logger.debug(f"{CL_module._group}.{CL_module._name}.{task_name} init by averaging {n} modules")
        for key in new_state:
            new_state[key] = sum(state[key].detach().cpu() for state in states_to_avg) / n
    new_module.load_state_dict(new_state)
    return new_module

def _average_parameters(new_module: nn.Parameter, CL_module: CLParameter, task_name: str, init_list: list[str | CLInitSpecials]):
    logger.debug(f"Try to init weights of {CL_module._group}.{CL_module._name}.{task_name} by averaging {init_list!r}")
    previous_task_names = [*CL_module.adapters]
    tensors_to_avg = []
    for init_name in init_list:
        if init_name == CLInitSpecials.Self:
            tensors_to_avg.append(new_module.detach().cpu())
        elif init_name == CLInitSpecials.First:
            if previous_task_names:
                tensors_to_avg.append(CL_module.adapters[previous_task_names[0]].detach().cpu())
        elif init_name == CLInitSpecials.Last:
            if previous_task_names:
                tensors_to_avg.append(CL_module.adapters[previous_task_names[-1]].detach().cpu())
        elif init_name == CLInitSpecials.All:
            tensors_to_avg.extend(m.detach().cpu() for m in CL_module.adapters.values())
        elif init_name in CL_module.adapters:
            tensors_to_avg.append(CL_module.adapters[init_name].detach().cpu())
        else:
            # logger.debug(f"{init_name!r} not found during init of {CL_module._group}.{CL_module._name}.{task_name}, skipped")
            pass

    n = len(tensors_to_avg)
    if n == 0:
        # logger.debug(f"{CL_module._group}.{CL_module._name}.{task_name} init without averaging")
        return new_module
    
    if n == 1:
        # logger.debug(f"{CL_module._group}.{CL_module._name}.{task_name} init by overriding")
        new_module.data = tensors_to_avg[0].clone()
    else:
        # logger.debug(f"{CL_module._group}.{CL_module._name}.{task_name} init by averaging {n} modules")
        new_module.data = sum(tensors_to_avg) / n
    return new_module

def new_e_adapter(module: CLAdapter, task_name: str, config: CLWavLMConfig):
    adapter = AdapterLayer(config)
    adapter._init_weights(config)
    adapter = _average_weights(adapter, module, task_name, config.cl_e_adapter_init)
    return adapter

def new_l_adapter(module: CLAdapter, task_name: str, config: CLWavLMConfig):
    adapter = AdapterToOutputLayer(config)
    adapter._init_weights(config)
    adapter = _average_weights(adapter, module, task_name, config.cl_l_adapter_init)
    return adapter

def new_layer_weights(num_layers: int) -> nn.Parameter:
    return nn.Parameter(torch.ones(num_layers) / num_layers)

def new_adapter_to_output_layer_weights(module: CLParameter, task_name: str, config: CLWavLMConfig):
    num_adapter_to_output_layers = config.num_hidden_layers
    adapter_to_output_layer_weights = new_layer_weights(num_adapter_to_output_layers)
    # length = 12, not same with WavLMForSequenceClassification.layer_weights (length = 13)
    config.layerdrop = 0.0  # Don't use LayerDrop at here
    adapter_to_output_layer_weights = _average_parameters(adapter_to_output_layer_weights, module, task_name, config.cl_layer_weights_init)
    return adapter_to_output_layer_weights

class AdaWavLMEncoderLayer(WavLMEncoderLayer):
    # def __init__(self, config: CustomWavLMConfig, layer: int, has_relative_position_bias = True):
    def __init__(self, config: CLWavLMConfig, has_relative_position_bias = True):
        super().__init__(config, has_relative_position_bias)

        if config.e_adapter_state == AdapterState.Missing:
            self.adapter_layer_ff = lambda x: x
        else:
            self.adapter_layer_ff = CLAdapter(new_e_adapter, config.e_adapter_state).manage_by(cl_modules, "e_adapter")

    def forward(self, hidden_states, attention_mask=None, position_bias=None, output_attentions=False, index=0):
        attn_residual = hidden_states
        hidden_states, attn_weights, position_bias = self.attention(  # May have UserWarning from pytorch
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            output_attentions=output_attentions,
            index=index,
        )
        hidden_states = self.dropout(hidden_states)
        # if self.config.use_adapter_attn:
        #     hidden_states = self.adapter_layer_attn(hidden_states)
        hidden_states = attn_residual + hidden_states

        hidden_states = self.layer_norm(hidden_states)
        
        res = hidden_states
        hidden_states = self.feed_forward(hidden_states)

        # if self.config.use_adapter_ff:
        hidden_states = res + self.adapter_layer_ff(hidden_states)

        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states, position_bias, )

        if output_attentions:
            outputs += (attn_weights,)
        
        return outputs

def _encoders_gen(config: CLWavLMConfig):
    yield AdaWavLMEncoderLayer(config, has_relative_position_bias=True)
    for i in range(1, config.num_hidden_layers):
        yield AdaWavLMEncoderLayer(config, has_relative_position_bias=False)

class AdaLayerToOutWavLMEncoder(WavLMEncoder):

    @property
    def use_l_adapter(self):
        return self.config.l_adapter_state != AdapterState.Missing

    @property
    def use_layer_weights_only(self):
        # return self.config.layer_weights_only
        return self.config.l_adapter_component == LComponentState.LayerWeights

    @property
    def use_l_adapters_only(self):
        return self.config.l_adapter_component == LComponentState.Adapters

    def __init__(self, config: CLWavLMConfig):
        super().__init__(config)
        self.config: CLWavLMConfig

        if config.e_adapter_state != AdapterState.Missing:
            self._edit_layers()

        if self.use_l_adapter:
            if not self.use_layer_weights_only:
                num_adapter_to_output_layers = config.num_hidden_layers
                self.adapter_to_output = nn.ModuleDict({
                    str(layer): CLAdapter(new_l_adapter, config.l_adapter_state).manage_by(cl_modules, "l_adapter")
                    for layer in range(num_adapter_to_output_layers)
                })
            if not self.use_l_adapters_only:
                self._adapter_to_output_layer_weights = CLParameter(new_adapter_to_output_layer_weights, config.l_adapter_state).manage_by(cl_modules, "l_adapter", "layer_weights")
            else:
                dummy_layer_weights = new_layer_weights(config.num_hidden_layers)
                dummy_layer_weights.requires_grad_(False)
                self._adapter_to_output_layer_weights = lambda: dummy_layer_weights
            # self.adapter_to_output_layer_weights = nn.Parameter(torch.ones(num_adapter_to_output_layers) / num_adapter_to_output_layers)
            # config.layerdrop = 0.0
            # Don't use LayerDrop at here

    def _edit_layers(self):
        # for layer in self.layers:
        #     layer: WavLMEncoderLayer
        #     layer.feed_forward = nn.Sequential(OrderedDict({
        #         "feed_forward": layer.feed_forward, 
        #         "adapter_layer_ff": CLAdapter(new_e_adapter)
        #     }))
        self.layers = nn.ModuleList(_encoders_gen(self.config))

    @property
    def adapter_to_output_layer_weights(self) -> nn.Parameter:
        return self._adapter_to_output_layer_weights()

    def _cl_e_adapter_gen(self) -> Iterable[CLAdapter]:
        # if self.config.e_adapter_state == AdapterState.Missing:
        #     return
        # for layer in self.layers:
        #     layer: AdaWavLMEncoderLayer
        #     yield layer.adapter_layer_ff
        yield from cl_modules.get("e_adapter", {}).values()

    def _cl_l_adapter_gen(self) -> Iterable[CLAdapter | CLParameter]:
        # if self.config.l_adapter_state == AdapterState.Missing:
        #     return
        # if not self.config.layer_weights_only:
        #     for adapter in self.adapter_to_output.values():
        #         yield adapter
        # yield self._adapter_to_output_layer_weights
        yield from cl_modules.get("l_adapter", {}).values()

    def _cl_adapter_gen(self) -> Iterable[CLAdapter | CLParameter]:
        # if self.config.use_e_adapter:
        # if self.config.e_adapter_state != AdapterState.Missing:
        #     for layer in self.layers:
        #         layer: AdaWavLMEncoderLayer
        #         # adapter: CLAdapter = layer.feed_forward[-1]
        #         # yield adapter
        #         yield layer.adapter_layer_ff
        # if self.config.l_adapter_state != AdapterState.Missing:
        #     for adapter in self.adapter_to_output.values():
        #         yield adapter
        #     yield self._adapter_to_output_layer_weights
        yield from self._cl_e_adapter_gen()
        yield from self._cl_l_adapter_gen()

    def add_task(self, task_name: str):
        for adapter in self._cl_adapter_gen():
            adapter.add_adapter(task_name, self.config)

    # def set_task(self, task_name: str):
    #     for adapter in self._cl_adapter_gen():
    #         adapter.current_task = task_name

    # def set_task_back(self):
    #     for adapter in self._cl_adapter_gen():
    #         adapter.current_task = adapter._previous_task

    # def set_task_grad(self, task_name: str | None = None, freeze=True):
    #     for adapter in self._cl_adapter_gen():
    #         adapter.set_grad(task_name, freeze)

    # def set_average_task(self):
    #     for adapter in self._cl_adapter_gen():
    #         adapter.average_adapter(self.config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        
        residual_adapter = ()
        # l_adapter_states = None
        
        if attention_mask is not None:
            # make sure padded tokens output 0
            hidden_states[~attention_mask] = 0.0

        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()
        position_bias = None

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                if residual_adapter:
                    all_hidden_states = all_hidden_states + residual_adapter[-1:]
                else:
                    all_hidden_states = all_hidden_states + (hidden_states,)

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            # don't use LayerDrop in AdaToOutputWavLM
            # dropout_probability = np.random.uniform(0, 1)
            dropout_probability = torch.rand([])

            skip_the_layer = self.training and i > 0 and (dropout_probability < self.config.layerdrop)
            if not skip_the_layer or deepspeed_zero3_is_enabled:
                # # under deepspeed zero3 all gpus must run in sync
                # if self.gradient_checkpointing and self.training:
                #     # create gradient checkpointing function
                #     def create_custom_forward(module):
                #         def custom_forward(*inputs):
                #             return module(*inputs, output_attentions)

                #         return custom_forward

                #     layer_outputs = torch.utils.checkpoint.checkpoint(
                #         create_custom_forward(layer),
                #         hidden_states,
                #         attention_mask,
                #         position_bias,
                #     )

                # under deepspeed zero3 all gpus must run in sync
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        layer.__call__,
                        hidden_states,
                        attention_mask,
                        position_bias,
                        output_attentions,
                    )
                else:
                    layer_outputs = layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_bias=position_bias,
                        output_attentions=output_attentions,
                        index=i,
                    )

                hidden_states, position_bias = layer_outputs[:2]
                
                if self.use_l_adapter:
                    # adapt output of FeedForad module
                    # layer_ada_keys = list(self.config.adapter_to_output_layer_size.keys()) 
                    # if str(i) in layer_ada_keys:        
                    #     residual_adapter += (self.adapter_to_output[str(i)](hidden_states),)
                    # l_adapter_states = self.adapter_to_output[str(i)](hidden_states)
                    if not self.use_layer_weights_only:
                        residual_adapter += (self.adapter_to_output[str(i)](hidden_states),)
                    else:
                        residual_adapter += (hidden_states,)
                    
            if skip_the_layer:
                # layer_outputs = (None, None)
                layer_outputs = (None, None, None)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[2],)

        if output_hidden_states:
            # if l_adapter_states:
            if residual_adapter:
                all_hidden_states = all_hidden_states + residual_adapter[-1:]
            else:
                all_hidden_states = all_hidden_states + (hidden_states,)
        
        # ====
        if self.use_l_adapter:
            hidden_states = torch.stack(residual_adapter, dim=1)
        
            # if self.config.use_adapter_to_output_weighted_sum:
            #     norm_weights = F.softmax(self.adapter_to_output_layer_weights, dim=-1)
            #     hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
            # else:
            #     hidden_states = hidden_states.mean(dim=1)
            if self.use_l_adapters_only:
                hidden_states = hidden_states.mean(dim=1)
            else:
                norm_weights = F.softmax(self.adapter_to_output_layer_weights, dim=-1)
                hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
            
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

class AdaWavLMModel(AdaPreTrainedModelOverride, WavLMModel):

    def __init__(self, config: CLWavLMConfig):
        super().__init__(config)

        self.encoder = AdaLayerToOutWavLMEncoder(config)

        self.post_init()


class ConfigHolder:
    
    def __init__(self, config: CustomWavLMConfig):
        self.config = config

def _mimic_init_weights(module: nn.Module, config: CustomWavLMConfig):
    fake_self = ConfigHolder(config)
    # AdaPreTrainedModelOverride._init_weights(fake_self, module)
    if isinstance(module, nn.Sequential):
        for m in module:
            _mimic_init_weights(m, config)
    WavLMPreTrainedModel._init_weights(fake_self, module)

def _projector(config: CLWavLMConfig):
    return nn.Linear(config.output_hidden_size, config.classifier_proj_size)

def new_projector(module: CLAdapter, task_name: str, config: CLWavLMConfig):
    projector = _projector(config)
    _mimic_init_weights(projector, config)
    projector = _average_weights(projector, module, task_name, config.cl_head_init)
    return projector

def _classifier(model: "AdaWavLMForSequenceClassification"):
    config = model.config
    if config.use_adaptive_pool:
        # return nn.Sequential(
        #     nn.AdaptiveAvgPool1d(config.head_adaptive_pool_size),
        #     nn.Flatten(),
        #     _classifier(config),
        # )
        if config.head_adaptive_pool == AdaptivePoolState.Max:
            pool = ModelAdaptiveMaxPool1d(model)  # need model as reference
        else:
            pool = ModelAdaptiveAvgPool1d(model)
        return nn.Sequential(
            nn.Linear(config.classifier_proj_size, config.head_adaptive_pool_size),
            pool,
            # nn.Flatten(),
        )
    return nn.Linear(config.classifier_proj_size, config.num_labels)

def _expand_head(module: CLAdapter, task_name: str, config: CLWavLMConfig):
    if module.current_task:
        head: ExpandingHead = module.adapters[module.current_task]
    else:
        head = ExpandingHead(config.classifier_proj_size)
    if task_name not in head.slices:
        head.expand(task_name, config)
    else:
        head.current_task = task_name
    head.requires_grad_(True)
    return head

def new_classifier(module: CLAdapter, task_name: str, config: CLWavLMConfig, model: "AdaWavLMForSequenceClassification"):
    if config.head_expanding:
        return _expand_head(module, task_name, config)
    else:
        classifier = _classifier(model)
        _mimic_init_weights(classifier, config)
        classifier = _average_weights(classifier, module, task_name, config.cl_head_init)
        return classifier

# _first_task_name = None
# def check_cl_init(func: Callable[Concatenate[str, P], None]) -> Callable[Concatenate[str, P], None]:
# def check_cl_init(func):
    
#     def inner(self: "AdaWavLMForSequenceClassification", task_name: str | None = None, *args, **kw):
#         global _first_task_name
#         if self.config.use_cl_init_only:
#             if _first_task_name:
#                 return  # do cl stuffs only at firs task
#             elif _first_task_name is None:
#                 _first_task_name = task_name  # cache firs task name
        
#         return func(self, task_name, *args, **kw)
    
#     return inner
                

class AdaWavLMForSequenceClassification(AdaPreTrainedModelOverride, WavLMForSequenceClassification):
    config: CLWavLMConfig

    def __init__(self, config: CLWavLMConfig):
        # if not config.use_upsampling:
        #     config.output_hidden_size = list(config.adapter_to_output_layer_size.values())[0]
        super().__init__(config)

        self.wavlm = AdaWavLMModel(config)
        # self.projector = _projector(config)
        # self.classifier = _classifier(self)

        def _new_classifier(module: CLAdapter, task_name: str, config: CLWavLMConfig):
            return new_classifier(module, task_name, config, self)
        
        self.projector = CLAdapter(new_projector, self.config.head_state).manage_by(cl_modules, "head", "projector")
        self.classifier = CLAdapter(_new_classifier, self.config.head_state).manage_by(cl_modules, "head", "classifier")

        self.post_init()

    def _cl_head_adapter_gen(self):
        # if self.config.head_state == AdapterState.Missing:
        #     return
        # assert isinstance(self.projector, CLAdapter)
        # yield self.projector
        # assert isinstance(self.classifier, CLAdapter)
        # yield self.classifier
        yield from cl_modules.get("head", {}).values()

    def _cl_adapter_gen(self, f: Callable[[CLAdapter], bool] | None = None):
        adapters = chain(
            self.wavlm.encoder._cl_e_adapter_gen(), 
            self.wavlm.encoder._cl_l_adapter_gen(),
            self._cl_head_adapter_gen()
        )
        if f:
            adapters = filter(f, adapters)

        yield from adapters

    def _add_task_head(self, task_name: str):
        # assert self.config.head_state != AdapterState.Missing  # head is required
        
        # if not isinstance(self.projector, CLAdapter):
        #     old_projector = self.projector
        #     self.projector = CLAdapter(new_projector, self.config.head_state).manage_by(cl_modules, "head", "projector")
        #     self.projector.set_adapter(task_name, old_projector)
        # else:
        #     self.projector.add_adapter(task_name, self.config)
        
        # if not isinstance(self.classifier, CLAdapter):
        #     old_classifier = self.classifier
        #     def _new_classifier(task_name: str, config: CLWavLMConfig):
        #         return new_classifier(task_name, config, self)
            
        #     self.classifier = CLAdapter(_new_classifier, self.config.head_state).manage_by(cl_modules, "head", "classifier")
        #     self.classifier.set_adapter(task_name, old_classifier)
        # else:
        #     self.classifier.add_adapter(task_name, self.config)
        return _add_task_head(self, task_name)

    def ensure_task_head(self, task_name: str, device: torch.device | None = None) -> str | None:
        # if task_name not in self.projector.adapters or task_name not in self.classifier.adapters:
        #     self._add_task_head(task_name)
        #     if device:
        #         self.projector.to(device)
        #         self.classifier.to(device)
        # else:
        #     if self.projector.current_task == task_name and self.classifier.current_task == task_name:
        #         return
        #     self.projector.current_task = task_name
        #     self.classifier.current_task = task_name
        # return self.classifier._previous_task
        return ensure_task_head(self, task_name, device)

    # def _set_task_head(self, task_name: str):
    #     if self.config.head_state == AdapterState.Missing:
    #         return
        
    #     assert isinstance(self.projector, CLAdapter)
    #     assert isinstance(self.classifier, CLAdapter)
    #     self.projector.current_task = task_name
    #     self.classifier.current_task = task_name

    # def _set_task_back_head(self):
    #     if self.config.head_state == AdapterState.Missing:
    #         return
        
    #     assert isinstance(self.projector, CLAdapter)
    #     assert isinstance(self.classifier, CLAdapter)
    #     self.projector.current_task = self.projector._previous_task
    #     self.classifier.current_task = self.classifier._previous_task

    # def _set_task_grad_head(self, task_name: str | None = None, freeze=True):
    #     if self.config.head_state == AdapterState.Missing:
    #         return
        
    #     assert isinstance(self.projector, CLAdapter)
    #     assert isinstance(self.classifier, CLAdapter)
    #     self.projector.set_grad(task_name, freeze)
    #     self.classifier.set_grad(task_name, freeze)

    # def _set_average_task_head(self):
    #     if self.config.head_state == AdapterState.Missing:
    #         return
        
    #     assert isinstance(self.projector, CLAdapter)
    #     assert isinstance(self.classifier, CLAdapter)
    #     self.projector.average_adapter(self.config)
    #     self.classifier.average_adapter(self.config)

    # # @check_cl_init
    def add_task(self, task_name: str):
        # self.wavlm.encoder.add_task(task_name)
        # self._add_task_head(task_name)
        return add_task(self, task_name)
    
    
    def set_task(self, task_name: str, f: Callable[[CLAdapter], bool] | None = None):
        # non_set = 0
        # for adapter in self._cl_adapter_gen(f):
        #     if task_name in adapter.adapters:
        #         adapter.current_task = task_name
        #     else:
        #         non_set += 1
        # if non_set:
        #     print(f"{task_name} not exists in {non_set} modules. Unable to set {task_name} for them.")
        return cl_modules.set_task(task_name, _f_with_names(f))
    
    # @check_cl_init  
    # def set_task(self, task_name: str):
    #     self.wavlm.encoder.set_task(task_name)
    #     self._set_task_head(task_name)

    # def set_task_back(self):
    #     self.wavlm.encoder.set_task_back()
    #     self._set_task_back_head()

    def set_task_back(self, f: Callable[[CLAdapter], bool] | None = None):
        # for adapter in self._cl_adapter_gen(f):
        #     adapter.current_task = adapter._previous_task
        return cl_modules.set_task_back(_f_with_names(f))

    def set_task_grad(self, task_name: str | None = None, freeze=True, f: Callable[[CLAdapter], bool] | None = None):
        # for adapter in self._cl_adapter_gen(f):
        #     adapter.set_grad(task_name, freeze)
        return cl_modules.set_task_grad(task_name, freeze, _f_with_names(f))

    # for legacy tasks
    def set_average_task(self, f: Callable[[CLAdapter], bool] | None = None):
        for adapter in self._cl_adapter_gen(f):
            adapter.average_adapter(self.config)

    # @check_cl_init
    # def set_task_grad(self, task_name: str | None = None, freeze=True):
    #     self.wavlm.encoder.set_task_grad(task_name, freeze)
    #     self._set_task_grad_head(task_name, freeze)

    # def set_average_task(self):
    #     self.wavlm.encoder.set_average_task()
    #     self._set_average_task_head()

    # for legacy tasks        
    def _task_names_gen(self):

        tasks: set[str] = set()
        for adapter in self._cl_adapter_gen():
            tasks.update(filter(not_special_task, adapter.adapters))
        # encoder_task_names = chain.from_iterable(adapter.adapters for adapter in self.wavlm.encoder._cl_adapter_gen())
        # tasks.update(filter(not_special_task, encoder_task_names))
        # if isinstance(self.projector, CLAdapter):
        #     tasks.update(filter(not_special_task, self.projector.adapters))
        # if isinstance(self.classifier, CLAdapter):
        #     tasks.update(filter(not_special_task, self.classifier.adapters))
        
        yield from sorted(tasks)
    
    @property
    def current_layer_weights(self):
        # return F.softmax(self.layer_weights, dim=-1)
        return F.softmax(self.wavlm.encoder.adapter_to_output_layer_weights.detach().cpu(), dim=-1)

def _f_with_names(f: Callable[[CLAdapter], bool] | None = None):
    if f is None:
        return
    def _deco_f(module, *_):
        return f(module)
    return _deco_f

def _add_task_head(model: AdaWavLMForSequenceClassification, task_name: str):
    assert model.config.head_state != AdapterState.Missing  # head is required

    # model.projector.add_adapter(task_name, model.config)
    # model.classifier.add_adapter(task_name, model.config)
    for module in cl_modules.get("head", {}).values():
        module.add_adapter(task_name, model.config)

def add_task(model: AdaWavLMForSequenceClassification, task_name: str, manager=cl_modules, f: Callable[[CLModule, str, str], bool] | None = None):
    for module in manager.module_gen(f):
        module.add_adapter(task_name, model.config)

def ensure_task_head(model: AdaWavLMForSequenceClassification, task_name: str, device: torch.device | None = None) -> str | None:
    projector = model.projector
    classifier = model.classifier
    if not isinstance(projector, CLAdapter):
        return
    if model.config.head_state != AdapterState.CL:
        return
    if task_name not in projector.adapters or task_name not in classifier.adapters:
        _add_task_head(model, task_name)
        if device:
            projector.to(device)
            classifier.to(device)
    else:
        to_change = (projector.current_task != task_name or classifier.current_task != task_name)
        if to_change:
            projector.current_task = task_name
            classifier.current_task = task_name
        if model.config.head_expanding:
            head: ExpandingHead = classifier.adapters[classifier.current_task]
            head.current_task = task_name
        if not to_change:
            return
    return classifier._previous_task

class ModelAdaptiveAvgPool1d(nn.AdaptiveAvgPool1d):

    model: AdaWavLMForSequenceClassification

    def __init__(self, model: AdaWavLMForSequenceClassification) -> None:
        super(nn.Module, self).__setattr__("model", model)  # do not register model as parameters
        super().__init__(self.model.config.num_labels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool1d(input, self.model.config.num_labels)  # Adaptive with different num_labels

class ModelAdaptiveMaxPool1d(nn.AdaptiveMaxPool1d):

    model: AdaWavLMForSequenceClassification

    def __init__(self, model: AdaWavLMForSequenceClassification, return_indices: bool = False) -> None:
        super(nn.Module, self).__setattr__("model", model)  # do not register model as parameters
        super().__init__(self.model.config.num_labels, return_indices)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.adaptive_max_pool1d(input, self.model.config.num_labels, self.return_indices) # Adaptive with different num_labels  


class ExpandingHead(nn.Module):

    def __init__(self, in_features: int, 
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.slices: dict[str, slice] = {}
        self.current_task: str | None = None
        self.in_features = in_features
        self._device = device
        self._dtype = dtype
        self.total_out_features = 0

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)

            self.linear = nn.Linear(self.in_features, 0, device=self._device, dtype=self._dtype)

    def expand(self, task_name: str, config: CLWavLMConfig):
        num_labels = config.num_labels
        self.current_task = task_name
        old_linear = self.linear
        old_end = self.total_out_features
        self.total_out_features += num_labels
        self.slices[task_name] = slice(old_end, self.total_out_features)

        self.linear = nn.Linear(self.in_features, self.total_out_features, device=self._device, dtype=self._dtype)
        with torch.no_grad():
            _mimic_init_weights(self.linear, config)
            self.linear.weight[:old_end, :] = old_linear.weight  
            self.linear.bias[:old_end] = old_linear.bias

        # self.linear.requires_grad_(True)

        logger.info(f"Expand head with {task_name}. Total num_labels: {old_end} => {self.total_out_features}")
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.linear.out_features:
            hidden_states = self.linear(hidden_states)
            _slice: slice = self.slices.get(self.current_task, slice(None))
            # if len(self.slices) > 1 and self.training:
            #     breakpoint()
            # if _slice.start and _slice.start > 0:
            #     left = hidden_states[:, 0:_slice.start]
            #     left.grad = None
            return hidden_states[:, _slice]
        return hidden_states

    
