
from typing import Callable, Iterable
# from collections import OrderedDict
# from functools import cached_property
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

from clacl.model.wavml import CustomWavLMConfig
from clacl.model.wavml import AdaPreTrainedModelOverride, AdapterLayer, AdapterToOutputLayer

class AdapterState(str, Enum):
    Missing = "missing"
    Freeze = "freeze"
    TuneOnce = "tune_once"
    TuneAll = "tune_all"
    CL = "cl"

class AdaptivePoolState(str, Enum):
    Missing = "missing"
    Avg = "avg"
    Max = "max"

class CLWavLMConfig(CustomWavLMConfig):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.e_adapter_state = AdapterState(kw.pop("e_adapter_state", AdapterState.CL))
        self.l_adapter_state = AdapterState(kw.pop("l_adapter_state", AdapterState.CL))
        self.head_state = AdapterState(kw.pop("head_state", AdapterState.CL))

        # self.use_e_adapter = (self.e_adapter_state != AdapterState.Missing)
        # self.use_l_adapter = (self.l_adapter_state != AdapterState.Missing) or self.use_l_adapter

        # if self.l_adapter_state == AdapterState.Missing:
        #     self.output_hidden_size = self.hidden_size  # default 768

        # self.use_cl_head = (self.head_state != AdapterState.Missing)

        # self.use_cl_init_only = False

        self.head_adaptive_pool: AdaptivePoolState = AdaptivePoolState(kw.pop("head_adaptive_pool", AdaptivePoolState.Missing))
        self.head_adaptive_pool_size: int = kw.pop("head_adaptive_pool_size", 0)
        if self.use_adaptive_pool:
            assert self.head_adaptive_pool_size > 0

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
    def use_adaptive_pool(self):
        return self.head_adaptive_pool != AdaptivePoolState.Missing

AdaPreTrainedModelOverride.config_class = CLWavLMConfig

def not_special_task(task_name: str):
    return not task_name.startswith("_")

class CLAdapter(nn.Module):            

    def __init__(self, factory: Callable[[str, CLWavLMConfig], nn.Module], state=AdapterState.CL):
        super().__init__()
        self.adapters = self._init_adapters()
        self._current_task: str | None = None
        self.module_factory = factory
        self.state = state
        self._previous_task: str | None = None

    def _init_adapters(self):
        return nn.ModuleDict()

    @property
    def current_task(self):
        return self._current_task
    
    @current_task.setter
    def current_task(self, val: str | None):
        self._previous_task = self._current_task
        self._current_task = val

    def add_adapter(self, task_name: str, config: CLWavLMConfig):
        if (self.state == AdapterState.TuneOnce and self.current_task):
            self.set_grad(self.current_task, freeze=True)
        
        if (not_special_task(task_name)
                and self.adapters 
                and self.state != AdapterState.CL):
            return  # only one module

        adapter = self.module_factory(task_name, config)
        self.set_adapter(task_name, adapter)

        if (self.state == AdapterState.Freeze):
            self.set_grad(task_name, freeze=True)

    def set_adapter(self, task_name: str, adapter: nn.Module):
        if (not_special_task(task_name)
                and self.adapters 
                and self.state != AdapterState.CL 
                and task_name not in self.adapters):
            return  # only one module
        
        # self.previous_task = self.current_task
        self.adapters[task_name] = adapter
        self.current_task = task_name

    def set_grad(self, task_name: str | None = None, freeze=True):
        if task_name is None:
            task_name = self.current_task
        assert task_name
        self.adapters[task_name].requires_grad_(not freeze)

    def forward(self, hidden_states):
        if not self.current_task:
            warnings.warn(f"{self.current_task = !r} in {self!r}")
            return hidden_states
        return self.adapters[self.current_task](hidden_states)
    
    def average_adapter(self, config: CLWavLMConfig):
        task_name = "_average_"
        if self.current_task == task_name:
            return
        if self.state != AdapterState.CL:
            return
        if (task_name not in self.adapters):
            self.add_adapter(task_name, config)  # will also change current task
        else:
            self.set_adapter(task_name, self.adapters[task_name])
        self.set_grad(task_name, freeze=True)
        average = self.adapters[task_name]
        if self._previous_task:
            previous_device = next(self.adapters[self._previous_task].parameters()).device
            average.to(previous_device)
        average_state = average.state_dict()
        states = [adapter.state_dict() for t_name, adapter in self.adapters.items() if not_special_task(t_name)]
        n = len(states)
        for key in average_state:
            average_state[key] = sum(state[key] for state in states) / n
        average.load_state_dict(average_state)


class CLParameter(nn.Module):

    def __init__(self, factory: Callable[[str, CLWavLMConfig], nn.Parameter], state=AdapterState.CL):
        super().__init__() 
        self.adapters = self._init_adapters()
        self._current_task: str | None = None
        self.module_factory = factory
        self.state = state
        self._previous_task: str | None = None

    def _init_adapters(self):
        return nn.ParameterDict()

    current_task = CLAdapter.current_task

    add_adapter = CLAdapter.add_adapter
    set_adapter = CLAdapter.set_adapter
    set_grad = CLAdapter.set_grad

    def forward(self) -> nn.Parameter:
        assert self.current_task, f"{self.current_task = !r} in {self!r}"
        return self.adapters[self.current_task]

    def average_adapter(self, config: CLWavLMConfig):
        task_name = "_average_"
        if self.current_task == task_name:
            return
        if self.state != AdapterState.CL:
            return
        if (task_name not in self.adapters):
            self.add_adapter(task_name, config)  # will also change current task
        else:
            self.set_adapter(task_name, self.adapters[task_name])
        self.set_grad(task_name, freeze=True)
        average: nn.Parameter = self.adapters[task_name]
        if self._previous_task:
            previous: nn.Parameter = self.adapters[self._previous_task]
            average.to(previous.device)
        tensors = [adapter.data for t_name, adapter in self.adapters.items() if not_special_task(t_name)]
        n = len(tensors)
        average.data = sum(tensors) / n

def new_e_adapter(task_name: str, config: CustomWavLMConfig):
    adapter = AdapterLayer(config)
    adapter._init_weights(config)
    return adapter

def new_l_adapter(task_name: str, config: CustomWavLMConfig):
    adapter = AdapterToOutputLayer(config)
    adapter._init_weights(config)
    return adapter

def new_adapter_to_output_layer_weights(task_name: str, config: CustomWavLMConfig):
    num_adapter_to_output_layers = config.num_hidden_layers
    adapter_to_output_layer_weights = nn.Parameter(torch.ones(num_adapter_to_output_layers) / num_adapter_to_output_layers)
    config.layerdrop = 0.0  # Don't use LayerDrop at here
    return adapter_to_output_layer_weights

class AdaWavLMEncoderLayer(WavLMEncoderLayer):
    # def __init__(self, config: CustomWavLMConfig, layer: int, has_relative_position_bias = True):
    def __init__(self, config: CLWavLMConfig, has_relative_position_bias = True):
        super().__init__(config, has_relative_position_bias)

        if config.e_adapter_state == AdapterState.Missing:
            self.adapter_layer_ff = lambda x: x
        else:
            self.adapter_layer_ff = CLAdapter(new_e_adapter, config.e_adapter_state)

    def forward(self, hidden_states, attention_mask=None, position_bias=None, output_attentions=False, index=0):
        attn_residual = hidden_states
        hidden_states, attn_weights, position_bias = self.attention(
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

def _encoders_gen(config: CustomWavLMConfig):
    yield AdaWavLMEncoderLayer(config, has_relative_position_bias=True)
    for i in range(1, config.num_hidden_layers):
        yield AdaWavLMEncoderLayer(config, has_relative_position_bias=False)

class AdaLayerToOutWavLMEncoder(WavLMEncoder):

    def __init__(self, config: CLWavLMConfig):
        super().__init__(config)
        self.config: CLWavLMConfig

        if config.e_adapter_state != AdapterState.Missing:
            self._edit_layers()
        
        self.use_l_adapter = (config.l_adapter_state != AdapterState.Missing)
        if self.use_l_adapter:
            num_adapter_to_output_layers = config.num_hidden_layers
            self.adapter_to_output = nn.ModuleDict({
                str(layer): CLAdapter(new_l_adapter, config.l_adapter_state)
                for layer in range(num_adapter_to_output_layers)
            })
            self._adapter_to_output_layer_weights = CLParameter(new_adapter_to_output_layer_weights, config.l_adapter_state)
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
        if self.config.e_adapter_state == AdapterState.Missing:
            return
        for layer in self.layers:
            layer: AdaWavLMEncoderLayer
            yield layer.adapter_layer_ff

    def _cl_l_adapter_gen(self) -> Iterable[CLAdapter | CLParameter]:
        if self.config.l_adapter_state == AdapterState.Missing:
            return
        for adapter in self.adapter_to_output.values():
                yield adapter
        yield self._adapter_to_output_layer_weights

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
                    residual_adapter += (self.adapter_to_output[str(i)](hidden_states),)
                    
            if skip_the_layer:
                # layer_outputs = (None, None)
                layer_outputs = (None, None, None)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        # ====
        if self.use_l_adapter:
            hidden_states = torch.stack(residual_adapter, dim=1)
        
            # if self.config.use_adapter_to_output_weighted_sum:
            #     norm_weights = F.softmax(self.adapter_to_output_layer_weights, dim=-1)
            #     hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
            # else:
            #     hidden_states = hidden_states.mean(dim=1)
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

    def __init__(self, config: CustomWavLMConfig):
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

def new_projector(task_name: str, config: CLWavLMConfig):
    projector = _projector(config)
    _mimic_init_weights(projector, config)
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
            pool = ModelAdaptiveMaxPool1d(model)
        else:
            pool = ModelAdaptiveAvgPool1d(model)
        return nn.Sequential(
            nn.Linear(config.classifier_proj_size, config.head_adaptive_pool_size),
            pool,
            # nn.Flatten(),
        )
    return nn.Linear(config.classifier_proj_size, config.num_labels)

def new_classifier(task_name: str, config: CLWavLMConfig, model: "AdaWavLMForSequenceClassification"):
    classifier = _classifier(model)
    _mimic_init_weights(classifier, config)
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
        self.projector = _projector(config)
        self.classifier = _classifier(self)
        self.post_init()

    def _cl_head_adapter_gen(self):
        if self.config.head_state == AdapterState.Missing:
            return
        assert isinstance(self.projector, CLAdapter)
        yield self.projector
        assert isinstance(self.classifier, CLAdapter)
        yield self.classifier

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
        assert self.config.head_state != AdapterState.Missing  # head is required
        
        if not isinstance(self.projector, CLAdapter):
            old_projector = self.projector
            self.projector = CLAdapter(new_projector, self.config.head_state)
            self.projector.set_adapter(task_name, old_projector)
        else:
            self.projector.add_adapter(task_name, self.config)
        
        if not isinstance(self.classifier, CLAdapter):
            old_classifier = self.classifier
            def _new_classifier(task_name: str, config: CLWavLMConfig):
                return new_classifier(task_name, config, self)
            
            self.classifier = CLAdapter(_new_classifier, self.config.head_state)
            self.classifier.set_adapter(task_name, old_classifier)
        else:
            self.classifier.add_adapter(task_name, self.config)

    def ensure_task_head(self, task_name: str, device: torch.device | None = None) -> str | None:
        if task_name not in self.projector.adapters or task_name not in self.classifier.adapters:
            self._add_task_head(task_name)
            if device:
                self.projector.to(device)
                self.classifier.to(device)
        else:
            if self.projector.current_task == task_name and self.classifier.current_task == task_name:
                return
            self.projector.current_task = task_name
            self.classifier.current_task = task_name
        return self.classifier._previous_task

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

    # @check_cl_init
    def add_task(self, task_name: str):
        self.wavlm.encoder.add_task(task_name)
        self._add_task_head(task_name)
    
    
    def set_task(self, task_name: str, f: Callable[[CLAdapter], bool] | None = None):
        non_set = 0
        for adapter in self._cl_adapter_gen(f):
            if task_name in adapter.adapters:
                adapter.current_task = task_name
            else:
                non_set += 1
        if non_set:
            print(f"{task_name} not exists in {non_set} modules. Unable to set {task_name} for them.")
    
    # @check_cl_init  
    # def set_task(self, task_name: str):
    #     self.wavlm.encoder.set_task(task_name)
    #     self._set_task_head(task_name)

    # def set_task_back(self):
    #     self.wavlm.encoder.set_task_back()
    #     self._set_task_back_head()

    def set_task_back(self, f: Callable[[CLAdapter], bool] | None = None):
        for adapter in self._cl_adapter_gen(f):
            adapter.current_task = adapter._previous_task

    def set_task_grad(self, task_name: str | None = None, freeze=True, f: Callable[[CLAdapter], bool] | None = None):
        for adapter in self._cl_adapter_gen(f):
            adapter.set_grad(task_name, freeze)

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
        return F.softmax(self.wavlm.encoder.adapter_to_output_layer_weights.detach().cpu(), dim=-1)

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
