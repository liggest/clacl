# Code partially derived from https://github.com/sinhat98/adapter-wavlm/blob/main/modeling.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.activations import GELUActivation
from transformers.modeling_outputs import BaseModelOutput
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

from transformers.models.wavlm.modeling_wavlm import (
    WavLMConfig,
    WavLMEncoderLayer, 
    WavLMEncoder,
    WavLMPreTrainedModel,   
    WavLMModel, 
    WavLMForSequenceClassification
)

class CustomWavLMConfig(WavLMConfig):

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

        # L-Adapter & E-Adapter case
        # self.use_upsampling = False
        self.e_adapter_size = 256

        # self.use_e_adapter = kw.pop("use_e_adapter", True)
        # self.use_e_adapter = False

        self.use_l_adapter = kw.pop("use_l_adapter", True)  # use_adapter_to_output
        # self.use_l_adapter = False  # use_adapter_to_output
        if self.use_l_adapter:
            self.output_hidden_size = 512  # default is self.hidden_size (768)

        # self.use_cl_head = kw.pop("use_cl_head", True)
        # self.use_cl_head = False

        # self.use_cl_init_only = False  
        # self.use_cl_init_only = True  
        # just init model with CLAdapter at 1st task, no subsequent model modification

        self.adapter_init_mean = 0
        self.adapter_init_std = 1e-3
        self.l_adapter_size = 512
        # self.adapter_init = 'normal'

def _init_adapter_linear(module: nn.Linear, config: CustomWavLMConfig):
    module.weight.data.normal_(mean=config.adapter_init_mean, std=config.adapter_init_std)
    if module.bias is not None:
        module.bias.data.zero_()

def _init_layer_norm(module: nn.LayerNorm):
    module.bias.data.zero_()
    module.weight.data.fill_(1.0)

class AdapterLayer(nn.Module):
    # def __init__(self, config: CustomWavLMConfig, layer: int):
    def __init__(self, config: CustomWavLMConfig):
        super().__init__()
        # self.config = config
        # self.linear_down = nn.Linear(config.hidden_size, config.adapter_embedding_size[layer])
        self.linear_down = nn.Linear(config.hidden_size, config.e_adapter_size)
        # self.act = ACT2FN[config.eadapter_act] if config.eadapter_act else None
        self.act = GELUActivation()
        # self.linear_up = nn.Linear(config.adapter_embedding_size[layer], config.hidden_size)
        self.linear_up = nn.Linear(config.e_adapter_size, config.hidden_size)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    
    def _init_weights(self, config: CustomWavLMConfig):
        _init_adapter_linear(self.linear_down, config)
        _init_adapter_linear(self.linear_up, config)
        _init_layer_norm(self.layernorm)

    def forward(self, hidden_states):
        res = hidden_states
        hidden_states = self.linear_down(hidden_states)
        if self.act:
            hidden_states = self.act(hidden_states)
        # hidden_states = self.act(self.linear_down(hidden_states)) \
        #                     if self.act else self.linear_down(hidden_states)
        hidden_states = self.linear_up(hidden_states)
        hidden_states = self.layernorm(hidden_states)
        hidden_states = hidden_states + res
        return hidden_states

class AdaWavLMEncoderLayer(WavLMEncoderLayer):
    # def __init__(self, config: CustomWavLMConfig, layer: int, has_relative_position_bias = True):
    def __init__(self, config: CustomWavLMConfig, has_relative_position_bias = True):
        super().__init__(config, has_relative_position_bias)
        # self.config = config
        # self.attention = WavLMAttention(
        #     embed_dim=config.hidden_size,
        #     num_heads=config.num_attention_heads,
        #     dropout=config.attention_dropout,
        #     num_buckets=config.num_buckets,
        #     max_distance=config.max_bucket_distance,
        #     has_relative_position_bias=has_relative_position_bias,
        # )
        # self.dropout = nn.Dropout(config.hidden_dropout)

        # if config.use_adapter_attn:
        #     self.adapter_layer_attn = AdapterLayer(config, layer) 
        
        # self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # self.feed_forward = WavLMFeedForward(config)

        # if config.use_adapter_ff:
        # self.adapter_layer_ff = AdapterLayer(config, layer)
        self.adapter_layer_ff = AdapterLayer(config)
        # self.feed_forward = nn.Sequential(self.feed_forward, self.adapter_layer_ff)
        
        # self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

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


class AdaWavLMEncoder(WavLMEncoder):
    def __init__(self, config: CustomWavLMConfig):
        super().__init__(config)
        # self.layers = nn.ModuleList(
        #     [AdaWavLMEncoderLayer(config, layer=str(i), has_relative_position_bias=(i == 0)) if str(i) in list(config.adapter_embedding_size.keys())\
        #      else WavLMEncoderLayer(config, has_relative_position_bias=(i == 0)) for i in range(config.num_hidden_layers) ])
        self.layers = nn.ModuleList(_encoders_gen(config))

class AdapterToOutputLayer(nn.Module):
    # def __init__(self, config: CustomWavLMConfig, layer):
    def __init__(self, config: CustomWavLMConfig):
        super().__init__()
        # self.config = config
  
        # if config.use_adapter_fc:
        #     self.linear_down = nn.Linear(config.hidden_size, config.adapter_to_output_layer_size[layer])
        self.linear_down = nn.Linear(config.hidden_size, config.l_adapter_size)
        # self.act = ACT2FN[config.ladapter_act] if config.ladapter_act else None 
        self.act = GELUActivation()
    
        # if config.use_upsampling:
        #     self.linear_up = nn.Linear(config.adapter_to_output_layer_size[layer], config.hidden_size)
        
        # if config.adapter_dropproba:
        #     self.dropout = nn.Dropout(config.adapter_dropproba)
        
        # if config.use_adapter_norm:
        #     self.layernorm = nn.LayerNorm(config.output_hidden_size, eps=config.layer_norm_eps)
        self.layernorm = nn.LayerNorm(config.output_hidden_size, eps=config.layer_norm_eps)

    def _init_weights(self, config: CustomWavLMConfig):
        _init_adapter_linear(self.linear_down, config)
        _init_layer_norm(self.layernorm)

    def forward(self, hidden_states):
        # res = hidden_states
        # if  self.config.use_adapter_fc:
        #     hidden_states = self.act(self.linear_down(hidden_states)) if self.act else self.linear_down(hidden_states)
        # else:
        #     if self.act:
        #         hidden_states = self.act(hidden_states)
        hidden_states = self.act(self.linear_down(hidden_states))
        
        # if self.config.use_upsampling:
        #     hidden_states = self.linear_up(hidden_states)
        #     if self.config.use_adapter_postact and self.config.adapter_act:
        #         hidden_states = self.act(hidden_states)
        
        # if self.config.adapter_dropproba:
        #     hidden_states = self.dropout(hidden_states)
        
        # if self.config.use_adapter_norm:
        #     hidden_states = self.layernorm(hidden_states)
        hidden_states = self.layernorm(hidden_states)
            
        # if self.config.use_residual and self.config.use_upsampling:
        #     hidden_states = hidden_states + res

        return hidden_states
        
class AdaLayerToOutWavLMEncoder(WavLMEncoder):
    def __init__(self, config: CustomWavLMConfig):
        super().__init__(config)
        self.config: CustomWavLMConfig
        # self.layers = nn.ModuleList(
        #     [AdaWavLMEncoderLayer(config, layer=str(i), has_relative_position_bias=(i == 0)) if str(i) in list(config.adapter_embedding_size.keys())\
        #     else WavLMEncoderLayer(config, has_relative_position_bias=(i == 0)) for i in range(config.num_hidden_layers) ]
        #     )
        self.layers = nn.ModuleList(_encoders_gen(config))
        
        # self.adapter_to_output = nn.ModuleDict(
        #     {layer:AdapterToOutputLayer(config, layer) for layer in list(config.adapter_to_output_layer_size.keys())}
        # )
        num_adapter_to_output_layers = config.num_hidden_layers
        self.adapter_to_output = nn.ModuleDict({
            str(layer): AdapterToOutputLayer(config) for layer in range(num_adapter_to_output_layers)
        })
        
        # self.num_adapter_to_output = len(config.adapter_to_output_layer_size.keys())
        # self.num_adapter_layer = len(config.adapter_embedding_size.keys())
        # if config.use_adapter_to_output_weighted_sum:
        #     if self.num_adapter_to_output:
        #         num_adapter_to_output_layers = self.num_adapter_to_output
        #     else:
        #         num_adapter_to_output_layers = self.num_adapter_layer
            
            # self.adapter_to_output_layer_weights = nn.Parameter(torch.ones(num_adapter_to_output_layers) / num_adapter_to_output_layers)
            # config.layerdrop=0.0
        self.adapter_to_output_layer_weights = nn.Parameter(torch.ones(num_adapter_to_output_layers) / num_adapter_to_output_layers)
        config.layerdrop = 0.0
        # Don't use LayerDrop at here

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


# class AdaPreTrainedModelOverride:
#     config_class = CustomWavLMConfig
#     config: CustomWavLMConfig

#     def _init_weights(self, module: nn.Module):
#         if not isinstance(module, (AdaLayerToOutWavLMEncoder, AdaWavLMEncoder)):
#             # return super()._init_weights(module)
#             return WavLMPreTrainedModel._init_weights(self, module)
#         # initialize adapter layer
#         for name, param in module.named_parameters():
#             if 'adapter' in name and 'linear' in name:
#                 if 'weight' in name:
#                     # if self.config.adapter_init == 'normal': 
#                         # initialize adapter as near-identity function
#                     param.data.normal_(mean=self.config.adapter_init_mean, std=self.config.adapter_init_std)
#                     # elif self.config.adapter_init == 'uniform':
#                     #     param.data.uniform_(a=-self.config.adapter_init_value, b=self.config.adapter_init_value)
#                     # elif self.config.adapter_init == 'constant':
#                     #     nn.init.constant_(param, self.config.adapter_init_value)
#                     # elif self.config.adapter_init == 'eye':
#                     #     nn.init.eye_(param)
#                     # elif self.config.adapter_init == 'zero':
#                     #     param.data.zero_()
#                     # elif self.config.adapter_init == 'he':
#                     #     nn.init.kaiming_uniform_(param, a=math.sqrt(5)) 
#                     # else:
#                     #     raise ValueError('error') 
#                 elif 'bias' in name:
#                     param.data.zero_()

class AdaPreTrainedModelOverride(WavLMPreTrainedModel):
    config_class = CustomWavLMConfig
    config: CustomWavLMConfig

    def _init_weights(self, module: nn.Module):
        super()._init_weights(module)
        # initialize adapter layer
        if isinstance(module, (AdapterLayer, AdapterToOutputLayer)):
            module._init_weights(self.config)

class AdaWavLMModel(AdaPreTrainedModelOverride, WavLMModel):

    def __init__(self, config: CustomWavLMConfig):
        super().__init__(config)

        if config.use_l_adapter:
            self.encoder = AdaLayerToOutWavLMEncoder(config)
        elif isinstance(self.encoder, WavLMEncoder):
            self.encoder = AdaWavLMEncoder(config)

        self.post_init()

class AdaWavLMForSequenceClassification(AdaPreTrainedModelOverride, WavLMForSequenceClassification):

    def __init__(self, config: CustomWavLMConfig):
        # if not config.use_upsampling:
        #     config.output_hidden_size = list(config.adapter_to_output_layer_size.values())[0]
        super().__init__(config)

        self.wavlm = AdaWavLMModel(config)
        self.projector = nn.Linear(config.output_hidden_size, config.classifier_proj_size)

        self.post_init()
