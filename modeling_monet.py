# fmt: off
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.utils.checkpoint
from scipy.stats import norm
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_utils import PreTrainedModel
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LLAMA_ATTENTION_CLASSES,
    LlamaRMSNorm,
)
from transformers.utils import ModelOutput, logging

logger = logging.get_logger(__name__)


@dataclass
class MonetModelOutputWithPast(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: tuple[tuple[torch.FloatTensor]] | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    router_probs: tuple[tuple[torch.FloatTensor, ...], ...] | None = None


@dataclass
class MonetCausalLMOutputWithPast(ModelOutput):
    loss: torch.FloatTensor | None = None
    aux_loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor = None
    past_key_values: tuple[tuple[torch.FloatTensor]] | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    router_probs: tuple[tuple[torch.FloatTensor, ...], ...] | None = None


class MonetConfig(LlamaConfig):
    model_type = "monet"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=None,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="relu2",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=None,
        moe_dim=8,
        moe_heads=8,
        moe_experts=512,
        moe_topk=32,
        moe_groups=4,
        moe_decompose="vertical",
        output_router_probs=False,
        **kwargs,
    ):
        self.moe_dim = moe_dim
        self.moe_heads = moe_heads
        self.moe_experts = moe_experts
        self.moe_topk = moe_topk
        self.moe_groups = moe_groups
        self.moe_decompose = moe_decompose
        self.output_router_probs = output_router_probs

        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pretraining_tp=pretraining_tp,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            mlp_bias=mlp_bias,
            **kwargs,
        )


class MonetRouter(nn.Module):
    def __init__(self, config: MonetConfig):
        super().__init__()
        self.config = config
        flatten_shape = config.moe_heads * config.moe_experts

        self.w1 = nn.Linear(config.hidden_size, flatten_shape, bias=False)
        self.w2 = nn.Linear(config.hidden_size, flatten_shape, bias=False)
        self.norm1 = nn.BatchNorm1d(config.moe_heads, affine=False)
        self.norm2 = nn.BatchNorm1d(config.moe_heads, affine=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        g1z = self.w1(x).unflatten(-1, (self.config.moe_heads, -1)).float()
        g2z = self.w2(x).unflatten(-1, (self.config.moe_heads, -1)).float()

        g1n = self.norm1(g1z.transpose(2, 3).flatten(0, -2))
        g2n = self.norm2(g2z.transpose(2, 3).flatten(0, -2))
        g1n = g1n.view(g1z.size(0), g1z.size(1), g1z.size(3), -1).transpose(2, 3)
        g2n = g2n.view(g2z.size(0), g2z.size(1), g2z.size(3), -1).transpose(2, 3)

        sigma = float(norm.ppf(1 - self.config.moe_topk / self.config.moe_experts))
        g1s = g1n.amax(-1, keepdim=True).clamp_max_(sigma)
        g2s = g2n.amax(-1, keepdim=True).clamp_max_(sigma)

        g1 = nn.functional.softmax(torch.where(g1n >= g1s, g1z, -1e10), dim=-1)
        g2 = nn.functional.softmax(torch.where(g2n >= g2s, g2z, -1e10), dim=-1)
        return g1, g2


class MonetMoVDE(nn.Module):
    def __init__(self, config: MonetConfig):
        super().__init__()
        self.config = config
        self.act_fn = ACT2FN[config.hidden_act]
        flatten_shape = config.moe_experts * config.moe_dim // 2

        self.u1 = nn.Linear(config.hidden_size, flatten_shape)
        self.u2 = nn.Linear(config.hidden_size, flatten_shape)

        self.v11 = nn.Linear(flatten_shape, config.hidden_size // 2, bias=False)
        self.v12 = nn.Linear(flatten_shape, config.hidden_size // 2, bias=False)
        self.v21 = nn.Linear(flatten_shape, config.hidden_size // 2, bias=False)
        self.v22 = nn.Linear(flatten_shape, config.hidden_size // 2, bias=False)

        self.b1 = nn.Parameter(torch.zeros(config.moe_experts, config.hidden_size // 2))
        self.b2 = nn.Parameter(torch.zeros(config.moe_experts, config.hidden_size // 2))

    def forward(
        self, x: torch.Tensor, g1: torch.Tensor, g2: torch.Tensor
    ) -> torch.Tensor:
        g1, g2 = g1.type_as(x), g2.type_as(x)
        x1 = self.act_fn(self.u1(x).unflatten(-1, (self.config.moe_experts, -1)))
        x2 = self.act_fn(self.u2(x).unflatten(-1, (self.config.moe_experts, -1)))

        x11 = self.v11(torch.einsum("btim,bthi->btim", x1, g1).flatten(-2))
        x12 = self.v12(torch.einsum("btjm,bthj,bthi->btim", x2, g2, g1).flatten(-2))
        x13 = torch.einsum("bthi,id->btd", g1, self.b1.type_as(x))

        x21 = self.v21(torch.einsum("btim,bthi,bthj->btjm", x1, g1, g2).flatten(-2))
        x22 = self.v22(torch.einsum("btjm,bthj->btjm", x2, g2).flatten(-2))
        x23 = torch.einsum("bthj,jd->btd", g2, self.b2.type_as(x))

        return torch.cat((x11 + x12 + x13, x21 + x22 + x23), dim=-1)


class MonetMoHDE(nn.Module):
    def __init__(self, config: MonetConfig):
        super().__init__()
        self.config = config
        self.act_fn = ACT2FN[config.hidden_act]
        flatten_shape = config.moe_experts * config.moe_dim

        self.u = nn.Linear(config.hidden_size, flatten_shape)
        self.v = nn.Linear(flatten_shape, config.hidden_size, bias=False)
        self.b = nn.Parameter(torch.zeros(config.moe_experts, config.hidden_size))

    def forward(
        self, x: torch.Tensor, g1: torch.Tensor, g2: torch.Tensor
    ) -> torch.Tensor:
        g1, g2 = g1.type_as(x), g2.type_as(x)
        x = self.act_fn(self.u(x).unflatten(-1, (self.config.moe_experts, -1)))
        x = self.v(torch.einsum("btim,bthi,bthj->btjm", x, g1, g2).flatten(-2))
        return x + torch.einsum("bthj,jd->btd", g2, self.b)


class MonetDecoderLayer(nn.Module):
    def __init__(self, config: MonetConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](
            config=config, layer_idx=layer_idx
        )
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        if config.moe_decompose == "vertical":
            self.moe = MonetMoVDE(config)
        elif config.moe_decompose == "horizontal":
            self.moe = MonetMoHDE(config)
        if layer_idx % config.moe_groups == 0:
            self.router = MonetRouter(config).requires_grad_(False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Cache | None = None,
        previous_router_probs: tuple[torch.Tensor, torch.Tensor] | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[torch.FloatTensor, ...]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        g1, g2 = (
            self.router(hidden_states)
            if hasattr(self, "router")
            else previous_router_probs
        )
        hidden_states = self.moe(hidden_states, g1, g2)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs + ((g1, g2) if hasattr(self, "router") else None,)


class MonetPreTrainedModel(PreTrainedModel):
    config_class = MonetConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MonetDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class MonetModel(MonetPreTrainedModel):
    def __init__(self, config: MonetConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)  # noqa
        self.layers = nn.ModuleList([MonetDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])  # noqa
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_probs: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
    ) -> tuple[torch.Tensor, ...] | MonetModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions  # noqa
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states  # noqa
        output_router_probs = output_router_probs if output_router_probs is not None else self.config.output_router_probs  # noqa
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict  # noqa

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one")  # noqa

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.")  # noqa
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):  # kept for BC (non `Cache` `past_key_values` inputs)  # noqa
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "  # noqa
                "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"  # noqa
            )

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device)  # noqa
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions)  # noqa

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_probs = () if output_router_probs else None
        previous_router_probs, next_decoder_cache = None, None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    previous_router_probs,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    previous_router_probs=previous_router_probs,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
            if output_router_probs:
                all_router_probs += (layer_outputs[-1],)
            previous_router_probs = (
                layer_outputs[-1]
                if layer_outputs[-1] is not None
                else previous_router_probs
            )

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_router_probs] if v is not None)  # noqa
        return MonetModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_probs=all_router_probs,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0  # noqa
        using_static_cache = isinstance(past_key_values, StaticCache)

        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:  # noqa
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        if attention_mask is not None and attention_mask.dim() == 4:
            if attention_mask.max() != 0:
                raise ValueError("Custom 4D attention mask should be passed in inverted form with max==0`")  # noqa
            causal_mask = attention_mask
        else:
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device  # noqa
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)  # noqa
            causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)  # noqa
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit  # noqa
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]  # noqa
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(padding_mask, min_dtype)  # noqa
        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)  # noqa

        return causal_mask


class MonetForCausalLM(MonetPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = MonetModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_probs: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
    ) -> tuple[torch.Tensor, ...] | MonetCausalLMOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions  # noqa
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states  # noqa
        output_router_probs = output_router_probs if output_router_probs is not None else self.config.output_router_probs  # noqa
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict  # noqa

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_probs=output_router_probs,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return MonetCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_probs=outputs.router_probs,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        use_cache=True,
        **kwargs,
    ):
        past_length = 0
        if past_key_values is not None:
            past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()  # noqa
            max_cache_length = (
                torch.tensor(past_key_values.get_max_length(), device=input_ids.device)
                if past_key_values.get_max_length() is not None
                else None
            )
            cache_length = past_length if max_cache_length is None else torch.min(max_cache_length, past_length)  # noqa

            # Keep only the unprocessed tokens:
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:  # noqa
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]

            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        if inputs_embeds is not None and past_length == 0:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}

        input_length = position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]  # noqa
        if cache_position is None:
            cache_position = torch.arange(past_length, past_length + input_length, device=input_ids.device)  # noqa
        elif use_cache:
            cache_position = cache_position[-input_length:]

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),  # noqa
            )
        return reordered_past
