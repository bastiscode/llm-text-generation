import copy
import functools
import time
import os
import sys
from typing import Any, Optional
from copy import deepcopy

import torch
from torch import nn
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.hooks import RemovableHandle
from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel,
    LlamaForCausalLM,
    GPT2LMHeadModel,
    MistralForCausalLM,
    MixtralForCausalLM
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithCrossAttentions,
    CausalLMOutputWithPast,
    MoeCausalLMOutputWithPast
)
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
from transformers.models.mixtral.modeling_mixtral import MixtralDecoderLayer
from transformers.models.phi.modeling_phi import (
    PhiForCausalLM,
    PhiDecoderLayer
)
from transformers.utils import logging as hf_logging
from braceexpand import braceexpand

from text_utils.api import utils
from text_utils.api.trainer import ShardingPolicy

hf_logging.disable_progress_bar()


def _register_hook(
    hooks: list[RemovableHandle],
    m: nn.Module,
    device: torch.device,
):
    m = m.to(device)

    def _pre_hook(
        m: nn.Module,
        args: tuple,
        kwargs: dict
    ) -> tuple[tuple, dict]:
        m = m.to(device)
        return utils.to(args, device), utils.to(kwargs, device)

    hook = m.register_forward_pre_hook(_pre_hook, with_kwargs=True)
    hooks.append(hook)


class Model(nn.Module):
    model: nn.Module
    hooks: list[RemovableHandle]

    def __init__(self):
        super().__init__()
        self.hooks = []

    def forward(
        self,
        token_ids: torch.Tensor,
        **kwargs: Any
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        raise NotImplementedError

    def get_sharding_policy(self) -> ShardingPolicy | None:
        return None

    def compile(
        self,
        **kwargs: Any
    ) -> "Model":
        return self

    def distribute(
        self,
        devices: list[torch.device]
    ) -> "Model":
        assert len(devices) == 1, "only single device is supported"
        return self.to(devices[0])


PRETRAINED_DECODERS = [
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
    "llama-2-7b",
    "llama-2-30b",
    "llama-2-70b",
    "llama-2-7b-chat",
    "llama-2-30b-chat",
    "llama-2-70b-chat",
    "llama-3-8b",
    "llama-3-8b-instruct",
    "llama-3-70b",
    "llama-3-70b-instruct",
    "mistral-7b",
    "mistral-7b-instruct",
    "mixtral-8x7b",
    "mixtral-8x7b-instruct",
    "mixtral-8x22b",
    "mixtral-8x22b-4bit",
    "phi-2"
]


class PretrainedDecoder(Model):
    def __init__(
        self,
        name: str | PreTrainedModel,
        **kwargs: Any
    ):
        super().__init__()
        if kwargs.get("device_map") is not None:
            kwargs["device_map"] = brace_expand_keys(kwargs["device_map"])

        if isinstance(name, PreTrainedModel):
            assert isinstance(name, PreTrainedModel)
            self.model = name
        else:
            assert name in PRETRAINED_DECODERS, f"unknown model {name}"
            if name.startswith("llama-3"):
                name = name.lower()
                if name.endswith("8b"):
                    name = "Meta-Llama-3-8B"
                elif name.endswith("8b-instruct"):
                    name = "Meta-Llama-3-8B-Instruct"
                elif name.endswith("70b"):
                    name = "Meta-Llama-3-70B"
                else:
                    name = "Meta-Llama-3-70B-Instruct"
                self.model = LlamaForCausalLM.from_pretrained(
                    f"meta-llama/{name}",
                    torch_dtype=kwargs.pop("torch_dtype", "auto"),
                    **kwargs
                )  # type: ignore
            elif name.startswith("llama-2"):
                self.model = LlamaForCausalLM.from_pretrained(
                    f"meta-llama/{name.capitalize()}-hf",
                    torch_dtype=kwargs.pop("torch_dtype", "auto"),
                    **kwargs
                )  # type: ignore
            elif name.startswith("mistral-7b"):
                if name.endswith("instruct"):
                    name = "Mistral-7B-Instruct-v0.2"
                else:
                    name = "Mistral-7B-v0.1"
                self.model = MistralForCausalLM.from_pretrained(
                    f"mistralai/{name}",
                    torch_dtype=kwargs.pop("torch_dtype", "auto"),
                    **kwargs
                )  # type: ignore
            elif name.startswith("mixtral-8x7b"):
                if name.endswith("instruct"):
                    name = "Mixtral-8x7B-Instruct-v0.1"
                else:
                    name = "Mixtral-8x7B-v0.1"
                self.model = MixtralForCausalLM.from_pretrained(
                    f"mistralai/{name}",
                    torch_dtype=kwargs.pop("torch_dtype", "auto"),
                    **kwargs
                )  # type: ignore
            elif name.startswith("mixtral-8x22b"):
                if name.endswith("4bit"):
                    name = "Mixtral-8x22B-v0.1-4bit"
                else:
                    name = "Mixtral-8x22B-v0.1"
                self.model = MixtralForCausalLM.from_pretrained(
                    f"mistral-community/{name}",
                    torch_dtype=kwargs.pop("torch_dtype", "auto"),
                    **kwargs
                )  # type: ignore
            elif name == "phi-2":
                self.model = PhiForCausalLM.from_pretrained(
                    "microsoft/phi-2",
                    torch_dtype=kwargs.pop("torch_dtype", "auto"),
                    **kwargs
                )  # type: ignore
            else:
                self.model = GPT2LMHeadModel.from_pretrained(
                    name,
                    torch_dtype=kwargs.pop("torch_dtype", "auto"),
                    **kwargs
                )  # type: ignore

        if isinstance(self.model, LlamaForCausalLM):
            self.layer_cls = LlamaDecoderLayer
        elif isinstance(self.model, MistralForCausalLM):
            self.layer_cls = MistralDecoderLayer
        elif isinstance(self.model, MixtralForCausalLM):
            self.layer_cls = MixtralDecoderLayer
        elif isinstance(self.model, PhiForCausalLM):
            self.layer_cls = PhiDecoderLayer
        elif isinstance(self.model, GPT2LMHeadModel):
            self.layer_cls = GPT2Block
        else:
            raise RuntimeError(f"unkown model type {type(self.model)}")

    def get_sharding_policy(self) -> ShardingPolicy | None:
        return functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                self.layer_cls
            }  # type: ignore
        )

    def enable_gradient_checkpointing(self) -> None:
        assert isinstance(self.model, PreTrainedModel)
        self.model.config.use_cache = False
        self.model.gradient_checkpointing_enable()

    def forward(
        self,
        token_ids: torch.Tensor,
        **_: Any
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        output = self.model(input_ids=token_ids)  # type: ignore
        return output.logits, {}

    def compile(
        self,
        **cfg: Any
    ) -> "PretrainedDecoder":
        cfg = copy.deepcopy(cfg)
        typ = cfg.pop("type", None)
        if typ == "torch":
            self.model = torch.compile(
                self.model,
                **cfg
            )  # type: ignore
        elif typ == "tensorrt":
            import torch_tensorrt as trt
            path = cfg.pop("path", None)
            if path is not None:
                self.model = torch.jit.load(path)
                return self
            max_bs = cfg.get("max_batch_size", 16)
            max_len = cfg.get("max_length", 512)
            name = cfg.get("name", "input_ids")
            self.model = trt.compile(
                self.model,
                inputs=[trt.Input(
                    min_shape=(1, 1),
                    opt_shape=(max_bs // 2, max_len // 2),
                    max_shape=(max_bs, max_len),
                    dtype=torch.long,
                    name=name
                )],
                enabled_precisions={next(self.model.parameters()).dtype},
                **cfg
            )  # type: ignore
            if path is not None:
                torch.jit.save(self.model, path)

        return self

    def decode(
        self,
        token_ids: torch.Tensor,
        lengths: torch.Tensor,
        kv_cache: Optional[tuple[tuple[torch.Tensor]]] = None,
        use_cache: bool = True
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor]]]:
        if use_cache and kv_cache is not None:
            b, s = token_ids.shape
            assert token_ids.ndim == 2 and s > 0
            token_ids = token_ids[torch.arange(
                b, device=token_ids.device
            ), lengths - 1, None]
        start = time.perf_counter()
        output = self.model(  # type: ignore
            input_ids=token_ids,
            past_key_values=kv_cache,
            use_cache=use_cache
        )
        end = time.perf_counter()
        # time in ms
        debug_print(f"decode time: {(end - start) * 1000:.2f} ms ")
        assert isinstance(
            output,
            (BaseModelOutputWithPast, CausalLMOutputWithPast,
             CausalLMOutputWithCrossAttentions, MoeCausalLMOutputWithPast)
        ), f"unexpected output type {type(output)}"
        return output.logits, output.past_key_values  # type: ignore

    def distribute(
        self,
        devices: list[torch.device]
    ) -> "Model":
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        assert len(devices) > 0
        if len(devices) == 1:
            return self.to(devices[0])

        # distribute the layers
        layers = [
            m
            for m in self.model.modules()
            if isinstance(m, self.layer_cls)
        ]
        assert len(layers) > 0 and len(devices) <= len(layers), \
            f"{len(devices)} devices for {len(layers)} layers not supported"

        # distribute evenly among devices
        layers_per_device = [len(layers) // len(devices)] * len(devices)
        for i in range(len(layers) % len(devices)):
            layers_per_device[i] += 1

        device_idx = 0
        for i, m in enumerate(layers):
            _register_hook(self.hooks, m, devices[device_idx])
            if i + 1 == sum(layers_per_device[:device_idx + 1]):
                device_idx += 1

        assert device_idx == len(devices)
        # add additional hooks for modules outside the regular
        # transformer layers
        if isinstance(self.model, LlamaForCausalLM):
            _register_hook(
                self.hooks,
                self.model.model.embed_tokens,
                devices[0]
            )
            _register_hook(
                self.hooks,
                self.model.model.norm,
                devices[-1]
            )
            _register_hook(
                self.hooks,
                self.model.lm_head,
                devices[-1]
            )
        elif isinstance(self.model, (MistralForCausalLM, MixtralForCausalLM)):
            _register_hook(
                self.hooks,
                self.model.model.embed_tokens,
                devices[0]
            )
            _register_hook(
                self.hooks,
                self.model.model.norm,
                devices[-1]
            )
            _register_hook(
                self.hooks,
                self.model.lm_head,
                devices[-1]
            )
        elif isinstance(self.model, PhiForCausalLM):
            _register_hook(
                self.hooks,
                self.model.model.embed_tokens,
                devices[0]
            )
            _register_hook(
                self.hooks,
                self.model.model.final_layer_norm,
                devices[-1]
            )
            _register_hook(
                self.hooks,
                self.model.lm_head,
                devices[-1]
            )
        else:
            assert isinstance(self.model, GPT2LMHeadModel)
            _register_hook(
                self.hooks,
                self.model.transformer.wte,
                devices[0]
            )
            _register_hook(
                self.hooks,
                self.model.transformer.wpe,
                devices[0]
            )
            _register_hook(
                self.hooks,
                self.model.transformer.ln_f,
                devices[-1]
            )
            _register_hook(
                self.hooks,
                self.model.lm_head,
                devices[-1]
            )
        return self


def model_from_config(
    cfg: dict[str, Any],
) -> Model:
    cfg = copy.deepcopy(cfg)
    model_type = cfg.pop("type")

    if model_type == "pretrained_decoder":
        return PretrainedDecoder(**cfg)
    elif model_type == "custom_pretrained_decoder":
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"],
            torch_dtype="auto"
        )
        return PretrainedDecoder(model)
    else:
        raise ValueError(f"unknown model type {model_type}")


def brace_expand_keys(in_dict: dict[str, Any]):
    """
    Expands keys of the input dict using bash braceexpand.

    Note that this is not recursive, so nested dicts will not be expanded.
    To avoid reference issues, all values are deepcopied.

    Example:
    >>> brace_expand_keys({"layer.{1..3}": 0, "head": 1})
    {'layer.1': 0, 'layer.2': 0, 'layer.3': 0, 'head': 1}
    """
    if not isinstance(in_dict, dict):
        return in_dict
    out_dict = {}
    for k, v in in_dict.items():
        if "{" not in k:
            out_dict[k] = deepcopy(v)
            continue
        new_keys = braceexpand(k)
        for new_key in new_keys:
            out_dict[new_key] = deepcopy(v)
    return out_dict


def debug_print(*args, **kwargs):
    debug_flag = os.environ.get("TEXT_UTILS_DEBUG", "")
    if debug_flag == "" or debug_flag == "0" or debug_flag.lower() == "false":
        return
    print(*args, **kwargs, file=sys.stderr)
