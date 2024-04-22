import copy
import functools
import time
import os
import sys
import tempfile
from typing import Dict, Any, Optional, Tuple, List
from copy import deepcopy

import torch
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
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
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raise NotImplementedError

    def get_sharding_policy(self) -> ShardingPolicy | None:
        return None

    def compile(
        self,
        **kwargs: Any
    ) -> None:
        pass

    def distribute(
        self,
        devices: list[torch.device]
    ) -> "Model":
        assert len(devices) == 1, "only single device is supported"
        return self.to(devices[0])


QUANTIZATION_SCHEMES = [
    "w8a16",
    "w4a16"
]
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

        assert isinstance(self.model, PreTrainedModel)
        if kwargs.get("gradient_checkpointing", False):
            self.model.config.use_cache = False
            self.model.gradient_checkpointing_enable()

    def get_sharding_policy(self) -> ShardingPolicy | None:
        return functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                self.layer_cls
            }  # type: ignore
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        **_: Any
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        output = self.model(input_ids=token_ids)  # type: ignore
        return output.logits, {}

    def compile(
        self,
        **kwargs: Any
    ):
        self.model = torch.compile(
            self.model,
            **kwargs
        )  # type: ignore

    def decode(
        self,
        token_ids: torch.Tensor,
        lengths: torch.Tensor,
        kv_cache: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: bool = True
    ) -> Tuple[torch.Tensor, Tuple[Tuple[torch.Tensor]]]:
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

    def quantize(
        self,
        scheme: str,
        output_dir: str,
        examples: Optional[
            List[Dict[str, List[int] | torch.LongTensor]]
        ] = None,
        batch_size: int = 16,
        use_triton: bool = False,
        cache_on_gpu: bool = True,
        **kwargs: Any
    ) -> None:
        assert scheme in QUANTIZATION_SCHEMES, \
            f"unknown quantization scheme {scheme}, must be one of " \
            f"{QUANTIZATION_SCHEMES}"
        assert examples is not None

        if scheme == "w8a16":
            bits = 8
        elif scheme == "w4a16":
            bits = 4
        else:
            raise ValueError(f"unknown quantization scheme {scheme}")

        config = BaseQuantizeConfig(
            bits=bits,
            group_size=128,
            desc_act=False
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            self.model: PreTrainedModel
            self.model.save_pretrained(tmpdir)
            quant_model = AutoGPTQForCausalLM.from_pretrained(
                tmpdir,
                config,
            )
        quant_model.quantize(
            examples,
            batch_size,
            use_triton=use_triton,
            cache_examples_on_gpu=cache_on_gpu,
        )
        quant_model.save_quantized(output_dir)


def model_from_config(
    cfg: Dict[str, Any],
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
    elif model_type == "quantized_decoder":
        quant = AutoGPTQForCausalLM.from_quantized(
            cfg["path"],
            torch_dtype="auto"
        )
        assert isinstance(quant.model, PreTrainedModel)
        return PretrainedDecoder(quant.model)
    else:
        raise ValueError(f"unknown model type {model_type}")


def brace_expand_keys(in_dict: Dict[str, Any]):
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
