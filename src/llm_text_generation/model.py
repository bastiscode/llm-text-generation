import copy
import functools
import os
import re
import sys
import time
from copy import deepcopy
from typing import Any, Optional

import torch
from braceexpand import braceexpand
from peft.mapping import get_peft_model
from peft.peft_model import PeftModel
from peft.tuners.lora import LoraConfig
from text_utils.api import utils
from text_utils.api.trainer import ShardingPolicy
from torch import nn
from torch.distributed.fsdp.wrap import (
    _or_policy,
    lambda_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
from torch.utils.hooks import RemovableHandle
from transformers import AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithCrossAttentions,
    CausalLMOutputWithPast,
    MoeCausalLMOutputWithPast,
)
from transformers.utils import logging as hf_logging

hf_logging.disable_progress_bar()


def _register_hook(
    hooks: list[RemovableHandle],
    m: nn.Module,
    device: torch.device,
):
    m = m.to(device)

    def _pre_hook(m: nn.Module, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
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
        self, token_ids: torch.Tensor, **kwargs: Any
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        raise NotImplementedError

    def get_sharding_policy(self) -> ShardingPolicy | None:
        return None

    def enable_gradient_checkpointing(self) -> None:
        raise NotImplementedError

    def compile(self, **kwargs: Any) -> "Model":
        return self

    def distribute(self, devices: list[torch.device]) -> "Model":
        assert len(devices) == 1, "only single device is supported"
        return self.to(devices[0])


class PretrainedDecoderForClassification(Model):
    def __init__(
        self,
        model: str | PreTrainedModel,
        output_layer_name: str,
        classes: list[str],
        output_position: str | int = "last",
        **kwargs: Any,
    ):
        super().__init__()
        self.model = PretrainedDecoder(model, **kwargs)
        # replace final layer with a linear layer
        assert hasattr(
            self.model, output_layer_name
        ), f"output layer {output_layer_name} not found in model"
        layer = getattr(self.model, output_layer_name)
        assert isinstance(
            layer, nn.Linear
        ), f"output layer {output_layer_name} is not a linear layer"
        setattr(
            self.model,
            output_layer_name,
            nn.Linear(
                layer.in_features,
                len(classes),
                device=layer.weight.device,
            ),
        )
        self.classes = classes
        self.idx_to_class = {i: c for i, c in enumerate(classes)}
        self.output_position = output_position

    def forward(
        self,
        token_ids: torch.Tensor,
        lengths: torch.Tensor | None = None,
        **_: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        assert lengths is not None, "lengths must be provided"
        logits, _ = self.model(token_ids)
        assert torch.all(lengths > 0), "lengths must be greater than 0"
        if self.output_position == "first":
            logits = logits[:, 0]
        elif self.output_position == "last":
            logits = logits[torch.arange(len(logits)), lengths - 1]
        elif isinstance(self.output_position, int):
            assert torch.all(
                lengths > self.output_position
            ), "all lengths must be greater than output_position"
            logits = logits[:, self.output_position]
        else:
            raise ValueError(f"unknown output_position {self.output_position}")

        return logits, {}

    def classify(
        self, token_ids: torch.Tensor, lengths: torch.Tensor, top_k: int = 1
    ) -> list[list[tuple[str, float]]]:
        logits, _ = self(token_ids, lengths)

        probs = torch.softmax(logits, dim=1)
        top = torch.topk(probs, min(top_k, len(self.classes)), dim=1)

        outputs = []
        for indices, values in zip(top.indices.tolist(), top.values.tolist()):
            classes_and_probs = [
                (self.idx_to_class[i], v) for i, v in zip(indices, values)
            ]
            outputs.append(classes_and_probs)

        return outputs

    def enable_gradient_checkpointing(self) -> None:
        self.model.enable_gradient_checkpointing()

    def compile(self, **cfg: Any) -> "PretrainedDecoderForClassification":
        self.model.compile(**cfg)
        return self

    def distribute(self, devices: list[torch.device]) -> "Model":
        self.model = self.model.distribute(devices)
        return self


class PretrainedDecoder(Model):
    model: PreTrainedModel | PeftModel
    layer_module: str | None = None
    other_modules: dict[str, int] | None = None

    def __init__(self, model: str, **kwargs: Any):
        super().__init__()
        if kwargs.get("device_map") is not None:
            kwargs["device_map"] = brace_expand_keys(kwargs["device_map"])

        # set layer modules and other modules
        # for distributing model across devices
        # (or use device_map auto, the default)
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=kwargs.pop("torch_dtype", "auto"),
            **kwargs,
        )

    def get_sharding_policy(self) -> ShardingPolicy | None:
        policies = []
        if isinstance(self.model, PeftModel):

            def find_peft_modules(
                m: nn.Module, peft_modules: set[nn.Module]
            ) -> set[nn.Module]:
                # if module has no children, all parameters trainable
                # and at least one parameter, we wrap it
                if (
                    next(m.children(), None) is None
                    and all(p.requires_grad for p in m.parameters())
                    and next(m.parameters(), None) is not None
                ):
                    peft_modules.add(m)
                else:
                    for module in m.children():
                        find_peft_modules(module, peft_modules)
                return peft_modules

            peft_modules = find_peft_modules(self.model, set())

            policies.append(
                functools.partial(
                    lambda_auto_wrap_policy, lambda_fn=lambda m: m in peft_modules
                )
            )

        policies.append(
            functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={self.layer_cls},  # type: ignore
            )
        )
        return functools.partial(_or_policy, policies=policies)

    def enable_gradient_checkpointing(self) -> None:
        assert isinstance(self.model, PreTrainedModel)
        self.model.config.use_cache = False
        self.model.gradient_checkpointing_enable()

    def forward(
        self, token_ids: torch.Tensor, **_: Any
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        output = self.model(input_ids=token_ids)  # type: ignore
        return output.logits, {}

    def compile(self, **cfg: Any) -> "PretrainedDecoder":
        cfg = copy.deepcopy(cfg)
        typ = cfg.pop("type", None)
        if typ == "torch":
            self.model = torch.compile(self.model, **cfg)  # type: ignore
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
                inputs=[
                    trt.Input(
                        min_shape=(1, 1),
                        opt_shape=(max_bs // 2, max_len // 2),
                        max_shape=(max_bs, max_len),
                        dtype=torch.long,
                        name=name,
                    )
                ],
                enabled_precisions={next(self.model.parameters()).dtype},
                **cfg,
            )  # type: ignore
            if path is not None:
                torch.jit.save(self.model, path)

        return self

    def decode(
        self,
        token_ids: torch.Tensor,
        lengths: torch.Tensor,
        kv_cache: Optional[tuple[tuple[torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor]]]:
        if use_cache and kv_cache is not None:
            b, s = token_ids.shape
            assert token_ids.ndim == 2 and s > 0
            token_ids = token_ids[
                torch.arange(b, device=token_ids.device), lengths - 1, None
            ]
        start = time.perf_counter()
        output = self.model(  # type: ignore
            input_ids=token_ids, past_key_values=kv_cache, use_cache=use_cache
        )
        end = time.perf_counter()
        # time in ms
        debug_print(f"decode time: {(end - start) * 1000:.2f} ms ")
        assert isinstance(
            output,
            (
                BaseModelOutputWithPast,
                CausalLMOutputWithPast,
                CausalLMOutputWithCrossAttentions,
                MoeCausalLMOutputWithPast,
            ),
        ), f"unexpected output type {type(output)}"
        return output.logits, output.past_key_values  # type: ignore

    def distribute(self, devices: list[torch.device]) -> "Model":
        for hook in self.hooks:
            hook.remove()

        self.hooks = []
        assert len(devices) > 0
        if len(devices) == 1:
            return self.to(devices[0])

        assert self.layer_module is not None and self.other_modules is not None, (
            "layer modules not set, specify it or use device_map auto "
            "when using Huggingface models"
        )

        if isinstance(self.model, PeftModel):
            # unwrap peft model
            model = self.model.base_model.model
        else:
            model = self.model

        layer_pattern = re.compile(self.layer_module)
        # distribute the layers
        layers = [m for n, m in model.named_modules() if layer_pattern.match(n)]
        assert len(layers) > 0 and len(devices) <= len(
            layers
        ), f"{len(devices)} devices for {len(layers)} layers not supported"

        # distribute evenly among devices
        layers_per_device = [len(layers) // len(devices)] * len(devices)
        for i in range(len(layers) % len(devices)):
            layers_per_device[i] += 1

        device_idx = 0
        for i, m in enumerate(layers):
            _register_hook(self.hooks, m, devices[device_idx])
            if i + 1 == sum(layers_per_device[: device_idx + 1]):
                device_idx += 1

        assert device_idx == len(devices)
        # add additional hooks for modules outside the regular
        # layers
        for n, m in model.named_modules():
            for pattern, device_idx in self.other_modules.items():
                if re.match(pattern, n):
                    _register_hook(self.hooks, m, devices[device_idx])
                    break

        return self


def peft_model_from_config(model: Model, cfg: dict[str, Any]) -> Model:
    peft = copy.deepcopy(cfg)
    typ = peft.pop("type")
    if typ == "lora":
        peft_cfg = LoraConfig(**peft)
    else:
        raise ValueError(f"unknown peft type: {typ}")

    assert isinstance(model.model, PreTrainedModel)
    model.model = get_peft_model(model.model, peft_cfg)
    return model


def model_from_config(
    cfg: dict[str, Any],
) -> Model:
    cfg = copy.deepcopy(cfg)
    model_type = cfg.pop("type")

    if model_type == "pretrained_decoder":
        return PretrainedDecoder(**cfg)
    elif model_type == "pretrained_decoder_for_classification":
        return PretrainedDecoderForClassification(**cfg)
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
