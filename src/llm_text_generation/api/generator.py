from typing import Any, Iterable, Iterator

import torch
from grammar_utils.constrain import Constraint, LR1Constraint, RegexConstraint
from text_utils import data, tokenization
from text_utils.api.processor import ModelInfo, TextProcessor
from text_utils.api.utils import (
    Device,
    device_info,
    get_devices,
)
from text_utils.inference import beam_search
from text_utils.inference import utils as inference_utils
from text_utils.inference.utils import Beam
from torch import nn

from llm_text_generation.api.utils import format_chat
from llm_text_generation.model import (
    Model,
    PretrainedDecoder,
    model_from_config,
    peft_model_from_config,
)

_BASE_URL = ""
_NAME_TO_ZIP = {}

Chat = list[dict[str, str]]
Const = str | tuple[str, str, bool]


class TextGenerator(TextProcessor):
    task = "Text Generation"

    @classmethod
    def available_models(cls) -> list[ModelInfo]:
        return []

    @classmethod
    def _model_url(cls, model: str) -> str:
        return f"{_BASE_URL}/{_NAME_TO_ZIP[model]}"

    @property
    def name(self) -> str:
        return self.cfg["experiment"]["name"]

    @classmethod
    def _model_from_config(cls, cfg: dict[str, Any], device: Device) -> nn.Module:
        model = model_from_config(cfg["model"])
        peft = cfg.get("train", {}).get("peft", None)
        if peft is not None:
            model = peft_model_from_config(model, peft)
        return model

    @property
    def max_length(self) -> int:
        cfg_max_length = self.cfg["inference"].get("max_length", 512)
        return min(self._max_length or cfg_max_length, cfg_max_length)

    def __init__(self, model: Model, cfg: dict[str, Any], device: Device) -> None:
        super().__init__(model, cfg, device)
        assert isinstance(model, PretrainedDecoder), "only decoder models are supported"

        self.logger.debug(f"Got model config:\n{self.cfg['model']}")
        self.logger.info(
            f"Running {self.name} text generator "
            f"on devices {[device_info(d) for d in self.devices]}"
        )
        self.tokenizer = tokenization.Tokenizer.from_config(
            self.cfg["inference"]["tokenizer"]
        )

        # continuations are the postprocessed tokens from the vocab
        # (already sorted by token id)
        self._continuations = self.tokenizer.get_continuations(initial=False)

        # some options for inference
        self._eos_token = self.cfg["inference"]["eos"]
        self._eos_token_id = self.tokenizer.token_to_id(self._eos_token)
        self._sample = False
        self._beam_width = 1
        self._temp: int | None = None
        self._top_k: int | None = None
        self._top_p: int | None = None
        self._min_p: int | None = None
        self._stop_condition = "estimated_score"
        self._repeat_penalty: float | None = None

        self._use_cache = False
        self._full_outputs = False
        self._max_length = None
        self._max_new_tokens = None
        self._constraint = None
        self._is_chat = self.cfg["inference"].get("chat_template", None) is not None

        self.model = self.model.compile(**self.cfg["inference"].get("compile", {}))

    def to(self, device: Device) -> "TextGenerator":
        self.devices = get_devices(device)
        if self.cfg["model"].get("device_map", None) is not None:
            return self
        assert isinstance(self.model, Model)
        self.model = self.model.distribute(self.devices)
        return self

    def _prepare_input(
        self,
        input: str | Chat,
        constraint: Const | None = None,
    ) -> tuple[str, dict[str, Any]]:
        info = {"const": constraint}

        template = self.cfg["inference"].get(
            "chat_template",
            {"roles": {"system": "{text}", "user": "{text}", "assistant": "{text}"}},
        )

        prompt = format_chat(input, template)
        return prompt, info

    @torch.inference_mode()
    def _live_inference(
        self, batch: data.InferenceBatch, infos: list[dict]
    ) -> Iterator[list[str]]:
        # decode fn gets in token ids and additional kwargs,
        # and return logits over next tokens and additional info
        def _decode_fn(
            token_ids: torch.Tensor, **kwargs: Any
        ) -> tuple[torch.Tensor, dict[str, Any]]:
            assert isinstance(self.model, PretrainedDecoder)
            dec, cache = self.model.decode(
                token_ids,
                kwargs["lengths"],
                kwargs.get("kv_cache", None),
                self._use_cache,
            )
            return dec, {"kv_cache": cache}

        def _kwargs_update_fn(
            kwargs: dict[str, Any], info: dict[str, Any], mask: torch.Tensor
        ) -> None:
            kv_cache = info.get("kv_cache", None)
            if kv_cache is None:
                return
            kwargs["kv_cache"] = tuple(
                tuple(c[mask.to(c.device)] for c in cache) for cache in info["kv_cache"]
            )

        initial_beams = []
        for token_ids, (index, _) in zip(batch.token_ids(), batch.indices()):
            beam_info = {}
            if infos[index].get("const"):
                beam_info["const"] = self._get_constraint(infos[index]["const"])
            elif self._constraint is not None:
                constraint = self._constraint.clone()
                constraint.reset()
                beam_info["const"] = constraint

            beam = Beam(token_ids, info=beam_info)
            initial_beams.append(beam)

        logit_fns = [
            inference_utils.constraint_logit_fn(
                lambda beam: beam.info.get("const", None), self._eos_token_id
            )
        ]

        def stop_fn(beam: Beam) -> bool:
            return beam.token_ids[-1] == self._eos_token_id

        def update_fn(beam: Beam) -> Beam | None:
            const = beam.info.get("const", None)
            if const is None:
                return beam
            elif const.is_invalid():
                return None

            const = const.clone()
            const.next(beam.token_ids[-1])
            beam.info["const"] = const
            return beam

        sample_fn = (
            inference_utils.sample() if self._sample else inference_utils.greedy()
        )

        keep_min = 2 if self._beam_width > 1 else 1

        if self._repeat_penalty is not None:
            logit_fns.append(inference_utils.repeat_penalty(self._repeat_penalty))

        if self._sample and self._temp is not None:
            logit_fns.append(inference_utils.temperature_scaling(self._temp))

        if self._sample and self._top_k is not None:
            assert (
                self._top_k > self._beam_width
            ), "top k must be greater than beam width"
            logit_fns.append(inference_utils.top_k_masking(self._top_k))

        if self._sample and self._top_p is not None:
            logit_fns.append(inference_utils.nucleus_masking(self._top_p, keep_min))

        if self._sample and self._min_p is not None:
            logit_fns.append(inference_utils.min_p_masking(self._min_p, keep_min))

        for output in beam_search(
            decode_fn=_decode_fn,
            initial=initial_beams,
            pad_token_id=self.tokenizer.pad_token_id(),
            max_length=self.max_length,
            stop_fn=stop_fn,
            device=self.devices[0],
            beam_width=self._beam_width,
            sample_fn=sample_fn,
            update_fn=update_fn,
            logit_fns=logit_fns,
            kwargs_update_fn=_kwargs_update_fn,
            stop_condition=self._stop_condition,
            max_new_tokens=self._max_new_tokens,
            return_incomplete=True,
            yield_intermediate=True,
        ):
            yield [
                self.tokenizer.de_tokenize(
                    beams[0].token_ids
                    if self._full_outputs
                    else beams[0].decoded_token_ids
                )
                for beams in output
            ]

    def _get_constraint(self, constraint: Const) -> Constraint:
        if isinstance(constraint, str):
            return RegexConstraint(constraint, self._continuations)
        else:
            gram, lexer, exact = constraint
            return LR1Constraint(gram, lexer, self._continuations, exact)

    def set_inference_options(
        self,
        sample: bool = False,
        repeat_penalty: float | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        min_p: float | None = None,
        beam_width: int = 1,
        stop_condition: str = "estimated_score",
        constraint: Const | None = None,
        max_length: int | None = None,
        max_new_tokens: int | None = None,
        use_cache: bool = False,
        full_outputs: bool = False,
    ) -> None:
        self._sample = sample
        self._repeat_penalty = repeat_penalty
        self._temp = temperature
        self._top_k = top_k
        self._top_p = top_p
        self._min_p = min_p
        self._beam_width = beam_width
        self._stop_condition = stop_condition

        if constraint is not None:
            self._constraint = self._get_constraint(constraint)
        else:
            self._constraint = None

        self._max_length = max_length
        self._max_new_tokens = max_new_tokens
        self._use_cache = use_cache
        self._full_outputs = full_outputs

    def generate_live(
        self,
        input: str | Chat,
        constraint: Const | None = None,
    ) -> Iterator[str]:
        input, info = self._prepare_input(input, constraint)

        batch = next(
            data.InferenceLoader.from_iterator(
                iter([input]),
                self.cfg["inference"]["tokenizer"],
                self.cfg["inference"].get("window", {"type": "full"}),
                ignore_special_tokens=self._is_chat,
                num_threads=0,
            )
        )

        yield from (outputs[0] for outputs in self._live_inference(batch, [info]))

    def generate(
        self,
        inputs: Iterable[str | Chat | tuple[str | Chat, Const | None]],
        batch_size: int = 16,
        batch_max_tokens: int | None = None,
        sort: bool = True,
        num_threads: int | None = None,
        show_progress: bool = False,
    ) -> Iterator[str]:
        inputs, infos = zip(
            *(self._prepare_input(text, constraint) for text, constraint in inputs)
        )

        def inference_fn(batch: data.InferenceBatch) -> list[str]:
            *_, last = self._live_inference(batch, infos)
            return last

        def postprocessing_fn(
            items: list[data.InferenceItem], outputs: list[str]
        ) -> str:
            assert len(items) == 1 and len(outputs) == 1
            return outputs[0]

        yield from self._process(
            iter(inputs),
            inference_fn,
            postprocessing_fn,
            "Generating text",
            batch_size,
            batch_max_tokens,
            sort,
            num_threads=num_threads,
            progress_total=len(inputs),
            show_progress=show_progress,
            ignore_special_tokens=self._is_chat,
        )
