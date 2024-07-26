from typing import Any, Iterable, Iterator

import torch
from torch import nn

from text_utils import data, tokenization, grammar
from text_utils.api.processor import ModelInfo, TextProcessor
from text_utils.api.utils import (
    Device,
    device_info,
    get_devices,
)
from text_utils.inference import (
    utils as inference_utils,
    beam_search
)
from text_utils.inference.utils import Beam
from text_utils.constraints import Constraint

from llm_text_generation.model import (
    Model,
    PretrainedDecoder,
    model_from_config
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
    def _model_from_config(
        cls,
        cfg: dict[str, Any],
        device: Device
    ) -> nn.Module:
        return model_from_config(cfg["model"])

    @property
    def max_length(self) -> int:
        cfg_max_length = self.cfg["inference"].get("max_length", 512)
        return min(
            self._max_length or cfg_max_length,
            cfg_max_length
        )

    def __init__(
        self,
        model: Model,
        cfg: dict[str, Any],
        device: Device
    ) -> None:
        super().__init__(model, cfg, device)
        assert isinstance(model, PretrainedDecoder)
        self.logger.debug(f"got model config:\n{self.cfg['model']}")
        self.logger.info(
            f"running {self.name} text generator "
            f"on devices {[device_info(d) for d in self.devices]}"
        )
        self.tokenizer = tokenization.Tokenizer.from_config(
            self.cfg["inference"]["tokenizer"]
        )

        # some options for inference
        self._eos_token = self.cfg["inference"]["eos_token"]
        self._eos_token_id = self.tokenizer.token_to_id(
            self._eos_token
        )

        # continuations are the postprocessed tokens from the vocab
        # (already sorted by token id)
        self._continuations = self.tokenizer.get_continuations(initial=False)
        self._sampling_strategy = "greedy"
        self._beam_width = 1
        self._temp = 1.0
        self._top_k = 5
        self._use_cache = False
        self._full_outputs = False
        self._max_length = None
        self._constraint = None
        self._is_chat = self.cfg["inference"].get(
            "chat_template", None
        ) is not None

        self.model = self.model.compile(
            **self.cfg["inference"].get("compile", {})
        )

    def to(self, device: Device) -> "TextGenerator":
        self.devices = get_devices(device)
        if self.cfg["model"].get("device_map", None) is not None:
            return self
        assert isinstance(self.model, Model)
        self.model = self.model.distribute(self.devices)
        return self

    def _prepare_input(
        self,
        ipt: str | Chat | tuple[str | Chat, Const],
    ) -> data.InferenceData:
        info = {}
        if isinstance(ipt, tuple):
            ipt, constraint = ipt
            info["constraint"] = constraint

        if isinstance(ipt, str):
            ipt = [{"role": "user", "text": ipt}]

        template = self.cfg["inference"].get(
            "chat_template",
            {"roles": {}}
        )

        assert len(ipt) > 0, "expected non-empty chat"
        assert ipt[-1]["role"] == "user", "expected user to be last"
        # initialize prompt
        text = template.get("start", "")

        # add messages
        for message in ipt:
            role = message["role"]
            if role not in template["roles"]:
                text += message["text"]
            else:
                msg = template["roles"][role].replace(
                    "{text}",
                    message["text"]
                )
                text += msg

        # add end
        text += template.get("end", "")

        return data.InferenceData(text, info)

    @torch.inference_mode()
    def _live_inference(
        self,
        batch: data.InferenceBatch,
    ) -> Iterator[list[str]]:
        # decode fn gets in token ids and additional kwargs,
        # and return logits over next tokens and additional info
        def _decode_fn(
            token_ids: torch.Tensor,
            **kwargs: Any
        ) -> tuple[torch.Tensor, dict[str, Any]]:
            assert isinstance(self.model, PretrainedDecoder)
            dec, cache = self.model.decode(
                token_ids,
                kwargs["lengths"],
                kwargs.get("kv_cache", None),
                self._use_cache
            )
            return dec, {"kv_cache": cache}

        def _kwargs_update_fn(
            kwargs: dict[str, Any],
            info: dict[str, Any],
            mask: torch.Tensor
        ) -> None:
            kv_cache = info.get("kv_cache", None)
            if kv_cache is None:
                return
            kwargs["kv_cache"] = tuple(
                tuple(c[mask.to(c.device)] for c in cache)
                for cache in info["kv_cache"]
            )

        logit_fns = []
        initial_beams = []
        initial_lengths = []
        for token_ids, info in zip(batch.token_ids(), batch.infos()):
            beam = Beam(token_ids, [0.0] * len(token_ids))
            initial_lengths.append(0 if self._full_outputs else len(token_ids))

            if "constraint" in info:
                beam.info["constraint"] = self._get_constraint(
                    info["constraint"]
                )
            elif self._constraint is not None:
                constraint = self._constraint.clone()
                constraint.reset()
                beam.info["constraint"] = constraint

            initial_beams.append(beam)

        # add constrain logit fn if any of the beams have a constraint
        if any(
            "constraint" in beam.info
            for beam in initial_beams
        ):
            logit_fns.append(inference_utils.constraint_logit_fn(
                lambda beam: beam.info.get(
                    "constraint", None
                ) if isinstance(beam, Beam) else None,
                self._eos_token_id
            ))

            def update_beam(
                beam: Beam,
                token_id: int,
                log_p: float
            ) -> Beam | None:
                beam = Beam.from_beam(beam, token_id, log_p)
                beam_const = beam.info.get("constraint", None)
                if token_id == self._eos_token_id or beam_const is None:
                    return beam
                elif beam_const.is_invalid():
                    return None

                beam_const = beam_const.clone()
                beam_const.next(token_id)
                beam.info["constraint"] = beam_const
                return beam

            candidate_fn = update_beam
        else:
            candidate_fn = inference_utils.default_beam_candidate_fn()

        if self._sampling_strategy == "greedy":
            sample_fn = inference_utils.greedy()
        elif self._sampling_strategy == "top_k":
            assert self._top_k >= self._beam_width, \
                "top k must be greater than or equal to beam width"
            logit_fns.append(inference_utils.top_k_masking(self._top_k))
            sample_fn = inference_utils.sample()
        else:
            logit_fns.append(inference_utils.nucleus_masking(self._top_p))
            sample_fn = inference_utils.sample()

        if self._sampling_strategy != "greedy" and self._temp != 1.0:
            logit_fns.append(inference_utils.temperature_scaling(
                self._temp
            ))

        def beam_stop_fn(beam: Beam) -> bool:
            return beam.token_ids[-1] == self._eos_token_id

        for output in beam_search(
            decode_fn=_decode_fn,
            initial=initial_beams,
            pad_token_id=self.tokenizer.pad_token_id(),
            max_length=self.max_length,
            stop_fn=beam_stop_fn,
            device=self.devices[0],
            normalize_by_length=True,
            alpha=1.0,
            beam_width=self._beam_width,
            sample_fn=sample_fn,
            candidate_fn=candidate_fn,
            logit_fns=logit_fns,
            kwargs_update_fn=_kwargs_update_fn,
            return_incomplete=True,
            yield_intermediate=True
        ):
            yield [
                self.tokenizer.de_tokenize(beams[0].token_ids[length:])
                for beams, length in zip(output, initial_lengths)
            ]

    def _get_constraint(
        self,
        constraint: Const
    ) -> Constraint:
        if isinstance(constraint, str):
            return grammar.RegexConstraint(
                constraint,
                self._continuations
            )
        else:
            gram, lexer, exact = constraint
            return grammar.LR1Constraint(
                gram,
                lexer,
                self._continuations,
                exact
            )

    def set_inference_options(
        self,
        sampling_strategy: str = "greedy",
        temperature: float = 1.0,
        top_k: int = 10,
        top_p: float = 0.95,
        beam_width: int = 1,
        constraint: Const | None = None,
        max_length: int | None = None,
        use_cache: bool = False,
        full_outputs: bool = False
    ) -> None:
        assert sampling_strategy in ["greedy", "top_k", "top_p"]
        self._sampling_strategy = sampling_strategy
        self._beam_width = beam_width
        self._temp = temperature
        self._top_k = top_k
        self._top_p = top_p
        if constraint is not None:
            self._constraint = self._get_constraint(constraint)
        else:
            self._constraint = None
        self._max_length = max_length
        self._use_cache = use_cache
        self._full_outputs = full_outputs

    def generate_live(
        self,
        ipt: str | Chat | tuple[str | Chat, Const],
    ) -> Iterator[str]:
        input = self._prepare_input(ipt)
        batch = next(data.InferenceLoader.from_iterator(
            iter([input]),
            self.cfg["inference"]["tokenizer"],
            self.cfg["inference"].get("window", {"type": "full"}),
            ignore_special_tokens=self._is_chat
        ))

        yield from (outputs[0] for outputs in self._live_inference(batch))

    def generate(
        self,
        inputs: Iterable[str | Chat | tuple[str | Chat, Const]],
        batch_size: int = 16,
        batch_max_tokens: int | None = None,
        sort: bool = True,
        num_threads: int | None = None,
        show_progress: bool = False,
    ) -> Iterator[str]:
        def inference_fn(
            batch: data.InferenceBatch
        ) -> list[str]:
            *_, last = self._live_inference(batch)
            return last

        def postprocessing_fn(
            items: list[data.InferenceItem],
            outputs: list[str]
        ) -> str:
            assert len(items) == 1 and len(outputs) == 1
            return outputs[0]

        yield from self._process(
            (self._prepare_input(ipt) for ipt in inputs),
            inference_fn,
            postprocessing_fn,
            "Generating text",
            batch_size,
            batch_max_tokens,
            sort,
            num_threads,
            show_progress=show_progress,
            ignore_special_tokens=self._is_chat
        )
