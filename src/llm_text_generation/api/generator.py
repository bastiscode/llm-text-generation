from io import TextIOWrapper
import os
import json
import sys
from typing import Any, Iterator

import torch
from torch import nn
from peft import get_peft_model

from text_utils import data, tokenization, grammar
from text_utils.api.processor import ModelInfo, TextProcessor
from text_utils.api.utils import (
    Device,
    device_info,
    get_devices,
    get_peft_config
)
from text_utils.inference import (
    utils as inference_utils,
    search,
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
        _: Device
    ) -> nn.Module:
        model = model_from_config(cfg["model"])
        peft = cfg["train"].get("peft", None)
        if peft is not None:
            peft_cfg = get_peft_config(peft)
            model.model = get_peft_model(
                model.model,  # type: ignore
                peft_cfg
            )
        return model

    @property
    def max_length(self) -> int:
        cfg_max_length = self.cfg["train"]["data"].get("max_length", 512)
        return min(
            self._max_length or cfg_max_length,
            cfg_max_length
        )

    @property
    def context_length(self) -> int:
        raise NotImplementedError

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
            self.cfg["input_tokenizer"]
        )

        # some options for inference
        self._eos_token = self.cfg["input_tokenizer"]["eos_token"]
        self._eos_token_id = self.tokenizer.special_token_to_id(
            self._eos_token
        )

        # continuations are the tokens from the vocab
        # (already sorted by token id)
        self._continuations = self.tokenizer.get_vocab()
        self._sampling_strategy = "greedy"
        self._beam_width = 1
        self._temp = 1.0
        self._top_k = 5
        self._use_cache = True
        self._full_outputs = False
        self._max_length = None
        self._constraint = None

        self.model = self.model.compile(**self.cfg.get("compile", {}))

    def to(self, device: Device) -> "TextGenerator":
        self.devices = get_devices(device)
        if self.cfg["model"].get("device_map", None) is not None:
            return self
        assert isinstance(self.model, Model)
        self.model = self.model.distribute(self.devices)
        return self

    def _build_inference_loader_config(self) -> dict[str, Any]:
        return {
            "tokenizer_config": self.cfg["input_tokenizer"],
            "window_config": {"type": "full"},
        }

    def _prepare_batch(self, batch: data.InferenceBatch) -> dict[str, Any]:
        token_ids_np, _, lengths, *_ = batch.tensors()
        return {
            "token_ids": token_ids_np,
            "lengths": lengths,
            "infos": batch.infos()
        }

    def _prepare_input(
        self,
        ipt: str | Chat | tuple[str | Chat, Const],
        json_decode: bool = False
    ) -> data.InferenceData:
        info = {}
        if isinstance(ipt, tuple):
            ipt, constraint = ipt
            info["constraint"] = constraint

        if json_decode:
            assert isinstance(ipt, str)
            ipt = json.loads(ipt)
            assert isinstance(ipt, (str, list)), \
                "expected text or chat as json encoded string"

        if isinstance(ipt, str):
            ipt = [{"role": "user", "text": ipt}]

        template = self.cfg.get("chat_template", {})

        assert len(ipt) > 0, "expected non-empty chat"
        assert ipt[-1]["role"] == "user", "expected user to be last"
        # initialize prompt
        text = template.get("start", "")

        # add messages
        for message in ipt:
            role = message["role"]
            if role not in template:
                text += message["text"]
            else:
                msg = template[role].replace("{text}", message["text"])
                text += msg

        # add end
        text += template.get("end", "")

        return data.InferenceData(text, info)

    @torch.inference_mode()
    def _inference(
        self,
        inputs: dict[str, Any],
        yield_intermediate: bool = False
    ) -> Iterator[Any]:
        initial_token_ids = [
            list(token_ids[:length])
            for token_ids, length in zip(
                inputs["token_ids"],
                inputs["lengths"]
            )
        ]

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

        if (self._beam_width or 1) > 1:
            assert self._beam_width is not None
            logit_fns = []
            initial_beams = []
            for token_ids, info in zip(initial_token_ids, inputs["infos"]):
                beam = Beam(token_ids, [0.0] * len(token_ids))

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

                def _update_beam(beam: Beam, token_id: int, log_p: float):
                    new_beam = Beam.from_beam(beam, token_id, log_p)
                    beam_const = beam.info.get("constraint", None)
                    if token_id == self._eos_token_id or beam_const is None:
                        return new_beam

                    beam_const = beam_const.clone()
                    beam_const.next(token_id)
                    new_beam.info["constraint"] = beam_const
                    return new_beam

                candidate_fn = _update_beam
            else:
                candidate_fn = inference_utils.default_beam_candidate_fn()

            if self._sampling_strategy == "greedy":
                sample_fn = inference_utils.beam_greedy()
            elif self._sampling_strategy == "top_k":
                assert self._top_k >= self._beam_width, \
                    "top k must be greater than or equal to beam width"
                logit_fns.append(inference_utils.top_k_masking(self._top_k))
                sample_fn = inference_utils.beam_sample()
            else:
                logit_fns.append(inference_utils.nucleus_masking(self._top_p))
                sample_fn = inference_utils.beam_sample()

            if self._sampling_strategy != "greedy" and self._temp != 1.0:
                logit_fns.append(inference_utils.temperature_scaling(
                    self._temp
                ))

            def beam_stop_fn(beam: Beam) -> bool:
                return beam.token_ids[-1] == self._eos_token_id

            yield from (
                [beam[0].token_ids for beam in beams]
                for beams in
                beam_search(
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
                    yield_intermediate=yield_intermediate
                )
            )
            return

        logit_fns = []
        constraints: dict[int, Constraint] = {}
        for i, info in enumerate(inputs["infos"]):
            if "constraint" in info:
                constraints[i] = self._get_constraint(info["constraint"])
            elif self._constraint is not None:
                constraint = self._constraint.clone()
                constraint.reset()
                constraints[i] = constraint

        # add constrain logit fn if any of the batch elements
        # has a constraint
        if len(constraints) > 0:
            logit_fns.append(inference_utils.constraint_logit_fn(
                lambda idx: constraints.get(
                    idx, None
                ) if isinstance(idx, int) else None,
                self._eos_token_id
            ))

        if self._sampling_strategy == "greedy":
            sample_fn = inference_utils.greedy()
        elif self._sampling_strategy == "top_k":
            logit_fns.append(inference_utils.top_k_masking(self._top_k))
            sample_fn = inference_utils.sample()
        else:
            logit_fns.append(inference_utils.nucleus_masking(self._top_p))
            sample_fn = inference_utils.sample()

        if self._sampling_strategy != "greedy" and self._temp != 1.0:
            logit_fns.append(inference_utils.temperature_scaling(
                self._temp
            ))

        # if there are constraints we need to update them
        # after sampling a token
        if len(constraints) > 0:
            sample_fn = inference_utils.constraint_sample_fn(
                lambda idx: constraints.get(idx, None),
                sample_fn,
                self._eos_token_id
            )

        def stop_fn(token_ids: torch.Tensor, _: list[int]) -> torch.Tensor:
            return token_ids == self._eos_token_id

        yield from search(
            decode_fn=_decode_fn,
            initial_token_ids=initial_token_ids,
            pad_token_id=self.tokenizer.pad_token_id(),
            max_length=self.max_length,
            sample_fn=sample_fn,
            logit_fns=logit_fns,
            stop_fn=stop_fn,
            device=self.devices[0],
            kwargs_update_fn=_kwargs_update_fn,
            yield_intermediate=yield_intermediate
        )

    def _process_results(
        self,
        items: list[data.InferenceItem],
        outputs: list[Any],
    ) -> data.InferenceData:
        assert len(outputs) == 1, "expected single output"
        output = outputs[0]
        item = items[0]

        text = self.tokenizer.de_tokenize(
            item.tokenization.token_ids + output
        )
        if not self._full_outputs:
            input_text = self.tokenizer.de_tokenize(
                item.tokenization.token_ids
            )
            text = text[len(input_text):]
        return data.InferenceData(text, item.data.info)

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
        beam_width: int | None = None,
        constraint: Const | None = None,
        max_length: int | None = None,
        use_cache: bool = True,
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

    def generate(
        self,
        inputs: list[str | Chat | tuple[str | Chat, Const]],
        batch_size: int = 16,
        batch_max_tokens: int | None = None,
        sort: bool = True,
        num_threads: int | None = None,
        show_progress: bool = False,
    ) -> str | list[str]:
        loader = self._get_loader(
            (
                self._prepare_input(ipt)
                for ipt in inputs
            ),
            batch_size,
            batch_max_tokens,
            sort,
            num_threads,
        )

        progress_desc = f"Generating text from " \
            f"{len(inputs)} sequences"
        progress_total = len(inputs)
        progress_unit = "seq"

        if sort:
            outputs = self._process_sorted(
                loader,
                progress_desc,
                progress_total,
                progress_unit,
                show_progress
            )
        else:
            outputs = self._process_unsorted(
                loader,
                progress_desc,
                progress_total,
                progress_unit,
                show_progress
            )

        return [
            output.text
            for output in outputs
        ]

    def generate_live(
        self,
        ipt: str | Chat | tuple[str | Chat, Const],
    ) -> Iterator[str]:
        batch = next(self._get_loader(
            iter([self._prepare_input(ipt)]),
            1,
        ))
        inputs = self._prepare_batch(batch)
        items = batch.items()
        for outputs in self._inference(inputs, yield_intermediate=True):
            yield self._process_results(items, outputs).text

    def generate_iter(
        self,
        iter: Iterator[str | Chat | tuple[str | Chat, Const]],
        batch_size: int = 16,
        batch_max_tokens: int | None = None,
        sort: bool = True,
        num_threads: int | None = None,
        show_progress: bool = False,
    ) -> Iterator[str]:
        loader = self._get_loader(
            (
                self._prepare_input(ipt)
                for ipt in iter
            ),
            batch_size,
            batch_max_tokens,
            sort,
            num_threads,
        )

        progress_desc = "Generating text from iterator"
        progress_total = sys.maxsize
        progress_unit = "byte"

        if sort:
            outputs = self._process_sorted(
                loader,
                progress_desc,
                progress_total,
                progress_unit,
                show_progress
            )
        else:
            outputs = self._process_unsorted(
                loader,
                progress_desc,
                progress_total,
                progress_unit,
                show_progress
            )

        return (output.text for output in outputs)

    def generate_file(
        self,
        input_file: str,
        output_file: TextIOWrapper | str | None = None,
        batch_size: int = 16,
        batch_max_tokens: int | None = None,
        sort: bool = True,
        num_threads: int | None = None,
        show_progress: bool = False,
        format: str = "jsonl"
    ) -> Iterator[str] | None:
        assert format in ["jsonl", "lines", "text"], \
            f"invalid format {format}, must be jsonl, lines, or text"

        if format == "lines" or format == "jsonl":
            inputs = (
                self._prepare_input(line.rstrip("\r\n"), format == "jsonl")
                for line in open(input_file, "r")
            )
        else:
            inputs = iter([self._prepare_input(open(input_file, "r").read())])

        loader = self._get_loader(
            inputs,
            batch_size,
            batch_max_tokens,
            sort,
            num_threads,
        )

        file_name = input_file \
            if len(input_file) < 32 else f"...{input_file[-29:]}"
        progress_desc = f"Generating text from {file_name}"
        progress_total = os.path.getsize(input_file)
        progress_unit = "byte"

        if sort:
            outputs = iter(self._process_sorted(
                loader,
                progress_desc,
                progress_total,
                progress_unit,
                show_progress
            ))
        else:
            outputs = self._process_unsorted(
                loader,
                progress_desc,
                progress_total,
                progress_unit,
                show_progress
            )

        if output_file is not None:
            output_file_is_str = isinstance(output_file, str)
            if output_file_is_str:
                output_dir = os.path.dirname(output_file)
                if output_dir != "":
                    os.makedirs(output_dir, exist_ok=True)
                output_file = open(output_file, "w", encoding="utf8")

            for output in outputs:
                text = output.text
                if format == "jsonl":
                    text = json.dumps(text)
                elif format == "lines" and "\n" in text:
                    raise ValueError(
                        "output contains newline, "
                        "lines format is not supported in this case"
                    )
                output_file.write(text + "\n")

            if output_file_is_str:
                output_file.close()

        else:
            return (output.text for output in outputs)
