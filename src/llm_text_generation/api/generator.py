from io import TextIOWrapper
import os
import sys
from typing import Any, Dict, List, Tuple, Optional, Union, Iterator

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

from llm_text_generation.model import (
    Model,
    PretrainedDecoder,
    model_from_config
)

_BASE_URL = ""
_NAME_TO_ZIP = {}

Chat = list[dict[str, str]]
Constraint = str | tuple[str, str, bool]


class TextGenerator(TextProcessor):
    task = "Text Generation"

    @classmethod
    def available_models(cls) -> List[ModelInfo]:
        return [
            ModelInfo(
                name="dummy",
                description="a dummy model",
                tags=["default", "dummy"]
            ),
        ]

    @classmethod
    def supported_input_formats(cls) -> List[str]:
        return ["text"]

    @classmethod
    def supported_output_formats(cls) -> List[str]:
        return ["text"]

    @classmethod
    def _model_url(cls, model: str) -> str:
        return f"{_BASE_URL}/{_NAME_TO_ZIP[model]}"

    @property
    def name(self) -> str:
        return self.cfg["experiment"]["name"]

    @classmethod
    def _model_from_config(
        cls,
        cfg: Dict[str, Any],
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

    def supported_languages(self) -> Optional[List[str]]:
        lang_cfg = self.cfg["input_tokenizer"].get("language")
        if lang_cfg is None:
            return None
        return lang_cfg["languages"]

    def __init__(
        self,
        model: Model,
        cfg: Dict[str, Any],
        device: Device
    ) -> None:
        super().__init__(model, cfg, device)
        assert isinstance(model, PretrainedDecoder)
        self.logger.debug(f"got model config:\n{self.cfg['model']}")
        self.logger.info(
            f"running {self.name} text generator "
            f"on devices {[device_info(d) for d in self.devices]}"
        )
        self.input_tokenizer = tokenization.Tokenizer.from_config(
            self.cfg["input_tokenizer"]
        )
        assert "output_tokenizer" in self.cfg
        self.output_tokenizer = tokenization.Tokenizer.from_config(
            self.cfg["output_tokenizer"]
        )

        # some options for inference
        self._eos_token = self.cfg["output_tokenizer"]["eos_token"]
        self._eos_token_id = self.output_tokenizer.special_token_to_id(
            self._eos_token
        )

        # continuations are the tokens from the vocab
        # (already sorted by token id)
        self._continuations = self.output_tokenizer.get_vocab()
        self._sampling_strategy = "greedy"
        self._beam_width = 5
        self._temp = 1.0
        self._top_k = 5
        self._use_cache = True
        self._full_outputs = False
        self._max_length = None
        self._constraint = None

    def to(self, device: Device) -> "TextGenerator":
        self.devices = get_devices(device)
        assert isinstance(self.model, Model)
        self.model = self.model.distribute(self.devices)
        return self

    def _build_inference_loader_config(self) -> Dict[str, Any]:
        return {
            "tokenizer_config": self.cfg["input_tokenizer"],
            "window_config": {"type": "full"},
            "clean_text": False
        }

    def _prepare_batch(self, batch: data.InferenceBatch) -> Dict[str, Any]:
        token_ids_np, _, lengths, *_ = batch.tensors()
        return {
            "token_ids": token_ids_np,
            "lengths": lengths,
            "infos": batch.infos()
        }

    def _format_input(
        self,
        ipt: str | Chat | tuple[str | Chat, Constraint],
    ) -> data.InferenceData:
        if isinstance(ipt, tuple):
            ipt, constraint = ipt
        else:
            constraint = None

        if isinstance(ipt, str):
            ipt = [{"role": "user", "text": ipt}]

        template = self.cfg.get("chat_template", {})
        text = ""
        for message in ipt:
            role = message["role"]
            if role not in template:
                text += message["text"]
            else:
                text += template[role].replace(
                    "{text}",
                    message["text"]
                )

        return data.InferenceData(text, {"constraint": constraint})

    def _constraint_logit_fn(
        self,
        constraints: list | None
    ) -> inference_utils.LogitFn:
        def _constrain_logits(
            logits: torch.Tensor,
            indices_or_beams:  list[int] | list[Beam]
        ) -> torch.Tensor:
            zeros = torch.full_like(logits, float("-inf"))

            batch_indices = []
            constrain_indices = []
            for i, idx_or_beam in enumerate(indices_or_beams):
                if isinstance(idx_or_beam, int):
                    assert constraints is not None
                    constraint = constraints[idx_or_beam]
                else:
                    assert constraints is None
                    constraint = idx_or_beam.info.get("constraint", None)

                if constraint is None:
                    zeros[i] = logits[i]
                    continue

                constrain_to, is_match = constraint.get()

                batch_indices.extend([i] * len(constrain_to))
                constrain_indices.extend(constrain_to)

                if len(constrain_to) == 0 or is_match:
                    batch_indices.append(i)
                    constrain_indices.append(self._eos_token_id)

            batch_indices = torch.tensor(batch_indices, device=logits.device)
            constrain_indices = torch.tensor(
                constrain_indices,
                device=logits.device
            )

            zeros[batch_indices, constrain_indices] = logits[
                batch_indices,
                constrain_indices
            ]

            return zeros

        return _constrain_logits

    def _constraint_sample_fn(
        self,
        constraints: list,
        sample_fn: inference_utils.SampleFn
    ) -> inference_utils.SampleFn:
        def _sample(
            logits: torch.Tensor,
            indices: list[int]
        ) -> torch.Tensor:
            token_ids = sample_fn(logits, indices)
            for idx, token_id in zip(indices, token_ids.tolist()):
                if token_id == self._eos_token_id:
                    continue

                constraint = constraints[idx]
                if constraint is not None:
                    constraint.next(token_id)

            return token_ids

        return _sample

    def _inference(
        self,
        inputs: Dict[str, Any],
    ) -> list[Any]:
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
        ) -> Tuple[torch.Tensor, Dict[str, Any]]:
            assert isinstance(self.model, PretrainedDecoder)
            dec, cache = self.model.decode(
                token_ids,
                kwargs["lengths"],
                kwargs.get("kv_cache", None),
                self._use_cache
            )
            return dec, {"kv_cache": cache}

        def _kwargs_update_fn(
            kwargs: Dict[str, Any],
            info: Dict[str, Any],
            mask: torch.Tensor
        ) -> None:
            kv_cache = info.get("kv_cache", None)
            if kv_cache is None:
                return
            kwargs["kv_cache"] = tuple(
                tuple(c[mask.to(c.device)] for c in cache)
                for cache in info["kv_cache"]
            )

        if self._beam_width is not None and self._beam_width > 1:
            logit_fns = []
            initial_beams = []
            for token_ids, info in zip(initial_token_ids, inputs["infos"]):
                beam = Beam(token_ids, [0.0] * len(token_ids))

                constraint = self._get_constraint(info.get("constraint", None))
                if constraint is None and self._constraint is not None:
                    constraint = self._constraint.clone()
                    constraint.reset()

                beam.info["constraint"] = constraint

                initial_beams.append(beam)

            if any(
                beam.info.get("constraint", None) is not None
                for beam in initial_beams
            ):
                logit_fns.append(self._constraint_logit_fn(None))

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

            outputs = beam_search(
                decode_fn=_decode_fn,
                initial=initial_beams,
                pad_token_id=self.output_tokenizer.pad_token_id(),
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
            )
            return [output[0].token_ids for output in outputs]

        logit_fns = []
        constraints: list | None = []
        for info in inputs["infos"]:
            constraint = self._get_constraint(info.get("constraint", None))
            if constraint is None and self._constraint is not None:
                constraint = self._constraint.clone()
                constraint.reset()
            constraints.append(constraint)

        if any(c is not None for c in constraints):
            # add a logit fn that masks out tokens that are
            # not in the constraint
            logit_fns.append(self._constraint_logit_fn(constraints))

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

        if constraints is not None:
            # after sampling a token, update the constraint
            sample_fn = self._constraint_sample_fn(constraints, sample_fn)

        def stop_fn(token_ids: torch.Tensor, _: list[int]) -> torch.Tensor:
            return token_ids == self._eos_token_id

        return search(
            decode_fn=_decode_fn,
            initial_token_ids=initial_token_ids,
            pad_token_id=self.output_tokenizer.pad_token_id(),
            max_length=self.max_length,
            sample_fn=sample_fn,
            logit_fns=logit_fns,
            stop_fn=stop_fn,
            device=self.devices[0],
            kwargs_update_fn=_kwargs_update_fn,
        )

    def _process_results(
        self,
        items: list[data.InferenceItem],
        outputs: list[Any],
    ) -> data.InferenceData:
        assert len(outputs) == 1, "expected single output"

        text = self.output_tokenizer.de_tokenize(
            [self._eos_token_id] + outputs[0][:-1], False
        )[len(self._eos_token):]
        if self._full_outputs:
            text = items[0].data.text + text
        return data.InferenceData(text)

    def _get_constraint(
        self,
        constraint: Constraint | None
    ) -> grammar.RegexConstraint | grammar.LR1Constraint | None:
        if isinstance(constraint, str):
            return grammar.RegexConstraint(
                constraint,
                self._continuations
            )
        elif isinstance(constraint, tuple):
            gram, lexer, exact = constraint
            return grammar.LR1Constraint(
                gram,
                lexer,
                self._continuations,
                exact
            )
        else:
            return None

    def set_inference_options(
        self,
        sampling_strategy: str = "greedy",
        temperature: float = 1.0,
        top_k: int = 10,
        top_p: float = 0.95,
        beam_width: int | None = None,
        constraint: Constraint | None = None,
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
        self._constraint = self._get_constraint(constraint)
        self._max_length = max_length
        self._use_cache = use_cache
        self._full_outputs = full_outputs

    def generate(
        self,
        inputs: list[str | Chat | tuple[str | Chat, Constraint]],
        batch_size: int = 16,
        batch_max_tokens: Optional[int] = None,
        sort: bool = True,
        num_threads: Optional[int] = None,
        show_progress: bool = False,
        regexes: list[str] | None = None,
        cfgs: list[tuple[str, str, bool]] | None = None
    ) -> Union[str, List[str]]:
        self._regexes = regexes
        self._cfgs = cfgs

        loader = self._get_loader(
            (
                self._format_input(ipt)
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

    def generate_iter(
        self,
        iter: Iterator[str | Chat | tuple[str | Chat, Constraint]],
        batch_size: int = 16,
        batch_max_tokens: Optional[int] = None,
        sort: bool = True,
        num_threads: Optional[int] = None,
        show_progress: bool = False,
        raw: bool = False
    ) -> Union[Iterator[str], Iterator[data.InferenceData]]:
        loader = self._get_loader(
            (
                self._format_input(ipt)
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

        yield from (output if raw else output.text for output in outputs)

    def generate_file(
        self,
        input_file: str,
        output_file: Optional[Union[TextIOWrapper, str]] = None,
        sort: bool = True,
        num_threads: Optional[int] = None,
        show_progress: bool = False,
    ) -> Optional[Iterator[str]]:
        with open(input_file, "r", encoding="utf8") as inf:
            text = inf.read()

        loader = self._get_loader(
            iter([self._format_input(text)]),
            batch_size=1,
            sort=sort,
            num_threads=num_threads,
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
                output_file.write(f"{output.to_str('text')}\n")

            if output_file_is_str:
                output_file.close()

        else:
            return (output.text for output in outputs)
