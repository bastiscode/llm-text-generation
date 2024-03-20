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
    Beam,
    BeamSelectFn,
    IdxSelectFn,
    beam_select_fn,
    greedy_select_fn,
    sample_select_fn,
    search,
    beam_search
)

from llm_text_generation.model import (
    Model,
    PretrainedDecoder,
    model_from_config
)

_BASE_URL = ""
_NAME_TO_ZIP = {}


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
        self._strategy = "greedy"
        self._beam_width = 5
        self._sample_top_k = 5
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
            "lengths": lengths
        }

    def format_chat(self, messages: list[dict[str, str]]) -> str:
        template = self.cfg.get("chat_template", {})
        text = ""
        for message in messages:
            role = message["role"]
            if role not in template:
                text += message["text"]
            else:
                text += template[role].replace(
                    "{text}",
                    message["text"]
                )
        return text

    def _get_constraints(self, n: int) -> list | None:
        if self._constraint is None:
            return None

        # this is non-blocking and runs heavy computations for
        # the constraint reset in the background on CPU
        self._constraint.reset()
        return [self._constraint.clone() for _ in range(n)]

    def _constrained_sample_select_fn(
        self,
        sample_top_k: int,
        constraints: list
    ) -> IdxSelectFn:
        def _sample(
            log_probs: torch.Tensor,
            indices: List[int]
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            assert log_probs.ndim == 2 and len(indices)
            k = min(sample_top_k, log_probs.shape[-1])

            top_k = torch.topk(log_probs, k, dim=-1)

            all_indices = []
            all_log_probs = []
            for idx, top_k_values, top_k_indices in zip(
                indices,
                top_k.values,
                top_k.indices
            ):
                length = max(1, constraints[idx].len())
                probs = torch.exp(top_k_values[:length])
                probs /= probs.sum(dim=-1, keepdim=True)
                sampled = torch.multinomial(probs, 1)
                sampled_idx = top_k_indices[sampled]
                sample_log_prob = top_k_values[sampled]
                all_indices.append(sampled_idx)
                all_log_probs.append(sample_log_prob)

            return torch.cat(all_indices), torch.cat(all_log_probs)

        return _sample

    def _constrain_idx_select_fn(
        self,
        select_fn: IdxSelectFn,
        constraints: list
    ) -> IdxSelectFn:
        def _constrained_select_fn(
            log_probs: torch.Tensor,
            indices: List[int]
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            batch_indices = []
            constrain_indices = []
            stops = []
            for i, idx in enumerate(indices):
                constraint = constraints[idx]
                constrain_to = constraint.get()
                can_stop = constraint.is_match()
                should_stop = constraint.should_stop()
                if not should_stop:
                    batch_indices.extend([i] * len(constrain_to))
                    constrain_indices.extend(constrain_to)
                can_stop = (
                    can_stop
                    or should_stop
                    or len(constrain_to) == 0
                )
                stops.append(can_stop)
                if can_stop:
                    batch_indices.append(i)
                    constrain_indices.append(self._eos_token_id)

            log_probs -= 10_000.0
            log_probs[
                torch.tensor(batch_indices),
                torch.tensor(constrain_indices)
            ] += 10_000.0

            tokens, scores = select_fn(log_probs, indices)
            for idx, can_stop, token in zip(
                indices,
                stops,
                tokens.tolist()
            ):
                if can_stop and token == self._eos_token_id:
                    continue

                # this is non-blocking and runs heavy computations for
                # the next constraint check in the background on CPU
                constraints[idx].next(token)

            return tokens, scores

        return _constrained_select_fn

    def _constrained_beam_select_fn(
        self,
        constraint: Any,
        beam_width: int
    ) -> BeamSelectFn:
        def _beam_select(
            log_probs: torch.Tensor,
            batch_beams: list[list[Beam]],
            _: list[int]
        ) -> list[list[Beam]]:
            batch_indices = []
            constrain_indices = []
            i = 0
            for beams in batch_beams:
                for beam in beams:
                    # initialize constraints
                    if "constraint" not in beam.info:
                        beam_const = constraint.clone()
                        beam_const.reset()
                        beam.info["constraint"] = beam_const

                    beam_const = beam.info["constraint"]

                    length = 0
                    constrain_to = beam_const.get()
                    can_stop = beam_const.is_match()
                    should_stop = beam_const.should_stop()
                    if not should_stop:
                        batch_indices.extend([i] * len(constrain_to))
                        constrain_indices.extend(constrain_to)
                        length += len(constrain_to)
                    can_stop = (
                        can_stop
                        or should_stop
                        or len(constrain_to) == 0
                    )
                    if can_stop:
                        batch_indices.append(i)
                        constrain_indices.append(self._eos_token_id)
                        length += 1

                    beam.info["can_stop"] = can_stop
                    beam.info["length"] = length
                    i += 1

            log_probs -= 10_000.0
            log_probs[
                torch.tensor(batch_indices),
                torch.tensor(constrain_indices)
            ] += 10_000.0

            num_beams = [len(b) for b in batch_beams]
            assert log_probs.ndim == 2 and log_probs.shape[0] == sum(num_beams)
            k = min(beam_width, log_probs.shape[1])
            top_k = torch.topk(log_probs, k, dim=1)

            # get new candidates
            batch_candidates = []
            for beams, indices, values in zip(
                batch_beams,
                torch.split(top_k.indices, num_beams),
                torch.split(top_k.values, num_beams)
            ):
                candidates = []
                for idx, (token_ids, log_probs) in enumerate(zip(
                    indices.tolist(),
                    values.tolist()
                )):
                    length = beams[idx].info["length"]
                    candidates.extend(
                        (idx, token_id, log_p)
                        for token_id, log_p in
                        zip(token_ids[:length], log_probs[:length])
                    )

                candidates = sorted(
                    candidates,
                    key=lambda item: -(beams[item[0]].log_prob + item[2]),
                )[:2 * beam_width]

                updated_candidates = []
                for idx, token_id, log_p in candidates:
                    beam = beams[idx]
                    new_beam = Beam.from_beam(beam, log_p, token_id)
                    beam_const = beam.info["constraint"].clone()
                    if not (
                        beam.info["can_stop"]
                        and token_id == self._eos_token_id
                    ):
                        beam_const.next(token_id)

                    new_beam.info["constraint"] = beam_const

                    updated_candidates.append(new_beam)

                batch_candidates.append(updated_candidates)

            return batch_candidates

        return _beam_select

    @torch.inference_mode()
    def _run_model(self, batch: data.InferenceBatch) -> list[Any]:
        inputs = self._prepare_batch(batch)
        return self._inference(inputs)

    def _inference(
        self,
        inputs: Dict[str, Any],
    ) -> list[Any]:
        batch_size = len(inputs["token_ids"])
        inference_kwargs = {}
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

        is_beam = self._strategy == "beam" and self._beam_width > 1
        is_sample = self._strategy == "sample" and self._sample_top_k > 1

        if is_beam:
            if self._constraint is not None:
                beam_select = self._constrained_beam_select_fn(
                    self._constraint,
                    self._beam_width
                )
            else:
                beam_select = beam_select_fn(self._beam_width)

            def beam_stop_fn(beam: Beam, _: int) -> bool:
                return beam.token_ids[-1] == self._eos_token_id

            outputs = beam_search(
                decode_fn=_decode_fn,
                initial_token_ids=initial_token_ids,
                pad_token_id=self.output_tokenizer.pad_token_id(),
                max_length=self.max_length,
                stop_fn=beam_stop_fn,
                device=self.devices[0],
                normalize_by_length=True,
                alpha=1.0,
                beam_width=self._beam_width,
                select_fn=beam_select,
                kwargs_update_fn=_kwargs_update_fn,
                **inference_kwargs
            )
            return [output[0].token_ids for output in outputs]

        else:
            idx_select = sample_select_fn(
                self._sample_top_k
            ) if is_sample else greedy_select_fn()

            constraints = self._get_constraints(batch_size)
            if constraints is not None:
                if is_sample:
                    idx_select = self._constrained_sample_select_fn(
                        self._sample_top_k,
                        constraints
                    )

                idx_select = self._constrain_idx_select_fn(
                    idx_select,
                    constraints
                )

            def stop_fn(token_ids: torch.Tensor, _: List[int]) -> torch.Tensor:
                return token_ids == self._eos_token_id

            return search(
                decode_fn=_decode_fn,
                initial_token_ids=initial_token_ids,
                pad_token_id=self.output_tokenizer.pad_token_id(),
                max_length=self.max_length,
                select_fn=idx_select,
                stop_fn=stop_fn,
                device=self.devices[0],
                kwargs_update_fn=_kwargs_update_fn,
                **inference_kwargs
            )

    def _process_results(
        self,
        items: List[data.InferenceItem],
        outputs: List[Any],
    ) -> data.InferenceData:
        assert len(outputs) == 1, "expected single output"

        text = self.output_tokenizer.de_tokenize(
            [self._eos_token_id] + outputs[0][:-1], False
        )[len(self._eos_token):]
        if self._full_outputs:
            text = items[0].data.text + text
        return data.InferenceData(
            text,
            language=items[0].data.language
        )

    def set_inference_options(
        self,
        strategy: str = "greedy",
        beam_width: int = 5,
        sample_top_k: int = 5,
        regex: str | None = None,
        regex_file: str | None = None,
        cfg: tuple[str, str, bool] | None = None,
        cfg_files: tuple[str, str, bool] | None = None,
        max_length: int | None = None,
        use_cache: bool = True,
        full_outputs: bool = False
    ) -> None:
        assert strategy in ["greedy", "beam", "sample"]
        self._strategy = strategy
        self._beam_width = beam_width
        self._sample_top_k = sample_top_k

        assert sum([
            regex is not None,
            regex_file is not None,
            cfg is not None,
            cfg_files is not None
        ]) <= 1, \
            "only one of regex, regex file, cfg, or cfg files can be specified"
        if regex is not None:
            self._constraint = grammar.RegexConstraint(
                regex,
                self._continuations
            )
        elif regex_file is not None:
            self._constraint = grammar.RegexConstraint.from_file(
                regex_file,
                self._continuations
            )
        elif cfg is not None:
            grammar_string, lexer_string, exact = cfg
            self._constraint = grammar.LR1Constraint(
                grammar_string,
                lexer_string,
                self._continuations,
                exact
            )
        elif cfg_files is not None:
            grammar_file, lexer_file, exact = cfg_files
            self._constraint = grammar.LR1Constraint.from_files(
                grammar_file,
                lexer_file,
                self._continuations,
                exact
            )
        else:
            self._constraint = None

        self._max_length = max_length
        self._use_cache = use_cache
        self._full_outputs = full_outputs

    def generate(
        self,
        inputs: Union[str, List[str]],
        batch_size: int = 16,
        batch_max_tokens: Optional[int] = None,
        sort: bool = True,
        num_threads: Optional[int] = None,
        show_progress: bool = False,
    ) -> Union[str, List[str]]:
        input_is_string = isinstance(inputs, str)
        assert (
            input_is_string
            or (
                isinstance(inputs, list)
                and all(isinstance(ipt, str) for ipt in inputs)
            )
        ), "input needs to be a string or a list of strings"

        if input_is_string:
            inputs = [inputs]

        loader = self._get_loader(
            (data.InferenceData(ipt) for ipt in inputs),
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

        if input_is_string:
            return next(iter(outputs)).text
        else:
            return [
                output.text
                for output in outputs
            ]

    def generate_iter(
        self,
        iter: Iterator[str],
        batch_size: int = 16,
        batch_max_tokens: Optional[int] = None,
        sort: bool = True,
        num_threads: Optional[int] = None,
        show_progress: bool = False,
        raw: bool = False
    ) -> Union[Iterator[str], Iterator[data.InferenceData]]:
        loader = self._get_loader(
            (data.InferenceData(ipt) for ipt in iter),
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
            iter([data.InferenceData(text)],),
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
