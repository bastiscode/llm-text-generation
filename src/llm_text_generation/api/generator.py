from io import TextIOWrapper
import os
import sys
from typing import Any, Dict, List, Tuple, Optional, Union, Iterator

import torch
from torch import nn
from peft import get_peft_model

from text_utils import data, tokenization, constraints
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

_BASE_URL = "https://ad-publications.informatik.uni-freiburg.de/" \
    "ACL_whitespace_procession_transformer_BHW_2023.materials"
_NAME_TO_ZIP = {
}


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
        return (
            self._max_length
            or self.cfg["train"]["data"].get("max_length", 512)
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
        eos_token = self.cfg["output_tokenizer"]["eos_token"]
        self._eos_token_id = self.output_tokenizer.special_token_to_id(
            eos_token
        )

        # preprocess continuations for constraints
        self._continuations = [
            self.output_tokenizer.de_tokenize(
                [self._eos_token_id, i, self._eos_token_id],
                False
            )[len(eos_token):-len(eos_token)].encode("utf8")
            for i in range(self.output_tokenizer.vocab_size())
        ]

        self._strategy = "greedy"
        self._beam_width = 5
        self._sample_top_k = 5
        self._use_cache = True
        self._full_outputs = False
        self._max_length = None
        self._regex_constraint = None
        self._cfg_constraint = None

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

    def _constrain_idx_select_fn(
        self,
        select_fn: IdxSelectFn,
        batch_size: int
    ) -> IdxSelectFn:
        if self._regex_constraint is not None:
            self._regex_constraint.reset()
            re_constraints = [
                self._regex_constraint.clone()
                for _ in range(batch_size)
            ]

            def _re_select_fn(
                log_probs: torch.Tensor,
                indices: List[int]
            ) -> Tuple[torch.Tensor, torch.Tensor]:
                batch_indices = []
                constrain_indices = []
                final = []
                for i, idx in enumerate(indices):
                    constrain_to = re_constraints[idx].get_constraint_indices()
                    batch_indices.extend((i for _ in range(len(constrain_to))))
                    constrain_indices.extend(constrain_to)
                    is_final = re_constraints[idx].is_final_state()
                    final.append(is_final)
                    if is_final or len(constrain_to) == 0:
                        batch_indices.append(i)
                        constrain_indices.append(self._eos_token_id)

                log_probs -= 10_000.0
                log_probs[
                    torch.tensor(batch_indices),
                    torch.tensor(constrain_indices)
                ] += 10_000.0

                tokens, scores = select_fn(log_probs, indices)
                for idx, is_final, token in zip(
                    indices,
                    final,
                    tokens.tolist()
                ):
                    if is_final and token == self._eos_token_id:
                        continue

                    re_constraints[idx].next(token)

                return tokens, scores

            return _re_select_fn

        elif self._cfg_constraint is not None:
            raise NotImplementedError

        else:
            return select_fn

    def _constrain_beam_select_fn(
        self,
        select_fn: BeamSelectFn
    ) -> BeamSelectFn:
        if self._regex_constraint is not None:
            raise NotImplementedError("regex constraint not implemented")

        elif self._cfg_constraint is not None:
            raise NotImplementedError("cfg constraint not implemented")

        return select_fn

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

            idx_select = self._constrain_idx_select_fn(idx_select, batch_size)

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
        text = self.output_tokenizer.de_tokenize(outputs[0][:-1])
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
        cfg: str | None = None,
        cfg_file: str | None = None,
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
            cfg_file is not None
        ]) <= 1, \
            "only one of regex, regex file, cfg, or cfg file can be specified"
        if regex is not None:
            self._regex_constraint = constraints.Regex(
                regex,
                self._continuations
            )
        elif regex_file is not None:
            self._regex_constraint = constraints.Regex.from_file(
                regex_file,
                self._continuations
            )
        elif cfg is not None:
            raise NotImplementedError
        elif cfg_file is not None:
            raise NotImplementedError
        else:
            self._regex_constraint = None
            self._cfg_constraint = None

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
