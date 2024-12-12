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
from torch.nn.utils.rnn import pad_sequence

from llm_text_generation.api.utils import format_chat
from llm_text_generation.model import (
    ClassAndProb,
    Model,
    PretrainedDecoder,
    PretrainedDecoderForClassification,
    model_from_config,
    peft_model_from_config,
)

_BASE_URL = ""
_NAME_TO_ZIP = {}

Chat = list[dict[str, str]]
Const = str | tuple[str, str, bool]


class TextClassifier(TextProcessor):
    task = "Text Classification"

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
        assert isinstance(
            model, PretrainedDecoderForClassification
        ), "only decoder models for classification are supported"

        self.logger.debug(f"Got model config:\n{self.cfg['model']}")
        self.logger.info(
            f"Running {self.name} text classifier "
            f"on devices {[device_info(d) for d in self.devices]}"
        )
        self.tokenizer = tokenization.Tokenizer.from_config(
            self.cfg["inference"]["tokenizer"]
        )

        # some options for inference
        self._eos_token = self.cfg["inference"]["eos"]
        self._eos_token_id = self.tokenizer.token_to_id(self._eos_token)

        self._is_chat = self.cfg["inference"].get("chat_template", None) is not None

        self.model = self.model.compile(**self.cfg["inference"].get("compile", {}))

    def to(self, device: Device) -> "TextClassifier":
        self.devices = get_devices(device)
        if self.cfg["model"].get("device_map", None) is not None:
            return self
        assert isinstance(self.model, Model)
        self.model = self.model.distribute(self.devices)
        return self

    def _prepare_input(
        self,
        ipt: str | Chat | tuple[str | Chat, Const],
    ) -> tuple[str, dict[str, Any]]:
        info = {}
        if isinstance(ipt, tuple):
            ipt, constraint = ipt
            info["const"] = constraint

        template = self.cfg["inference"].get(
            "chat_template",
            {"roles": {"system": "{text}", "user": "{text}", "assistant": "{text}"}},
        )

        prompt = format_chat(ipt, template) + self._eos_token
        return prompt, info

    @torch.inference_mode()
    def classify(
        self,
        inputs: Iterable[str | Chat],
        top_k: int = 1,
        batch_size: int = 16,
        batch_max_tokens: int | None = None,
        sort: bool = True,
        num_threads: int | None = None,
        show_progress: bool = False,
    ) -> Iterator[ClassAndProb | list[ClassAndProb]]:
        assert top_k > 0, "top_k must be greater than 0"
        assert isinstance(self.model, PretrainedDecoderForClassification)
        texts = []
        infos = []
        for ipt in inputs:
            text, info = self._prepare_input(ipt)
            texts.append(text + self._eos_token)
            infos.append(info)

        def inference_fn(
            batch: data.InferenceBatch,
        ) -> list[ClassAndProb | list[ClassAndProb]]:
            token_ids = batch.token_ids()
            lengths = batch.sizes()

            token_ids = pad_sequence(
                [torch.tensor(ids) for ids in batch.token_ids()],
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id(),
            ).to(self.devices[0], non_blocking=True)
            lengths = torch.tensor(lengths, dtype=torch.long, device=self.devices[0])

            top_classes = self.model.classify(token_ids, lengths, top_k)
            if top_k == 1:
                return [top[0] for top in top_classes]
            else:
                return top_classes

        def postprocessing_fn(
            items: list[data.InferenceItem],
            outputs: list[ClassAndProb | list[ClassAndProb]],
        ) -> ClassAndProb | list[ClassAndProb]:
            assert len(items) == 1 and len(outputs) == 1
            return outputs[0]

        yield from self._process(
            iter(texts),
            inference_fn,
            postprocessing_fn,
            "Classifying text",
            batch_size,
            batch_max_tokens,
            sort,
            num_threads,
            progress_total=len(texts),
            show_progress=show_progress,
            ignore_special_tokens=self._is_chat,
        )
