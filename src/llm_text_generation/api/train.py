import os
import copy
from typing import Any

import torch

from text_utils.api.trainer import ShardingPolicy, Trainer
from text_utils import data, tensorboard

from llm_text_generation.api.utils import InputOutputLogger
from llm_text_generation.model import Model, model_from_config, peft_model_from_config


class TextGenerationTrainer(Trainer):
    @classmethod
    def _model_from_config(cls, cfg: dict[str, Any]) -> Model:
        model = model_from_config(cfg["model"])
        if cfg.get("gradient_checkpointing", False):
            model.enable_gradient_checkpointing()
        return model

    @classmethod
    def _prepare_peft(  # type: ignore
        cls,
        model: Model,
        cfg: dict[str, Any],
    ) -> Model:
        return peft_model_from_config(model, cfg)

    @classmethod
    def _metric_from_config(
        cls, cfg: dict[str, Any], prefix: str
    ) -> tensorboard.TensorboardMetric:
        cfg = copy.deepcopy(cfg)
        metric_typ = cfg.pop("type", None)
        if metric_typ == "input_output":
            return InputOutputLogger(prefix)
        else:
            raise ValueError(f"unknown metric type {metric_typ}")

    @classmethod
    def _sharding_policy(  # type: ignore
        cls, model: Model
    ) -> ShardingPolicy | None:
        return model.get_sharding_policy()

    def _prepare_batch(
        self, batch: data.TrainBatch
    ) -> tuple[dict[str, Any], torch.Tensor]:
        assert len(batch) > 0, "got empty batch"

        inputs = batch.tensors()
        input_type = inputs.pop("type")
        assert input_type == "generation", f"unexpected input type: {input_type}"

        labels = torch.from_numpy(inputs.pop("labels")).to(
            non_blocking=True, dtype=torch.long, device=self.info.device
        )
        inputs = {
            k: torch.from_numpy(v).to(
                non_blocking=True, dtype=torch.int, device=self.info.device
            )
            for k, v in inputs.items()
        }
        return inputs, labels


def main():
    parser = TextGenerationTrainer.parser(
        "Text generation", "Train a model for generating text"
    )
    args = parser.parse_args()
    work_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..")
    if args.platform == "local":
        TextGenerationTrainer.train_local(
            work_dir, args.experiment, args.config, args.profile
        )
    else:
        TextGenerationTrainer.train_slurm(work_dir, args.experiment, args.config)


if __name__ == "__main__":
    main()
