import os
from typing import Dict, Any, Tuple

import torch
from torch import nn
from peft import (
    PeftConfig,
    prepare_model_for_kbit_training,
    get_peft_model
)

from text_utils.api.trainer import ShardingPolicy, Trainer
from text_utils import data

from llm_text_generation.model import (
    PretrainedDecoder,
    model_from_config
)


class TextGenerationTrainer(Trainer):
    @classmethod
    def _model_from_config(
        cls,
        cfg: Dict[str, Any]
    ) -> Tuple[nn.Module, ShardingPolicy | None]:
        model = model_from_config(cfg["model"])
        return model, model.get_sharding_policy()

    @classmethod
    def _prepare_peft(
        cls,
        model: nn.Module,
        peft_cfg: PeftConfig,
        use8_bit: bool = False
    ) -> nn.Module:
        if isinstance(model, PretrainedDecoder):
            if use8_bit:
                model.model = prepare_model_for_kbit_training(
                    model.model
                )
            model.model = get_peft_model(model.model, peft_cfg)
        else:
            raise RuntimeError(
                "peft is only supported for pretrained models"
            )
        return model

    def _prepare_batch(
        self,
        batch: data.DataBatch,
        train: bool = True
    ) -> Tuple[Dict[str, Any], torch.Tensor]:
        assert len(batch) > 0, "got empty batch"

        (
            token_ids_np,
            pad_mask_np,
            lengths,
            info,
            labels_np,
            label_info
        ) = batch.tensors()

        if self.cfg["model"]["type"] == "pretrained_decoder":
            mask_prefix = self.cfg["train"].get("mask_prefix", False)
            if (not train or mask_prefix) and "prefix_lengths" in label_info:
                # mask out the prefix in the labels with -1 to ignore it
                for i, pfx_l in enumerate(label_info["prefix_lengths"]):
                    if pfx_l <= 0:
                        continue
                    labels_np[i, :pfx_l - 1] = -1

            inputs = {
                "token_ids": torch.from_numpy(
                    label_info["token_ids"]
                ).to(
                    non_blocking=True,
                    device=self.info.device
                ),
                "padding_mask": torch.from_numpy(
                    label_info["padding_mask"]
                ).to(
                    non_blocking=True,
                    device=self.info.device
                ),
                "lengths": label_info["lengths"],
            }

        else:
            raise RuntimeError(
                f"unknown model type: {self.cfg['model']['type']}"
            )

        labels = torch.from_numpy(labels_np).to(
            non_blocking=True,
            dtype=torch.long,
            device=self.info.device
        )

        return inputs, labels


def main():
    parser = TextGenerationTrainer.parser(
        "Text generation", "Train a model for generating text"
    )
    args = parser.parse_args()
    work_dir = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        ".."
    )
    if args.platform == "local":
        TextGenerationTrainer.train_local(
            work_dir, args.experiment, args.config, args.profile
        )
    else:
        TextGenerationTrainer.train_slurm(
            work_dir, args.experiment, args.config
        )


if __name__ == "__main__":
    main()
