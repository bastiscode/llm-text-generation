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
        if cfg.get("gradient_checkpointing", False):
            model.enable_gradient_checkpointing()
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

    def _prepare_batch(self, batch: data.TrainBatch) -> Tuple[
        Dict[str, Any],
        torch.Tensor
    ]:
        assert len(batch) > 0, "got empty batch"

        inputs = batch.tensors()
        input_type = inputs.pop("type")
        assert input_type == "generation", \
            f"unexpected input type: {input_type}"

        # print([(item.data.input, item.data.target) for item in batch.items()])
        # from text_utils import tokenization
        # tok = tokenization.Tokenizer.from_config(
        #     self.cfg["inference"]["tokenizer"]
        # )
        # for lab in list(inputs["labels"]):
        #     print(tok.de_tokenize([l for l in lab if l >= 0], False))
        # exit()

        labels = torch.from_numpy(
            inputs.pop("labels")
        ).to(
            non_blocking=True,
            dtype=torch.long,
            device=self.info.device
        )
        inputs = {
            k: torch.from_numpy(v).to(
                non_blocking=True,
                dtype=torch.int,
                device=self.info.device
            )
            for k, v in inputs.items()
        }
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
