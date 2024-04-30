import argparse
import os
import shutil
import logging
import time
from transformers import PreTrainedModel
import yaml

from peft.peft_model import PeftModel
from peft.tuners.ia3 import IA3Model
from peft.tuners.lora import LoraModel


from text_utils.configuration import load_config


from deep_sparql.api.generator import SPARQLGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--experiment",
        type=str,
        required=True
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True
    )
    return parser.parse_args()


def export(args: argparse.Namespace):
    # disable logging
    logging.disable(logging.CRITICAL)

    start = time.perf_counter()
    gen = SPARQLGenerator.from_experiment(
        args.experiment,
        device="cpu"
    )

    info = load_config(os.path.join(args.experiment, "info.yaml"))
    cfg = load_config(os.path.join(args.experiment, info["config_name"]))
    model_type = cfg["model"]["type"]
    cfg["model"] = {
        "type": f"custom_{model_type}",
        "path": "relpath(model)",
    }
    if isinstance(gen.model.model, PeftModel):
        if isinstance(gen.model.model.base_model, (LoraModel, IA3Model)):
            print("found lora/ia3 model, merging adapters into base model")
            gen.model.model = gen.model.model.base_model.merge_and_unload()
            cfg["train"].pop("peft", None)
        else:
            raise ValueError("unsupported peft type model")

    os.makedirs(args.output, exist_ok=True)
    shutil.copy2(
        os.path.join(args.experiment, "info.yaml"),
        os.path.join(args.output, "info.yaml")
    )
    with open(os.path.join(args.output, info["config_name"]), "w") as of:
        of.write(yaml.safe_dump(cfg))

    assert isinstance(gen.model.model, PreTrainedModel)
    gen.model.model.save_pretrained(os.path.join(args.output, "model"))
    end = time.perf_counter()
    print(f"export took {end - start:.2f} seconds")


if __name__ == "__main__":
    export(parse_args())
