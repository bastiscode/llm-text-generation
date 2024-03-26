import argparse
import random

from tqdm import tqdm

from utils import (
    get_prompt_and_regex,
    load_samples,
    run_model,
    format_entity
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("model")
    parser.add_argument("-e", "--examples", type=str, default=None)
    parser.add_argument("-l", "--label", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace):
    examples = []
    if args.examples is not None:
        for entity, types in tqdm(
            load_samples(args.examples),
            "preparing examples",
            leave=False
        ):
            # only use in-context examples with unique natural types
            if len(types) > 1:
                continue
            example, _ = format_entity(entity, types)
            examples.append(example)

    with open(args.input) as inf:
        for line in inf:
            entity = line.strip()
            if entity == "":
                continue
            p, r = get_prompt_and_regex(
                entity,
                random.sample(examples, min(len(examples), 5))
            )
            results = run_model(p, r, args.model)
            assert len(results) == 1
            label, qid = results[0]
            s = f"{entity}\t{qid}"
            if args.label:
                s += f"\t{label}"
            print(s)


if __name__ == "__main__":
    run(parse_args())
