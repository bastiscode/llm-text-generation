import argparse
import random
from pprint import pformat

from tqdm import tqdm

from utils import (
    format_entity,
    get_prompt_and_regex,
    load_samples,
    run_model
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("model")
    parser.add_argument("-e", "--examples", type=str, default=None)
    parser.add_argument("-ne", "--num-examples", type=int, default=5)
    return parser.parse_args()


def evaluate(args: argparse.Namespace):
    test = load_samples(args.input)

    examples = []
    if args.examples is not None:
        for entity, types in tqdm(
            load_samples(args.examples),
            "preparing examples",
            leave=False
        ):
            example, _ = format_entity(entity, types)
            examples.append(example)

    correct = 0
    incorrect = []
    for entity, gt in tqdm(test, "evaluating", leave=False):
        p, r = get_prompt_and_regex(
            entity,
            random.sample(examples, min(len(examples), args.num_examples)),
        )
        pred = run_model(p, r, args.model)

        if any(p in gt for _, p in pred):
            correct += 1
        else:
            incorrect.append((entity, pred, gt))

    print(f"Accuracy: {correct / len(test):.2%}")
    print(f"{len(incorrect)} errors:\n{pformat(incorrect)}")


if __name__ == "__main__":
    evaluate(parse_args())
