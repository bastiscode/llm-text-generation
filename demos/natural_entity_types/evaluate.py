import argparse
import random
from pprint import pformat

from tqdm import tqdm

from utils import (
    format_entity,
    get_prompt_and_regex,
    load_benchmark,
    run_model
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("train")
    parser.add_argument("test")
    parser.add_argument("model")
    return parser.parse_args()


def evaluate(args: argparse.Namespace):
    train = load_benchmark(args.train)
    test = load_benchmark(args.test)

    examples = []
    for entity, types in tqdm(
        train,
        "preparing examples",
        leave=False
    ):
        # only use in-context examples with unique natural types
        if len(types) > 1:
            continue
        example, _ = format_entity(entity, types)
        examples.append(example)

    correct = 0
    incorrect = []
    for entity, gt in tqdm(test, "evaluating", leave=False):
        p, r = get_prompt_and_regex(
            entity,
            random.sample(examples, min(len(examples), 5)),
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
