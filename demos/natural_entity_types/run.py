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
    parser.add_argument("-a", "--api", type=str, default=None)
    parser.add_argument("-e", "--examples", type=str, default=None)
    parser.add_argument("-ne", "--num-examples", type=int, default=5)
    parser.add_argument("-l", "--label", action="store_true")
    parser.add_argument("-b", "--batch-size", type=int, default=8)
    return parser.parse_args()


def run(args: argparse.Namespace):
    examples = []
    if args.examples is not None:
        for entity, types in tqdm(
            load_samples(args.examples),
            "preparing examples",
            leave=False
        ):
            example, _ = format_entity(entity, types)
            examples.append(example)

    entities = []
    prompts = []
    regexes = []
    with open(args.input) as inf:
        for line in inf:
            entity = line.strip()
            if entity == "":
                continue

            entities.append(entity)
            p, r = get_prompt_and_regex(
                entity,
                random.sample(examples, min(len(examples), args.num_examples))
            )
            prompts.append(p)
            regexes.append(r)

    for i in range(0, len(prompts), args.batch_size):
        prompt_batch = prompts[i: i + args.batch_size]
        regex_batch = regexes[i: i + args.batch_size]
        entity_batch = entities[i: i + args.batch_size]

        results = run_model(
            prompt_batch,
            regex_batch,
            args.model,
            args.api
        )

        for entity, result in zip(entity_batch, results):
            s = f"{entity}\t"
            if len(result) == 0:
                if args.label:
                    s += "\t"
                print(s)

            assert len(result) == 1
            label, qid = result[0]
            s += qid
            if args.label:
                s += f"\t{label}"
            print(s)


if __name__ == "__main__":
    run(parse_args())
