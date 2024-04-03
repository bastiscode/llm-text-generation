import argparse
import os
import random
from multiprocessing import Pool

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
    parser.add_argument("-n", "--num-processes", type=int, default=None)
    parser.add_argument("-ne", "--num-examples", type=int, default=5)
    parser.add_argument("-l", "--label", action="store_true")
    parser.add_argument("-b", "--batch-size", type=int, default=8)
    return parser.parse_args()


def get_inputs(inputs):
    return get_prompt_and_regex(*inputs)


def run(args: argparse.Namespace):
    if args.num_processes is None:
        cpus = len(os.sched_getaffinity(0))
        args.num_processes = min(args.batch_size, cpus)

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
    with open(args.input) as inf:
        for line in inf:
            entities.append(line.strip())

    with Pool(args.num_processes) as pool:
        for i in range(0, len(entities), args.batch_size):
            batch_entities = entities[i: i + args.batch_size]
            batch_examples = [
                random.sample(examples, min(len(examples), args.num_examples))
                for _ in range(len(batch_entities))
            ]
            prompts, regexes = zip(*pool.map(
                get_inputs,
                zip(batch_entities, batch_examples)
            ))

            results = run_model(
                prompts,  # type: ignore
                regexes,  # type: ignore
                args.model,
                args.api
            )

            for entity, result in zip(entities[i: i + args.batch_size], results):
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
