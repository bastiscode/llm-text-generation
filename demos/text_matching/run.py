import argparse

from tqdm import tqdm

from utils import (
    format_sample,
    get_prompt,
    run_model
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("model")
    parser.add_argument("-e", "--examples", type=str, default=None)
    parser.add_argument("-b", "--batch-size", type=int, default=8)
    return parser.parse_args()


def run(args: argparse.Namespace):
    examples = []
    if args.examples is not None:
        with open(args.examples) as inf:
            for line in inf:
                line = line.rstrip("\r\n")
                score, query, entities, *texts = line.split("\t")
                example = format_sample(
                    query,
                    entities.split(),
                    texts,
                    int(score)
                )
                examples.append(example)

    prompts = []
    with open(args.input) as inf:
        for line in inf:
            query, entities, *texts = line.rstrip("\r\n").split("\t")
            p = get_prompt(query, entities.split(), texts, examples)
            prompts.append(p)

    for i in tqdm(
        range(0, len(prompts), args.batch_size),
        desc="getting scores",
        leave=False
    ):
        scores = run_model(prompts[i: i + args.batch_size], args.model)
        for score in scores:
            print(score)


if __name__ == "__main__":
    run(parse_args())
