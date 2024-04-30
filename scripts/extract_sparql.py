import argparse
import requests
import re

from deep_sparql.utils import QLEVER_URLS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=["qlever"],
        required=True
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True
    )
    return parser.parse_args()


def download_html(url: str) -> str:
    # download html from the given url
    return requests.get(url).text


def extract(args: argparse.Namespace):
    kg_samples = {}
    if args.source == "qlever":
        sparql_pattern = re.compile(
            r"examples.push\(`(.*?)`\)",
            flags=re.DOTALL
        )
        question_pattern = re.compile(
            r"<a onclick=\"example=1;.*?>\s*?<span .*?>(.*?)</span>\s*</a>",
            flags=re.DOTALL
        )
        for kg, url in QLEVER_URLS.items():
            samples = []
            html = download_html(url)
            for sparql, question in zip(
                sparql_pattern.finditer(html),
                question_pattern.finditer(html)
            ):
                question = question.group(1).strip()
                sparql = sparql.group(1).strip()
                samples.append((question, sparql))
            kg_samples[kg] = samples
    else:
        raise ValueError(f"unknown source: {args.source}")


if __name__ == "__main__":
    extract(parse_args())
