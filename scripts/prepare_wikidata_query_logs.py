import argparse
import re
import os
from functools import partial
from typing import TextIO
from urllib.parse import unquote_plus

from tqdm import tqdm

from deep_sparql.utils import (
    KgIndex,
    general_prefixes,
    load_kg_index,
    prefix_pattern
)


def prepare_file(
    file: str,
    files: dict[str, TextIO],
    entity_index: KgIndex,
    property_index: KgIndex,
    seen: set[str],
    args: argparse.Namespace
) -> tuple[int, int]:
    num_total = 0
    num_duplicate = 0

    prefixes = general_prefixes()
    prefixes.update(entity_index.prefixes)
    prefixes.update(property_index.prefixes)

    prefix_patterns = {
        short: re.compile(
            rf"\b{re.escape(short)}(\w+)\b|<{re.escape(long)}(\w+)>"
        )
        for short, long in prefixes.items()
    }

    entity_prefix_pattern = prefix_pattern(entity_index.prefixes)
    property_prefix_pattern = prefix_pattern(property_index.prefixes)

    clean_pattern = re.compile(r"\s+", flags=re.MULTILINE)

    with open(file, "r") as f:
        _ = next(f)  # forward headers
        for line in tqdm(
            f,
            desc=f"processing {os.path.basename(file)}",
            disable=not args.progress,
            leave=False
        ):
            sparql, _, source, _ = line.rstrip("\r\n").split("\t")
            if args.organic_only and source != "organic":
                continue

            num_total += 1
            if sparql in seen:
                num_duplicate += 1
                continue

            seen.add(sparql)

            sparql = clean_pattern.sub(" ", unquote_plus(sparql)).strip()
            sparql_natural = sparql
            sparql_raw = sparql

            def _replace_entity(match: re.Match) -> str:
                obj = match.group(0)
                ents = entity_index.get(obj)
                if ents is not None:
                    return f"<kge kg='wikidata'>{ents[0]}</kge>"
                return obj

            def _replace_property(match: re.Match) -> str:
                obj = match.group(0)
                props = property_index.get(obj)
                if props is not None:
                    return f"<kgp kg='wikidata'>{props[0]}</kgp>"
                return obj

            sparql_natural = entity_prefix_pattern.sub(
                _replace_entity,
                sparql_natural
            )
            sparql_natural = property_prefix_pattern.sub(
                _replace_property,
                sparql_natural
            )

            sparql_prefixes = set()
            sparql_natural_prefixes = set()
            for short, pattern in prefix_patterns.items():
                def _replace_prefix(match: re.Match, seen: set) -> str:
                    seen.add(short)
                    return short + (match.group(1) or match.group(2))

                sparql = pattern.sub(
                    partial(_replace_prefix, seen=sparql_prefixes),
                    sparql
                )
                sparql_natural = pattern.sub(
                    partial(_replace_prefix, seen=sparql_natural_prefixes),
                    sparql_natural
                )

            if len(sparql_prefixes) > 0:
                sparql = " ".join(
                    f"PREFIX {short} <{prefixes[short]}>"
                    for short in sparql_prefixes
                ) + " " + sparql

            files[source].write(sparql + "\n")

            if len(sparql_natural_prefixes) > 0:
                sparql_natural = " ".join(
                    f"PREFIX {short} <{prefixes[short]}>"
                    for short in sparql_natural_prefixes
                ) + " " + sparql_natural

            files[f"{source}_natural"].write(sparql_natural + "\n")
            files[f"{source}_raw"].write(sparql_raw + "\n")

    return num_total, num_duplicate


def prepare(args: argparse.Namespace):
    sources = []
    if not args.robotic_only:
        sources.append("organic")
    if not args.organic_only:
        sources.append("robotic")

    files = {}
    for source in sources:
        if any(
            os.path.exists(os.path.join(args.output_dir, f"{source}{ext}.txt"))
            for ext in ["", ".nl", ".raw"]
        ):
            raise FileExistsError(
                f"output files for {source} in {args.output_dir}"
                " already exist"
            )

    for source in sources:
        files[source] = open(
            os.path.join(args.output_dir, f"{source}.txt"), "w"
        )
        files[f"{source}_natural"] = open(
            os.path.join(args.output_dir, f"{source}.nl.txt"), "w"
        )
        files[f"{source}_raw"] = open(
            os.path.join(args.output_dir, f"{source}.raw.txt"), "w"
        )

    num_total = 0
    num_duplicate = 0
    seen = set()

    entity_index = load_kg_index(
        args.entity_index,
        args.entity_redirects,
        args.entity_prefixes,
        args.progress
    )

    property_index = load_kg_index(
        args.property_index,
        prefixes_path=args.property_prefixes,
        progress=args.progress
    )

    for file in tqdm(
        args.files,
        desc="processing files",
        leave=False,
        disable=not args.progress
    ):
        total, duplicate = prepare_file(
            file,
            files,
            entity_index,
            property_index,
            seen,
            args
        )
        num_total += total
        num_duplicate += duplicate

    for f in files.values():
        f.close()

    print(
        f"{num_duplicate:,} / {num_total:,} duplicate "
        f"({num_duplicate / num_total:.1%}, organic_only={args.organic_only})"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--files",
        nargs="+",
        type=str,
        required=True
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        required=True
    )
    parser.add_argument(
        "--progress",
        action="store_true"
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--organic-only", action="store_true")
    source.add_argument("--robotic-only", action="store_true")
    parser.add_argument("--entity-index", type=str, required=True)
    parser.add_argument("--entity-redirects", type=str, default=None)
    parser.add_argument("--entity-prefixes", type=str, default=None)
    parser.add_argument("--property-index", type=str, required=True)
    parser.add_argument("--property-prefixes", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
