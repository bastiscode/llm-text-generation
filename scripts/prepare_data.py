import argparse
import os
import re
import json
import collections

from tqdm import tqdm
from datasets import load_dataset


from deep_sparql.utils import (
    KgIndex,
    general_prefixes,
    load_kg_index,
    prefix_pattern,
    preprocess_natural_language_query,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    data = parser.add_mutually_exclusive_group(required=True)

    # wikidata
    data.add_argument("--wikidata-simple-questions", type=str)
    data.add_argument("--qald-10", type=str)
    data.add_argument("--time-questions", type=str)
    data.add_argument("--cron-questions", type=str)
    data.add_argument("--mkqa", type=str)
    data.add_argument("--mintaka", type=str)
    data.add_argument("--lc-quad2-wikidata", type=str)
    data.add_argument("--mcwq", type=str)
    data.add_argument("--qa-wiki", type=str)
    data.add_argument("--kqa-pro", type=str)

    # freebase
    data.add_argument("--graph-questions", type=str)
    data.add_argument("--wqsp", type=str)
    data.add_argument("--complex-web-questions", type=str)
    data.add_argument("--freebase-simple-questions", type=str)
    data.add_argument("--30mqa", type=str)
    data.add_argument("--cfq", type=str)
    data.add_argument("--grail-qa", type=str)
    data.add_argument("--freebase-qa", type=str)

    # dbpedia
    data.add_argument("--lc-quad2-dbpedia", type=str)
    data.add_argument("--qald-9-plus", type=str)
    data.add_argument("--simple-dbpedia-qa", type=str)
    data.add_argument("--mlpq", type=str)
    data.add_argument("--monument", type=str)

    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--entity-index", type=str, required=True)
    parser.add_argument("--entity-redirects", type=str, default=None)
    parser.add_argument("--entity-prefixes", type=str, required=True)
    parser.add_argument("--property-index", type=str, required=True)
    parser.add_argument("--property-prefixes", type=str, required=True)
    parser.add_argument("--version", choices=["v1", "v2"], default="v2")
    parser.add_argument("--skip-incomplete", action="store_true")
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


Sample = collections.namedtuple("Sample", ["question", "sparql", "result"])


SPLIT_RENAME = {
    "train": "train",
    "test": "test",
    "dev": "val",
    "valid": "val",
    "validation": "val",
}


def load_data(args: argparse.Namespace) -> tuple[str, dict[str, list[Sample]]]:
    output = {}
    if args.wikidata_simple_questions is not None:
        kg = "wikidata"
        data = load_dataset(args.wikidata_simple_questions)
        for split, items in data.items():
            split = SPLIT_RENAME.get(split, split)
            assert split in {"train", "val", "test"}
            samples = []
            for item in items:
                question = item["question"]
                subj = item["answer"]["subject"]
                obj = item["answer"]["object"]
                prop = item["answer"]["predicate"]

                if prop.startswith("R"):
                    subj, obj = obj, subj
                    subj = "x"
                    prop = "P" + prop[1:]
                else:
                    obj = "x"
                prop = "wdt:" + prop

                if subj == "x":
                    subj = "?" + subj
                    obj = "wd:" + obj
                else:
                    obj = "?" + obj
                    subj = "wd:" + subj

                sparql = f"SELECT ?x WHERE {{ {subj} {prop} {obj} . }}"
                samples.append(Sample(question, sparql, None))
            output[split] = samples

    elif args.qald_10 is not None:
        kg = "wikidata"
        data = load_dataset(args.qald_10)
        for split, items in data.items():
            split = SPLIT_RENAME.get(split, split)
            assert split in {"train", "val", "test"}
            samples = []
            for item in items:
                sparql = item["query"]["sparql"]
                questions = [
                    q["string"]
                    for q in json.loads(item["question"])
                    if q["language"] == "en"
                ]
                # replace entities and properties
                sparql = re.sub(
                    r"<http://www.wikidata.org/entity/(Q\d+?)>",
                    lambda match: "wd:" + match.group(1),
                    sparql
                )

                def _rep_prop(m: re.Match) -> str:
                    pfx = m.group(1)
                    if pfx == "direct":
                        pfx = "wdt"
                    else:
                        raise RuntimeError(f"unknown prefix {pfx}")
                    return f"{pfx}:{m.group(2)}"

                sparql = re.sub(
                    r"<http://www.wikidata.org/prop/(?:(\S+?)/)?(P\d+?)>",
                    _rep_prop,
                    sparql
                )
                for q in questions:
                    samples.append(Sample(q, sparql, None))

            output[split] = samples

    elif args.lc_quad2_wikidata is not None:
        kg = "wikidata"
        data = load_dataset(args.lc_quad2_wikidata, "lcquad2-wikidata")
        for split, items in data.items():
            split = SPLIT_RENAME.get(split, split)
            assert split in {"train", "val", "test"}
            samples = []
            for item in items:
                questions = [item["question"]]
                sparql = item["sparql"]
                for pq in item["paraphrased_question"]:
                    questions.append(pq)
                for q in questions:
                    if q is None or q.strip() == "" or "{" in q or "}" in q:
                        continue
                    samples.append(Sample(q, sparql, None))
            output[split] = samples

    elif args.mcwq is not None:
        kg = "wikidata"
        with open(os.path.join(args.mcwq, "dataset.json"), "r") as inf:
            train_data = json.load(inf)
        with open(os.path.join(args.mcwq, "gold_test.json"), "r") as inf:
            test_data = json.load(inf)
        for data, split in [(train_data, "train"), (test_data, "test")]:
            samples = []
            for item in data:
                question = item["questionWithBrackets"]
                # sub out brackets
                question = re.sub(
                    r"\[(.+?)\]",
                    lambda m: m.group(1),
                    question
                )
                # repair some whitespace issues
                # words followed by 's
                question = re.sub(
                    r"(\w+)\s+('s)(?:\s+|$)",
                    lambda m: m.group(1) + m.group(2) + " ",
                    question
                )
                # punctuation with surrounding spaces
                question = re.sub(
                    r"\s+([,.?!;])(?:\s+|$)",
                    lambda m: m.group(1) + " ",
                    question
                )
                sparql = item["sparql"]
                samples.append(Sample(question, sparql, None))
            output[split] = samples

    elif args.qa_wiki is not None:
        kg = "wikidata"
        samples = []
        with open(args.qa_wiki, "r") as inf:
            for line in inf:
                line = line.strip()
                sparql, question = line.split("\t")
                samples.append(Sample(question, sparql, None))
        output["train"] = samples

    else:
        raise RuntimeError("unknown dataset")

    return kg, output


def format_query(
    query: str,
    version: str,
    kg: str
) -> str:
    if version == "v1":
        return f"Generate a SPARQL query over {kg.capitalize()} for " \
            f"the question \"{query}\""

    query = preprocess_natural_language_query(query, kg)
    return json.dumps(query)


PREFIX_PATTERN = re.compile(r"PREFIX\s*(\w*:)\s*<[^>]+>")


def format_ent(ent: str, version: str, kg: str) -> str:
    if version == "v2":
        return f"<kge kg='{kg}'>{ent}</kge>"
    else:
        return f"<boe>{ent}<eoe>"


def format_prop(prop: str, version: str, kg: str) -> str:
    if version == "v2":
        return f"<kgp kg='{kg}'>{prop}</kgp>"
    else:
        return f"<bop>{prop}<eop>"


def clean_prefixes(
    sparql: str,
    prefix_patterns: dict[str, re.Pattern],
    prefixes: dict[str, str],
) -> str:
    exist = set(PREFIX_PATTERN.findall(sparql))
    seen = set()
    for short, pattern in prefix_patterns.items():
        def _replace_prefix(match: re.Match) -> str:
            nonlocal seen
            seen.add(short)
            return short + (match.group(1) or match.group(2))

        sparql = pattern.sub(_replace_prefix, sparql)

    diff = seen.difference(exist)
    if len(diff) > 0:
        sparql = " ".join(
            f"PREFIX {short} <{prefixes[short]}>"
            for short in diff
        ) + " " + sparql

    return sparql


def replace_vars_and_special_tokens(
    sparql: str,
    version: str,
) -> str:
    if version == "v1":
        # replace variables ?x or $x with <bov>x<eov>
        sparql = re.sub(
            r"\?(\w+)|\$(\w+)",
            lambda m: f"<bov>{m.group(1) or m.group(2)}<eov>",
            sparql
        )
        # replace brackets {, and } with <bob> and <eob>
        sparql = re.sub(
            r"{",
            "<bob>",
            sparql
        )
        sparql = re.sub(
            r"}",
            "<eob>",
            sparql
        )
    return sparql


def replace_entities_and_properties(
    sparql: str,
    kg: str,
    entity_index: KgIndex,
    property_index: KgIndex,
    entity_pattern: re.Pattern,
    property_pattern: re.Pattern,
    version: str,
    replacement: str = "only_first",
) -> tuple[list[str], bool]:
    assert replacement in [
        "only_first",
        "in_order"
    ]

    replacements = collections.Counter()
    incomplete = False
    done = False

    def _replace_ent(m: re.Match) -> str:
        nonlocal replacements
        obj = m.group(0)
        ents = entity_index.get(obj)
        if ents is not None:
            idx = replacements[obj]
            if idx < len(ents):
                replacements[obj] += 1
                obj = format_ent(ents[idx], version, kg)
            else:
                nonlocal done
                done = True
        else:
            nonlocal incomplete
            incomplete = True

        return obj

    def _replace_prop(m: re.Match) -> str:
        nonlocal replacements
        obj = m.group(0)
        props = property_index.get(obj)
        if props is not None:
            idx = replacements[obj]
            if idx < len(props):
                replacements[obj] += 1
                obj = format_prop(props[idx], version, kg)
            else:
                nonlocal done
                done = True
        else:
            nonlocal incomplete
            incomplete = True

        return obj

    org_sparql = sparql

    sparql = entity_pattern.sub(
        _replace_ent,
        sparql
    )

    sparql = property_pattern.sub(
        _replace_prop,
        sparql
    )

    sparqls = [sparql]
    if replacement == "only_first":
        return sparqls, incomplete

    while True:
        sparql = org_sparql

        sparql = entity_pattern.sub(
            _replace_ent,
            sparql
        )

        sparql = property_pattern.sub(
            _replace_prop,
            sparql
        )

        if done:
            break

        sparqls.append(sparql)

    return sparqls, incomplete


def prepare(args: argparse.Namespace):
    kg, data = load_data(args)
    prefixes = general_prefixes()

    ent_index = load_kg_index(
        args.entity_index,
        args.entity_redirects,
        args.entity_prefixes,
        args.progress
    )
    ent_pattern = prefix_pattern(ent_index.prefixes)
    prefixes.update(ent_index.prefixes)

    prop_index = load_kg_index(
        args.property_index,
        prefixes_path=args.property_prefixes,
        progress=args.progress
    )
    prop_pattern = prefix_pattern(prop_index.prefixes)
    prefixes.update(prop_index.prefixes)

    prefix_patterns = {
        short: re.compile(
            rf"\b{re.escape(short)}(\w+)\b|<{re.escape(long)}(\w+)>"
        )
        for short, long in prefixes.items()
    }

    os.makedirs(args.output, exist_ok=True)

    for split, samples in data.items():
        input = os.path.join(
            args.output,
            f"{split}_input.txt"
        )
        assert len(samples) > 0, f"no samples for split {split}"
        has_sparql = samples[0].sparql is not None
        target_name = "sparql" if has_sparql else "result"
        target = os.path.join(
            args.output,
            f"{split}_{target_name}.txt"
        )
        raw = os.path.join(
            args.output,
            f"{split}_raw.txt"
        )
        incomplete = 0
        total = 0
        with open(input, "w") as inf, \
                open(target, "w") as tf, \
                open(raw, "w") as rf:
            for sample in tqdm(
                samples,
                desc=f"processing and writing {split} samples",
                leave=False,
                disable=not args.progress
            ):
                # clean sample
                sample = Sample(
                    re.sub(
                        r"\s+",
                        " ",
                        sample.question,
                        flags=re.MULTILINE
                    ).strip(),
                    re.sub(
                        r"\s+",
                        " ",
                        sample.sparql,
                        flags=re.MULTILINE
                    ).strip()
                    if has_sparql else None,
                    None if has_sparql else sample.result
                )

                if has_sparql:
                    sparqls, inc = replace_entities_and_properties(
                        sample.sparql,
                        kg,
                        ent_index,
                        prop_index,
                        ent_pattern,
                        prop_pattern,
                        args.version,
                        "in_order" if split == "train" else "only_first"
                    )
                    incomplete += inc
                    total += len(sparqls)
                    if len(sparqls) == 0:
                        continue

                    # same as above, but without replacing
                    # with natural language entities
                    raw_sparql = clean_prefixes(
                        sample.sparql,
                        prefix_patterns,
                        prefixes,
                    )
                    raw_sparql = replace_vars_and_special_tokens(
                        raw_sparql,
                        args.version
                    )

                    for sparql in sparqls:
                        sparql = clean_prefixes(
                            sparql,
                            prefix_patterns,
                            prefixes,
                        )
                        sparql = replace_vars_and_special_tokens(
                            sparql,
                            args.version
                        )
                        tf.write(sparql + "\n")
                        rf.write(raw_sparql + "\n")
                        inf.write(format_query(
                            sample.question,
                            args.version,
                            kg
                        ) + "\n")
                else:
                    tf.write("\n")
                    rf.write(
                        " ".join(r.strip() for r in sample.result)
                        + "\n"
                    )
                    inf.write(format_query(
                        sample.question,
                        args.version,
                        kg
                    ) + "\n")

        print(
            f"Generated {total:,} SPARQL queries while "
            f"processing {len(samples):,} {split} samples with "
            f"{incomplete:,} ({incomplete / len(samples):.2%}) "
            "being incomplete"
        )


if __name__ == "__main__":
    prepare(parse_args())
