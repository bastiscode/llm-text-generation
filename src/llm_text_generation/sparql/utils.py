import re
import copy
import requests
from importlib import resources
from typing import Iterator

from tqdm import tqdm

from text_utils import text, grammar, continuations
from text_utils.constraints import Constraint, ContinuationConstraint
from text_utils.api.table import generate_table

ContIndex = continuations.ContinuationIndex

QLEVER_API = "https://qlever.cs.uni-freiburg.de/api"
QLEVER_URLS = {
    "wikidata": f"{QLEVER_API}/wikidata",
    "dbpedia": f"{QLEVER_API}/dbpedia",
    "freebase": f"{QLEVER_API}/freebase",
}


class KgIndex:
    def __init__(
        self,
        index: dict[str, list[str]],
        redirect: dict[str, str] | None = None,
        prefixes: dict[str, str] | None = None
    ):
        self.index = index
        self.redirect = redirect or {}
        self.prefixes = prefixes or {}
        self.reverse_prefixes = {
            long: short
            for short, long in self.prefixes.items()
        }
        self.long_key_pattern = re.compile(
            "|".join(
                rf"<(?P<long{i}>"
                + re.escape(long)
                + rf")(?P<long{i}_>.+)>"
                for i, long in enumerate(self.prefixes.values())
            )
        )

    def get(
        self,
        key: str,
        default: list[str] | None = None
    ) -> list[str] | None:
        match = self.long_key_pattern.fullmatch(key)
        if match is not None:
            # translate long key to short key
            d = match.groupdict()
            for k, v in d.items():
                if k.endswith("_") or v is None:
                    continue

                key = self.reverse_prefixes[v]
                key = key + d[k + "_"]
                break

        while key not in self.index and self.redirect.get(key, key) != key:
            key = self.redirect[key]

        if key in self.index:
            return self.index[key]

        return default


def load_kg_index(
    index_path: str,
    redirects_path: str | None = None,
    prefixes_path: str | None = None,
    progress: bool = False
) -> KgIndex:
    num_lines, _ = text.file_size(index_path)
    with open(index_path, "r", encoding="utf8") as f:
        index = {}
        for line in tqdm(
            f,
            total=num_lines,
            desc="loading kg index",
            disable=not progress,
            leave=False
        ):
            split = line.split("\t")
            assert len(split) >= 2
            short = split[0].strip()
            obj_names = [n.strip() for n in split[1:]]
            assert short not in index, \
                f"duplicate id {short}"
            index[short] = obj_names

    redirect = {}
    if redirects_path is not None:
        num_lines, _ = text.file_size(redirects_path)
        with open(redirects_path, "r", encoding="utf8") as f:
            for line in tqdm(
                f,
                total=num_lines,
                desc="loading kg redirects",
                disable=not progress,
                leave=False
            ):
                split = line.split("\t")
                assert len(split) >= 2
                short = split[0].strip()
                for redir in split[1:]:
                    redir = redir.strip()
                    assert redir not in redirect, \
                        f"duplicate redirect {redir}, should not happen"
                    redirect[redir] = short

    prefixes = {}
    if prefixes_path is not None:
        num_lines, _ = text.file_size(prefixes_path)
        with open(prefixes_path, "r", encoding="utf8") as f:
            for line in tqdm(
                f,
                total=num_lines,
                desc="loading kg prefixes",
                disable=not progress,
                leave=False
            ):
                split = line.split("\t")
                assert len(split) == 2
                short = split[0].strip()
                full = split[1].strip()
                assert short not in prefixes, \
                    f"duplicate prefix {short}"
                prefixes[short] = full

    return KgIndex(index, redirect, prefixes)


def load_inverse_index(path: str) -> dict[str, list[str]]:
    with open(path, "r", encoding="utf8") as f:
        index = {}
        for line in f:
            split = line.strip().split("\t")
            assert len(split) == 2
            obj_id_1 = split[0].strip()
            obj_id_2 = split[1].strip()
            if obj_id_1 not in index:
                index[obj_id_1] = [obj_id_2]
            else:
                index[obj_id_1].append(obj_id_2)
        return index


def general_prefixes() -> dict[str, str]:
    return {
        "bd:": "http://www.bigdata.com/rdf#",
        "cc:": "http://creativecommons.org/ns#",
        "dct:": "http://purl.org/dc/terms/",
        "geo:": "http://www.opengis.net/ont/geosparql#",
        "hint:": "http://www.bigdata.com/queryHints#",
        "ontolex:": "http://www.w3.org/ns/lemon/ontolex#",
        "owl:": "http://www.w3.org/2002/07/owl#",
        "prov:": "http://www.w3.org/ns/prov#",
        "rdf:": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "rdfs:": "http://www.w3.org/2000/01/rdf-schema#",
        "schema:": "http://schema.org/",
        "skos:": "http://www.w3.org/2004/02/skos/core#",
        "xsd:": "http://www.w3.org/2001/XMLSchema#",
        "wikibase:": "http://wikiba.se/ontology#",
    }


def prefix_pattern(prefixes: dict[str, str]) -> re.Pattern:
    return re.compile(
        "|".join(
            rf"\b(?P<short{i}>"
            + re.escape(short)
            + r")\w+\b"
            + "|"
            + rf"<(?P<long{i}>"
            + re.escape(long)
            + r")\w+>"
            for i, (short, long) in enumerate(prefixes.items())
        )
    )


def load_sparql_parser() -> grammar.LR1Parser:
    sparql_grammar = resources.read_text("deep_sparql.grammar", "sparql.y")
    sparql_lexer = resources.read_text("deep_sparql.grammar", "sparql.l")
    return grammar.LR1Parser(sparql_grammar, sparql_lexer)


def load_sparql_constraint(exact: bool) -> grammar.LR1Constraint:
    sparql_grammar = resources.read_text("deep_sparql.grammar", "sparql.y")
    sparql_lexer = resources.read_text("deep_sparql.grammar", "sparql.l")
    return grammar.LR1Constraint(sparql_grammar, sparql_lexer, exact)


class SPARQLConstraint(Constraint):
    def __init__(
        self,
        sparql_constraint: grammar.LR1Constraint,
        entity_indices: dict[str, ContIndex],
        property_indices: dict[str, ContIndex],
        cont: ContinuationConstraint | None = None
    ) -> None:
        self.sparql_constraint = sparql_constraint
        self.entity_indices = entity_indices
        self.property_indices = property_indices
        self.cont_constraint: ContinuationConstraint | None = cont

    @staticmethod
    def from_vocab_and_indices(
        vocab: list[bytes],
        entity_indices: dict[str, ContIndex],
        property_indices: dict[str, ContIndex],
        exact: bool = False,
        prefix: bytes | None = None
    ) -> 'SPARQLConstraint':
        sparql_grammar = resources.read_text("deep_sparql.grammar", "sparql.y")
        sparql_lexer = resources.read_text("deep_sparql.grammar", "sparql.l")
        sparql = grammar.LR1Constraint(
            sparql_grammar,
            sparql_lexer,
            vocab,
            exact
        )
        if prefix is not None:
            sparql.reset(prefix)
        return SPARQLConstraint(sparql, entity_indices, property_indices)

    def get(self) -> tuple[list[int], bool]:
        if self.cont_constraint is None:
            return self.sparql_constraint.get()

        raise NotImplementedError

    def is_match(self) -> bool:
        return self.sparql_constraint.is_match()

    def next(self, index: int) -> None:
        if self.cont_constraint is not None:
            self.cont_constraint.next(index)
        self.sparql_constraint.next(index)

    def reset(self, input: bytes | None = None) -> None:
        self.sparql_constraint.reset(input)
        raise NotImplementedError

    def clone(self) -> 'SPARQLConstraint':
        return SPARQLConstraint(
            self.sparql_constraint.clone(),
            self.entity_indices,
            self.property_indices,
            None if self.cont_constraint is None else
            self.cont_constraint.clone()
        )


def _parse_to_string(
    parse: dict,
    pretty: bool = False
) -> str:
    def _flatten(parse: dict) -> str:
        if "value" in parse:
            return parse["value"]
        return "".join(_flatten(p) for p in parse["children"])
    return _flatten(parse)


def _find_with_name(
    parse: dict,
    name: str,
    skip: set[str]
) -> dict | None:
    if parse["name"] in skip:
        return None
    elif parse["name"] == name:
        return parse
    for child in parse.get("children", []):
        t = _find_with_name(child, name, skip)
        if t is not None:
            return t
    return None


def _find_all_with_name(
    parse: dict,
    name: str,
    skip: set[str]
) -> Iterator[dict]:
    if parse["name"] in skip:
        return
    elif parse["name"] == name:
        yield parse
        return
    for child in parse.get("children", []):
        yield from _find_all_with_name(child, name, skip)


def _query_type(parse: dict) -> str | None:
    query_type = _find_with_name(parse, "QueryType", set())
    assert query_type is not None
    return query_type["children"][0]["name"]


def _ask_to_select(parse: dict) -> dict:
    parse = copy.deepcopy(parse)

    sub_parse = _find_with_name(parse, "QueryType", set())
    if sub_parse is None:
        return parse

    query = sub_parse["children"][0]
    if query["name"] != "AskQuery":
        return parse

    # we have an ask query
    # find the first var that is not in a subselect
    var_parse = _find_with_name(sub_parse, "Var", set())
    if var_parse is not None:
        # ask query has a var, convert to select
        query["name"] = "SelectQuery"
        # replace ASK terminal with SelectClause
        query["children"][0] = {
            'name': 'SelectClause',
            'children': [
                {
                    'name': 'SELECT',
                    'value': 'SELECT',
                },
                {
                    'name': '*',
                    'value': '*',
                }
            ],
        }
        return parse

    raise NotImplementedError


def prettify_sparql_query(
    sparql: str,
    parser: grammar.LR1Parser,
) -> str:
    parse = parser.parse(sparql)
    return _parse_to_string(parse, True)


_KG_PATTERN = re.compile("kg='(\\w+)'")


def preprocess_natural_language_query(query: str, kg: str) -> str:
    return f"""\
Task:
SPARQL question answering over {kg} knowledge graph

Query:
{query}

SPARQL:
"""


def postprocess_sparql_query(
    sparql: str,
    parser: grammar.LR1Parser,
    entity_indices: dict[str, ContIndex],
    property_indices: dict[str, ContIndex],
    pretty: bool = False,
) -> str:
    try:
        parse = parser.parse(sparql)
    except Exception:
        return sparql

    def _replace_entities_and_properties(parse: dict):
        item_name = parse["name"]
        if item_name == "KgEntity" or item_name == "KgProperty":
            open, obj = parse.pop("children")
            matches = list(_KG_PATTERN.finditer(open["value"]))
            assert len(matches) == 1
            kg = matches[0].group(1)
            if item_name == "KgEntity" and kg in entity_indices:
                index = entity_indices[kg]
            elif kg in property_indices:
                index = property_indices[kg]
            else:
                return

            value = index.get_value(obj["value"].encode("utf8"))
            if value is None:
                return

            parse["value"] = value
            parse["name"] = "PNAME_LN"
        else:
            for child in parse.get("children", []):
                _replace_entities_and_properties(child)

    _replace_entities_and_properties(parse)
    return _parse_to_string(parse, pretty)


class SelectRecord:
    def __init__(
        self,
        value: str | None,
        data_type: str | None,
        label: str | None = None
    ):
        self.value = value
        self.data_type = data_type
        self.label = label

    def __repr__(self) -> str:
        if self.data_type is None:
            return ""
        elif self.data_type == "uri":
            assert self.value is not None
            last = self.value.split("/")[-1]
            if self.label is not None:
                return f"{self.label} ({last})"
            return last
        else:
            return self.label or self.value or ""


AskResult = bool


class SelectResult:
    def __init__(
        self,
        vars: list[str],
        results: list[dict[str, SelectRecord]]
    ):
        self.vars = vars
        self.results = results

    def __len__(self) -> int:
        return len(self.results)

    def __repr__(self) -> str:
        return f"SPARQLResult({self.vars}, {self.results})"


def query_qlever(
    sparql_query: str,
    parser: grammar.LR1Parser,
    kg: str,
    qlever_endpoint: str | None
) -> SelectResult | AskResult:
    parse = parser.parse(sparql_query)
    query_type = _query_type(parse)
    if query_type == "AskQuery":
        sparql_query = _parse_to_string(_ask_to_select(parse))
    elif query_type != "SelectQuery":
        raise ValueError(f"unsupported query type {query_type}")

    if qlever_endpoint is None:
        assert kg in QLEVER_URLS, \
            f"no QLever endpoint for knowledge graph {kg}"
        qlever_endpoint = QLEVER_URLS[kg]

    response = requests.post(
        qlever_endpoint,
        headers={"Content-type": "application/sparql-query"},
        data=sparql_query
    )
    json = response.json()

    if response.status_code != 200:
        msg = json.get("exception", "unknown exception")
        raise RuntimeError(
            f"query {sparql_query} returned with "
            f"status code {response.status_code}:\n{msg}"
        )

    if query_type == "AskQuery":
        return AskResult(len(json["results"]["bindings"]) > 0)

    vars = json["head"]["vars"]
    results = []
    for binding in json["results"]["bindings"]:
        result = {}
        for var in vars:
            if binding is None or var not in binding:
                result[var] = SelectRecord(None, None)
                continue
            value = binding[var]
            result[var] = SelectRecord(
                value["value"],
                value["type"]
            )
        results.append(result)
    return SelectResult(vars, results)


def format_qlever_result(
    result: SelectResult | AskResult,
    max_column_width: int = 80,
) -> str:
    if isinstance(result, AskResult):
        return "yes" if result else "no"

    if len(result) == 0:
        return "no results"

    if len(result.vars) == 0:
        return "no bindings"

    data = []
    for record in result.results:
        data.append([
            str(record[var]) if var in record else "-"
            for var in result.vars
        ])

    return generate_table(
        headers=[result.vars],
        data=data,
        alignments=["left"] * len(result.vars),
        max_column_width=max_column_width,
    )


def query_entities(
    sparql: str,
    parser: grammar.LR1Parser,
    kg: str = "wikidata",
    qlever_endpoint: str | None = None
) -> set[tuple[str, ...]] | None:
    if qlever_endpoint is None:
        assert kg in QLEVER_URLS, \
            f"no QLever endpoint for knowledge graph {kg}"
        qlever_endpoint = QLEVER_URLS[kg]

    try:
        result = query_qlever(sparql, parser, kg, qlever_endpoint)
        if isinstance(result, AskResult):
            return {(f"{result}",)}
        return set(
            tuple(
                r[var].value or "" if var in r else ""
                for var in result.vars
            )
            for r in result.results
        )
    except Exception:
        return None


def calc_f1(
    pred: str,
    target: str,
    parser: grammar.LR1Parser,
    allow_empty_target: bool = True,
    kg: str = "wikidata",
    qlever_endpoint: str | None = None
) -> tuple[float | None, bool, bool]:
    pred_set = query_entities(pred, parser, kg, qlever_endpoint)
    target_set = query_entities(target, parser, kg, qlever_endpoint)
    if pred_set is None or target_set is None:
        return None, pred_set is None, target_set is None
    if len(target_set) == 0 and not allow_empty_target:
        return None, False, True
    if len(pred_set) == 0 and len(target_set) == 0:
        return 1.0, False, False
    tp = len(pred_set.intersection(target_set))
    fp = len(pred_set.difference(target_set))
    fn = len(target_set.difference(pred_set))
    # calculate precision, recall and f1
    if tp > 0:
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f1 = 2 * p * r / (p + r)
    else:
        f1 = 0.0
    return f1, False, False
