import requests
from re import escape
from json import dumps

QLEVER_URL = "https://qlever.cs.uni-freiburg.de/api/wikidata-sebastian"
MODEL_URL = "https://ad-llm.cs.uni-freiburg.de/api/generate"


def query_qlever(query: str) -> dict:
    response = requests.post(
        QLEVER_URL,
        headers={"Content-type": "application/sparql-query"},
        data=query
    )
    return response.json()


def get_entity_label_and_desc(entity: str) -> tuple[str, str | None]:
    query = f"""
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX schema: <http://schema.org/>
SELECT DISTINCT ?label ?desc WHERE {{
   wd:{entity} @en@rdfs:label ?label .
   OPTIONAL {{ wd:{entity} @en@schema:description ?desc . }}
}}
"""
    result = query_qlever(query)
    binding = result["results"]["bindings"][0]
    label = binding["label"]["value"]
    desc = binding.get("desc", {}).get("value", None)
    return label, desc


def get_types_for_entity(entity: str) -> list[tuple[str, str, str | None]]:
    query = f"""
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX schema: <http://schema.org/>
PREFIX wikibase: <http://wikiba.se/ontology#>
SELECT DISTINCT ?typ ?label ?desc WHERE {{
   wd:{entity} (
     wdt:P31
     |wdt:P31/wdt:P279
     |wdt:P31/wdt:P279/wdt:P279
     |wdt:P31/wdt:P279/wdt:P279/wdt:P279
     |wdt:P279
     |wdt:P279/wdt:P279
     |wdt:P279/wdt:P279/wdt:P279
     |wdt:P106
   ) ?typ .
   ?typ @en@rdfs:label ?label .
   OPTIONAL {{ ?typ @en@schema:description ?desc . }}
   OPTIONAL {{ ?typ ^schema:about/wikibase:sitelinks ?score . }}
}} ORDER BY DESC(?score)
"""
    result = query_qlever(query)
    types = []
    for binding in result["results"]["bindings"]:
        typ = binding["typ"]["value"]
        if not typ.startswith("http://www.wikidata.org/entity/Q"):
            continue
        typ = typ.split("/")[-1]
        label = binding["label"]["value"]
        desc = binding.get("desc", {}).get("value", None)
        types.append((typ, label, desc))
    return types


def _format(entity: str, label: str, desc: str | None) -> str:
    if desc is None:
        return f"{label} {entity}"
    return f"{label} ({desc}) {entity}"


PROMPT = """The natural type of an entity is the superclass \
of the entity that a human would most likely associate with it. \
The natural type is neither too general, nor too specific, and can be used to \
disambiguate the entity from other entities with the same name.

"""


def format_entity(
    entity: str,
    natural_types: list[str] | None = None,
) -> tuple[str, list[str]]:
    label, desc = get_entity_label_and_desc(entity)
    formatted = _format(entity, label, desc)
    types = get_types_for_entity(entity)
    selectable_types = []
    types_formatted = []
    for typ, label, desc in types:
        selectable_types.append(f"{label} {typ}")
        types_formatted.append(_format(typ, label, desc))
    types_formatted = "\n".join(types_formatted)
    s = f"""Entity:
{formatted}

List of types:
{types_formatted}

Natural type:
"""
    if natural_types is not None:
        types_formatted = []
        for entity in natural_types:
            label, desc = get_entity_label_and_desc(entity)
            types_formatted.append(f"{label} {entity}")
        formatted = "\n".join(types_formatted)
        s += formatted + "\n"
    return s, selectable_types


def get_prompt_and_regex(
    entity: str,
    examples: list[str],
) -> tuple[str, str]:
    formatted, selectable = format_entity(entity)

    prompt = PROMPT
    if len(examples) > 0:
        prompt += "\n\n".join(examples) + "\n\n"
    prompt += formatted

    regex = "(" + "|".join(escape(s) for s in selectable) + ")"
    return prompt, regex


def run_model(
    text: str,
    regex: str,
    model: str
) -> list[tuple[str, str]]:
    data = {
        "model": model,
        "texts": [text],
        "regex": regex,
        "beam_width": 1,
        "sampling_strategy": "greedy",
    }
    response = requests.post(
        MODEL_URL,
        headers={"Content-type": "application/json"},
        data=dumps(data)
    )
    json = response.json()
    outputs = []
    for t in json["texts"][0].split("\n"):
        if t == "":
            continue
        splits = t.strip().split(" ")
        outputs.append((" ".join(splits[:-1]), splits[-1]))
    return outputs


def load_benchmark(path: str) -> list[tuple[str, list[str]]]:
    samples = []
    with open(path) as inf:
        for line in inf:
            entity, types = line.rstrip("\r\n").split("\t")
            samples.append((entity, [t.strip() for t in types.split(" ")]))
    return samples
