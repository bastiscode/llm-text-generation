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


def get_infos(entity: str) -> tuple[str, str | None, list[str]]:
    query = f"""
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX schema: <http://schema.org/>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT DISTINCT
?label
(SAMPLE(?desc) AS ?d)
# (GROUP_CONCAT(?alias; SEPARATOR=";") AS ?a)
WHERE {{
   ?x @en@rdfs:label ?label .
   OPTIONAL {{ ?x @en@schema:description ?desc . }}
   OPTIONAL {{ ?x @en@skos:altLabel ?alias . }}
   VALUES ?x {{ wd:{entity} }}
}}
GROUP BY ?label
"""
    # uncomment aliases selction in the query above to use aliases
    result = query_qlever(query)
    binding = result["results"]["bindings"][0]
    label = binding["label"]["value"]
    desc = binding.get("d", {}).get("value", None)
    aliases = binding.get("a", {}).get("value", None)
    if aliases is None:
        aliases = []
    else:
        aliases = aliases.split(";")
    return label, desc, aliases


def get_type_list(entity: str) -> list[tuple[str, str, str | None, list[str]]]:
    query = f"""
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX schema: <http://schema.org/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT DISTINCT
?typ
(SAMPLE(?label) AS ?l)
(SAMPLE(?desc) AS ?d)
(MAX(?score) AS ?s)
# (GROUP_CONCAT(?alias; SEPARATOR=";") AS ?a)
WHERE {{
   wd:{entity} (
     wdt:P31
     | wdt:P31/wdt:P279
     | wdt:P31/wdt:P279/wdt:P279
     | wdt:P31/wdt:P279/wdt:P279/wdt:P279
     | wdt:P279
     | wdt:P279/wdt:P279
     | wdt:P279/wdt:P279/wdt:P279
     | wdt:P279/wdt:P279/wdt:P279/wdt:P279
   ) ?typ .
   ?typ @en@rdfs:label ?label .
   OPTIONAL {{ ?typ @en@schema:description ?desc . }}
   # OPTIONAL {{ ?typ @en@skos:altLabel ?alias . }}
   OPTIONAL {{ ?typ ^schema:about/wikibase:sitelinks ?score . }}
}}
GROUP BY ?typ
ORDER BY DESC(?s)
"""
    result = query_qlever(query)
    types = []
    for binding in result["results"]["bindings"]:
        typ = binding["typ"]["value"]
        if not typ.startswith("http://www.wikidata.org/entity/Q"):
            continue
        typ = typ.split("/")[-1]
        label = binding["l"]["value"]
        desc = binding.get("d", {}).get("value", None)

        # uncomment this and the corresponding parts in the query
        # to use aliases within the type list
        # aliases = binding.get("a", {}).get("value", None)
        # if aliases is None:
        #     aliases = []
        # else:
        #     aliases = aliases.split(";")

        types.append((typ, label, desc, []))
    return types


def _format(
    entity: str,
    label: str,
    desc: str | None,
    aliases: list[str]
) -> str:
    if desc is not None:
        label += f" ({desc})"
    if len(aliases) > 0:
        label += ", also known as " + ", ".join(aliases)
    label += f" {entity}"
    return label


PROMPT = """The natural type of an entity is the superclass \
of the entity that a human would most likely associate with it. \
The natural type is neither too general, nor too specific, and can be used to \
disambiguate the entity from other entities with the same name.

"""


def format_entity(
    entity: str,
    natural_types: list[str] | None = None,
) -> tuple[str, list[str]]:
    label, desc, aliases = get_infos(entity)
    formatted = _format(entity, label, desc, aliases)
    types = get_type_list(entity)
    selectable_types = []
    types_formatted = []
    for typ, label, desc, aliases in types:
        selectable_types.append(f"{label} {typ}")
        types_formatted.append(_format(typ, label, desc, aliases))
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
            label, desc, _ = get_infos(entity)
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


def load_samples(path: str) -> list[tuple[str, list[str]]]:
    samples = []
    with open(path) as inf:
        for line in inf:
            entity, types = line.rstrip("\r\n").split("\t")
            samples.append((entity, types.split()))
    return samples
