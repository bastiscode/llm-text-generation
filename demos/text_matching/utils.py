import requests
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
(GROUP_CONCAT(?alias; SEPARATOR=";") AS ?a)
WHERE {{
   ?x @en@rdfs:label ?label .
   OPTIONAL {{ ?x @en@schema:description ?desc . }}
   OPTIONAL {{ ?x @en@skos:altLabel ?alias . }}
   VALUES ?x {{ wd:{entity} }}
}}
GROUP BY ?label
"""
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


def get_regex() -> str:
    return r"[1-4]"


PROMPT = """Given a natural language query, the entities contained in it and \
a collection of texts, we assign a score from 1 (does not match) to 4 \
(definitely matches) to the query for how well it matches the text collection \
overall.
If there is only one text in the collection, the score describes how well \
this particular text fits to the query.

"""


def format_sample(
    query: str,
    entities: list[str],
    text_collection: list[str],
    score: int | None = None
) -> str:
    entity_strs = []
    for entity in entities:
        label, desc, aliases = get_infos(entity)
        if desc is not None:
            label += f" ({desc})"
        if len(aliases) > 0:
            label += ", also known as " + ", ".join(aliases)
        entity_strs.append(label)
    entity_str = "\n".join(entity_strs)
    texts = "\n".join(text_collection)
    s = f"""Query:
{query}

Contained entities:
{entity_str}

Text collection:
{texts}

Score:
"""
    if score is not None:
        s += f"{score}\n"
    return s


def get_prompt(
    query: str,
    entities: list[str],
    texts: list[str],
    examples: list[str]
) -> str:
    formatted = format_sample(query, entities, texts)

    prompt = PROMPT
    if len(examples) > 0:
        prompt += "\n\n".join(examples) + "\n\n"
    prompt += formatted

    return prompt


def run_model(
    texts: list[str],
    model: str
) -> list[int] | list[list[int]]:
    data = {
        "model": model,
        "texts": texts,
        "constraint": {
            "type": "regex",
            "regex": get_regex(),
        },
        "beam_width": 4,
        "sampling_strategy": "greedy",
    }
    response = requests.post(
        MODEL_URL,
        headers={"Content-type": "application/json"},
        data=dumps(data)
    )
    json = response.json()
    outputs = []
    for t in json["texts"]:
        score = int(t.strip())
        outputs.append(score)
    return outputs
