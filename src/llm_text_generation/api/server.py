import time
from typing import Dict, Any

from flask import Response, jsonify, request, abort

from text_utils.api.server import TextProcessingServer, Error
from text_utils.api.utils import ProgressIterator

from llm_text_generation.api.generator import Chat, Constraint, TextGenerator


def input_size(ipt: str | Chat | tuple[str | Chat, Constraint]) -> int:
    if isinstance(ipt, tuple):
        ipt, _ = ipt

    if isinstance(ipt, str):
        return len(ipt.encode("utf8"))

    return sum(
        len(m["text"].encode("utf8"))
        for m in ipt
    )


def parse_constraint(json: dict[str, Any] | None) -> Constraint | None:
    if json is None:
        return None
    typ = json["type"]
    if typ == "regex":
        return json["regex"]
    elif typ == "lr1":
        return (json["grammar"], json["lexer"], json.get("exact", False))
    else:
        raise ValueError(f"invalid constraint type: {typ}")


class TextGenerationServer(TextProcessingServer):
    text_processor_cls = TextGenerator

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.use_cache = self.config.get("kv_cache", True)
        self.batch_size = self.config.get("batch_size", 1)

        @self.server.route(f"{self.base_url}/generate", methods=["POST"])
        def _generate() -> Response:
            json = request.get_json()
            if json is None:
                return abort(Response("request body must be json", status=400))
            elif "model" not in json:
                return abort(Response("missing model in json", status=400))
            elif "inputs" not in json:
                return abort(Response("missing inputs in json", status=400))

            inputs = []
            for input in json["inputs"]:
                text = input.get("text", None)
                chat = input.get("chat", None)
                if text is None and chat is None:
                    return abort(Response(
                        "missing text or chat in input", status=400
                    ))
                elif text is not None and chat is not None:
                    return abort(Response(
                        "text and chat are mutually exclusive", status=400
                    ))
                else:
                    ipt = text or chat

                constraint = parse_constraint(input.get("constraint", None))
                if constraint is None:
                    inputs.append(ipt)
                else:
                    inputs.append((ipt, constraint))

            sampling_strategy = json.get("sampling_strategy", "greedy")
            beam_width = json.get("beam_width", None)
            top_k = json.get("top_k", 10)
            top_p = json.get("top_p", 0.95)
            temp = json.get("temperature", 1.0)
            max_length = json.get("max_length", None)

            constraint = parse_constraint(json.get("constraint", None))

            try:
                with self.text_processor(json["model"]) as gen:
                    if isinstance(gen, Error):
                        return abort(gen.to_response())
                    assert isinstance(gen, TextGenerator)
                    gen.set_inference_options(
                        sampling_strategy=sampling_strategy,
                        temperature=temp,
                        top_k=top_k,
                        top_p=top_p,
                        beam_width=beam_width,
                        constraint=constraint,
                        use_cache=self.use_cache,
                        max_length=max_length
                    )
                    start = time.perf_counter()

                    iter = ProgressIterator(
                        (ipt for ipt in inputs),
                        size_fn=input_size
                    )
                    outputs = list(gen.generate_iter(
                        iter,
                        batch_size=self.batch_size,
                    ))

                    end = time.perf_counter()
                    b = iter.total_size
                    s = end - start

                    output = {
                        "outputs": outputs,
                        "runtime": {"b": b, "s": s},
                    }
                    return jsonify(output)

            except Exception as error:
                return abort(
                    Response(
                        f"request failed with unexpected error: {error}",
                        status=500
                    )
                )
