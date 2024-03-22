import time
from typing import Dict, Any

from flask import Response, jsonify, request, abort

from text_utils.api.server import TextProcessingServer, Error
from text_utils.api.utils import ProgressIterator

from llm_text_generation.api.generator import Chat, TextGenerator


def input_size(text: str | Chat) -> int:
    if isinstance(text, str):
        return len(text.encode("utf8"))

    return sum(
        len(m["text"].encode("utf8"))
        for m in text
    )


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

            texts = json.get("texts", None)
            chats = json.get("chats", None)
            if texts is None and chats is None:
                return abort(Response(
                    "missing texts or chats in json", status=400
                ))
            elif chats is not None and texts is not None:
                return abort(Response(
                    "only one of texts or chats allowed",
                    status=400
                ))
            elif chats is not None:
                texts = chats

            sampling_strategy = json.get("sampling_strategy", "greedy")
            beam_width = json.get("beam_width", None)
            top_k = json.get("top_k", 10)
            top_p = json.get("top_p", 0.95)
            temp = json.get("temperature", 1.0)
            max_length = json.get("max_length", None)
            regex = json.get("regex", None)
            cfg = json.get("cfg", {})
            grammar = cfg.get("grammar", None)
            lexer = cfg.get("lexer", None)
            exact = cfg.get("exact", False)
            if regex is not None and len(cfg):
                return abort(Response(
                    "can only provide one of regex or cfg", status=400
                ))

            if grammar is not None and lexer is not None:
                cfg = (grammar, lexer, exact)
            else:
                cfg = None

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
                        regex=regex,
                        cfg=cfg,
                        use_cache=self.use_cache,
                        max_length=max_length
                    )
                    start = time.perf_counter()

                    iter = ProgressIterator(
                        (t for t in texts),
                        size_fn=input_size
                    )
                    generated = list(gen.generate_iter(
                        iter,
                        batch_size=self.batch_size,
                    ))

                    end = time.perf_counter()
                    b = iter.total_size
                    s = end - start

                    output = {
                        "texts": generated,
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
