import time
from typing import Dict, Any

from flask import Response, jsonify, request, abort

from text_utils.api.server import TextProcessingServer, Error
from text_utils.api.utils import ProgressIterator

from llm_text_generation.api.generator import TextGenerator


class TextGenerationServer(TextProcessingServer):
    text_processor_cls = TextGenerator

    def __init__(self, config: Dict[str, Any]):
        assert "feedback_file" in config, "missing feedback_file in config"
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
            elif "text" not in json:
                return abort(Response("missing texts in json", status=400))

            search_strategy = json.get("search_strategy", "greedy")
            beam_width = json.get("beam_width", 5)
            sample_top_k = json.get("sample_top_k", 5)
            regex = json.get("regex", None)
            cfg = json.get("cfg", {})
            grammar = cfg.get("grammar", None)
            lexer = cfg.get("lexer", None)
            if regex is not None and len(cfg):
                return abort(Response(
                    "can only provide one of regex or cfg", status=400
                ))

            if grammar is not None and lexer is not None:
                cfg = (grammar, lexer)
            else:
                cfg = None

            try:
                with self.text_processor(json["model"]) as gen:
                    if isinstance(gen, Error):
                        return abort(gen.to_response())
                    assert isinstance(gen, TextGenerator)
                    gen.set_inference_options(
                        strategy=search_strategy,
                        beam_width=beam_width,
                        sample_top_k=sample_top_k,
                        regex=regex,
                        cfg=cfg,
                        use_cache=self.use_cache,
                    )
                    start = time.perf_counter()

                    iter = ProgressIterator(
                        ((t, None) for t in json["text"]),
                        size_fn=lambda e: len(e[0].encode("utf8"))
                    )
                    generated = list(gen.generate_iter(
                        iter,
                        batch_size=self.batch_size,
                    ))

                    end = time.perf_counter()
                    b = iter.total_size
                    s = end - start

                    output = {
                        "text": generated,
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
