import time
import json as J
from typing import Dict, Any

from flask import Response, jsonify, request, abort
from flask_socketio import SocketIO, send, disconnect

from text_utils.api.server import TextProcessingServer, Error

from llm_text_generation.api.generator import Const, TextGenerator


def parse_constraint(json: dict[str, Any] | None) -> Const | None:
    if json is None:
        return None
    typ = json["type"]
    if typ == "regex":
        return json["regex"]
    elif typ == "lr1":
        return (json["grammar"], json["lexer"], json.get("exact", False))
    else:
        raise ValueError(f"invalid constraint type: {typ}")


def inference_options_from_json(json: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample": json.get("sample", False),
        "temperature": json.get("temperature"),
        "top_k": json.get("top_k"),
        "top_p": json.get("top_p"),
        "beam_width": json.get("beam_width", 1),
        "repeat_penalty": json.get("repeat_penalty"),
        "constraint": parse_constraint(json.get("constraint")),
        "max_new_tokens": json.get("max_new_tokens"),
    }


class TextGenerationServer(TextProcessingServer):
    text_processor_cls = TextGenerator

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        @self.server.route("/generate", methods=["POST"])
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
                text = input.get("text")
                chat = input.get("chat")
                if text is None and chat is None:
                    return abort(Response("missing text or chat in input", status=400))
                elif text is not None and chat is not None:
                    return abort(
                        Response("text and chat are mutually exclusive", status=400)
                    )
                else:
                    ipt = text or chat

                constraint = parse_constraint(input.get("constraint"))
                if constraint is None:
                    inputs.append(ipt)
                else:
                    inputs.append((ipt, constraint))

            inference_options = inference_options_from_json(json)

            try:
                name = json["model"]
                with self.text_processor(name) as gen:
                    if isinstance(gen, Error):
                        return abort(gen.to_response())
                    assert isinstance(gen, TextGenerator)
                    gen.set_inference_options(**inference_options)
                    start = time.perf_counter()

                    idx = self.name_to_idx[name]
                    model_cfg = self.config["models"][idx]
                    outputs = list(
                        gen.generate(
                            inputs,
                            model_cfg.get(
                                "batch_size", self.config.get("batch_size", 1)
                            ),
                        )
                    )

                    end = time.perf_counter()
                    b = sum(len(output.encode()) for output in outputs)
                    s = end - start

                    output = {
                        "outputs": outputs,
                        "runtime": {"b": b, "s": s},
                    }
                    return jsonify(output)

            except Exception as error:
                return abort(
                    Response(
                        f"request failed with unexpected error: {error}", status=500
                    )
                )

        self.socketio = SocketIO(
            self.server, path="live", cors_allowed_origins=self.allow_origin
        )

        self.connections = set()

        @self.socketio.on("connect")
        def _connect() -> None:
            self.connections.add(request.sid)  # type: ignore

        @self.socketio.on("disconnect")
        def _disconnect() -> None:
            self.connections.remove(request.sid)  # type: ignore

        @self.socketio.on("message")
        def _generate_live(data) -> None:
            try:
                json = J.loads(data)

                if "model" not in json:
                    send(J.dumps({"error": "missing model in json"}))
                    return

                text = json.get("text")
                chat = json.get("chat")
                if text is None and chat is None:
                    send(J.dumps({"error": "missing text or chat in input"}))
                    return
                elif text is not None and chat is not None:
                    send(J.dumps({"error": "text and chat are mutually exclusive"}))
                    return
                else:
                    ipt = text or chat

                inference_options = inference_options_from_json(json)

                with self.text_processor(json["model"]) as gen:
                    if isinstance(gen, Error):
                        send(J.dumps({"error": gen.msg}))
                        return

                    assert isinstance(gen, TextGenerator)
                    gen.set_inference_options(**inference_options)

                    start = time.perf_counter()
                    for text in gen.generate_live(ipt):  # type: ignore
                        if request.sid not in self.connections:
                            # early explicit disconnect by client
                            return

                        send(
                            J.dumps(
                                {
                                    "output": text,
                                    "runtime": {
                                        "b": len(text.encode()),
                                        "s": time.perf_counter() - start,
                                    },
                                }
                            )
                        )

            except Exception as error:
                send(
                    J.dumps({"error": f"request failed with unexpected error: {error}"})
                )

            finally:
                disconnect()

    def run(self) -> None:
        self.socketio.run(
            self.server,
            "0.0.0.0",
            self.port,
            debug=False,
            use_reloader=False,
            log_output=False,
        )
