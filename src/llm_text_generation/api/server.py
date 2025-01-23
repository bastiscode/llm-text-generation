import time
from typing import Any, Iterator, TypeVar

from fastapi import WebSocket, WebSocketDisconnect, status
from fastapi.responses import Response
from pydantic import BaseModel, ValidationError
from text_utils.api.server import Error, TextProcessingServer
from websockets.exceptions import ConnectionClosed

from llm_text_generation.api.generator import TextGenerator

HTTP_TO_WS = {
    status.HTTP_400_BAD_REQUEST: 1008,
    status.HTTP_503_SERVICE_UNAVAILABLE: 1013,
    status.HTTP_500_INTERNAL_SERVER_ERROR: 1011,
}


class Input(BaseModel):
    text: str | list[dict[str, Any]]
    constraint: str | tuple[str, str, bool] | None = None


class InferenceOptions(BaseModel):
    sample: bool = True
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    min_p: float | None = 0.1
    beam_width: int = 1
    repeat_penalty: float | None = 1.05
    constraint: str | tuple[str, str, bool] | None = None
    max_new_tokens: int | None = None


class GenerateRequest(BaseModel):
    model: str
    inputs: list[Input]
    inference_options: InferenceOptions = InferenceOptions()


class LiveRequest(BaseModel):
    model: str
    input: Input
    inference_options: InferenceOptions = InferenceOptions()


T = TypeVar("T")


class AsyncIterator:
    def __init__(self, iterator: Iterator[T]):
        self.iterator = iterator

    def __aiter__(self) -> Iterator[T]:
        return self

    async def __anext__(self) -> T:
        try:
            return next(self.iterator)
        except StopIteration:
            raise StopAsyncIteration


class TextGenerationServer(TextProcessingServer):
    text_processor_cls = TextGenerator

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        @self.server.post("/generate")
        def generate(request: GenerateRequest) -> Response:
            self.logger.info(f"Received generate request:\n{request.dict()}")
            try:
                with self.get_text_processor(request.model) as gen:
                    if isinstance(gen, Error):
                        return gen.to_response()

                    assert isinstance(gen, TextGenerator)

                    model_cfg = self.model_cfgs[request.model]
                    batch_size = max(
                        1,
                        model_cfg.get("batch_size", self.config.get("batch_size", 1)),
                    )
                    # override max new tokens
                    request.inference_options.max_new_tokens = min(
                        gen.max_length,
                        request.inference_options.max_new_tokens or gen.max_length,
                    )
                    gen.set_inference_options(**request.inference_options.dict())

                    inputs = [
                        (input.text, input.constraint) for input in request.inputs
                    ]

                    start = time.perf_counter()
                    outputs = list(gen.generate(inputs, batch_size))
                    end = time.perf_counter()

                    b = sum(len(output.encode()) for output in outputs)
                    s = end - start

                    return {
                        "outputs": outputs,
                        "runtime": {"b": b, "s": s},
                    }

            except Exception as error:
                return Error(
                    f"Request failed with unexpected error: {error}",
                    status.HTTP_500_INTERNAL_SERVER_ERROR,
                ).to_response()

        @self.server.websocket("/live")
        async def live(websocket: WebSocket):
            try:
                await websocket.accept()

                data = await websocket.receive_json()
                request = LiveRequest(**data)

                async with self.get_text_processor(request.model) as gen:
                    if isinstance(gen, Error):
                        await websocket.close(
                            HTTP_TO_WS.get(gen.status_code, 1011),
                            gen.error,
                        )
                        return

                    assert isinstance(gen, TextGenerator)
                    # override max new tokens
                    request.inference_options.max_new_tokens = min(
                        gen.max_length,
                        request.inference_options.max_new_tokens or gen.max_length,
                    )
                    gen.set_inference_options(**request.inference_options.dict())

                    start = time.perf_counter()
                    async for text in AsyncIterator(
                        gen.generate_live(
                            request.input.text,
                            request.input.constraint,
                        )
                    ):  # type: ignore
                        await websocket.send_json(
                            {
                                "output": text,
                                "runtime": {
                                    "b": len(text.encode()),
                                    "s": time.perf_counter() - start,
                                },
                            }
                        )

                await websocket.close()

            except WebSocketDisconnect as e:
                self.logger.info(f"Client disconnected: {e.code}, {e.reason}")

            except ConnectionClosed as e:
                self.logger.info(f"Connection closed: {e}")

            except ValidationError as e:
                self.logger.error(f"Request failed with validation error: {e}")
                await websocket.close(HTTP_TO_WS[status.HTTP_400_BAD_REQUEST])

            except Exception as e:
                self.logger.error(f"Request failed with unexpected error: {e}")
                await websocket.close(HTTP_TO_WS[status.HTTP_500_INTERNAL_SERVER_ERROR])
