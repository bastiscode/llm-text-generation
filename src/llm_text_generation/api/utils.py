from typing import Any
import logging

from text_utils.tensorboard import TensorboardMetric
from text_utils import data

import torch
from torch.utils.tensorboard import SummaryWriter


Chat = list[dict[str, str]]


class InputOutputLogger(TensorboardMetric):
    def __init__(self, prefix: str):
        self.items = []
        self.prefix = prefix

    def set_values(self, items: list[data.TrainItem], outputs: torch.Tensor):
        self.items = items

    def get_input_output(self) -> str:
        return "\n\n".join(
            f"input:\n{item.data.input}\n\noutput:\n{item.data.target}"
            for item in self.items
        )

    def log_tensorboard(self, writer: SummaryWriter, step: int):
        writer.add_text(f"{self.prefix}/input_output", self.get_input_output(), step)

    def log_info(self, logger: logging.Logger, step: int):
        logger.info(
            f"[step {step}] {self.prefix}/input_output:\n" f"{self.get_input_output()}"
        )


def format_chat(input: str | Chat, chat_template: dict[str, Any]) -> str:
    if isinstance(input, str):
        input = [{"role": "user", "text": input}]

    assert "user" in chat_template["roles"], "chat template must have a user role"
    assert (
        "assistant" in chat_template["roles"]
    ), "chat template must have an assistant role"

    s: str = chat_template.get("start", "")

    last_partial = False
    for i, message in enumerate(input):
        role = message["role"]
        text = message["text"]
        assert role in chat_template["roles"], f"role {role} not in chat template"
        template = chat_template["roles"][role]
        if message.get("partial", False):
            assert i == len(input) - 1, "partial message not last"
            pos = template.find("{text}")
            s += template[:pos] + text
            last_partial = True
        else:
            s += template.replace("{text}", text)

    if not last_partial:
        s += chat_template.get("end", "")

    return s
