from typing import Iterator
import random
import json

import torch

from text_utils.api.cli import TextProcessingCli
from text_utils.api.processor import TextProcessor

from llm_text_generation import version
from llm_text_generation.api.generator import TextGenerator
from llm_text_generation.api.server import TextGenerationServer


class TextGenerationCli(TextProcessingCli):
    text_processor_cls = TextGenerator
    text_processing_server_cls = TextGenerationServer

    def version(self) -> str:
        return version.__version__

    def setup(self) -> TextProcessor:
        gen = super().setup()
        # perform some additional setup
        assert isinstance(gen, TextGenerator)

        constraint = None
        if self.args.regex is not None:
            constraint = self.args.regex
        elif self.args.regex_file is not None:
            with open(self.args.regex_file, "r") as f:
                constraint = f.read()
        elif self.args.lr1_grammar is not None:
            constraint = (*self.args.lr1_grammar, self.args.lr1_exact)
        elif self.args.lr1_grammar_files is not None:
            with open(self.args.lr1_grammar_files[0], "r") as f1, open(
                self.args.lr1_grammar_files[1], "r"
            ) as f2:
                constraint = (f1.read(), f2.read(), self.args.lr1_exact)

        gen.set_inference_options(
            sample=self.args.sample,
            repeat_penalty=self.args.repeat_penalty,
            temperature=self.args.temperature,
            top_k=self.args.top_k,
            top_p=self.args.top_p,
            beam_width=self.args.beam_width,
            stop_condition=self.args.beam_stop_condition,
            constraint=constraint,
            max_length=self.args.max_length,
            max_new_tokens=self.args.max_new_tokens,
            use_cache=self.args.kv_cache,
            full_outputs=self.args.full_outputs,
        )
        return gen

    def process_iter(
        self, processor: TextProcessor, iter: Iterator[str]
    ) -> Iterator[str]:
        assert isinstance(processor, TextGenerator)
        jsonl_out = self.args.output_format == "jsonl"
        jsonl_in = self.args.input_format == "jsonl"
        yield from (
            json.dumps(output) if jsonl_out else output
            for output in processor.generate(
                (json.loads(item) if jsonl_in else item for item in iter),
                batch_size=self.args.batch_size,
                batch_max_tokens=self.args.batch_max_tokens,
                sort=not self.args.unsorted,
                num_threads=self.args.num_threads,
                show_progress=self.args.progress,
            )
        )


def main():
    parser = TextGenerationCli.parser(
        "Text generator", "Generate natural language text."
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use sampling during decoding; "
        "potentially uses temperature, top-k and top-p",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=1,
        help="Beam width to use for beam search decoding",
    )
    parser.add_argument(
        "--beam-stop-condition",
        type=str,
        choices=["max score", "estimated score", "max outputs"],
        default="estimated score",
        help="Stop condition for beam search decoding; "
        "in practice 'estimated score' is recommended",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Restrict to top k tokens during sampling",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Restrict to top p cumulative probability tokens during sampling",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Temperature to use during sampling",
    )
    parser.add_argument(
        "--repeat-penalty",
        type=float,
        default=None,
        help="Penalty to apply to repeated tokens",
    )
    parser.add_argument(
        "--kv-cache",
        action="store_true",
        help="Whether to use key and value caches during decoding",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Maximum supported input/output length in tokens",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Maximum number of new tokens to generate",
    )
    constraints = parser.add_mutually_exclusive_group()
    constraints.add_argument(
        "-re",
        "--regex",
        type=str,
        default=None,
        help="Regular expression to constrain text generation",
    )
    constraints.add_argument(
        "-ref",
        "--regex-file",
        type=str,
        default=None,
        help="Path to file containing a regular expression to constrain "
        "text generation",
    )
    constraints.add_argument(
        "-lr1",
        "--lr1-grammar",
        nargs=2,
        type=str,
        default=None,
        metavar=("GRAMMAR", "LEXER"),
        help="LR(1) grammar and lexer definitions to constrain text generation",
    )
    constraints.add_argument(
        "-lr1f",
        "--lr1-grammar-files",
        nargs=2,
        type=str,
        default=None,
        metavar=("GRAMMAR_FILE", "LEXER_FILE"),
        help="Paths to files containing a LR(1) grammar and lexer definitions "
        "to constrain text generation",
    )
    parser.add_argument(
        "-lr1e",
        "--lr1-exact",
        action="store_true",
        help="Whether to use exact constraining (respect terminal boundaries) "
        "with LR(1) grammars",
    )
    parser.add_argument(
        "-full",
        "--full-outputs",
        action="store_true",
        help="Whether to return input and generated text as output "
        "(default is only generated text)",
    )
    parser.add_argument(
        "--input-format",
        choices=["text", "jsonl"],
        default="text",
        help="Whether to treat input files as jsonl or text",
    )
    parser.add_argument(
        "--output-format",
        choices=["text", "jsonl"],
        default="text",
        help="Whether to format output as jsonl or text",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Seed for random number generator"
    )
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    # set default device to auto if not set
    # (different from underlying library which sets a single gpu as default)
    args.device = args.device or "auto"
    TextGenerationCli(args).run()
