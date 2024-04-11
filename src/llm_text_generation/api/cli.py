from io import TextIOWrapper
from typing import Iterator

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
            with open(self.args.lr1_grammar_files[0], "r") as f1, \
                    open(self.args.lr1_grammar_files[1], "r") as f2:
                constraint = (f1.read(), f2.read(), self.args.lr1_exact)

        gen.set_inference_options(
            sampling_strategy=self.args.sampling_strategy,
            temperature=self.args.temperature,
            top_k=self.args.top_k,
            top_p=self.args.top_p,
            beam_width=self.args.beam_width,
            constraint=constraint,
            max_length=self.args.max_length,
            use_cache=not self.args.no_kv_cache,
            full_outputs=self.args.full_outputs
        )
        return gen

    def process_iter(
        self,
        processor: TextGenerator,
        iter: Iterator[str]
    ) -> Iterator[str]:
        yield from processor.generate_iter(
            iter,
            batch_size=self.args.batch_size,
            batch_max_tokens=self.args.batch_max_tokens,
            sort=not self.args.unsorted,
            num_threads=self.args.num_threads,
            show_progress=self.args.progress,
        )

    def process_file(
        self,
        processor: TextGenerator,
        input_file: str,
        output_file: str | TextIOWrapper
    ):
        processor.generate_file(
            input_file,
            output_file,
            batch_size=self.args.batch_size,
            batch_max_tokens=self.args.batch_max_tokens,
            sort=not self.args.unsorted,
            num_threads=self.args.num_threads,
            show_progress=self.args.progress,
            format=self.args.file_format
        )


def main():
    parser = TextGenerationCli.parser(
        "Text generator",
        "Generate natural language text."
    )
    parser.add_argument(
        "--sampling-strategy",
        choices=["greedy", "top_k", "top_p"],
        type=str,
        default="greedy",
        help="Sampling strategy to use during decoding"
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=None,
        help="Beam width to use for beam search decoding"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Restrict to top k tokens during sampling"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Restrict to top p cumulative probability tokens during sampling"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature to use during sampling"
    )
    parser.add_argument(
        "--no-kv-cache",
        action="store_true",
        help="Whether to use key and value caches during decoding"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Maximum supported input/output length in tokens"
    )
    constraints = parser.add_mutually_exclusive_group()
    constraints.add_argument(
        "-re",
        "--regex",
        type=str,
        default=None,
        help="Regular expression to constrain text generation"
    )
    constraints.add_argument(
        "-ref",
        "--regex-file",
        type=str,
        default=None,
        help="Path to file containing a regular expression to constrain "
        "text generation"
    )
    constraints.add_argument(
        "-lr1",
        "--lr1-grammar",
        nargs=2,
        type=str,
        default=None,
        help="LR(1) grammar and lexer definitions to constrain text generation"
    )
    constraints.add_argument(
        "-lr1f",
        "--lr1-grammar-files",
        nargs=2,
        type=str,
        default=None,
        help="Paths to files containing a LR(1) grammar and lexer definitions "
        "to constrain text generation"
    )
    parser.add_argument(
        "-lr1e",
        "--lr1-exact",
        action="store_true",
        help="Whether to use exact constraining (respect terminal boundaries) "
        "with LR(1) grammars"
    )
    parser.add_argument(
        "-full",
        "--full-outputs",
        action="store_true",
        help="Whether to return input and generated text as output "
        "(default is only generated text)"
    )
    parser.add_argument(
        "--file-format",
        choices=["jsonl", "lines", "text"],
        default="lines",
        help="Whether to treat input/output files as jsonl, line-separated, "
        "or single piece of text"
    )
    args = parser.parse_args()
    # set default device to auto if not set
    # (different from underlying library which sets a single gpu as default)
    args.device = args.device or "auto"
    TextGenerationCli(args).run()
