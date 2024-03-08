from io import TextIOWrapper
from typing import Iterator, Optional, Union

from text_utils.api.cli import TextProcessingCli
from text_utils.api.processor import TextProcessor
from text_utils import data

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

        gen.set_inference_options(
            strategy=self.args.search_strategy,
            beam_width=self.args.beam_width,
            sample_top_k=self.args.sample_top_k,
            regex=self.args.regex,
            regex_file=self.args.regex_file,
            cfg=(*self.args.lr1_grammar, self.args.lr1_exact),
            cfg_files=(*self.args.lr1_grammar_files, self.args.lr1_exact),
            max_length=self.args.max_length,
            use_cache=not self.args.no_kv_cache,
            full_outputs=self.args.full_outputs
        )
        return gen

    def process_iter(
        self,
        processor: TextGenerator,
        input: Iterator[data.InferenceData]
    ) -> Iterator[data.InferenceData]:
        # collapse all text from iterator into a single sample
        text: str = "\n".join(item.text for item in input)

        yield from processor.generate_iter(
            iter([text]),
            sort=not self.args.unsorted,
            num_threads=self.args.num_threads,
            show_progress=self.args.progress,
            raw=True
        )

    def process_file(
        self,
        processor: TextGenerator,
        path: str,
        _: Optional[str],
        out_file: Union[str, TextIOWrapper]
    ):
        processor.generate_file(
            path,
            out_file,
            not self.args.unsorted,
            self.args.num_threads,
            show_progress=self.args.progress
        )


def main():
    parser = TextGenerationCli.parser(
        "Text generator",
        "Generate natural language text."
    )
    parser.add_argument(
        "--search-strategy",
        choices=["greedy", "beam", "sample"],
        type=str,
        default="greedy",
        help="Search strategy to use during decoding"
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=5,
        help="Beam width to use for beam search decoding"
    )
    parser.add_argument(
        "--sample-top-k",
        type=int,
        default=5,
        help="Sample from top k tokens during sampling decoding"
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
    args = parser.parse_args()
    # set default device to auto if not set
    # (different from underlying library which sets a single gpu as default)
    args.device = args.device or "auto"
    TextGenerationCli(args).run()
