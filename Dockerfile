FROM nvcr.io/nvidia/pytorch:23.11-py3

WORKDIR /llm-text-generation

COPY . .

RUN pip install .

ENV TEXT_GENERATION_DOWNLOAD_DIR=/llm-text-generation/download
ENV TEXT_GENERATION_CACHE_DIR=/llm-text-generation/cache
ENV PYTHONWARNINGS="ignore"

ENTRYPOINT ["/usr/local/bin/llm-gen"]
