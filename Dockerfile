FROM nvcr.io/nvidia/pytorch:23.11-py3

WORKDIR /deep-sparql

COPY . .

RUN pip install .

ENV SPARQL_GENERATION_DOWNLOAD_DIR=/deep-sparql/download
ENV SPARQL_GENERATION_CACHE_DIR=/deep-sparql/cache
ENV PYTHONWARNINGS="ignore"

ENTRYPOINT ["/usr/local/bin/deep-sparql"]
