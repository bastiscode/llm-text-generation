WD_ENT=data/kg-index/wikidata-entities-index.tsv
WD_ENT_REDIR=data/kg-index/wikidata-entities-index.redirects.tsv
WD_ENT_PREFIXES=data/kg-index/wikidata-entities-index.prefixes.tsv

WD_PROP=data/kg-index/wikidata-properties-index.tsv
WD_PROP_PREFIXES=data/kg-index/wikidata-properties-index.prefixes.tsv

ENT_SUFFIX="</kge>"
PROP_SUFFIX="</kgp>"

.PHONY: all data indices
all: data indices

data:
	@echo "Preparing simple questions"
	@python scripts/prepare_data.py \
	--wikidata-simple-questions third_party/KGQA-datasets/simple_wikidata_qa \
	--output data/wikidata-simplequestions \
	--entity-index $(WD_ENT) \
	--entity-redirects $(WD_ENT_REDIR) \
	--entity-prefixes $(WD_ENT_PREFIXES) \
	--property-index $(WD_PROP) \
	--property-prefixes $(WD_PROP_PREFIXES) \
	--progress
	@echo "Preparing lc quad wikidata"
	@python scripts/prepare_data.py \
	--lc-quad2-wikidata third_party/KGQA-datasets/lcquad_v2 \
	--output data/wikidata-lcquad2 \
	--entity-index $(WD_ENT) \
	--entity-redirects $(WD_ENT_REDIR) \
	--entity-prefixes $(WD_ENT_PREFIXES) \
	--property-index $(WD_PROP) \
	--property-prefixes $(WD_PROP_PREFIXES) \
	--progress
	@echo "Preparing qald 10"
	@python scripts/prepare_data.py \
	--qald-10 third_party/KGQA-datasets/qald/qald-10.py \
	--output data/wikidata-qald10 \
	--entity-index $(WD_ENT) \
	--entity-redirects $(WD_ENT_REDIR) \
	--entity-prefixes $(WD_ENT_PREFIXES) \
	--property-index $(WD_PROP) \
	--property-prefixes $(WD_PROP_PREFIXES) \
	--progress
	@echo "Preparing mcwq"
	@python scripts/prepare_data.py \
	--mcwq data/raw/mcwq \
	--output data/wikidata-mcwq \
	--entity-index $(WD_ENT) \
	--entity-redirects $(WD_ENT_REDIR) \
	--entity-prefixes $(WD_ENT_PREFIXES) \
	--property-index $(WD_PROP) \
	--property-prefixes $(WD_PROP_PREFIXES) \
	--progress
	@echo "Preparing qa wiki"
	@python scripts/prepare_data.py \
	--qa-wiki data/raw/qa_wiki/qa_wiki.tsv \
	--output data/wikidata-qa-wiki \
	--entity-index $(WD_ENT) \
	--entity-redirects $(WD_ENT_REDIR) \
	--entity-prefixes $(WD_ENT_PREFIXES) \
	--property-index $(WD_PROP) \
	--property-prefixes $(WD_PROP_PREFIXES) \
	--progress

indices:
	@echo "Creating wikidata continuation indices"
	@tu.create_continuation_index \
	--input-file data/kg-index/wikidata-properties-index.tsv \
	--output-file data/art-index/wikidata-properties.bin \
	--common-suffix $(PROP_SUFFIX)
	@tu.create_continuation_index \
	--input-file data/kg-index/wikidata-entities-index.tsv \
	--output-file data/art-index/wikidata-entities.bin \
	--common-suffix $(ENT_SUFFIX)
	@tu.create_continuation_index \
	--input-file data/kg-index/wikidata-entities-index.small.tsv \
	--output-file data/art-index/wikidata-entities.small.bin \
	--common-suffix $(ENT_SUFFIX)

