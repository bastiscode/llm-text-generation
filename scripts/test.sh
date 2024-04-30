#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --job-name=test
#SBATCH --open-mode=append
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --time=24:00:00

experiment=${EXPERIMENT?"env var EXPERIMENT not found"}
input=${INPUT?"env var INPUT not found"}
output=${OUTPUT?"env var OUTPUT not found"}
endpoint=${ENDPOINT:-https://qlever.cs.uni-freiburg.de/api/wikidata}
strategy=${STRATEGY:-beam}
beam_width=${BEAM_WIDTH:-5}
batch_size=${BATCH_SIZE:-16}
subgraph=${SUBGRAPH:-false}
max_length=${MAX_LENGTH:-512}
ent_index=${ENT_INDEX:-data/prefix-index/wikidata-entities.bin}
prop_index=${PROP_INDEX:-data/prefix-index/wikidata-properties.bin}

cmd="deep-sparql -e $experiment -f $input -o $output \
--search-strategy $strategy --beam-width $beam_width \
-E $ent_index -P $prop_index \
-b $batch_size --max-length $max_length \
--qlever-endpoint $endpoint \
--progress --report"

if [[ $subgraph == true ]]; then
  cmd="$cmd --subgraph-constraining"
fi

echo "Running $cmd"
bash -c "$cmd"
