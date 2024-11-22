## LLM Text Generation

Generate natural language text with large language models.

### Installation

This project requires Python 3.10 or higher.

#### From PyPI

```bash
pip install llm-text-generation
```

#### From source

```bash
git clone git@github.com:ad-freiburg/llm-text-generation.git
cd llm-text-generation
pip install -e .

```

### Usage

#### From Python

#### From command line

After installation the command `llm-gen` is available in your python environment. 
It lets you use the text generation models directly from the command line.
Below are examples of how to use `llm-gen`. See `llm-gen -h` for all options.

```bash
# print version
llm-gen -v

# list available models
llm-gen -l

# by default llm-gen tries to read stdin, complete the input it got line by line 
# and prints the completed lines back out
# therefore, you can for example use text generation with pipes
echo "The capital of Germany is" | llm-gen
cat "path/to/input/file.txt" | llm-gen > output.txt

# complete a string using
llm-gen -p "The capital of Germany is"

# complete a text file line by line and print the completed lines
llm-gen -f path/to/input/file.txt
# optionally specify an output file path where the completed lines are saved
llm-gen -f path/to/input/file.txt -o output.txt

# start an interactive text generation session
# where your input will be completeed and printed back out
llm-gen -i

# start a text generation server with the following endpoints:
### /models [GET] --> output: available models as json 
### /info [GET] --> output: info about backend as json
### /generate [POST] input: some input text --> output: continuation of the input text
### /live [WS] websocket endpoint for live text generation (only single unbatched requests)
llm-gen --server <config_file>

### OPTIONS
### Pass the following flags to the llm-gen command to customize its behaviour
-m <model_name> # use a different text generation model than the default one 
--cpu # force execution on CPU, by default a GPU is used if available
--progress # display a progress bar (always on when a file is repaired using -f)
-b <batch_size> # specify a different batch size
-batch-max-tokens <batch_max_tokens> # limit batch by a number of tokens and not by number of samples
-u # do not sort the inputs before completeing
-e <experiment_dir> # specify the path to an experiment directory to load the model from 
                    # (equivalent to TextGenerator.from_experiment(experiment_dir) in Python API)
--force-download # force download of the text generation model even if it was already downloaded
--progress # show a progress bar while completing
--report # print a report on the runtime of the model after finishing the completion
```

> Note: When first using `llm-gen` with a pretrained model, the model needs to be downloaded, so depending on
> your internet speed the command might take considerably longer.

> Note: Loading the text generation model requires an initial startup time each time you
> invoke the `llm-gen` command. CPU startup time is around 1s, GPU startup time around 3.5s, so for small
> inputs or files you should probably pass the `--cpu` flag to force CPU execution for best performance.

> See [configs/server.yaml](configs/server.yaml) for an exemplary server configuration file.

### Documentation

#### Use pretrained model

If you just want to use this project to complete whitespaces, this is the recommended way.

```python
from llm_text_generation import TextGenerator

gen = TextGenerator.from_pretrained(
    # pretrained model to load, get all available models from available_models(),
    # if None, loads the default model
    model=None,
    # the device to run the model on
    # ("cuda" by default)
    device="cuda",
    # optional path to a cache directory where downloaded models will be extracted to,
    # if None, we check the env variable TEXT_GENERATION_CACHE_DIR, if it is not set 
    # we use a default cache directory at <install_path>/api/.cache 
    # (None by default)
    cache_dir=None,
    # optional path to a download directory where pretrained models will be downloaded to,
    # if None, we check the env variable TEXT_GENERATION_DOWNLOAD_DIR, if it is not set 
    # we use a default download directory at <install_path>/api/.download
    # (None by default)
    download_dir=None,
    # force download of model even if it already exists in download dir
    # (False by default)
    force_download=False
)
```

When used for the first time with the command line interface or Python API the pretrained model will be automatically downloaded. 
However, you can also download our pretrained models first as zip files, put them in a directory on your local drive 
and set `TEXT_GENERATION_DOWNLOAD_DIR` (or the `download_dir` parameter above) to this directory.

#### Use own model

Once you trained your own model you can use it in the following way.

```python
from llm_text_generation import TextGenerator

gen = TextGenerator.from_experiment(
    # path to the experiment directory that is created by your training run
    experiment_dir="path/to/experiment_dir",
    # the device to run the model on
    # ("cuda" by default)
    device="cuda"
)
```

### Directory structure

The most important directories you might want to look at are:

```
configs -> (example yaml config files for training and server)
src -> (library code used by this project)
```

### Docker

You can also run this project using docker. Build the image using

`docker build -t llm-text-generation .`

If you have an older GPU build the image using

`docker build -t llm-text-generation -f Dockerfile.old .`

By default, the entrypoint is set to the `llm-gen` command, 
so you can use the Docker setup like described [here](#from-command-line) earlier.

You can mount /llm-gen/cache and /llm-gen/download to volumes on your machine, such that
you do not need to download the models every time.

```bash
# complete text
docker run llm-text-generation -p "completethisplease"

# complete file
docker run llm-text-generation -f path/to/file.txt

# start a server
docker run llm-text-generation --server path/to/config.yaml

# with volumes
docker run -v $(pwd)/.cache:/llm-gen/cache -v $(pwd)/.download:/llm-gen/download \
  llm-text-generation -c "completethisplease"

# optional parameters recommended when using a GPU:
# --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864

Note
----
Make sure you have docker version >= 19.03, a nvidia driver
and the nvidia container toolkit installed (see https://github.com/NVIDIA/nvidia-docker)
if you want to run the container with GPU support.
```
