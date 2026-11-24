# DattaBot

A novel AI agent/model from "scratch" (using libraries like pytorch). MAY try other approaches, like doing it in rust, if it serves value.

# How to install and setup (NVIDIA/AMD/APPLE)

1. Make sure you have `uv` installed
    - https://docs.astral.sh/uv/getting-started/installation/#installation-methods
2. Setup a virtual environment
    - `python3.10 -m venv <directory>`
    - example: `python3.10 -m venv dattabot_venv`
    - using uv: `uv venv dattabot_venv`
3. Activate the virtual environment
    - Windows cmd: `dattabot_venv\Scripts\activate.bat`
    - Windows powershell: `dattabot_venv\Scripts\Activate.ps1`
    - Linux/Mac/Other: `source dattabot_venv/bin/activate`
4. Pip install all dependencies
    - `export UV_PROJECT_ENVIRONMENT=./dattabot_venv && uv sync`
    - `uv pip install -r pyproject.toml`
5. Download the weights
    - `git clone https://huggingface.co/datapi/dattabot-weights`

# How to install and setup (TPU)
1. Download torch, torch_xla, and dependencies
    - `pip install torch torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html`
    - `pip install hydra-core`
    - `pip install tiktoken`
    ```
    gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --project=${PROJECT_ID} --worker=all --command="sudo apt-get update && sudo apt-get install libopenblas-dev -y && pip install numpy"

    gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --project=${PROJECT_ID} --worker=all --command="pip install torch torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html"

    gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --project=${PROJECT_ID} --worker=all --command="pip install hydra-core && pip install tiktoken && pip install tqdm && pip install safetensors && pip install psutil && pip install wandb && pip install datasets"
    ```
2. Download the weights
```
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=all \
  --command="cd ~/DattaBot && git clone https://oauth2:${HF_TOKEN}@huggingface.co/datapi/dattabot-weights"
```

# How to interact with bot (run either of them) via client

`python run.py`

For information on python virtual environment: [notes.md](notes.md)

For manual: Use `--help`, like so: `python run.py --help`

# Dependencies

-   Python version >= `3.10`
-   Cuda version >= `11.8`
-   Pytorch version >= `2.2.1`
-   Python packages: `requirements.txt`
    -   `pip install -r requirements.txt`

# Motivation

-   Have fun learning more about AI
-   Read/learn/implement papers (and any form of research)
-   Contribute back all of this research and implementation back to the open source community

# Goals

-   Fully safe and open source model - weights and all
-   Accessible to anyone and everyone, and can run on their own machines (don't need to hit an endpoint or give data away to a third party)
-   Make it the one of the best agents/models in the world
-   To make the above true, we have to find novel ideas

# License

**The MIT License**

-   Permissions (you are authorized to):

    -   Private use
    -   Commercial use
    -   Modification
    -   Distribution

-   Limitations:
    -   Liability
    -   Warranty

Feel free to contribute or fork, or use for money, or whatever. We don't mind, just have fun! We are not not liable for anything and will not provide any warranty for this.

# Note

If you are a cloud provider and just selling access to this model (on your own machines) to your customers, we would **appreciate** a royalty or some sort of support - but obviously **not required**.
