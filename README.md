# DattaBot

A novel AI agent/model from "scratch" (using libraries like pytorch). MAY try other approaches, like doing it in rust, if it serves value.

# How to install and setup

1. Setup a virtual environment
    - `python3.10 -m venv <directory>`
    - example: `python3.10 -m venv dattabot_venv`
2. Activate the virtual environment
    - Windows cmd: `dattabot_venv\Scripts\activate.bat`
    - Windows powershell: `dattabot_venv\Scripts\Activate.ps1`
    - Linux/Mac/Other: `source dattabot_venv/bin/activate`
3. Pip install all dependencies
    - `python -m pip install -r requirements.txt`
4. Git LFS install and pull the weights (DATTABOT_VERSION_x_x_weights.pt)
    - `git lfs install && git lfs pull`

# How to interact with bot (run either of them) via client

`python run.py`

For information on python virtual environment: [notes.md](notes.md)

For manual: Use `--help`, like so: `python run.py --help`

# How to run unit tests

`python run.py --test`

# How to run a specific unit test

`python run.py --test test_smoke_test.py`

# How to test api

`python run.py --api_cmd "<API COMMAND>" --api_args "<API_ARGS>"`

Examples:

-   `python run.py --api_cmd "get_encoding" --api_args "hi there"`
-   `python run.py --api_cmd "respond_to_queries" --api_args "hello there"`
-   `python run.py --api_cmd "get_random_validation_example"`


# How to train agent/model

-   `python run.py --api_cmd "train_agent"`

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

-   Fully open source - weights and all
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
