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

# How to run fast mock unit tests only

`python run.py --test`

# How to run a specific unit test

`python run.py --test --test-file test_model.py`

# How to run only slow integration tests (real model, ~5 minutes)
`python run.py --test integration`

# Hwo to run all tests (unit + integration)
`python run.py --test all`

# How to test api

`python run.py --api_cmd "<API COMMAND>" --api_args "<API_ARGS>"`

Examples:

-   `python run.py --api_cmd "get_encoding" --api_args "hi there"`
-   `python run.py --api_cmd "respond_to_queries" --api_args "hello there"`
-   `python run.py --api_cmd "get_random_validation_example"`

Example of long query:
```
python run.py --api_cmd "respond_to_queries" --api_args "
Question: what portion of the total lease payments is due in the next 12 months?

Context:
Table:
 | amount ( in thousands )
2009 | $ 47760
2010 | 48569
2011 | 49437
2012 | 49959
2013 | 50546
years thereafter | 103890
total | 350161
less : amount representing interest | 54857
present value of net minimum lease payments | $ 295304

entergy corporation and subsidiaries notes to financial statements as of december 31 , 2008 , system energy had future minimum lease payments ( reflecting an implicit rate of 5.13% ( 5.13 % ) ) , which are recorded as long-term debt as follows : amount ( in thousands ) .
"
```
Answer: 13.6%

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
