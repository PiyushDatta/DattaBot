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
python run.py --api_cmd "respond_to_queries" --api_args "Question: what is the increase observed in the liabilities incurred during 2017 and 2018? Answer: ? Context: Table: | 2018 | 2017 carrying amount at beginning of period | $ 946848 | $ 912926 liabilities incurred | 79057 | 54764 liabilities settled ( 1 ) | -70829 ( 70829 ) | -61871 ( 61871 ) accretion | 36622 | 34708 revisions | -38932 ( 38932 ) | -9818 ( 9818 ) foreign currency translations | 1611 | 16139 carrying amount at end of period | $ 954377 | $ 946848 current portion | $ 26214 | $ 19259 noncurrent portion | $ 928163 | $ 927589 eog utilized average prices per acre from comparable market transactions and estimated discounted cash flows as the basis for determining the fair value of unproved and proved properties , respectively , received in non-cash property exchanges . see note 10 . fair value of debt . at december 31 , 2018 and 2017 , respectively , eog had outstanding $ 6040 million and $ 6390 million aggregate principal amount of senior notes , which had estimated fair values of approximately $ 6027 million and $ 6602 million , respectively . the estimated fair value of debt was based upon quoted market prices and , where such prices were not available , other observable ( level 2 ) inputs regarding interest rates available to eog at year-end . 14 . accounting for certain long-lived assets eog reviews its proved oil and gas properties for impairment purposes by comparing the expected undiscounted future cash flows at a depreciation , depletion and amortization group level to the unamortized capitalized cost of the asset . the carrying values for assets determined to be impaired were adjusted to estimated fair value using the income approach described in the fair value measurement topic of the asc . in certain instances , eog utilizes accepted offers from third-party purchasers as the basis for determining fair value . during 2018 , proved oil and gas properties with a carrying amount of $ 139 million were written down to their fair value of $ 18 million , resulting in pretax impairment charges of $ 121 million . during 2017 , proved oil and gas properties with a carrying amount of $ 370 million were written down to their fair value of $ 146 million , resulting in pretax impairment charges of $ 224 million . impairments in 2018 , 2017 and 2016"
```

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
