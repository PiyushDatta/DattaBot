# DattaBot

A novel AI agent/model from "scratch" (using libraries like pytorch). MAY try other approaches, like doing it in rust, if it serves value.

# How to install and setup

1. Make sure you have `uv` installed
    - https://docs.astral.sh/uv/getting-started/installation/#installation-methods
2. Setup a virtual environment
    - `python3.10 -m venv <directory>`
    - example: `python3.10 -m venv dattabot_venv`
    - using uv: `uv venv -p 3.10 dattabot_venv`
3. Activate the virtual environment
    - Windows cmd: `dattabot_venv\Scripts\activate.bat`
    - Windows powershell: `dattabot_venv\Scripts\Activate.ps1`
    - Linux/Mac/Other: `source dattabot_venv/bin/activate`
4. Pip install all dependencies
    - `python -m pip install -r requirements.txt`
5. Download the weights
    - `git clone https://huggingface.co/datapi/dattabot-weights`

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

# How to train agent/model

`python run.py --api_cmd "train_agent"`

# How to run agent profiling

-   To use:

    `python run.py --api_cmd profile_agent_training --api_args ''`

-   To analyze:

    `tensorboard --logdir=/home/piydatta/DattaBot/dattabot_data_dir/agent_train_profiler`

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

# Versions and results:

### Base Transformer model (branch: `base_transformer`)

Final results from training standard GPT transformer.

Training run: https://wandb.ai/dattabot_team/DattaBotV1/runs/e3q134eq
Weights: https://huggingface.co/datapi/dattabot-weights/blob/main/base_transformer.pt
Validation loss: 0.083
Training loss: 0.135
Number of train tokens: 17 billion
Eval on HumanEval dataset: 0.0% (0/164)

Eval logs:
07:33:21 AM UTC - INFO - logger.py - dattabot.info - Saving results to /home/piydatta/DattaBot/dattabot_data_dir/humaneval_results.jsonl...
07:33:21 AM UTC - INFO - logger.py - dattabot.info - Running test cases...
07:33:21 AM UTC - INFO - logger.py - dattabot.info - Downloading HumanEval dataset from HuggingFace...
07:33:23 AM UTC - INFO - logger.py - dattabot.info - Loaded 164 HumanEval problems
07:33:23 AM UTC - INFO - logger.py - dattabot.info - Evaluating 164 samples across 164 tasks...
07:33:24 AM UTC - INFO - logger.py - dattabot.info - Completed 164 evaluations
07:33:24 AM UTC - INFO - logger.py - dattabot.info - Total passed: 0/164 (0.0%)
07:33:24 AM UTC - INFO - logger.py - dattabot.info - HumanEval evaluation completed successfully!
07:33:24 AM UTC - INFO - logger.py - dattabot.info - ======================================================================
07:33:24 AM UTC - INFO - logger.py - dattabot.info - ✓ HUMANEVAL evaluation completed!
07:33:24 AM UTC - INFO - logger.py - dattabot.info - Results saved to: /home/piydatta/DattaBot/dattabot_data_dir/humaneval_results.jsonl
07:33:24 AM UTC - INFO - logger.py - dattabot.info - ======================================================================
07:33:24 AM UTC - INFO - logger.py - dattabot.info -
======================================================================
07:33:24 AM UTC - INFO - logger.py - dattabot.info - HUMANEVAL EVALUATION RESULTS
07:33:24 AM UTC - INFO - logger.py - dattabot.info - ======================================================================
07:33:24 AM UTC - INFO - logger.py - dattabot.info - Problems Evaluated: 164
07:33:24 AM UTC - INFO - logger.py - dattabot.info - Samples per Problem: 1
07:33:24 AM UTC - INFO - logger.py - dattabot.info - ----------------------------------------------------------------------
07:33:24 AM UTC - INFO - logger.py - dattabot.info - SCORES:
07:33:24 AM UTC - INFO - logger.py - dattabot.info - pass@1: 0.00% [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]
07:33:24 AM UTC - INFO - logger.py - dattabot.info - ----------------------------------------------------------------------
07:33:24 AM UTC - INFO - logger.py - dattabot.info - Results saved to: /home/piydatta/DattaBot/dattabot_data_dir/humaneval_results.jsonl
07:33:24 AM UTC - INFO - logger.py - dattabot.info - ======================================================================

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
