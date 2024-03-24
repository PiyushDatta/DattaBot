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
    - `pip install -r requirements.txt`

# How to interact with bot (run either of them) via client

For windows:
`.\run.bat`

For linux/mac:
`./run.sh`

Run python client:
`python client.py`

For information on python virtual environment: [notes.md](notes.md)

For manual: Use `--help`, like so: `.\run.bat --help`

# How to run unit tests

For windows:
`.\run.bat --test`, specific test: `.\run.bat --test -k dattabot_smoke_test`

For linux/mac:
`./run.sh --test`, specific test: `./run.sh --test -k dattabot_smoke_test`

Run python api test file:
`python tests/test_api.py`

# Motivation

-   Have fun learning more about AI
-   Read/learn/implement papers (and any form of research)
-   Contribute back all of this research and implementation back to the community

# Goals

-   Fully open source - weights and all
-   Accessible to anyone and everyone, and can run on their own machines (don't need to hit an endpoint or give data away to a third party)
-   Make it work well on 1 single gpu (yes, just 1)
-   Make it the one of the best agents/models in the world
-   To make the above true we have to find novel ideas

# License

**The Unlicense**

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
