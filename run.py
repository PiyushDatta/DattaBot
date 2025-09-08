import argparse
import subprocess
import sys
import os


def main():
    parser = argparse.ArgumentParser(description="DattaBot start and run script!")
    # Unit tests.
    parser.add_argument("--test", nargs=argparse.REMAINDER, help="Run unit tests")
    args = parser.parse_args()

    # Run command.
    if args.test is not None:
        run_tests(arg_list=args.test)
    else:
        run_client()


def run_tests(arg_list: list[str] = None):
    if arg_list is None:
        arg_list = []

    command = [
        sys.executable,
        "-m",
        "pytest",
        "tests",  # test directory
        "-v",  # verbose
    ] + arg_list

    print(f"Running command: {' '.join(command)}")
    subprocess.run(command, check=True)


def run_client():
    command = [sys.executable, "client.py"]
    print(f"Running command: {' '.join(command)}")
    subprocess.run(command)


if __name__ == "__main__":
    main()
