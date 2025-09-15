import argparse
import subprocess
import sys
import os


def main():
    parser = argparse.ArgumentParser(description="DattaBot start and run script!")
    # Unit tests.
    parser.add_argument(
        "--test",
        nargs="?",
        const="all",
        help="Run tests (optionally specify test file, e.g. --test test_model.py)",
    )
    args = parser.parse_args()

    # Run command.
    if args.test is not None:
        run_tests(test_name=args.test)
    else:
        run_client()


def run_tests(test_name: str = None):
    """
    Run pytest on the tests directory, or on a specific test file if provided.
    """
    test_target = "tests"
    if test_name not in (None, "", "all"):
        test_target = os.path.join("tests", test_name)

    command = [
        sys.executable,
        "-m",
        "pytest",
        test_target,
        "-v",  # verbose
    ]

    print(f"Running command: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Tests failed with exit code {e.returncode}")
        sys.exit(e.returncode)


def run_client():
    command = [sys.executable, "client.py"]
    print(f"Running command: {' '.join(command)}")
    subprocess.run(command)


if __name__ == "__main__":
    main()
