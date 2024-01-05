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


def run_tests(arg_list: list[str] = [""]):
    command = [
        _get_python_executable(),
        "-m",
        "unittest",
        "-v",
        "tests.test_api",
    ] + arg_list
    print(f"Running command: {' '.join(command)}")
    subprocess.run(command)


def run_client():
    command = [_get_python_executable(), "client.py"]
    print(f"Running command: {' '.join(command)}")
    subprocess.run(command)


def _get_python_executable() -> str:
    # Check if in a virtual environment
    if hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    ):
        return (
            os.path.join(sys.prefix, "bin", "python")
            if sys.platform != "win32"
            else os.path.join(sys.prefix, "Scripts", "python.exe")
        )
    else:
        return "python"


if __name__ == "__main__":
    main()
