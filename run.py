import argparse
import os
import subprocess
import sys
from typing import Optional

from cli_parser import create_parser, EvalBenchmark
from src.util import APIActions


def detect_num_gpus() -> int:
    """Detect number of GPUs using nvidia-smi first, then PyTorch as fallback."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        num_gpus = int(result.stdout.strip() or 0)
        if num_gpus > 0:
            return num_gpus
    except Exception:
        pass

    # Try torch.
    try:
        import torch

        num_gpus = torch.cuda.device_count()
        return num_gpus
    except ImportError:
        return 0


def process_api_cmd(api_cmd: str, api_args: str):
    num_gpus = detect_num_gpus()
    nnodes = int(os.environ.get("NNODES", "1"))
    node_rank = int(os.environ.get("NODE_RANK", "0"))

    if num_gpus > 1:
        print(f"Detected {num_gpus} GPUs, running distributed with torchrun...")
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nnodes={nnodes}",
            f"--node_rank={node_rank}",
            f"--nproc_per_node={num_gpus}",
            "--rdzv_id=datta_bot_job",
            "--rdzv_backend=c10d",
            "--rdzv_endpoint=localhost:29500",
            "api_runner.py",
            "--api_cmd",
            api_cmd,
            "--api_args",
            api_args,
        ]
    elif num_gpus == 1:
        print("Single GPU detected, running without torchrun for faster startup.")
        cmd = [
            sys.executable,
            "api_runner.py",
            "--api_cmd",
            api_cmd,
            "--api_args",
            api_args,
        ]
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        print("No GPUs detected, running client normally on CPU.")
        cmd = [sys.executable, "client.py"]

    print("Running command:", " ".join(cmd))
    subprocess.run(cmd)


def run_client():
    num_gpus = detect_num_gpus()
    nnodes = int(os.environ.get("NNODES", "1"))
    node_rank = int(os.environ.get("NODE_RANK", "0"))

    if num_gpus > 1:
        print(f"Detected {num_gpus} GPUs, running distributed with torchrun...")
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nnodes={nnodes}",
            f"--node_rank={node_rank}",
            f"--nproc_per_node={num_gpus}",
            "--rdzv_id=datta_bot_job",
            "--rdzv_backend=c10d",
            "--rdzv_endpoint=localhost:29500",
            "client.py",
        ]
    elif num_gpus == 1:
        print("Single GPU detected, running client directly (no torchrun).")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        cmd = [sys.executable, "client.py"]
    else:
        print("No GPUs detected, running client on CPU.")
        cmd = [sys.executable, "client.py"]

    print("Running command:", " ".join(cmd))
    subprocess.run(cmd)


def run_tests(test_mode: str = "unit", test_file: Optional[str] = None):
    """
    Run tests with different modes.

    Args:
        test_mode: "unit" (fast, mocked), "integration" (slow, real), or "all"
        test_file: Optional specific test file to run
    """
    # Determine test target
    if test_file and test_file not in ("unit", "integration", "all"):
        # User specified a specific test file
        test_target = os.path.join("tests", test_file)
    else:
        test_target = "tests"

    # Build pytest command based on mode
    command = [sys.executable, "-m", "pytest", test_target, "-v"]

    if test_mode == "unit":
        # Run only fast unit tests (skip integration)
        print("Running UNIT tests (fast, with mocks)...")
        command.extend(["-m", "not integration"])
    elif test_mode == "integration":
        # Run only integration tests
        print("Running INTEGRATION tests (slow, real model)...")
        command.extend(["-m", "integration"])
    elif test_mode == "all":
        # Run all tests
        print("Running ALL tests (unit + integration)...")
        command.extend(["-m", ""])
    else:
        raise ValueError(
            f"Unknown test mode: {test_mode}. Use 'unit', 'integration', or 'all'"
        )

    env = os.environ.copy()
    env.update(
        {
            "LOCAL_RANK": "0",
            "RANK": "0",
            "WORLD_SIZE": "1",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29500",
        }
    )

    print(f"Running command: {' '.join(command)}")
    try:
        result = subprocess.run(command, check=True, env=env)
        print(f"\nTests passed!")
    except subprocess.CalledProcessError as e:
        print(f"\nTests failed with exit code {e.returncode}")
        sys.exit(e.returncode)


def main():
    parser = create_parser()
    args = parser.parse_args()
    try:
        if args.eval:
            eval_args_str = ",".join(
                [
                    str(args.eval),  # benchmark
                    str(args.batch_size),  # batch_size
                    str(args.num_samples),  # num_samples
                    str(args.output_file),  # eval output file
                    str(args.eval_args),  # extra_args
                ]
            )
            process_api_cmd(
                api_cmd=APIActions.RUN_EVALUATION.name, api_args=eval_args_str
            )
        elif args.test is not None:
            run_tests(test_mode=args.test, test_file=args.test_file)
        elif args.api_cmd:
            process_api_cmd(api_cmd=args.api_cmd, api_args=args.api_args)
        else:
            run_client()
    except Exception as e:
        print(f"\nError: {e}\n")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
