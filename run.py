import argparse
import os
import subprocess
import sys
from api_runner import APIActions


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


def run_tests(test_name: str = "all"):
    test_target = "tests"
    if test_name not in (None, "", "all"):
        test_target = os.path.join("tests", test_name)

    command = [sys.executable, "-m", "pytest", test_target, "-v"]

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
        subprocess.run(command, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"Tests failed with exit code {e.returncode}")
        sys.exit(e.returncode)


def main():
    parser = argparse.ArgumentParser(description="DattaBot start and run script!")
    group = parser.add_mutually_exclusive_group()

    group.add_argument(
        "--test",
        nargs="?",
        const="all",
        help="Run tests (optionally specify test file, e.g. --test test_model.py)",
    )
    group.add_argument(
        "--api_cmd",
        type=str,
        help=f"Run API command. Commands: {[action.value for action in APIActions]}",
    )
    parser.add_argument(
        "--api_args",
        type=str,
        default="",
        help="Optional arguments for the API command, comma separated if multiple",
    )

    args = parser.parse_args()
    if args.test is not None:
        run_tests(test_name=args.test)
    elif args.api_cmd:
        process_api_cmd(api_cmd=args.api_cmd, api_args=args.api_args)
    else:
        run_client()


if __name__ == "__main__":
    main()
