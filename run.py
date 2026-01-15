import atexit
import os
import signal
import subprocess
import sys
from typing import Optional

from cli_parser import create_parser
from src.util import APIActions

# Global list to track spawned processes
_spawned_processes: list[subprocess.Popen] = []


def cleanup_devices():
    """
    Cleanup spawned subprocesses and accelerator state.

    - CUDA / ROCm: terminate spawned + kill stray GPU processes
    - TPU: terminate spawned ONLY (no device killing)
    - MPS / CPU: terminate spawned only
    """
    print("\nCleaning up processes...")
    # --------------------
    # Kill spawned subprocesses (ALL backends)
    # --------------------
    for proc in list(_spawned_processes):
        try:
            if proc.poll() is None:
                print(f"Terminating spawned process {proc.pid}...")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print(f"Force killing process {proc.pid}...")
                    proc.kill()
        except Exception as e:
            print(f"Error terminating process {proc.pid}: {e}")
        finally:
            if proc in _spawned_processes:
                _spawned_processes.remove(proc)
    # --------------------
    # TPU cleanup (SAFE NO-OP for devices)
    # --------------------
    if os.environ.get("PJRT_DEVICE") == "TPU":
        print("TPU environment detected — skipping device-level cleanup")

        # Optional: polite XLA shutdown sync
        try:
            import torch_xla.core.xla_model as xm

            if xm.xrt_world_size() > 1:
                xm.rendezvous("cleanup_exit")
        except Exception:
            pass

        print("TPU cleanup complete!")
        return
    # --------------------
    # CUDA / ROCm cleanup
    # --------------------
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            timeout=5,
        )
        pids = [
            int(pid.strip())
            for pid in result.stdout.strip().split("\n")
            if pid.strip()
        ]
        if pids:
            print(f"Found {len(pids)} GPU processes to clean up...")
            for pid in pids:
                try:
                    os.kill(pid, signal.SIGTERM)
                    print(f"Terminated GPU process {pid}")
                except (ProcessLookupError, PermissionError) as e:
                    print(f"Could not kill process {pid}: {e}")
        else:
            print("No stray GPU processes found")
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        print("nvidia-smi not available or failed — skipping GPU cleanup")

    print("Cleanup complete!")


def signal_handler(signum, frame):
    """Handle Ctrl+C and other termination signals."""
    print("\n\nReceived interrupt signal, cleaning up...")
    cleanup_devices()
    sys.exit(0)


def register_cleanup_handlers():
    """Register signal handlers and atexit cleanup."""
    # Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    # Termination signal
    signal.signal(signal.SIGTERM, signal_handler)
    # Normal exit
    atexit.register(cleanup_devices)


def run_subprocess(cmd: list[str], check: bool = True):
    """
    Run a subprocess and track it for cleanup.

    Args:
        cmd: Command to run as list of strings
        check: Whether to check return code (passed to subprocess.run)
    """
    global _spawned_processes

    print("Running command:", " ".join(cmd))

    # Use Popen instead of run to get the process object immediately
    proc = subprocess.Popen(cmd)
    _spawned_processes.append(proc)

    try:
        returncode = proc.wait()
        if check and returncode != 0:
            raise subprocess.CalledProcessError(returncode, cmd)
    except KeyboardInterrupt:
        # Let the signal handler deal with it
        raise
    finally:
        # Remove from tracking list when done
        if proc in _spawned_processes:
            _spawned_processes.remove(proc)
    return returncode


def detect_num_devices() -> dict:
    """
    Detect available accelerator devices.

    Returns:
        {
            "device": one of {"cuda", "rocm", "tpu", "mps", "cpu"},
            "count": int,
        }
    """
    # --------------------
    # CUDA / ROCm (NVIDIA + AMD)
    # --------------------
    try:
        import torch

        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            backend = "rocm" if torch.version.hip is not None else "cuda"
            return {
                "device": backend,
                "count": count,
            }
    except Exception:
        pass
    # --------------------
    # Apple Silicon (MPS)
    # --------------------
    try:
        import torch

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # MPS is single-device by design
            return {
                "device": "mps",
                "count": 1,
            }
    except Exception:
        pass
    # --------------------
    # TPU (PJRT / torch-xla)
    # --------------------
    try:
        import os
        import torch_xla.core.xla_model as xm
        # Hard signal: TPU runtime explicitly enabled
        if os.environ.get("PJRT_DEVICE") == "TPU":
            count = 2
            return {
                "device": "tpu",
                "count": count,
            }
    except Exception:
        pass
    # --------------------
    # CPU fallback
    # --------------------
    return {
        "device": "cpu",
        "count": 0,
    }

def process_api_cmd(api_cmd: str, api_args: str):
    device_info = detect_num_devices()
    device = device_info["device"]
    num_devices = device_info["count"]
    nnodes = int(os.environ.get("NNODES", "1"))
    node_rank = int(os.environ.get("NODE_RANK", "0"))
    print(f"Processing API command: {api_cmd} with args: {api_args}, device: {device}, num_devices: {num_devices}")
    # --------------------
    # Distributed execution (CUDA / ROCm / TPU)
    # --------------------
    if device in {"cuda", "rocm", "tpu"} and num_devices > 1:
        print(
            f"Detected {num_devices} {device} devices, running distributed with torchrun..."
        )
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nnodes={nnodes}",
            f"--node_rank={node_rank}",
            f"--nproc_per_node={num_devices}",
            "--rdzv_id=datta_bot_job",
            "--rdzv_backend=c10d",
            "--rdzv_endpoint=localhost:29500",
            "api_runner.py",
            "--api_cmd",
            api_cmd,
            "--api_args",
            api_args,
        ]
    # --------------------
    # Single accelerator (CUDA / ROCm / TPU / MPS)
    # --------------------
    elif device in {"cuda", "rocm", "tpu", "mps"}:
        print(f"Single {device} device detected, running without torchrun.")
        if device in {"cuda", "rocm"}:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        cmd = [
            sys.executable,
            "api_runner.py",
            "--api_cmd",
            api_cmd,
            "--api_args",
            api_args,
        ]
    # --------------------
    # CPU fallback
    # --------------------
    else:
        print("No accelerators detected, running on CPU.")
        cmd = [sys.executable, "client.py"]
    run_subprocess(cmd, check=False)


def run_client():
    device_info = detect_num_devices()
    device = device_info["device"]
    num_devices = device_info["count"]
    nnodes = int(os.environ.get("NNODES", "1"))
    node_rank = int(os.environ.get("NODE_RANK", "0"))
    print(f"Running client: device: {device}, num_devices: {num_devices}")
    # --------------------
    # Distributed execution
    # --------------------
    if device in {"cuda", "rocm", "tpu"} and num_devices > 1:
        print(
            f"Detected {num_devices} {device} devices, running distributed with torchrun..."
        )
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nnodes={nnodes}",
            f"--node_rank={node_rank}",
            f"--nproc_per_node={num_devices}",
            "--rdzv_id=datta_bot_job",
            "--rdzv_backend=c10d",
            "--rdzv_endpoint=localhost:29500",
            "client.py",
        ]
    # --------------------
    # Single device
    # --------------------
    else:
        print(f"Running client on single {device} device.")
        if device in {"cuda", "rocm"}:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        cmd = [sys.executable, "client.py"]
    run_subprocess(cmd, check=False)


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
            f"Unknown test mode: {test_mode}. Use 'unit', 'integration', '--test-file', or 'all'"
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
        # For tests, we still use subprocess.run since we want to capture the result.
        proc = subprocess.Popen(command, env=env)
        _spawned_processes.append(proc)
        returncode = proc.wait()
        _spawned_processes.remove(proc)
        if returncode != 0:
            print(f"\nTests failed with exit code {returncode}")
            sys.exit(returncode)
        else:
            print("\nTests passed!")
    except subprocess.CalledProcessError as e:
        print(f"\nTests failed with exit code {e.returncode}")
        sys.exit(e.returncode)


def main():
    # Register cleanup handlers at the start.
    register_cleanup_handlers()
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
