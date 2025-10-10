"""
Command-line argument parser for DattaBot.
"""

import argparse
from enum import Enum

from api_runner import APIActions
from src.util import EvalBenchmark


def create_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser for DattaBot CLI.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="DattaBot start and run script!",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run fast unit tests (default)
  python run.py --test
  python run.py --test unit

  # Run slow integration tests
  python run.py --test integration

  # Run all tests
  python run.py --test all

  # Run specific test file (unit tests only)
  python run.py --test test_model.py

  # Run HumanEval evaluation with default settings
  python run.py --eval humaneval

  # Run HumanEval with custom batch size
  python run.py --eval humaneval --batch_size 8

  # Run HumanEval with multiple samples for pass@10
  python run.py --eval humaneval --num_samples 10

  # Full example with all options
  python run.py --eval humaneval \\
      --batch_size 8 \\
      --num_samples 10 \\
      --output_file my_results.jsonl

  # Available benchmarks: humaneval
        """,
    )
    group = parser.add_mutually_exclusive_group()
    # Testing arguments.
    group.add_argument(
        "--test",
        nargs="?",
        const="unit",
        choices=["unit", "integration", "all"],
        help="Run tests: 'unit' (fast, default), 'integration' (slow), or 'all'",
    )
    parser.add_argument(
        "--test-file",
        type=str,
        help="Specific test file to run (e.g., test_model.py)",
    )
    # API command arguments.
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
    # Evaluation arguments.
    group.add_argument(
        "--eval",
        type=str,
        help=f"Run evaluation benchmark. Options: {[b.value for b in EvalBenchmark]}",
    )
    # Evaluation-specific arguments.
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for evaluation inference (default: 4)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples per problem (default: 1). Use 10+ for pass@10 metrics on HumanEval",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output file for evaluation results (default: <benchmark>_results.jsonl)",
    )
    parser.add_argument(
        "--eval_args",
        type=str,
        default=None,
        help="Additional arguments to pass to evaluation script (comma-separated)",
    )
    return parser
