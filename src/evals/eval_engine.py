"""
Evaluation Engine for running benchmarks on DattaBot models.
"""

import os
from enum import Enum
from typing import Optional, TYPE_CHECKING

from src.agent_config import get_agent_config

from src.api_interface import DattaBotAPIResponse
from src.evals.eval_humaneval import run_humaneval
from src.logger import get_logger
from src.util import EvalBenchmark

if TYPE_CHECKING:
    from src.agent import Agent


class EvaluationEngine:
    """
    Evaluation engine for running benchmarks on trained models.
    Designed to be used as a class with static methods.
    """

    @classmethod
    def run_eval(
        cls,
        agent: "Agent",
        benchmark: str,
        batch_size: int = 4,
        num_samples: int = 1,
        output_file: Optional[str] = None,
    ) -> DattaBotAPIResponse:
        """
        Run evaluation on specified benchmark.

        Args:
            agent: The DattaBot agent instance to evaluate
            benchmark: Benchmark name (EvalBenchmark)
            batch_size: Batch size for inference
            num_samples: Number of samples per problem (for applicable benchmarks)
            output_file: Where to save results (optional)

        Returns:
            DattaBotAPIResponse with evaluation results
        """
        logger = get_logger()
        config = get_agent_config()
        response = DattaBotAPIResponse()
        # Validate benchmark
        try:
            benchmark_enum = EvalBenchmark(benchmark.lower())
        except ValueError:
            valid_benchmarks = [b.value for b in EvalBenchmark]
            error_msg = (
                f"Unknown benchmark: '{benchmark}'. "
                f"Valid options: {valid_benchmarks}"
            )
            logger.error(error_msg)
            response.text = error_msg
            return response

        benchmark = benchmark_enum.value
        # Set default output file if not provided
        if output_file is None:
            output_file = f"{config.agent.data_directory}/{benchmark}_results.jsonl"
            logger.info(f"Output file not provided, using default name: {output_file}")
        # Ensure output directory exists
        output_file = cls._ensure_output_file_exists(output_file)
        logger.info("=" * 70)
        logger.info(f"Running {benchmark.upper()} Evaluation")
        logger.info("=" * 70)
        logger.info(f"Benchmark:    {benchmark}")
        logger.info(f"Batch size:   {batch_size}")
        logger.info(f"Num samples:  {num_samples}")
        logger.info(f"Output file:  {output_file}")
        logger.info("=" * 70)
        # Route to appropriate benchmark evaluation
        try:
            match benchmark:
                case EvalBenchmark.HUMANEVAL.value:
                    response = run_humaneval(
                        agent=agent,
                        batch_size=batch_size,
                        num_samples=num_samples,
                        output_file=output_file,
                    )
                case _:
                    error_msg = f"Benchmark '{benchmark}' not yet implemented"
                    logger.error(error_msg)
                    response.text = error_msg
                    return response
            logger.info("=" * 70)
            logger.info(f"✓ {benchmark.upper()} evaluation completed!")
            logger.info(f"Results saved to: {output_file}")
            logger.info("=" * 70)
        except Exception as e:
            error_msg = f"Error during {benchmark.upper()} evaluation: {e}"
            logger.error(error_msg)
            logger.exception(e)
            response.text = error_msg

        return response

    @staticmethod
    def _ensure_output_file_exists(output_file: str) -> str:
        """
        Create the output file and any necessary parent directories.
        Args:
            output_file: Path to the output file
        Returns:
            Absolute path to the output file
        """
        output_file = os.path.abspath(output_file)
        output_dir = os.path.dirname(output_file)
        # Create directory if it doesn't exist
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        # Create empty file if it doesn't exist
        if not os.path.exists(output_file):
            with open(output_file, "w") as f:
                pass
        return output_file
