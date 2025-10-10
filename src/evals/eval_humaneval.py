"""
HumanEval benchmark evaluation for DattaBot.
Self-contained implementation with all necessary functions.
"""

import json
import os
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import as_completed, ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from src.api_interface import DattaBotAPIResponse
from src.logger import get_logger

if TYPE_CHECKING:
    from src.agent import Agent


def run_humaneval(
    agent: "Agent",
    batch_size: int,
    num_samples: int,
    output_file: str,
) -> DattaBotAPIResponse:
    """
    Run HumanEval evaluation.

    Args:
        agent: The agent to evaluate
        batch_size: Batch size for inference
        num_samples: Number of samples per problem
        output_file: Where to save results

    Returns:
        DattaBotAPIResponse with results
    """
    logger = get_logger()
    response = DattaBotAPIResponse()

    try:
        logger.info("Loading HumanEval problems...")
        problems = read_problems()
        logger.info(f"Found {len(problems)} problems")

        results = []

        # Collect all prompts
        prompts = []
        task_ids = []

        for task_id, problem in problems.items():
            for _ in range(num_samples):
                prompts.append(problem["prompt"])
                task_ids.append(task_id)

        logger.info(f"Generating {len(prompts)} completions...")

        # Generate completions in batches using the agent
        all_completions = []

        # Use tqdm if available
        try:
            from tqdm import tqdm

            progress_iter = tqdm(
                range(0, len(prompts), batch_size), desc="Generating completions"
            )
        except ImportError:
            progress_iter = range(0, len(prompts), batch_size)
            logger.info("tqdm not available, progress won't be shown")

        for i in progress_iter:
            batch_prompts = prompts[i : i + batch_size]

            # Use agent's respond_to_queries method
            responses = agent.respond_to_queries(batch_prompts)

            # Extract completions
            for prompt, resp in zip(batch_prompts, responses):
                completion = _clean_completion(resp.text, prompt)
                all_completions.append(completion)

        # Format results
        for task_id, completion in zip(task_ids, all_completions):
            results.append({"task_id": task_id, "completion": completion})

        # Save results
        logger.info(f"Saving results to {output_file}...")
        write_jsonl(output_file, results)

        # Run evaluation
        logger.info("Running test cases...")
        k_values = (
            [1, 10, 100]
            if num_samples >= 100
            else [1, 10] if num_samples >= 10 else [1]
        )
        scores = evaluate_functional_correctness(output_file, k=k_values)

        # Format response
        scores_text = "\n".join(
            [f"  {metric}: {value:.2%}" for metric, value in scores.items()]
        )
        response.text = (
            f"HumanEval evaluation completed!\n"
            f"Results:\n{scores_text}\n"
            f"Saved to: {output_file}"
        )
        response.metadata = {
            "benchmark": "humaneval",
            "scores": scores,
            "num_problems": len(problems),
            "num_samples": num_samples,
            "output_file": output_file,
        }

        logger.info("HumanEval evaluation completed successfully!")

    except ImportError as e:
        error_msg = (
            f"Required dependencies not installed. "
            f"Run: pip install datasets\n"
            f"Error: {e}"
        )
        logger.error(error_msg)
        response.text = error_msg

    except Exception as e:
        error_msg = f"Error during HumanEval evaluation: {e}"
        logger.error(error_msg)
        logger.exception(e)
        response.text = error_msg

    return response


def read_problems() -> Dict[str, Dict]:
    """
    Load HumanEval problems from HuggingFace datasets.

    Returns:
        Dictionary mapping task_id to problem data
    """
    logger = get_logger()

    try:
        from datasets import load_dataset

        logger.info("Downloading HumanEval dataset from HuggingFace...")
        dataset = load_dataset("openai_humaneval", split="test")

        problems = {}
        for item in dataset:
            problems[item["task_id"]] = {
                "task_id": item["task_id"],
                "prompt": item["prompt"],
                "canonical_solution": item["canonical_solution"],
                "test": item["test"],
                "entry_point": item["entry_point"],
            }

        logger.info(f"Loaded {len(problems)} HumanEval problems")
        return problems

    except ImportError:
        logger.error("datasets library not installed. Run: pip install datasets")
        raise
    except Exception as e:
        logger.error(f"Error loading HumanEval dataset: {e}")
        raise


def write_jsonl(filename: str, data: List[Dict], append: bool = False):
    """
    Write data to a JSONL file.

    Args:
        filename: Path to output file
        data: List of dictionaries to write
        append: Whether to append to existing file
    """
    mode = "a" if append else "w"
    with open(filename, mode, encoding="utf-8") as f:
        for item in data:
            json_str = json.dumps(item)
            f.write(json_str + "\n")


def read_jsonl(filename: str) -> List[Dict]:
    """
    Read data from a JSONL file.

    Args:
        filename: Path to input file

    Returns:
        List of dictionaries
    """
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def execute_code(code: str, timeout: float = 3.0) -> Tuple[bool, str]:
    """
    Execute Python code in a subprocess and return success status.

    Args:
        code: Python code to execute
        timeout: Timeout in seconds

    Returns:
        (success, error_message)
    """
    logger = get_logger()

    # Create temporary file for code
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        temp_file = f.name

    try:
        # Execute in subprocess with timeout
        result = subprocess.run(
            [sys.executable, temp_file], capture_output=True, text=True, timeout=timeout
        )

        # Check if execution was successful
        if result.returncode == 0:
            return True, ""
        else:
            return False, result.stderr

    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)
    finally:
        # Clean up temp file
        try:
            os.unlink(temp_file)
        except:
            pass


def check_correctness(
    task_id: str, completion: str, test: str, entry_point: str, timeout: float = 3.0
) -> Dict:
    """
    Check if a completion passes the test cases.

    Args:
        task_id: Problem identifier
        completion: Generated code completion
        test: Test cases
        entry_point: Function name to test
        timeout: Execution timeout

    Returns:
        Dictionary with test results
    """
    logger = get_logger()

    # Construct full program
    full_code = completion + "\n" + test + "\n" + f"check({entry_point})"

    # Execute and check
    success, error = execute_code(full_code, timeout=timeout)

    return {
        "task_id": task_id,
        "passed": success,
        "error": error if not success else None,
    }


def estimate_pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    """
    Estimate pass@k using the formula from the HumanEval paper.

    Args:
        num_samples: Total number of samples
        num_correct: Number of correct samples
        k: k value for pass@k

    Returns:
        Estimated pass@k probability
    """
    if num_samples - num_correct < k:
        return 1.0

    # Formula: 1 - (n - c choose k) / (n choose k)
    # Simplified to avoid overflow
    result = 1.0
    for i in range(1, k + 1):
        result *= (num_samples - num_correct - k + i) / (num_samples - k + i)
    return 1.0 - result


def evaluate_functional_correctness(
    sample_file: str,
    k: List[int] = [1, 10, 100],
    n_workers: int = 4,
    timeout: float = 3.0,
    problem_file: Optional[str] = None,
) -> Dict[str, float]:
    """
    Evaluate functional correctness of generated samples.

    Args:
        sample_file: Path to file with generated completions
        k: List of k values for pass@k metrics
        n_workers: Number of parallel workers
        timeout: Execution timeout per test
        problem_file: Path to problems file (optional)

    Returns:
        Dictionary with pass@k scores
    """
    logger = get_logger()

    # Load problems
    if problem_file and os.path.exists(problem_file):
        problems = {p["task_id"]: p for p in read_jsonl(problem_file)}
    else:
        problems = read_problems()

    # Load samples
    samples = read_jsonl(sample_file)

    # Group samples by task_id
    samples_by_task = defaultdict(list)
    for sample in samples:
        samples_by_task[sample["task_id"]].append(sample)

    logger.info(
        f"Evaluating {len(samples)} samples across {len(samples_by_task)} tasks..."
    )

    # Evaluate each sample
    results = []

    def eval_sample(sample):
        task_id = sample["task_id"]
        completion = sample["completion"]

        if task_id not in problems:
            logger.warning(f"Task {task_id} not found in problems")
            return None

        problem = problems[task_id]
        return check_correctness(
            task_id=task_id,
            completion=completion,
            test=problem["test"],
            entry_point=problem["entry_point"],
            timeout=timeout,
        )

    # Use ThreadPoolExecutor for parallel execution
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(eval_sample, sample) for sample in samples]

        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    logger.info(f"Completed {len(results)} evaluations")

    # Calculate pass@k for each task
    total_correct_by_task = Counter()
    total_samples_by_task = Counter()

    for result in results:
        task_id = result["task_id"]
        total_samples_by_task[task_id] += 1
        if result["passed"]:
            total_correct_by_task[task_id] += 1

    # Compute pass@k metrics
    pass_at_k = {}

    for k_val in k:
        pass_at_k_vals = []

        for task_id in total_samples_by_task:
            n_samples = total_samples_by_task[task_id]
            n_correct = total_correct_by_task[task_id]

            if n_samples >= k_val:
                pass_at_k_val = estimate_pass_at_k(n_samples, n_correct, k_val)
                pass_at_k_vals.append(pass_at_k_val)

        if pass_at_k_vals:
            pass_at_k[f"pass@{k_val}"] = sum(pass_at_k_vals) / len(pass_at_k_vals)
        else:
            logger.warning(f"Not enough samples for pass@{k_val}")

    # Log summary
    total_passed = sum(r["passed"] for r in results)
    logger.info(
        f"Total passed: {total_passed}/{len(results)} ({100*total_passed/len(results):.1f}%)"
    )

    return pass_at_k


def _clean_completion(full_response: str, prompt: str) -> str:
    """
    Extract only the generated completion, removing the prompt.

    Args:
        full_response: The full model output
        prompt: The original prompt

    Returns:
        Just the completion part
    """
    # Remove the prompt from the beginning
    if full_response.startswith(prompt):
        completion = full_response[len(prompt) :]
    else:
        completion = full_response

    # Stop at common function termination patterns
    stop_patterns = [
        "\nclass ",
        "\ndef ",
        "\n#",
        "\nif __name__",
    ]

    for pattern in stop_patterns:
        if pattern in completion:
            completion = completion.split(pattern)[0]

    return completion
