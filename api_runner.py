from argparse import ArgumentParser
from enum import Enum

from src.api import DattaBotAPI
from src.api_interface import DattaBotAPIResponse
from src.logger import get_logger
from src.util import APIActions

# Defines the delimiter to split the API args if its a query.
QUERIES_DELIMITER = "<|endoftext|>"
# Defines the delimiter to split the API args if not a query.
DELIMITER = ","


def print_eval_results(response: DattaBotAPIResponse):
    """Pretty print HumanEval evaluation results."""
    logger = get_logger()
    metadata = response.metadata
    scores = metadata.get("scores", {})
    # Create a nice formatted output
    logger.info("\n" + "=" * 70)
    logger.info("HUMANEVAL EVALUATION RESULTS")
    logger.info("=" * 70)
    logger.info(f"Problems Evaluated: {metadata.get('num_problems', 'N/A')}")
    logger.info(f"Samples per Problem: {metadata.get('num_samples', 'N/A')}")
    logger.info("-" * 70)
    logger.info("SCORES:")
    for metric, value in scores.items():
        percentage = value * 100
        bar_length = int(percentage / 2)  # Scale to 50 chars max
        bar = "█" * bar_length + "░" * (50 - bar_length)
        logger.info(f"   {metric:>10}: {percentage:6.2f}% [{bar}]")
    logger.info("-" * 70)
    logger.info(f"Results saved to: {metadata.get('output_file', 'N/A')}")
    logger.info("=" * 70 + "\n")


def process_api_cmd(api_cmd: str, api_args_str: str):
    if api_cmd is None:
        api_cmd = ""
    if api_args_str is None:
        api_args_str = ""

    api_cmd = api_cmd.strip().upper()
    api_client = DattaBotAPI()
    logger = get_logger()
    logger.info(f"API command: {api_cmd}")
    logger.info(f"API args: {api_args_str}")
    api_args = api_args_str.strip().split(DELIMITER)
    match api_cmd:
        case APIActions.RESPOND_TO_QUERIES.name:
            queries = api_args_str.split(QUERIES_DELIMITER)
            # Strip whitespace from each query.
            queries = [query.strip() for query in queries]
            responses: list[DattaBotAPIResponse] = api_client.respond_to_queries(
                queries=queries
            )
            # Only print on rank 0 (responses will be empty on other ranks)
            if responses:
                for i, response in enumerate(responses, start=1):
                    print(f"\nResponse {i}:\n{response.text}\n")
        case APIActions.GET_ENCODING.name:
            print(api_client.get_encoding(queries=api_args))
        case APIActions.GET_DECODING.name:
            print(api_client.get_decoding(queries=api_args))
        case APIActions.GET_TENSOR_ENCODING.name:
            print(api_client.get_tensor_encoding(queries=api_args))
        case APIActions.TRAIN_AGENT.name:
            print(api_client.train_agent())
        case APIActions.RUN_EVALUATION.name:
            responses: list[DattaBotAPIResponse] = api_client.run_evaluation(
                queries=api_args
            )
            for response in responses:
                print_eval_results(response)
        case APIActions.GET_RANDOM_VALIDATION_EXAMPLE.name:
            response: DattaBotAPIResponse = api_client.get_random_validation_example()
            raw_text = response.raw_text
            print(
                f"Raw text from validation example. Sequence length of raw text: {len(raw_text)}.\n=====START=====\n{raw_text}\n=====END====="
            )
        case APIActions.PROFILE_AGENT_TRAINING.name:
            num_steps = int(api_args[0]) if len(api_args) > 0 and api_args[0] else 20
            log_dir = api_args[1] if len(api_args) > 1 and api_args[1] else None
            responses: list[DattaBotAPIResponse] = api_client.profile_agent_training(
                queries=[str(num_steps), log_dir]
            )
            if responses:
                for i, response in enumerate(responses, start=1):
                    print(f"\nProfile Response {i}:\n{response.text}\n")
        case _:
            logger.error(
                f"API Command selected: {api_cmd}. This is an invalid API command."
            )
            logger.error(
                f"\nList of API commands:\n{[action.value for action in APIActions]}\n"
            )


def main():
    parser = ArgumentParser()
    # API Command.
    parser.add_argument(
        "--api_cmd",
        "--cmd",
        "-cmd",
        type=str,
        help=f"Required. The api command to run.\nList of API commands:\n{[action.value for action in APIActions]}\n",
    )
    # Input arguments (optional).
    parser.add_argument(
        "--api_args",
        "--args",
        "-args",
        type=str,
        nargs="?",
        help="Optional. Optional string arguments that can be used for the api command.",
    )
    # TODO: Check if help is specified.
    # if parser...:
    #     print(
    #         "Please set `--api_cmd <API command name>` when calling this script. It is required."
    #     )
    #     parser.print_help()
    #     return
    args = parser.parse_args()
    return process_api_cmd(api_cmd=args.api_cmd, api_args_str=args.api_args)


if __name__ == "__main__":
    main()
