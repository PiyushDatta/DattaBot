from enum import Enum
from src.api import DattaBotAPI
from argparse import ArgumentParser


class APIActions(Enum):
    RESPOND_TO_QUERIES = "respond_to_queries"
    GET_ENCODING = "get_encoding"
    GET_DECODING = "get_decoding"
    GET_TENSOR_ENCODING = "get_tensor_encoding"
    TRAIN_AGENT = "train_agent"


def process_api_cmd(api_cmd: str, api_args: list[str]):
    if api_cmd is None:
        api_cmd = ""
    if api_args is None:
        api_args = []

    api_cmd = api_cmd.strip().upper()
    api_client = DattaBotAPI()
    print(f"API command: {api_cmd}")
    print(f"API args: {api_args}")
    match api_cmd:
        case APIActions.RESPOND_TO_QUERIES.name:
            print(api_client.respond_to_queries(queries=api_args))
        case APIActions.GET_ENCODING.name:
            print(api_client.get_encoding(queries=api_args))
        case APIActions.GET_DECODING.name:
            print(api_client.get_decoding(queries=api_args))
        case APIActions.GET_TENSOR_ENCODING.name:
            print(api_client.get_tensor_encoding(queries=api_args))
        case APIActions.TRAIN_AGENT.name:
            print(api_client.train_agent())
        case _:
            print(f"API Command selected: {api_cmd}. This is an invalid API command.")
            print(
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
    return process_api_cmd(api_cmd=args.api_cmd, api_args=args.api_args)


if __name__ == "__main__":
    main()
