from src.logger import get_logger
from src.api import DattaBotAPI, DattaBotAPIResponse


class DattaCommandLineClient:
    def __init__(self) -> None:
        self.logger = get_logger()
        self.api = DattaBotAPI()
        self.logger.info("Interactive CLI Session. Type 'exit' to end the session.\n")

    def process_input(self, first_input: str, second_input: str) -> str:
        resp: DattaBotAPIResponse = self.api.respond_to_queries(
            queries=[first_input, second_input]
        )
        return resp.query_response

    def run(self) -> None:
        while True:
            try:
                # Get input from the user
                first_input = input("Enter text: ")
                second_input = input("Enter text: ")
                user_input = first_input
                # Check if the user wants to exit
                if user_input.lower() == "exit":
                    self.logger.info("Exiting the session. Goodbye!")
                    break
                # Process the input and get the output
                output_text: str = self.process_input(
                    first_input=first_input, second_input=second_input
                )
                self.logger.info(output_text + "\n")
            except KeyboardInterrupt:
                self.logger.info("KeyboardInterrupt. Exiting the session. Goodbye!")
                break


if __name__ == "__main__":
    cli_client = DattaCommandLineClient()
    cli_client.run()
