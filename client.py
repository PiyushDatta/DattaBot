from src.logger import get_logger
from src.api import DattaBotAPI, DattaBotAPIResponse


class DattaCommandLineClient:
    def __init__(self) -> None:
        self.logger = get_logger()
        self.api = DattaBotAPI()
        self.logger.info("Interactive CLI Session. Type 'exit' to end the session.\n")

    def process_input(self, input_queries: list[str]) -> str:
        responses: list[DattaBotAPIResponse] = self.api.respond_to_queries(
            queries=input_queries
        )
        return responses[0].text

    def run(self) -> None:
        while True:
            try:
                # Get input from the user
                user_input = input("Enter text: ")
                # Check if the user wants to exit
                if user_input.lower() == "exit":
                    self.logger.info("Exiting the session. Goodbye!\n")
                    break
                # Process the input and get the output
                output_text: str = self.process_input(input_queries=[user_input])
                self.logger.info(f"Response:\n{output_text}")
            except KeyboardInterrupt:
                self.logger.info("KeyboardInterrupt. Exiting the session. Goodbye!\n")
                break


if __name__ == "__main__":
    cli_client = DattaCommandLineClient()
    cli_client.run()
