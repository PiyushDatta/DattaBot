from src.util import get_default_logger
from src.api import DattaBotAPI


class DattaCommandLineClient:
    def __init__(self) -> None:
        self.logger = get_default_logger()
        self.api = DattaBotAPI()
        self.logger.info("Interactive CLI Session. Type 'exit' to end the session.")

    def process_input(self, input_text: str):
        return self.api.respond_to_query(query=input_text)

    def run(self) -> None:
        while True:
            try:
                # Get input from the user
                user_input = input("Enter text: ")
                # Check if the user wants to exit
                if user_input.lower() == "exit":
                    self.logger.info("Exiting the session. Goodbye!")
                    break
                # Process the input and get the output
                output_text = self.process_input(user_input)
                self.logger.info(output_text)
            except KeyboardInterrupt:
                self.logger.info("KeyboardInterrupt. Exiting the session. Goodbye!")
                break


if __name__ == "__main__":
    cli_client = DattaCommandLineClient()
    cli_client.run()
