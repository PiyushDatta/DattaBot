from src.api import DattaBotAPI


class DattaCommandLineClient:
    def __init__(self):
        self.api = DattaBotAPI()
        print("Interactive CLI Session. Type 'exit' to end the session.")

    def process_input(self, input_text):
        return self.api.respond_to_query(input_text)

    def run(self):
        while True:
            try:
                # Get input from the user
                user_input = input("Enter text: ")
                # Check if the user wants to exit
                if user_input.lower() == "exit":
                    print("Exiting the session. Goodbye!")
                    break
                # Process the input and get the output
                output_text = self.process_input(user_input)
                # Print the output
                print(output_text + "\n")
            except KeyboardInterrupt:
                print("\nKeyboardInterrupt. Exiting the session. Goodbye!")
                break


if __name__ == "__main__":
    cli_client = DattaCommandLineClient()
    cli_client.run()
