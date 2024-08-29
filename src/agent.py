from torch import Tensor, nn, no_grad as torch_no_grad, optim
from torch.utils.data import DataLoader as TorchDataLoader
from transformers import AutoTokenizer

from src.logger import get_logger
from src.agent_config import get_agent_config
from src.logger import get_logger
from src.agent_config import get_agent_config
from src.data_loader import DattaBotDataLoader
from src.data_loader import DattaBotDataset as DataSet
from src.model import DattaBotModel
from src.util import DattaBotAPIResponse, get_tensor_dtype_from_config


class Agent:
    def __init__(self) -> None:
        # Setup config and logger, both singletons.
        self.config = get_agent_config()
        self.logger = get_logger(logging_level=self.config.env.logging_level)
        # Setup tokenizer.
        tokenizer_model_name = "distilbert-base-uncased"
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
            tokenizer_model_name
        )
        # Add bos and eos tokens and ids
        self.tokenizer.bos_token = self.tokenizer.cls_token
        self.tokenizer.eos_token = self.tokenizer.sep_token
        self.tokenizer.bos_token_id = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.bos_token
        )
        self.tokenizer.eos_token_id = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.eos_token
        )
        # Things go bad when pad_id is -1, we can't even print an encoded tensor!
        # Pad token is -1, change it to eos token.
        assert self.tokenizer.pad_token_id != -1, f"Pad id can't be -1."
        self.logger.info(f"Loaded tokenizer model from path: {tokenizer_model_name}")
        # Setup data loader.
        self.data_loader = DattaBotDataLoader(tokenizer=self.tokenizer)
        # Setup model.
        self.model = DattaBotModel(tokenizer=self.tokenizer)
        # Batch size.
        self.batch_size = self.config.agent.batch_size
        self.logger.debug(f"Batch size: {self.batch_size}")
        # Model dimensions.
        self.model_dimensions = self.config.neural_net.model_dimensions
        self.logger.debug(f"Model dimensions: {self.model_dimensions}")
        # Max tokens for response.
        self.response_max_response_tokens = self.config.agent.max_response_tokens
        self.logger.debug(f"Max tokens: {self.response_max_response_tokens}")
        # Max epochs.
        self.max_epoch = self.config.agent.max_epoch
        # Learning rate.
        self.learning_rate = self.config.agent.learning_rate
        # Weight decay
        self.weight_decay = self.config.agent.weight_decay
        # Device (cpu or gpu)
        self.device = self.config.env.device
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def calc_loss_batch(self, input_batch, target_batch):
        input_batch, target_batch = input_batch.to(self.device), target_batch.to(
            self.device
        )
        logits = self.model(src_input=input_batch, src_mask=None)
        loss = nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
        return loss

    def calc_loss_loader(self, data_loader: TorchDataLoader, num_batches=None):
        total_loss = 0.0
        if len(data_loader) == 0:
            return float("nan")
        elif num_batches is None:
            num_batches = len(data_loader)
        else:
            num_batches = min(num_batches, len(data_loader))
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i < num_batches:
                loss = self.calc_loss_batch(input_batch, target_batch)
                total_loss += loss.item()
            else:
                break
        return total_loss / num_batches

    def evaluate_model(self, data_loader: TorchDataLoader, eval_iter):
        self.model.eval()
        with torch_no_grad():
            train_loss = self.calc_loss_loader(
                data_loader=data_loader, num_batches=eval_iter
            )
            # val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
        self.model.train()
        return train_loss
        # return train_loss, val_loss

    def train_agent(self) -> DattaBotAPIResponse:
        self.logger.info("\nStarting training...")
        self.model.train()
        epoch_loss = 0
        train_losses = []
        tokens_seen = 0
        global_step = -1
        eval_iter = 10
        eval_freq = 10
        response: DattaBotAPIResponse = DattaBotAPIResponse()
        response.query_response = "TODO(PIYUSHDATTA): Fix me. Training agent."
        batched_train_data: TorchDataLoader = (
            self.data_loader.get_formatted_batched_training_data()
        )
        self.model.train()
        for epoch in range(self.max_epoch):
            for input_data, target_data in batched_train_data:
                self.optimizer.zero_grad()
                loss = self.calc_loss_batch(input_data, target_data)
                loss.backward()
                self.optimizer.step()
                global_step += 1
                tokens_seen += input_data.numel()
                if (global_step % eval_freq) == 0:
                    train_loss = self.evaluate_model(
                        data_loader=batched_train_data, eval_iter=eval_iter
                    )
                    train_losses.append(train_loss)
                    print(
                        f"Ep {epoch+1} (Step {global_step:06d}): "
                        f"Train loss {train_loss:.3f}"
                    )
        self.logger.info("Finished training!\n")
        return response

    def respond_to_queries(self, queries: list[str]) -> DattaBotAPIResponse:
        # Encode the list of queries and convert them into a tensors.
        # Tensor, int
        input_tensor, total_batches = self.convert_queries_to_tensors(queries=queries)
        self.logger.debug(
            f"Output of self.convert_queries_to_tensors():\n{input_tensor}\nwith shape: {input_tensor.shape}\nTotal batches: {total_batches}"
        )
        # Feed the tensors to our model, in batches.
        batched_tensor_responses = []
        tensor_dtype = get_tensor_dtype_from_config(config=self.config)
        total_loss = 0
        for batch, i in enumerate(range(0, total_batches)):
            data, targets = self.data_loader.get_batch(
                input_tensor=input_tensor, idx=i, train=False
            )
            self.logger.debug(
                f"Data being fed to model:\n{data}\nwith shape: {data.shape}"
            )
            # Call our model.
            output = self.model(src_input=data, src_mask=None)
            # Apply softmax to convert logits to probabilities.
            softmax_output = nn.functional.softmax(output, dim=-1).to(tensor_dtype)
            # Flatten and add to our responses.
            output_flat = softmax_output.view(-1, self.response_max_response_tokens)
            batched_tensor_responses.append(output_flat)
            break
        # Decode the tensors and return a DattaBot API response for each tensor.
        response: DattaBotAPIResponse = self.convert_tensors_to_responses(
            tensor_resps=batched_tensor_responses
        )
        self.logger.debug(f"Output of self.respond_to_queries():\n{response}")
        return response

    def tokenizer_encode(self, decoded_queries: list[str]) -> list[list[int]]:
        return self.data_loader.tokenizer_encode(decoded_queries=decoded_queries)

    def tokenizer_decode(self, encoded_queries: list[list[int]]) -> list[str]:
        return self.data_loader.tokenizer_decode(encoded_queries=encoded_queries)

    def convert_queries_to_tensors(self, queries: list[str]) -> tuple[Tensor, int]:
        return self.data_loader.convert_queries_to_tensors(queries=queries)

    def convert_tensors_to_responses(
        self, tensor_resps: list[Tensor]
    ) -> DattaBotAPIResponse:
        return self.data_loader.convert_tensors_to_responses(tensor_resps=tensor_resps)
