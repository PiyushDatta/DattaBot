import os
import time
import traceback
from math import ceil
from typing import Optional

import torch
import torch.distributed as dist

# TPU support
try:
    import torch_xla.core.xla_model as xm

    HAS_XLA = True
except ImportError:
    HAS_XLA = False
import torch.multiprocessing as mp
from src.agent_components import DattaBotAgentComponentFactory, DattaBotAgentComponents
from src.agent_config import get_agent_config
from src.api_interface import DattaBotAPIResponse
from src.checkpointing import (
    load_agent,
    load_optimizer_state_from_checkpoint,
    save_agent,
)
from src.communication_mgr import DattaBotCommunicationManager
from src.data_loader import DattabotDataBuilder
from src.gpu_profiler import BackgroundGPUProfiler
from src.logger import get_logger
from src.metric_tracker import get_metric_tracker, MetricTracker
from src.model import DattaBotModel
from src.tokenizer import get_tokenizer
from src.training_engine import DattaBotTrainingEngine, TrainingMode
from src.util import (
    get_device_info,
    get_logging_level_from_config,
    get_tensor_dtype_from_config,
    is_autocast_enabled,
    is_rank_0,
    setup_backend_settings,
    setup_torch_dist_init,
)
from torch import nn, Tensor
from torch.distributed.fsdp import FSDPModule, fully_shard, MixedPrecisionPolicy
from torch.optim.lr_scheduler import OneCycleLR as TorchOneCycleLR
from torch.utils.data.distributed import DistributedSampler


class Agent:
    def __init__(self) -> None:
        # Setup config and logger, both singletons.
        self.config = get_agent_config()
        self.logger = get_logger(
            logging_level=get_logging_level_from_config(self.config)
        )
        self.agent_name = self.config.agent.agent_name
        self.tensor_dtype = get_tensor_dtype_from_config(self.config)
        self.autocast_dtype = torch.bfloat16
        # Initialize the default distributed process group before doing anything else.
        setup_torch_dist_init()
        self.local_rank = 0
        # Setup compute.
        self.device = None
        self.device_info = get_device_info()
        self._initialize_compute()
        assert self.device is not None
        # Setup communication manager (user and agent interactions manager).
        self.comm_manager = DattaBotCommunicationManager()
        # Setup tokenizer.
        self.tokenizer = get_tokenizer(encoding_name="o200k_harmony")
        tokenizer_model_name = repr(self.tokenizer)
        self.logger.info(f"Loaded tokenizer model from path: {tokenizer_model_name}")
        # Setup model.
        self.d_model = self.config.neural_net.model_dimensions
        self.model = self._create_model()
        # Setup optimizer.
        self.optimizer = self._create_optimizer()
        # Setup data loader.
        self.data_builder = DattabotDataBuilder()
        # Setup agent components, this includes the neural network model.
        factory = DattaBotAgentComponentFactory(
            config=self.config,
            tokenizer=self.tokenizer,
            optimizer=self.optimizer,
            device=self.device,
            device_info=self.device_info,
            local_rank=self.local_rank,
            tensor_dtype=self.tensor_dtype,
            autocast_dtype=self.autocast_dtype,
        )
        self.components: DattaBotAgentComponents = factory.create()
        # Load the agent and its components.
        load_agent(
            model=self.model,
            checkpoint_dir=self.components.weights_dir,
            optimizer=self.optimizer,
            device=self.device,
        )
        # Monitoring tools.
        self.metric_tracker: MetricTracker | None = None
        self.gpu_profiler: BackgroundGPUProfiler | None = BackgroundGPUProfiler(
            device=self.device
        )
        # Setup inference engine when needed.
        self.inference_engine = None
        # Setup training engine.
        self.training_engine = DattaBotTrainingEngine(
            device=self.device,
            tokenizer=self.tokenizer,
            model=self.model,
            optimizer=self.optimizer,
            components=self.components,
            autocast_dtype=self.autocast_dtype,
            metric_tracker=self.metric_tracker,
            gpu_profiler=self.gpu_profiler,
            d_model=self.d_model,
            save_checkpoint_fn=self._save_agent,
        )

    def __del__(self) -> None:
        self._cleanup()

    def _cleanup(self) -> None:
        """Clean up distributed process group if initialized."""
        if dist.is_available() and dist.is_initialized():
            self.logger.info("Destroying distributed process group...")
            dist.destroy_process_group()

    def _create_model(self) -> DattaBotModel:
        """Create neural network model."""
        # Setup model.
        # We pass in tensor_dtype to the model, but remember we may use
        # AMP autocast during model inference/training which will have
        # all tensors as float16/bfloat16.
        model = None
        with torch.device("meta"):
            model = DattaBotModel(device=self.device, dtype=self.tensor_dtype)
        model.to_empty(device=self.device)
        model.init_weights()
        self.logger.info(f"Model is on: {self.device}")
        self.logger.info(f"Model dimensions: {self.d_model}")
        self.logger.info(
            f"Total model parameters: {sum(p.numel() for p in model.parameters()):,}"
        )
        # Convert entire model to consistent dtype after loading
        if is_autocast_enabled(self.device):
            model = model.to(dtype=self.autocast_dtype)
            self.logger.info(f"Converted model to {self.autocast_dtype} dtype.")
        else:
            model = model.to(dtype=self.tensor_dtype)
            self.logger.info(f"Converted model to {self.tensor_dtype} dtype.")
        # Wrap with distributed layer if we have multiple gpus.
        if dist.is_available() and torch.cuda.device_count() > 1:
            if self.config.agent.fsdp:
                fsdp_kwargs = {
                    "mp_policy": MixedPrecisionPolicy(
                        param_dtype=self.autocast_dtype,
                        reduce_dtype=self.autocast_dtype,
                    )
                }
                for layer in model.layers:
                    fully_shard(layer, **fsdp_kwargs)
                fully_shard(model, **fsdp_kwargs)
                assert isinstance(model, FSDPModule)
            else:
                model = nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    find_unused_parameters=False,
                )
        if self.device_info["backend"] == "cuda":
            model.cuda()
        return model

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer."""
        # TODO(PiyushDatta): Try and get Muon optimizer to work.
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.agent.lr,
            weight_decay=self.config.agent.weight_decay,
            betas=(0.9, 0.95),
        )

    def _initialize_compute(self):
        """Main entry point for all compute setup."""
        # Detect available hardware
        self.logger.info(
            f"Detected backend: {self.device_info['backend']}, "
            f"device: {self.device_info['device_name']}, "
            f"count: {self.device_info['device_count']}"
        )
        # Setup device assignment (handles distributed if initialized)
        self._setup_device()
        # Configure backend-specific optimizations
        setup_backend_settings(backend=self.device_info["backend"])

    def _setup_device(self):
        """Assign the appropriate device based on backend and distributed config."""
        backend = self.device_info["backend"]
        is_distributed = dist.is_available() and dist.is_initialized()
        if is_distributed:
            self.local_rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.local_rank = 0
            self.world_size = 1
        if backend in ("cuda", "rocm"):
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f"cuda:{self.local_rank}")
        elif backend == "tpu":
            import torch_xla.core.xla_model as xm

            self.device = xm.xla_device()
        elif backend == "mps":
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.logger.debug(
            f"Device setup complete: device={self.device}, "
            f"local_rank={self.local_rank}, world_size={self.world_size}"
        )

    def _save_agent(self):
        """Helper method to save agent."""
        save_agent(
            model=self.model,
            checkpoint_dir=self.components.weights_dir,
            device=self.device,
            optimizer=self.optimizer,
            loss_fn=None,
            metadata=None,
        )

    def _setup_inference_engine(self):
        """Setup high-performance inference engine."""
        from src.inference_engine import DattaBotInferenceEngine

        if self.inference_engine is None:
            self.inference_engine = DattaBotInferenceEngine(
                model=self.model,
                device=self.device,
                adaptive_softmax=self.loss_fn,
            )
            self.logger.info("Inference engine initialized and ready for deployment!")

    def _setup_distributed(self):
        assert (
            dist.is_initialized() == True
        ), "Distributed process group is not initialized. Please call dist.is_initialized() first."
        # Ensure this is running under torchrun.
        self.local_rank = dist.get_rank()
        self.logger.debug(
            f"Setting up setup_distributed with local_rank={self.local_rank}"
        )

    def _setup_cpu_gpu_settings(self):
        # torch.manual seed(3407) is all you need
        # https://arxiv.org/pdf/2109.08203
        seed = 3407
        torch.manual_seed(seed)
        backend = self.device_info["backend"]
        # Setup backend.
        if backend in ("cuda", "rocm"):
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = True
            if backend == "cuda" and torch.cuda.get_device_capability()[0] >= 8:
                # TF32 settings (NVIDIA Ampere+ only)
                torch.backends.cuda.matmul.fp32_precision = "tf32"
                torch.backends.cudnn.conv.fp32_precision = "tf32"
            torch.cuda.empty_cache()
        elif backend == "tpu":
            import torch_xla.core.xla_model as xm

            xm.set_rng_state(seed)
        # Setup device.
        if self.device_info["backend"] in ("cuda", "rocm"):
            torch.cuda.set_device(self.local_rank)
            self.device = f"cuda:{self.local_rank}"
        elif self.device_info["backend"] == "tpu":
            import torch_xla.core.xla_model as xm

            self.device = xm.xla_device()
        else:
            self.device = "cpu"

    def _compute_loss(
        self, outputs: Tensor, targets: Tensor, mask: Optional[Tensor] = None
    ) -> tuple[Tensor, Optional[Tensor]]:
        """
        Compute loss using loss function.
        Args:
            outputs: Tensor of shape (batch_size, seq_len, d_model)
            targets: Tensor of shape (batch_size, seq_len)
            mask: Tensor of shape (batch_size, seq_len)
        Returns:
            main_loss: scalar tensor (cross-entropy loss)
            moe_loss: scalar tensor or None (load balancing loss)
        """
        # Flatten to (batch*seq_len, d_model).
        batch_size, seq_len, d_model = outputs.size()
        outputs_flat = outputs.view(batch_size * seq_len, d_model)
        targets_flat = targets.view(batch_size * seq_len)
        if mask is not None:
            mask_flat = mask.view(batch_size * seq_len)
            outputs_flat = outputs_flat[mask_flat]
            targets_flat = targets_flat[mask_flat]
        outputs_flat = outputs_flat.to(self.device)
        targets_flat = targets_flat.to(self.device)
        # Cast to float32 for loss.
        outputs_flat = outputs_flat.float()
        # Main cross-entropy loss.
        loss_output = self.loss_fn(outputs_flat, targets_flat)
        main_loss = loss_output.loss
        # MoE auxiliary loss (only during training).
        moe_loss = None
        if self.training and self.use_moe:
            if isinstance(
                self.model,
                (
                    torch.nn.DataParallel,
                    torch.nn.parallel.DistributedDataParallel,
                ),
            ):
                moe_loss = self.model.module.get_load_balancing_loss()
            else:
                moe_loss = self.model.get_load_balancing_loss()
        return main_loss, moe_loss

    def _get_attention_pad_mask(self, input_ids: Tensor) -> Tensor:
        """
        Get attention pad mask.
        Args:
            input_ids: Tensor of shape (batch_size, seq_len)
        Returns:
            mask: Tensor of shape (batch, 1, seq_len, seq_len)
        """
        # TODO(PiyushDatta): Figure out what to do with the mask, we are padding the sequences.
        #                    This is a temporary solution to unblock the flash attention training.
        return None
        # Attention mask (1 for real tokens, 0 for pad)
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        # [batch, seq_len]
        attention_pad_mask = (input_ids != self.tokenizer.pad_token_id).to(self.device)
        # unsqueeze to create head dim
        # [batch, seq_len] -> [batch, 1, seq_len]
        attention_pad_mask = attention_pad_mask.unsqueeze(1)
        # unsqueeze again to create key/value dim
        # [batch, 1, seq_len] -> [batch, 1, seq_len, 1]
        attention_pad_mask = attention_pad_mask.unsqueeze(-1)
        # expand the last dim to seq_len
        # [batch, 1, seq_len, 1] -> [batch, 1, seq_len, seq_len]
        attention_pad_mask = attention_pad_mask.expand(-1, -1, -1, seq_len)
        expected_shape = (batch_size, 1, seq_len, seq_len)
        assert attention_pad_mask.shape == expected_shape, (
            f"Attention mask shape mismatch! "
            f"Got {attention_pad_mask.shape}, expected {expected_shape}"
        )
        return attention_pad_mask

    def _get_gpu_info(self) -> list[str]:
        gpu_name = "Could not retrieve gpu_name"
        total_memory = "Could not retrieve total_memory"
        total_cores = "Could not retrieve total_cores"
        if torch.cuda.is_available() and self.device != "cpu":
            gpu_name = torch.cuda.get_device_name(self.device)
            # Convert bytes to MB
            total_memory = torch.cuda.get_device_properties(
                self.device
            ).total_memory // (1024**2)
            total_cores = torch.cuda.get_device_properties(
                self.device
            ).multi_processor_count
        return gpu_name, total_memory, total_cores

    @property
    def tokenizer_obj(self):
        """
        Accessor for the tokenizer object.
        Example usage: agent.tokenizer_obj.encode(["Hello"])
        """
        return self.tokenizer

    def train_agent(self) -> DattaBotAPIResponse:
        """Training entry point, just delegate most of the work to the training engine."""
        # Setup data
        self.gpu_profiler.log_gpu_memory("Training - before setup data")
        self.logger.info("Setting up data.")
        train_dataloader, val_dataloader, vocab = self.data_builder.setup_data()
        self.gpu_profiler.log_gpu_memory("Training - after setup data")
        # Run training
        result = self.training_engine.train(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            vocab=vocab,
            mode=TrainingMode.PRETRAIN,
        )
        # Log training details
        total_params = sum(p.numel() for p in self.model.parameters())
        self._log_training_details(
            num_train_tokens_processed=result.response.num_train_tokens_processed,
            num_val_tokens_processed=result.response.num_val_tokens_processed,
            num_train_batches=result.response.num_train_batches,
            num_val_batches=result.response.num_val_batches,
            training_score=result.train_loss,
            validation_score=result.val_loss,
            dataset_name=train_dataloader.dataset_type.value,
            model_name=self.agent_name,
            vocab=result.vocab,
            total_training_time=round(result.total_time, 2),
            total_params_millions=total_params // 1_000_000,
            interrupted=result.interrupted,
        )
        return result.response

    def _log_training_details(
        self,
        num_train_tokens_processed: int,
        num_val_tokens_processed: int,
        num_train_batches: int,
        num_val_batches: int,
        training_score: float,
        validation_score: float,
        dataset_name: str,
        model_name: str,
        vocab: dict[str, int],
        total_training_time: float,
        total_params_millions: int,
        interrupted=False,
    ):
        """Log final training summary using MetricTracker (W&B or other backend)."""
        gpu_name, total_memory, total_cores = self._get_gpu_info()

        if self.metric_tracker.active:
            self.metric_tracker.log_metrics(
                {
                    "summary/model_name": str(model_name),
                    "summary/dataset_name": str(dataset_name),
                    "summary/vocab_length": len(vocab),
                    "summary/batch_size": self.batch_size,
                    "summary/train_batches_completed": num_train_batches,
                    "summary/val_batches_completed": num_val_batches,
                    "summary/train_tokens_processed": num_train_tokens_processed,
                    "summary/val_tokens_processed": num_val_tokens_processed,
                    "summary/train_loss": training_score,
                    "summary/val_loss": validation_score,
                    "summary/total_params_millions": total_params_millions,
                    "summary/training_time_s": total_training_time,
                    "summary/gpu_name": gpu_name,
                    "summary/gpu_memory_MB": total_memory,
                    "summary/gpu_cores": total_cores,
                    "summary/interrupted": interrupted,
                },
                # Run-level summary, not per step
                step=None,
            )

        self.logger.info(
            f"Training summary logged to MetricTracker for model '{model_name}' on dataset '{dataset_name}'."
        )

    def respond_to_queries(self, queries: list[str]) -> list[DattaBotAPIResponse]:
        self.logger.info(f"Processing queries: {queries}")
        self.logger.info(f"Queries length: {len(queries)}")
        # Call the inference engine.
        if self.inference_engine is None:
            self._setup_inference_engine()
        assert self.inference_engine is not None, "Inference engine is not set up."
        # Use inference engine for batch generation.
        responses: list[DattaBotAPIResponse] = self.inference_engine.batch_generate(
            queries=queries
        )
        # Validate responses.
        assert isinstance(responses, list), f"Expected list, got {type(responses)}"
        assert all(
            isinstance(r, DattaBotAPIResponse) for r in responses
        ), "All responses must be DattaBotAPIResponse"
        # Log the results. Only on rank 0.
        if is_rank_0():
            self.logger.debug(f"Query Response for first response: {responses[0].text}")
            self.logger.debug(
                f"Number of Batches for first response: {responses[0].num_train_batches}"
            )
            self.logger.debug(
                f"Tensor Response for the first response: {responses[0].tensor_response}"
            )
            self.logger.debug(f"Number of responses: {len(responses)}")
        return responses

    def convert_queries_to_tensors(self, queries: list[str]) -> tuple[Tensor, int]:
        """
        Encode a list of queries and convert them to a padded tensor.
        Returns:
            - Tensor of shape (batch_size, max_sequence_len)
            - Total number of batches
        """
        if not queries:
            return torch.empty(0, 0, dtype=torch.long), 0
        # Tokenize queries
        tokens = [self.tokenizer.encode(query) for query in queries]
        max_seq_len = self.config.agent.max_response_tokens
        # Pad or truncate to max_seq_len
        padded_tokens = []
        for token_seq in tokens:
            if len(token_seq) > max_seq_len:
                token_seq = token_seq[:max_seq_len]
            else:
                token_seq += [self.tokenizer.pad_token_id] * (
                    max_seq_len - len(token_seq)
                )
            padded_tokens.append(token_seq)
        # Convert to tensor
        tensor = torch.tensor(padded_tokens, dtype=torch.long)
        # Compute number of batches
        num_batches = ceil(len(queries) / self.batch_size)
        return tensor, num_batches

    def profile_training(self, num_steps=20, log_dir: Optional[str] = None):
        """
        Create an AgentProfiler and run the training profiler.
        """
        from src.agent_profiler import AgentProfiler

        profiler = AgentProfiler(self, log_dir=log_dir)
        return profiler.profile_training(num_steps=num_steps)
