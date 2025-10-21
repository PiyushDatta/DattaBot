import os

import torch
import torch.profiler
from src.agent_config import get_agent_config
from src.logger import get_logger
from src.util import get_logging_level_from_config, is_device_cpu, is_rank_0


class AgentProfiler:
    def __init__(self, agent, log_dir=None):
        self.agent = agent
        self.config = get_agent_config()
        self.logger = get_logger(
            logging_level=get_logging_level_from_config(self.config)
        )
        if log_dir is None:
            data_dir = self.config.agent.data_directory
            log_dir = os.path.join(data_dir, "agent_train_profiler")
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    def profile_training(self, num_steps=20) -> None:
        """
        Profiles the agent's training loop for a limited number of steps.
        Only writes the profile if rank 0.
        """
        # Setup data
        self.agent.new_training_session()
        train_dataloader, _, _ = self.agent.data_builder.setup_data()
        self.agent.model.train()

        # Use PyTorch Profiler
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=8, repeat=1),
            on_trace_ready=(
                torch.profiler.tensorboard_trace_handler(self.log_dir)
                if is_rank_0()
                else None
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            for step, batch in enumerate(train_dataloader):
                if step >= num_steps:
                    break
                input_ids = batch[0].to(self.agent.agent_device)
                labels = batch[1].to(self.agent.agent_device)
                attention_pad_mask = self.agent._get_attention_pad_mask(
                    input_ids=input_ids
                )
                self.agent.optimizer.zero_grad()
                with torch.autocast(
                    device_type=self.agent.agent_device.type,
                    enabled=(not is_device_cpu(self.agent.agent_device.type)),
                    dtype=torch.bfloat16,
                ):
                    logits = self.agent.model(
                        input_ids=input_ids, attention_pad_mask=attention_pad_mask
                    )
                    padding_mask = labels != self.agent.tokenizer.pad_token_id
                    main_loss, moe_loss = self.agent._compute_loss(
                        outputs=logits, targets=labels, mask=padding_mask
                    )
                    loss = main_loss + (
                        self.agent.moe_weight * moe_loss if moe_loss is not None else 0
                    )
                if self.agent.scaler:
                    self.agent.scaler.scale(loss).backward()
                    self.agent.scaler.step(self.agent.optimizer)
                    self.agent.scaler.update()
                else:
                    loss.backward()
                    self.agent.optimizer.step()
                if self.agent.lr_scheduler:
                    self.agent.lr_scheduler.step()
                prof.step()
        if is_rank_0():
            self.logger.info(
                f"Profiling complete. View results in TensorBoard with:\n  tensorboard --logdir={self.log_dir}"
            )
