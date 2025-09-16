import os
from typing import Dict, Optional
import wandb
from src.logger import get_logger


class MetricTracker:
    """
    Singleton class to track and log metrics to Weights & Biases (wandb).
    """

    _instance: Optional["MetricTracker"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(MetricTracker, cls).__new__(cls)
        return cls._instance

    def __init__(
        self,
        project: str = "dattabot",
        run_name: Optional[str] = None,
        log_dir: str = "wandb_logs",
    ):
        if hasattr(self, "_initialized") and self._initialized:
            return

        self.logger = get_logger()
        self.project = project
        self.run_name = run_name
        self.log_dir = log_dir
        self.active = False
        self.global_step = 0  # Track global step internally

        os.makedirs(log_dir, exist_ok=True)

        try:
            wandb.init(
                project=self.project,
                name=self.run_name,
                dir=self.log_dir,
                mode="online" if os.getenv("WANDB_API_KEY") else "offline",
            )
            self.active = True
            run_name_safe = getattr(wandb.run, "name", "unknown")
            run_mode_safe = getattr(wandb.run, "mode", "unknown")
            self.logger.info(f"W&B run started: {run_name_safe} (mode={run_mode_safe})")
        except Exception as e:
            self.logger.error(f"Could not initialize W&B: {e}")
            self.active = False

        self._initialized = True

    def log_metrics(self, metrics: Dict, step: Optional[int] = None):
        """Log a dictionary of metrics to W&B, optionally passing a step."""
        if not self.active:
            return
        try:
            # If step is None, use internal global step
            if step is None:
                step = self.global_step
            wandb.log(metrics, step=step)
            self.global_step = step + 1
        except Exception as e:
            self.logger.error(f"Error logging metrics to W&B: {e}")

    def log_config(self, config: Dict):
        """Log training configuration/hyperparameters."""
        if self.active:
            wandb.config.update(config, allow_val_change=True)

    def finish(self):
        """Finish the W&B run."""
        if self.active:
            wandb.finish()
            self.logger.info("W&B run finished.")

    def get_run_url(self) -> str:
        """Return the W&B run URL, if active."""
        if self.active and wandb.run:
            return wandb.run.url
        return "W&B run not available."


# Global accessor
def get_metric_tracker(
    project: str = "dattabot",
    run_name: Optional[str] = None,
    log_dir: str = "wandb_logs",
) -> MetricTracker:
    return MetricTracker(project=project, run_name=run_name, log_dir=log_dir)
