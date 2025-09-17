import queue
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional

import psutil
import torch

from src.metric_tracker import MetricTracker, get_metric_tracker


@dataclass
class GPUMetrics:
    memory_allocated: float  # In MB
    memory_reserved: float  # In MB
    utilization: float  # In percentage
    temperature: Optional[float]  # In Celsius
    power_usage: Optional[float]  # In Watts
    cpu_percent: float  # CPU utilization
    ram_percent: float  # RAM utilization
    timestamp: float


class BackgroundGPUProfiler:
    def __init__(
        self,
        device: str,
        sample_every_x_seconds: float = 1.0,
        history_size: int = 3600,  # Keep 1 hour of history at 1s intervals
    ):
        self.device = device
        self.is_cuda = torch.cuda.is_available() and device != "cpu"
        self.sampling_interval = sample_every_x_seconds
        self.metrics_queue = queue.Queue()
        self.metrics_history = deque(maxlen=history_size)
        self.should_run = False
        self.profiler_thread = None
        self.metric_tracker: MetricTracker = get_metric_tracker()

    def _get_current_metrics(self) -> GPUMetrics:
        """Collect current GPU and system metrics"""
        if not self.is_cuda:
            return GPUMetrics(
                0,
                0,
                0,
                None,
                None,
                psutil.cpu_percent(),
                psutil.virtual_memory().percent,
                time.time(),
            )

        device_idx = torch.cuda.current_device()

        # Memory metrics (convert to MB)
        memory_allocated = torch.cuda.memory_allocated(device_idx) / 1024**2
        memory_reserved = torch.cuda.memory_reserved(device_idx) / 1024**2

        utilization, temperature, power_usage = 0, None, None
        try:
            # Use nvidia-smi for GPU utilization, temperature, and power
            utilization_output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                encoding="utf-8",
            )
            utilization_values = [
                float(x) for x in utilization_output.strip().split("\n") if x
            ]
            utilization = (
                sum(utilization_values) / len(utilization_values)
                if utilization_values
                else 0.0
            )

            temp_power_output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                encoding="utf-8",
            )
            temps, powers = [], []
            for line in temp_power_output.strip().split("\n"):
                if not line:
                    continue
                temp_str, power_str = line.split(",")
                temps.append(float(temp_str))
                powers.append(float(power_str))
            temperature = sum(temps) / len(temps) if temps else None
            power_usage = sum(powers) / len(powers) if powers else None

        except subprocess.CalledProcessError:
            pass

        return GPUMetrics(
            memory_allocated=memory_allocated,
            memory_reserved=memory_reserved,
            utilization=utilization,
            temperature=temperature,
            power_usage=power_usage,
            cpu_percent=psutil.cpu_percent(),
            ram_percent=psutil.virtual_memory().percent,
            timestamp=time.time(),
        )

    def _profiler_loop(self):
        """Main profiler loop running in background thread"""
        while self.should_run:
            try:
                metrics = self._get_current_metrics()
                self.metrics_history.append(metrics)

                # Log to global MetricTracker
                self.metric_tracker.log_metrics(
                    {
                        "system/gpu_memory_allocated_mb": metrics.memory_allocated,
                        "system/gpu_memory_reserved_mb": metrics.memory_reserved,
                        "system/gpu_utilization_percent": metrics.utilization,
                        "system/gpu_temperature_celsius": metrics.temperature,
                        "system/gpu_power_watts": metrics.power_usage,
                        "system/cpu_percent": metrics.cpu_percent,
                        "system/ram_percent": metrics.ram_percent,
                    },
                    step=None,
                )

                # Store in queue for programmatic access
                try:
                    self.metrics_queue.put_nowait(metrics)
                except queue.Full:
                    try:
                        self.metrics_queue.get_nowait()
                        self.metrics_queue.put_nowait(metrics)
                    except (queue.Empty, queue.Full):
                        pass

                time.sleep(self.sampling_interval)

            except Exception as e:
                print(f"Error in profiler loop: {e}")
                time.sleep(self.sampling_interval)

    def start(self):
        """Start the background profiler"""
        if self.profiler_thread is None or not self.profiler_thread.is_alive():
            self.should_run = True
            self.profiler_thread = threading.Thread(
                target=self._profiler_loop,
                daemon=True,
            )
            self.profiler_thread.start()
            return True
        return False

    def stop(self):
        """Stop the background profiler"""
        self.should_run = False
        if self.profiler_thread is not None:
            self.profiler_thread.join(timeout=5.0)
            self.profiler_thread = None

    def get_latest_metrics(self) -> Optional[GPUMetrics]:
        """Get the most recent metrics (non-blocking)"""
        try:
            return self.metrics_queue.get_nowait()
        except queue.Empty:
            return None

    def get_summary(self) -> Dict:
        """Generate summary statistics from collected metrics"""
        if not self.metrics_history:
            return {}
        return {
            "avg_memory_allocated": sum(
                m.memory_allocated for m in self.metrics_history
            )
            / len(self.metrics_history),
            "max_memory_allocated": max(
                m.memory_allocated for m in self.metrics_history
            ),
            "avg_utilization": sum(m.utilization for m in self.metrics_history)
            / len(self.metrics_history),
            "max_utilization": max(m.utilization for m in self.metrics_history),
            "avg_cpu_percent": sum(m.cpu_percent for m in self.metrics_history)
            / len(self.metrics_history),
            "max_cpu_percent": max(m.cpu_percent for m in self.metrics_history),
            "avg_ram_percent": sum(m.ram_percent for m in self.metrics_history)
            / len(self.metrics_history),
            "max_ram_percent": max(m.ram_percent for m in self.metrics_history),
        }
