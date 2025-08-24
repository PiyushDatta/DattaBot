import csv
import json
import queue
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import psutil
import torch


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
        log_dir: str = "gpu_metrics",
    ):
        self.device = device
        self.is_cuda = torch.cuda.is_available() and device != "cpu"
        self.sampling_interval = sample_every_x_seconds
        self.metrics_queue = queue.Queue()
        self.metrics_history = deque(maxlen=history_size)
        self.should_run = False
        self.profiler_thread = None
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Create CSV file for logging
        self.csv_file = (
            self.log_dir / f"gpu_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        self._initialize_csv()

    def _initialize_csv(self):
        """Initialize CSV file with headers"""
        with open(self.csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "memory_allocated_mb",
                    "memory_reserved_mb",
                    "gpu_utilization_percent",
                    "temperature_celsius",
                    "power_usage_watts",
                    "cpu_percent",
                    "ram_percent",
                ]
            )

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
        # Get utilization using nvidia-smi if available
        utilization = 0
        temperature = None
        power_usage = None
        # Execute nvidia-smi command to get utilization
        try:
            nvidia_smi_command = [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ]
            utilization_output = subprocess.check_output(
                nvidia_smi_command, encoding="utf-8"
            )
            utilization_values = [
                float(x) for x in utilization_output.strip().split("\n") if x
            ]
            utilization = (
                sum(utilization_values) / len(utilization_values)
                if utilization_values
                else 0.0
            )
            nvidia_smi_temp_command = [
                "nvidia-smi",
                "--query-gpu=temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ]
            temp_power_output = subprocess.check_output(
                nvidia_smi_temp_command, encoding="utf-8"
            )
            temp_power_lines = [
                line.strip() for line in temp_power_output.strip().split("\n") if line
            ]
            temps = []
            powers = []
            for line in temp_power_lines:
                temp_str, power_str = line.split(",")
                temps.append(float(temp_str))
                powers.append(float(power_str))
            temperature = sum(temps) / len(temps) if temps else None
            power_usage = sum(powers) / len(powers) if powers else None
        except subprocess.CalledProcessError:
            try:
                if hasattr(torch.cuda, "utilization"):
                    utilization = torch.cuda.utilization()
                if hasattr(torch.cuda, "temperature"):
                    temperature = torch.cuda.temperature()
                if hasattr(torch.cuda, "power_usage"):
                    power_usage = torch.cuda.power_usage()
            except:
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

    def _log_metrics(self, metrics: GPUMetrics):
        """Log metrics to CSV file"""
        with open(self.csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    metrics.timestamp,
                    metrics.memory_allocated,
                    metrics.memory_reserved,
                    metrics.utilization,
                    metrics.temperature,
                    metrics.power_usage,
                    metrics.cpu_percent,
                    metrics.ram_percent,
                ]
            )

    def _profiler_loop(self):
        """Main profiler loop running in background thread"""
        while self.should_run:
            try:
                # Collect metrics
                metrics = self._get_current_metrics()

                # Store in history
                self.metrics_history.append(metrics)

                # Log to CSV
                self._log_metrics(metrics)

                # Put in queue for real-time access
                try:
                    self.metrics_queue.put_nowait(metrics)
                except queue.Full:
                    # If queue is full, remove old item and try again
                    try:
                        self.metrics_queue.get_nowait()
                        self.metrics_queue.put_nowait(metrics)
                    except (queue.Empty, queue.Full):
                        pass

                # Sleep for sampling interval
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
                daemon=True,  # Thread will be killed when main program exits
            )
            self.profiler_thread.start()
            return True
        return False

    def stop(self):
        """Stop the background profiler"""
        self.should_run = False
        if self.profiler_thread is not None:
            self.profiler_thread.join(timeout=5.0)  # Wait up to 5 seconds
            self.profiler_thread = None

        # Save final summary
        self._save_summary()

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

        metrics = {
            "avg_memory_allocated": sum(
                m.memory_allocated for m in self.metrics_history
            )
            / len(self.metrics_history),
            "max_memory_allocated": max(
                m.memory_allocated for m in self.metrics_history
            ),
            "avg_memory_reserved": sum(m.memory_reserved for m in self.metrics_history)
            / len(self.metrics_history),
            "max_memory_reserved": max(m.memory_reserved for m in self.metrics_history),
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

        # Add temperature and power metrics if available
        temps = [
            m.temperature for m in self.metrics_history if m.temperature is not None
        ]
        if temps:
            metrics.update(
                {
                    "avg_temperature": sum(temps) / len(temps),
                    "max_temperature": max(temps),
                }
            )

        power = [
            m.power_usage for m in self.metrics_history if m.power_usage is not None
        ]
        if power:
            metrics.update(
                {
                    "avg_power_usage": sum(power) / len(power),
                    "max_power_usage": max(power),
                }
            )

        return metrics

    def _save_summary(self):
        """Save final summary to JSON file"""
        summary = self.get_summary()
        summary_file = (
            self.log_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=4)
