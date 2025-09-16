import threading
import time
from unittest.mock import patch, MagicMock
import pytest
import torch
import psutil

from src.gpu_profiler import BackgroundGPUProfiler, GPUMetrics
from src.metric_tracker import MetricTracker


@pytest.fixture(autouse=True)
def reset_singleton():
    # Ensure MetricTracker singleton is reset between tests
    MetricTracker._instance = None
    yield
    MetricTracker._instance = None


@patch("torch.cuda.is_available", return_value=False)
@patch("psutil.cpu_percent", return_value=10.0)
@patch("psutil.virtual_memory")
def test_get_current_metrics_cpu(mock_virtual_memory, mock_cuda, mock_cpu_percent):
    mock_virtual_memory.return_value.percent = 20.0
    profiler = BackgroundGPUProfiler(device="cpu")
    metrics = profiler._get_current_metrics()
    assert isinstance(metrics, GPUMetrics)
    assert metrics.memory_allocated == 0
    assert metrics.memory_reserved == 0
    assert metrics.utilization == 0
    assert metrics.cpu_percent == 10.0
    assert metrics.ram_percent == 20.0
    assert metrics.timestamp > 0


def test_profiler_loop_logs_metrics(monkeypatch):
    # Patch MetricTracker so no real logging happens
    dummy_metric_tracker = patch("src.gpu_profiler.get_metric_tracker").start()
    dummy_metric_tracker.return_value.log_metrics = lambda *a, **k: None
    profiler = BackgroundGPUProfiler(device="cpu", sample_every_x_seconds=0.01)

    # Patch _get_current_metrics to always return a dummy GPUMetrics
    dummy_metrics = GPUMetrics(
        memory_allocated=0,
        memory_reserved=0,
        utilization=0,
        temperature=None,
        power_usage=None,
        cpu_percent=0,
        ram_percent=0,
        timestamp=0,
    )
    monkeypatch.setattr(profiler, "_get_current_metrics", lambda: dummy_metrics)

    # Start profiler in a thread
    profiler.should_run = True

    def stop_after_first_iteration():
        profiler.should_run = False

    # Use monkeypatch to stop loop after first iteration
    monkeypatch.setattr(time, "sleep", lambda x: stop_after_first_iteration())

    t = threading.Thread(target=profiler._profiler_loop)
    t.start()
    t.join(timeout=1)

    # Test metrics were stored
    assert len(profiler.metrics_history) > 0
    assert isinstance(profiler.metrics_history[0], GPUMetrics)


@patch("time.sleep", return_value=None)
@patch.object(BackgroundGPUProfiler, "_profiler_loop")
def test_start_stop_profiler(mock_loop, mock_sleep):
    profiler = BackgroundGPUProfiler(device="cpu")
    started = profiler.start()
    assert started is True
    assert profiler.profiler_thread.is_alive() or mock_loop.called

    profiler.stop()
    assert profiler.profiler_thread is None
    assert profiler.should_run is False


def test_get_latest_metrics_queue():
    profiler = BackgroundGPUProfiler(device="cpu")
    sample_metric = GPUMetrics(
        memory_allocated=1,
        memory_reserved=2,
        utilization=3,
        temperature=None,
        power_usage=None,
        cpu_percent=4,
        ram_percent=5,
        timestamp=time.time(),
    )
    profiler.metrics_queue.put(sample_metric)
    latest = profiler.get_latest_metrics()
    assert latest == sample_metric


def test_get_summary_calculation():
    profiler = BackgroundGPUProfiler(device="cpu")
    metrics1 = GPUMetrics(1, 1, 10, None, None, 20, 30, time.time())
    metrics2 = GPUMetrics(2, 2, 20, None, None, 30, 40, time.time())
    profiler.metrics_history.extend([metrics1, metrics2])

    summary = profiler.get_summary()
    assert summary["avg_memory_allocated"] == 1.5
    assert summary["max_memory_allocated"] == 2
    assert summary["avg_utilization"] == 15
    assert summary["max_utilization"] == 20
    assert summary["avg_cpu_percent"] == 25
    assert summary["max_cpu_percent"] == 30
    assert summary["avg_ram_percent"] == 35
    assert summary["max_ram_percent"] == 40
