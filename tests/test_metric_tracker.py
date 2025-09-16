import pytest
from unittest.mock import patch, MagicMock
from src.metric_tracker import MetricTracker


def test_singleton_behavior():
    tracker1 = MetricTracker(project="test_project")
    tracker2 = MetricTracker(project="another_project")
    # Both instances should be the same object
    assert tracker1 is tracker2
    # Original project name should be retained from the first initialization
    assert tracker1.project == "test_project"


@patch("wandb.finish")
@patch("wandb.config", new_callable=MagicMock)
@patch("wandb.log")
@patch("wandb.init")
def test_metric_logging(mock_init, mock_log, mock_config, mock_finish):
    # Create a dummy run object
    dummy_run = MagicMock()
    dummy_run.name = "test_run"
    dummy_run.mode = "offline"
    dummy_run.url = "http://wandb.test/run"
    # Patch wandb.run globally
    with patch("wandb.run", dummy_run):
        # Reset singleton
        MetricTracker._instance = None
        tracker = MetricTracker(project="test_project", run_name="test_run")
        assert tracker.active is True
        # Log some metrics
        tracker.log_metrics({"loss": 0.5}, step=1)
        # Finish the run
        tracker.finish()
        # Check that log_metrics was called (mock_log)
        mock_log.assert_called()


def test_get_run_url(monkeypatch):
    # Reset singleton for testing
    MetricTracker._instance = None

    class DummyRun:
        url = "http://wandb.test/run"

    class DummyWandb:
        run = DummyRun()

    monkeypatch.setattr("src.metric_tracker.wandb", DummyWandb)
    tracker = MetricTracker()
    tracker.active = True

    assert tracker.get_run_url() == "http://wandb.test/run"


def test_get_run_url_inactive():
    MetricTracker._instance = None
    tracker = MetricTracker()
    tracker.active = False

    assert tracker.get_run_url() == "W&B run not available."
