import pytest
from src.api import DattaBotAPI


def test_smoke_test(monkeypatch):
    """
    Smoke test for DattaBotAPI initialization.
    """
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29500")
    # Instantiate main API class.
    api = DattaBotAPI()
    # Just ensure no exception occurs and object is created,
    assert api is not None
