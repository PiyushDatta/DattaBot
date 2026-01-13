import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from src.checkpointing import (
    _atomic_save_json,
    _atomic_save_pt,
    _atomic_save_safetensors,
    _checkpoint_exists,
    _clean_state_dict_keys,
    _ensure_directory,
    _extract_model_state,
    _extract_optimizer_state,
    _flatten_state_dict,
    _get_checkpoint_paths,
    _get_fsdp_state_dict_options,
    _is_fsdp_model,
    _load_json,
    _load_model_state,
    _load_optimizer_state,
    _log_checkpoint_components,
    _make_json_serializable,
    _should_save_on_this_rank,
    _unwrap_model,
    CheckpointComponents,
    load_agent,
    save_agent,
)
from torch.distributed.fsdp import FSDPModule


# ============================================================================
# Fixtures
# ============================================================================
@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5),
    )
    return model


@pytest.fixture
def simple_optimizer(simple_model):
    """Create a simple optimizer for testing."""
    return torch.optim.Adam(simple_model.parameters(), lr=0.001)


@pytest.fixture
def mock_fsdp_model():
    """Create a mock FSDP model."""
    model = Mock(spec=FSDPModule)
    model.fully_sharded = True
    return model


@pytest.fixture
def mock_logger():
    """Mock the logger."""
    with patch("src.checkpointing.get_logger") as mock_get_logger:
        logger_mock = Mock()
        mock_get_logger.return_value = logger_mock
        yield logger_mock


@pytest.fixture
def checkpoint_components():
    """Create sample checkpoint components."""
    return CheckpointComponents(
        model_state={"layer.weight": torch.randn(10, 5)},
        optimizer_state={"state": {}, "param_groups": [{"lr": 0.001}]},
        loss_fn_state={"decoder.weight": torch.randn(100, 50)},
        metadata={"epoch": 42, "step": 1000},
    )


# ============================================================================
# Unit Tests - FSDP Detection and Options
# ============================================================================


class TestFSDPDetection:
    """Test FSDP model detection."""

    def test_is_fsdp_model_with_fsdp_module(self, mock_fsdp_model):
        """Test detection of FSDPModule."""
        assert _is_fsdp_model(mock_fsdp_model) is True

    def test_is_fsdp_model_with_fully_sharded_attribute(self):
        """Test detection via fully_sharded attribute."""
        model = Mock()
        model.fully_sharded = True
        assert _is_fsdp_model(model) is True

    def test_is_fsdp_model_with_regular_model(self, simple_model):
        """Test that regular models are not detected as FSDP."""
        assert _is_fsdp_model(simple_model) is False

    def test_get_fsdp_state_dict_options_defaults(self):
        """Test default FSDP options."""
        options = _get_fsdp_state_dict_options()
        assert options.full_state_dict is True
        assert options.cpu_offload is True
        assert options.broadcast_from_rank0 is False

    def test_get_fsdp_state_dict_options_custom(self):
        """Test custom FSDP options."""
        options = _get_fsdp_state_dict_options(
            full_state_dict=False,
            cpu_offload=False,
            broadcast_from_rank0=True,
        )
        assert options.full_state_dict is False
        assert options.cpu_offload is False
        assert options.broadcast_from_rank0 is True


# ============================================================================
# Unit Tests - Path Management
# ============================================================================


class TestPathManagement:
    """Test checkpoint path management."""

    def test_get_checkpoint_paths(self, temp_dir):
        """Test getting checkpoint component paths."""
        base_path = Path(temp_dir)
        paths = _get_checkpoint_paths(base_path)

        assert paths["model"] == base_path / "model.safetensors"
        assert paths["optimizer"] == base_path / "optimizer.pt"
        assert paths["loss_fn"] == base_path / "loss_fn.pt"
        assert paths["metadata"] == base_path / "metadata.json"

    def test_ensure_directory_creates_new(self, temp_dir):
        """Test creating a new directory."""
        new_dir = Path(temp_dir) / "new_checkpoint_dir"
        assert not new_dir.exists()

        _ensure_directory(new_dir)

        assert new_dir.exists()
        assert new_dir.is_dir()

    def test_ensure_directory_existing(self, temp_dir):
        """Test that existing directory is handled gracefully."""
        existing_dir = Path(temp_dir)
        assert existing_dir.exists()

        # Should not raise
        _ensure_directory(existing_dir)
        assert existing_dir.exists()

    def test_ensure_directory_nested(self, temp_dir):
        """Test creating nested directories."""
        nested_dir = Path(temp_dir) / "a" / "b" / "c"
        assert not nested_dir.exists()

        _ensure_directory(nested_dir)

        assert nested_dir.exists()

    def test_checkpoint_exists_true(self, temp_dir):
        """Test checkpoint exists when model file present."""
        checkpoint_dir = Path(temp_dir)
        model_path = checkpoint_dir / "model.safetensors"

        # Create empty model file
        model_path.touch()

        assert _checkpoint_exists(checkpoint_dir) is True

    def test_checkpoint_exists_false(self, temp_dir):
        """Test checkpoint does not exist when model file missing."""
        checkpoint_dir = Path(temp_dir)

        assert _checkpoint_exists(checkpoint_dir) is False


# ============================================================================
# Unit Tests - SafeTensors Utilities
# ============================================================================


class TestSafeTensorsUtilities:
    """Test safetensors utility functions."""

    def test_flatten_state_dict_simple(self):
        """Test flattening a simple state dict."""
        state_dict = {
            "weight": torch.randn(5, 5),
            "bias": torch.randn(5),
        }
        flat = _flatten_state_dict(state_dict)

        assert "weight" in flat
        assert "bias" in flat
        assert len(flat) == 2

    def test_flatten_state_dict_nested(self):
        """Test flattening a nested state dict."""
        state_dict = {
            "layer1": {
                "weight": torch.randn(5, 5),
                "bias": torch.randn(5),
            },
            "layer2": {
                "weight": torch.randn(3, 3),
            },
        }
        flat = _flatten_state_dict(state_dict)

        assert "layer1.weight" in flat
        assert "layer1.bias" in flat
        assert "layer2.weight" in flat
        assert len(flat) == 3

    def test_flatten_state_dict_deeply_nested(self):
        """Test flattening a deeply nested state dict."""
        state_dict = {
            "encoder": {
                "layers": {
                    "0": {
                        "weight": torch.randn(5, 5),
                    },
                },
            },
        }
        flat = _flatten_state_dict(state_dict)

        assert "encoder.layers.0.weight" in flat
        assert len(flat) == 1

    def test_flatten_state_dict_with_prefix(self):
        """Test flattening with a prefix."""
        state_dict = {
            "weight": torch.randn(5, 5),
        }
        flat = _flatten_state_dict(state_dict, prefix="model.")

        assert "model.weight" in flat


# ============================================================================
# Unit Tests - JSON Serialization
# ============================================================================


class TestJSONSerialization:
    """Test JSON serialization utilities."""

    def test_make_json_serializable_dict(self):
        """Test serializing a dictionary."""
        data = {"key": "value", "number": 42}
        result = _make_json_serializable(data)

        assert result == data

    def test_make_json_serializable_tensor(self):
        """Test serializing a tensor."""
        tensor = torch.tensor([1, 2, 3])
        result = _make_json_serializable(tensor)

        assert result == [1, 2, 3]

    def test_make_json_serializable_dtype(self):
        """Test serializing a torch dtype."""
        dtype = torch.float32
        result = _make_json_serializable(dtype)

        assert result == "torch.float32"

    def test_make_json_serializable_nested(self):
        """Test serializing nested structure."""
        data = {
            "tensor": torch.tensor([1.0, 2.0]),
            "nested": {
                "dtype": torch.float16,
                "list": [torch.tensor(5)],
            },
        }
        result = _make_json_serializable(data)

        assert result["tensor"] == [1.0, 2.0]
        assert result["nested"]["dtype"] == "torch.float16"
        assert result["nested"]["list"] == [5]

    def test_make_json_serializable_tuple(self):
        """Test serializing a tuple."""
        data = (1, 2, torch.tensor(3))
        result = _make_json_serializable(data)

        assert result == [1, 2, 3]


# ============================================================================
# Unit Tests - State Dict Key Cleaning
# ============================================================================


class TestStateDictCleaning:
    def test_clean_state_dict_keys_ddp_to_regular(self, simple_model, mock_logger):
        """Test removal of 'module.' prefix when loading DDP checkpoint into regular model."""
        state_dict = {
            "module.layer1.weight": torch.randn(5, 5),
            "module.layer2.bias": torch.randn(5),
        }
        cleaned = _clean_state_dict_keys(state_dict, simple_model)

        assert "layer1.weight" in cleaned
        assert "layer2.bias" in cleaned
        assert "module.layer1.weight" not in cleaned
        mock_logger.info.assert_called_once()

    def test_clean_state_dict_keys_regular_to_ddp(self, simple_model, mock_logger):
        """Test addition of 'module.' prefix when loading regular checkpoint into DDP model."""
        state_dict = {
            "layer1.weight": torch.randn(5, 5),
            "layer2.bias": torch.randn(5),
        }

        # Mock DDP model
        ddp_model = Mock(spec=nn.parallel.DistributedDataParallel)

        cleaned = _clean_state_dict_keys(state_dict, ddp_model)

        assert "module.layer1.weight" in cleaned
        assert "module.layer2.bias" in cleaned
        assert "layer1.weight" not in cleaned
        mock_logger.info.assert_called_once()

    def test_clean_state_dict_keys_no_conversion_needed_unwrapped(
        self, simple_model, mock_logger
    ):
        """Test that matching unwrapped types need no conversion."""
        state_dict = {
            "layer1.weight": torch.randn(5, 5),
            "layer2.bias": torch.randn(5),
        }
        cleaned = _clean_state_dict_keys(state_dict, simple_model)

        assert cleaned == state_dict
        mock_logger.info.assert_not_called()

    def test_clean_state_dict_keys_no_conversion_needed_ddp(self, mock_logger):
        """Test that matching DDP types need no conversion."""
        state_dict = {
            "module.layer1.weight": torch.randn(5, 5),
            "module.layer2.bias": torch.randn(5),
        }

        # Mock DDP model
        ddp_model = Mock(spec=nn.parallel.DistributedDataParallel)

        cleaned = _clean_state_dict_keys(state_dict, ddp_model)

        assert cleaned == state_dict
        mock_logger.info.assert_not_called()

    def test_clean_state_dict_keys_dataparallel(self, simple_model, mock_logger):
        """Test that DataParallel is handled same as DDP."""
        state_dict = {
            "layer1.weight": torch.randn(5, 5),
        }

        # Mock DataParallel model
        dp_model = Mock(spec=nn.DataParallel)

        cleaned = _clean_state_dict_keys(state_dict, dp_model)

        assert "module.layer1.weight" in cleaned
        mock_logger.info.assert_called_once()


# ============================================================================
# Unit Tests - Model Unwrapping
# ============================================================================


class TestModelUnwrapping:
    """Test model unwrapping utilities."""

    def test_unwrap_model_ddp(self):
        """Test unwrapping DistributedDataParallel model."""
        base_model = nn.Linear(10, 5)
        # Create a mock DDP wrapper
        ddp_model = Mock(spec=nn.parallel.DistributedDataParallel)
        ddp_model.module = base_model

        unwrapped = _unwrap_model(ddp_model)
        assert unwrapped is base_model

    def test_unwrap_model_dataparallel(self):
        """Test unwrapping DataParallel model."""
        base_model = nn.Linear(10, 5)
        dp_model = Mock(spec=nn.DataParallel)
        dp_model.module = base_model

        unwrapped = _unwrap_model(dp_model)
        assert unwrapped is base_model

    def test_unwrap_model_already_unwrapped(self, simple_model):
        """Test that unwrapping a regular model returns itself."""
        unwrapped = _unwrap_model(simple_model)
        assert unwrapped is simple_model


# ============================================================================
# Unit Tests - Rank Detection
# ============================================================================


class TestRankDetection:
    """Test distributed rank detection."""

    @patch("src.checkpointing.dist.is_available", return_value=False)
    def test_should_save_when_dist_not_available(self, mock_is_available):
        """Test that we save when distributed is not available."""
        assert _should_save_on_this_rank() is True

    @patch("src.checkpointing.dist.is_initialized", return_value=False)
    @patch("src.checkpointing.dist.is_available", return_value=True)
    def test_should_save_when_dist_not_initialized(
        self, mock_is_available, mock_is_initialized
    ):
        """Test that we save when distributed is not initialized."""
        assert _should_save_on_this_rank() is True

    @patch("src.checkpointing.is_rank_0", return_value=True)
    @patch("src.checkpointing.dist.is_initialized", return_value=True)
    @patch("src.checkpointing.dist.is_available", return_value=True)
    def test_should_save_when_rank_0(
        self, mock_is_available, mock_is_initialized, mock_is_rank_0
    ):
        """Test that rank 0 should save."""
        assert _should_save_on_this_rank() is True

    @patch("src.checkpointing.is_rank_0", return_value=False)
    @patch("src.checkpointing.dist.is_initialized", return_value=True)
    @patch("src.checkpointing.dist.is_available", return_value=True)
    def test_should_not_save_when_not_rank_0(
        self, mock_is_available, mock_is_initialized, mock_is_rank_0
    ):
        """Test that non-rank-0 should not save."""
        assert _should_save_on_this_rank() is False


# ============================================================================
# Unit Tests - Atomic File Operations
# ============================================================================


class TestAtomicSaveSafetensors:
    """Test atomic safetensors save operations."""

    def test_atomic_save_safetensors_success(self, temp_dir, mock_logger):
        """Test successful atomic safetensors save."""
        filepath = Path(temp_dir) / "model.safetensors"
        state_dict = {"weight": torch.randn(5, 5), "bias": torch.randn(5)}

        _atomic_save_safetensors(state_dict, filepath)

        assert filepath.exists()
        mock_logger.debug.assert_called()

    def test_atomic_save_safetensors_nested(self, temp_dir, mock_logger):
        """Test saving nested state dict as safetensors."""
        filepath = Path(temp_dir) / "model.safetensors"
        state_dict = {
            "layer1": {
                "weight": torch.randn(5, 5),
            },
        }

        _atomic_save_safetensors(state_dict, filepath)

        assert filepath.exists()

    def test_atomic_save_safetensors_cleanup_on_error(self, temp_dir, mock_logger):
        """Test that temp file is cleaned up on error."""
        filepath = Path(temp_dir) / "model.safetensors"

        with patch(
            "src.checkpointing.save_safetensors",
            side_effect=RuntimeError("Save failed"),
        ):
            with pytest.raises(RuntimeError):
                _atomic_save_safetensors({"weight": torch.randn(5, 5)}, filepath)

        # Check no temp files left behind
        temp_files = [f for f in os.listdir(temp_dir) if f.startswith(".tmp_")]
        assert len(temp_files) == 0


class TestAtomicSavePt:
    """Test atomic PyTorch save operations."""

    def test_atomic_save_pt_success(self, temp_dir, mock_logger):
        """Test successful atomic .pt save."""
        filepath = Path(temp_dir) / "optimizer.pt"
        state_dict = {"state": {}, "param_groups": [{"lr": 0.001}]}

        _atomic_save_pt(state_dict, filepath)

        assert filepath.exists()
        loaded = torch.load(filepath)
        assert loaded["param_groups"][0]["lr"] == 0.001
        mock_logger.debug.assert_called()

    def test_atomic_save_pt_cleanup_on_error(self, temp_dir, mock_logger):
        """Test that temp file is cleaned up on error."""
        filepath = Path(temp_dir) / "optimizer.pt"

        with patch(
            "src.checkpointing.torch.save",
            side_effect=RuntimeError("Save failed"),
        ):
            with pytest.raises(RuntimeError):
                _atomic_save_pt({"data": "test"}, filepath)

        temp_files = [f for f in os.listdir(temp_dir) if f.startswith(".tmp_")]
        assert len(temp_files) == 0


class TestAtomicSaveJson:
    """Test atomic JSON save operations."""

    def test_atomic_save_json_success(self, temp_dir, mock_logger):
        """Test successful atomic JSON save."""
        filepath = Path(temp_dir) / "metadata.json"
        data = {"epoch": 42, "loss": 0.5}

        _atomic_save_json(data, filepath)

        assert filepath.exists()
        with open(filepath, "r") as f:
            loaded = json.load(f)
        assert loaded["epoch"] == 42
        mock_logger.debug.assert_called()

    def test_atomic_save_json_with_tensors(self, temp_dir, mock_logger):
        """Test saving JSON with tensor values (should be serialized)."""
        filepath = Path(temp_dir) / "metadata.json"
        data = {"tensor_val": torch.tensor([1, 2, 3]), "dtype": torch.float32}

        _atomic_save_json(data, filepath)

        assert filepath.exists()
        with open(filepath, "r") as f:
            loaded = json.load(f)
        assert loaded["tensor_val"] == [1, 2, 3]
        assert loaded["dtype"] == "torch.float32"

    def test_atomic_save_json_cleanup_on_error(self, temp_dir, mock_logger):
        """Test that temp file is cleaned up on error."""
        filepath = Path(temp_dir) / "metadata.json"

        # Create object that can't be serialized
        class NonSerializable:
            pass

        with patch(
            "src.checkpointing._make_json_serializable",
            return_value={"bad": NonSerializable()},
        ):
            with pytest.raises(TypeError):
                _atomic_save_json({"data": "test"}, filepath)

        temp_files = [f for f in os.listdir(temp_dir) if f.startswith(".tmp_")]
        assert len(temp_files) == 0


class TestLoadJson:
    """Test JSON loading."""

    def test_load_json_success(self, temp_dir):
        """Test loading a JSON file."""
        filepath = Path(temp_dir) / "test.json"
        data = {"key": "value", "number": 42}

        with open(filepath, "w") as f:
            json.dump(data, f)

        loaded = _load_json(filepath)

        assert loaded == data

    def test_load_json_unicode(self, temp_dir):
        """Test loading JSON with unicode content."""
        filepath = Path(temp_dir) / "test.json"
        data = {"message": "Hello, 世界! 🎉"}

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

        loaded = _load_json(filepath)

        assert loaded["message"] == "Hello, 世界! 🎉"


# ============================================================================
# Unit Tests - Logging
# ============================================================================


class TestLogging:
    """Test checkpoint component logging."""

    def test_log_checkpoint_components_all(self, checkpoint_components, mock_logger):
        """Test logging with all components."""
        _log_checkpoint_components(
            checkpoint_components, "saved", "/path/to/checkpoint"
        )

        call_args = mock_logger.info.call_args[0][0]
        assert "saved" in call_args
        assert "model" in call_args
        assert "optimizer" in call_args
        assert "loss_fn" in call_args
        assert "metadata" in call_args
        assert "/path/to/checkpoint" in call_args

    def test_log_checkpoint_components_model_only(self, mock_logger):
        """Test logging with only model."""
        components = CheckpointComponents(model_state={"weight": torch.randn(5, 5)})
        _log_checkpoint_components(components, "loaded", "/path/to/checkpoint")

        call_args = mock_logger.info.call_args[0][0]
        assert "loaded" in call_args
        assert "model" in call_args
        assert "optimizer" not in call_args
        assert "loss_fn" not in call_args


# ============================================================================
# Integration Tests - State Extraction and Loading
# ============================================================================


@pytest.mark.integration
class TestStateExtraction:
    """Integration tests for state extraction."""

    @patch("src.checkpointing.dist_barrier")
    def test_extract_model_state_regular(self, mock_barrier, simple_model, mock_logger):
        """Test extracting state from regular model."""
        device = torch.device("cpu")
        state_dict = _extract_model_state(simple_model, device)

        assert isinstance(state_dict, dict)
        assert len(state_dict) > 0
        mock_barrier.assert_not_called()  # No barrier for regular models

    @patch("src.checkpointing.dist_barrier")
    @patch("src.checkpointing.get_model_state_dict")
    def test_extract_model_state_fsdp(
        self, mock_get_state, mock_barrier, mock_fsdp_model, mock_logger
    ):
        """Test extracting state from FSDP model."""
        device = torch.device("cpu")
        mock_get_state.return_value = {"fsdp.weight": torch.randn(5, 5)}

        state_dict = _extract_model_state(mock_fsdp_model, device)

        assert isinstance(state_dict, dict)
        assert mock_barrier.call_count == 2  # Before and after
        mock_get_state.assert_called_once()

    @patch("src.checkpointing.dist_barrier")
    def test_extract_optimizer_state_regular(
        self, mock_barrier, simple_model, simple_optimizer, mock_logger
    ):
        """Test extracting optimizer state from regular setup."""
        device = torch.device("cpu")
        state_dict = _extract_optimizer_state(simple_model, simple_optimizer, device)

        assert isinstance(state_dict, dict)
        assert "param_groups" in state_dict
        mock_barrier.assert_not_called()

    @patch("src.checkpointing.dist_barrier")
    @patch("src.checkpointing.get_optimizer_state_dict")
    def test_extract_optimizer_state_fsdp(
        self, mock_get_state, mock_barrier, mock_fsdp_model, mock_logger
    ):
        """Test extracting optimizer state from FSDP setup."""
        device = torch.device("cpu")
        optimizer = Mock()
        mock_get_state.return_value = {"state": {}, "param_groups": []}

        state_dict = _extract_optimizer_state(mock_fsdp_model, optimizer, device)

        assert isinstance(state_dict, dict)
        assert mock_barrier.call_count == 2
        mock_get_state.assert_called_once()


@pytest.mark.integration
class TestStateLoading:
    """Integration tests for state loading."""

    @patch("src.checkpointing.dist_barrier")
    def test_load_model_state_regular(self, mock_barrier, simple_model, mock_logger):
        """Test loading state into regular model."""
        device = torch.device("cpu")
        original_state = simple_model.state_dict()
        new_state = {k: torch.randn_like(v) for k, v in original_state.items()}

        _load_model_state(simple_model, new_state, device, strict=True)

        loaded_state = simple_model.state_dict()
        for key in new_state.keys():
            assert torch.allclose(loaded_state[key], new_state[key])
        mock_barrier.assert_not_called()

    @patch("src.checkpointing.dist_barrier")
    @patch("src.checkpointing.set_model_state_dict")
    def test_load_model_state_fsdp(
        self, mock_set_state, mock_barrier, mock_fsdp_model, mock_logger
    ):
        """Test loading state into FSDP model."""
        device = torch.device("cpu")
        state_dict = {"fsdp.weight": torch.randn(5, 5)}

        _load_model_state(mock_fsdp_model, state_dict, device)

        assert mock_barrier.call_count == 2
        mock_set_state.assert_called_once()

    @patch("src.checkpointing.dist_barrier")
    def test_load_optimizer_state_regular(
        self, mock_barrier, simple_model, simple_optimizer, mock_logger
    ):
        """Test loading optimizer state into regular optimizer."""
        device = torch.device("cpu")
        original_state = simple_optimizer.state_dict()

        # Modify optimizer state
        new_state = original_state.copy()
        new_state["param_groups"][0]["lr"] = 0.999

        _load_optimizer_state(simple_model, simple_optimizer, new_state, device)

        loaded_state = simple_optimizer.state_dict()
        assert loaded_state["param_groups"][0]["lr"] == 0.999
        mock_barrier.assert_not_called()


# ============================================================================
# Integration Tests - Full Save/Load Cycle
# ============================================================================


@pytest.mark.integration
class TestFullSaveLoadCycle:
    """Integration tests for complete save/load workflows."""

    @patch("src.checkpointing.dist_barrier")
    @patch("src.checkpointing._should_save_on_this_rank", return_value=True)
    def test_save_and_load_regular_model(
        self,
        mock_should_save,
        mock_barrier,
        temp_dir,
        simple_model,
        simple_optimizer,
        mock_logger,
    ):
        """Test full save/load cycle with regular model."""
        checkpoint_dir = os.path.join(temp_dir, "checkpoint")
        device = torch.device("cpu")
        metadata = {"epoch": 10, "loss": 0.5}

        # Save
        save_agent(
            model=simple_model,
            checkpoint_dir=checkpoint_dir,
            device=device,
            optimizer=simple_optimizer,
            metadata=metadata,
        )

        # Verify checkpoint files exist
        assert os.path.exists(os.path.join(checkpoint_dir, "model.safetensors"))
        assert os.path.exists(os.path.join(checkpoint_dir, "optimizer.pt"))
        assert os.path.exists(os.path.join(checkpoint_dir, "metadata.json"))

        # Modify model and optimizer
        with torch.no_grad():
            for param in simple_model.parameters():
                param.fill_(0.0)
        simple_optimizer.param_groups[0]["lr"] = 0.999

        # Load
        loaded_metadata = load_agent(
            model=simple_model,
            checkpoint_dir=checkpoint_dir,
            optimizer=simple_optimizer,
            device="cpu",
        )

        assert loaded_metadata == metadata

        # Verify model weights restored
        for param in simple_model.parameters():
            assert not torch.all(param == 0.0)

        # Verify optimizer state restored
        assert simple_optimizer.param_groups[0]["lr"] == 0.001

    # Suppress PyTorch warning about zero-element tensor initialization
    @pytest.mark.filterwarnings(
        "ignore:Initializing zero-element tensors is a no-op:UserWarning"
    )
    @patch("src.checkpointing.dist_barrier")
    @patch("src.checkpointing._should_save_on_this_rank", return_value=True)
    def test_save_and_load_with_loss_fn(
        self,
        mock_should_save,
        mock_barrier,
        temp_dir,
        simple_model,
        simple_optimizer,
        mock_logger,
    ):
        """Test save/load with AdaptiveLogSoftmax."""
        checkpoint_dir = os.path.join(temp_dir, "checkpoint_with_loss")
        device = torch.device("cpu")

        # Create a simple loss function
        loss_fn = nn.AdaptiveLogSoftmaxWithLoss(
            in_features=5,
            n_classes=100,
            cutoffs=[10, 50],
        )

        # Save
        save_agent(
            model=simple_model,
            checkpoint_dir=checkpoint_dir,
            device=device,
            optimizer=simple_optimizer,
            loss_fn=loss_fn,
        )

        # Verify loss_fn file exists
        assert os.path.exists(os.path.join(checkpoint_dir, "loss_fn.pt"))

        # Create new loss function to load into
        new_loss_fn = nn.AdaptiveLogSoftmaxWithLoss(
            in_features=5,
            n_classes=100,
            cutoffs=[10, 50],
        )

        # Load
        load_agent(
            model=simple_model,
            checkpoint_dir=checkpoint_dir,
            optimizer=simple_optimizer,
            device="cpu",
            loss_fn=new_loss_fn,
        )

        # Verify loss function state matches
        original_state = loss_fn.state_dict()
        loaded_state = new_loss_fn.state_dict()
        for key in original_state.keys():
            assert torch.allclose(original_state[key], loaded_state[key])

    @patch("src.checkpointing.dist_barrier")
    @patch("src.checkpointing._should_save_on_this_rank", return_value=False)
    def test_save_non_rank_0_does_not_write(
        self,
        mock_should_save,
        mock_barrier,
        temp_dir,
        simple_model,
        simple_optimizer,
        mock_logger,
    ):
        """Test that non-rank-0 processes don't write files."""
        checkpoint_dir = os.path.join(temp_dir, "checkpoint_rank1")
        device = torch.device("cpu")

        save_agent(
            model=simple_model,
            checkpoint_dir=checkpoint_dir,
            device=device,
            optimizer=simple_optimizer,
        )

        # Directory should not be created by non-rank-0
        assert not os.path.exists(checkpoint_dir)

    @patch("src.checkpointing.dist_barrier")
    def test_load_nonexistent_checkpoint(
        self, mock_barrier, simple_model, simple_optimizer, mock_logger
    ):
        """Test loading from nonexistent checkpoint returns empty metadata."""
        metadata = load_agent(
            model=simple_model,
            checkpoint_dir="/nonexistent/path",
            optimizer=simple_optimizer,
            device="cpu",
        )

        assert metadata == {}
        mock_logger.error.assert_called()


# ============================================================================
# Integration Tests - Error Handling
# ============================================================================


@pytest.mark.integration
class TestErrorHandling:
    """Integration tests for error handling."""

    @patch("src.checkpointing.dist_barrier")
    @patch("src.checkpointing._should_save_on_this_rank", return_value=True)
    @patch(
        "src.checkpointing._atomic_save_safetensors",
        side_effect=RuntimeError("Disk full"),
    )
    def test_save_handles_disk_error(
        self,
        mock_safetensors_save,
        mock_should_save,
        mock_barrier,
        temp_dir,
        simple_model,
        simple_optimizer,
        mock_logger,
    ):
        """Test that save errors are properly caught and logged."""
        checkpoint_dir = os.path.join(temp_dir, "will_fail")
        device = torch.device("cpu")

        with pytest.raises(RuntimeError):
            save_agent(
                model=simple_model,
                checkpoint_dir=checkpoint_dir,
                device=device,
                optimizer=simple_optimizer,
            )

        mock_logger.error.assert_called()
        # Barrier should still be called in finally block
        mock_barrier.assert_called()

    @patch("src.checkpointing.dist_barrier")
    @patch("src.checkpointing._should_save_on_this_rank", return_value=True)
    def test_load_handles_corrupted_safetensors(
        self,
        mock_should_save,
        mock_barrier,
        temp_dir,
        simple_model,
        simple_optimizer,
        mock_logger,
    ):
        """Test loading corrupted safetensors file."""
        checkpoint_dir = os.path.join(temp_dir, "corrupted")
        os.makedirs(checkpoint_dir)

        # Create corrupted safetensors file
        model_path = os.path.join(checkpoint_dir, "model.safetensors")
        with open(model_path, "wb") as f:
            f.write(b"corrupted data")

        with pytest.raises(Exception):
            load_agent(
                model=simple_model,
                checkpoint_dir=checkpoint_dir,
                optimizer=simple_optimizer,
                device="cpu",
            )

        mock_logger.error.assert_called()

    @patch("src.checkpointing.dist_barrier")
    @patch("src.checkpointing._should_save_on_this_rank", return_value=True)
    def test_load_with_incompatible_state_dict(
        self,
        mock_should_save,
        mock_barrier,
        temp_dir,
        simple_model,
        simple_optimizer,
        mock_logger,
    ):
        """Test loading checkpoint with incompatible architecture."""
        checkpoint_dir = os.path.join(temp_dir, "incompatible")
        device = torch.device("cpu")

        # Save current model
        save_agent(
            model=simple_model,
            checkpoint_dir=checkpoint_dir,
            device=device,
            optimizer=simple_optimizer,
        )

        # Create different architecture
        different_model = nn.Sequential(
            nn.Linear(10, 30),  # Different size
            nn.ReLU(),
            nn.Linear(30, 5),
        )
        different_optimizer = torch.optim.Adam(different_model.parameters())

        # Load should fail with strict=True
        with pytest.raises(RuntimeError):
            load_agent(
                model=different_model,
                checkpoint_dir=checkpoint_dir,
                optimizer=different_optimizer,
                device="cpu",
                strict=True,
            )


# ============================================================================
# Integration Tests - Module Prefix Handling
# ============================================================================


@pytest.mark.integration
class TestModulePrefixHandling:
    """Test handling of DataParallel/DDP module prefixes."""

    @patch("src.checkpointing.dist_barrier")
    @patch("src.checkpointing._should_save_on_this_rank", return_value=True)
    def test_load_checkpoint_with_module_prefix(
        self, mock_should_save, mock_barrier, temp_dir, simple_model, mock_logger
    ):
        """Test loading checkpoint saved with DataParallel wrapper."""
        from safetensors.torch import save_file as save_safetensors

        checkpoint_dir = os.path.join(temp_dir, "checkpoint_with_prefix")
        os.makedirs(checkpoint_dir)

        # Create checkpoint with 'module.' prefix
        state_dict = simple_model.state_dict()
        prefixed_state = {f"module.{k}": v for k, v in state_dict.items()}

        # Save model with prefix
        model_path = os.path.join(checkpoint_dir, "model.safetensors")
        save_safetensors(prefixed_state, model_path)

        # Save empty metadata
        metadata_path = os.path.join(checkpoint_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump({}, f)

        # Load into regular model (should strip prefix)
        optimizer = torch.optim.Adam(simple_model.parameters())
        load_agent(
            model=simple_model,
            checkpoint_dir=checkpoint_dir,
            optimizer=optimizer,
            device="cpu",
        )

        # Should log that it's removing prefixes
        assert any("module." in str(call) for call in mock_logger.info.call_args_list)


# ============================================================================
# Performance Tests
# ============================================================================


@pytest.mark.integration
class TestPerformance:
    """Performance and stress tests."""

    @patch("src.checkpointing.dist_barrier")
    @patch("src.checkpointing._should_save_on_this_rank", return_value=True)
    def test_save_large_model(
        self, mock_should_save, mock_barrier, temp_dir, mock_logger
    ):
        """Test saving and loading a large model."""
        # Create larger model
        large_model = nn.Sequential(
            nn.Linear(1000, 2000),
            nn.ReLU(),
            nn.Linear(2000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 100),
        )
        optimizer = torch.optim.Adam(large_model.parameters())

        checkpoint_dir = os.path.join(temp_dir, "large_checkpoint")
        device = torch.device("cpu")

        # Save
        save_agent(
            model=large_model,
            checkpoint_dir=checkpoint_dir,
            device=device,
            optimizer=optimizer,
        )

        # Verify files were created and have reasonable size
        model_path = os.path.join(checkpoint_dir, "model.safetensors")
        assert os.path.exists(model_path)
        file_size = os.path.getsize(model_path)
        # Should be at least 1KB
        assert file_size > 1000

        # Load
        new_model = nn.Sequential(
            nn.Linear(1000, 2000),
            nn.ReLU(),
            nn.Linear(2000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 100),
        )
        new_optimizer = torch.optim.Adam(new_model.parameters())

        load_agent(
            model=new_model,
            checkpoint_dir=checkpoint_dir,
            optimizer=new_optimizer,
            device="cpu",
        )

        # Verify weights match
        for p1, p2 in zip(large_model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2)

    @patch("src.checkpointing.dist_barrier")
    @patch("src.checkpointing._should_save_on_this_rank", return_value=True)
    def test_multiple_save_load_cycles(
        self,
        mock_should_save,
        mock_barrier,
        temp_dir,
        simple_model,
        simple_optimizer,
        mock_logger,
    ):
        """Test multiple save/load cycles to ensure no corruption."""
        checkpoint_dir = os.path.join(temp_dir, "multi_cycle")
        device = torch.device("cpu")

        for i in range(5):
            # Modify model slightly
            with torch.no_grad():
                for param in simple_model.parameters():
                    param.add_(0.1)

            # Save
            save_agent(
                model=simple_model,
                checkpoint_dir=checkpoint_dir,
                device=device,
                optimizer=simple_optimizer,
                metadata={"cycle": i},
            )

            # Load back
            metadata = load_agent(
                model=simple_model,
                checkpoint_dir=checkpoint_dir,
                optimizer=simple_optimizer,
                device="cpu",
            )

            assert metadata["cycle"] == i


# ============================================================================
# Edge Cases
# ============================================================================


@pytest.mark.integration
class TestEdgeCases:
    @patch("src.checkpointing.dist_barrier")
    @patch("src.checkpointing._should_save_on_this_rank", return_value=True)
    def test_save_with_empty_metadata(
        self,
        mock_should_save,
        mock_barrier,
        temp_dir,
        simple_model,
        simple_optimizer,
        mock_logger,
    ):
        """Test saving with None metadata."""
        checkpoint_dir = os.path.join(temp_dir, "empty_metadata")
        device = torch.device("cpu")

        save_agent(
            model=simple_model,
            checkpoint_dir=checkpoint_dir,
            device=device,
            optimizer=simple_optimizer,
            metadata=None,
        )

        metadata = load_agent(
            model=simple_model,
            checkpoint_dir=checkpoint_dir,
            optimizer=simple_optimizer,
            device="cpu",
        )

        assert metadata == {}

    @patch("src.checkpointing.dist_barrier")
    @patch("src.checkpointing._should_save_on_this_rank", return_value=True)
    def test_save_with_complex_metadata(
        self,
        mock_should_save,
        mock_barrier,
        temp_dir,
        simple_model,
        simple_optimizer,
        mock_logger,
    ):
        """Test saving with complex metadata including tensors."""
        checkpoint_dir = os.path.join(temp_dir, "complex_metadata")
        device = torch.device("cpu")

        metadata = {
            "epoch": 42,
            "loss_history": [0.5, 0.4, 0.3],
            "config": {
                "learning_rate": 0.001,
                "batch_size": 32,
            },
            "tensor_stat": torch.tensor([1.0, 2.0, 3.0]),
        }

        save_agent(
            model=simple_model,
            checkpoint_dir=checkpoint_dir,
            device=device,
            optimizer=simple_optimizer,
            metadata=metadata,
        )

        loaded_metadata = load_agent(
            model=simple_model,
            checkpoint_dir=checkpoint_dir,
            optimizer=simple_optimizer,
            device="cpu",
        )

        assert loaded_metadata["epoch"] == 42
        assert loaded_metadata["loss_history"] == [0.5, 0.4, 0.3]
        assert loaded_metadata["config"]["learning_rate"] == 0.001
        # Tensor should have been converted to list
        assert loaded_metadata["tensor_stat"] == [1.0, 2.0, 3.0]

    @patch("src.checkpointing.dist_barrier")
    @patch("src.checkpointing._should_save_on_this_rank", return_value=True)
    def test_save_without_loss_fn_load_with_loss_fn(
        self,
        mock_should_save,
        mock_barrier,
        temp_dir,
        simple_model,
        simple_optimizer,
        mock_logger,
    ):
        """Test loading with loss_fn when checkpoint doesn't have one."""
        checkpoint_dir = os.path.join(temp_dir, "no_loss_fn")
        device = torch.device("cpu")

        # Save without loss function
        save_agent(
            model=simple_model,
            checkpoint_dir=checkpoint_dir,
            device=device,
            optimizer=simple_optimizer,
        )

        # Verify loss_fn file doesn't exist
        assert not os.path.exists(os.path.join(checkpoint_dir, "loss_fn.pt"))

        # Load with loss function
        loss_fn = nn.AdaptiveLogSoftmaxWithLoss(5, 100, [10, 50])
        load_agent(
            model=simple_model,
            checkpoint_dir=checkpoint_dir,
            optimizer=simple_optimizer,
            device="cpu",
            loss_fn=loss_fn,
        )

        # Should warn about missing loss function state
        warning_messages = [str(call) for call in mock_logger.warning.call_args_list]
        found_warning = any(
            "Loss function" in msg and "checkpoint" in msg.lower()
            for msg in warning_messages
        )
        assert found_warning

    @patch("src.checkpointing.dist_barrier")
    @patch("src.checkpointing._should_save_on_this_rank", return_value=True)
    def test_load_with_missing_optimizer_file(
        self,
        mock_should_save,
        mock_barrier,
        temp_dir,
        simple_model,
        simple_optimizer,
        mock_logger,
    ):
        """Test loading when optimizer file is missing."""
        from safetensors.torch import save_file as save_safetensors

        checkpoint_dir = os.path.join(temp_dir, "missing_optimizer")
        os.makedirs(checkpoint_dir)

        # Save only model and metadata
        model_path = os.path.join(checkpoint_dir, "model.safetensors")
        save_safetensors(simple_model.state_dict(), model_path)

        metadata_path = os.path.join(checkpoint_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump({"epoch": 1}, f)

        # Load should work, optimizer state should not be loaded
        metadata = load_agent(
            model=simple_model,
            checkpoint_dir=checkpoint_dir,
            optimizer=simple_optimizer,
            device="cpu",
        )

        assert metadata == {"epoch": 1}
        mock_logger.debug.assert_any_call("No optimizer state found in checkpoint")

    @patch("src.checkpointing.dist_barrier")
    @patch("src.checkpointing._should_save_on_this_rank", return_value=True)
    def test_load_with_missing_metadata_file(
        self,
        mock_should_save,
        mock_barrier,
        temp_dir,
        simple_model,
        simple_optimizer,
        mock_logger,
    ):
        """Test loading when metadata file is missing."""
        from safetensors.torch import save_file as save_safetensors

        checkpoint_dir = os.path.join(temp_dir, "missing_metadata")
        os.makedirs(checkpoint_dir)

        # Save only model and optimizer
        model_path = os.path.join(checkpoint_dir, "model.safetensors")
        save_safetensors(simple_model.state_dict(), model_path)

        optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
        torch.save(simple_optimizer.state_dict(), optimizer_path)

        # Load should work, metadata should be empty
        metadata = load_agent(
            model=simple_model,
            checkpoint_dir=checkpoint_dir,
            optimizer=simple_optimizer,
            device="cpu",
        )

        assert metadata == {}

    @patch("src.checkpointing.dist_barrier")
    @patch("src.checkpointing._should_save_on_this_rank", return_value=True)
    def test_overwrite_existing_checkpoint(
        self,
        mock_should_save,
        mock_barrier,
        temp_dir,
        simple_model,
        simple_optimizer,
        mock_logger,
    ):
        """Test overwriting an existing checkpoint."""
        checkpoint_dir = os.path.join(temp_dir, "overwrite")
        device = torch.device("cpu")

        # Save first checkpoint
        save_agent(
            model=simple_model,
            checkpoint_dir=checkpoint_dir,
            device=device,
            optimizer=simple_optimizer,
            metadata={"version": 1},
        )

        # Modify model
        with torch.no_grad():
            for param in simple_model.parameters():
                param.fill_(99.0)

        # Save second checkpoint (overwrite)
        save_agent(
            model=simple_model,
            checkpoint_dir=checkpoint_dir,
            device=device,
            optimizer=simple_optimizer,
            metadata={"version": 2},
        )

        # Load and verify it's the second version
        metadata = load_agent(
            model=simple_model,
            checkpoint_dir=checkpoint_dir,
            optimizer=simple_optimizer,
            device="cpu",
        )

        assert metadata["version"] == 2

        # Verify model weights are from second save
        for param in simple_model.parameters():
            assert torch.allclose(param, torch.full_like(param, 99.0))


# ============================================================================
# Safetensors-specific Tests
# ============================================================================


@pytest.mark.integration
class TestSafetensorsSpecific:
    """Tests specific to safetensors format."""

    @patch("src.checkpointing.dist_barrier")
    @patch("src.checkpointing._should_save_on_this_rank", return_value=True)
    def test_safetensors_file_can_be_loaded_directly(
        self,
        mock_should_save,
        mock_barrier,
        temp_dir,
        simple_model,
        simple_optimizer,
        mock_logger,
    ):
        """Test that saved safetensors can be loaded with safetensors library directly."""
        from safetensors.torch import load_file as load_safetensors

        checkpoint_dir = os.path.join(temp_dir, "direct_load")
        device = torch.device("cpu")

        save_agent(
            model=simple_model,
            checkpoint_dir=checkpoint_dir,
            device=device,
            optimizer=simple_optimizer,
        )

        # Load directly with safetensors
        model_path = os.path.join(checkpoint_dir, "model.safetensors")
        state_dict = load_safetensors(model_path)

        # Verify all model keys are present
        original_keys = set(simple_model.state_dict().keys())
        loaded_keys = set(state_dict.keys())
        assert original_keys == loaded_keys

    @patch("src.checkpointing.dist_barrier")
    @patch("src.checkpointing._should_save_on_this_rank", return_value=True)
    def test_safetensors_tensors_are_contiguous(
        self,
        mock_should_save,
        mock_barrier,
        temp_dir,
        mock_logger,
    ):
        """Test that saved tensors are contiguous (required by safetensors)."""
        from safetensors.torch import load_file as load_safetensors

        # Create model with potentially non-contiguous tensors
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Linear(20, 10),
        )

        # Make some tensors non-contiguous
        with torch.no_grad():
            model[0].weight = nn.Parameter(model[0].weight.t().t())  # Transpose twice

        optimizer = torch.optim.Adam(model.parameters())
        checkpoint_dir = os.path.join(temp_dir, "contiguous_test")
        device = torch.device("cpu")

        # This should not raise even with non-contiguous tensors
        save_agent(
            model=model,
            checkpoint_dir=checkpoint_dir,
            device=device,
            optimizer=optimizer,
        )

        # Verify the saved tensors can be loaded
        model_path = os.path.join(checkpoint_dir, "model.safetensors")
        state_dict = load_safetensors(model_path)
        assert len(state_dict) > 0
