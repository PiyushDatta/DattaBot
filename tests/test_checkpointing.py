"""
Tests for DattaBot checkpointing module.

Tests the singleton DattaBotCheckpointManager, CheckpointBundle,
and related dataclasses for saving/loading model states.
"""

import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn
from src.checkpointing import (
    CheckpointBundle,
    CheckpointMetadata,
    CheckpointPaths,
    DattaBotCheckpointManager,
    _clean_state_dict_keys,
    _flatten_state_dict,
    _make_json_serializable,
    _should_save_on_this_rank,
    get_checkpoint_manager,
)
from src.util import Singleton


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
def simple_loss_fn():
    """Create a simple loss function for testing."""
    return nn.AdaptiveLogSoftmaxWithLoss(
        in_features=5,
        n_classes=100,
        cutoffs=[10, 50],
    )


@pytest.fixture
def mock_logger():
    """Mock the logger."""
    with patch("src.checkpointing.get_logger") as mock_get_logger:
        logger_mock = Mock()
        mock_get_logger.return_value = logger_mock
        yield logger_mock


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the singleton before each test."""
    # Clear the singleton instance before each test
    if DattaBotCheckpointManager in Singleton._instances:
        del Singleton._instances[DattaBotCheckpointManager]
    yield
    # Clean up after test
    if DattaBotCheckpointManager in Singleton._instances:
        del Singleton._instances[DattaBotCheckpointManager]


# ============================================================================
# Unit Tests - CheckpointMetadata
# ============================================================================


class TestCheckpointMetadata:
    """Test CheckpointMetadata dataclass."""

    def test_default_values(self):
        """Test default metadata values."""
        metadata = CheckpointMetadata()
        assert metadata.version == "1.0"
        assert metadata.epoch == 0
        assert metadata.global_step == 0
        assert metadata.train_loss is None
        assert metadata.val_loss is None
        assert metadata.tokens_processed == 0
        assert metadata.agent_name == ""
        assert metadata.extra == {}

    def test_custom_values(self):
        """Test metadata with custom values."""
        metadata = CheckpointMetadata(
            epoch=10,
            global_step=1000,
            train_loss=0.5,
            val_loss=0.6,
            tokens_processed=1000000,
            agent_name="test_agent",
        )
        assert metadata.epoch == 10
        assert metadata.global_step == 1000
        assert metadata.train_loss == 0.5
        assert metadata.val_loss == 0.6
        assert metadata.tokens_processed == 1000000
        assert metadata.agent_name == "test_agent"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metadata = CheckpointMetadata(epoch=5, train_loss=0.3)
        data = metadata.to_dict()

        assert isinstance(data, dict)
        assert data["epoch"] == 5
        assert data["train_loss"] == 0.3
        assert "version" in data
        assert "created_at" in data

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "version": "1.0",
            "epoch": 15,
            "global_step": 500,
            "train_loss": 0.4,
            "created_at": "2024-01-01T00:00:00",
        }
        metadata = CheckpointMetadata.from_dict(data)

        assert metadata.epoch == 15
        assert metadata.global_step == 500
        assert metadata.train_loss == 0.4

    def test_from_dict_with_unknown_fields(self):
        """Test that unknown fields are stored in extra."""
        data = {
            "epoch": 5,
            "unknown_field": "value",
            "another_unknown": 123,
        }
        metadata = CheckpointMetadata.from_dict(data)

        assert metadata.epoch == 5
        assert metadata.extra["unknown_field"] == "value"
        assert metadata.extra["another_unknown"] == 123


# ============================================================================
# Unit Tests - CheckpointPaths
# ============================================================================


class TestCheckpointPaths:
    """Test CheckpointPaths dataclass."""

    def test_from_base_dir(self, temp_dir):
        """Test creating paths from base directory."""
        base_dir = Path(temp_dir)
        paths = CheckpointPaths.from_base_dir(base_dir)

        assert paths.base_dir == base_dir
        assert paths.model == base_dir / "model.safetensors"
        assert paths.optimizer == base_dir / "optimizer.pt"
        assert paths.loss_fn == base_dir / "loss_fn.pt"
        assert paths.metadata == base_dir / "metadata.json"

    def test_exists_true(self, temp_dir):
        """Test exists returns True when model file present."""
        base_dir = Path(temp_dir)
        paths = CheckpointPaths.from_base_dir(base_dir)

        # Create model file
        paths.model.touch()

        assert paths.exists() is True

    def test_exists_false(self, temp_dir):
        """Test exists returns False when model file missing."""
        base_dir = Path(temp_dir)
        paths = CheckpointPaths.from_base_dir(base_dir)

        assert paths.exists() is False


# ============================================================================
# Unit Tests - CheckpointBundle
# ============================================================================


class TestCheckpointBundle:
    """Test CheckpointBundle dataclass."""

    def test_default_values(self):
        """Test default bundle values."""
        bundle = CheckpointBundle()

        assert bundle.unwrapped_model is None
        assert bundle.wrapped_model is None
        assert bundle.optimizer is None
        assert bundle.loss_fn is None
        assert bundle.device is None
        assert bundle.epoch == 0
        assert bundle.global_step == 0
        assert bundle.tokens_processed == 0
        assert bundle.train_loss is None
        assert bundle.val_loss is None

    def test_update_state(self):
        """Test updating mutable state."""
        bundle = CheckpointBundle()
        bundle.update_state(
            epoch=5,
            global_step=1000,
            tokens_processed=50000,
            train_loss=0.5,
            val_loss=0.6,
        )

        assert bundle.epoch == 5
        assert bundle.global_step == 1000
        assert bundle.tokens_processed == 50000
        assert bundle.train_loss == 0.5
        assert bundle.val_loss == 0.6

    def test_update_state_partial(self):
        """Test partial state update."""
        bundle = CheckpointBundle()
        bundle.epoch = 10
        bundle.global_step = 500

        bundle.update_state(epoch=15)

        assert bundle.epoch == 15
        assert bundle.global_step == 500  # Unchanged

    def test_update_state_chaining(self):
        """Test that update_state returns self for chaining."""
        bundle = CheckpointBundle()
        result = bundle.update_state(epoch=5)

        assert result is bundle

    def test_is_configured_false(self):
        """Test is_configured returns False when not configured."""
        bundle = CheckpointBundle()
        assert bundle.is_configured() is False

    def test_is_configured_true(self, simple_model):
        """Test is_configured returns True when configured."""
        bundle = CheckpointBundle()
        bundle.unwrapped_model = simple_model
        bundle.device = torch.device("cpu")

        assert bundle.is_configured() is True

    def test_is_configured_partial(self, simple_model):
        """Test is_configured returns False with only model."""
        bundle = CheckpointBundle()
        bundle.unwrapped_model = simple_model

        assert bundle.is_configured() is False


# ============================================================================
# Unit Tests - DattaBotCheckpointManager
# ============================================================================


class TestCheckpointManagerInit:
    """Test DattaBotCheckpointManager initialization."""

    def test_init_with_checkpoint_dir(self, temp_dir, mock_logger):
        """Test initialization with checkpoint directory."""
        manager = DattaBotCheckpointManager(checkpoint_dir=temp_dir)

        assert manager.checkpoint_dir == Path(temp_dir)
        assert isinstance(manager.bundle, CheckpointBundle)
        assert isinstance(manager.paths, CheckpointPaths)

    def test_init_without_checkpoint_dir_raises(self, mock_logger):
        """Test initialization without checkpoint_dir raises error."""
        with pytest.raises(ValueError, match="requires checkpoint_dir"):
            DattaBotCheckpointManager()

    def test_singleton_behavior(self, temp_dir, mock_logger):
        """Test that manager is a singleton."""
        manager1 = DattaBotCheckpointManager(checkpoint_dir=temp_dir)
        manager2 = DattaBotCheckpointManager()

        assert manager1 is manager2

    def test_configure_reconfigures(self, temp_dir, mock_logger):
        """Test that configure updates the checkpoint directory."""
        manager = DattaBotCheckpointManager(checkpoint_dir=temp_dir)
        new_dir = Path(temp_dir) / "new_dir"

        manager.configure(new_dir)

        assert manager.checkpoint_dir == new_dir

    def test_get_checkpoint_manager_function(self, temp_dir, mock_logger):
        """Test the get_checkpoint_manager helper function."""
        manager = get_checkpoint_manager(checkpoint_dir=temp_dir)

        assert isinstance(manager, DattaBotCheckpointManager)
        assert manager.checkpoint_dir == Path(temp_dir)


# ============================================================================
# Unit Tests - Helper Functions
# ============================================================================


class TestFlattenStateDict:
    """Test _flatten_state_dict function."""

    def test_simple_state_dict(self):
        """Test flattening a simple state dict."""
        state_dict = {
            "weight": torch.randn(5, 5),
            "bias": torch.randn(5),
        }
        flat = _flatten_state_dict(state_dict)

        assert "weight" in flat
        assert "bias" in flat
        assert len(flat) == 2

    def test_nested_state_dict(self):
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

    def test_deeply_nested_state_dict(self):
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

    def test_with_prefix(self):
        """Test flattening with a prefix."""
        state_dict = {"weight": torch.randn(5, 5)}
        flat = _flatten_state_dict(state_dict, prefix="model.")

        assert "model.weight" in flat


class TestMakeJsonSerializable:
    """Test _make_json_serializable function."""

    def test_dict(self):
        """Test serializing a dictionary."""
        data = {"key": "value", "number": 42}
        result = _make_json_serializable(data)

        assert result == data

    def test_tensor(self):
        """Test serializing a tensor."""
        tensor = torch.tensor([1, 2, 3])
        result = _make_json_serializable(tensor)

        assert result == [1, 2, 3]

    def test_dtype(self):
        """Test serializing a torch dtype."""
        dtype = torch.float32
        result = _make_json_serializable(dtype)

        assert result == "torch.float32"

    def test_nested(self):
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

    def test_tuple(self):
        """Test serializing a tuple."""
        data = (1, 2, torch.tensor(3))
        result = _make_json_serializable(data)

        assert result == [1, 2, 3]


class TestCleanStateDictKeys:
    """Test _clean_state_dict_keys function."""

    def test_ddp_to_regular(self, simple_model, mock_logger):
        """Test removal of 'module.' prefix for DDP checkpoint into regular model."""
        state_dict = {
            "module.0.weight": torch.randn(20, 10),
            "module.0.bias": torch.randn(20),
        }
        cleaned = _clean_state_dict_keys(state_dict, simple_model)

        assert "0.weight" in cleaned
        assert "0.bias" in cleaned
        assert "module.0.weight" not in cleaned

    def test_regular_to_ddp(self, mock_logger):
        """Test addition of 'module.' prefix for regular checkpoint into DDP model."""
        state_dict = {
            "0.weight": torch.randn(20, 10),
            "0.bias": torch.randn(20),
        }
        ddp_model = Mock(spec=nn.parallel.DistributedDataParallel)

        cleaned = _clean_state_dict_keys(state_dict, ddp_model)

        assert "module.0.weight" in cleaned
        assert "module.0.bias" in cleaned

    def test_no_conversion_needed(self, simple_model, mock_logger):
        """Test that matching types need no conversion."""
        state_dict = {
            "0.weight": torch.randn(20, 10),
            "0.bias": torch.randn(20),
        }
        cleaned = _clean_state_dict_keys(state_dict, simple_model)

        assert cleaned == state_dict


class TestShouldSaveOnThisRank:
    """Test _should_save_on_this_rank function."""

    @patch("src.checkpointing.dist.is_available", return_value=False)
    def test_dist_not_available(self, mock_is_available):
        """Test returns True when distributed is not available."""
        assert _should_save_on_this_rank() is True

    @patch("src.checkpointing.dist.is_initialized", return_value=False)
    @patch("src.checkpointing.dist.is_available", return_value=True)
    def test_dist_not_initialized(self, mock_is_available, mock_is_initialized):
        """Test returns True when distributed is not initialized."""
        assert _should_save_on_this_rank() is True

    @patch("src.checkpointing.is_rank_0", return_value=True)
    @patch("src.checkpointing.dist.is_initialized", return_value=True)
    @patch("src.checkpointing.dist.is_available", return_value=True)
    def test_rank_0(self, mock_is_available, mock_is_initialized, mock_is_rank_0):
        """Test rank 0 should save."""
        assert _should_save_on_this_rank() is True

    @patch("src.checkpointing.is_rank_0", return_value=False)
    @patch("src.checkpointing.dist.is_initialized", return_value=True)
    @patch("src.checkpointing.dist.is_available", return_value=True)
    def test_not_rank_0(self, mock_is_available, mock_is_initialized, mock_is_rank_0):
        """Test non-rank-0 should not save."""
        assert _should_save_on_this_rank() is False


# ============================================================================
# Integration Tests - Save and Load
# ============================================================================


@pytest.mark.integration
class TestSaveAgent:
    """Integration tests for save_agent method."""

    @patch("src.checkpointing.dist_barrier")
    @patch("src.checkpointing._should_save_on_this_rank", return_value=True)
    def test_save_agent_basic(
        self,
        mock_should_save,
        mock_barrier,
        temp_dir,
        simple_model,
        simple_optimizer,
        mock_logger,
    ):
        """Test basic save_agent functionality."""
        manager = DattaBotCheckpointManager(checkpoint_dir=temp_dir)
        manager.bundle.unwrapped_model = simple_model
        manager.bundle.optimizer = simple_optimizer
        manager.bundle.device = torch.device("cpu")
        manager.bundle.update_state(epoch=5, global_step=100, train_loss=0.5)

        manager.save_agent()

        # Verify files exist
        assert (Path(temp_dir) / "model.safetensors").exists()
        assert (Path(temp_dir) / "optimizer.pt").exists()
        assert (Path(temp_dir) / "metadata.json").exists()

    @patch("src.checkpointing.dist_barrier")
    @patch("src.checkpointing._should_save_on_this_rank", return_value=True)
    def test_save_agent_with_loss_fn(
        self,
        mock_should_save,
        mock_barrier,
        temp_dir,
        simple_model,
        simple_optimizer,
        simple_loss_fn,
        mock_logger,
    ):
        """Test save_agent with loss function."""
        manager = DattaBotCheckpointManager(checkpoint_dir=temp_dir)
        manager.bundle.unwrapped_model = simple_model
        manager.bundle.optimizer = simple_optimizer
        manager.bundle.loss_fn = simple_loss_fn
        manager.bundle.device = torch.device("cpu")

        manager.save_agent()

        assert (Path(temp_dir) / "loss_fn.pt").exists()

    @patch("src.checkpointing.dist_barrier")
    @patch("src.checkpointing._should_save_on_this_rank", return_value=True)
    def test_save_agent_metadata(
        self,
        mock_should_save,
        mock_barrier,
        temp_dir,
        simple_model,
        mock_logger,
    ):
        """Test that metadata is saved correctly."""
        manager = DattaBotCheckpointManager(checkpoint_dir=temp_dir)
        manager.bundle.unwrapped_model = simple_model
        manager.bundle.device = torch.device("cpu")
        manager.bundle.update_state(
            epoch=10,
            global_step=500,
            tokens_processed=100000,
            train_loss=0.3,
            val_loss=0.4,
        )

        manager.save_agent()

        # Load and verify metadata
        with open(Path(temp_dir) / "metadata.json") as f:
            metadata = json.load(f)

        assert metadata["epoch"] == 10
        assert metadata["global_step"] == 500
        assert metadata["tokens_processed"] == 100000
        assert metadata["train_loss"] == 0.3
        assert metadata["val_loss"] == 0.4

    @patch("src.checkpointing.dist_barrier")
    @patch("src.checkpointing._should_save_on_this_rank", return_value=False)
    def test_save_agent_non_rank_0(
        self,
        mock_should_save,
        mock_barrier,
        temp_dir,
        simple_model,
        mock_logger,
    ):
        """Test that non-rank-0 does not write files."""
        manager = DattaBotCheckpointManager(checkpoint_dir=temp_dir)
        manager.bundle.unwrapped_model = simple_model
        manager.bundle.device = torch.device("cpu")

        manager.save_agent()

        # Directory should be empty (no files created)
        assert not (Path(temp_dir) / "model.safetensors").exists()

    def test_save_agent_unconfigured_raises(self, temp_dir, mock_logger):
        """Test that save_agent raises when bundle not configured."""
        manager = DattaBotCheckpointManager(checkpoint_dir=temp_dir)

        with pytest.raises(ValueError, match="Bundle not configured"):
            manager.save_agent()


@pytest.mark.integration
class TestLoadAgent:
    """Integration tests for load_agent method."""

    @patch("src.checkpointing.dist_barrier")
    @patch("src.checkpointing._should_save_on_this_rank", return_value=True)
    def test_load_agent_basic(
        self,
        mock_should_save,
        mock_barrier,
        temp_dir,
        simple_model,
        simple_optimizer,
        mock_logger,
    ):
        """Test basic load_agent functionality."""
        manager = DattaBotCheckpointManager(checkpoint_dir=temp_dir)
        manager.bundle.unwrapped_model = simple_model
        manager.bundle.optimizer = simple_optimizer
        manager.bundle.device = torch.device("cpu")
        manager.bundle.update_state(epoch=5, global_step=100, train_loss=0.5)

        # Save first
        manager.save_agent()

        # Modify model
        with torch.no_grad():
            for param in simple_model.parameters():
                param.fill_(0.0)

        # Reset state
        manager.bundle.update_state(epoch=0, global_step=0)

        # Load
        metadata = manager.load_agent()

        # Verify metadata restored
        assert metadata.epoch == 5
        assert metadata.global_step == 100
        assert metadata.train_loss == 0.5

        # Verify bundle state updated
        assert manager.bundle.epoch == 5
        assert manager.bundle.global_step == 100

        # Verify model weights restored (not all zeros)
        for param in simple_model.parameters():
            assert not torch.all(param == 0.0)

    @patch("src.checkpointing.dist_barrier")
    @patch("src.checkpointing._should_save_on_this_rank", return_value=True)
    def test_load_agent_optimizer_state(
        self,
        mock_should_save,
        mock_barrier,
        temp_dir,
        simple_model,
        simple_optimizer,
        mock_logger,
    ):
        """Test that optimizer state is restored."""
        manager = DattaBotCheckpointManager(checkpoint_dir=temp_dir)
        manager.bundle.unwrapped_model = simple_model
        manager.bundle.optimizer = simple_optimizer
        manager.bundle.device = torch.device("cpu")

        # Save
        manager.save_agent()

        # Modify optimizer lr
        simple_optimizer.param_groups[0]["lr"] = 0.999

        # Load
        manager.load_agent()

        # Verify optimizer state restored
        assert simple_optimizer.param_groups[0]["lr"] == 0.001

    @patch("src.checkpointing.dist_barrier")
    def test_load_agent_nonexistent(
        self,
        mock_barrier,
        temp_dir,
        simple_model,
        mock_logger,
    ):
        """Test loading from nonexistent checkpoint returns empty metadata."""
        manager = DattaBotCheckpointManager(checkpoint_dir=temp_dir)
        manager.bundle.unwrapped_model = simple_model
        manager.bundle.device = torch.device("cpu")

        metadata = manager.load_agent()

        assert metadata.epoch == 0
        assert metadata.global_step == 0

    def test_load_agent_unconfigured_raises(self, temp_dir, mock_logger):
        """Test that load_agent raises when bundle not configured."""
        manager = DattaBotCheckpointManager(checkpoint_dir=temp_dir)

        with pytest.raises(ValueError, match="Bundle not configured"):
            manager.load_agent()


@pytest.mark.integration
class TestFullSaveLoadCycle:
    """Integration tests for complete save/load workflows."""

    @patch("src.checkpointing.dist_barrier")
    @patch("src.checkpointing._should_save_on_this_rank", return_value=True)
    def test_full_cycle_with_all_components(
        self,
        mock_should_save,
        mock_barrier,
        temp_dir,
        simple_model,
        simple_optimizer,
        simple_loss_fn,
        mock_logger,
    ):
        """Test full save/load cycle with all components."""
        manager = DattaBotCheckpointManager(checkpoint_dir=temp_dir)
        manager.bundle.unwrapped_model = simple_model
        manager.bundle.optimizer = simple_optimizer
        manager.bundle.loss_fn = simple_loss_fn
        manager.bundle.device = torch.device("cpu")
        manager.bundle.update_state(
            epoch=10,
            global_step=1000,
            tokens_processed=500000,
            train_loss=0.25,
            val_loss=0.30,
        )

        # Save original state
        original_model_state = {
            k: v.clone() for k, v in simple_model.state_dict().items()
        }
        original_loss_fn_state = {
            k: v.clone() for k, v in simple_loss_fn.state_dict().items()
        }

        # Save checkpoint
        manager.save_agent()

        # Modify all components
        with torch.no_grad():
            for param in simple_model.parameters():
                param.fill_(0.0)
        simple_optimizer.param_groups[0]["lr"] = 0.999
        manager.bundle.update_state(epoch=0, global_step=0)

        # Load checkpoint
        metadata = manager.load_agent()

        # Verify everything restored
        assert metadata.epoch == 10
        assert metadata.global_step == 1000
        assert metadata.tokens_processed == 500000
        assert metadata.train_loss == 0.25
        assert metadata.val_loss == 0.30

        # Verify model weights
        for key, original_value in original_model_state.items():
            current_value = simple_model.state_dict()[key]
            assert torch.allclose(current_value, original_value)

        # Verify optimizer
        assert simple_optimizer.param_groups[0]["lr"] == 0.001

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
        """Test multiple save/load cycles."""
        manager = DattaBotCheckpointManager(checkpoint_dir=temp_dir)
        manager.bundle.unwrapped_model = simple_model
        manager.bundle.optimizer = simple_optimizer
        manager.bundle.device = torch.device("cpu")

        for i in range(5):
            # Update state
            manager.bundle.update_state(epoch=i, global_step=i * 100)

            # Modify model
            with torch.no_grad():
                for param in simple_model.parameters():
                    param.add_(0.1)

            # Save
            manager.save_agent()

            # Load
            metadata = manager.load_agent()

            assert metadata.epoch == i
            assert metadata.global_step == i * 100


# ============================================================================
# Tests - Exists and GetMetadata
# ============================================================================


class TestExistsAndGetMetadata:
    """Test exists and get_metadata methods."""

    @patch("src.checkpointing.dist_barrier")
    @patch("src.checkpointing._should_save_on_this_rank", return_value=True)
    def test_exists_true(
        self,
        mock_should_save,
        mock_barrier,
        temp_dir,
        simple_model,
        mock_logger,
    ):
        """Test exists returns True after save."""
        manager = DattaBotCheckpointManager(checkpoint_dir=temp_dir)
        manager.bundle.unwrapped_model = simple_model
        manager.bundle.device = torch.device("cpu")

        manager.save_agent()

        assert manager.exists() is True

    def test_exists_false(self, temp_dir, mock_logger):
        """Test exists returns False when no checkpoint."""
        manager = DattaBotCheckpointManager(checkpoint_dir=temp_dir)

        assert manager.exists() is False

    @patch("src.checkpointing.dist_barrier")
    @patch("src.checkpointing._should_save_on_this_rank", return_value=True)
    def test_get_metadata(
        self,
        mock_should_save,
        mock_barrier,
        temp_dir,
        simple_model,
        mock_logger,
    ):
        """Test get_metadata returns metadata without loading weights."""
        manager = DattaBotCheckpointManager(checkpoint_dir=temp_dir)
        manager.bundle.unwrapped_model = simple_model
        manager.bundle.device = torch.device("cpu")
        manager.bundle.update_state(epoch=7, global_step=700)

        manager.save_agent()

        # Get metadata without loading
        metadata = manager.get_metadata()

        assert metadata.epoch == 7
        assert metadata.global_step == 700

    def test_get_metadata_none_when_missing(self, temp_dir, mock_logger):
        """Test get_metadata returns None when no checkpoint."""
        manager = DattaBotCheckpointManager(checkpoint_dir=temp_dir)

        metadata = manager.get_metadata()

        assert metadata is None


# ============================================================================
# Tests - Error Handling
# ============================================================================


@pytest.mark.integration
class TestErrorHandling:
    """Test error handling scenarios."""

    @patch("src.checkpointing.dist_barrier")
    @patch("src.checkpointing._should_save_on_this_rank", return_value=True)
    def test_load_corrupted_checkpoint(
        self,
        mock_should_save,
        mock_barrier,
        temp_dir,
        simple_model,
        mock_logger,
    ):
        """Test loading corrupted checkpoint raises error."""
        manager = DattaBotCheckpointManager(checkpoint_dir=temp_dir)
        manager.bundle.unwrapped_model = simple_model
        manager.bundle.device = torch.device("cpu")

        # Create corrupted model file
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(temp_dir) / "model.safetensors", "wb") as f:
            f.write(b"corrupted data")

        with pytest.raises(Exception):
            manager.load_agent()

    @patch("src.checkpointing.dist_barrier")
    @patch("src.checkpointing._should_save_on_this_rank", return_value=True)
    def test_load_incompatible_model(
        self,
        mock_should_save,
        mock_barrier,
        temp_dir,
        simple_model,
        mock_logger,
    ):
        """Test loading checkpoint with incompatible architecture."""
        manager = DattaBotCheckpointManager(checkpoint_dir=temp_dir)
        manager.bundle.unwrapped_model = simple_model
        manager.bundle.device = torch.device("cpu")

        # Save checkpoint
        manager.save_agent()

        # Create different model
        different_model = nn.Sequential(
            nn.Linear(10, 30),  # Different size
            nn.ReLU(),
            nn.Linear(30, 5),
        )
        manager.bundle.unwrapped_model = different_model

        # Should fail with strict=True
        with pytest.raises(RuntimeError):
            manager.load_agent(strict=True)


# ============================================================================
# Tests - Large Model
# ============================================================================


@pytest.mark.integration
class TestLargeModel:
    """Test with larger models."""

    @patch("src.checkpointing.dist_barrier")
    @patch("src.checkpointing._should_save_on_this_rank", return_value=True)
    def test_save_load_large_model(
        self,
        mock_should_save,
        mock_barrier,
        temp_dir,
        mock_logger,
    ):
        """Test saving and loading a larger model."""
        large_model = nn.Sequential(
            nn.Linear(1000, 2000),
            nn.ReLU(),
            nn.Linear(2000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 100),
        )
        optimizer = torch.optim.Adam(large_model.parameters())

        manager = DattaBotCheckpointManager(checkpoint_dir=temp_dir)
        manager.bundle.unwrapped_model = large_model
        manager.bundle.optimizer = optimizer
        manager.bundle.device = torch.device("cpu")

        # Save
        manager.save_agent()

        # Verify file size is reasonable
        model_path = Path(temp_dir) / "model.safetensors"
        assert model_path.exists()
        assert model_path.stat().st_size > 1000  # At least 1KB

        # Modify model
        with torch.no_grad():
            for param in large_model.parameters():
                param.fill_(0.0)

        # Load
        manager.load_agent()

        # Verify weights restored
        for param in large_model.parameters():
            assert not torch.all(param == 0.0)


# ============================================================================
# Edge Cases - Tensor Types and Formats
# ============================================================================


@pytest.mark.integration
class TestTensorEdgeCases:
    """Test edge cases with different tensor types and formats."""

    @patch("src.checkpointing.dist_barrier")
    @patch("src.checkpointing._should_save_on_this_rank", return_value=True)
    def test_non_contiguous_tensors(
        self,
        mock_should_save,
        mock_barrier,
        temp_dir,
        mock_logger,
    ):
        """Test saving model with non-contiguous tensors."""
        model = nn.Linear(10, 5)

        # Make weight non-contiguous by using a view with different strides
        with torch.no_grad():
            # Create a larger tensor and take a non-contiguous slice
            large_tensor = torch.randn(10, 10)
            non_contig = large_tensor[::2, :]  # Every other row is non-contiguous
            model.weight = nn.Parameter(non_contig)
            assert not model.weight.is_contiguous()

        manager = DattaBotCheckpointManager(checkpoint_dir=temp_dir)
        manager.bundle.unwrapped_model = model
        manager.bundle.device = torch.device("cpu")

        # Should not raise
        manager.save_agent()

        # Load and verify
        new_model = nn.Linear(10, 5)
        manager.bundle.unwrapped_model = new_model
        manager.load_agent()

        assert torch.allclose(model.weight, new_model.weight)

    @patch("src.checkpointing.dist_barrier")
    @patch("src.checkpointing._should_save_on_this_rank", return_value=True)
    def test_different_dtypes(
        self,
        mock_should_save,
        mock_barrier,
        temp_dir,
        mock_logger,
    ):
        """Test saving tensors with different dtypes supported by safetensors."""
        # Create model with mixed precision layers
        # Note: safetensors supports float16, bfloat16, float32 (not float64)
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Linear(20, 10),
        )

        # Convert to different supported dtypes
        with torch.no_grad():
            model[0].weight.data = model[0].weight.data.to(torch.float32)
            model[1].weight.data = model[1].weight.data.to(torch.float16)

        original_weight0 = model[0].weight.data.clone()
        original_weight1 = model[1].weight.data.clone()

        manager = DattaBotCheckpointManager(checkpoint_dir=temp_dir)
        manager.bundle.unwrapped_model = model
        manager.bundle.device = torch.device("cpu")

        manager.save_agent()

        # Load into new model with matching dtypes
        new_model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Linear(20, 10),
        )
        # Match dtypes before loading
        new_model[1] = new_model[1].to(torch.float16)

        manager.bundle.unwrapped_model = new_model
        manager.load_agent()

        # Verify values are preserved
        assert torch.allclose(new_model[0].weight.data, original_weight0)
        assert torch.allclose(new_model[1].weight.data, original_weight1)

    @patch("src.checkpointing.dist_barrier")
    @patch("src.checkpointing._should_save_on_this_rank", return_value=True)
    def test_model_with_buffers(
        self,
        mock_should_save,
        mock_barrier,
        temp_dir,
        mock_logger,
    ):
        """Test saving model with registered buffers (not just parameters)."""

        class ModelWithBuffer(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
                self.register_buffer("running_mean", torch.zeros(5))
                self.register_buffer("running_var", torch.ones(5))

            def forward(self, x):
                return self.linear(x)

        model = ModelWithBuffer()
        model.running_mean.fill_(3.14)
        model.running_var.fill_(2.71)

        manager = DattaBotCheckpointManager(checkpoint_dir=temp_dir)
        manager.bundle.unwrapped_model = model
        manager.bundle.device = torch.device("cpu")

        manager.save_agent()

        # Load into new model
        new_model = ModelWithBuffer()
        manager.bundle.unwrapped_model = new_model
        manager.load_agent()

        # Verify buffers restored
        assert torch.allclose(new_model.running_mean, torch.full((5,), 3.14))
        assert torch.allclose(new_model.running_var, torch.full((5,), 2.71))

    @patch("src.checkpointing.dist_barrier")
    @patch("src.checkpointing._should_save_on_this_rank", return_value=True)
    def test_empty_tensors(
        self,
        mock_should_save,
        mock_barrier,
        temp_dir,
        mock_logger,
    ):
        """Test saving model with empty tensors (zero-size dimensions)."""

        class ModelWithEmptyTensor(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
                self.register_buffer("empty_buffer", torch.zeros(0, 10))

            def forward(self, x):
                return self.linear(x)

        model = ModelWithEmptyTensor()

        manager = DattaBotCheckpointManager(checkpoint_dir=temp_dir)
        manager.bundle.unwrapped_model = model
        manager.bundle.device = torch.device("cpu")

        manager.save_agent()

        new_model = ModelWithEmptyTensor()
        manager.bundle.unwrapped_model = new_model
        manager.load_agent()

        assert new_model.empty_buffer.shape == torch.Size([0, 10])

    @patch("src.checkpointing.dist_barrier")
    @patch("src.checkpointing._should_save_on_this_rank", return_value=True)
    def test_scalar_tensors(
        self,
        mock_should_save,
        mock_barrier,
        temp_dir,
        mock_logger,
    ):
        """Test saving model with scalar (0-dim) tensors."""

        class ModelWithScalar(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
                self.register_buffer("scalar", torch.tensor(42.0))

            def forward(self, x):
                return self.linear(x) * self.scalar

        model = ModelWithScalar()

        manager = DattaBotCheckpointManager(checkpoint_dir=temp_dir)
        manager.bundle.unwrapped_model = model
        manager.bundle.device = torch.device("cpu")

        manager.save_agent()

        new_model = ModelWithScalar()
        manager.bundle.unwrapped_model = new_model
        manager.load_agent()

        assert new_model.scalar.dim() == 0
        assert new_model.scalar.item() == 42.0

    @patch("src.checkpointing.dist_barrier")
    @patch("src.checkpointing._should_save_on_this_rank", return_value=True)
    def test_shared_parameters(
        self,
        mock_should_save,
        mock_barrier,
        temp_dir,
        mock_logger,
    ):
        """Test saving model with shared/tied parameters."""

        class ModelWithSharedParams(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(100, 32)
                self.output = nn.Linear(32, 100, bias=False)
                # Tie weights
                self.output.weight = self.embedding.weight

            def forward(self, x):
                return self.output(self.embedding(x))

        model = ModelWithSharedParams()

        # Verify weights are actually shared
        assert model.embedding.weight.data_ptr() == model.output.weight.data_ptr()

        manager = DattaBotCheckpointManager(checkpoint_dir=temp_dir)
        manager.bundle.unwrapped_model = model
        manager.bundle.device = torch.device("cpu")

        # Should handle shared tensors gracefully
        manager.save_agent()

        new_model = ModelWithSharedParams()
        manager.bundle.unwrapped_model = new_model
        manager.load_agent()

        # Verify values match
        assert torch.allclose(model.embedding.weight, new_model.embedding.weight)

    @patch("src.checkpointing.dist_barrier")
    @patch("src.checkpointing._should_save_on_this_rank", return_value=True)
    def test_tensor_views_shared_storage(
        self,
        mock_should_save,
        mock_barrier,
        temp_dir,
        mock_logger,
    ):
        """Test saving tensors that are views of a larger tensor (shared storage)."""

        class ModelWithViews(nn.Module):
            def __init__(self):
                super().__init__()
                # Create a base tensor
                self._base = torch.randn(100)
                # Create parameters that are views of the base
                self.w1 = nn.Parameter(self._base[:50].view(10, 5))
                self.w2 = nn.Parameter(self._base[50:].view(10, 5))

            def forward(self, x):
                return x @ self.w1 + x @ self.w2

        model = ModelWithViews()

        manager = DattaBotCheckpointManager(checkpoint_dir=temp_dir)
        manager.bundle.unwrapped_model = model
        manager.bundle.device = torch.device("cpu")

        # Should handle views correctly
        manager.save_agent()

        new_model = ModelWithViews()
        manager.bundle.unwrapped_model = new_model
        manager.load_agent()

        assert torch.allclose(model.w1, new_model.w1)
        assert torch.allclose(model.w2, new_model.w2)

    @patch("src.checkpointing.dist_barrier")
    @patch("src.checkpointing._should_save_on_this_rank", return_value=True)
    def test_transposed_tensors(
        self,
        mock_should_save,
        mock_barrier,
        temp_dir,
        mock_logger,
    ):
        """Test saving model with transposed weight tensors."""

        class ModelWithTranspose(nn.Module):
            def __init__(self):
                super().__init__()
                base = torch.randn(10, 5)
                # Transposed tensor shares storage with original
                self.weight = nn.Parameter(base.T)

            def forward(self, x):
                return x @ self.weight

        model = ModelWithTranspose()
        # Verify weight is not contiguous (transposed)
        assert not model.weight.is_contiguous()

        manager = DattaBotCheckpointManager(checkpoint_dir=temp_dir)
        manager.bundle.unwrapped_model = model
        manager.bundle.device = torch.device("cpu")

        manager.save_agent()

        new_model = ModelWithTranspose()
        manager.bundle.unwrapped_model = new_model
        manager.load_agent()

        assert torch.allclose(model.weight, new_model.weight)

    @patch("src.checkpointing.dist_barrier")
    @patch("src.checkpointing._should_save_on_this_rank", return_value=True)
    def test_numpy_shared_memory_tensors(
        self,
        mock_should_save,
        mock_barrier,
        temp_dir,
        mock_logger,
    ):
        """Test saving tensors created from numpy arrays (may share memory)."""
        import numpy as np

        class ModelFromNumpy(nn.Module):
            def __init__(self):
                super().__init__()
                # Create from numpy (default shares memory)
                arr = np.random.randn(5, 10).astype(np.float32)
                self.weight = nn.Parameter(torch.from_numpy(arr))
                self.linear = nn.Linear(10, 5)

            def forward(self, x):
                return self.linear(x) + self.weight

        model = ModelFromNumpy()

        manager = DattaBotCheckpointManager(checkpoint_dir=temp_dir)
        manager.bundle.unwrapped_model = model
        manager.bundle.device = torch.device("cpu")

        manager.save_agent()

        new_model = ModelFromNumpy()
        manager.bundle.unwrapped_model = new_model
        manager.load_agent()

        assert torch.allclose(model.weight, new_model.weight)

    @patch("src.checkpointing.dist_barrier")
    @patch("src.checkpointing._should_save_on_this_rank", return_value=True)
    def test_expand_broadcast_tensors(
        self,
        mock_should_save,
        mock_barrier,
        temp_dir,
        mock_logger,
    ):
        """Test saving tensors created via expand (shared storage, different strides)."""

        class ModelWithExpand(nn.Module):
            def __init__(self, use_expand: bool = True):
                super().__init__()
                if use_expand:
                    # Expanded tensors share storage
                    base = torch.randn(1, 10)
                    self.register_buffer("expanded", base.expand(5, 10))
                else:
                    # Regular contiguous buffer for loading
                    self.register_buffer("expanded", torch.zeros(5, 10))
                self.linear = nn.Linear(10, 5)

            def forward(self, x):
                return self.linear(x) + self.expanded

        model = ModelWithExpand(use_expand=True)
        # Verify buffer is not contiguous (expanded)
        assert not model.expanded.is_contiguous()

        manager = DattaBotCheckpointManager(checkpoint_dir=temp_dir)
        manager.bundle.unwrapped_model = model
        manager.bundle.device = torch.device("cpu")

        manager.save_agent()

        # Create new model with contiguous buffer for loading
        new_model = ModelWithExpand(use_expand=False)
        manager.bundle.unwrapped_model = new_model
        manager.load_agent()

        assert torch.allclose(model.expanded, new_model.expanded)

    @patch("src.checkpointing.dist_barrier")
    @patch("src.checkpointing._should_save_on_this_rank", return_value=True)
    def test_tensor_with_storage_offset(
        self,
        mock_should_save,
        mock_barrier,
        temp_dir,
        mock_logger,
    ):
        """Test saving tensors with non-zero storage offset."""

        class ModelWithOffset(nn.Module):
            def __init__(self):
                super().__init__()
                # Create tensor from middle of storage
                base = torch.randn(100)
                self.weight = nn.Parameter(base[25:75].view(10, 5))
                # Verify it has a storage offset
                assert self.weight.storage_offset() > 0

            def forward(self, x):
                return x @ self.weight

        model = ModelWithOffset()

        manager = DattaBotCheckpointManager(checkpoint_dir=temp_dir)
        manager.bundle.unwrapped_model = model
        manager.bundle.device = torch.device("cpu")

        manager.save_agent()

        new_model = ModelWithOffset()
        manager.bundle.unwrapped_model = new_model
        manager.load_agent()

        assert torch.allclose(model.weight, new_model.weight)

    @patch("src.checkpointing.dist_barrier")
    @patch("src.checkpointing._should_save_on_this_rank", return_value=True)
    def test_as_strided_tensors(
        self,
        mock_should_save,
        mock_barrier,
        temp_dir,
        mock_logger,
    ):
        """Test saving tensors created with as_strided (custom strides)."""

        class ModelWithStrides(nn.Module):
            def __init__(self):
                super().__init__()
                base = torch.randn(100)
                # Create with custom strides
                custom = torch.as_strided(base, size=(5, 10), stride=(2, 1))
                self.weight = nn.Parameter(custom)
                self.linear = nn.Linear(10, 5)

            def forward(self, x):
                return self.linear(x) + self.weight

        model = ModelWithStrides()

        manager = DattaBotCheckpointManager(checkpoint_dir=temp_dir)
        manager.bundle.unwrapped_model = model
        manager.bundle.device = torch.device("cpu")

        manager.save_agent()

        new_model = ModelWithStrides()
        manager.bundle.unwrapped_model = new_model
        manager.load_agent()

        assert torch.allclose(model.weight, new_model.weight)

    @patch("src.checkpointing.dist_barrier")
    @patch("src.checkpointing._should_save_on_this_rank", return_value=True)
    def test_to_clean_cpu_tensor_helper(
        self,
        mock_should_save,
        mock_barrier,
        temp_dir,
        mock_logger,
    ):
        """Test the _to_clean_cpu_tensor helper directly with various edge cases."""
        from src.checkpointing import _to_clean_cpu_tensor

        # Test basic tensor
        t1 = torch.randn(10, 5)
        clean1 = _to_clean_cpu_tensor(t1)
        assert clean1.is_contiguous()
        assert torch.allclose(t1, clean1)

        # Test non-contiguous tensor
        t2 = torch.randn(10, 10)[::2, :]
        assert not t2.is_contiguous()
        clean2 = _to_clean_cpu_tensor(t2)
        assert clean2.is_contiguous()
        assert torch.allclose(t2, clean2)

        # Test tensor from numpy
        import numpy as np

        arr = np.random.randn(5, 5).astype(np.float32)
        t3 = torch.from_numpy(arr)
        clean3 = _to_clean_cpu_tensor(t3)
        assert clean3.is_contiguous()
        assert torch.allclose(t3, clean3)

        # Test tensor with storage offset
        base = torch.randn(100)
        t4 = base[25:75].view(10, 5)
        clean4 = _to_clean_cpu_tensor(t4)
        assert clean4.is_contiguous()
        assert torch.allclose(t4, clean4)
        # Clean tensor should have zero storage offset
        assert clean4.storage_offset() == 0

        # Test transposed tensor
        t5 = torch.randn(5, 10).T
        clean5 = _to_clean_cpu_tensor(t5)
        assert clean5.is_contiguous()
        assert torch.allclose(t5, clean5)


# ============================================================================
# Edge Cases - Optimizer State
# ============================================================================


@pytest.mark.integration
class TestOptimizerEdgeCases:
    """Test edge cases with optimizer state."""

    @patch("src.checkpointing.dist_barrier")
    @patch("src.checkpointing._should_save_on_this_rank", return_value=True)
    def test_optimizer_with_momentum(
        self,
        mock_should_save,
        mock_barrier,
        temp_dir,
        mock_logger,
    ):
        """Test saving optimizer with momentum buffers."""
        model = nn.Linear(10, 5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        # Run some training steps to populate momentum buffers
        for _ in range(5):
            optimizer.zero_grad()
            loss = model(torch.randn(2, 10)).sum()
            loss.backward()
            optimizer.step()

        # Verify momentum buffers exist
        assert len(optimizer.state) > 0

        manager = DattaBotCheckpointManager(checkpoint_dir=temp_dir)
        manager.bundle.unwrapped_model = model
        manager.bundle.optimizer = optimizer
        manager.bundle.device = torch.device("cpu")

        manager.save_agent()

        # Create new model and optimizer
        new_model = nn.Linear(10, 5)
        new_optimizer = torch.optim.SGD(new_model.parameters(), lr=0.01, momentum=0.9)

        manager.bundle.unwrapped_model = new_model
        manager.bundle.optimizer = new_optimizer

        manager.load_agent()

        # Verify momentum state restored
        assert len(new_optimizer.state) > 0

    @patch("src.checkpointing.dist_barrier")
    @patch("src.checkpointing._should_save_on_this_rank", return_value=True)
    def test_optimizer_multiple_param_groups(
        self,
        mock_should_save,
        mock_barrier,
        temp_dir,
        mock_logger,
    ):
        """Test saving optimizer with multiple parameter groups."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Linear(20, 5),
        )

        # Create optimizer with different lr for each layer
        optimizer = torch.optim.Adam(
            [
                {"params": model[0].parameters(), "lr": 0.001},
                {"params": model[1].parameters(), "lr": 0.0001},
            ]
        )

        manager = DattaBotCheckpointManager(checkpoint_dir=temp_dir)
        manager.bundle.unwrapped_model = model
        manager.bundle.optimizer = optimizer
        manager.bundle.device = torch.device("cpu")

        manager.save_agent()

        # Create new model and optimizer
        new_model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Linear(20, 5),
        )
        new_optimizer = torch.optim.Adam(
            [
                {"params": new_model[0].parameters(), "lr": 0.999},
                {"params": new_model[1].parameters(), "lr": 0.888},
            ]
        )

        manager.bundle.unwrapped_model = new_model
        manager.bundle.optimizer = new_optimizer

        manager.load_agent()

        # Verify param groups restored
        assert new_optimizer.param_groups[0]["lr"] == 0.001
        assert new_optimizer.param_groups[1]["lr"] == 0.0001

    @patch("src.checkpointing.dist_barrier")
    @patch("src.checkpointing._should_save_on_this_rank", return_value=True)
    def test_adamw_with_weight_decay(
        self,
        mock_should_save,
        mock_barrier,
        temp_dir,
        mock_logger,
    ):
        """Test saving AdamW optimizer with weight decay."""
        model = nn.Linear(10, 5)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=0.001,
            weight_decay=0.01,
            betas=(0.9, 0.999),
        )

        # Run training steps
        for _ in range(3):
            optimizer.zero_grad()
            loss = model(torch.randn(2, 10)).sum()
            loss.backward()
            optimizer.step()

        manager = DattaBotCheckpointManager(checkpoint_dir=temp_dir)
        manager.bundle.unwrapped_model = model
        manager.bundle.optimizer = optimizer
        manager.bundle.device = torch.device("cpu")

        manager.save_agent()

        new_model = nn.Linear(10, 5)
        new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=0.001)

        manager.bundle.unwrapped_model = new_model
        manager.bundle.optimizer = new_optimizer

        manager.load_agent()

        # Verify optimizer state has exp_avg and exp_avg_sq
        for state in new_optimizer.state.values():
            assert "exp_avg" in state
            assert "exp_avg_sq" in state


# ============================================================================
# Edge Cases - Metadata
# ============================================================================


@pytest.mark.integration
class TestMetadataEdgeCases:
    """Test edge cases with metadata handling."""

    @patch("src.checkpointing.dist_barrier")
    @patch("src.checkpointing._should_save_on_this_rank", return_value=True)
    def test_metadata_unicode(
        self,
        mock_should_save,
        mock_barrier,
        temp_dir,
        simple_model,
        mock_logger,
    ):
        """Test metadata with unicode characters."""
        manager = DattaBotCheckpointManager(checkpoint_dir=temp_dir)
        manager.bundle.unwrapped_model = simple_model
        manager.bundle.device = torch.device("cpu")
        manager.bundle.agent_name = "测试模型 🤖 Тест"

        manager.save_agent()

        metadata = manager.get_metadata()
        assert metadata.agent_name == "测试模型 🤖 Тест"

    @patch("src.checkpointing.dist_barrier")
    @patch("src.checkpointing._should_save_on_this_rank", return_value=True)
    def test_metadata_extreme_values(
        self,
        mock_should_save,
        mock_barrier,
        temp_dir,
        simple_model,
        mock_logger,
    ):
        """Test metadata with extreme numeric values."""
        manager = DattaBotCheckpointManager(checkpoint_dir=temp_dir)
        manager.bundle.unwrapped_model = simple_model
        manager.bundle.device = torch.device("cpu")
        manager.bundle.update_state(
            epoch=999999999,
            global_step=2**31 - 1,  # Max 32-bit int
            tokens_processed=2**40,  # Very large
            train_loss=1e-10,  # Very small
            val_loss=1e10,  # Very large
        )

        manager.save_agent()

        metadata = manager.get_metadata()
        assert metadata.epoch == 999999999
        assert metadata.global_step == 2**31 - 1
        assert metadata.tokens_processed == 2**40
        assert metadata.train_loss == pytest.approx(1e-10)
        assert metadata.val_loss == pytest.approx(1e10)

    @patch("src.checkpointing.dist_barrier")
    @patch("src.checkpointing._should_save_on_this_rank", return_value=True)
    def test_metadata_zero_and_none(
        self,
        mock_should_save,
        mock_barrier,
        temp_dir,
        simple_model,
        mock_logger,
    ):
        """Test metadata with zero and None values."""
        manager = DattaBotCheckpointManager(checkpoint_dir=temp_dir)
        manager.bundle.unwrapped_model = simple_model
        manager.bundle.device = torch.device("cpu")
        manager.bundle.update_state(
            epoch=0,
            global_step=0,
            tokens_processed=0,
            train_loss=0.0,  # Explicitly zero
            val_loss=None,  # Explicitly None
        )

        manager.save_agent()

        metadata = manager.get_metadata()
        assert metadata.epoch == 0
        assert metadata.global_step == 0
        assert metadata.tokens_processed == 0
        assert metadata.train_loss == 0.0
        assert metadata.val_loss is None

    def test_metadata_backward_compatibility(self):
        """Test loading metadata with extra/missing fields."""
        # Old format with missing fields
        old_data = {
            "epoch": 5,
            "step": 100,  # Old field name
            "loss": 0.5,  # Old field name
        }
        metadata = CheckpointMetadata.from_dict(old_data)

        assert metadata.epoch == 5
        assert metadata.extra["step"] == 100
        assert metadata.extra["loss"] == 0.5

    def test_metadata_special_float_values(self):
        """Test metadata handles special float values."""
        metadata = CheckpointMetadata(
            train_loss=float("inf"),
            val_loss=float("-inf"),
        )

        data = metadata.to_dict()
        # Should be serializable (though inf becomes string in JSON)
        assert data["train_loss"] == float("inf")


# ============================================================================
# Edge Cases - File Operations
# ============================================================================


@pytest.mark.integration
class TestFileOperationEdgeCases:
    """Test edge cases with file operations."""

    @patch("src.checkpointing.dist_barrier")
    @patch("src.checkpointing._should_save_on_this_rank", return_value=True)
    def test_overwrite_checkpoint(
        self,
        mock_should_save,
        mock_barrier,
        temp_dir,
        simple_model,
        mock_logger,
    ):
        """Test overwriting existing checkpoint preserves atomicity."""
        manager = DattaBotCheckpointManager(checkpoint_dir=temp_dir)
        manager.bundle.unwrapped_model = simple_model
        manager.bundle.device = torch.device("cpu")

        # Save first version
        manager.bundle.update_state(epoch=1)
        manager.save_agent()

        # Save second version (overwrite)
        manager.bundle.update_state(epoch=2)
        manager.save_agent()

        # Verify second version
        metadata = manager.get_metadata()
        assert metadata.epoch == 2

        # No temp files should be left
        temp_files = [f for f in os.listdir(temp_dir) if f.startswith(".tmp_")]
        assert len(temp_files) == 0

    @patch("src.checkpointing.dist_barrier")
    @patch("src.checkpointing._should_save_on_this_rank", return_value=True)
    def test_long_path(
        self,
        mock_should_save,
        mock_barrier,
        temp_dir,
        simple_model,
        mock_logger,
    ):
        """Test checkpoint with deeply nested directory path."""
        long_path = Path(temp_dir) / "a" / "b" / "c" / "d" / "e" / "checkpoints"

        manager = DattaBotCheckpointManager(checkpoint_dir=long_path)
        manager.bundle.unwrapped_model = simple_model
        manager.bundle.device = torch.device("cpu")

        manager.save_agent()

        assert (long_path / "model.safetensors").exists()

    @patch("src.checkpointing.dist_barrier")
    @patch("src.checkpointing._should_save_on_this_rank", return_value=True)
    def test_special_chars_in_path(
        self,
        mock_should_save,
        mock_barrier,
        temp_dir,
        simple_model,
        mock_logger,
    ):
        """Test checkpoint with special characters in path."""
        special_path = Path(temp_dir) / "checkpoint-v1.0_test (final)"

        manager = DattaBotCheckpointManager(checkpoint_dir=special_path)
        manager.bundle.unwrapped_model = simple_model
        manager.bundle.device = torch.device("cpu")

        manager.save_agent()

        assert (special_path / "model.safetensors").exists()

    @patch("src.checkpointing.dist_barrier")
    @patch("src.checkpointing._should_save_on_this_rank", return_value=True)
    def test_rapid_successive_saves(
        self,
        mock_should_save,
        mock_barrier,
        temp_dir,
        simple_model,
        mock_logger,
    ):
        """Test many rapid successive saves don't cause issues."""
        manager = DattaBotCheckpointManager(checkpoint_dir=temp_dir)
        manager.bundle.unwrapped_model = simple_model
        manager.bundle.device = torch.device("cpu")

        for i in range(20):
            manager.bundle.update_state(epoch=i, global_step=i * 10)
            manager.save_agent()

        # Verify final state
        metadata = manager.get_metadata()
        assert metadata.epoch == 19
        assert metadata.global_step == 190


# ============================================================================
# Edge Cases - Model Architecture
# ============================================================================


@pytest.mark.integration
class TestModelArchitectureEdgeCases:
    """Test edge cases with different model architectures."""

    @patch("src.checkpointing.dist_barrier")
    @patch("src.checkpointing._should_save_on_this_rank", return_value=True)
    def test_deeply_nested_modules(
        self,
        mock_should_save,
        mock_barrier,
        temp_dir,
        mock_logger,
    ):
        """Test saving model with deeply nested module structure."""

        class DeepModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.level1 = nn.ModuleDict(
                    {
                        "branch_a": nn.Sequential(
                            nn.Linear(10, 20),
                            nn.ModuleList([nn.Linear(20, 20) for _ in range(3)]),
                        ),
                        "branch_b": nn.Sequential(
                            nn.Linear(10, 15),
                            nn.Linear(15, 20),
                        ),
                    }
                )
                self.output = nn.Linear(40, 5)

            def forward(self, x):
                a = self.level1["branch_a"][0](x)
                for layer in self.level1["branch_a"][1]:
                    a = layer(a)
                b = self.level1["branch_b"](x)
                return self.output(torch.cat([a, b], dim=-1))

        model = DeepModel()

        manager = DattaBotCheckpointManager(checkpoint_dir=temp_dir)
        manager.bundle.unwrapped_model = model
        manager.bundle.device = torch.device("cpu")

        manager.save_agent()

        new_model = DeepModel()
        manager.bundle.unwrapped_model = new_model
        manager.load_agent()

        # Verify all nested weights match
        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(), new_model.named_parameters()
        ):
            assert n1 == n2
            assert torch.allclose(p1, p2)

    @patch("src.checkpointing.dist_barrier")
    @patch("src.checkpointing._should_save_on_this_rank", return_value=True)
    def test_model_with_no_parameters(
        self,
        mock_should_save,
        mock_barrier,
        temp_dir,
        mock_logger,
    ):
        """Test saving model with no learnable parameters."""

        class NoParamModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("const", torch.tensor([1.0, 2.0, 3.0]))

            def forward(self, x):
                return x * self.const

        model = NoParamModel()

        manager = DattaBotCheckpointManager(checkpoint_dir=temp_dir)
        manager.bundle.unwrapped_model = model
        manager.bundle.device = torch.device("cpu")

        manager.save_agent()

        new_model = NoParamModel()
        manager.bundle.unwrapped_model = new_model
        manager.load_agent()

        assert torch.allclose(new_model.const, torch.tensor([1.0, 2.0, 3.0]))

    @patch("src.checkpointing.dist_barrier")
    @patch("src.checkpointing._should_save_on_this_rank", return_value=True)
    def test_model_with_lazy_modules(
        self,
        mock_should_save,
        mock_barrier,
        temp_dir,
        mock_logger,
    ):
        """Test saving model with LazyLinear (uninitialized dimensions)."""
        # First initialize the lazy module
        model = nn.Sequential(
            nn.LazyLinear(20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )
        # Initialize by doing a forward pass
        model(torch.randn(1, 10))

        manager = DattaBotCheckpointManager(checkpoint_dir=temp_dir)
        manager.bundle.unwrapped_model = model
        manager.bundle.device = torch.device("cpu")

        manager.save_agent()

        # Create new model and initialize it
        new_model = nn.Sequential(
            nn.LazyLinear(20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )
        new_model(torch.randn(1, 10))

        manager.bundle.unwrapped_model = new_model
        manager.load_agent()

        # Verify weights match
        assert torch.allclose(model[0].weight, new_model[0].weight)


# ============================================================================
# Edge Cases - State Dict Key Cleaning
# ============================================================================


class TestStateDictKeyEdgeCases:
    """Test edge cases with state dict key handling."""

    def test_multiple_module_prefixes(self, mock_logger):
        """Test handling of multiple 'module.' prefixes."""
        state_dict = {
            "module.module.layer.weight": torch.randn(5, 5),
        }
        model = nn.Linear(5, 5)

        cleaned = _clean_state_dict_keys(state_dict, model)

        # Should only remove one 'module.' prefix
        assert "module.layer.weight" in cleaned

    def test_module_in_layer_name(self, simple_model, mock_logger):
        """Test that 'module' within layer name is preserved."""
        state_dict = {
            "my_module_layer.weight": torch.randn(5, 5),
        }

        cleaned = _clean_state_dict_keys(state_dict, simple_model)

        # 'my_module_layer' should not be affected
        assert "my_module_layer.weight" in cleaned


# ============================================================================
# Edge Cases - Flatten State Dict
# ============================================================================


class TestFlattenStateDictEdgeCases:
    """Test edge cases with state dict flattening."""

    def test_mixed_tensor_and_dict(self):
        """Test flattening with mixed tensor and dict values."""
        state_dict = {
            "weight": torch.randn(5, 5),
            "nested": {
                "bias": torch.randn(5),
            },
            "another_weight": torch.randn(3, 3),
        }
        flat = _flatten_state_dict(state_dict)

        assert "weight" in flat
        assert "nested.bias" in flat
        assert "another_weight" in flat
        assert len(flat) == 3

    def test_empty_state_dict(self):
        """Test flattening empty state dict."""
        flat = _flatten_state_dict({})
        assert flat == {}

    def test_empty_nested_dict(self):
        """Test flattening with empty nested dicts."""
        state_dict = {
            "weight": torch.randn(5, 5),
            "empty_nested": {},
        }
        flat = _flatten_state_dict(state_dict)

        assert "weight" in flat
        assert len(flat) == 1  # Empty nested dict produces nothing


# ============================================================================
# Edge Cases - JSON Serialization
# ============================================================================


class TestJsonSerializationEdgeCases:
    """Test edge cases with JSON serialization."""

    def test_numpy_values(self):
        """Test serializing numpy-like values."""
        import numpy as np

        data = {
            "numpy_array": np.array([1, 2, 3]),
            "numpy_scalar": np.float32(3.14),
        }
        result = _make_json_serializable(data)

        assert result["numpy_array"] == [1, 2, 3]
        assert result["numpy_scalar"] == pytest.approx(3.14, rel=1e-5)

    def test_boolean_values(self):
        """Test serializing boolean values."""
        data = {"flag": True, "other_flag": False}
        result = _make_json_serializable(data)

        assert result["flag"] is True
        assert result["other_flag"] is False

    def test_none_values(self):
        """Test serializing None values."""
        data = {"value": None, "nested": {"also_none": None}}
        result = _make_json_serializable(data)

        assert result["value"] is None
        assert result["nested"]["also_none"] is None

    def test_integer_tensor_keys(self):
        """Test that integer dict keys become strings."""
        data = {0: "first", 1: "second", 2: torch.tensor([1, 2])}
        result = _make_json_serializable(data)

        assert "0" in result
        assert "1" in result
        assert "2" in result
        assert result["2"] == [1, 2]


# ============================================================================
# Edge Cases - Bundle State
# ============================================================================


class TestBundleStateEdgeCases:
    """Test edge cases with bundle state management."""

    def test_update_state_with_zero_loss(self):
        """Test updating bundle with zero loss (not None)."""
        bundle = CheckpointBundle()
        bundle.update_state(train_loss=0.0)

        assert bundle.train_loss == 0.0
        assert bundle.train_loss is not None

    def test_update_state_negative_values(self):
        """Test that negative values are handled (edge case for debugging)."""
        bundle = CheckpointBundle()
        bundle.update_state(
            epoch=-1,  # Shouldn't happen but test it
            global_step=-100,
        )

        assert bundle.epoch == -1
        assert bundle.global_step == -100

    def test_bundle_references_not_copied(self, simple_model):
        """Test that bundle holds references, not copies."""
        bundle = CheckpointBundle()
        bundle.unwrapped_model = simple_model

        # Modify original model
        with torch.no_grad():
            list(simple_model.parameters())[0].fill_(999.0)

        # Bundle should see the change (same reference)
        assert torch.all(list(bundle.unwrapped_model.parameters())[0] == 999.0)

    def test_bundle_chained_updates(self):
        """Test chaining multiple update_state calls."""
        bundle = CheckpointBundle()
        bundle.update_state(epoch=1).update_state(global_step=100).update_state(
            train_loss=0.5
        )

        assert bundle.epoch == 1
        assert bundle.global_step == 100
        assert bundle.train_loss == 0.5


# ============================================================================
# Edge Cases - Strict Loading
# ============================================================================


@pytest.mark.integration
class TestStrictLoadingEdgeCases:
    """Test edge cases with strict/non-strict loading."""

    @patch("src.checkpointing.dist_barrier")
    @patch("src.checkpointing._should_save_on_this_rank", return_value=True)
    def test_non_strict_missing_keys(
        self,
        mock_should_save,
        mock_barrier,
        temp_dir,
        mock_logger,
    ):
        """Test non-strict loading with missing keys."""

        # Save smaller model (same structure as subset of larger)
        class SmallerModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)

            def forward(self, x):
                return self.linear(x)

        small_model = SmallerModel()
        original_weight = small_model.linear.weight.clone()

        manager = DattaBotCheckpointManager(checkpoint_dir=temp_dir)
        manager.bundle.unwrapped_model = small_model
        manager.bundle.device = torch.device("cpu")
        manager.save_agent()

        # Load into larger model (has extra keys)
        class LargerModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
                self.extra = nn.Linear(5, 3)

            def forward(self, x):
                return self.extra(self.linear(x))

        larger_model = LargerModel()
        manager.bundle.unwrapped_model = larger_model

        # Should not raise with strict=False
        manager.load_agent(strict=False)

        # Linear weights should be loaded from checkpoint
        assert torch.allclose(original_weight, larger_model.linear.weight)

    @patch("src.checkpointing.dist_barrier")
    @patch("src.checkpointing._should_save_on_this_rank", return_value=True)
    def test_non_strict_unexpected_keys(
        self,
        mock_should_save,
        mock_barrier,
        temp_dir,
        mock_logger,
    ):
        """Test non-strict loading with unexpected keys in checkpoint."""

        # Save larger model
        class LargerModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
                self.extra = nn.Linear(5, 3)

            def forward(self, x):
                return self.extra(self.linear(x))

        larger_model = LargerModel()
        manager = DattaBotCheckpointManager(checkpoint_dir=temp_dir)
        manager.bundle.unwrapped_model = larger_model
        manager.bundle.device = torch.device("cpu")
        manager.save_agent()

        # Load into smaller model
        small_model = nn.Linear(10, 5)
        manager.bundle.unwrapped_model = small_model

        # Should not raise with strict=False
        manager.load_agent(strict=False)
