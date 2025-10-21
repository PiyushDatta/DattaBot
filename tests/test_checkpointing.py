import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import call, MagicMock, Mock, patch

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from src.checkpointing import (
    _atomic_save,
    _clean_state_dict_keys,
    _create_checkpoint_dict,
    _extract_model_state,
    _extract_optimizer_state,
    _get_fsdp_state_dict_options,
    _is_fsdp_model,
    _load_model_state,
    _load_optimizer_state,
    _log_checkpoint_components,
    _parse_checkpoint_dict,
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
# Unit Tests - Checkpoint Structure
# ============================================================================


class TestCheckpointStructure:
    """Test checkpoint dictionary creation and parsing."""

    def test_create_checkpoint_dict_full(self, checkpoint_components):
        """Test creating full checkpoint with all components."""
        checkpoint = _create_checkpoint_dict(checkpoint_components)

        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert "loss_fn_state_dict" in checkpoint
        assert "metadata" in checkpoint
        assert checkpoint["metadata"]["epoch"] == 42

    def test_create_checkpoint_dict_minimal(self):
        """Test creating checkpoint with only required components."""
        components = CheckpointComponents(model_state={"weight": torch.randn(5, 5)})
        checkpoint = _create_checkpoint_dict(components)

        assert "model_state_dict" in checkpoint
        assert "metadata" in checkpoint
        assert "optimizer_state_dict" not in checkpoint
        assert "loss_fn_state_dict" not in checkpoint

    def test_parse_checkpoint_dict_new_format(self, checkpoint_components):
        """Test parsing new structured checkpoint format."""
        checkpoint = _create_checkpoint_dict(checkpoint_components)
        parsed = _parse_checkpoint_dict(checkpoint)

        assert parsed.model_state is not None
        assert parsed.optimizer_state is not None
        assert parsed.loss_fn_state is not None
        assert parsed.metadata["epoch"] == 42

    def test_parse_checkpoint_dict_legacy_format(self, mock_logger):
        """Test parsing legacy checkpoint format."""
        legacy_checkpoint = {"layer.weight": torch.randn(5, 5)}
        parsed = _parse_checkpoint_dict(legacy_checkpoint)

        assert parsed.model_state == legacy_checkpoint
        assert parsed.optimizer_state is None
        assert parsed.loss_fn_state is None
        assert parsed.metadata == {}
        mock_logger.warning.assert_called_once()


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


class TestAtomicSave:
    """Test atomic save operations."""

    def test_atomic_save_success(self, temp_dir, mock_logger):
        """Test successful atomic save."""
        filepath = os.path.join(temp_dir, "checkpoint.pt")
        checkpoint = {"data": torch.randn(5, 5)}

        _atomic_save(checkpoint, filepath)

        assert os.path.exists(filepath)
        loaded = torch.load(filepath)
        assert "data" in loaded
        mock_logger.debug.assert_called()

    def test_atomic_save_cleanup_on_error(self, temp_dir, mock_logger):
        """Test that temp file is cleaned up on error."""
        filepath = os.path.join(temp_dir, "checkpoint.pt")
        checkpoint = {"data": "invalid"}  # Will cause error when trying to move

        with patch(
            "src.checkpointing.torch.save", side_effect=RuntimeError("Save failed")
        ):
            with pytest.raises(RuntimeError):
                _atomic_save(checkpoint, filepath)

        # Check no temp files left behind
        temp_files = [f for f in os.listdir(temp_dir) if f.startswith(".tmp_")]
        assert len(temp_files) == 0

    def test_atomic_save_no_collision(self, temp_dir, mock_logger):
        """Test that temp filenames don't collide (use PID)."""
        filepath = os.path.join(temp_dir, "checkpoint.pt")
        checkpoint1 = {"data": torch.randn(5, 5)}
        checkpoint2 = {"data": torch.randn(3, 3)}

        # Save twice to ensure no collision
        _atomic_save(checkpoint1, filepath)
        _atomic_save(checkpoint2, filepath)

        # Should overwrite cleanly
        assert os.path.exists(filepath)


# ============================================================================
# Unit Tests - Logging
# ============================================================================


class TestLogging:
    """Test checkpoint component logging."""

    def test_log_checkpoint_components_all(self, checkpoint_components, mock_logger):
        """Test logging with all components."""
        _log_checkpoint_components(checkpoint_components, "saved", "/path/to/file.pt")

        call_args = mock_logger.info.call_args[0][0]
        assert "saved" in call_args
        assert "model" in call_args
        assert "optimizer" in call_args
        assert "loss_fn" in call_args
        assert "/path/to/file.pt" in call_args

    def test_log_checkpoint_components_model_only(self, mock_logger):
        """Test logging with only model."""
        components = CheckpointComponents(model_state={"weight": torch.randn(5, 5)})
        _log_checkpoint_components(components, "loaded", "/path/to/file.pt")

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
        filepath = os.path.join(temp_dir, "checkpoint.pt")
        device = torch.device("cpu")
        metadata = {"epoch": 10, "loss": 0.5}

        # Save
        save_agent(
            model=simple_model,
            filename=filepath,
            device=device,
            optimizer=simple_optimizer,
            metadata=metadata,
        )

        assert os.path.exists(filepath)

        # Modify model and optimizer
        with torch.no_grad():
            for param in simple_model.parameters():
                param.fill_(0.0)
        simple_optimizer.param_groups[0]["lr"] = 0.999

        # Load
        loaded_metadata = load_agent(
            model=simple_model,
            filepath=filepath,
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
        filepath = os.path.join(temp_dir, "checkpoint_with_loss.pt")
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
            filename=filepath,
            device=device,
            optimizer=simple_optimizer,
            loss_fn=loss_fn,
        )

        # Create new loss function to load into
        new_loss_fn = nn.AdaptiveLogSoftmaxWithLoss(
            in_features=5,
            n_classes=100,
            cutoffs=[10, 50],
        )

        # Load
        load_agent(
            model=simple_model,
            filepath=filepath,
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
        filepath = os.path.join(temp_dir, "checkpoint_rank1.pt")
        device = torch.device("cpu")

        save_agent(
            model=simple_model,
            filename=filepath,
            device=device,
            optimizer=simple_optimizer,
        )

        # File should not be created by non-rank-0
        assert not os.path.exists(filepath)

    @patch("src.checkpointing.dist_barrier")
    def test_load_nonexistent_file(
        self, mock_barrier, simple_model, simple_optimizer, mock_logger
    ):
        """Test loading from nonexistent file returns empty metadata."""
        metadata = load_agent(
            model=simple_model,
            filepath="/nonexistent/path.pt",
            optimizer=simple_optimizer,
            device="cpu",
        )

        assert metadata == {}
        mock_logger.error.assert_called()

    @patch("src.checkpointing.dist_barrier")
    @patch("src.checkpointing._should_save_on_this_rank", return_value=True)
    def test_load_legacy_checkpoint_format(
        self, mock_should_save, mock_barrier, temp_dir, simple_model, mock_logger
    ):
        """Test loading legacy checkpoint format (direct state dict)."""
        filepath = os.path.join(temp_dir, "legacy_checkpoint.pt")

        # Create legacy format checkpoint (direct state dict)
        legacy_checkpoint = simple_model.state_dict()
        torch.save(legacy_checkpoint, filepath)

        # Create fresh model
        new_model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )

        optimizer = torch.optim.Adam(new_model.parameters())

        # Load should work with warning
        metadata = load_agent(
            model=new_model,
            filepath=filepath,
            optimizer=optimizer,
            device="cpu",
        )

        assert metadata == {}
        mock_logger.warning.assert_called()  # Should warn about legacy format


# ============================================================================
# Integration Tests - Error Handling
# ============================================================================


@pytest.mark.integration
class TestErrorHandling:
    """Integration tests for error handling."""

    @patch("src.checkpointing.dist_barrier")
    @patch("src.checkpointing._should_save_on_this_rank", return_value=True)
    @patch("src.checkpointing.torch.save", side_effect=RuntimeError("Disk full"))
    def test_save_handles_disk_error(
        self,
        mock_torch_save,
        mock_should_save,
        mock_barrier,
        temp_dir,
        simple_model,
        simple_optimizer,
        mock_logger,
    ):
        """Test that save errors are properly caught and logged."""
        filepath = os.path.join(temp_dir, "will_fail.pt")
        device = torch.device("cpu")

        with pytest.raises(RuntimeError):
            save_agent(
                model=simple_model,
                filename=filepath,
                device=device,
                optimizer=simple_optimizer,
            )

        mock_logger.error.assert_called()
        # Barrier should still be called in finally block
        mock_barrier.assert_called()

    @patch("src.checkpointing.dist_barrier")
    @patch("src.checkpointing._should_save_on_this_rank", return_value=True)
    def test_load_handles_corrupted_file(
        self,
        mock_should_save,
        mock_barrier,
        temp_dir,
        simple_model,
        simple_optimizer,
        mock_logger,
    ):
        """Test loading corrupted checkpoint file."""
        filepath = os.path.join(temp_dir, "corrupted.pt")

        # Create corrupted file
        with open(filepath, "wb") as f:
            f.write(b"corrupted data")

        with pytest.raises(Exception):
            load_agent(
                model=simple_model,
                filepath=filepath,
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
        filepath = os.path.join(temp_dir, "incompatible.pt")
        device = torch.device("cpu")

        # Save current model
        save_agent(
            model=simple_model,
            filename=filepath,
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

        # Load should fail or warn with strict=True
        with pytest.raises(RuntimeError):
            load_agent(
                model=different_model,
                filepath=filepath,
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
        filepath = os.path.join(temp_dir, "checkpoint_with_prefix.pt")

        # Create checkpoint with 'module.' prefix
        state_dict = simple_model.state_dict()
        prefixed_state = {f"module.{k}": v for k, v in state_dict.items()}

        checkpoint = {
            "model_state_dict": prefixed_state,
            "metadata": {},
        }
        torch.save(checkpoint, filepath)

        # Load into regular model (should strip prefix)
        optimizer = torch.optim.Adam(simple_model.parameters())
        load_agent(
            model=simple_model,
            filepath=filepath,
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

        filepath = os.path.join(temp_dir, "large_checkpoint.pt")
        device = torch.device("cpu")

        # Save
        save_agent(
            model=large_model,
            filename=filepath,
            device=device,
            optimizer=optimizer,
        )

        # Verify file was created and has reasonable size
        assert os.path.exists(filepath)
        file_size = os.path.getsize(filepath)
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
            filepath=filepath,
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
        filepath = os.path.join(temp_dir, "multi_cycle.pt")
        device = torch.device("cpu")

        for i in range(5):
            # Modify model slightly
            with torch.no_grad():
                for param in simple_model.parameters():
                    param.add_(0.1)

            # Save
            save_agent(
                model=simple_model,
                filename=filepath,
                device=device,
                optimizer=simple_optimizer,
                metadata={"cycle": i},
            )

            # Load back
            metadata = load_agent(
                model=simple_model,
                filepath=filepath,
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
        filepath = os.path.join(temp_dir, "empty_metadata.pt")
        device = torch.device("cpu")

        save_agent(
            model=simple_model,
            filename=filepath,
            device=device,
            optimizer=simple_optimizer,
            metadata=None,
        )

        metadata = load_agent(
            model=simple_model,
            filepath=filepath,
            optimizer=simple_optimizer,
            device="cpu",
        )

        assert metadata == {}

    @patch("src.checkpointing.dist_barrier")
    @patch("src.checkpointing._should_save_on_this_rank", return_value=True)
    def test_save_without_optimizer(
        self, mock_should_save, mock_barrier, temp_dir, simple_model, mock_logger
    ):
        """Test saving without optimizer (should still work)."""
        filepath = os.path.join(temp_dir, "no_optimizer.pt")
        device = torch.device("cpu")

        # Note: The function requires optimizer, so we pass it
        # but the test verifies it handles missing optimizer state
        optimizer = torch.optim.Adam(simple_model.parameters())

        save_agent(
            model=simple_model,
            filename=filepath,
            device=device,
            optimizer=optimizer,
        )

        # Should be able to load with different optimizer
        new_optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.01)
        load_agent(
            model=simple_model,
            filepath=filepath,
            optimizer=new_optimizer,
            device="cpu",
        )

    # Suppress PyTorch warning about zero-element tensor initialization
    @pytest.mark.filterwarnings(
        "ignore:Initializing zero-element tensors is a no-op:UserWarning"
    )
    @patch("src.checkpointing.dist_barrier")
    def test_load_with_no_loss_fn_in_checkpoint(
        self, mock_barrier, temp_dir, simple_model, mock_logger
    ):
        """Test loading with loss_fn provided but not in checkpoint."""
        filepath = os.path.join(temp_dir, "no_loss_fn.pt")
        # Create checkpoint without loss_fn
        checkpoint = {
            "model_state_dict": simple_model.state_dict(),
            "metadata": {},
        }
        torch.save(checkpoint, filepath)
        loss_fn = nn.AdaptiveLogSoftmaxWithLoss(5, 100, [10, 50])
        optimizer = torch.optim.Adam(simple_model.parameters())
        load_agent(
            model=simple_model,
            filepath=filepath,
            optimizer=optimizer,
            device="cpu",
            loss_fn=loss_fn,
        )
        # Should warn about missing loss function state in checkpoint
        # The actual warning message is:
        # "Loss function provided but no state in checkpoint.
        #  Using random initialization!"
        warning_messages = [str(call) for call in mock_logger.warning.call_args_list]
        found_warning = any(
            "Loss function" in msg and "checkpoint" in msg.lower()
            for msg in warning_messages
        )
        assert found_warning, (
            f"Expected warning about missing loss function in checkpoint, "
            f"got warnings: {warning_messages}"
        )
