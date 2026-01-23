# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import threading
import time
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import torch
import zmq
import zmq.asyncio

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_connector import (
    MooncakeAgentMetadata,
    MooncakeConnector,
    MooncakeConnectorMetadata,
    MooncakeConnectorScheduler,
    MooncakeConnectorWorker,
    RecvReqMeta,
    SendBlockMeta,
    group_concurrent_contiguous,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole


@pytest.fixture
def mock_vllm_config():
    """Create a mock VllmConfig for testing."""
    config = MagicMock(spec=VllmConfig)
    config.kv_transfer_config = MagicMock()
    config.kv_transfer_config.engine_id = "test_engine"
    config.kv_transfer_config.kv_role = "kv_producer"
    config.kv_transfer_config.kv_connector_extra_config = {
        "mooncake_protocol": "tcp",
        "num_workers": 2
    }
    config.cache_config = MagicMock()
    config.cache_config.block_size = 16
    config.model_config = MagicMock()
    config.model_config.use_mla = False
    config.model_config.get_total_num_kv_heads.return_value = 8
    config.parallel_config = MagicMock()
    config.parallel_config.data_parallel_index = 0
    config.parallel_config.tensor_parallel_size = 1
    return config


@pytest.fixture
def mock_vllm_config_consumer():
    """Create a mock VllmConfig for consumer role."""
    config = MagicMock(spec=VllmConfig)
    config.kv_transfer_config = MagicMock()
    config.kv_transfer_config.engine_id = "test_engine_consumer"
    config.kv_transfer_config.kv_role = "kv_consumer"
    config.kv_transfer_config.kv_connector_extra_config = {
        "mooncake_protocol": "tcp",
        "num_workers": 2
    }
    config.cache_config = MagicMock()
    config.cache_config.block_size = 16
    config.model_config = MagicMock()
    config.model_config.use_mla = False
    config.model_config.get_total_num_kv_heads.return_value = 8
    config.parallel_config = MagicMock()
    config.parallel_config.data_parallel_index = 0
    config.parallel_config.tensor_parallel_size = 1
    return config


class TestMooncakeConnector:
    """Tests for the main MooncakeConnector class."""

    @pytest.mark.parametrize("role", [KVConnectorRole.SCHEDULER, KVConnectorRole.WORKER])
    def test_init(self, mock_vllm_config, role):
        """Test MooncakeConnector initialization."""
        connector = MooncakeConnector(mock_vllm_config, role)

        assert connector.engine_id == "test_engine"
        assert connector.role == role

        if role == KVConnectorRole.SCHEDULER:
            assert connector.connector_scheduler is not None
            assert connector.connector_worker is None
        else:
            assert connector.connector_scheduler is None
            assert connector.connector_worker is not None

    def test_scheduler_side_methods(self, mock_vllm_config):
        """Test that scheduler methods delegate correctly."""
        connector = MooncakeConnector(mock_vllm_config, KVConnectorRole.SCHEDULER)

        # Mock request and blocks
        mock_request = MagicMock()
        mock_blocks = MagicMock()

        with patch.object(connector.connector_scheduler, 'get_num_new_matched_tokens') as mock_method:
            mock_method.return_value = (100, True)
            result = connector.get_num_new_matched_tokens(mock_request, 50)
            mock_method.assert_called_once_with(mock_request, 50)
            assert result == (100, True)

        with patch.object(connector.connector_scheduler, 'update_state_after_alloc') as mock_method:
            connector.update_state_after_alloc(mock_request, mock_blocks, 10)
            mock_method.assert_called_once_with(mock_request, mock_blocks, 10)

        with patch.object(connector.connector_scheduler, 'build_connector_meta') as mock_method:
            mock_scheduler_output = MagicMock()
            mock_meta = MagicMock()
            mock_method.return_value = mock_meta
            result = connector.build_connector_meta(mock_scheduler_output)
            mock_method.assert_called_once_with(mock_scheduler_output)
            assert result == mock_meta

        with patch.object(connector.connector_scheduler, 'request_finished') as mock_method:
            mock_method.return_value = (True, {"key": "value"})
            result = connector.request_finished(mock_request, [1, 2, 3])
            mock_method.assert_called_once_with(mock_request, [1, 2, 3])
            assert result == (True, {"key": "value"})

    def test_worker_side_methods(self, mock_vllm_config):
        """Test that worker methods delegate correctly."""
        connector = MooncakeConnector(mock_vllm_config, KVConnectorRole.WORKER)

        # Mock kv_caches
        kv_caches = {"layer1": torch.randn(10, 8, 16, 8)}

        with patch.object(connector.connector_worker, 'register_kv_caches') as mock_method:
            connector.register_kv_caches(kv_caches)
            mock_method.assert_called_once_with(kv_caches)

        with patch.object(connector.connector_worker, 'get_finished') as mock_method:
            mock_method.return_value = ({"req1"}, {"req2"})
            result = connector.get_finished({"req1", "req2"})
            mock_method.assert_called_once_with()
            assert result == ({"req1"}, {"req2"})

        with patch.object(connector.connector_worker, 'start_load_kv') as mock_method:
            mock_forward_context = MagicMock()
            connector.start_load_kv(mock_forward_context)
            mock_method.assert_called_once_with(connector._connector_metadata)

    def test_worker_side_noops(self, mock_vllm_config):
        """Test that noop methods work correctly."""
        connector = MooncakeConnector(mock_vllm_config, KVConnectorRole.WORKER)

        # These methods should be no-ops
        connector.wait_for_layer_load("layer1")
        connector.save_kv_layer("layer1", torch.randn(10, 8), MagicMock())
        connector.wait_for_save()


class TestMooncakeConnectorScheduler:
    """Tests for MooncakeConnectorScheduler."""

    def test_init(self, mock_vllm_config):
        """Test scheduler initialization."""
        scheduler = MooncakeConnectorScheduler(mock_vllm_config, "test_engine")

        assert scheduler.engine_id == "test_engine"
        assert scheduler.kv_role == "kv_producer"
        assert scheduler._reqs_need_recv == {}
        assert scheduler._reqs_need_send == {}

    def test_get_num_new_matched_tokens_no_prefill(self, mock_vllm_config):
        """Test get_num_new_matched_tokens when no remote prefill."""
        scheduler = MooncakeConnectorScheduler(mock_vllm_config, "test_engine")
        mock_request = MagicMock()
        mock_request.kv_transfer_params = None

        result = scheduler.get_num_new_matched_tokens(mock_request, 10)
        assert result == (0, False)

    def test_get_num_new_matched_tokens_with_prefill(self, mock_vllm_config):
        """Test get_num_new_matched_tokens with remote prefill."""
        scheduler = MooncakeConnectorScheduler(mock_vllm_config, "test_engine")
        mock_request = MagicMock()
        mock_request.kv_transfer_params = {
            "do_remote_prefill": True
        }
        mock_request.prompt_token_ids = ["token1", "token2", "token3"]

        result = scheduler.get_num_new_matched_tokens(mock_request, 1)
        assert result == (2, True)  # 3 tokens - 1 computed = 2 remaining

    def test_update_state_after_alloc_no_params(self, mock_vllm_config):
        """Test update_state_after_alloc with no transfer params."""
        scheduler = MooncakeConnectorScheduler(mock_vllm_config, "test_engine")
        mock_request = MagicMock()
        mock_request.kv_transfer_params = None
        mock_blocks = MagicMock()

        scheduler.update_state_after_alloc(mock_request, mock_blocks, 10)
        # Should not add to any queues
        assert scheduler._reqs_need_recv == {}
        assert scheduler._reqs_need_send == {}

    def test_update_state_after_alloc_remote_prefill(self, mock_vllm_config):
        """Test update_state_after_alloc with remote prefill."""
        scheduler = MooncakeConnectorScheduler(mock_vllm_config, "test_engine")
        mock_request = MagicMock()
        mock_request.request_id = "req1"
        mock_request.kv_transfer_params = {
            "do_remote_prefill": True,
            "remote_host": "192.168.1.100",
            "remote_port": 8080
        }
        mock_blocks = MagicMock()
        mock_blocks.get_unhashed_block_ids.return_value = [1, 2, 3]

        scheduler.update_state_after_alloc(mock_request, mock_blocks, 5)

        # Should add to recv queue
        assert "req1" in scheduler._reqs_need_recv
        req_data = scheduler._reqs_need_recv["req1"]
        assert req_data[0] == mock_request
        assert req_data[1] == [1, 2, 3]

        # Should clear do_remote_prefill flag
        assert mock_request.kv_transfer_params["do_remote_prefill"] is False

    def test_update_state_after_alloc_remote_decode(self, mock_vllm_config):
        """Test update_state_after_alloc with remote decode."""
        scheduler = MooncakeConnectorScheduler(mock_vllm_config, "test_engine")
        mock_request = MagicMock()
        mock_request.request_id = "req2"
        mock_request.kv_transfer_params = {
            "do_remote_decode": True
        }
        mock_blocks = MagicMock()

        scheduler.update_state_after_alloc(mock_request, mock_blocks, 5)

        # Should add to send queue
        assert "req2" in scheduler._reqs_need_send
        assert scheduler._reqs_need_send["req2"] == []

    def test_build_connector_meta(self, mock_vllm_config):
        """Test build_connector_meta."""
        scheduler = MooncakeConnectorScheduler(mock_vllm_config, "test_engine")

        # Add some requests
        mock_request = MagicMock()
        mock_request.kv_transfer_params = {
            "remote_host": "192.168.1.100",
            "remote_port": 8080
        }
        scheduler._reqs_need_recv["req1"] = (mock_request, [1, 2, 3])
        scheduler._reqs_need_send["req2"] = [4, 5, 6]

        mock_scheduler_output = MagicMock()

        meta = scheduler.build_connector_meta(mock_scheduler_output)

        assert isinstance(meta, MooncakeConnectorMetadata)
        assert "req1" in meta.reqs_to_recv
        assert "req2" in meta.reqs_to_send

        # Should clear internal queues
        assert scheduler._reqs_need_recv == {}
        assert scheduler._reqs_need_send == {}

    def test_request_finished_no_params(self, mock_vllm_config):
        """Test request_finished with no transfer params."""
        scheduler = MooncakeConnectorScheduler(mock_vllm_config, "test_engine")
        mock_request = MagicMock()
        mock_request.kv_transfer_params = None

        result = scheduler.request_finished(mock_request, [1, 2, 3])
        assert result == (False, None)

    def test_request_finished_remote_decode_finished(self, mock_vllm_config):
        """Test request_finished with completed remote decode."""
        scheduler = MooncakeConnectorScheduler(mock_vllm_config, "test_engine")
        mock_request = MagicMock()
        mock_request.request_id = "req1"
        mock_request.status = MagicMock()
        mock_request.status.name = "FINISHED_LENGTH_CAPPED"
        mock_request.kv_transfer_params = {
            "do_remote_decode": True
        }

        result = scheduler.request_finished(mock_request, [1, 2, 3])

        # Should delay free and add to send queue
        assert result[0] is True  # delay_free_blocks
        assert result[1]["do_remote_prefill"] is True
        assert result[1]["do_remote_decode"] is False
        assert "req1" in scheduler._reqs_need_send
        assert scheduler._reqs_need_send["req1"] == [1, 2, 3]

    def test_request_finished_prefill_still_active(self, mock_vllm_config):
        """Test request_finished when prefill flag is still active."""
        scheduler = MooncakeConnectorScheduler(mock_vllm_config, "test_engine")
        mock_request = MagicMock()
        mock_request.request_id = "req1"
        mock_request.kv_transfer_params = {
            "do_remote_prefill": True
        }

        result = scheduler.request_finished(mock_request, [1, 2, 3])

        # Should add empty blocks to recv queue to notify remote
        assert result == (False, None)
        assert "req1" in scheduler._reqs_need_recv
        assert scheduler._reqs_need_recv["req1"] == (mock_request, [])


class TestMooncakeConnectorWorker:
    """Tests for MooncakeConnectorWorker."""

    @patch('vllm.distributed.kv_transfer.kv_connector.v1.mooncake_connector.TransferEngine')
    @patch('vllm.distributed.kv_transfer.kv_connector.v1.mooncake_connector.get_current_attn_backend')
    @patch('vllm.distributed.kv_transfer.kv_connector.v1.mooncake_connector.get_mooncake_side_channel_port')
    @patch('vllm.distributed.kv_transfer.kv_connector.v1.mooncake_connector.get_ip')
    @patch('vllm.distributed.kv_transfer.kv_connector.v1.mooncake_connector.get_tp_group')
    @patch('vllm.distributed.kv_transfer.kv_connector.v1.mooncake_connector.get_tensor_model_parallel_rank')
    @patch('vllm.distributed.kv_transfer.kv_connector.v1.mooncake_connector.get_tensor_model_parallel_world_size')
    def test_init_producer(self, mock_world_size, mock_rank, mock_tp_group,
                          mock_get_ip, mock_get_port, mock_backend, mock_engine_cls,
                          mock_vllm_config):
        """Test worker initialization for producer role."""
        # Setup mocks
        mock_world_size.return_value = 2
        mock_rank.return_value = 0
        mock_tp_group.return_value = MagicMock()
        mock_get_ip.return_value = "127.0.0.1"
        mock_get_port.return_value = 8080
        mock_backend.return_value = MagicMock()
        mock_backend.return_value.get_name.return_value = "test_backend"

        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine
        mock_engine.initialize.return_value = 0
        mock_engine.get_rpc_port.return_value = 9090

        worker = MooncakeConnectorWorker(mock_vllm_config, "test_engine")

        # Verify initialization
        mock_engine.initialize.assert_called_once()
        assert worker.engine == mock_engine
        assert worker.hostname == "127.0.0.1"
        assert worker.rpc_port == 9090
        assert worker.kv_role == "kv_producer"

        # Should start background threads for producer
        assert worker._sender_executor is not None
        assert worker.sender_loop is not None
        assert worker._sender_listener_t is not None

    @patch('vllm.distributed.kv_transfer.kv_connector.v1.mooncake_connector.TransferEngine')
    @patch('vllm.distributed.kv_transfer.kv_connector.v1.mooncake_connector.get_current_attn_backend')
    @patch('vllm.distributed.kv_transfer.kv_connector.v1.mooncake_connector.get_mooncake_side_channel_port')
    @patch('vllm.distributed.kv_transfer.kv_connector.v1.mooncake_connector.get_ip')
    @patch('vllm.distributed.kv_transfer.kv_connector.v1.mooncake_connector.get_tp_group')
    @patch('vllm.distributed.kv_transfer.kv_connector.v1.mooncake_connector.get_tensor_model_parallel_rank')
    @patch('vllm.distributed.kv_transfer.kv_connector.v1.mooncake_connector.get_tensor_model_parallel_world_size')
    def test_init_consumer(self, mock_world_size, mock_rank, mock_tp_group,
                          mock_get_ip, mock_get_port, mock_backend, mock_engine_cls,
                          mock_vllm_config_consumer):
        """Test worker initialization for consumer role."""
        # Setup mocks
        mock_world_size.return_value = 2
        mock_rank.return_value = 0
        mock_tp_group.return_value = MagicMock()
        mock_get_ip.return_value = "127.0.0.1"
        mock_get_port.return_value = 8080
        mock_backend.return_value = MagicMock()
        mock_backend.return_value.get_name.return_value = "test_backend"

        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine
        mock_engine.initialize.return_value = 0
        mock_engine.get_rpc_port.return_value = 9090

        worker = MooncakeConnectorWorker(mock_vllm_config_consumer, "test_engine")

        # Should not start sender threads for consumer
        assert worker._sender_executor is None
        assert worker.sender_loop is None
        assert worker._sender_listener_t is None

        # Should start receiver thread
        assert worker.receiver_loop is not None
        assert worker._mooncake_receiver_t is not None

    @patch('vllm.distributed.kv_transfer.kv_connector.v1.mooncake_connector.TransferEngine')
    def test_register_kv_caches_simple(self, mock_engine_cls, mock_vllm_config_consumer):
        """Test register_kv_caches with simple case."""
        # Setup mocks
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine
        mock_engine.batch_register_memory.return_value = 0

        # Setup required attributes
        worker = MagicMock()
        worker.kv_role = "kv_consumer"  # Skip sender setup
        worker.kv_topo = MagicMock()
        worker.kv_topo.split_k_and_v = False
        worker.block_size = 16
        worker.num_blocks = 0
        worker.kv_caches_base_addr = []
        worker.device_kv_caches = {}

        # Create kv caches
        kv_caches = {
            "layer1": torch.randn(10, 8, 16, 8),  # shape: [num_blocks, num_heads, block_size, head_size]
        }

        # Call register_kv_caches
        MooncakeConnectorWorker.register_kv_caches(worker, kv_caches)

        # Verify registration
        assert worker.num_blocks == 10
        assert worker.block_len == kv_caches["layer1"].nbytes // 10
        assert worker.device_kv_caches == kv_caches
        assert len(worker.kv_caches_base_addr) == 1

        mock_engine.batch_register_memory.assert_called_once()

    @patch('vllm.distributed.kv_transfer.kv_connector.v1.mooncake_connector.TransferEngine')
    def test_register_kv_caches_split_kv(self, mock_engine_cls, mock_vllm_config_consumer):
        """Test register_kv_caches with split K/V."""
        # Setup mocks
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine
        mock_engine.batch_register_memory.return_value = 0

        # Setup required attributes
        worker = MagicMock()
        worker.kv_role = "kv_consumer"
        worker.kv_topo = MagicMock()
        worker.kv_topo.split_k_and_v = True
        worker.block_size = 16
        worker.num_blocks = 0
        worker.kv_caches_base_addr = []
        worker.device_kv_caches = {}

        # Create split K/V caches
        kv_caches = {
            "layer1": [torch.randn(10, 4, 16, 8), torch.randn(10, 4, 16, 8)]  # K and V separate
        }

        # Call register_kv_caches
        MooncakeConnectorWorker.register_kv_caches(worker, kv_caches)

        # Should register both K and V tensors
        assert len(worker.kv_caches_base_addr) == 2
        mock_engine.batch_register_memory.assert_called_once()

    @patch('vllm.distributed.kv_transfer.kv_connector.v1.mooncake_connector.TransferEngine')
    def test_register_kv_caches_registration_failure(self, mock_engine_cls, mock_vllm_config_consumer):
        """Test register_kv_caches with registration failure."""
        # Setup mocks
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine
        mock_engine.batch_register_memory.return_value = -1  # Failure

        worker = MagicMock()
        worker.kv_role = "kv_consumer"
        worker.kv_topo = MagicMock()
        worker.kv_topo.split_k_and_v = False

        kv_caches = {"layer1": torch.randn(10, 8, 16, 8)}

        # Should raise RuntimeError
        with pytest.raises(RuntimeError, match="Mooncake batch memory registration failed"):
            MooncakeConnectorWorker.register_kv_caches(worker, kv_caches)

    @patch('vllm.distributed.kv_transfer.kv_connector.v1.mooncake_connector.TransferEngine')
    def test_register_kv_caches_size_mismatch(self, mock_engine_cls, mock_vllm_config_consumer):
        """Test register_kv_caches with size mismatch."""
        # Setup mocks
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine

        worker = MagicMock()
        worker.kv_role = "kv_consumer"
        worker.kv_topo = MagicMock()
        worker.kv_topo.split_k_and_v = False

        # Create caches with different sizes
        kv_caches = {
            "layer1": torch.randn(10, 8, 16, 8),
            "layer2": torch.randn(10, 8, 32, 8),  # Different size
        }

        # Should raise AssertionError
        with pytest.raises(AssertionError, match="All kv cache tensors must have the same size"):
            MooncakeConnectorWorker.register_kv_caches(worker, kv_caches)

    @patch('vllm.distributed.kv_transfer.kv_connector.v1.mooncake_connector.TransferEngine')
    @patch('vllm.distributed.kv_transfer.kv_connector.v1.mooncake_connector.asyncio')
    def test_get_finished(self, mock_asyncio, mock_engine_cls, mock_vllm_config_consumer):
        """Test get_finished method."""
        # Setup mocks
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine

        worker = MagicMock()
        worker.kv_role = "kv_both"  # Test both send and recv
        worker.finished_recving_reqs = {"recv1", "recv2"}
        worker.finished_sending_reqs = {"send1", "send2"}

        # Mock the async calls
        mock_recv_fut = MagicMock()
        mock_recv_fut.result.return_value = {"recv1", "recv2"}
        mock_send_fut = MagicMock()
        mock_send_fut.result.return_value = {"send1", "send2"}

        mock_asyncio.run_coroutine_threadsafe.side_effect = [mock_recv_fut, mock_send_fut]

        result = MooncakeConnectorWorker.get_finished(worker)

        assert result == ({"send1", "send2"}, {"recv1", "recv2"})

        # Verify async calls were made
        assert mock_asyncio.run_coroutine_threadsafe.call_count == 2

    @patch('vllm.distributed.kv_transfer.kv_connector.v1.mooncake_connector.TransferEngine')
    def test_get_finished_only_consumer(self, mock_engine_cls, mock_vllm_config_consumer):
        """Test get_finished for consumer-only role."""
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine

        worker = MagicMock()
        worker.kv_role = "kv_consumer"
        worker.finished_recving_reqs = {"recv1"}

        # Mock the async call
        with patch('vllm.distributed.kv_transfer.kv_connector.v1.mooncake_connector.asyncio') as mock_asyncio:
            mock_fut = MagicMock()
            mock_fut.result.return_value = {"recv1"}
            mock_asyncio.run_coroutine_threadsafe.return_value = mock_fut

            result = MooncakeConnectorWorker.get_finished(worker)

            assert result == (None, {"recv1"})

    @patch('vllm.distributed.kv_transfer.kv_connector.v1.mooncake_connector.TransferEngine')
    @patch('vllm.distributed.kv_transfer.kv_connector.v1.mooncake_connector.asyncio')
    def test_start_load_kv(self, mock_asyncio, mock_engine_cls, mock_vllm_config_consumer):
        """Test start_load_kv method."""
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine

        worker = MagicMock()
        worker.kv_role = "kv_both"

        # Create mock metadata
        metadata = MagicMock()
        metadata.reqs_to_recv = {"req1": MagicMock()}
        metadata.reqs_to_send = {"req2": [1, 2, 3]}

        # Mock group_kv_pull and record_send_reqs
        with patch.object(MooncakeConnectorWorker, 'group_kv_pull') as mock_group:
            with patch.object(MooncakeConnectorWorker, 'record_send_reqs') as mock_record:
                mock_group.return_value = {"path1": [("req1", [1, 2])]}
                mock_record.return_value = None

                MooncakeConnectorWorker.start_load_kv(worker, metadata)

                # Verify calls
                mock_group.assert_called_once_with(metadata)
                mock_record.assert_called_once_with(metadata)

                # Verify async calls for receiving
                mock_asyncio.run_coroutine_threadsafe.assert_called()

    @patch('vllm.distributed.kv_transfer.kv_connector.v1.mooncake_connector.TransferEngine')
    def test_group_kv_pull(self, mock_engine_cls, mock_vllm_config_consumer):
        """Test group_kv_pull method."""
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine

        worker = MagicMock()
        worker.tp_rank = 0

        # Create mock metadata
        metadata = MagicMock()
        recv_meta1 = MagicMock()
        recv_meta1.remote_host = "host1"
        recv_meta1.remote_port = 8080
        recv_meta1.local_block_ids = [1, 2]

        recv_meta2 = MagicMock()
        recv_meta2.remote_host = "host1"
        recv_meta2.remote_port = 8080
        recv_meta2.local_block_ids = [3, 4]

        metadata.reqs_to_recv = {
            "req1": recv_meta1,
            "req2": recv_meta2
        }

        with patch('vllm.distributed.kv_transfer.kv_connector.v1.mooncake_connector.make_zmq_path') as mock_make_path:
            mock_make_path.return_value = "tcp://host1:8080"

            result = MooncakeConnectorWorker.group_kv_pull(worker, metadata)

            # Should group by path
            expected = {"tcp://host1:8080": [("req1", [1, 2]), ("req2", [3, 4])]}
            assert result == expected

    @patch('vllm.distributed.kv_transfer.kv_connector.v1.mooncake_connector.TransferEngine')
    @patch('vllm.distributed.kv_transfer.kv_connector.v1.mooncake_connector.asyncio')
    def test_record_send_reqs(self, mock_asyncio, mock_engine_cls, mock_vllm_config_consumer):
        """Test record_send_reqs method."""
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine

        worker = MagicMock()
        worker.reqs_need_send = {}

        # Create mock metadata
        metadata = MagicMock()
        metadata.reqs_to_send = {
            "req1": [1, 2, 3],  # Has blocks
            "req2": []  # No blocks
        }

        MooncakeConnectorWorker.record_send_reqs(worker, metadata)

        # Should create SendBlockMeta entries
        assert "req1" in worker.reqs_need_send
        assert "req2" in worker.reqs_need_send

        assert worker.reqs_need_send["req1"].local_block_ids == [1, 2, 3]
        assert worker.reqs_need_send["req2"].local_block_ids == []

        # Verify async call for recording
        mock_asyncio.run_coroutine_threadsafe.assert_called_once()


class TestMooncakeConnectorMetadata:
    """Tests for MooncakeConnectorMetadata."""

    def test_init(self):
        """Test metadata initialization."""
        meta = MooncakeConnectorMetadata()
        assert meta.reqs_to_recv == {}
        assert meta.reqs_to_send == {}

    def test_add_new_req_recv(self):
        """Test adding receive request."""
        meta = MooncakeConnectorMetadata()

        meta.add_new_req(
            request_id="req1",
            local_block_ids=[1, 2, 3],
            kv_transfer_params={"remote_host": "host1", "remote_port": 8080},
            load_remote_cache=True
        )

        assert "req1" in meta.reqs_to_recv
        recv_meta = meta.reqs_to_recv["req1"]
        assert recv_meta.local_block_ids == [1, 2, 3]
        assert recv_meta.remote_host == "host1"
        assert recv_meta.remote_port == 8080

    def test_add_new_req_send(self):
        """Test adding send request."""
        meta = MooncakeConnectorMetadata()

        meta.add_new_req(
            request_id="req2",
            local_block_ids=[4, 5, 6],
            kv_transfer_params={},
            load_remote_cache=False
        )

        assert "req2" in meta.reqs_to_send
        assert meta.reqs_to_send["req2"] == [4, 5, 6]


class TestGroupConcurrentContiguous:
    """Tests for group_concurrent_contiguous utility function."""

    def test_empty_indices(self):
        """Test with empty indices."""
        result = group_concurrent_contiguous([], [])
        assert result == ([], [])

    def test_single_group(self):
        """Test with contiguous indices."""
        src_indices = [1, 2, 3, 4]
        dst_indices = [5, 6, 7, 8]

        src_groups, dst_groups = group_concurrent_contiguous(src_indices, dst_indices)

        assert src_groups == [[1, 2, 3, 4]]
        assert dst_groups == [[5, 6, 7, 8]]

    def test_multiple_groups(self):
        """Test with non-contiguous indices."""
        src_indices = [1, 2, 4, 5, 7]
        dst_indices = [10, 11, 14, 15, 17]

        src_groups, dst_groups = group_concurrent_contiguous(src_indices, dst_indices)

        assert src_groups == [[1, 2], [4, 5], [7]]
        assert dst_groups == [[10, 11], [14, 15], [17]]

    def test_single_indices(self):
        """Test with single indices."""
        src_indices = [1]
        dst_indices = [5]

        src_groups, dst_groups = group_concurrent_contiguous(src_indices, dst_indices)

        assert src_groups == [[1]]
        assert dst_groups == [[5]]


class TestMooncakeAgentMetadata:
    """Tests for MooncakeAgentMetadata."""

    def test_init_and_serialization(self):
        """Test metadata initialization and serialization."""
        metadata = MooncakeAgentMetadata(
            remote_hostname="127.0.0.1",
            remote_port=9090,
            request_ids=["req1", "req2"],
            kv_caches_base_addr=[1000, 2000],
            block_ids=[[1, 2], [3, 4]]
        )

        assert metadata.remote_hostname == "127.0.0.1"
        assert metadata.remote_port == 9090
        assert metadata.request_ids == ["req1", "req2"]
        assert metadata.kv_caches_base_addr == [1000, 2000]
        assert metadata.block_ids == [[1, 2], [3, 4]]

    def test_msgpack_serialization(self):
        """Test msgpack serialization/deserialization."""
        from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_connector import msgspec

        metadata = MooncakeAgentMetadata(
            remote_hostname="127.0.0.1",
            remote_port=9090,
            request_ids=["req1", "req2"],
            kv_caches_base_addr=[1000, 2000],
            block_ids=[[1, 2], [3, 4]]
        )

        encoder = msgspec.msgpack.Encoder()
        decoder = msgspec.msgpack.Decoder(MooncakeAgentMetadata)

        serialized = encoder.encode(metadata)
        deserialized = decoder.decode(serialized)

        assert deserialized.remote_hostname == metadata.remote_hostname
        assert deserialized.remote_port == metadata.remote_port
        assert deserialized.request_ids == metadata.request_ids
        assert deserialized.kv_caches_base_addr == metadata.kv_caches_base_addr
        assert deserialized.block_ids == metadata.block_ids


class TestSendBlockMeta:
    """Tests for SendBlockMeta."""

    def test_init(self):
        """Test SendBlockMeta initialization."""
        meta = SendBlockMeta(local_block_ids=[1, 2, 3])

        assert meta.local_block_ids == [1, 2, 3]
        assert meta.expire_time == float("inf")
        assert isinstance(meta.ready, asyncio.Event)

    def test_custom_expire_time(self):
        """Test SendBlockMeta with custom expire time."""
        expire_time = time.time() + 60
        meta = SendBlockMeta(local_block_ids=[1, 2], expire_time=expire_time)

        assert meta.expire_time == expire_time


class TestRecvReqMeta:
    """Tests for RecvReqMeta."""

    def test_init(self):
        """Test RecvReqMeta initialization."""
        meta = RecvReqMeta(
            local_block_ids=[1, 2, 3],
            remote_host="192.168.1.100",
            remote_port=8080
        )

        assert meta.local_block_ids == [1, 2, 3]
        assert meta.remote_host == "192.168.1.100"
        assert meta.remote_port == 8080


# Integration test helpers
class MockTransferEngine:
    """Mock TransferEngine for testing."""

    def __init__(self):
        self.registered_memory = []
        self.transfers = []

    def initialize(self, hostname, handshake, protocol, extra):
        return 0

    def get_rpc_port(self):
        return 9090

    def batch_register_memory(self, ptrs, lens):
        self.registered_memory.append((ptrs, lens))
        return 0

    def batch_transfer_sync_write(self, remote_session, src_ptrs, dst_ptrs, lengths):
        self.transfers.append((remote_session, src_ptrs, dst_ptrs, lengths))
        return 0


@pytest.fixture
def mock_transfer_engine():
    """Create a mock TransferEngine for testing."""
    return MockTransferEngine()


class TestIntegration:
    """Integration tests for MooncakeConnector workflow."""

    @patch('vllm.distributed.kv_transfer.kv_connector.v1.mooncake_connector.TransferEngine')
    def test_full_workflow_producer(self, mock_engine_cls, mock_vllm_config):
        """Test full producer workflow."""
        # Setup mock engine
        mock_engine = MockTransferEngine()
        mock_engine_cls.return_value = mock_engine

        # Create connector and register KV caches
        connector = MooncakeConnector(mock_vllm_config, KVConnectorRole.WORKER)
        kv_caches = {"layer1": torch.randn(10, 8, 16, 8)}
        connector.register_kv_caches(kv_caches)

        # Verify registration
        assert len(mock_engine.registered_memory) == 1

        # Create metadata for a request
        meta = MooncakeConnectorMetadata()
        meta.add_new_req(
            request_id="test_req",
            local_block_ids=[1, 2, 3],
            kv_transfer_params={"remote_host": "remote", "remote_port": 8080},
            load_remote_cache=False
        )

        # Bind metadata and start load
        connector.bind_connector_metadata(meta)
        mock_forward_context = MagicMock()
        connector.start_load_kv(mock_forward_context)

        # Verify the workflow completed without errors
        assert connector._connector_metadata == meta

    @patch('vllm.distributed.kv_transfer.kv_connector.v1.mooncake_connector.TransferEngine')
    def test_full_workflow_consumer(self, mock_engine_cls, mock_vllm_config_consumer):
        """Test full consumer workflow."""
        # Setup mock engine
        mock_engine = MockTransferEngine()
        mock_engine_cls.return_value = mock_engine

        # Create connector and register KV caches
        connector = MooncakeConnector(mock_vllm_config_consumer, KVConnectorRole.WORKER)
        kv_caches = {"layer1": torch.randn(10, 8, 16, 8)}
        connector.register_kv_caches(kv_caches)

        # Verify registration
        assert len(mock_engine.registered_memory) == 1

        # Consumer should not have sender threads
        assert connector.connector_worker._sender_executor is None