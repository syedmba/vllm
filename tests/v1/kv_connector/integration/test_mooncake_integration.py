# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import multiprocessing as mp
import os
import time
from unittest.mock import MagicMock

import torch

# Optional pytest import
try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    # Create dummy pytest decorators for when pytest is not available
    class _pytest:
        @staticmethod
        def fixture(func):
            return func
        @staticmethod
        def mark(**kwargs):
            def decorator(func):
                return func
            return decorator
        class skipif:
            def __init__(self, condition, reason):
                self.condition = condition
                self.reason = reason
            def __call__(self, func):
                if self.condition:
                    print(f"Skipping {func.__name__}: {self.reason}")
                    return lambda *args, **kwargs: None
                return func
    pytest = _pytest()

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_connector import (
    MooncakeConnector,
    MooncakeConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole


def create_minimal_vllm_config(kv_role: str, engine_id: str = "test_engine"):
    """Create minimal VllmConfig for MooncakeConnector testing."""
    config = MagicMock(spec=VllmConfig)
    config.kv_transfer_config = MagicMock()
    config.kv_transfer_config.engine_id = engine_id
    config.kv_transfer_config.kv_role = kv_role
    config.kv_transfer_config.kv_connector_extra_config = {
        "mooncake_protocol": "tcp",  # Use TCP for local testing
        "num_workers": 1  # Minimal workers for testing
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


def producer_process(queue, gpu_id: int):
    """Producer process that sends KV cache."""
    try:
        # Set GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        torch.cuda.set_device(gpu_id)

        print(f"Producer: Starting on GPU {gpu_id}")

        # Create minimal config for producer
        config = create_minimal_vllm_config("kv_producer", "producer_engine")

        # Initialize MooncakeConnector as producer
        connector = MooncakeConnector(config, KVConnectorRole.WORKER)
        print("Producer: MooncakeConnector initialized")

        # Create simple KV cache (zeros tensor)
        kv_caches = {
            "layer0": torch.zeros(10, 8, 16, 8, dtype=torch.float16, device=f"cuda:{gpu_id}")
            # Shape: [num_blocks, num_heads, block_size, head_size]
        }
        print(f"Producer: Created KV cache with shape {kv_caches['layer0'].shape}")

        # Register KV cache with Mooncake
        connector.register_kv_caches(kv_caches)
        print("Producer: KV cache registered with Mooncake")

        # Create metadata for sending
        metadata = MooncakeConnectorMetadata()
        metadata.add_new_req(
            request_id="test_request_1",
            local_block_ids=[0, 1, 2],  # Send first 3 blocks
            kv_transfer_params={
                "remote_host": "127.0.0.1",  # Connect to consumer
                "remote_port": 8081  # Consumer's side channel port
            },
            load_remote_cache=False  # This is a send operation
        )

        # Bind metadata
        connector.bind_connector_metadata(metadata)
        print("Producer: Metadata bound, starting transfer")

        # Create mock forward context
        forward_context = MagicMock()
        forward_context.virtual_engine = 0

        # Start the transfer
        connector.start_load_kv(forward_context)
        print("Producer: Transfer started")

        # Wait for completion
        max_wait = 30  # 30 seconds timeout
        start_time = time.time()

        while time.time() - start_time < max_wait:
            finished_sending, finished_receiving = connector.get_finished({"test_request_1"})

            if finished_sending and "test_request_1" in finished_sending:
                print("Producer: Transfer completed successfully!")
                queue.put(("SUCCESS", "Transfer completed"))
                break

            time.sleep(0.1)
        else:
            print("Producer: Transfer timed out")
            queue.put(("TIMEOUT", "Transfer did not complete within timeout"))

    except Exception as e:
        print(f"Producer: Error - {e}")
        queue.put(("ERROR", str(e)))

    finally:
        # Cleanup
        try:
            connector.connector_worker.shutdown()
            print("Producer: Cleanup completed")
        except:
            pass


def consumer_process(queue, gpu_id: int):
    """Consumer process that receives KV cache."""
    try:
        # Set GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        torch.cuda.set_device(gpu_id)

        print(f"Consumer: Starting on GPU {gpu_id}")

        # Create minimal config for consumer
        config = create_minimal_vllm_config("kv_consumer", "consumer_engine")

        # Initialize MooncakeConnector as consumer
        connector = MooncakeConnector(config, KVConnectorRole.WORKER)
        print("Consumer: MooncakeConnector initialized")

        # Create KV cache buffer to receive into
        kv_caches = {
            "layer0": torch.zeros(10, 8, 16, 8, dtype=torch.float16, device=f"cuda:{gpu_id}")
            # Same shape as producer
        }
        print(f"Consumer: Created KV cache buffer with shape {kv_caches['layer0'].shape}")

        # Register KV cache with Mooncake
        connector.register_kv_caches(kv_caches)
        print("Consumer: KV cache registered with Mooncake")

        # Consumer waits for incoming transfers (handled by background threads)
        print("Consumer: Waiting for incoming transfers...")

        # Wait for transfer to complete or timeout
        max_wait = 60  # 60 seconds timeout (longer since consumer waits for producer)
        start_time = time.time()

        while time.time() - start_time < max_wait:
            finished_sending, finished_receiving = connector.get_finished(set())

            if finished_receiving:
                print(f"Consumer: Received transfer for requests: {finished_receiving}")
                # Check if data was actually transferred
                if torch.sum(kv_caches['layer0']) != 0:
                    print("Consumer: KV cache data received successfully!")
                    queue.put(("SUCCESS", f"Received data for requests: {finished_receiving}"))
                else:
                    print("Consumer: No data received in KV cache")
                    queue.put(("NO_DATA", "KV cache remained zeros"))
                break

            time.sleep(0.5)

        if time.time() - start_time >= max_wait:
            print("Consumer: Timeout waiting for transfer")
            queue.put(("TIMEOUT", "No transfer received within timeout"))

    except Exception as e:
        print(f"Consumer: Error - {e}")
        queue.put(("ERROR", str(e)))

    finally:
        # Cleanup
        try:
            connector.connector_worker.shutdown()
            print("Consumer: Cleanup completed")
        except:
            pass


class TestMooncakeIntegration:
    """Integration tests for MooncakeConnector with real transfers."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for integration test")
    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Need at least 2 GPUs for integration test")
    def test_kv_cache_transfer_between_gpus(self):
        """Test complete KV cache transfer between two GPUs using Mooncake."""
        # Create queues for inter-process communication
        producer_queue = mp.Queue()
        consumer_queue = mp.Queue()

        # Start consumer process (on GPU 0)
        consumer_proc = mp.Process(target=consumer_process, args=(consumer_queue, 0))
        consumer_proc.start()

        # Give consumer time to initialize
        time.sleep(2)

        # Start producer process (on GPU 1)
        producer_proc = mp.Process(target=producer_process, args=(producer_queue, 1))
        producer_proc.start()

        try:
            # Wait for both processes to complete
            producer_proc.join(timeout=60)
            consumer_proc.join(timeout=60)

            # Check results
            if not producer_queue.empty():
                producer_result = producer_queue.get()
                print(f"Producer result: {producer_result}")
            else:
                producer_result = ("NO_RESULT", "Producer process did not report result")

            if not consumer_queue.empty():
                consumer_result = consumer_queue.get()
                print(f"Consumer result: {consumer_result}")
            else:
                consumer_result = ("NO_RESULT", "Consumer process did not report result")

            # Assert successful transfer
            assert producer_result[0] == "SUCCESS", f"Producer failed: {producer_result}"
            assert consumer_result[0] == "SUCCESS", f"Consumer failed: {consumer_result}"

            print("✅ Integration test passed: KV cache successfully transferred between GPUs")

        except Exception as e:
            # Cleanup processes if test fails
            if producer_proc.is_alive():
                producer_proc.terminate()
            if consumer_proc.is_alive():
                consumer_proc.terminate()
            raise e

        finally:
            # Ensure processes are cleaned up
            if producer_proc.is_alive():
                producer_proc.terminate()
            if consumer_proc.is_alive():
                consumer_proc.terminate()


    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for test")
    def test_single_gpu_kv_cache_registration(self):
        """Test KV cache registration on a single GPU (simpler test)."""
        # Set GPU
        gpu_id = 0
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        torch.cuda.set_device(gpu_id)

        # Create config
        config = create_minimal_vllm_config("kv_producer", "single_gpu_test")

        # Initialize connector
        connector = MooncakeConnector(config, KVConnectorRole.WORKER)

        # Create KV cache
        kv_caches = {
            "layer0": torch.zeros(5, 4, 16, 8, dtype=torch.float16, device=f"cuda:{gpu_id}")
        }

        # Register KV cache
        connector.register_kv_caches(kv_caches)

        # Verify registration worked (no exceptions thrown)
        assert connector.connector_worker.device_kv_caches == kv_caches
        assert len(connector.connector_worker.kv_caches_base_addr) > 0

        # Cleanup
        connector.connector_worker.shutdown()

        print("✅ Single GPU registration test passed")


    def test_config_creation(self):
        """Test that config creation works correctly."""
        config = create_minimal_vllm_config("kv_producer", "test_engine")

        assert config.kv_transfer_config.engine_id == "test_engine"
        assert config.kv_transfer_config.kv_role == "kv_producer"
        assert config.kv_transfer_config.kv_connector_extra_config["mooncake_protocol"] == "tcp"
        assert config.cache_config.block_size == 16

        print("✅ Config creation test passed")


if __name__ == "__main__":
    # Allow running integration test directly
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--integration":
        # Run the full integration test
        test = TestMooncakeIntegration()

        # Check GPU availability
        if not torch.cuda.is_available():
            print("❌ CUDA not available, skipping integration test")
            sys.exit(1)

        if torch.cuda.device_count() < 2:
            print("❌ Need at least 2 GPUs for integration test, skipping")
            sys.exit(1)

        print("Running Mooncake integration test...")
        test.test_kv_cache_transfer_between_gpus()
        print("✅ Integration test completed successfully")

    else:
        # Run basic tests
        test = TestMooncakeIntegration()
        test.test_config_creation()

        if torch.cuda.is_available():
            test.test_single_gpu_kv_cache_registration()
        else:
            print("⚠️  CUDA not available, skipping GPU tests")