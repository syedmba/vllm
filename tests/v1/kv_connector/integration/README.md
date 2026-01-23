# MooncakeConnector Integration Tests

This directory contains integration tests for the MooncakeConnector that test real KV cache transfers between GPUs using the actual Mooncake TransferEngine.

## Overview

The integration tests create minimal vLLM configurations and test the complete workflow of:
1. Initializing MooncakeConnector instances on different GPUs
2. Registering KV caches with Mooncake
3. Transferring KV cache data between instances
4. Verifying successful transfer completion

## Test Files

- `test_mooncake_integration.py` - Main integration test suite
- `run_mooncake_integration.py` - Helper script to run tests
- `README.md` - This documentation

## Requirements

### Hardware
- At least 2 GPUs (for full integration test)
- CUDA-compatible system

### Software
- Mooncake TransferEngine installed
- PyTorch with CUDA support
- vLLM dependencies

### Installation

Install Mooncake following the [official instructions](https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md).

## Test Types

### 1. Config Test
Basic test that validates configuration creation works correctly.

```bash
python run_mooncake_integration.py config
```

### 2. Single GPU Test
Tests KV cache registration on a single GPU.

```bash
python run_mooncake_integration.py single_gpu
```

### 3. Full Integration Test
Tests complete KV cache transfer between two GPUs.

```bash
python run_mooncake_integration.py integration
```

### 4. Run All Tests
Runs all tests in sequence.

```bash
python run_mooncake_integration.py all
```

## How It Works

### Test Architecture

The integration test uses multiprocessing to create two separate processes:

1. **Producer Process** (GPU 1):
   - Initializes as `kv_producer` role
   - Creates zeros KV cache tensor
   - Registers KV cache with Mooncake
   - Initiates transfer to consumer

2. **Consumer Process** (GPU 0):
   - Initializes as `kv_consumer` role
   - Creates KV cache buffer
   - Registers buffer with Mooncake
   - Receives transfer from producer

### Communication

- **ZMQ Side Channel**: Used for coordination between producer and consumer
- **Mooncake TransferEngine**: Handles the actual RDMA/TCP data transfer
- **Multiprocessing Queues**: Used for test coordination and result reporting

### Data Flow

```
Producer (GPU 1)                  Consumer (GPU 0)
    |                                 |
    |-- Create zeros KV cache        |
    |-- Register with Mooncake       |
    |                                 |-- Create KV buffer
    |                                 |-- Register with Mooncake
    |-- Start transfer --------------->|
    |<------------------- Transfer ---|
    |-- Wait for completion           |
    |                                 |-- Receive completion
    |-- Report success                |-- Report success
```

## Configuration

The tests use minimal vLLM configurations:

```python
config = {
    "kv_role": "kv_producer" | "kv_consumer",
    "engine_id": "producer_engine" | "consumer_engine",
    "mooncake_protocol": "tcp",  # For local testing
    "num_workers": 1,  # Minimal for testing
}
```

## Debugging

### Enable Debug Logging

Set environment variables for detailed logging:

```bash
export VLLM_LOG_LEVEL=DEBUG
export PYTHONPATH=/path/to/vllm:$PYTHONPATH
python run_mooncake_integration.py integration
```

### Common Issues

1. **Mooncake not installed**: Follow installation instructions
2. **Insufficient GPUs**: Need at least 2 GPUs for full integration test
3. **Port conflicts**: Tests use hardcoded ports (8080-8081)
4. **CUDA errors**: Ensure CUDA_VISIBLE_DEVICES is properly set
5. **Timeout**: Increase timeout values if transfers take longer

### Manual Testing

You can also run the test directly:

```bash
cd tests/v1/kv_connector/integration
python test_mooncake_integration.py
```

Or run specific pytest tests:

```bash
pytest test_mooncake_integration.py::TestMooncakeIntegration::test_kv_cache_transfer_between_gpus -v -s
```

## Expected Output

Successful integration test output:

```
============================================================
Running MooncakeConnector Integration Test
============================================================
Checking system requirements...
✅ CUDA available, 2 device(s)
✅ Sufficient GPUs available
✅ Mooncake TransferEngine available

🚀 Starting integration test...
Producer: Starting on GPU 1
Producer: MooncakeConnector initialized
Producer: Created KV cache with shape torch.Size([10, 8, 16, 8])
Producer: KV cache registered with Mooncake
Producer: Metadata bound, starting transfer
Producer: Transfer started
Consumer: Starting on GPU 0
Consumer: MooncakeConnector initialized
Consumer: Created KV cache buffer with shape torch.Size([10, 8, 16, 8])
Consumer: KV cache registered with Mooncake
Consumer: Waiting for incoming transfers...
Consumer: Received transfer for requests: {...}
Consumer: KV cache data received successfully!
Producer: Transfer completed successfully!

✅ Integration test PASSED: KV cache successfully transferred between GPUs
```

## Performance Notes

- **Transfer Size**: Tests transfer 10 blocks × 8 heads × 16 tokens × 8 dims = ~80KB per layer
- **Protocol**: Uses TCP for local testing (change to "rdma" for production)
- **Timeout**: 30-60 seconds depending on network conditions
- **Cleanup**: Automatic cleanup of Mooncake resources on test completion