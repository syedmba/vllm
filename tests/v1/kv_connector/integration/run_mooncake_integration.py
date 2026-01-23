#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Runner script for MooncakeConnector integration tests.

This script provides an easy way to run the Mooncake integration tests
that require multiple GPUs and real network communication.
"""

import argparse
import sys
import torch


def check_requirements():
    """Check if system meets requirements for integration testing."""
    print("Checking system requirements...")

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    print(f"✅ CUDA available, {torch.cuda.device_count()} device(s)")

    # Check GPU count
    if torch.cuda.device_count() < 2:
        print(f"❌ Need at least 2 GPUs, found {torch.cuda.device_count()}")
        return False
    print("✅ Sufficient GPUs available")

    # Check if Mooncake is importable
    try:
        from mooncake.engine import TransferEngine
        print("✅ Mooncake TransferEngine available")
    except ImportError:
        print("❌ Mooncake TransferEngine not available")
        print("   Please install Mooncake following:")
        print("   https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md")
        return False

    return True


def run_integration_test():
    """Run the full integration test."""
    print("\n" + "="*60)
    print("Running MooncakeConnector Integration Test")
    print("="*60)

    if not check_requirements():
        print("\n❌ Requirements not met, exiting")
        return False

    print("\n🚀 Starting integration test...")

    # Import and run the test
    from test_mooncake_integration import TestMooncakeIntegration

    test_instance = TestMooncakeIntegration()

    try:
        test_instance.test_kv_cache_transfer_between_gpus()
        print("\n✅ Integration test PASSED!")
        return True
    except Exception as e:
        print(f"\n❌ Integration test FAILED: {e}")
        return False


def run_single_gpu_test():
    """Run the single GPU registration test."""
    print("\n" + "="*60)
    print("Running MooncakeConnector Single GPU Test")
    print("="*60)

    if not torch.cuda.is_available():
        print("❌ CUDA not available, cannot run single GPU test")
        return False

    print("🚀 Starting single GPU test...")

    from test_mooncake_integration import TestMooncakeIntegration

    test_instance = TestMooncakeIntegration()

    try:
        test_instance.test_single_gpu_kv_cache_registration()
        print("\n✅ Single GPU test PASSED!")
        return True
    except Exception as e:
        print(f"\n❌ Single GPU test FAILED: {e}")
        return False


def run_config_test():
    """Run the basic config creation test."""
    print("\n" + "="*60)
    print("Running MooncakeConnector Config Test")
    print("="*60)

    print("🚀 Starting config test...")

    from test_mooncake_integration import TestMooncakeIntegration

    test_instance = TestMooncakeIntegration()

    try:
        test_instance.test_config_creation()
        print("\n✅ Config test PASSED!")
        return True
    except Exception as e:
        print(f"\n❌ Config test FAILED: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="MooncakeConnector Integration Test Runner")
    parser.add_argument(
        "test_type",
        choices=["integration", "single_gpu", "config", "all"],
        default="all",
        nargs="?",
        help="Type of test to run"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    success = True

    if args.test_type in ["config", "all"]:
        success &= run_config_test()

    if args.test_type in ["single_gpu", "all"]:
        success &= run_single_gpu_test()

    if args.test_type in ["integration", "all"]:
        success &= run_integration_test()

    print("\n" + "="*60)
    if success:
        print("🎉 All requested tests PASSED!")
        return 0
    else:
        print("💥 Some tests FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())