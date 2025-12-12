"""
Test script for RSNA 2023 1st Place Solution.

This script performs basic sanity checks on the codebase to ensure:
1. All utility modules can be imported
2. Key functions work correctly
3. No syntax errors in major modules
"""

import sys
import os
import traceback

# Add project root to path
PROJECT_ROOT = "/home/dingziming/PycharmProjects/AI-Practices/09-practical-projects/05-kaggle-competitions/03-RSNA-2023-1st-Place"
sys.path.insert(0, PROJECT_ROOT)

def test_imports():
    """Test that all utility modules can be imported."""
    print("=" * 60)
    print("Testing imports...")
    print("=" * 60)

    try:
        from paths import PATHS
        print("✓ paths module imported successfully")
    except Exception as e:
        print(f"✗ Failed to import paths: {e}")
        return False

    try:
        from utils.dicom_utils import (
            glob_sorted,
            get_standardized_pixel_array,
            get_windowed_image,
            get_rescaled_image,
        )
        print("✓ utils.dicom_utils imported successfully")
    except Exception as e:
        print(f"✗ Failed to import utils.dicom_utils: {e}")
        traceback.print_exc()
        return False

    try:
        from utils.data_utils import (
            rle_encode,
            rle_decode,
            get_volume_data,
            process_volume,
        )
        print("✓ utils.data_utils imported successfully")
    except Exception as e:
        print(f"✗ Failed to import utils.data_utils: {e}")
        traceback.print_exc()
        return False

    try:
        from utils.distributed_utils import (
            seed_everything,
            is_main_process,
            get_rank,
        )
        print("✓ utils.distributed_utils imported successfully")
    except Exception as e:
        print(f"✗ Failed to import utils.distributed_utils: {e}")
        traceback.print_exc()
        return False

    print("\n✓ All imports successful!\n")
    return True


def test_dicom_utils():
    """Test DICOM utility functions."""
    print("=" * 60)
    print("Testing DICOM utilities...")
    print("=" * 60)

    try:
        import numpy as np
        from utils.dicom_utils import get_windowed_image

        # Test windowing function with dummy data
        img = np.random.randn(512, 512) * 500  # Simulate HU values
        windowed = get_windowed_image(img, WL=50, WW=400)

        assert windowed.shape == (512, 512), "Shape mismatch"
        assert windowed.dtype == np.uint8, "dtype mismatch"
        assert windowed.min() >= 0 and windowed.max() <= 255, "Value range error"

        print("✓ get_windowed_image works correctly")
    except Exception as e:
        print(f"✗ DICOM utils test failed: {e}")
        traceback.print_exc()
        return False

    print("\n✓ DICOM utilities test passed!\n")
    return True


def test_data_utils():
    """Test data processing utilities."""
    print("=" * 60)
    print("Testing data utilities...")
    print("=" * 60)

    try:
        import numpy as np
        from utils.data_utils import rle_encode, rle_decode

        # Test RLE encoding and decoding
        mask = np.array([[0, 0, 1, 1], [1, 1, 0, 0]], dtype=np.uint8)
        rle = rle_encode(mask)
        decoded = rle_decode(rle, (2, 4))

        assert np.array_equal(mask, decoded), "RLE encode/decode mismatch"
        print("✓ RLE encode/decode works correctly")

    except Exception as e:
        print(f"✗ Data utils test failed: {e}")
        traceback.print_exc()
        return False

    print("\n✓ Data utilities test passed!\n")
    return True


def test_distributed_utils():
    """Test distributed training utilities."""
    print("=" * 60)
    print("Testing distributed utilities...")
    print("=" * 60)

    try:
        from utils.distributed_utils import seed_everything, is_main_process

        # Test seed setting (should not raise errors)
        seed_everything(42)
        print("✓ seed_everything works correctly")

        # Test main process check (should return True when not in DDP mode)
        is_main = is_main_process()
        assert is_main == True, "Should be main process in non-distributed mode"
        print("✓ is_main_process works correctly")

    except Exception as e:
        print(f"✗ Distributed utils test failed: {e}")
        traceback.print_exc()
        return False

    print("\n✓ Distributed utilities test passed!\n")
    return True


def test_config_imports():
    """Test that config files can be imported without errors."""
    print("=" * 60)
    print("Testing config imports...")
    print("=" * 60)

    configs = [
        "Configs.segmentation_config",
        "Configs.coat_lite_medium_bs2_lr125e6_cfg",
    ]

    for config_module in configs:
        try:
            module = __import__(config_module, fromlist=['CFG'])
            CFG = module.CFG
            print(f"✓ {config_module} imported successfully")
        except Exception as e:
            print(f"✗ Failed to import {config_module}: {e}")
            traceback.print_exc()
            return False

    print("\n✓ All config imports successful!\n")
    return True


def run_all_tests():
    """Run all test suites."""
    print("\n")
    print("=" * 60)
    print(" RSNA 2023 1st Place Solution - Test Suite")
    print("=" * 60)
    print("\n")

    results = []

    results.append(("Imports", test_imports()))
    results.append(("DICOM Utils", test_dicom_utils()))
    results.append(("Data Utils", test_data_utils()))
    results.append(("Distributed Utils", test_distributed_utils()))
    results.append(("Config Imports", test_config_imports()))

    # Summary
    print("\n")
    print("=" * 60)
    print(" Test Summary")
    print("=" * 60)

    all_passed = True
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name:.<40} {status}")
        if not result:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n🎉 All tests passed!\n")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the errors above.\n")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
