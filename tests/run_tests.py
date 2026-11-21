"""
Zero-dependency test runner using standard library.
Can run standalone or via pytest.
"""

import sys
import os

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.test_filters import (
    test_rgb_to_gray_weights,
    test_rgb_to_gray_uint8_scaling,
    test_padding_modes,
    test_convolve_identity,
    test_convolve_loop_matches_vectorized,
    test_gaussian_kernel_properties,
    test_gaussian_separability,
    test_median_filter_denoising,
    test_sobel_gradient_ramp,
    test_histogram_equalization,
    test_frequency_domain_masks
)

def run_all():
    tests = [
        ("Step 1: RGB to Gray Luminance Weights", test_rgb_to_gray_weights),
        ("Step 1: RGB to Gray Uint8 Scaling", test_rgb_to_gray_uint8_scaling),
        ("Step 2: 2D Padding Modes", test_padding_modes),
        ("Step 3: Convolution with Identity Kernel", test_convolve_identity),
        ("Step 3: Loop vs Vectorized Convolution Equivalence", test_convolve_loop_matches_vectorized),
        ("Step 4: Gaussian Kernel Sum & Symmetry", test_gaussian_kernel_properties),
        ("Step 4: Gaussian Separability Equivalence", test_gaussian_separability),
        ("Step 5: Median Filter on Impulse Noise", test_median_filter_denoising),
        ("Step 6: Sobel Gradient Direction & Magnitude", test_sobel_gradient_ramp),
        ("Step 7: Histogram Equalization CDF & Range", test_histogram_equalization),
        ("Step 8: Frequency Domain Filter Masks", test_frequency_domain_masks)
    ]

    print("=" * 70)
    print("RUNNING ALL IMAGE PROCESSING UNIT TESTS (PURE NUMPY)")
    print("=" * 70)
    passed = 0
    for name, test_fn in tests:
        try:
            test_fn()
            print(f"[PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"[FAIL] {name}: {e}")

    print("=" * 70)
    print(f"RESULTS: {passed}/{len(tests)} tests passed.")
    print("=" * 70)
    return passed == len(tests)

if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
