"""
Comprehensive Unit Tests for imgproc
====================================
Tests basic functions, spatial filters, separability, median noise removal,
Sobel gradients, histogram equalization, frequency domain filtering, and OpenCV alignment.
"""

import numpy as np
import pytest

from imgproc.basics import rgb_to_gray, pad
from imgproc.filters import (
    IDENTITY_3X3,
    BOX_BLUR_3X3,
    SHARPEN_3X3,
    convolve2d_loop,
    convolve2d_vectorized,
    convolve2d,
    gaussian_kernel,
    gaussian_kernel_1d,
    get_gaussian_kernel_size,
    gaussian_blur,
    gaussian_blur_separable,
    add_salt_pepper,
    add_gaussian_noise,
    median_filter,
    psnr,
    sobel
)
from imgproc.histogram import compute_histogram, compute_cdf, equalize
from imgproc.frequency import (
    compute_dft_spectrum,
    ideal_lowpass_mask,
    gaussian_lowpass_mask,
    apply_frequency_filter
)


# ---------------------------------------------------------------------------
# Test 1: Color to Grayscale
# ---------------------------------------------------------------------------
def test_rgb_to_gray_weights():
    # Test pure red, pure green, pure blue
    red = np.zeros((2, 2, 3), dtype=np.float64)
    red[:, :, 0] = 1.0
    gray_r = rgb_to_gray(red)
    assert np.allclose(gray_r, 0.299)

    green = np.zeros((2, 2, 3), dtype=np.float64)
    green[:, :, 1] = 1.0
    gray_g = rgb_to_gray(green)
    assert np.allclose(gray_g, 0.587)

    blue = np.zeros((2, 2, 3), dtype=np.float64)
    blue[:, :, 2] = 1.0
    gray_b = rgb_to_gray(blue)
    assert np.allclose(gray_b, 0.114)

    # Green must be larger than red and blue combined (0.587 > 0.299 + 0.114 = 0.413)
    assert gray_g[0, 0] > (gray_r[0, 0] + gray_b[0, 0])


def test_rgb_to_gray_uint8_scaling():
    # Single pixel [100, 200, 50]
    img = np.array([[[100, 200, 50]]], dtype=np.uint8)
    expected_val = (0.299 * 100 + 0.587 * 200 + 0.114 * 50) / 255.0
    res = rgb_to_gray(img)
    assert np.allclose(res, expected_val, atol=1e-5)
    assert res.dtype == np.float64


# ---------------------------------------------------------------------------
# Test 2: Padding Modes
# ---------------------------------------------------------------------------
def test_padding_modes():
    img = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ], dtype=np.float64)

    # Zero pad (1 row top/bottom, 1 col left/right)
    p_zero = pad(img, 1, 1, mode="zero")
    assert p_zero.shape == (4, 4)
    assert p_zero[0, 0] == 0.0
    assert np.array_equal(p_zero[1:3, 1:3], img)

    # Replicate pad
    p_rep = pad(img, 1, 1, mode="replicate")
    assert p_rep.shape == (4, 4)
    assert p_rep[0, 0] == 1.0  # Corner replicated
    assert p_rep[3, 3] == 4.0

    # Reflect pad
    p_ref = pad(img, 1, 1, mode="reflect")
    assert p_ref.shape == (4, 4)
    assert np.array_equal(p_ref[1:3, 1:3], img)


# ---------------------------------------------------------------------------
# Test 3: Convolution (Loop vs Vectorized, Identity Kernel)
# ---------------------------------------------------------------------------
def test_convolve_identity():
    rng = np.random.default_rng(42)
    img = rng.uniform(0.0, 1.0, (16, 16))

    out_loop = convolve2d_loop(img, IDENTITY_3X3, pad_mode="reflect")
    out_vec = convolve2d_vectorized(img, IDENTITY_3X3, pad_mode="reflect")

    # Identity filter must return the exact input image
    assert np.allclose(out_loop, img, atol=1e-12)
    assert np.allclose(out_vec, img, atol=1e-12)


def test_convolve_loop_matches_vectorized():
    rng = np.random.default_rng(123)
    img = rng.uniform(0.0, 1.0, (20, 25))
    kernel = rng.uniform(-1.0, 1.0, (5, 5))

    out_loop = convolve2d_loop(img, kernel, pad_mode="reflect")
    out_vec = convolve2d_vectorized(img, kernel, pad_mode="reflect")

    assert np.allclose(out_loop, out_vec, atol=1e-10)


# ---------------------------------------------------------------------------
# Test 4: Gaussian Blur & Separability
# ---------------------------------------------------------------------------
def test_gaussian_kernel_properties():
    for sigma in [0.5, 1.0, 2.0, 3.5]:
        size = get_gaussian_kernel_size(sigma)
        assert size % 2 == 1  # Must be odd
        k = gaussian_kernel(size, sigma)
        assert np.isclose(np.sum(k), 1.0, atol=1e-7)
        # Symmetry check
        assert np.allclose(k, k.T)
        assert np.allclose(k, k[::-1, ::-1])


def test_gaussian_separability():
    rng = np.random.default_rng(99)
    img = rng.uniform(0.0, 1.0, (30, 30))
    sigma = 1.5

    blur_2d = gaussian_blur(img, sigma, pad_mode="reflect")
    blur_sep = gaussian_blur_separable(img, sigma, pad_mode="reflect")

    # Separable filtering must match 2D convolution within numerical float precision
    assert np.allclose(blur_2d, blur_sep, atol=1e-7)


# ---------------------------------------------------------------------------
# Test 5: Median Filter vs Gaussian on Salt-and-Pepper Noise
# ---------------------------------------------------------------------------
def test_median_filter_denoising():
    # Create clean synthetic image
    clean = np.full((32, 32), 0.5, dtype=np.float64)
    # Add heavy salt & pepper noise
    np.random.seed(42)
    corrupted = add_salt_pepper(clean, amount=0.1, salt_vs_pepper=0.5)

    # Filter with median vs gaussian
    med_filtered = median_filter(corrupted, size=3, pad_mode="reflect")
    gauss_filtered = gaussian_blur(corrupted, sigma=1.0, pad_mode="reflect")

    psnr_noisy = psnr(clean, corrupted)
    psnr_med = psnr(clean, med_filtered)
    psnr_gauss = psnr(clean, gauss_filtered)

    # Median filter should drastically outperform Gaussian on impulse noise
    assert psnr_med > psnr_noisy
    assert psnr_med > psnr_gauss
    assert psnr_med > 35.0  # Clean uniform image recovered almost perfectly


# ---------------------------------------------------------------------------
# Test 6: Sobel Gradient Direction & Magnitude
# ---------------------------------------------------------------------------
def test_sobel_gradient_ramp():
    # Image with horizontal ramp: intensity increases with column x
    ramp = np.tile(np.linspace(0.0, 1.0, 20), (20, 1))

    res = sobel(ramp, pad_mode="reflect")
    # Horizontal gradient Gx must be positive across interior
    assert np.all(res["gx"][2:-2, 2:-2] > 0.0)
    # Vertical gradient Gy must be practically zero (no intensity change across rows)
    assert np.allclose(res["gy"][2:-2, 2:-2], 0.0, atol=1e-5)
    # Direction should be 0 radians (angle along x axis)
    assert np.allclose(res["direction"][2:-2, 2:-2], 0.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Test 7: Histogram Equalization
# ---------------------------------------------------------------------------
def test_histogram_equalization():
    # Low-contrast image concentrated in [0.2, 0.4]
    rng = np.random.default_rng(7)
    dark_img = rng.uniform(0.2, 0.4, (50, 50))

    eq, cdf_orig, cdf_eq = equalize(dark_img, bins=256)

    # Contrast should be expanded: min close to 0, max close to 1
    assert eq.min() < 0.05
    assert eq.max() > 0.95
    # CDF of equalized image should be approximately linear
    assert cdf_eq[-1] == 1.0


# ---------------------------------------------------------------------------
# Test 8: Frequency Domain
# ---------------------------------------------------------------------------
def test_frequency_domain_masks():
    shape = (32, 32)
    cutoff = 8.0

    ilpf = ideal_lowpass_mask(shape, cutoff)
    glpf = gaussian_lowpass_mask(shape, cutoff)

    # Center is at (16, 16)
    assert ilpf[16, 16] == 1.0
    assert np.isclose(glpf[16, 16], 1.0)

    # Far corner must have lower filter response
    assert ilpf[0, 0] == 0.0
    assert glpf[0, 0] < 0.05


# ---------------------------------------------------------------------------
# Test 9: OpenCV Cross-Verification (when cv2 is available)
# ---------------------------------------------------------------------------
def test_opencv_filter2d_match():
    try:
        import cv2
    except ImportError:
        pytest.skip("OpenCV not installed, skipping test.")

    rng = np.random.default_rng(42)
    img = rng.uniform(0.0, 1.0, (25, 25))

    # Test with box blur
    kernel = BOX_BLUR_3X3

    # Important note: OpenCV filter2D computes cross-correlation, not convolution!
    # For a symmetric kernel like BOX_BLUR, kernel == flipped_kernel, so correlation == convolution.
    # OpenCV BORDER_DEFAULT is BORDER_REFLECT_101.
    cv_out = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REFLECT)
    my_out = convolve2d_vectorized(img, kernel, pad_mode="reflect")

    # Compare interior (where border effects don't dominate)
    diff = np.abs(cv_out[2:-2, 2:-2] - my_out[2:-2, 2:-2])
    assert np.mean(diff) < 1e-4
