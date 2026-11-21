"""
Spatial Filters & Convolution
=============================
2D Convolution, Separable Gaussian Blur, Median Filter, Noise, PSNR, and Sobel Gradients.
Part of the IPCVAI Portfolio Project.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from imgproc.basics import pad


# ---------------------------------------------------------------------------
# Standard Reference Kernels
# ---------------------------------------------------------------------------
IDENTITY_3X3 = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0]
], dtype=np.float64)

BOX_BLUR_3X3 = np.ones((3, 3), dtype=np.float64) / 9.0

SHARPEN_3X3 = np.array([
    [ 0.0, -1.0,  0.0],
    [-1.0,  5.0, -1.0],
    [ 0.0, -1.0,  0.0]
], dtype=np.float64)


# ---------------------------------------------------------------------------
# 2D Convolution (Loop vs Vectorized)
# ---------------------------------------------------------------------------
def convolve2d_loop(image: np.ndarray, kernel: np.ndarray, pad_mode: str = "reflect") -> np.ndarray:
    """
    Compute 2D discrete convolution using explicit loops.

    Convolution vs. Cross-Correlation:
    -----------------------------------
    Mathematical convolution requires flipping the kernel by 180 degrees
    (both horizontally and vertically) before computing the inner product:
        (I * K)[y, x] = sum_m sum_n I[y - m, x - n] * K[m, n]
                      = sum_m sum_n I[y + m, x + n] * K_flipped[m, n]

    Flipping ensures mathematical properties such as commutativity:
        f * g = g * f
    and associativity:
        (f * g) * h = f * (g * h)
    Cross-correlation does NOT flip the kernel and is not commutative.

    Parameters
    ----------
    image : np.ndarray
        2D array of shape (H, W).
    kernel : np.ndarray
        2D filter kernel of shape (Kh, Kw). Typically odd dimensions.
    pad_mode : str
        Border padding mode ('zero', 'replicate', 'reflect').

    Returns
    -------
    np.ndarray
        Convolved 2D array of same shape (H, W).
    """
    H, W = image.shape
    Kh, Kw = kernel.shape
    pad_h = Kh // 2
    pad_w = Kw // 2

    # Step 1: Flip kernel 180 degrees (both axes)
    flipped_k = kernel[::-1, ::-1]

    # Step 2: Pad the image so output has the same size
    padded = pad(image, pad_h, pad_w, mode=pad_mode)

    # Step 3: Slide kernel across each pixel
    out = np.zeros((H, W), dtype=np.float64)
    for r in range(H):
        for c in range(W):
            # Extract window of size (Kh, Kw)
            window = padded[r : r + Kh, c : c + Kw]
            # Element-wise multiply and sum
            out[r, c] = np.sum(window * flipped_k)

    return out


def convolve2d_vectorized(image: np.ndarray, kernel: np.ndarray, pad_mode: str = "reflect") -> np.ndarray:
    """
    Fast 2D discrete convolution using NumPy sliding_window_view and tensor dot product.

    Parameters
    ----------
    image : np.ndarray
        2D array of shape (H, W).
    kernel : np.ndarray
        2D filter kernel of shape (Kh, Kw).
    pad_mode : str
        Border padding mode.

    Returns
    -------
    np.ndarray
        Convolved 2D array of same shape (H, W).
    """
    H, W = image.shape
    Kh, Kw = kernel.shape
    pad_h = Kh // 2
    pad_w = Kw // 2

    # Flip kernel 180 degrees
    flipped_k = kernel[::-1, ::-1].astype(np.float64)

    # Pad image
    padded = pad(image, pad_h, pad_w, mode=pad_mode)

    # Extract all windows at once with stride tricks: shape (H, W, Kh, Kw)
    windows = sliding_window_view(padded, (Kh, Kw))

    # Compute sum over the last two dimensions (Kh, Kw) via tensordot
    out = np.tensordot(windows, flipped_k, axes=([2, 3], [0, 1]))
    return out


def convolve2d(image: np.ndarray, kernel: np.ndarray, pad_mode: str = "reflect", method: str = "vectorized") -> np.ndarray:
    """Convenience wrapper for 2D convolution."""
    if method == "vectorized":
        return convolve2d_vectorized(image, kernel, pad_mode=pad_mode)
    elif method == "loop":
        return convolve2d_loop(image, kernel, pad_mode=pad_mode)
    else:
        raise ValueError(f"Unknown convolution method '{method}'. Choose 'vectorized' or 'loop'.")


# ---------------------------------------------------------------------------
# Gaussian Blur & Separability
# ---------------------------------------------------------------------------
def gaussian_kernel_1d(size: int, sigma: float) -> np.ndarray:
    """
    Generate a 1D normalized Gaussian kernel.
    Formula: G(x) = exp(-x^2 / (2 * sigma^2))
    Normalized so sum of elements equals 1.0.
    """
    radius = size // 2
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel_1d = np.exp(-0.5 * (x / sigma) ** 2)
    return kernel_1d / np.sum(kernel_1d)


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """
    Generate a 2D isotropic Gaussian kernel.
    Formula: G(x, y) = exp(-(x^2 + y^2) / (2 * sigma^2))
    Normalized so sum of elements equals 1.0.
    """
    k1d = gaussian_kernel_1d(size, sigma)
    # 2D Gaussian is the outer product of two 1D Gaussians
    k2d = np.outer(k1d, k1d)
    return k2d / np.sum(k2d)


def get_gaussian_kernel_size(sigma: float) -> int:
    """
    Rule of thumb for Gaussian kernel size:
    ~6 * sigma captures 99.73% of the Gaussian area (+- 3 standard deviations).
    Always rounded up to the nearest odd integer.
    """
    size = int(np.ceil(6.0 * sigma))
    if size % 2 == 0:
        size += 1
    return max(3, size)


def gaussian_blur(image: np.ndarray, sigma: float, pad_mode: str = "reflect") -> np.ndarray:
    """Apply 2D Gaussian blur with automatic kernel sizing."""
    size = get_gaussian_kernel_size(sigma)
    kernel = gaussian_kernel(size, sigma)
    return convolve2d_vectorized(image, kernel, pad_mode=pad_mode)


def gaussian_blur_separable(image: np.ndarray, sigma: float, pad_mode: str = "reflect") -> np.ndarray:
    """
    Fast Gaussian blur exploiting separability:
        G_2D(x, y) = G_1D(x) * G_1D(y)

    Instead of convolving an (H, W) image with an (N, N) kernel (O(H * W * N^2) ops),
    we convolve horizontally with (1, N) then vertically with (N, 1) (O(H * W * 2N) ops).
    Speedup ratio: N / 2.
    """
    size = get_gaussian_kernel_size(sigma)
    k1d = gaussian_kernel_1d(size, sigma)

    # 1D horizontal kernel: shape (1, size)
    k_horiz = k1d.reshape(1, size)
    # 1D vertical kernel: shape (size, 1)
    k_vert = k1d.reshape(size, 1)

    # Convolve rows, then convolve columns
    temp = convolve2d_vectorized(image, k_horiz, pad_mode=pad_mode)
    out = convolve2d_vectorized(temp, k_vert, pad_mode=pad_mode)
    return out


# ---------------------------------------------------------------------------
# Noise Generation, Median Filter & PSNR
# ---------------------------------------------------------------------------
def add_salt_pepper(image: np.ndarray, amount: float = 0.05, salt_vs_pepper: float = 0.5) -> np.ndarray:
    """
    Add impulse (salt and pepper) noise to an image.
    Amount: proportion of pixels corrupted (e.g. 0.05 = 5%).
    Salt = 1.0 (white), Pepper = 0.0 (black).
    """
    noisy = image.copy()
    num_corrupt = int(amount * image.size)

    # Pick random coordinates
    coords_r = np.random.randint(0, image.shape[0], num_corrupt)
    coords_c = np.random.randint(0, image.shape[1], num_corrupt)

    # Split between salt and pepper
    num_salt = int(num_corrupt * salt_vs_pepper)
    noisy[coords_r[:num_salt], coords_c[:num_salt]] = 1.0
    noisy[coords_r[num_salt:], coords_c[num_salt:]] = 0.0

    return noisy


def add_gaussian_noise(image: np.ndarray, std: float = 0.05, mean: float = 0.0) -> np.ndarray:
    """Add additive zero-mean Gaussian noise N(mean, std^2) clipped to [0.0, 1.0]."""
    noise = np.random.normal(mean, std, image.shape)
    noisy = image + noise
    return np.clip(noisy, 0.0, 1.0)


def median_filter(image: np.ndarray, size: int = 3, pad_mode: str = "reflect") -> np.ndarray:
    """
    Non-linear spatial filter replacing each pixel with the median of its local neighborhood.
    Highly effective against salt-and-pepper noise because outliers (0.0 or 1.0)
    are pushed to the extremes of the sorted list, leaving the true median intact.
    """
    if size % 2 == 0:
        raise ValueError("Median filter size must be an odd integer.")

    pad_amount = size // 2
    padded = pad(image, pad_amount, pad_amount, mode=pad_mode)
    windows = sliding_window_view(padded, (size, size))
    # Compute median across the window axes
    return np.median(windows, axis=(-2, -1))


def psnr(original: np.ndarray, target: np.ndarray) -> float:
    """
    Peak Signal-to-Noise Ratio (in dB).
    Higher is better.
        PSNR = 10 * log10(MAX_I^2 / MSE)
    For normalized float images, MAX_I = 1.0.
    """
    mse = np.mean((original - target) ** 2)
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


# ---------------------------------------------------------------------------
# Edge Detection with Sobel Operator
# ---------------------------------------------------------------------------
def sobel(image: np.ndarray, pad_mode: str = "reflect") -> dict:
    """
    Compute edge gradients using the 3x3 Sobel operators.

    Discrete approximations to derivatives:
        Gx approximates dI/dx (vertical edges)
        Gy approximates dI/dy (horizontal edges)

    Sobel combines smoothing along one axis with central differentiation along the other:
        Gx = [-1, 0, 1] (derivative) (x) [1, 2, 1]^T (smoothing)
        Gy = [1, 2, 1] (smoothing) (x) [-1, 0, 1]^T (derivative)

    Returns:
    --------
    dict containing:
        'gx': horizontal gradient
        'gy': vertical gradient
        'magnitude': sqrt(gx^2 + gy^2)
        'direction': arctan2(gy, gx) in radians [-pi, pi]
    """
    # Note: When convolving with convolve2d, the kernel is flipped by 180 degrees.
    # To compute standard forward/backward central difference [I(x+1) - I(x-1)],
    # the un-flipped correlation kernel is:
    # Kx_corr = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    # Ky_corr = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    # To keep mathematical convolution convention consistent, we pass the correlation kernels
    # or perform cross-correlation directly:
    kx = np.array([
        [-1.0, 0.0, 1.0],
        [-2.0, 0.0, 2.0],
        [-1.0, 0.0, 1.0]
    ], dtype=np.float64)

    ky = np.array([
        [-1.0, -2.0, -1.0],
        [ 0.0,  0.0,  0.0],
        [ 1.0,  2.0,  1.0]
    ], dtype=np.float64)

    # In spatial domain, cross-correlation with kx gives I(x+1) - I(x-1):
    # To get this using convolve2d (which flips), we pass flipped(kx):
    gx = convolve2d_vectorized(image, kx[::-1, ::-1], pad_mode=pad_mode)
    gy = convolve2d_vectorized(image, ky[::-1, ::-1], pad_mode=pad_mode)

    magnitude = np.hypot(gx, gy)
    direction = np.arctan2(gy, gx)

    return {
        "gx": gx,
        "gy": gy,
        "magnitude": magnitude,
        "direction": direction
    }


def threshold(image: np.ndarray, thresh: float) -> np.ndarray:
    """Simple binary threshold: 1.0 if pixel >= thresh else 0.0."""
    return (image >= thresh).astype(np.float64)
