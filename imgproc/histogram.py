"""
Histogram & Histogram Equalization
==================================
Histogram computation, Cumulative Distribution Function (CDF), and Histogram Equalization.
Part of the IPCVAI Portfolio Project.
"""

import numpy as np


def compute_histogram(image: np.ndarray, bins: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the intensity histogram of a grayscale image in [0.0, 1.0].

    Parameters
    ----------
    image : np.ndarray
        Grayscale image of shape (H, W) with values in [0.0, 1.0].
    bins : int
        Number of intensity bins (default 256).

    Returns
    -------
    counts : np.ndarray
        Array of length `bins` with pixel counts for each bin.
    bin_centers : np.ndarray
        Array of length `bins` with center intensity value of each bin.
    """
    # Quantize [0.0, 1.0] to integer indices in range [0, bins - 1]
    indices = np.clip(np.floor(image * (bins - 1)).astype(int), 0, bins - 1)
    counts = np.bincount(indices.ravel(), minlength=bins)
    bin_centers = (np.arange(bins) + 0.5) / bins
    return counts, bin_centers


def compute_cdf(counts: np.ndarray) -> np.ndarray:
    """
    Compute the normalized Cumulative Distribution Function (CDF).
    CDF(k) = sum_{i=0}^k P(i) where P is the probability distribution.
    Ranges from CDF[0] > 0 up to CDF[-1] == 1.0.
    """
    cum = np.cumsum(counts, dtype=np.float64)
    total = cum[-1]
    if total == 0:
        return np.zeros_like(cum)
    return cum / total


def equalize(image: np.ndarray, bins: int = 256) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Histogram Equalization from scratch using Cumulative Distribution Function (CDF).

    Theoretical Intuition:
    ----------------------
    In continuous probability theory, if a continuous random variable X has
    cumulative distribution function F_X(x), then the transformed random
    variable Y = F_X(X) is uniformly distributed on [0, 1]!
    
    In digital images, mapping pixel intensity v through its normalized CDF
    stretches high-probability (frequent) intensity ranges across a wider
    spectrum while compressing low-probability ranges. This maximizes image
    contrast and information entropy.

    Formula (with minimum non-zero CDF offset):
    --------------------------------------------
        T(v) = (CDF(v) - CDF_min) / (1.0 - CDF_min)
    This guarantees that the lowest present intensity maps to 0.0,
    preventing washed-out black levels.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image in [0.0, 1.0].
    bins : int
        Number of quantization bins (default 256).

    Returns
    -------
    equalized : np.ndarray
        Contrast-enhanced image in [0.0, 1.0].
    cdf_original : np.ndarray
        Normalized CDF of the original image (length `bins`).
    cdf_equalized : np.ndarray
        Normalized CDF of the equalized image (length `bins`).
    """
    # Quantize image to bin indices
    indices = np.clip(np.floor(image * (bins - 1)).astype(int), 0, bins - 1)
    counts = np.bincount(indices.ravel(), minlength=bins)
    cdf_orig = compute_cdf(counts)

    # Find the smallest non-zero cumulative probability
    non_zero_cdfs = cdf_orig[cdf_orig > 0]
    cdf_min = non_zero_cdfs[0] if len(non_zero_cdfs) > 0 else 0.0

    # Build the transformation lookup table (LUT)
    if 1.0 - cdf_min > 0:
        lut = (cdf_orig - cdf_min) / (1.0 - cdf_min)
        lut = np.clip(lut, 0.0, 1.0)
    else:
        lut = cdf_orig

    # Map each pixel through the LUT using integer index indexing
    equalized = lut[indices]

    # Compute CDF of equalized image
    eq_indices = np.clip(np.floor(equalized * (bins - 1)).astype(int), 0, bins - 1)
    eq_counts = np.bincount(eq_indices.ravel(), minlength=bins)
    cdf_eq = compute_cdf(eq_counts)

    return equalized, cdf_orig, cdf_eq
