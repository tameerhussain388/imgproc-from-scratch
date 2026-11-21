"""
Fourier Domain Filtering
========================
2D Discrete Fourier Transform (DFT), Frequency Spectrum, Ideal & Gaussian Filters,
and Ringing Artifacts Analysis.
Part of the IPCVAI Portfolio Project.
"""

import numpy as np


def compute_dft_spectrum(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute 2D Discrete Fourier Transform (DFT) and center the low frequencies.

    Intuition of 2D Frequency Domain:
    ---------------------------------
    Any image can be decomposed into a linear combination of 2D sinusoids
    of varying spatial frequencies, orientations, amplitudes, and phases:
    - Low frequencies (near center after fftshift): Smooth gradients, uniform surfaces, overall illumination.
    - High frequencies (far from center): Sharp edges, fine textures, noise, rapid transitions.

    Returns
    -------
    F_shift : np.ndarray
        Complex 2D array of shifted DFT coefficients.
    log_magnitude : np.ndarray
        Log-scaled magnitude spectrum log(1 + |F|) for visual inspection.
    phase : np.ndarray
        Phase spectrum in radians [-pi, pi].
    """
    # 2D Fast Fourier Transform
    F = np.fft.fft2(image)
    # Shift DC component (frequency (0, 0)) to the center of the array
    F_shift = np.fft.fftshift(F)

    magnitude = np.abs(F_shift)
    # Log compression because DC component is typically orders of magnitude larger than high frequencies
    log_magnitude = np.log1p(magnitude)
    phase = np.angle(F_shift)

    return F_shift, log_magnitude, phase


def distance_matrix(shape: tuple[int, int]) -> np.ndarray:
    """
    Compute Euclidean distance D(u, v) from each frequency point (u, v)
    to the frequency center (H/2, W/2).
    """
    H, W = shape
    center_u = H / 2.0
    center_v = W / 2.0
    u = np.arange(H) - center_u
    v = np.arange(W) - center_v
    U, V = np.meshgrid(u, v, indexing="ij")
    return np.sqrt(U ** 2 + V ** 2)


def ideal_lowpass_mask(shape: tuple[int, int], cutoff_d0: float) -> np.ndarray:
    """
    Ideal Low-Pass Filter (ILPF) transfer function H(u, v):
        H(u, v) = 1 if D(u, v) <= D0 else 0
    Sharp cutoff in frequency domain causes ringing artifacts in spatial domain.
    """
    D = distance_matrix(shape)
    return (D <= cutoff_d0).astype(np.float64)


def gaussian_lowpass_mask(shape: tuple[int, int], cutoff_d0: float) -> np.ndarray:
    """
    Gaussian Low-Pass Filter (GLPF) transfer function H(u, v):
        H(u, v) = exp(-D(u, v)^2 / (2 * D0^2))
    Smooth transition in frequency domain guarantees ZERO ringing artifacts in spatial domain.
    """
    D = distance_matrix(shape)
    return np.exp(-(D ** 2) / (2.0 * (cutoff_d0 ** 2)))


def ideal_highpass_mask(shape: tuple[int, int], cutoff_d0: float) -> np.ndarray:
    """Ideal High-Pass Filter (IHPF): 1 - ILPF."""
    return 1.0 - ideal_lowpass_mask(shape, cutoff_d0)


def gaussian_highpass_mask(shape: tuple[int, int], cutoff_d0: float) -> np.ndarray:
    """Gaussian High-Pass Filter (GHPF): 1 - GLPF."""
    return 1.0 - gaussian_lowpass_mask(shape, cutoff_d0)


def apply_frequency_filter(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply a frequency domain mask and reconstruct the image via Inverse DFT.

    Convolution Theorem in Practice:
    --------------------------------
    Spatial convolution of image f with filter h corresponds to element-wise
    multiplication in frequency domain:
        F{f * h} = F{f} . F{h}
        f * h = F^{-1}{ F{f} . F{h} }
    """
    # Step 1: Forward DFT and center frequencies
    F = np.fft.fft2(image)
    F_shift = np.fft.fftshift(F)

    # Step 2: Multiply by filter transfer function
    filtered_shift = F_shift * mask

    # Step 3: Shift back and compute Inverse DFT
    filtered_F = np.fft.ifftshift(filtered_shift)
    spatial = np.fft.ifft2(filtered_F)

    # Step 4: Take real part and clip to [0.0, 1.0]
    result = np.real(spatial)
    return np.clip(result, 0.0, 1.0)
