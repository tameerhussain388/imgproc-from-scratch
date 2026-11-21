"""
Image Processing Basics
=======================
Handles image loading, color conversion, and array padding from scratch in NumPy.
Part of the IPCVAI Portfolio Project.
"""

import numpy as np


def rgb_to_gray(rgb_image: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image (H, W, 3) to a grayscale image (H, W)
    using the ITU-R BT.601 luminance formula:
        Y = 0.299 * R + 0.587 * G + 0.114 * B

    Why the weights?
    ----------------
    Human photopic vision is most sensitive to green wavelengths (~555 nm)
    due to the density and spectral response of M and L retinal cone cells.
    Green contributes the most (58.7%) to perceived brightness, followed by
    Red (29.9%) and Blue (11.4%).

    Parameters
    ----------
    rgb_image : np.ndarray
        Array of shape (H, W, 3). Can be uint8 in [0, 255] or float in [0.0, 1.0].

    Returns
    -------
    np.ndarray
        Grayscale image of shape (H, W) as float64 with values strictly in [0.0, 1.0].
    """
    # Ensure float64
    img = rgb_image.astype(np.float64)

    # Normalize to [0.0, 1.0] if in uint8 [0, 255] range
    if img.max() > 1.0:
        img = img / 255.0

    # Extract individual channels: (H, W)
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    # Compute weighted luminance
    gray = 0.299 * r + 0.587 * g + 0.114 * b

    # Clip for safety against floating point inaccuracies
    return np.clip(gray, 0.0, 1.0)


def load_image(filepath: str, max_dim: int = None) -> np.ndarray:
    """
    Load an image from disk and return it as a normalized grayscale
    float64 array with values in [0.0, 1.0].

    Parameters
    ----------
    filepath : str
        Path to image file (PNG, JPG, BMP, etc.).
    max_dim : int, optional
        If set, resize image so its maximum dimension is at most max_dim,
        preserving aspect ratio.

    Returns
    -------
    np.ndarray
        Grayscale image of shape (H, W) as float64 in range [0.0, 1.0].
    """
    try:
        from PIL import Image
        pil_img = Image.open(filepath)
        if max_dim is not None and max(pil_img.size) > max_dim:
            scale = max_dim / max(pil_img.size)
            new_size = (int(pil_img.size[0] * scale), int(pil_img.size[1] * scale))
            pil_img = pil_img.resize(new_size, Image.Resampling.BILINEAR)
        img = np.array(pil_img)
    except ImportError:
        import matplotlib.pyplot as plt
        img = plt.imread(filepath)

    # Case 1: Already grayscale (H, W)
    if img.ndim == 2:
        gray = img.astype(np.float64)
        if gray.max() > 1.0:
            gray = gray / 255.0
        return np.clip(gray, 0.0, 1.0)

    # Case 2: RGBA (H, W, 4) - discard alpha channel
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    # Case 3: RGB (H, W, 3)
    if img.ndim == 3 and img.shape[2] == 3:
        return rgb_to_gray(img)

    raise ValueError(f"Unsupported image shape: {img.shape}")


def pad(image: np.ndarray, pad_h: int, pad_w: int, mode: str = "zero") -> np.ndarray:
    """
    Pad a 2D grayscale image array with borders.

    Modes implemented:
    ------------------
    - 'zero':
        Borders are filled with 0.0.
        [0, 0, | a, b, c | 0, 0]
    - 'replicate' (clamp):
        Border pixels are extended outward.
        [a, a, | a, b, c | c, c]
    - 'reflect' (symmetric across edge):
        Reflects pixels across the outer edge without duplicating the edge pixel.
        [c, b, | a, b, c | b, a]

    Parameters
    ----------
    image : np.ndarray
        2D array of shape (H, W).
    pad_h : int
        Number of rows to add on top and bottom.
    pad_w : int
        Number of columns to add on left and right.
    mode : str
        One of {'zero', 'replicate', 'reflect'}.

    Returns
    -------
    np.ndarray
        Padded array of shape (H + 2*pad_h, W + 2*pad_w).
    """
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D image array, got ndim={image.ndim}")

    H, W = image.shape
    new_H = H + 2 * pad_h
    new_W = W + 2 * pad_w

    if pad_h == 0 and pad_w == 0:
        return image.copy()

    # Pre-allocate output canvas
    out = np.zeros((new_H, new_W), dtype=image.dtype)

    # Place original image in center
    out[pad_h : pad_h + H, pad_w : pad_w + W] = image

    if mode == "zero":
        # Canvas was pre-filled with zeros, so nothing more to do
        return out

    elif mode == "replicate":
        # Top and bottom strips
        for r in range(pad_h):
            out[r, pad_w : pad_w + W] = image[0, :]
            out[pad_h + H + r, pad_w : pad_w + W] = image[-1, :]

        # Left and right strips (including corners)
        for c in range(pad_w):
            out[:, c] = out[:, pad_w]
            out[:, pad_w + W + c] = out[:, pad_w + W - 1]

        return out

    elif mode == "reflect":
        # Reflect top and bottom:
        # e.g., if image has rows 0..H-1, row -1 becomes row 1, row -2 becomes row 2
        for r in range(pad_h):
            # Top: reflects row (r + 1) if available, capped at H-1
            src_r_top = min(r + 1, H - 1)
            out[pad_h - 1 - r, pad_w : pad_w + W] = image[src_r_top, :]

            # Bottom: reflects row (H - 2 - r) if available, capped at 0
            src_r_bot = max(H - 2 - r, 0)
            out[pad_h + H + r, pad_w : pad_w + W] = image[src_r_bot, :]

        # Reflect left and right across entire height (including padded top/bottom)
        for c in range(pad_w):
            src_c_left = pad_w + min(c + 1, W - 1)
            out[:, pad_w - 1 - c] = out[:, src_c_left]

            src_c_right = pad_w + max(W - 2 - c, 0)
            out[:, pad_w + W + c] = out[:, src_c_right]

        return out

    else:
        raise ValueError(f"Unknown pad mode '{mode}'. Choose 'zero', 'replicate', or 'reflect'.")
