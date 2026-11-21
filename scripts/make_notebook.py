"""Script to generate the comprehensive demo.ipynb notebook."""

import os
import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Image Processing From Scratch in NumPy\n",
                "### Preparation for IPCVAI Erasmus Mundus Master\n",
                "\n",
                "This notebook implements and explores fundamental image processing algorithms from first principles using **only NumPy**:\n",
                "1. **Images as Arrays & Luminance Representation** (ITU-R BT.601)\n",
                "2. **2D Padding Modes** (Zero, Replicate, Reflect)\n",
                "3. **2D Discrete Convolution** (Correlation vs. Convolution, Loop vs. Vectorized)\n",
                "4. **Gaussian Blur & Separability** ($O(N^2)$ vs. $O(2N)$)\n",
                "5. **Median Filter vs. Gaussian on Noise** (Salt & Pepper vs. Gaussian noise, PSNR metric)\n",
                "6. **Edge Gradients with Sobel Operators** ($G_x, G_y$, Magnitude, Direction, Pre-smoothing)\n",
                "7. **Histogram Equalisation** (CDF transformation, contrast stretching)\n",
                "8. **Fourier Domain Filtering** (2D DFT, Log Spectrum, Gibbs Ringing Artifacts, Convolution Theorem)\n",
                "9. **Numerical Benchmarking against OpenCV** (`filter2D`, `GaussianBlur`, `medianBlur`, `Sobel`, `equalizeHist`)\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Setup environment and imports\n",
                "import os\n",
                "import sys\n",
                "import time\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "from skimage import data\n",
                "import cv2\n",
                "\n",
                "# Add parent folder to path so imgproc package is importable\n",
                "sys.path.append(os.path.abspath('..'))\n",
                "\n",
                "from imgproc.basics import rgb_to_gray, load_image, pad\n",
                "from imgproc.filters import (\n",
                "    IDENTITY_3X3, BOX_BLUR_3X3, SHARPEN_3X3,\n",
                "    convolve2d_loop, convolve2d_vectorized, convolve2d,\n",
                "    gaussian_kernel, get_gaussian_kernel_size, gaussian_blur, gaussian_blur_separable,\n",
                "    add_salt_pepper, add_gaussian_noise, median_filter, psnr, sobel, threshold\n",
                ")\n",
                "from imgproc.histogram import compute_histogram, compute_cdf, equalize\n",
                "from imgproc.frequency import (\n",
                "    compute_dft_spectrum, ideal_lowpass_mask, gaussian_lowpass_mask,\n",
                "    ideal_highpass_mask, gaussian_highpass_mask, apply_frequency_filter\n",
                ")\n",
                "\n",
                "print(\"All modules loaded successfully!\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 1: Images as Arrays & Color Conversion\n",
                "\n",
                "A digital grayscale image is represented as a 2D matrix of shape `(H, W)`. We store pixels as `float64` strictly normalized to `[0.0, 1.0]`.\n",
                "\n",
                "The ITU-R BT.601 formula converts RGB to luminance $Y$:\n",
                "$$Y = 0.299 \\cdot R + 0.587 \\cdot G + 0.114 \\cdot B$$\n",
                "\n",
                "**Why Green has the highest weight (58.7%)**: Human photopic vision relies on retinal M- and L-cone photoreceptors whose composite spectral sensitivity peaks around 555 nm (green-yellow). Green light contributes vastly more to subjective perceived brightness than blue or red."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load sample camera image\n",
                "cam = data.camera().astype(np.float64) / 255.0\n",
                "print(f\"Shape: {cam.shape}\")\n",
                "print(f\"Dtype: {cam.dtype}\")\n",
                "print(f\"Intensity Range: [{cam.min():.3f}, {cam.max():.3f}]\")\n",
                "\n",
                "# Hand-check test with a single RGB pixel [100, 200, 50]\n",
                "test_pixel = np.array([[[100, 200, 50]]], dtype=np.uint8)\n",
                "gray_res = rgb_to_gray(test_pixel)\n",
                "expected = (0.299 * 100 + 0.587 * 200 + 0.114 * 50) / 255.0\n",
                "print(f\"Hand-check pixel: Computed = {gray_res[0, 0]:.4f}, Expected = {expected:.4f}\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 2: 2D Boundary Padding\n",
                "\n",
                "Spatial filtering requires neighborhood pixels. At the image borders, we implement three padding strategies:\n",
                "- **Zero Padding**: Surrounds boundaries with 0.0 (`[0, 0 | a, b | 0, 0]`).\n",
                "- **Replicate (Clamp)**: Extends outer edge pixels outward (`[a, a | a, b | b, b]`).\n",
                "- **Reflect (Symmetric)**: Mirrors pixels across the outer edge without repeating the boundary pixel (`[b, a | a, b | b, a]`)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 4x4 matrix demonstration\n",
                "mat4x4 = np.array([\n",
                "    [10, 20, 30, 40],\n",
                "    [50, 60, 70, 80],\n",
                "    [90, 100, 110, 120],\n",
                "    [130, 140, 150, 160]\n",
                "], dtype=np.float64) / 255.0\n",
                "\n",
                "print(\"--- Zero Pad (Border 1) ---\")\n",
                "print((pad(mat4x4, 1, 1, 'zero') * 255).round().astype(int))\n",
                "print(\"\\n--- Replicate Pad (Border 1) ---\")\n",
                "print((pad(mat4x4, 1, 1, 'replicate') * 255).round().astype(int))\n",
                "print(\"\\n--- Reflect Pad (Border 1) ---\")\n",
                "print((pad(mat4x4, 1, 1, 'reflect') * 255).round().astype(int))\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 3: 2D Convolution (Loop vs Vectorized)\n",
                "\n",
                "**Convolution vs. Cross-Correlation**:\n",
                "$$\\text{Convolution: } (I * K)(y, x) = \\sum_m \\sum_n I(y - m, x - n) K(m, n)$$\n",
                "$$\\text{Cross-Correlation: } (I \\star K)(y, x) = \\sum_m \\sum_n I(y + m, x + n) K(m, n)$$\n",
                "\n",
                "Mathematical convolution requires flipping the kernel by 180° ($K_{flipped}[m, n] = K[-m, -n]$). This guarantees commutativity ($f * g = g * f$) and associativity."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Timing benchmark: Loop vs Vectorized\n",
                "k_test = BOX_BLUR_3X3\n",
                "\n",
                "t0 = time.perf_counter()\n",
                "out_loop = convolve2d_loop(cam, k_test)\n",
                "t_loop = (time.perf_counter() - t0) * 1000\n",
                "\n",
                "t0 = time.perf_counter()\n",
                "out_vec = convolve2d_vectorized(cam, k_test)\n",
                "t_vec = (time.perf_counter() - t0) * 1000\n",
                "\n",
                "print(f\"Loop Convolve Time: {t_loop:.2f} ms\")\n",
                "print(f\"Vectorized Convolve Time: {t_vec:.2f} ms\")\n",
                "print(f\"Speedup: {t_loop / t_vec:.1f}x\")\n",
                "print(f\"Max difference: {np.max(np.abs(out_loop - out_vec)):.2e}\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 4: Gaussian Blur & Separability\n",
                "\n",
                "A 2D isotropic Gaussian kernel is defined as:\n",
                "$$G(x, y; \\sigma) = \\frac{1}{2\\pi\\sigma^2} \\exp\\left(-\\frac{x^2 + y^2}{2\\sigma^2}\\right)$$\n",
                "\n",
                "Rule of thumb for kernel size: $6\\sigma$ captures 99.73% of the Gaussian area ($\\pm 3\\sigma$).\n",
                "\n",
                "**Separability**: Since $G(x, y) = G_{1D}(x) \\cdot G_{1D}(y)$, convolving horizontally with $(1, K)$ then vertically with $(K, 1)$ drops computational complexity from $O(H \\cdot W \\cdot K^2)$ to $O(H \\cdot W \\cdot 2K)$."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Separable benchmark\n",
                "sigma = 2.0\n",
                "t0 = time.perf_counter()\n",
                "res_2d = gaussian_blur(cam, sigma=sigma)\n",
                "t_2d = (time.perf_counter() - t0) * 1000\n",
                "\n",
                "t0 = time.perf_counter()\n",
                "res_sep = gaussian_blur_separable(cam, sigma=sigma)\n",
                "t_sep = (time.perf_counter() - t0) * 1000\n",
                "\n",
                "print(f\"2D Gaussian Blur Time: {t_2d:.2f} ms\")\n",
                "print(f\"Separable Gaussian Blur Time: {t_sep:.2f} ms\")\n",
                "print(f\"Speedup: {t_2d / t_sep:.1f}x\")\n",
                "print(f\"Max difference: {np.max(np.abs(res_2d - res_sep)):.2e}\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 5: Median Filter vs. Gaussian on Noise\n",
                "\n",
                "- **Impulse (Salt & Pepper)**: Outliers are extreme (0.0 or 1.0). Linear averaging blurs corrupt pixels into the surrounding area. The median filter sorts local values and picks the median, completely ignoring extreme outliers at the tails.\n",
                "- **Gaussian Noise**: Additive zero-mean i.i.d. noise. Averaging via Gaussian blur is statistically optimal by the Law of Large Numbers."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "sp_noisy = add_salt_pepper(cam, amount=0.08)\n",
                "sp_med = median_filter(sp_noisy, size=3)\n",
                "sp_gauss = gaussian_blur(sp_noisy, sigma=1.0)\n",
                "\n",
                "print(f\"Salt & Pepper Raw PSNR: {psnr(cam, sp_noisy):.2f} dB\")\n",
                "print(f\"Salt & Pepper Median Filter PSNR: {psnr(cam, sp_med):.2f} dB [WINNER]\")\n",
                "print(f\"Salt & Pepper Gaussian Blur PSNR: {psnr(cam, sp_gauss):.2f} dB\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 6: Edge Gradients with Sobel Operators\n",
                "\n",
                "Sobel approximates spatial partial derivatives:\n",
                "$$K_x = \\begin{bmatrix} -1 & 0 & 1 \\\\ -2 & 0 & 2 \\\\ -1 & 0 & 1 \\end{bmatrix}, \\quad K_y = \\begin{bmatrix} -1 & -2 & -1 \\\\ 0 & 0 & 0 \\\\ 1 & 2 & 1 \\end{bmatrix}$$\n",
                "\n",
                "Gradient magnitude is $M = \\sqrt{G_x^2 + G_y^2}$ and direction is $\\theta = \\arctan2(G_y, G_x)$."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "edges = sobel(cam)\n",
                "print(\"Sobel gradients computed successfully!\")\n",
                "print(f\"Gx range: [{edges['gx'].min():.2f}, {edges['gx'].max():.2f}]\")\n",
                "print(f\"Gy range: [{edges['gy'].min():.2f}, {edges['gy'].max():.2f}]\")\n",
                "print(f\"Magnitude max: {edges['magnitude'].max():.2f}\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 7: Histogram Equalisation\n",
                "\n",
                "From probability theory, the probability integral transform states that applying the CDF of a random variable produces a uniform distribution on $[0, 1]$. We map pixel intensity $v$ using the scaled empirical CDF:\n",
                "$$T(v) = \\frac{\\text{CDF}(v) - \\text{CDF}_{\\min}}{1.0 - \\text{CDF}_{\\min}}$$"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "low_contrast = 0.2 + 0.3 * cam\n",
                "equalized, cdf_orig, cdf_eq = equalize(low_contrast, bins=256)\n",
                "print(f\"Low Contrast Range: [{low_contrast.min():.3f}, {low_contrast.max():.3f}]\")\n",
                "print(f\"Equalized Range: [{equalized.min():.3f}, {equalized.max():.3f}]\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 8: Fourier Domain Filtering & Gibbs Ringing\n",
                "\n",
                "The **Convolution Theorem** states that:\n",
                "$$\\mathcal{F}\\{f * h\\} = \\mathcal{F}\\{f\\} \\cdot \\mathcal{F}\\{h\\}$$\n",
                "\n",
                "**Why Ideal Low-Pass Filters Ring (Gibbs Phenomenon)**:\n",
                "A sharp cylinder in the frequency domain transforms to a 2D sinc function in the spatial domain. Convolving with an oscillating sinc function creates concentric ripples (ringing) around edges. A Gaussian filter decays smoothly and monotonically without side lobes, eliminating ringing."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "F_shift, log_mag, phase = compute_dft_spectrum(cam)\n",
                "ilpf = ideal_lowpass_mask(cam.shape, cutoff_d0=25.0)\n",
                "glpf = gaussian_lowpass_mask(cam.shape, cutoff_d0=25.0)\n",
                "\n",
                "ideal_out = apply_frequency_filter(cam, ilpf)\n",
                "gauss_out = apply_frequency_filter(cam, glpf)\n",
                "\n",
                "print(\"Frequency domain filtering completed!\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 9: OpenCV Parity & Error Analysis\n",
                "\n",
                "We compare our NumPy implementations with OpenCV functions and audit numerical differences."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Run OpenCV benchmark\n",
                "from scripts.generate_results import run_opencv_comparison\n",
                "run_opencv_comparison(cam)\n"
            ]
        }
    ],
    "metadata": {
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

out_path = os.path.join(os.path.dirname(__file__), "..", "notebooks", "demo.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"Notebook generated successfully at {out_path}")
