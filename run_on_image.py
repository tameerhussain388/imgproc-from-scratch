"""
Interactive Image Processing Playground
=======================================
Run this script on ANY image (PNG, JPG, BMP) or with sample images
to see all algorithms in action with visual before-and-after plots.

Usage:
------
    # Run with standard sample image (camera):
    python run_on_image.py

    # Run with sample coins image:
    python run_on_image.py --sample coins

    # Run with your own photo:
    python run_on_image.py "path/to/my_photo.jpg"
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from skimage import data

# Ensure imgproc package is discoverable
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from imgproc.basics import load_image, pad
from imgproc.filters import (
    BOX_BLUR_3X3,
    SHARPEN_3X3,
    convolve2d_vectorized,
    gaussian_blur,
    add_salt_pepper,
    add_gaussian_noise,
    median_filter,
    psnr,
    sobel,
    threshold
)
from imgproc.histogram import compute_histogram, compute_cdf, equalize
from imgproc.frequency import (
    compute_dft_spectrum,
    ideal_lowpass_mask,
    gaussian_lowpass_mask,
    apply_frequency_filter
)


def process_image(img: np.ndarray, output_dir: str = "custom_results", image_name: str = "sample"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[1] Image Loaded: shape={img.shape}, dtype={img.dtype}, range=[{img.min():.3f}, {img.max():.3f}]")

    # -------------------------------------------------------------
    # 1. Spatial Filtering (Blur, Sharpen)
    # -------------------------------------------------------------
    print("[2] Running 2D Convolutions (Box Blur & Sharpening)...")
    box_blur = convolve2d_vectorized(img, BOX_BLUR_3X3)
    sharpen = np.clip(convolve2d_vectorized(img, SHARPEN_3X3), 0.0, 1.0)
    gauss_blur = gaussian_blur(img, sigma=2.0)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Original Grayscale")
    axes[1].imshow(box_blur, cmap="gray")
    axes[1].set_title("Box Blur (Averaging)")
    axes[2].imshow(sharpen, cmap="gray")
    axes[2].set_title("Sharpen (Edge Boost)")
    axes[3].imshow(gauss_blur, cmap="gray")
    axes[3].set_title("Gaussian Blur (σ=2.0)")
    for ax in axes: ax.axis("off")
    plt.tight_layout()
    p1 = os.path.join(output_dir, f"{image_name}_1_filters.png")
    plt.savefig(p1, dpi=200)
    plt.close()
    print(f"    -> Saved: {p1}")

    # -------------------------------------------------------------
    # 2. Noise & Denoising Comparison
    # -------------------------------------------------------------
    print("[3] Testing Noise & Denoising (Median vs Gaussian)...")
    sp_noisy = add_salt_pepper(img, amount=0.06)
    sp_cleaned_med = median_filter(sp_noisy, size=3)
    sp_cleaned_gauss = gaussian_blur(sp_noisy, sigma=1.0)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Original Clean")
    axes[1].imshow(sp_noisy, cmap="gray")
    axes[1].set_title(f"Salt & Pepper ({psnr(img, sp_noisy):.1f} dB)")
    axes[2].imshow(sp_cleaned_med, cmap="gray")
    axes[2].set_title(f"Median Filter ({psnr(img, sp_cleaned_med):.1f} dB) [WINNER]")
    axes[3].imshow(sp_cleaned_gauss, cmap="gray")
    axes[3].set_title(f"Gaussian Blur ({psnr(img, sp_cleaned_gauss):.1f} dB)")
    for ax in axes: ax.axis("off")
    plt.tight_layout()
    p2 = os.path.join(output_dir, f"{image_name}_2_denoising.png")
    plt.savefig(p2, dpi=200)
    plt.close()
    print(f"    -> Saved: {p2}")

    # -------------------------------------------------------------
    # 3. Edge Detection (Sobel)
    # -------------------------------------------------------------
    print("[4] Detecting Edges with Sobel Operator...")
    edges = sobel(img)
    edge_map = threshold(edges["magnitude"], thresh=0.25)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Original")
    axes[1].imshow(edges["gx"], cmap="coolwarm")
    axes[1].set_title("Gx (Vertical Edges)")
    axes[2].imshow(edges["gy"], cmap="coolwarm")
    axes[2].set_title("Gy (Horizontal Edges)")
    axes[3].imshow(edge_map, cmap="gray")
    axes[3].set_title("Edge Map (Magnitude > 0.25)")
    for ax in axes: ax.axis("off")
    plt.tight_layout()
    p3 = os.path.join(output_dir, f"{image_name}_3_edges.png")
    plt.savefig(p3, dpi=200)
    plt.close()
    print(f"    -> Saved: {p3}")

    # -------------------------------------------------------------
    # 4. Histogram Equalization (Contrast Boost)
    # -------------------------------------------------------------
    print("[5] Enhancing Contrast with Histogram Equalization...")
    # Simulate low-contrast condition
    low_contrast = 0.3 + 0.4 * img
    eq_img, cdf_orig, cdf_eq = equalize(low_contrast, bins=256)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].imshow(low_contrast, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("Low Contrast (Before)")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(eq_img, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("Equalized Contrast (After)")
    axes[0, 1].axis("off")

    counts_orig, bins_orig = compute_histogram(low_contrast, 256)
    counts_eq, bins_eq = compute_histogram(eq_img, 256)

    axes[1, 0].plot(bins_orig, counts_orig, color="navy")
    axes[1, 0].set_title("Original Histogram (Narrow peak)")
    axes[1, 0].set_xlim([0, 1])

    axes[1, 1].plot(bins_eq, counts_eq, color="navy")
    axes[1, 1].set_title("Equalized Histogram (Spread evenly)")
    axes[1, 1].set_xlim([0, 1])

    plt.tight_layout()
    p4 = os.path.join(output_dir, f"{image_name}_4_histogram.png")
    plt.savefig(p4, dpi=200)
    plt.close()
    print(f"    -> Saved: {p4}")

    # -------------------------------------------------------------
    # 5. Fourier Frequency Filtering
    # -------------------------------------------------------------
    print("[6] Transforming to 2D Fourier Frequency Domain...")
    _, log_mag, _ = compute_dft_spectrum(img)
    cutoff = min(img.shape) * 0.1  # 10% cutoff
    ilpf = ideal_lowpass_mask(img.shape, cutoff)
    glpf = gaussian_lowpass_mask(img.shape, cutoff)
    ideal_filtered = apply_frequency_filter(img, ilpf)
    gauss_filtered = apply_frequency_filter(img, glpf)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Original")
    axes[1].imshow(log_mag, cmap="magma")
    axes[1].set_title("Log Frequency Spectrum")
    axes[2].imshow(ideal_filtered, cmap="gray")
    axes[2].set_title(f"Ideal Low-Pass (D0={cutoff:.0f})\n[Has Ringing!]")
    axes[3].imshow(gauss_filtered, cmap="gray")
    axes[3].set_title(f"Gaussian Low-Pass (D0={cutoff:.0f})\n[Clean, No Ringing]")
    for ax in axes: ax.axis("off")
    plt.tight_layout()
    p5 = os.path.join(output_dir, f"{image_name}_5_frequency.png")
    plt.savefig(p5, dpi=200)
    plt.close()
    print(f"    -> Saved: {p5}")

    print(f"\n Done! All 5 comparison figures are ready in '{output_dir}/'!\n")


def main():
    parser = argparse.ArgumentParser(description="Run image processing from scratch on any image or folder.")
    parser.add_argument("image_path", nargs="?", default=None, help="Path to your custom image (JPG/PNG) or folder of images.")
    parser.add_argument("--sample", choices=["camera", "coins"], default="camera", help="Built-in skimage sample image.")
    parser.add_argument("--outdir", default="custom_results", help="Folder where result figures will be saved.")
    parser.add_argument("--max-dim", type=int, default=640, help="Max image dimension for fast processing (default 640).")
    args = parser.parse_args()

    if args.image_path:
        if os.path.isdir(args.image_path):
            valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
            files = [os.path.join(args.image_path, f) for f in os.listdir(args.image_path)
                     if os.path.splitext(f)[1].lower() in valid_exts]
            if not files:
                print(f"No image files found in '{args.image_path}'.")
                sys.exit(1)
            print(f"Found {len(files)} images in '{args.image_path}'. Processing all...")
            for idx, filepath in enumerate(files, 1):
                name = os.path.splitext(os.path.basename(filepath))[0]
                print(f"\n[{idx}/{len(files)}] Processing: {os.path.basename(filepath)}")
                img = load_image(filepath, max_dim=args.max_dim)
                process_image(img, output_dir=args.outdir, image_name=name)
            print(f"\nAll {len(files)} images processed! Check the '{args.outdir}/' folder.")
            return

        elif os.path.isfile(args.image_path):
            name = os.path.splitext(os.path.basename(args.image_path))[0]
            print(f"Loading custom image from: {args.image_path}")
            img = load_image(args.image_path, max_dim=args.max_dim)
            process_image(img, output_dir=args.outdir, image_name=name)
        else:
            print(f"Error: Path not found at '{args.image_path}'")
            sys.exit(1)
    else:
        name = args.sample
        print(f"Using standard benchmark image: skimage.data.{args.sample}()")
        if args.sample == "camera":
            img = data.camera().astype(np.float64) / 255.0
        else:
            img = data.coins().astype(np.float64) / 255.0

        process_image(img, output_dir=args.outdir, image_name=name)


if __name__ == "__main__":
    main()
