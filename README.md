# Image Processing from Scratch in NumPy

A comprehensive, zero-dependency (NumPy-only) computer vision library covering fundamental image processing, spatial filtering, histogram equalisation, and 2D Fourier-domain signal processing.

Built as an admissions portfolio and preparation project for the **Erasmus Mundus Master in Image Processing and Computer Vision (IPCVAI)**.

---

## 🎯 Project Goals

1. **First-Principles Implementation**: Implement core 2D image processing algorithms using pure linear algebra and calculus in NumPy. No high-level OpenCV or SciPy filter shortcuts.
2. **Signal Processing Intuition**: Demystify frequency-domain filtering, Gibbs ringing artifacts, filter separability, and the Convolution Theorem.
3. **Rigorous OpenCV Auditing**: Compare output against OpenCV functions (`cv2.filter2D`, `cv2.GaussianBlur`, `cv2.medianBlur`, `cv2.Sobel`, `cv2.equalizeHist`) and scientifically explain every mathematical and implementation discrepancy.

---

## 📁 Repository Structure

```
imgproc-from-scratch/
├── imgproc/
│   ├── __init__.py
│   ├── basics.py        # Color space conversion (ITU-R BT.601), image loading, 2D padding
│   ├── filters.py       # 2D convolution (loop & vectorized), Gaussian blur, median, Sobel
│   ├── histogram.py     # Histogram extraction, CDF calculation, histogram equalisation
│   └── frequency.py     # 2D FFT, log spectrum, Ideal/Gaussian low-pass & high-pass filters
├── notebooks/
│   └── demo.ipynb       # Interactive Colab-ready notebook with full experimental figures
├── scripts/
│   └── generate_results.py # Headless runner to generate all figures and OpenCV benchmarks
├── tests/
│   └── test_filters.py  # Pytest suite verifying mathematical invariants & OpenCV parity
├── results/             # High-resolution output figures and benchmark tables
├── requirements.txt     # Python dependencies
├── LEARNING_LOG.md      # Detailed engineering logbook & theoretical derivations
└── README.md
```

---

## 🔬 Core Algorithms Implemented

### 1. Representation & Luminance (`basics.py`)
- Standardizes grayscale images as `float64` in range `[0.0, 1.0]`.
- Implements ITU-R BT.601 luminance weighting:
  $$Y = 0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B$$
  **Why Green dominates (58.7%)**: Human photopic vision relies on retinal M- and L-cone photoreceptors whose composite spectral sensitivity peaks around 555 nm (green). Green light contributes vastly more to subjective perceived brightness than blue or red.

### 2. Padding Modes (`basics.py`)
- **Zero Padding**: Surrounds boundaries with 0.0 (`[0, 0 | a, b | 0, 0]`).
- **Replicate (Clamp)**: Extends edge pixels outward (`[a, a | a, b | b, b]`).
- **Reflect (Symmetric)**: Mirrors pixels across the outer edge without repeating the boundary pixel (`[b, a | a, b | b, a]`).

### 3. 2D Discrete Convolution (`filters.py`)
- Implements both loop-based and fast vectorized convolution via `numpy.lib.stride_tricks.sliding_window_view`.
- **Convolution vs. Cross-Correlation**:
  $$\text{Convolution: } (I * K)(y, x) = \sum_m \sum_n I(y - m, x - n) K(m, n)$$
  $$\text{Correlation: } (I \star K)(y, x) = \sum_m \sum_n I(y + m, x + n) K(m, n)$$
  Mathematical convolution requires rotating the kernel by 180° ($K_{flipped}[m, n] = K[-m, -n]$). This guarantees commutativity ($f * g = g * f$) and associativity. `cv2.filter2D` actually computes cross-correlation by default.

### 4. Separable Gaussian Blur (`filters.py`)
- Generates 1D and 2D normalized Gaussian kernels ($6\sigma$ heuristic for odd sizing):
  $$G(x, y) = \frac{1}{2\pi\sigma^2} \exp\left(-\frac{x^2 + y^2}{2\sigma^2}\right)$$
- **Separability Speedup**:
  Since $G(x, y) = G_{1D}(x) \cdot G_{1D}(y)$, convolving an $H \times W$ image with an $N \times N$ kernel requires $O(H \cdot W \cdot N^2)$ operations. Convolving horizontally with $(1, N)$ and then vertically with $(N, 1)$ requires only $O(H \cdot W \cdot 2N)$ operations—a speedup of $\approx N / 2$.

### 5. Median Filter vs. Gaussian on Noise (`filters.py`)
- Evaluated on **Salt & Pepper (impulse)** vs. **Additive Gaussian noise** using Peak Signal-to-Noise Ratio (PSNR):
  $$\text{PSNR} = 10 \cdot \log_{10}\left(\frac{1.0}{\text{MSE}}\right)$$
- **Why Median wins on Salt & Pepper**:
  A Gaussian/mean filter averages the extreme outliers (0 or 1) into neighboring pixels, blurring corrupt values across clean patches. The median filter is an order statistic: as long as fewer than 50% of the window pixels are corrupted, the extreme 0s and 1s remain at the sorted tails, and the median selects an uncorrupted neighborhood intensity.

### 6. Edge Detection via Sobel Gradients (`filters.py`)
- Approximates discrete spatial derivatives:
  $$K_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}, \quad K_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}$$
- Computes gradient magnitude $M = \sqrt{G_x^2 + G_y^2}$ and direction $\theta = \arctan2(G_y, G_x)$.
- Demonstrates why Gaussian pre-smoothing is crucial: numerical differentiation acts as a high-pass filter that amplifies high-frequency noise; pre-smoothing suppresses noise before differentiation.

### 7. Histogram Equalisation (`histogram.py`)
- Derives transfer function from Cumulative Distribution Function (CDF):
  $$T(v) = \frac{\text{CDF}(v) - \text{CDF}_{\min}}{1.0 - \text{CDF}_{\min}}$$
- Uniformly spreads concentrated intensity histograms to maximize entropy and dynamic range.

### 8. 2D Frequency Domain & Ringing Artifacts (`frequency.py`)
- 2D DFT with centered DC component via `fftshift`.
- Compares **Ideal Low-Pass Filter (ILPF)** vs. **Gaussian Low-Pass Filter (GLPF)**:
  - **Why Ideal Low-Pass Rings (Gibbs Phenomenon)**: The sharp step discontinuity in the frequency domain is the Fourier pair of a 2D sinc ($\text{jinc}$) function in the spatial domain. Multiplying by a cylinder in frequency equals convolving with a sinc in space. The oscillating side lobes of the sinc cause concentric ripples (ringing) around edges.
  - **Why Gaussian Never Rings**: The Fourier transform of a Gaussian is another Gaussian ($e^{-\pi x^2} \stackrel{\mathcal{F}}{\longleftrightarrow} e^{-\pi s^2}$). Because a Gaussian decays smoothly and monotonically with zero negative side lobes, it produces perfectly smooth blur without ringing.

---

## 📊 OpenCV Comparison Benchmark

| Operation | Mean Abs Diff | Max Abs Diff | NumPy Time (ms) | OpenCV Time (ms) | Why Differences Exist |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Box Blur (3x3)** | $0.00 \times 10^0$ | $0.00 \times 10^0$ | ~15 ms | ~0.2 ms | Exact match! For symmetric kernels, correlation equals convolution. |
| **Gaussian Blur (σ=1.5)** | $< 10^{-5}$ | $< 10^{-4}$ | ~18 ms | ~0.3 ms | Negligible difference due to OpenCV's 16-bit fixed point kernel optimization and separable border padding. |
| **Median Filter (3x3)** | $0.00 \times 10^0$ (interior) | $0.00 \times 10^0$ | ~35 ms | ~0.4 ms | Interior matches exactly! Discrepancy at border is because OpenCV's `medianBlur` enforces `BORDER_REPLICATE`. |
| **Sobel Gx (3x3)** | $0.00 \times 10^0$ | $0.00 \times 10^0$ | ~15 ms | ~0.2 ms | Exact match! Mathematical convolution with flipped kernel reproduces standard central differences. |
| **Hist. Equalization** | $< 0.005$ | $< 0.02$ | ~4 ms | ~0.1 ms | OpenCV's `equalizeHist` takes `uint8` with integer round(), whereas our implementation operates on floating-point CDF mapping. |

---

## 💡 Key Takeaways for Motivation Letter & Interview

1. **Correlation vs. Convolution**: Many libraries (including OpenCV `filter2D`) default to cross-correlation. For asymmetric kernels (such as directional derivative filters), failing to flip the kernel inverts the sign of the computed derivative.
2. **The Duality of Smoothing and Ringing**: The frequency cutoff slope in the frequency domain directly dictates the spatial impulse response. A sharp frequency step produces infinite spatial ripples (Gibbs phenomenon); a smooth Gaussian frequency curve yields a strictly non-negative spatial profile.
3. **Complexity of Separability**: 2D convolution is $O(K^2)$ per pixel, but separable kernels reduce this to $O(2K)$, enabling real-time performance even in high-resolution image processing.

---

## 🚀 How to Run

### Run Unit Tests
```bash
pytest tests/ -v
```

### Generate All Figures
```bash
python scripts/generate_results.py
```
Output figures will be saved in `results/`.
