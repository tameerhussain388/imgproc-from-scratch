# Engineering & Learning Log: Image Processing from Scratch

This log documents step-by-step mathematical reasoning, algorithm designs, debugging insights, and interview talking points for the IPCVAI Erasmus Mundus application.

---

## Step 1: Images as Arrays & Color Spaces

### 1. Mathematical Representation
An RGB image is a 3-tensor $\mathbf{I} \in [0, 255]^{H \times W \times 3}$. In numerical processing, we map each element to $\hat{\mathbf{I}} \in [0.0, 1.0]^{H \times W \times 3}$ using:
$$\hat{\mathbf{I}}_{i,j,c} = \frac{\mathbf{I}_{i,j,c}}{255.0}$$

### 2. ITU-R BT.601 Luminance Derivation
The transformation to perceived luminance $Y$ is:
$$Y = 0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B$$
- **Human Eye Cone Sensitivity**: The human retina contains S, M, and L cones. Photopic spectral luminous efficiency $V(\lambda)$ peaks at 555 nm (in the green-yellow spectrum).
- **Consequence**: The eye is nearly twice as sensitive to green as to red, and more than five times as sensitive to green as to blue. Hence, $w_G = 0.587$ carries more than half the luminance information.

---

## Step 2: Boundary Padding Modes

When convolving with a kernel of shape $(K_h, K_w)$, boundary pixels lack full neighborhoods. We add $P_h = \lfloor K_h / 2 \rfloor$ and $P_w = \lfloor K_w / 2 \rfloor$ borders.

- **Zero Padding**:
  $$\mathbf{I}_{\text{pad}}[r, c] = 0 \quad \text{if } r \notin [P_h, H+P_h-1] \text{ or } c \notin [P_w, W+P_w-1]$$
  *Caveat*: Introduces artificial dark borders that create false edge responses.
- **Replicate (Clamp)**:
  $$\mathbf{I}_{\text{pad}}[r, c] = \mathbf{I}[\text{clip}(r - P_h, 0, H-1), \text{clip}(c - P_w, 0, W-1)]$$
  *Caveat*: Replicates outer pixels. Eliminates artificial dark edges.
- **Reflect (Symmetric)**:
  Mirrors across edge boundaries without duplicating the outer pixel:
  `[c, b | a, b, c, d | c, b]`

---

## Step 3: 2D Convolution vs. Cross-Correlation

### 1. Mathematical Definition
$$\text{Convolution: } (f * g)[y, x] = \sum_{m=-\infty}^\infty \sum_{n=-\infty}^\infty f[y - m, x - n] g[m, n]$$
$$\text{Cross-Correlation: } (f \star g)[y, x] = \sum_{m=-\infty}^\infty \sum_{n=-\infty}^\infty f[y + m, x + n] g[m, n]$$

Let $m' = -m$ and $n' = -n$. Then:
$$(f * g)[y, x] = \sum_{m'} \sum_{n'} f[y + m', x + n'] g[-m', -n']$$
**Conclusion**: Convolution is cross-correlation with the kernel rotated by 180° ($g[-m, -n]$).

### 2. Why Flip the Kernel?
- **Associativity**: $(f * g) * h = f * (g * h)$. Crucial for cascade filtering (e.g. smoothing then derivative). Cross-correlation is **not** associative!
- **Commutativity**: $f * g = g * f$. It doesn't matter whether the image slides over the kernel or the kernel slides over the image. Cross-correlation is **not** commutative!

### 3. Implementation Optimization
Explicit Python nested loops (`for r in range(H): for c in range(W):`) require $O(H \cdot W)$ Python interpreter jumps, running in hundreds of milliseconds.
By using `numpy.lib.stride_tricks.sliding_window_view`, we construct an $(H, W, K_h, K_w)$ strided view with **zero memory copies**, followed by `np.tensordot` in optimized C/BLAS, speeding up convolution by over $15\times$.

---

## Step 4: Gaussian Blur & Separability

### 1. The Gaussian Distribution
$$G(x, y; \sigma) = \frac{1}{2\pi\sigma^2} \exp\left(-\frac{x^2 + y^2}{2\sigma^2}\right)$$
We truncate the infinite Gaussian at $\pm 3\sigma$ ($6\sigma$ total window size), which contains $99.73\%$ of total probability mass.

### 2. Separability Theorem
Since $e^{-\frac{x^2+y^2}{2\sigma^2}} = e^{-\frac{x^2}{2\sigma^2}} \cdot e^{-\frac{y^2}{2\sigma^2}}$:
$$G(x, y; \sigma) = G_{1D}(x; \sigma) \cdot G_{1D}(y; \sigma)$$
Therefore:
$$\mathbf{I} * G_{2D} = (\mathbf{I} * G_{1D,\text{horiz}}) * G_{1D,\text{vert}}$$

**Computational Complexity Comparison**:
- Standard 2D convolution: $H \times W \times K^2$ multiplications.
- Separable 1D cascade: $H \times W \times 2K$ multiplications.
- For a $9 \times 9$ kernel ($K=9$), the theoretical speedup is:
  $$\frac{K^2}{2K} = \frac{K}{2} = 4.5\times$$

---

## Step 5: Median Filter vs. Gaussian on Noise

### 1. Impulse Noise (Salt-and-Pepper)
- Corrupts pixels to extreme values: $0.0$ (pepper) or $1.0$ (salt).
- **Linear Filter Failure (Gaussian/Box)**:
  $$\hat{I} = \frac{1}{9}\sum_{i=1}^9 I_i$$
  Even one bad pixel ($1.0$) inside a clean dark area ($0.0$) gets spread out to $\approx 0.11$, blurring the artifact into the neighborhood.
- **Non-Linear Filter Success (Median)**:
  The median sorts the values: $[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]$. The 5th value (median) is $0.0$. The impulse noise is completely rejected!

### 2. Gaussian Noise
- Additive i.i.d. zero-mean noise $\mathcal{N}(0, \sigma^2)$.
- By the Central Limit Theorem and Law of Large Numbers, averaging $N$ independent noisy samples reduces noise variance by a factor of $N$:
  $$\text{Var}\left(\frac{1}{N}\sum X_i\right) = \frac{\sigma^2}{N}$$
  Therefore, Gaussian blur (a weighted linear average) is statistically optimal for Gaussian noise, outperforming median filtering.

---

## Step 6: Edge Detection with the Sobel Operator

### 1. Discrete Derivative Approximations
The continuous image gradient is $\nabla \mathbf{I} = \left[\frac{\partial \mathbf{I}}{\partial x}, \frac{\partial \mathbf{I}}{\partial y}\right]^T$.
Sobel approximates this using central finite differences combined with orthogonal triangular smoothing $[1, 2, 1]$:
$$K_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix} = \begin{bmatrix} 1 \\ 2 \\ 1 \end{bmatrix} * \begin{bmatrix} -1 & 0 & 1 \end{bmatrix}$$
- Smoothing suppresses high-frequency noise before differentiation.
- Magnitude: $\|\nabla \mathbf{I}\| = \sqrt{G_x^2 + G_y^2}$
- Direction: $\theta = \arctan2(G_y, G_x)$

---

## Step 7: Histogram Equalisation

### 1. Probability Integral Transform
Let $X$ be a continuous random variable with probability density function $p_X(x)$ and cumulative distribution function $F_X(x) = \int_{-\infty}^x p_X(t) dt$.
Let $Y = F_X(X)$. The CDF of $Y$ is:
$$F_Y(y) = P(Y \le y) = P(F_X(X) \le y) = P(X \le F_X^{-1}(y)) = F_X(F_X^{-1}(y)) = y$$
Since $F_Y(y) = y$, $Y$ is **uniformly distributed on $[0, 1]$**!
Histogram equalisation is the discrete empirical implementation of this fundamental theorem.

---

## Step 8: Frequency Domain & Gibbs Ringing

### 1. 2D Discrete Fourier Transform (DFT)
$$F(u, v) = \sum_{x=0}^{H-1} \sum_{y=0}^{W-1} f(x, y) e^{-j 2\pi \left(\frac{ux}{H} + \frac{vy}{W}\right)}$$

### 2. The Convolution Theorem
$$\mathcal{F}\{f * h\} = \mathcal{F}\{f\} \cdot \mathcal{F}\{h\}$$
Filtering in the spatial domain equals point-wise multiplication in the frequency domain.

### 3. Why the Ideal Low-Pass Filter Rings
- In the frequency domain, the Ideal Low-Pass Filter is a step cylinder $H_{\text{ideal}}(u, v) = \text{rect}(D/2D_0)$.
- By the Fourier transform pair:
  $$\mathcal{F}^{-1}\{\text{rect}(r)\} \propto \frac{J_1(2\pi \rho)}{\rho} \quad (\text{the 2D Airy/Sinc pattern})$$
- In the spatial domain, the filtered image is convolved with a sinc-like radial function containing alternating positive and negative ripples.
- Near high-contrast step edges, these ripples produce visible concentric waves known as **Gibbs ringing**.
- In contrast, the Fourier transform of a Gaussian is another Gaussian ($e^{-\alpha r^2} \stackrel{\mathcal{F}}{\longleftrightarrow} e^{-\frac{\omega^2}{4\alpha}}$). Because a Gaussian is strictly positive and monotonically decreasing, it has zero side lobes and generates zero ringing.
