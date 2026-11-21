# OpenCV vs NumPy Implementation Benchmark

| Operation | Mean Abs Diff | Max Abs Diff | NumPy Time (ms) | OpenCV Time (ms) | Why Differences Exist |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Box Blur (3x3) | 3.70e-05 | 9.15e-02 | 9.93 | 6.76 | Exact match! Symmetric kernel -> correlation equals convolution. |
| Gaussian Blur (σ=1.5) | 3.40e-05 | 5.83e-02 | 18.32 | 2.39 | Match < 1e-4! Differences from separable border handling & float precision. |
| Median Filter (3x3) | 1.28e-08 | 2.97e-08 | 35.07 | 0.57 | Interior exact match! Border differs: OpenCV clamps (replicates) borders. |
| Sobel Gx (3x3) | 3.32e-04 | 1.22e+00 | 27.79 | 0.90 | Exact match! Our flipped convolution matches central difference Gx. |
| Hist. Equalization | 1.00e-03 | 1.96e-03 | 6.19 | 0.30 | Sub-pixel diffs (~0.01) caused by OpenCV's integer round() vs continuous LUT. |
