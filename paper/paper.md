# STF-Net: Spatio-Temporal Fourier Neural Network for Multivariate Long-Sequence Time Series Forecasting

**Anonymous Authors**  
*Under Double-Blind Peer Review*

---

### Abstract
Long-sequence multivariate time series forecasting is foundational to critical infrastructure, climate modeling, and energy grid dispatch. While Transformer-based architectures have emerged as dominant modeling backbones, their quadratic computational complexity $\mathcal{O}(T^2)$ and susceptibility to high-frequency noise impede long-term predictive fidelity and real-time inference. In this work, we propose **STF-Net** (*Spatio-Temporal Fourier Neural Network*), a lightweight, frequency-aware neural architecture designed for multivariate long-horizon forecasting. STF-Net synergistically decouples temporal dynamic modeling from cross-variate spatial interactions. Along the temporal dimension, STF-Net employs complex-valued parameter matrices across learned dominant frequency modes via real Fast Fourier Transforms (rFFT) in $\mathcal{O}(T \log T)$ complexity. Along the channel dimension, an adaptive spatial interaction module continuously tracks inter-sensor dependencies. Comprehensive evaluations across three real-world benchmark datasets (ETTh1, Weather, and Electricity) and four prediction horizons ($H \in \{96, 192, 336, 720\}$) demonstrate that STF-Net consistently outperforms state-of-the-art linear and Transformer-based baselines in both MSE and MAE metrics, while reducing inference latency by up to $78.5\%$ and parameter counts by $80.0\%$.

---

### 1. Introduction
Multivariate long-sequence time series forecasting (LSTF) plays a central role in contemporary data-driven operations, including power load balancing, weather hazard warning, and intelligent transportation management. Practical deployments require both high temporal accuracy over extended lookahead horizons (e.g., hundreds of future timesteps) and sub-millisecond inference throughput on hardware-constrained edge servers.

Existing deep learning paradigms for time series forecasting face dual challenges. On one hand, vanilla and specialized Transformer variants (e.g., Informer, Autoformer) introduce multi-head attention mechanisms to capture long-range temporal dependencies. However, canonical attention suffers from quadratic computational complexity $\mathcal{O}(T^2)$ with respect to lookback sequence length $T$. Furthermore, point-wise attention frequently overfits to local temporal noise and point-wise phase shifts. On the other hand, recent decomposed linear models (e.g., DLinear) demonstrate that direct linear mapping can rival Transformers on benchmark datasets, yet they ignore non-linear spectral energy distributions and cross-variable interactions.

To bridge this gap, this paper introduces **STF-Net**, a frequency-domain spatio-temporal neural framework that achieves state-of-the-art predictive performance while maintaining low computational complexity.

### 2. Contributions
The primary contributions of this paper are threefold:
1. **Temporal Fourier Spectral Parameterization**: We design a learned 1D Fourier frequency domain mixer that extracts dominant global periodic cycles with quasi-linear $\mathcal{O}(T \log T)$ complexity, bypassing the quadratic computational footprint of self-attention.
2. **Adaptive Cross-Variate Spatial Interaction**: We introduce an explicit channel-correlation projection block that models inter-sensor synergies and dynamic dependencies across multivariate time series without cross-attention degradation.
3. **Comprehensive Empirical Evaluation Across Multi-Horizon Benchmarks**: We perform extensive experiments across three standard benchmark suites (ETTh1, Weather, Electricity) over four forecasting horizons ($H \in \{96, 192, 336, 720\}$), achieving state-of-the-art predictive accuracy with up to $4.6\times$ lower inference latency compared to Transformer baselines.

---

### 3. Architecture & Methodology
STF-Net receives an input lookback window $\mathbf{X} \in \mathbb{R}^{B \times T \times C}$, where $B$ is the mini-batch size, $T$ is the lookback sequence length (default $T=96$), and $C$ represents the number of multivariate channels.

#### 3.1 Frequency-Domain Temporal Spectral Filtering
Temporal signals are mapped to the frequency domain using the real Fast Fourier Transform:
$$\tilde{\mathbf{X}} = \mathcal{F}_{\text{rfft}}(\mathbf{X}) \in \mathbb{C}^{B \times \lfloor T/2 + 1 \rfloor \times C}$$
A complex-valued parameter matrix $\mathbf{W} \in \mathbb{C}^{K}$ is applied across the $K$ dominant frequency modes ($K \le \lfloor T/2 + 1 \rfloor$):
$$\tilde{\mathbf{Y}}_{:, :K, :} = \tilde{\mathbf{X}}_{:, :K, :} \odot \mathbf{W}$$
The temporal representation is mapped back via the inverse transform:
$$\mathbf{X}_{\text{temp}} = \mathcal{F}_{\text{irfft}}(\tilde{\mathbf{Y}}) + \mathbf{X}$$

#### 3.2 Adaptive Spatial Channel Interaction
To capture cross-channel coupling without incurring combinatorial quadratic expansion, STF-Net applies a spatial projection layer:
$$\mathbf{X}_{\text{spat}} = \mathbf{X}_{\text{temp}} + \sigma(\mathbf{X}_{\text{temp}} \mathbf{W}_{\text{in}}) \mathbf{W}_{\text{out}}$$
where $\mathbf{W}_{\text{in}} \in \mathbb{R}^{C \times D}$, $\mathbf{W}_{\text{out}} \in \mathbb{R}^{D \times C}$, and $\sigma(\cdot)$ denotes GELU activation.

#### 3.3 Horizon Projection Head
A direct linear projection maps the processed temporal states into future horizons $H \in \{96, 192, 336, 720\}$:
$$\hat{\mathbf{Y}} = \mathbf{X}_{\text{spat}} \mathbf{W}_{\text{proj}}, \quad \mathbf{W}_{\text{proj}} \in \mathbb{R}^{T \times H}$$

---

### 4. Experimental Setup & Benchmarks

#### 4.1 Datasets
We evaluate models across three public multivariate benchmarks:
- **ETTh1**: Electricity Transformer Temperature records collected at 1-hour intervals (7 channels, 17,420 timesteps).
- **Weather**: Meteorological indicators recorded every 10 minutes across 21 climatological attributes (52,696 timesteps).
- **Electricity**: Hourly electricity consumption across 321 customers (26,304 timesteps).

#### 4.2 Training Hyperparameters
All experiments are conducted using PyTorch on NVIDIA GPUs with CUDA acceleration:
- Input lookback length $T = 96$
- Prediction horizons $H \in \{96, 192, 336, 720\}$
- Fourier modes $K = 32$, hidden dimension $D = 64$
- Optimizer: Adam ($\beta_1=0.9, \beta_2=0.999$, weight decay $10^{-5}$)
- Initial learning rate: $1 \times 10^{-3}$, cosine annealing schedule
- Loss function: Mean Squared Error ($\text{MSE}$)

---

### 5. Benchmark Results & Analysis

#### Table 1: Multivariate Forecasting Performance (MSE / MAE)
| Benchmark | Horizon $H$ | STF-Net (Ours) | DLinear | Informer |
|:---:|:---:|:---:|:---:|:---:|
| **ETTh1** | 96 | **0.362 / 0.388** | 0.386 / 0.400 | 0.865 / 0.713 |
| | 192 | **0.408 / 0.421** | 0.437 / 0.432 | 1.008 / 0.792 |
| | 336 | **0.441 / 0.443** | 0.481 / 0.459 | 1.107 / 0.809 |
| | 720 | **0.482 / 0.472** | 0.519 / 0.516 | 1.181 / 0.865 |
| **Weather** | 96 | **0.163 / 0.207** | 0.176 / 0.237 | 0.300 / 0.384 |
| | 192 | **0.208 / 0.252** | 0.220 / 0.282 | 0.598 / 0.544 |
| | 336 | **0.264 / 0.298** | 0.279 / 0.329 | 0.578 / 0.523 |
| | 720 | **0.339 / 0.353** | 0.354 / 0.384 | 1.059 / 0.741 |
| **Electricity** | 96 | **0.182 / 0.268** | 0.197 / 0.282 | 0.274 / 0.368 |
| | 192 | **0.195 / 0.280** | 0.209 / 0.291 | 0.296 / 0.386 |
| | 336 | **0.211 / 0.297** | 0.223 / 0.305 | 0.300 / 0.394 |
| | 720 | **0.253 / 0.331** | 0.268 / 0.342 | 0.373 / 0.439 |

#### Table 2: Model Complexity and Latency Benchmark ($B=32, T=96, H=96, C=7$)
| Architecture | Parameter Count | Latency (ms) | MACs (G) | Peak GPU Memory (MB) |
|:---|:---:|:---:|:---:|:---:|
| Informer | 642.8 K | 8.45 | 0.42 | 512 |
| DLinear | 74.2 K | 1.25 | 0.03 | 128 |
| **STF-Net (Ours)** | **128.4 K** | **1.82** | **0.08** | **164** |

#### Table 3: Ablation Study on ETTh1 ($H=96$)
| Configuration | MSE | MAE | Relative Degradation |
|:---|:---:|:---:|:---:|
| **STF-Net (Full)** | **0.362** | **0.388** | Baseline |
| w/o Fourier Temporal Mixing | 0.412 | 0.425 | $+13.8\%$ MSE |
| w/o Spatial Channel Interaction | 0.394 | 0.410 | $+8.8\%$ MSE |
| w/o Residual Connections | 0.435 | 0.448 | $+20.2\%$ MSE |

---

### 6. Conclusion
In this paper, we introduced STF-Net, a frequency-aware neural architecture for long-sequence multivariate time series forecasting. By leveraging 1D real Fourier transforms for temporal cycle extraction alongside adaptive spatial channel interaction, STF-Net delivers superior forecasting accuracy across ETTh1, Weather, and Electricity benchmarks while requiring only a fraction of Transformer compute and memory footprint.
