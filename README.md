# STF-Net: Spatio-Temporal Fourier Neural Network for Multivariate Long-Sequence Time Series Forecasting

Official benchmark codebase for STF-Net, a high-efficiency spatio-temporal Fourier neural network tailored for multivariate long-sequence time series forecasting.

## Architecture Highlights
- **Temporal Fourier Mixing**: Applies 1D spectral parameterization across the frequency domain with $O(T \log T)$ complexity.
- **Adaptive Spatial Channel Interaction**: Disentangles and models inter-variate sensor correlations without quadratic self-attention blowup.
- **Direct Horizon Projection**: Multi-horizon forecasting ($H \in \{96, 192, 336, 720\}$).

## Requirements & Environment
- Python >= 3.8
- PyTorch >= 2.0.0 (with CUDA support)
- NVIDIA GPU with >= 8GB VRAM
- Unix/Linux or macOS environment

```bash
pip install -r requirements.txt
pip install -e .
```

## Dataset Acquisition
Download the standard multivariate forecasting benchmarks (ETTh1, Weather, Electricity):
```bash
bash scripts/download_data.sh ./data
```

## Reproduction Pipeline
To run the full end-to-end training and evaluation pipeline:
```bash
bash scripts/reproduce_all.sh
```

Or execute training individually:
```bash
# Train STF-Net on ETTh1 (horizon 96)
python scripts/train.py --dataset ETTh1 --seq_len 96 --pred_len 96 --epochs 10

# Train STF-Net on Weather (horizon 192)
python scripts/train.py --dataset Weather --seq_len 96 --pred_len 192 --epochs 10

# Benchmark inference latency and computational overhead
python scripts/benchmark_efficiency.py

# Verify reproduction consistency against paper references
python scripts/verify_reproduction.py
```

## Benchmark Summary
| Benchmark | Lookback | Horizon | STF-Net MSE | STF-Net MAE | DLinear MSE | DLinear MAE |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ETTh1 | 96 | 96 | **0.362** | **0.388** | 0.386 | 0.400 |
| ETTh1 | 96 | 192 | **0.408** | **0.421** | 0.437 | 0.432 |
| ETTh1 | 96 | 336 | **0.441** | **0.443** | 0.481 | 0.459 |
| ETTh1 | 96 | 720 | **0.482** | **0.472** | 0.519 | 0.516 |
| Weather | 96 | 96 | **0.163** | **0.207** | 0.176 | 0.237 |
| Weather | 96 | 192 | **0.208** | **0.252** | 0.220 | 0.282 |
| Electricity | 96 | 96 | **0.182** | **0.268** | 0.197 | 0.282 |
| Electricity | 96 | 720 | **0.253** | **0.331** | 0.268 | 0.342 |
