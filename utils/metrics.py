import numpy as np
import torch

def MSE(pred, true):
    return np.mean((pred - true) ** 2)

def MAE(pred, true):
    return np.mean(np.abs(pred - true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / (true + 1e-5)))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / (true + 1e-5)))

def compute_metrics(pred, true):
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(true, torch.Tensor):
        true = true.detach().cpu().numpy()

    mse = float(MSE(pred, true))
    mae = float(MAE(pred, true))
    mape = float(MAPE(pred, true))
    mspe = float(MSPE(pred, true))

    return {
        "MSE": round(mse, 4),
        "MAE": round(mae, 4),
        "MAPE": round(mape, 4),
        "MSPE": round(mspe, 4)
    }
