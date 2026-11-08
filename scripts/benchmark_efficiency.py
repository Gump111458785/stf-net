import time
import torch
from models.stf_net import STFNet
from models.baselines import DLinear, SimpleInformer

def benchmark_model(model_name, model, x, iters=100):
    model.eval()
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(iters):
            _ = model(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_time = (time.time() - start) * 1000 / iters

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_time, num_params

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU device is required for latency benchmarking.")
    device = torch.device("cuda")

    B, T, C = 32, 96, 7
    pred_len = 96
    x = torch.randn(B, T, C, device=device)

    models = {
        "STF-Net": STFNet(seq_len=T, pred_len=pred_len, num_channels=C).to(device),
        "DLinear": DLinear(seq_len=T, pred_len=pred_len, num_channels=C).to(device),
        "Informer": SimpleInformer(seq_len=T, pred_len=pred_len, num_channels=C).to(device)
    }

    print("=" * 60)
    print(f"{'Model':<15} | {'Params (K)':<12} | {'Latency (ms)':<15}")
    print("-" * 60)
    for name, m in models.items():
        lat, params = benchmark_model(name, m, x)
        print(f"{name:<15} | {params / 1e3:<12.2f} | {lat:<15.2f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
