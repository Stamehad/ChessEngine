import argparse
import os
import time
from typing import List, Tuple

import torch

from chessengine.rl.tests.test_self_play_pipeline import (
    DummyModel,
    load_real_model,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Profile model forward pass.")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device string understood by torch.device (e.g. cpu, cuda, mps).",
    )
    parser.add_argument(
        "--use-real-model",
        action="store_true",
        help="Load the real ChessLightningModule checkpoint instead of the dummy model.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Wrap the model with torch.compile before profiling.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of timing iterations per batch size.",
    )
    parser.add_argument(
        "--max-batch",
        type=int,
        default=10_000,
        help="Largest batch size (multiple of 1k) to profile.",
    )
    return parser.parse_args()


def prepare_model(device: torch.device, use_real: bool, compile_model: bool):
    if use_real:
        model = load_real_model(device)
    else:
        model = DummyModel().to(device)
    model.eval()
    if compile_model:
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is not available in this build of PyTorch.")
        model = torch.compile(model, mode="default")
    return model


def benchmark_forward(
    model: torch.nn.Module,
    batch_sizes: List[int],
    device: torch.device,
    n_iters: int,
) -> List[Tuple[int, float, float]]:
    results = []
    for bs in batch_sizes:
        features = torch.randn(bs, 8, 8, 21, device=device, dtype=torch.float32)
        with torch.inference_mode():
            model(features)
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        with torch.inference_mode():
            for _ in range(n_iters):
                model(features)
            if device.type == "cuda":
                torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / n_iters
        throughput = bs / elapsed if elapsed > 0 else float("inf")
        results.append((bs, elapsed, throughput))
    return results


def main():
    args = parse_args()
    device = torch.device(args.device)
    batch_sizes = list(range(1000, args.max_batch + 1, 1000))
    batch_sizes = list(range(1, args.max_batch + 1, 10))  # --- IGNORE ---
    model = prepare_model(device, args.use_real_model, args.compile)
    results = benchmark_forward(model, batch_sizes, device, args.iterations)

    header = f"{'Batch':>8} | {'Latency (s)':>12} | {'Throughput (samples/s)':>24}"
    print(header)
    print("-" * len(header))
    for bs, latency, throughput in results:
        print(f"{bs:8d} | {latency:12.6f} | {throughput:24.2f}")


if __name__ == "__main__":
    main()
