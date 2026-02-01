import torch
import time
import argparse
from naive_attention import NaiveAttention

def benchmark_attention(batch, heads, seq_len, dim, device='cuda'):
    """
    Measures latency and peak VRAM for naive attention on ROCm.
    Using 'cuda' device string as ROCm maps HIP to torch.cuda.
    """
    
    q = torch.randn(batch, heads, seq_len, dim, device=device, dtype=torch.float16)
    k = torch.randn(batch, heads, seq_len, dim, device=device, dtype=torch.float16)
    v = torch.randn(batch, heads, seq_len, dim, device=device, dtype=torch.float16)
    
    model = NaiveAttention().to(device).half()
    
    # Warmup
    for _ in range(10):
        _ = model(q, k, v)
    
    torch.cuda.synchronize()
    
    # Timing
    start_time = time.time()
    iters = 100
    for _ in range(iters):
        _ = model(q, k, v)
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_latency = (end_time - start_time) / iters * 1000 # ms
    peak_memory = torch.cuda.max_memory_allocated(device) / (1024**2) # MB
    
    return avg_latency, peak_memory

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AMD ROCm Attention Benchmark")
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    
    print(f"Benchmarking Naive Attention on {torch.cuda.get_device_name(0)}")
    print(f"Config: Batch={args.batch}, Seq={args.seq_len}, Heads=12, Dim=64")
    
    try:
        latency, mem = benchmark_attention(args.batch, 12, args.seq_len, 64)
        print(f"Avg Latency: {latency:.2f} ms")
        print(f"Peak VRAM: {mem:.2f} MB")
    except torch.cuda.OutOfMemoryError:
        print("Result: OOM (Out of Memory)")
