import os
import subprocess
import argparse

def run_experiment(seq_len, batch_size):
    print(f"--- Running Experiment: Seq={seq_len}, Batch={batch_size} ---")
    
    # Run Baseline
    print("Running Baseline Benchmark...")
    subprocess.run([
        "python", "baselines/benchmark.py", 
        "--seq_len", str(seq_len), 
        "--batch", str(batch_size)
    ])
    
    # Run Prototype (when ready)
    print("\nRunning Tiled Prototype...")
    # subprocess.run(["python", "experiments/forward_attention_fp16.py"])
    print("TODO: Integrate tiled prototype results collection.")
    
    print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Run a sweep of experiments")
    args = parser.parse_args()
    
    if args.all:
        configs = [
            (1024, 8),
            (2048, 4),
            (4096, 2),
            (8192, 1)
        ]
        for seq, b in configs:
            run_experiment(seq, b)
    else:
        run_experiment(1024, 8)
