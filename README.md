# FlashAttention-AMD-RDNA2

A research-oriented prototype for enabling memory-efficient attention (FlashAttention-style) on AMD RDNA2 GPUs (specifically Radeon RX 6800) using ROCm.

> [!IMPORTANT]
> This project is currently a **Work In Progress (WIP)**. It does not yet provide a full FlashAttention implementation. The goal is to benchmark, study, and prototype memory-efficient attention mechanisms on AMD hardware.

## Project Goal
The primary objective is to explore the feasibility and performance of online softmax and tiling-based attention on AMD's RDNA2 architecture. This involves:
- Benchmarking standard PyTorch attention on ROCm.
- Documenting RDNA2-specific hardware constraints (LDS size, CU architecture).
- Implementing prototype kernels using HIP or OpenAI Triton (future).

## Project Structure
- `baselines/`: PyTorch-native attention implementations and benchmark scripts.
- `experiments/`: Proof-of-concept for tiling and memory-efficient kernels.
- `notes/`: Architectural research and theoretical notes.
- `profiling/`: Performance data and VRAM utilization reports.
- `setup/`: Environment configuration guides for AMD GPUs.

## Prerequisites
- Ubuntu 22.04 LTS
- ROCm 5.7+ or 6.x
- AMD Radeon RX 6800 (RDNA2) or similar
- Python 3.10+

## Getting Started
See [setup/rocm_install.md](file:///c:/Users/moham/Desktop/DEV/Research/AIML/FlashAttention/setup/rocm_install.md) for installation instructions.

## Disclaimer
This project is for research purposes. Performance may vary significantly compared to NVIDIA counterparts due to hardware architectural differences and software stack maturity.
