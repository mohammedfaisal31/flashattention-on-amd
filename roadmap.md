# Project Roadmap

This roadmap outlines the milestones for implementing memory-efficient attention on AMD RDNA2 hardware.

## Phase 1: Foundation & Baselines `[/]`
- [x] Repository structure setup with placeholder skeletons.
- [ ] Baseline performance profiling of standard PyTorch attention.
- [ ] Documentation of RX 6800 hardware specifications (LDS, Cache latency).

## Phase 2: Theory & Prototyping `[ ]`
- [ ] Formalize Online Softmax logic in Python (No tiling).
- [ ] Implement a basic tiled forward pass in FP16 using PyTorch (Manual tiling).
- [ ] Compare VRAM usage between naive and tiled prototypes.

## Phase 3: Hardware-Specific Optimization `[ ]`
- [ ] Explore HIP/C++ kernels for manual memory management.
- [ ] Research AMD Triton backend compatibility for RDNA2.
- [ ] Implement manual LDS (Local Data Share) management for tiling.

## Phase 4: Verification & Benchmarking `[ ]`
- [ ] Rigorous correctness checks against PyTorch native attention.
- [ ] Scaling analysis (Sequence length 512 -> 8k).
- [ ] Final performance comparison highlighting RDNA2 bottlenecks.

---
**Status**: Research & Exploration
**Current Focus**: Baseline benchmarking and architectural study.
