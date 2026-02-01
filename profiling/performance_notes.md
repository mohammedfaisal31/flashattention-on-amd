# Performance Notes

Analysis of execution time and hardware utilization on RDNA2.

## Latency Benchmarks
| Implementation | Latency (ms) | Sequence Length | Hardware |
|----------------|--------------|-----------------|----------|
| Naive          |              | 1024            | RX 6800  |
| Naive          |              | 4096            | RX 6800  |
| Prototype      |              | 1024            | RX 6800  |

## Architectural Insights
### LDS (Local Data Share) Usage
- RDNA2's 64KB LDS limit suggests a maximum block size of $64 \times 64$ when storing $Q, K, V$ tiles in FP16 concurrently ($64 \times 64 \times 2 \times 3 \approx 24$ KB). 
- Expanding to $128 \times 128$ tiles will require careful LDS reuse or double buffering.

### Compute vs IO Bound
- Standard attention is heavily IO-bound due to the $S^2$ memory writes for softmax.
- FlashAttention aims to move it towards being compute-bound by keeping tiles in LDS.

## Optimization Ideas for RX 6800
- Use `wave32` for better occupancy in small tiles.
- Leverage RDNA2's fast shared cache (Infinity Cache) for $K, V$ blocks.
