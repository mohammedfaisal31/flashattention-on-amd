# Memory Profile Log

Detailed breakdown of VRAM utilization for different attention configurations on AMD Radeon RX 6800.

## Naive Attention Baseline (FP16)
| Seq Length | Batch Size | Peak VRAM (MB) | Status |
|------------|------------|----------------|--------|
| 1024       | 8          |                |        |
| 2048       | 4          |                |        |
| 4096       | 2          |                |        |
| 8192       | 1          |                |        |

## Tiled Attention Prototype (FP16)
| Seq Length | Batch Size | Peak VRAM (MB) | Savings vs Baseline |
|------------|------------|----------------|---------------------|
| 1024       | 8          |                |                     |
| 2048       | 4          |                |                     |
| 4096       | 2          |                |                     |
| 8192       | 1          |                |                     |

## Memory Bottleneck Observations
- **Activation Memory**: Standard softmax creates a $[B, H, S, S]$ tensor. At $S=8192$, this is $1 \times 12 \times 8192 \times 8192 \times 2$ bytes $\approx 1.6$ GB for one head-layer.
- **Gradient Memory**: Requirements for backprop in naive mode.
