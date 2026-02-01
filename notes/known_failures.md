# Known Failures & Troubleshooting

This document tracks consistent failures, ROCm-specific bugs, and hardware-specific limitations encountered during the project.

## Software Stack Failures
| Issue | Cause | Fix / Workaround |
|-------|-------|------------------|
| MIOpen benchmark freeze | Improperly cached kernels | Clean `~/.cache/miopen` |
| Illegal Memory Access | GFX version mismatch | `H_OVERRIDE_GFX_VERSION=10.3.0` |

## Algorithmic Failures
| Issue | Observation | Context |
|-------|-------------|---------|
| FP16 Numerical Instability | Online softmax diverging | Need to verify $m_i$ initialization |

## Hardware Limitations
- **LDS Overflow**: Attempting block sizes > 128 for $Q$ and $K$ simultaneously often leads to registration failures.
- **Infinity Cache Misses**: Large sequence lengths (>4096) show sharp performance drops once blocks leave Infinity Cache.
