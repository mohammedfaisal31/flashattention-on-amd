# RDNA2 Architectural Constraints (gfx1030)

Researching the hardware limits of the Radeon RX 6800 for compute tasks.

## Hardware Specifications
- **Architecture**: RDNA 2 (gfx1030)
- **Compute Units (CUs)**: 60
- **VRAM**: 16 GB GDDR6
- **Infinity Cache**: 128 MB

## Memory Hierarchy Constraints

### 1. LDS (Local Data Share)
- **Size**: 64 KB per CU (Dual Compute Unit structure share).
- **Critical for FlashAttention**: Tiling block sizes must fit within this 64KB limit (shared between Q, K, V blocks and intermediate results).
- **NVIDIA Comparison**: A100 has up to 164 KB of shared memory per SM. RDNA2 is significantly more restricted.

### 2. Registers
- **VGPRs (Vector General Purpose Registers)**: Large file, but high usage reduces occupancy.
- **SGPRs (Scalar General Purpose Registers)**: Used for uniform variables and control flow.

### 3. Warp/Wavefront Size
- **Wavefront Size**: 32 (Wave32) or 64 (Wave64). RDNA2 typically defaults to Wave32 for many shaders.
- Performance implications for parallel reductions in online softmax.

## Known Bottlenecks
- **FP16 Throughput**: RX 6800 has high FP16 throughput, but memory bandwidth can be a bottleneck for large sequence lengths.
- **ROCm/HIP maturity**: Optimization for RDNA2 (consumer/workstation) is often secondary to CDNA (Instinct/datacenter) in the ROCm stack.
