# FlashAttention Theory Refresher

FlashAttention is based on the idea of making attention memory-efficient by avoiding the instantiation of the large $N \times N$ attention matrix.

## Key Concepts

### 1. Tiling
The $Q$, $K$, and $V$ matrices are divided into blocks that fit into the fast On-Chip memory (LDS on AMD, Shared Memory on NVIDIA).
- Standard Attention: $O(N^2)$ memory reads/writes.
- FlashAttention: $O(N^2/M)$ memory reads/writes (where $M$ is the size of fast memory).

### 2. Online Softmax
To compute softmax accurately over blocks, we must track:
- $m_i$: The maximum value seen so far for a row.
- $l_i$: The running sum of exponentials.

Formula for merging blocks:
$m_{new} = \max(m_{old}, m_{block})$
$l_{new} = l_{old} \cdot e^{m_{old} - m_{new}} + l_{block} \cdot e^{m_{block} - m_{new}}$

### 3. Recomputation (Backward Pass)
Instead of storing the $N \times N$ matrix for the backward pass, FlashAttention recomputes it block-by-block using stored statistics ($m_i, l_i$).

## Implementation Checklist for AMD
- [ ] Determine optimal block size for GCN/RDNA2 CUs.
- [ ] Map LDS (Local Data Share) to the FlashAttention tiling buffers.
- [ ] Handle FP16 precision and potential overflow in online softmax.
