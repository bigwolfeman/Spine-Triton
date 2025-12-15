"""
SpacemiT SMT Attention Kernels
================================

Two implementations demonstrating proper spine-triton SMT primitive usage:
1. smt_qk_kernel: Clean reference for Q @ K^T matmul using SMT
2. smt_attention_fused_kernel: Full attention with online softmax (hybrid approach)

Key patterns demonstrated:
- Proper smt.descriptor_load for both operands
- Correct K^T handling via stride swapping
- smt.view for accumulator management
- smt.parallel loop iteration
- Online softmax for memory efficiency
"""

import torch
import triton
import triton.language as tl
import triton.language.extra.smt as smt
from triton.backends.spine_triton.driver import CPUDriver


# =============================================================================
# KERNEL 1: Pure SMT Matrix Multiply (Q @ K^T)
# =============================================================================
# This is the reference implementation - demonstrates CORRECT SMT usage
# without the complexity of full attention.

@triton.jit
def smt_qk_kernel(
    q_ptr, k_ptr, out_ptr,
    # Q is [M, D], K is [N, D], Output is [M, N]
    M, N, D,
    stride_qm, stride_qd,
    stride_kn, stride_kd,
    stride_om, stride_on,
    sm_scale,  # 1/sqrt(d) scaling factor
    # Block sizes - must be divisible by micro sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SUB_BLK_M: tl.constexpr,  # Sub-block size for M dimension
    # Hardware micro-tile sizes (typically 8x8 or 8x16 for K1?)
    MICRO_M: tl.constexpr,
    MICRO_N: tl.constexpr,
    MICRO_K: tl.constexpr,
):
    """
    Compute scaled Q @ K^T for attention scores.

    This kernel demonstrates the correct pattern for SMT primitives:
    - Both Q and K loaded via smt.descriptor_load (both operands packed)
    - K loaded once outside loop and reused
    - Proper K^T handling via stride manipulation
    - Sub-block iteration with smt.parallel
    - Accumulator views via smt.view

    Grid: (cdiv(M, BLOCK_M), cdiv(N, BLOCK_N))
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Step 1: Create block pointers

    # Q block pointer: [BLOCK_M, BLOCK_D]
    q_block_ptr = tl.make_block_ptr(
        base=q_ptr,
        shape=[M, D],
        strides=[stride_qm, stride_qd],
        offsets=[pid_m * BLOCK_M, 0],
        block_shape=[BLOCK_M, BLOCK_D],
        order=[1, 0],
    )

    # K^T block pointer: K is stored as [N, D], we want K^T as [D, N]
    # Swap strides to get transpose without explicit transpose operation
    k_block_ptr = tl.make_block_ptr(
        base=k_ptr,
        shape=[D, N],  # Transposed shape
        strides=[stride_kd, stride_kn],  # Swapped strides for implicit transpose
        offsets=[0, pid_n * BLOCK_N],
        block_shape=[BLOCK_D, BLOCK_N],
        order=[1, 0],
    )

    # Step 2: Initialize accumulator (flat 2D)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)


    # Step 3: Load K^T ONCE (reused across all Q sub-blocks)
    # packs K^T into 4D format:
    # [BLOCK_D, BLOCK_N] → [BLOCK_D/MICRO_K, BLOCK_N/MICRO_N, MICRO_K, MICRO_N]
    k_packed = smt.descriptor_load(
        k_block_ptr,
        (0, 0),  # Load from start of block
        (BLOCK_D, BLOCK_N),
        (MICRO_K, MICRO_N),
    )

    # Step 4: Iterate over M dimension in sub-blocks (parallelize with spine-triton smt.parallel)
    sub_num = tl.cdiv(BLOCK_M, SUB_BLK_M)

    for s in smt.parallel(0, sub_num):
        # Load Q sub-block and pack to 4D
        q_packed = smt.descriptor_load(
            q_block_ptr,
            (s * SUB_BLK_M, 0),  # Offset to s-th sub-block
            (SUB_BLK_M, BLOCK_D),
            (MICRO_M, MICRO_K),
        )

        # Create view into accumulator for this sub-block
        # This prepares the accumulator slice for the packed mmt4d operation
        acc_view = smt.view(
            acc,
            (s * SUB_BLK_M, 0),
            (SUB_BLK_M, BLOCK_N),
            (MICRO_M, MICRO_N),
        )

        # Matrix multiply: acc[sub] += Q[sub] @ K^T
        # This lowers to linalg.mmt4d at the IR level
        acc_view = smt.dot(q_packed, k_packed, acc_view)



    # Step 5: Scale and store output
    out = acc * sm_scale
    out = out.to(out_ptr.dtype.element_ty)

    out_block_ptr = tl.make_block_ptr(
        base=out_ptr,
        shape=[M, N],
        strides=[stride_om, stride_on],
        offsets=[pid_m * BLOCK_M, pid_n * BLOCK_N],
        block_shape=[BLOCK_M, BLOCK_N],
        order=[1, 0],
    )
    tl.store(out_block_ptr, out, boundary_check=(0, 1))


# =============================================================================
# KERNEL 2: Full Attention with Hybrid Approach
# =============================================================================
# full attention with online softmax.
# Q @ K^T uses SMT, P @ V uses tl.dot (fallback) due to format transition complexity.
# Spine-Triton: The "backbone" of the attention kernel.

@triton.jit
def smt_attention_fused_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    M, N, D,  # M=sequence length (Q), N=sequence length (K,V), D=head dimension
    stride_qm, stride_qd,
    stride_kn, stride_kd,
    stride_vn, stride_vd,
    stride_om, stride_od,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SUB_BLK_M: tl.constexpr,
    MICRO_M: tl.constexpr,
    MICRO_N: tl.constexpr,
    MICRO_K: tl.constexpr,
):
    """
    Full scaled dot-product attention with online softmax.

    Pattern: softmax(Q @ K^T / sqrt(d)) @ V

    Implementation notes:
    - Q @ K^T: Uses SMT primitives (proper matrix engine usage)
    - Softmax: Standard Triton ops (element-wise, runs on vector units)
    - P @ V: Uses tl.dot as fallback (format transition unclear)

    This hybrid approach is honest about architectural constraints.
    """
    pid_m = tl.program_id(0)

    # =========================================================================
    # Setup Q block pointer
    # =========================================================================
    q_block_ptr = tl.make_block_ptr(
        base=q_ptr,
        shape=[M, D],
        strides=[stride_qm, stride_qd],
        offsets=[pid_m * BLOCK_M, 0],
        block_shape=[BLOCK_M, BLOCK_D],
        order=[1, 0],
    )

    # =========================================================================
    # Initialize running statistics for online softmax
    # =========================================================================
    # Running maximum for numerical stability
    m_i = tl.full([BLOCK_M], value=float("-inf"), dtype=tl.float32)
    # Running sum of exponentials
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    # Accumulator for output
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    # =========================================================================
    # Outer loop: iterate over K/V sequence in blocks
    # =========================================================================
    num_kv_blocks = tl.cdiv(N, BLOCK_N)

    for block_kv in range(num_kv_blocks):
        kv_offset = block_kv * BLOCK_N

        # PHASE 1: Compute Q @ K^T using SMT
        # K^T block pointer for this iteration
        k_block_ptr = tl.make_block_ptr(
            base=k_ptr,
            shape=[D, N],
            strides=[stride_kd, stride_kn],
            offsets=[0, kv_offset],
            block_shape=[BLOCK_D, BLOCK_N],
            order=[1, 0],
        )

        # Load K^T in packed format
        # NOTE: K is streamed per KV block (not hoisted/cached across blocks).
        # Online softmax requires iterating over all KV positions, so we cannot
        # reuse a single packed K across the outer loop. This differs from the
        # smt_qk_kernel where K is loaded once per output tile and reused across Q sub-blocks.
        k_packed = smt.descriptor_load(
            k_block_ptr,
            (0, 0),
            (BLOCK_D, BLOCK_N),
            (MICRO_K, MICRO_N),
        )

        # Compute QK scores using SMT dot
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        sub_num = tl.cdiv(BLOCK_M, SUB_BLK_M)

        for s in smt.parallel(0, sub_num):
            q_packed = smt.descriptor_load(
                q_block_ptr,
                (s * SUB_BLK_M, 0),
                (SUB_BLK_M, BLOCK_D),
                (MICRO_M, MICRO_K),
            )
            qk_view = smt.view(
                qk,
                (s * SUB_BLK_M, 0),
                (SUB_BLK_M, BLOCK_N),
                (MICRO_M, MICRO_N),
            )
            qk_view = smt.dot(q_packed, k_packed, qk_view)

        # Scale scores
        qk = qk * sm_scale


        # PHASE 2: Online softmax (standard Triton)
        # Update running max
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)

        # Compute exponentials
        alpha = tl.exp(m_i - m_new)
        p_ij = tl.exp(qk - m_new[:, None])

        # update running sum
        l_new = alpha * l_i + tl.sum(p_ij, axis=1)

        # PHASE 3: Multiply by V and accumulate
        # Load V block (standard load)
        v_block_ptr = tl.make_block_ptr(
            base=v_ptr,
            shape=[N, D],
            strides=[stride_vn, stride_vd],
            offsets=[kv_offset, 0],
            block_shape=[BLOCK_N, BLOCK_D],
            order=[1, 0],
        )
        v = tl.load(v_block_ptr, boundary_check=(0, 1))

        # Update accumulator with correction factor
        # NOTE: Using tl.dot here as fallback (not SMT)
        # The format transition from 2D probs to packed 4D is unclear
        acc = acc * alpha[:, None]
        acc += tl.dot(p_ij.to(v.dtype), v)

        # Update statistics
        l_i = l_new
        m_i = m_new

    # =========================================================================
    # Final normalization and store
    # =========================================================================
    acc = acc / l_i[:, None]
    out = acc.to(out_ptr.dtype.element_ty)

    out_block_ptr = tl.make_block_ptr(
        base=out_ptr,
        shape=[M, D],
        strides=[stride_om, stride_od],
        offsets=[pid_m * BLOCK_M, 0],
        block_shape=[BLOCK_M, BLOCK_D],
        order=[1, 0],
    )
    tl.store(out_block_ptr, out, boundary_check=(0, 1))


# =============================================================================
# Python wrapper functions
# =============================================================================

def run_smt_qk(Q, K, sm_scale=None):
    """
    Compute scaled Q @ K^T using SMT primitives.

    Args:
        Q: [M, D] query tensor
        K: [N, D] key tensor
        sm_scale: scaling factor (default: 1/sqrt(D))

    Returns:
        scores: [M, N] attention scores (before softmax)
    """
    # Extract dimensions from input tensors
    M, D = Q.shape  # M = sequence length, D = head dimension
    N = K.shape[0]  # N = key sequence length

    # Default scaling factor for attention: 1/sqrt(d_k)
    # This prevents dot products from growing too large and causing softmax saturation
    if sm_scale is None:
        sm_scale = 1.0 / (D ** 0.5)

    # Allocate output tensor for attention scores
    # Shape: [M, N] - each query position attends to all key positions
    out = torch.empty((M, N), device=Q.device, dtype=Q.dtype)

    # =========================================================================
    # BLOCK SIZE CONFIGURATION (spine-triton SMT specific)
    # =========================================================================
    # Traditional Triton only needs BLOCK_M, BLOCK_N, BLOCK_K for tiling.
    # spine-triton's SMT API introduces hierarchical blocking for RISC-V targets:
    
    BLOCK_M = 64      # Outer block: M dimension (queries)
    BLOCK_N = 64      # Outer block: N dimension (keys)
    BLOCK_D = 64      # Outer block: K dimension (reduction over head_dim)
    
    # SMT-specific sub-blocking parameters:
    SUB_BLK_M = 32    # Middle-tier block size for M
    MICRO_M = 8       # Microkernel tile size for M (matches RISC-V register capacity)
    MICRO_N = 16      # Microkernel tile size for N
    MICRO_K = 8       # Microkernel tile size for K (reduction dimension)
    
    # Note: Block % Micro = 0 is important and should be validated.

    # These MICRO sizes correspond to the actual compute granularity that gets
    # lowered to RISC-V vector instructions. Think of them as the "unroll factors"
    # that map directly to hardware vector length and register file size.

    # Standard Triton grid definition - this part is the same as regular Triton
    # Create a 2D grid where each block handles BLOCK_M x BLOCK_N output elements
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(N, META['BLOCK_N']),
    )

    # =========================================================================
    # SPINE-TRITON DRIVER SETUP 
    # =========================================================================
    # Regular Triton auto-detects GPU backends (CUDA/ROCm).
    # spine-triton requires explicit CPU driver initialization for RISC-V targets.
    
    driver = CPUDriver()  # spine-triton initialize CPU backend driver
    
    # Set architecture ID to "0xA03C" - this identifies the Spacemit K1 RISC-V chip
    # The driver uses this to select appropriate instruction scheduling and vector extensions
    driver.set_current_arch_id("0xA03C")  # spine-triton target specific RISC-V arch
    
    # Activate this driver for the current thread
    # This replaces Triton's default GPU driver with the RISC-V driver
    triton.runtime.driver.set_active(driver)  # spine-triton set active backend

    # =========================================================================
    # KERNEL LAUNCH
    # =========================================================================
    # The launch syntax is identical to regular Triton (kernel[grid](...)),
    # but under the hood, spine-triton is:
    # 1. Compiling to RISC-V instructions instead of PTX/AMDGPU
    # 2. Using SMT lowering for matrix operations
    # 3. Scheduling for CPU cache hierarchy instead of GPU shared memory
    
    smt_qk_kernel[grid](
        Q, K, out,
        M, N, D,
        Q.stride(0), Q.stride(1),        # Query tensor strides
        K.stride(0), K.stride(1),        # Key tensor strides
        out.stride(0), out.stride(1),    # Output tensor strides
        sm_scale,                         # Attention scaling factor
        # Block size parameters (passed as constexpr to enable compile-time optimization)
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        # SMT-specific parameters (not used in standard Triton):
        SUB_BLK_M=SUB_BLK_M,  # spine-triton specific SMT parameter
        MICRO_M=MICRO_M,      # spine-triton specific SMT parameter
        MICRO_N=MICRO_N,      # spine-triton specific SMT parameter
        MICRO_K=MICRO_K,      # spine-triton specific SMT parameter
    )

    return out


def run_smt_attention(Q, K, V, sm_scale=None):
    """
    Compute full scaled dot-product attention using SMT+hybrid approach.

    Args:
        Q: [M, D] query tensor
        K: [N, D] key tensor
        V: [N, D] value tensor
        sm_scale: scaling factor (default: 1/sqrt(D))

    Returns:
        out: [M, D] attention output
    """
    # Verify we have spine-triton's SMT support available
    if not HAS_SMT:
        raise RuntimeError("spine-triton not available")

    # Extract input dimensions
    M, D = Q.shape  # M = query sequence length, D = head dimension
    N = K.shape[0]  # N = key/value sequence length

    # Attention scaling factor (standard transformer practice)
    if sm_scale is None:
        sm_scale = 1.0 / (D ** 0.5)

    # Allocate output tensor
    # Shape: [M, D] - same as Q, each query position produces a D-dimensional output
    out = torch.empty((M, D), device=Q.device, dtype=Q.dtype)

    # =========================================================================
    # BLOCK SIZE CONFIGURATION (spine-triton SMT specific)
    # =========================================================================
    # Same hierarchical blocking strategy as run_smt_qk, but now we're computing
    # the full attention operation: softmax(Q @ K^T) @ V
    
    BLOCK_M = 64      # Process 64 query positions at a time
    BLOCK_N = 64      # Iterate over 64 key/value positions per inner loop
    BLOCK_D = 64      # Handle 64 dimensions of the head at a time
    
    # SMT microkernel parameters (spine-triton specific):
    SUB_BLK_M = 32    # Sub-blocking for better cache locality
    MICRO_M = 8       # Microkernel M dimension (maps to RISC-V vector registers)
    MICRO_N = 16      # Microkernel N dimension
    MICRO_K = 8       # Microkernel K dimension (reduction)
    
    # Note: These MICRO parameters are critical for performance on RISC-V
    # They determine how the SMT compiler generates vector instructions.

    # =========================================================================
    # GRID CONFIGURATION
    # =========================================================================
    # For the fused attention kernel, we use a 1D grid over query positions only.
    # Each block iterates over all key/value positions internally (for online softmax).
    # This differs from the QK-only kernel which used a 2D grid.
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']),)  # 1D grid over queries

    # =========================================================================
    # SPINE-TRITON DRIVER SETUP
    # =========================================================================
    # Same driver initialization as run_smt_qk
    # In a real env, you'd probably init this once globally
    # rather than per-function, its redone here for clarity 
    
    driver = CPUDriver()                    # spine-triton initialize CPU backend driver
    driver.set_current_arch_id("0xA03C")  # spine-triton target specific RISC-V arch
    triton.runtime.driver.set_active(driver)  # spine-triton activate CPU backend

    # =========================================================================
    # KERNEL LAUNCH - FUSED ATTENTION
    # =========================================================================
    # This kernel performs the complete attention operation:
    # 1. Q @ K^T (using SMT matrix multiplication)
    # 2. Online softmax (keeping running max/sum for numerical stability)
    # 3. P @ V (using fallback tl.dot due to format conversion complexity)
    #
    # All in a single kernel pass. This is more efficient than three separate
    # kernel launches because we avoid materializing intermediate tensors.
    
    smt_attention_fused_kernel[grid](
        Q, K, V, out,                     # Input and output tensors
        M, N, D,                          # Dimension parameters
        Q.stride(0), Q.stride(1),         # Query strides
        K.stride(0), K.stride(1),         # Key strides
        V.stride(0), V.stride(1),         # Value strides
        out.stride(0), out.stride(1),     # Output strides
        sm_scale,                         # Attention scale factor
        # Standard Triton block parameters:
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        # spine-triton SMT-specific parameters:
        SUB_BLK_M=SUB_BLK_M,  # SMT sub-block size
        MICRO_M=MICRO_M,      # SMT microkernel M
        MICRO_N=MICRO_N,      # SMT microkernel N
        MICRO_K=MICRO_K,      # SMT microkernel K
    )

    return out

# Demo
if __name__ == "__main__":
    if not HAS_SMT:
        print("spine-triton not available - cannot run demo")
    else:
        print("SMT Attention Demo")
        print("=" * 60)

        # Example usage
        L, D = 128, 64
        Q = torch.randn((L, D), dtype=torch.float32, device='cpu')
        K = torch.randn((L, D), dtype=torch.float32, device='cpu')
        V = torch.randn((L, D), dtype=torch.float32, device='cpu')

        print(f"\nInput: Q={Q.shape}, K={K.shape}, V={V.shape}")

        # Test Q @ K^T kernel
        print("\n1. Testing smt_qk_kernel (Q @ K^T)...")
        scores = run_smt_qk(Q, K)
        print(f"   Output shape: {scores.shape}")

        # Test full attention kernel
        print("\n2. Testing smt_attention_fused_kernel (full attention)...")
        out = run_smt_attention(Q, K, V)
        print(f"   Output shape: {out.shape}")

        print("\n" + "=" * 60)
        print("Demo complete. Set SPINE_TRITON_DUMP_PATH to see IR dumps.")
