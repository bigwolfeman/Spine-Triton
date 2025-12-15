When you write standard Triton, you think in terms of threads and warps operating on GPU SIMD units. When you write spine-triton, you're mental model ought to be a little different: a tiled matrix engine with explicit format control. It is actually better this way, you'd be surprised how often the compiler sucks at handling this. It also lets you handle batch tricks and unsupported datatypes, like GEMV. This post walks through what I learned building attention for the K1, focusing on why the SMT primitives exist and how they map to the underlying linalg.mmt4d operations. 

My opinion after all of this... It was an interesting middle ground between high level and low level. The primary thing the programmer needs out of low level is this granular control. I think as spine-triton matures it will be an excellent DSL for its platform. I cover some usage ambiguities in this post. I think good answers to these ambiguities will provide important pieces of what else is needed from low level while keeping the code easier to write, namely controlling the flow of data through computation; while also keeping the really painful parts out of the code, like core scheduling.

## The Matrix Engine Mental Model

Most ML accelerators (Tensor Cores, AMX, TPU) share a common pattern: they want data in a specific tiled format, they compute on fixed-size micro-tiles, and they accumulate into special registers. The K1 Matrix Engine follows this model, but spine-triton makes the tiling explicit rather than hidden in intrinsics. My opinion is this is a great decision for a DSL for niche hardware. We get the higher level abstraction of Triton, with more control than Triton normally offers. With out this someone would come along and port another DSL for the hardware and fragment the ecosystem later. For the immediate future of RISCV, spine-triton is a brilliant decision.

The compilation flow looks like:

```
Triton kernel with smt.*
    ↓
TTIR with xsmt.* ops
    ↓  [XSMTToLinalg]
linalg.mmt4d + linalg.pack/unpack
    ↓
LLVM IR → RISC-V
```

Smt primitives exist to generate linalg.mmt4d - the 4D tiled matmul that maps directly to hardware instructions.

### What mmt4d Actually Means

When you see `linalg.mmt4d`, you're looking at a matmul that operates on 4D tensors:

```
C[M/m, N/n, m, n] += A[M/m, K/k, m, k] @ B[K/k, N/n, k, n]
```

Where:
- Capital letters (M, N, K) are the logical dimensions
- Lowercase letters (m, n, k) are the micro-tile sizes (hardware-specific)
    - These must cleanly divide with no remainder.

This format lets the hardware:
1. Stream in micro-tiles sequentially (better memory patterns, potentially less fragmentation compared to standard triton)
2. Keep micro-tiles in special registers (avoid shuffle overhead)
3. Compute with fixed-size matrix units (simpler datapath)

Based on my research for the K1, the micro-tiles are typically `8×8` or `8×16`. This matches the matrix engine's native compute granularity.

## Building Block 1: The Descriptor Load

Let's start with the most fundamental operation: getting data into the matrix engine:

```python
a_packed = smt.descriptor_load(
    block_ptr,
    (offset_m, offset_k),    # Where in the block
    (SUB_BLK_M, BLOCK_K),    # Logical shape: e.g., [32, 64]
    (MICRO_M, MICRO_K)       # Micro shape: e.g., [8, 8]
)
```

This takes a 2D tile `[32, 64]` and repacks it to `[4, 8, 8, 8]`:
- Outer dimensions: `[32/8, 64/8]` = `[4, 8]` (number of micro-tiles)
- Inner dimensions: `[8, 8]` (micro-tile size)

Why does this matter? The hardware matrix unit expects data in this packed format. If you tried to use standard `tl.load`, you'd get flat 2D data, and the compiler would have to insert pack operations - or worse, fall back to scalar loops. I can't instrument this to be certain, but it is very likely the case.

From reading `semantic.py` in the source:

```python
result_shape = [
    shape[0] // micro_size[0],
    shape[1] // micro_size[1],
    micro_size[0],
    micro_size[1]
]
```

The descriptor load does this packing explicitly, giving the compiler maximum flexibility for DMA scheduling and prefetch. Enabling potentially lower memory fragmentation compared to standard triton.

## Building Block 2: The Accumulator View

For matrix multiply, you need: `C = A @ B + C`. The `+ C` part requires special handling because C is also in packed format:

```python
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)  # Start flat

# Create a view with micro-tiling structure
acc_view = smt.view(
    acc,
    (offset_m, offset_n),     # Which part of C
    (SUB_BLK_M, SUB_BLK_N),   # View shape
    (MICRO_M, MICRO_N)        # Micro sizes
)

# Now acc_view has the 4D structure for mmt4d
acc_view = smt.dot(a_packed, b_packed, acc_view)
```

The `smt.view` creates a packed 4D view of a portion of the accumulator. AT the IR level, it lowers to a combination of slice extraction and linalg.pack operations, preparing the data for the tiled matmul.

## Building Block 3: Putting It Together for Q @ K^T

Here's where it gets interesting. For attention, we compute `Q @ K^T`:
- Q is `[M, D]` (sequence length × head dimension)
- K is `[N, D]` (same)
- K^T is conceptually `[D, N]`

We want to compute this in tiles, loading K once and reusing it across Q sub-blocks:

```python
# Load K^T as packed 4D tensor (ONCE, outside the loop)
k_block_ptr = tl.make_block_ptr(
    base=k_ptr,
    shape=[D, N],                    # Logical shape of K^T
    strides=[stride_kd, stride_kn],  # Swapped for transpose
    offsets=[0, pid_n * BLOCK_N],
    block_shape=[BLOCK_D, BLOCK_N],
    order=[1, 0],
)

k_packed = smt.descriptor_load(
    k_block_ptr, (0, 0),
    (BLOCK_D, BLOCK_N),
    (MICRO_K, MICRO_N)
)

# Initialize accumulator for output tile [BLOCK_M, BLOCK_N]
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

# Iterate over M in sub-blocks (parallelizable)
for s in smt.parallel(0, num_sub_blocks):
    # Load Q sub-block
    q_packed = smt.descriptor_load(
        q_block_ptr,
        (s * SUB_BLK_M, 0),
        (SUB_BLK_M, BLOCK_D),
        (MICRO_M, MICRO_K)
    )

    # Create view into accumulator for this sub-block
    acc_view = smt.view(
        acc,
        (s * SUB_BLK_M, 0),
        (SUB_BLK_M, BLOCK_N),
        (MICRO_M, MICRO_N)
    )

    # Accumulate: C[sub] += Q[sub] @ K^T
    acc_view = smt.dot(q_packed, k_packed, acc_view)
```

The pattern: load once, reuse, accumulate. The K matrix stays in packed format across iterations.

Note: The "load once, reuse" pattern works for the QK kernel because K is loaded once per output tile and reused across Q sub-blocks. However, the fused attention kernel below must reload K for each KV block. Why? The online softmax needs to iterate over the entire sequence dimension, so we can't cache K across blocks. This is the tradeoff that comes with FlashAttention-style streaming.

### Why smt.parallel?

The `smt.parallel(0, n)` wrapper is a hint to the compiler that these iterations are independent. Unlike standard `range()`, it signals:
- No loop-carried dependencies
- Can vectorize or parallelize across cores
- Each iteration computes on a disjoint memory region

Whether the backend actually parallelizes depends on hardware resources and the Linalg → LLVM lowering strategy.

## The Format Transition Problem: Why Full Attention is Hard

Let's think about complete attention: `softmax(Q @ K^T) @ V`

1. Q @ K^T: Perfect for SMT - pure matmul, both operands in packed format 
2. Softmax: Element-wise (`max`, `exp`, `sum`, `div`) - runs on vector units, not matrix engine
3. P @ V: Could use SMT, but there's a catch...

After softmax, the probability matrix `P` lives in flat 2D registers. To use `smt.dot` for `P @ V`, you'd need to:

```python
# 1. Allocate shared memory (4D layout)
#smt.alloc returns a 4D tensor: [BLOCK_M/m, BLOCK_N/n, m, n]
p_scratch = smt.alloc((BLOCK_M, BLOCK_N), (MICRO_M, MICRO_N), dtype=tl.float32)

# 2. Reshape 2D probs into 4D packed format
num_m_tiles = BLOCK_M // MICRO_M
num_n_tiles = BLOCK_N // MICRO_N

for m_tile in range(num_m_tiles):
    for n_tile in range(num_n_tiles):
        # Extrat microtiles
        m_start = m_tile * MICRO_M
        n_start = n_tile * MICRO_N
        micro_tile = p[m_start:m_start+MICRO_M, n_start:n_start+MICRO_N]

        # Assign to 4D
        p_scratch[m_tile, n_tile] = micro_tile

# 3. Reload as packed tensor
# p_packed = smt.descriptor_load(p_scratch_ptr, ...) pointer conversion?
p_packed = p_scratch #if not pointer, then just copy and let the compiler solve it

# 4. Now can use smt.dot
acc = smt.dot(p_packed, v_packed, acc)
```

There are no examples of this pattern in the codebase. The `smt.alloc` primitive exists, but the store/reload dance isn't documented. Here is how to reason through this if you have hardware to compile on. 

- Does `smt.alloc` return a pointer you can `tl.store` to?
- Or does it return a special tensor type?
- Can `smt.descriptor_load` operate on local allocations?
- What's the latency penalty for this round-trip?

Without hardware to test, I can't answer these. So the implementation makes a pragmatic choice.

## Implementation Strategy: Two Kernels

### Kernel 1: Pure SMT for Q @ K^T

```python
@triton.jit
def smt_qk_kernel(...):
    """Clean demonstration of SMT primitives for matrix multiply."""
    # ... setup block pointers ...

    k_packed = smt.descriptor_load(k_block_ptr, ...)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for s in smt.parallel(0, sub_num):
        q_packed = smt.descriptor_load(q_block_ptr, ...)
        acc_view = smt.view(acc, ...)
        acc_view = smt.dot(q_packed, k_packed, acc_view)

    out = acc * sm_scale
    tl.store(out_ptr, out)
```

This is the reference implementation - it shows the correct SMT pattern without complications. Both operands packed, proper accumulator views, clean loop structure.

### Kernel 2: Full Attention with Hybrid Path

For the fused kernel, the online softmax requirement changes the memory access pattern. We must iterate over all KV positions to compute the global softmax statistics, which means K is streamed per KV block rather than loaded once and reused.

```python
@triton.jit
def smt_attention_fused_kernel(...):
    """Full attention with online softmax, using SMT where clear."""

    # Outer loop over K/V blocks (FlashAttention style)
    for block_idx in range(num_kv_blocks):
        # PHASE 1: Q @ K^T via SMT
        # (K is loaded per KV block; not cached across blocks)
        k_packed = smt.descriptor_load(k_block_ptr, ...)
        qk = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for s in smt.parallel(0, sub_num):
            q_packed = smt.descriptor_load(q_block_ptr, ...)
            qk_view = smt.view(qk, ...)
            qk_view = smt.dot(q_packed, k_packed, qk_view)

        qk = qk * sm_scale

        # PHASE 2: Online softmax (standard Triton)
        # Update running max
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        
        # Compute exponentials with correction factor
        alpha = tl.exp(m_i - m_new)
        p_ij = tl.exp(qk - m_new[:, None])
        
        # Update running sum
        l_new = alpha * l_i + tl.sum(p_ij, axis=1)

        # PHASE 3: P @ V
        # Using tl.dot instead of SMT because the
        # p_ij to packed format transition is unclear
        # (see smt.alloc discussion earlier for potential SMT approach)
        v = tl.load(v_block_ptr)
        acc = acc * alpha[:, None]  # Apply correction factor
        acc += tl.dot(p_ij.to(v.dtype), v)
        
        # Update statistics
        l_i = l_new
        m_i = m_new

    # Final output
    out = acc / l_i[:, None]
    tl.store(out_ptr, out)
```

## What the IR Should Look Like

Even without hardware to compile, we can reason about the TTIR → Linalg transformation:

```
xsmt.descriptor_load → linalg.pack
xsmt.dot → linalg.mmt4d
xsmt.view → memrfe.subview + linalg.pack
Standard tl.dot → linalg.matmul (unpacked)
```

The compiler inserts `linalg.unpack` operations where needed to convert between packed 4D and flat 2D representations. With `SPINE_TRITON_DUMP_PATH` set, you'd see:

```mlir
// After XSMTToLinalg pass
%packed_a = linalg.pack %a inner_dims_pos = [0, 1]
                             inner_tiles = [8, 8] ...
%packed_b = linalg.pack %b ...
%result = linalg.mmt4d ins(%packed_a, %packed_b : ...)
                       outs(%packed_c : ...)
%unpacked = linalg.unpack %result ...
```

The beauty of this design: the tiling metadata (`[8, 8]`) is explicit in the IR, so backend code generators can make informed decisions about register allocation and instruction selection.

## Open Questions for SpacemiT

If you're reading this and want to help the community:

1. **Scratchpad pattern**: Can you show an example of:
   ```python
   scratch = smt.alloc(...)
   # store something to scratch
   # reload via smt.descriptor_load
   ```

2. **Descriptor lifetime**: Is `k_packed` consumed by `smt.dot`, or can it be reused across multiple dot calls?

3. **Mixed precision**: How do you control accumulator precision? FP16 inputs with FP32 accumulation?

4. **Micro-tile constraints**: What are the valid `(MICRO_M, MICRO_N, MICRO_K)` combinations for different K1 variants? Generally CuTe provides this as a table in the repo, I did not find an equivalent. 

5. **Performance model**: What's the theoretical peak GFLOPS for `linalg.mmt4d` on K1? This helps validate whether kernels are hardware-bound or software-bound.