GPU Convolution Kernel Generator
===========================================

# Generalized Convolution Algorithm

See [oneDNN documentation](https://uxlfoundation.github.io/oneDNN/dev_guide_convolution.html)
for the naming conventions that are used below.

Convolution has more variations than GEMM but for simplicity we will rely on
the GEMM naming conventions to come up with a generalized convolution
algorithm. GEMM performs the following operation: `C += A * B`, where:

- `A` is `(M x K)` matrix
- `B` is `(K x N)` matrix
- `C` is `(M x N)` matrix

Algorithmically, any convolution can be expressed in a form which is very
similar to GEMM:

```python
for i0 in range(0, I0):
    ...
    for j0 in range(0, J0):
        ...
        c_val = 0
        for k0 in range(0, K0):
            ...
            a_val = load_A(i0, ..., k0)
            b_val = load_B(k0, ..., j0)
            c_val += a_val * b_val
        store_C(i0, ..., j0, ..., c_val)
```

`i0`, `j0`, and `k0` are `M`, `N` and `K` dimensions respectively:
- `M` dimensions are shared between `A` and `C`
- `N` dimensions are shared between `B` and `C`
- `K` dimensions are shared between `A` and `B`

Convolution may have many `I`/`J`/`K` dimensions.

Let's consider 1D forward convolution:

```python
for mb in range(0, MB):                                          # M
    for ow in range(0, OW):                                      # M
        for oc in range(0, OC):                                  # N
            dst_val = 0
            for ic in range(0, IC):                              # K
                for kw in range(0, KW):                          # K
                    src_val = load_src(mb, ic, ow, kw)
                    wei_val = load_wei(oc, ic, kw)
                    dst_val += src_val * wei_val
            store_dst(mb, oc, ow, dst_val)
```

Here `load_wei()` and `store_dst()` translate ND indices into 1D to calculate
the offset to load from/store to. However `load_src()` should also performs
convolution-specific `4D` to `3D` translation in the beginning. `ow` and `kw`
are translated to `iw` using the following expression `iw = ow * SW + kw * (DW + 1) - PW`.

Another convolution-specific feature is the logical padding support.
`load_src()` should return `0` in cases when `iw` index is out of bounds.

Considering all the above mentioned, load/store functions for `ncw`/`oiw`/`ncw`
layouts can be implemented as follows:

```python
def load_src(mb, ic, ow, kw):
    iw = ow * SW + kw * (DW + 1) - PW
    if iw < 0 or iw >= IW:
        return 0
    off = 0
    off += mb * IC * IW
    off += ic * IW
    off += iw
    return src[off]

def load_wei(oc, ic, kw):
    off = 0
    off += oc * IC * KW
    off += ic * KW
    off += kw
    return wei[off]

def store_dst(mb, oc, ow, val):
    off = 0
    off += mb * OC * OW
    off += oc * OW
    off += ow
    dst[off] = val
```

Backward by data and backward by weights convolutions can be expressed in the
same, GEMM-like form.

Here are the steps needed to transform any convolution to the GEMM-like
form:

- Identify M/N/K loops and map convolution tensors to A/B/C
    - For forward: source -> A, weights -> B, destination -> C
    - For backward by data: destination -> A, weights -> B, source -> C
    - For backward by weights: source -> A, destination -> B, weights -> C
- Describe load/store functions for the tensors. There are two parts:
    - Underlying layout (e.g. blocked layout `NChw32n32c`)
    - Access condition or mask (e.g. `iw >= 0 and iw < IW`). Load/store the
      element if the mask is true. Otherwise, for loads: return zero, for stores:
      drop the store.

Both steps depend on whether the convolution is forward/backward and should be
specialized accordingly. To properly do loads/stores, the generator introduces
the "view" abstraction which contains information about the tensor: its
underlying layout and corresponding masks (see the detailed description below).

## GPU Convolution Optimizations

GPU convolution requires a number of optimizations to reach close to roofline
performance:

- High-level optimizations
    - Loop order and blocking
        - Including assigning loops to the kernel grid and thread group grid
    - Single SLM buffering
        - Including selecting the SLM layout
    - Load/store decomposition into instructions
        - Block vs scattered messages
        - This also includes mask handling (for out-of-bound and stride
          conditions)
    - Multiplication decomposition into instructions (mad, dp4a, dpas(w))
        - This also includes optional GRF blocking (load -> compute -> load to
          the same GRF buffer -> compute)
        - May require auxiliary reorders to match the instruction layout
- Middle-level optimizations
    - Double/triple SLM buffering with split barriers
    - GRF buffering for SLM loads
- Low-level optimizations
    - Loop unrolling
    - Assigning GRF banks for multiplication buffers to avoid bank conflicts
    - SLM layout padding to avoid SLM write conflicts
    - Transforming dpas to dpasw to reduce SLM traffic and SLM consumption
    - Offset/address arithmetic optimizations

## Kernel Generation Flow

### Configuring Kernel Parameters

This is performed during primitive descriptor initialization.

Kernel configuration is defined by `config_t` object. The configuration
includes:

- Convolution problem description (`conv_problem_t`)
    - Propagation kind, batch size, input/output channels, etc
- Implementation-specific kernel parameters. Examples:
    - Block sizes
    - SLM buffering parameters: enabled/disabled, single/double/triple
    - FMA instruction to use: mad, dp4a, dpas(w)

Kernel parameters are set depending on the architecture, propagation, problem
shape, etc.

The configuration object is further passed to the kernel builder which
generates the kernel according to the configured parameters.

# IR Generation 

This and further steps are performed during primitive initialization.

`conv_ir_builder_t` class is responsible for the whole kernel IR generation.
There are other builder classes which are responsible for more specialized
functionality, for example:

- Builder to construct load-multiply statements
- Builder to decompose view load/store into messages

## Forming Loop Nest and A/B/C Views

The generation starts with creating the outer loop nest by forming
corresponding IR statements. The order of loops and blocking schemes are
hard-coded for different propagation kinds though some parameters are
configurable such as block sizes and the thread group size.

For simplicity there are some conventions and limitations:

- The outer loop nest is mapped to the kernel grid (grid of thread groups,
  supports up to 3 dimensions)
    - In many cases this requires packing of several problem-specific
      dimensions into one. In this case the kernel must implement unpacking
      using division and modulus operations.
- The outer loop nest typically includes M and N dimensions. It may also
  include K dimensions - in this case we have partial reduction, and the kernel
  must use atomic stores for C updates.
- Convention: thread group doesn't have outer loops across M or N dimensions,
  only across K dimensions.

According to these rules, the generic looping scheme looks as follows (assume
one dimension per M/N/K, for simplicity):

```python
for m_tg_idx in range(0, m, m_tg_blk):         # Mapped to the kernel grid
    for n_tg_idx in range(0, n, n_tg_blk):     # Mapped to the kernel grid
        for k_tg_idx in range(0, k, k_tg_blk): # Mapped to the kernel grid
            ...
            for k_idx in range(k_tg_idx, k_tg_idx + k_tg_blk, k_blk): # Loop inside thread
                ...
               # Perform C += A * B multiplication cooperatively by a thread group
               # A is (m_tg_blk x    k_blk)
               # B is (   k_blk x n_tg_blk)
               # C is (m_tg_blk x n_tg_blk)
```

After this step we have the following blocks:
- Let statements to unpack M/N/K indices from the kernel grid
- IR statements for the explicit reduction loops (inside a thread)
- A/B/C views describing thread group level blocked matrix multiplication
    - These views contain sizes, M/N/K semantics, access masks and the
      underlying layout

With these blocks the representation is generic enough so that all further
steps in the flow are common between different propagation kinds.

## SLM Buffering, Loads, Blocked Multiplication, Epilogue and Final Store

`compute_builder_t` is responsible for generation of the innermost blocked
computation and the final store of tensor C. According to `config_t` object the
builder generates the following kernel parts:

- SLM loads and stores (when SLM buffering is enabled):
    - Define SLM layout. Use FMA-friendly layout, if reorders are necessary,
      perform them earlier, between loads from global memory and stores to SLM
    - Split a view between all threads in the thread group for cooperative loads/stores
        - Sometimes only part of the thread group should participate in
          loads/stores - use thread group sub-grid and conditions to guard loads/stores
    - Add barriers before and after stores to SLM
- Loads from SLM/global memory and multiplication
    - Split A/B thread group views across the thread group grid (X x Y x 1)
        - Split M dimension across Y dimension of the grid
        - Split N dimension across X dimension of the grid
        - Each thread computes `(m_thr x n_thr)` tile of C tensor
            - `m_thr = m_tg / Y`
            - `n_thr = n_tg / X`
    - Split per-thread blocked multiplication to subtiles according to the
      configuration (to reduce GRF consumption and reuse GRF buffers)
        - Typically `b_subtiles > 1` is used (B tile is split into subtiles)
    - Generate loads (from SLM or global memory) for A/B tensors
    - Generate GRF-to-GRF reorders (when needed) to match FMA layout. This is
      mainly needed for dpas.
    - Generate IR function calls matching FMA instructions
    - Apply dpas to dpasw transformation (if possible). Save GRF permutation
      `grf_permutator_t` to restore registers back after applying dpasw. This
      permutation will be applied to the final C result.
    - Restore per-thread C tensor in terms of problem layout/dimensions
        - Per-thread C tensor is `(m_thr x n_thr)` but the way M/N dimensions
          are mapped back to the problem dimensions completely depends on how A/B
          were split across the grid. `mnk_mapper_t` is used to track that.
- Epilogue (`epilogue_builder_t`) in case when convolution includes bias, post-ops or output scales
    - Bias/post-ops/output scales are handled similarly by `post_op_builder_t`
    - C is split into blocks to apply post-ops
    - Flow for applying post-ops is:
        - Pre-load the corresponding blocks of right-hand side tensors to GRF
          (for binary post-ops or for bias/output scales)
        - Convert C to `f32`
        - Generate IR statements to handle all post-ops step by step
        - Convert the updated C to the final data type
- Final stores to global memory
    - Generate stores (maybe atomic stores, for partial reduction)

## More Optimizations and IR Passes

At this step the kernel is functionally correct. Now, more transformations and
optimizations need to be applied, some of which are listed in `jit/ir/README.md`.
