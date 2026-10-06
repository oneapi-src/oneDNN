Level-3 BLAS Support
====================

gemmstone can generate kernels for the full set of level-3 BLAS routines:

* GEMM: general * general matrix multiplication
* GEMMT family: general * general matrix multiplication, updating only the lower or upper triangle of C. As special cases, GEMMT includes:
  - SYRK: $B=A^T$.
  - HERK: $B=A^H$.
  - SYR2K: $C \leftarrow \alpha (AB^T+BA^T) + \beta C$. This can be implemented as a series of two GEMMT operations.
  - HER2K: $C \leftarrow \alpha AB^H + \overline\alpha BA^H + \beta C$. Sequence of two GEMMT operations.
* SYMM: Like GEMM, but one of A or B is a (real or complex) symmetric matrix, and only its upper or lower triangle is stored in memory.
* HEMM: Like SYMM, but with a Hermitian matrix.
* TRMM: Like GEMM, but one of A or B is a triangular matrix, and the other matrix is overwritten by the matrix product.
* TRSM: Solves a triangular system of equations $AC=B$ or $CB=A$ for matrix $C$, writing $C$ back to $B$ or $A$.

This document covers some of the algorithms used for the non-GEMM routines, and other MKL-specific functionality.

> [!NOTE]
> gemmstone always uses A to refer the matrix on the left of the multiplication, B for the matrix on the right, and C for the product.
>
> Note this differs from the BLAS naming conventions for SYMM/HEMM/TRMM/TRSM, where A is the special format matrix. Internally, gemmstone uses U to refer to the special format matrix, and V to refer to the general matrix in these routines.
>
> For TRSM, gemmstone's naming comes from interpreting the solve as a multiplication: $A^{-1}B=C$ (left-side TRSM) or $AB^{-1}=C$ (right-side TRSM).


GEMMT (SYRK, HERK, etc.) Kernels
--------------------------------

In most respects, a GEMMT kernel is identical with a GEMM kernel. There are two points of modification:
* Only workgroups intersecting the relevant triangle (either lower or upper) of C need to execute.
* Only the lower or upper triangle of C needs to be computed and written out.
  - For HERK/HER2K, the imaginary part of the diagonal additionally must be manually set to zero.

Ideally, a GEMMT of size `m` x `n` x `k` takes approximately 1/2 the time of the corresponding GEMM. In practice, there is some overhead because the triangular C matrix must be covered by rectangular C tiles, and also because A/B data reuse is reduced near the upper-left and lower-right corners of the triangle. 
 
As GEMMT kernels are so similar to regular GEMM kernels, the standard GEMM strategies (in the `gemm.db` catalog) are reused for GEMMT kernels.


### Workgroup Tile Selection

The C triangle selection is controlled by `GEMMProblem::updateType`. There is a check early within the kernel (prior to the mainloop) to see if the workgroup C tile is completely outside this C triangle, and if so, exit. That way, we can dispatch a rectangular grid of workgroups, as for GEMM, and skip the out-of-bounds workgroups, with only a small amount of overhead.

When persistent threading is enabled, such out-of-bounds workgroups can cause extreme load balance issues, where some workgroups have many more in-bounds tiles than others. In this case, it's recommended to use gemmstone's Hilbert-like walk order (`GEMMStrategy::cWalkOrder = WalkOrder::Hilbertlike`), which will iterate only through in-bounds workgroup tiles, using a recursive algorithm. (If a GEMM catalog entry uses panel boustrophedon instead of Hilbert-like walk order, it will automatically be converted to Hilbert-like for a GEMMT kernel).


### C Update

For thread tiles crossing the diagonal of C, the only supported path is the alternate C remainder path. This is enforced automatically during strategy preflight. The alternate C remainder path will use masked loads/stores to restrict C accesses to the proper triangle.


SYMM/HEMM
---------

gemmstone supports no-copy and copy-based implementations for SYMM/HEMM.

In a copy-based implementation (standard on CPU), both A and B matrices are packed, and data in the symmetric/Hermitian matrix is mirrored across the diagonal in the process. A regular GEMM kernel is used for computation. The only difference then from copy-based GEMM is a special copy kernel for the symmetric/Hermitian matrix, which gemmstone supports via `copyls/copyus/copylh/copyuh` copy kernels.

On GPU, a no-copy implementation is generally more efficient, though also more complicated since compute kernels must read symmetric/Hermitian matrices in place. In gemmstone, this is handled by splitting the k loop in three parts: lower triangle, diagonal block, and upper triangle (implementation: [`symmKLoop`](../../generator/pieces/symm.cxx)). As an example, taking the A lower triangular (left lower) case, a thread's A panel will cross through the lower triangle, then diagonal block, then upper triangle (where we reflect and read the lower triangle):

<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/a8972211-434d-4c3c-821c-d3dcd6d845a2" />

Note that the bounds of these k loops varies between threads within a workgroup. For this reason, special care is needed when the k loop has workgroup-cooperative operations (periodic barriers, prefetches, SLM pipelines, etc.). Support for SLM pipelines is particularly involved, and not yet supported in gemmstone &mdash; this would require additional k-loop segments around the diagonal.

In general, we need different A read operations in the upper and lower triangle. Continuing the lower triangular A example above, if A and C are both column-major, then block loads are needed for the lower triangle (non-transposing), but scattered loads for the upper triangle. This has effects throughout the gemmstone code, because the layouts of A/B tiles in memory are derived from the message type. Because of this, gemmstone sets up registers, layouts, prefetching, etc. to accomodate both regular A loads and reflected A loads. State variables with `Alt` in their name (e.g. [`GEMMState::A_alt`](../../generator/pieces/state.hpp)) correspond to the reflected loads.

The diagonal block in SYMM/HEMM requires special attention, because we need to reflect the A/B matrix across the diagonal. Following gemmstone conventions, let U be the symmetric/Hermitian matrix, and V the other matrix. There are two cases:

* Easy case: U is the broadcast matrix (i.e., B matrix if C is column-major; A matrix if C is row-major). Each FMA picks a single element from U, so we can handle reflection/conjugation inside the FMA. This is handled by `outerProduct`.
* Hard case: U is the vectorized matrix (A matrix if C is column-major). Reflection/conjugation of the U tile must occur prior to FMAs. In gemmstone, reflection is handled by using scattered loads with reflected addresses. Care is required to properly set up and maintain these addresses (see [`symmUIncrement`](../../generator/pieces/symm.cxx)). For HEMM, we also need to conjugate the reflected portion of the U tile in registers prior to the computation, which is performed by [`hemmPreprocessU`](../../generator/pieces/symm.cxx).

To facilitate reflection handling, the k loop for the diagonal block is completely unrolled. k remainders need to be checked only on the m/n remainder paths.

SYMM/HEMM kernel strategies are stored in a separate kernel database, [symm.db](../../selector/db/symm.db), because of potentially increased register pressure (relative to plain GEMM), as well as some implementation limitations. The main limitations of the current implementation are:
* No block 2D support
* No SLM pipelines for A/B

When converting GEMM strategies to SYMM/HEMM strategies, sometimes strategies can run out of registers due to extra register pressure in the diagonal block k-loop. A/B tile sizes can be reduced for the diagonal block only using the `uaN`/`ubN` strategy modifiers.


TRMM
----

gemmstone currently implements only copy-based TRMM, where both A/B are copied and packed into temporary memory prior to computation. There are three main differences compared to copy-based GEMM:
1. The copy kernel for the triangular matrix writes zeros to the unoccupied triangle.
2. The compute kernel restricts the k range dynamically to the actual range covered by the triangular matrix.
3. To allow for large-scale blocking in m/n dimensions (multiple kernel launches), the compute kernel dynamically adjusts beta depending on whether it handles the diagonal: 0 if this kernel's k range crosses the diagonal of triangular matrix (initial write) and 1 otherwise (accumulate with prior computations).

Note that (2) implies that in (1) it's not necessary to write zeros to the full unoccupied triangle, only to the parts that will be read during computation. Taking the left lower case as an example:

<img width="526" height="442" alt="image" src="https://github.com/user-attachments/assets/5eee66c8-3901-4080-8c30-fc5f43bbabb3" />

To facilitate this optimization, the copy kernel needs to be aware of the workgroup tile size of the compute kernel. This is configured by `CopyProblem::targetWG` (`twgN` strategy modifier). TRMM compute kernels are configured by setting `GEMMProblem::trxmMode = TRXMMode::TRMM`, along with proper `GEMMProblem::{left,lower}` settings.

Copy-based TRMM does not perform well on GPU and should be replaced by no-copy or partial no-copy algorithms. See the TODO list for a brief discussion of potential future improvements.


TRSM
----

Triangular solves are the most complex functionality in the level-3 BLAS, due to internal data dependencies in the calculation.

At its heart, a TRSM kernel looks like a more-or-less normal GEMM k loop, followed by a small triangular solve on a fixed size block. To see why, let's consider a lower left TRSM, solving $AC=B$, and divide it into two blocks:

<img width="768" height="315" alt="image" src="https://github.com/user-attachments/assets/3fef5f99-fd97-4929-98cc-a9cd7517b160" />

Solving for the blocks $C_1,\, C_2$:

```math
\begin{align}
  C_1 &= A_{11}^{-1}B_1,\\
  C_2 &= A_{22}^{-1}\left[B_2 - A_{12}C_1\right].
\end{align}
```

In other words, once we solve the first $k$ rows of a TRSM problem ($C_1$), solving the next $k'$ rows is achieved via matrix multiplication ($A_{12}C_1$) followed by an $k'$-sized triangular solve ($A_{22}^{-1}$).

In this way we can reduce a large TRSM into GEMMs plus TRSM operations on fixed-size diagonal blocks of $A$. A typical TRSM kernel maintains a fixed-size C tile, and starts by performing a normal GEMM accumulation with the previously-solved C data. After the GEMM, the new block is solved and written to global memory for use in further blocking steps.

Similar to GEMM, we can parallelize the computation in two dimensions:
* Rectangular outer dimension ($y$ dimension in gemmstone terminology). In the example above, this is $n$, the column dimension of $C$. TRSM is embarassingly parallel in $y$ since there are no data dependencies.
* Triangular outer dimension ($x$ dimension). Due to the data dependencies, this dimension is both much harder and less efficient to parallelize.

gemmstone supports both thread and workgroup-level parallelization for the $y$ dimension, but only parallelization within a workgroup for the $x$ dimension.

### Tile-Level TRSM

Calculating a tile-level TRSM boils down to applying the TRSM blocking formula described above, with blocks of size 1. Matrix inversion becomes scalar inversion (a no-op for unit triangles), and GEMM becomes a sequence of fused multiply-subtract operations.

The most efficient case for left-side TRSM is $y$-vectorization, where C is row-major in registers (for right-side TRSM, column-major). Diagonal elements of the triangle are associated with divides, non-diagonal elements with multiply-subtract. If $c_i$ is the $i^{\text{th}}$ row of $C$ (column for right-side), then the sequence of vector operations looks like

```math
\begin{align}
c_1 &\leftarrow c_1 / a_{11}\\
c_2 &\leftarrow c_2 - a_{21}c_1\\
c_3 &\leftarrow c_3 - a_{31}c_1\\
    &\vdots\\
c_2 &\leftarrow c_2 / a_{22}\\
c_3 &\leftarrow c_3 - a_{32}c_2\\
    &\vdots\\
c_3 &\leftarrow c_3 / a_{33}\\
    &\vdots
\end{align}
```

$x$-vectorization is also possible &mdash; C column-major in registers (for right-side TRSM, row-major) &mdash; but less efficient, and requires masked vector operations. gemmstone automatically decides which method to use based on the layout of C in registers; however, only $y$-vectorization is used in production, and the $x$-vectorization path is not well-tested. As a result of this, scattered stores (`aS`) are used for left-side TRSM strategies in the catalog.


### Thread-Level $x$-Parallelization

gemmstone supports thread-level $x$-parallelization, enabled via `GEMMStrategy::xParallel`/`xp`. The GEMM section of the k-loop is unaffected by parallelization, but the solves require much more effort. In particular, we have data dependencies where some threads' GEMMs require data from other threads' TRSM solves. Accordingly, gemm breaks up the TRSM k loop into three phases:
* `KLoop::TRSMMainGEMMXParallel`: all threads performing GEMM, no data dependencies (M in picture)
* `KLoop::TRSMExtraGEMMs`: section of GEMM computation that is dependent on data from other threads performing TRSM (E)
* `KLoop::TRSMSolve`: TRSM tile solve (T)

<img width="720" height="302" alt="image" src="https://github.com/user-attachments/assets/ed59b0fa-8f6e-4aaa-9ed1-2e106d4635c4" />

By default, the data from TRSM solves is communicated between threads via global memory only, but by enabling `GEMMStrategy::slmVSolve`/`cv`, it will be shared over SLM to reduce communication latency. Barriers are used for synchronization, and named barriers (`nb MxN`) can be used to allow each $y$-block of threads to operate independently. Furthermore, it can be helpful to reorder work-items (`ql`) so that a $y$-block of threads is scattered across EUs. Note that `ql` can reduce GEMM performance in some cases and thus may be beneficial only for smaller matrices.


### Outer Loop

Since each workgroup only covers a fixed number of rows/columns in the $x$ dimension, we typically need to loop over this dimension in the kernel to calculate the full TRSM. This outer $x$ loop is part of the `trsm` top-level routine.


### Copy-Based and No-Copy TRSM

gemmstone supports both copy-based and no-copy TRSM implementations. Like other level-3 BLAS routines, no-copy TRSM usually provides better performance.

In a copy-based TRSM kernel:
* A/B data is prepacked in separate copy kernels. Additionally, the triangular matrix's diagonal is pre-inverted (also negated) to reduce calculation in the compute kernel.
* The compute kernel writes solved data back to both the $C$ matrix and the temporary packed rectangular matrix.

In a no-copy TRSM kernel:
* A/B data is read in place.
* During the GEMM phase of the kernel, previously solved C data takes the place of A/B data (orchestrated by [`trsmMoveCToV`](../../generator/pieces/trsm.cxx)).

gemmstone's no-copy TRSM kernels support both in-place (normal TRSM) and out-of-place (TRSM3) operation, where the result matrix is written to a separate matrix.


### Implementation Limitations

gemmstone's TRSM implementation supports most GEMM strategy options, including SLM pipelines for both A and B. However, it does not currently support block 2D messages.


Complex Kernels
---------------

When A/B/C are complex-valued, there are several algorithmic options for handling the complex-valued product
```math
\begin{align}
(C_r + iC_i) &= (A_r + iA_i)\cdot (B_r + iB_i) \\
             &= (A_rB_r - A_iB_i) + i(A_rB_i + A_iB_r).
\end{align}
```

- **4M** (split-complex): Directly implement the standard complex multiplication equation. To perform this efficiently in hardware, real and imaginary parts of C must be split and placed in separate registers. Also the vectorized matrix (A if C is column-major, B if not) must also be split. "4M" refers to the 4 real multiplies.
- **4M** (interleaved): Accumulate the 4 real products $A_rB_r$, $A_rB_i$, $A_iB_r$, $A_iB_i$ separately and combine them only after the main loop. Real and imaginary parts of C and the vectorized matrix will remain interleaved in registers, matching their layout in global memory. The k loop of the kernel looks like a regular real-valued GEMM.
- **3M**: Rewrite the complex multiplication formula so that only three real multiplications $A_rB_r$, $A_iB_i$, $(A_r+B_r)(A_i+B_i)$ are necessary. These 3 real products are accumulated in the k loop, and combined to form the complex product after the loop completes:

```math
C = (A_rB_r - A_iB_i) + i((A_r + B_r)(A_i + B_i) - A_rB_r - A_iB_i).
```

By default, the interleaved 4M algorithm is used. Split-complex 4M is enabled using the `xc` strategy modifier, and 3M is enabled with `3m`. When the systolic array is enabled, only split-complex 4M is supported. 3M can be supported with systolic, but this is not yet implemented. 

Complex-valued level-3 BLAS APIs usually allow A or B to conjugated, generally in connection with transposition. In gemmstone, transposition is handled as part of the memory layout, while conjugation of A/B is independently configured via flags passed to the kernel.

More details on each algorithm's implementation in gemmstone can be found in the following sections:


### Split-Complex 4M

To support split-complex GEMMs, gemmstone has special machinery to handle split real/imaginary storage, both in registers and SLM.

In registers, gemmstone distinguishes three varieties of complex register data:

| `ComplexMode` | Description | `RegisterBlock::splitComplex` | `RegisterBlock::cxComponent` |
| - | - | - | - |
| `Interleaved` | alternating real/complex elements | `false` | `-1` |
| `Split` | alternating rows/columns of real and complex data within each `RegisterBlock` | `true` | `-1` |
| `FullySplit` | real and complex data stored in separate `RegisterBlock`s | `false` | `0` (real), `1` (imag) |

Loading complex data from memory is governed by `MatrixAddressingStrategy::cmode`, which is also a `ComplexMode`. Some send messages (e.g. `AccessType::Block`) naturally work interleaved data in registers, while others (e.g. `AccessType::Block2DTranspose`) naturally produce split-complex data, and some (e.g. `AccessType::Scattered`) can do either. In addition, `cmode` may be `ComplexMode::Any` to allow automatic selection. Note that `GEMMStrategy::C.cmode` should be set to `Interleaved` to enable split-complex 4M. 

When SLM pipelines are in use, A/B data may need to be stored split-complex in SLM. gemmstone expresses this with dedicated split-complex types, e.g. `Type::scf{16,32,64}`, which only work with packed layouts.

Split-complex GEMMs need runtime dispatching of the k loop in order to handle optional conjugation of A and/or B. gemmstone synthesizes two loops, one for $A\cdot B$ and one for either $A\cdot\overline B$ or $B\cdot\overline A$, depending on which is more efficient. This covers two of the four conjugation cases; the other two can be handled by conjugating the C accumulator after the k-loop.


### Interleaved 4M

For interleaved 4M, gemmstone needs to keep two copies of the C tile in registers. If C is column-major (so FMA reads vectors from A and scalars from B), tile 0 holds products with $B_r$ and tile 1 holds products with $B_i$:

```math
\begin{align}
\texttt{state.C\_regs[0]: } & A_rB_r\quad A_iB_r\quad A_rB_r\quad A_iB_r \ldots\\
\texttt{state.C\_regs[1]: } & A_rB_i\quad A_iB_i\quad A_rB_i\quad A_iB_i \ldots
\end{align}
```

For C row-major, A/B are swapped.

After the k-loop, [`complexCombineC`](../../generator/pieces/c_complex.cxx) is responsible for swizzling and combining the parts with proper signs, depending on whether A/B are conjugated, leaving the result in tile 0.

During C update, tile 1 is sometimes used temporarily in order to handle scaling by complex-valued alpha/beta. `GEMMState::cSwapActive` tracks whether tile 1 is in use in this way.


### 3M

The current 3M implementation uses two C tiles, though it only uses the real part of tile 0, since only three real accumulators are needed. The $A_r + B_r$ and $A_i + B_i$ sums are done on-the-fly as part of [`outerProductComplex3m`](../../generator/pieces/matrix_multiply.cxx). After the k loop the final result is computed by [`complex3mCombineC`](../../generator/pieces/c_complex.cxx).

The ideal speedup for a 3M GEMM is 4/3. In practice, actual speedup is somewhat smaller, due to the overhead of $A+B$ additions and the reduced tile sizes (compared to a 4M split-complex GEMM).

Future optimization possibilities include:
* Reducing C register usage by only storing the 3 real accumulators.
* Support systolic array usage and multicomponent GEMMs (bf16x2/bf16x3/tf32x2).
* Reduce cost of adding $A_r + B_r$ and $A_i + B_i$ by sharing the work among threads when using an SLM pipeline.


### Mixed Real/Complex Types

gemmstone also allows mixed-complex kernels (FMA only), where one of A/B is complex and the other real. Aside from the layout of elements in memory, these kernels are equivalent to purely real GEMMs.
