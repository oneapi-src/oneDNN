# IR-Based JIT Kernel Generation for x64

This directory holds an intermediate representation (IR) for x64 JIT CPU kernels
and the shared passes that lower it to machine code. A kernel is written as
target-neutral operations over virtual registers. The shared pipeline allocates
registers and an ISA-specific emitter lowers the IR to instructions. Kernel code
carries no manual register management and no data-type or ISA branching.

IR-based kernels coexist with the existing Xbyak generators and are enabled
incrementally. It is not an optimizing compiler, has no IR-to-IR passes, and
does not target non-x64 backends.

## Architecture

The kernel is a class derived from `jit_generator_t`. It knows the concrete ISA
and data types and can use them directly, for example to set up post-op
callbacks. Its `generate()` runs the builder and passes the IR to
`generate_kernel()`, which runs the shared stages below in order. Only the
*builder* stage is kernel-specific.

```
generate()
|
|   kernel-specific
+-> +-------------+
|   | Builder     |  emits target-neutral IR: param loads, loops, math
|   +-------------+
|
|   shared, generate_kernel()
+-> +-------------+
|   | Reg config  |  builds ISA-agnostic register pools
|   +-------------+
|
+-> +-------------+
|   | Reg alloc   |  liveness and linear scan over gpr/vec/mask
|   +-------------+
|
+-> +-------------+
|   | Emitter     |  lowers IR to instructions, one backend per ISA
|   +-------------+
|
+-> +-------------+
    | Static data |  mask and post-op tables, after the postamble
    +-------------+
```

The ABI preamble, the spill-frame reservation, and the postamble wrap the emit
step. They are omitted above for clarity.

### Components

* **Builder.** The only kernel-specific part. It loads parameters, builds the
  loop nest, and expresses the math in target-neutral operations. It is data-type
  agnostic (the type is only a tag on vector values) and ISA agnostic (masks,
  vector, and general-purpose registers are abstract).
* **IR.** A linear list of operations over *virtual registers*. One fixed
  struct represents every operation, and an `op` kind selects which fields it
  uses. Every virtual register has a kind (`gpr`, `vec`, or `mask`), and a `vec`
  register also carries the element data type it holds. A vector load or store
  carries a second data type, `mem_dt`, for what memory holds. The two differ
  when the access converts.
* **Register configuration.** An ISA-aware step that produces ISA-agnostic
  register pools (integer indices per register kind) plus the reserved registers:
  the stack pointer and the kernel-argument pointer. It encodes per-ISA facts
  such as register counts and whether the target has dedicated mask (k)
  registers.
* **Register allocator.** Maps unlimited virtual registers onto physical ones,
  spilling to the stack under pressure. It knows only register kinds and control
  flow. Liveness analysis (which values are still needed at each operation) is
  backward data-flow over the IR's trivial control-flow graph, iterated to a fixed
  point so loop back-edges propagate. Linear scan then
  reduces each value to a single live interval and spills to the stack when the
  active set outgrows the register file. A spilled value still needs a register
  while an operation reads or writes it, so the scan also gives each operation a
  temp register for every spilled operand. The temps per kind (2 gpr, 3 vector)
  match the widest operation of that kind, so only `inject_postops`, which takes
  any number of accumulators, can run out. Masks get no temps and are never
  spilled.
* **Emitter.** The only part aware of the ISA and data types, because it produces
  the code. It walks the allocated IR once and lowers each operation to
  instructions using the physical registers the allocator chose. Spilled values
  are loaded into their temps around each use. There is one backend per ISA
  family (for example, AVX2\* and AVX-512\*), and a dispatch step selects the
  matching backend.
* **Static data.** Some lowerings need constants, such as the AVX2 mask tables,
  that must live after the ABI postamble, though the references to them are
  emitted during lowering. The emitter fills a data-section structure with the
  bytes and an unresolved label. After the postamble, the bytes are written and
  the labels are bound.
* **Post-ops injector.** The existing Xbyak post-ops injector is reused unchanged,
  plugged in as a single `inject_postops` operation. Its variable-length arguments
  live in a side table indexed from the operation's immediate field, so the
  operation itself carries only that index. The table also holds each
  accumulator's output offset, which a binary post-op needs to address its
  right-hand-side argument and a sum post-op needs to read the previous
  destination value. The injector saves and restores its own registers and does
  not participate in IR allocation.

### Control and Data Flow

The IR is produced in full, then consumed read-only by the allocator and the
emitter. `generate()` builds the IR and passes it to `generate_kernel()`, which
runs a fixed sequence: build the register configuration, allocate registers,
emit the ABI preamble, reserve the spill frame, emit the lowered code, tear down
the frame, emit the postamble, write the static data, and print the debug
output with `ir::print_kernel_dump()` (see Debug Output).

## Design Principles

* **The whole kernel exists before lowering.** This is what distinguishes the
  approach from single-pass generation and what makes global register allocation
  possible.
* **Flat, fixed-struct IR.** Instructions do not nest, and one struct fits every
  operation. This keeps the builder, allocator, and emitter simple and avoids any
  rewrite-pass machinery.
* **No IR-to-IR optimization passes.** The developer writes the structural
  optimizations (blocking, unrolling, loop order), which are emitted as written.
  The IR only takes over register allocation.
* **Mutable virtual registers.** A virtual register is a named value that may be
  written more than once (for example, a pointer advanced each
  iteration), and it occupies a single physical location for its entire live
  range. The cost is that live-range splitting is ruled out.
* **Single-register addressing.** A memory operand is a base register plus a
  build-time constant displacement, with no index and no scale. Any distance known
  only at run time is folded into the base pointer with an explicit `add`. Every
  memory operation then reads at most one pointer, which lowers register pressure
  and simplifies spilling.
* **Structured loops.** Loops are explicit nodes with a runtime counter.
* **Separation of concerns is the invariant to protect.** The builder is
  target-neutral, ISA and data-type knowledge lives only in the emitter and the
  register configuration, and the allocator knows only kinds and control flow.
* **`def_use()` must match the emitter.** Liveness is computed from the reads and
  writes reported by `def_use()`. If a lowering reads or writes a virtual register
  that `def_use()` does not report, allocation is wrong, so the two must stay in
  sync.

## Directory Structure

* `ir.hpp`, `ir.cpp`: the IR itself, that is, the operation kinds, virtual
  registers, the builder helpers, `def_use()`, and the loop-emission helpers.
* `reg_config.hpp`, `reg_config.cpp`: builds the per-ISA register configuration,
  that is, the allocatable pools plus the reserved registers.
* `reg_alloc.hpp`, `reg_alloc.cpp`: liveness analysis and the linear-scan
  allocator, producing the assignment for each virtual register and the size of
  the spill frame.
* `emitter/emitter.hpp`, `emitter/emitter.cpp`: the lowering pass, which walks the
  allocated IR, dispatches by ISA family, and manages the static-data section.
* `emitter/backend_avx2.hpp`, `emitter/backend_avx512.hpp`: the per-ISA-family
  backends, which do the per-operation instruction selection. They are peers,
  each self-contained.
* `postops_injector.hpp`, `postops_injector.cpp`: the driver for the JIT post-ops
  injector, which lowers the `inject_postops` operation.
* `codegen.hpp`, `codegen.cpp`: `generate_kernel()`, which runs every stage
  after the builder.
* `dump.hpp`, `dump.cpp`: the debug output that `ONEDNN_VERBOSE=x64ir` enables
  (see Debug Output).

The kernel-specific builders live outside this directory. For example,
`src/cpu/x64/brgemm/brgemv_ir.{hpp,cpp}` holds the GEMV builder and shows how
`generate()` passes the IR and the post-ops inputs to `generate_kernel()`.

## Developer Guidelines

New functionality maps to a specific layer. The IR infrastructure grows by adding
data types, ISAs, and the fused instructions each ISA supports, and each kind of
change belongs in one place.

1. **A new data type** is a tag on a `vec` register, plus the `mem_dt` on the
   loads and stores that reach it. The builder stays the same, and the emitter
   maps the operation and the data types to the right instruction.
2. **A new ISA** adds a new emitter backend or extends an existing one, along with
   the matching register-configuration facts.
3. **A new variant of an existing instruction** is a lowering rule in the emitter,
   not a new IR operation, and is invisible to the builder. For example,
   `vload_masked` already lowers to `vmaskmovps` on AVX2 and to `vmovups` under
   an EVEX write mask on AVX-512.
4. **A new behavior** that no existing operation can express becomes a new IR
   operation, with its own definition, `def_use()` entry, and lowering.

When adding new operations conform to the following rules:

* **Keep parameters out of the name.** An operation takes its data types as
  parameters, so the name must not mention a data type. There is one `vdot`,
  not `vdot_f32` and `vdot_bf16`, and the emitter selects `vfmadd231ps` or
  `vdpbf16ps` from the operand's `dt`. The same applies to the ISA. If the
  operation cannot be named without naming a data type, it is a lowering rule
  for an existing operation, not a new operation.
* **Decide how many operations the family needs.** Related forms can be one
  operation whose parameters select the form, or one operation per form. A
  single load taking an element count and an optional mask would cover the
  full-vector, single-element, and masked cases, and it would work. The cost
  is that the meaning of a call depends on the combination of arguments, and
  some arguments are unused in some combinations. This IR uses separate
  operations: `vload`, `vload_scalar`, `vload_bcast`, and `vload_masked`. Each
  signature takes only the arguments its form needs. Both options are valid.
  Prefer separate operations, because they are harder to call incorrectly.
* **Check that the operation works for other data types and ISAs.** A new
  operation usually has one lowering at first, and it is tempting to name it
  after that instruction. `fma` would have been a reasonable name while the
  only lowering was `vfmadd231ps`. That name does not work for bf16, where
  `vdpbf16ps` adds two products per f32 lane, or for int8, where VNNI adds
  four. The operation is named `vdot` because all of these compute a dot
  product, and the length depends on the data type. Before adding an
  operation, look up the instruction you would emit for every ISA and data
  type you expect to support, and define the operation at the level where they
  agree. If they do not agree at any level, these are two operations.

Additional rules to follow.

* Keep the builder target-neutral. No ISA or data-type conditionals in builder
  code, since that is exactly the branching this design removes.
* Update `def_use()` with every change to an operation so liveness stays correct.
* Confine ISA and data-type knowledge to the emitter and the register
  configuration.
* Prefer a running pointer over a computed index, to respect single-register
  addressing.
* The public entry points in these headers are exported for unit testing.

## Tests

The IR, allocator, and emitter have dedicated unit tests
(`test_internals_cpu_ir`). IR-based kernels are also tested through benchdnn.

## Debug Output

The IR pipeline can print the IR of the kernel that it generates with a short
summary of the kernel.

The output is useful in these cases:

* **Checking a builder.** The IR dump shows the loops with their counters and
  iteration counts, the pointer increments (`add_imm`), the memory offsets, and
  the data types that the builder produced. A developer can compare them with
  what the builder code is supposed to produce.
* **Comparing two versions.** The output does not change from run to run. A
  `diff` of the output before and after a change shows how the change affected
  the kernel.

### Enabling the Output

The output can be enabled only in dev-mode builds (`ONEDNN_DEV_MODE=ON`). Add
the `x64ir` token to `ONEDNN_VERBOSE` to enable it, for example,
`ONEDNN_VERBOSE=x64ir` or `ONEDNN_VERBOSE=dispatch,x64ir`.
`ONEDNN_VERBOSE=all` and `debuginfo=` do not enable it. `all` enables the
standard verbose output, and backend-specific dumps does not mix with it.

The output is printed only when a kernel is generated. A primitive cache hit
generates no kernel, so it prints nothing.

### Output Format

The example below is a part of the output for a BRGEMM kernel. The lines
marked `...` are left out.

```
begin x64ir jit_brgemm_ir_kernel_t isa=avx512_core
code: 1339 bytes (instructions 1339 bytes, static data 0 bytes)

    0 | load g0, [param+24]
    1 | load g1, [param+16]
    2 | load g2, [param+96]
...
    9 | jz g8, L0
   10 | load g0, [param+40]
   11 | L0:
   12 | loop g9 = 2 {
   13 |   vzero f32:v10
...
   29 |   mov_reg g28, g1
   30 |   jz g2, L1
   31 |   loop g30 = g2 {
   32 |     load g31, [g28+0]
...
   39 |     loop g33 = 16 {
   40 |       vload f32:v26, f32:[g32+0]
   41 |       vload f32:v27, f32:[g32+64]
   42 |       vload_bcast f32:v29, f32:[g31+0]
   43 |       prefetch [g32+512]
   44 |       vdot f32:v10, f32:v26, f32:v29
...
  154 |     } // g33 -= 1, repeat while > 0
  155 |   } // g30 -= 1, repeat while > 0
  156 |   L1:
...
  177 |   vstore f32:[g0+0], f32:v10
...
  195 | } // g9 -= 1, repeat while > 0
end x64ir
```

The lines mean the following:

* `begin x64ir <name> isa=<isa>` starts the output for one kernel. `end x64ir`
  ends it.
* `code:` is the size of the kernel in bytes, the same size that
  `ONEDNN_JIT_DUMP` writes. It is split into the instructions and the static
  data.
* Each line of the IR dump has the operation index and the operation. An
  operation inside a loop is indented by two spaces for each loop around it.

An operation prints as its `op_kind_t` name, then its operands. The
destination comes first. Operands print as follows:

* `g<id>` and `m<id>` are gpr and mask virtual registers. `<id>` is the
  `vreg_t` value.
* `<dt>:v<id>` is a vec virtual register that holds the data type `<dt>`.
* `[g<id>+<disp>]` is the memory at `g<id>` plus a decimal byte offset.
  `[param+<disp>]` is a field of the kernel argument struct.
* `<dt>:[...]` specifies the in-memory data type of a vector load or store. A
  converting access shows two types, for example, `vload f32:v3, bf16:[g0+0]`.
* `L<id>` is an IR label, the target of `jmp` and `jz`.

Two kinds of operations print in a special form:

* `loop g9 = 2 {` is `loop_begin` with the counter `g9` and the iteration count
  `2`. The count can also be a virtual register, as in `loop g30 = g2 {`.
  `} // g9 -= 1, repeat while > 0` is `loop_end`.
* `L0:` is the `label` operation for the label `L0`.

## References

* Design document (RFC): *IR-Based JIT Kernel Generation for x64 CPUs*, at
  https://github.com/uxlfoundation/oneDNN/pull/5460
  (`rfcs/20260630-ir-x64/README.md`). It covers the motivation, goals, and
  design rationale.
