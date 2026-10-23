[//]: # (Base for all permalinks below: alexandrelimassantana/oneDNN@c2ea612b037f9bbe91bc087f53abed7b7a491209, the exact head commit of PR #5500 at time of writing.)

# `rvjit`: A Component Library for RV64 JIT Kernels

Related: [PR #5500](https://github.com/uxlfoundation/oneDNN/pull/5500)

## Introduction

[`rvjit`](https://github.com/alexandrelimassantana/oneDNN/tree/c2ea612b037f9bbe91bc087f53abed7b7a491209/src/cpu/rv64/rvjit) is a small set of code-generation components to produce reusable code segments as simple as branches and constant value additions, and as complex as auto-tuned matmul dense loops.
RV64 `jit_generator_t` objects can delegate code generation to `rvjit` components, leveraging flexible implementations of reusable patterns exposed by a descriptive API.
The library grew out of our personal experience working on a local oneDNN fork at BSC over 5 years, with the goal of producing vendor-agnostic code for diverse systems.

The proposal of `rvjit` starts from empirical evidence obtained during our research entitled "Just-in-Time Convolution Code Generation for Vector Architectures", published at [IPDPS'26](https://ssl.linklings.net/conferences/ipdps/ipdps2026_program/views/at_a_glance.html).
In this work, we find that optimal choices for variables such as [`LMUL`](https://github.com/alexandrelimassantana/oneDNN/blob/c2ea612b037f9bbe91bc087f53abed7b7a491209/src/cpu/rv64/rvjit/rvjit_types.hpp#L46) and `unrolling` are not necessarily the same across different processors.
In pursuit of configurable micro-kernels, we identified recurrent segments of code implementing concepts such as loops, or operations over register blocks widespread across oneDNN.
We studied how these concepts transformed under different parameters, derived from optimization regimes, and identified a way to represent these strips of code as reusable code emitters.
Ultimately, implementing these configurable components allowed us to compose them and materialize micro-kernels adaptable to different circumstances, be they variations of [`SEW`](https://github.com/alexandrelimassantana/oneDNN/blob/c2ea612b037f9bbe91bc087f53abed7b7a491209/src/cpu/rv64/rvjit/rvjit_types.hpp#L47) and `LMUL` or optimization variable values.

PR #5500 currently bundles the library itself together with three kernel migrations (gemm, brgemm, 1x1-conv) and its own gtest suite.
This RFC is meant to settle the open architectural questions first, so the follow-up PRs can be reviewed on their own merits.

## Proposal

`rvjit` is a tool for micro-kernel developers that automates the emission of software design patterns that are frequent in the context of oneDNN.
These patterns are exposed as configurable components, or code emitters, that use callbacks to switch between developer-supplied code blocks and `rvjit` pattern code generation.
The components are made fully configurable in order to account for micro-architectural variability across current and future processors.

The split of code-generation responsibility between the library and the caller is explicit and caller-determined, at two different levels.
First, the user chooses the components they want to use.
Highly-reusable components like [`control_flow_t`](https://github.com/alexandrelimassantana/oneDNN/blob/c2ea612b037f9bbe91bc087f53abed7b7a491209/src/cpu/rv64/rvjit/rvjit_control_flow.hpp) may be used directly.
In this case, the caller must supply register and loop unrolling parameters.
Some components encapsulate decision logic, such as [`register_pool_t::new_loop`](https://github.com/alexandrelimassantana/oneDNN/blob/c2ea612b037f9bbe91bc087f53abed7b7a491209/src/cpu/rv64/rvjit/rvjit_register_pool.hpp#L259), which allocates whatever registers a loop still needs.
At the API level, methods like [`rvv_matmul_engine_t::configure`](https://github.com/alexandrelimassantana/oneDNN/blob/c2ea612b037f9bbe91bc087f53abed7b7a491209/src/cpu/rv64/rvjit/rvjit_matmul.hpp#L145) derive unroll factors automatically from the `rvv_t` hardware model whenever the caller hasn't set them explicitly.
In other words, callers pick how much to delegate on a per-call basis.

### Relationship to Xbyak

`rvjit` does not replace or duplicate Xbyak.
[`jit_generator_t`](https://github.com/alexandrelimassantana/oneDNN/blob/c2ea612b037f9bbe91bc087f53abed7b7a491209/src/cpu/rv64/jit_generator.hpp#L104)
still inherits `Xbyak_riscv::CodeGenerator` directly and unchanged.
[`rvjit_t`](https://github.com/alexandrelimassantana/oneDNN/blob/c2ea612b037f9bbe91bc087f53abed7b7a491209/src/cpu/rv64/rvjit/rvjit.hpp#L41)
is built by composition: it takes a [`codegen_t&`](https://github.com/alexandrelimassantana/oneDNN/blob/c2ea612b037f9bbe91bc087f53abed7b7a491209/src/cpu/rv64/rvjit/rvjit_types.hpp#L41) (a `Xbyak_riscv::CodeGenerator&`),
and every component it owns holds an [`emitter_t`](https://github.com/alexandrelimassantana/oneDNN/blob/c2ea612b037f9bbe91bc087f53abed7b7a491209/src/cpu/rv64/rvjit/rvjit_types.hpp#L53) wrapping that same reference.
In this way, every component eventually defers code generation to Xbyak_riscv as if the primitive had called it directly.
`rvjit_t` never subclasses `CodeGenerator` and never emits an instruction Xbyak doesn't expose; it only packages up patterns (e.g. "zero an accumulator tile, run a pipelined
K-loop, store the result") that were previously copy-pasted across primitive kernels.

This isn't a new software design pattern for this codebase.
Injectors, such as [`src/cpu/x64/injectors/`](https://github.com/uxlfoundation/oneDNN/tree/main/src/cpu/x64/injectors)
(`jit_uni_eltwise_injector`, `jit_uni_binary_injector`, `jit_uni_postops_injector`), are cross-kernel JIT helpers composed into `jit_generator_t`-derived kernels the same way.
That is, a kernel holds an injector as a member and calls it to emit a reusable instruction sequence, rather than each kernel re-implementing eltwise/postop codegen by hand.

`rvjit`'s scope is broader; it covers optimizations like constant folding, control flow, register blocks, register allocation, and type-adaptable meta-instructions, which are everywhere in oneDNN.
These components are applied in PR #5500 to produce a universal matrix multiplication code-generation engine, capable of adapting code emission to available GPR, VR, FPR resources, application/optimization parameters, and loop strategies, from a series of decisions taken by components at different abstraction levels.

### Relationship to the x64 IR proposal

Both proposals target the same underlying issues with Xbyak-based kernels: duplicated boilerplate, and data-type/ISA branches scattered through the kernel body.
The optimizations are structural (loop order, blocking, unrolling), decided mainly by the kernel author, though `rvjit` can optionally derive some of them automatically.
Both are also opt-in, coexisting with the existing per-kernel generator model.
The mechanisms diverge from there.

#### Difference 1: implementation

[PR #5460](https://github.com/uxlfoundation/oneDNN/pull/5460) builds a complete intermediate representation — a flat list of fixed-size instructions over virtual registers — before any machine code is emitted; register allocation (global liveness analysis, then linear scan) and instruction lowering are separate passes that run only once the whole kernel is known.
`rvjit` has no equivalent intermediate structure and no build-then-lower staging: every component call emits real `Xbyak_riscv` instructions immediately, synchronously, as the call happens.
What plays the IR's structural role in `rvjit` is the nested hierarchy of lambda callbacks at C++ call time itself — components hand control to caller-supplied callbacks and to each other directly, rather than recording anything for a later pass to walk.

#### Difference 2: target problem

The x64 IR's two-stage design exists to give an automatic register allocator a global view.
The `rvjit` component for register allocation, [`register_pool_t`](https://github.com/alexandrelimassantana/oneDNN/blob/c2ea612b037f9bbe91bc087f53abed7b7a491209/src/cpu/rv64/rvjit/rvjit_register_pool.hpp), is deliberately simpler.
It implements a forward-allocating cursor with no liveness analysis and no spilling, because `rvjit`'s primary goal is cross-microarchitecture portability: reusing patterns and logic to determine and adapt optimization values such as `LMUL`/`SEW`/unrolling choices, rather than solving register pressure through global allocation.

#### Difference 3: meta-instructions and lowering

The x64 IR's builder is written once, fully data-type- and ISA-neutral, with the instruction choice deferred to a later point.
`rvjit` resolves data-type dispatch immediately at each call site (e.g. [`memory_move_t::vle(vd, addr, dt)`](https://github.com/alexandrelimassantana/oneDNN/blob/c2ea612b037f9bbe91bc087f53abed7b7a491209/src/cpu/rv64/rvjit/rvjit_memory_move.hpp) picks its mnemonic on the spot) — there's no deferred, whole-kernel lowering step, since every `rvjit` call already knows its concrete dtype/ISA context when it runs.
Support for other codegen strategies, such as portability across vector and matrix ISAs, is designed to be integrated in the future as alternative component instances (e.g. generalizing the existing `rvv_matmul_t` for RVV into a `matmul_engine_t` also implemented by a new `ime_matmul_t` for IME).

#### Difference 4: caller/library boundary

The x64 IR has one fixed boundary: the builder is entirely target-neutral, and everything past it (allocation, lowering) is entirely automatic and shared.
`rvjit` instead exposes a continuum of boundaries, chosen per component and per call (see "Proposal" above) — from `control_flow_t`, which a caller drives directly, to `rvv_matmul_engine_t::configure`, which derives its own unroll factors unless the caller overrides them.

### Dependency model

`rvjit` is opt-in at the kernel level and the `rvjit` abstractions are completely detached from other oneDNN structures.
Our proposed integration touches 3 kernels:
[`gemm/jit_rvv_gemm_kernel.cpp`](https://github.com/alexandrelimassantana/oneDNN/blob/c2ea612b037f9bbe91bc087f53abed7b7a491209/src/cpu/rv64/gemm/jit_rvv_gemm_kernel.cpp#L55-L58),
[`brgemm/jit_brgemm_kernel.cpp`](https://github.com/alexandrelimassantana/oneDNN/blob/c2ea612b037f9bbe91bc087f53abed7b7a491209/src/cpu/rv64/brgemm/jit_brgemm_kernel.cpp#L74-L78),
and
[`jit_rvv_1x1_conv_kernel.cpp`](https://github.com/alexandrelimassantana/oneDNN/blob/c2ea612b037f9bbe91bc087f53abed7b7a491209/src/cpu/rv64/jit_rvv_1x1_conv_kernel.cpp#L210-L214).
There is no reference to `rvjit` anywhere in
[`cpu_isa_traits.hpp`](https://github.com/alexandrelimassantana/oneDNN/blob/c2ea612b037f9bbe91bc087f53abed7b7a491209/src/cpu/rv64/cpu_isa_traits.hpp)
or
[`src/cpu/rv64/CMakeLists.txt`](https://github.com/alexandrelimassantana/oneDNN/blob/c2ea612b037f9bbe91bc087f53abed7b7a491209/src/cpu/rv64/CMakeLists.txt)
— every other RV64 kernel builds, registers, and runs exactly as it did before
`rvjit`.

Going forward we'd like to keep it that way explicitly: adoption should be
gradual and justified case by case, not required. Concretely:

- New RV64 kernel contributions are **not** expected to use `rvjit`.
- Existing kernels are refactored to use `rvjit` if it provides a measurable
  benefit in performance or code quality.

Matrix multiplications constitute the main target already covered in the proposal.
The migration of `brgemm` alone
([diff](https://github.com/alexandrelimassantana/oneDNN/commit/3b28251765))
results in a net deletion of ~1,535 lines from
[`jit_brgemm_kernel.cpp`](https://github.com/alexandrelimassantana/oneDNN/blob/c2ea612b037f9bbe91bc087f53abed7b7a491209/src/cpu/rv64/brgemm/jit_brgemm_kernel.cpp), while retaining all functionality.
The primitive is also extended with runtime-configurable optimizations, increasing flexibility to support other architectures (such as long vectors).
The optimization values are further refined using the empirical values we obtained from our research, resulting in performance improvements measurable on RV64 platforms previously used to evaluate oneDNN.

### Alternatives considered

- **Extend the existing Xbyak-based approach directly**, i.e. add
  helper functions to `jit_generator_t` or to individual kernel files,
  without a dedicated component library. Pros: no new abstraction layer to
  review or maintain. Cons: the `jit_generator_t` aggregates the different
  responsibilities that are now split between components.

We believe a shared component library composed on top of Xbyak — the
`rvjit` proposal — is the best balance. It keeps Xbyak as the sole
instruction emitter, removes duplicated code, and allows a flexible
degree of adoption as the developer decides which components to use.

### API and ABI backwards compatibility

`rvjit` is entirely internal to `dnnl::impl::cpu::rv64::rvjit`, compiled
only when `DNNL_TARGET_ARCH=RV64`, and is not exposed through any public
oneDNN header, the C API, or the C++ API wrappers. This proposal changes no
public symbol, struct layout, or documented primitive interface. Since
oneDNN's semantic versioning covers the public API/ABI surface, this
proposal has no impact on it and requires no version bump beyond the
ordinary release cadence.

### Build system changes

The proposal's build-system footprint is limited to the RV64-only build:

- `rvjit/` is header-only, so no new object-library target is required for
  the library itself.
- The three kernel migrations (`gemm`, `brgemm`, `1x1-conv`) add no new
  CMake targets; they only change which headers the existing kernel `.cpp`
  files include.
- The new component will be supported by a new gtest suite, using a testing
  strategy yet to be determined (See **Open Questions**).
- No changes to the default (non-RV64) build, and no changes to
  `ONEDNN_BUILD_TESTS`/`ONEDNN_BUILD_EXAMPLES` or other cross-cutting CMake
  options.

### Dependencies and support matrix

`rvjit` introduces no new external dependencies; it's built entirely on the
already-vendored `third_party/xbyak_riscv` and on oneDNN's existing
`dnnl::impl` utilities (e.g. `platform::get_cache_line_size()`). It only
affects the RV64 target (`DNNL_TARGET_ARCH=RV64`); the support matrix for
x64, AArch64, and other backends is unchanged, and none of their build
outputs link against `rvjit`.

`rvjit` introduces a flexible optimization scheme that classifies implementations across short-, mid-, and long-vector processors.
We will expand QEMU correctness coverage across VLEN settings of 128-bit and 256-bit in addition to 1024-bit and 4096-bit to stress each of these hardware categories.

The first `rvjit` PRs introducing the matmul kernel refactors will include the default benchdnn correctness coverage for the migrated primitive, plus performance tests derived from a list of problem shapes obtained from open-source AI model implementations in PyTorch via tracing oneDNN library calls.
We will contribute performance evaluations on Bananapi-F3 and long-vector architectures emulated on FPGAs.
`zhangjian29` has offered to run the SG2044 side and share benchdnn inputs/methodology so results are comparable.
We consider that collaboration, and evaluations on other distinct microarchitectures such as Pioneer and K3, valuable for de-risking the "vendor-agnostic" claims of this proposal.
Nonetheless, we will add instructions to reproduce our performance analysis on each PR related to `rvjit` so that other stakeholders can easily evaluate each PR's impact.

We are willing to carry out further analyses if access to other hardware platforms is granted to us.
Alternatively, other evaluation platforms might be obtained from community clouds such as [cloud-v.co](https://cloud-v.co/).

### `rvjit` components

Each component below is a plain class living in `dnnl::impl::cpu::rv64::rvjit`, orchestrated by an `rvjit_t` object managing the components used by a primitive.
Kernels interested in using `rvjit` may do so by constructing an instance (`rvjit_t m(*this)`) within `generate()`.

#### `const_folding_t` — fold constant arithmetic into immediates

`rvjit` defines the [`const_t`](https://github.com/alexandrelimassantana/oneDNN/blob/c2ea612b037f9bbe91bc087f53abed7b7a491209/src/cpu/rv64/rvjit/rvjit_types.hpp#L103) structure to represent a value that can be expressed as a 12-bit immediate or a register.
Using this structure, rvjit introduces the `const_folding` component, [`rvjit_const_folding.hpp`](https://github.com/alexandrelimassantana/oneDNN/blob/c2ea612b037f9bbe91bc087f53abed7b7a491209/src/cpu/rv64/rvjit/rvjit_const_folding.hpp), exposing operations over constants.

The API consists of `init_constant`, `add_const`, `div_const`, and `round_down`.
Each takes a `const_t` as one operand and adapts code emission conditionally based on how the constant is expressed, folding arithmetic into immediate forms when possible.
In cases such as `div_const` and `round_down`, optimized paths are provided for immediates that are powers of two, favoring `slli` instructions over `div` to realize divisions.

```cpp
// Obtain a reference to the constant arithmetics component
auto &ca = m.const_folding();

// Initialize a constant value LDB, loading to a register if not simm12
ca.init_constant(/*const_t*/ ldb);

// Materializes as either `addi` or `add` depending on ldb parameters
ca.add_const(rd, rs1, /*const_t*/ ldb);

// tail = largest multiple of `unroll` <= len
// power-of-two immediate values fold into a shift instead of div
ca.round_down(tail, len, unroll);
```

This component is used upstream by `brgemm` and `1x1-conv` to emit pointer arithmetic code.
Packing constants into values assists portability to long vector architectures which may define large `vlenb`-derived strides that, if encoded blindly into `addi` instructions, may result in a failure to emit code.
Stride- and block-sizing variables derived from the vector length are common across the adopting kernels, e.g.
[`m` in `gemm`](https://github.com/alexandrelimassantana/oneDNN/blob/c2ea612b037f9bbe91bc087f53abed7b7a491209/src/cpu/rv64/gemm/rvv_gemm_utils_f32.hpp#L60-L61),
[`bd_block` in `brgemm`](https://github.com/alexandrelimassantana/oneDNN/blob/c2ea612b037f9bbe91bc087f53abed7b7a491209/src/cpu/rv64/brgemm/brgemm.cpp#L85-L86),
and
[`oc_block` in `1x1-conv`](https://github.com/alexandrelimassantana/oneDNN/blob/c2ea612b037f9bbe91bc087f53abed7b7a491209/src/cpu/rv64/jit_rvv_1x1_conv_kernel.cpp#L69-L71),
each of which feeds directly into a pointer stride the `const_folding` component then has to fold or materialize.
Blindly using registers for constant strides results in fewer registers to apply important optimizations like iterators with multiple pointer registers (as in the use of pivot pointers to B rows in `brgemm`).
The `const_folding` component is suitable for integration into any kernel with pointer arithmetic, or generators handling variables that may be either codegen constants or runtime values.

#### `control_flow_t` — structured branches, loops, and dispatch

The `control_flow_t` component ([`rvjit_control_flow.hpp`](https://github.com/alexandrelimassantana/oneDNN/blob/c2ea612b037f9bbe91bc087f53abed7b7a491209/src/cpu/rv64/rvjit/rvjit_control_flow.hpp)) aims to abstract the emission of `Label` and branch instructions.

The API consists of the `if_`/`while_`/`switch_case`/`unrolled_loop` methods, which take a user-supplied lambda to generate the code region related to the body of such constructs.
Conditions are expressed with a new basic struct,
[`branch_t`](https://github.com/alexandrelimassantana/oneDNN/blob/c2ea612b037f9bbe91bc087f53abed7b7a491209/src/cpu/rv64/rvjit/rvjit_types.hpp#L220),
and its static factories.

`unrolled_loop` alone materializes several distinct unrolling strategies through [`loop_t::mode_t`](https://github.com/alexandrelimassantana/oneDNN/blob/c2ea612b037f9bbe91bc087f53abed7b7a491209/src/cpu/rv64/rvjit/rvjit_types.hpp#L361): a fully unrolled loop by a fixed factor (`loop_t::unroll`), a fully unrolled loop by a dynamic factor `< N` dispatched via `switch_case` (`loop_t::switch_`), and a main unrolled loop followed by a runtime dispatch to a tail (`loop_t::unroll_and_switch`). This is deliberate performance infrastructure, not only a de-duplication mechanism: which strategy a kernel picks changes the emitted code's register pressure and branch count, and `rvjit` exposes the choice explicitly instead of hard-coding one.

These patterns are used to implement all sorts of control flow expressions such as
[`jit_brgemm_kernel.cpp#L126`](https://github.com/alexandrelimassantana/oneDNN/blob/c2ea612b037f9bbe91bc087f53abed7b7a491209/src/cpu/rv64/brgemm/jit_brgemm_kernel.cpp#L126)
and
[`jit_rvv_1x1_conv_kernel.cpp#L285`](https://github.com/alexandrelimassantana/oneDNN/blob/c2ea612b037f9bbe91bc087f53abed7b7a491209/src/cpu/rv64/jit_rvv_1x1_conv_kernel.cpp#L285):

```cpp
// Obtain the control flow component
auto &cf = m.control_flow();

// Emit a C-like if-then expression
cf.if_(branch_t::nez(bias), [&] {
    // body callback is emitted on the branch taken path
    vle32_v(bias_tile, bias);
    // ...
});

// Emit a C-like `while (oc_left > 0) { cb(); }`
cf.while_(branch_t::gtz(oc_left), [&] {
    ...
    ca.add_const(oc_left, oc_left, -oc_block);
});
```

Expressions such as the `switch_case` are extremely useful as dispatch mechanisms.
On [`gemm`](https://github.com/alexandrelimassantana/oneDNN/blob/c2ea612b037f9bbe91bc087f53abed7b7a491209/src/cpu/rv64/gemm/jit_rvv_gemm_kernel.cpp#L86-L87) (`plan.n_loop = loop_t::switch_().id(...)`), it is used to unify all kernels with different `N_unrolling` into a single micro-kernel, dispatching based on a runtime-provided `n` value.
On [`brgemm`](https://github.com/alexandrelimassantana/oneDNN/blob/c2ea612b037f9bbe91bc087f53abed7b7a491209/src/cpu/rv64/brgemm/jit_brgemm_kernel.cpp#L97-L99) and [`1x1-conv`](https://github.com/alexandrelimassantana/oneDNN/blob/c2ea612b037f9bbe91bc087f53abed7b7a491209/src/cpu/rv64/jit_rvv_1x1_conv_kernel.cpp#L249-L250) (both `plan.n_loop = loop_t::unroll_and_switch()...`), it is used to dispatch tail loops after iterations of a main unrolled loop.
This pattern can be applied to any micro-kernel providing code generation for optimized tail loops.
In addition, raw, hand-rolled `Label`/branch-based control flow shows up in essentially every RV64 kernel.

#### `register_pool_t` — allocation plus callee-saved bookkeeping

The `register_pool_t` component, [`rvjit_register_pool.hpp`](https://github.com/alexandrelimassantana/oneDNN/blob/c2ea612b037f9bbe91bc087f53abed7b7a491209/src/cpu/rv64/rvjit/rvjit_register_pool.hpp), defines the concepts of templated blocks of architectural state, [`block_t<T>`](https://github.com/alexandrelimassantana/oneDNN/blob/c2ea612b037f9bbe91bc087f53abed7b7a491209/src/cpu/rv64/rvjit/rvjit_types.hpp#L178), which can be allocated from a [`partition_t<T>`](https://github.com/alexandrelimassantana/oneDNN/blob/c2ea612b037f9bbe91bc087f53abed7b7a491209/src/cpu/rv64/rvjit/rvjit_register_pool.hpp#L47), and provides specializations of `T` for integer, float, and vector registers.

The API includes methods to allocate `block_t<T>` objects or single registers (`new_int`/`new_float`/`new_vector`), a `new_const` helper that returns a value as-is when it fits a 12-bit immediate or otherwise allocates a register, and `new_loop`, which allocates whatever registers a [`loop_t`](https://github.com/alexandrelimassantana/oneDNN/blob/c2ea612b037f9bbe91bc087f53abed7b7a491209/src/cpu/rv64/rvjit/rvjit_types.hpp#L358) still needs.
A pair of `preserve()`/`restore()` methods emit the prologue/epilogue for whichever `s`-registers actually got allocated, avoiding code duplication and automating the management of save/restore lists.

To avoid obfuscating important variables with automatic name allocation, the API allows the callers to exclude names from the `partition_t<T>`.
Such registers are essentially invisible to the `register_pool_t` and must be managed by the caller.
Variants for `preserve()`/`restore()` with caller-supplied lists are provided if needed for manual management.
This contract offers a configurable split of responsibilities between caller and `rvjit` aimed at retaining legibility of important variables in the JIT-generated code.

```cpp
auto &pool = m.register_pool();

// Live registers (reserved names)
const Reg ptra = a1;
const Reg ptrb = a2;
const Reg ptrc = a3;

// Setup integer partition pool excluding live names
pool.int_register_file({ptra, ptrb, ptrc});

// Allocate temporaries upfront
Reg tmp = pool.new_int();

// Allocate a 4-register block_t<T> of accumulators
x_block_t acc = pool.new_int(4);
...

// Preserve callee-saved registers allocated with `new_int`
pool.preserve();
... // Kernel body
pool.restore();   // matches ld + stack deallocation
ret();
```

Manual `s`-register save/restore sequences exist in
[`jit_uni_group_normalization.cpp#L325`](https://github.com/alexandrelimassantana/oneDNN/blob/c2ea612b037f9bbe91bc087f53abed7b7a491209/src/cpu/rv64/jit_uni_group_normalization.cpp#L325),
[`jit_uni_pool_kernel.cpp#L182`](https://github.com/alexandrelimassantana/oneDNN/blob/c2ea612b037f9bbe91bc087f53abed7b7a491209/src/cpu/rv64/jit_uni_pool_kernel.cpp#L182),
[`rvv_winograd_convolution.cpp#L123`](https://github.com/alexandrelimassantana/oneDNN/blob/c2ea612b037f9bbe91bc087f53abed7b7a491209/src/cpu/rv64/rvv_winograd_convolution.cpp#L123),
[`reorder/jit_uni_reorder_kernel.cpp#L174`](https://github.com/alexandrelimassantana/oneDNN/blob/c2ea612b037f9bbe91bc087f53abed7b7a491209/src/cpu/rv64/reorder/jit_uni_reorder_kernel.cpp#L174),
and
[`injectors/injector_utils.cpp#L40`](https://github.com/alexandrelimassantana/oneDNN/blob/c2ea612b037f9bbe91bc087f53abed7b7a491209/src/cpu/rv64/injectors/injector_utils.cpp#L40).

#### `memory_move_t` — width-typed load/store

The `memory_move_t` component, [`rvjit_memory_move.hpp`](https://github.com/alexandrelimassantana/oneDNN/blob/c2ea612b037f9bbe91bc087f53abed7b7a491209/src/cpu/rv64/rvjit/rvjit_memory_move.hpp), creates meta memory instructions which adapt to a configurable data-type.

The API includes integer and float scalar loads (`fload`/`xload`) in addition to three vector memory instruction modes: unit-stride (`vle`/`vse`), register-strided (`vlse`/`vsse`), and indexed (`vloxei`/`vluxei`/`vsoxei`/`vsuxei`).
The vector instructions are further unified behind mode-adaptable `vload`/`vstore` that take a [`vaddr_t`](https://github.com/alexandrelimassantana/oneDNN/blob/c2ea612b037f9bbe91bc087f53abed7b7a491209/src/cpu/rv64/rvjit/rvjit_memory_move.hpp#L42) parameter describing the vector memory mode and additional parameters (i.e. stride or index register).

Kernels may convert from an oneDNN `data_type_t` to `SEW` via the [`to_rvjit_sew()`](https://github.com/alexandrelimassantana/oneDNN/blob/c2ea612b037f9bbe91bc087f53abed7b7a491209/src/cpu/rv64/jit_generator.hpp#L85) helper and pass the `SEW` in, as in [`jit_brgemm_kernel.cpp#L127`](https://github.com/alexandrelimassantana/oneDNN/blob/c2ea612b037f9bbe91bc087f53abed7b7a491209/src/cpu/rv64/brgemm/jit_brgemm_kernel.cpp#L127):

```cpp
auto &mem = m.memory_move();

// Type-adaptable BIAS float addition
mem.vle(vbias, bias_ptr, to_rvjit_sew(dt_c));
vfadd_vv(vdest, vdest, vbias);
mem.vse(vdest, ptrc, to_rvjit_sew(dt_c));
```

This component is used to unify implementations for different data types and may be used by any kernel since vector memory instructions in RISC-V are differentiated statically (e.g. [`jit_rvv_inner_product_kernel.cpp#L108-L116`](https://github.com/alexandrelimassantana/oneDNN/blob/c2ea612b037f9bbe91bc087f53abed7b7a491209/src/cpu/rv64/jit_rvv_inner_product_kernel.cpp#L108-L116)).

#### `rvv_arithmetic_t` — multiply-accumulate

The `rvjit_arithmetic` component, [`rvjit_arithmetic.hpp`](https://github.com/alexandrelimassantana/oneDNN/blob/c2ea612b037f9bbe91bc087f53abed7b7a491209/src/cpu/rv64/rvjit/rvjit_arithmetic.hpp), introduces meta-arithmetic instructions adapted to uniform- and mixed-precision scenarios.

The API currently consists of templated `fmacc_int`/`fmacc_float` functions, specialized to RVV to use vector operands and scalar-broadcast overloads, covering `vmacc`/`vwmacc{,u,su,us}` (int) and `vfmacc`/`vfwmacc`/`vfwmaccbf16` (float).
Dispatch is driven by an [`fma_t`](https://github.com/alexandrelimassantana/oneDNN/blob/c2ea612b037f9bbe91bc087f53abed7b7a491209/src/cpu/rv64/rvjit/rvjit_arithmetic.hpp#L41) struct describing the accumulator/operand dtypes.
This object can be constructed with [`fma_t`](https://github.com/alexandrelimassantana/oneDNN/blob/c2ea612b037f9bbe91bc087f53abed7b7a491209/src/cpu/rv64/rvjit/rvjit_arithmetic.hpp#L41)`::uniform(dt)` for same-width macc, or `fma_t::widening(dt)` for a 2x-wide accumulator.

```cpp
auto &mc = m.macc();

// uniform f32 x f32 -> f32: emits vfmacc.vv
mc.fmacc_float(acc, a_tile, b_tile, fma_t::uniform(f32));

// widening bf16 x bf16 -> f32: emits vfwmaccbf16.vv
mc.fmacc_float(acc, a_tile, b_tile, fma_t::widening(bf16));
```

Used by all matrix multiplication kernels.
The semantics behind `fmacc_float` and `fmacc_int` may be reused to support matrix extensions, as `T` may be specialized to whatever form of architectural storage these operations can assume under the four official extensions.
Using this scheme, we can also unify future matrix code generators for different types under this component.

#### `rvv_t` — hardware model

The `rvv_t` component, [`rvjit_rvv.hpp`](https://github.com/alexandrelimassantana/oneDNN/blob/c2ea612b037f9bbe91bc087f53abed7b7a491209/src/cpu/rv64/rvjit/rvjit_rvv.hpp), describes an analytical model of the vector unit and memory system ([`vpu_t`](https://github.com/alexandrelimassantana/oneDNN/blob/c2ea612b037f9bbe91bc087f53abed7b7a491209/src/cpu/rv64/rvjit/rvjit_rvv.hpp#L93)/[`memory_t`](https://github.com/alexandrelimassantana/oneDNN/blob/c2ea612b037f9bbe91bc087f53abed7b7a491209/src/cpu/rv64/rvjit/rvjit_rvv.hpp#L122)).
This component centralizes preferences for code generation, currently obtained from guesses of micro-architectural features such as the `vector lane width`.
These feed into derived concepts like `vector arithmetic latency` used to drive optimization variables such as the minimum unrolling to saturate the VPU.

Providing this unifying layer across platforms is the primary objective of `rvjit`.
During our research, we found that long-vector processors need aggressive unrolling to hide large vector-instruction latencies, while short-vector processors benefit more from a larger `LMUL` and only modest unrolling — and that this difference propagates into how supporting concepts like iterators must be implemented.
Current `brgemm`/`rvv_matmul_engine_t` B addressing, for instance, can allocate one [pivot pointer register per unrolled B column](https://github.com/alexandrelimassantana/oneDNN/blob/c2ea612b037f9bbe91bc087f53abed7b7a491209/src/cpu/rv64/rvjit/rvjit_matmul.hpp#L428-L434) so that the corresponding scalar factors can be addressed via a fixed base with an immediate offset; that scheme breaks down at large unroll factors (e.g. 16) purely from register pressure.
By providing flexible loop-unrolling implementations with configurable parameters and validation rules, `rvjit` lets code-generator subsystems reason about these optimization choices instead of relying on per-kernel developer expertise or duplicated code.

This component is built once via `rvv_t::from_hardware()` using information available to oneDNN via `get_platform_vlen()`/cache-size queries.
A helper method `vpu_t::max_n_accumulators(sew_inp, sew_acc)` determines how many independent accumulator tiles the architecture may use under the preferred settings.
Alternatively, the model may be constructed manually and fed into `rvjit`.

```cpp
auto &model = m.model(); // default-constructed

LMUL lmul  = model.vpu.lmul_preference;
int  n_ur  = model.vpu.max_n_accumulators(SEW::e32, SEW::e32);
```

Any kernel currently hardcoding an unroll factor or LMUL choice may leverage this component.
For instance, [`jit_rvv_batch_normalization_kernel.cpp#L105`](https://github.com/alexandrelimassantana/oneDNN/blob/c2ea612b037f9bbe91bc087f53abed7b7a491209/src/cpu/rv64/jit_rvv_batch_normalization_kernel.cpp#L105)
(`vsetvli(reg_vl, reg_len, SEW::e32, LMUL::m1)` fixed at `m1`).

#### `rvv_matmul_engine_t` — the shared dense N/K-loop engine

The `rvjit_matmul_engine` component, [`rvjit_matmul.hpp`](https://github.com/alexandrelimassantana/oneDNN/blob/c2ea612b037f9bbe91bc087f53abed7b7a491209/src/cpu/rv64/rvjit/rvjit_matmul.hpp), builds upon all previous pieces to materialize matrix multiplication dense loops.

The API revolves around constructing a declarative [`matmul_plan_t`](https://github.com/alexandrelimassantana/oneDNN/blob/c2ea612b037f9bbe91bc087f53abed7b7a491209/src/cpu/rv64/rvjit/rvjit_matmul.hpp#L87) describing pointers, [`strides`](https://github.com/alexandrelimassantana/oneDNN/blob/c2ea612b037f9bbe91bc087f53abed7b7a491209/src/cpu/rv64/rvjit/rvjit_matmul.hpp#L51) (a [`matmul_strides_t`](https://github.com/alexandrelimassantana/oneDNN/blob/c2ea612b037f9bbe91bc087f53abed7b7a491209/src/cpu/rv64/rvjit/rvjit_matmul.hpp#L51)), N/K loop shapes (each a [`loop_t`](https://github.com/alexandrelimassantana/oneDNN/blob/c2ea612b037f9bbe91bc087f53abed7b7a491209/src/cpu/rv64/rvjit/rvjit_types.hpp#L358)), and dtype.
The caller invokes `configure()` once to resolve SEW/LMUL/unrolling/register
allocation, delegating optimization decisions and code emission strategies to the respective components.
The `generate()` method emits the pipelined loop nest, receiving the finished accumulator tile back through a callback for post-ops.

```cpp
matmul_plan_t plan;
plan.dti = plan.dto = f32; // uniform precision
plan.ptra = a_ptr; // operand pointer
plan.ptrb = b_ptr; // operand pointer
plan.ptrc = c_ptr; // operand pointer

// Strides are constants known at codegen time
plan.strides = matmul_strides_t::from_bytes(lda, ldb, ldc);

// The N loop is implemented by a main unrolled loop
// followed by a switch_case to dispatch the tail
// When unroll factor is not set, the matmul_t plan optimizes it.
plan.n_loop = loop_t::unroll_and_switch();

// The K loop is implemented by a main unrolled loop
// followed by a tail loop with unroll=1.
// When unroll factor is not set, the matmul_t plan optimizes it.
plan.k_loop = loop_t::unroll();

// How the component accesses the application vector length
plan.avl = avl;

// Analyze parameters to check if they produce valid code
auto &eng = m.matmul_engine();
if (!eng.configure(plan)) return;

// Generate micro-kernel structures with supplied post-op callback
eng.generate([&](v_block_t c_tile, v_block_t scratch) {
    mem.vse(c_tile(0), ptrc, to_rvjit_sew(plan.dto));
});
```

This is the shared dense N/K-loop engine behind gemm/brgemm/1x1-conv.
It is the most specific component, applied specifically to matrix multiplications.

Because `n_loop`/`k_loop` are independently configurable [`loop_t`](https://github.com/alexandrelimassantana/oneDNN/blob/c2ea612b037f9bbe91bc087f53abed7b7a491209/src/cpu/rv64/rvjit/rvjit_types.hpp#L358) instances built on top of the 2D register blocking exposed by [`block_t<T>`](https://github.com/alexandrelimassantana/oneDNN/blob/c2ea612b037f9bbe91bc087f53abed7b7a491209/src/cpu/rv64/rvjit/rvjit_types.hpp#L178), the same `matmul_engine_t` component call can materialize either a "dense reduce loop" shape (K only) or a full "inner loop" shape (N and K) purely from different `matmul_plan_t` parameters, with no separate code path per shape.

### Testing

We include testing that verifies the emission of the `rvjit` recurrent patterns.
Due to its broad applicability, `rvjit` expressions must generate code that meets the expectations of developers.
The tests validate the pattern emissions under different parameters, verifying parameters around acceptance margins of optimizations.
Our strategy is to test `rvjit` in isolation from oneDNN since the tests consist of defining simple code generators to verify correctness of concepts such as loop trip counts.

The current PR adds
[`tests/gtests/internals/rv64/`](https://github.com/alexandrelimassantana/oneDNN/tree/c2ea612b037f9bbe91bc087f53abed7b7a491209/tests/gtests/internals/rv64),
built into a separate `test_internals_rvjit` binary
([`tests/gtests/internals/CMakeLists.txt#L71-L81`](https://github.com/alexandrelimassantana/oneDNN/blob/c2ea612b037f9bbe91bc087f53abed7b7a491209/tests/gtests/internals/CMakeLists.txt#L71-L81)).
These must be materialized in a different way, but the content of the tests is already enough to support the current component system.

### Performance

Early measurements report geometric-mean speedups of **~1.28x on
convolution workloads** (measured on a Bananapi platform) and **~1.06x on
matrix-multiplication workloads** (a custom test set derived from
transformer models), relative to the pre-`rvjit` RV64 kernels.
The brgemm migration's ~1,535-line net deletion is a secondary,
non-performance benefit of the same change.
The performance benefits stem from the standardization of optimizations applied across the three matmul kernels as well as the refinement of unrolling factors and better instruction ordering (i.e., interleaving unrelated instructions in pairs).

The latest set of performance data was executed on a Bananapi-F3 node provided by [cloud-v.co](https://cloud-v.co/), a community platform launched by 10xEngineers to accelerate RISC-V software development.

### Ownership and maintenance

BSC pledges to maintain `rvjit` and provide support for new use cases throughout its development in oneDNN.
The development of runtime systems for RV64 and vendor-agnostic code generation is an active research line at the institution.
We are interested in researching the challenge of integrating RV64 platforms into oneDNN at the community level, and we are open to assisting with optimization tuning efforts.
To do so, it is in our best interest to first address the risk of software fragmentation in RV64 with abstraction and reusability.
Whether that maintenance happens inside a standalone `rvjit` repository or fully within oneDNN is intentionally left open — see "Standalone repository vs. full integration" under Open Questions below.

#### Documentation

Documentation for `rvjit` shall be provided with the intent of setting clear expectations from caller code.
Doxygen notation will be used to express pre- and post-conditions of `rvjit` expressions.
This documentation is to be updated alongside any future `rvjit` updates and shall specify:

- Register parameters that are potentially `clobbered` on `rvjit` calls, or any conditions that result in a `clobbered` register (e.g. `round_down` clobbers the `Reg tmp` register if the [`const_t`](https://github.com/alexandrelimassantana/oneDNN/blob/c2ea612b037f9bbe91bc087f53abed7b7a491209/src/cpu/rv64/rvjit/rvjit_types.hpp#L103) `c` parameter is immediate and is not a power-of-two); register parameters that are not specified as `clobbered` are assumed to be `preserved` unless aliased by the caller.
- Any restrictions concerning `rvjit` call parameters such as invalid aliasing of registers, illegal uses of `x0` as destination, and others that result in code emission failure or undefined behavior.
- Any `rvjit` call used to implement expression through composition (e.g. `control_flow_t::unrolled_loop` uses `const_folding_t::round_down` to compute the limit of unrolled loops); only the components directly used by the documented method are to be specified (e.g. `rvv_matmul_engine_t::generate` leverages `control_flow_t::unrolled_loop` which delegates to `const_folding_t::round_down`, but only `control_flow_t` should be linked to `rvv_matmul_engine_t::generate`).

#### Testing philosophy

We will propose and maintain a set of tests validating the expectations of `rvjit` caller code.
These shall include both failure and success cases so that the API behaves as documented.
Future contributions to `rvjit` must update the tests accordingly to validate the updated caller expectations.

#### Debugging `rvjit`

To enhance the developer experience, all failure conditions of `rvjit` are accompanied by `DEBUG` messages emitted on developer builds describing the failure stack of component calls.
For instance, an `rvjit` failure to emit code in response to a call to `rvv_matmul_engine_t::generate` due to an illegal alias in a downstream `const_folding_t::round_down` call should print a failure trace including all involved components (e.g. failed `rvjit_t::rvv_matmul_engine_t::generate` due to invalid `unrolled_loop` parameters; failed `rvjit_t::control_flow_t::unrolled_loop` due to invalid `round_down` parameters; failed `rvjit_t::const_folding_t::round_down` due to illegal aliasing of `rd` and `rs1` registers).

## Open Questions

- **Standalone repository vs. full integration.** Whether `rvjit` should
  eventually live in an independent, BSC-maintained repository or stay
  fully internal to oneDNN is not settled:

  - *Standalone repository*. Pros: clear ownership from being a separate
    project. Cons: optional integration of `rvjit` may result in
    duplicated primitives with equivalent semantics, possibly competing
    for priority.
  - *oneDNN internal project*. Pros: simpler development under familiar,
    extended guidelines; BSC pledges full maintenance (documentation,
    support, tests, features, integrations, evaluations). Cons: weaker
    alignment with how oneDNN already treats comparable code (e.g.
    `third_party/xbyak_riscv`) as a vendored external dependency rather
    than in-tree.

  Our preference is an *internal project*.
  Nevertheless, we are open to bootstrapping `rvjit` as a standalone
  project if that's what maintainers prefer.
  We are open to a future split-out if requested by the RV64 community.
  In that scenario, BSC assumes responsibility for the effort in a way
  that preserves oneDNN's primitive and performance characteristics.

  Either way, `rvjit` integration plans involve a series of smaller PRs,
  first introducing the library and tests (if *internal project*), then refactoring primitives one by
  one, backed by reproducible experiments to support performance claims.

- **Test structure.** If `rvjit` is accepted as an *internal project*, the
  `rvjit` testing infrastructure must be revised.
