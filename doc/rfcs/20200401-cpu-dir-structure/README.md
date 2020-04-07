# DNNL CPU Code Organization Adjustments

## Motivation

There are ongoing efforts of porting DNNL library to non-x64 CPU:
- To NEC SX maintained by @kruus: https://github.com/necla-ml/gen-dnn
- To ARM maintained by @nSircombe: https://github.com/nSircombe/mkl-dnn

The biggest obstacle for supporting those ports or making the official DNNL to
be buildable on non-x64 platforms is the tight interleaving of x64-specific
(Xbyak) and generic code. The main objective of this RFC is to suggested
changes to directory and code structure to make the sources more modular.

The high-level suggestion is to introduce `cpu/x64` directory and move all the
specific code there, leaving the generic code as it is now. Allow building the
library without x64-specific code.

## Current State

Current directory structure looks like this:

```
cpu
├── gemm/
├── jit_utils/
├── rnn/
├── xbyak/
├── cpu_batch_normalization_utils.cpp
├── cpu_engine.hpp
├── cpu_lrn_pd.hpp
├── cpu_lrn_list.cpp
├── jit_*.*
├── nchw_pooling.cpp
└── ref_eltwise.cpp
```

Conceptually, the code in `src/cpu` consists of the following groups:

| Group                               | Examples                                                                  | Any x64-specific code
| :--                                 | :--                                                                       | :--
| Basic DNNL abstractions             | `cpu_engine.hpp`, `cpu_memory_storage.hpp`                                | No
| Base primitive descriptor classes   | `cpu_lrn_pd.hpp`                                                          | No
| Implementation lists                | `cpu_lrn_list.cpp`                                                        | Listing the implementations themselves
| Code shared between implementations | `cpu_batch_normalization_utils.cpp`                                       | Minimal
| Reference implementations           | `ref_lrn.cpp`                                                             | Minimal
| Simple C-based implementations      | `nchw_pooling.hpp`                                                        | Moderate: same as above + querying cache sizes, gemm-based Inner Product and Convolutions use JIT to perform post-processing
| Semi-jitted implementations         | `jit_uni_layer_normalization_kernels.hpp`, `gemm_inner_product_utils.cpp` | Yes: a half fully relies on jit, while the other half is reference code
| JIT-based implementations           | `jit_uni_eltwise.cpp`                                                     | Yes
| JIT dump utilities                  | `jit_utils/`                                                              | Yes
| GEMM                                | `gemm/`                                                                   | Yes, but there is generic-reference code as well
| RNN                                 | `rnn/`                                                                    | Yes: jitted element-wise, int8 packed gemm

Some observations:
- Lots of places depend on `mayiuse(isa)` feature to check if bf16 data type is
  supported. This should be abstracted out by a specialized function
  `platform::has_bf16()` or `engine->has_bf16()`.
- Functions to check the cache size should also be abstract. For x64 the
  implementation may use Xbyak, but such kind of dependencies should be
  guarded, so that the library can be built on non x64 platforms.
- Semi-jitted implementations, like `gemm_inner_product_utils.cpp`, currently
  hide some unsupported case by providing the fallback C-based implementation.
  Typically, within the same class.

## Proposal

### Directory Structure

The proposed directory structure:

```
cpu
├── gemm/
├── x64
│   ├── gemm/
│   ├── jit_utils/
│   ├── rnn/
│   ├── xbyak/
│   └── jit_*.*
├── cpu_engine.hpp
├── cpu_lrn_pd.hpp
├── cpu_lrn_list.cpp
├── nchw_pooling.cpp
├── platform.hpp
└── ref_eltwise.cpp
```

Notes:
- General rule of thumb is that `cpu/x64` may depend on the files from `cpu/`,
  the opposite should be true in very limited cases, mostly in dispatch codes
  and guarded by special macros (see below). Examples:
  - Implementation list;
  - Implementation of CPU hardware characteristics queries;
  - Implementation of auxiliary sub-kernels that would have generic and jitted
    version. The base class and a dispatcher will be located in `cpu/`, while
    the jitted implementation will live in `cpu/x64`. Example: inner product
    post processing kernel, that is used by inner product and matmul.
- `cpu/x64/jit_*.*` keep `jit_` prefix, as there might be other implementations
  that are mostly not jitted, but there is no sense in these implementations
  without the jitted parts. Examples: semi-optimized plain formats, with jitted
  sub-kernels.
- New `platform.hpp` that roughly corresponds to the current
  `cpu_isa_traits.hpp`. It would contain queries like cache size, number of
  cores, etc. The implementation will depend on `x64` but will be guarded by
  the corresponding macros.
- Binary and resampling will lose their own directories. Rationale: after the
  split each directory will contain very few files and probably not worth it.
- However, gemm, matmul, and rnn will continue using their own subdirectories.
  That's mostly because they already have lots of files in them, and the number
  will continue growing.
- As RNN is currently very x64-specific, there won't be an implementation in
  the `cpu/` yet. Only `cpu_rnn_list.hpp` and `cpu_rnn_pd.hpp`. This can be
  enhanced later.
- Most of the gemm code will go to `cpu/x64/gemm` directory. But the gemm
  internal API and reference implementation will live in `cpu/gemm`.
- The `cpu_isa_bit_t` and `cpu_isa_t` enums will live in `cpu/platform.hpp` and
  will contain all isa (generic, like `all` / `any`, and x64 specific, like
  `avx`) without any macro guards. However, the traits (like `vreg_len` etc)
  will be moved to `cpu/x64/platform.hpp`.

### Refactoring

- Introduce nested namespaces mapping to directory structure
    - Ex: `dnnl::impl::cpu::x64` for x64-specific files
- Macros:
    ``` cpp
    // src/cpu/platform.hpp (preferable) or src/common/utils.hpp (currently)
    #define DNNL_ARCH_UNDEF 0
    #define DNNL_X64 1
    #define DNNL_AARCH64 2

    #if defined(__x86_64__) || defined(_M_X64)
    #define DNNL_ARCH DNNL_X64
    #else
    #define DNNL_ARCH DNNL_ARCH_UNDEF
    #endif

    // expand __VA_ARGS__ only if arch matches the current arch
    #define DNNL_ARCH_ONLY(arch, ...) IF(arch == DNNL_ARCH) __VA_ARGS__
    ```
- The implementations lists are kept in common `cpu` directory, and enabling
  happens through `DNNL_ARCH_ONLY(DNNL_X64, impl)`.
  - Maybe later we will need more advanced implementation iterator method. But
    not now.


---

EOD
