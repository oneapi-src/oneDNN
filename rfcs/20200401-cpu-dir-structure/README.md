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
  `platform::has_data_type_support(data_type::bf16)`.
- Functions to check the cache size should also be abstract. For x64 the
  implementation may use Xbyak, but such kind of dependencies should be
  guarded, so that the library can be built on non x64 platforms.
- Semi-jitted implementations, like `gemm_inner_product_utils.cpp`, currently
  hide some unsupported cases by providing the fallback C-based implementation.
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
- Most of the gemm code will go to `cpu/x64/gemm` directory. But the gemm
  internal API and reference implementation will live in `cpu/gemm`.
- Add `src/cpu/README.md` file that describes the file organization and helper
  macros.

### Refactoring

- Introduce nested namespaces mapping to directory structure
    - Ex: `dnnl::impl::cpu::x64` for x64-specific files
- Macros:
    - `DNNL_X64` is 1 on x64 architecture;
    - `DNNL_AARCH64` is 1 on ARM AArch64 architecture;
    - `DNNL_ARCH_GENERIC` is 1 on other platforms.

    Only one of the macros above is defined to 1. All others are **defined** to
    0.

    Usage example:

    ``` cpp
    #include "cpu/platform.hpp" // IMPORTANT: INCLUDE THIS FILE!

    int generic_foo() {
    #if DNNL_X64
        return x64_impl_foo();
    #else
        return generic_impl_foo();
    #endif
    }
    ```

- The implementations lists are kept in common `cpu` directory, and enabling
  happens through `INSTANCE_x64()` macro that expands in to its parameters on
  X64, and to nothing otherwise.
  - Maybe later we will need more advanced implementation iterator method. But
    not now.

---

EOD
