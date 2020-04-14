oneDNN CPU Implementation
=========================

## Directory structure

```
cpu
├── gemm/               # Generic GEMM implementation (opt. depends on x64/gemm)
├── rnn/                # Generic RNN implementation (opt. depends on x64/rnn)
├── x64                 # x64-specific sub-directory
│   ├── gemm/           # x64-specific GEMM implementation
│   ├── jit_utils/      # JIT-related utilities, such as support of profilers
│   ├── rnn/            # JIT-related kernels for rnn primitive
│   ├── xbyak/          # Xbyak sources
│   └── jit_*.*         # x64-specific implementations
├── cpu_engine.hpp      # Basic oneDNN abstractions
├── cpu_lrn_pd.hpp      # Base cpu primitive descriptor classes
├── cpu_lrn_list.cpp    # Implementation lists
├── nchw_pooling.cpp    # Semi-optimized (aka simple) implementations
├── platform.hpp        # Platform-related utility functions
└── ref_eltwise.cpp     # Reference implementations
```

## Target architectures

The source code is organized in a modular way to separate generic architecture
independent (or weakly dependent) implementations from the arch-specific ones.
- Generic code is located under `cpu/`;
- Architecture specific code lives in `cpu/$arch/` sub-directory.

Currently, the only architecture specific directory is `cpu/x64` which contains
Intel 64 / AMD64 implementations, that mostly use JIT assembler
[Xbyak](https://github.com/herumi/xbyak) to produce highly optimized code.

The architecture specific code can easily access the generic code, but the
opposite should be limited as much as possible. However, sometimes it is
absolutely necessary for generic code to access architecture specific one. For
instance, the list of implementations that live in `cpu/*_list.cpp` should
conditionally include the specific implementations on the corresponding
architecture. Hence, for portability reasons [`cpu/platform.hpp`](platform.hpp)
header file provides a set of helpers macros that could help conditionally
enable or disable parts of code. There the following macros defined:
- `DNNL_X64` is 1 on x64 architecture, and 0 otherwise;
- `DNNL_AARCH64` is 1 on ARM AArch64 architecture, and 0 otherwise;
- `DNNL_ARCH_GENERIC` is 1 on non of the above architectures;

Note, that one and only of the macros above are defined to 1, while others are
defined to 0. Usage example:

``` cpp
#include "cpu/platform.hpp" // IMPORTANT: INCLUDE THIS FILE!

int generic_foo() {
#if DNNL_X64
    return x64_specific_foo();
#endif
    return generic_impl_foo();
}
```

Additionally, there is `DNNL_${ARCH}_ONLY(...)` macro that expand the
parameters only on the corresponding architectures. Hence, the following
snippet corresponds to the one above:

``` cpp
#include "cpu/platform.hpp" // IMPORTANT: INCLUDE THIS FILE!

int generic_foo() {
    DNNL_X64_ONLY(return x64_specific_foo());
    return generic_impl_foo();
}
```

See more details in [`platform.hpp`](platform.hpp).
Also check `DNNL_TARGET_ARCH` cmake variable.
