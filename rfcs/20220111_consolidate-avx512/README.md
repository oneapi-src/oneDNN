Proposal for consolidating AVX512 ISA
====================================

## Introduction

With the drop of support for KNX and removal of `avx512_mic` and
`avx512_mic_4ops` code path from OneDNN, there is no longer a need to
differentiate between `avx512_common` and `avx512_core`. The former (i.e.
`avx512_common`) was used to indicate an overlap between `avx512_mic[_4ops]`
for KNX and `avx512_core` for SKX or newer models. This RFC proposes the
consolidation of `avx512_common` and `avx512_core` into a single definition and
use.

## Proposal & Implementation

The primary intent of these changes is to consolidate `avx512_common` and
`avx512_core` into a single definition, their utilities, their properties, and
the code path for the cpu/x64 F32 optimized primitives.

### Option 1) *Recommended
Use `avx512_core` as the definitive ISA type for AVX512 code path
and properties:

### Pros:
* Minimal changes required to implement. Either remove or replace
  `avx1512_common` text with `avx512_core` throughout the code.
* No OneDNN API changes since `avx512_core` is already defined while
  `avx512_common` is not:

[include/oneapi/dnnl/dnnl_types.h](https://github.com/oneapi-src/oneDNN/tree/master/include/oneapi/dnnl/dnnl_types.h):
```diff
/// CPU instruction set flags
typedef enum {
    (…)
    /// Intel AVX-512 subset for Intel Xeon Scalable processor family
    /// and Intel Core processor family.
    dnnl_cpu_isa_avx512_core = 0x27,
}
```

* Because `avx512_core` is already a superset of `avx512_common`, the properties
  will be consolidated into `avx512_core`:

[src/cpu/x64/cpu_isa_traits.hpp](https://github.com/oneapi-src/oneDNN/tree/master/src/cpu/x64/cpu_isa_traits.hpp):
```diff
bool mayiuse(const cpu_isa_t cpu_isa) {
    (...)
-    case avx512_common: return cpu().has(Cpu::tAVX512F);
    case avx512_core: return cpu().has(Cpu::tAVX512F)
        && cpu().has(Cpu::tAVX512BW) && cpu().has(Cpu::tAVX512VL)
        && cpu().has(Cpu::tAVX512DQ);
    (...)
}

enum cpu_isa_t {
    (…)
-    avx512_common = avx512_common_bit | avx2,
-    avx512_core = avx512_core_bit | avx512_common,
+    avx512_core = avx512_core_bit | avx2,
    (…)
}

- template <>
- struct cpu_isa_traits<avx512_common> {
-     (...)
- };

template <> struct cpu_isa_traits<avx512_core> {
+    typedef Xbyak::Zmm Vmm;
+    static constexpr int vlen_shift = 6;
+    static constexpr int vlen = 64;
+    static constexpr int n_vregs = 32;
    static constexpr dnnl_cpu_isa_t user_option_val = dnnl_cpu_isa_avx512_core;
    static constexpr const char *user_option_env = "avx512_core";
};
```

* Additionally, the use of `avx512_core` is aligned with the Intel Compiler
  definition ([link](
  https://www.intel.com/content/www/us/en/developer/articles/technical/performance-tools-compiler-options-for-sse-generation-and-processor-specific-optimizations.html),
  see mention of `avx512_common` and `avx512_core`).

### Cons:
* Overspecification of the definition, instead of a simpler label like
avx512 since there are no alternate definitions for avx512.

## Option 2)
Use `avx512` as the CPU ISA definition since there are no other avx512 F32
definitions in OneDNN.

### Pros:
* Follows the simpler nomenclature of `avx2`, `avx` and `sse41`. The renaming
  and replacement would follow the approach detailed in Option 1.

### Cons:
* Naming discrepancy for other definitions using `avx512_core*` for `vnni`,
  `bf16` and `amx`.

* Requires updating OneDNN's API, potentially leaving `dnnl_cpu_isa_avx512_core`
  for backward compatibility:

[include/oneapi/dnnl/dnnl_types.h](https://github.com/oneapi-src/oneDNN/tree/master/include/oneapi/dnnl/dnnl_types.h):
```diff
/// CPU instruction set flags
typedef enum {
    (…)
    /// Intel AVX-512 subset for Intel Xeon Scalable processor family
    /// and Intel Core processor family.
    dnnl_cpu_isa_avx512_core = 0x27, // backward compatibility
+    dnnl_cpu_isa_avx512 = 0x27
}
```

## Option 3)
Replace every `avx512_core*` entry with avx512 for all cpu ISA
definitions (e.g. `avx512_core_vnni` will become `avx512_vnni`,
`avx512_core_bf16` will become `avx512_bf16`, etc.).

### Pros:
* Removes unnecessary `_core` label from ISA definitions now that there's no
  other avx512 kind (i.e. `avx512_common`, `avx512_mic`, etc.).

### Cons:
* There may be some adoption time from developers for the new nomenclature.

* Requires the most API changes and throughout OneDNNs cpu/x64 source:

[include/oneapi/dnnl/dnnl_types.h](https://github.com/oneapi-src/oneDNN/tree/master/include/oneapi/dnnl/dnnl_types.h)

```diff
/// CPU instruction set flags
typedef enum {
    (…)
    /// Intel AVX-512 subset for Intel Xeon Scalable processor family
    /// and Intel Core processor family.
    dnnl_cpu_isa_avx512_core = 0x27, // backward compatibility
+    dnnl_cpu_isa_avx512 = 0x27,

    /// Intel AVX-512 and Intel Deep Learning Boost (Intel DL Boost) support
    /// for Intel Xeon Scalable processor family
    /// and Intel Core processor family.
    dnnl_cpu_isa_avx512_core_vnni = 0x67, // backward compatibility
+    dnnl_cpu_isa_avx512_vnni = 0x67,

    /// Intel AVX-512, Intel DL Boost and bfloat16 support
    /// for Intel Xeon Scalable processor family
    /// and Intel Core processor family.
    dnnl_cpu_isa_avx512_core_bf16 = 0xe7, // backward compatibility
+    dnnl_cpu_isa_avx512_bf16 = 0xe7,

    /// Intel AVX-512, Intel DL Boost and bfloat16 support and
    /// Intel AMX with 8-bit integer and bfloat16 support
    dnnl_cpu_isa_avx512_core_amx = 0x3e7, // backward compatibility
+    dnnl_cpu_isa_avx512_amx = 0x3e7,
}
```

## Additional Changes
With the removal of `ver_4fma` in `enum conv_version_t` (defined in
[src/cpu/x64/jit_primitive_conf.hpp](https://github.com/oneapi-src/oneDNN/tree/master/src/cpu/x64/jit_primitive_conf.hpp)),
differentiating between the `fma` and `ver_avx512_core` entries is redundant.
Furthermore, there is an additional value called `ver_vnni` used for the
`avx512_core_vnni` and `avx2_vnni` code paths when int8 instructions are
supported natively.

Hence, this code can be reformated in the following way:
* Remove any conditional code path for `ver_fma` or `ver_avx512_core` in
  `jit_avx512*` files since the implementations are now equal -and having no
  alternative KNX code paths.

* Replace the field `conv_version_t ver;` with `bool has_vnni`. This will
  differentiate between ISAs supporting VNNI either natively or emulated. All
  of the instances of `jcp.ver` are currently only used in a condition against
  `jcp.ver == ver_vnni` so this would simplify its use.

* Lastly, there is no need to have the enum `conv_version_t` definition and can
  be removed.
