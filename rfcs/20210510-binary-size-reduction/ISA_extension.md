# Build knobs controlling binary size extension.

## Problem Statement.
This follows up this
[RFC](https://github.com/oneapi-src/oneDNN/tree/rfcs/rfcs/20210510-binary-size-reduction)
and proposes additional build knob to control functionality by ISA. It may
provide additional save up to 5 MB, which is around ten percent (~10%) from
current full binary size, based on selected ISA.

## Proposal
The proposal is to introduce one more build option,
`DNNL_ENABLE_PRIMITIVE_CPU_ISA`. It would take effect if `DNNL_X64` build
setting was specified.

Library supported ISA are only partially ordered, that is why the suggestion is
to group them, so that these groups are in linear order. Here they are:
* `SSE41`  - supports sse41 ISA only and all compiler-based optimized
             implementations.
* `AVX2`   - supports `SSE41` group and avx and avx2 ISA extensions on top.
* `AVX512` - supports `AVX2` group and most AVX512 ISA extensions: avx512_mic,
             avx512_mic_4ops, avx512_core, avx512_core_vnni, avx512_core_bf16
             and avx2_vnni.
* `AMX`    - supports `AVX512` group and avx512_core_amx ISA extension on top.
* `ALL`    - is an alias to the latest group (supports all groups altogether).

Using groups would simplify maintenance burden on the library and keeps the
implementation logic straight.

## Implementation Details
Build system will define the following set of macros:

```cpp
...
// Primitives ISA controls
#cmakedefine01 BUILD_PRIMITIVE_ISA_ALL
#cmakedefine01 BUILD_AVX512
#cmakedefine01 BUILD_AVX2
#cmakedefine01 BUILD_SSE41
...
```

Based on this `BUILD_XXX` macros, the following internal macros would be set:

```cpp
// src/common/impl_registration.hpp:

#if BUILD_PRIMITIVE_ISA_ALL
#define REG_ISA_AMX(...) __VA_ARGS__
#define REG_ISA_AVX512(...) __VA_ARGS__
#define REG_ISA_AVX2(...) __VA_ARGS__
#define REG_ISA_SSE41(...) __VA_ARGS__
#elif BUILD_AVX512
#define REG_ISA_AMX(...)
#define REG_ISA_AVX512(...) __VA_ARGS__
#define REG_ISA_AVX2(...) __VA_ARGS__
#define REG_ISA_SSE41(...) __VA_ARGS__
#elif BUILD_AVX2
#define REG_ISA_AMX(...)
#define REG_ISA_AVX512(...)
#define REG_ISA_AVX2(...) __VA_ARGS__
#define REG_ISA_SSE41(...) __VA_ARGS__
#elif BUILD_SSE41
#define REG_ISA_AMX(...)
#define REG_ISA_AVX512(...)
#define REG_ISA_AVX2(...)
#define REG_ISA_SSE41(...) __VA_ARGS__
#endif
```

Based on this set of macros, `CPU_INSTANCE_X64` would be extended with the
following list:
```cpp
// src/cpu/cpu_engine.hpp:

#define CPU_INSTANCE_X64_AVX2(...) REG_ISA_AVX2(DNNL_X64_ONLY(CPU_INSTANCE(__VA_ARGS__)))
#define CPU_INSTANCE_X64_AVX512(...) REG_ISA_AVX512(DNNL_X64_ONLY(CPU_INSTANCE(__VA_ARGS__)))
#define CPU_INSTANCE_X64_AMX(...) REG_ISA_AMX(DNNL_X64_ONLY(CPU_INSTANCE(__VA_ARGS__)))
```

Finally, these internal macros are applied inside primitive implementation
lists:

```cpp
// src/cpu/cpu_convolution_list.cpp:
const std::map<conv_impl_key_t, std::vector<pd_create_f>> impl_list_map {
    {{forward, s8, s8, f32}, {
        REG_CONV_P_FWD(REG_IP_P_FWD(CPU_INSTANCE_X64(ip_convolution_fwd_t)))
        REG_CONV_P_FWD(CPU_INSTANCE_X64_AMX(jit_avx512_core_amx_1x1_convolution_fwd_t))
        REG_CONV_P_FWD(CPU_INSTANCE_X64_AVX512(jit_avx512_core_amx_convolution_fwd_t))
        REG_CONV_P_FWD(CPU_INSTANCE_X64_AVX512(jit_avx512_core_x8s8s32x_1x1_convolution_fwd_t))
        REG_CONV_P_FWD(CPU_INSTANCE_X64_AVX512(jit_avx512_core_x8s8s32x_convolution_fwd_t))
        REG_CONV_P_FWD(CPU_INSTANCE_X64_AVX2(jit_uni_x8s8s32x_1x1_convolution_fwd_t<avx2>))
        REG_CONV_P_FWD(CPU_INSTANCE_X64_AVX2(jit_uni_x8s8s32x_convolution_fwd_t<avx2>))
        REG_CONV_P_FWD(CPU_INSTANCE_X64(jit_uni_x8s8s32x_1x1_convolution_fwd_t<sse41>))
        REG_CONV_P_FWD(CPU_INSTANCE_X64(jit_uni_x8s8s32x_convolution_fwd_t<sse41>))
        CPU_INSTANCE_AARCH64(jit_sve_512_x8s8s32x_convolution_fwd_t<s8, f32>)
        REG_CONV_P_FWD(CPU_INSTANCE(_gemm_x8s8s32x_convolution_fwd_t<s8, f32>))
        REG_CONV_P_FWD(CPU_INSTANCE(ref_convolution_int8_fwd_t))
        REG_CONV_P_FWD(CPU_INSTANCE(ref_fused_convolution_fwd_t))
        nullptr,
    }},
```

Additionally, to overcome some of [2], instantiations of same classes for
different ISA should be wrapped with macros as well. Otherwise, those instances
would still present in the final binary:

```cpp
// src/cpu/x64/jit_uni_x8s8s32x_convolution.cpp:

...
REG_ISA_SSE41(template struct jit_uni_x8s8s32x_convolution_fwd_t<sse41>);
REG_ISA_AVX2(template struct jit_uni_x8s8s32x_convolution_fwd_t<avx2>);
```

Implementation details are subject to change and only for educational purposes
here to shed a light on a mechanism.

EOD.
