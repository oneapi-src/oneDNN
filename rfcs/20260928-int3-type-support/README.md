# Support for 3-bit unsigned integer type

## Introduction

Weight-only quantization (WOQ) is an important mechanism to fit LLMs into the memory of client devices.
Currently oneDNN supports 8-bit, 4-bit (`s4`/`u4`) and 2-bit (`u2`) integer weights.
Idea for 3-bit weights (`u3`) is that they would need 25% less memory than 4-bit weights while providing
the same accuracy.

Current driver request is OpenVINO GPU plugin, so that one could run LLMs, both dense and Mixture-of-Experts (MoE), on client platforms.

This RFC proposes `dnnl_u3`, specify storage scheme, and share some configurations and preliminary results
for 3-bit unsigned integer data type for matmul weights.

## Motivation

- **Memory.** Compressed weights are 25% smaller with, e.g. whole-model GPU memory dropped by 18%
  (as measured by OV).
- **Accuracy.** int3 is similar to int4.
- **Performance expectations.** With 25% less weight data to read, `u3` can be
  up to about 1.3x faster than `u4`, where reading memory is the
  bottleneck, e.g. MoE expert layers or memory-bound dense shapes.
  Compute-heavy layers will be on par with `u4`.
- **Interim performance.** Early kernel measurements show the MoE expert
  layers 1.16x-1.30x faster than `u4`, and dense layers on par, that supports expectations.

## Proposal

### API

A new value is appended to the data type enum:

```c
typedef enum {
    ...
    /// 2-bit unsigned integer.
    dnnl_u2 = 17,
    /// 3-bit unsigned integer.
    dnnl_u3 = 18,
    ...
} dnnl_data_type_t;
```

The C++ API gets the matching `dnnl::memory::data_type::u3`.

### Storage format

`u3` is an unsigned integer type with the value range `[0, 7]`.

`u3` values are stored as a dense LSB-first bit stream.
The element with memory index `i` occupies bits `[3i, 3i + 2]`:

```
bit:     7    6    5    4    3    2    1    0
byte 0:  v2.1 v2.0 v1.2 v1.1 v1.0 v0.2 v0.1 v0.0
byte 1:  v5.0 v4.2 v4.1 v4.0 v3.2 v3.1 v3.0 v2.2
byte 2:  v7.2 v7.1 v7.0 v6.2 v6.1 v6.0 v5.2 v5.1
```

This is aligned with the OpenVINO `u3` layout [^1].

### Scope of the Initial Support

- matmul (dense layers), grouped matmul (MoE)
- Weights are in `u3`, plain layouts with `K` contiguous (`ba`, `acb`, `cab`)
  - Weights scales: per `N` or per `N` and per 64/128 elements along `K`; `f16`
  - Weight zero points: none; common (`u8` or `s8`); per `N` or per group (`u8` or `u4`)
- Activations are in `f16` (WOQ) as well as `s8` (dynamic quantization) with `f16` scales (per token or per 64 elements along `K`)
- Destination is in `f16`, `f32`
- Bias (`f16`, per `N`) and post-ops (dense matmul): `eltwise_logistic`, `eltwise_swish`, `binary_add`, `binary_mul`

Relevant benchdnn cases:

```sh
# Dense, WOQ: scales and zero points per 64 elements along K
benchdnn --matmul --engine=gpu --dt=f16:u3:f16 --wtag=ba --attr-scales=wei:3:f16:64x1 --attr-zero-points=wei:3:u8:64x1 --attr-fpmath=f16:true 64x2048:2048x512
# Dense, WOQ: scales per 128 elements along K, common s8 zero point, binary post-op
benchdnn --matmul --engine=gpu --dt=f16:u3:f16 --wtag=cab --attr-scales=wei:7:f16:128x1 --attr-zero-points=wei:common:1:s8 --attr-post-ops=binary_add:f16:6:abc --attr-fpmath=f16:true 1x16x12288:1x12288x4096
# Dense, dynamic quantization: s8 activations with per-token scales
benchdnn --matmul --engine=gpu --dt=s8:u3:f16 --stag=abc --wtag=acb --dtag=abc --attr-scales=src:7:f16:1x4096+wei:7:f16:128x1 --attr-zero-points=wei:common:1:s8 720x1x4096:1x4096x1024
# MoE: grouped matmul with 256 experts
benchdnn --matmul --engine=gpu --dt=f16:u3:f16 --wtag=acb --grouped=0:256:balanced --attr-scales=wei:7:f16:64x1 --attr-zero-points=wei:7:u4:64x1 --attr-fpmath=f16:true 8192x2048:256x2048x512
```

- PR 5726 [^2] (draft): API, CPU and GPU reference matmul and grouped
  matmul, CPU reorder, benchdnn.
- PR 5988 [^3] (draft): optimized Intel GPU matmul and grouped matmul, used
  for the measurements above.
- No optimized CPU implementation is planned for now.


[^1]: https://github.com/openvinotoolkit/openvino/pull/37430 (OpenVINO: "U3/U6 tight layout")
[^2]: https://github.com/uxlfoundation/oneDNN/pull/5726
[^3]: https://github.com/uxlfoundation/oneDNN/pull/5988
