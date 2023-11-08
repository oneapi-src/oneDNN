# Proposal for GPT-Quantization (GPT-Q) support

# 1. Introduction

The motivation for GPT-Q[[#1]][1] is to reduce memory requirements for storage and execution.
GPT-Q is an attempt to reduce model size without enforcing quantization of activation in Linear
layers. The idea is to compress weights once after training is completed. During inference,
weights are decompressed into f16/f32 data types and passed into f16/f32 Linear layers.
The decompression happens in local device memory or in registers to reduce global memory
size and bandwidth requirements.

The quantization details:
- The potential weights compression data types are integer 8, 4, 3 and even 2 bits.
- The quantization is asymmetric (both scales and zero points are used).
- The quantization scales and zero points are multidimensional and contain groups.

## OpenVino support

Recently OpenVino added support for the following receipts of weights compression to support GPT-Q:
1. int8 weights compression:
   - Scales are per-output channels, data type is f16/f32
   - Zero points are per-output channels, data type is u8/s8
2. int4 weights compression [[#2]][2] [[#3]][3]:
   - Scales are per-output and per-input channels with groups along input channels, data type is f16/f32
   - Zero points are per-output and per-input channels with groups along input channels, data type is u8/s8

# 2. Implications to oneDNN

There are 3 new quantization features that can't be supported by the current oneDNN API:
- scales and zero points data type. Currently oneDNN defines scales as f32 data and zero points as s32 data.
- scales and zero points groups support. Currently scales and zero points are set using a mask where each
  bit represents if for a given dimension multiple or a single scale/zero point is used.
- Weights quantization applied to primitives with non-integer activations. This part is covered by a separate RFC: [[#4]][4]

# 3. Proposal

## Scales and zero points data type support

### Option a (recommended)

Extend scales and zero points setting API with a parameter for data type. This additional information will allow kernels to generate
the best code for a given data type.

```cpp
// C API
status_t dnnl_primitive_attr_set_scales(
        primitive_attr_t *attr, int arg, ..., dnnl_data_type_t data_type);

// C++ API
void set_scales_mask(int arg, ..., data_type_t data_type);

// An example
primitive_attr attr;
attr.set_scales(DNNL_ARG_DST, ..., dt::f16);
...
auto dst_scales_md = memory::desc({1}, dt::f16, tag::x);
auto dst_scales = memory(dst_scales_md, engine);
...
p.execute(s, {{DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, dst_scales}, ...});
```

### Option b

Keep data types implicit during scales and zero points setup, but detect the data type from the memory objects passed at execution.

```cpp
// An example
primitive_attr attr;
attr.set_scales(DNNL_ARG_DST, ...);
...
auto dst_scales_md = memory::desc({1}, dt::f16, tag::x);
auto dst_scales = memory(dst_scales_md, engine);
...
p.execute(s, {{DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, dst_scales} ...});
```

## Scales and zero points groups support

## Option 1.a

Extend scales and zero points setting API with a parameter for groups.
The upside of this option is attributes are kept separate from the actual dimensions
and this might help in case of runtime dimensions are involved in the future.
`DNNL_RUNTIME_DIM_VAL` is a special group size value to indicate that the i-th
dimension uses a common scale/zero point. It is needed to avoid putting the actual
dimension.

```cpp
// C API
status_t dnnl_primitive_attr_set_scales(primitive_attr_t *attr,
        int arg, int ndims, const dims_t groups, dnnl_data_type_t data_type);

// C++ API
void set_scales(int arg, dims groups, memory::data_type data_type);

// Examples
// Common scales
primitive_attr attr_common;
attr_common.set_scales(DNNL_ARG_DST, {}, data_type);
attr_common.set_scales(DNNL_ARG_DST, {DNNL_RUNTIME_DIM_VAL}, data_type);
// PER_OC, no groups
primitive_attr attr_per_dim0;
attr_per_dim0.set_scales(DNNL_ARG_DST, {1}, data_type);
// PER_OC + groups
primitive_attr attr_per_dim0g;
attr_per_dim0g.set_scales(DNNL_ARG_DST, {G}, data_type);
// PER_IC
primitive_attr attr_per_dim1;
attr_per_dim1.set_scales(DNNL_ARG_DST, {DNNL_RUNTIME_DIM_VAL, 1}, data_type);
// PER_OC and PER_IC + groups
primitive_attr attr_per_dim0_dim1g;
attr_per_dim0_dim1g.set_scales(DNNL_ARG_DST, {1, G}, data_type);
// PER_OC + groups and PER_IC + groups
primitive_attr attr_per_dim0g_dim1g;
attr_per_dim0g_dim1g.set_scales(DNNL_ARG_DST, {G, G}, data_type);
```

## Option 1.b (recommended)

Extend scales and zero points setting API with a parameter for groups.
The upside of this option is attributes are kept separate from the actual dimensions
and this might help in case of runtime dimensions are involved in the future.
To remove the need for a special group size value mask is kept as a parameter:
- If i-th bit in the mask is not set, the groups[i] is ignored.
- If i-th bit in the mask is set, the group_size is taken into account and a user
  would use groups[i]=1 for per_dim semantics or groups[i]=G for groups scales/zero points

Another benefit of this option is it cover the existing API `set_scales_mask`, so the existing API
can be retired in the future releases.

```cpp
// C API
status_t dnnl_primitive_attr_set_scales(primitive_attr_t *attr,
        int arg, int mask, int ndims, const dims_t groups, dnnl_data_type_t data_type);

// C++ API
void set_scales(int arg, int mask, dims groups, memory::data_type data_type);

// Examples
// Common scales
primitive_attr attr_common;
attr_common.set_scales(DNNL_ARG_DST, 0, data_type);
// PER_OC, no groups
primitive_attr attr_per_dim0;
attr_per_dim0.set_scales(DNNL_ARG_DST, 1 << 0, data_type);
attr_per_dim0.set_scales(DNNL_ARG_DST, 1 << 0, {1}, data_type);
// PER_OC + groups
primitive_attr attr_per_dim0g;
attr_per_dim0g.set_scales(DNNL_ARG_DST, 1 << 0, {G}, data_type);
// PER_IC
primitive_attr attr_per_dim1;
// groups[0] is ignored because mask & (1 << 0) is 0
attr_per_dim1.set_scales(DNNL_ARG_DST, 1 << 1, {/* ignored */0, 1}, data_type);
// PER_OC and PER_IC + groups
primitive_attr attr_per_dim0_dim1g;
attr_per_dim0_dim1g.set_scales(DNNL_ARG_DST, (1 << 1) | (1 << 0), {1, G}, data_type);
// PER_OC + groups and PER_IC + groups
primitive_attr attr_per_dim0g_dim1g;
attr_per_dim0g_dim1g.set_scales(DNNL_ARG_DST, (1 << 1) | (1 << 0), {G, G}, data_type);
```


## Option 2

Extend scales and zero points setting API with a parameter for dims.
This option is implemented in the fork of oneDNN in OpenVino side.
The downside of this option is that the dimensions of a primitive are now part of attributes.
As a result the same attributes can't be reused for multiple primitives or pasted as is into
nested primitives that compute a part of dimensions (for example, BRGEMM-based primitives).

```cpp
// C API
status_t dnnl_primitive_attr_set_scales(primitive_attr_t *attr, int arg,
        int ndims, const dims_t dims, dnnl_data_type_t data_type);
// C++ API
void set_scales(int arg, dims dims, memory::data_type data_type);

// Examples
// Common scales
primitive_attr attr_common;
attr_common.set_scales_mask(DNNL_ARG_DST, {}, data_type);
attr_common.set_scales(DNNL_ARG_DST, {1}, data_type);
// PER_OC, no groups
primitive_attr attr_per_dim0;
attr_per_dim0.set_scales(DNNL_ARG_DST, {OC}, data_type);
// PER_OC + groups
primitive_attr attr_per_dim0g;
attr_per_dim0g.set_scales(DNNL_ARG_DST, {OC / G}, data_type);
// PER_IC
primitive_attr attr_per_dim1;
attr_per_dim1.set_scales(DNNL_ARG_DST, {1, IC}, data_type);
// PER_OC and PER_IC + groups
primitive_attr attr_per_dim0_dim1g;
attr_per_dim0_dim1g.set_scales(DNNL_ARG_DST, {OC, IC / G}, data_type);
// PER_OC + groups and PER_IC + groups
primitive_attr attr_per_dim0g_dim1g;
attr_per_dim0g_dim1g.set_scales(DNNL_ARG_DST, {OC / G, IC / G}, data_type);
```

## Option 3

Extend scales and zero points setting API with a parameter for groups dimension and groups value.
This option is very limited and is focused on the current needs with an assumption that there will
be no groups across multiple dimensions.

```cpp
// C API
status_t dnnl_primitive_attr_set_scales(primitive_attr_t *attr, int arg,
        int mask, int group_dim, dim_t group_size, dnnl_data_type_t data_type);
// C++ API
void set_scales(int arg, int mask, int group_dim, dim group, memory::data_type data_type);

// Examples
// Common scales
primitive_attr attr_common;
attr_common.set_scales(DNNL_ARG_DST, 0,
    /* group dim is ignored when mask is 0 */ 0,
    /* group is ignored when mask is 0 */ 0, data_type);
// PER_OC, no groups
primitive_attr attr_per_dim0;
attr_per_dim0.set_scales(DNNL_ARG_DST, 1 << 0, 0, 1, data_type);
// PER_OC + groups
primitive_attr attr_per_dim0g;
attr_per_dim0g.set_scales(DNNL_ARG_DST, 1 << 0, 0, G, data_type);
// PER_IC
primitive_attr attr_per_dim1;
attr_per_dim1.set_scales(DNNL_ARG_DST, 1 << 1, 0, 1, data_type);
// PER_OC and PER_IC + groups
primitive_attr attr_per_dim0_dim1g;
attr_per_dim0_dim1g.set_scales(DNNL_ARG_DST, (1 << 1) | (1 << 0), 1, G, data_type);
// PER_OC + groups and PER_IC + groups
// Not possible
```

# References

1. [GPTQ: ACCURATE POST-TRAINING QUANTIZATION FOR GENERATIVE PRE-TRAINED TRANSFORMERS][1]
2. [OpenVino: [CPU] FullyConnected acceleration with 4bit weights decompression][2]
3. [OpenVino: [GPU] FC with 4-bit weights compression support draft][3]
4. [oneDNN: Proposal for weight decompression][4]

[1]: https://arxiv.org/pdf/2210.17323.pdf
[2]: https://github.com/openvinotoolkit/openvino/pull/20607
[3]: https://github.com/openvinotoolkit/openvino/pull/20572
[4]: https://github.com/oneapi-src/oneDNN/pull/1736 

---

EOD
