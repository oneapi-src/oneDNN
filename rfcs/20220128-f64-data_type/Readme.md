# RFC: f64 data type support

## Introduction & Motivation

Some deep learning applications require higher numerical accuracy than traditionally 
supported 32-bit floating points, especially for training phase. This RFC addresses
this requirement by introducing 64-bit, double precision float support in oneDNN.
We propose adding f64 data type support for convolution on the GPU. Other primitives, 
or engines are not impacted at this point; although library infrastrcture is modified
to recognize double-precision floating-point data type.

## API changes

Here's the list of of the changes in library header files:

include/oneapi/dnnl/dnnl.hpp:

```c
    /// Data type specification.
    enum class data_type {
        /// Undefined data type (used for empty memory descriptors).
        undef = dnnl_data_type_undef,
        /// [16-bit/half-precision floating point](https://en.wikipedia.org/wiki/Half-precision_floating-point_format).
        f16 = dnnl_f16,
        /// non-standard
        /// [16-bit floating point with 7-bit mantissa](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format).
        bf16 = dnnl_bf16,
        /// [32-bit/single-precision floating point](https://en.wikipedia.org/wiki/Single-precision_floating-point_format).
        f32 = dnnl_f32,
        //// [64-bit/double-precision floating point](https://en.wikipedia.org/wiki/Double-precision_floating-point_format).
        f64 = dnnl_f64,
        /// 32-bit signed integer.
        s32 = dnnl_s32,
        /// 8-bit signed integer.
        s8 = dnnl_s8,
        /// 8-bit unsigned integer.
        u8 = dnnl_u8,
    };
```

include/oneapi/dnnl/dnnl_types.h:

```c
    typedef enum {
        /// Undefined data type, used for empty memory descriptors.
        dnnl_data_type_undef = 0,
        /// 16-bit/half-precision floating point.
        dnnl_f16 = 1,
        /// non-standard 16-bit (bfloat16 w/ 7 bit mantissa) floating point.
        dnnl_bf16 = 2,
        /// 32-bit/single-precision floating point.
        dnnl_f32 = 3,
        /// 32-bit signed integer.
        dnnl_s32 = 4,
        /// 8-bit signed integer.
        dnnl_s8 = 5,
        /// 8-bit unsigned integer.
        dnnl_u8 = 6,
        /// 64-bit/double-precision floating point.
        dnnl_f64 = 7,
    } dnnl_data_type_t;
```

## Some details on Convolution primitive

Convolution primitive in particular has several configurations and parameters that 
deserves to be discussed in more detail with regards to f64 data type.
- Table in https://oneapi-src.github.io/oneDNN/dev_guide_convolution.html describes the supported configurations in the convolution api. We suggest adding one row, for which source, weights, destination and bias are all f64; for both forward and backward propagation.
- Data Representation: f64 conv will be suppored for all the representations which f32 was supported on.
- Postops & attributes: all post-ops and attributes supported for f32 configuration will be supported for f64 on GPU.
- Scales: scaling values will remain in f32 in order to minimize API changes. These values will be upconverted to f64 (accumulate data type) and the math will occur in f64.

## Known limitations

- Reference computation in benchdnn is remained in f32. This decision is made in order to minimiaze changes in benchdnn infrastructure which uses f32 for reference computation, until some feedback is collected on this feature.
