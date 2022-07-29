# Data Types {#dev_guide_data_types}

Besides 32-bit IEEE single-precision floating-point data type, oneDNN Graph can
also support other data types. For low precision support, oneDNN Graph API
expects users to convert the computation graph to low precision representation
and specify the data’s precision and quantization parameters. oneDNN Graph API
implementation should strictly respect the numeric precision of the computation.

Data type | Description
-- | --|
f16 | [16-bit/half-precision floating point](https://en.wikipedia.org/wiki/Half-precision_floating-point_format)
bf16 | [non-standard 16-bit floating point with 7-bit mantissa](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format)
f32 | [32-bit/single-precision floating point](https://en.wikipedia.org/wiki/Single-precision_floating-point_format)
s32 | 32-bit signed integer
s8 | 8-bit signed integer
u8 | 8-bit unsigned integer
boolean | boolean data type. The tensor element will be interpreted with C++ bool type. Note that the size of C++ bool type is language implementation defined
