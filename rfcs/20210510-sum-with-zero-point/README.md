# Proposal for support sum with zero-point

## Motivation
The motivation to provide such support is to enable dequantize the
asymmetrically quantized sum's src1 tensor to f32 domain before performing the
sum operation. In general we want to support following operation:
`dst[:] := scale * (dst[:] - zero_point) + op(...)`

## Proposal
The proposal is to extend API by adding the possibility of using such
functionality. Support for reference solutions should also be added. For
consistency also need to implement zero points support in standalone sum
primitive. The solution should also include the benchdnn extension.

## Design

Currently, the sum operation also supports scale. Zero point support should
be designed to keep backward compatibility. This function is targeting for
quantization and there is no needed to add support for selection of parameter.

### API extension

1. Overload `append_sum(scale, dt)` by adding next `append_sum(scale, zp, dt)`
function.
    - API:
        ```cpp
        // include/oneapi/dnnl/dnnl.h
        dnnl_status_t DNNL_API dnnl_post_ops_append_sum_v3(
                dnnl_post_ops_t post_ops, float scale, int32_t zero_point,
                dnnl_data_type_t data_type);

        // include/oneapi/dnnl/dnnl.hpp
        void append_sum(float scale = 1.f, int32_t zero_point,
                 memory::data_type data_type = memory::data_type::undef) {
            if (data_type == memory::data_type::undef)
                error::wrap_c_api(dnnl_post_ops_append_sum(get(), scale),
                        "could not append a sum post-op");
            else
                    error::wrap_c_api(dnnl_post_ops_append_sum_v3(
                                              get(), scale, zero_point,
                                              memory::convert_to_c(data_type)),
                            "could not append a sum post-op");
        }
        ```
    - Pros:
        - It is backwards compatible.
        - Allows to use zero point without scale (scale == 1.0).
    - Cons:
        - It is not the simplest solution
    - Example:
        ```cpp
        // Zero point is the last parameter to keep existing code working.
        attr.append_sum(scale, dt);
        attr.append_sum(scale, dt, zp);
        ```

2. Add possibility to use `dnnl_primitive_attr_set_zero_points` with sum.
    - Pros:
        - It will allow to use mask and array of zero-point values (instead of
          one scalar value).
    - Cons:
        - More complicated solution.
        - Probably requires backward incompatibility or adding a new function.
        - Probably not needed in real topologies.
    - Example:
        ```cpp
        attr.set_zero_points(
            DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx) | DNNL_ARG_SRC1, mask,
            zero_points);
        ```
### Benchdnn extension

1. Overwrite `SUM[:SCALE[:DATA_TYPE]]` post-ops parameter by
`SUM[:SCALE[:ZEROPOINT[:DATA_TYPE]]]` instance. (preferred)
    - Pros:
        - More compatible with api function - more intuitive.
    - Cons:
        - Existing tests must be changed

2. Overwrite `SUM[:SCALE[:DATA_TYPE]]` post-ops parameter by
`SUM[:SCALE[:DATA_TYPE[:ZEROPOINT]]]` instance.
    - Pros:
        - It does not require changing existing tests
    - Cons:
        - Less compatible with api function.
