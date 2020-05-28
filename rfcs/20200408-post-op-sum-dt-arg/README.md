# Introducing data type argument for sum post operation

## Introduction

When executing an int8 ResNet50 and other networks it is a frequent situation
that the convolutions getting summed have different input types -- u8 and s8
respectively. In such situation frameworks have to add extra reorders to be
able to use sum post operation.

In current implementation data type for sum is taken as destination data type,
it would be more useful to set data type for sum input buffer.

## Proposal

Additional argument for `append_sum`(C++ API) and additional function
`dnnl_post_ops_append_sum_v2`(C API)

```c
dnnl_status_t DNNL_API dnnl_post_ops_append_sum_v2(
        dnnl_post_ops_t post_ops, float scale, dnnl_data_type_t data_type);
```

```c++
void append_sum(float scale = 1.,
        memory::data_type data_type = memory::data_type::undef);
```

And additional functions for getting parameters

```c
dnnl_status_t DNNL_API dnnl_post_ops_get_params_sum_v2(
        const_dnnl_post_ops_t post_ops, int index, float *scale,
        dnnl_data_type_t *data_type);
```

```c++
void get_params_sum(
        int index, float &scale, memory::data_type &data_type)
```

In case of undefined data type post option will work as before (with
destination data type).  When using sum v2 with `get_params_sum` function,
result will be scale value as for sum v1. Sum kind will have an additional
member, which will have undefined value in case of default/old sum behavior.

```c
struct {
    float scale;
    dnnl::impl::data_type_t dt;
} sum;

```

For internal usage for implementations without such functionality execution can
be skipped using `skip_mask_t::sum_dt`. By default, situations with different
data type sizes for sum and destination can be filtered by `has_default_values`
function.
