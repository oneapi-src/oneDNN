# RFC: Depthwise post-op extension

## Introduction
oneDNN provides a performance optimization for a chain of convolutions, when 1x1 convolution is followed by 3x3 depthwise convolution with stride of 1 or 2.
This optimization is provided through an attribute post-op API.
To keep usability in place, the decision in the past was to hardcode depthwise convolution settings into a post-op name.
Thus, `k3s1p1` or `k3s2p1` options are only allowed.
This optimization targeted only MobileNet topologies popular back then.
Now there are models that have same pattern of 1x1 + depthwise, but they might have different depthwise convolution settings.
E.g., they could have kernel size of 3, 5, or 7; stride size of 1, or 2; (left) padding size of 0, or 1.
In addition, it may happen that certain convolutions have right padding greater than left, especially when padding size is 0.
Extended API will allow to fuse those convolutions and may provide 5%-15% speed up on a model level as measured by TensorFlow benchmarks.

## Proposal
Support more shapes that appear for depthwise convolutions through a more general API for depthwise post-op in oneDNN.
Instead of hardcoding parameters in the name, add them as an explicit argument for a call.
If parameters form a valid convolution setup, at least the implementation that calls two optimized implementations should be invoked.
Performance assumptions hold for extended API - it provides better, or on-par performance compared to unfused implementation.
Data type coverage remains the same as it was before.
The API targets the oneDNN v2.6 Update release to intercept the next TensorFlow v2.10 release.
The next oneDNN v3.0 Major release will have previous version of API removed since new one covers previously supported cases.

## API
```c
/// dnnl.h

dnnl_status_t DNNL_API dnnl_post_ops_append_dw(dnnl_post_ops_t post_ops,
        dnnl_data_type_t weights_data_type, dnnl_data_type_t bias_data_type,
        dnnl_data_type_t dst_data_type, dnnl_dim_t kernel_size,
        dnnl_dim_t stride_size, dnnl_dim_t padding_l_size, dnnl_dim_t count,
        int mask, const float *scales);

dnnl_status_t DNNL_API dnnl_post_ops_get_params_dw(
        const_dnnl_post_ops_t post_ops, int index,
        dnnl_data_type_t *weights_data_type, dnnl_data_type_t *bias_data_type,
        dnnl_data_type_t *dst_data_type, dnnl_dim_t *kernel_size,
        dnnl_dim_t *stride_size, dnnl_dim_t *padding_l_size, dnnl_dim_t *count,
        int *mask, const float **scales);

```

```cpp
/// dnnl.hpp

struct post_ops : public handle<dnnl_post_ops_t> {
    ...

    void append_dw(memory::data_type weights_data_type,
            memory::data_type bias_data_type, memory::data_type dst_data_type,
            memory::dim kernel_size, memory::dim stride_size,
            memory::dim padding_l_size, int mask,
            const std::vector<float> &scales) { ... }

    void get_params_dw(int index, memory::data_type &weights_data_type,
            memory::data_type &bias_data_type, memory::data_type &dst_data_type,
            memory::dim &kernel_size, memory::dim &stride_size,
            memory::dim &padding_l_size, int &mask,
            std::vector<float> &scales) const { ... }
    ...
}
```

## Validation
Benchdnn parser will be extended to accept any kernel, stride or padding values and will trigger new API to construct a post-op.
