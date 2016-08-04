#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#include <assert.h>

#include "mkl_dnn.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"

uint32_t tensor_volume(const mkl_dnn::tensor::dims &t)
{
    uint32_t x = 1;
    for (uint32_t i = 0; i < t.size(); ++i) x *= t[i];
    return x;
}

int doit(bool lazy) {
    using namespace mkl_dnn;

    /* AlexNet: c3
     * {2, 256, 13, 13} (x) {384, 256, 3, 3} -> {2, 384, 13, 13}
     * pad: {1, 1}
     * strides: {1, 1}
     */

    printf("There are %zu CPU engines\n", engine::get_count(engine::cpu));
    auto cpu_engine = engine(lazy ? engine::cpu_lazy : engine::cpu, 0);

    // TODO: make tensor desc optional and default to N C X1 .. XN

    // XXX: descs for memory should be not necessary!

    const uint32_t mb = 2;
    const uint32_t groups = 2;
    tensor::dims input_tz = {mb, 256, 13, 13};
    tensor::dims weights_tz = {groups, 384/groups, 256/groups, 3, 3};
    tensor::dims bias_tz = {384};
    tensor::dims strides = {1, 1};
    tensor::nd_offset padding = {0, 0};
    tensor::dims output_tz = {mb, 384,
        (input_tz[2] + 2*padding[0] - weights_tz[3])/strides[0] + 1,
        (input_tz[3] + 2*padding[1] - weights_tz[4])/strides[1] + 1,
    };

    /* prepare actual data */
    float *input = new float[tensor_volume(input_tz)];
    for (size_t i = 0; i < tensor_volume(input_tz); ++i) input[i] = 0;
    float *weights = new float[tensor_volume(weights_tz)];
    for (size_t i = 0; i < tensor_volume(weights_tz); ++i) weights[i] = 0;
    float *bias = new float[tensor_volume(bias_tz)];
    for (size_t i = 0; i < tensor_volume(bias_tz); ++i) bias[i] = i;
    float *output = new float[tensor_volume(output_tz)];
    for (size_t i = 0; i < tensor_volume(output_tz); ++i) output[i] = 0;

    /* mkl-dnn starts here */
    auto c3_src_desc = memory::desc({1, 1, 2, input_tz}, memory::precision::f32, memory::format::nchw);
    auto c3_weights_desc = memory::desc({1, 2, 2, weights_tz}, memory::precision::f32, memory::format::goihw);
    auto c3_bias_desc = memory::desc({0, 0, 1, bias_tz}, memory::precision::f32, memory::format::n);
    auto c3_dst_desc = memory::desc({1, 1, 2, output_tz}, memory::precision::f32, memory::format::nchw);

    auto c3_src = memory({c3_src_desc, cpu_engine}, input);
    auto c3_weights = memory({c3_weights_desc, cpu_engine}, weights);
    auto c3_bias = memory({c3_bias_desc, cpu_engine});
    auto c3_dst = memory({c3_dst_desc, cpu_engine});

#if 0
    auto c3_desc = convolution::desc(prop_kind::forward, convolution::direct,
            c3_src_desc, c3_weights_desc, c3_bias_desc, c3_dst_desc,
            {0, 0, 1, 1}, {0, 0, 1, 1}, padding_kind::zero);
    auto c3_primitive_desc = convolution::primitive_desc(c3_desc, cpu_engine);
    auto c3 = convolution(c3_primitive_desc,
            c3_src, c3_weights, c3_bias, c3_dst);
#else
    auto c3 = convolution(prop_kind::forward, convolution::direct,
            c3_src, c3_weights, c3_bias, c3_dst,
            strides, padding, padding_kind::zero);
#endif

    stream().submit({c3}).wait();

    return 0;
}

#pragma GCC diagnostic pop

int main(int argc, char **argv) {
    int rc = doit(false);
    printf("eager: %s\n", rc ? "failed" : "passed");
    rc = doit(true);
    printf("lazy: %s\n", rc ? "failed" : "passed");
    return rc;
}
