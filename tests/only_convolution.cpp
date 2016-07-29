#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#include <assert.h>

#include "mkl_dnn.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"

int doit() {
    using namespace mkl_dnn;

    /* AlexNet: c3
     * {2, 256, 13, 13} (x) {384, 256, 3, 3} -> {2, 384, 13, 13}
     * pad: {1, 1}
     * strides: {1, 1}
     */

    printf("There are %zu CPU engines\n", engine::get_count(engine::cpu));
    auto cpu_engine = engine(engine::cpu, 0);

    // TODO: make tensor desc optional and default to N C X1 .. XN

    // XXX: descs for memory should be not necessary!

    auto c3_input_desc = memory::desc({1, 1, 2, {2, 256, 13, 13}}, memory::precision::f32, memory::format::nchw);
    auto c3_weights_desc = memory::desc({0, 2, 2, {384, 256, 3, 3}}, memory::precision::f32, memory::format::oihw);
    auto c3_bias_desc = memory::desc({0, 0, 1, {384}}, memory::precision::f32, memory::format::n);
    auto c3_output_desc = memory::desc({1, 1, 2, {2, 384, 13, 13}}, memory::precision::f32, memory::format::nchw);

    auto c3_input = memory({c3_input_desc, cpu_engine});
    auto c3_weights = memory({c3_weights_desc, cpu_engine});
    auto c3_bias = memory({c3_bias_desc, cpu_engine});
    auto c3_output = memory({c3_output_desc, cpu_engine});

#if 0
    auto c3_input = memory({{{1, 1, 2, {2, 256, 13, 13}}, memory::format::nchw}, cpu_engine});
    auto c3_weights = memory({{{0, 2, 2, {384, 256, 3, 3}}, memory::format::oihw}, cpu_engine});
    auto c3_bias = memory({{{0, 0, 1, {384}}, memory::format::n}, cpu_engine});
    auto c3_output = memory({{{1, 1, 2, {2, 384, 13, 13}}, memory::format::nchw}, cpu_engine});

    auto c3_desc = convolution::desc(prop_kind::forward, convolution::direct,
            c3_input, c3_weights, c3_bias, c3_output,
            {0, 0, 1, 1}, {0, 0, 1, 1}, padding_kind::zero);
    auto c3_primitive_desc = convolution::primitive_desc(c3_desc, cpu_engine);
    auto c3 = convolution(c3_primitive_desc,
            c3_input, c3_weights, c3_bias, c3_output);
#endif

#if 0
    auto c3_desc = convolution::desc(prop_kind::forward, convolution::direct,
            c3_input_desc, c3_weights_desc, c3_bias_desc, c3_output_desc,
            {0, 0, 1, 1}, {0, 0, 1, 1}, padding_kind::zero);
    auto c3_primitive_desc = convolution::primitive_desc(c3_desc, cpu_engine);
    auto c3 = convolution(c3_primitive_desc,
            c3_input, c3_weights, c3_bias, c3_output);
#else
    auto c3 = convolution(prop_kind::forward, convolution::direct,
            c3_input, c3_weights, c3_bias, c3_output,
            {1, 1}, {1, 1}, padding_kind::zero);
#endif

    stream().submit({c3}).wait();

    return 0;
}

#pragma GCC diagnostic pop

int main(int argc, char **argv) {
    int rc = doit();
    printf("%s\n", rc ? "failed" : "passed");
    return rc;
}
