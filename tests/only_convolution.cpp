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
     * {256, 256, 13, 13} (x) {384, 256, 3, 3} -> {256, 384, 13, 13}
     * pad: {1, 1}
     * strides: {1, 1}
     */

    printf("There are %zu CPU engines\n", engine::get_count(engine::cpu));
    auto cpu_engine = engine(engine::cpu, 0);

    // TODO: make tensor desc optional and default to N C X1 .. XN

    auto c3_input_tensor_desc = tensor::desc(1, 1, 2, {256, 256, 13, 13});
    auto c3_input_memory_desc = memory::desc({1, 1, 2, {256, 256, 13, 13}}, memory::format::nchw_f32);
    auto c3_input_primitive_desc = memory::primitive_desc({{1, 1, 2, {256, 256, 13, 13}}, memory::format::nchw_f32}, cpu_engine);

    auto c3_input = memory({{{1, 1, 2, {256, 256, 13, 13}}, memory::format::nchw_f32}, cpu_engine});
    auto c3_weights = memory({{{0, 2, 2, {384, 256, 3, 3}}, memory::format::oihw_f32}, cpu_engine});
    auto c3_bias = memory({{{0, 0, 1, {384}}, memory::format::n_f32}, cpu_engine});
    auto c3_output = memory({{{1, 1, 2, {256, 384, 13, 13}}, memory::format::nchw_f32}, cpu_engine});

    // auto c3 = convolution::create({propagation::forward, algorithm::direct, {0, 0, 1, 1}, {0, 0, 1, 1}, padding::zero}, {input, weights, bias}, {output});

    // stream::create().submit({c3}).wait();

    return 0;
}

#pragma GCC diagnostic pop

int main(int argc, char **argv) {
    int rc = doit();
    printf("%s\n", rc ? "failed" : "passed");
    return rc;
}
