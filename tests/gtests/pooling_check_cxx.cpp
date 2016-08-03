#include <cstddef>
#include <cstdio>

#include "mkl_dnn.hpp"

typedef float real_t;
void init_input(const mkl_dnn::tensor::dims &dim, real_t *x);
int check_output(const mkl_dnn::tensor::dims &dim, const real_t *x);
uint32_t tensor_volume(const mkl_dnn::tensor::dims &t);
int doit(bool lazy);

using std::ptrdiff_t;
#include "gtest/gtest.h"

TEST(pooling_tests, pooling_check_cxx) {
    int n_errors = doit(false);
    printf("Eager engine\n");
    EXPECT_EQ(n_errors, 0);
    n_errors = doit(true);
    printf("Lazy engine\n");
    EXPECT_EQ(n_errors, 0);
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"

void init_input(const mkl_dnn::tensor::dims &dim, real_t *x)
{
    uint32_t N = dim[0], C = dim[1], H = dim[2], W = dim[3];
    for (uint32_t n = 0; n < N; n += 1)
    for (uint32_t c = 0; c < C; c += 1)
    for (uint32_t h = 2; h+2 <= H; h += 2)
    for (uint32_t w = 2; w+2 <= W; w += 2)
        x[w + W*h + c*W*H + n*W*H*C] = c*n;
}

int check_output(const mkl_dnn::tensor::dims &dim, const real_t *x)
{
    int n_errors = 0;
    uint32_t N = dim[0], C = dim[1], H = dim[2], W = dim[3];
    for (uint32_t n = 0; n < N; ++n)
    for (uint32_t c = 0; c < C; ++c)
    for (uint32_t h = 0; h < H; ++h)
    for (uint32_t w = 0; w < W; ++w)
    {
        if (x[w + W*h + c*W*H + n*W*H*C] != c*n) n_errors += 1;
    }
    return n_errors;
}

uint32_t tensor_volume(const mkl_dnn::tensor::dims &t)
{
    uint32_t x = 1;
    for (uint32_t i = 0; i < t.size(); ++i) x *= t[i];
    return x;
}

int doit(bool lazy) {
    using namespace mkl_dnn;

    /* AlexNet: p1
     * {16, 96, 55, 55} -> {16, 96, 27, 27}
     * pad: {0, 0}
     * strides: {2, 2}
     * kernel: {3, 3}
     */

    printf("There are %zu CPU engines\n", engine::get_count(engine::cpu));
    auto cpu_engine = engine(lazy ? engine::cpu_lazy : engine::cpu, 0);

    auto p1_input_desc   = memory::desc({1, 1, 2, {16, 96, 55, 55}}, memory::precision::f32, memory::format::nchw);
    auto p1_indices_desc = memory::desc({1, 1, 2, {16, 96, 27, 27}}, memory::precision::f32, memory::format::nchw);
    auto p1_output_desc  = memory::desc({1, 1, 2, {16, 96, 27, 27}}, memory::precision::f32, memory::format::nchw);

    real_t *input   = new real_t[tensor_volume({ 16, 96, 55, 55 })];
    real_t *indices = new real_t[tensor_volume({ 16, 96, 27, 27 })];
    real_t *output  = new real_t[tensor_volume({ 16, 96, 27, 27 })];

    auto p1_input   = memory({ p1_input_desc , cpu_engine}, input  );
    auto p1_indices = memory({p1_indices_desc, cpu_engine}, indices);
    auto p1_output  = memory({p1_output_desc , cpu_engine}, output );

    auto p1 = pooling(prop_kind::forward, pooling::max, p1_input, p1_indices, p1_output,
        {2, 2}, {3, 3}, {0, 0}, padding_kind::zero);

    init_input({16, 96, 55, 55}, input);
    stream().submit({p1}).wait();
    int n_errors = check_output({ 16, 96, 27, 27 }, output);

    delete[] input;
    delete[] indices;
    delete[] output;

    return n_errors;
}

#pragma GCC diagnostic pop
