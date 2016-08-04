#include <cstddef>
#include <cstdio>

#include "mkl_dnn.hpp"

int doit(bool lazy);

using std::ptrdiff_t;
#include "gtest/gtest.h"

TEST(pooling_tests, AlexNet_p1) {
    int n_errors = 0;

    printf("Eager engine\n");
    n_errors = doit(false);
    EXPECT_EQ(n_errors, 0);

    printf("Lazy engine\n");
    n_errors = doit(true);
    EXPECT_EQ(n_errors, 0);
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"

typedef float real_t;

void init_src(const mkl_dnn::tensor::dims &dim, real_t *x)
{
    uint32_t N = dim[0], C = dim[1], H = dim[2], W = dim[3];
#   pragma omp parallel for collapse(2)
    for (uint32_t n = 0; n < N; n += 1)
    for (uint32_t c = 0; c < C; c += 1)
    for (uint32_t h = 2; h+2 <= H; h += 2)
    for (uint32_t w = 2; w+2 <= W; w += 2)
        x[w + W*h + c*W*H + n*W*H*C] = c*n;
}

int check_dst(const mkl_dnn::tensor::dims &dim, const real_t *x)
{
    int n_errors = 0;
    uint32_t N = dim[0], C = dim[1], H = dim[2], W = dim[3];
#   pragma omp parallel for collapse(4)
    for (uint32_t n = 0; n < N; ++n)
    for (uint32_t c = 0; c < C; ++c)
    for (uint32_t h = 0; h < H; ++h)
    for (uint32_t w = 0; w < W; ++w)
    {
        if (x[w + W*h + c*W*H + n*W*H*C] != c*n)
#           pragma omp atomic
            n_errors += 1;
    }
    return n_errors;
}

int doit(bool lazy) {
    using namespace mkl_dnn;

    /* AlexNet: p1
     * {16, 96, 55, 55} -> {16, 96, 27, 27}
     * strides: {2, 2}
     * kernel : {3, 3}
     * padding: {0, 0}
     */

    printf("There are %zu CPU engines\n", engine::get_count(engine::cpu));
    auto cpu_engine = engine(lazy ? engine::cpu_lazy : engine::cpu, 0);

    auto p1_src_desc     = memory::desc({1, 1, 2, {16, 96, 55, 55}}, memory::precision::f32, memory::format::nchw);
    auto p1_indices_desc = memory::desc({1, 1, 2, {16, 96, 27, 27}}, memory::precision::f32, memory::format::nchw);
    auto p1_dst_desc     = memory::desc({1, 1, 2, {16, 96, 27, 27}}, memory::precision::f32, memory::format::nchw);

    real_t *src     = new real_t[16*96*55*55]();
    real_t *indices = new real_t[16*96*27*27]();
    real_t *dst     = new real_t[16*96*27*27]();

    auto p1_src     = memory({p1_src_desc    , cpu_engine}, src    );
    auto p1_indices = memory({p1_indices_desc, cpu_engine}, indices);
    auto p1_dst     = memory({p1_dst_desc    , cpu_engine}, dst    );

    auto p1 = pooling(prop_kind::forward, pooling::max, p1_src, p1_indices, p1_dst,
        {2, 2}, {3, 3}, {0, 0}, padding_kind::zero);

    init_src({16, 96, 55, 55}, src);
    stream().submit({p1}).wait();
    int n_errors = check_dst({ 16, 96, 27, 27 }, dst);

    delete[] src;
    delete[] indices;
    delete[] dst;

    return n_errors;
}

#pragma GCC diagnostic pop
