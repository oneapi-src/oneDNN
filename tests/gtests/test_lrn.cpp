#include <cstddef>
#include <cstdio>

#include "mkl_dnn.hpp"

static int doit(bool lazy);

using std::ptrdiff_t;
#include "gtest/gtest.h"
#include <cmath>

TEST(normalization_tests, AlexNet_n1) {
    int n_errors = 0;

    n_errors = doit(false);
    EXPECT_EQ(n_errors, 0);

    n_errors = doit(true);
    EXPECT_EQ(n_errors, 0);
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"

typedef float real_t;

static void init_src(const mkl_dnn::tensor::dims &dim, real_t *x)
{
    uint32_t N = dim[0], C = dim[1], H = dim[2], W = dim[3];
#   pragma omp parallel for collapse(2)
    for (uint32_t n = 0; n < N; n += 1)
    for (uint32_t c = 0; c < C; c += 1)
    for (uint32_t h = 0; h < H; h += 1)
    for (uint32_t w = 0; w < W; w += 1)
        x[w + W*h + c*W*H + n*W*H*C] = 1;
}

static int check_dst(const mkl_dnn::tensor::dims &dim, double a, double b, double n, const real_t *x)
{
    uint32_t N = dim[0], C = dim[1], H = dim[2], W = dim[3];
    int n_errors = 0;
#   pragma omp parallel for collapse(4)
    for (uint32_t n = 0; n < N; ++n)
    for (uint32_t c = 0; c < C; ++c)
    for (uint32_t h = 0; h < H; ++h)
    for (uint32_t w = 0; w < W; ++w)
    {
        if (std::abs(x[w + W*h + c*W*H + n*W*H*C] - std::pow(1 + a, b)) >= 1e-7)
#           pragma omp atomic
            n_errors += 1;
    }
    return n_errors;
}

static int doit(bool lazy) {
    using namespace mkl_dnn;

    /* AlexNet: n1
     * {16, 96, 55, 55} -> {16, 96, 55, 55}
     * alpha: 0.0001
     * beta : 0.75
     * size : 5
     */

    auto cpu_engine = engine(lazy ? engine::cpu_lazy : engine::cpu, 0);

    auto n1_src_desc     = memory::desc({1, 1, 2, {16, 96, 55, 55}}, memory::precision::f32, memory::format::nchw);
    auto n1_scratch_desc = memory::desc({1, 1, 2, {16, 96, 55, 55}}, memory::precision::f32, memory::format::nchw);
    auto n1_dst_desc     = memory::desc({1, 1, 2, {16, 96, 55, 55}}, memory::precision::f32, memory::format::nchw);

    real_t *src     = new real_t[16*96*55*55]();
    real_t *scratch = new real_t[16*96*55*55]();
    real_t *dst     = new real_t[16*96*55*55]();

    auto n1_src     = memory({n1_src_desc    , cpu_engine}, src    );
    auto n1_scratch = memory({n1_scratch_desc, cpu_engine}, scratch);
    auto n1_dst     = memory({n1_dst_desc    , cpu_engine}, dst    );

    auto n1 = lrn(prop_kind::forward, lrn::across_channels, n1_src, n1_scratch, n1_dst, 0.0001, 0.75, 5);

    init_src({16, 96, 55, 55}, src);
    stream().submit({n1}).wait();
    int n_errors = check_dst({ 16, 96, 55, 55 }, 0.0001, 0.75, 5, dst);

    delete[] src;
    delete[] scratch;
    delete[] dst;

    return n_errors;
}

#pragma GCC diagnostic pop
