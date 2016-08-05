#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "mkl_dnn.hpp"
#include "gtest/gtest.h"

struct test_inner_product_descr_t {
    uint32_t mb;
    uint32_t ic;
    uint32_t oc;
    uint32_t kh, kw;
};

static uint32_t doFill(size_t index, double sparsity)
{
    const size_t group_size = (size_t)(1. / sparsity);
    const size_t group = index / group_size;
    const size_t in_group = index % group_size;
    return in_group == ((group % 1637) % group_size);
}

template <typename data_t>
static void fillData(const uint32_t size, data_t *data, double sparsity = 1.)
{
#pragma omp parallel for
    for (uint32_t n = 0; n < size; n++) {
        data[n] = doFill(n, sparsity) ? 1. + 2e-1 * sin(n % 37) : 0;
    }
}

template <typename data_t>
static void computeRefInnerProductFwd(
        test_inner_product_descr_t ipd, data_t *in, data_t *filt, data_t *out)
{
#pragma omp parallel for collapse(2)
    for (uint32_t n = 0; n < ipd.mb; n++) {
        for (uint32_t oc = 0; oc < ipd.oc; oc++) {
            uint32_t oidx = n * ipd.oc + oc;
            out[oidx] = 0.0;
            for (uint32_t ic = 0; ic < ipd.ic; ic++) {
                for (uint32_t kh = 0; kh < ipd.kh; kh++) {
                    for (uint32_t kw = 0; kw < ipd.kw; kw++) {
                        uint32_t iidx = n * ipd.ic * ipd.kh * ipd.kw
                                + ic * ipd.kh * ipd.kw + kh * ipd.kw + kw;
                        uint32_t fidx = oc * ipd.ic * ipd.kh * ipd.kw
                                + ic * ipd.kh * ipd.kw + kh * ipd.kw + kw;
                        out[oidx] += in[iidx] * filt[fidx];
                    }
                }
            }
        }
    }
}

template <typename data_t>
static int doit(test_inner_product_descr_t ipd, bool lazy)
{
    using namespace mkl_dnn;

    auto cpu_engine = engine(lazy ? engine::cpu_lazy : engine::cpu, 0);
    EXPECT_EQ(sizeof(data_t), 4);
    memory::precision testPrecision = memory::precision::f32;

    data_t *src_data = new data_t[ipd.mb * ipd.ic * ipd.kh * ipd.kw];
    fillData(ipd.mb * ipd.ic * ipd.kh * ipd.kw, src_data);
    data_t *weights_data = new data_t[ipd.oc * ipd.ic * ipd.kh * ipd.kw];
    fillData(ipd.oc * ipd.ic * ipd.kh * ipd.kw, weights_data);
    data_t *dst_data = new data_t[ipd.mb * ipd.oc];
    data_t *dst_ref_data = new data_t[ipd.mb * ipd.oc];

    auto c_src_desc = ipd.kh > 1 || ipd.kw > 1 ?
            memory::desc({ 1, 1, 2, { ipd.mb, ipd.ic, ipd.kh, ipd.kw } },
                    testPrecision, memory::format::nchw) :
            memory::desc({ 1, 1, 0, { ipd.mb, ipd.ic } }, testPrecision,
                    memory::format::nc);

    auto c_weights_desc = ipd.kh > 1 || ipd.kw > 1 ?
            memory::desc({ 0, 2, 2, { ipd.oc, ipd.ic, ipd.kh, ipd.kw } },
                    testPrecision, memory::format::oihw) :
            memory::desc({ 0, 2, 0, { ipd.oc, ipd.ic } }, testPrecision,
                    memory::format::oi);

    auto c_dst_desc = memory::desc(
            { 1, 1, 0, { ipd.mb, ipd.oc } }, testPrecision, memory::format::nc);

    auto c_src = memory({ c_src_desc, cpu_engine }, src_data);
    auto c_weights = memory({ c_weights_desc, cpu_engine }, weights_data);
    auto c_dst = memory({ c_dst_desc, cpu_engine }, dst_data);

    auto ip = inner_product(prop_kind::forward, c_src, c_weights, c_dst);

    stream().submit({ ip }).wait();

    computeRefInnerProductFwd(ipd, src_data, weights_data, dst_ref_data);

#pragma omp parallel for
    for (uint32_t i = 0; i < ipd.mb * ipd.oc; ++i) {
        EXPECT_NEAR(dst_data[i], dst_ref_data[i], 1e-4);
    }

    delete[] src_data;
    delete[] dst_data;
    delete[] weights_data;
    delete[] dst_ref_data;
    return 0;
}
typedef ::testing::Types<float> testDataTypes;

template <typename data_t>
class innerProductTest : public ::testing::Test {
public:
protected:
    innerProductTest() {}
    virtual ~innerProductTest() {}
};

TYPED_TEST_CASE(innerProductTest, testDataTypes);

TYPED_TEST(innerProductTest, simpleTest)
{
    doit<TypeParam>({ 2, 32, 48, 6, 6 }, 1);
    doit<TypeParam>({ 2, 2, 4, 1, 1 }, 1);
}
