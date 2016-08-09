#include "mkl_dnn_test_common.hpp"
#include "gtest/gtest.h"

#include "mkl_dnn.hpp"

namespace mkl_dnn {

struct test_inner_product_descr_t {
    uint32_t mb;
    uint32_t ic;
    uint32_t oc;
    uint32_t kh, kw;
};

template <typename data_t>
void compute_ref_inner_product_fwd_nchw(
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

struct inprod_test_params {
    prop_kind aprop_kind;
    const engine::kind engine_kind;
    memory::format src_format;
    memory::format weights_format;
    memory::format dst_format;
    test_inner_product_descr_t test_ipd;
};

template <typename data_t>
class inner_product_test : public ::testing::TestWithParam<inprod_test_params> {
protected:
    virtual void SetUp()
    {
        inprod_test_params p
                = ::testing::TestWithParam<inprod_test_params>::GetParam();
        test_inner_product_descr_t ipd = p.test_ipd;
        bool has_spatial = ipd.kh > 1 && ipd.kw > 1;
        ASSERT_TRUE(p.src_format == memory::format::nchw
                || (p.src_format == memory::format::nc && !has_spatial));
        ASSERT_TRUE(p.weights_format == memory::format::oihw
                || (p.weights_format == memory::format::oi && !has_spatial));
        ASSERT_EQ(p.dst_format, memory::format::nc);
        ASSERT_TRUE(p.engine_kind == engine::kind::cpu
                || p.engine_kind == engine::kind::cpu_lazy);
        ASSERT_EQ(p.aprop_kind, prop_kind::forward);
        auto eng = engine(p.engine_kind, 0);
        memory::precision prec = data_traits<data_t>::prec;
        ASSERT_EQ(prec, mkl_dnn::memory::precision::f32);

        size_t src_size = has_spatial ? ipd.mb * ipd.ic * ipd.kh * ipd.kw :
                                        ipd.mb * ipd.ic;
        data_t *src_data = new data_t[src_size];
        fill_data(src_size, src_data);
        size_t weights_size = has_spatial ? ipd.oc * ipd.ic * ipd.kh * ipd.kw :
                                            ipd.oc * ipd.ic;
        data_t *weights_data = new data_t[weights_size];
        fill_data(weights_size, weights_data);
        size_t dst_size = ipd.mb * ipd.oc;
        data_t *dst_data = new data_t[dst_size];
        // fillData(dst_size, output_data);
        data_t *dst_ref_data = new data_t[dst_size];
        // fillData(dst_size, output_ref_data);

        auto c_src_desc = has_spatial ?
                create_md({ ipd.mb, ipd.ic, ipd.kh, ipd.kw }, prec,
                        p.src_format) :
                create_md({ ipd.mb, ipd.ic }, prec, p.src_format);
        auto c_weights_desc = has_spatial ?
                create_md({ ipd.oc, ipd.ic, ipd.kh, ipd.kw }, prec,
                        p.weights_format) :
                create_md({ ipd.oc, ipd.ic }, prec, p.weights_format);
        auto c_dst_desc = create_md({ ipd.mb, ipd.oc }, prec, p.dst_format);

        auto c_src = memory(
                memory::primitive_desc(c_src_desc, eng), (void *)src_data);
        auto c_weights = memory(memory::primitive_desc(c_weights_desc, eng),
                (void *)weights_data);
        auto c_dst = memory(
                memory::primitive_desc(c_dst_desc, eng), (void *)dst_data);

        auto ip = inner_product(p.aprop_kind, c_src, c_weights, c_dst);

        stream().submit({ ip }).wait();

        compute_ref_inner_product_fwd_nchw(
                ipd, src_data, weights_data, dst_ref_data);
        compare_data(dst_ref_data, dst_data, dst_size);
    }
};

using inner_product_test_float = inner_product_test<float>;
using inprod_test_params_float = inprod_test_params;

TEST_P(inner_product_test_float, TestsInnerProduct)
{
}
INSTANTIATE_TEST_CASE_P(
        TestInnerProductForward, inner_product_test_float,
        ::testing::Values(
                inprod_test_params_float{ prop_kind::forward, engine::kind::cpu,
                        memory::format::nchw, memory::format::oihw,
                        memory::format::nc, { 2, 32, 48, 6, 6 } },
                inprod_test_params_float{ prop_kind::forward, engine::kind::cpu,
                        memory::format::nc, memory::format::oi,
                        memory::format::nc, { 2, 2, 4, 1, 1 } }));
}
