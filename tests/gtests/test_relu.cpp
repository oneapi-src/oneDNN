#include "gtest/gtest.h"
#include "mkl_dnn_test_common.hpp"

#include "mkl_dnn.hpp"

namespace mkl_dnn {

template <typename T> void assert_eq(T a, T b);
template <> void assert_eq<float>(float a, float b) { ASSERT_FLOAT_EQ(a, b); }

template <typename data_t>
void check_relu(prop_kind aprop_kind,
        data_t negative_slope, memory::desc md,
        const data_t *src, const data_t *dst)
{
    ASSERT_EQ(aprop_kind, prop_kind::forward);

    ASSERT_EQ(md.data.tensor_desc.ndims_batch, 1U);
    ASSERT_EQ(md.data.tensor_desc.ndims_channels, 1U);
    ASSERT_EQ(md.data.tensor_desc.ndims_spatial, 2U);
    ASSERT_EQ(md.data.format, memory::convert_to_c(memory::format::nchw));
    ASSERT_EQ(md.data.precision, memory::convert_to_c(memory::precision::f32)); // TODO: type assert

    size_t N = md.data.tensor_desc.dims[0];
    size_t C = md.data.tensor_desc.dims[1];
    size_t H = md.data.tensor_desc.dims[2];
    size_t W = md.data.tensor_desc.dims[3];
    for (size_t i = 0; i < N * C *H * W; ++i) {
        data_t s = src[i];
        assert_eq(dst[i], s >= 0 ? s : s * negative_slope);
    }
}

template <typename data_t>
struct relu_test_params {
    data_t negative_slope;
    prop_kind aprop_kind;
    engine::kind engine_kind;
    memory::format memory_format;
    tensor::dims dims;
};

template <typename data_t>
class relu_test : public ::testing::TestWithParam<relu_test_params<data_t>> {
protected:
    virtual void SetUp() {
        relu_test_params<data_t> p
            = ::testing::TestWithParam<relu_test_params<data_t>>::GetParam();

        ASSERT_EQ(p.memory_format, memory::format::nchw);
        ASSERT_TRUE(p.engine_kind == engine::kind::cpu || p.engine_kind ==
                engine::kind::cpu_lazy);
        auto eng = engine(p.engine_kind, 0);

        ASSERT_EQ(p.dims.size(), 4U);
        size_t size = p.dims[0] * p.dims[1] * p.dims[2] * p.dims[3];
        auto src_nchw_data = new data_t[size];
        // TODO: random fill src_nchw_data
        auto dst_nchw_data = new data_t[size];

        memory::precision prec = data_traits<data_t>::prec;
        auto td = tensor::desc(1, 1, 2, p.dims);

        auto nchw_mem_desc = memory::desc(td, prec, memory::format::nchw);
        auto nchw_mem_prim_desc = memory::primitive_desc(nchw_mem_desc, eng);
        auto src_nchw = memory(nchw_mem_prim_desc, src_nchw_data);
        auto dst_nchw = memory(nchw_mem_prim_desc, dst_nchw_data);

        auto test_desc = memory::desc(td, prec, p.memory_format);
        auto relu_prim_desc = relu::primitive_desc(
                {p.aprop_kind, p.negative_slope, test_desc, test_desc}, eng);

        // Need better accessors to descriptors from C++ ?
        // XXX: hacks to avoid reorders
        auto src = src_nchw;
        auto src_format = static_cast<memory::format>(
                relu_prim_desc.data.src_primitive_desc.memory_desc.format);
        if (src_format != memory::format::nchw)
            src = memory({{td, prec, src_format}, eng});
        fill_data<data_t>(src.get_primitive_desc().get_number_of_elements(),
                (data_t *)src.get_data_handle());

        auto dst_format = static_cast<memory::format>(
                relu_prim_desc.data.dst_primitive_desc.memory_desc.format);
        auto dst = dst_nchw;
        if (dst_format != memory::format::nchw)
            dst = memory({{td, prec, dst_format}, eng});

        std::vector<primitive> pipeline;
//        if (src != src_nchw)
//            pipeline.push_back(reorder(src_nchw, src));
        pipeline.push_back(relu(relu_prim_desc, src, dst));
//        if (dst != dst_nchw)
//            pipeline.push_back(reorder(dst, dst_nchw));
        stream().submit(pipeline).wait();

        check_relu(p.aprop_kind, p.negative_slope,
                nchw_mem_desc, src_nchw_data, dst_nchw_data);
    }
};

using relu_test_float = relu_test<float>;
using relu_test_params_float = relu_test_params<float>;

TEST_P(relu_test_float, TestsReLU) { }
INSTANTIATE_TEST_CASE_P(TestReLUForward, relu_test_float,
        ::testing::Values(
            relu_test_params_float{0, prop_kind::forward, engine::kind::cpu,
            memory::format::nchw, {10, 10, 10, 10}}));

}
