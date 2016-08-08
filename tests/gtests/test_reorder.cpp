#include <numeric>

#include "gtest/gtest.h"
#include "mkl_dnn_test_common.hpp"

#include "mkl_dnn.hpp"

namespace mkl_dnn {

template <typename A, typename B> struct two_types {
    typedef A data_i_t;
    typedef B data_o_t;
};

template <typename data_i_t, typename data_o_t>
inline void check_reorder(memory::desc md_i, memory::desc md_o,
        const data_i_t *src, const data_o_t *dst)
{
    const uint32_t ndims = md_i.data.tensor_desc.ndims_batch
        + md_i.data.tensor_desc.ndims_channels
        + md_i.data.tensor_desc.ndims_spatial;
    const uint32_t *dims = md_i.data.tensor_desc.dims;
    const size_t nelems = std::accumulate(
            dims, dims + ndims, size_t(1), std::multiplies<size_t>());

    for (size_t i = 0; i < nelems; ++i) {
        data_i_t s_raw = src[map_index(md_i, i)];
        data_o_t s = static_cast<data_o_t>(s_raw);
        data_o_t d = dst[map_index(md_o, i)];
        ASSERT_EQ(s, d) << "mismatch at position " << i;
    }
}

template <typename reorder_types>
struct test_simple_params {
    engine::kind engine_kind;
    memory::format fmt_i;
    memory::format fmt_o;
    tensor::dims dims;
};

template <typename reorder_types>
class reorder_simple_test:
    public ::testing::TestWithParam<test_simple_params<reorder_types>>
{
protected:
    virtual void SetUp() {
        using data_i_t = typename reorder_types::data_i_t;
        using data_o_t = typename reorder_types::data_o_t;

        test_simple_params<reorder_types> p
            = ::testing::TestWithParam<decltype(p)>::GetParam();

        ASSERT_TRUE(p.engine_kind == engine::kind::cpu || p.engine_kind ==
                engine::kind::cpu_lazy);
        auto eng = engine(p.engine_kind, 0);

        const size_t nelems_i = std::accumulate(p.dims.begin(), p.dims.end(),
                size_t(1), std::multiplies<size_t>());
        const size_t nelems_o = std::accumulate(p.dims.begin(), p.dims.end(),
                size_t(1), std::multiplies<size_t>());
        ASSERT_EQ(nelems_i, nelems_o);

        auto src_data = new data_i_t[nelems_i];
        auto dst_data = new data_o_t[nelems_o];

        memory::precision prec_i = data_traits<data_i_t>::prec;
        memory::precision prec_o = data_traits<data_o_t>::prec;
        auto mpd_i = memory::primitive_desc(create_md(p.dims, prec_i, p.fmt_i),
                eng);
        auto mpd_o = memory::primitive_desc(create_md(p.dims, prec_o, p.fmt_o),
                eng);

        /* initialize input data */
        for (size_t i = 0; i < nelems_i; ++i)
            src_data[map_index(mpd_i.desc(), i)] = data_i_t(i);

        auto src = memory(mpd_i, src_data);
        auto dst = memory(mpd_o, dst_data);
        auto r = reorder(src, dst);
        stream().submit({r}).wait();

        check_reorder(mpd_i.desc(), mpd_o.desc(), src_data, dst_data);

        delete[] src_data;
        delete[] dst_data;
    }
};

using f32_f32 = two_types<float, float>;
using reorder_simple_test_f32_f32 = reorder_simple_test<f32_f32>;
using test_simple_params_f32_f32 = test_simple_params<f32_f32>;

using eng = engine::kind;
using fmt = memory::format;
using cfg = test_simple_params_f32_f32;

TEST_P(reorder_simple_test_f32_f32, TestsReorder) { }
INSTANTIATE_TEST_CASE_P(TestReorder, reorder_simple_test_f32_f32,
        ::testing::Values(
            cfg{eng::cpu, fmt::nchw, fmt::nchw, {10, 10, 10, 10}},
            cfg{eng::cpu, fmt::nchw, fmt::nhwc, {10, 10, 10, 10}},
            cfg{eng::cpu, fmt::nhwc, fmt::nchw, {10, 10, 10, 10}},
            cfg{eng::cpu, fmt::nhwc, fmt::nhwc, {10, 10, 10, 10}}
            )
        );

}
