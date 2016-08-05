#include <numeric>

#include "gtest/gtest.h"

#include "mkl_dnn.hpp"

namespace mkl_dnn {

template <typename reorder_types> struct reorder_typesraits { };
template <> struct reorder_typesraits<float> {
    static const memory::precision prec = memory::precision::f32;
};

template <typename T> inline void assert_eq(T a, T b);
template <> inline void assert_eq<float>(float a, float b) {
    ASSERT_FLOAT_EQ(a, b);
}

template <typename A, typename B> struct two_types {
    typedef A data_i_t;
    typedef B data_o_t;
};

inline size_t map_index(memory::desc md, size_t index) {
    return index;
}

template <typename data_i_t, typename data_o_t>
inline void check_reorder(memory::desc md_i, memory::desc md_o,
        const data_i_t *src,
        const data_o_t *dst)
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
struct test_params {
    engine::kind engine_kind;
    memory::format fmt_i, fmt_o;
    tensor::dims dims_i, dims_o;
};

template <typename reorder_types>
class reorder_test: public ::testing::TestWithParam<test_params<reorder_types>> {
protected:
    virtual void SetUp() {
        using data_i_t = typename reorder_types::data_i_t;
        using data_o_t = typename reorder_types::data_o_t;

        test_params<reorder_types> p
            = ::testing::TestWithParam<test_params<reorder_types>>::GetParam();

        ASSERT_TRUE(p.engine_kind == engine::kind::cpu || p.engine_kind ==
                engine::kind::cpu_lazy);
        auto eng = engine(p.engine_kind, 0);

        const size_t nelems_i = std::accumulate(p.dims_i.begin(), p.dims_i.end(),
                size_t(1), std::multiplies<size_t>());
        const size_t nelems_o = std::accumulate(p.dims_o.begin(), p.dims_o.end(),
                size_t(1), std::multiplies<size_t>());
        ASSERT_EQ(nelems_i, nelems_o);

        auto src_data = new data_i_t[nelems_i];
        auto dst_data = new data_o_t[nelems_o];

        memory::precision prec_i = reorder_typesraits<data_i_t>::prec;
        memory::precision prec_o = reorder_typesraits<data_o_t>::prec;
        auto md_i = memory::primitive_desc({{1, 1, 2, p.dims_i}, prec_i,
                memory::format::nchw}, eng);
        auto md_o = memory::primitive_desc({{1, 1, 2, p.dims_o}, prec_o,
                memory::format::nchw}, eng);

        /* initialize input data */
        for (size_t i = 0; i < nelems_i; ++i)
            src_data[map_index(md_i.desc(), i)] = data_i_t(i);

        auto src = memory(md_i, src_data);
        auto dst = memory(md_o, dst_data);
        auto r = reorder(src, dst);
        stream().submit({r}).wait();

        check_reorder(md_i.desc(), md_o.desc(), src_data, dst_data);

        delete[] src_data;
        delete[] dst_data;
    }
};

using f32_f32 = two_types<float, float>;
using reorder_test_f32_f32 = reorder_test<f32_f32>;
using test_params_f32_f32 = test_params<f32_f32>;

TEST_P(reorder_test_f32_f32, TestsReorder) { }
INSTANTIATE_TEST_CASE_P(TestReorder, reorder_test_f32_f32,
        ::testing::Values(
            test_params_f32_f32{engine::kind::cpu, memory::format::nchw,
            memory::format::nchw, {10, 10, 10, 10}, {10, 10, 10, 10}}
            )
        );

}
