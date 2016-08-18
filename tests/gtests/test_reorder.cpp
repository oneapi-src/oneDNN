/*******************************************************************************
* Copyright 2016 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <utility>
#include <numeric>

#include "gtest/gtest.h"
#include "mkldnn_test_common.hpp"

#include "mkldnn.hpp"

namespace mkldnn {

template <typename data_i_t, typename data_o_t>
inline void check_reorder(const memory::desc &md_i, const memory::desc &md_o,
        const data_i_t *src, const data_o_t *dst)
{
    const uint32_t ndims = md_i.data.tensor_desc.ndims;
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
        using data_i_t = typename reorder_types::first_type;
        using data_o_t = typename reorder_types::second_type;

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

using f32_f32 = std::pair<float, float>;
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
            cfg{eng::cpu, fmt::nhwc, fmt::nhwc, {10, 10, 10, 10}},
            cfg{eng::cpu, fmt::nchw, fmt::nChw8c, {2, 32, 4, 4}},
            cfg{eng::cpu, fmt::nChw8c, fmt::nchw, {2, 32, 4, 4}},
            cfg{eng::cpu, fmt::oihw, fmt::OIhw8i8o, {32, 32, 3, 3}},
            cfg{eng::cpu, fmt::OIhw8i8o, fmt::oihw, {32, 32, 3, 3}},
            cfg{eng::cpu, fmt::goihw, fmt::gOIhw8i8o, {2, 32, 32, 3, 3}},
            cfg{eng::cpu, fmt::gOIhw8i8o, fmt::goihw, {2, 32, 32, 3, 3}}
            )
        );

}
