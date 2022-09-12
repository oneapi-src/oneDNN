/*******************************************************************************
* Copyright 2022 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain src copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "dnnl_test_common.hpp"
#include "src/common/broadcast_strategy.cpp"
#include "gtest/gtest.h"

#include "oneapi/dnnl/dnnl.hpp"

#define CASE(ndims, tag) \
    case ndims: return memory::format_tag::tag;

namespace dnnl {

memory::format_tag plain_format_tag(size_t ndims) {
    assert(ndims <= 12);
    switch (ndims) {
        CASE(1, a)
        CASE(2, ab)
        CASE(3, abc)
        CASE(4, abcd)
        CASE(5, abcde)
        CASE(6, abcdef)
        CASE(7, abcdefg)
        CASE(8, abcdefgh)
        CASE(9, abcdefghi)
        CASE(10, abcdefghij)
        CASE(11, abcdefghijk)
        CASE(12, abcdefghijkl)
        default: return memory::format_tag::any;
    }
}

#undef CASE

struct bcast_strategy_test_t
    : public ::testing::TestWithParam<std::tuple<memory::dims, memory::dims,
              impl::broadcasting_strategy_t>> {};

HANDLE_EXCEPTIONS_FOR_TEST_P(bcast_strategy_test_t, TestBroadcastStrategy) {
    const auto &dst_dims = std::get<0>(GetParam());
    const auto &rhs_arg_dims = std::get<1>(GetParam());
    ASSERT_EQ(dst_dims.size(), rhs_arg_dims.size());

    const size_t ndims = dst_dims.size();
    constexpr auto defualt_dt = memory::data_type::f32;
    const auto default_format = plain_format_tag(ndims);
    auto rhs_md = memory::desc(rhs_arg_dims, defualt_dt, default_format, true);
    auto dst_md = memory::desc(dst_dims, defualt_dt, default_format, true);
    auto dst_mdw = impl::memory_desc_wrapper(dst_md.get());
    const auto bcast_type
            = impl::get_rhs_arg_broadcasting_strategy(*rhs_md.get(), dst_mdw);
    const auto expected_bcast_type = std::get<2>(GetParam());
    ASSERT_EQ(bcast_type, expected_bcast_type);
}

INSTANTIATE_TEST_SUITE_P(SupportedStrategies, bcast_strategy_test_t,
        ::testing::Values(
                // 5d cases
                std::make_tuple(memory::dims {2, 2, 2, 2, 2},
                        memory::dims {1, 1, 1, 1, 1},
                        impl::broadcasting_strategy_t::scalar),
                std::make_tuple(memory::dims {2, 2, 2, 2, 2},
                        memory::dims {1, 2, 1, 1, 1},
                        impl::broadcasting_strategy_t::per_oc_spatial),
                std::make_tuple(memory::dims {1, 2, 2, 2, 2},
                        memory::dims {1, 2, 1, 1, 1},
                        impl::broadcasting_strategy_t::per_oc_spatial),
                std::make_tuple(memory::dims {1, 2, 1, 2, 2},
                        memory::dims {1, 2, 1, 1, 1},
                        impl::broadcasting_strategy_t::per_oc_spatial),
                std::make_tuple(memory::dims {1, 2, 1, 1, 2},
                        memory::dims {1, 2, 1, 1, 1},
                        impl::broadcasting_strategy_t::per_oc_spatial),
                std::make_tuple(memory::dims {2, 2, 1, 1, 2},
                        memory::dims {1, 2, 1, 1, 1},
                        impl::broadcasting_strategy_t::per_oc_spatial),
                std::make_tuple(memory::dims {2, 2, 1, 2, 2},
                        memory::dims {1, 2, 1, 1, 1},
                        impl::broadcasting_strategy_t::per_oc_spatial),
                std::make_tuple(memory::dims {1, 2, 1, 1, 1},
                        memory::dims {1, 2, 1, 1, 1},
                        impl::broadcasting_strategy_t::no_broadcast),
                std::make_tuple(memory::dims {2, 2, 2, 2, 2},
                        memory::dims {2, 2, 2, 2, 2},
                        impl::broadcasting_strategy_t::no_broadcast),
                std::make_tuple(memory::dims {2, 2, 2, 2, 2},
                        memory::dims {1, 1, 1, 1, 2},
                        impl::broadcasting_strategy_t::per_w),
                std::make_tuple(memory::dims {2, 2, 2, 1, 2},
                        memory::dims {1, 1, 1, 1, 2},
                        impl::broadcasting_strategy_t::per_w),
                std::make_tuple(memory::dims {2, 2, 1, 1, 2},
                        memory::dims {1, 1, 1, 1, 2},
                        impl::broadcasting_strategy_t::per_w),
                std::make_tuple(memory::dims {2, 1, 1, 1, 2},
                        memory::dims {1, 1, 1, 1, 2},
                        impl::broadcasting_strategy_t::per_w),
                std::make_tuple(memory::dims {1, 2, 1, 1, 2},
                        memory::dims {1, 1, 1, 1, 2},
                        impl::broadcasting_strategy_t::per_w),
                std::make_tuple(memory::dims {1, 1, 2, 1, 2},
                        memory::dims {1, 1, 1, 1, 2},
                        impl::broadcasting_strategy_t::per_w),
                std::make_tuple(memory::dims {1, 1, 1, 2, 2},
                        memory::dims {1, 1, 1, 1, 2},
                        impl::broadcasting_strategy_t::per_w),
                std::make_tuple(memory::dims {1, 1, 1, 1, 2},
                        memory::dims {1, 1, 1, 1, 2},
                        impl::broadcasting_strategy_t::no_broadcast),
                std::make_tuple(memory::dims {2, 2, 2, 2, 2},
                        memory::dims {2, 1, 1, 1, 2},
                        impl::broadcasting_strategy_t::per_mb_w),
                std::make_tuple(memory::dims {2, 2, 2, 1, 2},
                        memory::dims {2, 1, 1, 1, 2},
                        impl::broadcasting_strategy_t::per_mb_w),
                std::make_tuple(memory::dims {2, 2, 1, 1, 2},
                        memory::dims {2, 1, 1, 1, 2},
                        impl::broadcasting_strategy_t::per_mb_w),
                std::make_tuple(memory::dims {2, 1, 1, 1, 2},
                        memory::dims {2, 1, 1, 1, 2},
                        impl::broadcasting_strategy_t::no_broadcast),
                std::make_tuple(memory::dims {2, 1, 2, 2, 2},
                        memory::dims {2, 1, 2, 2, 2},
                        impl::broadcasting_strategy_t::no_broadcast),
                std::make_tuple(memory::dims {2, 2, 2, 2, 2},
                        memory::dims {2, 1, 2, 2, 2},
                        impl::broadcasting_strategy_t::per_mb_spatial),
                std::make_tuple(memory::dims {1, 2, 2, 2, 2},
                        memory::dims {1, 1, 2, 2, 2},
                        impl::broadcasting_strategy_t::per_mb_spatial),
                std::make_tuple(memory::dims {2, 2, 1, 2, 2},
                        memory::dims {2, 1, 1, 2, 2},
                        impl::broadcasting_strategy_t::per_mb_spatial),
                std::make_tuple(memory::dims {2, 2, 2, 1, 2},
                        memory::dims {2, 1, 2, 1, 2},
                        impl::broadcasting_strategy_t::per_mb_spatial),
                std::make_tuple(memory::dims {2, 2, 2, 2, 1},
                        memory::dims {2, 1, 2, 2, 1},
                        impl::broadcasting_strategy_t::per_mb_spatial),
                std::make_tuple(memory::dims {2, 2, 1, 2, 1},
                        memory::dims {2, 1, 1, 2, 1},
                        impl::broadcasting_strategy_t::per_mb_spatial),
                std::make_tuple(memory::dims {1, 2, 1, 2, 1},
                        memory::dims {1, 1, 1, 2, 1},
                        impl::broadcasting_strategy_t::per_mb_spatial),
                // 4d cases
                std::make_tuple(memory::dims {2, 2, 2, 2},
                        memory::dims {1, 1, 1, 1},
                        impl::broadcasting_strategy_t::scalar),
                std::make_tuple(memory::dims {2, 2, 2, 2},
                        memory::dims {1, 2, 1, 1},
                        impl::broadcasting_strategy_t::per_oc_spatial),
                std::make_tuple(memory::dims {1, 2, 2, 2},
                        memory::dims {1, 2, 1, 1},
                        impl::broadcasting_strategy_t::per_oc_spatial),
                std::make_tuple(memory::dims {1, 2, 1, 2},
                        memory::dims {1, 2, 1, 1},
                        impl::broadcasting_strategy_t::per_oc_spatial),
                std::make_tuple(memory::dims {2, 2, 1, 2},
                        memory::dims {1, 2, 1, 1},
                        impl::broadcasting_strategy_t::per_oc_spatial),
                std::make_tuple(memory::dims {2, 2, 2, 1},
                        memory::dims {1, 2, 1, 1},
                        impl::broadcasting_strategy_t::per_oc_spatial),
                std::make_tuple(memory::dims {1, 2, 1, 1},
                        memory::dims {1, 2, 1, 1},
                        impl::broadcasting_strategy_t::no_broadcast),
                std::make_tuple(memory::dims {2, 2, 2, 2},
                        memory::dims {2, 2, 2, 2},
                        impl::broadcasting_strategy_t::no_broadcast),
                std::make_tuple(memory::dims {2, 2, 2, 2},
                        memory::dims {1, 1, 1, 2},
                        impl::broadcasting_strategy_t::per_w),
                std::make_tuple(memory::dims {2, 2, 1, 2},
                        memory::dims {1, 1, 1, 2},
                        impl::broadcasting_strategy_t::per_w),
                std::make_tuple(memory::dims {2, 1, 1, 2},
                        memory::dims {1, 1, 1, 2},
                        impl::broadcasting_strategy_t::per_w),
                std::make_tuple(memory::dims {1, 2, 1, 2},
                        memory::dims {1, 1, 1, 2},
                        impl::broadcasting_strategy_t::per_w),
                std::make_tuple(memory::dims {1, 1, 2, 2},
                        memory::dims {1, 1, 1, 2},
                        impl::broadcasting_strategy_t::per_w),
                std::make_tuple(memory::dims {1, 1, 1, 2},
                        memory::dims {1, 1, 1, 2},
                        impl::broadcasting_strategy_t::no_broadcast),
                std::make_tuple(memory::dims {2, 2, 2, 2},
                        memory::dims {2, 1, 1, 2},
                        impl::broadcasting_strategy_t::per_mb_w),
                std::make_tuple(memory::dims {2, 2, 1, 2},
                        memory::dims {2, 1, 1, 2},
                        impl::broadcasting_strategy_t::per_mb_w),
                std::make_tuple(memory::dims {2, 1, 1, 2},
                        memory::dims {2, 1, 1, 2},
                        impl::broadcasting_strategy_t::no_broadcast),
                std::make_tuple(memory::dims {2, 1, 2, 2},
                        memory::dims {2, 1, 2, 2},
                        impl::broadcasting_strategy_t::no_broadcast),
                std::make_tuple(memory::dims {2, 2, 2, 2},
                        memory::dims {2, 1, 2, 2},
                        impl::broadcasting_strategy_t::per_mb_spatial),
                std::make_tuple(memory::dims {1, 2, 2, 2},
                        memory::dims {1, 1, 2, 2},
                        impl::broadcasting_strategy_t::per_mb_spatial),
                std::make_tuple(memory::dims {1, 2, 2, 1},
                        memory::dims {1, 1, 2, 1},
                        impl::broadcasting_strategy_t::per_mb_spatial),
                std::make_tuple(memory::dims {2, 2, 2, 1},
                        memory::dims {2, 1, 2, 1},
                        impl::broadcasting_strategy_t::per_mb_spatial),
                // 3d cases
                std::make_tuple(memory::dims {2, 2, 2}, memory::dims {1, 1, 1},
                        impl::broadcasting_strategy_t::scalar),
                std::make_tuple(memory::dims {2, 2, 2}, memory::dims {1, 2, 1},
                        impl::broadcasting_strategy_t::per_oc_spatial),
                std::make_tuple(memory::dims {2, 2, 1}, memory::dims {1, 2, 1},
                        impl::broadcasting_strategy_t::per_oc_spatial),
                std::make_tuple(memory::dims {1, 2, 2}, memory::dims {1, 2, 1},
                        impl::broadcasting_strategy_t::per_oc_spatial),
                std::make_tuple(memory::dims {1, 2, 1}, memory::dims {1, 2, 1},
                        impl::broadcasting_strategy_t::no_broadcast),
                std::make_tuple(memory::dims {2, 2, 2}, memory::dims {2, 2, 2},
                        impl::broadcasting_strategy_t::no_broadcast),
                std::make_tuple(memory::dims {2, 2, 2}, memory::dims {1, 1, 2},
                        impl::broadcasting_strategy_t::per_w),
                std::make_tuple(memory::dims {2, 1, 2}, memory::dims {1, 1, 2},
                        impl::broadcasting_strategy_t::per_w),
                std::make_tuple(memory::dims {1, 2, 2}, memory::dims {1, 1, 2},
                        impl::broadcasting_strategy_t::per_w),
                std::make_tuple(memory::dims {1, 1, 2}, memory::dims {1, 1, 2},
                        impl::broadcasting_strategy_t::no_broadcast),
                std::make_tuple(memory::dims {2, 2, 2}, memory::dims {2, 1, 2},
                        impl::broadcasting_strategy_t::per_mb_w),
                std::make_tuple(memory::dims {2, 1, 2}, memory::dims {2, 1, 2},
                        impl::broadcasting_strategy_t::no_broadcast),
                std::make_tuple(memory::dims {2, 2, 1}, memory::dims {2, 1, 1},
                        impl::broadcasting_strategy_t::per_mb_spatial),
                // 2d cases
                std::make_tuple(memory::dims {2, 2}, memory::dims {1, 1},
                        impl::broadcasting_strategy_t::scalar),
                std::make_tuple(memory::dims {2, 2}, memory::dims {1, 2},
                        impl::broadcasting_strategy_t::per_oc),
                std::make_tuple(memory::dims {1, 2}, memory::dims {1, 2},
                        impl::broadcasting_strategy_t::no_broadcast),
                std::make_tuple(memory::dims {2, 2}, memory::dims {2, 1},
                        impl::broadcasting_strategy_t::per_mb_spatial),
                // 1d cases
                std::make_tuple(memory::dims {2}, memory::dims {1},
                        impl::broadcasting_strategy_t::scalar),
                std::make_tuple(memory::dims {2}, memory::dims {2},
                        impl::broadcasting_strategy_t::no_broadcast)));
} // namespace dnnl
