/*******************************************************************************
* Copyright 2022 Intel Corporation
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
#include <memory>

#include "interface/c_types_map.hpp"

#include "backend/dnnl/common.hpp"
#include "backend/dnnl/dnnl_backend.hpp"
#include "backend/dnnl/op_executable.hpp"
#include "backend/dnnl/passes/lower.hpp"

#include "gtest/gtest.h"

#include "cpp/unit/backend/dnnl/dnnl_test_common.hpp"
#include "cpp/unit/unit_test_common.hpp"
#include "cpp/unit/utils.hpp"

namespace impl = dnnl::graph::impl;
namespace utils = dnnl::graph::tests::unit::utils;
namespace dnnl_impl = impl::dnnl_impl;

TEST(OpExecutable, DummyArgIndicesGetter) {
    impl::op_t op {0, impl::op_kind::Wildcard, "op"};
    dnnl_impl::fusion_info_mgr_t mgr;
#ifndef NDEBUG
    EXPECT_DEATH(dnnl_impl::dummy_arg_indices_getter(&op, mgr),
            "dummy getter shoule never be called");
#endif
}

TEST(OpExecutable, DummyExecutableCreator) {
    impl::engine_t &eng = get_engine();
    dnnl::engine p_engine = dnnl_impl::make_dnnl_engine(eng);
    dnnl_impl::fusion_info_mgr_t mgr;
    dnnl_impl::pd_cache_t pd_cache;

    auto op = std::make_shared<impl::op_t>(0, impl::op_kind::Wildcard, "op");
#ifndef NDEBUG
    EXPECT_DEATH(
            dnnl_impl::dummy_executable_creator(op, p_engine, mgr, pd_cache),
            "dummy executable creator shoule never be called");
#endif
}
