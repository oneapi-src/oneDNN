/*******************************************************************************
* Copyright 2022-2023 Intel Corporation
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

#include "graph/unit/backend/dnnl/dnnl_test_common.hpp"
#include "graph/unit/unit_test_common.hpp"
#include "graph/unit/utils.hpp"

namespace graph = dnnl::impl::graph;
namespace dnnl_impl = graph::dnnl_impl;

TEST(OpExecutableDeathTest, DummyArgIndicesGetter) {
    graph::op_t op {0, graph::op_kind::Wildcard, "op"};
    dnnl_impl::fusion_info_mgr_t mgr;
#ifndef NDEBUG
    EXPECT_DEATH(dnnl_impl::dummy_arg_indices_getter(&op, mgr),
            "dummy getter should never be called");
#endif
}

TEST(OpExecutableDeathTest, DummyExecutableCreator) {
    dnnl::engine p_engine;
    dnnl_impl::fusion_info_mgr_t mgr;
    dnnl_impl::pd_cache_t pd_cache;
    auto op = std::make_shared<graph::op_t>(0, graph::op_kind::Wildcard, "op");
    EXPECT_DEBUG_DEATH(
            dnnl_impl::dummy_executable_creator(op, p_engine, mgr, pd_cache),
            "dummy executable creator should never be called");
}

#ifdef DNNL_WITH_SYCL
TEST(OpExecutable, DummyImpl) {
#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_SYCL
    graph::engine_kind_t kind = get_test_engine_kind();
    SKIP_IF(kind == graph::engine_kind::cpu,
            "skip sycl api test for native cpu runtime.");
#endif
    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();
    dnnl::engine p_engine = dnnl_impl::make_dnnl_engine(*engine);
    dnnl::stream p_stream = dnnl_impl::make_dnnl_stream(p_engine, *strm);
    auto op_exec = std::make_shared<dnnl_impl::dummy_impl_t>();

    // test empty input events
    auto returned_event0 = op_exec->execute_sycl(p_stream, {}, {});
    const auto &event_list0 = returned_event0.get_wait_list();
    ASSERT_EQ(event_list0.size(), 0U);
    ASSERT_EQ(
            returned_event0
                    .get_info<::sycl::info::event::command_execution_status>(),
            ::sycl::info::event_command_status::complete);

    // test one input event
    ::sycl::event input_event0;
    auto returned_event1 = op_exec->execute_sycl(p_stream, {}, {input_event0});
    ASSERT_EQ(returned_event1, input_event0);
    ASSERT_EQ(
            returned_event1
                    .get_info<::sycl::info::event::command_execution_status>(),
            ::sycl::info::event_command_status::complete);

    // test two input events
    ::sycl::event input_event1;
    auto returned_event2 = op_exec->execute_sycl(
            p_stream, {}, {std::move(input_event0), std::move(input_event1)});
    const auto &event_list2 = returned_event2.get_wait_list();
    ASSERT_GT(event_list2.size(), 0U);
    returned_event2.wait();
    ASSERT_EQ(
            returned_event2
                    .get_info<::sycl::info::event::command_execution_status>(),
            ::sycl::info::event_command_status::complete);
}
#endif
