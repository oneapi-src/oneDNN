/*******************************************************************************
 * Copyright 2020-2023 Intel Corporation
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

#include <compiler/ir/transform/static_memory_planner.hpp>

#include <iostream>
#include "gtest/gtest.h"

using namespace dnnl::impl::graph::gc;
TEST(GCCore_CPU_static_memory_planner, TestStaticMemoryPlanning) {
    /*
    {0}                   {160}               {280}                 {450}
    |           0         |           1        |          2          |
    |    3    |   4/5     |                                          |
    |    7    |                                     6                |
              {100}
    */
    std::vector<memory_optim::memory_alloc_trace_t> traces
            = {{0, 100}, {1, 120}, {2, 100}, {0, 0}, {3, 50}, {4, 60}, {2, 0},
                    {4, 0}, {8, 100}, {8, 0}, {5, 60}, {5, 0}, {1, 0}, {6, 350},
                    {3, 0}, {7, 100}};
    std::unordered_map<uintptr_t, size_t> out;
    std::unordered_map<uintptr_t, std::vector<uintptr_t>> inplace_selection;
    size_t total = memory_optim::schedule_memory_allocations(
            traces, 1, false, {}, out, inplace_selection);
    std::unordered_map<uintptr_t, size_t> expected_out = {{0, 0}, {1, 160},
            {2, 280}, {3, 0}, {4, 100}, {5, 100}, {6, 100}, {7, 0}, {8, 280}};
    EXPECT_EQ(total, 450UL);
    EXPECT_EQ(out, expected_out);

    total = memory_optim::schedule_memory_allocations(
            traces, 1, true, {}, out, inplace_selection);
    expected_out = {{0, 0}, {1, 160}, {2, 280}, {3, 0}, {4, 100}, {5, 100},
            {6, 100}, {7, 0}, {8, 280}};
    EXPECT_EQ(total, 450UL);
    EXPECT_EQ(out, expected_out);
}

TEST(GCCore_CPU_static_memory_planner, TestStaticMemoryPlanningInplace) {
    using inplace_outdata
            = std::unordered_map<uintptr_t, std::vector<uintptr_t>>;
    using inplace_data = std::unordered_map<uintptr_t,
            std::vector<memory_optim::inplace_info>>;

    // simple inplace (need merge + split)
    {
        std::vector<memory_optim::memory_alloc_trace_t> traces = {{1, 120},
                {2, 100}, {3, 200}, {1, 0}, {2, 0}, {3, 0}, {4, 220}, {4, 0}};
        std::unordered_map<uintptr_t, size_t> out;
        inplace_outdata inplace_selection;
        inplace_data inplace_hint
                = {{3, {{1, inplace_kind::FREE}, {2, inplace_kind::FREE}}}};
        size_t total = memory_optim::schedule_memory_allocations(
                traces, 1, false, inplace_hint, out, inplace_selection);
        std::unordered_map<uintptr_t, size_t> expected_out
                = {{1, 0}, {2, 120}, {3, 0}, {4, 0}};
        EXPECT_EQ(total, 220UL);
        EXPECT_EQ(out, expected_out);

        inplace_outdata expected_inplace = {{3, {1, 2}}};
        EXPECT_EQ(inplace_selection, expected_inplace);
    }

    // inplace extend
    {
        std::vector<memory_optim::memory_alloc_trace_t> traces = {{1, 120},
                {2, 100}, {4, 250}, {3, 250}, {1, 0}, {2, 0}, {3, 0}, {4, 0}};
        std::unordered_map<uintptr_t, size_t> out;
        inplace_outdata inplace_selection;
        inplace_data inplace_hint
                = {{3, {{1, inplace_kind::FREE}, {2, inplace_kind::FREE}}}};
        size_t total = memory_optim::schedule_memory_allocations(
                traces, 1, false, inplace_hint, out, inplace_selection);
        std::unordered_map<uintptr_t, size_t> expected_out
                = {{1, 0}, {2, 120}, {3, 0}, {4, 250}};
        EXPECT_EQ(total, 500UL);
        EXPECT_EQ(out, expected_out);

        inplace_outdata expected_inplace = {{3, {1, 2}}};
        EXPECT_EQ(inplace_selection, expected_inplace);
    }

    // inplace 2 buffers into one
    {
        std::vector<memory_optim::memory_alloc_trace_t> traces
                = {{1, 120}, {2, 100}, {3, 150}, {4, 50}, {5, 10}, {1, 0},
                        {2, 0}, {3, 0}, {4, 0}, {5, 0}};
        std::unordered_map<uintptr_t, size_t> out;
        inplace_outdata inplace_selection;
        inplace_data inplace_hint = {
                {3, {{1, inplace_kind::FREE}, {2, inplace_kind::FREE}}},
                {4, {{1, inplace_kind::FREE}, {2, inplace_kind::FREE}}}};
        size_t total = memory_optim::schedule_memory_allocations(
                traces, 1, false, inplace_hint, out, inplace_selection);
        std::unordered_map<uintptr_t, size_t> expected_out
                = {{1, 0}, {2, 120}, {3, 0}, {4, 150}, {5, 220}};
        EXPECT_EQ(total, 230UL);
        EXPECT_EQ(out, expected_out);

        inplace_outdata expected_inplace = {{3, {1, 2}}, {4, {2}}};
        EXPECT_EQ(inplace_selection, expected_inplace);
    }

    // inplace 2 buffers into one, but require zero offset
    {
        std::vector<memory_optim::memory_alloc_trace_t> traces
                = {{1, 120}, {2, 100}, {3, 150}, {4, 50}, {5, 10}, {1, 0},
                        {2, 0}, {3, 0}, {4, 0}, {5, 0}};
        std::unordered_map<uintptr_t, size_t> out;
        inplace_outdata inplace_selection;
        inplace_data inplace_hint = {
                {3, {{1, inplace_kind::FREE}, {2, inplace_kind::ZERO_OFFSET}}},
                {4, {{1, inplace_kind::FREE}, {2, inplace_kind::FREE}}}};
        size_t total = memory_optim::schedule_memory_allocations(
                traces, 1, false, inplace_hint, out, inplace_selection);
        std::unordered_map<uintptr_t, size_t> expected_out
                = {{1, 0}, {2, 150}, {3, 0}, {4, 150}, {5, 250}};
        EXPECT_EQ(total, 260UL);
        EXPECT_EQ(out, expected_out);

        inplace_outdata expected_inplace = {{3, {1}}, {4, {2}}};
        EXPECT_EQ(inplace_selection, expected_inplace);
    }

    // inplace 2 buffers into one, but require zero offset for split buffer
    // buffer4 cannot reuse buffer 2 because it requires zero offset
    {
        std::vector<memory_optim::memory_alloc_trace_t> traces
                = {{1, 120}, {2, 100}, {3, 150}, {4, 50}, {5, 10}, {1, 0},
                        {2, 0}, {3, 0}, {4, 0}, {5, 0}};
        std::unordered_map<uintptr_t, size_t> out;
        inplace_outdata inplace_selection;
        inplace_data inplace_hint = {
                {3, {{1, inplace_kind::FREE}, {2, inplace_kind::FREE}}},
                {4, {{1, inplace_kind::FREE}, {2, inplace_kind::ZERO_OFFSET}}}};
        size_t total = memory_optim::schedule_memory_allocations(
                traces, 1, false, inplace_hint, out, inplace_selection);
        std::unordered_map<uintptr_t, size_t> expected_out
                = {{1, 0}, {2, 120}, {3, 0}, {4, 220}, {5, 270}};
        EXPECT_EQ(total, 280UL);
        EXPECT_EQ(out, expected_out);

        inplace_outdata expected_inplace = {{3, {1, 2}}};
        EXPECT_EQ(inplace_selection, expected_inplace);
    }

    // merge free to the right
    {
        std::vector<memory_optim::memory_alloc_trace_t> traces
                = {{1, 120}, {2, 100}, {3, 150}, {2, 0}, {4, 150}, {5, 10},
                        {1, 0}, {3, 0}, {4, 0}, {5, 0}};
        std::unordered_map<uintptr_t, size_t> out;
        inplace_outdata inplace_selection;
        inplace_data inplace_hint = {{4, {{1, inplace_kind::FREE}}}};
        size_t total = memory_optim::schedule_memory_allocations(
                traces, 1, false, inplace_hint, out, inplace_selection);
        std::unordered_map<uintptr_t, size_t> expected_out
                = {{1, 0}, {2, 120}, {3, 220}, {4, 0}, {5, 150}};
        EXPECT_EQ(total, 370UL);
        EXPECT_EQ(out, expected_out);

        inplace_outdata expected_inplace = {{4, {1}}};
        EXPECT_EQ(inplace_selection, expected_inplace);
    }

    // perfect matches
    {
        std::vector<memory_optim::memory_alloc_trace_t> traces
                = {{1, 120}, {2, 100}, {3, 100}, {4, 120}, {1, 0}, {2, 0},
                        {3, 0}, {4, 0}, {5, 200}, {5, 0}};
        std::unordered_map<uintptr_t, size_t> out;
        inplace_outdata inplace_selection;
        inplace_data inplace_hint = {
                {3, {{1, inplace_kind::FREE}, {2, inplace_kind::FREE}}},
                {4, {{1, inplace_kind::FREE}, {2, inplace_kind::FREE}}}};
        size_t total = memory_optim::schedule_memory_allocations(
                traces, 1, false, inplace_hint, out, inplace_selection);
        std::unordered_map<uintptr_t, size_t> expected_out
                = {{1, 0}, {2, 120}, {3, 120}, {4, 0}, {5, 0}};
        EXPECT_EQ(total, 220UL);
        EXPECT_EQ(out, expected_out);

        inplace_outdata expected_inplace = {{3, {2}}, {4, {1}}};
        EXPECT_EQ(inplace_selection, expected_inplace);
    }

    // selected inputs
    {
        std::vector<memory_optim::memory_alloc_trace_t> traces
                = {{1, 120}, {2, 100}, {3, 100}, {4, 120}, {1, 0}, {2, 0},
                        {3, 0}, {4, 0}, {5, 200}, {5, 0}};
        std::unordered_map<uintptr_t, size_t> out;
        inplace_outdata inplace_selection;
        inplace_data inplace_hint = {
                {3, {{1, inplace_kind::FREE}}}, {4, {{2, inplace_kind::FREE}}}};
        size_t total = memory_optim::schedule_memory_allocations(
                traces, 1, false, inplace_hint, out, inplace_selection);
        std::unordered_map<uintptr_t, size_t> expected_out
                = {{1, 0}, {2, 120}, {3, 0}, {4, 120}, {5, 0}};
        EXPECT_EQ(total, 240UL);
        EXPECT_EQ(out, expected_out);

        inplace_outdata expected_inplace = {{3, {1}}, {4, {2}}};
        EXPECT_EQ(inplace_selection, expected_inplace);
    }
}
