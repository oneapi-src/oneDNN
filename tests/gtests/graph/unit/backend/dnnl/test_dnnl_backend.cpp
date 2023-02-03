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
#include "gtest/gtest.h"

#include "backend/dnnl/dnnl_backend.hpp"

#include "utils/any.hpp"
#include "utils/utils.hpp"

namespace graph = dnnl::impl::graph;

TEST(LayoutIdManager, GetMemDesc) {
    class layout_id_manager_test_impl
        : public graph::dnnl_impl::layout_id_manager_t {
        bool is_mem_desc_equal(const graph::utils::any_t &mem_desc1,
                const graph::utils::any_t &mem_desc2) const override {
            return true;
        }
    };
    layout_id_manager_test_impl manager;
    size_t layout_id1 = 1;
    ASSERT_FALSE(manager.get_mem_desc(layout_id1).has_value());
    graph::utils::any_t mem_desc(int64_t(12));
    graph::utils::optional_t<size_t> layout_id2
            = manager.set_mem_desc(mem_desc);
    ASSERT_TRUE(layout_id2.has_value());

    ASSERT_TRUE(manager.get_mem_desc(layout_id2.value()).has_value());
    ASSERT_EQ(graph::utils::any_cast<int64_t>(
                      manager.get_mem_desc(layout_id2.value()).value()),
            graph::utils::any_cast<int64_t>(mem_desc));
}

TEST(LargetPartition, LargerPartitionKernelCreator) {
    ASSERT_NO_THROW(graph::dnnl_impl::large_partition_kernel_creator());
}
