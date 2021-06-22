/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include <thread>

#include "oneapi/dnnl/dnnl_graph.hpp"
#include "gtest/gtest.h"

#include "test_api_common.hpp"

#include "interface/partition_cache.hpp"

namespace dnnl {
namespace graph {

TEST(compiled_partition_cache_mt_test, SingleOpCase) {
    using data_type = logical_tensor::data_type;
    using layout_type = logical_tensor::layout_type;

    const size_t max_batch = 4;

    engine::kind engine_kind = engine::kind::cpu;
    engine eng {engine_kind, 0};
    std::vector<op::kind> kind_set {
            op::kind::ReLU, op::kind::ReLU, op::kind::Tanh};

    const size_t num_eltwise_kind = kind_set.size();

    // Flush the cache
    set_compiled_partition_cache_capacity(0);
    set_compiled_partition_cache_capacity(1024);

    const int n_compiled_partitions
            = static_cast<int>(num_eltwise_kind * max_batch);

    std::vector<std::thread> tasks;
    tasks.reserve(n_compiled_partitions * 2);

    for (size_t batch = 0; batch < max_batch; ++batch) {
        for (size_t op_i = 0; op_i < kind_set.size(); ++op_i) {
            op::kind kind = kind_set[op_i];
            logical_tensor input {0, data_type::f32,
                    {(int64_t)(batch * op_i + 1), 1, 1, 1},
                    layout_type::strided};
            logical_tensor output {1, data_type::f32,
                    {(int64_t)(batch * op_i + 1), 1, 1, 1},
                    layout_type::strided};

            // Create op
            op elt {batch * op_i, kind, {input}, {output}, "elt"};

            // Create single-op partition
            partition par {elt, engine_kind};

            // highly possibly cache_miss
            tasks.emplace_back([&kind_set, &eng, par, input, output]() {
                // Partition compilation
                auto cp = par.compile({input}, {output}, eng);
            });

            // highly possibly cache_hit
            tasks.emplace_back([&kind_set, &eng, par, input, output]() {
                // Partition compilation
                auto cp = par.compile({input}, {output}, eng);
            });
        }
    }

    // join tasks
    for (auto &t : tasks)
        t.join();

#ifdef DNNL_GRAPH_DISABLE_COMPILED_PARTITION_CACHE
    ASSERT_EQ(get_compiled_partition_cache_size(), 0);
#else
    ASSERT_EQ(get_compiled_partition_cache_size(), n_compiled_partitions);
#endif

    // test evict(n_compiled_partitions - 2)
    const int new_capacity = 2;
    set_compiled_partition_cache_capacity(new_capacity);

#ifdef DNNL_GRAPH_DISABLE_COMPILED_PARTITION_CACHE
    ASSERT_EQ(get_compiled_partition_cache_size(), 0);
#else
    ASSERT_EQ(get_compiled_partition_cache_size(), new_capacity);
#endif
}

} // namespace graph
} // namespace dnnl
