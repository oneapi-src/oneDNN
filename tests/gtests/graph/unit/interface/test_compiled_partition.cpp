/*******************************************************************************
* Copyright 2021-2023 Intel Corporation
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

#include "gtest/gtest.h"

#include "graph/unit/unit_test_common.hpp"
#include "graph/unit/utils.hpp"

#include "interface/backend.hpp"
#include "interface/logical_tensor.hpp"
#include "interface/op.hpp"
#include "interface/partition.hpp"
#include "interface/partition_cache.hpp"

#include "oneapi/dnnl/dnnl.hpp"

namespace utils = dnnl::graph::tests::unit::utils;

namespace dnnl {
namespace graph {

TEST(CompiledPartitionCache, SingleOpCase) {
    const size_t max_batch = 4;

    impl::engine_t *eng = get_engine();
    std::vector<impl::graph::op_kind_t> kind_set {impl::graph::op_kind::ReLU,
            impl::graph::op_kind::ReLU, impl::graph::op_kind::Tanh};

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
            impl::graph::op_kind_t kind = kind_set[op_i];
            impl::graph::logical_tensor_t input = utils::logical_tensor_init(0,
                    {(int64_t)(batch * op_i + 1), 1, 1, 1},
                    impl::graph::data_type::f32,
                    impl::graph::layout_type::strided);
            impl::graph::logical_tensor_t output = utils::logical_tensor_init(1,
                    {(int64_t)(batch * op_i + 1), 1, 1, 1},
                    impl::graph::data_type::f32,
                    impl::graph::layout_type::strided);
            // Create op
            impl::graph::op_t elt {batch * op_i, kind, "elt"};
            elt.add_input(input);
            elt.add_output(output);

            // Create graph
            impl::graph::graph_t g {eng->kind()};
            g.add_op(&elt);
            g.finalize();

            // Create single-op partition
            std::vector<const impl::graph::backend_t *> &backends
                    = impl::graph::backend_registry_t::get_singleton()
                              .get_registered_backends();
            for (const auto &cbkd : backends) {
                impl::graph::backend_t *bkd
                        = const_cast<impl::graph::backend_t *>(cbkd);
                bkd->get_partitions(g, impl::graph::partition_policy::fusion);
            }

            // wrap into the partition
            impl::graph::partition_t par = impl::graph::partition_t();
            std::vector<impl::graph::partition_t *> parts {&par};
            g.get_ordered_partitions(parts);

            // highly possibly cache_miss
            tasks.emplace_back([eng, par, input, output]() {
                impl::graph::compiled_partition_t cp(par);
                std::pair<impl::graph::compiled_partition_t *, bool> cpcache {
                        &cp, false};
                std::vector<const impl::graph::logical_tensor_t *> inputs {
                        &input};
                std::vector<const impl::graph::logical_tensor_t *> outputs {
                        &output};
                // Partition compilation
                par.compile(cpcache, inputs, outputs, eng);
            });

            // highly possibly cache_hit
            tasks.emplace_back([eng, par, input, output]() {
                impl::graph::compiled_partition_t cp(par);
                std::pair<impl::graph::compiled_partition_t *, bool> cpcache {
                        &cp, false};
                std::vector<const impl::graph::logical_tensor_t *> inputs {
                        &input};
                std::vector<const impl::graph::logical_tensor_t *> outputs {
                        &output};
                // Partition compilation
                par.compile(cpcache, inputs, outputs, eng);
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
    ASSERT_EQ(impl::graph::compiled_partition_cache().get_capacity(),
            new_capacity);
#endif
}

TEST(LruCompiledPartitionCache, Method) {
    namespace graph = dnnl::impl::graph;

    graph::engine_t &eng = *get_engine();
    std::vector<graph::op_kind_t> kind_set {
            graph::op_kind::ReLU, graph::op_kind::ReLU, graph::op_kind::Tanh};

    graph::logical_tensor_t input = utils::logical_tensor_init(0, {1, 1, 1, 1},
            graph::data_type::f32, graph::layout_type::strided);
    graph::logical_tensor_t output = utils::logical_tensor_init(1, {1, 1, 1, 1},
            graph::data_type::f32, graph::layout_type::strided);
    // Create op
    auto elt = std::make_shared<graph::op_t>(1, graph::op_kind::Abs, "elt");
    elt->add_input(input);
    elt->add_output(output);

    // Create graph
    graph::graph_t g {eng.kind()};
    g.add_op(elt.get());
    g.finalize();

    // Create single-op partition
    std::vector<const graph::backend_t *> &backends
            = graph::backend_registry_t::get_singleton()
                      .get_registered_backends();
    for (const auto &cbkd : backends) {
        graph::backend_t *bkd = const_cast<graph::backend_t *>(cbkd);
        bkd->get_partitions(g, graph::partition_policy::fusion);
    }

    // wrap into the partition
    graph::partition_t par = graph::partition_t();
    std::vector<graph::partition_t *> parts {&par};
    g.get_ordered_partitions(parts);

    graph::compiled_partition_t cp(par);
    std::pair<graph::compiled_partition_t *, bool> cpcache {&cp, false};
    std::vector<const graph::logical_tensor_t *> inputs {&input};
    std::vector<const graph::logical_tensor_t *> outputs {&output};
    // Partition compilation
    par.compile(cpcache, inputs, outputs, &eng);

#ifndef DNNL_GRAPH_DISABLE_COMPILED_PARTITION_CACHE
    graph::partition_hashing::key_t key {
            par.id(), &eng, {elt}, inputs, outputs};
    auto &cache_mapper = graph::compiled_partition_cache();
    ASSERT_NO_THROW(cache_mapper.get_partition(key));

#endif
}

TEST(CompiledPartition, InvalidArguments) {
    namespace graph = dnnl::impl::graph;

    graph::partition_t pti;
    std::vector<const graph::logical_tensor_t *> inputs;
    std::vector<const graph::logical_tensor_t *> outputs;
    ASSERT_EQ(graph::status::invalid_arguments,
            pti.compile(nullptr, inputs, outputs, nullptr));
}

} // namespace graph
} // namespace dnnl
