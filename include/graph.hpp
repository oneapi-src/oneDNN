/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#ifndef LLGA_GRAPH_HPP
#define LLGA_GRAPH_HPP

#include <memory>
#include <vector>

#include "engine.hpp"
#include "llga_api_detail.hpp"
#include "op.hpp"
#include "partition.hpp"

namespace llga {
namespace api {

/// A graph session to start analysis of computational DAG
class graph {
public:
    /// Constructs a graph session using device information
    ///
    /// @param engine_kind Can be cpu, gpu or any supported engine.
    graph(engine::kind engine_kind);

    /// Add an op to the graph session to construct DAG for analysis
    ///
    /// @param op An operator/node that represents the entry of frameworks'
    ///    graph
    void add_op(const op &op);

    /// Vector to store the partitions
    using partition_vec = std::vector<partition>;
    /// Get filtered partitions
    ///
    /// @param policy Partition policy
    /// @return partition_vec A vector storing the partitions
    partition_vec get_partitions(llga_partition_policy policy);
};
} // namespace api
} // namespace llga

#endif
