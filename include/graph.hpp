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
    /// Constructs a graph session using the engine context
    ///
    /// @param aengine The engine context of this session
    graph(const engine &aengine);

    /// Constructs a graph session using device information
    ///
    /// @param device_type The device type of this session, can be cpu, gpu
    ///     or any supported device.
    /// @param device_id The device id
    graph(engine::kind engine_kind, int32_t device_id);

    /// Selects an op to the graph session to construct DAG for analysis
    ///
    /// @param aop An operator/node that represents the entry of frameworks'
    ///    graph
    bool select(const op &aop);

    /// Vector to store the partitions
    using partition_vec = std::vector<std::unique_ptr<partition>>;
    /// Filter each partitions to apply some pass
    ///
    /// @param policy Partition policy
    /// @return partition_vec A vector storing the partitions
    partition_vec filter_partitions(llga_partition_policy policy);
};
} // namespace api
} // namespace llga

#endif
