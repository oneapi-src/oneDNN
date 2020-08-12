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

#ifndef LLGA_PARTITION_HPP
#define LLGA_PARTITION_HPP

#include <utility>
#include <vector>
#include <unordered_set>

#include "engine.hpp"
#include "llga_api_detail.hpp"
#include "op.hpp"
#include "tensor.hpp"

namespace llga {
namespace api {

class compiled_partition;

class partition {
public:
    /// Returns the number of nodes in the partition
    ///
    /// @returns Number of nodes
    uint64_t get_nodes_num() const;

    /// Returns all node's id of the partition
    ///
    /// @returns An unordered set of node ids
    std::vector<uint64_t> get_nodes();

    /// Returns a list of input logical tensor in the partition
    ///
    /// @returns A list of logical tensor
    std::vector<logical_tensor> get_inputs() const;

    /// Returns a list of output logical tensor in the partition
    ///
    /// @returns A list of logical tensor
    std::vector<logical_tensor> get_outputs() const;

    /// Returns the unique id of the partition
    ///
    /// @returns Unique id
    uint64_t get_id() const;

    /// Compile the partition to generate compiled partition based
    /// on the input/output logical tensors. The order of these two lists
    /// may have already been changed according to the fwk fused node.
    ///
    /// @param inputs A list of input logical tensors
    /// @param outputs A list of output logical tensors
    /// @returns A compiled partition
    compiled_partition compile(std::vector<logical_tensor> &inputs,
            std::vector<logical_tensor> &outputs);
};

class compiled_partition {
public:
    /// Constructs a compiled partition object
    compiled_partition(llga_compiled_partition_t *compiled_partition);

    /// Returns the source partition from a compiled partition
    ///
    /// @returns A copy of partition object
    partition get_partition();

    /// Returns the underlying execution kernel's handle of the compiled partition
    ///
    /// @returns Handle of the kernel function
    void *get_handle();
};

} // namespace api
} // namespace llga

#endif
