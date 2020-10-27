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

class compiled_partition {
public:
    /// Default constructor. Constructs an empty object.
    compiled_partition() = default;

    /// Constructs a compiled partition object
    ///
    /// @param compiled_partition A raw pointer of C API handle
    compiled_partition(llga_compiled_partition_t *compiled_partition);

    /// Returns the logical tensor according to tensor id
    ///
    /// @param tid The unique id of required tensor
    /// @returns The logical tensor
    logical_tensor query_logical_tensor(uint64_t tid);

    /// Execute a compiled partition.
    ///
    /// @param astream The stream used for execution
    /// @param inputs A list of input tensors
    /// @param outputs A list of output tensors
    void execute(stream &astream, const std::vector<tensor> &inputs, 
        const std::vector<tensor> &outputs);

    /// Execute a compiled partition
    ///
    /// @param astream Stream object to run over
    /// @param inputs A list of input tensors in the partition
    /// @param outputs A list of output tensors in the partition
    /// @param deps Optional vector with `cl::sycl::event` dependencies.
    /// @returns event Output event
    cl::sycl::event execute_sycl(stream &astream,
            const std::vector<tensor> &inputs, std::vector<tensor> &outputs,
            const std::vector<cl::sycl::event> &deps = {}) const;
};

class partition {
public:
    /// Default constructor for partition
    partition() = default;

    /// Constructs a partition object
    ///
    /// @param p A raw pointer to the C API handle
    partition(llga_partition_t *p);

    /// Returns the number of llga ops in the partition
    ///
    /// @returns Number of ops
    uint64_t get_ops_num() const;

    /// Returns all op’s id of the partition
    ///
    /// @returns An unordered set of op ids
    std::vector<uint64_t> get_ops();

    /// Sets layout id to the specified tensor in the partition
    ///
    /// @param tensor_id The unique id of tensor
    /// @param layout_id The layout id
    /// @returns @c true if the specified tensor is found
    ///     @c false if the specified tensor is not found
    void set_layout_id(uint64_t tid, uint64_t lid);

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
    /// @param e The engine used to compile the partition
    /// @returns A compiled partition
    compiled_partition compile(const std::vector<logical_tensor> &inputs,
            const std::vector<logical_tensor> &outputs, const engine &e);

    /// Infer the shape of outputs
    ///
    /// @param inputs A list of input logical tensors
    /// @param outputs A list of output logical tensors
    /// @returns @c true if the shape is inferred successfully
    ///          @c false if the shape is not inferred 
    bool infer_shape(const std::vector<logical_tensor> &inputs,
            std::vector<logical_tensor> &outputs) const;
};

} // namespace api
} // namespace llga

#endif
