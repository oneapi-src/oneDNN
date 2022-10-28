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

#ifndef BENCHDNN_GRAPH_EXECUTION_CONTEXT_HPP
#define BENCHDNN_GRAPH_EXECUTION_CONTEXT_HPP

#include <algorithm>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <unordered_set>

#include "oneapi/dnnl/dnnl_graph.hpp"
#include "utils.hpp"

#ifdef DNNL_WITH_SYCL
#include <CL/sycl.hpp>
#endif

namespace graph {

/// A mapping from id to tensor is used to manage the lifecycle of all created
/// tensors since these tensors need to be held until all compiled partitions'
/// execution finished
class tensor_map {
    using data_type = dnnl::graph::logical_tensor::data_type;
    using layout_type = dnnl::graph::logical_tensor::layout_type;

private:
    /// mapping from id to tensor
    std::unordered_map<size_t, dnnl::graph::tensor> data_;
    std::unordered_map<size_t, std::shared_ptr<void>> buffer_map_;

    /// containing ids of in-placed tensors
    std::unordered_set<size_t> inplace_tensor_ids_;
#ifdef DNNL_WITH_SYCL
    /// q_ is for deallocation of USM memory buffer
    ::sycl::queue q_;
#endif

    /// Add or insert a new tensor into the tensor map
    ///
    /// @param id Unique id to add or insert.
    /// @param ts A tensor to add or insert.
    void insert_or_replace(size_t id, const dnnl::graph::tensor &ts);

    /// Return a flag to indicate whether this tensor map contains such a tensor
    /// according to the given id
    ///
    /// @param id the given unique id
    /// @return @c true if this tensor map contains the tensor with this id
    ///     @c false if this tensor map doesn't contain the tensor
    bool has(size_t id) const { return data_.find(id) != data_.end(); }

public:
    tensor_map() = default;

#ifdef DNNL_WITH_SYCL
    tensor_map(const ::sycl::queue &q) : q_(q) {}
#endif

    /// destructor - free all of allocated memory buffer
    ~tensor_map() = default;

    /// Create and initialize input/output tensors for the compiled partition
    ///
    /// @param lts a list of logical tensors to be constructed and initialized
    /// @param c_partition target compiled partition
    /// @param eng target engine for the tensor
    /// @param value an initialization value of buffer
    /// @return a list of copies of tensors
    std::vector<dnnl::graph::tensor> construct_and_initialize_tensors(
            const std::vector<dnnl::graph::logical_tensor> &lts,
            const dnnl::graph::compiled_partition &c_partition,
            const dnnl::engine &eng, int value);
};

} // namespace graph
#endif
