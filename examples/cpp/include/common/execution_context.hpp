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

#ifndef COMMON_EXECUTION_CONTEXT_HPP
#define COMMON_EXECUTION_CONTEXT_HPP

#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <unordered_set>

#include "oneapi/dnnl/dnnl_graph.hpp"

#include "utils.hpp"

#ifdef DNNL_GRAPH_WITH_SYCL
#include <CL/sycl.hpp>
#endif

/// A mapping from id to tensor is used to manage the lifecycle of all created
/// tensors since these tensors need to be held until all compiled partitions'
/// execution finished
class tensor_map {
    using data_type = dnnl::graph::logical_tensor::data_type;
    using layout_type = dnnl::graph::logical_tensor::layout_type;
    using dims_t = dnnl::graph::logical_tensor::dims_t;

private:
    /// mapping from id to tensor
    std::unordered_map<size_t, dnnl::graph::tensor> data_;

    /// workaround: need to know the data type of tensor's handle
    /// need to further enhance tensor.get_data_handle() API
    std::unordered_map<size_t, data_type> id_to_dtypes_;

    /// containing ids of in-placed tensors
    std::unordered_set<size_t> inplace_tensor_ids_;
#ifdef DNNL_GRAPH_WITH_SYCL
    /// q is for deallocation of USM memory buffer
    cl::sycl::queue q_;
#endif

public:
    tensor_map() = default;

#ifdef DNNL_GRAPH_WITH_SYCL
    tensor_map(const cl::sycl::queue &q) : q_(q) {}
#endif

    /// destructor - free all of allocated memory buffer
    ~tensor_map() {
        for (const auto &v : data_) {
            // address double-free issue for inplace scenario
            // if two tensors share the same memory buffer, need skip the second
            // time of freeing memory
            if (inplace_tensor_ids_.find(v.first) != inplace_tensor_ids_.end())
                continue;

            void *mem_ptr
                    = get_handle_from_tensor(v.second, id_to_dtypes_[v.first]);
            if (mem_ptr) {
#ifdef DNNL_GRAPH_WITH_SYCL
                cl::sycl::free(mem_ptr, q_.get_context());
#else
                free(mem_ptr);
#endif
            }
        }
    }

    /// Add or insert a new tensor into the tensor map
    ///
    /// @param id the given unique id
    /// @param ts a new tensor
    void insert_or_replace(size_t id, const dnnl::graph::tensor &ts) {
        auto iter = data_.find(id);
        if (iter != data_.end()) {
            std::cout << "Warning: inserting the same tensor twice time, is it "
                         "intended?\n";
            // since this will replace old tensor with a new one, so here need
            // to free the memory buffer of the old tensor
            void *old_mem_ptr
                    = get_handle_from_tensor(iter->second, id_to_dtypes_[id]);
            if (old_mem_ptr)
#ifdef DNNL_GRAPH_WITH_SYCL
                cl::sycl::free(old_mem_ptr, q_.get_context());
#else
                free(old_mem_ptr);
#endif
        }
        data_[id] = ts;
    }

    /// Return a flag to indicate whether this tensor map contains such a tensor
    /// according to the given id
    ///
    /// @param id the given unique id
    /// @return @c true if this tensor map contains the tensor with this id
    ///     @c false if this tensor map doesn't contain the tensor
    bool has(size_t id) const { return data_.find(id) != data_.end(); }

    /// Retrieve the tensor from this tensor map according to the given id
    ///
    /// @param id the given unique id
    /// @return A copy of tensor
    const dnnl::graph::tensor get(size_t id) const { return data_.at(id); }

    /// Create and initialize input/output tensors for the compiled partition
    ///
    /// @param lts a list of logical tensors to be constructed and initialized
    /// @param c_partition target compiled partition
    /// @param value an initialization value of buffer
    /// @return a list of copies of tensors
    std::vector<dnnl::graph::tensor> construct_and_initialize_tensors(
            const std::vector<dnnl::graph::logical_tensor> &lts,
            dnnl::graph::compiled_partition &c_partition, int value) {
        std::vector<dnnl::graph::tensor> ret;
        ret.reserve(lts.size());
        for (auto &lt : lts) {
            size_t id = lt.get_id();
            if (this->has(id)) {
                // this tensor map has already contained a tensor, just take
                // it out
                ret.emplace_back(this->get(id));
            } else {
                // this tensor map doesn't contain this tensor, need create from
                // logical tensor queried from compiled partition
                dnnl::graph::logical_tensor new_lt
                        = c_partition.query_logical_tensor(id);
                // allocate and initialize memory buffer
#ifdef DNNL_GRAPH_WITH_SYCL
                void *mem_ptr = cl::sycl::malloc_shared(new_lt.get_mem_size(),
                        q_.get_device(), q_.get_context());
                EXAMPLE_SWITCH_TYPE(new_lt.get_data_type(), dtype, {
                    fill_buffer<dtype>(q_, mem_ptr,
                            static_cast<size_t>(
                                    new_lt.get_mem_size() / sizeof(dtype)),
                            value);
                });
#else
                void *mem_ptr = malloc(new_lt.get_mem_size());
                EXAMPLE_SWITCH_TYPE(new_lt.get_data_type(), dtype, {
                    fill_buffer<dtype>(mem_ptr, new_lt.get_mem_size(), value);
                });
#endif
                ret.emplace_back(dnnl::graph::tensor {new_lt, mem_ptr});
                this->insert_or_replace(new_lt.get_id(), ret.back());
                id_to_dtypes_[new_lt.get_id()] = new_lt.get_data_type();
            }
        }
        return ret;
    }

    /// Convert tensor to a queried format that is determined by backend
    ///
    /// @param ts a list of tensors
    /// @param lts a list of logical tensors in the same order as given tensors
    /// @param queried_lt the querid logical tensors with another format
    /// @param eng an engine used to compile the partition
    /// @param strm a stream used to execute the compiled partition
    void convert_tensor_with_queried_format(
            std::vector<dnnl::graph::tensor> &ts,
            const std::vector<dnnl::graph::logical_tensor> &lts,
            const dnnl::graph::logical_tensor &queried_lt,
            dnnl::graph::engine &eng, dnnl::graph::stream &strm) {
        size_t lid = queried_lt.get_id();
        auto found_lt_iter = std::find_if(lts.begin(), lts.end(),
                [lid](const dnnl::graph::logical_tensor &lt) {
                    return lid == lt.get_id();
                });
        auto idx = static_cast<size_t>(
                std::distance(lts.begin(), found_lt_iter));

        // firstly create a logical tensor with plain format
        dims_t ori_dims = lts[idx].get_dims();
        data_type ori_dtype = lts[idx].get_data_type();
        dnnl::graph::logical_tensor ori_lt {
                lid, ori_dtype, ori_dims, layout_type::strided};

        void *buffer = nullptr;
        if (!queried_lt.has_same_layout_and_dtype(lts[idx])) {
#ifdef DNNL_GRAPH_WITH_SYCL
            buffer = cl::sycl::malloc_shared(queried_lt.get_mem_size(),
                    q_.get_device(), q_.get_context());
#else
            buffer = malloc(queried_lt.get_mem_size());
#endif
            // create a conversion partition
            dnnl::graph::conversion convert {};
            // compile to compiled partition
            dnnl::graph::compiled_partition convert_executable
                    = convert.compile(ori_lt, queried_lt, eng);
            // real tensor with queried layout
            dnnl::graph::tensor tensor_r {queried_lt, buffer};
            // execute the conversion
            convert_executable.execute(strm, {ts[idx]}, {tensor_r});
            // replace the original tensor with the new one
            ts[idx] = tensor_r;
            this->insert_or_replace(lid, tensor_r);
            id_to_dtypes_[lid] = ori_dtype;
        }
    }

    /// Update tensor's handle according to the inplace pairs
    ///
    /// @param inputs a list of input tensors
    /// @param outputs a list of output tensors
    /// @param input_lts a list of input logical tensors in the same order as
    ///     inputs
    /// @param output_lts a list of input logical tensors in the same order as
    ///     outputs
    /// @param inplace_options the given inplace pairs (input id, output id)
    void update_tensor_handle_by_inplace_options(
            const std::vector<dnnl::graph::tensor> &inputs,
            std::vector<dnnl::graph::tensor> &outputs,
            const std::vector<dnnl::graph::logical_tensor> &input_lts,
            const std::vector<dnnl::graph::logical_tensor> &output_lts,
            const std::vector<std::pair<size_t, size_t>> &inplace_options) {
        for (const auto &p : inplace_options) {
            size_t input_id = p.first;
            size_t output_id = p.second;
            auto input_lt_iter
                    = std::find_if(input_lts.begin(), input_lts.end(),
                            [input_id](const dnnl::graph::logical_tensor &lt) {
                                return input_id == lt.get_id();
                            });
            auto input_lt_idx = static_cast<size_t>(
                    std::distance(input_lts.begin(), input_lt_iter));
            auto output_lt_iter
                    = std::find_if(output_lts.begin(), output_lts.end(),
                            [output_id](const dnnl::graph::logical_tensor &lt) {
                                return output_id == lt.get_id();
                            });
            auto output_lt_idx = static_cast<size_t>(
                    std::distance(output_lts.begin(), output_lt_iter));

            // free original output buffer if it can be in-placed.
            void *ori_mem_ptr = get_handle_from_tensor(outputs[output_lt_idx],
                    output_lts[output_lt_idx].get_data_type());
            if (ori_mem_ptr) {
#ifdef DNNL_GRAPH_WITH_SYCL
                cl::sycl::free(ori_mem_ptr, q_.get_context());
#else
                free(ori_mem_ptr);
#endif
            }
            void *new_mem_ptr = get_handle_from_tensor(inputs[input_lt_idx],
                    input_lts[input_lt_idx].get_data_type());
            outputs[output_lt_idx].set_data_handle(new_mem_ptr);
            inplace_tensor_ids_.insert(output_id);
        }
    }
}; // class tensor_map
#endif
