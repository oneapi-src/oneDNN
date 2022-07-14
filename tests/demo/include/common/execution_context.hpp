/*******************************************************************************
* Copyright 2021-2022 Intel Corporation
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
#include <memory>
#include <unordered_map>
#include <unordered_set>

#include "oneapi/dnnl/dnnl_graph.hpp"

#include "utils.hpp"

#ifdef DNNL_GRAPH_WITH_SYCL
#include <CL/sycl.hpp>
#endif

struct cpu_deletor {
    cpu_deletor() = default;
    void operator()(void *ptr) {
        if (ptr) free(ptr);
    }
};

#ifdef DNNL_GRAPH_WITH_SYCL
struct sycl_deletor {
    ::sycl::context ctx_;
    sycl_deletor() = delete;
    sycl_deletor(const ::sycl::context &ctx) : ctx_(ctx) {}
    void operator()(void *ptr) {
        if (ptr) ::sycl::free(ptr, ctx_);
    }
};
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
    std::unordered_map<size_t, std::shared_ptr<void>> buffer_map_;

    /// containing ids of in-placed tensors
    std::unordered_set<size_t> inplace_tensor_ids_;
#ifdef DNNL_GRAPH_WITH_SYCL
    /// q is for deallocation of USM memory buffer
    ::sycl::queue q_;
#endif

public:
    tensor_map() = default;

#ifdef DNNL_GRAPH_WITH_SYCL
    tensor_map(const ::sycl::queue &q) : q_(q) {}
#endif

    /// destructor - free all of allocated memory buffer
    ~tensor_map() = default;

    /// Add or insert a new tensor into the tensor map
    ///
    /// @param id the given unique id
    /// @param ts a new tensor
    void insert_or_replace(size_t id, const dnnl::graph::tensor &ts) {
        auto iter = data_.find(id);
        if (iter != data_.end()) {
            std::cout << "Warning: inserting the same tensor twice time, is it "
                         "intended?\n";
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
    /// @param eng target engine for the tensor
    /// @param value an initialization value of buffer
    /// @return a list of copies of tensors
    std::vector<dnnl::graph::tensor> construct_and_initialize_tensors(
            const std::vector<dnnl::graph::logical_tensor> &lts,
            dnnl::graph::compiled_partition &c_partition,
            dnnl::graph::engine &eng, int value) {
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
                // memory buffer allocation and initialization
                void *mem_ptr = nullptr;
                if (eng.get_kind() == dnnl::graph::engine::kind::cpu) {
                    // cpu
#ifdef DNNL_GRAPH_CPU_SYCL
                    mem_ptr = ::sycl::malloc_shared(new_lt.get_mem_size(),
                            q_.get_device(), q_.get_context());
                    buffer_map_[new_lt.get_id()].reset(
                            mem_ptr, sycl_deletor {q_.get_context()});
                    EXAMPLE_SWITCH_TYPE(new_lt.get_data_type(), dtype, {
                        fill_buffer<dtype>(q_, mem_ptr,
                                static_cast<size_t>(
                                        new_lt.get_mem_size() / sizeof(dtype)),
                                value);
                    });
#else
                    mem_ptr = malloc(new_lt.get_mem_size());
                    buffer_map_[new_lt.get_id()].reset(mem_ptr, cpu_deletor {});
                    EXAMPLE_SWITCH_TYPE(new_lt.get_data_type(), dtype, {
                        fill_buffer<dtype>(
                                mem_ptr, new_lt.get_mem_size(), value);
                    });
#endif
                } else {
                    // gpu
#ifdef DNNL_GRAPH_GPU_SYCL
                    mem_ptr = ::sycl::malloc_shared(new_lt.get_mem_size(),
                            q_.get_device(), q_.get_context());
                    buffer_map_[new_lt.get_id()].reset(
                            mem_ptr, sycl_deletor {q_.get_context()});
                    EXAMPLE_SWITCH_TYPE(new_lt.get_data_type(), dtype, {
                        fill_buffer<dtype>(q_, mem_ptr,
                                static_cast<size_t>(
                                        new_lt.get_mem_size() / sizeof(dtype)),
                                value);
                    });
#endif
                }
                ret.emplace_back(dnnl::graph::tensor {new_lt, eng, mem_ptr});
                this->insert_or_replace(new_lt.get_id(), ret.back());
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

        if (!queried_lt.has_same_layout(lts[idx])) {
            std::shared_ptr<void> buffer;
            if (eng.get_kind() == dnnl::graph::engine::kind::cpu) {
                // cpu
#ifdef DNNL_GRAPH_CPU_SYCL
                buffer.reset(::sycl::malloc_shared(queried_lt.get_mem_size(),
                                     q_.get_device(), q_.get_context()),
                        sycl_deletor {q_.get_context()});
#else
                buffer.reset(malloc(queried_lt.get_mem_size()), cpu_deletor {});
#endif
            } else {
                // gpu
#ifdef DNNL_GRAPH_GPU_SYCL
                buffer.reset(::sycl::malloc_shared(queried_lt.get_mem_size(),
                                     q_.get_device(), q_.get_context()),
                        sycl_deletor {q_.get_context()});
#endif
            }
            // create a conversion partition
            dnnl::graph::op reorder_op {0, dnnl::graph::op::kind::Reorder,
                    {ori_lt}, {queried_lt}, "reorder"};
            dnnl::graph::partition convert {reorder_op, eng.get_kind()};
            // compile to compiled partition
            dnnl::graph::compiled_partition convert_executable
                    = convert.compile({ori_lt}, {queried_lt}, eng);
            // real tensor with queried layout
            dnnl::graph::tensor tensor_r {queried_lt, eng, buffer.get()};
            // execute the conversion
            convert_executable.execute(strm, {ts[idx]}, {tensor_r});
            // replace the original tensor with the new one
            ts[idx] = tensor_r;
            this->insert_or_replace(lid, tensor_r);
            buffer_map_[lid] = buffer;
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
            buffer_map_[output_lts[output_lt_idx].get_id()].reset();
            void *new_mem_ptr = inputs[input_lt_idx].get_data_handle();
            outputs[output_lt_idx].set_data_handle(new_mem_ptr);
            inplace_tensor_ids_.insert(output_id);
        }
    }
}; // class tensor_map
#endif
