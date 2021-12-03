/*******************************************************************************
 * Copyright 2020-2021 Intel Corporation
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

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_TRAITS_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_TRAITS_HPP

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "graph.hpp"
namespace sc {

class fusion_manager;
struct fusion_anchor_data;

namespace op_traits {
struct may_broadcast_t : public virtual op_base_trait_t {
    // returns the input index of the logical tensor that will be broadcast
    // returns -1 when it cannot broadcast
    virtual int get_broadcast_input() const = 0;
    virtual std::vector<int> infer_broadcast_axis() const = 0;

protected:
    std::vector<int> plain_bc_axis_;
};

struct copyable_t : public virtual op_base_trait_t {
    virtual sc_op_ptr copy(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, sc_graph_t &mgr)
            = 0;
};

/**
 * A class is auto-copyable if we can construct a valid copy of the node with
 * the in/out tensors and the attrs of the node. The op name should be in the op
 * registery
 * */
struct auto_copyable_t : public copyable_t {
    sc_op_ptr copy(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs,
            sc_graph_t &mgr) override;
};

// the OP can be optimized if some of the inputs are constants
struct constant_optimizable_t : public virtual op_base_trait_t {
    // do optimization and return the new optmized op. If no optimization can be
    // applied, return null
    virtual sc_op_ptr constant_optimize(sc_graph_t &graph) = 0;
};

// the part of OP's workload can be computed, e.g. intrisics(brgemm), tensor
// slice.
struct workload_computable_t : public virtual op_base_trait_t {
    using shape_dtype_pair = std::pair<sc_dims, sc_data_type_t>;
    static const size_t read_weight = 1UL;
    static const size_t write_weight = 1UL;
    static constexpr const char *workload_number = "workload_number";
    // compute workload with given input and output tensor pointers, according
    // to read/write times and operator numbers.
    virtual size_t compute_workload(const std::vector<shape_dtype_pair> &ins,
            const std::vector<shape_dtype_pair> &outs)
            = 0;
};

// the OP can accept a fusion manager to do post fusion
struct post_fusion_acceptable_t : public virtual op_base_trait_t {
    virtual ir_module_ptr get_func(context_ptr ctx,
            const std::shared_ptr<fusion_manager> &fuse_mgr,
            const std::string &func_name)
            = 0;
};

// quantize
struct may_quantize_t : public virtual op_base_trait_t {
    virtual sc_op_ptr do_compensations(
            sc_graph_t &mgr, const context_ptr &ctx) {
        need_compensation_ = false;
        return sc_op_ptr();
    }
    bool should_quantized_ = false;
    bool is_quantized_ = false;
    bool need_compensation_ = true;
};

struct data_compensation_t : public virtual op_base_trait_t {};
struct weight_compensation_t : public virtual op_base_trait_t {};
struct constant_compensation_t : public virtual op_base_trait_t {};

} // namespace op_traits

} // namespace sc

#endif
