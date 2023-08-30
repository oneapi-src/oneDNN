/*******************************************************************************
 * Copyright 2022-2023 Intel Corporation
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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_FUSIBLE_BINARY_ELEMWISE_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_FUSIBLE_BINARY_ELEMWISE_HPP

#include <utility>
#include <vector>
#include <compiler/ir/graph/fusible_op.hpp>
#include <runtime/microkernel/cpu/brgemm_alg_kind.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

enum class elt_operator {
    ADD,
    SUB,
    MUL,
    DIV,
    MIN,
    MAX,
    SQD_DIFF,
    PRELU,
};

/**
 * The binary_elementwise op, including add, sub, mul, div, min, max
 * squared_diff, and prelu
 * Inputs:
 *  - in[0] - the lhs input
 *  - in[1] - the rhs input
 * Outputs:
 *  - The result tensors
 * Attrs:
 *  - auto_broadcast: std::string - default = "numpy". If set to "none", lhs
 *    and rhs shall strictly match. If set to numpy, it will follow broadcast
 *    semantics
 *  - bc_axis: std::vector<int> (optional). If it is not set, the calculation
 *    will strictly follow auto-broadcast semantics. If set, the broadcast axis
 *    will follow the specified "bc_axis", which gives binary_elementwise op an
 *    opportunity to override auto-broadcast semantics. Notice that in this
 *    case, bc_input must have the same length as specified bc_axis to avoid
 *    semantic ambiguity.
 * */
class binary_elementwise_op_impl_t : public binary_elementwise_op_t {
public:
    DECLARE_QUERY_AND_COMPUTE();
    std::vector<std::pair<int, std::vector<tensor_inplace_info_t>>>
    get_inplace_map() override;
    binary_elementwise_op_impl_t(graph_tensor_ptr lhs, graph_tensor_ptr rhs,
            elt_operator elt_op, int inplace = 0);
    binary_elementwise_op_impl_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs);

    std::vector<int> get_non_broadcast_input_index(
            bool assert_non_empty) const override;

    void set_elt_operator(elt_operator elt_op) { elt_op_ = elt_op; }

    uint32_t get_lanes() const { return vx_info_.lanes; }

    int get_ref_input_index(bool assert_determined) const override;

    // Legacy function only required for old inplace info design
    void set_inplace_info();

    void query_format(context_ptr ctx,
            std::vector<std::vector<format_stride_pair>> &supported_ins,
            std::vector<std::vector<format_stride_pair>> &supported_outs)
            override;
    bool register_brgemm_fusion(const context_ptr &ctx,
            const std::vector<tensor_slice *> &outputs,
            const std::vector<const tensor_slice *> &inputs,
            brgemm_fusion_register &brg_reg) override;
    shape_rl_vec get_dynamic_shape_relations() const override;
    // get real broadcast axis, generally, you should set bc_axis on plain
    // format semantics if necessary.
    std::vector<int> get_bc_axis() const;
    vectorized_info_t &get_vx_info() { return vx_info_; }

    sc_dims get_bwise_fuse_shrink_dims() override;

    void collect_shrinked_lt_map(int bw_size, gt2gt_map &bw_lt_map) override;

    void collect_shrinked_axis_map(
            int bw_size, gt2axis_map &bw_axis_map) override;

    void infer_binding_axis(bound_axis_map &bdax_map) override;

    void pre_binding_axis(bound_axis_map &bdax_map) override;

private:
    elt_operator elt_op_;
    int inplace_;
    vectorized_info_t vx_info_;
};

class add_op_t : public binary_elementwise_op_impl_t {
public:
    add_op_t(graph_tensor_ptr lhs, graph_tensor_ptr rhs,
            bool vectorized = false, int inplace = 0)
        : binary_elementwise_op_impl_t(
                std::move(lhs), std::move(rhs), elt_operator::ADD, inplace) {
        alg_kind_ = brgemm::binary_add;
    }
    add_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
        : binary_elementwise_op_impl_t(ins, outs, attrs) {
        alg_kind_ = brgemm::binary_add;
        set_elt_operator(elt_operator::ADD);
        op_name_ = "add";
    }
};

class sub_op_t : public binary_elementwise_op_impl_t {
public:
    sub_op_t(graph_tensor_ptr lhs, graph_tensor_ptr rhs,
            bool vectorized = false, int inplace = 0)
        : binary_elementwise_op_impl_t(
                std::move(lhs), std::move(rhs), elt_operator::SUB, inplace) {
        alg_kind_ = brgemm::binary_sub;
    }
    sub_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
        : binary_elementwise_op_impl_t(ins, outs, attrs) {
        alg_kind_ = brgemm::binary_sub;
        set_elt_operator(elt_operator::SUB);
        op_name_ = "sub";
    }
};

class mul_op_t : public binary_elementwise_op_impl_t {
public:
    mul_op_t(graph_tensor_ptr lhs, graph_tensor_ptr rhs,
            bool vectorized = false, int inplace = 0)
        : binary_elementwise_op_impl_t(
                std::move(lhs), std::move(rhs), elt_operator::MUL, inplace) {
        alg_kind_ = brgemm::binary_mul;
    }
    mul_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
        : binary_elementwise_op_impl_t(ins, outs, attrs) {
        alg_kind_ = brgemm::binary_mul;
        set_elt_operator(elt_operator::MUL);
        op_name_ = "mul";
    }
};

class div_op_t : public binary_elementwise_op_impl_t {
public:
    div_op_t(graph_tensor_ptr lhs, graph_tensor_ptr rhs,
            bool vectorized = false, int inplace = 0)
        : binary_elementwise_op_impl_t(
                std::move(lhs), std::move(rhs), elt_operator::DIV, inplace) {
        alg_kind_ = brgemm::binary_div;
    }
    div_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
        : binary_elementwise_op_impl_t(ins, outs, attrs) {
        alg_kind_ = brgemm::binary_div;
        set_elt_operator(elt_operator::DIV);
        op_name_ = "div";
    }
};

class min_op_t : public binary_elementwise_op_impl_t {
public:
    min_op_t(graph_tensor_ptr lhs, graph_tensor_ptr rhs,
            bool vectorized = false, int inplace = 0)
        : binary_elementwise_op_impl_t(
                std::move(lhs), std::move(rhs), elt_operator::MIN, inplace) {
        alg_kind_ = brgemm::binary_min;
    }
    min_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
        : binary_elementwise_op_impl_t(ins, outs, attrs) {
        alg_kind_ = brgemm::binary_min;
        set_elt_operator(elt_operator::MIN);
        op_name_ = "min";
    }
};

class max_op_t : public binary_elementwise_op_impl_t {
public:
    max_op_t(graph_tensor_ptr lhs, graph_tensor_ptr rhs,
            bool vectorized = false, int inplace = 0)
        : binary_elementwise_op_impl_t(
                std::move(lhs), std::move(rhs), elt_operator::MAX, inplace) {
        alg_kind_ = brgemm::binary_max;
    }
    max_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
        : binary_elementwise_op_impl_t(ins, outs, attrs) {
        alg_kind_ = brgemm::binary_max;
        set_elt_operator(elt_operator::MAX);
        op_name_ = "max";
    }
};

// squared_difference: (x-mean)^2
// squared_diff should support both elementwise and broad-cast mode.
class squared_diff_op_t : public binary_elementwise_op_impl_t {
public:
    squared_diff_op_t(graph_tensor_ptr lhs, graph_tensor_ptr rhs,
            bool vectorized = false, int inplace = 0)
        : binary_elementwise_op_impl_t(std::move(lhs), std::move(rhs),
                elt_operator::SQD_DIFF, inplace) {}
    squared_diff_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
        : binary_elementwise_op_impl_t(ins, outs, attrs) {
        set_elt_operator(elt_operator::SQD_DIFF);
        op_name_ = "squared_diff";
    }
};

// parameter version of leaky_relu.
class prelu_op_t : public binary_elementwise_op_impl_t {
public:
    prelu_op_t(graph_tensor_ptr lhs, graph_tensor_ptr rhs,
            bool vectorized = false, int inplace = 0)
        : binary_elementwise_op_impl_t(
                std::move(lhs), std::move(rhs), elt_operator::PRELU, inplace) {}
    prelu_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
        : binary_elementwise_op_impl_t(ins, outs, attrs) {
        set_elt_operator(elt_operator::PRELU);
        op_name_ = "prelu";
    }
};

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
#endif
