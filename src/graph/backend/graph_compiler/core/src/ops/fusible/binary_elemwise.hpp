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
    ABS_BWD,
    CLAMP_BWD,
    ELU_BWD,
    HARDSWISH_BWD,
    HARDSIGMOID_BWD,
    SQRT_BWD,
    MISH_BWD,
    TANH_BWD,
    SOFTPLUS_BWD,
};

/**
 * The binary_elementwise op, including add, sub, mul, div, min, max
 * squared_diff
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
    binary_elementwise_op_impl_t(
            graph_tensor_ptr lhs, graph_tensor_ptr rhs, elt_operator elt_op);
    binary_elementwise_op_impl_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs);

    std::vector<int> get_non_broadcast_input_index(
            bool assert_non_empty) const override;

    void set_elt_operator(elt_operator elt_op) { elt_op_ = elt_op; }
    elt_operator get_elt_operator() { return elt_op_; }

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
    void set_plain_bc_axis();
    vectorized_info_t &get_vx_info() { return vx_info_; }

    void infer_binding_axis(binding_axis_map &bdax_map) override;

    void pre_infer_binding_axis(binding_axis_map &bdax_map) override;

private:
    elt_operator elt_op_;
    vectorized_info_t vx_info_;
};

class add_op_t : public binary_elementwise_op_impl_t {
public:
    add_op_t(graph_tensor_ptr lhs, graph_tensor_ptr rhs)
        : binary_elementwise_op_impl_t(
                std::move(lhs), std::move(rhs), elt_operator::ADD) {
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
    sub_op_t(graph_tensor_ptr lhs, graph_tensor_ptr rhs)
        : binary_elementwise_op_impl_t(
                std::move(lhs), std::move(rhs), elt_operator::SUB) {
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
    mul_op_t(graph_tensor_ptr lhs, graph_tensor_ptr rhs)
        : binary_elementwise_op_impl_t(
                std::move(lhs), std::move(rhs), elt_operator::MUL) {
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
    div_op_t(graph_tensor_ptr lhs, graph_tensor_ptr rhs)
        : binary_elementwise_op_impl_t(
                std::move(lhs), std::move(rhs), elt_operator::DIV) {
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
    min_op_t(graph_tensor_ptr lhs, graph_tensor_ptr rhs)
        : binary_elementwise_op_impl_t(
                std::move(lhs), std::move(rhs), elt_operator::MIN) {
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
    max_op_t(graph_tensor_ptr lhs, graph_tensor_ptr rhs)
        : binary_elementwise_op_impl_t(
                std::move(lhs), std::move(rhs), elt_operator::MAX) {
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
    squared_diff_op_t(graph_tensor_ptr lhs, graph_tensor_ptr rhs)
        : binary_elementwise_op_impl_t(
                std::move(lhs), std::move(rhs), elt_operator::SQD_DIFF) {}
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
    prelu_op_t(graph_tensor_ptr lhs, graph_tensor_ptr rhs)
        : binary_elementwise_op_impl_t(
                std::move(lhs), std::move(rhs), elt_operator::PRELU) {}
    prelu_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
        : binary_elementwise_op_impl_t(ins, outs, attrs) {
        set_elt_operator(elt_operator::PRELU);
        op_name_ = "prelu";
    }
};

class unary_backward_base_t : public binary_elementwise_op_impl_t {
public:
    unary_backward_base_t(
            graph_tensor_ptr lhs, graph_tensor_ptr rhs, elt_operator elt_opt);
    unary_backward_base_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
        : binary_elementwise_op_impl_t(ins, outs, attrs) {
        COMPILE_ASSERT(gc::graph::check_shape_equal(
                               info_.inputs_[0]->details_.get_plain_dims(),
                               info_.inputs_[1]->details_.get_plain_dims())
                        && info_.inputs_[0]->details_.dtype_
                                == info_.inputs_[1]->details_.dtype_,
                "unary backward op's inputs is not set correctly.");
    };
    void compute_block(context_ptr ctx, const std::vector<tensor_slice *> &dst,
            const std::vector<const tensor_slice *> &inputs) override;
};

class abs_bwd_op_t : public unary_backward_base_t {
public:
    abs_bwd_op_t(
            graph_tensor_ptr lhs, graph_tensor_ptr rhs, bool vectorized = false)
        : unary_backward_base_t(
                std::move(lhs), std::move(rhs), elt_operator::ABS_BWD) {}
    // ins: ins[0] is src, ins[1] is diff_dst
    abs_bwd_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
        : unary_backward_base_t(ins, outs, attrs) {
        set_elt_operator(elt_operator::ABS_BWD);
        op_name_ = "abs_bwd";
    }
};

class clamp_bwd_op_t : public unary_backward_base_t {
public:
    clamp_bwd_op_t(graph_tensor_ptr src, graph_tensor_ptr dst)
        : unary_backward_base_t(
                std::move(src), std::move(dst), elt_operator::CLAMP_BWD) {}
    // ins: ins[0] is src or dst, ins[1] is diff_dst
    clamp_bwd_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
        : unary_backward_base_t(ins, outs, attrs) {
        set_elt_operator(elt_operator::CLAMP_BWD);
        op_name_ = "clamp_bwd";
        // should set min max attrs.
        assert(attrs.get_or_null<float>("min")
                && attrs.get_or_null<float>("max"));
        use_dst_ = attrs.get_or_else<bool>("use_dst", true);
    }
    bool use_dst_ = true;
};

class elu_bwd_op_t : public unary_backward_base_t {
public:
    elu_bwd_op_t(graph_tensor_ptr src_dst, graph_tensor_ptr diff_dst)
        : unary_backward_base_t(std::move(src_dst), std::move(diff_dst),
                elt_operator::ELU_BWD) {}
    // ins: ins[0] is src or dst, ins[1] is diff_dst
    elu_bwd_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
        : unary_backward_base_t(ins, outs, attrs) {
        assert(attrs.get_or_null<float>("alpha"));
        set_elt_operator(elt_operator::ELU_BWD);
        op_name_ = "elu_bwd";
        alpha_ = attrs_.get<float>("alpha");
        use_dst_ = attrs_.get_or_else<bool>("use_dst", true);
    }
    float alpha_;
    bool use_dst_;
};

class hardswish_bwd_op_t : public unary_backward_base_t {
public:
    hardswish_bwd_op_t(graph_tensor_ptr src_dst, graph_tensor_ptr diff_dst)
        : unary_backward_base_t(std::move(src_dst), std::move(diff_dst),
                elt_operator::HARDSWISH_BWD) {}
    // ins: ins[0] is src, ins[1] is diff_dst
    hardswish_bwd_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
        : unary_backward_base_t(ins, outs, attrs) {
        set_elt_operator(elt_operator::HARDSWISH_BWD);
        op_name_ = "hardswish_bwd";
        alpha_ = attrs_.get_or_else<float>("alpha", 1.f / 6.f);
        beta_ = attrs_.get_or_else<float>("beta", 0.5f);
    }
    float alpha_;
    float beta_;
};

class hardsigmoid_bwd_op_t : public unary_backward_base_t {
public:
    hardsigmoid_bwd_op_t(graph_tensor_ptr src_dst, graph_tensor_ptr diff_dst)
        : unary_backward_base_t(std::move(src_dst), std::move(diff_dst),
                elt_operator::HARDSIGMOID_BWD) {}
    // ins: ins[0] is src, ins[1] is diff_dst
    hardsigmoid_bwd_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
        : unary_backward_base_t(ins, outs, attrs) {
        assert(attrs.get_or_null<float>("beta")
                && attrs.get_or_null<float>("alpha"));
        set_elt_operator(elt_operator::HARDSIGMOID_BWD);
        op_name_ = "hardsigmoid_bwd";
        alpha_ = attrs_.get<float>("alpha");
        beta_ = attrs_.get<float>("beta");
    }
    float alpha_;
    float beta_;
};

class sqrt_bwd_op_t : public unary_backward_base_t {
public:
    sqrt_bwd_op_t(graph_tensor_ptr src_dst, graph_tensor_ptr diff_dst)
        : unary_backward_base_t(std::move(src_dst), std::move(diff_dst),
                elt_operator::SQRT_BWD) {}
    // ins: ins[0] is src, ins[1] is diff_dst
    sqrt_bwd_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
        : unary_backward_base_t(ins, outs, attrs) {
        set_elt_operator(elt_operator::SQRT_BWD);
        op_name_ = "sqrt_bwd";
        use_dst_ = attrs_.get_or_else<bool>("use_dst", true);
    }
    bool use_dst_;
};

class mish_bwd_op_t : public unary_backward_base_t {
public:
    mish_bwd_op_t(graph_tensor_ptr src_dst, graph_tensor_ptr diff_dst)
        : unary_backward_base_t(std::move(src_dst), std::move(diff_dst),
                elt_operator::MISH_BWD) {}
    // ins: ins[0] is src, ins[1] is diff_dst
    mish_bwd_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
        : unary_backward_base_t(ins, outs, attrs) {
        set_elt_operator(elt_operator::MISH_BWD);
        op_name_ = "mish_bwd";
    }
};

class tanh_bwd_op_t : public unary_backward_base_t {
public:
    tanh_bwd_op_t(graph_tensor_ptr src_dst, graph_tensor_ptr diff_dst)
        : unary_backward_base_t(std::move(src_dst), std::move(diff_dst),
                elt_operator::TANH_BWD) {}
    // ins: ins[0] is src, ins[1] is diff_dst
    tanh_bwd_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
        : unary_backward_base_t(ins, outs, attrs) {
        set_elt_operator(elt_operator::TANH_BWD);
        op_name_ = "tanh_bwd";
        use_dst_ = attrs_.get_or_else<bool>("use_dst", true);
    }
    bool use_dst_;
};

class softplus_bwd_op_t : public unary_backward_base_t {
public:
    softplus_bwd_op_t(graph_tensor_ptr src_dst, graph_tensor_ptr diff_dst)
        : unary_backward_base_t(std::move(src_dst), std::move(diff_dst),
                elt_operator::SOFTPLUS_BWD) {}
    // ins: ins[0] is src, ins[1] is diff_dst
    softplus_bwd_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
        : unary_backward_base_t(ins, outs, attrs) {
        set_elt_operator(elt_operator::SOFTPLUS_BWD);
        op_name_ = "soft_plus_bwd";
        beta_ = attrs_.get_or_else<float>("beta", 1.f);
    }
    bool beta_;
};

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
#endif
