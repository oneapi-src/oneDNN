/*******************************************************************************
 * Copyright 2020-2022 Intel Corporation
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

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_FUSIBLE_UNARY_ELEMWISE_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_FUSIBLE_UNARY_ELEMWISE_HPP

#include <string>
#include <utility>
#include <vector>
#include <compiler/ir/graph/fusible_op.hpp>

#define DECLARE_COMPUTE_ELEMENT() expr compute_element(expr in) override;

namespace sc {

class unary_elementwise_op_impl_t : public unary_elementwise_op_t {
public:
    void infer_slice_ranges(
            fslice_map &fsmap, infer_status_map_t &stat_map) override;
    void pre_slice_ranges(
            fslice_map &fsmap, infer_status_map_t &stat_map) override;
    void prepare_fusion_data(fdata_map &fdmap) override;

    void compute_block(context_ptr ctx, const std::vector<tensor_slice *> &dst,
            const std::vector<const tensor_slice *> &inputs) override;

    bool register_brgemm_fusion(const context_ptr &ctx,
            const std::vector<tensor_slice *> &outputs,
            const std::vector<const tensor_slice *> &inputs,
            brgemm_fusion_register &brg_reg) override;

    unary_elementwise_op_impl_t(graph_tensor_ptr v, const std::string &op_name);
    unary_elementwise_op_impl_t(const std::string &op_name,
            const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs);
    vectorized_info_t &get_vx_info() { return vx_info_; }

    virtual expr compute_element(expr in) = 0;

    sc_dims get_bwise_fuse_shrink_dims() override;

private:
    vectorized_info_t vx_info_;
};

class relu_op_t : public unary_elementwise_op_impl_t {
public:
    DECLARE_COMPUTE_ELEMENT();
    relu_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
        : unary_elementwise_op_impl_t("relu", ins, outs, attrs) {
        alg_kind_ = brgemm::eltwise_relu;
    }
    relu_op_t(const std::vector<graph_tensor_ptr> &ins, const any_map_t &attrs);
    relu_op_t(graph_tensor_ptr v)
        : unary_elementwise_op_impl_t(std::move(v), "relu") {
        alg_kind_ = brgemm::eltwise_relu;
    };
};

class leaky_relu_op_t : public unary_elementwise_op_impl_t {
public:
    DECLARE_COMPUTE_ELEMENT();
    leaky_relu_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
        : unary_elementwise_op_impl_t("leaky_relu", ins, outs, attrs) {
        COMPILE_ASSERT(attrs.has_key("alpha"), "Cannot find attr `alpha`");
        alpha_ = attrs.get<float>("alpha");
    }
    leaky_relu_op_t(
            const std::vector<graph_tensor_ptr> &ins, const any_map_t &attrs);
    leaky_relu_op_t(graph_tensor_ptr v, float alpha)
        : unary_elementwise_op_impl_t(std::move(v), "leaky_relu")
        , alpha_(alpha) {};

private: // coefficient of leakage
    float alpha_;
};

class select_one_op_t : public unary_elementwise_op_impl_t {
public:
    DECLARE_COMPUTE_ELEMENT();
    select_one_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
        : unary_elementwise_op_impl_t("select_one", ins, outs, attrs) {}
    select_one_op_t(
            const std::vector<graph_tensor_ptr> &ins, const any_map_t &attrs);
    select_one_op_t(graph_tensor_ptr v)
        : unary_elementwise_op_impl_t(std::move(v), "select_one") {};
};

class round_op_t : public unary_elementwise_op_impl_t {
public:
    DECLARE_COMPUTE_ELEMENT();

    round_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
        : unary_elementwise_op_impl_t("round", ins, outs, attrs) {
        alg_kind_ = brgemm::eltwise_round;
    }
    round_op_t(
            const std::vector<graph_tensor_ptr> &ins, const any_map_t &attrs);
    round_op_t(graph_tensor_ptr v)
        : unary_elementwise_op_impl_t(std::move(v), "round") {
        alg_kind_ = brgemm::eltwise_round;
    };
};

class sigmoid_op_t : public unary_elementwise_op_impl_t {
public:
    DECLARE_COMPUTE_ELEMENT();

    sigmoid_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
        : unary_elementwise_op_impl_t("sigmoid", ins, outs, attrs) {
        alg_kind_ = brgemm::eltwise_logsigmoid;
    }
    sigmoid_op_t(graph_tensor_ptr v)
        : unary_elementwise_op_impl_t(std::move(v), "sigmoid") {
        alg_kind_ = brgemm::eltwise_logsigmoid;
    };
};

class exp_op_t : public unary_elementwise_op_impl_t {
public:
    DECLARE_COMPUTE_ELEMENT();
    exp_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
        : unary_elementwise_op_impl_t("exp", ins, outs, attrs) {
        alg_kind_ = brgemm::eltwise_exp;
    }
    exp_op_t(graph_tensor_ptr v)
        : unary_elementwise_op_impl_t(std::move(v), "exp") {
        alg_kind_ = brgemm::eltwise_exp;
    };
};

class tanh_op_t : public unary_elementwise_op_impl_t {
public:
    DECLARE_COMPUTE_ELEMENT();
    tanh_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
        : unary_elementwise_op_impl_t("tanh", ins, outs, attrs) {
        alg_kind_ = brgemm::eltwise_tanh;
    }
    tanh_op_t(graph_tensor_ptr v)
        : unary_elementwise_op_impl_t(std::move(v), "tanh") {
        alg_kind_ = brgemm::eltwise_tanh;
    };
};

class erf_op_t : public unary_elementwise_op_impl_t {
public:
    DECLARE_COMPUTE_ELEMENT();
    erf_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
        : unary_elementwise_op_impl_t("erf", ins, outs, attrs) {
        alg_kind_ = brgemm::eltwise_gelu_erf;
    }
    erf_op_t(graph_tensor_ptr v)
        : unary_elementwise_op_impl_t(std::move(v), "erf") {
        alg_kind_ = brgemm::eltwise_gelu_erf;
    };
};

// squared_root: sqrt(x)
class squared_root_op_t : public unary_elementwise_op_impl_t {
public:
    DECLARE_COMPUTE_ELEMENT();

    squared_root_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
        : unary_elementwise_op_impl_t("squared_root", ins, outs, attrs) {
        alg_kind_ = brgemm::eltwise_sqrt;
        reciprocal_ = attrs.get_or_else("reciprocal", false);
    };
    squared_root_op_t(graph_tensor_ptr v, bool reciprocal = false)
        : unary_elementwise_op_impl_t(std::move(v), "squared_root")
        , reciprocal_(reciprocal) {
        alg_kind_ = brgemm::eltwise_sqrt;
    }

private:
    // This flag decides return sqrt or rsqrt.
    bool reciprocal_;
};

class cast_op_t : public unary_elementwise_op_impl_t {
public:
    DECLARE_COMPUTE_ELEMENT();

    cast_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs);
    cast_op_t(graph_tensor_ptr v, sc_data_type_t out_dtype,
            bool saturated = false);

private:
    sc_data_type_t dtype_;
    bool saturated_;
};

class clamp_op_t : public unary_elementwise_op_impl_t {
public:
    DECLARE_COMPUTE_ELEMENT();

    clamp_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
        : unary_elementwise_op_impl_t("clamp", ins, outs, attrs) {
        alg_kind_ = brgemm::eltwise_clip;
    }
    clamp_op_t(graph_tensor_ptr v, float clamp_min = 0.0, float clamp_max = 1.0)
        : unary_elementwise_op_impl_t(std::move(v), "clamp") {
        alg_kind_ = brgemm::eltwise_clip;
        attrs_.set("clamp_min", clamp_min);
        attrs_.set("clamp_max", clamp_max);
    };
};

} // namespace sc
#endif
