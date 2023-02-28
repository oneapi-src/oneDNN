/*******************************************************************************
 * Copyright 2020-2023 Intel Corporation
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

#include <assert.h>

#include <memory>
#include <string>
#include <utility>
#include "unary_elemwise.hpp"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/graph/brgemm_fusion.hpp>
#include <compiler/ir/graph/fusible_op.hpp>
#include <compiler/ir/graph/fusible_op_utils.hpp>
#include <runtime/microkernel/cpu/brgemm_alg_kind.hpp>
#include <util/utils.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

unary_elementwise_op_impl_t::unary_elementwise_op_impl_t(
        graph_tensor_ptr v, const std::string &op_name)
    : unary_elementwise_op_impl_t(op_name, {std::move(v)}, {}, {}) {}

unary_elementwise_op_impl_t::unary_elementwise_op_impl_t(
        const std::string &op_name, const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs) {
    COMPILE_ASSERT(ins.size() == 1, "Wrong op input size.\n");
    op_name_ = op_name;
    info_.inputs_ = ins;
    attrs_ = attrs;
    if (outs.empty()) {
        info_.outputs_.emplace_back(
                std::make_shared<graph_tensor>(this, ins[0]->details_));
    } else {
        COMPILE_ASSERT(outs.size() == 1, "Wrong op output size.\n");
        COMPILE_ASSERT(outs[0]->details_.get_blocking_dims()
                        == ins[0]->details_.get_blocking_dims(),
                "Wrong op output shapes.\n");
        info_.outputs_ = outs;
    }
    info_.tensor_share_info_ = {{0, {0}}};
    attrs_ = attrs;
}

std::vector<std::pair<int, std::vector<tensor_inplace_info_t>>>
unary_elementwise_op_impl_t::get_inplace_map() {
    return {{0, {{0, inplace_kind::ZERO_OFFSET}}}};
}

void unary_elementwise_op_impl_t::compute_block(context_ptr ctx,
        const std::vector<tensor_slice *> &dst,
        const std::vector<const tensor_slice *> &inputs) {
    size_t wkld = compute_fusible_workload(ctx, dst, inputs);
    // set default vectorized information
    vx_info_.axis = dst[0]->get_shape().size() - 1;
    for (int64_t i = dst[0]->nslice_dims() - 1; i >= 0; --i) {
        auto &dim = dst.at(0)->get_shape().at(i);
        if (!dim.isa<constant>() || 1 != get_expr_as_int(dim)) {
            vx_info_.axis = i;
            break;
        }
    }
    vx_info_.lanes
            = vectorize_step(ctx, info_.inputs_[0]->details_.dtype_.type_code_);
    auto func = [&](const std::vector<expr> &in,
                        std::vector<expr::lvalue_proxy_t> &out) -> stmt {
        return builder::make_assign_unattached(out[0], compute_element(in[0]));
    };
    // Currenly only support for exp
    bool use_mask = op_name_ == "exp";
    compute_vectorized_op(get_owner_graph(), inputs, *dst[0], info_, vx_info_,
            mask_compute_func_t(func), mask_compute_func_t(func), attrs_, wkld,
            use_mask);
}

void unary_elementwise_op_impl_t::prepare_fusion_data(fdata_map &fdmap) {
    COMPILE_ASSERT(info_.outputs_.size() == 1, "Wrong op output size.\n");
    auto &in_detail0 = fdmap.get(info_.inputs_[0]);
    in_detail0.use_count_++;
}

static void infer_unary_slice_ranges(
        fusible_op_t *cur, fslice_map &fsmap, infer_status_map_t &stat_map) {
    COMPILE_ASSERT(cur->get_inputs().size() == 1, "unary op is expected");
    // search known ranges from any input of cur fusbile op
    slice_range_map known_ranges_map
            = search_known_slice_ranges(cur, fsmap, stat_map);
    if (known_ranges_map.empty()) return;
    // set outputs slice range
    fsmap.get(cur->get_outputs()[0]) = known_ranges_map[0];
}

void unary_elementwise_op_impl_t::infer_slice_ranges(
        fslice_map &fsmap, infer_status_map_t &stat_map) {
    infer_unary_slice_ranges(this, fsmap, stat_map);
}

static void pre_unary_slice_ranges(
        fusible_op_t *cur, fslice_map &fsmap, infer_status_map_t &stat_map) {
    auto &input = cur->get_inputs()[0];
    auto &out_ranges = fsmap.get(cur->get_outputs()[0]);
    if (out_ranges.empty()) {
        stat_map.append_ops_by_status(cur, infer_status_code::RETRY);
        return;
    }
    auto &in_ranges = fsmap.get(input);
    if (in_ranges.empty()) {
        in_ranges = out_ranges;
        if (stat_map.is_recursive_mode()) {
            input->producer_owner_->dyn_cast<fusible_op_t>()->pre_slice_ranges(
                    fsmap, stat_map);
        }
    }
}

void unary_elementwise_op_impl_t::pre_slice_ranges(
        fslice_map &fsmap, infer_status_map_t &stat_map) {
    pre_unary_slice_ranges(this, fsmap, stat_map);
}

void identical_infer_binding_axis(fusible_op_t *cur, bound_axis_map &bdax_map) {
    auto known_axis_map = search_known_bound_axis(cur, bdax_map);
    if (!bdax_map.get(cur->get_outputs()[0]).empty()) return;
    bdax_map.get(cur->get_outputs()[0]) = known_axis_map[0];
    set_unknown_axis_binding(cur, known_axis_map, bdax_map);
}

void unary_elementwise_op_impl_t::infer_binding_axis(bound_axis_map &bdax_map) {
    identical_infer_binding_axis(this, bdax_map);
}

void identical_pre_binding_axis(fusible_op_t *cur, bound_axis_map &bdax_map) {
    auto &outaxis = bdax_map.get(cur->get_outputs()[0]);
    COMPILE_ASSERT(!outaxis.empty(),
            "Unknown output axis found, could not pre bind axis")
    auto &input = cur->get_inputs()[0];
    auto &inpaxis = bdax_map.get(input);
    if (inpaxis.empty()) {
        inpaxis = outaxis;
        if (auto bd_op
                = input->producer_owner_
                          ->dyn_cast<op_traits::mixed_partition_acceptable>()) {
            bd_op->pre_binding_axis(bdax_map);
        }
    }
}

void unary_elementwise_op_impl_t::pre_binding_axis(bound_axis_map &bdax_map) {
    identical_pre_binding_axis(this, bdax_map);
}

shape_rl_vec unary_elementwise_op_impl_t::get_dynamic_shape_relations() const {
    shape_rl_vec ret;
    auto &in_dims = get_inputs()[0]->details_.get_plain_dims();
    auto &out_dims = get_outputs()[0]->details_.get_plain_dims();
    for (size_t i = 0; i < in_dims.size(); i++) {
        if (is_dynamic_dim(in_dims[i])) {
            ret.emplace_back(in_dims[i], out_dims[i]);
        }
    }
    return ret;
}

bool unary_elementwise_op_impl_t::register_brgemm_fusion(const context_ptr &ctx,
        const std::vector<tensor_slice *> &outputs,
        const std::vector<const tensor_slice *> &inputs,
        brgemm_fusion_register &brg_reg) {
    if (!fuse_in_brgemm_) { return false; }
    return brg_reg.register_op_infos(
            shared_from_this(), outputs[0]->get_tensor_ptr());
}

sc_dims unary_elementwise_op_impl_t::get_bwise_fuse_shrink_dims() {
    auto output_dims = info_.outputs_[0]->details_.get_blocking_dims();
    int offset = op_traits::batchwise_shrinkable_t::get_shrinkable_offset(
            info_.outputs_[0]);
    return {output_dims.begin(), output_dims.begin() + offset};
}

expr relu_op_t::compute_element(expr in) {
    return builder::make_max(
            in, make_expr<constant_node>((int64_t)0, in->dtype_));
}

expr leaky_relu_op_t::compute_element(expr in) {
    expr alpha = make_expr<constant_node>((float)alpha_, in->dtype_);
    if (in->dtype_.type_code_ == sc_data_etype::BF16) {
        alpha = builder::make_cast(
                sc_data_type_t(sc_data_etype::BF16, in->dtype_.lanes_), alpha);
    }
    return builder::make_select(
            in > make_expr<constant_node>((int64_t)0, in->dtype_), in,
            alpha * in);
}

expr select_one_op_t::compute_element(expr in) {
    return builder::make_select(
            in > make_expr<constant_node>((float)0.0f, in->dtype_),
            make_expr<constant_node>((float)1.0f, in->dtype_),
            make_expr<constant_node>((float)0.0f, in->dtype_));
}

expr round_op_t::compute_element(expr in) {
    return builder::make_round(in);
}

expr sigmoid_op_t::compute_element(expr in) {
    auto bld = builder::get_current_builder();
    // constants
    auto lanes = in->dtype_.lanes_;
    bool is_bf16 = in->dtype_.is_etype(sc_data_etype::BF16);
    if (is_bf16) {
        sc_data_type_t fp32ty = sc_data_type_t::f32(lanes);
        in = builder::make_cast(fp32ty, in);
    }
    expr f_one = make_expr<constant_node>(1.0f, sc_data_type_t::f32(lanes));
    expr sign_mask = make_expr<constant_node>(
            0x80000000UL, sc_data_type_t::u32(lanes));
    // temp vars
    auto f_neg_x = builder::make_var(
            sc_data_type_t::f32(lanes), "f_neg_x" + fusion_create_var_idx());
    bld->push_var_tensor_def(f_neg_x);

    auto f_exp_neg_x = builder::make_var(sc_data_type_t::f32(lanes),
            "f_exp_neg_x" + fusion_create_var_idx());
    bld->push_var_tensor_def(f_exp_neg_x);

    // make negative x
    bld->push_assign(f_neg_x,
            builder::make_reinterpret(
                    builder::make_int_xor(builder::make_reinterpret(in,
                                                  sc_data_type_t::u32(lanes)),
                            sign_mask),
                    sc_data_type_t::f32(lanes)));

    // out = 1 / ( 1 + exp(-x) )
    bld->push_assign(f_exp_neg_x, builder::make_exp(f_neg_x));

    if (is_bf16) {
        sc_data_type_t bf16ty = sc_data_type_t::bf16(lanes);
        return builder::make_cast(
                bf16ty, builder::make_div(f_one, f_one + f_exp_neg_x));
    } else {
        return builder::make_div(f_one, f_one + f_exp_neg_x);
    }
}

expr exp_op_t::compute_element(expr in) {
    return builder::make_exp(in);
}

expr tanh_op_t::compute_element(expr in) {
    auto lanes = in->dtype_.lanes_;
#define DECL_VEC_CONSTANT(name, dtype, value) \
    expr name = make_expr<constant_node>(value, sc_data_type_t::dtype(lanes));

// clang-format off
// NOLINTNEXTLINE
#define DECL_VEC_VAR(name, dtype) auto name = builder::make_var( \
            sc_data_type_t::dtype(lanes), #name + fusion_create_var_idx()); \
    builder::get_current_builder()->push_var_tensor_def(name);
// clang-format on
#define DECL_CONSTANT(name, dtype, value) \
    expr name = make_expr<constant_node>(value, datatypes::dtype);
// clang-format off
// NOLINTNEXTLINE
#define DECL_VAR(name, dtype) auto name = builder::make_var( \
            datatypes::dtype, #name + fusion_create_var_idx()); \
    builder::get_current_builder()->push_var_tensor_def(name);
    // clang-format on

    auto bld = builder::get_current_builder();
    DECL_VEC_CONSTANT(uint_saturate_ubound, u32, 0x41b00000UL);
    DECL_VEC_CONSTANT(positive_mask, u32, 0x7fffffffUL);
    DECL_VEC_CONSTANT(sign_mask, u32, 0x80000000UL);
    DECL_VEC_CONSTANT(f_one, f32, 1.0f);
    DECL_VEC_CONSTANT(f_two, f32, 2.0f);
    DECL_VEC_VAR(abs_a, u32);
    DECL_VEC_VAR(sign, u32);
    DECL_VEC_VAR(f_abs_a, f32);
    DECL_VEC_VAR(f_2a, f32);
    DECL_VEC_VAR(f_exp_2a, f32);
    DECL_VEC_VAR(f_tmp, f32);
    DECL_VEC_VAR(f_out, f32);

    bld->push_assign(abs_a,
            builder::make_int_and(
                    builder::make_reinterpret(in, sc_data_type_t::u32(lanes)),
                    positive_mask));
    bld->push_assign(f_abs_a,
            builder::make_reinterpret(abs_a, sc_data_type_t::f32(lanes)));
    bld->push_assign(sign,
            builder::make_int_and(
                    builder::make_reinterpret(in, sc_data_type_t::u32(lanes)),
                    sign_mask));
    bld->push_assign(f_2a, builder::make_mul(f_abs_a, f_two));
    bld->push_assign(f_exp_2a, builder::make_exp(f_2a));
    bld->push_assign(
            f_tmp, builder::make_div(f_exp_2a - f_one, f_exp_2a + f_one));
    bld->push_assign(f_out,
            builder::make_select(abs_a > uint_saturate_ubound, f_one, f_tmp));
    return builder::make_reinterpret(
            builder::make_int_xor(builder::make_reinterpret(
                                          f_out, sc_data_type_t::u32(lanes)),
                    sign),
            sc_data_type_t::f32(lanes));

#undef DECL_VEC_CONSTANT
#undef DECL_VEC_VAR
#undef DECL_CONSTANT
#undef DECL_VAR
}

expr erf_op_t::compute_element(expr in) {
    auto lanes = in->dtype_.lanes_;

    auto bld = builder::get_current_builder();
    expr const_a1 = make_expr<constant_node>(
            0.254829592f, sc_data_type_t::f32(lanes));
    expr const_a2 = make_expr<constant_node>(
            -0.284496736f, sc_data_type_t::f32(lanes));
    expr const_a3 = make_expr<constant_node>(
            1.421413741f, sc_data_type_t::f32(lanes));
    expr const_a4 = make_expr<constant_node>(
            -1.453152027f, sc_data_type_t::f32(lanes));
    expr const_a5 = make_expr<constant_node>(
            1.061405429f, sc_data_type_t::f32(lanes));
    expr ONE_f = make_expr<constant_node>(1.0f, sc_data_type_t::f32(lanes));
    expr ZERO_f = make_expr<constant_node>(0.0f, sc_data_type_t::f32(lanes));
    expr const_p
            = make_expr<constant_node>(0.3275911f, sc_data_type_t::f32(lanes));
    expr sign_mask = make_expr<constant_node>(
            0x80000000UL, sc_data_type_t::u32(lanes));

    auto temp = builder::make_var(
            sc_data_type_t::f32(lanes), "temp" + fusion_create_var_idx());
    auto Q = builder::make_var(
            sc_data_type_t::f32(lanes), "Q" + fusion_create_var_idx());
    auto t = builder::make_var(
            sc_data_type_t::f32(lanes), "t" + fusion_create_var_idx());
    auto result = builder::make_var(
            sc_data_type_t::f32(lanes), "result" + fusion_create_var_idx());
    auto sign = builder::make_var(
            sc_data_type_t::u32(lanes), "sign" + fusion_create_var_idx());
    bld->push_var_tensor_def(temp);
    bld->push_var_tensor_def(Q);
    bld->push_var_tensor_def(t);
    bld->push_var_tensor_def(result);
    bld->push_var_tensor_def(sign);

    bld->push_assign(sign,
            builder::make_int_and(
                    builder::make_reinterpret(in, sc_data_type_t::u32(lanes)),
                    sign_mask));
    bld->push_assign(temp, builder::make_abs(in));
    bld->push_assign(Q, ZERO_f - builder::make_exp(ZERO_f - in * in));
    bld->push_assign(t,
            builder::make_div(
                    ONE_f, builder::make_fmadd(const_p, temp, ONE_f)));
    bld->push_assign(temp, builder::make_mul(Q, t));
    bld->push_assign(result, const_a5);
    bld->push_assign(result, builder::make_fmadd(result, t, const_a4));
    bld->push_assign(result, builder::make_fmadd(result, t, const_a3));
    bld->push_assign(result, builder::make_fmadd(result, t, const_a2));
    bld->push_assign(result, builder::make_fmadd(result, t, const_a1));
    bld->push_assign(result, builder::make_fmadd(result, temp, ONE_f));

    return builder::make_reinterpret(
            builder::make_int_xor(sign,
                    builder::make_reinterpret(
                            result, sc_data_type_t::u32(lanes))),
            sc_data_type_t::f32(lanes));
}

expr squared_root_op_t::compute_element(expr in) {
    if (reciprocal_) { return builder::make_rsqrt(in); }
    return builder::make_sqrt(in);
}

std::vector<std::pair<int, std::vector<tensor_inplace_info_t>>>
cast_op_t::get_inplace_map() {
    // if out size == in size, we can do inplace. Otherwise, we can't
    COMPILE_ASSERT(info_.outputs_.size() == 1 && info_.inputs_.size() == 1,
            "bad number of in/outs for cast op");
    if (utils::get_sizeof_type(info_.outputs_[0]->details_.dtype_)
            == utils::get_sizeof_type(info_.inputs_[0]->details_.dtype_)) {
        return unary_elementwise_op_impl_t::get_inplace_map();
    }
    return {};
}

cast_op_t::cast_op_t(const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
    : unary_elementwise_op_impl_t("cast", ins, outs, attrs) {
    dtype_ = attrs.get<sc_data_type_t>("dtype");
    saturated_ = attrs.get_or_else("saturated", false);
    info_.outputs_[0]->details_.dtype_ = dtype_;
    info_.tensor_share_info_.clear();
    alg_kind_ = brgemm::out_dtype;
}

cast_op_t::cast_op_t(
        graph_tensor_ptr v, sc_data_type_t out_dtype, bool saturated)
    : unary_elementwise_op_impl_t(std::move(v), "cast")
    , dtype_(out_dtype)
    , saturated_(saturated) {
    info_.outputs_[0]->details_.dtype_ = out_dtype;
    info_.tensor_share_info_.clear();
}

expr cast_op_t::compute_element(expr in) {
    sc_data_type_t vectorize_out_dtype = dtype_;
    vectorize_out_dtype.lanes_ = in->dtype_.lanes_;
    return saturated_ ? builder::make_saturated_cast(in, vectorize_out_dtype)
                      : builder::make_cast(vectorize_out_dtype, in);
}

expr clamp_op_t::compute_element(expr in) {
    auto dtype = in->dtype_;
    COMPILE_ASSERT(dtype.type_code_ == sc_data_etype::F32,
            "clamp_op_t currently only supports fp32");
    float clamp_min = attrs_.get<float>("clamp_min");
    float clamp_max = attrs_.get<float>("clamp_max");
    return builder::make_max(
            builder::make_min(in, make_expr<constant_node>(clamp_max, dtype)),
            make_expr<constant_node>(clamp_min, dtype));
}

expr reciprocal_op_t::compute_element(expr in) {
    // TODO(xxx): return approximate reciprocal when fast math enabled
    auto dtype = in->dtype_;
    COMPILE_ASSERT(dtype.type_code_ == sc_data_etype::F32
                    || dtype.type_code_ == sc_data_etype::BF16,
            "reciprocal_op_t currently only supports fp32/bf16");
    auto f_one = make_expr<constant_node>((float)1.f, dtype);
    return builder::make_div(f_one, in);
}

OP_REGISTER(sigmoid_op_t, sigmoid)
OP_REGISTER(exp_op_t, exp)
OP_REGISTER(erf_op_t, erf)
OP_REGISTER(tanh_op_t, tanh)
OP_REGISTER(relu_op_t, relu)
OP_REGISTER(leaky_relu_op_t, leaky_relu)
OP_REGISTER(select_one_op_t, select_one)
OP_REGISTER(round_op_t, round)
OP_REGISTER(squared_root_op_t, squared_root)
OP_REGISTER(cast_op_t, cast)
OP_REGISTER(clamp_op_t, clamp)
OP_REGISTER(reciprocal_op_t, reciprocal)

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
