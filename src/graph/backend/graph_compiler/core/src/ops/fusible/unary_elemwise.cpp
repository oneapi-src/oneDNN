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

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include "unary_elemwise.hpp"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/graph/brgemm_fusion.hpp>
#include <compiler/ir/graph/fusible_op.hpp>
#include <compiler/ir/graph/fusible_op_utils.hpp>
#include <runtime/dynamic_dispatch/ops/impl_type.hpp>
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
        COMPILE_ASSERT(outs[0]->details_.get_plain_dims()
                        == ins[0]->details_.get_plain_dims(),
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
    auto cur_cpu_flags = ctx->machine_.cpu_flags_;
    if (!cur_cpu_flags.fAVX512F && cur_cpu_flags.fAVX2) {
        // In avx2, bf16x16 can't cast to f32x16. Maximum lanes must be 8.
        if (op_name_ == "cast") {
            const sc_data_etype dst_etype
                    = dst[0]->tptr_->base_->dtype_.type_code_;
            auto size = utils::get_sizeof_etype(dst_etype);
            assert(size * 8 <= 256 && "Bad type for cast");
            const uint32_t avx2_max_lanes = 256 / (size * 8);
            vx_info_.lanes = std::min(avx2_max_lanes, vx_info_.lanes);
        }
    }
    auto func = [&](const std::vector<expr> &in,
                        std::vector<expr::lvalue_proxy_t> &out) -> stmt {
        return builder::make_assign_unattached(out[0], compute_element(in[0]));
    };
    // default use mask
    bool use_mask = attrs_.get_or_else(op_attr_key::use_padded_mask, true);
    if (get_owner_graph().is_dynamic()) {
        use_mask &= info_.cur_impl_ != impl_kind_t::no_padding;
    }
    compute_vectorized_op(ctx, get_owner_graph(), inputs, *dst[0], info_,
            vx_info_, mask_compute_func_t(func), mask_compute_func_t(func),
            attrs_, wkld, use_mask);
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

void infer_identical_binding_axis(fusible_op_t *cur, bound_axis_map &bdax_map) {
    auto known_axis_map = search_known_bound_axis(cur, bdax_map);
    if (!bdax_map.get(cur->get_outputs()[0]).empty()) return;
    bdax_map.get(cur->get_outputs()[0]) = known_axis_map[0];
    set_unknown_axis_binding(cur, known_axis_map, bdax_map);
}

void unary_elementwise_op_impl_t::infer_binding_axis(bound_axis_map &bdax_map) {
    infer_identical_binding_axis(this, bdax_map);
}

void pre_identical_binding_axis(fusible_op_t *cur, bound_axis_map &bdax_map) {
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
    pre_identical_binding_axis(this, bdax_map);
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

static expr compute_sigmoid(expr in) {
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

expr sigmoid_op_t::compute_element(expr in) {
    return compute_sigmoid(in);
}

expr exp_op_t::compute_element(expr in) {
    auto out = builder::make_exp(in);
    out->attr().set(
            "overflow_check", attrs_.get_or_else("overflow_check", true));
    return out;
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
    DECL_VEC_VAR(f_fin, f32);

    bool is_bf16 = in->dtype_.is_etype(sc_data_etype::BF16);
    if (is_bf16) {
        sc_data_type_t fp32ty = sc_data_type_t::f32(lanes);
        in = builder::make_cast(fp32ty, in);
    }
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
    bld->push_assign(f_fin,
            builder::make_reinterpret(
                    builder::make_int_xor(builder::make_reinterpret(f_out,
                                                  sc_data_type_t::u32(lanes)),
                            sign),
                    sc_data_type_t::f32(lanes)));

    if (is_bf16) {
        sc_data_type_t bf16ty = sc_data_type_t::bf16(lanes);
        return builder::make_cast(bf16ty, f_fin);
    } else {
        return f_fin;
    }

#undef DECL_VEC_CONSTANT
#undef DECL_VEC_VAR
#undef DECL_CONSTANT
#undef DECL_VAR
}

expr erf_op_t::compute_element(expr in) {
    auto lanes = in->dtype_.lanes_;
    bool is_bf16 = in->dtype_.is_etype(sc_data_etype::BF16);
    if (is_bf16) {
        return builder::make_cast(in->dtype_,
                builder::make_erf(builder::make_cast(
                        sc_data_type_t::f32(in->dtype_.lanes_), in)));
    }
    return builder::make_erf(in);
}

expr square_op_t::compute_element(expr in) {
    return in * in;
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
    float clamp_min = attrs_.get<float>("min");
    float clamp_max = attrs_.get<float>("max");
    return builder::make_max(
            builder::make_min(in, make_expr<constant_node>(clamp_max, dtype)),
            make_expr<constant_node>(clamp_min, dtype));
}

#define DEFINE_AND_ASSERT_DTYPE(op) \
    auto dtype = in->dtype_; \
    COMPILE_ASSERT(dtype.type_code_ == sc_data_etype::F32 \
                    || dtype.type_code_ == sc_data_etype::BF16, \
            (op) << "_op_t currently only supports fp32/bf16");

#define DEFINE_ALPHA_BASED_ON_DTYPE(op) \
    DEFINE_AND_ASSERT_DTYPE(op) \
    auto f_alpha = make_expr<constant_node>(alpha_, dtype);

#define DEFINE_ALPHA_AND_BETA_BASED_ON_DTYPE(op) \
    DEFINE_ALPHA_BASED_ON_DTYPE(op) \
    auto f_beta = make_expr<constant_node>(beta_, dtype);

expr reciprocal_op_t::compute_element(expr in) {
    // TODO(xxx): return approximate reciprocal when fast math enabled
    DEFINE_AND_ASSERT_DTYPE("reciprocal");
    auto f_one = make_expr<constant_node>((float)1.f, dtype);
    return builder::make_div(f_one, in);
}

expr abs_op_t::compute_element(expr in) {
    return builder::make_abs(in);
}

expr elu_op_t::compute_element(expr in) {
    DEFINE_ALPHA_BASED_ON_DTYPE("elu");
    auto f_one = make_expr<constant_node>(1.f, dtype);
    auto f_zero = make_expr<constant_node>(0.f, dtype);
    auto f_tail = (builder::make_exp(in) - f_one) * f_alpha;
    return builder::make_select(in > f_zero, in, f_tail);
}

expr hardswish_op_t::compute_element(expr in) {
    DEFINE_ALPHA_AND_BETA_BASED_ON_DTYPE("hardswish");
    auto f_one = make_expr<constant_node>(1.f, dtype);
    auto f_zero = make_expr<constant_node>(0.f, dtype);
    return in
            * builder::make_max(f_zero,
                    builder::make_min(
                            f_one, builder::make_fmadd(in, f_alpha, f_beta)));
}

// todo: find approximately faster algorithm(onednn impl).
expr log_op_t::compute_element(expr in) {
    return builder::make_log(in);
}

// mish(x) = x * ((e^x + 1)^2 - 1)/((e^x + 1)^2 + 1).
expr mish_op_t::compute_element(expr in) {
    DEFINE_AND_ASSERT_DTYPE("mish");
    auto f_one = make_expr<constant_node>(1.f, dtype);
    auto f_temp = builder::make_exp(in) + f_one;
    f_temp = f_temp * f_temp;
    return in * ((f_temp - f_one) / (f_temp + f_one));
}

expr soft_plus_op_t::compute_element(expr in) {
    DEFINE_AND_ASSERT_DTYPE("soft_plus");
    auto f_one = make_expr<constant_node>(1.f, dtype);
    auto f_beta_r = make_expr<constant_node>(beta_, dtype);
    return f_one / f_beta_r
            * builder::make_log(builder::make_exp(f_beta_r * in) + f_one);
}

expr swish_op_t::compute_element(expr in) {
    DEFINE_ALPHA_BASED_ON_DTYPE("swish");
    auto bld = builder::get_current_builder();
    auto f_sigmoid_in = builder::make_var(
            dtype, "f_sigmoid_in" + fusion_create_var_idx());
    bld->push_var_tensor_def(f_sigmoid_in, linkage::local, f_alpha * in);
    return in * compute_sigmoid(f_sigmoid_in);
}

expr hardsigmoid_op_t::compute_element(expr in) {
    DEFINE_ALPHA_AND_BETA_BASED_ON_DTYPE("hardsigmoid");
    auto f_one = make_expr<constant_node>(1.f, dtype);
    auto f_zero = make_expr<constant_node>(0.f, dtype);
    return builder::make_max(f_zero,
            builder::make_min(f_one, builder::make_fmadd(in, f_alpha, f_beta)));
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
OP_REGISTER(abs_op_t, abs)
OP_REGISTER(elu_op_t, elu)
OP_REGISTER(hardswish_op_t, hardswish)
OP_REGISTER(log_op_t, log)
OP_REGISTER(mish_op_t, mish)
OP_REGISTER(soft_plus_op_t, soft_plus)
OP_REGISTER(square_op_t, square)
OP_REGISTER(swish_op_t, swish)
OP_REGISTER(hardsigmoid_op_t, hardsigmoid)

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
