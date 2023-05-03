/*******************************************************************************
* Copyright 2017-2023 Intel Corporation
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

#include "oneapi/dnnl/dnnl.h"

#include "c_types_map.hpp"
#include "primitive_attr.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::status;
using namespace dnnl::impl::utils;

namespace dnnl {
namespace impl {

const primitive_attr_t &default_attr() {
    static const primitive_attr_t default_attr_instance;
    return default_attr_instance;
}

void scales_t::set_single_scale(float scale) {
    count_ = 1;
    mask_ = 0;
    scales_ = scales_buf_;
    if (is_runtime_value(scale)) {
        scales_[0] = scale;
    } else {
        utils::array_set(scales_, scale, scales_buf_size);
    }
}

status_t scales_t::set(dim_t count, int mask, const float *scales) {
    cleanup();

    count_ = count;
    mask_ = mask;

    if (is_runtime_value(*scales)) {
        scales_ = scales_buf_;
        scales_[0] = *scales;
    } else if (count_ == 1) {
        set_single_scale(scales[0]);
    } else {
        scales_ = (float *)impl::malloc(count_ * sizeof(*scales_), 64);
        if (scales_ == nullptr) return status::out_of_memory;

        for (dim_t c = 0; c < count_; ++c)
            scales_[c] = scales[c];
    }

    return status::success;
}

status_t zero_points_t::get(int arg, int *mask) const {
    if (mask) *mask = get_mask(arg);
    return status::success;
}

int zero_points_t::get(int arg) const {
    return get_mask(arg);
}

status_t zero_points_t::set(int arg, int mask) {
    const bool supported_arg
            = utils::one_of(arg, DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST);
    if (!supported_arg) return status::unimplemented;

    switch (arg) {
        case DNNL_ARG_SRC:
            is_set_src = true;
            mask_src = mask;
            break;
        case DNNL_ARG_WEIGHTS:
            is_set_wei = true;
            mask_wei = mask;
            break;
        case DNNL_ARG_DST:
            is_set_dst = true;
            mask_dst = mask;
            break;
    }
    return status::success;
}

} // namespace impl
} // namespace dnnl

bool primitive_attr_t::has_default_values(dnnl_primitive_attr::skip_mask_t mask,
        dnnl::impl::data_type_t dst_dt) const {
    using smask_t = skip_mask_t;
    // prepare mask for runtime-parameters check
    smask_t defined_mask = smask_t::none;
    if ((mask & smask_t::oscale_runtime) == smask_t::oscale_runtime)
        defined_mask |= smask_t::oscale;
    if ((mask & smask_t::scales_runtime) == smask_t::scales_runtime)
        defined_mask |= smask_t::scales;
    if ((mask & smask_t::zero_points_runtime) == smask_t::zero_points_runtime)
        defined_mask |= smask_t::zero_points;
    bool ok = true;

#define CHECK_ARG(x) ok = ok && (x)
#define CHECK_MASK(mask_name, mask_field) \
    CHECK_ARG(IMPLICATION( \
            (bool)(~mask & (mask_name)), (mask_field).has_default_values()))
    CHECK_MASK(smask_t::oscale_runtime, output_scales_);
    CHECK_MASK(smask_t::scales, scales_);
    CHECK_MASK(smask_t::zero_points, zero_points_);
    CHECK_MASK(smask_t::post_ops, post_ops_);
    CHECK_MASK(smask_t::rnn_data_qparams, rnn_data_qparams_);
    CHECK_MASK(smask_t::rnn_weights_qparams, rnn_weights_qparams_);
    CHECK_MASK(smask_t::rnn_weights_projection_qparams,
            rnn_weights_projection_qparams_);
    CHECK_ARG(IMPLICATION((bool)(~mask & smask_t::sum_dt),
            post_ops_.sum_with_default_dt(dst_dt)));
    bool gpu_attr_ok = IMPLICATION((bool)(~mask & smask_t::gpu_attr),
            !gpu_attr_ || gpu_attr_->has_default_values());
    CHECK_ARG(gpu_attr_ok);
    CHECK_ARG(this->defined(defined_mask));
    return ok;
#undef CHECK_MASK
#undef CHECK_ARG
}

bool primitive_attr_t::defined(dnnl_primitive_attr::skip_mask_t mask) const {
    using smask_t = skip_mask_t;
    bool ok = true;
#define CHECK_ARG(x) ok = ok && (x)
#define CHECK_MASK(mask_name, mask_field) \
    CHECK_ARG(IMPLICATION((bool)(~mask & (mask_name)), (mask_field).defined()))
    CHECK_MASK(smask_t::oscale, output_scales_);
    CHECK_MASK(smask_t::scales, scales_);
    CHECK_MASK(smask_t::zero_points, zero_points_);
    CHECK_MASK(smask_t::post_ops, post_ops_);
    CHECK_MASK(smask_t::rnn_data_qparams, rnn_data_qparams_);
    CHECK_MASK(smask_t::rnn_weights_qparams, rnn_weights_qparams_);
    CHECK_MASK(smask_t::rnn_weights_projection_qparams,
            rnn_weights_projection_qparams_);
    return ok;
#undef CHECK_MASK
#undef CHECK_ARG
}

status_t post_ops_t::append_sum(
        float scale, int32_t zero_point, data_type_t dt) {
    entry_.emplace_back();
    auto &e = entry_.back();
    e.kind = primitive_kind::sum;
    e.sum.scale = scale;
    e.sum.zero_point = zero_point;
    e.sum.dt = dt;
    return success;
}

status_t post_ops_t::append_eltwise(
        float scale, alg_kind_t alg, float alpha, float beta) {
    if (!math::is_eltwise_ok(data_type::f32, alg, alpha, beta))
        return invalid_arguments;

    entry_.emplace_back();
    auto &e = entry_.back();
    e.kind = primitive_kind::eltwise;
    e.eltwise.scale = scale;
    e.eltwise.alg = alg;
    e.eltwise.alpha = alpha;
    e.eltwise.beta = beta;
    return success;
}

status_t post_ops_t::append_dw(data_type_t wei_dt, data_type_t bias_dt,
        data_type_t dst_dt, dim_t kernel_size, dim_t stride_size,
        dim_t padding_l_size) {
    if (len() == post_ops_limit) return out_of_memory;
    bool ok = wei_dt != data_type::undef && dst_dt != data_type::undef;
    if (!ok) return invalid_arguments;

    ok = ok && kernel_size > 0 && stride_size > 0;
    if (!ok) return invalid_arguments;

    // Avoiding cases when kernel in pad area
    ok = ok && (padding_l_size + 1) <= kernel_size;
    if (!ok) return invalid_arguments;

    entry_.emplace_back();
    auto &e = entry_.back();
    e.kind = primitive_kind::convolution;
    auto &d = e.depthwise_conv;
    d.kernel = kernel_size;
    d.stride = stride_size;
    d.padding = padding_l_size;
    d.wei_dt = wei_dt;
    d.bias_dt = bias_dt;
    d.dst_dt = dst_dt;

    return success;
}

status_t post_ops_t::validate_binary(
        alg_kind_t alg, const memory_desc_t *user_src1_desc) const {

    if (len() == post_ops_limit) return out_of_memory;
    using namespace alg_kind;
    bool alg_ok = one_of(alg, binary_add, binary_mul, binary_max, binary_min,
            binary_div, binary_sub, binary_ge, binary_gt, binary_le, binary_lt,
            binary_eq, binary_ne);
    if (!alg_ok) return invalid_arguments;
    if (!memory_desc_sanity_check(*user_src1_desc)) return invalid_arguments;

    // Additional check to restrict run-time dimension usage until supported.
    for (int d = 0; d < user_src1_desc->ndims; ++d) {
        if (user_src1_desc->dims[d] == DNNL_RUNTIME_DIM_VAL)
            return invalid_arguments;
    }

    return success;
}

status_t post_ops_t::append_binary(
        alg_kind_t alg, const memory_desc_t *user_src1_desc) {
    auto status = validate_binary(alg, user_src1_desc);
    if (status != success) return status;

    entry_.emplace_back();
    auto &e = entry_.back();
    e.kind = primitive_kind::binary;
    e.binary.alg = alg;
    e.binary.user_src1_desc = *user_src1_desc;
    e.binary.src1_desc = *user_src1_desc;
    return success;
}

status_t post_ops_t::prepend_binary(
        alg_kind_t alg, const memory_desc_t *user_src1_desc) {
    auto status = validate_binary(alg, user_src1_desc);
    if (status != success) return status;

    entry_.emplace(entry_.begin());
    auto &e = entry_[0];
    e.kind = primitive_kind::binary;
    e.binary.alg = alg;
    e.binary.user_src1_desc = *user_src1_desc;
    e.binary.src1_desc = *user_src1_desc;
    return success;
}

status_t post_ops_t::append_prelu(int mask) {
    if (len() == post_ops_limit) return out_of_memory;

    auto it_entry = entry_.emplace(entry_.end());
    it_entry->kind = primitive_kind::prelu;
    it_entry->prelu.mask = mask;

    return success;
}

bool post_ops_t::defined() const {
    for (int idx = 0; idx < len(); ++idx) {
        auto kind = entry_[idx].kind;
        if (kind == primitive_kind::sum) {
            if (is_runtime_value(entry_[idx].sum.scale)) return false;
        } else if (kind == primitive_kind::eltwise) {
            const auto &e = entry_[idx].eltwise;
            if (is_runtime_value(e.scale) || is_runtime_value(e.alpha)
                    || is_runtime_value(e.beta))
                return false;
        } else if (utils::one_of(kind, primitive_kind::binary,
                           primitive_kind::prelu,
                           primitive_kind::convolution)) {
            // binary is always defined
        } else {
            assert(!"unreachable");
        }
    }
    return true;
}

status_t post_ops_t::set_default_formats(const memory_desc_t *dst_md) {
    for (int idx = 0; idx < len(); ++idx) {
        if (!contain(primitive_kind::binary, idx)) continue;

        auto &src1_md = entry_[idx].binary.src1_desc;
        const memory_desc_wrapper src1_mdw(src1_md);
        if (!src1_mdw.format_any()) continue;

        const memory_desc_wrapper dst_mdw(dst_md);
        assert(!dst_mdw.format_any());

        // 1D tensors should be plain abx.
        if (src1_mdw.count_non_unit_dims(1))
            CHECK(memory_desc_init_by_strides(src1_md, nullptr));
        else
            CHECK(memory_desc_init_by_blocking_desc(
                    src1_md, dst_mdw.blocking_desc()));
    }

    return status::success;
}

bool post_ops_t::check_sum_consistent_dt(const data_type_t dst_dt,
        const bool diverse_sum_dt_is_supported) const {
    int sum_ind = find(dnnl::impl::primitive_kind::sum);
    if (sum_ind == -1) return true;
    const auto sum_dt = entry_[sum_ind].sum.dt;

    // sum dt and dst dt must have the same size
    const bool compatible_dt_size = IMPLICATION(
            !utils::one_of(dnnl_data_type_undef, sum_dt, dst_dt),
            types::data_type_size(dst_dt) == types::data_type_size(sum_dt));
    if (!compatible_dt_size) return false;
    if (diverse_sum_dt_is_supported) return true;

    bool ok = true;
    while ((sum_ind = find(dnnl::impl::primitive_kind::sum, sum_ind + 1)) != -1)
        ok = ok && entry_[sum_ind].sum.dt == sum_dt;
    return ok;
}

bool post_ops_t::check_sum_consistent_quantization(
        const data_type_t dst_dt, const bool is_int8) const {
    using namespace data_type;
    using namespace primitive_kind;
    bool ok = true;
    int sum_ind = -1;
    while ((sum_ind = find(sum, sum_ind + 1)) != -1) {
        const auto &sum_e = entry_[sum_ind].sum;
        // validate interface requirements
        ok = ok && IMPLICATION(!is_int8, sum_e.zero_point == 0)
                && IMPLICATION(sum_e.zero_point != 0,
                        one_of(get_sum_dt(dst_dt, sum_ind), s8, u8, s32));
    }
    return ok;
}

bool post_ops_t::check_sum_consistency(const data_type_t dst_dt,
        const bool is_int8, const bool diverse_sum_dt_is_supported) const {

    return check_sum_consistent_dt(dst_dt, diverse_sum_dt_is_supported)
            && check_sum_consistent_quantization(dst_dt, is_int8);
}

status_t primitive_attr_t::set_fpmath_mode(fpmath_mode_t fpmath_mode) {
    auto st = check_fpmath_mode(fpmath_mode);
    if (st == success) fpmath_mode_ = fpmath_mode;
    return st;
}

status_t primitive_attr_t::set_scratchpad_mode(
        scratchpad_mode_t scratchpad_mode) {
    const bool ok = one_of(
            scratchpad_mode, scratchpad_mode::library, scratchpad_mode::user);
    if (!ok) return invalid_arguments;

    scratchpad_mode_ = scratchpad_mode;
    return success;
}

status_t primitive_attr_t::set_post_ops(const post_ops_t &post_ops) {
    post_ops_.copy_from(post_ops);
    return status::success;
}

status_t primitive_attr_t::set_default_formats(const memory_desc_t *dst_md) {
    return post_ops_.set_default_formats(dst_md);
}

status_t primitive_attr_t::set_gpu_attr(const primitive_attr_item_t &gpu_attr) {
    gpu_attr_ = gpu_attr.clone();
    return status::success;
}

/* Public C API */

status_t dnnl_primitive_attr_create(primitive_attr_t **attr) {
    if (attr == nullptr) return invalid_arguments;

    return safe_ptr_assign(*attr, new dnnl_primitive_attr);
}

status_t dnnl_primitive_attr_clone(
        primitive_attr_t **attr, const primitive_attr_t *existing_attr) {
    if (any_null(attr, existing_attr)) return invalid_arguments;

    auto new_attr = utils::make_unique<primitive_attr_t>(*existing_attr);
    if (!new_attr->is_initialized()) return out_of_memory;

    return safe_ptr_assign(*attr, new_attr.release());
}

status_t dnnl_primitive_attr_destroy(primitive_attr_t *attr) {
    delete attr;

    return success;
}

status_t dnnl_primitive_attr_get_fpmath_mode(
        const primitive_attr_t *attr, fpmath_mode_t *mode) {
    if (any_null(attr, mode)) return invalid_arguments;
    *mode = attr->fpmath_mode_;
    return success;
}

status_t dnnl_primitive_attr_set_fpmath_mode(
        primitive_attr_t *attr, fpmath_mode_t mode) {
    if (any_null(attr)) return invalid_arguments;
    return attr->set_fpmath_mode(mode);
}

status_t dnnl_primitive_attr_get_scratchpad_mode(
        const primitive_attr_t *attr, scratchpad_mode_t *scratchpad_mode) {
    if (any_null(attr, scratchpad_mode)) return invalid_arguments;

    *scratchpad_mode = attr->scratchpad_mode_;

    return success;
}

status_t dnnl_primitive_attr_set_scratchpad_mode(
        primitive_attr_t *attr, scratchpad_mode_t scratchpad_mode) {
    if (any_null(attr)) return invalid_arguments;

    return attr->set_scratchpad_mode(scratchpad_mode);
}

status_t dnnl_primitive_attr_set_scales_mask(
        primitive_attr_t *attr, int arg, int mask) {
    bool ok = attr && mask >= 0 && arg >= 0
            && attr->output_scales_.has_default_values();
    if (!ok) return invalid_arguments;
    return attr->scales_.set(arg, mask);
}

status_t dnnl_primitive_attr_set_zero_points_mask(
        primitive_attr_t *attr, int arg, int mask) {
    bool ok = attr && mask >= 0;
    if (!ok) return invalid_arguments;

    return attr->zero_points_.set(arg, mask);
}

status_t dnnl_primitive_attr_get_post_ops(
        const primitive_attr_t *attr, const post_ops_t **post_ops) {
    if (any_null(attr, post_ops)) return invalid_arguments;

    *post_ops = &attr->post_ops_;
    return success;
}

status_t dnnl_primitive_attr_set_post_ops(
        primitive_attr_t *attr, const post_ops_t *post_ops) {
    if (any_null(attr, post_ops)) return invalid_arguments;

    return attr->set_post_ops(*post_ops);
}

status_t dnnl_post_ops_create(post_ops_t **post_ops) {
    if (post_ops == nullptr) return invalid_arguments;

    return safe_ptr_assign(*post_ops, new dnnl_post_ops);
}

status_t dnnl_post_ops_clone(
        post_ops_t **post_ops, const post_ops_t *existing_post_ops) {
    if (any_null(post_ops, existing_post_ops)) return invalid_arguments;

    auto new_post_ops = utils::make_unique<post_ops_t>(*existing_post_ops);
    if (!new_post_ops->is_initialized()) return out_of_memory;

    return safe_ptr_assign(*post_ops, new_post_ops.release());
}

status_t dnnl_post_ops_destroy(post_ops_t *post_ops) {
    delete post_ops;

    return success;
}

int dnnl_post_ops_len(const post_ops_t *post_ops) {
    if (post_ops) return post_ops->len();

    return 0;
}

primitive_kind_t dnnl_post_ops_get_kind(const post_ops_t *post_ops, int index) {
    bool ok = post_ops && 0 <= index && index < post_ops->len();
    if (!ok) return primitive_kind::undefined;

    return post_ops->entry_[index].kind;
}

status_t dnnl_post_ops_append_sum(
        post_ops_t *post_ops, float scale, int32_t zero_point, data_type_t dt) {
    if (post_ops == nullptr) return invalid_arguments;
    if (post_ops->len() >= post_ops_t::post_ops_limit) return out_of_memory;

    return post_ops->append_sum(scale, zero_point, dt);
}

namespace {
bool simple_get_params_check(
        const post_ops_t *post_ops, int index, primitive_kind_t kind) {
    bool ok = true && post_ops != nullptr && 0 <= index
            && index < post_ops->len() && post_ops->entry_[index].kind == kind;
    return ok;
}
} // namespace

status_t dnnl_post_ops_get_params_sum(const post_ops_t *post_ops, int index,
        float *scale, int32_t *zero_point, data_type_t *dt) {
    bool ok = true
            && simple_get_params_check(post_ops, index, primitive_kind::sum);
    if (!ok) return invalid_arguments;

    if (scale) *scale = post_ops->entry_[index].sum.scale;
    if (zero_point) *zero_point = post_ops->entry_[index].sum.zero_point;
    if (dt) *dt = post_ops->entry_[index].sum.dt;
    return success;
}

status_t dnnl_post_ops_append_eltwise(
        post_ops_t *post_ops, alg_kind_t kind, float alpha, float beta) {
    if (post_ops == nullptr) return invalid_arguments;
    if (post_ops->len() >= post_ops_t::post_ops_limit) return out_of_memory;

    return post_ops->append_eltwise(1.0f, kind, alpha, beta);
}

status_t dnnl_post_ops_get_params_eltwise(const post_ops_t *post_ops, int index,
        alg_kind_t *alg, float *alpha, float *beta) {
    bool ok = true
            && simple_get_params_check(post_ops, index, primitive_kind::eltwise)
            && !any_null(alpha, beta);
    if (!ok) return invalid_arguments;

    const auto &e = post_ops->entry_[index].eltwise;
    *alg = e.alg;
    *alpha = e.alpha;
    *beta = e.beta;

    return success;
}

status_t dnnl_post_ops_append_dw(post_ops_t *post_ops, data_type_t wei_dt,
        data_type_t bias_dt, data_type_t dst_dt, dim_t kernel_size,
        dim_t stride_size, dim_t padding_l_size) {
    if (post_ops == nullptr) return invalid_arguments;

    return post_ops->append_dw(
            wei_dt, bias_dt, dst_dt, kernel_size, stride_size, padding_l_size);
}

status_t dnnl_post_ops_get_params_dw(const post_ops_t *post_ops, int index,
        data_type_t *wei_dt, data_type_t *bias_dt, data_type_t *dst_dt,
        dim_t *kernel, dim_t *stride, dim_t *padding) {

    if (!simple_get_params_check(post_ops, index, primitive_kind::convolution))
        return invalid_arguments;

    const auto &d = post_ops->entry_[index].depthwise_conv;
    if (wei_dt) *wei_dt = d.wei_dt;
    if (bias_dt) *bias_dt = d.bias_dt;
    if (dst_dt) *dst_dt = d.dst_dt;
    if (kernel) *kernel = d.kernel;
    if (stride) *stride = d.stride;
    if (padding) *padding = d.padding;

    return success;
}

status_t dnnl_post_ops_append_binary(post_ops_t *post_ops, alg_kind_t alg_kind,
        const memory_desc_t *user_src1_desc) {
    if (post_ops == nullptr) return invalid_arguments;

    return post_ops->append_binary(alg_kind, user_src1_desc);
}

status_t dnnl_post_ops_get_params_binary(const post_ops_t *post_ops, int index,
        alg_kind_t *alg_kind, const memory_desc_t **user_src1_desc) {
    if (!simple_get_params_check(post_ops, index, primitive_kind::binary))
        return invalid_arguments;

    const auto &b = post_ops->entry_[index].binary;
    if (alg_kind) *alg_kind = b.alg;
    if (user_src1_desc) *user_src1_desc = &b.user_src1_desc;

    return success;
}

status_t dnnl_post_ops_append_prelu(post_ops_t *post_ops, int mask) {
    if (post_ops == nullptr) return invalid_arguments;

    return post_ops->append_prelu(mask);
}

status_t dnnl_post_ops_get_params_prelu(
        const post_ops_t *post_ops, int index, int *mask) {
    if (post_ops == nullptr || index >= post_ops->len())
        return invalid_arguments;

    const auto &prelu_entry = post_ops->entry_[index].prelu;
    if (mask) *mask = prelu_entry.mask;

    return success;
}

status_t dnnl_primitive_attr_set_rnn_data_qparams(
        primitive_attr_t *attr, const float scale, const float shift) {
    if (attr == nullptr) return invalid_arguments;

    return attr->rnn_data_qparams_.set(scale, shift);
}

status_t dnnl_primitive_attr_get_rnn_data_qparams(
        const primitive_attr_t *attr, float *scale, float *shift) {
    if (attr == nullptr) return invalid_arguments;

    const auto qparams = attr->rnn_data_qparams_;
    if (scale) *scale = qparams.scale_;
    if (shift) *shift = qparams.shift_;

    return success;
}

status_t dnnl_primitive_attr_set_rnn_weights_qparams(
        primitive_attr_t *attr, dim_t count, int mask, const float *scales) {
    bool ok = !any_null(attr, scales) && count > 0 && mask >= 0;
    if (!ok) return invalid_arguments;

    return attr->rnn_weights_qparams_.set(count, mask, scales);
}

status_t dnnl_primitive_attr_get_rnn_weights_qparams(
        const primitive_attr_t *attr, dim_t *count, int *mask,
        const float **scales) {
    if (attr == nullptr) return invalid_arguments;

    const auto &qparams = attr->rnn_weights_qparams_;
    if (count) *count = qparams.count_;
    if (mask) *mask = qparams.mask_;
    if (scales) *scales = qparams.scales_;

    return success;
}

status_t dnnl_primitive_attr_set_rnn_weights_projection_qparams(
        primitive_attr_t *attr, dim_t count, int mask, const float *scales) {
    bool ok = !any_null(attr, scales) && count > 0 && mask >= 0;
    if (!ok) return invalid_arguments;

    return attr->rnn_weights_projection_qparams_.set(count, mask, scales);
}

status_t dnnl_primitive_attr_get_rnn_weights_projection_qparams(
        const primitive_attr_t *attr, dim_t *count, int *mask,
        const float **scales) {
    if (attr == nullptr) return invalid_arguments;

    const auto &qparams = attr->rnn_weights_projection_qparams_;
    if (count) *count = qparams.count_;
    if (mask) *mask = qparams.mask_;
    if (scales) *scales = qparams.scales_;

    return success;
}

status_t DNNL_API dnnl_primitive_attr_set_rnn_tparams(
        dnnl_primitive_attr_t attr, bool mode, dim_t ngates,
        const float *scales, float cscale) {
    if (attr == nullptr) return invalid_arguments;

    return attr->rnn_tparams_.set(mode, ngates, scales, cscale);
}
