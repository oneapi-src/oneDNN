/*******************************************************************************
* Copyright 2017-2019 Intel Corporation
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

#include "dnnl.h"

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

status_t scales_t::set(dim_t count, int mask, const float *scales) {
    cleanup();

    count_ = count;
    mask_ = mask;

    if (is_runtime_value(*scales)) {
        scales_ = scales_buf_;
        scales_[0] = *scales;
    } else if (count_ == 1) {
        scales_ = scales_buf_;
        utils::array_set(scales_, scales[0], scales_buf_size);
    } else {
        scales_ = (float *)impl::malloc(count_ * sizeof(*scales_), 64);
        if (scales_ == nullptr) return status::out_of_memory;

        for (dim_t c = 0; c < count_; ++c)
            scales_[c] = scales[c];
    }

    return status::success;
}

status_t arg_scales_t::set(
        int arg, dim_t count, int mask, const float *scales) {
    if (!check_arg(arg)) return status::invalid_arguments;

    scales_[arg] = scales_t(count, mask, scales);
    return status::success;
}

status_t arg_scales_t::get(
        int arg, dim_t *count, int *mask, const float **scales) const {
    if (!check_arg(arg)) return status::invalid_arguments;
    const auto &s = get(arg);

    *count = s.count_;
    *mask = s.mask_;
    *scales = s.scales_;
    return status::success;
}

status_t zero_points_t::get(
        int arg, dim_t *count, int *mask, const int **zero_points) const {
    if (count) *count = 1;
    if (mask) *mask = 0;
    if (zero_points) *zero_points = get(arg);
    return status::success;
}

status_t zero_points_t::set(
        int arg, dim_t count, int mask, const int *zero_points) {
    if (zero_points == nullptr) return status::invalid_arguments;

    const bool supported_arg
            = utils::one_of(arg, DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST);
    const bool ok = count == 1 && mask == 0
            && IMPLICATION(!supported_arg, *zero_points == 0);
    if (!ok) return status::unimplemented;

    switch (arg) {
        case DNNL_ARG_SRC: zero_point_src = *zero_points; break;
        case DNNL_ARG_WEIGHTS: zero_point_wei = *zero_points; break;
        case DNNL_ARG_DST: zero_point_dst = *zero_points; break;
    }
    return status::success;
}

} // namespace impl
} // namespace dnnl

bool primitive_attr_t::has_default_values(
        dnnl_primitive_attr::skip_mask_t mask) const {
    // prepare mask for runtime-parameters check
    skip_mask_t defined_mask {};
    if ((mask & skip_mask_t::oscale_runtime) == skip_mask_t::oscale_runtime)
        defined_mask |= skip_mask_t::oscale;
    if ((mask & skip_mask_t::zero_points_runtime)
            == skip_mask_t::zero_points_runtime)
        defined_mask |= skip_mask_t::zero_points;

    return true
            && IMPLICATION((bool)(~mask & skip_mask_t::oscale),
                    output_scales_.has_default_values())
            && IMPLICATION((bool)(~mask & skip_mask_t::scales),
                    scales_.has_default_values())
            && IMPLICATION((bool)(~mask & skip_mask_t::zero_points),
                    zero_points_.has_default_values())
            && IMPLICATION((bool)(~mask & skip_mask_t::post_ops),
                    post_ops_.has_default_values())
            && IMPLICATION((bool)(~mask & skip_mask_t::rnn_data_qparams),
                    rnn_data_qparams_.has_default_values())
            && IMPLICATION((bool)(~mask & skip_mask_t::rnn_weights_qparams),
                    rnn_weights_qparams_.has_default_values())
            && this->defined(defined_mask);
}

bool primitive_attr_t::defined(dnnl_primitive_attr::skip_mask_t mask) const {
    return true
            && IMPLICATION((bool)(~mask & skip_mask_t::oscale),
                    output_scales_.defined())
            && IMPLICATION((bool)(~mask & skip_mask_t::zero_points),
                    zero_points_.defined())
            && IMPLICATION(
                    (bool)(~mask & skip_mask_t::post_ops), post_ops_.defined())
            && IMPLICATION((bool)(~mask & skip_mask_t::rnn_data_qparams),
                    rnn_data_qparams_.defined())
            && IMPLICATION((bool)(~mask & skip_mask_t::rnn_weights_qparams),
                    rnn_weights_qparams_.defined());
}

status_t post_ops_t::append_sum(float scale) {
    if (len_ == capacity) return out_of_memory;

    entry_[len_].kind = primitive_kind::sum;
    entry_[len_].sum.scale = scale;

    len_++;

    return success;
}

status_t post_ops_t::append_eltwise(
        float scale, alg_kind_t alg, float alpha, float beta) {
    using namespace dnnl::impl::alg_kind;
    bool known_alg = one_of(alg, eltwise_relu, eltwise_tanh, eltwise_elu,
            eltwise_square, eltwise_abs, eltwise_sqrt, eltwise_linear,
            eltwise_bounded_relu, eltwise_soft_relu, eltwise_logistic,
            eltwise_exp, eltwise_gelu, eltwise_swish, eltwise_log,
            eltwise_clip);
    if (!known_alg) return invalid_arguments;

    bool ok = true && IMPLICATION(alg == eltwise_bounded_relu, alpha >= 0)
            && IMPLICATION(alg == eltwise_clip, beta >= alpha);
    if (!ok) return invalid_arguments;

    if (len_ == capacity) return out_of_memory;

    entry_[len_].kind = primitive_kind::eltwise;
    entry_[len_].eltwise.scale = scale;
    entry_[len_].eltwise.alg = alg;
    entry_[len_].eltwise.alpha = alpha;
    entry_[len_].eltwise.beta = beta;

    len_++;

    return success;
}

bool post_ops_t::defined() const {
    for (int idx = 0; idx < len_; ++idx) {
        if (entry_[idx].kind == primitive_kind::sum) {
            if (is_runtime_value(entry_[idx].sum.scale)) return false;
        } else if (entry_[idx].kind == primitive_kind::eltwise) {
            const auto &e = entry_[idx].eltwise;
            if (is_runtime_value(e.scale) || is_runtime_value(e.alpha)
                    || is_runtime_value(e.beta))
                return false;
        } else {
            assert(!"unreachable");
        }
    }
    return true;
}

status_t primitive_attr_t::set_scratchpad_mode(
        scratchpad_mode_t scratchpad_mode) {
    using namespace dnnl::impl::scratchpad_mode;

    const bool ok = one_of(scratchpad_mode, library, user);
    if (!ok) return invalid_arguments;

    scratchpad_mode_ = scratchpad_mode;
    return success;
}

status_t primitive_attr_t::set_post_ops(const post_ops_t &post_ops) {
    this->post_ops_ = post_ops;
    return success;
}

/* Public C API */

status_t dnnl_primitive_attr_create(primitive_attr_t **attr) {
    if (attr == nullptr) return invalid_arguments;

    return safe_ptr_assign<dnnl_primitive_attr>(*attr, new dnnl_primitive_attr);
}

status_t dnnl_primitive_attr_clone(
        primitive_attr_t **attr, const primitive_attr_t *existing_attr) {
    if (any_null(attr, existing_attr)) return invalid_arguments;

    return safe_ptr_assign<dnnl_primitive_attr>(*attr, existing_attr->clone());
}

status_t dnnl_primitive_attr_destroy(primitive_attr_t *attr) {
    if (attr) delete attr;

    return success;
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

status_t dnnl_primitive_attr_get_output_scales(const primitive_attr_t *attr,
        dim_t *count, int *mask, const float **scales) {
    if (any_null(attr, count, mask, scales)) return invalid_arguments;

    *count = attr->output_scales_.count_;
    *mask = attr->output_scales_.mask_;
    *scales = attr->output_scales_.scales_;

    return success;
}

status_t dnnl_primitive_attr_set_output_scales(
        primitive_attr_t *attr, dim_t count, int mask, const float *scales) {
    bool ok = !any_null(attr, scales) && count > 0 && mask >= 0
            && attr->scales_.has_default_values();
    if (!ok) return invalid_arguments;

    return attr->output_scales_.set(count, mask, scales);
}

status_t dnnl_primitive_attr_set_scales(primitive_attr_t *attr, int arg,
        dim_t count, int mask, const float *scales) {
    bool ok = !any_null(attr, scales) && count > 0 && mask >= 0 && arg >= 0
            && attr->output_scales_.has_default_values();
    if (!ok) return invalid_arguments;

    return attr->scales_.set(arg, count, mask, scales);
}

status_t dnnl_primitive_attr_get_scales(primitive_attr_t *attr, int arg,
        dim_t *count, int *mask, const float **scales) {
    bool ok = !any_null(attr, count, mask, scales) && arg >= 0;
    if (!ok) return invalid_arguments;

    return attr->scales_.get(arg, count, mask, scales);
}

status_t dnnl_primitive_attr_get_zero_points(const primitive_attr_t *attr,
        int arg, dim_t *count, int *mask, const int **scales) {
    if (attr == nullptr) return invalid_arguments;
    return attr->zero_points_.get(arg, count, mask, scales);
}

status_t dnnl_primitive_attr_set_zero_points(primitive_attr_t *attr, int arg,
        dim_t count, int mask, const int *scales) {
    bool ok = !any_null(attr, scales) && count > 0 && mask >= 0;
    if (!ok) return invalid_arguments;
    return attr->zero_points_.set(arg, count, mask, scales);
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

    return safe_ptr_assign<dnnl_post_ops>(*post_ops, new dnnl_post_ops);
}

status_t dnnl_post_ops_destroy(post_ops_t *post_ops) {
    if (post_ops) delete post_ops;

    return success;
}

int dnnl_post_ops_len(const post_ops_t *post_ops) {
    if (post_ops) return post_ops->len_;

    return 0;
}

primitive_kind_t dnnl_post_ops_get_kind(const post_ops_t *post_ops, int index) {
    bool ok = post_ops && 0 <= index && index < post_ops->len_;
    if (!ok) return primitive_kind::undefined;

    return post_ops->entry_[index].kind;
}

status_t dnnl_post_ops_append_sum(post_ops_t *post_ops, float scale) {
    if (post_ops == nullptr) return invalid_arguments;

    return post_ops->append_sum(scale);
}

namespace {
bool simple_get_params_check(
        const post_ops_t *post_ops, int index, primitive_kind_t kind) {
    bool ok = true && post_ops != nullptr && 0 <= index
            && index < post_ops->len_ && post_ops->entry_[index].kind == kind;
    return ok;
}
} // namespace

status_t dnnl_post_ops_get_params_sum(
        const post_ops_t *post_ops, int index, float *scale) {
    bool ok = true
            && simple_get_params_check(post_ops, index, primitive_kind::sum)
            && !any_null(scale);
    if (!ok) return invalid_arguments;

    *scale = post_ops->entry_[index].sum.scale;
    return success;
}

status_t dnnl_post_ops_append_eltwise(post_ops_t *post_ops, float scale,
        alg_kind_t kind, float alpha, float beta) {
    if (post_ops == nullptr) return invalid_arguments;

    return post_ops->append_eltwise(scale, kind, alpha, beta);
}

status_t dnnl_post_ops_get_params_eltwise(const post_ops_t *post_ops, int index,
        float *scale, alg_kind_t *alg, float *alpha, float *beta) {
    bool ok = true
            && simple_get_params_check(post_ops, index, primitive_kind::eltwise)
            && !any_null(scale, alpha, beta);
    if (!ok) return invalid_arguments;

    const auto &e = post_ops->entry_[index].eltwise;
    *scale = e.scale;
    *alg = e.alg;
    *alpha = e.alpha;
    *beta = e.beta;

    return success;
}

status_t dnnl_primitive_attr_set_rnn_data_qparams(
        primitive_attr_t *attr, const float scale, const float shift) {
    if (attr == nullptr) return invalid_arguments;

    return attr->rnn_data_qparams_.set(scale, shift);
}

status_t dnnl_primitive_attr_set_rnn_weights_qparams(
        primitive_attr_t *attr, dim_t count, int mask, const float *scales) {
    bool ok = !any_null(attr, scales) && count > 0 && mask >= 0;
    if (!ok) return invalid_arguments;

    return attr->rnn_weights_qparams_.set(count, mask, scales);
}

status_t DNNL_API dnnl_primitive_attr_set_rnn_tparams(
        dnnl_primitive_attr_t attr, bool mode, dim_t ngates,
        const float *scales, float cscale) {
    if (attr == nullptr) return invalid_arguments;

    return attr->rnn_tparams_.set(mode, ngates, scales, cscale);
}
