/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
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

#include <memory>

#include "common/math_utils.hpp"

#include "cpu/primitive_attr_postops.hpp"
#include "cpu/simple_q10n.hpp"

#if DNNL_X64
#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/x64/jit_gemm_inner_product_utils.hpp"
#endif

#include "cpu/gemm_inner_product_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace inner_product_utils {

template <data_type_t acc_type, data_type_t dst_type>
struct ref_pp_kernel_t : public pp_kernel_t<acc_type, dst_type> {
    ref_pp_kernel_t(size_t OC, size_t MB, dim_t dst_mb_stride,
            const primitive_attr_t *attr, data_type_t bias_dt, bool skip_sum)
        : pp_kernel_t<acc_type, dst_type>(
                OC, MB, dst_mb_stride, attr, bias_dt, skip_sum)
        , ref_post_ops_(this->do_sum_ || this->do_eltwise_ || this->do_binary_
                          ? utils::make_unique<ref_post_ops_t>(
                                  this->post_ops_, skip_sum)
                          : nullptr) {}

    using acc_data_t = typename prec_traits<acc_type>::type;
    using dst_data_t = typename prec_traits<dst_type>::type;

    void operator()(dst_data_t *dst, const acc_data_t *acc, const char *bias,
            const float *scales, size_t start, size_t dst_logical_offs,
            size_t dst_row_idx, size_t end, size_t runtime_oc,
            dim_t dst_mb_stride, const float *dst_zero_points,
            const void *post_ops_binary_rhs_arg_vec, const void *dst_orig,
            const exec_ctx_t &ctx, const memory_desc_t &dst_md) const override;

private:
    std::unique_ptr<ref_post_ops_t> ref_post_ops_;
};

template <data_type_t acc_type, data_type_t dst_type>
void ref_pp_kernel_t<acc_type, dst_type>::operator()(dst_data_t *dst,
        const acc_data_t *acc, const char *bias, const float *scales,
        size_t start, size_t dst_row_idx, size_t dst_logical_off, size_t end,
        size_t runtime_oc, dim_t dst_mb_stride, const float *dst_zero_points,
        const void * /* post_ops_binary_rhs_arg_vec */,
        const void * /* dst_orig */, const exec_ctx_t &ctx,
        const memory_desc_t &dst_md) const {
    using math::get_bias;

    if (end <= start) return;

    const size_t OC = this->runtime_oc() ? runtime_oc : this->OC_;

    ref_post_ops_t::args_t args;
    args.ctx = &ctx;
    args.dst_md = &dst_md;
    const bool apply_postops
            = this->do_sum_ || this->do_eltwise_ || this->do_binary_;
    auto calculate_dst_value_and_increment_oc
            = [&](const acc_data_t &acc_value, dst_data_t &dst_value,
                      size_t &oc_value, const size_t dst_offset) {
                  float d = (float)acc_value;
                  if (this->do_bias())
                      d += get_bias(bias, oc_value, this->bias_data_type_);
                  if (this->do_scale_)
                      d *= scales[oc_value * this->scale_idx_mult_];
                  if (apply_postops) {
                      if (this->do_sum_) args.dst_val = dst_value;
                      args.l_offset = dst_offset;
                      ref_post_ops_->execute(d, args);
                  }
                  if (this->do_dst_zero_points_) d += dst_zero_points[0];
                  dst_value = qz_a1b0<float, dst_data_t>()(d);
                  oc_value = (oc_value == OC - 1) ? 0 : oc_value + 1;
              };

    size_t oc = start % OC;
    dim_t offt = (start / OC) * dst_mb_stride + oc;
    if (this->has_trivial_mb_stride()) {
        // keep separate code path to avoid performance degradations
        for (size_t i = start; i < end; i++) {
            calculate_dst_value_and_increment_oc(acc[i], dst[i], oc, offt);
            ++offt;
        }
    } else {
        const bool acc_is_dst = dst == (dst_data_t *)acc;
        dst = dst + offt;
        // if dst and acc point to same address (inplace), then strides
        // must be similar, else assume acc buffer is dense.
        acc = acc + (acc_is_dst ? offt : start);
        while (start < end) {
            calculate_dst_value_and_increment_oc(*acc, *dst, oc, offt);
            if (oc == 0) {
                dst = dst + dst_mb_stride - OC;
                offt += dst_mb_stride - OC;
                // if dst and acc point to same address (inplace), then strides
                // must be similar, else assume acc buffer is dense.
                if (acc_is_dst) acc = acc + dst_mb_stride - OC;
            }
            ++dst;
            ++offt;
            ++acc;
            ++start;
        }
    }
}

// Interface section

template <data_type_t acc_type, data_type_t dst_type>
pp_kernel_t<acc_type, dst_type>::pp_kernel_t(size_t OC, size_t MB,
        dim_t dst_mb_stride, const primitive_attr_t *attr, data_type_t bias_dt,
        bool skip_sum)
    : OC_(OC)
    , MB_(MB)
    , dst_mb_stride_(dst_mb_stride)
    , bias_data_type_(bias_dt) {
    do_scale_ = !attr->output_scales_.has_default_values();
    if (do_scale_) scale_idx_mult_ = (attr->output_scales_.mask_ == (1 << 1));

    post_ops_ = attr->post_ops_;
    const int eltwise_ind = post_ops_.find(primitive_kind::eltwise);
    do_eltwise_ = eltwise_ind != -1;

    const int binary_ind = post_ops_.find(primitive_kind::binary);
    do_binary_ = binary_ind != -1;

    const int sum_ind = post_ops_.find(primitive_kind::sum);
    do_sum_ = sum_ind != -1 && !skip_sum;
    if (do_sum_) sum_scale_ = post_ops_.entry_[sum_ind].sum.scale;

    if (do_bias())
        bias_data_type_size_ = types::data_type_size(bias_data_type_);

    if (!attr->zero_points_.has_default_values(DNNL_ARG_DST))
        do_dst_zero_points_ = true;
}

template <data_type_t acc_type, data_type_t dst_type>
pp_kernel_t<acc_type, dst_type> *pp_kernel_t<acc_type, dst_type>::create(
        size_t OC, size_t MB, dim_t dst_mb_stride, const primitive_attr_t *attr,
        data_type_t bias_dt, const memory_desc_t *dst_md, bool skip_sum) {
#if DNNL_X64
    auto *res = x64::inner_product_utils::jit_pp_kernel_create<acc_type,
            dst_type>(OC, MB, dst_mb_stride, attr, bias_dt, dst_md, skip_sum);
    if (res) return res;
#endif

    return new ref_pp_kernel_t<acc_type, dst_type>(
            OC, MB, dst_mb_stride, attr, bias_dt, skip_sum);
}

using namespace data_type;
template struct pp_kernel_t<f32, f32>;
template struct pp_kernel_t<s32, f32>;
template struct pp_kernel_t<s32, s32>;
template struct pp_kernel_t<s32, s8>;
template struct pp_kernel_t<s32, u8>;
template struct pp_kernel_t<f32, bf16>;

bool post_ops_ok(const post_ops_t &post_ops, const memory_desc_wrapper *dst_d) {
#if DNNL_X64
    static constexpr auto isa_supported
            = x64::inner_product_utils::jit_pp_kernel_supported_isa();
    using namespace cpu::x64;
    if (mayiuse(isa_supported)) {
        using namespace x64::injector;
        static constexpr bool sum_at_pos_0_only = true;
        static constexpr bool sum_requires_scale_one = false;
        static const bcast_set_t enabled_bcast_strategy
                = {broadcasting_strategy_t::scalar,
                        broadcasting_strategy_t::per_oc,
                        broadcasting_strategy_t::per_oc_spatial,
                        broadcasting_strategy_t::no_broadcast};
        return injector::post_ops_ok({isa_supported, {binary, eltwise, sum},
                post_ops, dst_d, sum_at_pos_0_only, sum_requires_scale_one,
                enabled_bcast_strategy});
    }
#endif
    for (size_t i = 0; i < post_ops.entry_.size(); i++) {
        const auto &post_op = post_ops.entry_[i];
        const bool sum_postop_present = post_op.is_sum(false);
        if (sum_postop_present && i > 0) return false;
        if (!(sum_postop_present || post_op.is_eltwise()
                    || post_op.is_binary()))
            return false;
    }
    return true;
}

bool post_ops_ok(const post_ops_t &post_ops, const memory_desc_t *dst_d) {
    const auto dst_md = memory_desc_wrapper(dst_d);
    return post_ops_ok(post_ops, &dst_md);
}

} // namespace inner_product_utils
} // namespace cpu
} // namespace impl
} // namespace dnnl
