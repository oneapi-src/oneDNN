/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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
#include "cpu/x64/jit_gemm_inner_product_utils.hpp"
#endif

#include "cpu/gemm_inner_product_utils.hpp"
#include "ref_depthwise_injector.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace inner_product_utils {

template <data_type_t acc_type, data_type_t dst_type>
struct ref_pp_kernel_t : public pp_kernel_t<acc_type, dst_type> {
    ref_pp_kernel_t(size_t OC, size_t MB, const primitive_attr_t *attr,
            data_type_t bias_dt, bool skip_sum)
        : pp_kernel_t<acc_type, dst_type>(OC, MB, attr, bias_dt, skip_sum) {
        for (int i = 0; i < this->post_ops_.len(); i++) {
            auto &post_op = this->post_ops_.entry_[i];
            if (post_op.is_eltwise()) {
                ref_eltwise_injectors_.push_back(new ref_eltwise_scalar_fwd_t(post_op.eltwise));
            } else if (post_op.is_depthwise()) {
                ref_depthwise_injectors_.push_back(new ref_depthwise_scalar_fwd_t(
                        post_op.depthwise.alg));
            }
        }
    }
    ~ref_pp_kernel_t() {
        for (auto impl : ref_eltwise_injectors_)
            delete impl;
        ref_eltwise_injectors_.clear();
        for (auto impl : ref_depthwise_injectors_)
            delete impl;
        ref_depthwise_injectors_.clear();
    }

    using acc_data_t = typename prec_traits<acc_type>::type;
    using dst_data_t = typename prec_traits<dst_type>::type;

    void operator()(dst_data_t *dst, const acc_data_t *acc, const char *bias,
            const float *scales, size_t start, size_t end, size_t runtime_oc,
            const float *dst_zero_points) const override;

private:
    nstl::vector<ref_eltwise_scalar_fwd_t*> ref_eltwise_injectors_;
    nstl::vector<ref_depthwise_scalar_fwd_t*> ref_depthwise_injectors_;
};

template <data_type_t acc_type, data_type_t dst_type>
void ref_pp_kernel_t<acc_type, dst_type>::operator()(dst_data_t *dst,
        const acc_data_t *acc, const char *bias, const float *scales,
        size_t start, size_t end, size_t runtime_oc,
        const float *dst_zero_points) const {
    using math::get_bias;

    if (end <= start) return;

    const size_t OC = this->runtime_oc() ? runtime_oc : this->OC_;

    size_t oc = start % OC;
    for (size_t i = start; i < end; i++) {
        float d = (float)acc[i];
        if (this->do_bias_) d += get_bias(bias, oc, this->bias_data_type_);
        if (this->do_scale_) d *= scales[oc * this->scale_idx_mult_];
        if (this->do_sum_) d += this->sum_scale_ * dst[i];

        int eltwise_inj_idx = 0;
        int depthwise_inj_idx = 0;

        for (int j = 0; j < this->post_ops_.len(); j++) {
            auto &post_op = this->post_ops_.entry_[j];
            if (post_op.is_eltwise()) {
                d = ref_eltwise_injectors_[eltwise_inj_idx]->compute_scalar(d);
                eltwise_inj_idx++;
            } else if (post_op.is_depthwise()) {
                auto depthwise_weights = post_op.depthwise.weights_data;
                auto depthwise_bias = post_op.depthwise.biases_data;
                d = ref_depthwise_injectors_[depthwise_inj_idx]->compute_scalar(d, depthwise_weights + oc, depthwise_bias + oc);
                depthwise_inj_idx++;
            } else if (post_op.is_quantization()) {
                bool do_dequantization = post_op.quantization.alg == alg_kind::quantization_quantize_dequantize;
                bool do_rounding = do_dequantization || dst_type == dnnl_f32 || j != this->post_ops_.len() - 1;

                auto quant = post_op.quantization;
                auto pcl = quant.crop_low_data->shifts_;
                auto pch = quant.crop_high_data->shifts_;
                auto pisc = quant.input_scale_data->scales_;
                auto pish = quant.input_shift_data->shifts_;
                auto posc = quant.output_scale_data->scales_;
                auto posh = quant.output_shift_data->shifts_;

                int cl_idx = quant.crop_low_data->count_ == 1 ? 0 : oc;
                int ch_idx = quant.crop_high_data->count_ == 1 ? 0 : oc;
                int isc_idx = quant.input_scale_data->count_ == 1 ? 0 : oc;
                int ish_idx = quant.input_shift_data->count_ == 1 ? 0 : oc;
                int osc_idx = quant.output_scale_data->count_ == 1 ? 0 : oc;
                int osh_idx = quant.output_shift_data->count_ == 1 ? 0 : oc;

                d = nstl::min(pch[ch_idx], nstl::max(pcl[cl_idx], d));
                d = d * pisc[isc_idx] + pish[ish_idx];

                if (do_rounding)
                    d = roundf(d);

                if (do_dequantization)
                    d = d * posc[osc_idx] + posh[osh_idx];
            }
        }
        dst[i] = qz_a1b0<float, dst_data_t>()(d);
        oc = (oc == OC - 1) ? 0 : oc + 1;
    }
}

// Interface section

template <data_type_t acc_type, data_type_t dst_type>
pp_kernel_t<acc_type, dst_type>::pp_kernel_t(size_t OC, size_t MB,
        const primitive_attr_t *attr, data_type_t bias_dt, bool skip_sum)
    : OC_(OC), MB_(MB), bias_data_type_(bias_dt) {
    do_scale_ = !attr->output_scales_.has_default_values();
    if (do_scale_) scale_idx_mult_ = (attr->output_scales_.mask_ == (1 << 1));

    post_ops_ = attr->post_ops_;

    // todo:
    const int sum_ind = post_ops_.find(primitive_kind::sum);
    do_sum_ = sum_ind != -1 && !skip_sum;
    if (do_sum_) sum_scale_ = post_ops_.entry_[sum_ind].sum.scale;

    do_bias_ = do_bias();
}

template <data_type_t acc_type, data_type_t dst_type>
pp_kernel_t<acc_type, dst_type> *pp_kernel_t<acc_type, dst_type>::create(
        size_t OC, size_t MB, const primitive_attr_t *attr, data_type_t bias_dt,
        bool skip_sum) {
#if DNNL_X64
    auto *res = x64::inner_product_utils::jit_pp_kernel_create<acc_type,
            dst_type>(OC, MB, attr, bias_dt, skip_sum);
    if (res) return res;
#endif

    return new ref_pp_kernel_t<acc_type, dst_type>(
            OC, MB, attr, bias_dt, skip_sum);
}

using namespace data_type;
template struct pp_kernel_t<f32, f32>;
template struct pp_kernel_t<s32, f32>;
template struct pp_kernel_t<s32, s32>;
template struct pp_kernel_t<s32, s8>;
template struct pp_kernel_t<s32, u8>;
template struct pp_kernel_t<f32, bf16>;

} // namespace inner_product_utils
} // namespace cpu
} // namespace impl
} // namespace dnnl
