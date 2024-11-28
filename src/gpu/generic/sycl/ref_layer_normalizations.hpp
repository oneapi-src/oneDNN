/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
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

#ifndef GPU_GENERIC_SYCL_REF_LAYER_NORMALIZATION_HPP
#define GPU_GENERIC_SYCL_REF_LAYER_NORMALIZATION_HPP

#include "common/c_types_map.hpp"
#include "common/utils.hpp"
#include "gpu/generic/sycl/sycl_gpu_primitive.hpp"
#include "gpu/generic/sycl/sycl_io_helper.hpp"
#include "gpu/generic/sycl/sycl_primitive_conf.hpp"
#include "gpu/generic/sycl/sycl_q10n.hpp"
#include "gpu/generic/sycl/sycl_utils.hpp"
#include "gpu/gpu_layer_normalization_pd.hpp"
#include "xpu/sycl/types.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

struct ref_layer_normalization_fwd_t : public gpu::generic::sycl::primitive_t {
    using gpu::generic::sycl::primitive_t::primitive_t;

    struct pd_t : public gpu_layer_normalization_fwd_pd_t {
        using gpu_layer_normalization_fwd_pd_t::
                gpu_layer_normalization_fwd_pd_t;

        DECLARE_COMMON_PD_T("dpcpp:ref:any", ref_layer_normalization_fwd_t);

        status_t init(impl::engine_t *engine) {
            using namespace data_type;
            using sm = primitive_attr_t::skip_mask_t;

            const memory_desc_wrapper data_d(src_md(0));
            const memory_desc_wrapper stat_d(src_md(1));
            const memory_desc_wrapper data_scaleshift_d(weights_md());
            const memory_desc_wrapper dst_d(dst_md(0));
            const memory_desc_wrapper var_d(src_md(2));

            VDISPATCH_LNORM(is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_LNORM((src_md(0)->format_desc.blocking.inner_nblks == 0),
                    VERBOSE_UNSUPPORTED_FORMAT_KIND);
            VDISPATCH_LNORM(is_supported_type(src_md(0)->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_LNORM(is_supported_type(dst_md(0)->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_LNORM(is_supported_type(stat_md()->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_LNORM(check_scale_shift_data_type({f32, bf16, f16}),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_LNORM(attr()->has_default_values(sm::scales_runtime),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_LNORM(IMPLICATION(!attr()->scales_.has_default_values(),
                                    scales_ok()),
                    VERBOSE_UNSUPPORTED_SCALES_CFG);
            VDISPATCH_LNORM(attr_scales_ok(), VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_LNORM(
                    set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_LNORM(md_dims_in_range(src_md()), VERBOSE_UNSUPPORTED_DT);
            return init_conf();
        }

        bool scales_ok() const {
            const std::vector<int> supported_args
                    = {DNNL_ARG_SRC, DNNL_ARG_DST};

            const auto &scales = attr()->scales_;
            for (auto arg : supported_args) {
                auto dt = scales.get(arg).data_type_;
                if (!is_supported_type(dt)) { return false; }
            }
            return true;
        }

        status_t init_conf();
        sycl_layer_normalization_conf_t conf_;
    };

    status_t init(impl::engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    kernel_t kernel_;
};

struct ref_layer_normalization_bwd_t : public gpu::generic::sycl::primitive_t {
    using gpu::generic::sycl::primitive_t::primitive_t;

    struct pd_t : public gpu_layer_normalization_bwd_pd_t {
        using gpu_layer_normalization_bwd_pd_t::
                gpu_layer_normalization_bwd_pd_t;

        DECLARE_COMMON_PD_T("dpcpp:ref:any", ref_layer_normalization_bwd_t);

        status_t init(impl::engine_t *engine) {
            using namespace data_type;

            const memory_desc_wrapper data_d(src_md(0));
            const memory_desc_wrapper diff_data_d(diff_src_md(0));
            const memory_desc_wrapper stat_d(src_md(1));
            const memory_desc_wrapper diff_data_scaleshift_d(
                    diff_weights_md(0));
            const memory_desc_wrapper diff_dst_d(diff_dst_md(0));
            const memory_desc_wrapper var_d(src_md(2));

            VDISPATCH_LNORM(!is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_LNORM((src_md(0)->format_desc.blocking.inner_nblks == 0),
                    VERBOSE_UNSUPPORTED_FORMAT_KIND);
            VDISPATCH_LNORM(
                    (diff_dst_md(0)->format_desc.blocking.inner_nblks == 0),
                    VERBOSE_UNSUPPORTED_FORMAT_KIND);
            VDISPATCH_LNORM(is_supported_type(src_md(0)->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_LNORM(is_supported_type(diff_dst_md(0)->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_LNORM(is_supported_type(diff_src_md(0)->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_LNORM(is_supported_type(stat_md()->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_LNORM(check_scale_shift_data_type({f32, bf16, f16}),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_LNORM(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_LNORM(
                    set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_LNORM(md_dims_in_range(diff_dst_md()),
                    VERBOSE_OUT_OF_RANGE_DIMS, "diff_dst");
            return init_conf();
        }

        status_t init_conf();
        sycl_layer_normalization_conf_t conf_;
    };

    status_t init(impl::engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward(ctx);
    }

private:
    status_t execute_backward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    kernel_t kernel_;
    kernel_t kernel2_;
};

} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
