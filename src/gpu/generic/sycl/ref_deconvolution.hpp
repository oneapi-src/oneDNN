/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#ifndef GPU_SYCL_REF_DECONVOLUTION_HPP
#define GPU_SYCL_REF_DECONVOLUTION_HPP

#include "gpu/generic/sycl/ref_convolution.hpp"
#include "gpu/generic/sycl/sycl_gpu_primitive.hpp"
#include "gpu/generic/sycl/sycl_io_helper.hpp"
#include "gpu/generic/sycl/sycl_post_ops.hpp"
#include "gpu/generic/sycl/sycl_primitive_conf.hpp"
#include "gpu/generic/sycl/sycl_q10n.hpp"
#include "gpu/generic/sycl/sycl_utils.hpp"
#include "gpu/gpu_deconvolution_pd.hpp"
#include "xpu/sycl/types.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

struct ref_deconvolution_bwd_weights_t
    : public gpu::generic::sycl::primitive_t {
    using gpu::generic::sycl::primitive_t::primitive_t;

    struct pd_t : public deconvolution_bwd_weights_pd_t {
        using deconvolution_bwd_weights_pd_t::deconvolution_bwd_weights_pd_t;

        DECLARE_COMMON_PD_T("dpcpp:ref:any", ref_deconvolution_bwd_weights_t);

        status_t init(impl::engine_t *engine) {
            using namespace data_type;

            const memory_desc_wrapper data_d(src_md());
            const memory_desc_wrapper diff_weights_d(diff_weights_md());
            const memory_desc_wrapper diff_dst_d(diff_dst_md());

            VDISPATCH_DECONVOLUTION(
                    desc()->prop_kind == prop_kind::backward_weights,
                    VERBOSE_BAD_PROPKIND);
            VDISPATCH_DECONVOLUTION(
                    check_convolution_work_amount(diff_weights_d, OC()),
                    VERBOSE_IMPL_HEURISTIC_FAIL,
                    "number of elements exceeds threshold");
            VDISPATCH_DECONVOLUTION(md_dims_in_range(src_md()),
                    VERBOSE_OUT_OF_RANGE_DIMS, "src");
            VDISPATCH_DECONVOLUTION(
                    set_default_formats(), VERBOSE_UNSUPPORTED_TAG_S);
            VDISPATCH_DECONVOLUTION(check_convolution_data_types(
                                            data_d, diff_weights_d, diff_dst_d),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_DECONVOLUTION(check_convolution_formats(
                                            data_d, diff_weights_d, diff_dst_d),
                    VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_DECONVOLUTION(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_DECONVOLUTION(
                    desc()->alg_kind == alg_kind::deconvolution_direct,
                    VERBOSE_BAD_ALGORITHM);

            return init_conf();
        }

        sycl_convolution_bwd_weights_conf_t conf_;

    private:
        status_t init_conf();

        bool set_default_formats_common_template(memory_desc_t &src_md,
                format_tag_t src_tag, memory_desc_t &wei_md,
                format_tag_t wei_tag, memory_desc_t &dst_md,
                format_tag_t dst_tag, memory_desc_t &bia_md) {
            using namespace format_tag;

#define IS_OK(f) \
    do { \
        if ((f) != status::success) return false; \
    } while (0)

            if (src_md.format_kind == format_kind::any
                    && !utils::one_of(src_tag, any, undef))
                IS_OK(memory_desc_init_by_tag(src_md, src_tag));
            if (dst_md.format_kind == format_kind::any
                    && !utils::one_of(dst_tag, any, undef))
                IS_OK(memory_desc_init_by_tag(dst_md, dst_tag));
            if (wei_md.format_kind == format_kind::any
                    && !utils::one_of(wei_tag, any, undef))
                IS_OK(memory_desc_init_by_tag(wei_md, wei_tag));
            if (with_bias() && bia_md.format_kind == format_kind::any)
                IS_OK(memory_desc_init_by_tag(bia_md, x));
#undef IS_OK

            return true;
        }

        bool set_default_formats() {
            using namespace format_tag;
            auto dat_tag = utils::pick(ndims() - 3, nwc, nhwc, ndhwc);
            auto wei_tag = with_groups()
                    ? utils::pick(ndims() - 3, goiw, goihw, goidhw)
                    : utils::pick(ndims() - 3, oiw, oihw, oidhw);
            return set_default_formats_common_template(src_md_, dat_tag,
                    diff_weights_md_, wei_tag, diff_dst_md_, dat_tag,
                    diff_bias_md_);
        }
    };

    status_t init(impl::engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    kernel_t kernel_;
};

} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
