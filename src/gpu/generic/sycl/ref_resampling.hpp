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

#ifndef GPU_GENERIC_SYCL_REF_RESAMPLING_HPP
#define GPU_GENERIC_SYCL_REF_RESAMPLING_HPP

#include "gpu/generic/sycl/sycl_gpu_primitive.hpp"
#include "gpu/generic/sycl/sycl_io_helper.hpp"
#include "gpu/generic/sycl/sycl_post_ops.hpp"
#include "gpu/generic/sycl/sycl_primitive_conf.hpp"
#include "gpu/generic/sycl/sycl_q10n.hpp"
#include "gpu/generic/sycl/sycl_utils.hpp"
#include "gpu/gpu_resampling_pd.hpp"
#include "xpu/sycl/types.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

struct ref_resampling_fwd_t : public gpu::generic::sycl::primitive_t {
    using gpu::generic::sycl::primitive_t::primitive_t;

    struct pd_t : public gpu_resampling_fwd_pd_t {
        using gpu_resampling_fwd_pd_t::gpu_resampling_fwd_pd_t;

        DECLARE_COMMON_PD_T("dpcpp:ref:any", ref_resampling_fwd_t);

        status_t init(impl::engine_t *engine) {
            using namespace prop_kind;
            using namespace alg_kind;
            using sm = primitive_attr_t::skip_mask_t;
            const memory_desc_wrapper src_d(src_md(0));
            const memory_desc_wrapper dst_d(dst_md(0));

            VDISPATCH_RESAMPLING(is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_RESAMPLING(is_supported_type(src_md(0)->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_RESAMPLING(is_supported_type(dst_md(0)->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_RESAMPLING(attr()->has_default_values(sm::post_ops),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_RESAMPLING(sycl_post_ops_t::post_ops_ok(attr()),
                    VERBOSE_UNSUPPORTED_POSTOP);
            VDISPATCH_RESAMPLING_SC(
                    set_default_params(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_RESAMPLING_SC(attr_.set_default_formats(dst_md(0)),
                    VERBOSE_UNSUPPORTED_POSTOP);
            VDISPATCH_RESAMPLING(
                    (src_md(0)->format_desc.blocking.inner_nblks == 0),
                    VERBOSE_UNSUPPORTED_FORMAT_KIND);
            VDISPATCH_RESAMPLING(md_dims_in_range(src_md()),
                    VERBOSE_OUT_OF_RANGE_DIMS, "src");
            return init_conf();
        }

        status_t init_conf();
        sycl_resampling_conf_t conf_;
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

struct ref_resampling_bwd_t : public gpu::generic::sycl::primitive_t {
    using gpu::generic::sycl::primitive_t::primitive_t;

    struct pd_t : public gpu_resampling_bwd_pd_t {
        using gpu_resampling_bwd_pd_t::gpu_resampling_bwd_pd_t;

        DECLARE_COMMON_PD_T("dpcpp:ref:any", ref_resampling_bwd_t);

        status_t init(impl::engine_t *engine) {
            using namespace data_type;

            const memory_desc_wrapper diff_dst_d(diff_dst_md(0));
            const memory_desc_wrapper diff_src_d(diff_src_md(0));

            VDISPATCH_RESAMPLING(!is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_RESAMPLING(is_supported_type(diff_src_md(0)->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_RESAMPLING(is_supported_type(diff_dst_md(0)->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_RESAMPLING_SC(
                    set_default_params(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_RESAMPLING(
                    (diff_src_md(0)->format_desc.blocking.inner_nblks == 0),
                    VERBOSE_UNSUPPORTED_FORMAT_KIND);
            VDISPATCH_RESAMPLING(
                    (diff_dst_md(0)->format_desc.blocking.inner_nblks == 0),
                    VERBOSE_UNSUPPORTED_FORMAT_KIND);
            VDISPATCH_RESAMPLING(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_RESAMPLING(md_dims_in_range(diff_dst_md()),
                    VERBOSE_OUT_OF_RANGE_DIMS, "src");
            return init_conf();
        }

        status_t init_conf();
        sycl_resampling_conf_t conf_;
    };

    status_t init(impl::engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward(ctx);
    }

private:
    status_t execute_backward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    kernel_t kernel_;
};

} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
