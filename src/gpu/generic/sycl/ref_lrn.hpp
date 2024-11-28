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
#ifndef GPU_GENERIC_SYCL_REF_LRN_HPP
#define GPU_GENERIC_SYCL_REF_LRN_HPP

#include "gpu/generic/sycl/sycl_gpu_primitive.hpp"
#include "gpu/generic/sycl/sycl_primitive_conf.hpp"
#include "gpu/generic/sycl/sycl_utils.hpp"
#include "gpu/gpu_lrn_pd.hpp"
#include "xpu/sycl/types.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

struct ref_sycl_lrn_fwd_t : public gpu::generic::sycl::primitive_t {
    using gpu::generic::sycl::primitive_t::primitive_t;

    struct pd_t : public gpu_lrn_fwd_pd_t {
        using gpu_lrn_fwd_pd_t::gpu_lrn_fwd_pd_t;

        DECLARE_COMMON_PD_T("dpcpp:ref:any", ref_sycl_lrn_fwd_t);

        status_t init(impl::engine_t *engine) {
            using namespace format_tag;
            using namespace data_type;

            const memory_desc_wrapper src_d(src_md());
            const memory_desc_wrapper dst_d(dst_md());

            VDISPATCH_LRN(is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_LRN(utils::one_of(src_md()->data_type, f32, bf16, f16),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_LRN(utils::everyone_is(
                                  src_md()->data_type, dst_md()->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_LRN((src_md(0)->format_desc.blocking.inner_nblks == 0),
                    VERBOSE_UNSUPPORTED_FORMAT_KIND);
            VDISPATCH_LRN(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_LRN(
                    set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_LRN(
                    src_d == dst_d, VERBOSE_INCONSISTENT_MDS, "src", "dst");
            VDISPATCH_LRN(md_dims_in_range(src_md()), VERBOSE_OUT_OF_RANGE_DIMS,
                    "src");

            dat_tag_ = memory_desc_matches_one_of_tag(*src_md(), nchw, nhwc);
            return init_conf();
        }

        sycl_lrn_conf_t conf_;
        format_tag_t dat_tag_;

    private:
        status_t init_conf();
    };

    status_t init(impl::engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    status_t execute_forward(const exec_ctx_t &ctx) const;
    kernel_t kernel_;
};

struct ref_sycl_lrn_bwd_t : public gpu::generic::sycl::primitive_t {
    using gpu::generic::sycl::primitive_t::primitive_t;

    struct pd_t : public gpu_lrn_bwd_pd_t {
        using gpu_lrn_bwd_pd_t::gpu_lrn_bwd_pd_t;

        DECLARE_COMMON_PD_T("dpcpp:ref:any", ref_sycl_lrn_bwd_t);

        status_t init(impl::engine_t *engine) {
            using namespace format_tag;
            using namespace data_type;

            const memory_desc_wrapper diff_src_d(diff_src_md());
            const memory_desc_wrapper diff_dst_d(diff_dst_md());

            VDISPATCH_LRN(!is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_LRN(utils::one_of(src_md()->data_type, f32, bf16, f16),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_LRN(
                    utils::everyone_is(src_md()->data_type,
                            diff_src_md()->data_type, diff_dst_md()->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_LRN(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_LRN(
                    set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_LRN(diff_dst_d == diff_src_d, VERBOSE_INCONSISTENT_MDS,
                    "diff_src", "diff_dst");
            VDISPATCH_LRN(md_dims_in_range(diff_dst_md()),
                    VERBOSE_OUT_OF_RANGE_DIMS, "diff_dst");

            dat_tag_ = memory_desc_matches_one_of_tag(*src_md(), nchw, nhwc);
            return init_conf();
        }

        sycl_lrn_conf_t conf_;
        format_tag_t dat_tag_;

    private:
        status_t init_conf();
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
