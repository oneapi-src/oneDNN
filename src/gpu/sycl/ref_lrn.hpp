/*******************************************************************************
* Copyright 2023 Intel Corporation
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
#ifndef GPU_SYCL_REF_LRN_HPP
#define GPU_SYCL_REF_LRN_HPP

#include "gpu/gpu_lrn_pd.hpp"
#include "gpu/sycl/sycl_gpu_primitive.hpp"
#include "gpu/sycl/sycl_primitive_conf.hpp"
#include "gpu/sycl/sycl_types.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace sycl {

struct ref_sycl_lrn_fwd_t : public sycl_gpu_primitive_t {
    using sycl_gpu_primitive_t::sycl_gpu_primitive_t;

    struct pd_t : public gpu_lrn_fwd_pd_t {
        using gpu_lrn_fwd_pd_t::gpu_lrn_fwd_pd_t;

        DECLARE_COMMON_PD_T("dpcpp:ref:any", ref_sycl_lrn_fwd_t);

        status_t init(engine_t *engine) {
            using namespace format_tag;
            using namespace data_type;

            const memory_desc_wrapper src_d(src_md());
            const memory_desc_wrapper dst_d(dst_md());

            bool ok = is_fwd()
                    && utils::one_of(src_md()->data_type, f32, bf16, f16)
                    && utils::everyone_is(
                            src_md()->data_type, dst_md()->data_type)
                    && (src_md(0)->format_desc.blocking.inner_nblks == 0)
                    && attr()->has_default_values()
                    && set_default_formats_common() && src_d == dst_d;

            if (!ok) return status::unimplemented;
            dat_tag_ = memory_desc_matches_one_of_tag(*src_md(), nchw, nhwc);
            return init_conf();
        }

        sycl_lrn_conf_t conf_;
        format_tag_t dat_tag_;

    private:
        status_t init_conf();
    };

    status_t init(engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    status_t execute_forward(const exec_ctx_t &ctx) const;
    compute::kernel_t kernel_;
};

struct ref_sycl_lrn_bwd_t : public sycl_gpu_primitive_t {
    using sycl_gpu_primitive_t::sycl_gpu_primitive_t;

    struct pd_t : public gpu_lrn_bwd_pd_t {
        using gpu_lrn_bwd_pd_t::gpu_lrn_bwd_pd_t;

        DECLARE_COMMON_PD_T("dpcpp:ref:any", ref_sycl_lrn_bwd_t);

        status_t init(engine_t *engine) {
            using namespace format_tag;
            using namespace data_type;

            const memory_desc_wrapper diff_src_d(diff_src_md());
            const memory_desc_wrapper diff_dst_d(diff_dst_md());

            bool ok = !is_fwd()
                    && utils::one_of(src_md()->data_type, f32, bf16, f16)
                    && utils::everyone_is(src_md()->data_type,
                            diff_src_md()->data_type, diff_dst_md()->data_type)
                    && attr()->has_default_values()
                    && set_default_formats_common() && diff_dst_d == diff_src_d;

            if (!ok) return status::unimplemented;

            dat_tag_ = memory_desc_matches_one_of_tag(*src_md(), nchw, nhwc);
            return init_conf();
        }

        sycl_lrn_conf_t conf_;
        format_tag_t dat_tag_;

    private:
        status_t init_conf();
    };

    status_t init(engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward(ctx);
    }

private:
    status_t execute_backward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    compute::kernel_t kernel_;
};

} // namespace sycl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
