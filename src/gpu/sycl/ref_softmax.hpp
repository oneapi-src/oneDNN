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
#ifndef GPU_SYCL_REF_SOFTMAX_HPP
#define GPU_SYCL_REF_SOFTMAX_HPP

#include "gpu/gpu_softmax_pd.hpp"
#include "gpu/sycl/sycl_gpu_primitive.hpp"
#include "gpu/sycl/sycl_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace sycl {

struct ref_sycl_softmax_fwd_t : public sycl_gpu_primitive_t {
    using sycl_gpu_primitive_t::sycl_gpu_primitive_t;

    struct pd_t : public gpu_softmax_fwd_pd_t {
        using gpu_softmax_fwd_pd_t::gpu_softmax_fwd_pd_t;

        DECLARE_COMMON_PD_T("dpcpp:ref:any", ref_sycl_softmax_fwd_t);

        status_t init(engine_t *engine) {
            using sm = primitive_attr_t::skip_mask_t;

            bool ok = is_fwd() && check_data_types(src_md()->data_type)
                    && check_data_types(dst_md()->data_type)
                    && (src_md(0)->format_desc.blocking.inner_nblks == 0)
                    && attr()->has_default_values(sm::scales_runtime)
                    && attr_oscale_ok()
                    && set_default_formats() == status::success;

            if (!ok) return status::unimplemented;
            return init_conf();
        }

        sycl_softmax_conf_t conf_;
        status_t init_conf();

        bool attr_oscale_ok() const {
            const auto &scales = attr()->scales_;
            bool ok = true;
            for (const auto &e : scales.scales_) {
                ok = ok && e.second.mask_ == 0;
            }
            return ok;
        }

        bool check_data_types(data_type_t src) {
            return utils::one_of(src, data_type::f32, data_type::bf16,
                    data_type::f16, data_type::s8, data_type::u8);
        }
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

struct ref_sycl_softmax_bwd_t : public sycl_gpu_primitive_t {
    using sycl_gpu_primitive_t::sycl_gpu_primitive_t;

    struct pd_t : public gpu_softmax_bwd_pd_t {
        using gpu_softmax_bwd_pd_t::gpu_softmax_bwd_pd_t;

        DECLARE_COMMON_PD_T("dpcpp:ref:any", ref_sycl_softmax_bwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            bool ok = !is_fwd()
                    && utils::one_of(dst_md()->data_type, f32, bf16, f16)
                    && utils::one_of(diff_src_md()->data_type, f32, bf16, f16)
                    && (dst_md(0)->format_desc.blocking.inner_nblks == 0)
                    && dst_md()->data_type == diff_dst_md()->data_type
                    && attr()->has_default_values()
                    && set_default_formats() == status::success;

            if (!ok) return status::unimplemented;
            return init_conf();
        }

        sycl_softmax_conf_t conf_;
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
