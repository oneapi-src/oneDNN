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

#ifndef GPU_SYCL_REF_PRELU_HPP
#define GPU_SYCL_REF_PRELU_HPP

#include "common/broadcast_strategy.hpp"
#include "gpu/gpu_prelu_pd.hpp"
#include "gpu/sycl/prelu_kernels.hpp"
#include "gpu/sycl/sycl_gpu_primitive.hpp"
#include "gpu/sycl/sycl_io_helper.hpp"
#include "gpu/sycl/sycl_post_ops.hpp"
#include "gpu/sycl/sycl_primitive_conf.hpp"
#include "gpu/sycl/sycl_q10n.hpp"
#include "gpu/sycl/sycl_types.hpp"
#include "sycl/sycl_stream.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace sycl {

struct ref_prelu_fwd_t : public sycl_gpu_primitive_t {
    using sycl_gpu_primitive_t::sycl_gpu_primitive_t;

    struct pd_t : public gpu_prelu_fwd_pd_t {
        using gpu_prelu_fwd_pd_t::gpu_prelu_fwd_pd_t;

        DECLARE_COMMON_PD_T("dpcpp:ref:any", ref_prelu_fwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;

            const memory_desc_wrapper data_d(src_md(0));
            const memory_desc_wrapper weights_d(weights_md(0));
            const memory_desc_wrapper dst_d(dst_md(0));

            const bool ok = is_fwd() && set_default_formats()
                    && (src_md(0)->format_desc.blocking.inner_nblks == 0)
                    && (weights_md(0)->format_desc.blocking.inner_nblks == 0);

            if (!ok) return status::unimplemented;
            return init_conf();
        }

        status_t init_conf();
        sycl_prelu_conf_t conf_;
    };

    status_t init(engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    compute::kernel_t kernel_;
};

struct ref_prelu_bwd_t : public sycl_gpu_primitive_t {
    using sycl_gpu_primitive_t::sycl_gpu_primitive_t;

    struct pd_t : public gpu_prelu_bwd_pd_t {
        using gpu_prelu_bwd_pd_t::gpu_prelu_bwd_pd_t;

        DECLARE_COMMON_PD_T("dpcpp:ref:any", ref_prelu_bwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            const memory_desc_wrapper data_d(src_md(0));
            const memory_desc_wrapper weights_d(weights_md(0));
            const memory_desc_wrapper diff_data_d(diff_src_md(0));
            const memory_desc_wrapper diff_weights_d(diff_weights_md(0));
            const memory_desc_wrapper diff_dst_d(diff_dst_md(0));

            const bool ok = !is_fwd() && set_default_formats()
                    && (src_md(0)->format_desc.blocking.inner_nblks == 0)
                    && (weights_md(0)->format_desc.blocking.inner_nblks == 0)
                    && diff_src_md(0)->data_type == src_md(0)->data_type
                    && diff_weights_md(0)->data_type
                            == weights_md(0)->data_type;

            if (!ok) return status::unimplemented;
            return init_conf();
        }

        status_t init_conf();
        sycl_prelu_conf_t conf_;
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
