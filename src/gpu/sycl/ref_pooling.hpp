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

#ifndef GPU_SYCL_REF_POOLING_HPP
#define GPU_SYCL_REF_POOLING_HPP

#include "common/c_types_map.hpp"
#include "common/utils.hpp"
#include "gpu/gpu_pooling_pd.hpp"
#include "gpu/primitive_conf.hpp"
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

struct ref_pooling_fwd_t : public sycl_gpu_primitive_t {
    using sycl_gpu_primitive_t::sycl_gpu_primitive_t;

    struct pd_t : public gpu_pooling_fwd_pd_t {
        using gpu_pooling_fwd_pd_t::gpu_pooling_fwd_pd_t;

        DECLARE_COMMON_PD_T("dpcpp:ref:any", ref_pooling_fwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            using namespace prop_kind;
            using namespace alg_kind;
            using sm = primitive_attr_t::skip_mask_t;
            const memory_desc_wrapper src_d(src_md(0));
            const memory_desc_wrapper dst_d(dst_md(0));
            const memory_desc_wrapper ws_d(workspace_md(0));

            const bool ok = is_fwd() && set_default_params() == status::success
                    && (src_md(0)->format_desc.blocking.inner_nblks == 0)
                    && (utils::everyone_is(
                                s8, src_md(0)->data_type, dst_md(0)->data_type)
                            || utils::everyone_is(u8, src_md(0)->data_type,
                                    dst_md(0)->data_type)
                            || utils::everyone_is(f32, src_md(0)->data_type,
                                    dst_md(0)->data_type)
                            || utils::everyone_is(bf16, src_md(0)->data_type,
                                    dst_md(0)->data_type)
                            || utils::everyone_is(f16, src_md(0)->data_type,
                                    dst_md(0)->data_type)
                            || utils::everyone_is(s32, src_md(0)->data_type,
                                    dst_md(0)->data_type))
                    && attr()->has_default_values(sm::post_ops)
                    && attr_.set_default_formats(dst_md(0)) == status::success;
            if (!ok) return status::unimplemented;
            bool is_training = desc_.prop_kind == prop_kind::forward_training;
            if (desc()->alg_kind == alg_kind::pooling_max && is_training)
                init_default_ws();
            return init_conf();
        }

        status_t init_conf();
        sycl_pooling_conf_t conf_;
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

struct ref_pooling_bwd_t : public sycl_gpu_primitive_t {
    using sycl_gpu_primitive_t::sycl_gpu_primitive_t;

    struct pd_t : public gpu_pooling_bwd_pd_t {
        using gpu_pooling_bwd_pd_t::gpu_pooling_bwd_pd_t;

        DECLARE_COMMON_PD_T("dpcpp:ref:any", ref_pooling_bwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;

            const memory_desc_wrapper diff_dst_d(diff_dst_md(0));
            const memory_desc_wrapper diff_src_d(diff_src_md(0));
            const memory_desc_wrapper ws_d(workspace_md(0));

            const bool ok = !is_fwd() && set_default_params() == status::success
                    && (utils::everyone_is(f32, diff_src_md(0)->data_type,
                                diff_dst_md(0)->data_type)
                            || utils::everyone_is(bf16,
                                    diff_src_md(0)->data_type,
                                    diff_dst_md(0)->data_type)
                            || utils::everyone_is(f16,
                                    diff_src_md(0)->data_type,
                                    diff_dst_md(0)->data_type))
                    && (src_md(0)->format_desc.blocking.inner_nblks == 0)
                    && attr()->has_default_values();

            if (!ok) return status::unimplemented;
            if (desc()->alg_kind == alg_kind::pooling_max) {
                init_default_ws();
                if (!compare_ws(hint_fwd_pd_)) return status::unimplemented;
            }
            return init_conf();
        }

        status_t init_conf();
        sycl_pooling_conf_t conf_;
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
