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

#ifndef GPU_GENERIC_SYCL_REF_POOLING_HPP
#define GPU_GENERIC_SYCL_REF_POOLING_HPP

#include "common/c_types_map.hpp"
#include "common/utils.hpp"
#include "gpu/generic/sycl/sycl_gpu_primitive.hpp"
#include "gpu/generic/sycl/sycl_io_helper.hpp"
#include "gpu/generic/sycl/sycl_post_ops.hpp"
#include "gpu/generic/sycl/sycl_primitive_conf.hpp"
#include "gpu/generic/sycl/sycl_q10n.hpp"
#include "gpu/generic/sycl/sycl_utils.hpp"
#include "gpu/gpu_pooling_pd.hpp"
#include "gpu/intel/primitive_conf.hpp"
#include "xpu/sycl/types.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

struct ref_pooling_fwd_t : public gpu::generic::sycl::primitive_t {
    using gpu::generic::sycl::primitive_t::primitive_t;

    struct pd_t : public gpu_pooling_fwd_pd_t {
        using gpu_pooling_fwd_pd_t::gpu_pooling_fwd_pd_t;

        DECLARE_COMMON_PD_T("dpcpp:ref:any", ref_pooling_fwd_t);

        status_t init(impl::engine_t *engine) {
            using namespace data_type;
            using namespace prop_kind;
            using namespace alg_kind;
            using sm = primitive_attr_t::skip_mask_t;
            const memory_desc_wrapper src_d(src_md(0));
            const memory_desc_wrapper dst_d(dst_md(0));
            const memory_desc_wrapper ws_d(workspace_md(0));

            VDISPATCH_POOLING(is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_POOLING_SC(set_default_params(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_POOLING(
                    (src_md(0)->format_desc.blocking.inner_nblks == 0),
                    VERBOSE_UNSUPPORTED_FORMAT_KIND);
            VDISPATCH_POOLING((!utils::one_of(f64, src_md(0)->data_type,
                                      dst_md(0)->data_type))
                            && (IMPLICATION(src_md(0)->data_type == bf16,
                                    dst_md(0)->data_type == bf16))
                            && (IMPLICATION(src_md(0)->data_type == s8,
                                    dst_md(0)->data_type != u8))
                            && (IMPLICATION(src_md(0)->data_type == u8,
                                    dst_md(0)->data_type != s8))
                            && (IMPLICATION(src_md(0)->data_type
                                            != dst_md(0)->data_type,
                                    desc()->prop_kind == forward_inference)),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_POOLING(attr()->has_default_values(sm::post_ops),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_POOLING(sycl_post_ops_t::post_ops_ok(attr(), true, false),
                    VERBOSE_UNSUPPORTED_POSTOP);
            VDISPATCH_POOLING_SC(attr_.set_default_formats(dst_md(0)),
                    VERBOSE_UNSUPPORTED_POSTOP);
            VDISPATCH_POOLING(md_dims_in_range(src_md()),
                    VERBOSE_OUT_OF_RANGE_DIMS, "src");

            bool is_training = desc_.prop_kind == prop_kind::forward_training;
            if (desc()->alg_kind == alg_kind::pooling_max && is_training)
                init_default_ws();
            return init_conf();
        }

        status_t init_conf();
        sycl_pooling_fwd_conf_t conf_;
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

struct ref_pooling_bwd_t : public gpu::generic::sycl::primitive_t {
    using gpu::generic::sycl::primitive_t::primitive_t;

    struct pd_t : public gpu_pooling_bwd_pd_t {
        using gpu_pooling_bwd_pd_t::gpu_pooling_bwd_pd_t;

        DECLARE_COMMON_PD_T("dpcpp:ref:any", ref_pooling_bwd_t);

        status_t init(impl::engine_t *engine) {
            using namespace data_type;

            const memory_desc_wrapper diff_dst_d(diff_dst_md(0));
            const memory_desc_wrapper diff_src_d(diff_src_md(0));
            const memory_desc_wrapper ws_d(workspace_md(0));

            VDISPATCH_POOLING(!is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_POOLING_SC(set_default_params(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_POOLING(
                    (utils::everyone_is(f32, diff_src_md(0)->data_type,
                             diff_dst_md(0)->data_type)
                            || utils::everyone_is(bf16,
                                    diff_src_md(0)->data_type,
                                    diff_dst_md(0)->data_type)
                            || utils::everyone_is(f16,
                                    diff_src_md(0)->data_type,
                                    diff_dst_md(0)->data_type)),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_POOLING(
                    (src_md(0)->format_desc.blocking.inner_nblks == 0),
                    VERBOSE_UNSUPPORTED_FORMAT_KIND);
            VDISPATCH_POOLING(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_POOLING(md_dims_in_range(diff_dst_md()),
                    VERBOSE_OUT_OF_RANGE_DIMS, "src");

            if (desc()->alg_kind == alg_kind::pooling_max) {
                init_default_ws();
                VDISPATCH_POOLING(compare_ws(hint_fwd_pd_), VERBOSE_WS_INIT);
            }
            return init_conf();
        }

        status_t init_conf();
        sycl_pooling_bwd_conf_t conf_;
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
