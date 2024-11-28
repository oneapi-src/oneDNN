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

#ifndef GPU_GENERIC_SYCL_REF_PRELU_HPP
#define GPU_GENERIC_SYCL_REF_PRELU_HPP

#include "common/broadcast_strategy.hpp"
#include "common/primitive_desc_iterator.hpp"
#include "common/reduction_pd.hpp"
#include "gpu/generic/sycl/prelu_kernels.hpp"
#include "gpu/generic/sycl/sycl_gpu_primitive.hpp"
#include "gpu/generic/sycl/sycl_io_helper.hpp"
#include "gpu/generic/sycl/sycl_post_ops.hpp"
#include "gpu/generic/sycl/sycl_primitive_conf.hpp"
#include "gpu/generic/sycl/sycl_q10n.hpp"
#include "gpu/generic/sycl/sycl_utils.hpp"
#include "gpu/gpu_prelu_pd.hpp"
#include "xpu/sycl/types.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

struct ref_prelu_fwd_t : public gpu::generic::sycl::primitive_t {
    using gpu::generic::sycl::primitive_t::primitive_t;

    struct pd_t : public gpu_prelu_fwd_pd_t {
        using gpu_prelu_fwd_pd_t::gpu_prelu_fwd_pd_t;

        DECLARE_COMMON_PD_T("dpcpp:ref:any", ref_prelu_fwd_t);

        status_t init(impl::engine_t *engine) {
            using namespace data_type;

            const memory_desc_wrapper data_d(src_md(0));
            const memory_desc_wrapper weights_d(weights_md(0));
            const memory_desc_wrapper dst_d(dst_md(0));

            VDISPATCH_PRELU(is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_PRELU(set_default_formats(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_PRELU((src_md(0)->format_desc.blocking.inner_nblks == 0),
                    VERBOSE_UNSUPPORTED_FORMAT_KIND);
            VDISPATCH_PRELU(
                    (weights_md(0)->format_desc.blocking.inner_nblks == 0),
                    VERBOSE_UNSUPPORTED_FORMAT_KIND);
            VDISPATCH_PRELU(check_data_types(data_d, weights_d, dst_d),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_PRELU(md_dims_in_range(src_md()),
                    VERBOSE_OUT_OF_RANGE_DIMS, "src");
            VDISPATCH_PRELU(md_dims_in_range(weights_md()),
                    VERBOSE_OUT_OF_RANGE_DIMS, "weights");
            VDISPATCH_PRELU(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            return init_conf();
        }

        status_t init_conf();
        sycl_prelu_conf_t conf_;

        static bool check_data_types(const memory_desc_wrapper &src,
                const memory_desc_wrapper &wei,
                const memory_desc_wrapper &dst) {
            for (const auto &mdw : {src, wei, dst}) {
                if (!is_supported_type(mdw.data_type())) return false;
            }
            return true;
        }
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

struct ref_prelu_bwd_t : public gpu::generic::sycl::primitive_t {
    using gpu::generic::sycl::primitive_t::primitive_t;

    struct pd_t : public gpu_prelu_bwd_pd_t {
        using gpu_prelu_bwd_pd_t::gpu_prelu_bwd_pd_t;

        DECLARE_COMMON_PD_T("dpcpp:ref:any", ref_prelu_bwd_t);

        status_t init(impl::engine_t *engine) {
            using namespace data_type;
            const memory_desc_wrapper data_d(src_md(0));
            const memory_desc_wrapper weights_d(weights_md(0));
            const memory_desc_wrapper diff_data_d(diff_src_md(0));
            const memory_desc_wrapper diff_weights_d(diff_weights_md(0));
            const memory_desc_wrapper diff_dst_d(diff_dst_md(0));

            VDISPATCH_PRELU(!is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_PRELU(set_default_formats(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_PRELU((src_md(0)->format_desc.blocking.inner_nblks == 0),
                    VERBOSE_UNSUPPORTED_FORMAT_KIND);
            VDISPATCH_PRELU(
                    (weights_md(0)->format_desc.blocking.inner_nblks == 0),
                    VERBOSE_UNSUPPORTED_FORMAT_KIND);
            VDISPATCH_PRELU(diff_src_md(0)->data_type == src_md(0)->data_type,
                    VERBOSE_INCONSISTENT_DT, "src", "diff_src");
            VDISPATCH_PRELU(
                    diff_weights_md(0)->data_type == weights_md(0)->data_type,
                    VERBOSE_INCONSISTENT_DT, "weights", "diff_weights");
            VDISPATCH_PRELU(check_data_types(data_d, weights_d, diff_dst_d),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_PRELU(md_dims_in_range(diff_src_md()),
                    VERBOSE_OUT_OF_RANGE_DIMS, "diff_src");
            VDISPATCH_PRELU(md_dims_in_range(weights_md()),
                    VERBOSE_OUT_OF_RANGE_DIMS, "weights");
            VDISPATCH_PRELU(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);

            CHECK(init_conf());
            CHECK(init_reduction(engine));
            init_scratchpad();

            return status::success;
        }

        status_t init_conf();
        status_t init_reduction(impl::engine_t *engine);
        void init_scratchpad();

        static bool check_data_types(const memory_desc_wrapper &src,
                const memory_desc_wrapper &wei,
                const memory_desc_wrapper &dst) {
            for (const auto &mdw : {src, wei, dst}) {
                if (!is_supported_type(mdw.data_type())) return false;
            }

            return true;
        }

        sycl_prelu_conf_t conf_;
        bool reduce_diff_weights_ = false;
        memory_desc_t scratch_md_;
        std::shared_ptr<primitive_desc_t> reduction_pd_;
    };

    status_t init(impl::engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward(ctx);
    }

private:
    status_t execute_backward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    kernel_t kernel_;
    std::shared_ptr<impl::primitive_t> reduction_p_;
};

} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
