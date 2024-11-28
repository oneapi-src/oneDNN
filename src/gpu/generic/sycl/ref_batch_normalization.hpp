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

#ifndef GPU_GENERIC_SYCL_REF_BATCH_NORMALIZATION_HPP
#define GPU_GENERIC_SYCL_REF_BATCH_NORMALIZATION_HPP

#include "common/c_types_map.hpp"
#include "common/utils.hpp"
#include "gpu/generic/sycl/sycl_gpu_primitive.hpp"
#include "gpu/generic/sycl/sycl_io_helper.hpp"
#include "gpu/generic/sycl/sycl_post_ops.hpp"
#include "gpu/generic/sycl/sycl_primitive_conf.hpp"
#include "gpu/generic/sycl/sycl_q10n.hpp"
#include "gpu/generic/sycl/sycl_utils.hpp"
#include "gpu/gpu_batch_normalization_pd.hpp"
#include "xpu/sycl/types.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

struct ref_batch_normalization_fwd_t : public gpu::generic::sycl::primitive_t {
    using gpu::generic::sycl::primitive_t::primitive_t;

    struct pd_t : public gpu_batch_normalization_fwd_pd_t {
        using gpu_batch_normalization_fwd_pd_t::
                gpu_batch_normalization_fwd_pd_t;
        DECLARE_COMMON_PD_T("dpcpp:ref:any", ref_batch_normalization_fwd_t);

        status_t init(impl::engine_t *engine) {
            using namespace data_type;

            const memory_desc_wrapper data_d(src_md(0));
            const memory_desc_wrapper dst_d(dst_md(0));

            VDISPATCH_BNORM(is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_BNORM((src_md(0)->format_desc.blocking.inner_nblks == 0),
                    VERBOSE_UNSUPPORTED_FORMAT_KIND);
            VDISPATCH_BNORM(
                    (utils::everyone_is(
                             f32, src_md()->data_type, dst_md()->data_type)
                            || utils::everyone_is(bf16, src_md()->data_type,
                                    dst_md()->data_type)
                            || utils::everyone_is(f16, src_md()->data_type,
                                    dst_md()->data_type)
                            || utils::everyone_is(s8, src_md()->data_type,
                                    dst_md()->data_type)),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_BNORM(
                    check_scale_shift_data_type(), VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_BNORM((attr()->has_default_values()
                                    || with_relu_post_op(is_training())),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_BNORM(
                    set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_BNORM(memory_desc_wrapper(src_md(0))
                            == memory_desc_wrapper(dst_md(0)),
                    VERBOSE_INCONSISTENT_MDS, "src", "dst");
            VDISPATCH_BNORM(md_dims_in_range(src_md()),
                    VERBOSE_OUT_OF_RANGE_DIMS, "src");

            VDISPATCH_BNORM(!(src_md(0)->data_type == s8 && !stats_is_src()),
                    VERBOSE_UNSUPPORTED_DT);

            if (is_training() && (fuse_norm_relu() || fuse_norm_add_relu()))
                init_default_ws(8);
            return init_conf();
        }
        status_t init_conf();
        sycl_batch_normalization_conf_t conf_;
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

struct ref_batch_normalization_bwd_t : public gpu::generic::sycl::primitive_t {
    using gpu::generic::sycl::primitive_t::primitive_t;

    struct pd_t : public gpu_batch_normalization_bwd_pd_t {
        using gpu_batch_normalization_bwd_pd_t::
                gpu_batch_normalization_bwd_pd_t;

        DECLARE_COMMON_PD_T("dpcpp:ref:any", ref_batch_normalization_bwd_t);

        status_t init(impl::engine_t *engine) {
            using namespace data_type;

            const memory_desc_wrapper data_d(src_md(0));
            const memory_desc_wrapper diff_data_d(diff_src_md(0));
            const memory_desc_wrapper stat_d(src_md(1));
            const memory_desc_wrapper diff_data_scaleshift_d(
                    diff_weights_md(0));
            const memory_desc_wrapper diff_dst_d(diff_dst_md(0));
            const memory_desc_wrapper var_d(src_md(2));

            VDISPATCH_BNORM(!is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_BNORM((src_md(0)->format_desc.blocking.inner_nblks == 0),
                    VERBOSE_UNSUPPORTED_FORMAT_KIND);
            VDISPATCH_BNORM(
                    (diff_dst_md(0)->format_desc.blocking.inner_nblks == 0),
                    VERBOSE_UNSUPPORTED_FORMAT_KIND);
            VDISPATCH_BNORM(
                    (utils::everyone_is(f32, src_md()->data_type,
                             diff_dst_md()->data_type, diff_src_md()->data_type)
                            || utils::everyone_is(bf16, src_md()->data_type,
                                    diff_dst_md()->data_type,
                                    diff_src_md()->data_type)
                            || utils::everyone_is(f16, src_md()->data_type,
                                    diff_dst_md()->data_type,
                                    diff_src_md()->data_type)),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_BNORM(
                    check_scale_shift_data_type(), VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_BNORM(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_BNORM(
                    set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_BNORM(memory_desc_wrapper(diff_src_md())
                            == memory_desc_wrapper(diff_dst_md()),
                    VERBOSE_INCONSISTENT_MDS, "diff_src", "diff_dst");
            VDISPATCH_BNORM(md_dims_in_range(diff_src_md()),
                    VERBOSE_OUT_OF_RANGE_DIMS, "diff_src");

            if (fuse_norm_relu() || fuse_norm_add_relu()) {
                init_default_ws(8);
                VDISPATCH_BNORM(compare_ws(hint_fwd_pd_), VERBOSE_WS_INIT);
            }
            return init_conf();
        }

        status_t init_conf();
        sycl_batch_normalization_conf_t conf_;
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
