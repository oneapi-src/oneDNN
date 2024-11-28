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
#ifndef GPU_GENERIC_SYCL_REF_SOFTMAX_HPP
#define GPU_GENERIC_SYCL_REF_SOFTMAX_HPP

#include "gpu/generic/sycl/sycl_gpu_primitive.hpp"
#include "gpu/generic/sycl/sycl_post_ops.hpp"
#include "gpu/generic/sycl/sycl_primitive_conf.hpp"
#include "gpu/generic/sycl/sycl_utils.hpp"
#include "gpu/gpu_softmax_pd.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

struct ref_sycl_softmax_fwd_t : public gpu::generic::sycl::primitive_t {
    using gpu::generic::sycl::primitive_t::primitive_t;

    struct pd_t : public gpu_softmax_fwd_pd_t {
        using gpu_softmax_fwd_pd_t::gpu_softmax_fwd_pd_t;

        DECLARE_COMMON_PD_T("dpcpp:ref:any", ref_sycl_softmax_fwd_t);

        status_t init(impl::engine_t *engine) {
            using sm = primitive_attr_t::skip_mask_t;

            VDISPATCH_SOFTMAX(is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_SOFTMAX(check_data_types(src_md()->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_SOFTMAX(check_data_types(dst_md()->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_SOFTMAX(
                    (src_md(0)->format_desc.blocking.inner_nblks == 0),
                    VERBOSE_UNSUPPORTED_FORMAT_KIND);
            VDISPATCH_SOFTMAX(attr()->has_default_values(
                                      sm::scales_runtime | sm::post_ops),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_SOFTMAX(attr_oscale_ok(), VERBOSE_UNSUPPORTED_SCALES_CFG);
            VDISPATCH_SOFTMAX(sycl_post_ops_t::post_ops_ok(attr(), true, false),
                    VERBOSE_UNSUPPORTED_POSTOP);
            VDISPATCH_SOFTMAX_SC(
                    set_default_formats(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_SOFTMAX_SC(attr_.set_default_formats(dst_md()),
                    VERBOSE_UNSUPPORTED_TAG_S, "dst");
            VDISPATCH_SOFTMAX(check_formats(src_md(), dst_md()),
                    VERBOSE_UNSUPPORTED_TAG_S, "src,dst");
            VDISPATCH_SOFTMAX(md_dims_in_range(src_md()),
                    VERBOSE_OUT_OF_RANGE_DIMS, "src");
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

        static bool check_formats(const memory_desc_wrapper &src,
                const memory_desc_wrapper &dst) {
            for (const auto &mdw : {src, dst}) {
                if (!mdw.is_plain()) return false;
            }

            return true;
        }
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

struct ref_sycl_softmax_bwd_t : public gpu::generic::sycl::primitive_t {
    using gpu::generic::sycl::primitive_t::primitive_t;

    struct pd_t : public gpu_softmax_bwd_pd_t {
        using gpu_softmax_bwd_pd_t::gpu_softmax_bwd_pd_t;

        DECLARE_COMMON_PD_T("dpcpp:ref:any", ref_sycl_softmax_bwd_t);

        status_t init(impl::engine_t *engine) {
            using namespace data_type;
            VDISPATCH_SOFTMAX(!is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_SOFTMAX(
                    utils::one_of(dst_md()->data_type, f32, bf16, f16),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_SOFTMAX(
                    utils::one_of(diff_src_md()->data_type, f32, bf16, f16),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_SOFTMAX(
                    (dst_md(0)->format_desc.blocking.inner_nblks == 0),
                    VERBOSE_UNSUPPORTED_FORMAT_KIND);
            VDISPATCH_SOFTMAX(dst_md()->data_type == diff_dst_md()->data_type,
                    VERBOSE_INCONSISTENT_DT, "dst", "diff_dst");
            VDISPATCH_SOFTMAX(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_SOFTMAX_SC(
                    set_default_formats(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_SOFTMAX(memory_desc_wrapper(diff_src_md()).is_plain(),
                    VERBOSE_UNSUPPORTED_TENSOR_LAYOUT, "diff_src");
            VDISPATCH_SOFTMAX(memory_desc_wrapper(diff_dst_md()).is_plain(),
                    VERBOSE_UNSUPPORTED_TENSOR_LAYOUT, "diff_dst");
            VDISPATCH_SOFTMAX(memory_desc_wrapper(dst_md()).is_plain(),
                    VERBOSE_UNSUPPORTED_TENSOR_LAYOUT, "dst");
            VDISPATCH_SOFTMAX(md_dims_in_range(diff_dst_md()),
                    VERBOSE_OUT_OF_RANGE_DIMS, "diff_dst");
            return init_conf();
        }

        sycl_softmax_conf_t conf_;
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
