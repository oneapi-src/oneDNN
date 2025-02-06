/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#ifndef GPU_GENERIC_SYCL_REF_MATMUL_HPP
#define GPU_GENERIC_SYCL_REF_MATMUL_HPP

#include "gpu/generic/sycl/sycl_gpu_primitive.hpp"
#include "gpu/generic/sycl/sycl_io_helper.hpp"
#include "gpu/generic/sycl/sycl_post_ops.hpp"
#include "gpu/generic/sycl/sycl_primitive_conf.hpp"
#include "gpu/generic/sycl/sycl_q10n.hpp"
#include "gpu/generic/sycl/sycl_utils.hpp"
#include "gpu/gpu_matmul_pd.hpp"
#include "xpu/sycl/types.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

struct ref_matmul_t : public gpu::generic::sycl::primitive_t {
    using gpu::generic::sycl::primitive_t::primitive_t;

    struct pd_t : public gpu_matmul_pd_t {
        using gpu_matmul_pd_t::gpu_matmul_pd_t;

        DECLARE_COMMON_PD_T("dpcpp:ref:any", ref_matmul_t);

        status_t init(impl::engine_t *engine) {
            using namespace data_type;
            using sm = primitive_attr_t::skip_mask_t;

            const memory_desc_wrapper src_d(src_md());
            const memory_desc_wrapper weights_d(weights_md(0));
            const memory_desc_wrapper bias_d(weights_md(1));
            const memory_desc_wrapper dst_d(dst_md());

            VDISPATCH_MATMUL_SC(set_default_params(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_MATMUL_SC(attr_.set_default_formats(dst_md()),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_MATMUL(check_data_types(src_d, weights_d, dst_d),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_MATMUL(check_formats(src_d, weights_d, dst_d),
                    VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_MATMUL(
                    attr()->has_default_values(sm::scales_runtime
                            | sm::zero_points_runtime | sm::post_ops
                            | sm::dropout | sm::scales_runtime_data_type
                            | sm::zero_points_runtime_data_type),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_MATMUL(IMPLICATION(!attr()->scales_.has_default_values(),
                                     scales_ok()),
                    VERBOSE_UNSUPPORTED_SCALES_CFG);
            VDISPATCH_MATMUL(sycl_post_ops_t::post_ops_ok(attr()),
                    VERBOSE_UNSUPPORTED_POSTOP);
            VDISPATCH_MATMUL(md_dims_in_range(src_md()),
                    VERBOSE_OUT_OF_RANGE_DIMS, "src");
            VDISPATCH_MATMUL(md_dims_in_range(weights_md()),
                    VERBOSE_OUT_OF_RANGE_DIMS, "weights");

            init_conf();
            return status::success;
        }

        sycl_matmul_conf_t conf_;
        bool any_runtime_params_ = false;

        void init_rt_conf(sycl_matmul_conf_t &conf,
                const memory_desc_wrapper src_d,
                const memory_desc_wrapper weights_d,
                const memory_desc_wrapper dst_d,
                const memory_desc_wrapper bias_d) const;

    private:
        void init_conf();

        status_t set_default_params() {
            if (src_md_.format_kind == format_kind::any) {
                auto src_tag = utils::pick(ndims() - 2, format_tag::ab,
                        format_tag::abc, format_tag::abcd);
                CHECK(memory_desc_init_by_tag(src_md_, src_tag));
            }
            const memory_desc_wrapper src_d(src_md());
            if (src_d.is_blocking_desc()) {
                if (weights_md_.format_kind == format_kind::any) {
                    CHECK(memory_desc_init_by_blocking_desc(
                            weights_md_, src_d.blocking_desc()));
                }
                if (dst_md_.format_kind == format_kind::any) {
                    CHECK(memory_desc_init_by_blocking_desc(
                            dst_md_, src_d.blocking_desc()));
                }
            }
            const memory_desc_wrapper dst_d(dst_md());
            if (dst_d.is_blocking_desc()) {
                if (bias_md_.format_kind == format_kind::any) {
                    CHECK(memory_desc_init_by_blocking_desc(
                            bias_md_, dst_d.blocking_desc()));
                }
            }
            return status::success;
        }

        bool scales_ok() const {
            const std::vector<int> supported_args
                    = {DNNL_ARG_SRC_0, DNNL_ARG_WEIGHTS_0, DNNL_ARG_DST};

            const auto &scales = attr()->scales_;
            bool dt_ok = true;
            for (auto arg : supported_args) {
                if (!scales.get(arg).has_default_values()) {
                    dt_ok = dt_ok
                            && is_supported_type(scales.get_data_type(arg));
                }
            }
            return dt_ok && attr_scales_ok(supported_args);
        }

        static bool check_data_types(const memory_desc_wrapper &src,
                const memory_desc_wrapper &weights,
                const memory_desc_wrapper &dst) {
            using namespace data_type;

            const auto src_dt = src.data_type();
            const auto weights_dt = weights.data_type();
            const auto dst_dt = dst.data_type();

            for (auto t : {src_dt, weights_dt, dst_dt}) {
                if (!utils::one_of(t, f32, bf16, f16, s8, u8, s32))
                    return false;
            }

            return true;
        }

        static bool check_formats(const memory_desc_wrapper &src,
                const memory_desc_wrapper &weights,
                const memory_desc_wrapper &dst) {
            using namespace format_tag;

            for (const auto &mdw : {src, weights, dst}) {
                if (!mdw.is_plain()) { return false; }
            }
            return true;
        }
    };

    status_t init(impl::engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    kernel_t kernel_;
};

} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
