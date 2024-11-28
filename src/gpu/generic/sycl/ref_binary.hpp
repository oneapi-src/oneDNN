/*******************************************************************************
* Copyright 2022-2024 Intel Corporation
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

#ifndef GPU_GENERIC_SYCL_REF_BINARY_HPP
#define GPU_GENERIC_SYCL_REF_BINARY_HPP

#include "gpu/generic/sycl/sycl_gpu_primitive.hpp"
#include "gpu/generic/sycl/sycl_io_helper.hpp"
#include "gpu/generic/sycl/sycl_post_ops.hpp"
#include "gpu/generic/sycl/sycl_primitive_conf.hpp"
#include "gpu/generic/sycl/sycl_q10n.hpp"
#include "gpu/generic/sycl/sycl_utils.hpp"
#include "gpu/gpu_binary_pd.hpp"
#include "xpu/sycl/types.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

struct ref_binary_t : public gpu::generic::sycl::primitive_t {
    using gpu::generic::sycl::primitive_t::primitive_t;

    struct pd_t : public gpu_binary_pd_t {
        using gpu_binary_pd_t::gpu_binary_pd_t;

        DECLARE_COMMON_PD_T("dpcpp:ref:any", ref_binary_t);

        status_t init(impl::engine_t *engine) {
            using namespace data_type;
            using sm = primitive_attr_t::skip_mask_t;

            const memory_desc_wrapper src0_d(src_md(0));
            const memory_desc_wrapper src1_d(src_md(1));
            const memory_desc_wrapper dst_d(dst_md());

            VDISPATCH_BINARY_SC(
                    set_default_params(), VERBOSE_UNSUPPORTED_FORMAT_KIND);
            VDISPATCH_BINARY_SC(attr_.set_default_formats(dst_md()),
                    VERBOSE_UNSUPPORTED_TAG_S, "dst");
            VDISPATCH_BINARY(check_data_types(src0_d, src1_d, dst_d),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_BINARY(check_formats(src0_d, src1_d, dst_d),
                    VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_BINARY(attr()->has_default_values(
                                     sm::scales_runtime | sm::post_ops),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_BINARY(!is_ternary_op(), VERBOSE_BAD_ALGORITHM);
            VDISPATCH_BINARY(IMPLICATION(!attr()->scales_.has_default_values(),
                                     scales_ok()),
                    VERBOSE_UNSUPPORTED_SCALES_CFG);
            VDISPATCH_BINARY(sycl_post_ops_t::post_ops_ok(attr()),
                    VERBOSE_UNSUPPORTED_POSTOP);
            VDISPATCH_BINARY(md_dims_in_range(src_md(0)),
                    VERBOSE_OUT_OF_RANGE_DIMS, "src(0)");
            VDISPATCH_BINARY(md_dims_in_range(src_md(1)),
                    VERBOSE_OUT_OF_RANGE_DIMS, "src(1)");
            VDISPATCH_BINARY(md_dims_in_range(dst_md()),
                    VERBOSE_OUT_OF_RANGE_DIMS, "dst");

            return init_conf();
        }

        sycl_binary_conf_t conf_;

    private:
        status_t init_conf();

        bool scales_ok() const {
            const std::vector<int> supported_args
                    = {DNNL_ARG_SRC_0, DNNL_ARG_SRC_1};

            const auto &scales = attr()->scales_;
            bool dt_ok = true;
            for (auto arg : supported_args) {
                auto &s = scales.get(arg);
                dt_ok = dt_ok && is_supported_type(s.data_type_);
            }
            return dt_ok && attr_scales_ok(supported_args);
        }

        static bool check_data_types(const memory_desc_wrapper &src0,
                const memory_desc_wrapper &src1,
                const memory_desc_wrapper &dst) {
            using namespace data_type;

            const auto src0_dt = src0.data_type();
            const auto src1_dt = src1.data_type();
            const auto dst_dt = dst.data_type();

            for (auto t : {src0_dt, src1_dt, dst_dt}) {
                if (!utils::one_of(t, f32, bf16, f16, s8, u8, s32))
                    return false;
            }

            return true;
        }

        static bool check_formats(const memory_desc_wrapper &src0,
                const memory_desc_wrapper &src1,
                const memory_desc_wrapper &dst) {
            using namespace format_tag;

            for (const auto &mdw : {src0, src1, dst}) {
                if (!(mdw.is_plain() || mdw.matches_tag(format_tag::Ab32a)
                            || mdw.matches_tag(format_tag::aBc32b)))
                    return false;
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
