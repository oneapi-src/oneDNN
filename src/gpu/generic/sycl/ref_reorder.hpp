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

#ifndef GPU_GENERIC_SYCL_REF_REORDER_HPP
#define GPU_GENERIC_SYCL_REF_REORDER_HPP

#include "gpu/generic/sycl/sycl_gpu_primitive.hpp"
#include "gpu/generic/sycl/sycl_io_helper.hpp"
#include "gpu/generic/sycl/sycl_post_ops.hpp"
#include "gpu/generic/sycl/sycl_primitive_conf.hpp"
#include "gpu/generic/sycl/sycl_q10n.hpp"
#include "gpu/generic/sycl/sycl_utils.hpp"
#include "gpu/gpu_reorder_pd.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

struct ref_reorder_t : public gpu::generic::sycl::primitive_t {
    using gpu::generic::sycl::primitive_t::primitive_t;

    struct pd_t : public gpu_reorder_pd_t {
        using gpu_reorder_pd_t::gpu_reorder_pd_t;

        DECLARE_COMMON_PD_T("dpcpp:ref:any", ref_reorder_t);

        status_t init(impl::engine_t *engine, impl::engine_t *src_engine,
                impl::engine_t *dst_engine) {
            using namespace data_type;
            using sm = primitive_attr_t::skip_mask_t;

            const memory_desc_wrapper src_d(src_md());
            const memory_desc_wrapper dst_d(dst_md());

            VDISPATCH_REORDER(src_engine == dst_engine,
                    "engine mismatch for src and dst tensors");
            VDISPATCH_REORDER(!src_d.has_runtime_dims_or_strides(),
                    VERBOSE_RUNTIMEDIM_UNSUPPORTED);
            VDISPATCH_REORDER(!dst_d.has_runtime_dims_or_strides(),
                    VERBOSE_RUNTIMEDIM_UNSUPPORTED);
            VDISPATCH_REORDER(
                    check_data_types(src_d, dst_d), VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_REORDER(
                    check_formats(src_d, dst_d), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_REORDER(attr()->has_default_values(
                                      sm::scales_runtime | sm::post_ops),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_REORDER(IMPLICATION(!attr()->scales_.has_default_values(),
                                      scales_ok()),
                    VERBOSE_UNSUPPORTED_SCALES_CFG);
            VDISPATCH_REORDER(sycl_post_ops_t::post_ops_ok(attr()),
                    VERBOSE_UNSUPPORTED_POSTOP);
            VDISPATCH_REORDER(md_dims_in_range(dst_md()),
                    VERBOSE_OUT_OF_RANGE_DIMS, "dst");

            return init_conf();
        }

        sycl_reorder_conf_t conf_;

    private:
        DECLARE_GPU_REORDER_CREATE();

        status_t init_conf();

        static bool check_data_types(const memory_desc_wrapper &src,
                const memory_desc_wrapper &dst) {
            for (const auto &mdw : {src, dst}) {
                if (!is_supported_type(mdw.data_type())) return false;
            }

            return true;
        }

        static bool check_formats(const memory_desc_wrapper &src,
                const memory_desc_wrapper &dst) {
            using namespace format_tag;

            for (const auto &mdw : {src, dst}) {
                if (!mdw.is_plain()) { return false; }
            }
            return true;
        }

        bool scales_ok() const {
            const std::vector<int> supported_args
                    = {DNNL_ARG_SRC, DNNL_ARG_DST};

            const auto &scales = attr()->scales_;
            for (auto arg : supported_args) {
                auto dt = scales.get(arg).data_type_;
                if (!is_supported_type(dt)) { return false; }
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
