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

#ifndef GPU_SYCL_REF_SUM_HPP
#define GPU_SYCL_REF_SUM_HPP

#include "common/primitive.hpp"
#include "common/stream.hpp"
#include "gpu/generic/sycl/sycl_gpu_primitive.hpp"
#include "gpu/generic/sycl/sycl_io_helper.hpp"
#include "gpu/generic/sycl/sycl_primitive_conf.hpp"
#include "gpu/generic/sycl/sycl_q10n.hpp"
#include "gpu/gpu_sum_pd.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

struct ref_sum_t : public gpu::generic::sycl::primitive_t {
    using gpu::generic::sycl::primitive_t::primitive_t;

    struct pd_t : public gpu_sum_pd_t {
        using gpu_sum_pd_t::gpu_sum_pd_t;

        DECLARE_SUM_PD_T("dpcpp:ref:any", ref_sum_t);

        status_t init(impl::engine_t *engine) {
            using namespace format_tag;

            const memory_desc_wrapper dst_d(dst_md());
            VDISPATCH_SUM(is_supported_type(dst_d.data_type()),
                    VERBOSE_UNSUPPORTED_DT);
            // Block formats are not yet supported
            // Dimensions can not be > 6
            VDISPATCH_SUM(!(!dst_d.is_plain()
                                  || dst_d.ndims() > xpu::sycl::md_t::max_dims),
                    VERBOSE_UNSUPPORTED_TENSOR_LAYOUT, "dst");

            const int n = n_inputs();
            const auto &scales = attr()->scales_;
            for (auto i = 0; i < n; ++i) {
                const memory_desc_wrapper src_d(src_md(i));
                VDISPATCH_SUM(is_supported_type(src_d.data_type()),
                        VERBOSE_UNSUPPORTED_DT);

                // Block formats are not yet supported
                // Dimensions can not be > 6
                VDISPATCH_SUM(
                        !(!src_d.is_plain()
                                || src_d.ndims() > xpu::sycl::md_t::max_dims),
                        VERBOSE_UNSUPPORTED_TENSOR_LAYOUT, "src");

                VDISPATCH_SUM(!(!attr()->scales_.has_default_values()
                                      && !is_supported_type(
                                              scales.get(DNNL_ARG_SRC + i)
                                                      .data_type_)),
                        VERBOSE_UNSUPPORTED_ATTR);
            }

            VDISPATCH_SUM_SC(set_default_params(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_SUM(n <= DNNL_REF_SUM_MAX_NUM_TENSORS, VERBOSE_BAD_PARAM,
                    "n_inputs");

            return init_conf();
        }

        sycl_sum_conf_t conf_;

    private:
        status_t init_conf();

        inline bool equal(float in_value, float in_compare_to) {
            return std::fabs(in_value - in_compare_to) <= FLT_EPSILON;
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
