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

#ifndef GPU_SYCL_REF_SUM_MANY_INPUTS_HPP
#define GPU_SYCL_REF_SUM_MANY_INPUTS_HPP

#include "common/primitive.hpp"
#include "gpu/generic/sycl/sycl_gpu_primitive.hpp"
#include "gpu/generic/sycl/sycl_io_helper.hpp"
#include "gpu/generic/sycl/sycl_post_ops.hpp"
#include "gpu/generic/sycl/sycl_primitive_conf.hpp"
#include "gpu/generic/sycl/sycl_q10n.hpp"
#include "gpu/gpu_sum_pd.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

struct ref_sum_many_inputs_t : public gpu::generic::sycl::primitive_t {
    using gpu::generic::sycl::primitive_t::primitive_t;

    struct pd_t : public gpu_sum_pd_t {
        using gpu_sum_pd_t::gpu_sum_pd_t;

        DECLARE_SUM_PD_t("dpcpp:ref:sum_many_inputs", ref_sum_many_inputs_t);

        status_t init(impl::engine_t *engine) {
            using namespace data_type;

            const memory_desc_wrapper dst_d(dst_md());

            const int n = n_inputs();
            VDISPATCH_SUM_SC(set_default_params(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_SUM(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            // prevent inf recursion
            VDISPATCH_SUM(n > DNNL_REF_SUM_MAX_NUM_TENSORS, VERBOSE_BAD_PARAM,
                    "n_inputs");

            // the first kernel handles up to 8 inputs and remaining ones up to 7
            const int n_kernels = n == 1
                    ? 1
                    : utils::div_up(n - 1, DNNL_REF_SUM_MAX_NUM_TENSORS - 1);
            base_pds_.resize(n_kernels);
            int in_arg_offset = 0;
            int n_remaining = n;
            for (auto i = 0; i < n_kernels; ++i) {
                bool pass_in_dst = i > 0;
                int max_n_child_inputs
                        = DNNL_REF_SUM_MAX_NUM_TENSORS - pass_in_dst;
                int n_child_inputs = std::min(n_remaining, max_n_child_inputs);
                const memory_desc_t *src[DNNL_REF_SUM_MAX_NUM_TENSORS];
                if (pass_in_dst) { src[0] = dst_md(); }
                for (int j = 0; j < n_child_inputs; j++) {
                    src[j + pass_in_dst] = src_md(j + in_arg_offset);
                }
                in_arg_offset += n_child_inputs;
                n_remaining -= n_child_inputs;

                primitive_attr_t r_attr;
                CHECK(dnnl_sum_primitive_desc_create(&base_pds_[i], engine,
                        dst_md(), n_child_inputs + pass_in_dst, scales(), src,
                        &r_attr));
            }

            return init_conf();
        }

        sycl_sum_conf_t conf_;
        std::vector<primitive_desc_iface_t *> base_pds_;

    private:
        status_t init_conf();
    };

    status_t init(impl::engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::vector<std::shared_ptr<impl::primitive_t>> base_prims_;
};

} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
