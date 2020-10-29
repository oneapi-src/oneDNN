/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#ifndef GPU_OCL_GEMM_MATMUL_HPP
#define GPU_OCL_GEMM_MATMUL_HPP

#include "common/gemm_utils.hpp"
#include "common/primitive.hpp"
#include "common/primitive_iterator.hpp"
#include "gpu/gemm/gpu_gemm.hpp"
#include "gpu/gpu_matmul_pd.hpp"
#include "gpu/gpu_primitive.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

struct gemm_matmul_t : public gpu_primitive_t {
    struct pd_t : public gpu_matmul_pd_t {
        pd_t(const matmul_desc_t *adesc, const primitive_attr_t *attr,
                const matmul_pd_t *hint_pd)
            : gpu_matmul_pd_t(adesc, attr, hint_pd) {}

        pd_t(const pd_t &other)
            : gpu_matmul_pd_t(other), gemm_pd_(other.gemm_pd_->clone()) {}

        DECLARE_COMMON_PD_T("ocl:gemm:any", gemm_matmul_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            auto prepare_gemm_attributes
                    = [](const primitive_attr_t &attr,
                              primitive_attr_t &gemm_attr) {
                          if (!attr.output_scales_.has_default_values()) {
                              gemm_attr.output_scales_.copy_from(
                                      attr.output_scales_);
                          }

                          auto map_gemm_zp = [&attr, &gemm_attr](
                                                     int arg, int gemm_arg) {
                              if (!attr.zero_points_.has_default_values(arg)) {
                                  dim_t count = 0;
                                  int mask = 0;
                                  const int *zero_points = nullptr;
                                  attr.zero_points_.get(
                                          arg, &count, &mask, &zero_points);
                                  gemm_attr.zero_points_.set(
                                          gemm_arg, count, mask, zero_points);
                              }
                          };

                          if (!attr.zero_points_.has_default_values()) {
                              map_gemm_zp(DNNL_ARG_SRC, DNNL_ARG_B);
                              map_gemm_zp(DNNL_ARG_WEIGHTS, DNNL_ARG_A);
                              map_gemm_zp(DNNL_ARG_DST, DNNL_ARG_C);
                          }

                          if (!attr.post_ops_.has_default_values()) {
                              gemm_attr.post_ops_ = attr.post_ops_;
                          }
                      };

            primitive_attr_t gemm_attr;
            prepare_gemm_attributes(*attr(), gemm_attr);

            const auto acc_dt = desc()->accum_data_type;

            // We create a gemm_pd and resolve 'any' desc
            // by querying gemm_pd
            bool ok = status::success
                    == create_gemm_pd(gemm_pd_, engine, src_md(), weights_md(),
                            dst_md(), weights_md(1), acc_dt, &gemm_attr);
            if (!ok) return status::unimplemented;

            ok = status::success == set_default_params();
            if (!ok) return status::unimplemented;

            init_scratchpad();

            return status::success;
        }

        std::unique_ptr<primitive_desc_t> gemm_pd_;

    private:
        status_t set_default_params() {
            src_md_ = *gemm_pd_->arg_md(DNNL_ARG_SRC_0);
            weights_md_ = *gemm_pd_->arg_md(DNNL_ARG_SRC_1);
            bias_md_ = *gemm_pd_->arg_md(DNNL_ARG_BIAS);
            dst_md_ = *gemm_pd_->arg_md(DNNL_ARG_DST);
            return status::success;
        }

        void init_scratchpad() {
            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.book(memory_tracking::names::key_nested,
                    gemm_pd_->scratchpad_registry());
        }
    };

    gemm_matmul_t(const pd_t *apd) : gpu_primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        status_t gemm_status = pd()->gemm_pd_->create_primitive(gemm_, engine);
        return gemm_status;
    }

    status_t execute(const exec_ctx_t &ctx) const override;

protected:
    primitive_list_t nested_primitives() const override {
        return {gemm_.get()};
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::shared_ptr<primitive_t> gemm_;
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
