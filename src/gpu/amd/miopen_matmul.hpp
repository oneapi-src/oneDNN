/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
* Copyright 2020 Codeplay Software Limited
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

#ifndef GPU_AMD_MIOPEN_MATMUL_HPP
#define GPU_AMD_MIOPEN_MATMUL_HPP

#include <assert.h>

#include "common/matmul_pd.hpp"
#include "common/primitive.hpp"

#include "gpu/amd/miopen_matmul_executor.hpp"
#include "gpu/amd/miopen_matmul_impl.hpp"
#include "gpu/amd/sycl_hip_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

struct miopen_matmul_t : public primitive_t {
    using primitive_t::primitive_t;
    struct pd_t : public matmul_pd_t {
        using matmul_pd_t::matmul_pd_t;

        DECLARE_COMMON_PD_T("hip:miopen:any", miopen_matmul_t);

        status_t init(engine_t *) {
            using namespace data_type;
            using smask_t = primitive_attr_t::skip_mask_t;

            data_type_t src_dt = src_md()->data_type;
            data_type_t dst_dt = dst_md()->data_type;
            data_type_t wei_dt = weights_md(0)->data_type;
            data_type_t bia_dt
                    = with_bias() ? weights_md(1)->data_type : data_type::f32;

            bool f32_case = utils::everyone_is(f32, src_dt, wei_dt, dst_dt);
            bool f16_case = utils::everyone_is(f16, src_dt, wei_dt, dst_dt);
            bool s8_case
                    = utils::everyone_is(s8, src_dt, wei_dt) && (dst_dt == s32);

            bool bf16_case = batched()
                    ? utils::everyone_is(bf16, src_dt, wei_dt, dst_dt)
                    : utils::everyone_is(bf16, src_dt, wei_dt)
                            && utils::one_of(dst_dt, f32, bf16);

            bool ok = blocking_ok()
                    && attr()->has_default_values(smask_t::post_ops)
                    && attr_oscale_ok() && attr_post_ops_ok(s8_case)
                    && set_default_formats()
                    && (f32_case || f16_case || s8_case || bf16_case)
                    && IMPLICATION(with_bias(),
                            (IMPLICATION(f32_case, (bia_dt == f32))
                                    && IMPLICATION(f16_case, (bia_dt == f16))
                                    && IMPLICATION(s8_case, (bia_dt == s32))
                                    && IMPLICATION(bf16_case,
                                            dst_dt == f32 && bia_dt == f32)));

            if (!ok) return status::unimplemented;

            if (src_md()->ndims > 3) return status::unimplemented;

            return status::success;
        }

    private:
        bool attr_oscale_ok() const {
            const auto &oscale = attr()->output_scales_;
            return oscale.mask_ == 0;
        }

        bool attr_post_ops_ok(bool s8_case) const {
            using namespace primitive_kind;
            const auto &p = attr()->post_ops_;
            const int eltwise_idx = p.find(eltwise);
            const int sum_idx = p.find(sum);

            if (eltwise_idx != -1) {
                using namespace alg_kind;
                const bool ok = utils::one_of(p.entry_[eltwise_idx].eltwise.alg,
                        eltwise_relu, eltwise_tanh, eltwise_elu,
                        eltwise_logistic);
                if (!ok) return false;
            }

            if ((sum_idx != -1) && s8_case) {
                float sum_scale = p.entry_[sum_idx].sum.scale;
                if (sum_scale != (int)sum_scale) return false;
            }

            switch (p.len()) {
                case 0: return true;
                case 1: return p.contain(sum, 0) || p.contain(eltwise, 0);
                case 2: return p.contain(sum, 0) && p.contain(eltwise, 1);
                default: return false;
            }
        }

        bool blocking_ok() const {
            std::vector<const memory_desc_t *> mds
                    = {src_md(), dst_md(), weights_md(0)};
            if (with_bias()) mds.push_back(weights_md(1));
            for (const memory_desc_t *md : mds) {
                memory_desc_wrapper mdw(md);
                if (mdw.is_blocking_desc()) {
                    if (mdw.blocking_desc().inner_nblks != 0) { return false; }
                }
            }
            return true;
        }
    };

    status_t init(engine_t *engine) override {
        matmul_impl_.reset(new miopen_matmul_impl_t());
        const auto status
                = matmul_impl_->init((matmul_pd_t *)primitive_t::pd().get());

        const bool with_bias = matmul_impl_->with_bias();
        const bool has_runtime_args = matmul_impl_->has_runtime_params();
        const bool with_scratchpad = matmul_impl_->with_scratchpad();

        if (with_scratchpad && has_runtime_args && with_bias) {
            executor_.reset(new miopen_matmul_scratch_runtime_args_bias_exec_t);
        } else if (with_scratchpad && has_runtime_args) {
            executor_.reset(new miopen_matmul_runtime_args_scratch_exec_t);
        } else if (has_runtime_args && with_bias) {
            executor_.reset(new miopen_matmul_runtime_args_bias_exec_t);
        } else if (has_runtime_args) {
            executor_.reset(new miopen_matmul_runtime_args_exec_t);
        } else if (with_bias && with_scratchpad) {
            executor_.reset(new miopen_matmul_bias_scratch_exec_t);
        } else if (with_scratchpad) {
            executor_.reset(new miopen_matmul_scratch_exec_t);
        } else if (with_bias) {
            executor_.reset(new miopen_matmul_bias_exec_t);
        } else if (!with_scratchpad && !has_runtime_args && !with_bias) {
            executor_.reset(new miopen_matmul_exec_t);
        } else {
            return status::unimplemented;
        }

        return status;
    }

    status_t execute(const exec_ctx_t &ctx) const override;

    virtual ~miopen_matmul_t() {}

    std::shared_ptr<miopen_matmul_impl_t> matmul_impl_;
    std::shared_ptr<miopen_matmul_exec_base_t> executor_;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace amd
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
