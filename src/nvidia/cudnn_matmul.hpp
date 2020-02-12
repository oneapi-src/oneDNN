/***************************************************************************
 *  Copyright 2020 Codeplay Software Limited
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  For your convenience, a copy of the License has been included in this
 *  repository.
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 **************************************************************************/

#ifndef CUDNN_MATMUL_HPP
#define CUDNN_MATMUL_HPP

#include <assert.h>

#include "common/matmul_pd.hpp"
#include "common/primitive.hpp"
#include "cudnn_matmul_executor.hpp"
#include "nvidia/cudnn_matmul_impl.hpp"
#include "nvidia/sycl_cuda_utils.hpp"

namespace dnnl {
namespace impl {
namespace cuda {

struct cudnn_matmul_t : public primitive_t {
    struct pd_t : public matmul_pd_t {
        using matmul_pd_t::matmul_pd_t;

        DECLARE_COMMON_PD_T("cuda:cudnn:any", cudnn_matmul_t);

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
            bool s8_case = utils::everyone_is(s8, src_dt, wei_dt)
                    && utils::one_of(dst_dt, s8, f32);

            bool ok = attr()->has_default_values(
                              smask_t::oscale_runtime | smask_t::post_ops)
                    && attr_oscale_ok() && attr_post_ops_ok()
                    && set_default_formats()
                    && (f32_case || f16_case || s8_case)
                    && IMPLICATION(with_bias(),
                            (IMPLICATION(f32_case, utils::one_of(bia_dt, f32))
                                    && IMPLICATION(f16_case,
                                            utils::one_of(bia_dt, f16, f32))
                                    && IMPLICATION(s8_case,
                                            utils::one_of(bia_dt, s8, f32))));

            if (!ok) return status::unimplemented;
            return status::success;
        }

    private:
        bool attr_oscale_ok() const {
            const auto &oscale = attr()->output_scales_;
            return oscale.mask_ == 0 || oscale.mask_ == (1 << (batched() + 1));
        }

        bool attr_post_ops_ok() const {
            using namespace primitive_kind;
            const auto &p = attr()->post_ops_;
            switch (p.len_) {
                case 0: return true;
                case 1: return p.contain(sum, 0) || p.contain(eltwise, 0);
                case 2: return p.contain(sum, 0) && p.contain(eltwise, 1);
                default: return false;
            }
        }
    };

    cudnn_matmul_t(const pd_t *apd) : primitive_t(apd) {}

    virtual status_t init(engine_t *engine) override {
        matmul_impl.reset(new cudnn_matmul_impl_t());
        const auto status
                = matmul_impl->init((matmul_pd_t *)primitive_t::pd().get());

        if (pd()->attr()->output_scales_.defined()) {
            output_scale_ = pd()->attr()->output_scales_.scales_;
        } else {
            // Only single-element scale is supported
            output_scale_ = new float;
        }

        const bool with_bias = matmul_impl->with_bias();
        const bool has_runtime_args = matmul_impl->has_runtime_params();
        const bool with_scratchpad = matmul_impl->with_scratchpad();

        if (with_scratchpad && has_runtime_args && with_bias) {
            executor.reset(new cudnn_matmul_scratch_runtime_args_bias_exec_t);
        } else if (with_scratchpad && has_runtime_args) {
            executor.reset(new cudnn_matmul_runtime_args_scratch_exec_t);
        } else if (has_runtime_args && with_bias) {
            executor.reset(new cudnn_matmul_runtime_args_bias_exec_t);
        } else if (has_runtime_args) {
            executor.reset(new cudnn_matmul_runtime_args_exec_t);
        } else if (with_bias && with_scratchpad) {
            executor.reset(new cudnn_matmul_bias_scratch_exec_t);
        } else if (with_scratchpad) {
            executor.reset(new cudnn_matmul_scratch_exec_t);
        } else if (with_bias) {
            executor.reset(new cudnn_matmul_bias_exec_t);
        } else if (!with_scratchpad && !has_runtime_args && !with_bias) {
            executor.reset(new cudnn_matmul_exec_t);
        } else {
            return status::unimplemented;
        }

        return status;
    }

    virtual status_t execute(const exec_ctx_t &ctx) const override;

    virtual ~cudnn_matmul_t() {
        if (!pd()->attr()->output_scales_.defined()) { delete output_scale_; }
    }

    std::shared_ptr<cudnn_matmul_impl_t> matmul_impl;
    std::shared_ptr<cudnn_matmul_exec_base_t> executor;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    float *output_scale_;
};

} // namespace cuda
} // namespace impl
} // namespace dnnl

#endif
