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

#ifndef GPU_OCL_GEMM_REF_GEMM_HPP
#define GPU_OCL_GEMM_REF_GEMM_HPP

#include "gpu/gpu_gemm_pd.hpp"
#include "gpu/ocl/gemm/ocl_gemm.hpp"
#include "gpu/ocl/gemm/ocl_gemm_utils.hpp"
#include "gpu/ocl/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

struct ref_gemm_t : public ocl_gemm_t {
    struct pd_t : public gpu_gemm_pd_t {
        using gpu_gemm_pd_t::gpu_gemm_pd_t;

        DECLARE_COMMON_PD_T("ocl:ref:any", ref_gemm_t);

        status_t init() {
            using namespace data_type;
            using smask_t = primitive_attr_t::skip_mask_t;

            const auto a_dt = desc()->a_type;
            const auto b_dt = desc()->b_type;
            const auto c_dt = desc()->c_type;
            const auto acc_dt = desc()->acc_type;
            const auto bia_dt = desc()->bias_type;

            bool ok = IMPLICATION(acc_dt == s32, attr()->zero_points_.common())
                    && IMPLICATION(acc_dt != s32,
                            attr()->zero_points_.has_default_values())
                    && attr()->has_default_values(smask_t::oscale_runtime
                            | smask_t::zero_points_runtime | smask_t::post_ops)
                    && attr_oscale_ok() && attr_zp_ok() && attr_post_ops_ok()
                    && ((utils::one_of(a_dt, u8, s8)
                                && utils::one_of(b_dt, u8, s8)
                                && utils::one_of(c_dt, f32, s8, u8, s32)
                                && IMPLICATION(with_bias(),
                                        utils::one_of(
                                                bia_dt, f32, u8, s8, s32)))
                            || (utils::everyone_is(f32, a_dt, b_dt, c_dt)
                                    && IMPLICATION(with_bias(), bia_dt == f32))
                            || (utils::everyone_is(f16, a_dt, b_dt, c_dt)
                                    && IMPLICATION(with_bias(), bia_dt == f16))
                            || (utils::everyone_is(bf16, a_dt, b_dt)
                                    && utils::one_of(c_dt, bf16, f32)
                                    && IMPLICATION(with_bias(),
                                            utils::one_of(bia_dt, bf16, f32))));

            if (!ok) return status::unimplemented;

            return status::success;
        }

        bool attr_oscale_ok() const {
            const auto &oscale = attr()->output_scales_;
            return oscale.mask_ == 0;
        }

        bool attr_zp_ok() const {
            return this->attr()->zero_points_.common(DNNL_ARG_A)
                    && this->attr()->zero_points_.common(DNNL_ARG_B)
                    && this->attr()->zero_points_.common(DNNL_ARG_C);
        }

        bool attr_post_ops_ok() const {
            using namespace primitive_kind;
            using namespace data_type;
            const auto &p = attr()->post_ops_;
            switch (p.len_) {
                case 0: return true;
                case 1: return p.contain(sum, 0) || p.contain(eltwise, 0);
                case 2: return p.contain(sum, 0) && p.contain(eltwise, 1);
                default: return false;
            }
        }

        bool with_eltwise(int position) const {
            return attr()->post_ops_.contain(primitive_kind::eltwise, position);
        }

        float eltwise_alpha() const {
            const int eltwise_idx
                    = attr()->post_ops_.find(primitive_kind::eltwise);
            return with_eltwise(0) || with_eltwise(1)
                    ? attr()->post_ops_.entry_[eltwise_idx].eltwise.alpha
                    : 1.0f;
        }

        float eltwise_beta() const {
            const int eltwise_idx
                    = attr()->post_ops_.find(primitive_kind::eltwise);
            return with_eltwise(0) || with_eltwise(1)
                    ? attr()->post_ops_.entry_[eltwise_idx].eltwise.beta
                    : 0.0f;
        }

        alg_kind_t eltwise_alg_kind() const {
            const int eltwise_idx
                    = attr()->post_ops_.find(primitive_kind::eltwise);
            return with_eltwise(0) || with_eltwise(1)
                    ? attr()->post_ops_.entry_[eltwise_idx].eltwise.alg
                    : alg_kind::undef;
        }

        float sum_scale() const {
            using namespace primitive_kind;
            const auto &p = attr()->post_ops_;
            return p.contain(sum, 0) ? p.entry_[0].sum.scale : 0.f;
        }

        bool with_bias() const { return desc()->bias_type != data_type::undef; }
    };

    status_t init() override {
        using namespace gemm_utils;

        const auto attr = pd()->attr();
        auto e = pd()->engine();

        status_t s = status::success;

        s = prepare_zero_points(attr, e, DNNL_ARG_A, a0_mem_storage_);
        if (s != status::success) return s;

        s = prepare_zero_points(attr, e, DNNL_ARG_B, b0_mem_storage_);
        if (s != status::success) return s;

        s = prepare_zero_points(attr, e, DNNL_ARG_C, c0_mem_storage_);
        if (s != status::success) return s;

        s = prepare_scales(attr, e, s_mem_storage_);
        if (s != status::success) return s;

        auto *compute_engine
                = utils::downcast<compute::compute_engine_t *>(engine());
        compute::kernel_ctx_t kernel_ctx;

        kernel_ctx.define_int("WITH_BIAS", pd()->with_bias());
        kernel_ctx.define_int(
                "NON_DEFAULT_ATTRS", !pd()->attr()->has_default_values());
        kernel_ctx.define_int(
                "DO_SUM", attr->post_ops_.contain(primitive_kind::sum, 0));
        kernel_ctx.define_int(
                "WITH_ELTWISE", pd()->with_eltwise(0) || pd()->with_eltwise(1));

        const auto d = pd()->desc();
        kernel_ctx.set_data_type(d->c_type);
        def_postops(kernel_ctx, pd()->eltwise_alg_kind());

        const auto bias_type = d->bias_type != data_type::undef
                ? d->bias_type
                : data_type::f32;
        def_data_type(kernel_ctx, d->a_type, "A");
        def_data_type(kernel_ctx, d->b_type, "B");
        def_data_type(kernel_ctx, d->c_type, "C");
        def_data_type(kernel_ctx, d->acc_type, "ACC");
        def_data_type(kernel_ctx, bias_type, "BIAS");
        compute_engine->create_kernel(&kernel_, "ref_gemm", kernel_ctx);
        if (!kernel_) return status::runtime_error;

        return status::success;
    }

    ref_gemm_t(const pd_t *apd) : ocl_gemm_t(apd) {}

    virtual status_t execute(const gemm_exec_ctx_t &ctx) const override;

    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }

private:
    compute::kernel_t kernel_;
    std::unique_ptr<memory_storage_t> a0_mem_storage_;
    std::unique_ptr<memory_storage_t> b0_mem_storage_;
    std::unique_ptr<memory_storage_t> c0_mem_storage_;
    std::unique_ptr<memory_storage_t> s_mem_storage_;
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
