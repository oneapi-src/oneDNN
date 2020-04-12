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

#include "gpu/gemm/gpu_gemm.hpp"
#include "gpu/gemm/gpu_gemm_utils.hpp"
#include "gpu/gpu_gemm_pd.hpp"
#include "gpu/ocl/ocl_resource.hpp"
#include "gpu/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

struct ref_gemm_t : public gpu_gemm_t {
    struct pd_t : public gpu_gemm_pd_t {
        using gpu_gemm_pd_t::gpu_gemm_pd_t;

        DECLARE_COMMON_PD_T("ocl:ref:any", ref_gemm_t);

        status_t init(engine_t *engine) {
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
            return eltwise_idx != -1
                    ? attr()->post_ops_.entry_[eltwise_idx].eltwise.alpha
                    : 1.0f;
        }

        float eltwise_beta() const {
            const int eltwise_idx
                    = attr()->post_ops_.find(primitive_kind::eltwise);
            return eltwise_idx != -1
                    ? attr()->post_ops_.entry_[eltwise_idx].eltwise.beta
                    : 0.0f;
        }

        float eltwise_scale() const {
            const int eltwise_idx
                    = attr()->post_ops_.find(primitive_kind::eltwise);
            return eltwise_idx != -1
                    ? attr()->post_ops_.entry_[eltwise_idx].eltwise.scale
                    : 1.0f;
        }

        alg_kind_t eltwise_alg_kind() const {
            const int eltwise_idx
                    = attr()->post_ops_.find(primitive_kind::eltwise);
            return eltwise_idx != -1
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

    ref_gemm_t(const pd_t *apd) : gpu_gemm_t(apd) {}

    status_t init(engine_t *engine) override {
        compute::kernel_ctx_t kernel_ctx;

        kernel_ctx.define_int("WITH_BIAS", pd()->with_bias());
        kernel_ctx.define_int(
                "NON_DEFAULT_ATTRS", !pd()->attr()->has_default_values());
        kernel_ctx.define_int("DO_SUM",
                pd()->attr()->post_ops_.contain(primitive_kind::sum, 0));
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
        create_binary(engine, &binary_, "ref_gemm", kernel_ctx);
        if (!binary_) return status::runtime_error;

        return status::success;
    }

    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const override {
        using namespace gemm_utils;
        if (mapper.has_resource(this)) return status::success;
        auto r = utils::make_unique<ocl_resource_t>();
        if (!r) return status::out_of_memory;
        const auto attr = pd()->attr();
        std::unique_ptr<memory_storage_t> tmp_mem_storage;
        for (const auto idx : {A0_, B0_, C0_}) {
            CHECK(prepare_zero_points(attr, engine, idx, tmp_mem_storage));
            r->add_memory_storage(idx, std::move(tmp_mem_storage));
        }

        CHECK(prepare_scales(attr, engine, tmp_mem_storage));
        r->add_memory_storage(SCALES_, std::move(tmp_mem_storage));
        CHECK(r->create_kernel_and_add(engine, binary_));
        mapper.add(this, std::move(r));
        return status::success;
    }

    virtual status_t execute(const gemm_exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    compute::binary_t binary_;
    enum {
        A0_ = DNNL_ARG_A,
        B0_ = DNNL_ARG_B,
        C0_ = DNNL_ARG_C,
        SCALES_ = DNNL_ARG_ATTR_OUTPUT_SCALES
    };
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
