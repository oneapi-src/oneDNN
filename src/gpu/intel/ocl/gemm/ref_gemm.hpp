/*******************************************************************************
* Copyright 2020-2025 Intel Corporation
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

#ifndef GPU_INTEL_OCL_GEMM_REF_GEMM_HPP
#define GPU_INTEL_OCL_GEMM_REF_GEMM_HPP

#include "gpu/gpu_gemm_pd.hpp"
#include "gpu/gpu_resource.hpp"
#include "gpu/intel/gemm/gpu_gemm.hpp"
#include "gpu/intel/primitive_conf.hpp"
#include "gpu/intel/serialization.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

struct ref_gemm_jit_params_t
    : public trivially_serializable_t<ref_gemm_jit_params_t> {
    status_t create_generator(const compute::compute_engine_t &engine,
            compute::kernel_bundle_t &bundle) const {
        return engine.create_kernel_bundle(
                bundle, get_kernel_names(), get_kernel_ctx());
    }

    const std::vector<const char *> &get_kernel_names() const {
        static const std::vector<const char *> names {"ref_gemm"};
        return names;
    }

    compute::kernel_ctx_t get_kernel_ctx() const {
        compute::kernel_ctx_t kernel_ctx;
        kernel_ctx.set_data_type(c_dt);
        def_data_type(kernel_ctx, a_dt, "A");
        def_data_type(kernel_ctx, b_dt, "B");
        def_data_type(kernel_ctx, c_dt, "C");
        def_data_type(kernel_ctx, bia_dt, "BIA");
        def_data_type(kernel_ctx, acc_dt, "ACC");

        kernel_ctx.define_int("WITH_BIAS", bia_dt != data_type::undef);
        kernel_ctx.define_int("NON_DEFAULT_ATTRS",
                with_post_ops || with_sum || with_dst_zpoints);

        kernel_ctx.define_int("WITH_POST_OP", with_post_ops);
        if (with_post_ops) {
            def_binary_alg_kinds(kernel_ctx);
            def_eltwise_alg_kinds(kernel_ctx);
            kernel_ctx.define_int("ELTWISE_ALG", eltwise_alg);
        }
        kernel_ctx.define_int("WITH_SUM", with_sum);
        kernel_ctx.define_int("WITH_DST_ZPOINTS", with_dst_zpoints);

        return kernel_ctx;
    };

    data_type_t a_dt = {};
    data_type_t b_dt = {};
    data_type_t c_dt = {};
    data_type_t bia_dt = {};
    data_type_t acc_dt = {};
    bool with_post_ops = {};
    bool with_sum = {};
    bool with_dst_zpoints = {};
    uint8_t pad[1] = {};
    int eltwise_alg = {};
};

struct ref_gemm_t : public gpu_gemm_t {
    using gpu_gemm_t::gpu_gemm_t;
    struct pd_t : public gpu_gemm_pd_t {
        using gpu_gemm_pd_t::gpu_gemm_pd_t;

        DECLARE_COMMON_PD_T("ocl:ref:any", ref_gemm_t);

        status_t init(impl::engine_t *engine) {
            using namespace data_type;
            using smask_t = primitive_attr_t::skip_mask_t;

            VDISPATCH_GEMM(set_default_formats(), VERBOSE_UNSUPPORTED_TAG);

            const auto a_dt = desc()->a_type();
            const auto b_dt = desc()->b_type();
            const auto c_dt = desc()->c_type();
            const auto acc_dt = desc()->acc_type;
            const auto bia_dt = desc()->bias_type();
            const bool wei_decompress = utils::one_of(b_dt, f32, f16, bf16)
                    && utils::one_of(c_dt, f32, f16, bf16)
                    && utils::one_of(a_dt, u8, s8);

            const auto ndims = desc()->c_desc.ndims;
            const auto a_strides = desc()->a_desc.format_desc.blocking.strides;
            const auto b_strides = desc()->b_desc.format_desc.blocking.strides;
            const auto c_strides = desc()->c_desc.format_desc.blocking.strides;

            VDISPATCH_GEMM(!has_blocks(), VERBOSE_UNSUPPORTED_FEATURE,
                    "blocked format");
            VDISPATCH_GEMM(desc()->c_desc.ndims <= 3, VERBOSE_BAD_NDIMS,
                    "desc()->c_desc.ndims", desc()->c_desc.ndims);
            VDISPATCH_GEMM(
                    (a_strides[ndims - 1] == 1 || a_strides[ndims - 2] == 1),
                    VERBOSE_UNSUPPORTED_MEM_STRIDE);
            VDISPATCH_GEMM(
                    (b_strides[ndims - 1] == 1 || b_strides[ndims - 2] == 1),
                    VERBOSE_UNSUPPORTED_MEM_STRIDE);
            VDISPATCH_GEMM((c_strides[ndims - 1] == 1),
                    VERBOSE_UNSUPPORTED_MEM_STRIDE);
            VDISPATCH_GEMM(
                    IMPLICATION(desc()->is_batched(),
                            desc()->a_desc.dims[0] == desc()->b_desc.dims[0]),
                    VERBOSE_INCONSISTENT_NDIMS, "desc()->a_desc.dims[0]",
                    "desc()->b_desc.dims[0]");
            VDISPATCH_GEMM(IMPLICATION(acc_dt != s32 && !wei_decompress,
                                   attr()->zero_points_.has_default_values()),
                    VERBOSE_UNSUPPORTED_ZP_CFG);
            VDISPATCH_GEMM(
                    attr()->has_default_values(smask_t::zero_points_runtime
                            | smask_t::post_ops | smask_t::fpmath_mode),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_GEMM(attr_oscale_ok(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_GEMM(attr_zp_ok(), VERBOSE_UNSUPPORTED_ZP_CFG);
            VDISPATCH_GEMM(attr_post_ops_ok(), VERBOSE_UNSUPPORTED_POSTOP);
            VDISPATCH_GEMM(desc()->sum_ab == sum_ab::sum_none,
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_GEMM(
                    ((utils::everyone_is(f32, a_dt, b_dt, c_dt)
                             && IMPLICATION(with_bias(), bia_dt == f32))
                            || (utils::everyone_is(f16, a_dt, b_dt)
                                    && utils::one_of(c_dt, u8, s8, f16)
                                    && IMPLICATION(with_bias(),
                                            utils::one_of(bia_dt, f16, f32)))
                            || (utils::everyone_is(bf16, a_dt, b_dt)
                                    && utils::one_of(c_dt, bf16, f32)
                                    && IMPLICATION(with_bias(),
                                            utils::one_of(bia_dt, bf16, f32)))
                            || (utils::one_of(a_dt, f32, bf16, f16)
                                    && utils::one_of(b_dt, u8, s8)
                                    && utils::one_of(c_dt, f32, bf16, f16)
                                    && IMPLICATION(with_bias(),
                                            utils::one_of(f32, bf16, f16)))
                            || (utils::one_of(a_dt, u8, s8)
                                    && utils::one_of(b_dt, u8, s8)
                                    && utils::one_of(c_dt, f32, s8, u8, s32)
                                    && IMPLICATION(with_bias(),
                                            utils::one_of(
                                                    bia_dt, f32, u8, s8, s32)))
                            || (utils::one_of(a_dt, f8_e5m2, f8_e4m3)
                                    && utils::one_of(b_dt, f8_e5m2, f8_e4m3)
                                    && utils::one_of(c_dt, f32, f16, bf16,
                                            f8_e5m2, f8_e4m3)
                                    && IMPLICATION(with_bias(),
                                            utils::one_of(bia_dt, f32, f8_e5m2,
                                                    f8_e4m3)))
                            || (utils::one_of(a_dt, u8, s8)
                                    && (utils::everyone_is(f32, b_dt, c_dt)
                                            || utils::everyone_is(
                                                    f16, b_dt, c_dt))
                                    && IMPLICATION(with_bias(),
                                            utils::one_of(
                                                    bia_dt, f32, u8, s8, s32)))
                            || wei_decompress),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            attr_info = attr_info_t::create(attr());

            conf.a_dt = a_dt;
            conf.b_dt = b_dt;
            conf.c_dt = c_dt;
            conf.bia_dt = bia_dt;
            conf.acc_dt = acc_dt;
            conf.with_post_ops = attr()->post_ops_.len() > 0;
            conf.with_sum = attr_info.with_sum;
            conf.with_dst_zpoints = attr_info.with_dst_zpoints;
            conf.eltwise_alg = attr_info.eltwise_alg;

            return status::success;
        }

        bool set_default_formats() {
            return gpu_gemm_pd_t::set_default_formats();
        }

        bool attr_oscale_ok() const {
            const auto &scales = attr()->scales_;
            const bool src_scale_ok = scales.has_default_values(DNNL_ARG_SRC)
                    || scales.get_mask(DNNL_ARG_SRC) == 0;
            const bool wei_scale_ok
                    = scales.has_default_values(DNNL_ARG_WEIGHTS)
                    || scales.get_mask(DNNL_ARG_WEIGHTS) == 0;
            const bool dst_scale_ok = scales.has_default_values(DNNL_ARG_DST)
                    || scales.get_mask(DNNL_ARG_DST) == 0;
            return src_scale_ok && wei_scale_ok && dst_scale_ok;
        }

        bool attr_zp_ok() const {
            const auto &zp = attr()->zero_points_;

            bool ok = IMPLICATION(desc()->acc_type != data_type::s32,
                    zp.has_default_values());
            if (!ok) return false;

            static const std::vector<int> supported_args {
                    DNNL_ARG_A, DNNL_ARG_B, DNNL_ARG_C};
            for (int arg : supported_args) {
                if (!zp.has_default_values(arg)) {
                    const int mask = zp.get_mask(arg);
                    if (mask > 0) return false;
                }
            }
            return true;
        }

        bool attr_post_ops_ok() const {
            using namespace primitive_kind;
            using namespace data_type;
            const auto &p = attr()->post_ops_;
            switch (p.len()) {
                case 0: return true;
                case 1: return p.contain(sum, 0) || p.contain(eltwise, 0);
                case 2: return p.contain(sum, 0) && p.contain(eltwise, 1);
                default: return false;
            }
        }

        bool with_bias() const {
            return desc()->bias_type() != data_type::undef;
        }

        attr_info_t attr_info = {};
        ref_gemm_jit_params_t conf = {};
    };

    status_t init(impl::engine_t *engine) override {
        CHECK(create_kernel(engine, kernel_, "ref_gemm", pd()->conf));
        if (!kernel_) return status::runtime_error;

        return status::success;
    }

    status_t execute(const gemm_exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    compute::kernel_t kernel_;
};

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
