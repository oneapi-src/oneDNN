/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
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

#ifndef GPU_INTEL_OCL_REF_REORDER_HPP
#define GPU_INTEL_OCL_REF_REORDER_HPP

#include "common/c_types_map.hpp"
#include "common/memory.hpp"
#include "common/primitive.hpp"
#include "common/utils.hpp"
#include "gpu/gpu_reorder_pd.hpp"
#include "gpu/gpu_resource.hpp"
#include "gpu/intel/gpu_primitive.hpp"
#include "gpu/intel/ocl/utils.hpp"
#include "gpu/intel/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

struct ref_reorder_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_reorder_pd_t {
        using gpu_reorder_pd_t::gpu_reorder_pd_t;

        DECLARE_COMMON_PD_T("ocl:ref:any", ref_reorder_t);

        status_t init(impl::engine_t *engine, impl::engine_t *src_engine,
                impl::engine_t *dst_engine) {
            using namespace data_type;
            using smask_t = dnnl_primitive_attr::skip_mask_t;
            using compute::device_ext_t;
            const auto sdt = src_md()->data_type;
            const auto ddt = dst_md()->data_type;

            VDISPATCH_REORDER(
                    src_engine == dst_engine, VERBOSE_BAD_ENGINE_KIND);
            VDISPATCH_REORDER(!memory_desc_ndims_ok(src_md(), dst_md()),
                    VERBOSE_INCONSISTENT_NDIMS, "src", "dst");
            VDISPATCH_REORDER(src_engine->kind() == engine_kind::gpu,
                    VERBOSE_BAD_ENGINE_KIND);
            VDISPATCH_REORDER(
                    attr()->has_default_values(smask_t::scales_runtime_data_type
                            | smask_t::scales_runtime_groups
                            | smask_t::zero_points_runtime_data_type
                            | smask_t::zero_points_runtime_groups
                            | smask_t::post_ops | smask_t::rounding_mode)
                            && post_ops_ok(),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_REORDER(
                    IMPLICATION(!attr()->rounding_mode_.has_default_values(),
                            utils::one_of(sdt, f32, bf16, f16)
                                    && utils::one_of(ddt, f8_e4m3, f8_e5m2)),
                    VERBOSE_UNSUPPORTED_ATTR);

            VDISPATCH_REORDER(
                    extra_ok(), VERBOSE_UNSUPPORTED_MD_FLAG, "extra_ok");

            VDISPATCH_REORDER(!(memory_desc_wrapper(src_md())
                                              .has_runtime_dims_or_strides()),
                    VERBOSE_RUNTIMEDIM_UNSUPPORTED);

            VDISPATCH_REORDER(
                    utils::one_of(sdt, f32, f16, bf16, f8_e5m2, f8_e4m3,
                            f4_e2m1, f4_e3m0, s32, s8, u8, s4, u4, f64),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_REORDER(
                    utils::one_of(ddt, f32, f16, bf16, f8_e5m2, f8_e4m3,
                            f4_e2m1, f4_e3m0, s32, s8, u8, s4, u4, f64),
                    VERBOSE_UNSUPPORTED_DT);

            VDISPATCH_REORDER(
                    IMPLICATION(utils::one_of(ddt, f8_e4m3, f8_e5m2, f4_e2m1,
                                        f4_e3m0),
                            utils::one_of(sdt, f64, f32, f16, bf16, f8_e5m2,
                                    f8_e4m3, f4_e2m1, f4_e3m0, ddt)),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_REORDER(
                    IMPLICATION(utils::one_of(sdt, f8_e4m3, f8_e5m2, f4_e2m1,
                                        f4_e3m0),
                            utils::one_of(ddt, f64, f32, f16, bf16, f8_e5m2,
                                    f8_e4m3, f4_e2m1, f4_e3m0, sdt)),
                    VERBOSE_UNSUPPORTED_DT);

            auto *compute_engine = utils::downcast<compute::compute_engine_t *>(
                    dst_engine->kind() == engine_kind::gpu ? dst_engine
                                                           : src_engine);

            VDISPATCH_REORDER(compute_engine->mayiuse(
                                      compute::device_ext_t::intel_subgroups),
                    VERBOSE_UNSUPPORTED_DEVICE_FEATURE, "subgroups");
            VDISPATCH_REORDER(
                    IMPLICATION(utils::one_of(data_type::f16, sdt, ddt),
                            compute_engine->mayiuse(device_ext_t::khr_fp16)
                                    && compute_engine->mayiuse(device_ext_t::
                                                    intel_subgroups_short)),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_REORDER(
                    IMPLICATION(utils::one_of(data_type::f64, sdt, ddt),
                            compute_engine->mayiuse(device_ext_t::khr_fp64)
                                    && attr()->post_ops_.has_default_values()),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_REORDER(
                    IMPLICATION(
                            (utils::one_of(data_type::u4, sdt, ddt)
                                    || utils::one_of(data_type::s4, sdt, ddt)),
                            attr()->post_ops_.has_default_values()),
                    VERBOSE_UNSUPPORTED_DT_CFG);

            VDISPATCH_REORDER_SC(init_conf(engine), "init_conf()");
            VDISPATCH_REORDER_SC(maybe_create_zp_precompute_conv_pd(dst_engine),
                    "failed to create nested zp precompute convolution");

            init_scratchpad();
            return status::success;
        }

        status_t init_conf(impl::engine_t *engine);
        void init_scratchpad();
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;

        reorder_conf_t conf;

    private:
        DECLARE_GPU_REORDER_CREATE();
    };

    status_t init(impl::engine_t *engine) override {
        CHECK(pd()->maybe_create_zp_precompute_conv(
                zp_precomp_conv_, engine, this));

        compute::kernel_ctx_t kernel_ctx;

        auto status = pd()->init_kernel_ctx(kernel_ctx);
        if (status != status::success) return status;

        const auto &conf = pd()->conf;
        if (conf.nelems == 0) return status::success;
        kernels_.resize(2);

        CHECK(create_kernel(engine, &kernels_[0], "ref_reorder", kernel_ctx));
        if (conf.subbyte_pack)
            CHECK(create_kernel(
                    engine, &kernels_[1], "subbyte_pack", kernel_ctx));

        if (!kernels_[0]) return status::runtime_error;
        if (conf.subbyte_pack && !kernels_[1]) return status::runtime_error;
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::vector<compute::kernel_t> kernels_;
    std::shared_ptr<impl::primitive_t> zp_precomp_conv_;
};

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
