/*******************************************************************************
* Copyright 2019-2024 Intel Corporation
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

#ifndef GPU_INTEL_OCL_REF_MATMUL_HPP
#define GPU_INTEL_OCL_REF_MATMUL_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "gpu/gpu_matmul_pd.hpp"
#include "gpu/intel/gpu_primitive.hpp"
#include "gpu/intel/gpu_resource.hpp"
#include "gpu/intel/ocl/ocl_utils.hpp"
#include "gpu/intel/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

struct ref_matmul_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_matmul_pd_t {
        using gpu_matmul_pd_t::gpu_matmul_pd_t;

        DECLARE_COMMON_PD_T("ocl:ref:any", ref_matmul_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            using smask_t = primitive_attr_t::skip_mask_t;

            src_dt_ = src_md()->data_type;
            dst_dt_ = dst_md()->data_type;
            wei_dt_ = weights_md(0)->data_type;
            bia_dt_ = with_bias() ? weights_md(1)->data_type : data_type::f32;
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);

            auto dev_info_ = compute_engine->device_info();

            VDISPATCH_MATMUL(
                    is_dense_format_kind(), VERBOSE_UNSUPPORTED_SPARSE_CFG);
            VDISPATCH_MATMUL(IMPLICATION(desc()->accum_data_type == s32,
                                     attr()->zero_points_.common()),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_MATMUL(IMPLICATION(desc()->accum_data_type != s32,
                                     attr()->zero_points_.has_default_values()),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_MATMUL(
                    attr()->has_default_values(smask_t::scales_runtime
                            | smask_t::zero_points_runtime | smask_t::post_ops),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_MATMUL(attr_scales_ok(), VERBOSE_UNSUPPORTED_SCALES_CFG);
            VDISPATCH_MATMUL(set_default_formats(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_MATMUL(IMPLICATION(has_blocks(), dst_md()->ndims < 6),
                    VERBOSE_BAD_NDIMS, "dst", dst_md()->ndims);
            VDISPATCH_MATMUL(
                    ((utils::one_of(src_dt_, u8, s8)
                             && utils::one_of(wei_dt_, u8, s8)
                             && utils::one_of(dst_dt_, f32, s8, u8, s32, f16)
                             && IMPLICATION(with_bias(),
                                     utils::one_of(bia_dt_, f32, u8, s8, s32)))
                            || ((utils::everyone_is(
                                         f32, src_dt_, wei_dt_, dst_dt_)
                                        || utils::everyone_is(
                                                f64, src_dt_, wei_dt_, dst_dt_)
                                        || (utils::everyone_is(
                                                    f16, src_dt_, wei_dt_)
                                                && utils::one_of(
                                                        dst_dt_, u8, s8, f16))
                                        || ((utils::everyone_is(
                                                     f8_e5m2, src_dt_, wei_dt_)
                                                    || utils::everyone_is(
                                                            f8_e4m3, src_dt_,
                                                            wei_dt_))
                                                && utils::one_of(dst_dt_, f32,
                                                        bf16, f16, src_dt_))
                                        || (utils::everyone_is(
                                                    bf16, src_dt_, wei_dt_)
                                                && utils::one_of(
                                                        dst_dt_, bf16, f32)))
                                    && IMPLICATION(with_bias(),
                                            utils::one_of(bia_dt_, f32)))),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_MATMUL(post_ops_with_binary_ok(attr(), dst_dt_, 6),
                    VERBOSE_UNSUPPORTED_POSTOP);
            VDISPATCH_MATMUL_SC(attr_.set_default_formats(dst_md(0)),
                    VERBOSE_UNSUPPORTED_POSTOP);
            VDISPATCH_MATMUL(
                    IMPLICATION(utils::one_of(f64, src_dt_, wei_dt_, dst_dt_),
                            dev_info_->has_native(f64)),
                    VERBOSE_UNSUPPORTED_DT);

            non_default_attrs_ = !attr()->has_default_values();
            attr_info_ = attr_info_t::create(attr());

            return status::success;
        }

        bool non_default_attrs_ = false;
        data_type_t bia_dt_ = data_type::undef;
        data_type_t src_dt_ = data_type::undef;
        data_type_t dst_dt_ = data_type::undef;
        data_type_t wei_dt_ = data_type::undef;

        attr_info_t attr_info_ = {};
    };

    status_t init(engine_t *engine) override {
        compute::kernel_ctx_t kernel_ctx;

        int ndims = pd()->dst_md()->ndims;
        kernel_ctx.define_int("DST_NDIMS", ndims);
        kernel_ctx.define_int("WITH_BIAS", pd()->with_bias());
        kernel_ctx.define_int("NON_DEFAULT_ATTRS", pd()->non_default_attrs_);

        kernel_ctx.set_data_type(pd()->dst_dt_);
        CHECK(def_attr_info(kernel_ctx, pd()->attr_info_,
                pd()->attr()->post_ops_, *pd()->dst_md()));

        bool runtime_dims = pd()->has_runtime_dims_or_strides() || ndims > 5;
        if (!runtime_dims) {
            const memory_desc_wrapper src_d(pd()->src_md(0));
            const memory_desc_wrapper wei_d(pd()->weights_md(0));
            const memory_desc_wrapper dst_d(pd()->dst_md(0));
            offsets_t off;
            set_offsets(src_d, off.src_off);
            set_offsets(wei_d, off.wei_off);
            set_offsets(dst_d, off.dst_off);
            def_offsets(off.src_off, kernel_ctx, "SRC", ndims);
            def_offsets(off.wei_off, kernel_ctx, "WEI", ndims);
            def_offsets(off.dst_off, kernel_ctx, "DST", ndims);
            kernel_ctx.define_int("NDIMS", ndims);
        }
        kernel_ctx.define_int("RUNTIME_DIMS", runtime_dims);

        def_data_type(kernel_ctx, pd()->src_dt_, "SRC");
        def_data_type(kernel_ctx, pd()->wei_dt_, "WEI");
        def_data_type(kernel_ctx, pd()->dst_dt_, "DST");
        def_data_type(kernel_ctx, pd()->bia_dt_, "BIA");
        def_data_type(kernel_ctx, pd()->desc()->accum_data_type, "ACC");
        CHECK(create_kernel(engine, &kernel_, "ref_matmul", kernel_ctx));
        if (!kernel_) return status::runtime_error;
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_ref(ctx);
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    status_t execute_ref(const exec_ctx_t &ctx) const;
    compute::kernel_t kernel_;
};

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
