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

#ifndef GPU_INTEL_OCL_REF_MATMUL_HPP
#define GPU_INTEL_OCL_REF_MATMUL_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "gpu/gpu_matmul_pd.hpp"
#include "gpu/gpu_resource.hpp"
#include "gpu/intel/gpu_primitive.hpp"
#include "gpu/intel/ocl/utils.hpp"
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

        status_t init(impl::engine_t *engine) {
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
            VDISPATCH_MATMUL(
                    attr()->has_default_values(smask_t::scales_runtime_data_type
                            | smask_t::scales_runtime_groups | smask_t::dropout
                            | smask_t::zero_points_runtime_data_type
                            | smask_t::zero_points_runtime_groups
                            | smask_t::post_ops | smask_t::fpmath_mode
                            | smask_t::rounding_mode),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_MATMUL(attr_scales_ok(), VERBOSE_UNSUPPORTED_SCALES_CFG);
            VDISPATCH_MATMUL(zero_points_ok(), VERBOSE_UNSUPPORTED_ZP_CFG);
            VDISPATCH_MATMUL(set_default_formats(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_MATMUL(IMPLICATION(has_blocks(), dst_md()->ndims < 6),
                    VERBOSE_BAD_NDIMS, "dst", dst_md()->ndims);

            const bool is_f64
                    = utils::everyone_is(f64, src_dt_, wei_dt_, dst_dt_);
            const bool is_f32 = src_dt_ == f32
                    && utils::one_of(wei_dt_, f32, s8, u8, s4, u4)
                    && dst_dt_ == f32;
            const bool is_f16 = src_dt_ == f16
                    && utils::one_of(wei_dt_, f16, s8, u8, s4, u4)
                    && utils::one_of(dst_dt_, u8, s8, f16, f32);
            const bool is_f8
                    = (utils::one_of(src_dt_, f8_e5m2, f8_e4m3)
                              || utils::one_of(wei_dt_, f8_e5m2, f8_e4m3))
                    && utils::one_of(dst_dt_, f32, bf16, f16, src_dt_);
            const bool is_f4
                    = ((utils::one_of(src_dt_, f4_e2m1, f4_e3m0)
                               || utils::everyone_is(wei_dt_, f4_e2m1, f4_e3m0))
                            && utils::one_of(dst_dt_, f32, bf16, f16, src_dt_));
            const bool is_bf16 = src_dt_ == bf16
                    && utils::one_of(wei_dt_, bf16, s8, u8, s4, u4)
                    && utils::one_of(dst_dt_, bf16, f32);
            const bool is_int8 = utils::one_of(src_dt_, u8, s8)
                    && utils::one_of(wei_dt_, u8, s8, u4, s4)
                    && utils::one_of(dst_dt_, f32, s8, u8, s32, f16);
            VDISPATCH_MATMUL((is_int8
                                     || ((is_f32 || is_f64 || is_f16 || is_f8
                                                 || is_f4 || is_bf16)
                                             && IMPLICATION(with_bias(),
                                                     utils::one_of(bia_dt_, f32,
                                                             dst_dt_)))),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_MATMUL_SC(attr_.set_default_formats(dst_md(0)),
                    VERBOSE_UNSUPPORTED_POSTOP);
            VDISPATCH_MATMUL(post_ops_with_binary_ok(attr(), dst_dt_, 6),
                    VERBOSE_UNSUPPORTED_POSTOP);
            const memory_desc_wrapper dropout_md(attr_.dropout_.dropout_desc_);
            VDISPATCH_MATMUL(
                    IMPLICATION(!attr_.dropout_.has_default_values(),
                            dropout_md.similar_to(dst_md(), true, false)),
                    VERBOSE_INCONSISTENT_MDS, "dropout", "dst");
            VDISPATCH_MATMUL(
                    IMPLICATION(!attr_.dropout_.has_default_values(),
                            utils::one_of(dropout_md.data_type(), u8, s8)),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_MATMUL(
                    IMPLICATION(utils::one_of(f64, src_dt_, wei_dt_, dst_dt_),
                            dev_info_->has_native(f64)),
                    VERBOSE_UNSUPPORTED_DT);
            subbyte_pack_ = utils::one_of(
                    dst_dt_, data_type::f4_e2m1, data_type::f4_e3m0);
            if (subbyte_pack_) {
                using namespace dnnl::impl::memory_tracking::names;
                const memory_desc_wrapper dst_mdw(dst_md(0));
                const auto &padded_dims = dst_mdw.padded_dims();
                const dim_t ndims = dst_mdw.ndims();
                const dim_t nelems = utils::array_product(padded_dims, ndims);
                auto scratchpad = scratchpad_registry().registrar();
                scratchpad.book(memory_tracking::names::key_matmul_pack_space,
                        nelems, sizeof(char), OCL_BUFFER_ALIGNMENT);
            }

            non_default_attrs_ = !attr()->has_default_values();
            attr_info_ = attr_info_t::create(attr());

            return status::success;
        }

        bool non_default_attrs_ = false;
        bool subbyte_pack_ = false;
        data_type_t bia_dt_ = data_type::undef;
        data_type_t src_dt_ = data_type::undef;
        data_type_t dst_dt_ = data_type::undef;
        data_type_t wei_dt_ = data_type::undef;

        attr_info_t attr_info_ = {};

    private:
        bool zero_points_ok() const {
            int mask_src = 0, mask_wei = 0, mask_dst = 0;
            CHECK_BOOL(attr()->zero_points_.get(DNNL_ARG_SRC, &mask_src));
            CHECK_BOOL(attr()->zero_points_.get(DNNL_ARG_WEIGHTS, &mask_wei));
            CHECK_BOOL(attr()->zero_points_.get(DNNL_ARG_DST, &mask_dst));

            const auto src_group_ndims
                    = attr()->zero_points_.get_groups_ndims(DNNL_ARG_SRC);
            const auto src_group_dims
                    = attr()->zero_points_.get_groups(DNNL_ARG_SRC);
            const bool src_m_group_ok
                    = IMPLICATION(src_group_ndims == 2, src_group_dims[0] == 1);
            const bool src_k_group_ok
                    = IMPLICATION(src_group_ndims == 2 && src_group_dims[1] > 1,
                            K() % src_group_dims[1] == 0);

            const auto wei_group_ndims
                    = attr()->zero_points_.get_groups_ndims(DNNL_ARG_WEIGHTS);
            const auto wei_group_dims
                    = attr()->zero_points_.get_groups(DNNL_ARG_WEIGHTS);
            const bool wei_k_group_ok
                    = IMPLICATION(wei_group_ndims == 2 && wei_group_dims[0] > 1,
                            K() % wei_group_dims[0] == 0);
            const bool wei_n_group_ok
                    = IMPLICATION(wei_group_ndims == 2 && wei_group_dims[1] > 1,
                            N() % wei_group_dims[1] == 0);

            bool mask_src_ok = utils::one_of(
                    mask_src, 0, src_qmask_K(), src_qmask_M() + src_qmask_K());
            bool mask_dst_ok = mask_dst == 0;

            return mask_src_ok && mask_dst_ok
                    && utils::one_of(wei_group_ndims, 0, 2)
                    && IMPLICATION(wei_group_ndims == 2,
                            utils::one_of(
                                    1, wei_group_dims[0], wei_group_dims[1])
                                    && wei_k_group_ok && wei_n_group_ok)
                    && IMPLICATION(src_group_ndims == 2,
                            utils::one_of(
                                    1, src_group_dims[0], src_group_dims[1])
                                    && src_m_group_ok && src_k_group_ok);
        }
    };

    status_t init(impl::engine_t *engine) override {
        compute::kernel_ctx_t kernel_ctx;

        int ndims = pd()->dst_md()->ndims;
        kernel_ctx.define_int("DST_NDIMS", ndims);
        kernel_ctx.define_int("WITH_BIAS", pd()->with_bias());
        kernel_ctx.define_int(
                "WITH_DROPOUT", !pd()->attr()->dropout_.has_default_values());
        kernel_ctx.define_int("NON_DEFAULT_ATTRS", pd()->non_default_attrs_);

        auto dst_rnd_mode = pd()->attr()->rounding_mode_.get(DNNL_ARG_DST);
        kernel_ctx.define_int(
                "WITH_SROUND", dst_rnd_mode == rounding_mode::stochastic);
        kernel_ctx.define_int("DST_DT_DIGITS",
                dnnl::impl::types::digits<uint32_t>(pd()->dst_dt_));

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
        def_data_type(kernel_ctx,
                pd()->attr()->scales_.get_data_type(DNNL_ARG_WEIGHTS),
                "WEI_SCALES");
        def_data_type(kernel_ctx,
                pd()->attr()->zero_points_.get_data_type(DNNL_ARG_WEIGHTS),
                "WEI_ZP");
        def_data_type(kernel_ctx,
                pd()->attr()->scales_.get_data_type(DNNL_ARG_SRC),
                "SRC_SCALES");
        def_data_type(kernel_ctx,
                pd()->attr()->zero_points_.get_data_type(DNNL_ARG_SRC),
                "SRC_ZP");
        def_data_type(kernel_ctx,
                pd()->attr()->scales_.get_data_type(DNNL_ARG_DST),
                "DST_SCALES");
        kernels_.resize(2);
        CHECK(create_kernel(engine, &kernels_[0], "ref_matmul", kernel_ctx));
        if (pd()->subbyte_pack_)
            CHECK(create_kernel(
                    engine, &kernels_[1], "subbyte_pack", kernel_ctx));
        if (!kernels_[0]) return status::runtime_error;
        if (pd()->subbyte_pack_ && !kernels_[1]) return status::runtime_error;
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_ref(ctx);
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    status_t execute_ref(const exec_ctx_t &ctx) const;
    std::vector<compute::kernel_t> kernels_;
};

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
