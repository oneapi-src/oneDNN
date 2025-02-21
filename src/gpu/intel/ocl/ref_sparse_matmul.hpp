/*******************************************************************************
* Copyright 2024-2025 Intel Corporation
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

#ifndef GPU_INTEL_OCL_REF_SPARSE_MATMUL_HPP
#define GPU_INTEL_OCL_REF_SPARSE_MATMUL_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
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

struct ref_sparse_matmul_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_matmul_pd_t {
        using gpu_matmul_pd_t::gpu_matmul_pd_t;

        DECLARE_COMMON_PD_T("ocl:ref:any", ref_sparse_matmul_t);

        status_t init(impl::engine_t *engine) {
            using namespace data_type;

            src_dt_ = src_md()->data_type;
            dst_dt_ = dst_md()->data_type;
            wei_dt_ = weights_md(0)->data_type;

            memory_desc_wrapper src_d(src_md());
            memory_desc_wrapper wei_d(weights_md(0));

            bool is_f16_dt = utils::everyone_is(f16, src_dt_, wei_dt_, dst_dt_);
            bool is_f32_dt = utils::everyone_is(f32, src_dt_, wei_dt_, dst_dt_);
            VDISPATCH_MATMUL(
                    is_f32_dt || is_f16_dt, VERBOSE_UNSUPPORTED_DT_CFG);

            bool is_src_coo_sparse = src_d.is_sparse_desc()
                    && (src_d.encoding() == sparse_encoding::coo);
            VDISPATCH_MATMUL(is_src_coo_sparse, VERBOSE_UNSUPPORTED_SPARSE_CFG);

            bool is_meta_data_valid = src_d.metadata_type(0) == s32;
            VDISPATCH_MATMUL(
                    is_meta_data_valid, VERBOSE_UNSUPPORTED_SPARSE_CFG);

            VDISPATCH_MATMUL(set_default_formats(), VERBOSE_UNSUPPORTED_TAG);
            bool wei_tag_check
                    = wei_d.matches_one_of_tag(format_tag::ab, format_tag::ba);
            VDISPATCH_MATMUL(wei_tag_check, VERBOSE_UNSUPPORTED_TAG);

            VDISPATCH_MATMUL(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_MATMUL(!with_bias(), VERBOSE_UNSUPPORTED_BIAS_CFG);

            return status::success;
        }

        data_type_t src_dt_ = data_type::undef;
        data_type_t dst_dt_ = data_type::undef;
        data_type_t wei_dt_ = data_type::undef;
    };

    status_t init(impl::engine_t *engine) override {
        compute::kernel_ctx_t kernel_ctx;

        int ndims = pd()->dst_md()->ndims;

        kernel_ctx.set_data_type(pd()->dst_dt_);

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

        def_data_type(kernel_ctx, pd()->src_dt_, "SRC");
        def_data_type(kernel_ctx, pd()->wei_dt_, "WEI");
        def_data_type(kernel_ctx, pd()->dst_dt_, "DST");
        def_data_type(kernel_ctx, pd()->desc()->accum_data_type, "ACC");

        CHECK(create_kernel(engine, &kernel_, "ref_sparse_matmul", kernel_ctx));
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
