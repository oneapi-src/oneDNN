/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#ifndef GPU_OCL_REF_GATED_MLP_HPP
#define GPU_OCL_REF_GATED_MLP_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/gated_mlp_pd.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "gpu/gpu_resource.hpp"
#include "gpu/intel/gpu_primitive.hpp"
#include "gpu/intel/ocl/ocl_utils.hpp"
#include "gpu/intel/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

struct ref_gated_mlp_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gated_mlp_pd_t {
        using gated_mlp_pd_t::gated_mlp_pd_t;

        DECLARE_COMMON_PD_T("ocl:ref:any", ref_gated_mlp_t);

        status_t init(impl::engine_t *engine) {
            using namespace data_type;
            //using smask_t = primitive_attr_t::skip_mask_t;

            /* Reference gated mlp is only enabled on-demand, for testing. */
            bool enable_ref
                    = gpu_utils::dev_getenv("enable_ref_gated_mlp", false);
            VDISPATCH_GATED_MLP(enable_ref, VERBOSE_SKIP_PRIMITIVE_IMPL);

            //VDISPATCH_GATED_MLP(
                    //attr()->has_default_values(smask_t::scales_runtime),
                    //VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_GATED_MLP(utils::everyone_is(2, src_md()->ndims,
                                        W_gate_md()->ndims, W_up_md()->ndims,
                                        W_down_md()->ndims, dst_md()->ndims),
                    VERBOSE_UNSUPPORTED_TAG);
            //if (with_attn_mask()) { //TODO: STF add scale + zp + activation?
            //VDISPATCH_GATED_MLP(
            //attn_mask_md()->ndims == 4, VERBOSE_UNSUPPORTED_TAG);
            //}
            VDISPATCH_GATED_MLP(set_default_formats(), VERBOSE_UNSUPPORTED_TAG);

            return status::success;
        }
    };

    status_t init(impl::engine_t *engine) override {
        compute::kernel_ctx_t kernel_ctx;

        kernel_ctx.set_data_type(pd()->dst_md()->data_type);

        int ndims = 2;

        const memory_desc_wrapper src_mdw(pd()->src0_md());
        const memory_desc_wrapper W_gate_mdw(pd()->W_gate_md());
        const memory_desc_wrapper W_up_mdw(pd()->W_up_md());
        const memory_desc_wrapper W_down_mdw(pd()->W_down_md());
        const memory_desc_wrapper dst_mdw(pd()->dst_md());

        using offset_t = decltype(offsets_t().src_off);
        offset_t src_off, W_gate_off, W_up_off, W_down_off, dst_off;
        set_offsets(src_mdw, src_off);
        set_offsets(W_gate_mdw, W_gate_off);
        set_offsets(W_up_mdw, W_up_off);
        set_offsets(W_down_mdw, W_down_off);
        set_offsets(dst_mdw, dst_off);

        def_offsets(src_off, kernel_ctx, "SRC", ndims);
        def_offsets(W_gate_off, kernel_ctx, "W_GATE", ndims);
        def_offsets(W_up_off, kernel_ctx, "W_UP", ndims);
        def_offsets(W_down_off, kernel_ctx, "W_DOWN", ndims);
        def_offsets(dst_off, kernel_ctx, "DST", ndims);
        kernel_ctx.define_int("NDIMS", ndims);

        //kernel_ctx.define_int("SIZE_MB", pd()->desc()->mb_sz());
        //TODO: STF here in context, or just kernel arg? ^^ above for nregisters, batch size relatively stable
        //kernel_ctx.define_int("SIZE_IC", pd()->desc()->ic_sz());
        kernel_ctx.define_int("SIZE_OC", pd()->desc()->oc_sz());

        //kernel_ctx.define_int("INVERT_SCALE", pd()->desc()->invert_scale); //TODO: scale + zp + activation type
        //kernel_ctx.define_int("WITH_ATTN_SCALE", pd()->with_attn_scale());
        //kernel_ctx.define_int("WITH_ATTN_MASK", pd()->with_attn_mask());

        def_data_type(kernel_ctx, pd()->src0_md()->data_type, "SRC");
        def_data_type(kernel_ctx, pd()->W_gate_md()->data_type, "W_GATE");
        def_data_type(kernel_ctx, pd()->W_up_md()->data_type, "W_UP");
        def_data_type(kernel_ctx, pd()->W_down_md()->data_type, "W_DOWN");
        def_data_type(kernel_ctx, pd()->dst_md()->data_type, "DST");
        // def_data_type(kernel_ctx, pd()->desc()->scale_dt, "SCALE");

        CHECK(create_kernel(engine, &kernel_, "ref_gated_mlp", kernel_ctx));
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
