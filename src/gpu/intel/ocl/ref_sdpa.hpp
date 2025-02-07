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

#ifndef GPU_OCL_REF_SDPA_HPP
#define GPU_OCL_REF_SDPA_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/sdpa_pd.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "gpu/gpu_resource.hpp"
#include "gpu/intel/gpu_primitive.hpp"
#include "gpu/intel/ocl/utils.hpp"
#include "gpu/intel/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

struct ref_sdpa_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public sdpa_pd_t {
        using sdpa_pd_t::sdpa_pd_t;

        DECLARE_COMMON_PD_T("ocl:ref:any", ref_sdpa_t);

        status_t init(impl::engine_t *engine) {
            using namespace data_type;
            using smask_t = primitive_attr_t::skip_mask_t;

            /* Reference SDPA is only enabled on-demand, for testing. */
            bool enable_ref = gpu_utils::dev_getenv("enable_ref_sdpa", false);
            VDISPATCH_SDPA(enable_ref, VERBOSE_SKIP_PRIMITIVE_IMPL);

            VDISPATCH_SDPA(attr()->has_default_values(smask_t::scales_runtime),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_SDPA(
                    utils::everyone_is(4, qry_md()->ndims, key_md()->ndims,
                            val_md()->ndims, dst_md()->ndims),
                    VERBOSE_UNSUPPORTED_TAG);
            if (with_attn_mask()) {
                VDISPATCH_SDPA(
                        attn_mask_md()->ndims == 4, VERBOSE_UNSUPPORTED_TAG);
            }
            VDISPATCH_SDPA(set_default_formats(), VERBOSE_UNSUPPORTED_TAG);

            return status::success;
        }
    };

    status_t init(impl::engine_t *engine) override {
        compute::kernel_ctx_t kernel_ctx;

        kernel_ctx.set_data_type(pd()->dst_md()->data_type);

        int ndims = 4;

        const memory_desc_wrapper qry_mdw(pd()->qry_md());
        const memory_desc_wrapper key_mdw(pd()->key_md());
        const memory_desc_wrapper val_mdw(pd()->val_md());
        const memory_desc_wrapper dst_mdw(pd()->dst_md());
        const memory_desc_wrapper msk_mdw(pd()->attn_mask_md());
        using offset_t = decltype(offsets_t().src_off);
        offset_t qry_off, key_off, val_off, dst_off, msk_off;
        set_offsets(qry_mdw, qry_off);
        set_offsets(key_mdw, key_off);
        set_offsets(val_mdw, val_off);
        set_offsets(dst_mdw, dst_off);
        set_offsets(msk_mdw, msk_off);
        def_offsets(qry_off, kernel_ctx, "QRY", ndims);
        def_offsets(key_off, kernel_ctx, "KEY", ndims);
        def_offsets(val_off, kernel_ctx, "VAL", ndims);
        def_offsets(dst_off, kernel_ctx, "DST", ndims);
        def_offsets(msk_off, kernel_ctx, "MSK", ndims);
        kernel_ctx.define_int("NDIMS", ndims);

        kernel_ctx.define_int("SIZE_K", pd()->desc()->keys());
        kernel_ctx.define_int("INVERT_SCALE", pd()->desc()->invert_scale);
        kernel_ctx.define_int("WITH_ATTN_SCALE", pd()->with_attn_scale());
        kernel_ctx.define_int("WITH_ATTN_MASK", pd()->with_attn_mask());

        def_data_type(kernel_ctx, pd()->qry_md()->data_type, "QRY");
        def_data_type(kernel_ctx, pd()->key_md()->data_type, "KEY");
        def_data_type(kernel_ctx, pd()->val_md()->data_type, "VAL");
        def_data_type(kernel_ctx, pd()->dst_md()->data_type, "DST");
        def_data_type(kernel_ctx, pd()->attn_mask_md()->data_type, "MSK");
        def_data_type(kernel_ctx, pd()->desc()->scale_dt, "SCALE");
        CHECK(create_kernel(engine, &kernel_, "ref_sdpa", kernel_ctx));
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
