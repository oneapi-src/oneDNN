/*******************************************************************************
* Copyright 2024 Intel Corporation
* Copyright 2024 Arm Ltd. and affiliates
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

#ifndef GPU_INTEL_OCL_DIRECT_COPY_HPP
#define GPU_INTEL_OCL_DIRECT_COPY_HPP

#include "common/c_types_map.hpp"
#include "common/memory.hpp"
#include "common/primitive.hpp"
#include "common/utils.hpp"
#include "gpu/gpu_reorder_pd.hpp"
#include "gpu/intel/gpu_primitive.hpp"
#include "gpu/intel/gpu_resource.hpp"
#include "gpu/intel/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

struct direct_copy_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_reorder_pd_t {
        using gpu_reorder_pd_t::gpu_reorder_pd_t;

        DECLARE_COMMON_PD_T("ocl:direct_copy", direct_copy_t);

        status_t init(engine_t *engine, engine_t * /*src_engine*/,
                engine_t * /*dst_engine*/) {
            VDISPATCH_REORDER(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_REORDER(
                    extra_ok(), VERBOSE_UNSUPPORTED_MD_FLAG, "extra_ok");

            memory_desc_wrapper src_mdw(src_md()), dst_mdw(dst_md());
            VDISPATCH_REORDER(!src_mdw.has_runtime_dims_or_strides(),
                    VERBOSE_RUNTIMEDIM_UNSUPPORTED);
            VDISPATCH_REORDER(!dst_mdw.has_runtime_dims_or_strides(),
                    VERBOSE_RUNTIMEDIM_UNSUPPORTED);
            VDISPATCH_REORDER(src_mdw.data_type() == dst_mdw.data_type(),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_REORDER(src_mdw.offset0() == 0,
                    VERBOSE_UNSUPPORTED_PAD_FEATURE, "src offset");
            VDISPATCH_REORDER(dst_mdw.offset0() == 0,
                    VERBOSE_UNSUPPORTED_PAD_FEATURE, "dst offset");

            block_layout_t src_layout {src_mdw}, dst_layout {dst_mdw};
            auto src_it = src_layout.begin(), dst_it = dst_layout.begin();
            const auto src_end = src_layout.end(), dst_end = dst_layout.end();

            dim_t stride = 1;
            for (; src_it != src_end && dst_it != dst_end; ++src_it, ++dst_it) {
                if (*src_it != *dst_it) break;
                VDISPATCH_REORDER((dim_t)src_it->stride == stride,
                        VERBOSE_UNSUPPORTED_MEM_STRIDE);
                stride *= src_it->block;
            }

            if (src_it == src_end) {
                VDISPATCH_REORDER(dst_it == dst_end, VERBOSE_INCONSISTENT_MDS,
                        "src", "dst");
                return status::success;
            }

            // Fallthrough is a special case where we can trim padding when
            // it covers the outermost dimension. E.g. A4a -> a for a tensor
            // of size 3.
            VDISPATCH_REORDER(std::distance(src_it, src_end) == 1,
                    VERBOSE_INCONSISTENT_MDS, "src", "dst");
            VDISPATCH_REORDER((dim_t)src_it->stride == stride,
                    VERBOSE_UNSUPPORTED_MEM_STRIDE);
            if (dst_it == dst_end) return status::success;
            VDISPATCH_REORDER(std::distance(dst_it, dst_end) == 1,
                    VERBOSE_INCONSISTENT_MDS, "src", "dst");
            VDISPATCH_REORDER((dim_t)dst_it->stride == stride,
                    VERBOSE_UNSUPPORTED_MEM_STRIDE);
            VDISPATCH_REORDER(dst_it->block <= src_it->block,
                    VERBOSE_INCONSISTENT_MDS, "src", "dst");
            return status::success;
        }

    private:
        DECLARE_GPU_REORDER_CREATE();
    };

    status_t init(engine_t *engine) override { return status::success; }

    status_t execute(const exec_ctx_t &ctx) const override {
        auto *compute_stream
                = utils::downcast<compute::compute_stream_t *>(ctx.stream());

        size_t size = memory_desc_wrapper(pd()->dst_md()).size();
        auto &input = CTX_IN_STORAGE(DNNL_ARG_FROM);
        auto &output = CTX_OUT_STORAGE(DNNL_ARG_TO);
        auto &deps = compute_stream->ctx().get_deps();
        return compute_stream->copy(input, output, size, deps, deps);
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
