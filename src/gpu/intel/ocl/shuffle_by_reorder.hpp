/*******************************************************************************
* Copyright 2020-2024 Intel Corporation
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

#ifndef GPU_INTEL_OCL_SHUFFLE_BY_REORDER_HPP
#define GPU_INTEL_OCL_SHUFFLE_BY_REORDER_HPP

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/reorder.hpp"
#include "gpu/gpu_shuffle_pd.hpp"
#include "gpu/intel/gpu_primitive.hpp"
namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

// Implements shuffle using reorder kernel.
// Pretends that instead of the one dimension to be shuffled there are two
// smaller dimensions, then reorders the tensor to swap those two.
// Reorder kernel is used more often so is expected to be better optimized.
struct shuffle_by_reorder_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_shuffle_pd_t {
        using gpu_shuffle_pd_t::gpu_shuffle_pd_t;

        DECLARE_COMMON_PD_T("ocl:reorder:any", shuffle_by_reorder_t);

        status_t init(impl::engine_t *engine) {
            const auto &md_src = is_fwd() ? src_md() : diff_src_md();
            const auto &md_dst = is_fwd() ? dst_md() : diff_dst_md();
            const memory_desc_wrapper src_d(md_src);
            const memory_desc_wrapper dst_d(md_dst);

            VDISPATCH_SHUFFLE(src_d.data_type() == dst_d.data_type(),
                    VERBOSE_INCONSISTENT_DT, "src_d", "dst_d");
            VDISPATCH_SHUFFLE(md_src->format_kind == format_kind::blocked,
                    VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_SHUFFLE(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_SHUFFLE(
                    set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_SHUFFLE(
                    src_d == dst_d, VERBOSE_INCONSISTENT_MDS, "src", "dst");
            VDISPATCH_SHUFFLE(src_d.is_dense(), VERBOSE_UNSUPPORTED_SPARSE_CFG);

            // Abort if there's blocking on the dimension that's going to be
            // shuffled; such shuffle cannot be reduced to simple reorder.
            // TODO: if both group_size and groups are multiples of blocking it
            // still could be possible to use reorder.
            for (int i = 0; i < md_src->format_desc.blocking.inner_nblks; i++) {
                VDISPATCH_SHUFFLE(
                        md_src->format_desc.blocking.inner_idxs[i] != axis(),
                        VERBOSE_BAD_DIM,
                        "md_src->format_desc.blocking.inner_idxs[i]",
                        (int)md_src->format_desc.blocking.inner_idxs[i]);
            }

            auto tensor_size
                    = utils::array_product(md_src->dims, md_src->ndims);
            // groups, group_size() are sizes of the two fake dimensions
            // groups * group_size() == size of the original single dimension
            auto groups = md_src->dims[axis()] / group_size();
            // prepare 2 dimensions to be reordered
            auto tr_rows = is_fwd() ? group_size() : groups;
            auto tr_cols = is_fwd() ? groups : group_size();
            // combine all dimensions below axis() together with all blocks
            // into a single dimension that's not going to be reordered
            auto stride_of_axis = md_src->format_desc.blocking.strides[axis()];
            // combine all dimensions above axis into a single dimension
            // that's not going to be reordered
            auto remaining = tensor_size
                    / md_src->format_desc.blocking.strides[axis()] / tr_cols
                    / tr_rows;

            memory_desc_t fake_src;
            memory_desc_t fake_dst;

            dims_t d = {remaining, tr_cols, tr_rows, stride_of_axis};
            dims_t strides_src = {d[3] * d[2] * d[1], d[3] * d[2], d[3], 1};
            dims_t strides_dst = {d[3] * d[2] * d[1], d[3], d[1] * d[3], 1};

            VDISPATCH_SHUFFLE_SC(memory_desc_init_by_strides(fake_src, 4, d,
                                         md_src->data_type, strides_src),
                    "memory_desc_init_by_strides()");
            VDISPATCH_SHUFFLE_SC(memory_desc_init_by_strides(fake_dst, 4, d,
                                         md_src->data_type, strides_dst),
                    "memory_desc_init_by_strides()");

            VDISPATCH_SHUFFLE_SC(reorder_primitive_desc_create(reorder_pd_,
                                         engine, &fake_src, &fake_dst),
                    "reorder_primitive_desc_create()");
            return status::success;
        }

        std::shared_ptr<primitive_desc_t> reorder_pd_;
    };

    status_t init(impl::engine_t *engine) override {
        return create_nested_primitive(reorder_, pd()->reorder_pd_, engine);
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        using namespace memory_tracking::names;
        exec_args_t r_args;

        auto src = pd()->is_fwd() ? DNNL_ARG_SRC : DNNL_ARG_DIFF_DST;
        auto dst = pd()->is_fwd() ? DNNL_ARG_DST : DNNL_ARG_DIFF_SRC;

        r_args[DNNL_ARG_SRC] = ctx.args().at(src);
        r_args[DNNL_ARG_DST] = ctx.args().at(dst);
        exec_ctx_t r_ctx(ctx, std::move(r_args));

        return reorder_->execute(r_ctx);
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::shared_ptr<impl::primitive_t> reorder_;
};
} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
