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

#ifndef GPU_GENERIC_DIRECT_COPY_HPP
#define GPU_GENERIC_DIRECT_COPY_HPP

#include "gpu/gpu_primitive.hpp"
#include "gpu/gpu_reorder_pd.hpp"
#include "gpu/gpu_stream.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {

struct direct_copy_t : public primitive_t {
    using primitive_t::primitive_t;
    struct pd_t : public gpu_reorder_pd_t {
        using gpu_reorder_pd_t::gpu_reorder_pd_t;

        DECLARE_COMMON_PD_T("gpu:direct_copy", direct_copy_t);

        status_t init(impl::engine_t *engine, impl::engine_t * /*src_engine*/,
                impl::engine_t * /*dst_engine*/) {
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

            std::vector<block_t> src_blocks, dst_blocks;
            VDISPATCH_REORDER_SC(normalize(src_mdw, src_blocks),
                    VERBOSE_UNSUPPORTED_MEM_STRIDE);
            VDISPATCH_REORDER_SC(normalize(dst_mdw, dst_blocks),
                    VERBOSE_UNSUPPORTED_MEM_STRIDE);
            auto src_it = src_blocks.begin(), dst_it = dst_blocks.begin();
            const auto src_end = src_blocks.end(), dst_end = dst_blocks.end();

            for (; src_it != src_end && dst_it != dst_end; ++src_it, ++dst_it) {
                if (*src_it != *dst_it) break;
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
            if (dst_it == dst_end) return status::success;
            VDISPATCH_REORDER(std::distance(dst_it, dst_end) == 1,
                    VERBOSE_INCONSISTENT_MDS, "src", "dst");
            VDISPATCH_REORDER(dst_it->second <= src_it->second,
                    VERBOSE_INCONSISTENT_MDS, "src", "dst");
            return status::success;
        }

    private:
        DECLARE_GPU_REORDER_CREATE();
        using block_t = std::pair<int, dim_t>;

        status_t normalize(
                const memory_desc_wrapper &mdw, std::vector<block_t> &blocks) {
            if (mdw.ndims() == 0) return status::success;
            blocks.clear();
            auto &blocking = mdw.blocking_desc();
            blocks.reserve(mdw.ndims() + blocking.inner_nblks);

            dim_t stride = 1;
            std::vector<dim_t> dim_blocking(mdw.ndims(), 1);
            for (int i = blocking.inner_nblks - 1; i >= 0; --i) {
                int dim_idx = blocking.inner_idxs[i];
                dim_t block = blocking.inner_blks[i];
                if (block == 1) continue;
                if (blocks.empty() || blocks.back().first != dim_idx)
                    blocks.emplace_back(dim_idx, block);
                else
                    blocks.back().second *= block;
                dim_blocking[dim_idx] *= block;
                stride *= block;
            }

            size_t offset = blocks.size();
            for (int i = 0; i < mdw.ndims(); ++i) {
                dim_t block = mdw.padded_dims()[i] / dim_blocking[i];
                if (block == 1) continue;
                blocks.emplace_back(i, block);
            }
            auto cmp = [&](const block_t &l, const block_t &r) {
                auto &l_stride = blocking.strides[l.first];
                auto &r_stride = blocking.strides[r.first];
                return l_stride < r_stride
                        || (l_stride == r_stride && l.first > r.first);
            };
            std::sort(blocks.begin() + offset, blocks.end(), cmp);
            if (offset > 0 && blocks.size() > offset
                    && blocks[offset].first == blocks[offset - 1].first
                    && blocking.strides[blocks[offset].first] == stride) {
                blocks[offset - 1].second *= blocks[offset].second;
                stride *= blocks[offset].second;
                blocks.erase(blocks.begin() + offset);
            }

            for (; offset < blocks.size(); ++offset) {
                int dim_idx = blocks[offset].first;
                dim_t block = blocks[offset].second;

                if (blocking.strides[dim_idx] != stride)
                    return status::unimplemented;
                stride *= block;
            }
            return status::success;
        }
    };

    status_t init(impl::engine_t *engine) override { return status::success; }

    status_t execute(const exec_ctx_t &ctx) const override {
        auto *stream = utils::downcast<stream_t *>(ctx.stream());

        size_t size = memory_desc_wrapper(pd()->dst_md()).size();
        auto &input = CTX_IN_STORAGE(DNNL_ARG_FROM);
        auto &output = CTX_OUT_STORAGE(DNNL_ARG_TO);
        auto &deps = stream->ctx().get_deps();
        return stream->copy(input, output, size, deps, deps);
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
