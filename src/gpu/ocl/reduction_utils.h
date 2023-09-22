/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#ifndef GPU_REDUCTION_UTILS_H
#define GPU_REDUCTION_UTILS_H

#include <assert.h>
#include <sstream>
#include <vector>

#include "common/c_types_map.hpp"
#include "gpu/block_structure.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

// Zero padding splits one block into two, filling with
// zeros. This is a kind of reorder that can be used
// to short-circuit calculations to avoid reading/writing zeros.
struct zero_padding_t {
    zero_padding_t(const int dim_idx, const dim_t data_size,
            const dim_t outer_stride, const dim_t outer_size,
            const dim_t inner_stride, const dim_t inner_size)
        : dim_idx(dim_idx)
        , data_size(data_size)
        , outer_stride(outer_stride)
        , outer_size(outer_size)
        , inner_stride(inner_stride)
        , inner_size(inner_size) {}
    zero_padding_t(
            const dim_t data_size, const block_t &outer, const block_t &inner)
        : data_size(data_size)
        , outer_stride(outer.stride)
        , outer_size(outer.block)
        , inner_stride(inner.stride)
        , inner_size(inner.block) {
        assert(outer.dim_idx == inner.dim_idx);
        assert(outer.block * inner.block >= data_size);
        dim_idx = outer.dim_idx;
    }

    // Prints the indexing this zero-padding enforces. e.g.:
    // (idx / 1) % 16 + [(idx / 256) % 2] * 16 < 30 (aren't zeros)
    std::string str() const {
        std::stringstream os;
        os << dim_idx << ": ";
        os << "(idx / " << inner_stride << ") % " << inner_size;
        os << " + " << inner_size << " * (";
        os << "(idx / " << outer_stride << ") % " << outer_size;
        os << ") < " << data_size;
        return os.str();
    }

    dim_t dim_idx;
    dim_t data_size;
    dim_t outer_stride, outer_size;
    dim_t inner_stride, inner_size;
};

class reduction_subproblem_t {
public:
    reduction_subproblem_t(
            dim_t inner_size, dim_t reduction_size, dim_t outer_size)
        : inner_block(0, inner_size, 1)
        , reduction_block(1, reduction_size, inner_size)
        , outer_block(2, outer_size, inner_size * reduction_size) {}

    reduction_subproblem_t(std::vector<block_t> blocks, size_t red_start_idx,
            size_t red_end_idx, int reduced_mask)
        : inner_block(0, 1, 1)
        , reduction_block(1, 1, blocks[red_start_idx].stride)
        , outer_block(2, 1,
                  red_end_idx == blocks.size()
                          ? blocks.back().stride * blocks.back().block
                          : blocks[red_end_idx].stride) {

        // Compute the size of the inner, reduction, and outer blocks
        // Assume that any reduced dims before red_start_idx have already
        // been reduced to 1 element (heuristic in how these problems are generated)
        dim_t ignored_inner_elems = 1;
        for (size_t i = 0; i < blocks.size(); i++) {
            dim_t block_size = blocks[i].block;
            if (i < red_start_idx) {
                if (reduced_mask & (1 << blocks[i].dim_idx)) {
                    ignored_inner_elems *= block_size;
                } else {
                    inner_block.block *= block_size;
                }
            } else if (i < red_end_idx) {
                assert(reduced_mask & (1 << blocks[i].dim_idx));
                reduction_block.block *= block_size;
            } else {
                outer_block.block *= block_size;
            }
        }

        // For ignored inner dims, we have to reduce the stride of the other dims
        reduction_block.stride /= ignored_inner_elems;
        outer_block.stride /= ignored_inner_elems;
    }

    std::string str() const {
        std::stringstream os;
        os << "subproblem:" << std::endl;
        os << "outer: " << outer_block.str() << std::endl;
        os << "reduction: " << reduction_block.str() << std::endl;
        os << "inner: " << inner_block.str() << std::endl;
        os << " -- src zero pads: " << src_zpads.size() << std::endl;
        for (const auto &zp : src_zpads) {
            os << "    + " << zp.str() << std::endl;
        }
        os << " -- dst zero pads: " << dst_zpads.size() << std::endl;
        for (const auto &zp : dst_zpads) {
            os << "    + " << zp.str() << std::endl;
        }
        return os.str();
    }

    block_t inner_block;
    block_t reduction_block;
    block_t outer_block;

    std::vector<zero_padding_t> src_zpads;
    std::vector<zero_padding_t> dst_zpads;
};

status_t generate_reduction_phases(const memory_desc_t *src,
        const memory_desc_t *dst, std::vector<reduction_subproblem_t> &subprbs);

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
