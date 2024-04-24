/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
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

#include "gpu/intel/compute/block_manipulation.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace compute {

// All blocks in a bin are:
// - for each buffer: either broadcasted or not (no mixing)
// - for each buffer: form a dense block
// - if it's "indexed", all blocks have the same dim idx
bool mapped_block_t::can_merge(
        const mapped_block_t &other, bool require_all_match) const {
    bool same_broadcast = true;
    bool all_match = true;
    bool any_match = false;
    for (size_t idx = 0; idx < DNNL_MAX_NDIMS; idx++) {

        // Check that broadcasting is the same
        bool new_broadcast = other.is_broadcasted(idx);
        bool old_broadcast = is_broadcasted(idx);
        same_broadcast &= (new_broadcast == old_broadcast);

        if (new_broadcast || old_broadcast) {
            // No further checks required
            continue;
        }

        // Check that it forms a dense block with the last block
        const block_t &new_block = other.get_buffer_blocks().at(idx);
        const block_t &old_block = get_buffer_blocks().at(idx);
        all_match &= old_block.can_merge(new_block, false);
        any_match |= old_block.can_merge(new_block, false);
    }

    if (require_all_match) return same_broadcast && all_match;
    return same_broadcast && any_match;
}

std::string block_bin_t::str() const {
    std::ostringstream ss;
    ss << "block bin (dim_idx: " << dim_idx;
    ss << ", num_layouts: " << num_layouts;
    ss << ", size: " << size() << ")" << std::endl;
    ss << std::setw(50) << "broadcast:";
    for (bool bc : is_broadcasted_) {
        ss << std::setw(50) << bc;
    }
    ss << std::endl;
    for (const mapped_block_t &blocks : mapped_blocks) {
        ss << std::setw(50) << "block:";
        for (size_t i = 0; i < num_layouts; i++) {
            if (blocks.is_broadcasted(i))
                ss << std::setw(50) << "broadcast";
            else
                ss << std::setw(50) << blocks.get_buffer_blocks().at(i).str();
        }
        ss << std::endl;
    }
    return ss.str();
}

} // namespace compute
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
