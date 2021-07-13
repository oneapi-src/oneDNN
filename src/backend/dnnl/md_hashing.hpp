/*******************************************************************************
 * Copyright 2021 Intel Corporation
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
#ifndef BACKEND_DNNL_MD_HASHING_HPP
#define BACKEND_DNNL_MD_HASHING_HPP

#include <memory>
#include <unordered_map>

#include <dnnl.hpp>

#include "interface/partition_hashing.hpp"
#include "utils/utils.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

// Copied from oneDNN:
// https://github.com/oneapi-src/oneDNN/blob/master/src/common/primitive_hashing.cpp#L123
// Combine hash of each memory_desc_t data member
inline size_t get_md_hash(const dnnl::memory::desc &mem_desc) {
    using namespace impl::partition_hashing;
    using namespace impl::utils;

    auto md = mem_desc.data;

    size_t seed = 0;
    seed = get_array_hash(seed, md.dims, md.ndims);
    seed = hash_combine(seed, static_cast<size_t>(md.data_type));
    seed = get_array_hash(seed, md.padded_dims, md.ndims);
    seed = get_array_hash(seed, md.padded_offsets, md.ndims);
    seed = hash_combine(seed, md.offset0);
    seed = hash_combine(seed, static_cast<size_t>(md.format_kind));
    // format desc
    switch (md.format_kind) {
        case dnnl_format_kind_undef:
        case dnnl_format_kind_any: break;
        case dnnl_blocked:
            for (int i = 0; i < md.ndims; i++) {
                if (md.dims[i] == 1 && md.padded_dims[i] == 1) continue;
                seed = hash_combine(seed, md.format_desc.blocking.strides[i]);
            }
            seed = hash_combine(seed, md.format_desc.blocking.inner_nblks);
            seed = get_array_hash(seed, md.format_desc.blocking.inner_blks,
                    md.format_desc.blocking.inner_nblks);
            seed = get_array_hash(seed, md.format_desc.blocking.inner_idxs,
                    md.format_desc.blocking.inner_nblks);
            break;
        case dnnl_format_kind_wino:
            seed = hash_combine(seed,
                    static_cast<size_t>(md.format_desc.wino_desc.wino_format));
            seed = hash_combine(seed, md.format_desc.wino_desc.r);
            seed = hash_combine(seed, md.format_desc.wino_desc.alpha);
            seed = hash_combine(seed, md.format_desc.wino_desc.ic);
            seed = hash_combine(seed, md.format_desc.wino_desc.oc);
            seed = hash_combine(seed, md.format_desc.wino_desc.ic_block);
            seed = hash_combine(seed, md.format_desc.wino_desc.oc_block);
            seed = hash_combine(seed, md.format_desc.wino_desc.ic2_block);
            seed = hash_combine(seed, md.format_desc.wino_desc.oc2_block);
            seed = hash_combine(seed, md.format_desc.wino_desc.adj_scale);
            seed = hash_combine(seed, md.format_desc.wino_desc.size);
            break;
        case dnnl_format_kind_rnn_packed:
            seed = hash_combine(seed,
                    static_cast<size_t>(md.format_desc.rnn_packed_desc.format));
            seed = hash_combine(seed, md.format_desc.rnn_packed_desc.n_parts);
            seed = hash_combine(seed, md.format_desc.rnn_packed_desc.n);
            seed = hash_combine(seed, md.format_desc.rnn_packed_desc.ldb);
            {
                int n_parts = md.format_desc.rnn_packed_desc.n_parts;
                seed = get_array_hash(
                        seed, md.format_desc.rnn_packed_desc.parts, n_parts);
                seed = get_array_hash(seed,
                        md.format_desc.rnn_packed_desc.part_pack_size, n_parts);
                seed = get_array_hash(seed,
                        md.format_desc.rnn_packed_desc.pack_part, n_parts);
            }
            seed = hash_combine(
                    seed, md.format_desc.rnn_packed_desc.offset_compensation);
            seed = hash_combine(seed, md.format_desc.rnn_packed_desc.size);
            break;
        default: assert(!"unknown format_kind");
    }

    if (md.extra.flags != dnnl_memory_extra_flag_none) {
        seed = hash_combine(seed, md.extra.flags);
        if (md.extra.flags
                & (dnnl_memory_extra_flag_compensation_conv_s8s8
                        | dnnl_memory_extra_flag_rnn_u8s8_compensation)) {
            seed = hash_combine(seed, md.extra.compensation_mask);
        }

        if (md.extra.flags & dnnl_memory_extra_flag_scale_adjust) {
            seed = hash_combine(seed, md.extra.scale_adjust);
        }

        if (md.extra.flags
                & dnnl_memory_extra_flag_compensation_conv_asymmetric_src) {
            seed = hash_combine(seed, md.extra.asymm_compensation_mask);
        }
    }
    // Combined hash for a memory descriptor
    return seed;
}

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl
#endif
