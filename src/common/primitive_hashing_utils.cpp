/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
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

#include "utils.hpp"
#include "primitive_hashing_utils.hpp"

namespace dnnl {
namespace impl {
namespace primitive_hashing {

// Combine hash of each memory_desc_t data member
size_t get_md_hash(const memory_desc_t &md) {
    size_t seed = 0;
    seed = get_array_hash(seed, md.dims, md.ndims);
    seed = hash_combine(seed, static_cast<size_t>(md.data_type));
    seed = get_array_hash(seed, md.padded_dims, md.ndims);
    seed = get_array_hash(seed, md.padded_offsets, md.ndims);
    seed = hash_combine(seed, md.offset0);
    seed = hash_combine(seed, static_cast<size_t>(md.format_kind));
    // format desc
    switch ((int)md.format_kind) {
        case format_kind::undef:
        case format_kind::any: break;
        case format_kind::blocked:
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
        case format_kind::wino:
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
        case format_kind::rnn_packed:
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
#ifdef DNNL_EXPERIMENTAL_SPARSE
        case format_kind::sparse:
            seed = hash_combine(seed,
                    static_cast<size_t>(md.format_desc.sparse_desc.encoding));
            seed = hash_combine(seed, md.format_desc.sparse_desc.nnz);
            seed = get_array_hash(seed,
                    md.format_desc.sparse_desc.metadata_types,
                    sparse_desc_t::max_metadata_types);
            break;
#endif
        case format_kind::sparse:
            seed = hash_combine(seed,
                    static_cast<size_t>(md.format_desc.sparse_desc.encoding));
            // User cannot initialize `packed_desc` therefore therefore
            // at this point `packed_desc` is always zero initialized.
            break;
        default: assert(!"unknown format_kind");
    }

    if (md.extra.flags != dnnl_memory_extra_flag_none) {
        seed = hash_combine(seed, md.extra.flags);
        if ((md.extra.flags
                    & (dnnl_memory_extra_flag_compensation_conv_s8s8
                            | dnnl_memory_extra_flag_rnn_u8s8_compensation))
                && !types::extra_flag_rnn_s8s8_compensation_is_set(
                        md.extra.flags)) {
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

// Combine hash of each primitive_attr_t data member
size_t get_attr_hash(const primitive_attr_t &attr) {
    size_t seed = 0;
    // scratchpad_mode
    seed = hash_combine(seed, static_cast<size_t>(attr.scratchpad_mode_));
    // fpmath_mode
    seed = hash_combine(seed, static_cast<size_t>(attr.fpmath_mode_));

    if (!attr.output_scales_.has_default_values()) {
        // output_scales: mask
        seed = hash_combine(seed, attr.output_scales_.mask_);
    } else if (!attr.scales_.has_default_values()) {
        // go through scales for all arguments
        for (const auto &p : attr.scales_.scales_) {
            // scales: arg
            seed = hash_combine(seed, p.first);
            // scales: mask
            seed = hash_combine(seed, p.second.mask_);
        }
    }
    // zero_points
    for (int arg : {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST})
        if (!attr.zero_points_.has_default_values(arg)) {
            // zero_points: arg
            seed = hash_combine(seed, arg);
            int mask = 0;
            attr.zero_points_.get(arg, &mask);
            // zero_points: mask
            seed = hash_combine(seed, mask);
        }
    // post_ops: entry[:]
    seed = get_post_op_hash(seed, attr.post_ops_);
    // rnn_data_qparams: scale, shift
    seed = hash_combine(seed, attr.rnn_data_qparams_.scale_);
    seed = hash_combine(seed, attr.rnn_data_qparams_.shift_);
    if (!attr.rnn_weights_qparams_.has_default_values()) {
        // rnn_weights_qparams: mask
        seed = hash_combine(seed, attr.rnn_weights_qparams_.mask_);
        // rnn_weights_qparams: count
        seed = hash_combine(seed, attr.rnn_weights_qparams_.count_);
        // rnn_weights_qparams: scales[:]
        seed = get_array_hash(seed, attr.rnn_weights_qparams_.scales_,
                attr.rnn_weights_qparams_.count_);
    }
    if (attr.gpu_attr_) {
        seed = hash_combine(seed, attr.gpu_attr_->get_hash());
    }
    // Combined hash for attributes
    return seed;
}

// Combine hash of each post_ops::entry_
size_t get_post_op_hash(size_t seed, const post_ops_t &post_ops) {
    for (int i = 0; i < post_ops.len(); i++) {
        const auto &entry = post_ops.entry_[i];
        switch (entry.kind) {
            case primitive_kind::eltwise:
                seed = hash_combine(
                        seed, static_cast<size_t>(entry.eltwise.alg));
                seed = hash_combine(seed, entry.eltwise.scale);
                seed = hash_combine(seed, entry.eltwise.alpha);
                seed = hash_combine(seed, entry.eltwise.beta);
                break;
            case primitive_kind::sum:
                seed = hash_combine(seed, entry.sum.scale);
                seed = hash_combine(seed, static_cast<size_t>(entry.sum.dt));
                break;
            case primitive_kind::convolution:
                seed = hash_combine(seed, static_cast<size_t>(entry.depthwise_conv_old.in_h));
                seed = hash_combine(seed, static_cast<size_t>(entry.depthwise_conv_old.in_w));
                seed = hash_combine(seed, static_cast<size_t>(entry.depthwise_conv_old.ker_h));
                seed = hash_combine(seed, static_cast<size_t>(entry.depthwise_conv_old.ker_w));
                seed = hash_combine(seed, static_cast<size_t>(entry.depthwise_conv_old.str_h));
                seed = hash_combine(seed, static_cast<size_t>(entry.depthwise_conv_old.str_w));
                seed = hash_combine(seed, static_cast<size_t>(entry.depthwise_conv_old.in_dt));
                break;
            case primitive_kind::binary:
                seed = hash_combine(
                        seed, static_cast<size_t>(entry.binary.alg));
                seed = hash_combine(
                        seed, get_md_hash(entry.binary.user_src1_desc));
                break;
            case primitive_kind::prelu:
                seed = hash_combine(
                        seed, static_cast<size_t>(entry.prelu.mask));
                break;
            case primitive_kind::depthwise:
                seed = hash_combine(seed, static_cast<size_t>(entry.depthwise.alg));
                seed = get_array_hash(seed, entry.depthwise.offset, entry.depthwise.fields_count);
                break;
            case primitive_kind::quantization:
                seed = hash_combine(seed, static_cast<size_t>(entry.quantization.alg));
                seed = get_array_hash(seed, entry.quantization.per_channel, entry.quantization.fields_count);
                seed = get_array_hash(seed, entry.quantization.all_default, entry.quantization.fields_count);
                seed = get_array_hash(seed, entry.quantization.offset, entry.quantization.fields_count);
                break;
            default: assert(!"unknown post_op");
        }
    }

    return seed;
}

} // namespace primitive_hashing
} // namespace impl
} // namespace dnnl
