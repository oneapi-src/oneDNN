/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#ifndef PRIMITIVE_HASHING_HPP
#define PRIMITIVE_HASHING_HPP

#include <typeindex>

#include "c_types_map.hpp"
#include "dnnl.h"
#include "type_helpers.hpp"

namespace dnnl {
namespace impl {
namespace primitive_hashing {

struct key_t {
    key_t(const primitive_desc_t *pd, int impl_nthr);

    bool operator==(const key_t &rhs) const;

    dnnl_primitive_kind_t primitive_kind_;
    const op_desc_t *op_desc_;
    const primitive_attr_t *attr_;
    std::type_index impl_id_;
    int impl_nthr_;
    std::vector<memory_desc_t> mds;

private:
    template <typename T>
    bool cast_and_compare(const op_desc_t *lhs, const op_desc_t *rhs) const {
        return *(reinterpret_cast<const T *>(lhs))
                == *(reinterpret_cast<const T *>(rhs));
    }

    void init_mds(const primitive_desc_t *pd);
};

// The following code is derived from Boost C++ library
// Copyright 2005-2014 Daniel James.
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
template <typename T>
static size_t hash_combine(size_t seed, const T &v) {
    return seed ^= std::hash<T> {}(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

template <typename T>
static size_t get_array_hash(size_t seed, const T *v, int size) {
    for (int i = 0; i < size; i++) {
        seed = hash_combine(seed, v[i]);
    }
    return seed;
}

static inline size_t get_md_hash(const memory_desc_t &md);
// Specialization for an array of mds
template <>
size_t get_array_hash<memory_desc_t>(
        size_t seed, const memory_desc_t *v, int size) {
    for (int i = 0; i < size; i++) {
        seed = hash_combine(seed, get_md_hash(v[i]));
    }
    return seed;
}

// Combine hash of each primitive_attr_t data member
static inline size_t get_attr_hash(const primitive_attr_t *attr) {
    size_t seed = 0;
    // scratchpad_mode
    seed = hash_combine(seed, static_cast<size_t>(attr->scratchpad_mode_));

    if (!attr->output_scales_.has_default_values()) {
        // output_scales: mask
        seed = hash_combine(seed, attr->output_scales_.mask_);
        // output_scales: scales[:]
        if (attr->output_scales_.scales_) {
            for (int i = 0; i < attr->output_scales_.count_; i++) {
                seed = hash_combine(seed, attr->output_scales_.scales_[i]);
            }
        }
    } else if (!attr->scales_.has_default_values()) {
        // go through scales for all arguments
        for (const auto &p : attr->scales_.scales_) {
            seed = hash_combine(seed, p.second.mask_);
            for (int i = 0; i < p.second.count_; i++) {
                seed = hash_combine(seed, p.second.scales_[i]);
            }
        }
    }
    // zero_points
    for (int arg : {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST})
        seed = hash_combine(seed, *attr->zero_points_.get(arg));
    // post_ops: entry[:]
    for (int i = 0; i < attr->post_ops_.len_; i++) {
        const auto &entry = attr->post_ops_.entry_[i];
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
                break;
            default: assert(!"unknown post_op");
        }
    }
    // rnn_data_qparams: scale, shift
    seed = hash_combine(seed, attr->rnn_data_qparams_.scale_);
    seed = hash_combine(seed, attr->rnn_data_qparams_.shift_);
    // rnn_weights_qparams: mask
    seed = hash_combine(seed, attr->rnn_weights_qparams_.mask_);
    // rnn_weights_qparams_: scales[:]
    if (attr->rnn_weights_qparams_.scales_) {
        for (int i = 0; i < attr->rnn_weights_qparams_.count_; i++) {
            seed = hash_combine(seed, attr->rnn_weights_qparams_.scales_[i]);
        }
    }
    // Combined hash for attributes
    return seed;
}

// Combine hash of each memory_desc_t data member
static inline size_t get_md_hash(const memory_desc_t &md) {
    size_t seed = 0;
    seed = get_array_hash(seed, md.dims, DNNL_MAX_NDIMS);
    seed = hash_combine(seed, static_cast<size_t>(md.data_type));
    seed = get_array_hash(seed, md.padded_dims, DNNL_MAX_NDIMS);
    seed = get_array_hash(seed, md.padded_offsets, DNNL_MAX_NDIMS);
    seed = hash_combine(seed, md.offset0);
    seed = hash_combine(seed, static_cast<size_t>(md.format_kind));
    // format desc
    switch (md.format_kind) {
        case format_kind::undef:
        case format_kind::any: break;
        case format_kind::blocked:
            seed = get_array_hash(
                    seed, md.format_desc.blocking.strides, DNNL_MAX_NDIMS);
            seed = hash_combine(seed, md.format_desc.blocking.inner_nblks);
            seed = get_array_hash(
                    seed, md.format_desc.blocking.inner_blks, DNNL_MAX_NDIMS);
            seed = get_array_hash(
                    seed, md.format_desc.blocking.inner_idxs, DNNL_MAX_NDIMS);
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
            seed = get_array_hash(seed, md.format_desc.rnn_packed_desc.parts,
                    DNNL_RNN_MAX_N_PARTS);
            seed = get_array_hash(seed,
                    md.format_desc.rnn_packed_desc.part_pack_size,
                    DNNL_RNN_MAX_N_PARTS);
            seed = get_array_hash(seed,
                    md.format_desc.rnn_packed_desc.pack_part,
                    DNNL_RNN_MAX_N_PARTS);
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
                        | dnnl_memory_extra_flag_gpu_rnn_u8s8_compensation)) {
            seed = hash_combine(seed, md.extra.compensation_mask);
        }

        if (md.extra.flags & dnnl_memory_extra_flag_scale_adjust) {
            seed = hash_combine(seed, md.extra.scale_adjust);
        }
    }
    // Combined hash for a memory descriptor
    return seed;
}

// Functions that compute hash for different op_descs
template <typename T>
static size_t get_desc_hash(const op_desc_t *op_desc);

template <>
size_t get_desc_hash<concat_desc_t>(const op_desc_t *op_desc) {
    const auto *desc = reinterpret_cast<const concat_desc_t *>(op_desc);
    size_t seed = 0;
    // Kinds
    seed = hash_combine(seed, static_cast<size_t>(desc->primitive_kind));
    // Memory descriptors
    seed = hash_combine(seed, get_md_hash(desc->dst_md));
    // N
    seed = hash_combine(seed, desc->n);
    // Concat dimension
    seed = hash_combine(seed, desc->concat_dimension);
    // Array of mds
    seed = get_array_hash(seed, desc->src_mds.data(), desc->n);
    // Combined hash for concat desc
    return seed;
}

template <>
size_t get_desc_hash<batch_normalization_desc_t>(const op_desc_t *op_desc) {
    const auto *desc
            = reinterpret_cast<const batch_normalization_desc_t *>(op_desc);
    size_t seed = 0;
    // Kinds
    seed = hash_combine(seed, static_cast<size_t>(desc->primitive_kind));
    seed = hash_combine(seed, static_cast<size_t>(desc->prop_kind));
    // Memory descriptors
    seed = hash_combine(seed, get_md_hash(desc->data_desc));
    seed = hash_combine(seed, get_md_hash(desc->diff_data_desc));
    seed = hash_combine(seed, get_md_hash(desc->data_scaleshift_desc));
    seed = hash_combine(seed, get_md_hash(desc->diff_data_scaleshift_desc));
    seed = hash_combine(seed, get_md_hash(desc->stat_desc));
    // Epsilon
    seed = hash_combine(seed, desc->batch_norm_epsilon);
    // Flags
    seed = hash_combine(seed, desc->flags);
    // Combined hash for batch normalization desc
    return seed;
}

template <>
size_t get_desc_hash<binary_desc_t>(const op_desc_t *op_desc) {
    const auto *desc = reinterpret_cast<const binary_desc_t *>(op_desc);
    size_t seed = 0;
    // Kinds
    seed = hash_combine(seed, static_cast<size_t>(desc->primitive_kind));
    seed = hash_combine(seed, static_cast<size_t>(desc->alg_kind));
    // Memory descriptors
    seed = hash_combine(seed, get_md_hash(desc->src_desc[0]));
    seed = hash_combine(seed, get_md_hash(desc->src_desc[1]));
    seed = hash_combine(seed, get_md_hash(desc->dst_desc));
    // Combined hash for binary op desc
    return seed;
}

// (De-)Convolution
template <>
size_t get_desc_hash<convolution_desc_t>(const op_desc_t *op_desc) {
    const auto *desc = reinterpret_cast<const convolution_desc_t *>(op_desc);
    size_t seed = 0;
    // Kinds
    seed = hash_combine(seed, static_cast<size_t>(desc->primitive_kind));
    seed = hash_combine(seed, static_cast<size_t>(desc->prop_kind));
    seed = hash_combine(seed, static_cast<size_t>(desc->alg_kind));
    // Memory descriptors
    seed = hash_combine(seed, get_md_hash(desc->src_desc));
    seed = hash_combine(seed, get_md_hash(desc->diff_src_desc));
    seed = hash_combine(seed, get_md_hash(desc->weights_desc));
    seed = hash_combine(seed, get_md_hash(desc->diff_weights_desc));
    seed = hash_combine(seed, get_md_hash(desc->bias_desc));
    seed = hash_combine(seed, get_md_hash(desc->diff_bias_desc));
    seed = hash_combine(seed, get_md_hash(desc->dst_desc));
    seed = hash_combine(seed, get_md_hash(desc->diff_dst_desc));
    // Strides, dilates, padding
    seed = get_array_hash(seed, desc->strides, DNNL_MAX_NDIMS);
    seed = get_array_hash(seed, desc->dilates, DNNL_MAX_NDIMS);
    seed = get_array_hash(seed, desc->padding[0], DNNL_MAX_NDIMS);
    seed = get_array_hash(seed, desc->padding[1], DNNL_MAX_NDIMS);
    // Accumulator type
    seed = hash_combine(seed, static_cast<size_t>(desc->accum_data_type));
    // Combined hash for (de-)convolution desc
    return seed;
}

// Eltwise
template <>
size_t get_desc_hash<eltwise_desc_t>(const op_desc_t *op_desc) {
    const auto *desc = reinterpret_cast<const eltwise_desc_t *>(op_desc);
    size_t seed = 0;
    // Kinds
    seed = hash_combine(seed, static_cast<size_t>(desc->primitive_kind));
    seed = hash_combine(seed, static_cast<size_t>(desc->prop_kind));
    seed = hash_combine(seed, static_cast<size_t>(desc->alg_kind));
    // Memory descriptors
    seed = hash_combine(seed, get_md_hash(desc->data_desc));
    seed = hash_combine(seed, get_md_hash(desc->diff_data_desc));
    // Alpha, beta
    seed = hash_combine(seed, desc->alpha);
    seed = hash_combine(seed, desc->beta);
    // Combined hash for eltwise desc
    return seed;
}

template <>
size_t get_desc_hash<gemm_desc_t>(const op_desc_t *op_desc) {
    const auto *desc = reinterpret_cast<const gemm_desc_t *>(op_desc);
    size_t seed = 0;
    // Kinds
    seed = hash_combine(seed, static_cast<size_t>(desc->primitive_kind));
    // Trans
    seed = hash_combine(seed, static_cast<size_t>(desc->transa));
    seed = hash_combine(seed, static_cast<size_t>(desc->transb));
    // M, N, K
    seed = hash_combine(seed, desc->m);
    seed = hash_combine(seed, desc->n);
    seed = hash_combine(seed, desc->k);
    // LDA, LDB, LDC
    seed = hash_combine(seed, desc->lda);
    seed = hash_combine(seed, desc->ldb);
    seed = hash_combine(seed, desc->ldc);
    // Alpha, beta
    seed = hash_combine(seed, desc->alpha);
    seed = hash_combine(seed, desc->beta);
    // a_type, b_type, c_type, acc_type
    seed = hash_combine(seed, static_cast<size_t>(desc->a_type));
    seed = hash_combine(seed, static_cast<size_t>(desc->b_type));
    seed = hash_combine(seed, static_cast<size_t>(desc->c_type));
    seed = hash_combine(seed, static_cast<size_t>(desc->acc_type));
    // Combined hash for gemm desc
    return seed;
}

template <>
size_t get_desc_hash<inner_product_desc_t>(const op_desc_t *op_desc) {
    const auto *desc = reinterpret_cast<const inner_product_desc_t *>(op_desc);
    size_t seed = 0;
    // Kinds
    seed = hash_combine(seed, static_cast<size_t>(desc->primitive_kind));
    seed = hash_combine(seed, static_cast<size_t>(desc->prop_kind));
    // Memory descriptors
    seed = hash_combine(seed, get_md_hash(desc->src_desc));
    seed = hash_combine(seed, get_md_hash(desc->diff_src_desc));
    seed = hash_combine(seed, get_md_hash(desc->weights_desc));
    seed = hash_combine(seed, get_md_hash(desc->diff_weights_desc));
    seed = hash_combine(seed, get_md_hash(desc->bias_desc));
    seed = hash_combine(seed, get_md_hash(desc->diff_bias_desc));
    seed = hash_combine(seed, get_md_hash(desc->dst_desc));
    seed = hash_combine(seed, get_md_hash(desc->diff_dst_desc));
    // Accumulator type
    seed = hash_combine(seed, static_cast<size_t>(desc->accum_data_type));
    // Combined hash for inner_product desc
    return seed;
}

// Layer normalization
template <>
size_t get_desc_hash<layer_normalization_desc_t>(const op_desc_t *op_desc) {
    const auto *desc
            = reinterpret_cast<const layer_normalization_desc_t *>(op_desc);
    size_t seed = 0;
    // Kinds
    seed = hash_combine(seed, static_cast<size_t>(desc->primitive_kind));
    seed = hash_combine(seed, static_cast<size_t>(desc->prop_kind));
    // Memory descriptors
    seed = hash_combine(seed, get_md_hash(desc->data_desc));
    seed = hash_combine(seed, get_md_hash(desc->diff_data_desc));
    seed = hash_combine(seed, get_md_hash(desc->data_scaleshift_desc));
    seed = hash_combine(seed, get_md_hash(desc->diff_data_scaleshift_desc));
    seed = hash_combine(seed, get_md_hash(desc->stat_desc));
    // Epsilon
    seed = hash_combine(seed, desc->layer_norm_epsilon);
    // Flags
    seed = hash_combine(seed, desc->flags);
    // Combined hash for layer_normalization desc
    return seed;
}

template <>
size_t get_desc_hash<lrn_desc_t>(const op_desc_t *op_desc) {
    const auto *desc = reinterpret_cast<const lrn_desc_t *>(op_desc);
    size_t seed = 0;
    // Kinds
    seed = hash_combine(seed, static_cast<size_t>(desc->primitive_kind));
    seed = hash_combine(seed, static_cast<size_t>(desc->prop_kind));
    seed = hash_combine(seed, static_cast<size_t>(desc->alg_kind));
    // Memory descriptors
    seed = hash_combine(seed, get_md_hash(desc->data_desc));
    seed = hash_combine(seed, get_md_hash(desc->diff_data_desc));
    // Local size
    seed = hash_combine(seed, desc->local_size);
    // Alpha, beta
    seed = hash_combine(seed, desc->lrn_alpha);
    seed = hash_combine(seed, desc->lrn_beta);
    // k
    seed = hash_combine(seed, desc->lrn_k);
    // Combined hash for lrn desc
    return seed;
}

template <>
size_t get_desc_hash<matmul_desc_t>(const op_desc_t *op_desc) {
    const auto *desc = reinterpret_cast<const matmul_desc_t *>(op_desc);
    size_t seed = 0;
    // Kinds
    seed = hash_combine(seed, static_cast<size_t>(desc->primitive_kind));
    // Memory descriptors
    seed = hash_combine(seed, get_md_hash(desc->src_desc));
    seed = hash_combine(seed, get_md_hash(desc->weights_desc));
    seed = hash_combine(seed, get_md_hash(desc->bias_desc));
    seed = hash_combine(seed, get_md_hash(desc->dst_desc));
    // Accumulator type
    seed = hash_combine(seed, static_cast<size_t>(desc->accum_data_type));
    // Combined hash for matmul op desc
    return seed;
}

template <>
size_t get_desc_hash<pooling_desc_t>(const op_desc_t *op_desc) {
    const auto *desc = reinterpret_cast<const pooling_desc_t *>(op_desc);
    size_t seed = 0;
    // Kinds
    seed = hash_combine(seed, static_cast<size_t>(desc->primitive_kind));
    seed = hash_combine(seed, static_cast<size_t>(desc->prop_kind));
    seed = hash_combine(seed, static_cast<size_t>(desc->alg_kind));
    // Memory descriptors
    seed = hash_combine(seed, get_md_hash(desc->src_desc));
    seed = hash_combine(seed, get_md_hash(desc->diff_src_desc));
    seed = hash_combine(seed, get_md_hash(desc->dst_desc));
    seed = hash_combine(seed, get_md_hash(desc->diff_dst_desc));
    // Strides, dilates, padding
    seed = get_array_hash(seed, desc->strides, DNNL_MAX_NDIMS);
    seed = get_array_hash(seed, desc->kernel, DNNL_MAX_NDIMS);
    seed = get_array_hash(seed, desc->padding[0], DNNL_MAX_NDIMS);
    seed = get_array_hash(seed, desc->padding[1], DNNL_MAX_NDIMS);
    // Accumulator type
    seed = hash_combine(seed, static_cast<size_t>(desc->accum_data_type));
    // Combined hash for pooling desc
    return seed;
}

template <>
size_t get_desc_hash<reorder_desc_t>(const op_desc_t *op_desc) {
    const auto *desc = reinterpret_cast<const reorder_desc_t *>(op_desc);
    size_t seed = 0;
    // Kinds
    seed = hash_combine(seed, static_cast<size_t>(desc->primitive_kind));
    // Memory descriptors
    seed = hash_combine(seed, get_md_hash(desc->src_md));
    seed = hash_combine(seed, get_md_hash(desc->dst_md));
    // Kinds of source and destination engines
    seed = hash_combine(seed, static_cast<size_t>(desc->src_engine_kind));
    seed = hash_combine(seed, static_cast<size_t>(desc->dst_engine_kind));
    // Combined hash for reorder desc
    return seed;
}

template <>
size_t get_desc_hash<resampling_desc_t>(const op_desc_t *op_desc) {
    const auto *desc = reinterpret_cast<const resampling_desc_t *>(op_desc);
    size_t seed = 0;
    // Kinds
    seed = hash_combine(seed, static_cast<size_t>(desc->primitive_kind));
    seed = hash_combine(seed, static_cast<size_t>(desc->alg_kind));
    // Memory descriptors
    seed = hash_combine(seed, get_md_hash(desc->src_desc));
    seed = hash_combine(seed, get_md_hash(desc->diff_src_desc));
    seed = hash_combine(seed, get_md_hash(desc->dst_desc));
    seed = hash_combine(seed, get_md_hash(desc->diff_dst_desc));
    // Factors
    seed = get_array_hash(seed, desc->factors, DNNL_MAX_NDIMS);
    // Combined hash for resampling op desc
    return seed;
}

template <>
size_t get_desc_hash<rnn_desc_t>(const op_desc_t *op_desc) {
    const auto *desc = reinterpret_cast<const rnn_desc_t *>(op_desc);
    size_t seed = 0;
    // Kinds
    seed = hash_combine(seed, static_cast<size_t>(desc->primitive_kind));
    seed = hash_combine(seed, static_cast<size_t>(desc->prop_kind));
    seed = hash_combine(seed, static_cast<size_t>(desc->cell_kind));
    seed = hash_combine(seed, static_cast<size_t>(desc->direction));
    // Memory descriptors
    seed = hash_combine(seed, get_md_hash(desc->src_layer_desc));
    seed = hash_combine(seed, get_md_hash(desc->src_iter_desc));
    seed = hash_combine(seed, get_md_hash(desc->src_iter_c_desc));
    seed = hash_combine(seed, get_md_hash(desc->weights_layer_desc));
    seed = hash_combine(seed, get_md_hash(desc->weights_iter_desc));
    seed = hash_combine(seed, get_md_hash(desc->bias_desc));
    seed = hash_combine(seed, get_md_hash(desc->dst_layer_desc));
    seed = hash_combine(seed, get_md_hash(desc->dst_iter_desc));
    seed = hash_combine(seed, get_md_hash(desc->dst_iter_c_desc));
    seed = hash_combine(seed, get_md_hash(desc->placeholder_desc));
    seed = hash_combine(seed, get_md_hash(desc->placeholder2_desc));
    seed = hash_combine(seed, get_md_hash(desc->diff_src_layer_desc));
    seed = hash_combine(seed, get_md_hash(desc->diff_src_iter_desc));
    seed = hash_combine(seed, get_md_hash(desc->diff_src_iter_c_desc));
    seed = hash_combine(seed, get_md_hash(desc->diff_weights_layer_desc));
    seed = hash_combine(seed, get_md_hash(desc->diff_weights_iter_desc));
    seed = hash_combine(seed, get_md_hash(desc->diff_bias_desc));
    seed = hash_combine(seed, get_md_hash(desc->diff_dst_layer_desc));
    seed = hash_combine(seed, get_md_hash(desc->diff_dst_iter_desc));
    seed = hash_combine(seed, get_md_hash(desc->diff_dst_iter_c_desc));
    seed = hash_combine(seed, get_md_hash(desc->diff_placeholder_desc));
    seed = hash_combine(seed, get_md_hash(desc->diff_placeholder2_desc));
    // Flags
    seed = hash_combine(seed, desc->flags);
    // Activation kind
    seed = hash_combine(seed, static_cast<size_t>(desc->activation_kind));
    // Alpha, beta
    seed = hash_combine(seed, desc->alpha);
    seed = hash_combine(seed, desc->beta);
    // Combined hash for rnn desc
    return seed;
}

// Shuffle
template <>
size_t get_desc_hash<shuffle_desc_t>(const op_desc_t *op_desc) {
    const auto *desc = reinterpret_cast<const shuffle_desc_t *>(op_desc);
    size_t seed = 0;
    // Kinds
    seed = hash_combine(seed, static_cast<size_t>(desc->primitive_kind));
    seed = hash_combine(seed, static_cast<size_t>(desc->prop_kind));
    // Memory descriptors
    seed = hash_combine(seed, get_md_hash(desc->data_desc));
    // Axis
    seed = hash_combine(seed, desc->axis);
    // Groupe size
    seed = hash_combine(seed, desc->group_size);
    // Combined hash for shuffle desc
    return seed;
}

template <>
size_t get_desc_hash<softmax_desc_t>(const op_desc_t *op_desc) {
    const auto *desc = reinterpret_cast<const softmax_desc_t *>(op_desc);
    size_t seed = 0;
    // Kinds
    seed = hash_combine(seed, static_cast<size_t>(desc->primitive_kind));
    seed = hash_combine(seed, static_cast<size_t>(desc->prop_kind));
    // Memory descriptors
    seed = hash_combine(seed, get_md_hash(desc->data_desc));
    seed = hash_combine(seed, get_md_hash(desc->diff_desc));
    // Axis
    seed = hash_combine(seed, desc->softmax_axis);
    // Combined hash for softmax desc
    return seed;
}

template <>
size_t get_desc_hash<sum_desc_t>(const op_desc_t *op_desc) {
    const auto *desc = reinterpret_cast<const sum_desc_t *>(op_desc);
    size_t seed = 0;
    // Kinds
    seed = hash_combine(seed, static_cast<size_t>(desc->primitive_kind));
    // Memory descriptors
    seed = hash_combine(seed, get_md_hash(desc->dst_md));
    // N
    seed = hash_combine(seed, desc->n);
    // Scales
    if (!desc->scales.empty()) {
        seed = get_array_hash(seed, desc->scales.data(), desc->n);
    }
    // Array of mds
    seed = get_array_hash(seed, desc->src_mds.data(), desc->n);
    // Combined hash for sum desc
    return seed;
}

} // namespace primitive_hashing
} // namespace impl
} // namespace dnnl

// inject a specialization of std::hash for key_t in std namespace
namespace std {
template <>
struct hash<dnnl::impl::primitive_hashing::key_t> {
    using argument_type = dnnl::impl::primitive_hashing::key_t;
    using result_type = std::size_t;
    result_type operator()(const argument_type &key) const {
        using namespace dnnl::impl;
        using namespace dnnl::impl::primitive_hashing;
        size_t seed = 0;
        // Compute hash for primitive_kind_, attr_, impl_id_ and impl_nthr_
        seed = hash_combine(seed,
                hash_combine(0, static_cast<size_t>(key.primitive_kind_)));
        seed = hash_combine(seed, get_attr_hash(key.attr_));
        seed = hash_combine(seed, hash_combine(0, key.impl_id_));
        seed = hash_combine(seed, hash_combine(0, key.impl_nthr_));
        // Combine hash for op_desc with the computed hash
        switch (key.primitive_kind_) {
            case primitive_kind::batch_normalization:
                seed = hash_combine(seed,
                        get_desc_hash<batch_normalization_desc_t>(
                                key.op_desc_));
                break;
            case primitive_kind::binary:
                seed = hash_combine(
                        seed, get_desc_hash<binary_desc_t>(key.op_desc_));
                break;
            case primitive_kind::concat:
                seed = hash_combine(
                        seed, get_desc_hash<concat_desc_t>(key.op_desc_));
                break;
            case primitive_kind::convolution:
                seed = hash_combine(
                        seed, get_desc_hash<convolution_desc_t>(key.op_desc_));
                break;
            case primitive_kind::deconvolution:
                seed = hash_combine(seed,
                        get_desc_hash<deconvolution_desc_t>(key.op_desc_));
                break;
            case primitive_kind::eltwise:
                seed = hash_combine(
                        seed, get_desc_hash<eltwise_desc_t>(key.op_desc_));
                break;
            case primitive_kind::gemm:
                seed = hash_combine(
                        seed, get_desc_hash<gemm_desc_t>(key.op_desc_));
                break;
            case primitive_kind::inner_product:
                seed = hash_combine(seed,
                        get_desc_hash<inner_product_desc_t>(key.op_desc_));
                break;
            case primitive_kind::layer_normalization:
                seed = hash_combine(seed,
                        get_desc_hash<layer_normalization_desc_t>(
                                key.op_desc_));
                break;
            case primitive_kind::logsoftmax:
                seed = hash_combine(
                        seed, get_desc_hash<logsoftmax_desc_t>(key.op_desc_));
                break;
            case primitive_kind::lrn:
                seed = hash_combine(
                        seed, get_desc_hash<lrn_desc_t>(key.op_desc_));
                break;
            case primitive_kind::matmul:
                seed = hash_combine(
                        seed, get_desc_hash<matmul_desc_t>(key.op_desc_));
                break;
            case primitive_kind::pooling:
                seed = hash_combine(
                        seed, get_desc_hash<pooling_desc_t>(key.op_desc_));
                break;
            case primitive_kind::reorder:
                seed = hash_combine(
                        seed, get_desc_hash<reorder_desc_t>(key.op_desc_));
                break;
            case primitive_kind::resampling:
                seed = hash_combine(
                        seed, get_desc_hash<resampling_desc_t>(key.op_desc_));
                break;
            case primitive_kind::rnn:
                seed = hash_combine(
                        seed, get_desc_hash<rnn_desc_t>(key.op_desc_));
                break;
            case primitive_kind::shuffle:
                seed = hash_combine(
                        seed, get_desc_hash<shuffle_desc_t>(key.op_desc_));
                break;
            case primitive_kind::softmax:
                seed = hash_combine(
                        seed, get_desc_hash<softmax_desc_t>(key.op_desc_));
                break;
            case primitive_kind::sum:
                seed = hash_combine(
                        seed, get_desc_hash<sum_desc_t>(key.op_desc_));
                break;
            default: assert(!"unknown primitive_kind");
        }

        seed = get_array_hash(seed, key.mds.data(), (int)key.mds.size());

        return seed;
    }
};

} // namespace std

#endif
