/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#ifndef COMMON_PRIMITIVE_HASHING_HPP
#define COMMON_PRIMITIVE_HASHING_HPP

#include <typeindex>
#include <type_traits>

#include "c_types_map.hpp"
#include "dnnl.h"
#include "primitive_attr.hpp"
#include "type_helpers.hpp"

namespace dnnl {
namespace impl {

struct primitive_desc_t;

namespace primitive_hashing {

// TODO: Consider replacing op_desc_t with cached_op_desc_t
struct cached_op_desc_t {
    cached_op_desc_t(primitive_kind_t kind, const op_desc_t *op_desc)
        : kind_(get_kind(kind)) {

#define CASE(pkind) \
    case primitive_kind::pkind: \
        new (&pkind) pkind##_desc_t( \
                *(reinterpret_cast<const pkind##_desc_t *>(op_desc))); \
        break;

        switch (kind_) {
            CASE(batch_normalization)
            CASE(binary)
            CASE(convolution)
            CASE(eltwise)
            CASE(gemm)
            CASE(inner_product)
            CASE(layer_normalization)
            CASE(lrn)
            CASE(matmul)
            CASE(pooling)
            CASE(reorder)
            CASE(resampling)
            CASE(rnn)
            CASE(shuffle)
            CASE(softmax)
            CASE(concat)
            CASE(sum)
            default: assert(!"unknown primitive kind");
        }
#undef CASE
    }

    cached_op_desc_t(const cached_op_desc_t &other) : kind_(other.kind_) {

#define CASE(pkind) \
    case primitive_kind::pkind: new (&pkind) pkind##_desc_t(other.pkind); break;

        switch (kind_) {
            CASE(batch_normalization)
            CASE(binary)
            CASE(convolution)
            CASE(eltwise)
            CASE(gemm)
            CASE(inner_product)
            CASE(layer_normalization)
            CASE(lrn)
            CASE(matmul)
            CASE(pooling)
            CASE(reorder)
            CASE(resampling)
            CASE(rnn)
            CASE(shuffle)
            CASE(softmax)
            CASE(concat)
            CASE(sum)
            default: assert(!"unknown primitive kind");
        }
#undef CASE
    }

    bool operator==(const cached_op_desc_t &other) const {
        if (kind_ != other.kind_) return false;

#define CASE(pkind) \
    case primitive_kind::pkind: ret = pkind == other.pkind; break;

        bool ret = true;
        switch (kind_) {
            CASE(batch_normalization)
            CASE(binary)
            CASE(concat)
            CASE(convolution)
            CASE(eltwise)
            CASE(gemm)
            CASE(inner_product)
            CASE(layer_normalization)
            CASE(lrn)
            CASE(matmul)
            CASE(pooling)
            CASE(reorder)
            CASE(resampling)
            CASE(rnn)
            CASE(shuffle)
            CASE(softmax)
            CASE(sum)
            default: assert(!"unknown primitive kind");
        }
#undef CASE
        return ret;
    }

#define DECLARE_CONVERSION_OPERATOR(pkind) \
    operator pkind##_desc_t() const { \
        assert(kind_ == primitive_kind::pkind); \
        return pkind; \
    }
    DECLARE_CONVERSION_OPERATOR(batch_normalization)
    DECLARE_CONVERSION_OPERATOR(binary)
    DECLARE_CONVERSION_OPERATOR(concat)
    DECLARE_CONVERSION_OPERATOR(convolution)
    DECLARE_CONVERSION_OPERATOR(eltwise)
    DECLARE_CONVERSION_OPERATOR(gemm)
    DECLARE_CONVERSION_OPERATOR(inner_product)
    DECLARE_CONVERSION_OPERATOR(layer_normalization)
    DECLARE_CONVERSION_OPERATOR(lrn)
    DECLARE_CONVERSION_OPERATOR(matmul)
    DECLARE_CONVERSION_OPERATOR(pooling)
    DECLARE_CONVERSION_OPERATOR(reorder)
    DECLARE_CONVERSION_OPERATOR(resampling)
    DECLARE_CONVERSION_OPERATOR(rnn)
    DECLARE_CONVERSION_OPERATOR(shuffle)
    DECLARE_CONVERSION_OPERATOR(softmax)
    DECLARE_CONVERSION_OPERATOR(sum)
#undef DECLARE_CONVERSION_OPERATOR

    ~cached_op_desc_t() {
        switch (kind_) {
            case primitive_kind::batch_normalization:
            case primitive_kind::binary:
            case primitive_kind::convolution:
            case primitive_kind::eltwise:
            case primitive_kind::gemm:
            case primitive_kind::inner_product:
            case primitive_kind::layer_normalization:
            case primitive_kind::logsoftmax:
            case primitive_kind::lrn:
            case primitive_kind::matmul:
            case primitive_kind::pooling:
            case primitive_kind::reorder:
            case primitive_kind::resampling:
            case primitive_kind::rnn:
            case primitive_kind::shuffle:
            case primitive_kind::softmax: break;
            case primitive_kind::concat: concat.~dnnl_concat_desc_t(); break;
            case primitive_kind::sum: sum.~dnnl_sum_desc_t(); break;
            default: assert(!"unknown primitive_kind");
        }
    }

private:
    cached_op_desc_t() = delete;
    cached_op_desc_t &operator=(const cached_op_desc_t &) = delete;

    primitive_kind_t get_kind(primitive_kind_t kind) {
        auto k = primitive_kind::undefined;
        switch (kind) {
            case primitive_kind::softmax:
            case primitive_kind::logsoftmax: k = primitive_kind::softmax; break;
            case primitive_kind::convolution:
            case primitive_kind::deconvolution:
                k = primitive_kind::convolution;
                break;
            default: k = kind;
        }
        return k;
    }

    primitive_kind_t kind_;
    union {
        batch_normalization_desc_t batch_normalization;
        binary_desc_t binary;
        concat_desc_t concat;
        // This desc is common for convolution and deconvolution
        convolution_desc_t convolution;
        eltwise_desc_t eltwise;
        gemm_desc_t gemm;
        inner_product_desc_t inner_product;
        layer_normalization_desc_t layer_normalization;
        lrn_desc_t lrn;
        matmul_desc_t matmul;
        pooling_desc_t pooling;
        reorder_desc_t reorder;
        resampling_desc_t resampling;
        rnn_desc_t rnn;
        shuffle_desc_t shuffle;
        // This desc is common for softmax and logsoftmax
        softmax_desc_t softmax;
        sum_desc_t sum;
    };
};

struct key_t {
    key_t(const primitive_desc_t *pd, const engine_t *engine, int impl_nthr);

    // XXX: this ctor is used to create keys to compare pds
    // in 1x1 convolution + dw
    key_t(const primitive_desc_t *pd, int impl_nthr);

    bool operator==(const key_t &other) const;

    dnnl_primitive_kind_t primitive_kind_;
    cached_op_desc_t op_desc_;
    primitive_attr_t attr_;
    std::type_index impl_id_;
    int impl_nthr_;
    std::vector<memory_desc_t> mds;
    engine_kind_t kind_;
    runtime_kind_t runtime_kind_;
    intptr_t device_id_;

private:
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

inline size_t get_md_hash(const memory_desc_t &md);
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
inline size_t get_attr_hash(const primitive_attr_t &attr) {
    size_t seed = 0;
    // scratchpad_mode
    seed = hash_combine(seed, static_cast<size_t>(attr.scratchpad_mode_));

    if (!attr.output_scales_.has_default_values()) {
        // output_scales: mask
        seed = hash_combine(seed, attr.output_scales_.mask_);
        // output_scales: count
        seed = hash_combine(seed, attr.output_scales_.count_);
        // output_scales: scales[:]
        seed = get_array_hash(
                seed, attr.output_scales_.scales_, attr.output_scales_.count_);
    } else if (!attr.scales_.has_default_values()) {
        // go through scales for all arguments
        for (const auto &p : attr.scales_.scales_) {
            seed = hash_combine(seed, p.second.mask_);
            seed = hash_combine(seed, p.second.count_);
            seed = get_array_hash(seed, p.second.scales_, p.second.count_);
        }
    }
    // zero_points
    for (int arg : {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST})
        seed = hash_combine(seed, *attr.zero_points_.get(arg));
    // post_ops: entry[:]
    for (int i = 0; i < attr.post_ops_.len_; i++) {
        const auto &entry = attr.post_ops_.entry_[i];
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
            case primitive_kind::convolution:
                seed = hash_combine(
                        seed, static_cast<size_t>(entry.depthwise_conv.stride));
                seed = hash_combine(
                        seed, static_cast<size_t>(entry.depthwise_conv.wei_dt));
                seed = hash_combine(seed,
                        static_cast<size_t>(entry.depthwise_conv.bias_dt));
                seed = hash_combine(
                        seed, static_cast<size_t>(entry.depthwise_conv.dst_dt));
                if (entry.depthwise_conv.scales) {
                    seed = hash_combine(seed, entry.depthwise_conv.mask);
                    seed = hash_combine(seed, entry.depthwise_conv.count);
                    seed = get_array_hash(seed, entry.depthwise_conv.scales,
                            entry.depthwise_conv.count);
                }
                break;
            default: assert(!"unknown post_op");
        }
    }
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
    // Combined hash for attributes
    return seed;
}

// Combine hash of each memory_desc_t data member
inline size_t get_md_hash(const memory_desc_t &md) {
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
inline size_t get_desc_hash(const concat_desc_t &desc) {
    size_t seed = 0;
    // Kinds
    seed = hash_combine(seed, static_cast<size_t>(desc.primitive_kind));
    // Memory descriptors
    seed = hash_combine(seed, get_md_hash(desc.dst_md));
    // N
    seed = hash_combine(seed, desc.n);
    // Concat dimension
    seed = hash_combine(seed, desc.concat_dimension);
    // Array of mds
    seed = get_array_hash(seed, desc.src_mds.data(), desc.n);
    // Combined hash for concat desc
    return seed;
}

inline size_t get_desc_hash(const batch_normalization_desc_t &desc) {
    size_t seed = 0;
    // Kinds
    seed = hash_combine(seed, static_cast<size_t>(desc.primitive_kind));
    seed = hash_combine(seed, static_cast<size_t>(desc.prop_kind));
    // Memory descriptors
    seed = hash_combine(seed, get_md_hash(desc.data_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_data_desc));
    seed = hash_combine(seed, get_md_hash(desc.data_scaleshift_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_data_scaleshift_desc));
    seed = hash_combine(seed, get_md_hash(desc.stat_desc));
    // Epsilon
    seed = hash_combine(seed, desc.batch_norm_epsilon);
    // Flags
    seed = hash_combine(seed, desc.flags);
    // Combined hash for batch normalization desc
    return seed;
}

inline size_t get_desc_hash(const binary_desc_t &desc) {
    size_t seed = 0;
    // Kinds
    seed = hash_combine(seed, static_cast<size_t>(desc.primitive_kind));
    seed = hash_combine(seed, static_cast<size_t>(desc.alg_kind));
    // Memory descriptors
    seed = hash_combine(seed, get_md_hash(desc.src_desc[0]));
    seed = hash_combine(seed, get_md_hash(desc.src_desc[1]));
    seed = hash_combine(seed, get_md_hash(desc.dst_desc));
    // Combined hash for binary op desc
    return seed;
}

// (De-)Convolution
inline size_t get_desc_hash(const convolution_desc_t &desc) {
    size_t seed = 0;
    // Kinds
    seed = hash_combine(seed, static_cast<size_t>(desc.primitive_kind));
    seed = hash_combine(seed, static_cast<size_t>(desc.prop_kind));
    seed = hash_combine(seed, static_cast<size_t>(desc.alg_kind));
    // Memory descriptors
    seed = hash_combine(seed, get_md_hash(desc.src_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_src_desc));
    seed = hash_combine(seed, get_md_hash(desc.weights_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_weights_desc));
    seed = hash_combine(seed, get_md_hash(desc.bias_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_bias_desc));
    seed = hash_combine(seed, get_md_hash(desc.dst_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_dst_desc));
    // Strides, dilates, padding
    seed = get_array_hash(seed, desc.strides, DNNL_MAX_NDIMS);
    seed = get_array_hash(seed, desc.dilates, DNNL_MAX_NDIMS);
    seed = get_array_hash(seed, desc.padding[0], DNNL_MAX_NDIMS);
    seed = get_array_hash(seed, desc.padding[1], DNNL_MAX_NDIMS);
    // Accumulator type
    seed = hash_combine(seed, static_cast<size_t>(desc.accum_data_type));
    // Combined hash for (de-)convolution desc
    return seed;
}

// Eltwise
inline size_t get_desc_hash(const eltwise_desc_t &desc) {
    size_t seed = 0;
    // Kinds
    seed = hash_combine(seed, static_cast<size_t>(desc.primitive_kind));
    seed = hash_combine(seed, static_cast<size_t>(desc.prop_kind));
    seed = hash_combine(seed, static_cast<size_t>(desc.alg_kind));
    // Memory descriptors
    seed = hash_combine(seed, get_md_hash(desc.data_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_data_desc));
    // Alpha, beta
    seed = hash_combine(seed, desc.alpha);
    seed = hash_combine(seed, desc.beta);
    // Combined hash for eltwise desc
    return seed;
}

inline size_t get_desc_hash(const gemm_desc_t &desc) {
    size_t seed = 0;
    // Kinds
    seed = hash_combine(seed, static_cast<size_t>(desc.primitive_kind));
    // Trans
    seed = hash_combine(seed, static_cast<size_t>(desc.transa));
    seed = hash_combine(seed, static_cast<size_t>(desc.transb));
    // M, N, K
    seed = hash_combine(seed, desc.batch);
    seed = hash_combine(seed, desc.m);
    seed = hash_combine(seed, desc.n);
    seed = hash_combine(seed, desc.k);
    // Strides
    seed = hash_combine(seed, desc.stride_a);
    seed = hash_combine(seed, desc.stride_b);
    seed = hash_combine(seed, desc.stride_c);
    // LDA, LDB, LDC
    seed = hash_combine(seed, desc.lda);
    seed = hash_combine(seed, desc.ldb);
    seed = hash_combine(seed, desc.ldc);
    // bias mask
    seed = hash_combine(seed, static_cast<size_t>(desc.bias_mask));
    // a_type, b_type, c_type, acc_type, bias_type
    seed = hash_combine(seed, static_cast<size_t>(desc.a_type));
    seed = hash_combine(seed, static_cast<size_t>(desc.b_type));
    seed = hash_combine(seed, static_cast<size_t>(desc.c_type));
    seed = hash_combine(seed, static_cast<size_t>(desc.acc_type));
    seed = hash_combine(seed, static_cast<size_t>(desc.bias_type));
    // Combined hash for gemm desc
    return seed;
}

inline size_t get_desc_hash(const inner_product_desc_t &desc) {
    size_t seed = 0;
    // Kinds
    seed = hash_combine(seed, static_cast<size_t>(desc.primitive_kind));
    seed = hash_combine(seed, static_cast<size_t>(desc.prop_kind));
    // Memory descriptors
    seed = hash_combine(seed, get_md_hash(desc.src_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_src_desc));
    seed = hash_combine(seed, get_md_hash(desc.weights_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_weights_desc));
    seed = hash_combine(seed, get_md_hash(desc.bias_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_bias_desc));
    seed = hash_combine(seed, get_md_hash(desc.dst_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_dst_desc));
    // Accumulator type
    seed = hash_combine(seed, static_cast<size_t>(desc.accum_data_type));
    // Combined hash for inner_product desc
    return seed;
}

// Layer normalization
inline size_t get_desc_hash(const layer_normalization_desc_t &desc) {
    size_t seed = 0;
    // Kinds
    seed = hash_combine(seed, static_cast<size_t>(desc.primitive_kind));
    seed = hash_combine(seed, static_cast<size_t>(desc.prop_kind));
    // Memory descriptors
    seed = hash_combine(seed, get_md_hash(desc.data_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_data_desc));
    seed = hash_combine(seed, get_md_hash(desc.data_scaleshift_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_data_scaleshift_desc));
    seed = hash_combine(seed, get_md_hash(desc.stat_desc));
    // Epsilon
    seed = hash_combine(seed, desc.layer_norm_epsilon);
    // Flags
    seed = hash_combine(seed, desc.flags);
    // Combined hash for layer_normalization desc
    return seed;
}

inline size_t get_desc_hash(const lrn_desc_t &desc) {
    size_t seed = 0;
    // Kinds
    seed = hash_combine(seed, static_cast<size_t>(desc.primitive_kind));
    seed = hash_combine(seed, static_cast<size_t>(desc.prop_kind));
    seed = hash_combine(seed, static_cast<size_t>(desc.alg_kind));
    // Memory descriptors
    seed = hash_combine(seed, get_md_hash(desc.data_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_data_desc));
    // Local size
    seed = hash_combine(seed, desc.local_size);
    // Alpha, beta
    seed = hash_combine(seed, desc.lrn_alpha);
    seed = hash_combine(seed, desc.lrn_beta);
    // k
    seed = hash_combine(seed, desc.lrn_k);
    // Combined hash for lrn desc
    return seed;
}

inline size_t get_desc_hash(const matmul_desc_t &desc) {
    size_t seed = 0;
    // Kinds
    seed = hash_combine(seed, static_cast<size_t>(desc.primitive_kind));
    // Memory descriptors
    seed = hash_combine(seed, get_md_hash(desc.src_desc));
    seed = hash_combine(seed, get_md_hash(desc.weights_desc));
    seed = hash_combine(seed, get_md_hash(desc.bias_desc));
    seed = hash_combine(seed, get_md_hash(desc.dst_desc));
    // Accumulator type
    seed = hash_combine(seed, static_cast<size_t>(desc.accum_data_type));
    // Combined hash for matmul op desc
    return seed;
}

inline size_t get_desc_hash(const pooling_desc_t &desc) {
    size_t seed = 0;
    // Kinds
    seed = hash_combine(seed, static_cast<size_t>(desc.primitive_kind));
    seed = hash_combine(seed, static_cast<size_t>(desc.prop_kind));
    seed = hash_combine(seed, static_cast<size_t>(desc.alg_kind));
    // Memory descriptors
    seed = hash_combine(seed, get_md_hash(desc.src_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_src_desc));
    seed = hash_combine(seed, get_md_hash(desc.dst_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_dst_desc));
    // Strides, dilates, padding
    seed = get_array_hash(seed, desc.strides, DNNL_MAX_NDIMS);
    seed = get_array_hash(seed, desc.kernel, DNNL_MAX_NDIMS);
    seed = get_array_hash(seed, desc.padding[0], DNNL_MAX_NDIMS);
    seed = get_array_hash(seed, desc.padding[1], DNNL_MAX_NDIMS);
    // Accumulator type
    seed = hash_combine(seed, static_cast<size_t>(desc.accum_data_type));
    // Combined hash for pooling desc
    return seed;
}

inline size_t get_desc_hash(const reorder_desc_t &desc) {
    size_t seed = 0;
    // Kinds
    seed = hash_combine(seed, static_cast<size_t>(desc.primitive_kind));
    // Memory descriptors
    seed = hash_combine(seed, get_md_hash(desc.src_md));
    seed = hash_combine(seed, get_md_hash(desc.dst_md));
    // Kinds of source and destination engines
    seed = hash_combine(seed, static_cast<size_t>(desc.src_engine_kind));
    seed = hash_combine(seed, static_cast<size_t>(desc.dst_engine_kind));
    // Combined hash for reorder desc
    return seed;
}

inline size_t get_desc_hash(const resampling_desc_t &desc) {
    size_t seed = 0;
    // Kinds
    seed = hash_combine(seed, static_cast<size_t>(desc.primitive_kind));
    seed = hash_combine(seed, static_cast<size_t>(desc.alg_kind));
    // Memory descriptors
    seed = hash_combine(seed, get_md_hash(desc.src_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_src_desc));
    seed = hash_combine(seed, get_md_hash(desc.dst_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_dst_desc));
    // Factors
    seed = get_array_hash(seed, desc.factors, DNNL_MAX_NDIMS);
    // Combined hash for resampling op desc
    return seed;
}

inline size_t get_desc_hash(const rnn_desc_t &desc) {
    size_t seed = 0;
    // Kinds
    seed = hash_combine(seed, static_cast<size_t>(desc.primitive_kind));
    seed = hash_combine(seed, static_cast<size_t>(desc.prop_kind));
    seed = hash_combine(seed, static_cast<size_t>(desc.cell_kind));
    seed = hash_combine(seed, static_cast<size_t>(desc.direction));
    // Memory descriptors
    seed = hash_combine(seed, get_md_hash(desc.src_layer_desc));
    seed = hash_combine(seed, get_md_hash(desc.src_iter_desc));
    seed = hash_combine(seed, get_md_hash(desc.src_iter_c_desc));
    seed = hash_combine(seed, get_md_hash(desc.weights_layer_desc));
    seed = hash_combine(seed, get_md_hash(desc.weights_iter_desc));
    seed = hash_combine(seed, get_md_hash(desc.bias_desc));
    seed = hash_combine(seed, get_md_hash(desc.dst_layer_desc));
    seed = hash_combine(seed, get_md_hash(desc.dst_iter_desc));
    seed = hash_combine(seed, get_md_hash(desc.dst_iter_c_desc));
    seed = hash_combine(seed, get_md_hash(desc.weights_peephole_desc));
    seed = hash_combine(seed, get_md_hash(desc.weights_projection_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_src_layer_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_src_iter_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_src_iter_c_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_weights_layer_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_weights_iter_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_bias_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_dst_layer_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_dst_iter_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_dst_iter_c_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_weights_peephole_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_weights_projection_desc));
    // Flags
    seed = hash_combine(seed, desc.flags);
    // Activation kind
    seed = hash_combine(seed, static_cast<size_t>(desc.activation_kind));
    // Alpha, beta
    seed = hash_combine(seed, desc.alpha);
    seed = hash_combine(seed, desc.beta);
    // Combined hash for rnn desc
    return seed;
}

// Shuffle
inline size_t get_desc_hash(const shuffle_desc_t &desc) {
    size_t seed = 0;
    // Kinds
    seed = hash_combine(seed, static_cast<size_t>(desc.primitive_kind));
    seed = hash_combine(seed, static_cast<size_t>(desc.prop_kind));
    // Memory descriptors
    seed = hash_combine(seed, get_md_hash(desc.data_desc));
    // Axis
    seed = hash_combine(seed, desc.axis);
    // Groupe size
    seed = hash_combine(seed, desc.group_size);
    // Combined hash for shuffle desc
    return seed;
}

inline size_t get_desc_hash(const softmax_desc_t &desc) {
    size_t seed = 0;
    // Kinds
    seed = hash_combine(seed, static_cast<size_t>(desc.primitive_kind));
    seed = hash_combine(seed, static_cast<size_t>(desc.prop_kind));
    // Memory descriptors
    seed = hash_combine(seed, get_md_hash(desc.data_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_desc));
    // Axis
    seed = hash_combine(seed, desc.softmax_axis);
    // Combined hash for softmax desc
    return seed;
}

inline size_t get_desc_hash(const sum_desc_t &desc) {
    size_t seed = 0;
    // Kinds
    seed = hash_combine(seed, static_cast<size_t>(desc.primitive_kind));
    // Memory descriptors
    seed = hash_combine(seed, get_md_hash(desc.dst_md));
    // N
    seed = hash_combine(seed, desc.n);
    // Scales
    if (!desc.scales.empty()) {
        seed = get_array_hash(seed, desc.scales.data(), desc.n);
    }
    // Array of mds
    seed = get_array_hash(seed, desc.src_mds.data(), desc.n);
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
        seed = hash_combine(
                seed, hash_combine(0, static_cast<size_t>(key.kind_)));
        seed = hash_combine(
                seed, hash_combine(0, static_cast<size_t>(key.runtime_kind_)));
        seed = hash_combine(
                seed, hash_combine(0, static_cast<size_t>(key.device_id_)));
        // Combine hash for op_desc with the computed hash
#define CASE(pkind) \
    case primitive_kind::pkind: \
        seed = hash_combine( \
                seed, get_desc_hash((pkind##_desc_t)key.op_desc_)); \
        break;

        switch (key.primitive_kind_) {
            CASE(batch_normalization)
            CASE(binary)
            CASE(concat)
            CASE(convolution)
            CASE(deconvolution)
            CASE(eltwise)
            CASE(gemm)
            CASE(inner_product)
            CASE(layer_normalization)
            CASE(lrn)
            CASE(matmul)
            CASE(pooling)
            CASE(reorder)
            CASE(resampling)
            CASE(rnn)
            CASE(shuffle)
            CASE(softmax)
            CASE(sum)
            default: assert(!"unknown primitive_kind");
        }
#undef CASE
        seed = get_array_hash(seed, key.mds.data(), (int)key.mds.size());

        return seed;
    }
};

} // namespace std

#endif
