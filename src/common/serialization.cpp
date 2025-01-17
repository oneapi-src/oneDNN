/*******************************************************************************
* Copyright 2021-2025 Intel Corporation
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

#include "common/serialization.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

namespace dnnl {
namespace impl {
namespace serialization {

status_t serialize_desc(
        serialization_stream_t &sstream, const op_desc_t *op_desc) {
#define CASE(pkind) \
    case primitive_kind::pkind: \
        serialize_desc(sstream, *(const pkind##_desc_t *)op_desc); \
        break;

    switch ((int)op_desc->primitive_kind) {
        CASE(batch_normalization)
        CASE(binary)
        CASE(concat)
        CASE(convolution)
        CASE(deconvolution)
        CASE(eltwise)
        CASE(gemm)
        CASE(group_normalization)
        CASE(inner_product)
        CASE(layer_normalization)
        CASE(lrn)
        CASE(matmul)
        CASE(pooling)
        CASE(prelu)
        CASE(reduction)
        CASE(reorder)
        CASE(resampling)
        CASE(rnn)
        CASE(sdpa)
        CASE(shuffle)
        CASE(softmax)
        CASE(sum)
        default: return status::invalid_arguments;
    }
#undef CASE
    return status::success;
}

void serialize_md(serialization_stream_t &sstream, const memory_desc_t &md) {
    sstream.write(&md.ndims);
    sstream.write(md.dims, md.ndims);
    sstream.write(&md.data_type);
    sstream.write(md.padded_dims, md.ndims);
    sstream.write(md.padded_offsets, md.ndims);
    sstream.write(&md.offset0);
    sstream.write(&md.format_kind);
    // format desc
    switch ((int)md.format_kind) {
#ifdef DNNL_EXPERIMENTAL_SPARSE
        case format_kind::sparse:
#endif
        case format_kind::undef:
        case format_kind::any: break;
        case format_kind::blocked:
            sstream.write(md.format_desc.blocking.strides, md.ndims);
            sstream.write(&md.format_desc.blocking.inner_nblks);
            sstream.write(md.format_desc.blocking.inner_blks,
                    md.format_desc.blocking.inner_nblks);
            sstream.write(md.format_desc.blocking.inner_idxs,
                    md.format_desc.blocking.inner_nblks);
            break;
        case format_kind::wino:
            sstream.write(&md.format_desc.wino_desc.wino_format);
            sstream.write(&md.format_desc.wino_desc.r);
            sstream.write(&md.format_desc.wino_desc.alpha);
            sstream.write(&md.format_desc.wino_desc.ic);
            sstream.write(&md.format_desc.wino_desc.oc);
            sstream.write(&md.format_desc.wino_desc.ic_block);
            sstream.write(&md.format_desc.wino_desc.oc_block);
            sstream.write(&md.format_desc.wino_desc.ic2_block);
            sstream.write(&md.format_desc.wino_desc.oc2_block);
            sstream.write(&md.format_desc.wino_desc.adj_scale);
            sstream.write(&md.format_desc.wino_desc.size);
            break;
        case format_kind::cublaslt_blocked:
            sstream.write(
                    &md.format_desc.cublaslt_blocked_desc.cublaslt_format);
            sstream.write(&md.format_desc.cublaslt_blocked_desc.size);
            break;
        case format_kind::rnn_packed:
            sstream.write(&md.format_desc.rnn_packed_desc.format);
            sstream.write(&md.format_desc.rnn_packed_desc.n_parts);
            sstream.write(&md.format_desc.rnn_packed_desc.n);
            sstream.write(&md.format_desc.rnn_packed_desc.ldb);
            {
                int n_parts = md.format_desc.rnn_packed_desc.n_parts;
                sstream.write(md.format_desc.rnn_packed_desc.parts, n_parts);
                sstream.write(
                        md.format_desc.rnn_packed_desc.part_pack_size, n_parts);
                sstream.write(
                        md.format_desc.rnn_packed_desc.pack_part, n_parts);
            }
            sstream.write(&md.format_desc.rnn_packed_desc.offset_compensation);
            sstream.write(&md.format_desc.rnn_packed_desc.size);
            break;
        default: assert(!"unknown format_kind");
    }

    if (md.extra.flags != dnnl_memory_extra_flag_none) {
        sstream.write(&md.extra.flags);
        if (md.extra.flags
                & (dnnl_memory_extra_flag_compensation_conv_s8s8
                        | dnnl_memory_extra_flag_rnn_u8s8_compensation)) {
            sstream.write(&md.extra.compensation_mask);
        }
        if (md.extra.flags & dnnl_memory_extra_flag_scale_adjust) {
            sstream.write(&md.extra.scale_adjust);
        }
        if (md.extra.flags
                & dnnl_memory_extra_flag_compensation_conv_asymmetric_src) {
            sstream.write(&md.extra.asymm_compensation_mask);
        }
        if (md.extra.flags
                & dnnl_memory_extra_flag_compensation_gpu_conv_asymmetric_src) {
            sstream.write(md.extra.idhw, 3);
            sstream.write(md.extra.odhw, 3);
            sstream.write(md.extra.pdhw, 3);
            sstream.write(md.extra.ddhw, 3);
            sstream.write(&md.extra.dst_size);
        }
    }
}

void serialize_post_ops(
        serialization_stream_t &sstream, const post_ops_t &post_ops) {
    // post_ops: entry[:]
    for (int i = 0; i < post_ops.len(); i++) {
        const auto &entry = post_ops.entry_[i];
        switch (entry.kind) {
            case primitive_kind::eltwise:
                sstream.write(&entry.eltwise.alg);
                sstream.write(&entry.eltwise.scale);
                sstream.write(&entry.eltwise.alpha);
                sstream.write(&entry.eltwise.beta);
                break;
            case primitive_kind::sum:
                sstream.write(&entry.sum.scale);
                sstream.write(&entry.sum.zero_point);
                sstream.write(&entry.sum.dt);
                break;
            case primitive_kind::convolution:
                sstream.write(&entry.depthwise_conv.kernel);
                sstream.write(&entry.depthwise_conv.stride);
                sstream.write(&entry.depthwise_conv.padding);
                sstream.write(&entry.depthwise_conv.wei_dt);
                sstream.write(&entry.depthwise_conv.bias_dt);
                sstream.write(&entry.depthwise_conv.dst_dt);
                break;
            case primitive_kind::binary:
                sstream.write(&entry.binary.alg);
                serialize_md(sstream, entry.binary.user_src1_desc);
                break;
            case primitive_kind::prelu: sstream.write(&entry.prelu.mask); break;
            default: assert(!"unknown post_op");
        }
    }
}

void serialize_attr(
        serialization_stream_t &sstream, const primitive_attr_t &attr) {
    // scratchpad_mode
    sstream.write(&attr.scratchpad_mode_);
    // fpmath_mode
    sstream.write(&attr.fpmath_.mode_);
    sstream.write(&attr.fpmath_.apply_to_int_);
    // deterministic
    sstream.write(&attr.deterministic_);
    // acc_mode
    sstream.write(&attr.acc_mode_);

    if (!attr.scales_.has_default_values()) {
        sstream.write("scale:");
        attr.scales_.serialize(sstream);
    }
    // zero_points
    if (!attr.zero_points_.has_default_values()) {
        sstream.write("zp:");
        attr.zero_points_.serialize(sstream);
    }

    // Rounding modes
    if (!attr.rounding_mode_.has_default_values()) sstream.write("rm:");
    for (const auto &e : attr.rounding_mode_.rounding_modes_map_) {
        if (!attr.rounding_mode_.has_default_values(e.first)) {
            sstream.write(&e.first);
            sstream.write(&e.second);
        }
    }

    if (!attr.dropout_.has_default_values()) {
        sstream.write("dropout:");
        serialize_md(sstream, attr.dropout_.user_dropout_desc_);
    }

    serialize_post_ops(sstream, attr.post_ops_);

    // rnn_data_qparams: scale, shift
    sstream.write(&attr.rnn_data_qparams_.scale_);
    sstream.write(&attr.rnn_data_qparams_.shift_);
    if (!attr.rnn_weights_qparams_.has_default_values()) {
        // rnn_weights_qparams: mask
        sstream.write(&attr.rnn_weights_qparams_.mask_);
        // rnn_weights_qparams: count
        sstream.write(&attr.rnn_weights_qparams_.count_);
        // rnn_weights_qparams: scales[:]
        sstream.write(attr.rnn_weights_qparams_.scales_,
                attr.rnn_weights_qparams_.count_);
    }
    if (attr.gpu_attr_) {
        attr.gpu_attr_->serialize(sstream);
    } else {
        int zero = 0;
        sstream.write(&zero);
    }
}

void serialize_desc(
        serialization_stream_t &sstream, const concat_desc_t &desc) {
    // Kinds
    sstream.write(&desc.primitive_kind);
    // Memory descriptors
    serialize_md(sstream, *desc.dst_md);
    // N
    sstream.write(&desc.n);
    // Concat dimension
    sstream.write(&desc.concat_dimension);
    // Array of mds
    for (int i = 0; i < desc.n; i++)
        serialize_md(sstream, *desc.src_mds[i]);
}

void serialize_desc(serialization_stream_t &sstream,
        const batch_normalization_desc_t &desc) {
    // Kinds
    sstream.write(&desc.primitive_kind);
    sstream.write(&desc.prop_kind);
    // Memory descriptors
    serialize_md(sstream, desc.src_desc);
    serialize_md(sstream, desc.dst_desc);
    serialize_md(sstream, desc.diff_src_desc);
    serialize_md(sstream, desc.diff_dst_desc);
    serialize_md(sstream, desc.scaleshift_desc);
    serialize_md(sstream, desc.diff_scaleshift_desc);
    serialize_md(sstream, desc.stat_desc);
    // Epsilon
    sstream.write(&desc.batch_norm_epsilon);
    // Flags
    sstream.write(&desc.flags);
}

void serialize_desc(
        serialization_stream_t &sstream, const binary_desc_t &desc) {
    // Kinds
    sstream.write(&desc.primitive_kind);
    sstream.write(&desc.alg_kind);
    // Memory descriptors
    serialize_md(sstream, desc.src_desc[0]);
    serialize_md(sstream, desc.src_desc[1]);
    serialize_md(sstream, desc.src_desc[2]);
    serialize_md(sstream, desc.dst_desc);
}

// (De-)Convolution
void serialize_desc(
        serialization_stream_t &sstream, const convolution_desc_t &desc) {
    // Kinds
    sstream.write(&desc.primitive_kind);
    sstream.write(&desc.prop_kind);
    sstream.write(&desc.alg_kind);
    // Memory descriptors
    serialize_md(sstream, desc.src_desc);
    serialize_md(sstream, desc.diff_src_desc);
    serialize_md(sstream, desc.weights_desc);
    serialize_md(sstream, desc.diff_weights_desc);
    serialize_md(sstream, desc.bias_desc);
    serialize_md(sstream, desc.diff_bias_desc);
    serialize_md(sstream, desc.dst_desc);
    serialize_md(sstream, desc.diff_dst_desc);
    // Strides, dilates, padding
    sstream.write(desc.strides, DNNL_MAX_NDIMS);
    sstream.write(desc.dilates, DNNL_MAX_NDIMS);
    sstream.write(desc.padding[0], DNNL_MAX_NDIMS);
    sstream.write(desc.padding[1], DNNL_MAX_NDIMS);
    // Accumulator type
    sstream.write(&desc.accum_data_type);
    // Internal member
    sstream.write(&desc.use_inversion);
}

// Eltwise
void serialize_desc(
        serialization_stream_t &sstream, const eltwise_desc_t &desc) {
    // Kinds
    sstream.write(&desc.primitive_kind);
    sstream.write(&desc.prop_kind);
    sstream.write(&desc.alg_kind);
    // Memory descriptors
    serialize_md(sstream, desc.src_desc);
    serialize_md(sstream, desc.dst_desc);
    serialize_md(sstream, desc.diff_src_desc);
    serialize_md(sstream, desc.diff_dst_desc);
    // Alpha, beta
    sstream.write(&desc.alpha);
    sstream.write(&desc.beta);
}

void serialize_desc(serialization_stream_t &sstream, const gemm_desc_t &desc) {
    // Kind
    sstream.write(&desc.primitive_kind);
    serialize_md(sstream, desc.a_desc);
    serialize_md(sstream, desc.b_desc);
    serialize_md(sstream, desc.c_desc);
    serialize_md(sstream, desc.bias_desc);
    // Accumulator type
    sstream.write(&desc.acc_type);
    sstream.write(&desc.sum_ab);
    sstream.write(&desc.sum_ab_type);
}

void serialize_desc(serialization_stream_t &sstream,
        const group_normalization_desc_t &desc) {
    // Kinds
    sstream.write(&desc.primitive_kind);
    sstream.write(&desc.prop_kind);
    // Memory descriptors
    serialize_md(sstream, desc.src_desc);
    serialize_md(sstream, desc.dst_desc);
    serialize_md(sstream, desc.diff_src_desc);
    serialize_md(sstream, desc.diff_dst_desc);
    serialize_md(sstream, desc.scaleshift_desc);
    serialize_md(sstream, desc.diff_scaleshift_desc);
    serialize_md(sstream, desc.stat_desc);
    // Groups
    sstream.write(&desc.groups);
    // Epsilon
    sstream.write(&desc.group_norm_epsilon);
    // Flags
    sstream.write(&desc.flags);
}

void serialize_desc(
        serialization_stream_t &sstream, const inner_product_desc_t &desc) {
    // Kinds
    sstream.write(&desc.primitive_kind);
    sstream.write(&desc.prop_kind);
    // Memory descriptors
    serialize_md(sstream, desc.src_desc);
    serialize_md(sstream, desc.diff_src_desc);
    serialize_md(sstream, desc.weights_desc);
    serialize_md(sstream, desc.diff_weights_desc);
    serialize_md(sstream, desc.bias_desc);
    serialize_md(sstream, desc.diff_bias_desc);
    serialize_md(sstream, desc.dst_desc);
    serialize_md(sstream, desc.diff_dst_desc);
    // Accumulator type
    sstream.write(&desc.accum_data_type);
}

void serialize_desc(serialization_stream_t &sstream,
        const layer_normalization_desc_t &desc) {
    // Kinds
    sstream.write(&desc.primitive_kind);
    sstream.write(&desc.prop_kind);
    // Memory descriptors
    serialize_md(sstream, desc.src_desc);
    serialize_md(sstream, desc.diff_src_desc);
    serialize_md(sstream, desc.data_scaleshift_desc);
    serialize_md(sstream, desc.diff_data_scaleshift_desc);
    serialize_md(sstream, desc.dst_desc);
    serialize_md(sstream, desc.diff_dst_desc);
    serialize_md(sstream, desc.stat_desc);
    // Epsilon
    sstream.write(&desc.layer_norm_epsilon);
    // Flags
    sstream.write(&desc.flags);
}

void serialize_desc(serialization_stream_t &sstream, const lrn_desc_t &desc) {
    // Kinds
    sstream.write(&desc.primitive_kind);
    sstream.write(&desc.prop_kind);
    sstream.write(&desc.alg_kind);
    // Memory descriptors
    serialize_md(sstream, desc.src_desc);
    serialize_md(sstream, desc.dst_desc);
    serialize_md(sstream, desc.diff_src_desc);
    serialize_md(sstream, desc.diff_dst_desc);
    // Local size
    sstream.write(&desc.local_size);
    // Alpha, beta
    sstream.write(&desc.lrn_alpha);
    sstream.write(&desc.lrn_beta);
    // k
    sstream.write(&desc.lrn_k);
}

void serialize_desc(
        serialization_stream_t &sstream, const matmul_desc_t &desc) {
    // Kinds
    sstream.write(&desc.primitive_kind);
    // Memory descriptors
    serialize_md(sstream, desc.src_desc);
    serialize_md(sstream, desc.weights_desc);
    serialize_md(sstream, desc.bias_desc);
    serialize_md(sstream, desc.dst_desc);
    // Accumulator type
    sstream.write(&desc.accum_data_type);
}

void serialize_desc(
        serialization_stream_t &sstream, const pooling_desc_t &desc) {
    // Kinds
    sstream.write(&desc.primitive_kind);
    sstream.write(&desc.prop_kind);
    sstream.write(&desc.alg_kind);
    // Memory descriptors
    serialize_md(sstream, desc.src_desc);
    serialize_md(sstream, desc.diff_src_desc);
    serialize_md(sstream, desc.dst_desc);
    serialize_md(sstream, desc.diff_dst_desc);
    // Strides, dilates, padding
    sstream.write(desc.strides, DNNL_MAX_NDIMS);
    sstream.write(desc.kernel, DNNL_MAX_NDIMS);
    sstream.write(desc.padding[0], DNNL_MAX_NDIMS);
    sstream.write(desc.padding[1], DNNL_MAX_NDIMS);
    sstream.write(desc.dilation, DNNL_MAX_NDIMS);
    // Accumulator type
    sstream.write(&desc.accum_data_type);
}

void serialize_desc(serialization_stream_t &sstream, const prelu_desc_t &desc) {
    // Kinds
    sstream.write(&desc.primitive_kind);
    sstream.write(&desc.prop_kind);
    // Memory descriptors
    serialize_md(sstream, desc.src_desc);
    serialize_md(sstream, desc.weights_desc);
    serialize_md(sstream, desc.dst_desc);
    serialize_md(sstream, desc.diff_src_desc);
    serialize_md(sstream, desc.diff_weights_desc);
    serialize_md(sstream, desc.diff_dst_desc);
}

void serialize_desc(
        serialization_stream_t &sstream, const reduction_desc_t &desc) {
    // Kinds
    sstream.write(&desc.primitive_kind);
    sstream.write(&desc.alg_kind);
    // Memory descriptors
    serialize_md(sstream, desc.src_desc);
    serialize_md(sstream, desc.dst_desc);
    // P, eps
    sstream.write(&desc.p);
    sstream.write(&desc.eps);
}

void serialize_desc(
        serialization_stream_t &sstream, const reorder_desc_t &desc) {
    // Kinds
    sstream.write(&desc.primitive_kind);
    // Memory descriptors
    serialize_md(sstream, *desc.src_md);
    serialize_md(sstream, *desc.dst_md);
    // Kinds of source and destination engines
    sstream.write(&desc.src_engine_kind);
    sstream.write(&desc.dst_engine_kind);
    sstream.write(&desc.is_cross_engine);
}

void serialize_desc(
        serialization_stream_t &sstream, const resampling_desc_t &desc) {
    // Kinds
    sstream.write(&desc.primitive_kind);
    sstream.write(&desc.alg_kind);
    // Memory descriptors
    serialize_md(sstream, desc.src_desc);
    serialize_md(sstream, desc.diff_src_desc);
    serialize_md(sstream, desc.dst_desc);
    serialize_md(sstream, desc.diff_dst_desc);
    // Factors
    sstream.write(desc.factors, DNNL_MAX_NDIMS);
}

void serialize_desc(serialization_stream_t &sstream, const rnn_desc_t &desc) {
    // Kinds
    sstream.write(&desc.primitive_kind);
    sstream.write(&desc.prop_kind);
    sstream.write(&desc.cell_kind);
    sstream.write(&desc.direction);
    // Memory descriptors
    serialize_md(sstream, desc.src_layer_desc);
    serialize_md(sstream, desc.src_iter_desc);
    serialize_md(sstream, desc.src_iter_c_desc);
    serialize_md(sstream, desc.weights_layer_desc);
    serialize_md(sstream, desc.weights_iter_desc);
    serialize_md(sstream, desc.bias_desc);
    serialize_md(sstream, desc.dst_layer_desc);
    serialize_md(sstream, desc.dst_iter_desc);
    serialize_md(sstream, desc.dst_iter_c_desc);
    serialize_md(sstream, desc.weights_peephole_desc);
    serialize_md(sstream, desc.weights_projection_desc);
    serialize_md(sstream, desc.diff_src_layer_desc);
    serialize_md(sstream, desc.diff_src_iter_desc);
    serialize_md(sstream, desc.diff_src_iter_c_desc);
    serialize_md(sstream, desc.diff_weights_layer_desc);
    serialize_md(sstream, desc.diff_weights_iter_desc);
    serialize_md(sstream, desc.diff_bias_desc);
    serialize_md(sstream, desc.diff_dst_layer_desc);
    serialize_md(sstream, desc.diff_dst_iter_desc);
    serialize_md(sstream, desc.diff_dst_iter_c_desc);
    serialize_md(sstream, desc.diff_weights_peephole_desc);
    serialize_md(sstream, desc.diff_weights_projection_desc);
    // Flags
    sstream.write(&desc.flags);
    // Activation kind
    sstream.write(&desc.activation_kind);
    // Alpha, beta
    sstream.write(&desc.alpha);
    sstream.write(&desc.beta);
}

// Shuffle
void serialize_desc(
        serialization_stream_t &sstream, const shuffle_desc_t &desc) {
    // Kinds
    sstream.write(&desc.primitive_kind);
    sstream.write(&desc.prop_kind);
    // Memory descriptors
    serialize_md(sstream, desc.src_desc);
    serialize_md(sstream, desc.dst_desc);
    // Axis
    sstream.write(&desc.axis);
    // Groupe size
    sstream.write(&desc.group_size);
}

void serialize_desc(
        serialization_stream_t &sstream, const softmax_desc_t &desc) {
    // Kinds
    sstream.write(&desc.primitive_kind);
    sstream.write(&desc.prop_kind);
    sstream.write(&desc.alg_kind);
    // Memory descriptors
    serialize_md(sstream, desc.src_desc);
    serialize_md(sstream, desc.diff_src_desc);
    serialize_md(sstream, desc.dst_desc);
    serialize_md(sstream, desc.diff_dst_desc);
    // Axis
    sstream.write(&desc.softmax_axis);
}

void serialize_desc(serialization_stream_t &sstream, const sum_desc_t &desc) {
    // Kinds
    sstream.write(&desc.primitive_kind);
    // Memory descriptors
    serialize_md(sstream, *desc.dst_md);
    // N
    sstream.write(&desc.n);
    // Scales
    sstream.write(desc.scales, desc.n);
    // Array of mds
    for (int i = 0; i < desc.n; i++)
        serialize_md(sstream, *desc.src_mds[i]);
}

void serialize_desc(serialization_stream_t &sstream, const sdpa_desc_t &desc) {
    // Kind
    sstream.write(&desc.primitive_kind);
    serialize_md(sstream, desc.q_desc);
    serialize_md(sstream, desc.k_desc);
    serialize_md(sstream, desc.v_desc);
    desc.kq_scales.serialize(sstream);
    desc.kq_zero_points.serialize(sstream);
    desc.vs_scales.serialize(sstream);
    desc.vs_zero_points.serialize(sstream);
    serialize_md(sstream, desc.dst_desc);
    serialize_md(sstream, desc.attn_mask_desc);
    sstream.write(&desc.scale_dt);
    sstream.write(&desc.invert_scale);
    sstream.write(&desc.kv_head_number);
    sstream.write(&desc.causal_mask);
}

} // namespace serialization
} // namespace impl
} // namespace dnnl
