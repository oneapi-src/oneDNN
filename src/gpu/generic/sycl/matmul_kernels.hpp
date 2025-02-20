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

#ifndef GPU_GENERIC_SYCL_MATMUL_KERNELS_HPP
#define GPU_GENERIC_SYCL_MATMUL_KERNELS_HPP

#include "common/primitive_exec_types.hpp"
#include "gpu/generic/sycl/sycl_io_helper.hpp"
#include "gpu/generic/sycl/sycl_math_utils.hpp"
#include "gpu/generic/sycl/sycl_post_ops.hpp"
#include "gpu/generic/sycl/sycl_primitive_conf.hpp"
#include "gpu/generic/sycl/sycl_utils.hpp"
#include "xpu/sycl/memory_storage_base.hpp"
#include "xpu/sycl/types.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

struct matmul_kernel_fwd_t {
    static constexpr int max_supported_ndims = 6;

    static constexpr int vec_len = 2;

    // block sizes must be a multiple of vec_len
    static constexpr int register_block_M = 4;
    static constexpr int register_block_N = 2;
    static constexpr int register_block_K = 2;

    static int transpose_mask(int mask, int ndims) {
        return (mask & ~(3 << (ndims - 2))) | ((mask & (1 << (ndims - 2))) << 1)
                | ((mask >> 1) & (1 << (ndims - 2)));
    }

    static uint get_dropout_threshold(float p) {
        if (p >= 1.f) return 0xFFFFFFFFu;
        char exponent = 126 - ((reinterpret_cast<uint &>(p) >> 23) & 0x7F);
        if ((p <= 0.f) || (exponent > 31)) return 0u;
        uint mantissa = (reinterpret_cast<uint &>(p) << 8) | 0x80000000u;
        if (!exponent) return (ulong(mantissa) * 0xFFFFFFFFuL) >> 32;
        return ((ulong(mantissa >> exponent) * 0xFFFFFFFFuL) >> 32)
                + !!(mantissa & ((1u << exponent) - 1u));
    }

    template <int Rows, int Cols>
    struct register_block {
        using Vec = ::sycl::vec<float, vec_len>;
        using Transposed = register_block<Cols, Rows>;
        static constexpr int size = Rows * Cols;
        Vec data[Rows][Cols / vec_len];

        void transpose_from(register_block<Cols, Rows> input) {
            for (int row = 0; row < Rows; row++) {
                for (int col = 0; col < Cols; col++) {
                    data[row][col / vec_len][col % vec_len]
                            = input.data[col][row / vec_len][row % vec_len];
                }
            }
        }

        template <::sycl::access_mode mode>
        static Vec load_vec_helper(
                const memory_tensor_t<mode> &input, int offset) {
            data_type_t type = input.md().data_type();
            char *offset_ptr = static_cast<char *>(input.ptr())
                    + data_type_size(type) * offset;
            return load_float_vec<vec_len>(type, offset_ptr, 0);
        }

        static void store_vec_helper(
                inout_memory_tensor_t &output, Vec data, int offset) {
            data_type_t type = output.md().data_type();
            char *offset_ptr = static_cast<char *>(output.ptr())
                    + data_type_size(type) * offset;
            return store_float_vec<vec_len>(type, data, offset_ptr, 0);
        }

        // offset and row_stride are in scalars
        template <::sycl::access_mode mode>
        void load(const memory_tensor_t<mode> &input, int offset,
                int row_stride) {
            for (int row = 0; row < Rows; row++) {
                for (int col = 0; col < Cols / vec_len; col++) {
                    data[row][col] = load_vec_helper(
                            input, offset + row * row_stride + col * vec_len);
                }
            }
        }

        template <::sycl::access_mode mode>
        void load_masked(const memory_tensor_t<mode> &input, int offset,
                int row_stride, int mask) {
            switch ((mask >> (input.md().ndims() - 2)) & 3) {
                default:
                case 3: load(input, offset, row_stride); break;
                case 2: load(input, offset, 0); break;
                case 1: {
                    register_block<Cols, Rows> tmp;
                    tmp.load(input, offset, 0);
                    transpose_from(tmp);
                    break;
                }
                case 0: {
                    float val = load_float_value(
                            input.md().data_type(), input.ptr(), offset);
                    eltwise([=](float &el) { el = val; });
                    break;
                }
            }
        }

        template <::sycl::access_mode mode>
        void load_edge(const memory_tensor_t<mode> &input, int offset,
                int row_stride, int rows, int cols) {
            for (int row = 0; row < rows; row++) {
                int col;
                for (col = 0; col < cols / vec_len; col++) {
                    data[row][col] = load_vec_helper(
                            input, offset + row * row_stride + col * vec_len);
                }
                int n_remaining = cols - col * vec_len;
                for (int vec_el = 0; vec_el < n_remaining; vec_el++) {
                    data[row][col][vec_el] = load_float_value(
                            input.md().data_type(), input.ptr(),
                            offset + row * row_stride + col * vec_len + vec_el);
                }
            }
        }

        template <::sycl::access_mode mode>
        void load_edge_masked(const memory_tensor_t<mode> &input, int offset,
                int row_stride, int rows, int cols, int mask) {
            switch ((mask >> (input.md().ndims() - 2)) & 3) {
                case 3: load_edge(input, offset, row_stride, rows, cols); break;
                case 2: load_edge(input, offset, 0, rows, cols); break;
                case 1: {
                    register_block<Cols, Rows> tmp;
                    tmp.load_edge(input, offset, 0, cols, rows);
                    transpose_from(tmp);
                    break;
                }
                case 0: {
                    float val = load_float_value(
                            input.md().data_type(), input.ptr(), offset);
                    eltwise([=](float &el) { el = val; });
                    break;
                }
            }
        }

        template <::sycl::access_mode mode>
        void load_generic(const memory_tensor_t<mode> &input, int offset,
                int row_stride, bool transpose, bool is_edge_block, int rows,
                int cols, int mask = ~0) {
            if (is_edge_block) {
                if (transpose) {
                    Transposed tmp;
                    tmp.load_edge_masked(input, offset, row_stride, cols, rows,
                            transpose_mask(mask, input.md().ndims()));
                    transpose_from(tmp);
                } else {
                    load_edge_masked(
                            input, offset, row_stride, rows, cols, mask);
                }
            } else {
                if (transpose) {
                    Transposed tmp;
                    tmp.load_masked(input, offset, row_stride,
                            transpose_mask(mask, input.md().ndims()));
                    transpose_from(tmp);
                } else {
                    load_masked(input, offset, row_stride, mask);
                }
            }
        }

        void store(inout_memory_tensor_t &output, int offset, int row_stride) {
            for (int row = 0; row < Rows; row++) {
                for (int col = 0; col < Cols / vec_len; col++) {
                    store_vec_helper(output, data[row][col],
                            offset + row * row_stride + col * vec_len);
                }
            }
        }

        void store_edge(inout_memory_tensor_t &output, int offset,
                int row_stride, int rows, int cols) {
            for (int row = 0; row < rows; row++) {
                int col;
                for (col = 0; col < cols / vec_len; col++) {
                    store_vec_helper(output, data[row][col],
                            offset + row * row_stride + col * vec_len);
                }
                int n_remaining = cols - col * vec_len;
                for (int vec_el = 0; vec_el < n_remaining; vec_el++) {
                    store_float_value(output.md().data_type(),
                            data[row][col][vec_el], output.ptr(),
                            offset + row * row_stride + col * vec_len + vec_el);
                }
            }
        }

        void store_generic(inout_memory_tensor_t &output, int offset,
                int row_stride, bool transpose, bool is_edge_block, int rows,
                int cols) {
            if (is_edge_block) {
                if (transpose) {
                    Transposed dst_tmp;
                    dst_tmp.transpose_from(*this);
                    dst_tmp.store_edge(output, offset, row_stride, cols, rows);
                } else {
                    store_edge(output, offset, row_stride, rows, cols);
                }
            } else {
                if (transpose) {
                    Transposed dst_tmp;
                    dst_tmp.transpose_from(*this);
                    dst_tmp.store(output, offset, row_stride);
                } else {
                    store(output, offset, row_stride);
                }
            }
        }

        template <typename F>
        void eltwise(F funct) {
            for (int row = 0; row < Rows; row++) {
                for (int col = 0; col < Cols / vec_len; col++) {
                    for (int v_el = 0; v_el < vec_len; v_el++) {
                        funct(data[row][col][v_el]);
                    }
                }
            }
        }

        template <int K>
        void matmul_accumulate(
                register_block<Rows, K> lhs, register_block<K, Cols> rhs) {
            for (int row = 0; row < Rows; row++) {
                for (int k = 0; k < K / vec_len; k++) {
                    for (int k_el = 0; k_el < vec_len; k_el++) {
                        for (int col = 0; col < Cols / vec_len; col++) {
                            data[row][col] += Vec(lhs.data[row][k][k_el])
                                    * rhs.data[k * vec_len + k_el][col];
                        }
                    }
                }
            }
        }

        template <int K>
        void matmul_accumulate_edge_k(register_block<Rows, K> lhs,
                register_block<K, Cols> rhs, int k_max) {
            for (int row = 0; row < Rows; row++) {
                int k;
                for (k = 0; k < k_max / vec_len; k++) {
                    for (int k_el = 0; k_el < vec_len; k_el++) {
                        for (int col = 0; col < Cols / vec_len; col++) {
                            data[row][col] += Vec(lhs.data[row][k][k_el])
                                    * rhs.data[k * vec_len + k_el][col];
                        }
                    }
                }
                int last_vec_len = k_max - k * vec_len;
                for (int k_el = 0; k_el < last_vec_len; k_el++) {
                    for (int col = 0; col < Cols / vec_len; col++) {
                        data[row][col] += Vec(lhs.data[row][k][k_el])
                                * rhs.data[k * vec_len + k_el][col];
                    }
                }
            }
        }

        void dropout(xpu::sycl::out_memory_arg_t dropout_mask, uint threshold,
                uint seed, float inv_q, int offset, int row_stride) {
            for (int row = 0; row < Rows; row++) {
                for (int col = 0; col < Cols / vec_len; col++) {
                    for (int vec_el = 0; vec_el < vec_len; vec_el++) {
                        int dst_off = offset + row * row_stride + col * vec_len
                                + vec_el;
                        uint random
                                = ::dnnl::impl::math::philox4x32(dst_off, seed);
                        char dropout = random > threshold;
                        data[row][col][vec_el]
                                = dropout ? data[row][col][vec_el] * inv_q : 0;
                        static_cast<char *>(dropout_mask.get_pointer())[dst_off]
                                = dropout;
                    }
                }
            }
        }

        void apply_post_ops(sycl_post_ops_t post_ops,
                register_block<Rows, Cols> prev_dst, dims_t off_po, int dim1,
                const matmul_kernel_fwd_t *kernel) {
            for (int row = 0; row < Rows; row++) {
                for (int col = 0; col < Cols / vec_len; col++) {
                    for (int v_el = 0; v_el < vec_len; v_el++) {
                        off_po[dim1] += row;
                        off_po[dim1 + 1] += col * vec_len + v_el;
                        data[row][col][v_el]
                                = post_ops.apply(data[row][col][v_el],
                                        prev_dst.data[row][col][v_el],
                                        kernel->po_args_, off_po);
                        off_po[dim1] -= row;
                        off_po[dim1 + 1] -= col * vec_len + v_el;
                    }
                }
            }
        }

        void apply_post_ops_edge(sycl_post_ops_t post_ops,
                register_block<Rows, Cols> prev_dst, dims_t off_po, int dim1,
                const matmul_kernel_fwd_t *kernel, int rows, int cols) {
            for (int row = 0; row < rows; row++) {
                int col;
                for (col = 0; col < cols / vec_len; col++) {
                    for (int v_el = 0; v_el < vec_len; v_el++) {
                        off_po[dim1] += row;
                        off_po[dim1 + 1] += col * vec_len + v_el;
                        data[row][col][v_el]
                                = post_ops.apply(data[row][col][v_el],
                                        prev_dst.data[row][col][v_el],
                                        kernel->po_args_, off_po);
                        off_po[dim1] -= row;
                        off_po[dim1 + 1] -= col * vec_len + v_el;
                    }
                }
                int n_remaining = cols - col * vec_len;
                for (int v_el = 0; v_el < n_remaining; v_el++) {
                    off_po[dim1] += row;
                    off_po[dim1 + 1] += col * vec_len + v_el;
                    data[row][col][v_el] = post_ops.apply(data[row][col][v_el],
                            prev_dst.data[row][col][v_el], kernel->po_args_,
                            off_po);
                    off_po[dim1] -= row;
                    off_po[dim1 + 1] -= col * vec_len + v_el;
                }
            }
        }
    };

    matmul_kernel_fwd_t(const sycl_matmul_conf_t &conf, ::sycl::handler &cgh,
            const exec_ctx_t &ctx)
        : conf_(conf)
        , data_(CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_SRC_0))
        , weights_(CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_WEIGHTS))
        , bias_(CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_BIAS))
        , dst_(CTX_INOUT_SYCL_KERNEL_MEMORY(DNNL_ARG_DST))
        , data_scale_(CTX_IN_SYCL_KERNEL_MEMORY(
                  DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_0))
        , data_scales_dt_((conf_.do_scale_data)
                          ? ctx.memory_mdw(
                                       DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_0)
                                    .data_type()
                          : data_type_t::dnnl_f32)
        , weights_scale_(CTX_IN_SYCL_KERNEL_MEMORY(
                  DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS))
        , weights_scales_dt_((conf_.do_scale_weights)
                          ? ctx.memory_mdw(
                                       DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS)
                                    .data_type()
                          : data_type_t::dnnl_f32)
        , dst_scale_(CTX_IN_SYCL_KERNEL_MEMORY(
                  DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST))
        , dst_scales_dt_((conf_.do_scale_dst)
                          ? ctx.memory_mdw(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST)
                                    .data_type()
                          : data_type_t::dnnl_f32)
        , data_zeropoints_(CTX_IN_SYCL_KERNEL_MEMORY(
                  DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC_0))
        , data_zeropoints_dt_((conf_.use_data_zeropoints)
                          ? ctx.memory_mdw(DNNL_ARG_ATTR_ZERO_POINTS
                                       | DNNL_ARG_SRC_0)
                                    .data_type()
                          : data_type_t::dnnl_f32)
        , weights_zeropoints_(CTX_IN_SYCL_KERNEL_MEMORY(
                  DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS))
        , weights_zeropoints_dt_((conf_.use_weights_zeropoints)
                          ? ctx.memory_mdw(DNNL_ARG_ATTR_ZERO_POINTS
                                       | DNNL_ARG_WEIGHTS)
                                    .data_type()
                          : data_type_t::dnnl_f32)
        , dst_zeropoints_(CTX_IN_SYCL_KERNEL_MEMORY(
                  DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST))
        , dst_zeropoints_dt_((conf_.use_dst_zeropoints)
                          ? ctx.memory_mdw(
                                       DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST)
                                    .data_type()
                          : data_type_t::dnnl_f32)
        , dropout_mask_(CTX_OUT_SYCL_KERNEL_MEMORY(DNNL_ARG_ATTR_DROPOUT_MASK))
        , dropout_seed_(CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_ATTR_DROPOUT_SEED))
        , dropout_probability_(
                  CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_ATTR_DROPOUT_PROBABILITY))
        , po_args_(cgh, ctx, conf_.post_ops) {}

    void operator()(::sycl::nd_item<1> item) const {
        using data_block_t = register_block<register_block_M, register_block_K>;
        using weights_block_t
                = register_block<register_block_K, register_block_N>;
        using dst_block_t = register_block<register_block_M, register_block_N>;

        memory_tensor_t data_mem(data_, conf_.data_md);
        memory_tensor_t weights_mem(weights_, conf_.weights_md);
        memory_tensor_t bias_mem(bias_, conf_.bias_md);
        memory_tensor_t dst_mem(dst_, conf_.dst_md);
        memory_plain_t data_scale_mem(data_scale_, data_scales_dt_);
        memory_plain_t weights_scale_mem(weights_scale_, weights_scales_dt_);
        memory_plain_t dst_scale_mem(dst_scale_, dst_scales_dt_);
        memory_plain_t data_zeropoints_mem(
                data_zeropoints_, data_zeropoints_dt_);
        memory_plain_t weights_zeropoints_mem(
                weights_zeropoints_, weights_zeropoints_dt_);
        memory_plain_t dst_zeropoints_mem(dst_zeropoints_, dst_zeropoints_dt_);

        bool has_bias = bias_mem.md().ndims() != 0;

        float data_scale, weights_scale, dst_scale, data_zeropoint,
                weights_zeropoint, dst_zeropoint;
        if (conf_.do_scale_data) { data_scale = data_scale_mem.load(0); }
        if (conf_.do_scale_weights && conf_.single_weights_scale) {
            weights_scale = weights_scale_mem.load(0);
        }
        if (conf_.do_scale_dst) { dst_scale = dst_scale_mem.load(0); }

        if (conf_.use_data_zeropoints) {
            data_zeropoint = data_zeropoints_mem.load(0);
        }
        if (conf_.use_weights_zeropoints) {
            weights_zeropoint = weights_zeropoints_mem.load(0);
        }
        if (conf_.use_dst_zeropoints) {
            dst_zeropoint = dst_zeropoints_mem.load(0);
        }

        uint dropout_seed;
        float dropout_p;
        uint dropout_threshold;
        float dropout_inv_q;
        if (conf_.use_dropout) {
            dropout_seed = reinterpret_cast<const uint *>(
                    dropout_seed_.get_pointer())[0];
            dropout_p = reinterpret_cast<const float *>(
                    dropout_probability_.get_pointer())[0];
            dropout_threshold = get_dropout_threshold(dropout_p);
            dropout_inv_q = (dropout_p != 1.f) ? 1.f / (1.f - dropout_p) : 0.f;
        }

        // dimensions N/M/K depending on the tensor
        const int matmul_dim_1 = dst_mem.md().ndims() - 2;
        const int matmul_dim_2 = dst_mem.md().ndims() - 1;

        int M = dst_mem.md().dims()[matmul_dim_1];
        int N = dst_mem.md().dims()[matmul_dim_2];
        if (conf_.transpose_dst) { std::swap(M, N); }
        int K = data_mem.md().dims()[conf_.transpose_data ? matmul_dim_1
                                                          : matmul_dim_2];

        dims_t dst_dims, dst_blocks, dst_strides, off_dst_blocks, off_dst;
        for (int i = 0; i < max_supported_ndims; i++) {
            if (i < dst_mem.md().ndims()) {
                dst_dims[i] = dst_mem.md().dims()[i];
                dst_blocks[i] = dst_mem.md().dims()[i];
                dst_strides[i] = dst_mem.md().strides()[i];
            } else {
                dst_dims[i] = 1;
                dst_blocks[i] = 1;
                dst_strides[i] = INT_MAX;
            }
        }
        dst_blocks[matmul_dim_1] = math::div_up(dst_blocks[matmul_dim_1],
                conf_.transpose_dst ? register_block_N : register_block_M);
        dst_blocks[matmul_dim_2] = math::div_up(dst_blocks[matmul_dim_2],
                conf_.transpose_dst ? register_block_M : register_block_N);
        int n_blocks = 1;
        for (int i = 0; i < max_supported_ndims; i++) {
            n_blocks *= dst_blocks[i];
        }

        int dst_block_row_stride = dst_mem.md().strides()[matmul_dim_1];
        int bias_block_row_stride = bias_mem.md().strides()[matmul_dim_1];
        int data_block_row_stride = data_mem.md().strides()[matmul_dim_1];
        int weights_block_row_stride = weights_mem.md().strides()[matmul_dim_1];

        for (int block_idx = item.get_global_id(0); block_idx < n_blocks;
                block_idx += item.get_global_range(0)) {
            int idx_tmp = block_idx;
            for (int i = max_supported_ndims - 1; i >= 0; i--) {
                off_dst_blocks[i] = idx_tmp % dst_blocks[i];
                idx_tmp /= dst_blocks[i];
                off_dst[i] = off_dst_blocks[i];
            }
            bool is_dst_edge_block
                    = off_dst[matmul_dim_1] == dst_blocks[matmul_dim_1] - 1
                    || off_dst[matmul_dim_2] == dst_blocks[matmul_dim_2] - 1;
            off_dst[matmul_dim_1] *= conf_.transpose_dst ? register_block_N
                                                         : register_block_M;
            off_dst[matmul_dim_2] *= conf_.transpose_dst ? register_block_M
                                                         : register_block_N;
            int m = off_dst[conf_.transpose_dst ? matmul_dim_2 : matmul_dim_1];
            int n = off_dst[conf_.transpose_dst ? matmul_dim_1 : matmul_dim_2];

            dims_t off_src, off_weights, off_bias;
            for (int i = max_supported_ndims - 1; i >= 0; i--) {
                off_src[i] = off_dst[i];
                off_weights[i] = off_dst[i];
                off_bias[i] = off_dst[i];
            }

            int bias_mask = conf_.bias_mask;
            if (conf_.transpose_dst ^ conf_.transpose_bias) {
                std::swap(off_bias[matmul_dim_1], off_bias[matmul_dim_2]);
            }
            if (conf_.transpose_bias) {
                bias_mask = transpose_mask(bias_mask, data_mem.md().ndims());
            }

            int dst_block_start = dst_mem.md().off_v(off_dst);
            int bias_block_start
                    = bias_mem.md().off_v_masked(off_bias, bias_mask);

            int remaining_m = ::sycl::min(M - m, register_block_M);
            int remaining_n = ::sycl::min(N - n, register_block_N);

            dst_block_t dst_block;
            if (!has_bias) {
                dst_block.eltwise([=](float &el) { el = 0; });
            } else {
                // load bias
                dst_block.load_generic(bias_mem, bias_block_start,
                        bias_block_row_stride, conf_.transpose_bias,
                        is_dst_edge_block, remaining_m, remaining_n,
                        conf_.bias_mask);
            }

            for (int k = 0; k < K; k += register_block_K) {
                bool is_edge_k = k + register_block_K >= K;
                bool is_edge_block
                        = is_dst_edge_block || k + register_block_K >= K;
                off_src[matmul_dim_1] = conf_.transpose_data ? k : m;
                off_src[matmul_dim_2] = conf_.transpose_data ? m : k;
                off_weights[matmul_dim_1] = conf_.transpose_weights ? n : k;
                off_weights[matmul_dim_2] = conf_.transpose_weights ? k : n;

                int data_block_start
                        = data_mem.md().off_v_masked(off_src, conf_.data_mask);
                int weights_block_start = weights_mem.md().off_v_masked(
                        off_weights, conf_.weights_mask);

                data_block_t data_block;
                weights_block_t weights_block;

                int remaining_k = ::sycl::min(K - k, register_block_K);

                data_block.load_generic(data_mem, data_block_start,
                        data_block_row_stride, conf_.transpose_data,
                        is_edge_block, remaining_m, remaining_k,
                        conf_.data_mask);
                if (conf_.use_data_zeropoints) {
                    data_block.eltwise(
                            [=](float &el) { el -= data_zeropoint; });
                }
                if (conf_.do_scale_data) {
                    data_block.eltwise([=](float &el) { el *= data_scale; });
                }

                weights_block.load_generic(weights_mem, weights_block_start,
                        weights_block_row_stride, conf_.transpose_weights,
                        is_edge_block, remaining_k, remaining_n,
                        conf_.weights_mask);
                if (conf_.use_weights_zeropoints) {
                    weights_block.eltwise(
                            [=](float &el) { el -= weights_zeropoint; });
                }
                if (conf_.do_scale_weights) {
                    if (conf_.single_weights_scale) {
                        weights_block.eltwise(
                                [=](float &el) { el *= weights_scale; });
                    } else {
                        for (int n1 = 0; n1 < remaining_n; n1++) {
                            float scale_n = weights_scale_mem.load(n + n1);
                            for (int k1 = 0; k1 < remaining_k; k1++) {
                                weights_block
                                        .data[k1][n1 / vec_len][n1 % vec_len]
                                        *= scale_n;
                            }
                        }
                    }
                }

                if (is_edge_k) {
                    dst_block.matmul_accumulate_edge_k(
                            data_block, weights_block, remaining_k);
                } else {
                    dst_block.matmul_accumulate(data_block, weights_block);
                }
            }
            if (conf_.use_dropout) {
                dst_block.dropout(dropout_mask_, dropout_threshold,
                        dropout_seed, dropout_inv_q, dst_block_start,
                        dst_block_row_stride);
            }

            dst_block_t prev_dst;
            prev_dst.load_generic(dst_mem, dst_block_start,
                    dst_block_row_stride, conf_.transpose_dst,
                    is_dst_edge_block, remaining_m, remaining_n);
            dims_t off_po;
            for (int i = 0; i < max_supported_ndims; i++) {
                off_po[i] = off_dst[i];
            }
            if (conf_.transpose_dst) {
                std::swap(off_po[matmul_dim_1], off_po[matmul_dim_2]);
            }
            if (is_dst_edge_block) {
                dst_block.apply_post_ops_edge(conf_.post_ops, prev_dst, off_po,
                        matmul_dim_1, this, remaining_m, remaining_n);
            } else {
                dst_block.apply_post_ops(
                        conf_.post_ops, prev_dst, off_po, matmul_dim_1, this);
            }

            if (conf_.do_scale_dst) {
                dst_block.eltwise([=](float &el) { el /= dst_scale; });
            }
            if (conf_.use_dst_zeropoints) {
                dst_block.eltwise([=](float &el) { el += dst_zeropoint; });
            }
            dst_block.store_generic(dst_mem, dst_block_start,
                    dst_block_row_stride, conf_.transpose_dst,
                    is_dst_edge_block, remaining_m, remaining_n);
        }
    }

private:
    sycl_matmul_conf_t conf_;

    xpu::sycl::in_memory_arg_t data_;
    xpu::sycl::in_memory_arg_t weights_;
    xpu::sycl::in_memory_arg_t bias_;
    xpu::sycl::inout_memory_arg_t dst_;
    xpu::sycl::in_memory_arg_t data_scale_;
    data_type_t data_scales_dt_;
    xpu::sycl::in_memory_arg_t weights_scale_;
    data_type_t weights_scales_dt_;
    xpu::sycl::in_memory_arg_t dst_scale_;
    data_type_t dst_scales_dt_;
    xpu::sycl::in_memory_arg_t data_zeropoints_;
    data_type_t data_zeropoints_dt_;
    xpu::sycl::in_memory_arg_t weights_zeropoints_;
    data_type_t weights_zeropoints_dt_;
    xpu::sycl::in_memory_arg_t dst_zeropoints_;
    data_type_t dst_zeropoints_dt_;
    xpu::sycl::out_memory_arg_t dropout_mask_;
    xpu::sycl::in_memory_arg_t dropout_seed_;
    xpu::sycl::in_memory_arg_t dropout_probability_;
    post_op_input_args po_args_;
};

} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
