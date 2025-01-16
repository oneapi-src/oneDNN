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

#ifndef GPU_GENERIC_SYCL_REDUCTION_KERNELS_HPP
#define GPU_GENERIC_SYCL_REDUCTION_KERNELS_HPP

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/primitive_exec_types.hpp"
#include "common/utils.hpp"
#include "gpu/generic/sycl/sycl_io_helper.hpp"
#include "gpu/generic/sycl/sycl_math_utils.hpp"
#include "gpu/generic/sycl/sycl_primitive_conf.hpp"
#include "xpu/sycl/memory_storage_base.hpp"
#include "xpu/sycl/types.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

struct init_kernel_t {
    init_kernel_t(xpu::sycl::out_memory_arg_t &out_arg, alg_kind_t alg)
        : out_arg_(out_arg), alg_(alg) {}

    void operator()(::sycl::item<1> item) const {
        auto *out = reinterpret_cast<float *>(out_arg_.get_pointer());
        const int idx = item.get_linear_id();
        float val = 0;
        if (alg_ == alg_kind::reduction_min)
            val = std::numeric_limits<float>::max();
        else if (alg_ == alg_kind::reduction_max)
            val = std::numeric_limits<float>::lowest();
        else if (alg_ == alg_kind::reduction_mul)
            val = 1;
        else
            val = 0;

        out[idx] = val;
    }

private:
    xpu::sycl::out_memory_arg_t out_arg_;
    alg_kind_t alg_;
};

struct atomic_finalize_kernel_t {
    atomic_finalize_kernel_t(::sycl::handler &cgh, const exec_ctx_t &ctx,
            data_type_t dt, xpu::sycl::out_memory_arg_t &out_arg,
            alg_kind_t alg, float p, float eps, sycl_post_ops_t &post_ops,
            xpu::sycl::md_t dst_md, int reduce_size)
        : dt_(dt)
        , out_arg_(out_arg)
        , alg_(alg)
        , p_(p)
        , eps_(eps)
        , post_ops_(post_ops)
        , po_args_(cgh, ctx, post_ops)
        , dst_md_(dst_md)
        , reduce_size_(reduce_size) {}

    void operator()(::sycl::item<1> item) const {
        void *out_ptr = out_arg_.get_pointer();
        auto idx = item.get_linear_id();
        auto val = load_float_value(dt_, out_ptr, idx);
        if (alg_ == alg_kind::reduction_norm_lp_max) {
            val = ::sycl::rootn(::sycl::max(val, eps_), p_);
        } else if (alg_ == alg_kind::reduction_norm_lp_sum) {
            val = ::sycl::rootn(val + eps_, p_);
        } else if (alg_ == alg_kind::reduction_norm_lp_power_p_max) {
            val = ::sycl::max(val, eps_);
        } else if (alg_ == alg_kind::reduction_norm_lp_power_p_sum) {
            val = val + eps_;
        } else if (alg_ == alg_kind::reduction_mean) {
            val = val / reduce_size_;
        }

        auto prev_val = val;
        auto l_offset = idx;
        dims_t pos;
        for (int i = 0; i < dst_md_.ndims(); i++) {
            const int d = dst_md_.ndims() - 1 - i;
            const dim_t cur_dim = dst_md_.dims()[d];
            pos[d] = l_offset % cur_dim;
            l_offset = l_offset / cur_dim;
        }
        val = post_ops_.apply(val, prev_val, po_args_, pos);

        store_float_value(dt_, val, out_ptr, idx);
    }

private:
    data_type_t dt_;
    xpu::sycl::out_memory_arg_t out_arg_;
    alg_kind_t alg_;
    float p_, eps_;
    sycl_post_ops_t post_ops_;
    post_op_input_args po_args_;
    xpu::sycl::md_t dst_md_;
    int reduce_size_;
};

struct Reducer {
    alg_kind_t alg_;
    float p_, eps_;
    bool needs_prepare_;
    bool iter_needs_finalize_;

    Reducer(alg_kind_t alg, float p, float eps, bool needs_prepare,
            bool iter_needs_finalize)
        : alg_(alg)
        , p_(p)
        , eps_(eps)
        , needs_prepare_(needs_prepare)
        , iter_needs_finalize_(iter_needs_finalize) {}

    inline float identity() const {
        if (alg_ == alg_kind::reduction_min) {
            return std::numeric_limits<float>::max();
        } else if (alg_ == alg_kind::reduction_max) {
            return std::numeric_limits<float>::lowest();
        } else if (alg_ == alg_kind::reduction_mul) {
            return 1.f;
        }

        return 0.f;
    }

    inline bool needs_finalize() const {
        return (alg_ == alg_kind::reduction_mean
                || alg_ == alg_kind::reduction_norm_lp_max
                || alg_ == alg_kind::reduction_norm_lp_sum
                || alg_ == alg_kind::reduction_norm_lp_power_p_max
                || alg_ == alg_kind::reduction_norm_lp_power_p_sum);
    }

    inline float subgroup_reduce(
            ::sycl::sub_group &subgroup, float sg_input) const {
        if (alg_ == alg_kind::reduction_sum || alg_ == alg_kind::reduction_mean
                || alg_ == alg_kind::reduction_norm_lp_max
                || alg_ == alg_kind::reduction_norm_lp_sum
                || alg_ == alg_kind::reduction_norm_lp_power_p_max
                || alg_ == alg_kind::reduction_norm_lp_power_p_sum) {
            return ::sycl::reduce_over_group(
                    subgroup, sg_input, ::sycl::plus<float> {});
        } else if (alg_ == alg_kind::reduction_min) {
            return ::sycl::reduce_over_group(
                    subgroup, sg_input, ::sycl::minimum<float> {});
        } else if (alg_ == alg_kind::reduction_max) {
            return ::sycl::reduce_over_group(
                    subgroup, sg_input, ::sycl::maximum<float> {});
        } else if (alg_ == alg_kind::reduction_mul) {
            return ::sycl::reduce_over_group(
                    subgroup, sg_input, ::sycl::multiplies<float> {});
        }

        return ::sycl::nan(0U);
    }

    inline float reduce(float lhs, float rhs) const {
        if (alg_ == alg_kind::reduction_sum || alg_ == alg_kind::reduction_mean
                || alg_ == alg_kind::reduction_norm_lp_max
                || alg_ == alg_kind::reduction_norm_lp_sum
                || alg_ == alg_kind::reduction_norm_lp_power_p_max
                || alg_ == alg_kind::reduction_norm_lp_power_p_sum) {
            return lhs + rhs;
        } else if (alg_ == alg_kind::reduction_min) {
            return ::sycl::min(lhs, rhs);
        } else if (alg_ == alg_kind::reduction_max) {
            return ::sycl::max(lhs, rhs);
        } else if (alg_ == alg_kind::reduction_mul) {
            return lhs * rhs;
        }

        return ::sycl::nan(0U);
    }

    template <::sycl::memory_order Order, ::sycl::memory_scope Scope,
            ::sycl::access::address_space Space>
    void atomic_op(data_type_t dt, void *ref, int idx, float val, int size) {
        auto atomic_out = ::sycl::atomic_ref<float, Order, Scope, Space>(
                reinterpret_cast<float *>(ref)[idx]);
        if (alg_ == alg_kind::reduction_sum || alg_ == alg_kind::reduction_mean
                || alg_ == alg_kind::reduction_norm_lp_max
                || alg_ == alg_kind::reduction_norm_lp_sum
                || alg_ == alg_kind::reduction_norm_lp_power_p_max
                || alg_ == alg_kind::reduction_norm_lp_power_p_sum) {
            atomic_out.fetch_add(val);
        } else if (alg_ == alg_kind::reduction_min) {
            atomic_out.fetch_min(val);
        } else if (alg_ == alg_kind::reduction_max) {
            atomic_out.fetch_max(val);
        }
    }

    inline void prepare(float &val) {
        if (needs_prepare_
                && (alg_ == alg_kind::reduction_norm_lp_max
                        || alg_ == alg_kind::reduction_norm_lp_sum
                        || alg_ == alg_kind::reduction_norm_lp_power_p_max
                        || alg_ == alg_kind::reduction_norm_lp_power_p_sum)) {
            val = ::sycl::pow(::sycl::fabs(val), p_);
        }
    }

    inline void finalize(float &val, int size) {
        if (alg_ == alg_kind::reduction_mean) {
            val /= size;
        } else if (alg_ == alg_kind::reduction_norm_lp_max
                && iter_needs_finalize_) {
            val = ::sycl::rootn(::sycl::max(val, eps_), p_);
        } else if (alg_ == alg_kind::reduction_norm_lp_sum
                && iter_needs_finalize_) {
            val = ::sycl::rootn(val + eps_, p_);
        } else if (alg_ == alg_kind::reduction_norm_lp_power_p_max
                && iter_needs_finalize_) {
            val = ::sycl::max(val, eps_);
        } else if (alg_ == alg_kind::reduction_norm_lp_power_p_sum
                && iter_needs_finalize_) {
            val = val + eps_;
        }
    }
};

struct LocalMemTile {
    using T = float;
    using Index = int;

    static constexpr Index Dim = 3;
    // XXX: Set this depending on reduction size for optimisation
    static constexpr bool CheckBounds = true;
    static constexpr Index RowDim = Dim == 3 ? 1 : 0;
    static constexpr Index ColDim = Dim == 3 ? 2 : 1;
    Index row_tile_;
    Index col_tile_;
    Index row_lim_;
    Index col_lim_;
    Index local_id_;
    Index local_row_id_;
    Index local_col_id_;
    Index wg_id_;
    ::sycl::nd_item<Dim> &nd_item_;
    bool bank_offset_;
    T pad_val_;

    LocalMemTile(data_type_t src_dt, data_type_t dst_dt, Index row_tile,
            Index col_tile, Index row_lim, Index col_lim,
            ::sycl::nd_item<Dim> &nd_item, bool bank_offset, T pad_val = 0)
        : row_tile_ {row_tile}
        , col_tile_ {col_tile}
        , row_lim_ {row_lim}
        , col_lim_ {col_lim}
        , local_id_ {static_cast<Index>(nd_item.get_local_linear_id())}
        , local_row_id_ {static_cast<Index>(nd_item.get_local_id(RowDim))}
        , local_col_id_ {static_cast<Index>(nd_item.get_local_id(ColDim))}
        , wg_id_ {static_cast<Index>(nd_item.get_group(0))}
        , nd_item_(nd_item)
        , bank_offset_(bank_offset)
        , pad_val_ {pad_val} {}

    Index get_row_tile() { return row_tile_; }
    Index get_col_tile() { return col_tile_; }

    Index get_local_id(bool is_transposed = false) {
        if (!is_transposed) {
            return local_row_id_ * (col_tile_ + bank_offset_) + local_col_id_;
        } else {
            return local_col_id_ * (row_tile_ + bank_offset_) + local_row_id_;
        }
    }

    Index get_local_row_id() { return local_row_id_; }
    Index get_local_col_id() { return local_col_id_; }
    Index get_wg_id() { return wg_id_; }
    Index get_global_row_id() { return nd_item_.get_global_id(RowDim); }
    Index get_global_col_id() { return nd_item_.get_global_id(ColDim); }

    void load_memory(data_type_t in_dt, void *global_in, data_type_t local_dt,
            const xpu::sycl::md_t &in_md, void *local_out, Reducer &reducer,
            bool is_first_red_iter) {
        auto const local_id = get_local_id();
        auto const wg_batch_id = get_wg_id();
        auto const global_row_id = get_global_row_id();
        auto const global_col_id = get_global_col_id();

        auto const global_id = wg_batch_id * row_lim_ * col_lim_
                + global_row_id * col_lim_ + global_col_id;

        if constexpr (CheckBounds) {
            const auto within_bounds
                    = global_row_id < row_lim_ && global_col_id < col_lim_;
            int idx = is_first_red_iter ? in_md.off_l(global_id) : global_id;
            float val = within_bounds ? load_float_value(in_dt, global_in, idx)
                                      : pad_val_;
            reducer.prepare(val);
            store_float_value(local_dt, val, local_out, local_id);
        } else {
            // maybe do subgroup load
            auto val = load_float_value(in_dt, global_in, global_id);
            reducer.prepare(val);
            store_float_value(local_dt, val, local_out, local_id);
        }
        group_barrier(nd_item_.get_group());
    }

    void store_memory(data_type_t in_dt, void *local_in, data_type_t out_dt,
            void *global_out, const xpu::sycl::md_t &out_md, bool is_reduced,
            bool is_transposed, bool is_last_red_iter,
            const sycl_post_ops_t &post_ops,
            const post_op_input_args &po_args) {
        auto const local_id = get_local_id(is_transposed);
        auto const wg_batch_id = get_wg_id();
        auto const col_id = get_global_col_id();

        Index row_id;
        Index row_lim;
        Index global_id;
        if (is_reduced) {
            row_id = get_local_row_id();
            row_lim = get_row_tile();
            global_id = wg_batch_id * col_lim_ + col_id;
        } else {
            row_id = get_global_row_id();
            row_lim = row_lim_;
            if (is_transposed) {
                global_id
                        = (wg_batch_id * col_lim_ + col_id) * row_lim_ + row_id;
            } else {
                global_id
                        = (wg_batch_id * row_lim_ + row_id) * col_lim_ + col_id;
            }
        }

        if (row_id < row_lim && col_id < col_lim_) {
            auto val = load_float_value(in_dt, local_in, local_id);
            auto idx = is_last_red_iter ? out_md.off_l(global_id) : global_id;
            float prev_val = load_float_value(out_dt, global_out, idx);

            if (is_last_red_iter) {
                auto l_offset = global_id;
                dims_t pos;
                for (int i = 0; i < out_md.ndims(); i++) {
                    const int d = out_md.ndims() - 1 - i;
                    const dim_t cur_dim = out_md.dims()[d];
                    pos[d] = l_offset % cur_dim;
                    l_offset = l_offset / cur_dim;
                }
                val = post_ops.apply(val, prev_val, po_args, pos);
            }

            store_float_value(out_dt, val, global_out, idx);
        }
    }

    T load_local(data_type_t dt, void *input, int index) {
        return load_float_value(dt, input, index); // ((T *)input)[index];
    }

    void store_local(data_type_t dt, void *output, int index, T val) {
        store_float_value(dt, val, output, index);
    }

private:
    template <bool Inplace>
    void transpose_impl_(
            data_type_t in_dt, void *input, data_type_t out_dt, void *output) {
        auto const local_id = get_local_id(false);
        auto const trans_local_id = get_local_id(true);
        auto group = nd_item_.get_group();
        const auto val = load_local(in_dt, input, local_id);
        if constexpr (Inplace) { group_barrier(group); }
        store_local(out_dt, output, trans_local_id, val);
        group_barrier(group);
    }

public:
    void transpose(
            data_type_t in_dt, void *input, data_type_t out_dt, void *output) {
        transpose_impl_<false>(in_dt, input, out_dt, output);
    }

    void transpose(data_type_t dt, void *input) {
        transpose_impl_<true>(dt, input, dt, input);
    }

    void sg_reduce_impl(data_type_t in_dt, void *input, data_type_t out_dt,
            void *output, bool is_transposed, Reducer &reducer) {
        Index const reduce_lim = row_tile_;
        Index const stride_lim = col_tile_;
        if (row_tile_ == 1) { return; }

        auto group = nd_item_.get_group();
        auto subgroup = nd_item_.get_sub_group();

        Index const sg_size = subgroup.get_max_local_range()[0];
        Index const sg_group_id = subgroup.get_group_linear_id();
        Index const sg_local_id = subgroup.get_local_linear_id();

        // The tile is read in the shape [row_tile, col_tile] ->
        // [num_row_blocks*max_subgroup_size, col_tile]. Thus the local range has
        // the same shape (num_row_blocks*max_subgroup_size, col_tile), using this
        // we can assign num_row_blocks*col_tile subgroups to reduce a chunk of
        // memory size max_subgroup_size

        auto const num_reduce_blocks
                = ::sycl::max((reduce_lim) / sg_size, Index {1});
        auto const reduce_block_id = sg_group_id % num_reduce_blocks;
        auto const local_reduce_id = reduce_block_id * sg_size + sg_local_id;
        auto const local_stride_id = sg_group_id / num_reduce_blocks;

        Index input_id;
        Index output_id;
        if (!is_transposed) {
            input_id = local_reduce_id * (stride_lim + bank_offset_)
                    + local_stride_id;
            output_id = reduce_block_id * (stride_lim + bank_offset_)
                    + local_stride_id;
        } else {
            input_id = local_stride_id * (reduce_lim + bank_offset_)
                    + local_reduce_id;
            output_id = local_stride_id * (num_reduce_blocks + bank_offset_)
                    + reduce_block_id;
        }
        auto const sg_input = local_reduce_id < reduce_lim
                ? load_local(in_dt, input, input_id)
                : reducer.identity();
        float sg_output = reducer.subgroup_reduce(subgroup, sg_input);

        group_barrier(group);

        if (subgroup.leader()) {
            store_local(out_dt, output, output_id, sg_output);
        }
        row_tile_ = num_reduce_blocks;
        group_barrier(group);
    }

    void sg_reduce(data_type_t in_dt, void *input, data_type_t out_dt,
            void *output, int num_sg_reductions, bool is_transposed,
            Reducer &reducer) {
        for (auto i = 0; i < num_sg_reductions; i++) {
            sg_reduce_impl(
                    in_dt, input, out_dt, output, is_transposed, reducer);
        }
    }

    void sg_reduce(data_type_t dt, void *input, int num_sg_reductions,
            bool is_transposed, Reducer &reducer) {
        for (auto i = 0; i < num_sg_reductions; i++) {
            sg_reduce_impl(dt, input, dt, input, is_transposed, reducer);
        }
    }

    void wi_reduce_impl(data_type_t in_dt, void *input, data_type_t out_dt,
            void *output, bool is_transposed, Reducer &reducer) {
        if (row_lim_ == 1) { return; }

        auto output_id = get_local_id(is_transposed);
        if (get_local_row_id() == 0) {
            auto input_id = output_id;
            for (auto row_id = 1; row_id < row_tile_; row_id++) {
                if (is_transposed) {
                    input_id++;
                } else {
                    input_id += (col_tile_ + bank_offset_);
                }
                auto lhs = load_float_value(in_dt, input, input_id);
                auto rhs = load_float_value(out_dt, output, output_id);
                auto val = reducer.reduce(lhs, rhs);
                store_float_value(out_dt, val, output, output_id);
            }
        }

        group_barrier(nd_item_.get_group());
        row_tile_ = 1;

        if (get_local_row_id() == 0) {
            auto final_output_id
                    = get_local_col_id() * (row_tile_ + bank_offset_);
            auto val = load_float_value(out_dt, output, output_id);
            store_float_value(out_dt, val, output, final_output_id);
        }
        group_barrier(nd_item_.get_group());
    }

    void wi_reduce(
            data_type_t dt, void *input, bool is_transposed, Reducer &reducer) {
        wi_reduce_impl(dt, input, dt, input, is_transposed, reducer);
    }

    void finalize_reduce(data_type_t dt, void *input, int finalize_param,
            bool is_transposed, Reducer &reducer) {
        if (reducer.needs_finalize()) {
            const auto local_row_id = get_local_row_id();
            const auto local_id = get_local_id(is_transposed);
            if (local_row_id == 0) {
                auto val = load_float_value(dt, input, local_id);
                reducer.finalize(val, finalize_param);
                store_float_value(dt, val, input, local_id);
            }
        }
    }

    template <::sycl::memory_order Order, ::sycl::memory_scope Scope,
            ::sycl::access::address_space Space>
    void atomic_reduce(data_type_t in_dt, void *input, data_type_t out_dt,
            void *output, const xpu::sycl::md_t &out_md, Index finalize_param,
            Index batch_groups, bool is_transposed, Reducer &reducer) {
        constexpr bool GlobalReduce
                = Space == ::sycl::access::address_space::global_space;

        // Get local indexes
        const auto local_row_id = get_local_row_id();
        const auto local_col_id = get_local_col_id();

        // Get global column id and limit
        const auto global_col_id
                = GlobalReduce ? get_global_col_id() : local_col_id;
        const auto global_col_lim = GlobalReduce ? col_lim_ : col_tile_;

        const auto row_id_ok = local_row_id < row_tile_;
        const auto col_id_ok
                = GlobalReduce ? global_col_id < global_col_lim : true;

        if (row_id_ok && col_id_ok) {
            const auto outer_id = GlobalReduce ? get_wg_id() : 0;
            const auto output_id = out_md.off_l(
                    (outer_id % batch_groups) * global_col_lim + global_col_id);
            const auto input_id = get_local_id(is_transposed);
            reducer.atomic_op<Order, Scope, Space>(out_dt, output, output_id,
                    load_float_value(in_dt, input, input_id), finalize_param);
        }

        // Atomics do a full reduction of the row.
        row_tile_ = 1;
    }
};

struct reduction_kernel_fwd_t {
    using LocalMem = ::sycl::local_accessor<uint8_t, 1>;

    static auto constexpr Order = ::sycl::memory_order::relaxed;
    static auto constexpr DeviceScope = ::sycl::memory_scope::device;
    static auto constexpr WGScope = ::sycl::memory_scope::work_group;
    static auto constexpr GlobalSpace
            = ::sycl::access::address_space::global_space;
    static auto constexpr LocalSpace
            = ::sycl::access::address_space::local_space;

    reduction_kernel_fwd_t(const sycl_reduction_conf_t &conf, int row_tile,
            int col_tile, int batch_groups, bool needs_atomic_reduce,
            LocalMem &local_mem, ::sycl::handler &cgh, const exec_ctx_t &ctx)
        : conf_(conf)
        , src_(CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_SRC))
        , dst_(CTX_OUT_SYCL_KERNEL_MEMORY(DNNL_ARG_DST))
        , local_mem_(local_mem)
        , row_tile_(row_tile)
        , col_tile_(col_tile)
        , batch_groups_(batch_groups)
        , needs_atomic_reduce_(needs_atomic_reduce)
        , po_args_(cgh, ctx, conf_.post_ops) {}

    reduction_kernel_fwd_t(xpu::sycl::in_memory_arg_t &src_arg,
            xpu::sycl::out_memory_arg_t &dst_arg,
            const sycl_reduction_conf_t &conf, bool needs_atomic_reduce,
            LocalMem &local_mem, ::sycl::handler &cgh, const exec_ctx_t &ctx)
        : conf_(conf)
        , src_(src_arg)
        , dst_(dst_arg)
        , local_mem_(local_mem)
        , row_tile_(conf.tile_row)
        , col_tile_(conf.tile_col)
        , batch_groups_(conf.batch_groups)
        , needs_atomic_reduce_(needs_atomic_reduce)
        , po_args_(cgh, ctx, conf_.post_ops) {}

    void operator()(::sycl::nd_item<3> nd_item) const {
        Reducer reducer(conf_.alg, conf_.p, conf_.eps, conf_.is_first_iter,
                conf_.is_last_iter);

        LocalMemTile tile(conf_.src_dt, conf_.dst_dt, row_tile_, col_tile_,
                conf_.reduce_size, conf_.stride_size, nd_item,
                conf_.bank_offset, reducer.identity());

        // Copy values from global memory to local
        tile.load_memory(conf_.src_dt, src_ptr(), conf_.local_mem_dt,
                conf_.src_md, local_ptr(), reducer, conf_.is_first_iter);

        if (conf_.transpose) {
            // Transpose data in local memory
            tile.transpose(conf_.local_mem_dt, local_ptr());
        }

        // Reduce values using subgroup reducer
        tile.sg_reduce(conf_.local_mem_dt, local_ptr(), conf_.num_sg_reductions,
                conf_.transpose, reducer);

        if (conf_.alg == alg_kind::reduction_mean
                && conf_.num_sg_reductions == 0) {
            // Reduce values using work-item reducer
            tile.wi_reduce(
                    conf_.local_mem_dt, local_ptr(), conf_.transpose, reducer);
        }

        if (needs_atomic_reduce_) {
            // Reduce remaining values into global memory using global atomics
            tile.atomic_reduce<Order, DeviceScope, GlobalSpace>(
                    conf_.local_mem_dt, local_ptr(), conf_.dst_dt, dst_ptr(),
                    conf_.dst_md,
                    conf_.reduce_size * conf_.batch_size / batch_groups_,
                    batch_groups_, conf_.transpose, reducer);
        } else {
            // Finalize reduction
            tile.finalize_reduce(conf_.local_mem_dt, local_ptr(),
                    conf_.reduce_size, conf_.transpose, reducer);
            tile.store_memory(conf_.local_mem_dt, local_ptr(), conf_.dst_dt,
                    dst_ptr(), conf_.dst_md, true, conf_.transpose,
                    conf_.is_last_iter, conf_.post_ops, po_args_);
        }
    }

private:
    void *src_ptr() const { return src_.get_pointer(); }
    void *dst_ptr() const { return dst_.get_pointer(); }
    void *local_ptr() const {
        return local_mem_.get_multi_ptr<::sycl::access::decorated::no>().get();
    }

    sycl_reduction_conf_t conf_;
    xpu::sycl::in_memory_arg_t src_;
    xpu::sycl::out_memory_arg_t dst_;
    LocalMem local_mem_;
    int row_tile_;
    int col_tile_;
    int batch_groups_;
    bool needs_atomic_reduce_;
    post_op_input_args po_args_;
};

} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
