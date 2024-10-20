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
#pragma once
#include <algorithm>
#include <numeric>

#include "common/math_utils.hpp"
#include "gpu/intel/compute/dispatch.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

// For inputs and output that have the same format tag, we can reduce formats to
// three dimensions: the outer, concat, and inner dimensions.
// 1. All dimensions interior to the concat dimension, and any blocks of
//    non-concat dimensions belong to the inner dimension.
// 2. Anything remaining that is exterior to the concat dimension becomes the
//    outer dimension.
// 3. An additional interior block can be extracted from the concat dimension if
//    the GCD of the concat dimensions of all inputs and the output (AND in the
//    case of padded concat dimensions, the sizes of the concat dimensions'
//    zero-pads) is non-unit.
//
// E.g., ABC2a2b2c:ABC2a2b2c -> ABC2a2b2c
//         10x4x12:10x8x12   -> 10x12x12
// becomes
//             abc:abc       ->   abc
//          5x1x96:5x2x96    -> 5x3x96
// because the 2a block is absorbed into the inner ('c') dimension and there is
// a block of 4 that can be extracted from the concat ('b') dimension and
// absorbed into the inner dimension. This includes the 2b block, so the only
// remaining block is 2c, which can be combined safely into the inner block to
// get a plain format. Only blocking in the concat dimension matters, so if
// there is none, we will always get a plain format.
namespace axis {
enum normalized_axis_t { outer = 0, concat = 1, inner = 2, ndims = 3 };
}

class normalization_t {
    enum class padding_t { none = 0, external, internal };

    enum class striding_t {
        mismatched, // Formats are incompatible
        sparse, // Striding between concat dim/block and inner dim
        partitioned, // Striding between concat dim and exterior dim
        dense
    };

public:
    normalization_t(const memory_desc_t &md, int concat_dim)
        : ndims_(md.ndims)
        , concat_dim_(concat_dim)
        , extern_axis_idx_(ndims_)
        , data_type_(md.data_type)
        , blocking_(blocking(md))
        , chunk_size_(md.padded_dims[concat_dim])
        , padded_chunk_size_(math::gcd(md.dims[concat_dim], chunk_size_))
        , axis_order_(ndims_) {
        const auto &dims = md.padded_dims;
        const auto &strides = blocking_.strides;
        auto cmp = [&](int i, int j) {
            return strides[i] < strides[j]
                    || (strides[i] == strides[j] && dims[i] < dims[j]);
        };
        std::iota(axis_order_.begin(), axis_order_.end(), 0);
        std::sort(axis_order_.begin(), axis_order_.end(), cmp);

        std::vector<dim_t> blocks(ndims_, 1);
        for (int i = 0; i < blocking_.inner_nblks; ++i)
            blocks[blocking_.inner_idxs[i]] *= blocking_.inner_blks[i];

        int i = 0;
        for (; i < ndims_; ++i) {
            auto idx = axis_order_[i];
            if (idx == concat_dim) break;
            inner_size_ *= dims[idx];
        }

        for (++i; i < ndims_; ++i) {
            const auto &idx = axis_order_[i];
            if (dims[idx] == 1) continue;
            if (extern_axis_idx_ == ndims_) extern_axis_idx_ = idx;
            outer_size_ *= dims[idx] / blocks[idx];
            inner_size_ *= blocks[idx];
        }
    }

    bool add_source(const memory_desc_t &md) {
        if (md.padded_dims[concat_dim_] == 0) return true;
        if (md.data_type != data_type_) return false;
        if (md.format_kind != format_kind::blocked) return false;

        auto striding = validate(md);
        if (!striding_ok(striding)) return false;
        if (striding != striding_t::dense) can_read_past_concat_dim_ = false;

        auto dim = md.dims[concat_dim_];
        auto pdim = md.padded_dims[concat_dim_];
        auto source_chunk = math::gcd(dim, pdim - dim);

        // We can ignore padding if it only occurs on the final non-empty input
        if (padding_type_ == padding_t::external) {
            padding_type_ = padding_t::internal;
            chunk_size_ = padded_chunk_size_;
        } else if (padding_type_ == padding_t::none && dim != pdim)
            padding_type_ = padding_t::external;

        if (padding_type_ == padding_t::internal) {
            chunk_size_ = math::gcd(chunk_size_, source_chunk);
        } else {
            chunk_size_ = math::gcd(chunk_size_, pdim);
            padded_chunk_size_ = math::gcd(padded_chunk_size_, source_chunk);
        }
        return true;
    }

    data_type_t data_type() const { return data_type_; }
    size_t data_type_size() const { return types::data_type_size(data_type_); }

    dim_t max_write_size() const {
        dim_t write_size = 1;
        dim_t rem_chunk_size = chunk_size_;
        for (int i = blocking_.inner_nblks - 1; i >= 0; --i) {
            const auto &idx = blocking_.inner_idxs[i];
            const auto &size = blocking_.inner_blks[i];
            if (idx == concat_dim_) {
                if (rem_chunk_size < size)
                    return rem_chunk_size * write_size * data_type_size();
                rem_chunk_size /= size;
            }
            write_size *= size;
        }
        return inner_size_ * chunk_size_ * data_type_size();
    }

    dim_t max_read_size() const {
        dim_t size = inner_size_ * chunk_size_ * data_type_size();
        if (can_read_past_concat_dim_) size *= outer_size_;
        return size;
    }

    void operator()(memory_desc_t &) const;

private:
    static bool striding_ok(striding_t striding) {
        return striding == striding_t::partitioned
                || striding == striding_t::dense;
    }

    striding_t validate(const memory_desc_t &md) const {
        const auto blkg = blocking(md);
        if (blkg.inner_nblks != blocking_.inner_nblks)
            return striding_t::mismatched;

        dim_t exp_stride = 1;
        std::vector<dim_t> dim_blks(md.ndims, 1);
        // Check that inner blocking is the same
        for (int i = 0; i < blkg.inner_nblks; ++i) {
            const auto idx = blkg.inner_idxs[i];
            const auto size = blkg.inner_blks[i];
            if (idx != blocking_.inner_idxs[i]
                    || size != blocking_.inner_blks[i])
                return striding_t::mismatched;
            exp_stride *= size;
            dim_blks[idx] *= size;
        }

        // Check that outer dimensions are in the same order and that tensors
        // are the same shape except for the concat dim
        bool last_dim_was_concat_dim = false;
        striding_t striding = striding_t::dense;
        for (auto idx : axis_order_) {
            if (blkg.strides[idx] < exp_stride) return striding_t::mismatched;
            if (blkg.strides[idx] > exp_stride) {
                if (!last_dim_was_concat_dim) return striding_t::sparse;
                striding = striding_t::partitioned;
            }
            dim_t step = md.padded_dims[idx] / dim_blks[idx];
            exp_stride = step * blkg.strides[idx];
            last_dim_was_concat_dim = (idx == concat_dim_);
        }

        return striding;
    }

    static blocking_desc_t blocking(const memory_desc_t &md) {
        auto blkg = md.format_desc.blocking;
        const auto old_nblks = blkg.inner_nblks;
        int nblks = 0;

        for (int i = 0; i < old_nblks; ++i) {
            const auto idx = blkg.inner_idxs[i];
            const auto size = blkg.inner_blks[i];
            if (nblks && blkg.inner_idxs[nblks - 1] == idx)
                blkg.inner_blks[nblks - 1] *= size;
            else {
                blkg.inner_idxs[nblks] = idx;
                blkg.inner_blks[nblks] = size;
                nblks++;
            }
        }
        blkg.inner_nblks = nblks;
        return blkg;
    }

    int ndims_;
    int concat_dim_;
    int extern_axis_idx_;
    data_type_t data_type_;
    blocking_desc_t blocking_;
    padding_t padding_type_ = padding_t::none;
    bool can_read_past_concat_dim_ = true;

    dim_t chunk_size_; // concat-dim elements that can be written simultaneously
    dim_t padded_chunk_size_; // pessimistic chunk size for internal padding
    dim_t inner_size_ = 1;
    dim_t outer_size_ = 1;
    std::vector<int> axis_order_;
};

inline void normalization_t::operator()(memory_desc_t &md) const {
    auto chunk_size = chunk_size_;
    auto &blkg = md.format_desc.blocking;
    auto &dims = md.dims;
    auto &pdims = md.padded_dims;
    auto &poff = md.padded_offsets;
    auto &nblks = blkg.inner_nblks;
    const auto old_nblks = nblks;
    const auto old_ndims = md.ndims;
    dims_t inner_idxs, inner_blks; // temporaries

    dims[axis::concat] = utils::div_up(dims[concat_dim_], chunk_size);
    pdims[axis::concat] = pdims[concat_dim_] / chunk_size;
    dims[axis::outer] = pdims[axis::outer] = outer_size_;
    dims[axis::inner] = pdims[axis::inner] = inner_size_ * chunk_size;
    poff[axis::outer] = poff[axis::concat] = poff[axis::inner] = 0;
    md.ndims = axis::ndims;
    nblks = 0;

    auto add_blk = [&](int idx, dim_t size) {
        if (size == 1) return;
        if (nblks && inner_idxs[nblks - 1] == idx)
            inner_blks[nblks - 1] *= size;
        else {
            inner_idxs[nblks] = idx;
            inner_blks[nblks] = size;
            nblks++;
        }
    };

    dim_t stride = 1;
    // Temporarily store indices and blocks in reverse order so we can combine
    // blocks where needed without having to know the final block count.
    for (int i = old_nblks - 1; i >= 0; --i) {
        dim_t inner_size = blkg.inner_blks[i];
        dim_t concat_size = 1;
        stride *= inner_size;
        if (blkg.inner_idxs[i] == concat_dim_) {
            if (inner_size > chunk_size) {
                concat_size = inner_size / chunk_size;
                inner_size = chunk_size;
                chunk_size = 1;
            } else
                chunk_size /= inner_size;
        }

        add_blk(axis::inner, inner_size);
        add_blk(axis::concat, concat_size);
    }

    // Ignore outermost inner block
    if (nblks && inner_idxs[nblks - 1] == axis::inner) {
        stride /= inner_blks[nblks - 1];
        --nblks;
    }

    // Copy new indices and blocks in original order.
    for (int i = 0; i < nblks; ++i) {
        blkg.inner_idxs[nblks - 1 - i] = inner_idxs[i];
        blkg.inner_blks[nblks - 1 - i] = inner_blks[i];
    }

    auto &strides = blkg.strides;
    const auto concat_stride = strides[concat_dim_] * chunk_size;
    strides[axis::outer] = extern_axis_idx_ < old_ndims
            ? strides[extern_axis_idx_]
            : pdims[axis::inner] * pdims[axis::concat];
    strides[axis::inner] = stride;
    strides[axis::concat] = concat_stride;
}

struct prb_info_t {
    static constexpr int scattered_message_penalty = 2;

    int simd;
    int type_size;
    dim_t block;
    int messages;

    prb_info_t(int simd, int type_size, dim_t max_elems, dim_t max_read_size,
            dim_t max_write_size, compute::gpu_arch_t hw)
        : simd(simd), type_size(type_size) {
        dim_t best_block = 0;
        int best_messages = 1;
        const dim_t max_write_elems = max_write_size / type_size;
        for (dim_t read_elems = 1; read_elems <= max_elems; ++read_elems) {
            dim_t write_elems = std::min(max_write_elems, read_elems);
            dim_t read_block_size = read_elems * (dim_t)type_size;
            if (read_block_size > max_read_size) break;
            if (read_block_size > max_block_size(hw)) break;
            if (read_elems < max_write_elems && max_write_elems % read_elems)
                continue;
            if (read_elems > max_write_elems
                    && (read_elems % max_write_elems
                            || max_read_size % read_block_size))
                continue;
            int messages = subgroup_messages(
                    simd, read_elems, write_elems, type_size, hw);
            if (read_elems * best_messages < best_block * messages) continue;
            if (read_elems * best_messages == best_block * messages
                    && read_elems < best_block)
                continue;
            best_block = read_elems;
            best_messages = messages;
        }
        block = best_block;
        messages = best_messages;
    }

    static int register_bytes(compute::gpu_arch_t hw) {
        return hw >= compute::gpu_arch_t::xe_hpc ? 64 : 32;
    }

    static dim_t max_block_size(compute::gpu_arch_t hw) {
        return 16 * register_bytes(hw);
    }

    static int subgroup_messages(int simd, dim_t read_block, dim_t write_block,
            dim_t type_size, compute::gpu_arch_t hw) {
        const int reg_size = register_bytes(hw);
        const bool scattered_load = read_block * type_size % 4 != 0;
        const bool scattered_store = write_block * type_size % 16 != 0;
        const dim_t load_type_size
                = scattered_load ? std::max(type_size, (dim_t)4) : type_size;
        const dim_t store_type_size
                = scattered_store ? std::max(type_size, (dim_t)4) : type_size;
        const int load_regs = scattered_load ? 2 : 4;
        const int store_regs = scattered_store ? 2 : 8;
        const dim_t load_size = load_regs * reg_size;
        const dim_t store_size = store_regs * reg_size;
        return into<int>(utils::div_up(read_block * load_type_size, load_size)
                + (read_block / write_block)
                        * utils::div_up(
                                write_block * store_type_size, store_size));
    }

    bool operator<(const prb_info_t &other) const {
        auto average_bytes = type_size * block * other.messages;
        auto other_average_bytes = other.type_size * other.block * messages;
        if (average_bytes != other_average_bytes)
            return average_bytes > other_average_bytes;
        if (type_size * block != other.type_size * other.block)
            return type_size * block > other.type_size * other.block;
        if (simd != other.simd) return simd > other.simd;
        return type_size > other.type_size;
    }
};

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
