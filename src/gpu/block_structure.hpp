/*******************************************************************************
* Copyright 2021-2023 Intel Corporation
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

#ifndef GPU_BLOCK_STRUCTURE_HPP
#define GPU_BLOCK_STRUCTURE_HPP

#include <sstream>
#include <vector>

#include "common/c_types_map.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/utils.hpp"
#include "gpu/serialization.hpp"
#include "gpu/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {

class stride_t {
public:
    stride_t() = default;

    stride_t(dim_t stride) : stride_(stride) {}

    bool operator==(const stride_t &other) const {
        return stride_ == other.stride_;
    }

    bool operator!=(const stride_t &other) const { return !operator==(other); }

    size_t get_hash() const { return dnnl::impl::gpu::get_hash(this); }

    operator dim_t() const {
        assert(is_fixed());
        return stride_;
    }

    bool is_fixed() const { return !is_unknown() && !is_undefined(); }
    bool is_unknown() const { return stride_ == unknown_stride; }
    bool is_undefined() const { return stride_ == undefined_stride; }

    stride_t &operator*=(const stride_t &other) {
        if (is_fixed() && other.is_fixed()) {
            stride_ *= other.stride_;
        } else {
            stride_ = unknown_stride;
        }
        return *this;
    }

    // XXX: Ambiguous when coprime
    stride_t &operator/=(const stride_t &other) {
        if (is_fixed() && other.is_fixed()) {
            stride_ /= other.stride_;
        } else {
            stride_ = unknown_stride;
        }
        return *this;
    }

    std::string str() const {
        std::ostringstream oss;
        if (is_fixed()) {
            oss << stride_;
        } else if (is_unknown()) {
            oss << "(unknown)";
        } else {
            oss << "(invalid)";
        }
        return oss.str();
    }

    static stride_t unknown() { return stride_t(unknown_stride); }
    static stride_t undefined() { return stride_t(undefined_stride); }

private:
    // Both negative sentinels: won't interfere with valid strides
    static constexpr dim_t unknown_stride = std::numeric_limits<dim_t>::min();
    static constexpr dim_t undefined_stride = unknown_stride + 1;

    dim_t stride_ = undefined_stride;
};
assert_trivially_serializable(stride_t);

inline stride_t operator*(const stride_t &a, const stride_t &b) {
    stride_t tmp = a;
    return tmp *= b;
}

inline stride_t operator*(const stride_t &a, dim_t b) {
    return a * stride_t(b);
}

inline stride_t operator*(dim_t a, const stride_t &b) {
    return stride_t(a) * b;
}

static constexpr dim_t undefined_dim_idx = -1;

struct block_t {
    block_t() = default;

    block_t(dim_t dim_idx, dim_t block, const stride_t &stride)
        : dim_idx(dim_idx), block(block), stride(stride) {}

    bool can_merge(const block_t &other, bool same_dim_only = true) const {
        bool dim_ok = !same_dim_only || (dim_idx == other.dim_idx);
        bool is_dense = (stride * block == other.stride);
        return dim_ok && is_dense;
    }

#if __cplusplus >= 202002L
    // Enabling default operator== on C++20 for validation purposes.
    bool operator==(const block_t &) const = default;
#else
    bool operator==(const block_t &other) const {
        return (dim_idx == other.dim_idx) && (block == other.block)
                && (stride == other.stride);
    }
#endif
    bool operator!=(const block_t &other) const { return !(*this == other); }

    size_t get_hash() const { return dnnl::impl::gpu::get_hash(this); }

    std::string str() const {
        std::ostringstream oss;
        oss << "block_t(dim_idx = " << dim_idx;
        oss << ", block = " << block;
        oss << ", stride = " << stride.str();
        oss << ")";
        return oss.str();
    }

    bool is_empty() const { return dim_idx == undefined_dim_idx; }

    dim_t dim_idx = undefined_dim_idx; // Dimension index.
    dim_t block = 1; // Block size.
    stride_t stride; // Stride between elements of the block.
};
assert_trivially_serializable(block_t);

// Static-sized layout of blocks
struct block_layout_t {
#if __cplusplus >= 202002L
    bool operator==(const block_layout_t &) const = default;
#endif
    using value_type = std::array<block_t, DNNL_MAX_NDIMS>;
    using iterator = value_type::iterator;
    using reverse_iterator = value_type::reverse_iterator;
    using const_iterator = value_type::const_iterator;
    using const_reverse_iterator = value_type::const_reverse_iterator;

    block_layout_t() = default;
    block_layout_t(const memory_desc_wrapper &mdw, bool inner_only = false,
            bool do_normalize = true);

    size_t size() const { return num_blocks; }
    bool empty() const { return num_blocks == 0; }
    const block_t &front() const {
        gpu_assert(num_blocks > 0);
        return blocks[0];
    }
    block_t &front() {
        gpu_assert(num_blocks > 0);
        return blocks[0];
    }
    const block_t &back() const {
        gpu_assert(num_blocks > 0);
        return blocks[num_blocks - 1];
    }
    block_t &back() {
        gpu_assert(num_blocks > 0);
        return blocks[num_blocks - 1];
    }

    // Iterators only go up to num_blocks, not necessarily to DNNL_MAX_NDIMS
    iterator begin() { return blocks.begin(); }
    const_iterator begin() const { return blocks.begin(); }
    reverse_iterator rbegin() {
        return blocks.rbegin() + static_cast<long>(blocks.size() - num_blocks);
    }
    const_reverse_iterator rbegin() const {
        return blocks.rbegin() + static_cast<long>(blocks.size() - num_blocks);
    }
    iterator end() { return blocks.begin() + num_blocks; }
    const_iterator end() const { return blocks.begin() + num_blocks; }
    reverse_iterator rend() { return blocks.rend(); }
    const_reverse_iterator rend() const { return blocks.rend(); }

    void erase(size_t idx) {
        for (size_t i = idx + 1; i < num_blocks; i++) {
            blocks[i - 1] = blocks[i];
        }
        blocks[num_blocks] = block_t();
        num_blocks--;
    }

    void insert(size_t idx, block_t val) {
        assert(num_blocks + 1 < DNNL_MAX_NDIMS);
        for (size_t i = idx; i < num_blocks; i++) {
            std::swap(val, blocks[i]);
        }
        append(val);
    }

    const block_t &operator[](size_t idx) const { return blocks[idx]; }

    void append(const block_t &block) { blocks[num_blocks++] = block; }
    size_t get_hash() const { return dnnl::impl::gpu::get_hash(this); }

    block_t &operator[](size_t idx) {
        assert(idx < num_blocks);
        return blocks[idx];
    }

    std::string str() const {
        std::ostringstream ss;
        for (size_t i = 0; i < num_blocks; i++) {
            const auto &block = blocks[i];
            ss << block.str() << " ";
        }
        return ss.str();
    }

    block_layout_t normalized(bool remove_size_1_blocks = true) const;

private:
    size_t num_blocks = 0;
    value_type blocks;
};

// Alias for block_layout_t::normalized which should be removed once jit::ir
// supports block_layout_t in favor of std::vector<block_t>
std::vector<block_t> normalize_blocks(
        const std::vector<block_t> &blocks, bool remove_size_1_blocks = true);

} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
