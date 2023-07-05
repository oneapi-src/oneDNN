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

    block_t(int dim_idx, dim_t block, const stride_t &stride)
        : dim_idx(dim_idx), block(block), stride(stride) {}

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
        oss << ", stride = " << stride;
        oss << ")";
        return oss.str();
    }

    bool is_empty() const { return dim_idx == undefined_dim_idx; }

    dim_t dim_idx = undefined_dim_idx; // Dimension index.
    dim_t block = 1; // Block size.
    stride_t stride; // Stride between elements of the block.
};
assert_trivially_serializable(block_t);

std::vector<block_t> normalize_blocks(
        const std::vector<block_t> &blocks, bool remove_size_1_blocks = true);

std::vector<block_t> compute_block_structure(const memory_desc_wrapper &mdw,
        bool inner_only = false, bool do_normalize = true);

} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
