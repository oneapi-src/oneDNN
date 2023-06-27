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

namespace dnnl {
namespace impl {
namespace gpu {

enum class stride_kind_t {
    undef,
    fixed,
    unknown,
};

class stride_t {
public:
    stride_t() = default;

    stride_t(dim_t stride) : stride_t(stride_kind_t::fixed, stride) {}

    bool operator==(const stride_t &other) const {
        return (kind_ == other.kind_) && (stride_ == other.stride_);
    }

    bool operator!=(const stride_t &other) const { return !operator==(other); }

    size_t get_hash() const {
        size_t hash = 0;
        hash = hash_combine(hash, kind_);
        hash = hash_combine(hash, stride_);
        return hash;
    }

    operator dim_t() const {
        assert(kind_ == stride_kind_t::fixed);
        return stride_;
    }

    bool is_fixed() const { return kind_ == stride_kind_t::fixed; }

    bool is_unknown() const { return kind_ == stride_kind_t::unknown; }

    stride_t &operator*=(const stride_t &other) {
        if (is_fixed() && other.is_fixed()) {
            stride_ *= other.stride_;
        } else {
            set_unknown();
        }
        return *this;
    }

    stride_t &operator/=(const stride_t &other) {
        if (is_fixed() && other.is_fixed()) {
            stride_ /= other.stride_;
        } else {
            set_unknown();
        }
        return *this;
    }

    std::string str() const {
        std::ostringstream oss;
        if (is_fixed()) {
            oss << stride_;
        } else {
            oss << "(unknown)";
        }
        return oss.str();
    }

    static stride_t unknown() { return stride_t(stride_kind_t::unknown); }

private:
    stride_t(stride_kind_t kind, dim_t stride = 0)
        : kind_(kind), stride_(stride) {}

    void set_unknown() {
        kind_ = stride_kind_t::unknown;
        stride_ = 0;
    }

    stride_kind_t kind_ = stride_kind_t::undef;
    dim_t stride_ = 0;
};

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

struct block_t {
    block_t() = default;

    block_t(int dim_idx, dim_t block, const stride_t &stride)
        : dim_idx(dim_idx), block(block), stride(stride) {}

    bool is_equal(const block_t &other) const {
        return (dim_idx == other.dim_idx) && (block == other.block)
                && (stride == other.stride);
    }

    size_t get_hash() const {
        size_t hash = 0;
        hash = hash_combine(hash, dim_idx);
        hash = hash_combine(hash, block);
        hash = hash_combine(hash, stride.get_hash());
        return hash;
    }

    std::string str() const {
        std::ostringstream oss;
        oss << "block_t(dim_idx = " << dim_idx;
        oss << ", block = " << block;
        oss << ", stride = " << stride;
        oss << ")";
        return oss.str();
    }

    bool is_empty() const { return dim_idx == -1; }

    int dim_idx = -1; // Dimension index.
    dim_t block = 1; // Block size.
    stride_t stride; // Stride between elements of the block.
};

std::vector<block_t> normalize_blocks(
        const std::vector<block_t> &blocks, bool remove_size_1_blocks = true);

std::vector<block_t> compute_block_structure(const memory_desc_wrapper &mdw,
        bool inner_only = false, bool do_normalize = true);

} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
