/*******************************************************************************
 * Copyright 2022-2023 Intel Corporation
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
#include <assert.h>
#include <stdint.h>
#include <util/hash_utils.hpp>

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_DISPATCH_KEY_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_DISPATCH_KEY_HPP

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

namespace runtime {
// the compressed 64-bit dispatch_key
union dispatch_key {
    uint64_t storage_;
    struct {
        // the compressed encoding for 1st blocking; block_idx1=block1/16-1. It
        // is used for fast indexing on most frequently used blocking numbers
        unsigned block_idx1_ : 2;
        // the compressed encoding for 2nd blocking; block_idx2=block2/16-1. It
        // is used for fast indexing on most frequently used blocking numbers
        unsigned block_idx2_ : 2;
        // the compressed encoding for op implement algorithm type. There are 16
        // reserved algorithms for select.
        unsigned impl_alg_ : 11;
        // uncompressed 1st blocking number: 0-255. If compressed encoding is
        // used, it should be 0
        unsigned block1_ : 8;
        // uncompressed 2nd blocking number: 0-255. If compressed encoding is
        // used, it should be 0
        unsigned block2_ : 8;
        // the format kind is plain or not.
        unsigned is_plain_ : 1;
        // format_kind, see sc_data_format_kind_t. 4 bits per axis and we can
        // encode at most 8 axises
        unsigned format_kind_ : 32;
        // unused bits to pad to 64 bits. Should be 0
        // unsigned unused_ : 0;
    };
    struct meta {
        static constexpr int MAX_DIMS = 8;
        static constexpr int IMPL_ALG_BITS = 11;
        static constexpr int LINEAR_INDEX_BITS = 4 + IMPL_ALG_BITS;
        static constexpr int BLOCKS_BIT_OFFSET = LINEAR_INDEX_BITS;
        static constexpr int BLOCKS_BITS = 16;
        static constexpr int BLOCKS_MASK = ((1UL << BLOCKS_BITS) - 1)
                << BLOCKS_BIT_OFFSET;
        static constexpr int FORMAT_BITS_OFFSET = 32;
        static constexpr int FORMAT_BITS = 32;
        static constexpr uint64_t FORMAT_MASK
                = ((static_cast<uint64_t>(1) << FORMAT_BITS) - 1)
                << FORMAT_BITS_OFFSET;
        static constexpr int PLAIN_BIT_OFFSET = LINEAR_INDEX_BITS + BLOCKS_BITS;
        static constexpr uint64_t PLAIN_MASK = 1UL << PLAIN_BIT_OFFSET;
        static constexpr int BITS_PER_SLOT = 4;
    };

    dispatch_key() = default;
    constexpr dispatch_key(uint64_t storage) : storage_(storage) {}
    dispatch_key(unsigned format_kind, unsigned block1, unsigned block2,
            unsigned impl_alg, bool is_plain = false)
        : storage_ {0} {
        impl_alg_ = impl_alg;
        is_plain_ = is_plain;
        if (block1 % 16 == 0 && block1 <= 64) {
            block1_ = 0;
            block_idx1_ = block1 ? (block1 / 16 - 1) : 0;
        } else {
            block1_ = block1;
            block_idx1_ = 0;
        }
        if (block2 % 16 == 0 && block2 <= 64) {
            block2_ = 0;
            block_idx2_ = block2 ? (block2 / 16 - 1) : 0;
        } else {
            block2_ = block2;
            block_idx2_ = 0;
        }
        format_kind_ = format_kind;
    }

    constexpr int get(int idx) const {
        return 0xf & (get_format_bits() >> (idx * meta::BITS_PER_SLOT));
    }

    void set(int idx, int axis) {
        format_kind_ = (format_kind_ & ~(0xF << (idx * meta::BITS_PER_SLOT)))
                | (axis << (idx * meta::BITS_PER_SLOT));
    }

    int ndims() const {
        int idx = 0;
        while (idx < meta::MAX_DIMS && get(idx) != 0xF) {
            idx++;
        }
        return idx;
    }

    constexpr operator uint64_t() const { return storage_; }
    constexpr uint16_t get_block1() const {
        return (block1_ == 0) ? (block_idx1_ + 1) * 16 : block1_;
    }

    constexpr uint16_t get_block2() const {
        return (block2_ == 0) ? (block_idx2_ + 1) * 16 : block2_;
    }

    constexpr uint32_t get_linear_index() const { return storage_ & (0xff); }
    constexpr uint32_t get_impl_alg_type() const { return impl_alg_; }

    constexpr uint32_t get_format_bits() const {
        return (storage_ & meta::FORMAT_MASK) >> meta::FORMAT_BITS_OFFSET;
    }

    void set_block1(uint16_t block) {
        if (block % 16 == 0 && block <= 64) {
            block_idx1_ = block / 16 - 1;
            block1_ = 0;
        } else {
            block1_ = block;
            block_idx1_ = 0;
        }
    }
    void set_block2(uint16_t block) {
        if (block % 16 == 0 && block <= 64) {
            block_idx2_ = block / 16 - 1;
            block2_ = 0;
        } else {
            block2_ = block;
            block_idx2_ = 0;
        }
    }
    void set_impl_alg(unsigned impl_alg) { impl_alg_ = impl_alg; }
    void reset_blocks_and_impl() {
        impl_alg_ = 0;
        if (is_plain_) {
            block_idx1_ = 0;
            block_idx2_ = 0;
            block1_ = 0;
            block2_ = 0;
        }
    }
    constexpr bool is_blocks_uncompressed() const {
        return storage_ & meta::BLOCKS_MASK;
    }

    constexpr bool is_plain() const { return storage_ & meta::PLAIN_MASK; }

    // converter for format_kind => 0~N index
    template <uint64_t format_kind, uint64_t... args>
    struct linear_converter {
        static constexpr int idx = 1 + linear_converter<args...>::idx;
        static uint64_t call(dispatch_key v) {
            if (v.format_kind_ == uint32_t(format_kind)) { return idx; }
            return linear_converter<args...>::call(v);
        }
    };

    template <uint64_t format_kind>
    struct linear_converter<format_kind> {
        static constexpr int idx = 0;
        static uint64_t call(dispatch_key v) {
            assert(v.format_kind_ == uint32_t(format_kind));
            return 0;
        }
    };
};

} // namespace runtime
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
namespace std {
template <>
struct hash<::dnnl::impl::graph::gc::runtime::dispatch_key> {
    std::size_t operator()(
            const ::dnnl::impl::graph::gc::runtime::dispatch_key &in) const {
        return std::hash<uint64_t>()(uint64_t(in));
    }
};
} // namespace std

#endif
