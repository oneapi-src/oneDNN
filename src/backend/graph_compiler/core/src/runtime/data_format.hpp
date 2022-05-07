/*******************************************************************************
 * Copyright 2022 Intel Corporation
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

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_DATA_FORMAT_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_DATA_FORMAT_HPP

namespace sc {

namespace runtime {
// the compressed 64-bit data_format
union data_format {
    uint64_t storage_;
    struct {
        unsigned is_padding_ : 1;
        // the compressed encoding for 1st blocking; block_idx1=block1/16-1. It
        // is used for fast indexing on most frequently used blocking numbers
        unsigned block_idx1_ : 2;
        // the compressed encoding for 2nd blocking; block_idx2=block2/16-1. It
        // is used for fast indexing on most frequently used blocking numbers
        unsigned block_idx2_ : 2;
        // uncompressed 1st blocking number: 0-255. If compressed encoding is
        // used, it should be 0
        unsigned block1_ : 8;
        // uncompressed 2nd blocking number: 0-255. If compressed encoding is
        // used, it should be 0
        unsigned block2_ : 8;
        // format_kind, see sc_data_format_kind_t. 4 bits per axis and we can
        // encode at most 8 axises
        unsigned format_kind_ : 32;
        // unused bits to pad to 64 bits. Should be 0
        // unsigned unused_ : 11;
    };
    struct meta {
        static constexpr int MAX_DIMS = 8;
        static constexpr int LINEAR_INDEX_BITS = 5;
        static constexpr int BLOCKS_BIT_OFFSET = LINEAR_INDEX_BITS;
        static constexpr int BLOCKS_BITS = 16;
        static constexpr int BLOCKS_MASK = ((1UL << BLOCKS_BITS) - 1)
                << BLOCKS_BIT_OFFSET;
    };

    data_format() = default;
    constexpr data_format(uint64_t storage) : storage_(storage) {}
    data_format(unsigned format_kind, unsigned block1, unsigned block2,
            bool is_padding)
        : storage_ {0} {
        is_padding_ = is_padding;
        if (block1 % 16 == 0 && block2 % 16 == 0 && block1 <= 64
                && block2 <= 64) {
            block1_ = 0;
            block2_ = 0;
            block_idx1_ = block1 ? (block1 / 16 - 1) : 0;
            block_idx2_ = block2 ? (block2 / 16 - 1) : 0;
        } else {
            block1_ = block1;
            block2_ = block2;
            block_idx1_ = 0;
            block_idx2_ = 0;
        }
        format_kind_ = format_kind;
    }

    constexpr operator uint64_t() const { return storage_; }
    constexpr uint16_t get_block1() const {
        return (block1_ == 0) ? (block_idx1_ + 1) * 16 : block1_;
    }

    constexpr uint16_t get_block2() const {
        return (block2_ == 0) ? (block_idx2_ + 1) * 16 : block2_;
    }

    constexpr uint32_t get_linear_index() const { return storage_ & (0x1f); }
    constexpr bool is_blocks_uncompressed() const {
        return storage_ & meta::BLOCKS_MASK;
    }

    // converter for format_kind => 0~N index
    template <uint64_t format_kind, uint64_t... args>
    struct linear_converter {
        static constexpr int idx = 1 + linear_converter<args...>::idx;
        static uint64_t call(data_format v) {
            if (v.format_kind_ == uint32_t(format_kind)) { return idx; }
            return linear_converter<args...>::call(v);
        }
    };

    template <uint64_t format_kind>
    struct linear_converter<format_kind> {
        static constexpr int idx = 0;
        static uint64_t call(data_format v) {
            assert(v.format_kind_ == uint32_t(format_kind));
            return 0;
        }
    };
};

} // namespace runtime

} // namespace sc
#endif
