/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#ifndef GPU_COMPUTE_BLOCK_MANIPULATION_HPP
#define GPU_COMPUTE_BLOCK_MANIPULATION_HPP

#include <cstddef>
#include <unordered_map>

#include "gpu/block_structure.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace compute {

class mapped_block_t {
public:
    mapped_block_t(size_t size, size_t dim_idx)
        : size(size), dim_idx(dim_idx) {}
    mapped_block_t(size_t buffer_idx, const block_t &block)
        : size(static_cast<size_t>(block.block))
        , dim_idx(static_cast<size_t>(block.dim_idx)) {
        map(buffer_idx, block);
    }

    void map(size_t buffer_idx, const block_t &block) {
        blocks[buffer_idx] = block;
    }

    bool matches(const block_t &block) const {
        size_t block_size = static_cast<size_t>(block.block);
        size_t block_dim_idx = static_cast<size_t>(block.dim_idx);
        return block_size == size && block_dim_idx == dim_idx;
    }

    const std::unordered_map<size_t, block_t> &get_buffer_blocks() const {
        return blocks;
    }

    bool is_broadcasted(size_t buffer_idx) const {
        return blocks.find(buffer_idx) == blocks.end();
    }

    size_t get_dim_idx() const { return dim_idx; }
    size_t get_size() const { return size; }

    bool can_merge(
            const mapped_block_t &other, bool require_all_match = true) const;

    std::string str() const {
        std::ostringstream ss;
        ss << "<";
        ss << size << "/" << dim_idx << ": ";
        for (const auto &it : blocks) {
            ss << it.first << " -> " << it.second.str() << ", ";
        }
        ss << ">";
        return ss.str();
    }

    // Mutates the existing object and returns another one
    mapped_block_t split(size_t first_size) {
        assert(size % first_size == 0);
        size_t size_remaining = size / first_size;
        size = first_size;

        mapped_block_t res(size_remaining, dim_idx);
        for (auto &it : blocks) {
            it.second.block = static_cast<dim_t>(size);
            block_t new_block(it.second.dim_idx,
                    static_cast<dim_t>(size_remaining),
                    it.second.stride * static_cast<dim_t>(first_size));
            res.map(it.first, new_block);
        }

        return res;
    }

private:
    size_t size;
    size_t dim_idx;
    std::unordered_map<size_t, block_t> blocks;
};

// represents a combined block used for indexing
class block_bin_t {
public:
    block_bin_t(const mapped_block_t &blocks, size_t num_layouts)
        : dim_idx(blocks.get_dim_idx()), num_layouts(num_layouts) {
        mapped_blocks.emplace_back(blocks);
        is_broadcasted_.resize(num_layouts);
        for (size_t i = 0; i < num_layouts; i++) {
            is_broadcasted_[i] = blocks.is_broadcasted(i);
        }
    }

    bool is_broadcasted(size_t buffer_idx) const {
        assert(buffer_idx < num_layouts);
        return is_broadcasted_[buffer_idx];
    }

    void append(const mapped_block_t &new_blocks) {
        mapped_blocks.emplace_back(new_blocks);
    }

    size_t size() const {
        size_t res = 1;
        for (const mapped_block_t &blocks : mapped_blocks) {
            res *= blocks.get_size();
        }
        return res;
    }

    std::string str() const;

    block_t combined_block(size_t buffer_idx) const {
        assert(!is_broadcasted_[buffer_idx]);
        block_t front_block
                = mapped_blocks.front().get_buffer_blocks().at(buffer_idx);
        dim_t dim = static_cast<dim_t>(dim_idx);
        dim_t block_size = static_cast<dim_t>(size());
        return block_t(dim, block_size, front_block.stride);
    }

    size_t get_dim_idx() const { return dim_idx; }
    const std::vector<mapped_block_t> &get_blocks() const {
        return mapped_blocks;
    }

private:
    size_t dim_idx;
    size_t num_layouts;
    std::vector<mapped_block_t> mapped_blocks;
    std::vector<bool> is_broadcasted_;
};

} // namespace compute
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
