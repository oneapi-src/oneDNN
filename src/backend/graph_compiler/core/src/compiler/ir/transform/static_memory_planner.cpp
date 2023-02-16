/*******************************************************************************
 * Copyright 2020-2023 Intel Corporation
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
#include "static_memory_planner.hpp"
#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <util/utils.hpp>

SC_MODULE(pass.mem_plan)

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace memory_optim {

// how the buffer was created
enum class chunk_type {
    ORIGIN, // the chunk is directly allocated from the large buffer
    SPLIT, // the chunk is got by splitting another memory chunk
    MERGED, // the chunk is got by merging several consecutive memory chunks
};

struct memory_state;

struct memory_chunk_t {
    chunk_type type_;
    size_t size_;
    bool free_ = true;
    size_t last_freed_tick_ = 0;
    bool is_inplace_split_remainder_ = false;
    // splits the chunk and get the left hand side with size = size, registers
    // both the returned chunk and the rest of the chunk to the state
    void split(memory_state *state, size_t size, memory_chunk_t *&lhs,
            memory_chunk_t *&rhs);
    // move the buffer, propagate the message up to the parent chunk. It will
    // not update the siblings.
    virtual void move(int64_t start_diff) = 0;
    // extend the buffer, propagate the message up to the parent chunk. It will
    // not update the siblings.
    virtual void extend(int64_t size_diff) = 0;

    memory_chunk_t(chunk_type type, size_t size) : type_(type), size_(size) {}
    // there should be no updates to memory chunks after calling
    // get_start_offset
    size_t get_start_offset() {
        if (cached_start_offset == UNINITIALIZED) {
            cached_start_offset = get_start_offset_impl();
        }
        return cached_start_offset;
    }
    virtual ~memory_chunk_t() = default;

    virtual size_t get_start_offset_impl() = 0;

protected:
    static constexpr size_t UNINITIALIZED = std::numeric_limits<size_t>::max();
    size_t cached_start_offset = UNINITIALIZED;
};

// the memory chunk that is directly allocated from the large buffer
struct origin_chunk_t : public memory_chunk_t {
    // no parent
    // memory_chunk_t *parent_;
    size_t start_;
    origin_chunk_t(size_t start, size_t size)
        : memory_chunk_t {chunk_type::ORIGIN, size}, start_(start) {}
    void move(int64_t start_diff) override { start_ += start_diff; }
    void extend(int64_t size_diff) override { size_ += size_diff; }
    size_t get_start_offset_impl() override { return start_; }
};

// the memory chunk that is split from another chunk
struct split_chunk_t : public memory_chunk_t {
    memory_chunk_t *parent_;
    // if the chunk is the left hand side (smaller starting offset)
    bool is_lhs_;
    split_chunk_t(size_t size, memory_chunk_t *parent, bool is_lhs)
        : memory_chunk_t {chunk_type::SPLIT, size}
        , parent_(parent)
        , is_lhs_(is_lhs) {}
    void move(int64_t start_diff) override {
        if (is_lhs_) { parent_->move(start_diff); }
        // no need to pass message to parent for rhs, since lhs has done so
    }
    void extend(int64_t size_diff) override {
        size_ += size_diff;
        parent_->extend(size_diff);
        // if is_lhs, we will later call rhs->move(...)
    }
    size_t get_start_offset_impl() override {
        if (is_lhs_) {
            return parent_->get_start_offset();
        } else {
            return parent_->get_start_offset() + parent_->size_ - size_;
        }
    }
};

static size_t get_size_of_chunks(const std::vector<memory_chunk_t *> &c) {
    size_t v = 0;
    for (auto chk : c) {
        v += chk->size_;
    }
    return v;
}
// the memory chunk that is merged from another chunks
struct merged_chunk_t : public memory_chunk_t {
    std::vector<memory_chunk_t *> parent_;
    merged_chunk_t(std::vector<memory_chunk_t *> &&parent)
        : memory_chunk_t {chunk_type::MERGED, get_size_of_chunks(parent)}
        , parent_(std::move(parent)) {}
    void move(int64_t start_diff) override {
        for (auto v : parent_) {
            v->move(start_diff);
        }
    }
    void extend(int64_t size_diff) override {
        size_ += size_diff;
        parent_.back()->extend(size_diff);
    }
    size_t get_start_offset_impl() override {
        return parent_.front()->get_start_offset();
    }
};

struct memory_state {
    // buffer_id -> allocated memory chunk, used to collect the final result
    std::unordered_map<uintptr_t, memory_chunk_t *> allocations_;
    // buffer_id -> <current_alive_memory_chunk>, used when inplace
    // optimization, and when a buffer is inplace reused by another buffer. The
    // reused buffer will have unchanged memory_chunk_t in allocations_, because
    // allocations_ shows the final result of the buffer. cur_allocations_
    // tracks the current mapping of buffer_id to memory_chunk_t, which may be
    // different from allocations_
    std::unordered_map<uintptr_t, memory_chunk_t *> cur_allocations_;
    // all memory chunks that has been created, takes the ownerships of the
    // memory_chunk_t objects
    std::vector<std::unique_ptr<memory_chunk_t>> chunks_;
    // the current memory chunks, sorted by the starting offset
    std::vector<memory_chunk_t *> cur_chunks_;
    // free chunks sorted by size
    std::multimap<size_t, memory_chunk_t *> free_chunks_by_size_;
    // free chunks sorted by last freed tick
    std::multimap<size_t, memory_chunk_t *> free_chunks_by_tick_;
    // the current size of the large buffer, in number of elements
    size_t current_alloc_size_ = 0;
    // the alignment in number of elements
    size_t alignment_;
    // the map from a buffer-id to the buffer-ids that the buffer can inplace
    // reuse
    const inplace_info_map &inplace_map_;
    std::unordered_map<uintptr_t, std::vector<uintptr_t>>
            &out_inplace_selection_;
    int tick_ = 0;
    bool hot_first_;

    memory_state(size_t alignment, bool hot_first,
            const inplace_info_map &inplace_map,
            std::unordered_map<uintptr_t, std::vector<uintptr_t>>
                    &out_inplace_selection)
        : alignment_(alignment)
        , inplace_map_(inplace_map)
        , out_inplace_selection_(out_inplace_selection)
        , hot_first_(hot_first) {}

    void remove_chunk_from_map(memory_chunk_t *target, size_t t,
            std::multimap<size_t, memory_chunk_t *> &m) {
        auto mapitr = m.equal_range(t);
        assert(mapitr.first != mapitr.second);
        for (auto map_i = mapitr.first; map_i != mapitr.second; ++map_i) {
            if (map_i->second == target) {
                m.erase(map_i);
                break;
            }
        }
    }
    void remove_chunk_from_free_list(memory_chunk_t *target) {
        remove_chunk_from_map(target, target->size_, free_chunks_by_size_);
        remove_chunk_from_map(
                target, target->last_freed_tick_, free_chunks_by_tick_);
        target->free_ = false;
    }

    void add_chunk_to_free_list(memory_chunk_t *target) {
        free_chunks_by_size_.insert(std::make_pair(target->size_, target));
        free_chunks_by_tick_.insert(
                std::make_pair(target->last_freed_tick_, target));
        target->free_ = true;
    }

    void extend_alloc(memory_chunk_t *target, size_t aligned) {
        // remove the chunk from free list
        remove_chunk_from_free_list(target);
        int64_t size_diff = aligned - target->size_;
        assert(size_diff > 0);
        current_alloc_size_ += size_diff;
        // extend the target chunk, also move all buffers at the right of it
        target->extend(size_diff);
        bool found_target = false;
        for (auto v : cur_chunks_) {
            if (v == target) {
                found_target = true;
            } else if (found_target) {
                // v is at the right of the target
                v->move(size_diff);
            }
        }
        assert(found_target);
        target->free_ = false;
    }

    memory_chunk_t *split_alloc(
            memory_chunk_t *target, size_t aligned, memory_chunk_t *&rhs_ret) {
        // found a free chunk that is large enough
        if (target->size_ == aligned) {
            // a perfect match, no need to split
            auto ret = target;
            remove_chunk_from_free_list(target);
            return ret;
        }
        // split the larger chunk
        assert(target->size_ > aligned);
        auto lhs = utils::make_unique<split_chunk_t>(aligned, target, true);
        auto rhs = utils::make_unique<split_chunk_t>(
                target->size_ - aligned, target, false);
        rhs_ret = rhs.get();
        auto ret = lhs.get();

        auto old_itr_in_cur_chunks
                = std::find(cur_chunks_.begin(), cur_chunks_.end(), target);
        assert(old_itr_in_cur_chunks != cur_chunks_.end());
        // replace old chunk with rhs
        *old_itr_in_cur_chunks = rhs.get();
        // insert lhs before rhs
        cur_chunks_.insert(old_itr_in_cur_chunks, lhs.get());
        rhs->last_freed_tick_ = target->last_freed_tick_;
        // add rhs to free list
        add_chunk_to_free_list(rhs.get());

        // move ownership
        chunks_.emplace_back(std::move(lhs));
        chunks_.emplace_back(std::move(rhs));

        // remove old chunk in free list
        remove_chunk_from_free_list(target);
        ret->free_ = false;
        return ret;
    }

    float calculate_size_score(size_t chk_size, size_t alloc_size) const {
        // size_score = abs(chunk_size-alloc_size)/max(chunk_size, alloc_size)
        int64_t size_diff = static_cast<int64_t>(chk_size)
                - static_cast<int64_t>(alloc_size);
        float size_max = static_cast<float>(std::max(alloc_size, chk_size));
        float size_score = -std::abs(size_diff) / size_max;
        // if we don't need to extend the buffer, add a bounus score for it
        if (alloc_size <= chk_size) { size_score += 1; }
        // size_score and tick_score are normalized in [-1,1]. We set a weight
        // for these two scores: 1:1
        return size_score;
    }

    // calculates the score of a free chunk to help select the best chunk we
    // allocate memory from. It considers 2 factors: 1) the free chunk size and
    // the size of the current memory allocation request. The closer they are,
    // the better the chunk is. 2) the heat of the chunk. If the chunk's last
    // free'd tick is closer to the current tick, the chunk is better.
    // The better the chunk is, the greater the score is
    float calculate_chunk_score(
            memory_chunk_t *chk, size_t alloc_size, size_t last_tick) const {
        // if the buffer is free'd N ticks ago, it will have score max(0, 1 - N
        // * 0.1)
        float tick_score = static_cast<float>(tick_ - last_tick) / 10;
        tick_score = 1 - std::min(tick_score, 1.0f);
        // size_score and tick_score are normalized in [-1,1]. We set a weight
        // for these two scores: 1:1
        return 1 * calculate_size_score(chk->size_, alloc_size)
                + 1 * tick_score;
    }

    memory_chunk_t *alloc(uintptr_t bufferid, size_t size) {
        tick_++;
        auto ret = do_alloc(bufferid, size);
        allocations_[bufferid] = ret;
        cur_allocations_[bufferid] = ret;
        return ret;
    }

    // check if the buffer is split from a base tensor and check the
    // inplace_info for whether it requires zero offset
    bool check_buffer_offset_for_inplace(
            memory_chunk_t *chunk, const inplace_info *info) {
        // if the old memory chunk is splitted from the base tensor
        bool old_is_split = chunk->is_inplace_split_remainder_;
        // if the old memory chunk is based on a offset of the base tensor
        // and we require that we should use zero offset on that tensor, we
        // cannot reuse it
        if (old_is_split && info->second == inplace_kind::ZERO_OFFSET) {
            return false;
        }
        return true;
    }

    // find the range of chunks in cur_chunks_ that can be merged for inplace
    // reuse, returns the memory size of the range and the start/end iterators
    size_t find_inplace_merge_range(memory_chunk_t *victim, size_t aligned,
            const std::unordered_map<memory_chunk_t *, const inplace_info *>
                    &can_inplace,
            std::vector<memory_chunk_t *>::iterator &to_merge_start,
            std::vector<memory_chunk_t *>::iterator &to_merge_end) {
        // add_chunk_to_free_list(chk);
        auto itr_in_cur_chunks
                = std::find(cur_chunks_.begin(), cur_chunks_.end(), victim);
        assert(itr_in_cur_chunks != cur_chunks_.end());
        // merge right if they are free or can be inplaced
        to_merge_start = itr_in_cur_chunks;
        to_merge_end = itr_in_cur_chunks + 1;
        // remember the memory size we already collected. If
        // current_collected_size is greater than the memory size to alloc, we
        // can stop searching
        size_t current_collected_size = victim->size_;
        // look right to see any one we can merge with
        for (auto itr = itr_in_cur_chunks + 1;
                itr != cur_chunks_.end() && current_collected_size < aligned;
                ++itr) {
            // if the memory chunk is in use and is in can_inplace map, we may
            // reuse it now
            auto inplace_info_itr = can_inplace.find(*itr);
            if ((*itr)->free_
                    || (inplace_info_itr != can_inplace.end()
                            && inplace_info_itr->second->second
                                    == inplace_kind::FREE)) {
                to_merge_end = itr + 1;
                current_collected_size += (*itr)->size_;
            } else {
                break;
            }
        }
        return current_collected_size;
    }

    // inplace alloc memory on a chunk that is in use, but about to be freed.
    memory_chunk_t *do_inplace_alloc(uintptr_t bufferid, size_t aligned) {
        if (inplace_map_.empty()) { return nullptr; }
        auto itr_inplace = inplace_map_.find(bufferid);
        if (itr_inplace == inplace_map_.end()) { return nullptr; }
        // if the buffer can inplace reuse some other buffers that is
        // still in use but about to be freed
        const auto &buffer_can_inplace = itr_inplace->second;
        if (buffer_can_inplace.empty()) { return nullptr; }

        // reversed map, chunk --> buffer id for inplace candidates
        std::unordered_map<memory_chunk_t *, const inplace_info *> can_inplace;
        for (auto &v : buffer_can_inplace) {
            auto itr = cur_allocations_.find(v.first);
            if (itr != cur_allocations_.end()) {
                can_inplace[itr->second] = &v;
            }
        }

        // stage 1, find a victim based on the memory size that can be freed
        float target_score = -std::numeric_limits<float>::infinity();
        memory_chunk_t *victim = nullptr;
        std::vector<memory_chunk_t *>::iterator to_merge_start;
        std::vector<memory_chunk_t *>::iterator to_merge_end;
        size_t current_collected_size = 0;
        for (auto &bufinfo : buffer_can_inplace) {
            auto buf_id = bufinfo.first;
            auto old_buf_itr = cur_allocations_.find(buf_id);
            // if the buffer has already been reused by other buffers, skip
            if (old_buf_itr == cur_allocations_.end()) { continue; }
            // the old memory chunk
            auto old_buf = old_buf_itr->second;

            auto &old_inplace_info = can_inplace[old_buf];
            if (!check_buffer_offset_for_inplace(old_buf, old_inplace_info)) {
                continue;
            }

            std::vector<memory_chunk_t *>::iterator cur_merge_start;
            std::vector<memory_chunk_t *>::iterator cur_merge_end;
            auto cur_size = find_inplace_merge_range(old_buf, aligned,
                    can_inplace, cur_merge_start, cur_merge_end);
            float score = calculate_size_score(cur_size, aligned);
            if (score > target_score) {
                target_score = score;
                victim = old_buf;
                to_merge_start = cur_merge_start;
                to_merge_end = cur_merge_end;
                current_collected_size = cur_size;
            }
        }
        if (current_collected_size * 10 < aligned) {
            // if the memory can be reused is too small (less than 10% of the
            // target size), inplacing has no benifits, skip
            return nullptr;
        }
        if (!victim) { return nullptr; }
        assert(!victim->free_);

        victim->last_freed_tick_ = tick_;

        std::vector<memory_chunk_t *> merged_buffers(
                to_merge_start, to_merge_end);
        for (auto buf : merged_buffers) {
            auto itr = can_inplace.find(buf);
            if (itr != can_inplace.end()) {
                uintptr_t vic_buffer_id = itr->second->first;
                if (vic_buffer_id) {
                    out_inplace_selection_[bufferid].emplace_back(
                            vic_buffer_id);
                    SC_MODULE_INFO << "Buffer " << bufferid
                                   << " inplace reuses " << vic_buffer_id;
                }
            }
        }
        if (current_collected_size < aligned) {
            // if the collected memory size is still less than the size to
            // alloc, need to extend
            auto target_size = aligned - current_collected_size
                    + merged_buffers.back()->size_;
            if (!merged_buffers.back()->free_) {
                // if it is not free, we are inplacing it. Temporarily move to
                // free list
                add_chunk_to_free_list(merged_buffers.back());
            }
            extend_alloc(merged_buffers.back(), target_size);
            // after extension of the last buffer, the collected size is equal
            // to the size to alloc
            current_collected_size = aligned;
        }

        // remove from freelist and buffer_id->chunk map
        for (auto itr = to_merge_start; itr != to_merge_end; ++itr) {
            auto chunk = *itr;
            if (chunk->free_) { remove_chunk_from_free_list(chunk); }
            auto itr_chunk = can_inplace.find(chunk);
            if (itr_chunk != can_inplace.end()) {
                cur_allocations_.erase(itr_chunk->second->first);
            }
        }

        memory_chunk_t *merged_chunk;
        // if we need to merge multiple chunks
        if (to_merge_end - to_merge_start > 1) {
            // do merge
            chunks_.emplace_back(utils::make_unique<merged_chunk_t>(
                    std::vector<memory_chunk_t *>(merged_buffers)));
            merged_chunk = chunks_.back().get();
            // remove merged chunks from free list and cur_chunk list
            // add merged chunk to cur_chunks and free_chunks_by_size
            *to_merge_start = merged_chunk;
            merged_chunk->last_freed_tick_ = tick_;
            merged_chunk->free_ = false;
            cur_chunks_.erase(to_merge_start + 1, to_merge_end);
        } else {
            merged_chunk = victim;
            merged_chunk->last_freed_tick_ = tick_;
        }

        // merged_chunk is in cur_chunks_ and is removed from freelist and
        // cur_allocations_ map
        if (current_collected_size == aligned) {
            // if is extended, or perfect match, just return the chunk
            merged_chunk->free_ = false;
            return merged_chunk;
        } else {
            // otherwise, there are some unused memory in the chunk to be
            // reused. We need to split it. If the RHS of the chunk is from a
            // inplace reused buffer, need to add a mapping of the buffer id to
            // the RHS remaining chunk
            if (!merged_chunk->free_) { add_chunk_to_free_list(merged_chunk); }
            memory_chunk_t *rhs = nullptr;
            auto ret = split_alloc(merged_chunk, aligned, rhs);
            auto itr_chunk = can_inplace.find(merged_buffers.back());
            if (itr_chunk != can_inplace.end()) {
                // if the last chunk is from inplace map, the RHS chunk is not
                // really freed, need to remove from free list and mark it not
                // freed.
                remove_chunk_from_free_list(rhs);
                rhs->is_inplace_split_remainder_ = true;
                // update the buffer id -> chunk map, so that when freeing the
                // inplaced buffer, we can find the correct remaining buffer
                cur_allocations_[itr_chunk->second->first] = rhs;
            }
            return ret;
        }
    }

    memory_chunk_t *do_alloc(uintptr_t bufferid, size_t size) {
        auto aligned = utils::divide_and_ceil(size, alignment_) * alignment_;
        // try inplace
        if (auto inp_ret = do_inplace_alloc(bufferid, size)) { return inp_ret; }
        if (free_chunks_by_size_.empty()) {
            chunks_.emplace_back(utils::make_unique<origin_chunk_t>(
                    current_alloc_size_, aligned));
            current_alloc_size_ += aligned;
            auto ret = chunks_.back().get();
            cur_chunks_.emplace_back(ret);
            ret->free_ = false;
            return ret;
        }
        if (hot_first_) {
            memory_chunk_t *target = free_chunks_by_tick_.rbegin()->second;
            float target_score = calculate_chunk_score(
                    target, aligned, free_chunks_by_tick_.rbegin()->first);
            for (auto &kv : free_chunks_by_tick_) {
                float score
                        = calculate_chunk_score(kv.second, aligned, kv.first);
                if (score > target_score) {
                    target = kv.second;
                    target_score = score;
                }
            }
            if (target->size_ < aligned) {
                extend_alloc(target, aligned);
                return target;
            } else {
                memory_chunk_t *rhs;
                return split_alloc(target, aligned, rhs);
            }
        } else {
            // find a free chunk that best fits the current size
            // itr will be the smallest chunk whose size >= aligned
            auto itr = free_chunks_by_size_.lower_bound(aligned);
            if (itr == free_chunks_by_size_.end()) {
                memory_chunk_t *target;
                // itr points to the last element
                --itr;
                // if not found, this means that all free chunk is smaller than
                // aligned size, switch to the largest chunk
                target = itr->second;
                extend_alloc(target, aligned);
                return target;
            } else {
                memory_chunk_t *rhs;
                return split_alloc(itr->second, aligned, rhs);
            }
        }
    }

    void dealloc(memory_chunk_t *chk) {
        tick_++;
        chk->last_freed_tick_ = tick_;
        add_chunk_to_free_list(chk);
        auto itr_in_cur_chunks
                = std::find(cur_chunks_.begin(), cur_chunks_.end(), chk);
        assert(itr_in_cur_chunks != cur_chunks_.end());
        // merge left and right if they are free
        std::vector<memory_chunk_t *>::iterator to_merge_start
                = itr_in_cur_chunks;
        std::vector<memory_chunk_t *>::iterator to_merge_end
                = itr_in_cur_chunks + 1;
        // look left to see any one we can merge with
        for (auto itr = itr_in_cur_chunks;; --itr) {
            if ((*itr)->free_) {
                to_merge_start = itr;
            } else {
                break;
            }
            if (itr == cur_chunks_.begin()) { break; }
        }
        // look right to see any one we can merge with
        for (auto itr = itr_in_cur_chunks + 1; itr != cur_chunks_.end();
                ++itr) {
            if ((*itr)->free_) {
                to_merge_end = itr + 1;
            } else {
                break;
            }
        }
        if (to_merge_end - to_merge_start > 1) {
            // do merge
            chunks_.emplace_back(utils::make_unique<merged_chunk_t>(
                    std::vector<memory_chunk_t *>(
                            to_merge_start, to_merge_end)));

            // remove merged chunks from free list and cur_chunk list
            for (auto itr = to_merge_start; itr != to_merge_end; ++itr) {
                auto chunk = *itr;
                remove_chunk_from_free_list(chunk);
            }
            // add merged chunk to cur_chunks and free_chunks_by_size
            *to_merge_start = chunks_.back().get();
            chunks_.back()->last_freed_tick_ = tick_;
            add_chunk_to_free_list(chunks_.back().get());
            cur_chunks_.erase(to_merge_start + 1, to_merge_end);
        }
        // else, no chunks are merged, do nothing
    }

    void dealloc(uintptr_t bufferid) {
        auto alocitr = allocations_.find(bufferid);
        COMPILE_ASSERT(alocitr != allocations_.end(),
                "Cannot find buffer id in allocations");
        auto itr = cur_allocations_.find(bufferid);
        if (itr != cur_allocations_.end()) {
            itr->second->is_inplace_split_remainder_ = false;
            dealloc(itr->second);
            cur_allocations_.erase(itr);
        }
    }

    std::string to_string() const {
        std::stringstream ss;
        ss << "total size " << current_alloc_size_ << " ";
        size_t cur_offset = 0;
        for (auto buf : cur_chunks_) {
            ss << "| " << cur_offset << ',' << buf->size_ << ',' << buf->free_
               << " ";
            cur_offset += buf->size_;
        }
        return ss.str();
    }
};

size_t schedule_memory_allocations(
        const std::vector<memory_alloc_trace_t> &traces, size_t alignment,
        bool hot_first, const inplace_info_map &inplace_map,
        std::unordered_map<uintptr_t, size_t> &out_schedule,
        std::unordered_map<uintptr_t, std::vector<uintptr_t>>
                &out_inplace_selection) {
    memory_state planner {
            alignment, hot_first, inplace_map, out_inplace_selection};
    SC_MODULE_INFO << "Start of a function";
    for (auto &trace : traces) {
        if (trace.size_ > 0) {
            planner.alloc(trace.buffer_id_, trace.size_);
            SC_MODULE_INFO << "Alloc " << trace.buffer_id_
                           << ", sz=" << trace.size_;
            SC_MODULE_INFO << planner.to_string();
        } else {
            planner.dealloc(trace.buffer_id_);
            SC_MODULE_INFO << "Dealloc " << trace.buffer_id_;
            SC_MODULE_INFO << planner.to_string();
        }
    }
    for (auto &kv : planner.allocations_) {
        out_schedule[kv.first] = kv.second->get_start_offset();
    }
    return planner.current_alloc_size_;
}
} // namespace memory_optim
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
