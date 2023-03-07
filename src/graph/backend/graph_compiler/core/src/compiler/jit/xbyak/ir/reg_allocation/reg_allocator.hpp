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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_IR_REG_ALLOCATION_REG_ALLOCATOR_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_IR_REG_ALLOCATION_REG_ALLOCATOR_HPP

#include <deque>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <vector>

#include "virtual_slot.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace xbyak {

/* *
 * vrtual_reg dequeue priority:
 * [spill weight higher] -> [defined earlier] -> [ended earlier]
 * */
struct spill_weight_comparator_t {
    bool operator()(const virtual_reg_t *lhs, const virtual_reg_t *rhs) {
        assert(lhs != nullptr && rhs != nullptr);
        bool def_later = lhs->live_range_.start_ == rhs->live_range_.start_
                ? lhs->live_range_.end_ > rhs->live_range_.end_
                : lhs->live_range_.start_ > rhs->live_range_.start_;
        bool weight_equal = lhs->spill_weight_ == rhs->spill_weight_;
        bool weight_less = lhs->spill_weight_ < rhs->spill_weight_;
        return (weight_equal && def_later) || (!weight_equal && weight_less);
    }
};

/* *
 * Priority based register allocator.
 * All unassigned virtual regs need to be enqueued.
 * Dequeue each virtual reg based on priority, allocator will try to assign a
 * virtual slot for each virtual reg.
 * If no slot available, virtual reg will be spilled and if spill resolver find
 * the spilled operand is conflicted with intrinsics' instruction format,
 * conflicts will be resolved and small load/store intervals with infinite spill
 * weight will be created and put back into the priority queue.
 * */
class reg_allocator_t {
public:
    reg_allocator_t(const x86_64::target_profile_t &profile)
        : target_profile_(profile), require_resolve_(false) {
        // Win64 ABI considers registers XMM6-XMM15 nonvolatile, but the
        // upper portions of YMM0-YMM15 and ZMM0-ZMM15 still volatile.
        // This will introduce unnecessary design complexity, so we treat
        // all of them as volatile in xbyak generated callers.
        virtual_slots_map_
                = std::make_shared<virtual_slots_map_t>(target_profile_, true);
        virtual_slots_array_ = std::make_shared<virtual_slots_array_t>(
                virtual_slots_map_->get_slots_sum());
    }
    virtual ~reg_allocator_t() = default;

    bool queue_empty() { return virtual_reg_queue_.empty(); }

    void enqueue(virtual_reg_t *virt_reg) { virtual_reg_queue_.push(virt_reg); }

    virtual_reg_t *dequeue() {
        if (queue_empty()) { return nullptr; }
        virtual_reg_t *top = virtual_reg_queue_.top();
        virtual_reg_queue_.pop();
        return top;
    }

    // Allocation routine
    void run_allocator() {
        // Go through queue and allocate every unassigned virtual_regs
        while (!queue_empty()) {
            // Get the virtual_reg on front of queue
            auto virt_reg = dequeue();
            // Try to allocate virtual_reg, if interference exists
            std::set<virtual_reg_t *> evicted;
            auto index = try_assign(virt_reg, evicted);
            // Evict confilct virt_regs with less spill weight
            for (auto &vr : evicted) {
                unassign(vr);
            }
            // Assign or spill virtual reg
            if (index == virt_reg_const::invalid) {
                spill(virt_reg);
            } else {
                allocate(virt_reg, index);
            }
            // Resolve address mode and get new virtual_regs created
            if (require_resolve()) { resolve_spill(virt_reg->live_range_); }
        }
    }

    // Heuristics for determining virtual_reg assign/spill
    virt_reg_index_t try_assign(
            virtual_reg_t *virt_reg, std::set<virtual_reg_t *> &evicted) {
        // Initial value for interference check
        virt_reg_index_t index = virt_reg_const::invalid;
        spill_weight_t weight = spill_weight_const::infinity;

        // Check interference for hint reg
        auto check_interference_hint = [&](bool preserved) {
            auto &callee_save
                    = slots_map().get_callee_save_set(virt_reg->type_);
            index = virt_reg->index_hint_;
            bool check = preserved
                    ? callee_save.find(index) != callee_save.end()
                    : true;
            if (check) {
                weight = slots_array().interfered_weights(virt_reg, index);
            }
        };
        // Check interference for all regs
        auto check_interference_all = [&](bool preserved) {
            const auto &candidates = preserved
                    ? slots_map().get_callee_save(virt_reg->type_)
                    : slots_map().get_slots_index(virt_reg->type_);
            for (const auto &i : candidates) {
                auto w = slots_array().interfered_weights(virt_reg, i);
                if (w <= weight) {
                    index = i;
                    weight = w;
                }
                if (weight == spill_weight_const::null) { break; }
            }
        };

        // Check interference for different types of virtual reg
        switch (virt_reg->hint_) {
            case virt_reg_hint::strong: {
                check_interference_hint(virt_reg->preserved_);
            } break;
            case virt_reg_hint::weak: {
                check_interference_hint(virt_reg->preserved_);
                if (weight == spill_weight_const::null) { break; }
                check_interference_all(virt_reg->preserved_);
            } break;
            case virt_reg_hint::none: {
                check_interference_all(virt_reg->preserved_);
            } break;
        }

        // Final result
        if (weight == spill_weight_const::null) {
            // If slot aviliable
            return index;
        } else {
            if (virt_reg->spill_weight_ <= weight) {
                // If slot unaviliable
                assert(virt_reg->spill_weight_ < spill_weight_const::infinity);
                return virt_reg_const::invalid;
            } else {
                // If interfered virtual regs on slot have less spill weight
                // Evict interfered and assign current virtual reg instead
                assert(index != virt_reg_const::invalid);
                evicted = slots_array().interfered_regs(virt_reg, index);
                return index;
            }
        }
    }

    // Check instruction format and create new load/store interval if needed
    void resolve_spill(const live_range_t &spill_range) {
        std::vector<virtual_reg_t *> virtual_regs;
        resolve_spill_impl(spill_range, virtual_regs);
        for (auto &vr : virtual_regs) {
            enqueue(vr);
        }
        require_resolve_ = false;
    }

    // Unassign virtual_reg and put back to the queue
    void unassign(virtual_reg_t *virt_reg) {
        assert(virt_reg);
        switch (virt_reg->stat_) {
            case virt_reg_stat::spilled: {
                virt_reg->set_unassigned();
            } break;
            case virt_reg_stat::allocated: {
                slots_array().unassign_slot(virt_reg, virt_reg->index_);
                virt_reg->set_unassigned();
            } break;
            case virt_reg_stat::disabled:
            case virt_reg_stat::buffered:
            case virt_reg_stat::unassigned:
            case virt_reg_stat::designated: {
                assert(0 && "Invalid Unassign.");
            } break;
        }
        enqueue(virt_reg);
    }

    // Allocate physical_reg for virtual_reg
    void allocate(virtual_reg_t *virt_reg, virt_reg_index_t index) {
        assert(virt_reg);
        switch (virt_reg->stat_) {
            case virt_reg_stat::unassigned: {
                slots_array().assign_slot(virt_reg, index);
                virt_reg->set_allocated(index);
            } break;
            case virt_reg_stat::designated: {
                assert(index == virt_reg->index_);
                slots_array().assign_slot(virt_reg, index);
            } break;
            case virt_reg_stat::disabled:
            case virt_reg_stat::buffered:
            case virt_reg_stat::allocated:
            case virt_reg_stat::spilled: {
                assert(0 && "Invalid Allocate.");
            } break;
        }
    }

    // Spill virtual_reg on stack
    void spill(virtual_reg_t *virt_reg) {
        assert(virt_reg);
        switch (virt_reg->stat_) {
            case virt_reg_stat::unassigned: {
                virt_reg->set_spilled();
            } break;
            case virt_reg_stat::allocated:
            case virt_reg_stat::disabled:
            case virt_reg_stat::buffered:
            case virt_reg_stat::designated:
            case virt_reg_stat::spilled: {
                assert(0 && "Invalid Spill.");
            } break;
        }
        spilled_virt_regs_.insert(virt_reg);
        require_resolve_ = true;
    }

    // Current target profile
    const x86_64::target_profile_t &target_profile() { return target_profile_; }
    // Slots map to map between virtual slots and physical registers
    virtual_slots_map_t &slots_map() { return *virtual_slots_map_; }
    // Slots array to store allocated virtual registers and check interference
    virtual_slots_array_t &slots_array() { return *virtual_slots_array_; }
    // Spilled occurred
    bool require_resolve() { return require_resolve_; }
    // Spilled virt regs
    std::set<virtual_reg_t *> &spilled_virt_regs() {
        return spilled_virt_regs_;
    }

    // Virtual function for allocator_impl to create spill resolver
    virtual void resolve_spill_impl(const live_range_t &spill_range,
            std::vector<virtual_reg_t *> &virtual_regs)
            = 0;

private:
    const x86_64::target_profile_t &target_profile_;

    std::priority_queue<virtual_reg_t *, std::vector<virtual_reg_t *>,
            spill_weight_comparator_t>
            virtual_reg_queue_;

    bool require_resolve_;
    std::set<virtual_reg_t *> spilled_virt_regs_;
    std::shared_ptr<virtual_slots_map_t> virtual_slots_map_;
    std::shared_ptr<virtual_slots_array_t> virtual_slots_array_;
};

} // namespace xbyak
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
