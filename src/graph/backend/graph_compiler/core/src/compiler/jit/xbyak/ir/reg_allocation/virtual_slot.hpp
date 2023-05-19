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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_IR_REG_ALLOCATION_VIRTUAL_SLOT_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_IR_REG_ALLOCATION_VIRTUAL_SLOT_HPP

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <unordered_map>

#include <compiler/jit/xbyak/x86_64/target_profile.hpp>

#include "interval_tree.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace xbyak {

/* *
 * Represent each physical rigister and store live ranges using non-overlapping
 * balanced interval tree
 * */
class virtual_slot_t : public interval_tree_t {
public:
    virtual_slot_t() = default;
    virtual ~virtual_slot_t() = default;

    void insert(virtual_reg_t *virt_reg) {
        assert(virt_reg);
        auto &live_range = virt_reg->live_range_;
        interval_tree_t::insert(live_range.start_, live_range.end_, virt_reg);
    }

    void remove(virtual_reg_t *virt_reg) {
        assert(virt_reg);
        auto &live_range = virt_reg->live_range_;
        interval_tree_t::remove(live_range.start_, live_range.end_, virt_reg);
    }

    void divide(virtual_reg_t *virt_reg, const live_range_t &range) {
        assert(virt_reg);
        interval_tree_t::divide(range.start_, range.end_, virt_reg);
    }

    bool intersects(virtual_reg_t *virt_reg) {
        assert(virt_reg);
        auto &live_range = virt_reg->live_range_;
        return interval_tree_t::search(live_range.start_, live_range.end_);
    }

    spill_weight_t intersect_weights(virtual_reg_t *virt_reg) {
        assert(virt_reg);
        auto &live_range = virt_reg->live_range_;
        spill_weight_t weight = spill_weight_const::null;
        auto query_func = [&](virtual_reg_t *virt_reg) {
            // Sum of intersected weight
            weight = std::min(spill_weight_const::infinity,
                    weight + virt_reg->spill_weight_);
        };
        query(live_range.start_, live_range.end_, query_func);
        return weight;
    }

    std::set<virtual_reg_t *> intersect_regs(virtual_reg_t *virt_reg) {
        assert(virt_reg);
        auto &live_range = virt_reg->live_range_;
        std::set<virtual_reg_t *> virtual_regs;
        auto query_func = [&](virtual_reg_t *vreg) {
            // Set of intersected virtual regs
            virtual_regs.insert(vreg);
        };
        query(live_range.start_, live_range.end_, query_func);
        return virtual_regs;
    }

    virtual_reg_t *encompassing(const live_range_t &range) {
        std::set<virtual_reg_t *> virtual_regs;
        auto query_func = [&](virtual_reg_t *vreg) {
            // Set of virtual_regs encompassing range
            auto &live_range = vreg->live_range_;
            if (live_range.encompasses(range)) { virtual_regs.insert(vreg); }
        };
        query(range.start_, range.end_, query_func);
        if (virtual_regs.empty()) { return nullptr; }
        assert(virtual_regs.size() == 1);
        return *virtual_regs.begin();
    }
};

/* *
 * Represent all virtual slots available for assign
 * */
class virtual_slots_array_t {
public:
    virtual_slots_array_t(virt_reg_index_t slots_sum)
        : virtual_slots_sum_(slots_sum) {
        virtual_slots_.resize(virtual_slots_sum_);
    }
    virtual ~virtual_slots_array_t() = default;

    void assign_slot(virtual_reg_t *virt_reg, virt_reg_index_t index) {
        assert(virt_reg);
        assert(index >= 0 && index < virtual_slots_sum_);
        virtual_slots_[index].insert(virt_reg);
    }

    void unassign_slot(virtual_reg_t *virt_reg, virt_reg_index_t index) {
        assert(virt_reg);
        assert(index >= 0 && index < virtual_slots_sum_);
        virtual_slots_[index].remove(virt_reg);
    }

    void divide_interval(virtual_reg_t *virt_reg, const live_range_t &range,
            virt_reg_index_t index) {
        assert(virt_reg);
        assert(index >= 0 && index < virtual_slots_sum_);
        virtual_slots_[index].divide(virt_reg, range);
    }

    bool interfered_with(virtual_reg_t *virt_reg, virt_reg_index_t index) {
        assert(virt_reg);
        assert(index >= 0 && index < virtual_slots_sum_);
        return virtual_slots_[index].intersects(virt_reg);
    }

    spill_weight_t interfered_weights(
            virtual_reg_t *virt_reg, virt_reg_index_t index) {
        assert(virt_reg);
        assert(index >= 0 && index < virtual_slots_sum_);
        return virtual_slots_[index].intersect_weights(virt_reg);
    }

    std::set<virtual_reg_t *> interfered_regs(
            virtual_reg_t *virt_reg, virt_reg_index_t index) {
        assert(virt_reg);
        assert(index >= 0 && index < virtual_slots_sum_);
        return virtual_slots_[index].intersect_regs(virt_reg);
    }

    virtual_reg_t *encompassing_reg(
            const live_range_t &range, virt_reg_index_t index) {
        assert(!range.empty());
        assert(index >= 0 && index < virtual_slots_sum_);
        return virtual_slots_[index].encompassing(range);
    }

    std::set<virt_reg_index_t> utilized_slots() {
        std::set<virt_reg_index_t> ret_set;
        for (virt_reg_index_t index = 0; index < virtual_slots_sum_; index++) {
            if (!virtual_slots_[index].empty()) { ret_set.insert(index); }
        }
        return ret_set;
    }

private:
    virt_reg_index_t virtual_slots_sum_;
    std::vector<virtual_slot_t> virtual_slots_;
};

/* *
 * Map between physical registers and vistual slots.
 * */
class virtual_slots_map_t {
public:
    // When fp_regs_volatile is true
    // Ignore abi profile and does not treat fp regs as callee-saved
    virtual_slots_map_t(const x86_64::target_profile_t &profile,
            bool fp_regs_volatile = false) {
        // Map all allocatable regs to virtual slot indexes
        virt_reg_index_t virt_index = 0;
        // Allocatable slot indexes
        allocatable_indexes_.resize(static_cast<int>(virt_reg_type::NUM_TYPES));
        callee_save_indexes_.resize(static_cast<int>(virt_reg_type::NUM_TYPES));
        callee_save_sets_.resize(static_cast<int>(virt_reg_type::NUM_TYPES));
        // ========================================
        // Get allocatable regs from target profile
        // ========================================
        auto get_allocatable_regs =
                [&](const virt_reg_type &reg_type,
                        const std::vector<Xbyak::Reg> &alloc_regs) {
                    auto type_index = static_cast<int>(reg_type);
                    for (size_t i = 0; i < alloc_regs.size(); i++) {
                        allocatable_indexes_[type_index].push_back(virt_index);
                        allocatable_regs_.push_back(alloc_regs[i]);
                        allocatable_regs_name_.push_back(
                                alloc_regs[i].toString());
                        xbyak_regs_map_[alloc_regs[i]] = virt_index;
                        virt_index++;
                    }
                };
        auto get_allocatable_fp_vex_regs
                = [&](const virt_reg_type &reg_type,
                          const std::vector<Xbyak::Reg> &alloc_regs) {
                      auto type_index = static_cast<int>(reg_type);
                      for (size_t i = 0; i < alloc_regs.size(); i++) {
                          auto idx = get_reg_index(alloc_regs[i]);
                          allocatable_indexes_[type_index].push_back(idx);
                      }
                  };

        // Allocatable gp regs to virtual indexes
        get_allocatable_regs(virt_reg_type::gp_reg, profile.alloc_gp_regs_);
        // Allocatable fp regs to virtual indexes
        get_allocatable_regs(virt_reg_type::fp_reg, profile.alloc_xmm_regs_);
        // Allocatable mask regs to virtual indexes
        get_allocatable_regs(virt_reg_type::mask_reg, profile.alloc_mask_regs_);
        // Allocatable tile regs to virtual indexes
        get_allocatable_regs(virt_reg_type::tile_reg, profile.alloc_tile_regs_);
        // Allocatable fp_vex regs, without create new virt_index
        get_allocatable_fp_vex_regs(
                virt_reg_type::fp_vex_reg, profile.alloc_xmm_vex_regs_);

        // Check consitancy
        assert((size_t)virt_index == allocatable_regs_.size());
        slots_sum_ = virt_index;

        // =========================================
        // Get callee saved regs from target profile
        // =========================================
        auto get_callee_saved_regs
                = [&](const virt_reg_type &reg_type,
                          const std::vector<Xbyak::Reg> &callee_saved_regs) {
                      auto type_index = static_cast<int>(reg_type);
                      for (auto &reg : callee_saved_regs) {
                          auto index = get_reg_index(reg);
                          callee_save_indexes_[type_index].push_back(index);
                          callee_save_sets_[type_index].insert(index);
                      }
                  };
        // Allocatable gp regs to virtual indexes
        get_callee_saved_regs(
                virt_reg_type::gp_reg, profile.callee_saved_gp_regs_);
        // Allocatable fp regs to virtual indexes
        if (!fp_regs_volatile) {
            get_callee_saved_regs(
                    virt_reg_type::fp_reg, profile.callee_saved_xmm_regs_);
        }
    }

    Xbyak::Reg get_reg_physical(virt_reg_index_t idx) {
        assert(idx >= 0 && idx < slots_sum_);
        return allocatable_regs_[idx];
    }

    std::string get_reg_name(virt_reg_index_t idx) {
        if (idx >= 0 && idx < slots_sum_) {
            return allocatable_regs_name_[idx];
        } else {
            return "";
        }
    }

    virt_reg_index_t get_reg_index(const Xbyak::Reg &reg) {
        auto iter = xbyak_regs_map_.find(reg);
        if (iter == xbyak_regs_map_.end()) {
            assert(false && "Not valid reg.");
            return virt_reg_const::invalid;
        }
        return iter->second;
    }

    const std::vector<virt_reg_index_t> &get_slots_index(virt_reg_type type) {
        assert(type != virt_reg_type::NUM_TYPES);
        return allocatable_indexes_[static_cast<int>(type)];
    }

    const std::vector<virt_reg_index_t> &get_callee_save(virt_reg_type type) {
        assert(type != virt_reg_type::NUM_TYPES);
        return callee_save_indexes_[static_cast<int>(type)];
    }

    const std::set<virt_reg_index_t> &get_callee_save_set(virt_reg_type type) {
        assert(type != virt_reg_type::NUM_TYPES);
        return callee_save_sets_[static_cast<int>(type)];
    }

    virt_reg_index_t get_slots_sum() { return slots_sum_; }

private:
    std::vector<std::vector<virt_reg_index_t>> allocatable_indexes_;
    std::vector<std::vector<virt_reg_index_t>> callee_save_indexes_;
    std::vector<std::set<virt_reg_index_t>> callee_save_sets_;
    std::vector<std::string> allocatable_regs_name_;
    std::vector<Xbyak::Reg> allocatable_regs_;
    std::unordered_map<Xbyak::Reg, virt_reg_index_t, xbyak_reg_hasher_t>
            xbyak_regs_map_;
    virt_reg_index_t slots_sum_;
};

} // namespace xbyak
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
