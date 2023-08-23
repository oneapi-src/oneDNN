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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_IR_REG_ALLOCATION_VIRTUAL_REG_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_IR_REG_ALLOCATION_VIRTUAL_REG_HPP

#include <compiler/ir/sc_data_type.hpp>
#include <compiler/jit/xbyak/configured_xbyak.hpp>
#include <compiler/jit/xbyak/ir/util/utils.hpp>
#include <util/utils.hpp>

#include "live_range.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace xbyak {

using virt_reg_index_t = int32_t;
using spill_weight_t = int32_t;

namespace virt_reg_const {
constexpr virt_reg_index_t invalid = -1;
} // namespace virt_reg_const

namespace spill_weight_const {
constexpr spill_weight_t infinity = 65535;
constexpr spill_weight_t initial = 1;
constexpr spill_weight_t null = 0;
} // namespace spill_weight_const

enum class virt_reg_stat {
    disabled = 0, // no need for register allocation
    buffered, // tensor buffer at local stack
    designated, // designated physical registers
    unassigned, // to be assigned by register allcator
    allocated, // allocated by register allcator
    spilled, // spilled by register allcator/all addr operands
};

enum class virt_reg_type {
    gp_reg = 0, // x86 general purpose 64-bit registers
    fp_reg, // x86 AVX/AVX512(0-15/0-31) Extended SIMD registers
    fp_vex_reg, // x86 SSE/AVX(0-15) SIMD registers
    mask_reg, // x86 AVX512 mask registers
    tile_reg, // x86 AMX tile registers
    NUM_TYPES,
};

enum class virt_reg_hint {
    none = 0, // no hint
    weak, // perfer to be assigned at certain register
    strong, // must be assigned at certain register or spill
};

/* *
 * Xbyak::Reg hashing fucntion for hash map
 * */
struct xbyak_reg_hasher_t {
    size_t operator()(const Xbyak::Reg &r) const {
        size_t kind_mask = r.getKind() << 16;
        return kind_mask | r.getIdx();
    }
};

/* *
 * Virtual regsister for each expr, containing low level ir info, e.g. status,
 * type, allocation hint, spill weight, live range, assigned slot, etc.
 * */
struct virtual_reg_t {
    virt_reg_hint hint_ = virt_reg_hint::none;
    virt_reg_type type_ = virt_reg_type::gp_reg;
    virt_reg_stat stat_ = virt_reg_stat::disabled;

    live_range_t live_range_;

    virt_reg_index_t index_ = virt_reg_const::invalid;
    virt_reg_index_t index_hint_ = virt_reg_const::invalid;

    spill_weight_t spill_weight_ = spill_weight_const::initial;

    // need to be preserved across function calls
    bool preserved_ = false;
    // force fp_reg to only use sse/avx fp_reg_vex XMM0-15
    bool force_fp_vex_ = false;

    spill_weight_t extra_weight() {
        auto range = live_range_.end_ - live_range_.start_;
        spill_weight_t range_weight = range > 0 //
                ? 1 + (stmt_index_const::increment * 256 / range)
                : 0;
        spill_weight_t preserve_weight = preserved_ //
                ? stmt_index_const::increment * 128
                : 0;
        spill_weight_t hint_weight = hint_ != virt_reg_hint::none //
                ? stmt_index_const::increment * 16
                : 0;
        spill_weight_t fp_reg_weight = force_fp_vex_ //
                ? stmt_index_const::increment * 64
                : 0;
        return range_weight + preserve_weight + hint_weight + fp_reg_weight;
    }

    bool intersects(const virtual_reg_t &b) {
        return live_range_.intersects(b.live_range_);
    }

    bool disabled() { return stat_ == virt_reg_stat::disabled; }

    bool buffered() { return stat_ == virt_reg_stat::buffered; }

    bool spilled() { return stat_ == virt_reg_stat::spilled; }

    bool allocated() {
        return stat_ == virt_reg_stat::designated
                || stat_ == virt_reg_stat::allocated;
    }

    void set_preserved() { preserved_ = true; }
    void set_force_fp_vex() { force_fp_vex_ = true; }

    void set_type(virt_reg_type type) { type_ = type; }

    void set_hint(virt_reg_hint hint, virt_reg_index_t index) {
        if (static_cast<int>(hint) > static_cast<int>(hint_)) {
            hint_ = hint;
            index_hint_ = index;
        }
    }

    void reset_hint() {
        hint_ = virt_reg_hint::none;
        index_hint_ = virt_reg_const::invalid;
    }

    void set_buffered() {
        stat_ = virt_reg_stat::buffered;
        index_ = virt_reg_const::invalid;
    }

    void set_unassigned() {
        stat_ = virt_reg_stat::unassigned;
        index_ = virt_reg_const::invalid;
    }

    void set_designated(virt_reg_index_t index) {
        stat_ = virt_reg_stat::designated;
        index_ = index;
        spill_weight_ = spill_weight_const::infinity + 1;
        set_hint(virt_reg_hint::strong, index);
    }

    void set_allocated(virt_reg_index_t index) {
        stat_ = virt_reg_stat::allocated;
        index_ = index;
    }

    void set_spilled() {
        stat_ = virt_reg_stat::spilled;
        index_ = virt_reg_const::invalid;
    }

    void add_weight(spill_weight_t weight) {
        spill_weight_ = std::min(
                spill_weight_const::infinity - 1, spill_weight_ + weight);
        assert(spill_weight_ > 0);
    }

    friend std::ostream &operator<<(std::ostream &os, const virtual_reg_t &m) {
        static const char *type_enum_str[] = {"gp", "fp", "fpvex", "k", "tmm"};
        static const char *stat_enum_str[] = {"x", "B", "D", "U", "A", "S"};
        static const char *hint_enum_str[] = {"", "h", "H"};
        os << m.live_range_ << ": SW-" //
           << m.spill_weight_ << ": CP-" //
           << m.preserved_ << ": FV-" //
           << m.force_fp_vex_ << ": " //
           << stat_enum_str[static_cast<int>(m.stat_)]
           << hint_enum_str[static_cast<int>(m.hint_)];
        switch (m.stat_) {
            case virt_reg_stat::designated:
            case virt_reg_stat::allocated: {
                os << ": %" << type_enum_str[static_cast<int>(m.type_)]
                   << m.index_;
            } break;
            default: break;
        }
        return os;
    }

    virtual_reg_t() = default;
    virtual_reg_t(virt_reg_type type) : type_(type) {}
};

inline virt_reg_type get_virt_reg_type(
        const sc_data_type_t &t, bool is_avx512, bool force_fp_vex = false) {
    if (t.type_code_ == sc_data_etype::BOOLEAN && t.lanes_ > 1) {
        return is_avx512 ? virt_reg_type::mask_reg : virt_reg_type::gp_reg;
    } else if (force_fp_vex && is_x86_simd(t)) {
        return virt_reg_type::fp_vex_reg;
    } else if (is_x86_simd(t)) {
        return virt_reg_type::fp_reg;
    } else if (t.is_tile()) {
        return virt_reg_type::tile_reg;
    }

    return virt_reg_type::gp_reg;
}

} // namespace xbyak
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
