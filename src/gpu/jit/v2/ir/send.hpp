/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
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

#ifndef GPU_JIT_V2_IR_SEND_HPP
#define GPU_JIT_V2_IR_SEND_HPP

#include "gpu/jit/ir/block_2d_utils.hpp"
#include "gpu/jit/v2/ir/plan_utils.hpp"
#include "gpu/jit/v2/ir/reqs.hpp"
#include "gpu/jit/v2/ir/tensor.hpp"

#include <string>

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {
namespace v2 {

static const int max_slots = 32;
static const int max_slot_size = 8;

enum class send_op_t {
    undef,
    atomic_fadd,
    load,
    prefetch,
    store,
};

inline std::string to_string(send_op_t kind) {
    switch (kind) {
#define CASE(name) \
    case send_op_t::name: return #name
        CASE(undef);
        CASE(atomic_fadd);
        CASE(load);
        CASE(prefetch);
        CASE(store);
#undef CASE
        default: ir_error_not_expected();
    }
    return {};
}

struct addr_t {
    expr_t base;
    std::vector<expr_t> slot_incs;

    addr_t() = default;
    addr_t(const layout_t &layout, int slots, int elems_per_slot) {
        base = layout.base() * layout.type().size();
        slot_incs.resize(slots, 0);
        layout_iterator_t it(layout);
        for (int i = 1; i < slots; i++) {
            it.next(elems_per_slot);
            slot_incs[i] = layout.offset_in_bytes(it.block_offset());
        }
    }

    std::string str() const {
        using namespace ir_utils;
        std::ostringstream oss;
        oss << "base: " << base << std::endl;
        oss << "slot_incs: " << slot_incs;
        return oss.str();
    }

    IR_DEFINE_DUMP()
};

struct dim_mask_t {
    dim_mask_t() = default;

    dim_mask_t(const dim_mask_desc_t &dmd, int slots) : slot_incs(slots, 0) {
        dim = dmd.dim;
        base = dmd.base;
        bound = dmd.bound;
        do_zero_cmp = dmd.do_zero_cmp;
    }

    bool is_empty() const { return slot_incs.empty(); }
    int slots() const { return (int)slot_incs.size(); }

    std::string str() const {
        using namespace ir_utils;
        if (is_empty()) return "(empty)";
        std::ostringstream oss;
        oss << "[" << dim << "] " << base << " < " << bound << std::endl;
        oss << "slot_incs: " << slot_incs;
        return oss.str();
    }

    IR_DEFINE_DUMP()

    prb_dim_t dim;
    expr_t base;
    expr_t bound;
    std::vector<expr_t> slot_incs;
    bool do_zero_cmp = false;
};

struct mask_t {
    mask_t() = default;
    mask_t(const mask_desc_t &md, const layout_t &layout, int slots,
            int elems_per_slot) {
        for (int i = 0; i < md.nmasks(); i++) {
            dim_masks.emplace_back(md[i], slots);
        }
        if (layout.is_empty()) return;
        layout_iterator_t it(layout);
        for (int i = 1; i < slots; i++) {
            it.next(elems_per_slot);
            auto coord = it.coord();
            for (int j = 0; j < md.nmasks(); j++) {
                dim_masks[j].slot_incs[i]
                        = md[j].to_expr(coord, /*with_const=*/false);
            }
        }
    }
    mask_t(const mask_desc_t &md) : mask_t(md, layout_t(), 1, 0) {}

    // TODO: Rename.
    int nmasks() const { return static_cast<int>(dim_masks.size()); }
    int slots() const { return dim_masks[0].slots(); }
    void clear(const prb_dim_t &dim) {
        for (auto &dm : dim_masks) {
            if (dm.dim == dim) {
                dm = dim_mask_t();
                break;
            }
        }
    }

    std::string str() const {
        std::ostringstream oss;
        for (int i = 0; i < nmasks(); i++) {
            if (i != 0) oss << std::endl;
            auto tag = "#" + std::to_string(i);
            oss << ir_utils::add_tag(tag, dim_masks[i].str());
        }
        return oss.str();
    }

    IR_DEFINE_DUMP()

    std::vector<dim_mask_t> dim_masks;
};

struct multiply_hint_t {
    fma_kind_t fma = fma_kind_t::undef;
    int simd = 0;
    bool src1 = false;
    bool src2 = false;
    dim_map_t<prb_dim_t, prb_dim_kind_t> bmnk_map;

    bool is_k(const prb_dim_t &dim) const {
        return bmnk_map.get(dim, prb_dim_kind_t::undef) == prb_dim_kind_t::k;
    }
};

struct send_2d_hint_t {
    static const int any_block = 0;

    bool transpose = false;
    bool vnni = false;
    int width = 0;
    int height = 0;
    bool is_valid = false;

    send_2d_hint_t() = default;
    send_2d_hint_t(const view_t &view, send_op_t send_op,
            const multiply_hint_t &mul_hint = multiply_hint_t()) {
        auto &plane = view.plane();
        if (!plane) return;
        if (!utils::one_of(send_op, send_op_t::load, send_op_t::prefetch,
                    send_op_t::store))
            return;
        bool is_dpas = (mul_hint.fma == fma_kind_t::dpas);
        int w_blk = any_block;
        int h_blk = any_block;
        if (send_op == send_op_t::load && is_dpas
                && (mul_hint.src1 || mul_hint.src2)) {
            // Handle 4 cases (consider bf16):
            // src1, MxK: 16a16b -> 8a16b2a           (VNNI)
            // src1, KxM: 16a16b -> 16b16a -> 8b16a2b (transpose + VNNI)
            // src2, KxN: 16a16b -> 16b16a            (transpose)
            // src2, NxK: 16a16b -> 16a16b            ()
            int m_blk = mul_hint.simd;
            int n_blk = any_block;
            int mn_blk = (mul_hint.src1 ? m_blk : n_blk);
            int k_blk = 32 / plane.type.size();
            bool is_w_reduce = mul_hint.is_k(plane.w_dim);
            transpose = (mul_hint.src1 == is_w_reduce);
            vnni = mul_hint.src1;
            w_blk = is_w_reduce ? k_blk : mn_blk;
            h_blk = !is_w_reduce ? k_blk : mn_blk;
            if (vnni && transpose) return;
        }
        if (!init(send_op, plane.type, vnni, transpose, plane.w, plane.h, w_blk,
                    h_blk))
            return;

        is_valid = true;
    }

    operator bool() const { return is_valid; }

    bool init(send_op_t send_op, const type_t &type, bool vnni, bool transpose,
            int w_tile, int h_tile, int w_blk, int h_blk) {
        bool is_load_or_prefetch
                = utils::one_of(send_op, send_op_t::load, send_op_t::prefetch);
        bool is_store = (send_op == send_op_t::store);

        // Only D8, D16 and D32 are implemented.
        if (!utils::one_of(type.size(), 1, 2, 4)) return false;

        // VNNI and transpose are mutually exclusive.
        if (vnni && transpose) return false;

        // VNNI and transpose are supported with load only.
        if (is_store && (vnni || transpose)) return false;

        // VNNI is supported with D8 and D16 only.
        if (vnni && !utils::one_of(type.size(), 1, 2)) return false;

        // Transpose is supported with D32 only.
        if (transpose && type.size() != 4) return false;

        int w_min = (transpose ? 1 : 4 / type.size());
        int w_max = (transpose ? 8 : (vnni ? 16 : 64 / type.size()));
        int h_min = (vnni ? (4 / type.size()) : 1);
        int h_max = (is_load_or_prefetch ? 32 : 8);

        if (w_blk != any_block && (w_blk < w_min || w_blk > w_max))
            return false;
        if (h_blk != any_block && (h_blk < h_min || h_blk > h_max))
            return false;

        auto find_block = [&](int dim, int min, int max) {
            for (int b = max; b >= min; b--) {
                if (dim % b == 0) return b;
            }
            return -1;
        };

        if (w_blk == any_block) w_blk = find_block(w_tile, w_min, w_max);
        if (h_blk == any_block) h_blk = find_block(h_tile, h_min, h_max);
        if (w_blk == -1 || h_blk == -1) return false;

        if (vnni) h_blk = find_block(h_tile, h_blk, h_max);
        if (transpose && w_blk > 0) w_blk = find_block(w_tile, w_blk, w_max);

        width = w_blk;
        height = h_blk;
        return true;
    }
};

struct send_params_t {
    hw_t hw;
    send_kind_t kind = send_kind_t::undef;
    send_op_t op = send_op_t::undef;
    send_2d_hint_t hint_2d;
    // For register payload.
    int max_entry_reg_size = 0;
    std::vector<prb_dim_t> skip_mask;

    void init_max_entry_reg_size() {
        if (hint_2d) {
            max_entry_reg_size = 2048;
        } else {
            max_entry_reg_size = 512;
        }
    }

    void downgrade_to_1d() {
        hint_2d = send_2d_hint_t();
        init_max_entry_reg_size();
    }
};

struct send_1d_desc_t {
    hw_t hw;
    send_op_t op = send_op_t::undef;
    int type_size = 0;
    int slots = 0;

    operator bool() const { return op != send_op_t::undef; }

    bool base_alignment_ok(const expr_t &off, const prover_t &prover) {
        int align = (type_size >= 16 ? 8 : 1);
        if (!prover.prove(off % align == 0)) return false;
        return true;
    }

    bool base_alignment_ok(const addr_t &addr, const prover_t &prover) {
        if (!base_alignment_ok(addr.base, prover)) return false;
        for (auto &inc : addr.slot_incs) {
            if (!base_alignment_ok(inc, prover)) return false;
        }
        return true;
    }

    int header_size(int grf_size) const {
        return utils::rnd_up(8 * slots, grf_size);
    }

    std::string str() const {
        std::ostringstream oss;
        oss << to_string(op) << ".b" << type_size;
        if (slots != 1) oss << "x" << slots;
        return oss.str();
    }

    IR_DEFINE_DUMP()
};

struct send_1d_entry_t {
    expr_t addr_inc;
    std::vector<expr_t> mask_incs; // Per dimension mask.
    int reg_off = 0;
    prb_coord_t<int> coord;

    std::string str() const {
        using namespace ir_utils;
        std::ostringstream oss;
        oss << "mem[" << addr_inc << "] reg[" << reg_off << "] mask"
            << mask_incs;
        return oss.str();
    }
};

struct send_1d_plan_t : public base_plan_t {
    send_1d_desc_t desc;
    prb_reqs_t reqs;
    addr_t addr;
    mask_t mask;
    std::vector<send_1d_entry_t> entries;
    layout_t reg_layout;
    prb_tile_t entry_tile;

    using base_plan_t::base_plan_t;

    int nmasks() const { return mask.nmasks(); }
    int nentries() const { return static_cast<int>(entries.size()); }
    operator bool() const { return desc; }

    bool add_entry(const layout_iterator_t &it, const mask_desc_t &mask_desc,
            int reg_off, const prover_t &prover) {
        auto &layout = it.parent();
        auto &off = it.block_offset();
        expr_t addr_inc = layout.offset_in_bytes(off);
        if (!desc.base_alignment_ok(addr_inc, prover)) return false;
        std::vector<expr_t> mask_incs(nmasks());
        auto coord = it.coord();
        for (int i = 0; i < nmasks(); i++) {
            mask_incs[i] = mask_desc[i].to_expr(coord, /*with_const=*/false);
        }
        entries.emplace_back();
        auto &e = entries.back();
        e.addr_inc = addr_inc;
        e.mask_incs = mask_incs;
        e.reg_off = reg_off;
        e.coord = coord;
        ir_assert(reg_layout.offset_in_bytes(coord) == reg_off);
        return true;
    }

    int grf_usage_bytes() const {
        int ret = 0;
        ret += utils::rnd_up(reg_layout.size(), grf_size());
        ret += nentries() * desc.header_size(grf_size());
        return ret;
    }

    std::string str() const {
        std::ostringstream oss;
        oss << ir_utils::add_tag("addr", addr.str()) << std::endl;
        oss << ir_utils::add_tag("mask", mask.str()) << std::endl;
        oss << "reg_layout = " << reg_layout.str_with_size(hw) << std::endl;
        oss << desc << std::endl;
        for (int i = 0; i < nentries(); i++) {
            if (i != 0) oss << std::endl;
            oss << "  #" + std::to_string(i);
            oss << " " << entries[i].str();
        }
        return oss.str();
    }

    IR_DEFINE_DUMP()
};

struct send_2d_desc_t {
    hw_t hw;
    send_op_t op = send_op_t::undef;
    type_t type;
    bool transpose = false;
    bool vnni = false;
    expr_t W; // Surface width in elements.
    expr_t H; // Surface height in elements.
    expr_t P; // Pitch in elements.
    int w = 0; // Block width.
    int h = 0; // Block height.
    int c = 0; // Batch count.
    int w_rcount = 0;
    int h_rcount = 0;
    prb_dim_t w_dim;
    prb_dim_t h_dim;
    bool is_valid = false;
    expr_t base;

    send_2d_desc_t() = default;
    send_2d_desc_t(const view_t &view, const send_params_t &params,
            const prover_t &prover) {
        auto &plane = view.plane();
        if (!params.hint_2d) return;
        if (!plane) return;

        auto &hint = params.hint_2d;
        hw = params.hw;
        op = params.op;
        type = view.type();
        transpose = hint.transpose;
        vnni = hint.vnni;
        W = plane.W;
        H = plane.H;
        P = plane.P;
        w = hint.width;
        h = hint.height;
        c = 1;
        w_rcount = ir_utils::safe_div(plane.w, w);
        h_rcount = ir_utils::safe_div(plane.h, h);
        w_dim = plane.w_dim;
        h_dim = plane.h_dim;
        base = get_2d_base(view);
        try_promote_count();
        is_valid = is_supported(view, prover);
    }

    operator bool() const { return is_valid; }

    // Reduce the number of messages by increasing count per
    // message.
    void try_promote_count() {
        int max_count = block_2d_max_count(
                op == send_op_t::store, transpose, w, type.size());
        while (c * 2 <= max_count) {
            if (w_rcount % 2 != 0) break;
            c *= 2;
            w_rcount /= 2;
        }
    }

    bool is_supported(const view_t &view, const prover_t &prover) const {
        if (w % block_2d_x_alignment(type.size()) != 0) return false;

        auto &plane = view.plane();
        auto width_bytes = W * type.size();
        auto pitch_bytes = P * type.size();
        int base_align = block_2d_base_alignment(hw);
        int x_align = block_2d_x_alignment(type.size());
        if (!prover.prove(width_bytes >= 64)) return false;
        if (!prover.prove(width_bytes <= (1 << 24))) return false;
        if (!prover.prove(width_bytes % std::max(4, type.size()) == 0))
            return false;
        if (!prover.prove(H <= (1 << 24))) return false;
        if (!prover.prove(pitch_bytes >= 64)) return false;
        if (!prover.prove(pitch_bytes <= (1 << 24))) return false;
        if (!prover.prove(pitch_bytes % 8 == 0)) return false;
        if (!prover.prove(plane.y_stride == 1)) return false;
        if (!prover.prove(base % base_align == 0)) return false;
        if (!prover.prove(plane.x % x_align == 0)) return false;
        return true;
    }

    layout_t reg_layout(int grf_size, const layout_desc_t &desc) const {
        layout_t ret(desc, type);
        enum class pad_kind_t {
            none,
            dim_pow2,
            stride_grf,
        };
        int cur_stride = 1;
        auto add_block = [&](prb_dim_t dim, int size,
                                 pad_kind_t pad = pad_kind_t::none) {
            ret.add_block(dim, size, cur_stride);
            int stride = cur_stride * size;
            switch (pad) {
                case pad_kind_t::dim_pow2:
                    stride = cur_stride * utils::rnd_up_pow2(size);
                    break;
                case pad_kind_t::stride_grf:
                    stride = utils::rnd_up(stride, grf_size / type.size());
                    break;
                case pad_kind_t::none: break;
                default: ir_error_not_expected();
            }
            cur_stride = stride;
        };
        if (transpose) {
            add_block(h_dim, h, pad_kind_t::dim_pow2);
            add_block(w_dim, w, pad_kind_t::stride_grf);
        } else if (vnni) {
            int h_inner = 4 / type.size();
            int h_outer = ir_utils::safe_div(h, h_inner);
            add_block(h_dim, h_inner);
            add_block(w_dim, w, pad_kind_t::dim_pow2);
            add_block(h_dim, h_outer, pad_kind_t::stride_grf);
        } else {
            add_block(w_dim, w, pad_kind_t::dim_pow2);
            add_block(h_dim, h, pad_kind_t::stride_grf);
        }
        return ret;
    }

    int header_size(int grf_size) const { return grf_size; }

    std::string str() const {
        std::ostringstream oss;
        oss << to_string(op) << "_2d.";
        oss << c << "x" << h << "x" << w;
        if (vnni || transpose) {
            oss << ".";
            if (vnni) oss << "v";
            if (transpose) oss << "t";
        }
        return oss.str();
    }

    IR_DEFINE_DUMP()

    static expr_t get_2d_base(const view_t &view) {
        auto dim_mapper = view.dim_mapper();
        dim_mapper.set_dim(view.plane().x_dim, 0);
        dim_mapper.set_dim(view.plane().y_dim, 0);
        auto l = view.base_layout().map(dim_mapper, view.coord(), view.tile());
        return simplify_rewrite(l.base() * l.type().size());
    }
};

struct send_2d_entry_t {
    expr_t x_inc;
    expr_t y_inc;
    int reg_off = 0;
    prb_coord_t<int> coord;

    std::string str() const {
        std::ostringstream oss;
        oss << "reg[" << reg_off << "] ";
        oss << "x_inc = " << x_inc << " y_inc = " << y_inc;
        return oss.str();
    }

    IR_DEFINE_DUMP()
};

struct send_2d_plan_t : public base_plan_t {
    send_2d_desc_t desc;
    prb_reqs_t reqs;
    expr_t base;
    expr_t x_base;
    expr_t y_base;
    mask_t mask;
    std::vector<send_2d_entry_t> entries;
    layout_t reg_layout;
    prb_tile_t entry_tile;

    using base_plan_t::base_plan_t;

    int nentries() const { return static_cast<int>(entries.size()); }
    operator bool() const { return desc; }

    bool add_entry(const prb_coord_t<int> &coord, int reg_off,
            const prover_t &prover) {
        entries.emplace_back();
        auto &e = entries.back();
        e.x_inc = coord.at(desc.w_dim);
        e.y_inc = coord.at(desc.h_dim);
        e.reg_off = reg_off;
        e.coord = coord;
        return true;
    }

    int grf_usage_bytes() const {
        int ret = 0;
        ret += utils::rnd_up(reg_layout.size(), grf_size());
        ret += nentries() * desc.header_size(grf_size());
        return ret;
    }

    std::string str() const {
        std::ostringstream oss;
        oss << "base = " << base << std::endl;
        oss << "x_base = " << x_base << std::endl;
        oss << "y_base = " << y_base << std::endl;
        oss << ir_utils::add_tag("mask", mask.str()) << std::endl;
        oss << "reg_layout = " << reg_layout.str_with_size(hw) << std::endl;
        oss << desc << std::endl;
        for (int i = 0; i < nentries(); i++) {
            if (i != 0) oss << std::endl;
            oss << "  #" << std::to_string(i);
            oss << " " << entries[i].str();
        }
        return oss.str();
    }

    IR_DEFINE_DUMP()
};

struct send_plan_t : public base_plan_t {
    send_1d_plan_t _1d;
    send_2d_plan_t _2d;

    using base_plan_t::base_plan_t;

    bool is_1d() const { return _1d; }
    bool is_2d() const { return _2d; }
    send_1d_plan_t &get_1d() { return _1d; }
    const send_1d_plan_t &get_1d() const { return _1d; }
    send_2d_plan_t &get_2d() { return _2d; }
    const send_2d_plan_t &get_2d() const { return _2d; }

    const prb_reqs_t &reqs() const {
        if (is_1d()) return _1d.reqs;
        return _2d.reqs;
    }

    const layout_t &reg_layout() const {
        if (is_1d()) return _1d.reg_layout;
        return _2d.reg_layout;
    }

    const prb_tile_t &entry_tile() const {
        if (is_1d()) return _1d.entry_tile;
        return _2d.entry_tile;
    }

    int grf_usage_bytes() const {
        if (is_1d()) return _1d.grf_usage_bytes();
        return _2d.grf_usage_bytes();
    }

    std::string str() const {
        if (is_1d()) return _1d.str();
        return _2d.str();
    }

    IR_DEFINE_DUMP()
};

class send_plan_builder_t {
public:
    send_plan_builder_t() = default;
    send_plan_builder_t(const send_params_t &params, const view_t &view)
        : params_(params), view_(view), plan_(params_.hw) {}

    const send_plan_t &plan() const { return plan_; }

    void build() {
        if (try_build_2d()) return;
        if (try_build_1d()) return;
        ir_error_not_expected();
    }

    bool try_build_1d() {
        prb_reqs_t reqs;
        auto &layout = view_.layout();
        auto &mask_desc = view_.mask_desc();
        auto inner_last = find_inner_last(mask_desc, reqs);
        int type_size = layout.type().size();
        int inner_elems = inner_last.elems();
        int inner_bytes = type_size * inner_elems;
        int slot_size = ir_utils::max_pow2_divisor(inner_bytes);
        int grf_size = plan_.hw.grf_size();

        // TODO: Add oword block support.
        if (slot_size < grf_size)
            slot_size = std::min(max_slot_size, slot_size);

        ir_assert(inner_bytes % slot_size == 0);
        ir_assert(slot_size % type_size == 0);
        bool is_scattered = (slot_size <= max_slot_size);
        int slots = inner_bytes / slot_size;
        int elems_per_slot = slot_size / type_size;
        int slot_stride = std::max(4, slot_size);

        auto inner_end = inner_last + 1;
        auto middle_last = inner_last;
        auto outer_begin = end(layout);
        if (is_scattered) {
            // Add blocks to fill up slots in the scattered message.
            for (auto it = inner_end; it != end(layout); ++it) {
                int it_slots = ir_utils::safe_div(it.elems(), elems_per_slot);
                int entry_reg_size
                        = utils::rnd_up(it_slots * slot_stride, grf_size);
                if (it_slots > max_slots
                        || entry_reg_size > params_.max_entry_reg_size) {
                    outer_begin = it;
                    break;
                }
                slots = it_slots;
                middle_last = it;
            }
        }

        send_1d_desc_t desc;
        desc.hw = params_.hw;
        desc.op = params_.op;
        desc.type_size = slot_size;
        desc.slots = slots;

        addr_t addr(layout, slots, elems_per_slot);
        if (!desc.base_alignment_ok(addr, reqs.prover())) return false;

        auto reg_layout = middle_last.sub_layout();
        reg_layout.pad_bytes(grf_size);

        auto entry_tile = reg_layout.int_dim_sizes();
        add_remaining_blocks(reg_layout, middle_last);
        reg_layout.normalize();

        auto &plan = plan_.get_1d();
        plan = send_1d_plan_t(plan_.hw);
        plan.desc = desc;
        plan.addr = addr;
        plan.mask = mask_t(mask_desc, layout, slots, elems_per_slot);
        plan.reg_layout = reg_layout;
        plan.entry_tile = entry_tile;

        for (auto &d : params_.skip_mask)
            plan.mask.clear(d);

        int step_elems = slots * elems_per_slot;
        layout_iterator_t it(layout);
        int reg_off = 0;
        plan.add_entry(it, mask_desc, reg_off, reqs.prover());
        while (it.has_next(step_elems)) {
            it.next(step_elems);
            reg_off += slots * slot_stride;
            reg_off = utils::rnd_up(reg_off, grf_size);
            if (!plan.add_entry(it, mask_desc, reg_off, reqs.prover()))
                return false;
        }
        plan.reqs = reqs;
        return true;
    }

    bool try_build_2d() {
        if (params_.kind != send_kind_t::_2d) return false;

        prb_reqs_t reqs;
        send_2d_desc_t desc(view_, params_, reqs.prover());
        if (!desc) return false;

        auto &plane = view_.plane();
        int grf_size = params_.hw.grf_size();
        auto reg_layout = desc.reg_layout(grf_size, view_.layout().desc());
        int entry_reg_size = utils::rnd_up(reg_layout.size(), grf_size);
        ir_assert(entry_reg_size <= params_.max_entry_reg_size);
        reg_layout.pad_bytes(grf_size);

        auto entry_tile = reg_layout.int_dim_sizes();
        reg_layout.add_block(plane.w_dim, desc.w_rcount);
        reg_layout.add_block(plane.h_dim, desc.h_rcount);

        auto &plan = plan_.get_2d();
        plan = send_2d_plan_t(plan_.hw);
        plan.desc = desc;
        plan.reqs = reqs;
        plan.base = desc.base;
        plan.x_base = plane.x;
        plan.y_base = plane.y;
        plan.mask = mask_t(view_.mask_desc());
        plan.mask.clear(plane.x_dim);
        plan.mask.clear(plane.y_dim);
        for (auto &d : params_.skip_mask)
            plan.mask.clear(d);

        plan.reg_layout = reg_layout;
        plan.entry_tile = entry_tile;

        int reg_off = 0;
        for (int h = 0; h < plane.h; h += desc.h) {
            for (int w = 0; w < plane.w; w += desc.w) {
                prb_coord_t<int> coord;
                coord[plane.w_dim] = w;
                coord[plane.h_dim] = h;
                if (!plan.add_entry(coord, reg_off, reqs.prover()))
                    return false;
                reg_off += entry_reg_size;
            }
        }
        return true;
    }

private:
    block_iterator_t find_inner_last(
            const mask_desc_t &mask_desc, prb_reqs_t &reqs) const {
        auto &layout = view_.layout();
        auto inner_last = begin(layout);
        int type_size = layout.type().size();
        auto ok_to_return = [&]() {
            if (params_.kind != send_kind_t::block) return true;
            int grf_size = plan_.hw.grf_size();
            return type_size * inner_last.elems() >= grf_size;
        };
        for (auto it = begin(layout); it != end(layout); ++it) {
            auto prover = reqs.prover(!ok_to_return());
            if (!mask_desc.is_uniform(it, prover)) break;
            if (!it.is_dense(prover)) break;
            if (type_size * it.elems() > params_.max_entry_reg_size) break;
            inner_last = it;
        }
        return inner_last;
    }

    void normalize(send_1d_plan_t &plan) const {
        auto &desc = plan.desc;
        if (desc.slots != 1) return;

        const int max_type_size = 512;
        if (desc.type_size <= max_type_size) return;

        ir_assert(desc.type_size % max_type_size == 0);
        send_1d_plan_t new_plan;
        new_plan.desc = desc;
        new_plan.desc.type_size = max_type_size;
        new_plan.addr = plan.addr;
        new_plan.mask = plan.mask;
        new_plan.reg_layout = plan.reg_layout;
        for (auto &_e : plan.entries) {
            auto e = _e;
            for (int off = 0; off < desc.type_size; off += max_type_size) {
                e.reg_off = _e.reg_off + off;
                e.addr_inc = _e.addr_inc + off;
                new_plan.entries.push_back(e);
            }
        }
        plan = new_plan;
    }

    send_params_t params_;
    view_t view_;
    send_plan_t plan_;
};

inline send_plan_t create_send_plan(
        const send_params_t &params, const view_t &view) {
    send_plan_builder_t spb(params, view);
    spb.build();
    return spb.plan();
}

} // namespace v2
} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
