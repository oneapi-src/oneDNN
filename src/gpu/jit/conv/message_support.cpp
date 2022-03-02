/*******************************************************************************
* Copyright 2021-2022 Intel Corporation
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

#include "gpu/jit/conv/message_support.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

stmt_t send_t::create_offset_store(const expr_t &header_buf,
        const expr_t &mem_buf, const expr_t &mem_off,
        bool is_signed_offset) const {
    expr_t header_sub_buf;
    expr_t off;
    if (is_block() && (is_slm() || is_bts())) {
        // Convert byte offset to dwords/owords/hwords offset.
        off = mem_off / type.scalar().size();
        header_sub_buf = header_buf[2 * sizeof(uint32_t)];
    } else if (is_a64()) {
        // Convert buffer to 64-bit integer.
        off = cast(mem_buf, type_t::u64());
        if (mem_off.type().is_vector())
            off = shuffle_t::make_broadcast(off, mem_off.type().elems());
        off += mem_off;
        header_sub_buf = header_buf[0];
    } else if (is_bts()) {
        off = cast(mem_off, type_t::u32(mem_off.type().elems()));
        header_sub_buf = header_buf[0];
    } else {
        ir_error_not_expected();
    }
    off = cast(off, address_type(is_signed_offset, off.type().elems()));
    return store_t::make(header_sub_buf, 0, off);
}

bool send_t::is_supported() const {
    if (access_size() > 256) return false;

    // Block messages imply one slot.
    if (is_block() && slots != 1) return false;

    if (is_block() && !utils::one_of(type.elems(), 1, 2, 4, 8, 16))
        return false;

    // owordx8 is max supported unless accessing SLM.
    if (is_block() && !is_slm() && type.elems() > 8) return false;

    // hword is not supported with SLM.
    if (is_slm() && type.is_hword()) return false;

    // Allow only block messages for SLM to reduce offset-related arithmetic.
    if (is_slm() && !is_block()) return false;

    // Only load/store with SLM.
    if (is_slm() && !is_load() && !is_store()) return false;

    // No hword stores before XeHPC.
    if (is_store() && type.is_hword() && !is_xe_hpc_plus()) return false;

    // XXX: Half-GRF stores result in correctness issues on XeHPC.
    if (is_store() && is_block() && is_xe_hpc_plus()
            && type.size() % grf_size() != 0)
        return false;

    // Skip transposing messages, they need additional logic in message
    // decomposition to handle layouts.
    if (type.is_dword() && type.elems() != 1) return false;

    // XXX: Allow only hword x {1,2,4,8} prefetch for now.
    if (is_prefetch() && !type.is_hword()) return false;
    if (is_prefetch() && type.elems() > 8) return false;

    // Expect only float atomics.
    if (is_atomic() && !type.is_dword()) return false;

    if (is_atomic() && !is_xe_hpc_plus() && is_a64() && slots > 8) return false;

    // XXX: Tested only byte scattered messages.
    if (is_scattered() && !is_atomic() && !type.is_byte()) return false;

    if (is_scattered() && !is_atomic() && !utils::one_of(type.elems(), 1, 2, 4))
        return false;

    return true;
}

std::vector<func_t> send_t::get_all(ngen::HW hw, send_op_t op,
        send_address_t address, const type_t &mem_type) {
    std::vector<func_t> filtered;
    for (int slots : {1, 2, 4, 8, 16}) {
        for (int elems : {1, 2, 4, 8, 16}) {
            for (auto &type : {type_t::byte(), type_t::dword(), type_t::oword(),
                         type_t::hword()}) {
                // Require data type size exact match for atomic messages.
                if (op == send_op_t::atomic_fadd
                        && type.size() != mem_type.size())
                    continue;

                auto f = send_t::make(
                        hw, op, address, type.with_elems(elems), slots);
                if (!f.as<send_t>().is_supported()) continue;
                filtered.push_back(f);
            }
        }
    }

    // Sort by total size in descending order.
    std::sort(filtered.begin(), filtered.end(),
            [](const func_t &_a, const func_t &_b) {
                auto &a = _a.as<send_t>();
                auto &b = _b.as<send_t>();
                size_t a_sz = a.access_size();
                size_t b_sz = b.access_size();
                // Put block messages first.
                if (a.is_block() != b.is_block()) return a.is_block();
                if (a_sz == b_sz)
                    return a.type.scalar().size() > b.type.scalar().size();
                return a_sz > b_sz;
            });

    // Remove block messages with the same size (e.g. owordx4 and hwordx2).
    std::vector<func_t> ret;
    for (size_t i = 0; i < filtered.size(); i++) {
        if (i > 0) {
            auto &s_prev = filtered[i - 1].as<send_t>();
            auto &s_cur = filtered[i].as<send_t>();
            if (s_prev.is_block() && s_cur.is_block()
                    && (s_prev.type.size() == s_cur.type.size()))
                continue;
        }
        ret.push_back(filtered[i]);
    }

    return ret;
}

// Helper class to iterate through global memory offsets, provides queries to
// check whether given blocks are dense, properly aligned, etc.
class memory_walker_t {
public:
    memory_walker_t(const constraint_set_t &cset, const view_t &view)
        : view_(view)
        , type_size_(view.type().size())
        , mask_tensor_(view.create_mask_tensor(cset).reinterpret(view.type()))
        , full_size_(view.velems() * type_size_) {
        init_dense_blocks(cset);
        reset();
    }

    void reset() {
        cur_off_ = 0;
        remaining_size_ = full_size_;
    }

    bool has_next() const { return cur_off_ < full_size_; }

    int remaining_size() const { return remaining_size_; }

    int remaining_elems() const { return remaining_size_ / type_size_; }

    bool is_dense_and_aligned(int off, int size, int alignment) const {
        if (size % type_size_ != 0) return false;
        if (off + size > remaining_size_) return false;
        if (size == 0) return true;
        int beg = cur_off_ + off;
        int end = cur_off_ + off + size;
        if (get_block_index(beg) != get_block_index(end - 1)) return false;
        if (alignment != 0 && get_alignment(beg) < alignment) return false;
        return true;
    }

    // Returns true if each of the given slot regions is dense and aligned.
    bool check_region(int off, int slots, int slot_size, int alignment) const {
        for (int i = 0; i < slots; i++) {
            int off = i * slot_size;
            // Overflow is fine, expect it to be handled by proper masking.
            if (off >= remaining_size_) return true;
            if (!is_dense_and_aligned(off, slot_size, alignment)) return false;
        }
        return true;
    }

    // Returns true if the given region can be masked with `mask_size`
    // granularity and `nmasks` number of masks.
    bool check_mask_size(int off, int size, int mask_size, int nmasks) const {
        auto mask = get_mask(off, size, mask_size, nmasks, /*allow_fail=*/true);
        return !mask.is_empty();
    }

    expr_t get_offset(int off, expr_t &base, int &off_const) const {
        if (off >= remaining_size_) {
            base = expr_t(0);
            off_const = 0;
            return base;
        }
        int block_idx = get_block_index(cur_off_ + off);
        ir_assert(block_idx >= 0 && block_idx < int(block_offs_.size()));
        base = block_offs_[block_idx];
        off_const = (cur_off_ + off) % dense_block_size_;
        if (off_const == 0) return base;
        return base + off_const;
    }

    // Returns a boolean mask expression for the given region to access.
    expr_t get_mask(int off, int size, int mask_size, int nmasks,
            bool allow_fail = false) const {
        ir_assert(size % mask_size == 0) << "Incompatible mask size.";
        auto sub_mask_tensor = create_sub_mask_tensor(off, size);
        sub_mask_tensor = sub_mask_tensor.reinterpret(type_t::u8(mask_size));
        if (sub_mask_tensor.is_empty()) {
            if (allow_fail) return expr_t();
            ir_error_not_expected();
        }
        auto ret = sub_mask_tensor.to_expr(nmasks);
        if (ret.is_empty()) {
            if (allow_fail) return expr_t();
            ir_error_not_expected() << "Can't create mask.";
        }
        return ret;
    }

    // Moves the current position `size` bytes ahead.
    void advance(int size) {
        ir_assert(size % type_size_ == 0);
        size = std::min(size, remaining_size_);
        cur_off_ += size;
        remaining_size_ -= size;
    }

private:
    void init_dense_blocks(const constraint_set_t &cset) {
        auto l = view_.create_pseudo_vlayout();
        // Find the maximum innermost dense tile.
        stride_t stride = 1;
        std::vector<dim_t> dims(l.ndims(), 1);
        for (auto &b : l.blocks()) {
            if (b.stride != stride) break;
            dims[b.dim_idx] *= b.block;
            stride = b.block * b.stride;
        }
        tensor_t tile(dims);
        dense_block_size_ = tile.elems() * type_size_;
        // Split the memory view into dense blocks and precompute block offsets
        // and alignments.
        view_.for_each_tile(tile, [&](const std::vector<dim_t> &start) {
            auto off = view_.offset_in_bytes(expr_cast<expr_t>(start));
            off = simplify(off, cset);

            const int base_alignment = 128;
            int64_t f = get_max_const_factor(off, cset);
            int alignment = f ? ir_utils::max_pow2_divisor(f) : base_alignment;

            block_offs_.push_back(off);
            block_alignments_.push_back(alignment);
        });
    }

    mask_tensor_t create_sub_mask_tensor(int off, int size) const {
        ir_assert(off % type_size_ == 0);
        ir_assert(size % type_size_ == 0);

        std::vector<dim_t> sub_dims = {size / type_size_};
        layout_t sub_layout(view_.type(), 0, sub_dims);
        mask_tensor_t sub_mask_tensor(sub_layout);
        int beg = (cur_off_ + off) / type_size_;
        int end = (cur_off_ + off + size) / type_size_;
        for (int i = beg; i < end; i++) {
            auto mask = (i < mask_tensor_.elems()) ? mask_tensor_.mask(i)
                                                   : expr_t(false);
            sub_mask_tensor.set_mask(i - beg, mask);
        }
        return sub_mask_tensor;
    }

    int get_block_index(int off) const { return off / dense_block_size_; }

    int get_alignment(int off) const {
        int block_alignment = block_alignments_[off / dense_block_size_];
        return ir_utils::max_pow2_divisor(
                block_alignment + off % dense_block_size_);
    }

    view_t view_;
    int type_size_;
    mask_tensor_t mask_tensor_;
    std::vector<expr_t> block_offs_;
    std::vector<int> block_alignments_;
    int cur_off_ = 0;
    int full_size_ = 0;
    int remaining_size_ = 0;
    int dense_block_size_ = 0;
};

class layout_walker_t {
public:
    layout_walker_t() = default;
    layout_walker_t(const layout_t &layout, int grf_size)
        : layout_(layout)
        , grf_size_(grf_size)
        , type_size_(layout.type().size())
        , idxs_(layout.blocks().size()) {}

    int offset_bytes() const { return off_bytes_; }

    bool can_access(int size) const {
        int off = off_bytes_ + size;
        return off <= max_offset_bytes();
    }

    // Returns true if the next `elems` elements can be stored in the layout
    // given the following requirements:
    // - They must be uniformly strided with `stride` (specified in elements)
    // - The last element must be GRF boundary aligned (unless `is_last_region`
    //   is true)
    // - The last element must not cross the layout boundary
    bool can_advance(int stride, int elems, bool is_last_region) {
        if (is_last_region) elems = std::min(elems, remaining_elems());
        auto cur_idxs = idxs_;
        int cur_off_bytes = off_bytes_;
        for (int i = 0; i < elems - 1; i++) {
            int next_off_bytes = advance(cur_idxs, cur_off_bytes);
            if (next_off_bytes - cur_off_bytes != stride * type_size_)
                return false;
            cur_off_bytes = next_off_bytes;
        }
        cur_off_bytes = advance(cur_idxs, cur_off_bytes);
        if (cur_off_bytes > max_offset_bytes()) return false;
        if (!is_last_region && cur_off_bytes % grf_size_ != 0) return false;
        return true;
    }

    // Moves the current position `elems` elements ahead.
    void advance(int elems) {
        elems = std::min(elems, remaining_elems());
        for (int i = 0; i < elems; i++) {
            off_bytes_ = advance(idxs_, off_bytes_);
            elems_++;
        }
    }

private:
    int max_offset_bytes() const {
        return utils::rnd_up((int)layout_.size(), grf_size_);
    }

    int remaining_elems() const { return layout_.elems() - elems_; }

    int advance(std::vector<int> &idxs, int off_bytes) const {
        for (size_t i = 0; i < idxs.size(); i++) {
            if (++idxs[i] < layout_.blocks()[i].block) break;
            idxs[i] = 0;
        }
        int off = 0;
        for (size_t i = 0; i < idxs.size(); i++) {
            int stride = (int)layout_.blocks()[i].stride;
            off += idxs[i] * stride;
        }
        return off * type_size_;
    }

    layout_t layout_;
    int grf_size_;
    int type_size_;

    std::vector<int> idxs_;
    int elems_ = 0;
    int off_bytes_ = 0;
};

access_builder_t::access_builder_t(ngen::HW hw, ir_context_t &ir_ctx,
        const constraint_set_t &cset, const view_t &mem_view,
        const expr_t &mem_buf, const expr_t &reg_buf, send_op_t send_op,
        send_address_t send_address)
    : hw_(hw)
    , mem_view_(mem_view)
    , mem_buf_(mem_buf)
    , reg_buf_(reg_buf)
    , send_op_(send_op)
    , send_address_(send_address)
    , mem_type_(mem_view.type())
    , mem_walker_(utils::make_unique<memory_walker_t>(cset, mem_view)) {
    build();
}

access_builder_t::access_builder_t(access_builder_t &&) = default;
access_builder_t::~access_builder_t() = default;

void access_builder_t::build() {
    bool ok = false;
    for (auto &l : candidate_payload_layouts()) {
        // Try to find send decomposition with the given GRF payload layout.
        if (try_build(l)) {
            ok = true;
            break;
        }
    }
    if (!ok && send_op_ == send_op_t::prefetch) {
        // Do not treat as an error, skip prefetch messages during generation.
        ir_warning() << "Can't generate send decomposition for prefetch."
                     << std::endl;
        return;
    }
    ir_assert(ok) << "Can't generate send decomposition.";
}

bool access_builder_t::try_build(const layout_t &try_layout) {
    auto &try_layout_blocks = try_layout.blocks();
    int reg_stride
            = (try_layout_blocks.empty() ? 0
                                         : (int)try_layout_blocks[0].stride);
    auto send_list = send_t::get_all(hw_, send_op_, send_address_, mem_type_);
    int grf_size = ngen::GRF::bytes(hw_);
    reg_layout_walker_
            = utils::make_unique<layout_walker_t>(try_layout, grf_size);
    stmt_ = stmt_t();
    mem_walker_->reset();
    // Iterate through the memory view, greedily select messages according to
    // the sorted message list.
    while (mem_walker_->has_next()) {
        func_t _send;
        for (auto &_s : send_list) {
            auto &s = _s.as<send_t>();

            int slot_size = s.type.size();
            int alignment = s.alignment();
            int nmasks = s.nmasks();
            int payload_stride = s.payload_type_stride();
            int access_size = s.access_size();
            int access_elems = access_size / mem_type_.size();
            bool is_last_chunk = mem_walker_->remaining_size() <= access_size;

            if (reg_stride != 1 || payload_stride != slot_size) {
                // Detected strided GRF layout or strided payload. In this
                // case require full data type and stride match.
                if (reg_stride != 0
                        && payload_stride != reg_stride * mem_type_.size())
                    continue;
                if (s.type.size() != mem_type_.size()) continue;
            }
            // Prefetches don't have payload so skip these conditions for
            // prefetch.
            if (!s.is_prefetch()) {
                if (!reg_layout_walker_->can_advance(
                            reg_stride, access_elems, is_last_chunk))
                    continue;

                if (!reg_layout_walker_->can_access(s.payload_size())) continue;
            }

            // Check if slots are contiguous and aligned.
            if (!mem_walker_->check_region(0, s.slots, slot_size, alignment))
                continue;

            // Check mask requirements.
            // XXX: Mask is not generated for prefetch to reduce offset
            // arithmetic.
            if (!s.is_prefetch()
                    && !mem_walker_->check_mask_size(
                            0, access_size, s.mask_size(), nmasks))
                continue;

            _send = _s;
            break;
        }
        // Can't find a message - try another GRF layout for payload.
        if (_send.is_empty()) return false;

        auto &send = _send.as<send_t>();
        auto send_stmt = create_send_stmt(send);
        stmt_ = stmt_.append(send_stmt);

        reg_layout_walker_->advance(send.access_size() / mem_type_.size());
        mem_walker_->advance(send.access_size());
    }
    reg_layout_ = try_layout;
    return true;
}

std::vector<layout_t> access_builder_t::candidate_payload_layouts() const {
    int type_size = mem_type_.size();
    auto vlayout = mem_view_.create_dense_vlayout();

    std::vector<layout_t> ret;

    auto &vblocks = vlayout.blocks();
    auto &tblocks = mem_view_.tlayout().blocks();
    int grf_size = ngen::GRF::bytes(hw_);

    // This is to support layouts that are half-GRF blocked (e.g. u8/s8 32c on
    // XeHPC). In this case we can pad the block to full register and apply GRF
    // reorder later - this may be more efficient than using scattered
    // messages.
    if (send_address_ != send_address_t::slm && !vblocks.empty()
            && !tblocks.empty()
            && mem_view_.tlayout().innermost_block_layout().size() < grf_size) {
        auto &v0 = vblocks[0];
        auto &t0 = tblocks[0];
        int v0_size = type_size * v0.block;
        int t0_size = type_size * t0.block;
        int half_grf_size = grf_size / 2;
        if (v0.dim_idx == t0.dim_idx && (int)v0.stride == 1
                && (int)t0.stride == 1 && v0_size % half_grf_size == 0
                && t0_size % half_grf_size == 0 && v0_size < grf_size) {
            auto tmp = vlayout.make_strided(grf_size / type_size, 1);
            ret.push_back(tmp);
        }
    }

    // Dense payload layout directly mapping to the memory view.
    ret.push_back(vlayout);

    // These payload layouts are to match payload for byte x {1,2} scattered
    // messages (they are dword-strided).
    if (type_size == 2) ret.push_back(vlayout.make_strided(2));
    if (type_size == 1) ret.push_back(vlayout.make_strided(4));

    return ret;
}

stmt_t access_builder_t::create_send_stmt(const send_t &send) {
    std::vector<expr_t> off_vec;
    // Try to detect a common base and const vector offset to reduce further
    // arithmetic.
    expr_t off_base0;
    int off_const0 = -1;
    bool is_same_base = true;
    std::vector<expr_t> off_const_vec;
    for (int i = 0; i < send.slots; i++) {
        expr_t off_base;
        int off_const;
        auto off = mem_walker_->get_offset(
                i * send.type.size(), off_base, off_const);
        if (off_base0.is_empty()) {
            off_base0 = off_base;
            off_const0 = off_const;
        } else if (!off_base.is_equal(off_base0)) {
            is_same_base = false;
        }
        off_vec.push_back(off);
        off_const_vec.push_back(off_const - off_const0);
    }
    expr_t off;
    if (send.slots == 1 || !is_same_base) {
        off = shuffle_t::make(off_vec);
    } else {
        off = shuffle_t::make_broadcast(off_base0 + off_const0, send.slots)
                + shuffle_t::make(off_const_vec);
    }
    bool allow_fail = send.is_prefetch();
    auto _mask = mem_walker_->get_mask(
            0, send.access_size(), send.mask_size(), send.nmasks(), allow_fail);
    if (_mask.is_empty()) return stmt_t();

    auto _reg_buf = (send.is_prefetch()
                    ? expr_t()
                    : reg_buf_ + reg_layout_walker_->offset_bytes());
    auto ret = send(mem_buf_, off, _reg_buf, _mask);
    return ret;
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
