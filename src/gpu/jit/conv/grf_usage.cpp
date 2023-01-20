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

#include "gpu/jit/conv/grf_usage.hpp"

#include <sstream>

#include "gpu/jit/codegen/register_allocator.hpp"
#include "gpu/jit/ir/message.hpp"
#include "gpu/jit/ir/reorder.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

std::string to_string(grf_usage_label_t label) {
    switch (label) {
#define CASE(l) \
    case grf_usage_label_t::l: return #l;
        CASE(unknown)
        CASE(gmem_load)
        CASE(out_buf)
        CASE(reorder)
        CASE(reserved)
        CASE(reused_headers)
        CASE(slm_load)
        CASE(slm_store)
        CASE(tmp_vars)
        CASE(zero_points)
#undef CASE
        default: ir_error_not_expected();
    }
    return "";
}

std::ostream &operator<<(std::ostream &out, grf_usage_label_t label) {
    out << to_string(label);
    return out;
}

std::string grf_buf_usage_t::str() const {
    std::ostringstream oss;
    oss << "Buffers:";
    for (auto label : all_grf_usage_labels()) {
        int regs = total_regs(label);
        if (regs == 0) continue;
        oss << std::endl << " " << label << " (" << regs << "): ";
        bool is_first = true;
        for (auto &buf : sorted_bufs()) {
            if (get_label(buf) != label) continue;
            if (!is_first) oss << ", ";
            is_first = false;
            oss << buf << "[" << get_size(buf) << "]";
        }
    }
    return oss.str();
}

std::string grf_usage_t::str() const {
    std::vector<std::string> headers = {"Label", "Regs"};
    ir_utils::table_t table("GRF usage (registers):", headers);
    int total = 0;
    for (auto label : all_grf_usage_labels()) {
        int regs = regs_.at(label);
        if (regs == 0) continue;
        table << "  " + to_string(label) << regs << std::endl;
        total += regs;
    }
    table << "  Total" << total << std::endl;
    std::ostringstream oss;
    oss << table << std::endl;
    oss << buf_usage_;
    return oss.str();
}

class access_grf_usage_helper_t {
public:
    access_grf_usage_helper_t(const layout_t &mem_layout, int elems,
            int reg_bytes, bool is_slm, bool use_2d_send)
        : mem_type_size_(mem_layout.type().size())
        , reg_bytes_(reg_bytes)
        , is_slm_(is_slm)
        , use_2d_send_(use_2d_send) {
        init_message_size(mem_layout);
        init_payload_size(elems);
        init_header_size();
    }

    // This setting is related to dpasw loads. dpasw reuses registers between
    // fused threads so each of the fused threads need to load only half of the
    // data it will access.
    void enable_fused_eus_sharing() { enabled_fused_eus_sharing_ = true; }

    int payload_regs() const {
        int ret = payload_size_ / reg_bytes_;
        if (enabled_fused_eus_sharing_) ret = utils::div_up(ret, 2);
        return ret;
    }

    int header_regs_per_msg() const {
        return header_size_per_msg_ / reg_bytes_;
    }

    int header_regs() const {
        int ret = nmsgs_ * header_regs_per_msg();
        if (enabled_fused_eus_sharing_) ret = utils::div_up(ret, 2);
        return ret;
    }

private:
    void init_message_size(const layout_t &mem_layout) {
        auto l = mem_layout.innermost_block_layout();
        int block_bytes = (is_slm_ ? oword_bytes_ : hword_bytes_);
        int max_block_bytes = (is_slm_ ? 16 * oword_bytes_ : 8 * hword_bytes_);
        block_t b0;
        int b0_size = mem_type_size_;
        auto &mem_blocks = mem_layout.blocks();
        if (!mem_blocks.empty()) {
            b0 = mem_blocks[0];
            b0_size = b0.block * mem_type_size_;
        }
        if (use_2d_send_) {
            is_block_ = true;
            // It's hard to determine 2D block message decomposition at this
            // point but in general 2D block messages are larger so use 2x of a
            // regular block message (empirical estimate).
            msg_size_ = 2 * max_block_bytes;
            payload_bytes_per_elem_ = mem_type_size_;
        } else if (l.size() % block_bytes == 0) {
            is_block_ = true;
            msg_size_ = (l.size() % max_block_bytes == 0) ? max_block_bytes
                                                          : block_bytes;
            payload_bytes_per_elem_ = mem_type_size_;
        } else if (!b0.is_empty() && b0_size % block_bytes == 0) {
            is_block_ = true;
            msg_size_ = block_bytes;
            payload_bytes_per_elem_ = mem_type_size_;
        } else {
            ir_assert(!is_slm_) << "Unexpected scattered messages with SLM.";
            // Assume scattered byte SIMD16 load as the worst case. Check if
            // we can use byte x {1,2,4} messages.
            int slots = 16;
            int bytes_per_slot = 4;
            for (int x : {4, 2, 1}) {
                if (x < bytes_per_slot && mem_type_size_ != x) continue;
                if (b0_size % x == 0) {
                    msg_size_ = slots * x;
                    payload_bytes_per_elem_
                            = mem_type_size_ * (bytes_per_slot / x);
                    break;
                }
            }
            ir_assert(msg_size_ > 0);
        }
    }

    void init_payload_size(int elems) {
        int elems_per_msg = utils::div_up(msg_size_, mem_type_size_);
        int payload_per_msg = elems_per_msg * payload_bytes_per_elem_;
        int payload_per_msg_grf_aligned
                = utils::rnd_up(payload_per_msg, reg_bytes_);
        nmsgs_ = utils::div_up(elems * mem_type_size_, msg_size_);
        payload_size_ = nmsgs_ * payload_per_msg_grf_aligned;
    }

    void init_header_size() {
        if (is_block_) {
            // One register per header for block messages.
            header_size_per_msg_ = reg_bytes_;
        } else {
            // Assume SIMD16 with A64 address model.
            int slots = 16;
            int bytes_per_slot = sizeof(uint64_t);
            header_size_per_msg_
                    = utils::rnd_up(slots * bytes_per_slot, reg_bytes_);
        }
    }

    static const int oword_bytes_ = 16;
    static const int hword_bytes_ = 32;

    int mem_type_size_ = 0;
    int reg_bytes_ = 0;
    bool is_slm_ = false;
    bool use_2d_send_ = false;
    bool enabled_fused_eus_sharing_ = false;

    // Whether message is block or scattered.
    bool is_block_ = false;

    // Amount of memory that can be read by a single message from global memory.
    int msg_size_ = 0;

    // How many bytes are occupied by a single element in the message payload.
    int payload_bytes_per_elem_ = 0;

    // Size of GRF buffers for all messages to load data.
    int payload_size_ = 0;

    // Number of messages to load data.
    int nmsgs_ = 0;

    // Size of header buffer per message.
    int header_size_per_msg_ = 0;
};

// Helper class to provide GRF usage estimation.
class grf_usage_helper_t {
public:
    grf_usage_helper_t(const conv_config_t &cfg) : prb_(cfg.prb()), cfg_(cfg) {
        auto &tg_grid = cfg_.thread_group_grid();

        reg_bytes_ = cfg_.grf_size();
        tg_size_ = tg_grid.elems();

        bmnk_dim_helper_t h(cfg_);
        m_tg_dim_ = h.thread_group_dim('m');
        n_tg_dim_ = h.thread_group_dim('n');

        int b_iter_blk = h.iter_dim('b');
        int m_iter_blk = h.iter_dim('m');
        int n_iter_blk = h.iter_dim('n');
        int k_iter_blk = h.iter_dim('k');

        if (!cfg_.ow_kw_grf_cache()) {
            a_thr_elems_ = b_iter_blk * m_iter_blk * k_iter_blk;
        } else {
            ir_assert(!cfg_.slm().a());
            int a_m_blk = (prb_.sw * (m_iter_blk - 1)
                    + (prb_.kw - 1) * (1 + prb_.dw) + 1);
            int a_k_blk = utils::div_up(k_iter_blk, prb_.kw);
            a_thr_elems_ = b_iter_blk * a_m_blk * a_k_blk;
        }

        b_thr_elems_ = b_iter_blk * n_iter_blk * k_iter_blk;
        c_thr_elems_ = b_iter_blk * m_iter_blk * n_iter_blk;
        a_tg_elems_ = a_thr_elems_ * m_tg_dim_;
        b_tg_elems_ = b_thr_elems_ * n_tg_dim_;
        a_subtile_elems_ = utils::div_up(a_thr_elems_, cfg_.subtiles().a());
        b_subtile_elems_ = utils::div_up(b_thr_elems_, cfg_.subtiles().b());
        can_reliably_use_dpasw_ = can_reliably_use_dpasw(h);
    }

    grf_usage_t estimate() const {
        int max_reuse_header_regs = 0;
        int a_slm_store_payload_regs = 0;
        int b_slm_store_payload_regs = 0;

        int c_buf_regs = estimate_c_buf_regs();
        int gmem_load_regs = estimate_gmem_load_regs(max_reuse_header_regs);
        int slm_store_regs = estimate_slm_store_regs(a_slm_store_payload_regs,
                b_slm_store_payload_regs, max_reuse_header_regs);
        int slm_load_regs = estimate_slm_load_regs(max_reuse_header_regs);
        int reorder_regs = estimate_reorder_regs(
                a_slm_store_payload_regs, b_slm_store_payload_regs);
        int zp_regs = estimate_zero_point_regs();

        grf_usage_t info(cfg_.grf_size());
        info.add(grf_usage_label_t::out_buf, c_buf_regs);
        info.add(grf_usage_label_t::gmem_load, gmem_load_regs);
        info.add(grf_usage_label_t::slm_store, slm_store_regs);
        info.add(grf_usage_label_t::slm_load, slm_load_regs);
        info.add(grf_usage_label_t::reorder, reorder_regs);
        info.add(grf_usage_label_t::reused_headers, max_reuse_header_regs);
        info.add(grf_usage_label_t::reserved, constants::reserved_regs);
        info.add(grf_usage_label_t::zero_points, zp_regs);
        return info;
    }

private:
    int estimate_c_buf_regs() const {
        int c_bytes = c_thr_elems_ * prb_.acc_data_type_size;
        return utils::div_up(c_bytes, reg_bytes_);
    }

    int estimate_gmem_load_regs(int &max_reuse_header_regs) const {
        int regs = 0;
        bool use_a_2d_send = can_use_a_2d_send(cfg_);
        bool use_b_2d_send = can_use_b_2d_send(cfg_);
        for (bool is_a : {true, false}) {
            bool use_slm = ab_use_slm(is_a);
            int per_thr_elems = utils::div_up(ab_tg_elems(is_a), tg_size_);
            int load_elems = (use_slm ? per_thr_elems : ab_subtile_elems(is_a));
            auto layout = get_gmem_layout(is_a);
            bool use_2d_send = (is_a ? use_a_2d_send : use_b_2d_send);
            access_grf_usage_helper_t load(layout, load_elems, reg_bytes_,
                    /*is_slm=*/false, use_2d_send);
            if (is_a && !use_slm && can_reliably_use_dpasw_)
                load.enable_fused_eus_sharing();
            int mult = (use_slm ? cfg_.slm().gmem_bufs() : 1);
            regs += mult * load.payload_regs();
            if (cfg_.pipeline().reuse_headers()) {
                max_reuse_header_regs = std::max(
                        max_reuse_header_regs, load.header_regs_per_msg());
            } else {
                int subtiles
                        = (is_a ? cfg_.subtiles().a() : cfg_.subtiles().b());
                int mult = (use_slm ? 1 : subtiles);
                regs += mult * load.header_regs();
                if (cfg_.prefetch()) {
                    access_grf_usage_helper_t prefetch(layout, per_thr_elems,
                            reg_bytes_, /*is_slm=*/false, use_2d_send);
                    regs += prefetch.header_regs();
                }
            }
        }
        return regs;
    }

    int estimate_slm_store_regs(int &a_payload_regs, int &b_payload_regs,
            int &max_reuse_header_regs) const {
        int regs = 0;
        for (bool is_a : {true, false}) {
            if (!ab_use_slm(is_a)) continue;

            int per_thr_elems = utils::div_up(ab_tg_elems(is_a), tg_size_);
            int bytes = per_thr_elems * ab_type_size(is_a);
            auto slm_layout = dummy_slm_layout(bytes);
            access_grf_usage_helper_t store(slm_layout, bytes, reg_bytes_,
                    /*is_slm=*/true, /*use_2d_send=*/false);
            int &payload_regs = (is_a ? a_payload_regs : b_payload_regs);
            payload_regs = store.payload_regs();
            if (cfg_.pipeline().reuse_headers()) {
                max_reuse_header_regs = std::max(
                        max_reuse_header_regs, store.header_regs_per_msg());
            } else {
                regs += store.header_regs();
            }
        }
        return regs;
    }

    int estimate_slm_load_regs(int &max_reuse_header_regs) const {
        int regs = 0;
        for (bool is_a : {true, false}) {
            if (!ab_use_slm(is_a)) continue;

            int bytes = ab_subtile_elems(is_a) * ab_type_size(is_a);
            auto slm_layout = dummy_slm_layout(bytes);
            access_grf_usage_helper_t load(slm_layout, bytes, reg_bytes_,
                    /*is_slm=*/true, /*use_2d_send=*/false);
            if (is_a && can_reliably_use_dpasw_)
                load.enable_fused_eus_sharing();
            regs += load.payload_regs();
            if (cfg_.pipeline().reuse_headers()) {
                max_reuse_header_regs = std::max(
                        max_reuse_header_regs, load.header_regs_per_msg());
            } else {
                regs += load.header_regs();
            }
        }

        return regs;
    }

    // Extra registers for GRF <-> GRF reorders.
    // Estimates upper bound for A/B reorders to temporary buffers.
    int estimate_reorder_regs(int a_payload_regs, int b_payload_regs) const {
        if (!cfg_.allow_a_grf_reorder() && !cfg_.allow_b_grf_reorder())
            return 0;

        int regs = 0;
        if (prb_.is_bwd_w) {
            // Hardcode the size of the temporary reorder buffer for BWD_W to
            // avoid suboptimal performance.
            int bwd_w_reorder_regs = 16;
            regs += bwd_w_reorder_regs;
        }

        for (bool is_a : {true, false}) {
            bool allow_grf_reorder = (is_a ? cfg_.allow_a_grf_reorder()
                                           : cfg_.allow_b_grf_reorder());
            if (!allow_grf_reorder) continue;
            int reorder_regs = 0;
            if (ab_use_slm(is_a)) {
                int &payload_regs = (is_a ? a_payload_regs : b_payload_regs);
                reorder_regs = payload_regs;
            } else {
                int size = ab_subtile_elems(is_a) * ab_type_size(is_a);
                reorder_regs = utils::div_up(size, reg_bytes_);
            }
            regs += reorder_regs;
        }

        return regs;
    }

    int estimate_zero_point_regs() const {
        if (!prb_.zp_cfg.do_src_compensation) return 0;
        int sp_iter_dim = 1;
        for (auto *name : {"ow", "iw", "osp"}) {
            sp_iter_dim *= cfg_.iter_dim(name);
        }
        int subtiles = cfg_.subtiles().a() * cfg_.subtiles().b();
        int zp_mask0_regs = 2
                * utils::div_up(
                        sp_iter_dim * (int)sizeof(uint32_t), reg_bytes_);
        int zp_mask1_regs = subtiles
                * utils::div_up(
                        sp_iter_dim * (int)sizeof(uint16_t), reg_bytes_);
        int zp_buf_regs = subtiles * utils::div_up(128, reg_bytes_);
        int zp_header_regs = subtiles;
        int zp_let_regs = 4;
        return zp_mask0_regs + zp_mask1_regs + zp_buf_regs + zp_header_regs
                + zp_let_regs;
    }

    layout_t get_gmem_layout(bool is_a) const {
        auto layout = (is_a ? cfg_.a_layout() : cfg_.b_layout()).compute();
        bool is_src_dst = is_a || prb_.is_bwd_w;
        if (is_src_dst && prb_.is_dw) {
            auto &blocks = layout.blocks();
            if (!blocks.empty()) {
                auto &b0 = blocks[0];
                std::vector<block_t> new_blocks(
                        blocks.begin() + 1, blocks.end());
                // Remove the innermost block of channels for depthwise
                // convolution.
                if (b0.dim_idx == 2 && b0.block == 1) {
                    layout = layout_t(layout.type(), layout.ndims(),
                            layout.offset(), new_blocks,
                            /*do_normalize=*/false);
                }
            }
        }
        return layout;
    }

    int ab_type_size(bool is_a) const {
        auto ret = is_a ? prb_.a_data_type_size : prb_.b_data_type_size;
        if (prb_.is_s32_accumulator() && cfg_.fma_kind() == fma_kind_t::mad) {
            // s8/u8 is converted to dword-strided word for mad.
            ir_assert(ret == 1);
            ret = 4;
        }
        return ret;
    }

    int ab_tg_elems(bool is_a) const {
        return is_a ? a_tg_elems_ : b_tg_elems_;
    }

    int ab_thr_elems(bool is_a) const {
        return is_a ? a_thr_elems_ : b_thr_elems_;
    }

    int ab_subtile_elems(bool is_a) const {
        return is_a ? a_subtile_elems_ : b_subtile_elems_;
    }

    int ab_use_slm(bool is_a) const {
        return is_a ? cfg_.slm().a() : cfg_.slm().b();
    }

    bool can_reliably_use_dpasw(const bmnk_dim_helper_t &h) {
        if (cfg_.fma_kind() != fma_kind_t::dpasw) return false;
        if (!cfg_.slm().a()) return false;
        int m_tg_bytes = h.thread_group_dim('m') * h.iter_dim('m')
                * prb_.a_data_type_size;
        int m_thr_bytes
                = ir_utils::safe_divide(m_tg_bytes, h.thread_group_dim('m'));
        int owordx16_size = 256;
        if (cfg_.a_layout().compute().innermost_block_layout().size()
                < owordx16_size)
            return false;
        int k_iter_blk = h.iter_dim('k');
        if (m_thr_bytes * k_iter_blk % owordx16_size != 0) return false;
        int nmsgs = m_thr_bytes * k_iter_blk / owordx16_size;
        if (nmsgs % 2 != 0) return false;
        return true;
    }

    layout_t dummy_slm_layout(int size) const {
        int inner_block = 16; // In bytes.
        int outer_block = utils::div_up(size, inner_block);
        std::vector<block_t> blocks;
        blocks.emplace_back(0, inner_block, 1);
        blocks.emplace_back(1, outer_block, inner_block);
        blocks.emplace_back(0, 1, size);
        blocks.emplace_back(1, 1, size);
        return layout_t(type_t::byte(), 2, 0, blocks, /*do_normalize=*/false);
    }

    const conv_problem_t &prb_;
    const conv_config_t &cfg_;

    int reg_bytes_;
    int tg_size_;
    int m_tg_dim_;
    int n_tg_dim_;
    int a_tg_elems_;
    int b_tg_elems_;
    int a_thr_elems_;
    int b_thr_elems_;
    int c_thr_elems_;
    int a_subtile_elems_;
    int b_subtile_elems_;
    bool can_reliably_use_dpasw_;
};

grf_usage_t estimate_grf_usage(const conv_config_t &cfg) {
    grf_usage_helper_t helper(cfg);
    return helper.estimate();
}

class ir_usage_analyzer_t : public ir_visitor_t {
public:
    ir_usage_analyzer_t(int grf_size)
        : grf_size_(grf_size), buf_usage_(grf_size) {}

    void analyze(const stmt_t &stmt, bool allow_errors = false) {
        visit(stmt);
        if (!is_invalid_) {
            if (peak_headers_ <= 1) {
                std::vector<expr_t> header_bufs;
                for (auto &buf : buf_usage_.bufs()) {
                    if (is_header(buf)) header_bufs.push_back(buf);
                }
                int peak_header_usage_ = 0;
                for (auto &buf : header_bufs) {
                    peak_header_usage_ = std::max(
                            peak_header_usage_, buf_usage_.get_size(buf));
                    buf_usage_.remove(buf);
                }
            }
        }
        if (!verify(allow_errors)) is_invalid_ = true;
    }

    void _visit(const alloc_t &obj) override {
        if (is_invalid_) return;
        int size = (obj.kind == alloc_kind_t::grf ? obj.size : 0);
        size = utils::rnd_up(size, grf_size_);
        mem_usage_guard_t alloc_guard(&alloc_usage_, &peak_alloc_usage_, size);
        mem_usage_guard_t guard(&grf_usage_, &peak_grf_usage_, size);
        if (size > 0) {
            buf_usage_.add(obj.buf, obj.size, grf_usage_label_t::unknown);
            mark_known_bufs(obj.buf);
        }
        mem_usage_guard_t header_guard;
        if (is_header(obj.buf))
            header_guard = mem_usage_guard_t(&headers_, &peak_headers_, 1);
        ir_visitor_t::_visit(obj);
    }

    void _visit(const func_call_t &obj) override {
        if (is_invalid_) return;
        auto &func = obj.func;
        if (auto *reorder = func.as_ptr<reorder_t>()) {
            auto &src = get_base(reorder_t::arg_src_buf(obj));
            auto &dst = get_base(reorder_t::arg_dst_buf(obj));
            mark_bufs(*reorder, src, dst);
        } else if (auto *send = func.as_ptr<send_t>()) {
            if (!send_t::arg_header_buf(obj).is<var_t>()) {
                is_invalid_ = true;
                return;
            }
            auto &buf = get_base(send_t::arg_reg_buf(obj));
            auto &header = get_base(send_t::arg_header_buf(obj));
            mark_bufs(*send, buf, header);
        } else if (is_func_call<dpas_t>(obj)) {
            auto &dst = get_base(dpas_t::arg_dst(obj));
            auto &src1 = get_base(dpas_t::arg_src1(obj));
            auto &src2 = get_base(dpas_t::arg_src2(obj));
            mark_fma_bufs(dst, src1, src2);
        } else if (is_func_call<mad_t>(obj)) {
            auto &dst = get_base(mad_t::arg_dst(obj));
            auto &src1 = get_base(mad_t::arg_src1(obj));
            auto &src2 = get_base(mad_t::arg_src2(obj));
            mark_fma_bufs(dst, src1, src2);
        }
    }

    void _visit(const let_t &obj) override {
        if (is_invalid_) return;
        int size = (obj.value.is_empty() ? 0 : obj.var.type().size());
        size = utils::rnd_up(size, reg_allocator_t::granularity);
        mem_usage_guard_t guard(&grf_usage_, &peak_grf_usage_, size);
        ir_visitor_t::_visit(obj);
    }

    void _visit(const stmt_group_t &obj) override {
        if (is_invalid_) return;
        if (obj.label == stmt_label_t::c_store()) {
            // Do not analyze C store consumption for simplicity. Assume there
            // is enough space after releasing A/B and other buffers after the
            // main loop.
            return;
        }
        ir_visitor_t::_visit(obj);
    }

    void _visit(const store_t &obj) override {
        if (is_invalid_) return;
        auto loads = find_objects<load_t>(obj);
        for (auto &l : loads) {
            if (obj.buf.is_same(l.as<load_t>().buf)) {
                set_label(obj.buf, grf_usage_label_t::tmp_vars);
                break;
            }
        }
        ir_visitor_t::_visit(obj);
    }

    grf_usage_t get_grf_usage(int external_regs) const {
        if (is_invalid_) return grf_usage_t();
        grf_usage_t info(grf_size_);
        info.add(buf_usage_);
        info.add(grf_usage_label_t::reserved, external_regs);
        info.add(grf_usage_label_t::tmp_vars,
                utils::div_up(peak_grf_usage_ - peak_alloc_usage_, grf_size_));
        if (peak_headers_ <= 1) {
            info.add(grf_usage_label_t::reused_headers, peak_header_usage_);
        }
        return info;
    }

private:
    bool verify(bool allow_errors) const {
        if (is_invalid_) {
            if (!allow_errors)
                ir_error_not_expected() << "Can't collect GRF usage.";
            return false;
        }
        for (auto &buf : buf_usage_.bufs()) {
            if (buf_usage_.get_label(buf) != grf_usage_label_t::unknown)
                continue;
            if (!allow_errors)
                ir_error_not_expected() << "Buffer doesn't have label: " << buf;
            return false;
        }
        return true;
    }

    bool is_buffer(const expr_t &buf) const { return buf_usage_.has(buf); }

    bool is_header(const expr_t &buf) const {
        if (!is_buffer(buf)) return false;
        auto &name = buf.as<var_t>().name;
        return name.find("h_") == 0;
    }

    bool should_skip_if_set(const expr_t &buf, grf_usage_label_t label) const {
        if (is_known_buf(buf)) return true;
        switch (label) {
            case grf_usage_label_t::tmp_vars:
            case grf_usage_label_t::slm_store: return true;
            default: return false;
        }
    }

    void set_label(const expr_t &buf, grf_usage_label_t label) {
        if (is_invalid_) return;
        bool skip_if_set = should_skip_if_set(buf, label);
        auto buf_label = buf_usage_.get_label(buf);
        if (utils::one_of(buf_label, grf_usage_label_t::unknown, label)) {
            buf_usage_.set_label(buf, label);
        } else {
            if (skip_if_set) return;
            ir_error_not_expected()
                    << "Label already set. Buffer: " << buf
                    << ", old label: " << buf_label << ", new label: " << label;
        }
    }

    void mark_known_bufs(const expr_t &buf) {
        if (is_invalid_) return;
        ir_assert(is_buffer(buf));
        auto &name = buf.as<var_t>().name;
        if (name.find("b_reduce") == 0) {
            set_label(buf, grf_usage_label_t::out_buf);
        } else if (name.find("zp_") == 0 || name.find("src_zp") == 0) {
            set_label(buf, grf_usage_label_t::zero_points);
        }
    }

    bool is_known_buf(const expr_t &buf) const {
        ir_assert(is_buffer(buf));
        auto &name = buf.as<var_t>().name;
        if (name.find("zp_") == 0) return true;
        if (name.find("src_zp") == 0) return true;
        if (name.find("b_reduce")) return true;
        return false;
    }

    void mark_bufs(
            const reorder_t &reorder, const expr_t &src, const expr_t &dst) {
        if (is_invalid_) return;
        ir_assert(is_buffer(src));
        ir_assert(is_buffer(dst));
        set_label(dst, grf_usage_label_t::reorder);
    }

    void mark_bufs(
            const send_t &send, const expr_t &buf, const expr_t &header) {
        if (is_invalid_) return;
        if (!buf.is_empty()) ir_assert(is_buffer(buf));
        ir_assert(is_buffer(header));
        ir_assert(is_header(header));
        grf_usage_label_t label = grf_usage_label_t::unknown;
        if (buf.is_empty()) {
            label = grf_usage_label_t::gmem_load;
        } else if (buf_usage_.get_label(buf)
                == grf_usage_label_t::zero_points) {
            label = grf_usage_label_t::zero_points;
        } else if (send.is_slm()) {
            label = (send.is_load() ? grf_usage_label_t::slm_load
                                    : grf_usage_label_t::slm_store);
        } else {
            if (!send.is_load() && !send.is_load_2d()) {
                is_invalid_ = true;
                return;
            }
            label = grf_usage_label_t::gmem_load;
        }
        if (!buf.is_empty()) set_label(buf, label);
        set_label(header, label);
    }

    void mark_fma_bufs(
            const expr_t &dst, const expr_t &src1, const expr_t &src2) {
        if (is_invalid_) return;
        ir_assert(is_buffer(dst));
        ir_assert(is_buffer(src1));
        ir_assert(is_buffer(src2));
        set_label(dst, grf_usage_label_t::out_buf);
    }

    int grf_size_;
    bool is_invalid_ = false;
    grf_buf_usage_t buf_usage_;

    int grf_usage_ = 0;
    int alloc_usage_ = 0;

    int peak_grf_usage_ = 0;
    int peak_alloc_usage_ = 0;

    int headers_ = 0;
    int peak_headers_ = 0;
    int peak_header_usage_ = 0; // Unset when headers are not reused.
};

grf_usage_t get_grf_usage(const stmt_t &body, int grf_size) {
    ir_usage_analyzer_t analyzer(grf_size);
    analyzer.visit(body);
    return analyzer.get_grf_usage(0);
}

void compare(const grf_usage_t &est_usage, const grf_usage_t &ir_usage,
        const ir_usage_analyzer_t &analyzer) {
    std::vector<std::string> headers
            = {"Label", "Estimated regs", "IR regs", "Status"};
    ir_utils::table_t table("Compare GRF usage:", headers);
    int est_total = 0;
    int ir_total = 0;
    for (auto label : all_grf_usage_labels()) {
        int est_regs = est_usage.get(label);
        int ir_regs = ir_usage.get(label);
        table << "  " + to_string(label) << est_regs << ir_regs;
        table << (ir_regs > est_regs ? "FAIL" : "");
        table << std::endl;
        est_total += est_regs;
        ir_total += ir_regs;
    }
    table << "  Total" << est_total << ir_total;
    table << (ir_total > est_total ? "FAIL" : "");
    table << std::endl;
    ir_trace() << table << std::endl;
    ir_trace() << ir_usage.buf_usage() << std::endl;
}

void verify_grf_usage(
        const conv_config_t &cfg, const stmt_t &body, int external_usage) {
    ir_usage_analyzer_t analyzer(cfg.grf_size());
    analyzer.analyze(body);

    auto ir_info = analyzer.get_grf_usage(external_usage);
    auto est_info = estimate_grf_usage(cfg);
    compare(est_info, ir_info, analyzer);
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
