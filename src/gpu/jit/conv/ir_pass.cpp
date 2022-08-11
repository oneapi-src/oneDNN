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

#include "gpu/jit/conv/ir_pass.hpp"

#include "gpu/jit/conv/builder_utils.hpp"
#include "gpu/jit/conv/message_support.hpp"
#include "gpu/jit/conv/reorder_support.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

class dpasw_injector_t {
public:
    dpasw_injector_t(ngen::HW hw, const stmt_t &load_mul_stmt,
            const expr_t &c_buf, const stmt_t &c_store_stmt,
            alloc_updater_t &alloc_updater, const expr_t &tg_idx0)
        : hw_(hw)
        , load_mul_stmt_(load_mul_stmt)
        , c_buf_(c_buf)
        , c_store_stmt_(c_store_stmt)
        , alloc_updater_(alloc_updater)
        , tg_idx0_(tg_idx0) {}

    const stmt_t &load_mul_stmt() const { return load_mul_stmt_; }

    const stmt_t &c_store_stmt() const { return c_store_stmt_; }

    void inject() {
        expr_t src2_base;
        if (!extract_dpas_calls(src2_base)) return;

        grf_permutation_t grf_perm;

        bool was_injected = false;
        int dpas_count = int(dpas_infos_.size());
        for (int i = 0; i < dpas_count;) {
            if (i + 1 < dpas_count) {
                auto &a = dpas_infos_[i];
                auto &b = dpas_infos_[i + 1];
                if (try_convert_to_dpasw(a, b, grf_perm)) {
                    was_injected = true;
                    i += 2;
                    continue;
                }
            }
            if (try_convert_to_dpasw(dpas_infos_[i], grf_perm)) {
                was_injected = true;
            }
            ++i;
        }
        // Nothing to update, no dpas -> dpasw transformation.
        if (!was_injected) return;

        int src2_size = 0;
        object_map_t<stmt_t, int> send2off;
        std::function<int(const stmt_t &)> get_src2_off;
        get_src2_off = [&](const stmt_t &s) {
            auto &si = find_send_info(s);
            if (!si.base_call.is_empty()) return get_src2_off(si.base_call);
            if (!si.prev_send.is_empty()) return get_src2_off(si.prev_send);

            auto it = send2off.find(s);
            if (it != send2off.end()) return it->second;

            auto ret = send2off.insert({s, src2_size});
            if (!ret.second) return ret.first->second;

            int new_size = si.new_reg_buf_size();
            src2_size += new_size;
            return ret.first->second;
        };
        for (auto &si : send_infos_) {
            if (!si.reg_buf_base().is_equal(src2_base)) continue;

            int src2_off = get_src2_off(si.call);
            auto src2_sub = src2_base[src2_off];
            auto new_call = si.new_call;
            if (!new_call.is_empty()) {
                new_call = substitute(
                        new_call, send_t::arg_reg_buf(new_call), src2_sub, 1);
            }

            load_mul_stmt_ = substitute(load_mul_stmt_, si.call, new_call, 1);
            for (auto &d : si.dpas_consumers) {
                auto &di = find_dpas_info(d);
                ir_assert(si.promote_to_dpasw == di.promote_to_dpasw)
                        << "Both send and dpas must be updated.";
                if (di.update_applied) {
                    ir_error_not_expected() << "Can it happen?";
                    continue;
                }
                auto new_call = di.new_call;
                new_call = substitute(new_call, dpas_t::arg_src2(new_call),
                        src2_sub[di.src2_relative_off], 1);
                load_mul_stmt_
                        = substitute(load_mul_stmt_, di.call, new_call, 1);
                di.update_applied = true;
            }
        }

        // Update src2 size after applying send updates.
        alloc_updater_.resize(src2_base, src2_size);

        // Apply permutation to C buffer.
        alloc_updater_.add_attr(c_buf_,
                grf_permute_attr_t::make(
                        std::make_shared<grf_permutation_t>(grf_perm)));
    }

private:
    struct send_info_t {
        send_info_t() = default;

        send_info_t(const stmt_t &call) : call(call), new_call(call) {}

        const send_t &send() const {
            return call.as<func_call_t>().func.as<send_t>();
        }

        const send_t &new_send() const {
            ir_assert(!new_call.is_same(call));
            return new_call.as<func_call_t>().func.as<send_t>();
        }

        const std::vector<expr_t> &args() const {
            return call.as<func_call_t>().args;
        }

        const expr_t &reg_buf() const { return send_t::arg_reg_buf(call); }

        const expr_t &reg_buf_base() const {
            return reg_buf().as<ptr_t>().base;
        }

        int reg_buf_size() const { return send().payload_size(); }

        int new_reg_buf_size() const {
            if (new_call.is_same(call)) return reg_buf_size();
            return new_send().payload_size();
        }

        void set_new_call(const stmt_t &s, const stmt_t &base = stmt_t()) {
            if (!promote_to_dpasw) {
                promote_to_dpasw = true;
                new_call = s;
                base_call = base;
                return;
            }
            ir_assert(new_call.is_equal(s));
            ir_assert(base_call.is_equal(base));
        }

        void set_prev_send(const stmt_t &s) {
            int prev_size
                    = s.as<func_call_t>().func.as<send_t>().payload_size();
            if (reg_buf_size() != prev_size) return;
            prev_send = s;
        }

        stmt_t call;
        std::vector<stmt_t> dpas_consumers;

        bool promote_to_dpasw = false;
        stmt_t new_call;
        stmt_t base_call;
        stmt_t prev_send;
    };

    struct dpas_info_t {
        dpas_info_t() = default;

        dpas_info_t(const stmt_t &call) : call(call), new_call(call) {}

        const dpas_t &dpas() const {
            return call.as<func_call_t>().func.as<dpas_t>();
        }

        const std::vector<expr_t> &args() const {
            return call.as<func_call_t>().args;
        }

        const expr_t &src1_buf() const { return dpas_t::arg_src1(call); }

        const expr_t &src2_buf() const { return dpas_t::arg_src2(call); }

        int src2_size() const { return dpas().src2_size(); }

        void set_new_call(const stmt_t &s, int src2_relative_off) {
            if (!promote_to_dpasw) {
                promote_to_dpasw = true;
                this->src2_relative_off = src2_relative_off;
                new_call = s;
                return;
            }
            ir_assert(this->src2_relative_off == src2_relative_off);
            ir_assert(new_call.is_equal(s));
        }

        stmt_t call;
        stmt_t send_producer;

        bool promote_to_dpasw = false;
        bool update_applied = false;
        int src2_relative_off = 0;
        stmt_t new_call;
    };

    send_info_t &find_send_info(const stmt_t &s) {
        for (auto &si : send_infos_)
            if (si.call.is_same(s)) return si;
        ir_error_not_expected();
        return send_infos_.front();
    }

    dpas_info_t &find_dpas_info(const stmt_t &s) {
        for (auto &si : dpas_infos_)
            if (si.call.is_same(s)) return si;
        ir_error_not_expected();
        return dpas_infos_.front();
    }
    static bool is_send(const stmt_t &s, send_info_t &info) {
        if (!is_func_call<send_t>(s)) return false;
        info = send_info_t(s);
        return true;
    }

    static bool is_dpas(const stmt_t &s, dpas_info_t &info) {
        if (!is_func_call<dpas_t>(s)) return false;
        if (dpas_t::is_dp4a_call(s)) return false;
        info = dpas_info_t(s);
        return true;
    }

    bool extract_dpas_calls(expr_t &src2_base) {
        object_eq_map_t<expr_t, stmt_t> buf2send;

        auto set_src2_base = [&](const expr_t &ptr) {
            auto &ptr_base = ptr.as<ptr_t>().base;
            if (src2_base.is_empty()) {
                src2_base = ptr_base;
                return;
            }
            ir_assert(src2_base.is_same(ptr_base));
        };

        // Iterate through dpas and send calls.
        auto stmt_vec = flatten_statements(load_mul_stmt_);
        for (auto &s : stmt_vec) {
            send_info_t send_info;
            if (is_send(s, send_info)) {
                auto &buf = send_info.reg_buf();
                stmt_t prev_send;
                auto it = buf2send.find(buf);
                if (it != buf2send.end()) prev_send = it->second;
                buf2send[buf] = s;
                send_infos_.push_back(send_info);
                if (!prev_send.is_empty()) {
                    send_infos_.back().set_prev_send(prev_send);
                }
                continue;
            }
            dpas_info_t dpas_info;
            if (is_dpas(s, dpas_info)) {
                set_src2_base(dpas_info.src2_buf());
                auto &buf = dpas_info.src2_buf();
                auto it = buf2send.find(buf);
                if (it == buf2send.end()) continue;
                auto &send_info = find_send_info(it->second);
                // For simplicity require full size match between load and dpas
                // instructions. That is dpas src2 buffer should be fully
                // loaded by the corresponding send message.
                if (send_info.reg_buf_size() != dpas_info.src2_size()) {
                    ir_warning() << "Can't inject dpasw: different register "
                                    "sizes in send and dpas."
                                 << std::endl;
                    return false;
                }
                dpas_info.send_producer = send_info.call;
                send_info.dpas_consumers.push_back(s);
                dpas_infos_.push_back(dpas_info);
            }
        }
        return true;
    }

    // Checks for the following pattern:
    //    dpas.sxr(a_dst, a_src0, src1, src2)
    //    dpas.sxr(b_dst, b_src0, src1, src2 + s * r * 4)
    static bool can_convert_to_dpasw(
            const dpas_info_t &a, const dpas_info_t &b) {
        if (!a.dpas().is_equal(b.dpas())) return false;
        if (!a.src1_buf().is_equal(b.src1_buf())) return false;

        auto src2_off0 = to_cpp<int>(a.src2_buf().as<ptr_t>().off);
        auto src2_off1 = to_cpp<int>(b.src2_buf().as<ptr_t>().off);

        if (src2_off1 - src2_off0 != a.src2_size()) return false;

        return true;
    }

    bool try_convert_to_dpasw(
            dpas_info_t &a, dpas_info_t &b, grf_permutation_t &grf_perm) {
        if (hw_ >= ngen::HW::XeHPC) return false;

        // Check if DPAS -> DPASW transformation is possible.
        if (!can_convert_to_dpasw(a, b)) return false;

        // Perform the transformation:
        // Before:
        //   send(slm, a_off, src2[0])
        //   send(slm, b_off, src2[s * r * 4])
        //   dpas.sxr(a_dst, a_src0, src1, src2[0])
        //   dpas.sxr(b_dst, b_src0, src1, src2[s * r * 4])
        // After:
        //   send(slm, a_off + (tg_idx0 % 2) * (b_off - a_off), src2)
        //   dpasw.sxr(p_a_dst, p_a_src0, src1, src2[0])
        //   dpasw.sxr(p_b_dst, p_b_src0, src1, src2[s * r * 4 / 2])
        // Where:
        //   p_a_dst[:] = a_dst[0:rcount / 2] + b_dst[0:rcount / 2]
        //   p_b_dst[:] = a_dst[rcount / 2:rcount] + b_dst[rcount / 2:rcount]
        ir_assert(a.dpas().is_equal(b.dpas()));
        auto _dpasw = dpas_t::make_dpasw(a.dpas());
        auto &dpasw = _dpasw.as<dpas_t>();

        auto a_args = a.args();
        auto b_args = b.args();
        dpas_t::arg_src2(b_args) -= dpasw.src2_size();

        a.set_new_call(dpasw.call(a.args()), 0);
        b.set_new_call(dpasw.call(b_args), dpasw.src2_size());

        // Record permutation for registers to apply it for the destination
        // store later.
        const auto grf_size = ngen::GRF::bytes(hw_);
        const auto rcount = a.dpas().rcount;
        for (int j = 0; j < rcount; j++) {
            int k = j % (rcount / 2);
            auto a_old = dpas_t::arg_dst(a_args) + grf_size * j;
            auto b_old = dpas_t::arg_dst(b_args) + grf_size * j;
            expr_t grf_new;
            if (j < rcount / 2) {
                grf_new = dpas_t::arg_dst(a_args)[grf_size * k];
            } else {
                grf_new = dpas_t::arg_dst(b_args)[grf_size * k];
            }
            set_grf_permute(grf_perm, a_old, grf_new);
            set_grf_permute(grf_perm, b_old, grf_new + grf_size * rcount / 2);
        }

        auto &a_send = find_send_info(a.send_producer);
        auto &b_send = find_send_info(b.send_producer);

        auto &a_mem_off = send_t::arg_mem_off(a_send.call);
        auto &b_mem_off = send_t::arg_mem_off(b_send.call);
        auto ab_addr_diff = simplify(b_mem_off - a_mem_off);
        ir_assert(is_const(ab_addr_diff));

        auto new_send_args = a_send.args();
        send_t::arg_mem_off(new_send_args)
                += (tg_idx0_ % 2) * to_cpp<int64_t>(ab_addr_diff);

        a_send.set_new_call(a_send.send().call(new_send_args));
        b_send.set_new_call(stmt_t(), a_send.call);

        return true;
    }

    void set_grf_permute(grf_permutation_t &grf_perm, const expr_t &old_grf,
            const expr_t &new_grf) {
        int old_off = to_cpp<int>(old_grf.as<ptr_t>().off);
        int new_off = to_cpp<int>(new_grf.as<ptr_t>().off);

        const int grf_size = ngen::GRF::bytes(hw_);

        ir_assert(old_off % grf_size == 0)
                << "Must be aligned to GRF boundary.";
        ir_assert(new_off % grf_size == 0)
                << "Must be aligned to GRF boundary.";

        old_off /= grf_size;
        new_off /= grf_size;

        grf_perm.set_permute(old_off, new_off);
    }

    static bool can_convert_to_dpasw(const dpas_info_t &a_dpas,
            const send_info_t &a_send, const expr_t &tg_idx0) {
        if (contains_object(a_send.call, tg_idx0)) return false;
        return a_dpas.dpas().rcount % 2 == 0;
    }

    static func_t create_half_send(const send_t &send) {
        ir_assert(send.type.elems() % 2 == 0) << "Can't create half-send.";
        auto _s = send_t::make(send.hw, send.op, send.address,
                send.type.with_elems(send.type.elems() / 2), send.slots);
        auto &s = _s.as<send_t>();
        ir_assert(s.is_supported())
                << "Can't find send reading half of the original send.";
        MAYBE_UNUSED(s);
        return _s;
    }

    bool try_convert_to_dpasw(dpas_info_t &a, grf_permutation_t &grf_perm) {
        if (hw_ >= ngen::HW::XeHPC) return false;
        if (!can_convert_to_dpasw(a, find_send_info(a.send_producer), tg_idx0_))
            return false;

        // Perform the transformation:
        // Before:
        //   send(slm, a_off, src2[0])
        //   dpas.sxr(a_dst, a_src0, src1, src2[0])
        // After:
        //   send(slm, a_off + (tg_idx0 % 2) * (s * r * 4 / 2), src2)
        //   dpasw.sxr(a_dst, a_src0, src1, src2[0])

        auto _dpasw = dpas_t::make_dpasw(a.dpas());
        auto &dpasw = _dpasw.as<dpas_t>();

        a.set_new_call(dpasw.call(a.args()), 0);

        auto &a_send = find_send_info(a.send_producer);
        auto new_send_args = a_send.args();
        send_t::arg_mem_off(new_send_args)
                += (tg_idx0_ % 2) * (a.src2_size() / 2);
        a_send.set_new_call(
                create_half_send(a_send.send()).call(new_send_args));

        return true;
    }

    ngen::HW hw_;
    stmt_t load_mul_stmt_;
    expr_t c_buf_;
    stmt_t c_store_stmt_;
    alloc_updater_t &alloc_updater_;
    expr_t tg_idx0_;

    std::vector<dpas_info_t> dpas_infos_;
    std::vector<send_info_t> send_infos_;
};

void inject_dpasw(ngen::HW hw, stmt_t &load_mul_stmt, const expr_t &c_buf,
        stmt_t &c_store_stmt, alloc_updater_t &alloc_updater,
        const expr_t &tg_idx0) {
    dpasw_injector_t injector(
            hw, load_mul_stmt, c_buf, c_store_stmt, alloc_updater, tg_idx0);

    injector.inject();
    load_mul_stmt = injector.load_mul_stmt();
    c_store_stmt = injector.c_store_stmt();
}

stmt_t inject_atomic(const stmt_t &stmt) {
    stmt_t ret = stmt;
    auto stmt_vec = flatten_statements(stmt);
    for (size_t i = 0; i < stmt_vec.size(); i++) {
        bool ok = true;
        ok &= is_func_call<dpas_t>(stmt_vec[i]) // No atomics for DP4As!
                && !dpas_t::is_dp4a_call(stmt_vec[i]);
        ok &= (i + 1 < stmt_vec.size()) && is_func_call<dpas_t>(stmt_vec[i + 1])
                && !dpas_t::is_dp4a_call(stmt_vec[i + 1]);
        if (ok) {
            auto &cur_src1 = dpas_t::arg_src1(stmt_vec[i]);
            auto &next_src1 = dpas_t::arg_src1(stmt_vec[i + 1]);
            // Compare src1, apply {Atomic} if they are equal.
            if (cur_src1.is_equal(next_src1)) {
                auto &s = stmt_vec[i];
                auto atomic_attr = instruction_modifier_attr_t::make(
                        ngen_proxy::InstructionModifier().with_atomic());
                ret = substitute(ret, s, atomic_attr.apply_to(s));
            }
        }
    }
    return ret;
}

class external_var_visitor_t : public scope_visitor_t {
public:
    void _visit(const var_t &obj) {
        if (!is_expr_defined(obj)) external_vars.insert(obj);
    }

    object_eq_set_t<expr_t> external_vars;
};

stmt_t inject_external_var_let(const stmt_t &_stmt, ir_context_t &ir_ctx) {
    trace_start();
    auto stmt = _stmt;
    external_var_visitor_t v;
    v.visit(stmt);

    for (auto &var : v.external_vars)
        stmt = let_t::make(var, {}, stmt);

    trace_pass("inject_external_var_let", stmt, ir_ctx);
    return stmt;
}

class slm_buffer_merger_t : public ir_mutator_t {
public:
    slm_buffer_merger_t() {
        slm_base_ = make_buffer("slm");
        slm_off_.push_back(0);
    }

    const expr_t &slm_base() const { return slm_base_; }

    int slm_size() const { return slm_size_; }

    object_t _mutate(const alloc_t &obj) override {
        if (obj.kind != alloc_kind_t::slm) return ir_mutator_t::_mutate(obj);

        auto new_buf = push(obj);
        auto new_obj = ir_mutator_t::_mutate(obj);
        pop();

        auto &alloc = new_obj.as<alloc_t>();
        new_obj = substitute(alloc.body, alloc.buf, new_buf);

        return new_obj;
    }

private:
    expr_t push(const alloc_t &obj) {
        int cur_off = slm_off_.back();
        expr_t new_buf = slm_base_ + cur_off;
        slm_off_.push_back(cur_off + obj.size);
        slm_size_ = std::max(slm_size_, cur_off + obj.size);
        return new_buf;
    }

    void pop() { slm_off_.pop_back(); }

    expr_t slm_base_;
    std::vector<int> slm_off_;
    int slm_size_ = 0;
};

stmt_t merge_slm_buffers(const stmt_t &_stmt, ir_context_t &ir_ctx) {
    trace_start();
    stmt_t stmt = _stmt;
    slm_buffer_merger_t merger;
    stmt = merger.mutate(stmt);
    stmt = alloc_t::make(
            merger.slm_base(), merger.slm_size(), alloc_kind_t::slm, stmt);
    trace_pass("merge_slm_buffers", stmt, ir_ctx);
    return stmt;
}

class buffer_offset_lifter_t : public ir_mutator_t {
public:
    object_t _mutate(const func_call_t &obj) {
        if (!obj.func.is<send_t>()) return ir_mutator_t::_mutate(obj);

        auto &mem_buf = send_t::arg_mem_buf(obj);
        if (!mem_buf.is<ptr_t>()) return ir_mutator_t::_mutate(obj);

        auto &base = mem_buf.as<ptr_t>().base;
        auto &off = mem_buf.as<ptr_t>().off;

        std::vector<expr_t> new_args = obj.args;
        send_t::arg_mem_buf(new_args) = base;
        send_t::arg_mem_off(new_args) += off;
        return obj.func.call(new_args, obj.attr);
    }
};

stmt_t lift_buffer_offsets_in_send(const stmt_t &s, ir_context_t &ir_ctx) {
    trace_start();
    buffer_offset_lifter_t lifter;
    auto ret = lifter.mutate(s);
    trace_pass("lift_buffer_offsets_in_send", ret, ir_ctx);
    return ret;
}

stmt_t simplify_pass(const stmt_t &s, ir_context_t &ir_ctx) {
    trace_start();
    auto ret = simplify(s, ir_ctx.cset());
    trace_pass("simplify_pass", ret, ir_ctx);
    return ret;
}

class slm_reorder_injector_t : public ir_mutator_t {
public:
    slm_reorder_injector_t(const stmt_t &root, const conv_config_t &cfg,
            const grid_info_t &tg_grid)
        : cfg_(cfg), tg_grid_(tg_grid) {
        alloc_manager_t alloc_mgr(root);
        auto slm_buffers = alloc_mgr.find_buffers(alloc_kind_t::slm);
        ir_assert(slm_buffers.size() == 1);
        slm_base_ = slm_buffers[0];
        slm_size_ = alloc_mgr.total_size(alloc_kind_t::slm);
    }

    const expr_t &slm_base() const { return slm_base_; }

    int slm_size() const { return slm_size_; }

    object_t _mutate(const func_call_t &obj) override {
        if (!is_func_call<reorder_t>(obj)) return obj;

        auto &call = obj.as<func_call_t>();

        auto stmt = create_slm_reorder(call.func.as<reorder_t>(),
                reorder_t::arg_src_buf(call), reorder_t::arg_dst_buf(call));
        if (stmt.is_empty()) return obj;
        return std::move(stmt);
    }

private:
    stmt_t create_slm_reorder(const reorder_t &reorder, const expr_t &src_buf,
            const expr_t &dst_buf) {
        auto src = reorder.src_layout;
        auto dst = reorder.dst_layout;
        if (!src.is_dense() || !dst.is_dense()) return stmt_t();

        layout_t::try_reinterpret_to_wider_type(src, dst);
        if (src.type() != dst.type()) return stmt_t();
        if (src.type().size() != 4) return stmt_t();

        layout_iterator_t src_it(src);
        layout_iterator_t dst_it(dst);

        tensor_t max_tile;
        for (;;) {
            auto src_tile = src_it.tile();
            auto dst_tile = dst_it.tile();
            if (src_tile.is_equal(dst_tile)) {
                auto s = src.map(src_it.tile());
                auto d = dst.map(dst_it.tile());
                if (s.is_dense() && d.is_dense()
                        && src_it.outer_layout() == dst_it.outer_layout()) {
                    if (is_slm_reorder_ok(s, d)) { max_tile = src_tile; }
                }
                if (!src_it.has_next() || !dst_it.has_next()) break;
                ++src_it;
                ++dst_it;
            } else {
                if (src_tile.elems() <= dst_tile.elems()) {
                    if (!src_it.has_next()) break;
                    ++src_it;
                } else {
                    if (!dst_it.has_next()) break;
                    ++dst_it;
                }
            }
        }

        if (max_tile.is_empty()) return stmt_t();

        return create_slm_reorder(max_tile, src, dst, src_buf, dst_buf);
    }

    stmt_t create_slm_reorder(const tensor_t &tile, const layout_t &src,
            const layout_t &dst, const expr_t &src_buf, const expr_t &dst_buf) {
        auto src_tile = src.map(tile);
        auto &src_tile_blocks = src_tile.blocks();
        int simd = src_tile_blocks[0].block;
        int vect_size = src_tile_blocks[1].block;
        int tile_size = simd * vect_size * src.type().size();
        int slm_thr_size = (int)src.size();
        int dword_size = type_t::dword().size();
        int hword_size = type_t::hword().size();
        int hwords = tile_size / hword_size;

        ir_assert(tile_size % hword_size == 0);

        slm_size_ = std::max(slm_size_, slm_thr_size * tg_grid_.elems());

        auto store_send = send_t::make(cfg_.hw(), send_op_t::store,
                send_address_t::slm, type_t::dword(vect_size), simd);
        auto load_send = send_t::make(cfg_.hw(), send_op_t::load,
                send_address_t::slm, type_t::hword(hwords), 1);

        std::vector<expr_t> vec(simd);
        for (int i = 0; i < simd; i++)
            vec[i] = expr_t(i * vect_size * dword_size);
        auto vec_off = shuffle_t::make(vec);
        auto tid = tg_grid_.idx(1) * tg_grid_.dim(0) + tg_grid_.idx(0);
        expr_t off0 = tid * slm_thr_size;

        stmt_t store_stmt;
        stmt_t load_stmt;
        src.for_each_tile(tile, [&](const std::vector<dim_t> &start) {
            expr_t off = (int)src.offset_in_bytes(start);
            auto store = store_send.call({slm_base_,
                    shuffle_t::make_broadcast(off0 + off, simd) + vec_off,
                    src_buf + off, expr_t()});
            auto load = load_send.call(
                    {slm_base_, off0 + off, dst_buf + off, expr_t()});
            store_stmt = store_stmt.append(store);
            load_stmt = load_stmt.append(load);
        });

        auto ret = store_stmt.append(load_stmt);
        return ret;
    }

    bool is_slm_reorder_ok(const layout_t &src, const layout_t &dst) const {
        auto &src_blocks = src.blocks();
        auto &dst_blocks = dst.blocks();
        if (src_blocks.size() != 2 || dst_blocks.size() != 2) return false;
        auto &s0 = src_blocks[0];
        auto &s1 = src_blocks[1];
        auto &d0 = dst_blocks[0];
        auto &d1 = dst_blocks[1];

        if (s0.dim_idx != d1.dim_idx || s1.dim_idx != d0.dim_idx) return false;
        ir_assert(s0.block == d1.block);
        ir_assert(s1.block == d0.block);

        int simd = s0.block;
        int vec_size = s1.block;
        if (!utils::one_of(simd, 16)) return false;
        if (!utils::one_of(vec_size, 8)) return false;

        return true;
    }

    const conv_config_t &cfg_;
    grid_info_t tg_grid_;

    expr_t slm_base_;
    int slm_size_ = 0;
};

stmt_t inject_slm_reorder(const stmt_t &s, ir_context_t &ir_ctx,
        const conv_config_t &cfg, const grid_info_t &tg_grid) {
    trace_start();
    if (cfg.use_a_slm || cfg.use_b_slm) return s;
    if (cfg.hw() < ngen::HW::XeHPC) return s;
    slm_reorder_injector_t injector(s, cfg, tg_grid);
    stmt_t ret = injector.mutate(s);

    auto &slm_buf = injector.slm_base();
    int slm_size = injector.slm_size();
    alloc_updater_t alloc_updater;
    alloc_updater.resize(slm_buf, slm_size);
    ret = alloc_updater.update(ret);

    trace_pass("inject_slm_reorder", ret, ir_ctx);
    return ret;
}

class send_injector_t : public ir_mutator_t {
public:
    send_injector_t(ir_context_t &ir_ctx) : ir_ctx_(ir_ctx) {}

    object_t _mutate(const func_call_t &obj) {
        auto *send = obj.func.as_ptr<send_t>();
        if (!send) return ir_mutator_t::_mutate(obj);

        auto &mem_buf = send_t::arg_mem_buf(obj);
        auto &mem_off = send_t::arg_mem_off(obj);
        auto &reg_buf = send_t::arg_reg_buf(obj);
        auto &mask = send_t::arg_mask(obj);

        ir_assert(is_var(mem_buf)) << mem_buf;

        auto header_buf = ir_ctx_.create_tmp_var(type_t::byte_ptr(), "h");
        auto off_store = simplify_store(
                send->create_offset_store(header_buf, mem_buf, mem_off));

        if (send->is_2d()) {
            auto emit_store = [&](const expr_t &e, int off) {
                auto store = store_t::make(header_buf, off, e);
                off_store = off_store.append(store);
            };
            auto emit_store_s32 = [&](int value, int off) {
                emit_store(cast(value, type_t::s32()), off);
            };
            auto &info = send->block_2d_info;
            int type_size = send->type.size();
            emit_store_s32(info.surface_width * type_size - 1,
                    send_t::header_2d_off_surface_width());
            emit_store_s32(info.surface_height - 1,
                    send_t::header_2d_off_surface_height());
            emit_store_s32(info.surface_pitch * type_size - 1,
                    send_t::header_2d_off_surface_pitch());
            emit_store(send_t::arg_x(obj), send_t::header_2d_off_x());
            emit_store(send_t::arg_y(obj), send_t::header_2d_off_y());
            uint32_t w_enc = info.width - 1;
            uint32_t h_enc = info.height - 1;
            uint32_t count_enc = info.count - 1;
            emit_store_s32((count_enc << 16) + (h_enc << 8) + w_enc,
                    send_t::header_2d_off_whc());
        }

        auto new_call = func_call_t::make(
                obj.func, {mem_buf, header_buf, reg_buf, mask}, obj.attr);
        auto body = stmt_seq_t::make(off_store, new_call);

        // Allocate header.
        return alloc_t::make(
                header_buf, send->header_size(), alloc_kind_t::grf, body);
    }

private:
    stmt_t simplify_store(const stmt_t &_store) const {
        auto &store = _store.as<store_t>();

        auto value = store.value;
        value = simplify(value, ir_ctx_.cset());

        // Convert to N-ary form and back to expand multiplications. This
        // helps to find more common subexpressions during the pass.
        value = nary_op_canonicalize(value);
        value = nary_op_back_transform(value);

        return store_t::make(store.buf, store.off, value, store.stride);
    }

    ir_context_t &ir_ctx_;
};

stmt_t inject_send(const stmt_t &s, ir_context_t &ir_ctx) {
    trace_start();
    auto ret = send_injector_t(ir_ctx).mutate(s);
    trace_pass("inject_send", ret, ir_ctx);
    return ret;
}

class alloc_lifter_t : public ir_mutator_t {
public:
    alloc_lifter_t(const stmt_t &root, bool reuse_headers)
        : reuse_headers_(reuse_headers) {
        if (!reuse_headers_) return;
        auto calls = find_objects<func_call_t>(root);
        for (auto &c : calls) {
            if (!is_func_call<send_t>(c)) continue;
            auto header_buf = send_t::arg_mem_off(c);
            ir_assert(is_var(header_buf)) << header_buf;
            header_bufs_.insert(header_buf);
        }
    }

    object_t _mutate(const alloc_t &obj) override {
        if (!do_lift(obj)) return ir_mutator_t::_mutate(obj);
        // Remove alloc and insert it before the compute loop.
        allocs_.push_back(&obj);
        return obj.body;
    }

    object_t _mutate(const stmt_group_t &obj) override {
        bool is_compute_loop = (obj.label == stmt_label_t::compute_loop());
        if (is_compute_loop) in_compute_loop_ = true;
        auto new_obj = ir_mutator_t::_mutate(obj);
        if (is_compute_loop) {
            in_compute_loop_ = false;
            // Outermost loop.
            for (auto it = allocs_.rbegin(); it != allocs_.rend(); ++it) {
                auto &a = it->as<alloc_t>();
                new_obj = alloc_t::make(
                        a.buf, a.size, a.kind, a.attrs, new_obj);
            }
            allocs_.resize(0);
        }
        return new_obj;
    }

private:
    bool do_lift(const alloc_t &obj) const {
        if (!in_compute_loop_) return false;
        if (reuse_headers_) {
            bool is_header_alloc = (header_bufs_.count(obj.buf) != 0);
            return !is_header_alloc;
        }
        return true;
    }

    bool reuse_headers_;
    object_set_t<expr_t> header_bufs_;

    bool in_compute_loop_ = false;
    std::vector<stmt_t> allocs_;
};

stmt_t lift_alloc(
        const stmt_t &s, ir_context_t &ir_ctx, const conv_config_t &cfg) {
    trace_start();
    auto ret = alloc_lifter_t(s, cfg.reuse_headers).mutate(s);
    trace_pass("lift_alloc", ret, ir_ctx);
    return ret;
}

class send_2d_header_store_lifter_t : public ir_mutator_t {
public:
    send_2d_header_store_lifter_t(const stmt_t &root) {
        auto calls = find_objects<func_call_t>(root);
        for (auto &c : calls) {
            if (!is_func_call<send_t>(c)) continue;
            if (!c.as<func_call_t>().func.as<send_t>().is_2d()) continue;
            auto header_buf = send_t::arg_mem_off(c);
            ir_assert(is_var(header_buf)) << header_buf;
            header_bufs_.insert(header_buf);
        }
    }

    object_t _mutate(const alloc_t &obj) override {
        auto new_obj = ir_mutator_t::_mutate(obj);
        auto it = stores_.find(obj.buf);
        if (it == stores_.end()) return new_obj;

        auto &alloc = new_obj.as<alloc_t>();
        stmt_t header_store;
        for (auto &s : it->second)
            header_store = header_store.append(s);
        it->second.clear();

        auto new_body = header_store.append(alloc.body);
        return alloc_t::make(
                alloc.buf, alloc.size, alloc.kind, alloc.attrs, new_body);
    }

    object_t _mutate(const store_t &obj) override {
        if (header_bufs_.count(obj.buf) == 0) return obj;
        // Do not lift address assignments and non-const x and y.
        int off = to_cpp<int>(obj.off);
        if (off == 0) return obj;
        if (utils::one_of(
                    off, send_t::header_2d_off_x(), send_t::header_2d_off_y())
                && !is_const(obj.value))
            return obj;
        stores_[obj.buf].push_back(obj);
        return stmt_t();
    }

private:
    object_set_t<expr_t> header_bufs_;
    object_map_t<expr_t, std::vector<stmt_t>> stores_;
};

stmt_t lift_send_2d_header_store(const stmt_t &s, ir_context_t &ir_ctx) {
    trace_start();
    auto ret = send_2d_header_store_lifter_t(s).mutate(s);
    trace_pass("lift_send_2d_header_store", ret, ir_ctx);
    return ret;
}

class hoist_exprs_mutator_t : public ir_mutator_t {
public:
    hoist_exprs_mutator_t(ir_context_t &ir_ctx) : ir_ctx_(ir_ctx) {}

    ~hoist_exprs_mutator_t() override { ir_assert(let_vars_.empty()); }

    object_t _mutate(const func_call_t &obj) override {
        if (!obj.func.is<send_t>()) return ir_mutator_t::_mutate(obj);

        std::vector<expr_t> new_args;
        for (auto &e : obj.args) {
            new_args.push_back(hoist_expr(e));
        }

        if (ir_utils::is_equal(new_args, obj.args)) return obj;

        return func_call_t::make(obj.func, new_args, obj.attr);
    }

    object_t _mutate(const stmt_group_t &obj) override {
        if (obj.body.is<for_t>()) {
            loops_.emplace_back(obj.body.as<for_t>().var);
            const for_t *for_obj = obj.body.as_ptr<for_t>();
            auto body = for_obj ? ir_mutator_t::_mutate(*for_obj) : for_obj;
            if (body.is_same(obj.body)) return obj;
            auto new_obj = stmt_group_t::make(obj.label, body);
            return injects_lets_and_pop_loop(new_obj);
        }
        return ir_mutator_t::_mutate(obj);
    }

    object_t _mutate(const store_t &obj) override {
        auto value = hoist_expr(obj.value);
        if (value.is_equal(obj.value)) return obj;
        return store_t::make(obj.buf, obj.off, value, obj.stride);
    }

    object_t _mutate(const for_t &obj) override {
        loops_.emplace_back(obj.var);
        auto new_obj = ir_mutator_t::_mutate(obj);
        return injects_lets_and_pop_loop(new_obj);
    }

    object_t _mutate(const let_t &obj) override {
        bool fully_hoisted = false;
        expr_t new_value;
        bool is_const_let = is_const(obj.value) || is_shuffle_const(obj.value);
        if (is_const_let && loops_.size() > 0) {
            fully_hoisted = true;
            register_let(obj.var, obj.value);
            loops_[0].lets.push_back(let_t::make(obj.var, obj.value));
        } else {
            new_value = hoist_expr(obj.value, obj.var, &fully_hoisted);
        }
        if (fully_hoisted) return mutate(obj.body);
        register_let(obj.var, new_value);
        auto new_obj = let_t::make(
                obj.var, new_value, ir_mutator_t::mutate(obj.body));
        unregister_let(obj.var);
        return std::move(new_obj);
    }

private:
    struct loop_info_t {
        loop_info_t(const expr_t &var) : var(var) {}

        expr_t var;
        int var_count = 0;
        std::vector<stmt_t> lets;
    };

    expr_t hoist_expr(const expr_t &expr, const expr_t &expr_var = {},
            bool *fully_hoisted = nullptr) {
        if (expr.is_empty()) return expr;
        if (expr.type().is_ptr()) return expr;
        if (expr.type().is_bool()) return expr;
        if (is_const(expr) || is_shuffle_const(expr) || is_var(expr))
            return expr;

        auto hoisted_expr = hoist_expr_with_add(expr, expr_var, fully_hoisted);
        if (!hoisted_expr.is_equal(expr)) return hoisted_expr;

        // hoist_expr_with_add() doesn't handle cast so try to hoist it manually.
        auto *cast = expr.as_ptr<cast_t>();
        if (!cast) return hoisted_expr;

        auto hoisted_cast_expr = hoist_expr(cast->expr);
        if (!hoisted_cast_expr.is_equal(cast->expr)) {
            hoisted_expr = cast_t::make(
                    cast->type, hoisted_cast_expr, cast->saturate);
        }
        return hoisted_expr;
    }

    expr_t hoist_expr_with_add(const expr_t &expr, const expr_t &expr_var = {},
            bool *fully_hoisted = nullptr) {
        auto cur_expr = nary_op_canonicalize(expr);

        auto is_nary_add = [](const expr_t &e) {
            auto *nary = e.as_ptr<nary_op_t>();
            return nary && (nary->op_kind == op_kind_t::_add);
        };

        for (size_t i = 0; i < loops_.size(); i++) {
            std::vector<expr_t> invariant_args;
            std::vector<expr_t> other_args;
            std::vector<expr_t> nary_args;
            if (is_nary_add(cur_expr)) {
                nary_args = cvt_expr_to_nary_op_args(cur_expr);
            } else {
                nary_args.push_back(cur_expr);
            }
            for (auto &_a : nary_args) {
                auto a = nary_op_back_transform(_a);
                bool is_inv_arg = true;
                for (size_t j = i; j < loops_.size(); j++) {
                    if (!is_invariant(a, loops_[j].var)) is_inv_arg = false;
                }
                if (is_inv_arg) {
                    invariant_args.push_back(_a);
                } else {
                    other_args.push_back(_a);
                }
            }
            // Nothing to hoist for this loop, continue.
            if (invariant_args.empty()) continue;
            if (invariant_args.size() == 1 && is_var(invariant_args[0]))
                continue;

            // Introduce new variable for the invariant sub-expression.
            auto inv_expr = nary_op_back_transform(
                    make_nary_op(op_kind_t::_add, invariant_args));
            expr_t inv_var;
            if (!expr_var.is_empty() && other_args.empty()) {
                // If nothing to hoist further, reuse the old variable and
                // return.
                inv_var = expr_var;
            } else {
                inv_var = ir_ctx_.create_tmp_var(inv_expr.type());
            }
            auto let = let_t::make(inv_var, inv_expr);
            register_let(inv_var, inv_expr);
            loops_[i].lets.push_back(let);

            if (other_args.empty()) {
                if (fully_hoisted) *fully_hoisted = true;
                return inv_var;
            }

            other_args.push_back(inv_var);
            cur_expr = make_nary_op(op_kind_t::_add, other_args);
        }
        return nary_op_back_transform(cur_expr);
    }

    stmt_t injects_lets_and_pop_loop(const stmt_t &_s) {
        stmt_t s = _s;
        // Inject let statements if any.
        auto &lets = loops_.back().lets;
        for (auto it = lets.rbegin(); it != lets.rend(); ++it) {
            auto &let = it->as<let_t>();
            s = let_t::make(let.var, let.value, s);
            unregister_let(let.var);
        }
        loops_.pop_back();
        return s;
    }

    void register_let(const expr_t &var, const expr_t &value) {
        let_vars_.insert({var, value});
    }

    void unregister_let(const expr_t &var) { let_vars_.erase(var); }

    bool is_invariant(const expr_t &e, const expr_t &var) const {
        if (contains_object(e, var)) return false;
        if (!find_objects<load_t>(e).empty()) return false;

        // Check value if this is a let variable.
        auto it = let_vars_.find(e);
        if (it != let_vars_.end()) return is_invariant(it->second, var);

        if (is_var(e)) return true;

        // Check transitive dependencies.
        auto vars = find_unique_objects<var_t>(e);
        for (auto &v : vars) {
            if (!is_invariant(v, var)) return false;
        }
        return true;
    }

    ir_context_t &ir_ctx_;
    std::vector<loop_info_t> loops_;

    object_map_t<expr_t, expr_t> let_vars_;
};

stmt_t hoist_exprs(const stmt_t &s, ir_context_t &ir_ctx) {
    trace_start();
    auto ret = hoist_exprs_mutator_t(ir_ctx).mutate(s);
    trace_pass("hoist_exprs", ret, ir_ctx);
    return ret;
}

class hoist_send_masks_mutator_t : public ir_mutator_t {
public:
    hoist_send_masks_mutator_t(
            ir_context_t &ir_ctx, const stmt_label_t &label, bool split_by_and)
        : ir_ctx_(ir_ctx), label_(label), split_by_and_(split_by_and) {}

    object_t _mutate(const for_t &obj) override {
        loop_deps_.insert(obj.var);
        return ir_mutator_t::_mutate(obj);
    }

    object_t _mutate(const func_call_t &obj) override {
        if (!in_stmt_group || !is_func_call<send_t>(obj))
            return ir_mutator_t::_mutate(obj);

        auto &mask = send_t::arg_mask(obj);
        if (mask.is_empty()) return ir_mutator_t::_mutate(obj);

        auto new_args = obj.args;
        auto hoisted_mask = hoist_mask(mask);
        if (hoisted_mask.is_same(mask)) return ir_mutator_t::_mutate(obj);

        ir_assert(hoisted_mask.type().is_u16()) << hoisted_mask;

        send_t::arg_mask(new_args) = cast(hoisted_mask, mask.type());
        return func_call_t::make(obj.func, new_args, obj.attr);
    }

    object_t _mutate(const let_t &obj) override {
        auto value_vars = find_objects<var_t>(obj.value);
        for (auto &v : value_vars) {
            if (is_loop_dependency(v)) {
                loop_deps_.insert(obj.var);
                break;
            }
        }

        if (in_stmt_group) {
            ir_assert(!obj.value.is_empty());
            let_values_.emplace(obj.var, expand(obj.value, value_vars));
        }

        return ir_mutator_t::_mutate(obj);
    }

    object_t _mutate(const stmt_group_t &obj) override {
        bool is_stmt_group = (obj.label == label_);
        if (is_stmt_group) in_stmt_group = true;
        auto new_obj = ir_mutator_t::_mutate(obj);
        if (is_stmt_group) {
            in_stmt_group = false;
            return create_mask_stmt(new_obj);
        }
        return new_obj;
    }

private:
    bool is_loop_dependency(const expr_t &v) const {
        ir_assert(is_var(v)) << v;
        return loop_deps_.count(v) != 0;
    }

    expr_t hoist_mask(const expr_t &e) {
        ir_assert(e.type().is_bool()) << e;

        if (e.type().elems() > 16) return e;
        if (is_const(e) || is_shuffle_const(e)) return e;

        // Can't hoist a mask containing loop vars.
        auto vars = find_objects<var_t>(e);
        for (auto &v : vars) {
            if (is_loop_dependency(v)) return e;
        }

        auto e_expanded = expand(e, vars);

        // Can't hoist a mask containing loads.
        if (!find_objects<load_t>(e_expanded).empty()) return e;

        auto it = hoisted_masks_.find(e_expanded);
        if (it != hoisted_masks_.end()) return it->second;

        auto var = ir_ctx_.create_tmp_var(type_t::u16());
        hoisted_masks_.emplace(e_expanded, var);

        return var;
    }

    expr_t expand(const expr_t &_e, const std::vector<object_t> &e_vars) const {
        auto e = _e;
        for (auto &v : e_vars) {
            auto it = let_values_.find(v);
            if (it == let_values_.end()) continue;
            e = substitute(e, v, it->second);
        }
        return e;
    }

    stmt_t create_mask_stmt(const stmt_t &body) {
        stmt_t s = body;

        object_eq_map_t<expr_t, expr_t> and_ops;
        object_eq_map_t<expr_t, expr_t> mask_exprs;
        for (auto &kv : hoisted_masks_) {
            if (split_by_and_) {
                auto e = split_by_and_ops(kv.first, and_ops);
                mask_exprs.emplace(e, kv.second);
            }
        }
        if (and_ops.size() < mask_exprs.size()) {
            for (auto &kv : mask_exprs) {
                s = let_t::make(kv.second, cast(kv.first, kv.second.type()), s);
            }
            for (auto &kv : and_ops) {
                s = let_t::make(kv.second, cast(kv.first, kv.second.type()), s);
            }
        } else {
            for (auto &kv : hoisted_masks_)
                s = let_t::make(kv.second, cast(kv.first, kv.second.type()), s);
        }

        return s;
    }

    expr_t split_by_and_ops(
            const expr_t &e, object_eq_map_t<expr_t, expr_t> &ops) {
        auto *binary_op = e.as_ptr<binary_op_t>();
        if (!binary_op || binary_op->op_kind != op_kind_t::_and) {
            auto it = ops.find(e);
            if (it != ops.end()) return it->second;

            auto var = ir_ctx_.create_tmp_var(type_t::u16());
            ops.emplace(e, var);
            return var;
        }

        auto a = split_by_and_ops(binary_op->a, ops);
        auto b = split_by_and_ops(binary_op->b, ops);
        return binary_op_t::make(op_kind_t::_and, a, b);
    }

    bool in_stmt_group = false;
    object_set_t<expr_t> loop_deps_;
    object_eq_map_t<expr_t, expr_t> hoisted_masks_;
    object_map_t<expr_t, expr_t> let_values_;

    ir_context_t &ir_ctx_;
    stmt_label_t label_;
    bool split_by_and_;
};

stmt_t hoist_send_masks(const stmt_t &s, ir_context_t &ir_ctx,
        const stmt_label_t &label, bool split_by_and) {
    trace_start();
    hoist_send_masks_mutator_t mutator(ir_ctx, label, split_by_and);

    auto ret = mutator.mutate(s);
    trace_pass("hoist_send_masks", ret, ir_ctx);
    return ret;
}

class spurious_send_mask_cast_remover_t : public ir_mutator_t {
public:
    object_t _mutate(const cast_t &obj) override {
        if (in_send_ && obj.is_bool_vec_u16() && obj.expr.type().is_bool())
            return mutate(obj.expr);
        return ir_mutator_t::_mutate(obj);
    }

    object_t _mutate(const func_call_t &obj) override {
        if (!is_func_call<send_t>(obj)) return obj;

        in_send_ = true;
        auto new_obj = ir_mutator_t::_mutate(obj);
        in_send_ = false;
        return new_obj;
    }

private:
    bool in_send_ = false;
};

stmt_t remove_spurious_send_mask_cast(const stmt_t &s, ir_context_t &ir_ctx) {
    spurious_send_mask_cast_remover_t mutator;
    trace_start();
    auto ret = mutator.mutate(s);
    trace_pass("remove_spurious_send_mask_cast", ret, ir_ctx);
    return ret;
}

class loop_strength_reducer_t : public ir_mutator_t {
public:
    loop_strength_reducer_t() {
        // Create top-level dummy loop.
        loops_.emplace_back();
    }

    ~loop_strength_reducer_t() override {
        // Sanity check, all stores must be applied.
        ir_assert(post_inc_stores.empty());
    }

    object_t _mutate(const for_t &obj) override {
        loops_.emplace_back(obj);
        auto new_obj = ir_mutator_t::_mutate(obj);
        return inject_stores_and_pop_loop(new_obj);
    }

    object_t _mutate(const let_t &obj) override {
        int loop_level = int(loops_.size()) - 1;
        auto ret = lets_.insert(
                {obj.var, let_info_t(obj.var, obj.value, loop_level)});
        ir_assert(ret.second);
        MAYBE_UNUSED(ret);
        auto new_obj = ir_mutator_t::_mutate(obj);
        lets_.erase(obj.var);
        return new_obj;
    }

    object_t _mutate(const stmt_group_t &obj) override {
        if (obj.body.is<for_t>()) {
            loops_.emplace_back(obj.body);
            const for_t *for_obj = obj.body.as_ptr<for_t>();
            auto body = for_obj ? ir_mutator_t::_mutate(*for_obj) : for_obj;
            if (body.is_same(obj.body)) return obj;
            auto new_obj = stmt_group_t::make(obj.label, body);
            return inject_stores_and_pop_loop(new_obj);
        }
        return ir_mutator_t::_mutate(obj);
    }

    // Pattern to handle:
    //     for (...) {
    //         store(buf_ptr, ...) <- Write (producer).
    //         // ...
    //         stmt_t(..., buf_ptr, ...) <- Read (consumer).
    //     }
    object_t _mutate(const store_t &obj) override {
        if (loops_.size() == 1) return ir_mutator_t::_mutate(obj);

        // Try to reduce strength, moving the store up.
        int init_store_level = -1;
        stmt_t init_store_stmt = obj;
        post_inc_store_info_t post_inc_store(obj);
        for (int level = int(loops_.size()) - 1; level >= 1; level--) {
            auto &loop_info = loops_[level];
            int refs = count_object(loop_info.loop, obj.buf);
            // Producer and consumer - must be 2 references.
            if (refs != 2) break;

            // Try to insert the store before level-th loop.
            auto &store = init_store_stmt.as<store_t>();
            auto &store_value = store.value;
            auto &loop_var = loop_info.loop_var();

            auto cur_value = substitute_let(store_value, level);
            auto next_value = substitute(cur_value, loop_var, loop_var + 1);
            auto inc = simplify(next_value - cur_value);

            // Cannot eliminate loop variable, break.
            if (contains_object(inc, loop_var)) break;

            // Not scalar, break.
            if (!store_value.type().is_scalar()) break;

            // Success, replace store by post-increment store.
            init_store_level = level;

            auto new_store_value
                    = substitute(cur_value, loop_var, loop_info.loop_init());
            init_store_stmt = store_t::make(store.buf, store.off,
                    simplify(new_store_value), store.stride);

            post_inc_store.update(loop_info, inc);
        }

        // Can't do anything, return as is.
        if (init_store_level == -1) return ir_mutator_t::_mutate(obj);

        // Move this store up, remove from here.
        loops_[init_store_level].init_stores.push_back(init_store_stmt);
        if (!post_inc_store.is_empty()) {
            auto ret = post_inc_stores.insert({obj.buf, post_inc_store});
            ir_assert(ret.second);
            MAYBE_UNUSED(ret);
        }
        return stmt_t();
    }

    object_t _mutate(const func_call_t &obj) override {
        for (auto &kv : post_inc_stores) {
            int refs = count_object(obj, kv.first);
            if (refs == 1) {
                auto ret = stmt_seq_t::make(obj, kv.second.stmt());
                post_inc_stores.erase(kv.first);
                return std::move(ret);
            }
        }
        return ir_mutator_t::_mutate(obj);
    }

private:
    struct loop_info_t {
        loop_info_t(const stmt_t &loop = {}) : loop(loop) {}

        const expr_t &loop_var() const { return loop.as<for_t>().var; }

        const expr_t &loop_init() const { return loop.as<for_t>().init; }

        const expr_t &loop_bound() const { return loop.as<for_t>().bound; }

        expr_t loop_extent() const { return loop_bound() - loop_init(); }

        // Loop being analyzed.
        stmt_t loop;
        // Stores to insert before the loop.
        std::vector<stmt_t> init_stores;

        std::vector<stmt_t> lets;
    };

    struct let_info_t {
        let_info_t(const expr_t &var, const expr_t &value, int loop_level)
            : var(var), value(value), loop_level(loop_level) {}

        expr_t var;
        expr_t value;
        int loop_level;
    };

    struct post_inc_store_info_t {
        post_inc_store_info_t(const store_t &obj)
            : store(&obj), inc(0), last_iter_cond(true), compensation(0) {}

        stmt_t stmt() const {
            auto load
                    = load_t::make(store->value.type(), store->buf, store->off);
            return store_t::make(store->buf, store->off, load + inc);
        }

        bool is_empty() const { return is_zero(inc); }

        void update(const loop_info_t &loop, const expr_t &loop_inc) {
            inc = simplify(iif_t::make(
                    last_iter_cond, inc - compensation + loop_inc, inc));
            if (last_iter_cond.is_equal(expr_t(true))) {
                last_iter_cond = (loop.loop_var() == loop.loop_bound() - 1);
            } else {
                last_iter_cond = last_iter_cond
                        & (loop.loop_var() == loop.loop_bound() - 1);
            }
            compensation = simplify(loop.loop_extent() * loop_inc);
        }

        const store_t *store;
        expr_t inc;

        expr_t last_iter_cond;
        expr_t compensation;
    };

    // Recursively substitutes all variable from let statements located under
    // the given loop level.
    expr_t substitute_let(const expr_t &_e, int loop_level) const {
        auto e = _e;
        for (;;) {
            bool found = false;
            auto vars = find_unique_objects<var_t>(e);
            for (auto &v : vars) {
                auto it = lets_.find(v);
                if (it == lets_.end()) continue;
                auto &let_info = it->second;
                // Do not substitute top-level let variables.
                if (let_info.loop_level < loop_level) continue;
                found = true;
                e = substitute(e, v, let_info.value);
            }
            if (!found) break;
        }
        return e;
    }

    // Injects initial store statements if any.
    object_t inject_stores_and_pop_loop(const stmt_t &_s) {
        stmt_t s = _s;
        auto &stores = loops_.back().init_stores;
        for (auto it = stores.rbegin(); it != stores.rend(); ++it) {
            s = stmt_seq_t::make(*it, s);
        }
        loops_.pop_back();
        // The top-level dummy loop shouldn't be removed.
        ir_assert(loops_.size() >= 1);
        return std::move(s);
    }

    // Loops, ordered from outermost to innermost. The first loop is dummy, to
    // represent let statements in the top-level scope.
    std::vector<loop_info_t> loops_;

    // Buffers whose references are to be updated.
    object_map_t<expr_t, post_inc_store_info_t> post_inc_stores;

    // Let statements available at the current IR node.
    object_map_t<expr_t, let_info_t> lets_;
};

// Detects and converts expensive expression operations inside a loop to less
// expensive operations. Example:
// Before:
//     for (int j = 0; j < N; j++) {
//         int off = off_i + j * K;
//         a[off] = j;
//     }
// After:
//     int off = off_i;
//     for (int j = 0; j < N; j++) {
//         a[off] = j;
//         off += K;
//     }
stmt_t loop_strength_reduce(const stmt_t &s, ir_context_t &ir_ctx) {
    trace_start();
    auto ret = loop_strength_reducer_t().mutate(s);
    trace_pass("loop_strength_reduce", ret, ir_ctx);
    return ret;
}

class alloc_let_optimizer_t : public ir_mutator_t {
public:
    // Also track alloc_t and for_t to validate all variable usages.
    object_t _mutate(const alloc_t &obj) override {
        return mutate_scope(obj, obj.buf);
    }

    object_t _mutate(const for_t &obj) override {
        level_++;
        auto new_obj = mutate_scope(obj, obj.var);
        level_--;
        return new_obj;
    }

    object_t _mutate(const let_t &obj) override {
        return mutate_scope(obj, obj.var);
    }

    object_t _mutate(const store_t &obj) override {
        auto &base = (obj.buf.is<var_t>() ? obj.buf : obj.buf.as<ptr_t>().base);
        // Do not count store references. If there are only stores to a buffer
        // and no other usages, the buffer can be safely removed.
        skip_var_ = base;
        auto new_obj = ir_mutator_t::_mutate(obj);
        skip_var_ = expr_t();
        return new_obj;
    }

    object_t _mutate(const var_t &obj) override {
        ir_assert(refs_.count(obj) == 1)
                << "Variable is not defined: " << expr_t(&obj);
        if (!skip_var_.is_same(obj)) refs_[&obj].update(increment_, level_);
        return ir_mutator_t::_mutate(obj);
    }

private:
    struct ref_info_t {
        ref_info_t(int level = 0)
            : refs(0), min_level(level), max_level(level) {}

        void update(int increment, int level) {
            refs += increment;
            max_level = std::max(max_level, level);
        }

        bool is_same_level() const { return min_level == max_level; }

        int refs;
        int min_level;
        int max_level;
    };

    template <typename T>
    object_t mutate_scope(const T &obj, const expr_t &var) {
        auto ret = refs_.insert({var, ref_info_t(level_)});
        ir_assert(ret.second) << stmt_t(obj);
        MAYBE_UNUSED(ret);

        auto new_obj = ir_mutator_t::_mutate(obj);
        auto &ref_info = refs_[var];

        if (std::is_same<T, let_t>()) {
            new_obj = mutate_let(new_obj.template as<let_t>(), ref_info);
        } else if (std::is_same<T, alloc_t>()) {
            new_obj = mutate_alloc(new_obj.template as<alloc_t>(), ref_info);
        }

        refs_.erase(var);
        return new_obj;
    }

    object_t mutate_let(const let_t &obj, const ref_info_t &ref_info) {
        ir_assert(ref_info.refs >= 1);
        if (ref_info.refs == 1) {
            // Variable is not used.
            remove_refs(obj);
            return obj.body;
        }
        // Check following conditions to substitute let value:
        // - 2 references: one from producer, one from consumer - means single usage
        // - Consumer and producer are on the same level (same loop)
        // - Variable is not external
        if (ref_info.refs == 2 && ref_info.is_same_level()
                && !obj.value.is_empty()) {
            return substitute(obj.body, obj.var, obj.value);
        }
        return obj;
    }

    object_t mutate_alloc(const alloc_t &obj, const ref_info_t &ref_info) {
        ir_assert(ref_info.refs >= 1);
        // Buffer is not used, single reference from alloc_t itself. Remove
        // stores to the buffer if any.
        if (ref_info.refs == 1) return remove_stores(obj.body, obj.buf);
        return obj;
    }

    void remove_refs(const let_t &obj) {
        increment_ = -1;
        mutate(obj.value);
        increment_ = 1;
    }

    // Removes all nested stores to the buffer.
    stmt_t remove_stores(const stmt_t &stmt, const expr_t &buf) {
        auto ret = stmt;
        auto stores = find_objects<store_t>(stmt);
        for (auto &_s : stores) {
            auto &s = _s.as<store_t>();
            auto &base = (s.buf.is<var_t>() ? s.buf : s.buf.as<ptr_t>().base);
            if (base.is_same(buf)) ret = substitute(ret, _s, stmt_t());
        }
        return ret;
    }

    int increment_ = 1;
    int level_ = 0;

    expr_t skip_var_;
    object_map_t<expr_t, ref_info_t> refs_;
};

stmt_t optimize_alloc_let(const stmt_t &s, ir_context_t &ir_ctx) {
    trace_start();
    auto ret = alloc_let_optimizer_t().mutate(s);
    trace_pass("optimize_alloc_let", ret, ir_ctx);
    return ret;
}

class unrolling_updater_t : public ir_mutator_t {
public:
    object_t _mutate(const let_t &obj) override {
        if (level_ == 0) {
            // Skip top-level let statements.
            return ir_mutator_t::_mutate(obj);
        }
        lets_.push_back(&obj);
        auto new_body = mutate(obj.body);
        if (!lets_.back()) {
            // Let was moved to the innermost loop.
            lets_.pop_back();
            return new_body;
        }
        lets_.pop_back();
        if (new_body.is_same(obj.body)) return obj;
        return let_t::make(obj.var, obj.value, new_body);
    }

    object_t _mutate(const for_t &obj) override {
        if (in_compute_loop_) level_++;
        found_loop_ = false;
        auto new_obj = ir_mutator_t::_mutate(obj);
        if (in_compute_loop_) level_--;
        if (!found_loop_) {
            // Innermost loop, inject let statements.
            auto body = get_stmt_body(new_obj);
            for (auto it = lets_.rbegin(); it != lets_.rend(); ++it) {
                body = let_t::make((*it)->var, (*it)->value, body);
                *it = nullptr;
            }
            new_obj = replace_stmt_body(new_obj, body);
        }
        found_loop_ = true;
        return new_obj;
    }

    object_t _mutate(const stmt_group_t &obj) override {
        if (obj.label == stmt_label_t::compute_loop()) {
            in_compute_loop_ = true;
        }
        auto new_obj = ir_mutator_t::_mutate(obj);
        if (obj.label == stmt_label_t::compute_loop()) {
            in_compute_loop_ = false;
        }
        return new_obj;
    }

private:
    bool found_loop_ = false;
    bool in_compute_loop_ = false;
    int level_ = 0;
    std::vector<const let_t *> lets_;
};

stmt_t update_loops_for_unrolling(
        const stmt_t &s, ir_context_t &ir_ctx, const conv_config_t &cfg) {
    trace_start();
    auto ret = s;
    if (cfg.do_pipeline_unroll) ret = unrolling_updater_t().mutate(s);
    trace_pass("update_loops_for_unrolling", ret, ir_ctx);
    return ret;
}

class store_splitter_t : public ir_mutator_t {
public:
    store_splitter_t(ngen::HW hw) : hw_(hw) {}

    object_t _mutate(const store_t &obj) override {
        int elems = obj.value.type().elems();
        int elem_size = obj.value.type().scalar().size();
        int stride = (obj.has_default_stride() ? 1 : obj.stride / elem_size);
        int store_size = elem_size * stride * elems;
        const auto grf_size = ngen::GRF::bytes(hw_);
        if (store_size <= 2 * grf_size) return ir_mutator_t::_mutate(obj);

        int step = 2 * grf_size / (stride * elem_size);
        stmt_t new_stmt;
        for (int i = 0; i < elems; i += step) {
            int cur_elems = std::min(step, elems - i);
            ir_assert(math::is_pow2(cur_elems));
            int off = i * stride * elem_size;
            auto store = store_t::make(obj.buf, obj.off + off,
                    split_expr(obj.value, i, i + cur_elems), obj.stride);
            new_stmt = new_stmt.append(store);
        }
        return std::move(new_stmt);
    }

private:
    static expr_t split_expr(const expr_t &e, int beg, int end) {
        auto *shuffle = e.as_ptr<shuffle_t>();
        if (shuffle) return shuffle_t::make(shuffle, beg, end);

        auto *binary = e.as_ptr<binary_op_t>();
        if (binary) {
            auto a = split_expr(binary->a, beg, end);
            auto b = split_expr(binary->b, beg, end);
            return binary_op_t::make(binary->op_kind, a, b);
        }
        ir_error_not_expected();
        return expr_t();
    }

    ngen::HW hw_;
};

stmt_t split_wide_stores(const stmt_t &s, ir_context_t &ir_ctx) {
    trace_start();
    auto ret = store_splitter_t(ir_ctx.hw_cfg().hw()).mutate(s);
    trace_pass("split_wide_stores", ret, ir_ctx);
    return ret;
}

class overflow_bound_finder_t : public bound_finder_base_t {
public:
    bool has_var(const expr_t &e) const {
        ir_assert(is_var(e)) << "Expected variable, found: " << e;
        auto it = var_bounds_.find(e);
        return it != var_bounds_.end();
    }

    std::pair<int64_t, int64_t> find_bounds(const expr_t &e) const {
        int64_t lo = find_low_bound(e);
        int64_t hi = find_high_bound(e);
        return std::make_pair(lo, hi);
    }

    int64_t get_var_bound(const expr_t &e, bool is_low) const override {
        ir_assert(has_var(e)) << "Variable not found: " << e;
        auto &lo_hi = var_bounds_.at(e);
        return is_low ? lo_hi.first : lo_hi.second;
    }

    void set_var_bounds(
            const expr_t &e, const std::pair<int64_t, int64_t> &lo_hi) {
        ir_assert(is_good_bound(lo_hi.first))
                << "Can't compute low bound for " << e;
        ir_assert(is_good_bound(lo_hi.second))
                << "Can't compute high bound for " << e;
        var_bounds_.emplace(e, lo_hi);
    }

protected:
    int64_t find_bound_impl(const expr_t &e, bool is_low) const override {
        auto *cast = e.as_ptr<cast_t>();
        if (cast) {
            if (e.type().is_u64() && cast->expr.type().is_ptr()) {
                return is_low ? 0 : std::numeric_limits<uint32_t>::max();
            } else if (e.type().is_u32() && cast->expr.type().is_ptr()) {
                return is_low ? 0 : std::numeric_limits<uint16_t>::max();
            }
        }
        return bound_finder_base_t::find_bound_impl(e, is_low);
    }

private:
    object_map_t<expr_t, std::pair<int64_t, int64_t>> var_bounds_;
};

class overflow_fixer_t : public ir_mutator_t {
public:
    overflow_fixer_t(ir_context_t &ir_ctx) : ir_ctx_(ir_ctx) {
        for (auto &kv : ir_ctx.cset().relations()) {
            int64_t lo = bound_finder_base_t::unlimited_bound(true);
            int64_t hi = bound_finder_base_t::unlimited_bound(false);
            for (auto &rel : kv.second) {
                bool is_ge = (rel.op_kind() == op_kind_t::_ge);
                bool is_le = (rel.op_kind() == op_kind_t::_le);
                ir_assert(is_ge || is_le);
                if (rel.op_kind() == op_kind_t::_ge) {
                    lo = std::max(to_cpp<int64_t>(rel.rhs()), lo);
                } else if (rel.op_kind() == op_kind_t::_le) {
                    hi = std::min(to_cpp<int64_t>(rel.rhs()), hi);
                } else {
                    ir_error_not_expected()
                            << "Only >= or <= is expected, found: "
                            << to_string(rel.op_kind());
                }
            }
            bound_finder_.set_var_bounds(kv.first, {lo, hi});
        }
    }

    object_t _mutate(const alloc_t &obj) override {
        return ir_mutator_t::_mutate(obj);
    }

    object_t _mutate(const binary_op_t &obj) override {
        return mutate_expr(obj);
    }

    object_t _mutate(const for_t &obj) override {
        auto lo = to_cpp<int64_t>(obj.init);
        auto hi = to_cpp<int64_t>(obj.bound) - 1;
        bound_finder_.set_var_bounds(obj.var, {lo, hi});
        return ir_mutator_t::_mutate(obj);
    }

    object_t _mutate(const let_t &obj) override {
        bool ok = true;
        if (!obj.var.type().is_int()) ok = false;
        if (ok && obj.value.is_empty()) ok = false;
        if (ok && bound_finder_.has_var(obj.var)) ok = false;

        if (ok) {
            if (contains_load(obj.value)) {
                vars_with_load_.insert(obj.var);
                ok = false;
            }
        }

        if (ok) {
            int elems = obj.var.type().elems();
            vec_vars_[obj.var].reserve(elems);
            for (int i = 0; i < elems; i++) {
                auto var_i = make_vec_var(obj.var, elems, i);
                expr_scalarizer_t scalarizer(elems, i, vec_vars_);
                auto value_i = scalarizer.mutate(obj.value);
                auto lo_hi = bound_finder_.find_bounds(value_i);
                bound_finder_.set_var_bounds(var_i, lo_hi);
                vec_vars_[obj.var].push_back(var_i);
            }
        }
        expr_t var = obj.var;
        expr_t value = mutate(obj.value);
        stmt_t body = mutate(obj.body);
        if (value.is_same(obj.value) && body.is_same(obj.body)) return obj;
        if (!value.is_empty() && value.type() != obj.value.type()) {
            auto old_var = var;
            var = ir_ctx_.create_tmp_var(
                    value.type(), old_var.as<var_t>().name);
            body = substitute(body, old_var, var);
        }
        return let_t::make(var, value, body);
    }

    object_t _mutate(const unary_op_t &obj) override {
        return mutate_expr(obj);
    }

private:
    template <typename T>
    object_t mutate_expr(const T &obj) {
        expr_t new_obj = ir_mutator_t::_mutate(obj);
        if (!new_obj.type().is_x32()) return std::move(new_obj);
        if (contains_load(new_obj)) return std::move(new_obj);

        bool found_overflow = false;
        int elems = new_obj.type().elems();
        for (int i = 0; i < elems; i++) {
            expr_scalarizer_t scalarizer(elems, i, vec_vars_);
            expr_t value = scalarizer.mutate(new_obj);
            int64_t lo = bound_finder_.find_low_bound(value);
            int64_t hi = bound_finder_.find_high_bound(value);
            bool ok = bound_finder_base_t::is_good_bound(lo)
                    && bound_finder_base_t::is_good_bound(hi);
            if (ok) {
                int64_t type_lo = value.type().is_s32()
                        ? (int64_t)std::numeric_limits<int32_t>::min()
                        : (int64_t)std::numeric_limits<uint32_t>::min();
                int64_t type_hi = value.type().is_s32()
                        ? (int64_t)std::numeric_limits<int32_t>::max()
                        : (int64_t)std::numeric_limits<uint32_t>::max();

                bool is_overflow = (lo < type_lo || hi > type_hi);
                if (is_overflow) {
                    found_overflow = true;
                    ir_warning() << "Found overflow: " << value
                                 << " low bound: " << lo
                                 << " high bound: " << hi << std::endl;
                    break;
                }
            }
        }
        if (found_overflow) return fix_overflow(new_obj);
        return std::move(new_obj);
    }

    bool contains_load(const expr_t &e) const {
        if (!find_objects<load_t>(e).empty()) return true;
        for (auto &v : find_objects<var_t>(e)) {
            if (vars_with_load_.count(v) != 0) return true;
        }
        return false;
    }

    static expr_t make_vec_var(const expr_t &_var, int elems, int idx) {
        if (elems == 1) return _var;
        auto &var = _var.as<var_t>();
        auto vec_name = var.name + "_" + std::to_string(idx) + "_";
        return var_t::make(var.type.scalar(), vec_name);
    }

    static expr_t fix_overflow(const expr_t &e) {
        auto *binary = e.as_ptr<binary_op_t>();
        if (binary) {
            return binary_op_t::make(binary->op_kind,
                    cast(binary->a, type_t::u64(e.type().elems())), binary->b);
        }

        ir_error_not_expected() << "Can't fix overflow: " << e;
        return e;
    }

    ir_context_t &ir_ctx_;
    overflow_bound_finder_t bound_finder_;
    object_map_t<expr_t, std::vector<expr_t>> vec_vars_;
    object_set_t<expr_t> vars_with_load_;
};

stmt_t fix_int32_overflow(const stmt_t &s, ir_context_t &ir_ctx) {
    trace_start();
    auto ret = overflow_fixer_t(ir_ctx).mutate(s);
    trace_pass("fix_int32_overflow", ret, ir_ctx);
    return ret;
}

class peephole_optimizer_t : public ir_mutator_t {
public:
    object_t _mutate(const binary_op_t &obj) override {
        auto old_obj = ir_mutator_t::_mutate(obj);
        auto new_obj
                = simplify_rewrite_with_ternary(old_obj, /*recursive=*/false);
        auto *ternary = new_obj.as_ptr<ternary_op_t>();
        if (!ternary) return std::move(new_obj);

        switch (ternary->op_kind) {
            case op_kind_t::_add3: {
                bool ok = true;
                // Allowed form: add3(dword/word, dword/word, dword/word).
                ok &= add3_type_ok(ternary->a);
                ok &= add3_type_ok(ternary->b);
                ok &= add3_type_ok(ternary->c);
                ok &= !is_const(ternary->a);
                ok &= !is_const(ternary->b);
                if (!ok) new_obj = old_obj;
                break;
            }
            case op_kind_t::_mad: {
                bool ok = false;
                if (try_int_mad(ternary))
                    ok = true;
                else if (try_float_mad(ternary))
                    ok = true;
                if (!ok) new_obj = old_obj;
                break;
            }
            default: ir_error_not_expected();
        }
        return std::move(new_obj);
    }

private:
    static type_t real_type(const expr_t &e) {
        auto *imm = e.as_ptr<int_imm_t>();
        if (!imm) return e.type();
        if (int_imm_t::try_shrink_type<int16_t>(imm->value))
            return type_t::s16();
        if (int_imm_t::try_shrink_type<int32_t>(imm->value))
            return type_t::s32();
        return type_t::s64();
    }

    static bool try_int_mad(const ternary_op_t *ternary) {
        auto a_type = real_type(ternary->a);
        auto b_type = real_type(ternary->b);
        auto c_type = real_type(ternary->c);
        bool ok = true;
        // Allowed form: mad(dword, dword, word).
        ok &= utils::one_of(a_type, type_t::s32(), type_t::u32());
        ok &= utils::one_of(b_type, type_t::s32(), type_t::u32());
        ok &= utils::one_of(c_type, type_t::s16(), type_t::u16());
        return ok;
    }

    static bool try_float_mad(const ternary_op_t *ternary) {
        auto op_ok = [](const expr_t &e) {
            if (is_const(e) || is_const_broadcast(e)) return false;
            if (!e.type().is_f32()) return false;
            return true;
        };
        if (!op_ok(ternary->a)) return false;
        if (!op_ok(ternary->b)) return false;
        if (!op_ok(ternary->c)) return false;
        return true;
    }

    static bool add3_type_ok(const expr_t &e) {
        auto t = real_type(e);
        if (!t.is_scalar()) return false;
        switch (t.kind()) {
            case type_kind_t::s32:
            case type_kind_t::u32: return !is_const(e);
            case type_kind_t::s16:
            case type_kind_t::u16: return true;
            default: return false;
        }
    }
};

stmt_t optimize_peephole(const stmt_t &s, ir_context_t &ir_ctx) {
    trace_start();
    auto ret = peephole_optimizer_t().mutate(s);
    trace_pass("optimize_peephole", ret, ir_ctx);
    return ret;
}

class barrier_optimizer_t : public ir_mutator_t {
public:
    object_t _mutate(const for_t &obj) override {
        loop_level_++;
        auto new_obj = ir_mutator_t::_mutate(obj);
        loop_level_--;
        return new_obj;
    }

    object_t _mutate(const func_call_t &obj) override {
        if (is_func_call<send_t>(obj)) {
            auto &send = obj.func.as<send_t>();
            if (send.is_slm()) can_remove_barrier_ = false;
        } else if (obj.func.is_same(funcs::barrier_func())) {
            bool can_remove = can_remove_barrier_;
            can_remove_barrier_ = false;

            // If not in a loop and this is the first barrier -> can be removed.
            if (loop_level_ == 0 && can_remove) return stmt_t();
            return obj;
        }

        return obj;
    }

    // Store doesn't contain nested statements, return as is.
    object_t _mutate(const store_t &obj) override { return obj; }

private:
    int loop_level_ = 0;
    bool can_remove_barrier_ = true;
};

stmt_t optimize_barrier(const stmt_t &s, ir_context_t &ir_ctx) {
    trace_start();
    auto ret = barrier_optimizer_t().mutate(s);
    trace_pass("optimize_barrier", ret, ir_ctx);
    return ret;
}

class if_condition_fixer_t : public ir_mutator_t {
public:
    if_condition_fixer_t(int simd_size) : simd_size_(simd_size) {}

    object_t _mutate(const if_t &obj) override {
        auto _new_obj = ir_mutator_t::_mutate(obj);
        auto &new_obj = _new_obj.as<if_t>();
        auto cond = shuffle_t::make_broadcast(new_obj.cond, simd_size_);
        return if_t::make(cond, new_obj.body, new_obj.else_body);
    }

private:
    int simd_size_;
};

stmt_t fixup_if_conditions(const stmt_t &s, ir_context_t &ir_ctx) {
    trace_start();
    auto ret = if_condition_fixer_t(ir_ctx.hw_cfg().simd_size()).mutate(s);
    trace_pass("fixup_if_conditions", ret, ir_ctx);
    return ret;
}

class loop_unroller_t : public ir_mutator_t {
public:
    loop_unroller_t(ir_context_t &ir_ctx) : ir_ctx_(ir_ctx) {}

    object_t _mutate(const for_t &obj) override {
        auto new_obj = ir_mutator_t::_mutate(obj);
        auto &_for = new_obj.as<for_t>();
        // No unrolling.
        if (_for.unroll == 1) return new_obj;

        ir_assert(is_const(obj.init))
                << "Can't unroll loop with non-const bound: " << obj.init;
        ir_assert(is_const(obj.bound))
                << "Can't unroll loop with non-const bound: " << obj.bound;

        auto init = to_cpp<int>(obj.init);
        auto bound = to_cpp<int>(obj.bound);

        ir_assert(_for.unroll == (bound - init))
                << "Only full loop unroll is supported.";

        stmt_t ret;
        for (int i = init; i < bound; i++) {
            auto iter_stmt
                    = substitute(obj.body, obj.var, to_expr(i, obj.var.type()));
            iter_stmt = rename_let_alloc(iter_stmt, i - init);
            ret = ret.append(iter_stmt);
        }
        return std::move(ret);
    }

private:
    stmt_t rename_let_alloc(const stmt_t &s, int idx) {
        auto lets = find_objects<let_t>(s);
        auto ret = s;
        for (auto &_let : lets) {
            auto &let = _let.as<let_t>();
            auto &var = let.var.as<var_t>();
            auto new_var = ir_ctx_.create_tmp_var(var.type, var.name);
            ret = substitute(ret, let.var, new_var);
        }
        auto allocs = find_objects<alloc_t>(s);
        for (auto &_alloc : allocs) {
            auto &alloc = _alloc.as<alloc_t>();
            auto &buf = alloc.buf.as<var_t>();
            auto new_buf = ir_ctx_.create_tmp_var(buf.type, buf.name);
            ret = substitute(ret, alloc.buf, new_buf);
        }
        return ret;
    }

    ir_context_t &ir_ctx_;
};

stmt_t unroll_loops(const stmt_t &s, ir_context_t &ir_ctx) {
    trace_start();
    auto ret = loop_unroller_t(ir_ctx).mutate(s);
    trace_pass("unroll_loops", ret, ir_ctx);
    return ret;
}

class bank_conflict_attribute_injector_t : public ir_mutator_t {
public:
    object_t _mutate(const alloc_t &obj) override {
        all_buf_sizes_.emplace(obj.buf, obj.size);

        auto new_obj = ir_mutator_t::_mutate(obj);
        if (bufs_.count(obj.buf) == 0) return new_obj;

        init_attr();

        auto new_attrs = obj.attrs;
        new_attrs.push_back(attr_);
        auto &body = new_obj.as<alloc_t>().body;
        return alloc_t::make(obj.buf, obj.size, obj.kind, new_attrs, body);
    }

    object_t _mutate(const func_call_t &obj) override {
        if (is_frozen) return ir_mutator_t::_mutate(obj);

        bool is_mad = obj.func.is<mad_t>();
        bool is_dpas = obj.func.is<dpas_t>();
        auto *send = obj.func.as_ptr<send_t>();
        bool is_load = send && (send->is_load() || send->is_load_2d());

        if (is_mad || is_dpas) {
            auto dst_buf = ptr_base(obj.args[0]);
            auto src0_buf = ptr_base(obj.args[1]);
            auto src1_buf = ptr_base(obj.args[2]);
            auto src2_buf = ptr_base(obj.args[3]);

            // src0 may be null in some cases, skip it.
            if (!src0_buf.is_empty()) bufs_.insert(src0_buf);
            bufs_.insert(src1_buf);
            bufs_.insert(src2_buf);

            instructions_.insert(obj);
        } else if (is_load) {
            // Returns minimal 2^B so that there is x such that:
            //   x * 2^B <= a <= b < (x + 1) * 2^B
            auto min_pow2_span = [](int a, int b) {
                int same_left_bits = 0;
                for (int i = 31; i >= 0; i--) {
                    int b0 = ((uint32_t)a >> i) & 0x1;
                    int b1 = ((uint32_t)b >> i) & 0x1;
                    if (b0 != b1) break;
                    same_left_bits++;
                }
                return 1 << (32 - same_left_bits);
            };
            auto &buf = send_t::arg_reg_buf(obj);
            auto &base = (is_var(buf) ? buf : buf.as<ptr_t>().base);
            int off = (is_var(buf) ? 0 : to_cpp<int>(buf.as<ptr_t>().off));
            int size = send->payload_size();
            int span = min_pow2_span(off, off + size - 1);
            int &min_block_size = all_buf_min_block_sizes[base];
            min_block_size = std::max(min_block_size, span);
        }
        return ir_mutator_t::_mutate(obj);
    }

private:
    void init_attr() {
        if (!attr_.is_empty()) return;

        is_frozen = true;
        std::vector<stmt_t> instructions;
        for (auto &s : instructions_)
            instructions.push_back(s);

        std::vector<expr_t> buf_vec;
        std::vector<int> buf_sizes;
        std::vector<int> buf_min_block_sizes;
        for (auto &buf : bufs_) {
            buf_vec.push_back(buf);
            buf_sizes.push_back(all_buf_sizes_.at(buf));
            auto it = all_buf_min_block_sizes.find(buf);
            int min_block_size
                    = (it == all_buf_min_block_sizes.end() ? 0 : it->second);
            buf_min_block_sizes.push_back(min_block_size);
        }
        attr_ = bank_conflict_attr_t::make(
                buf_vec, buf_sizes, buf_min_block_sizes, instructions);
    }

    static expr_t ptr_base(const expr_t &e) {
        if (e.is<var_t>()) return e;
        auto *ptr = e.as_ptr<ptr_t>();
        if (ptr) return e.as<ptr_t>().base;
        return expr_t();
    }

    object_map_t<expr_t, int> all_buf_sizes_;
    object_map_t<expr_t, int> all_buf_min_block_sizes;
    object_eq_set_t<expr_t> bufs_;
    object_eq_set_t<stmt_t> instructions_;
    bool is_frozen = false;

    alloc_attr_t attr_;
};

stmt_t inject_bank_conflict_attribute(const stmt_t &s, ir_context_t &ir_ctx) {
    trace_start();
    auto ret = bank_conflict_attribute_injector_t().mutate(s);
    trace_pass("inject_bank_conflict_attribute", ret, ir_ctx);
    return ret;
}

class dp4a_injector_t : public ir_mutator_t {
public:
    object_t _mutate(const func_call_t &obj) {
        auto *dpas = obj.func.as_ptr<dpas_t>();
        if (!dpas) return obj;

        int M = dpas->exec_size;
        int N = dpas->rcount;
        int K = dpas->sdepth * 4;

        auto &dst = dpas_t::arg_dst(obj);
        auto &src0 = dpas_t::arg_src0(obj);
        auto &src1 = dpas_t::arg_src1(obj);
        auto &src2 = dpas_t::arg_src2(obj);
        int dst_size = dpas->dst_type.size();
        int src0_size = dpas->dst_type.size();
        int src1_size = dpas->src1_type.size();
        int src2_size = dpas->src2_type.size();
        auto dst_type = to_dp4a_type(dpas->dst_type);
        auto src1_type = to_dp4a_type(dpas->src1_type);
        auto src2_type = to_dp4a_type(dpas->src2_type);
        bool is_src0_zero = is_zero(src0);

        stmt_t stmt;
        auto _dp4a = dpas_t::make(
                /*is_dpasw=*/false, M, 1, 1, dst_type, src1_type, src2_type);
        auto &dp4a = _dp4a.as<dpas_t>();
        auto zero = shuffle_t::make_broadcast(0, M);
        int k0 = (is_src0_zero ? -4 : 0);
        for (int k = k0; k < K; k += 4) {
            for (int n = 0; n < N; n++) {
                int dst_off = n * M * dst_size;
                int src0_off = n * M * src0_size;
                int src1_off = k * M * src1_size;
                int src2_off = (n * K + k) * src2_size;
                auto _dst = dst + dst_off;
                auto _src0 = is_src0_zero ? _dst : (src0 + src0_off);
                auto _src1 = src1 + src1_off;
                auto _src2 = src2 + src2_off;
                if (k < 0) {
                    stmt = stmt.append(store_t::make(_dst, 0, zero));
                } else {
                    stmt = stmt.append(dp4a(_dst, _src0, _src1, _src2));
                }
            }
        }
        return std::move(stmt);
    }

private:
    static type_t to_dp4a_type(const type_t &type) {
        if (type.is_x32()) return type;
        if (type.is_s8()) return type_t::s32();
        if (type.is_u8()) return type_t::u32();
        ir_error_not_expected();
        return type_t();
    };
};

stmt_t inject_dp4a(const stmt_t &s, ir_context_t &ir_ctx) {
    trace_start();
    auto ret = dp4a_injector_t().mutate(s);
    trace_pass("inject_dp4a", ret, ir_ctx);
    return ret;
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
