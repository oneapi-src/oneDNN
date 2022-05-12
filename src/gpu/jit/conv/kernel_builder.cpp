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

#include "gpu/jit/conv/kernel_builder.hpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <limits>
#include <memory>
#include <utility>
#include <vector>
#include <unordered_map>

#include "gpu/jit/conv/config.hpp"
#include "gpu/jit/conv/fma_support.hpp"
#include "gpu/jit/conv/gemm_schedule.hpp"
#include "gpu/jit/conv/ir.hpp"
#include "gpu/jit/conv/message_support.hpp"
#include "gpu/jit/conv/post_op_support.hpp"
#include "gpu/jit/conv/reduce_support.hpp"
#include "gpu/jit/conv/reorder_support.hpp"
#include "gpu/jit/conv/tensor.hpp"

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
            if (new_call.is_same(call)) return 0;
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

// Transforms DPAS to DPASW.
void inject_dpasw(ngen::HW hw, stmt_t &load_mul_stmt, const expr_t &c_buf,
        stmt_t &c_store_stmt, alloc_updater_t &alloc_updater,
        const expr_t &tg_idx0) {
    dpasw_injector_t injector(
            hw, load_mul_stmt, c_buf, c_store_stmt, alloc_updater, tg_idx0);
    injector.inject();

    load_mul_stmt = injector.load_mul_stmt();
    c_store_stmt = injector.c_store_stmt();
}

// Adds {Atomic} modifier to DPAS/DPASW instructions when applicable.
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

// Trace for debugging purposes.
void trace_pass(const char *pass_name, const stmt_t &stmt) {
    ir_trace() << "=== After " << pass_name << std::endl;
    ir_trace() << stmt << std::endl;
}

class external_var_visitor_t : public scope_visitor_t {
public:
    void _visit(const var_t &obj) {
        if (!is_expr_defined(obj)) external_vars.insert(obj);
    }

    object_eq_set_t<expr_t> external_vars;
};

stmt_t inject_external_var_let(const stmt_t &_stmt) {
    auto stmt = _stmt;
    external_var_visitor_t v;
    v.visit(stmt);

    for (auto &var : v.external_vars)
        stmt = let_t::make(var, {}, stmt);

    trace_pass("inject_external_var_let", stmt);
    return stmt;
}

class slm_zp_mask_extractor_t : public ir_visitor_t {
public:
    slm_zp_mask_extractor_t(
            std::vector<stmt_t> &retn, object_eq_set_t<expr_t> &bufs)
        : retn_(retn), bufs_(bufs), outer_(true) {}

    void _visit(const store_t &obj) override {
        if (obj.buf.str().find("zp_mask") == 0) {
            if (outer_) retn_.emplace_back(obj);
            bufs_.insert(obj.buf);
        }
    }

    void _visit(const let_t &obj) override {
        if ((obj.var.str().find("zp_mask") == 0)) {
            if (outer_) retn_.emplace_back(obj);
            auto outer_prev = outer_;
            outer_ = false;
            visit(obj.body);
            outer_ = outer_prev;
        }
    }

private:
    std::vector<stmt_t> &retn_;
    object_eq_set_t<expr_t> &bufs_;
    bool outer_;
};

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

// Merges all SLM buffers into a single one.
stmt_t merge_slm_buffers(const stmt_t &_stmt) {
    stmt_t stmt = _stmt;
    slm_buffer_merger_t merger;
    stmt = merger.mutate(stmt);
    stmt = alloc_t::make(
            merger.slm_base(), merger.slm_size(), alloc_kind_t::slm, stmt);
    trace_pass("merge_slm_buffers", stmt);
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

stmt_t lift_buffer_offsets_in_send(const stmt_t &s) {
    buffer_offset_lifter_t lifter;
    auto ret = lifter.mutate(s);
    trace_pass("lift_buffer_offsets_in_send", ret);
    return ret;
}

stmt_t simplify_pass(const stmt_t &s, const constraint_set_t &cset) {
    auto ret = simplify(s, cset);
    trace_pass("simplify_pass", ret);
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

// Replaces some heavy GRF reorders by reorder through SLM (store and load).
stmt_t inject_slm_reorder(
        const stmt_t &s, const conv_config_t &cfg, const grid_info_t &tg_grid) {
    if (cfg.use_a_slm || cfg.use_b_slm) return s;
    slm_reorder_injector_t injector(s, cfg, tg_grid);
    stmt_t ret = injector.mutate(s);

    auto &slm_buf = injector.slm_base();
    int slm_size = injector.slm_size();
    alloc_updater_t alloc_updater;
    alloc_updater.resize(slm_buf, slm_size);
    ret = alloc_updater.update(ret);

    trace_pass("inject_slm_reorder", ret);
    return ret;
}

class send_injector_t : public ir_mutator_t {
public:
    send_injector_t(ir_context_t &ir_ctx, const constraint_set_t &cset)
        : ir_ctx_(ir_ctx), cset_(cset) {}

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
        value = simplify(value, cset_);

        // Convert to N-ary form and back to expand multiplications. This
        // helps to find more common subexpressions during the pass.
        value = nary_op_canonicalize(value);
        value = nary_op_back_transform(value);

        return store_t::make(store.buf, store.off, value, store.stride);
    }

    ir_context_t &ir_ctx_;
    const constraint_set_t &cset_;
};

stmt_t inject_send(
        const stmt_t &s, ir_context_t &ir_ctx, const constraint_set_t &cset) {
    auto ret = send_injector_t(ir_ctx, cset).mutate(s);
    trace_pass("inject_send", ret);
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

// Lifts alloc statements out of loops.
stmt_t lift_alloc(const stmt_t &s, const conv_config_t &cfg) {
    auto ret = alloc_lifter_t(s, cfg.reuse_headers).mutate(s);
    trace_pass("lift_alloc", ret);
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

// Lifts loop-invariant header assignments related to block 2D messages.
stmt_t lift_send_2d_header_store(const stmt_t &s) {
    auto ret = send_2d_header_store_lifter_t(s).mutate(s);
    trace_pass("lift_send_2d_header_store", ret);
    return ret;
}

// Common subexpression elimination support.

// Represents an expression-candidate to eliminate.
class cse_expr_t {
public:
    cse_expr_t(const expr_t &expr, const expr_t &orig_expr,
            const ir_path_t &path, int refs = 1, const expr_t &cse_var = {})
        : expr(expr)
        , orig_expr(orig_expr)
        , path(path)
        , refs(refs)
        , cse_var(cse_var) {
        ir_trace() << "cse_pass: add expression: " << expr << std::endl;
    }

    void add_usage(const ir_path_t &other_path, bool do_increment = true) {
        if (do_increment) refs++;
        path.merge(other_path);
        ir_trace() << "cse_pass: add usage: " << expr
                   << ", total refs: " << refs << std::endl;
    }

    // Expression to eliminate via let.
    expr_t expr;
    // Original expression to eliminate (doesn't contain any CSEed vars).
    expr_t orig_expr;
    // Path to the innermost IR node where the expression can be defined.
    ir_path_t path;
    // Number of references to the expression.
    int refs;
    // Variable assigned to the expression (if decided to eliminate).
    expr_t cse_var;
};

// Stores information about all expressions subject to CSEing.
class cse_context_t {
public:
    cse_context_t(ir_context_t &ir_ctx) : ir_ctx_(ir_ctx) {}

    int memory_usage() const { return memory_usage_; }

    ir_context_t &ir_ctx() { return ir_ctx_; }

    bool has(const expr_t &e) const { return cse_exprs_.count(e) != 0; }

    cse_expr_t &find_cse_expr(const expr_t &e) {
        ir_assert(has(e)) << e;
        return cse_exprs_.at(e);
    }

    const cse_expr_t &find_cse_expr(const expr_t &e) const {
        ir_assert(has(e)) << e;
        return cse_exprs_.at(e);
    }

    bool has_var(const expr_t &e) const {
        return !find_cse_expr(e).cse_var.is_empty();
    }

    int get_refs(const expr_t &e) const {
        if (!has(e)) return 0;
        return find_cse_expr(e).refs;
    }

    void register_expr(const expr_t &e, const ir_path_t &path) {
        if (e.type().is_bool()) return; // Ignore booleans.
        auto ret = cse_exprs_.insert({e, cse_expr_t(e, e, path)});
        ir_assert(ret.second) << e;
        MAYBE_UNUSED(ret);
    }

    void register_expr(const cse_expr_t &cse_expr) {
        auto ret = cse_exprs_.insert({cse_expr.expr, cse_expr});
        ir_assert(ret.second);
        MAYBE_UNUSED(ret);
    }

    expr_t get_or_assign_var(const expr_t &e) {
        auto &cse_expr = find_cse_expr(e);
        if (cse_expr.cse_var.is_empty()) {
            memory_usage_ += e.type().size();
            cse_expr.cse_var = ir_ctx_.create_tmp_var(e.type());
            ir_trace() << "cse_pass: assigning var: " << e << " -> "
                       << cse_expr.cse_var << std::endl;
        }
        return cse_expr.cse_var;
    }

    const expr_t &get_var(const expr_t &e) const {
        return find_cse_expr(e).cse_var;
    }

    const ir_path_t &get_path(const expr_t &e) const {
        return find_cse_expr(e).path;
    }

    void add_usage(
            const expr_t &e, const ir_path_t &path, bool do_increment = true) {
        if (e.type().is_bool()) return; // Ignore booleans.
        if (e.type().is_u64() && e.type().elems() >= 8) return;
        return find_cse_expr(e).add_usage(path, do_increment);
    }

    void update_expr(const expr_t &old_expr, const expr_t &new_expr) {
        auto it = cse_exprs_.find(old_expr);
        ir_assert(it != cse_exprs_.end()) << old_expr;
        auto &old_cse_expr = it->second;
        auto new_cse_expr = cse_expr_t(new_expr, old_cse_expr.orig_expr,
                old_cse_expr.path, old_cse_expr.refs, old_cse_expr.cse_var);
        cse_exprs_.erase(it);
        auto ret = cse_exprs_.insert({new_expr, new_cse_expr});
        ir_assert(ret.second);
        MAYBE_UNUSED(ret);
    }

    template <typename F>
    void for_each(const F &f) const {
        for (auto &kv : cse_exprs_)
            f(kv.first);
    }

    bool should_assign_var(const expr_t &e) const {
        if (get_refs(e) <= 1) return false;
        if (skip_exprs_.count(e) != 0) return false;
        return true;
    }

    void set_skip_exprs(int limit) {
        using cost_entry_t = std::pair<const cse_expr_t *, int>;
        std::vector<cost_entry_t> costs;
        for (auto &kv : cse_exprs_) {
            auto &cse_expr = kv.second;
            costs.emplace_back(&cse_expr, expr_cost(cse_expr.expr));
        }
        // Order expressions candidates for CSE by their computation cost.
        std::sort(costs.begin(), costs.end(),
                [&](const cost_entry_t &a, const cost_entry_t &b) {
                    return a.second > b.second;
                });
        // Add the most lightweight expressions to the skip list to prepare for
        // the retry run.
        int usage = 0;
        for (auto &e : costs) {
            auto &expr = e.first->orig_expr;
            usage += expr.type().size();
            if (usage > limit) skip_exprs_.insert(expr);
        }
    }

    void reset_cse_exprs() {
        cse_exprs_.clear();
        memory_usage_ = 0;
    }

private:
    static int expr_cost(const expr_t &e) {
        if (is_var(e)) return 0;
        if (is_const(e)) return 0;
        if (e.is<cast_t>()) return 0;
        if (auto *op = e.as_ptr<binary_op_t>()) {
            return expr_cost(op->a) + expr_cost(op->b) + 1;
        }
        if (auto *op = e.as_ptr<unary_op_t>()) { return expr_cost(op->a) + 1; }
        if (auto *s = e.as_ptr<shuffle_t>()) {
            if (s->is_broadcast()) return 0;
            return s->elems();
        }
        ir_error_not_expected() << "Unhandled expression: " << e;
        return 0;
    }

    ir_context_t &ir_ctx_;
    object_eq_map_t<expr_t, cse_expr_t> cse_exprs_;
    object_eq_set_t<expr_t> skip_exprs_;

    int memory_usage_ = 0;
};

// Collects statistics about expressions for common subexpression elimination.
class cse_visitor_t : public ir_visitor_t {
public:
    cse_visitor_t(cse_context_t &ctx) : ctx_(ctx) {}

    void _visit(const binary_op_t &obj) override { visit_expr(obj); }
    void _visit(const shuffle_t &obj) override {
        if (is_const_broadcast(obj)) return;
        visit_expr(obj);
    }
    void _visit(const unary_op_t &obj) override { visit_expr(obj); }

#define HANDLE_IR_OBJECT(type) \
    void _visit(const type &obj) override { visit_stmt(obj); }

    HANDLE_STMT_IR_OBJECTS()

#undef HANDLE_IR_OBJECT

private:
    template <typename T>
    void visit_expr(const T &obj) {
        // Exclude loads as they may have side effects.
        if (count_objects<load_t>(obj) > 0) {
            ir_visitor_t::_visit(obj);
            return;
        }

        if (std::is_same<T, shuffle_t>::value) {
            auto &shuffle = reinterpret_cast<const shuffle_t &>(obj);
            if (shuffle.is_broadcast()) {
                ir_visitor_t::_visit(obj);
                return;
            }
        }

        if (propagate_path_) {
            if (ctx_.has(obj))
                ctx_.add_usage(obj, root_path_, /*do_increment=*/false);
            ir_visitor_t::_visit(obj);
            return;
        }
        if (ctx_.has(obj)) {
            ctx_.add_usage(obj, root_path_);
            propagate_path_ = true;
            ir_visitor_t::_visit(obj);
            propagate_path_ = false;
            return;
        }
        ir_visitor_t::_visit(obj);
        ctx_.register_expr(obj, root_path_);
    }

    template <typename T>
    void visit_stmt(const T &obj) {
        if (std::is_same<T, for_t>::value) {
            visit_for((const object_impl_t &)obj);
            return;
        }
        if (std::is_same<T, let_t>::value) {
            visit_let((const object_impl_t &)obj);
            return;
        }
        root_path_.push(&obj);
        ir_visitor_t::_visit(obj);
        root_path_.pop();
    }

    void visit_for(const object_impl_t &_obj) {
        auto &obj = (const for_t &)_obj;

        visit(obj.var);
        visit(obj.init);
        visit(obj.bound);
        root_path_.push(&obj);
        visit(obj.body);
        root_path_.pop();
    }

    void visit_let(const object_impl_t &_obj) {
        auto &obj = (const let_t &)_obj;

        visit(obj.var);
        visit(obj.value);
        root_path_.push(&obj);
        visit(obj.body);
        root_path_.pop();
    }

    cse_context_t &ctx_;
    ir_path_t root_path_;

    bool propagate_path_ = false;
};

// Verifies all IR paths are correct (for debugging purposes).
class cse_verifier_t : public scope_visitor_t {
public:
    cse_verifier_t(cse_context_t &ctx) : ctx_(ctx) {}

    ~cse_verifier_t() override { ir_assert(to_check_.empty()); }

    void _visit(const binary_op_t &obj) override { visit_expr(obj); }
    void _visit(const shuffle_t &obj) override { return visit_expr(obj); }
    void _visit(const unary_op_t &obj) override { visit_expr(obj); }

#define HANDLE_IR_OBJECT(type) \
    void _visit(const type &obj) override { visit_stmt(obj); }

    HANDLE_STMT_IR_OBJECTS()

#undef HANDLE_IR_OBJECT

    void verify(const stmt_t &s) {
        // Phase 0: collect IR paths for expressions.
        phase_ = 0;
        visit(s);

        // Phase 1: verify all expressions are defined at their path.
        phase_ = 1;
        visit(s);
    }

private:
    template <typename T>
    void visit_expr(const T &obj) {
        // Expressions are not used during phase 1.
        if (phase_ == 1) return;
        if (ctx_.has(obj)) {
            auto &path = ctx_.get_path(obj);
            to_check_[path.back()].push_back(obj);
        }
        scope_visitor_t::_visit(obj);
    }

    template <typename T>
    void visit_stmt(const T &obj) {
        scope_visitor_t::_visit(obj);

        // Statements are not used during phase 0.
        if (phase_ == 0) return;

        // Phase 1: check that all attached expressions are defined at this
        // statement.
        auto it = to_check_.find(obj);
        if (it != to_check_.end()) {
            for (auto &e : it->second) {
                ir_assert(is_expr_defined(e))
                        << "Expression contains undefined variables: " << e;
                MAYBE_UNUSED(e);
            }
            to_check_.erase(it);
        }
    }

    cse_context_t &ctx_;

    int phase_ = 0;
    object_map_t<stmt_t, std::vector<expr_t>> to_check_;
};

// Generates let statements for expressions being eliminated.
class cse_let_generator_t : public ir_visitor_t {
public:
    cse_let_generator_t(const cse_context_t &ctx, const stmt_t &stmt)
        : ctx_(ctx), stmt_(stmt) {}

    void _visit(const binary_op_t &obj) override { visit_expr(obj); }
    void _visit(const shuffle_t &obj) override { visit_expr(obj); }
    void _visit(const unary_op_t &obj) override { visit_expr(obj); }
    void _visit(const var_t &obj) override {
        auto it = all_vars_.find(obj);
        if (it == all_vars_.end()) return;
        if (seen_vars_.count(obj) == 0) generate_for_expr(it->second);
    }

    stmt_t generate() {
        ctx_.for_each([&](const expr_t &e) {
            auto &cse_var = ctx_.get_var(e);
            auto ret = all_vars_.insert({cse_var, e});
            ir_assert(ret.second);
            MAYBE_UNUSED(ret);
        });
        ctx_.for_each([&](const expr_t &e) { generate_for_expr(e); });
        for (auto it = lets_.rbegin(); it != lets_.rend(); ++it) {
            auto &let = it->as<let_t>();
            stmt_ = let_t::make(let.var, let.value, stmt_);
        }
        return stmt_;
    }

private:
    void generate_for_expr(const expr_t &e) {
        auto &cse_var = ctx_.get_var(e);
        if (seen_vars_.count(cse_var) == 1) return;
        visit(e);
    }

    template <typename T>
    void visit_expr(const T &obj) {
        ir_visitor_t::_visit(obj);
        if (ctx_.has(obj) && ctx_.has_var(obj)) {
            auto &var = ctx_.get_var(obj);
            auto ret = seen_vars_.insert(var);
            if (ret.second) lets_.push_back(let_t::make(var, obj));
        }
    }

    const cse_context_t &ctx_;
    stmt_t stmt_;

    object_map_t<expr_t, expr_t> all_vars_; // Var -> expression.
    object_set_t<expr_t> seen_vars_;

    std::vector<stmt_t> lets_;
};

// Eliminates expressions from the statement.
class cse_mutator_t : public ir_mutator_t {
public:
    cse_mutator_t(cse_context_t &ctx) : ctx_(ctx) {}

    object_t _mutate(const binary_op_t &obj) override {
        return mutate_expr(obj);
    }
    object_t _mutate(const shuffle_t &obj) override { return mutate_expr(obj); }
    object_t _mutate(const unary_op_t &obj) override {
        return mutate_expr(obj);
    }

#define HANDLE_IR_OBJECT(type) \
    object_t _mutate(const type &obj) override { return mutate_stmt(obj); }

    HANDLE_STMT_IR_OBJECTS()

#undef HANDLE_IR_OBJECT

private:
    template <typename T>
    object_t mutate_expr(const T &obj) {
        auto new_obj = ir_mutator_t::_mutate(obj);
        if (ctx_.has(obj) && !new_obj.is_equal(obj)) {
            ctx_.update_expr(obj, new_obj);
        }
        if (ctx_.should_assign_var(new_obj)) {
            bool has_var = ctx_.has_var(new_obj);
            auto var = ctx_.get_or_assign_var(new_obj);
            auto &path = ctx_.get_path(new_obj);
            if (!has_var) to_update_[path.back()].push_back(new_obj);
            return std::move(var);
        }
        return new_obj;
    }

    template <typename T>
    object_t mutate_stmt(const T &obj) {
        auto new_obj = ir_mutator_t::_mutate(obj);
        auto it = to_update_.find(obj);
        if (it == to_update_.end()) return new_obj;

        cse_context_t local_ctx(ctx_.ir_ctx());
        for (auto &e : it->second) {
            local_ctx.register_expr(ctx_.find_cse_expr(e));
        }
        to_update_.erase(it);

        auto body = get_stmt_body(new_obj);
        cse_let_generator_t g(local_ctx, body);
        body = g.generate();
        new_obj = replace_stmt_body(new_obj, body);
        return new_obj;
    }

    cse_context_t &ctx_;
    object_map_t<stmt_t, std::vector<expr_t>> to_update_;
};

stmt_t eliminate_common_subexprs_impl(
        const stmt_t &_stmt, cse_context_t &ctx, int memory_usage_limit) {
    auto stmt = _stmt;

    // Collect statistics.
    cse_visitor_t visitor(ctx);
    visitor.visit(stmt);

#ifndef NDEBUG
    // Verify that collected IR paths are correct (cse_expr_t objects are
    // defined at those paths).
    cse_verifier_t verifier(ctx);
    verifier.verify(stmt);
#endif

    // Eliminate subexpressions.
    cse_mutator_t mutator(ctx);
    stmt = mutator.mutate(stmt);

    // If memory usage exceeds the limit, exclude some
    // expressions from CSE and retry the whole process from
    // scratch.
    int memory_usage = ctx.memory_usage();
    if (memory_usage > memory_usage_limit) {
        ir_trace() << "CSE exceeded GRF usage limit. Usage: " << memory_usage
                   << ", limit: " << memory_usage_limit
                   << ". Retry CSE and skip some expressions..." << std::endl;
        ctx.set_skip_exprs(memory_usage_limit);
        ctx.reset_cse_exprs();
        return stmt_t();
    }

    trace_pass("eliminate_common_subexprs", stmt);
    return stmt;
}

stmt_t eliminate_common_subexprs(const stmt_t &_stmt, ir_context_t &ir_ctx) {
    stmt_t stmt;
    cse_context_t cse_ctx(ir_ctx);

    // Max amount of GRF memory allowed to use for temporary variables.
    const int memory_usage_limit = 1024;

    stmt = eliminate_common_subexprs_impl(_stmt, cse_ctx, memory_usage_limit);
    // Retry if statement is empty, rely on the updated
    // skip_exprs from the CSE context.
    if (stmt.is_empty()) {
        stmt = eliminate_common_subexprs_impl(
                _stmt, cse_ctx, memory_usage_limit);
    }
    trace_pass("eliminate_common_subexprs", stmt);
    return stmt;
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

// Moves invariant expressions out of loops.
stmt_t hoist_exprs(const stmt_t &s, ir_context_t &ir_ctx) {
    auto ret = hoist_exprs_mutator_t(ir_ctx).mutate(s);
    trace_pass("hoist_exprs", ret);
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

// Moves boolean mask computation from send calls to the top of the statement
// group corresponding to `label`. This is done to reduce GRF consumption and
// to reuse masks between calls. A vector boolean mask is stored as u16 type
// and converted to bool type right before the call. Transformation is limited
// to the statement group corresponding to `label`.
// If `split_by_and` is true then any ((A & B) & C) mask is split into A, B, C
// sub-masks which are initialized independently. This allows reusing those
// sub-masks for other masks.
stmt_t hoist_send_masks(const stmt_t &s, ir_context_t &ir_ctx,
        const stmt_label_t &label, bool split_by_and) {
    hoist_send_masks_mutator_t mutator(ir_ctx, label, split_by_and);

    auto ret = mutator.mutate(s);
    trace_pass("hoist_send_masks", ret);
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

// Removes redundant u16 casts inside send masks which may appear after
// previous mask hoisting.
stmt_t remove_spurious_send_mask_cast(const stmt_t &s) {
    spurious_send_mask_cast_remover_t mutator;
    auto ret = mutator.mutate(s);
    trace_pass("remove_spurious_send_mask_cast", ret);
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
stmt_t loop_strength_reduce(const stmt_t &s) {
    auto ret = loop_strength_reducer_t().mutate(s);
    trace_pass("loop_strength_reduce", ret);
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

stmt_t optimize_alloc_let(const stmt_t &s) {
    auto ret = alloc_let_optimizer_t().mutate(s);
    trace_pass("optimize_alloc_let", ret);
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

// Eliminates let statements from the outer loops to be able to unroll loop
// nest for SLM buffering or prefetch injection. Example:
// Before:
//     for (int i = 0; i < I; i++) {
//         int tmp = TMP;
//         for (int j = 0; j < J; j++) {
//            ...
//         }
//     }
// After:
//     for (int i = 0; i < I; i++) {
//         for (int j = 0; j < J; j++) {
//             int tmp = TMP;
//             ...
//         }
//     }
stmt_t update_loops_for_unrolling(const stmt_t &s, const conv_config_t &cfg) {
    auto ret = s;
    if (cfg.do_pipeline_unroll) ret = unrolling_updater_t().mutate(s);
    trace_pass("update_loops_for_unrolling", ret);
    return ret;
}

// Helper structure for for_t.
struct loop_info_t {
    loop_info_t() = default;

    loop_info_t(const stmt_t &s) {
        ir_assert(s.is<for_t>()) << s;
        auto &loop = s.as<for_t>();
        stmt = s;
        var = loop.var;
        init_ = loop.init;
        bound_ = loop.bound;

        auto e_size = simplify(bound_ - init_);
        ir_assert(is_const(e_size));
        size_ = to_cpp<int>(e_size);
    }

    int init() const {
        ir_assert(is_const(init_));
        return to_cpp<int>(init_);
    }

    int bound() const {
        ir_assert(is_const(bound_));
        return to_cpp<int>(bound_);
    }

    int size() const { return size_; }
    const stmt_t &body() const { return stmt.as<for_t>().body; }
    int unroll() const { return stmt.as<for_t>().unroll; }

    stmt_t stmt;
    expr_t var;

private:
    expr_t init_;
    expr_t bound_;
    int size_;
};

// Iterates through multiple nested loops with fixed bounds. Used to unroll
// such nested loops.
class multi_loop_iterator_t {
public:
    // Ordered from innermost to outermost.
    multi_loop_iterator_t(const std::vector<loop_info_t> &loops)
        : loops_(loops) {
        for (auto &l : loops)
            var_values_.push_back(l.init());
    }

    int var_value(const expr_t &var) const {
        for (size_t i = 0; i < loops_.size(); i++) {
            if (loops_[i].var.is_same(var)) return var_values_[i];
        }
        ir_error_not_expected();
        return 0;
    }

    void advance(int n = 1) {
        if (loops_.empty()) return;
        for (int i_n = 0; i_n < n; i_n++) {
            for (size_t i = 0; i < loops_.size(); i++) {
                auto &l = loops_[i];
                if (++var_values_[i] < l.bound()) break;
                var_values_[i] = l.init();
            }
            ir_assert(var_values_.back() < loops_.back().bound());
        }
    }

    bool is_outer_loop_end() const {
        if (loops_.empty()) return true;
        for (size_t i = 0; i < loops_.size() - 1; i++) {
            auto &l = loops_[i];
            if (var_values_[i] != l.bound() - 1) return false;
        }
        return true;
    }

    std::string str() const {
        std::ostringstream oss;
        oss << "multi_loop_iterator_t(";
        for (size_t i = 0; i < loops_.size(); i++) {
            oss << (i != 0 ? ", " : "");
            oss << loops_[i].var << " = " << var_values_[i];
        }
        oss << ")";
        return oss.str();
    }

    IR_DEFINE_DUMP()

private:
    std::vector<loop_info_t> loops_;
    std::vector<int> var_values_;
};

// Extracts different parts of the compute iteration and verifies the loop nest
// is properly formed and can be further injected with SLM buffering.
class compute_step_visitor_t : public ir_visitor_t {
public:
    stmt_t find_stmt_group(const stmt_label_t &label) const {
        auto groups = find_stmt_groups(label);
        if (groups.empty()) return stmt_t();
        ir_assert(groups.size() == 1);
        return groups[0];
    }

    std::vector<stmt_t> find_stmt_groups(const stmt_label_t &label) const {
        std::vector<stmt_t> ret;
        for (auto &_g : stmt_groups_) {
            auto &g = _g.as<stmt_group_t>();
            if (g.label == label) ret.push_back(_g);
        }
        return ret;
    }

    const std::vector<std::pair<stmt_t, std::set<size_t>>> &if_stmts() const {
        return if_stmts_;
    }

    const std::vector<stmt_t> &inner_let_stmts() const {
        return inner_let_stmts_;
    }

#define HANDLE_IR_OBJECT(type) \
    void _visit(const type &obj) override { visit_stmt(obj); }

    HANDLE_STMT_IR_OBJECTS()

#undef HANDLE_IR_OBJECT

    template <typename T>
    void visit_stmt(const T &obj) {
        auto obj_type_id = T::_type_id();
        bool is_for = (obj_type_id == for_t::_type_id());
        bool is_stmt_group = (obj_type_id == stmt_group_t::_type_id());
        bool is_let = (obj_type_id == let_t::_type_id());
        bool is_stmt_seq = (obj_type_id == stmt_seq_t::_type_id());
        bool is_if = (obj_type_id == if_t::_type_id());

        // Loop may contain:
        // - Another loop
        // - Container statement (stmt_seq_t or stmt_group_t)
        // - Let statement (in the innermost loop only)
        // - Barrier
        if (loop_level_ > 0) {
            bool ok = false;
            if (is_for || is_let || is_stmt_group || is_stmt_seq || is_if) {
                ok = true;
            } else if (obj_type_id == func_call_t::_type_id()) {
                auto &call = obj.template as<func_call_t>();
                ok = call.func.is_equal(funcs::barrier_func());
            }
            if (is_if && !obj.template as<if_t>().else_body.is_empty())
                ir_error_not_expected() << "else stmt currently not supported "
                                           "with loop unrolling.\n"
                                        << stmt_t(obj);

            if (!ok) {
                ir_error_not_expected()
                        << "Found unexpected statement inside loop.\n"
                        << stmt_t(obj);
            }
        }

        bool is_compute_loop = false;
        if (is_stmt_group) {
            auto label = obj.template as<stmt_group_t>().label;
            stmt_groups_.push_back(obj);
            if (utils::one_of(label, stmt_label_t::g2s_load(),
                        stmt_label_t::g2s_store(), stmt_label_t::g2r_load(),
                        stmt_label_t::s2r_load(), stmt_label_t::prefetch(),
                        stmt_label_t::mul())) {
                stmt_list.push_back(label);
                // Leaf labels, do not visit them.
                return;
            }
            if (label == stmt_label_t::compute_loop()) {
                is_compute_loop = true;
                in_compute_loop_ = true;
            }
        }

        size_t if_size = 0;
        size_t stmt_size = 0;
        if (is_if) {
            if_size = if_stmts_.size() + 1;
            stmt_size = stmt_list.size();
            if_stmts_.push_back(std::pair<stmt_t, std::set<size_t>>(
                    obj, std::set<size_t>()));
            if_idx_++;
        }
        if (is_for && in_compute_loop_) loop_level_++;
        found_loop_ = false;
        ir_visitor_t::_visit(obj);
        if (is_if) {
            if_idx_--;
            for (size_t i = stmt_size; i < stmt_list.size(); i++) {
                if_stmts_[if_size - 1].second.emplace(stmt_list[i].get_hash());
            }
        }
        if (in_compute_loop_ && is_let) {
            if (found_loop_)
                ir_error_not_expected()
                        << "Let is allowed in the innermost loop only.";

            inner_let_stmts_.push_back(replace_stmt_body(obj, stmt_t()));
        }
        if (is_for && in_compute_loop_) {
            loop_level_--;
            found_loop_ = true;
        }

        if (is_compute_loop) in_compute_loop_ = false;
    }

private:
    bool found_loop_ = false;
    bool in_compute_loop_ = false;
    int loop_level_ = 0;
    int if_idx_ = -1;

    std::vector<stmt_label_t> stmt_list;
    std::vector<stmt_t> stmt_groups_;
    std::vector<stmt_t> inner_let_stmts_;
    std::vector<std::pair<stmt_t, std::set<size_t>>> if_stmts_;
};

// Provides access to different parts of the inner compute iteration.
class compute_step_t {
public:
    compute_step_t(const stmt_t &parent) {
        compute_step_visitor_t v;
        v.visit(parent);

        compute_loop_ = v.find_stmt_group(stmt_label_t::compute_loop());
        g2s_load_ = v.find_stmt_group(stmt_label_t::g2s_load());
        g2s_store_ = v.find_stmt_group(stmt_label_t::g2s_store());
        prefetch_ = v.find_stmt_group(stmt_label_t::prefetch());
        g2r_load_ = v.find_stmt_groups(stmt_label_t::g2r_load());
        s2r_load_ = v.find_stmt_groups(stmt_label_t::s2r_load());
        mul_ = v.find_stmt_groups(stmt_label_t::mul());
        c_zero_out_ = v.find_stmt_group(stmt_label_t::c_zero_out());
        inner_let_stmts_ = v.inner_let_stmts();
        if_stmts_ = v.if_stmts();

        ir_assert(g2r_load_.size() == mul_.size());
        ir_assert(s2r_load_.size() == mul_.size());

        // Assign preload/mul tags to let statements.
        for (auto &_let : inner_let_stmts_) {
            auto &var = _let.as<let_t>().var;
            bool is_preload = (count_object(g2s_load_, var) > 0)
                    || (count_object(prefetch_, var) > 0);
            bool is_mul = count_object(g2r_load_, var) > 0
                    || count_object(mul_, var) > 0;
            if (is_preload) preload_lets_.insert(_let);
            if (is_mul) mul_lets_.insert(_let);
        }

        // Propagate preload/mul tags up based on dependencies between let
        // statements.
        std::vector<let_info_t> let_infos;
        object_set_t<stmt_t> seen;
        std::function<void(const stmt_t &)> propagate;
        propagate = [&](const stmt_t &_let) {
            if (seen.count(_let) > 0) return;
            auto &let = _let.as<let_t>();
            for (auto &_child : inner_let_stmts_) {
                auto &child = _child.as<let_t>();
                if (_child.is_same(_let)) continue;
                if (contains_object(child.value, let.var)) {
                    // Visit child let statements first.
                    propagate(_child);
                    // Propagate child preload/mul values to this let statement.
                    if (is_preload_let(_child)) preload_lets_.insert(_let);
                    if (is_mul_let(_child)) mul_lets_.insert(_let);
                }
            }
            auto let_info = create_let_info(
                    let, is_preload_let(_let), is_mul_let(_let));
            let_infos.push_back(let_info);
            seen.insert(_let);
        };
        for (auto &_let : inner_let_stmts_)
            propagate(_let);

        // Duplicate lets that are used in both preload and mul contexts.
        duplicate_lets(let_infos);
    }

    // See ir_core.hpp for the description.
    const stmt_t &compute_loop() const { return compute_loop_; }
    const stmt_t &g2s_load() const { return g2s_load_; }
    const stmt_t &g2s_store() const { return g2s_store_; }
    const stmt_t &prefetch() const { return prefetch_; }
    const std::vector<stmt_t> &g2r_load() const { return g2r_load_; }
    const std::vector<stmt_t> &s2r_load() const { return s2r_load_; }
    const std::vector<stmt_t> &mul() const { return mul_; }
    const stmt_t &c_zero_out() const { return c_zero_out_; }
    const std::vector<stmt_t> &inner_let_stmts() const {
        return inner_let_stmts_;
    }
    const std::vector<std::pair<stmt_t, std::set<size_t>>> &if_stmts() const {
        return if_stmts_;
    }

    bool is_preload_let(const stmt_t &s) const {
        return preload_lets_.count(s) > 0;
    }
    bool is_mul_let(const stmt_t &s) const { return mul_lets_.count(s) > 0; }

private:
    struct let_info_t {
        let_info_t(const expr_t &var) : var(var) {}

        expr_t var;
        expr_t preload_var;
        expr_t mul_var;

        bool is_preload() const { return !preload_var.is_empty(); }
        bool is_mul() const { return !mul_var.is_empty(); }

        bool needs_update() const { return is_preload() && is_mul(); }
    };

    let_info_t create_let_info(const let_t &let, bool is_preload, bool is_mul) {
        let_info_t info(let.var);
        if (is_preload && !is_mul) {
            info.preload_var = let.var;
        } else if (!is_preload && is_mul) {
            info.mul_var = let.var;
        } else if (is_preload && is_mul) {
            info.preload_var = create_var_with_suffix(let.var, "p");
            info.mul_var = create_var_with_suffix(let.var, "m");
        }
        return info;
    }

    void duplicate_lets(const std::vector<let_info_t> &let_infos) {
        int nlets = int(inner_let_stmts_.size());
        ir_assert(int(let_infos.size()) == nlets);

        std::vector<stmt_t> new_lets;
        for (int i = nlets - 1; i >= 0; i--) {
            auto &info = let_infos[i];
            auto &old_let = inner_let_stmts_[i].as<let_t>();
            if (!info.needs_update()) {
                auto new_value = update_var(old_let.value, let_infos,
                        info.is_preload(), info.is_mul());
                auto new_let = inner_let_stmts_[i];
                if (!new_value.is_same(old_let.value)) {
                    new_let = let_t::make(old_let.var, new_value, old_let.body);
                }
                new_lets.push_back(new_let);
                continue;
            }

            preload_lets_.erase(&old_let);
            mul_lets_.erase(&old_let);

            auto preload_value
                    = update_var(old_let.value, let_infos, true, false);
            auto preload_let = let_t::make(
                    info.preload_var, preload_value, old_let.body);

            auto mul_value = update_var(old_let.value, let_infos, false, true);
            auto mul_let = let_t::make(info.mul_var, mul_value, old_let.body);

            preload_lets_.insert(preload_let);
            new_lets.push_back(preload_let);

            mul_lets_.insert(mul_let);
            new_lets.push_back(mul_let);

            // Update statements.
            g2s_load_ = update_var(g2s_load_, let_infos, true, false);
            g2s_store_ = update_var(g2s_store_, let_infos, true, false);
            prefetch_ = update_var(prefetch_, let_infos, true, false);
            g2r_load_ = update_var(g2r_load_, let_infos, false, true);
            s2r_load_ = update_var(s2r_load_, let_infos, false, true);
            mul_ = update_var(mul_, let_infos, false, true);
        }

        std::reverse(new_lets.begin(), new_lets.end());
        inner_let_stmts_ = new_lets;
    }

    template <typename T>
    static std::vector<T> update_var(const std::vector<T> &vec,
            const std::vector<let_info_t> &let_infos, bool is_preload,
            bool is_mul) {
        std::vector<T> ret;
        for (auto &v : vec)
            ret.push_back(update_var(v, let_infos, is_preload, is_mul));
        return ret;
    }

    static object_t update_var(const object_t &obj,
            const std::vector<let_info_t> &let_infos, bool is_preload,
            bool is_mul) {
        auto ret = obj;
        for (auto &info : let_infos) {
            if (!info.needs_update()) continue;
            if (!contains_object(ret, info.var)) continue;
            if (is_preload) {
                ir_assert(info.is_preload());
                ret = substitute(ret, info.var, info.preload_var);
            } else if (is_mul) {
                ir_assert(info.is_mul());
                ret = substitute(ret, info.var, info.mul_var);
            }
        }
        return ret;
    }

    static expr_t create_var_with_suffix(
            const expr_t &_var, const std::string &suffix) {
        auto &var = _var.as<var_t>();
        auto new_name = var.name + "_" + suffix;
        return var_t::make(var.type, new_name);
    }

    stmt_t compute_loop_;
    stmt_t g2s_load_;
    stmt_t g2s_store_;
    stmt_t prefetch_;
    std::vector<stmt_t> g2r_load_;
    std::vector<stmt_t> s2r_load_;
    std::vector<stmt_t> mul_;
    stmt_t c_zero_out_;

    std::vector<stmt_t> inner_let_stmts_;
    std::vector<std::pair<stmt_t, std::set<size_t>>> if_stmts_;

    // Due to loop unrolling the inner let statements may depend on different
    // indices of the outer loops. There are two contexts:
    // - "preload" loop iteration, e.g. index I
    // - "multiplication" loop iteration, e.g. index (I + nbuf)
    // Preloads (either via SLM or via prefetches) for the corresponding
    // multiplication are executed several iterations before the real
    // multiplication. That's why we need to know exactly in which context the
    // given let statement is used. It might be that the same variable is used
    // from two different contexts. In this case it is duplicated and
    // initialized with different values for each case.
    object_set_t<stmt_t> preload_lets_;
    object_set_t<stmt_t> mul_lets_;
};

// Helper class to access the outer loop index after pipelining. Pipelining
// in general requires tracking two versions of a loop index:
// - Multiplication version - corresponding to the iteration that is currently
//   used for multiplication
// - Preload version - corresponding to the iteration that is currently used
//   for preload for one of the next multiplications
// The multiplication version is a few steps behind the preload version.
class outer_loop_info_t : public loop_info_t {
public:
    outer_loop_info_t() = default;

    outer_loop_info_t(const stmt_t &s, ir_context_t &ir_ctx) : loop_info_t(s) {
        // Outer loop may not be used for unrolling hence loop iterations must
        // not use its index. If this doesn't hold, introduce a GRF buffer to
        // represent that variable and apply post-increment updates after each
        // outer loop iteration.
        if (count_object(s.as<for_t>().body, var) != 0) {
            has_var_refs_ = true;
            mul_var_buf_ = ir_ctx.create_tmp_var(
                    type_t::byte_ptr(), var.as<var_t>().name + "_mul_buf");
            preload_var_buf_ = ir_ctx.create_tmp_var(
                    type_t::byte_ptr(), var.as<var_t>().name + "_preload_buf");

            auto mul_alloc = alloc_t::make(
                    mul_var_buf_, var.type().size(), alloc_kind_t::grf);
            auto preload_alloc = alloc_t::make(
                    preload_var_buf_, var.type().size(), alloc_kind_t::grf);
            allocs_.push_back(mul_alloc);
            allocs_.push_back(preload_alloc);

            auto mul_init = store_t::make(mul_var_buf_, 0, init());
            auto preload_init = store_t::make(preload_var_buf_, 0, init());
            init_stmt_ = mul_init.append(preload_init);

            mul_post_inc_stmt_
                    = store_t::make(mul_var_buf_, 0, mul_var_load() + 1);
            preload_post_inc_stmt_ = store_t::make(
                    preload_var_buf_, 0, preload_var_load() + 1);
        }
    }

    bool has_var_refs() const { return has_var_refs_; }

    expr_t mul_var_load() const {
        return load_t::make(var.type(), mul_var_buf_, 0);
    }
    expr_t preload_var_load() const {
        return load_t::make(var.type(), preload_var_buf_, 0);
    }

    stmt_t inject_alloc_stmts(const stmt_t &stmt) const {
        return jit::inject_alloc_stmts(stmt, allocs_);
    }

    const stmt_t &init_stmt() const { return init_stmt_; }

    const stmt_t &mul_post_inc_stmt() const { return mul_post_inc_stmt_; }
    const stmt_t &preload_post_inc_stmt() const {
        return preload_post_inc_stmt_;
    }

private:
    bool has_var_refs_ = false;

    // Helper expressions/statements to partially unroll the loop.
    expr_t mul_var_buf_;
    expr_t preload_var_buf_;
    std::vector<stmt_t> allocs_;
    stmt_t init_stmt_;
    stmt_t mul_post_inc_stmt_;
    stmt_t preload_post_inc_stmt_;
};

class compute_loop_nest_visitor_t : public ir_visitor_t {
public:
    int compute_loop_level() const { return compute_loop_level_; }

    const std::vector<loop_info_t> &loops() const { return loops_; }

    void _visit(const stmt_group_t &obj) override {
        bool is_compute_loop = (obj.label == stmt_label_t::compute_loop());
        if (is_compute_loop) {
            in_compute_loop_ = true;
            compute_loop_level_ = level_;
        }
        ir_visitor_t::_visit(obj);
        if (is_compute_loop) in_compute_loop_ = false;
    }

    void _visit(const for_t &obj) override {
        level_++;
        ir_visitor_t::_visit(obj);
        if (in_compute_loop_) loops_.emplace_back(obj);
        level_--;
    }

private:
    bool in_compute_loop_ = false;
    int compute_loop_level_ = -1;
    std::vector<loop_info_t> loops_;
    int level_ = 0;
};

// Helper class to work with loop nest of the compute loop.
class compute_loop_nest_t {
public:
    compute_loop_nest_t() = default;

    compute_loop_nest_t(const stmt_t &root, ir_context_t &ir_ctx)
        : root_(root) {
        compute_loop_nest_visitor_t visitor;
        visitor.visit(root);

        compute_loop_level_ = visitor.compute_loop_level();
        loops_ = visitor.loops();

        if (loops_.empty()) {
            outer_loop_size_ = 1;
            return;
        }

        outer_loop_ = outer_loop_info_t(loops_.back().stmt, ir_ctx);
        outer_loop_size_ = outer_loop_.size();
    }

    // Returns the loop level of the compute_loop statement group corresponding
    // to the number of outer loops.
    int compute_loop_level() const { return compute_loop_level_; }

    // Returns loops inside compute_loop statement group.
    const std::vector<loop_info_t> &loops() const { return loops_; }

    // Number of iterations of all loops.
    int size() const {
        int ret = 1;
        for (auto &l : loops_)
            ret *= l.size();
        return ret;
    }

    // Number of iterations in the outermost loop (see comments in ctor).
    int outer_loop_size() const { return outer_loop_size_; }

    const outer_loop_info_t &outer_loop_info() const { return outer_loop_; }

    template <typename F>
    void for_each_loop_var(const F &f) const {
        for (auto &l : loops_)
            f(l.var);
    }

    // Number of iterations of all loops except the outermost.
    int inner_loops_size() const { return size() / outer_loop_size(); }

private:
    stmt_t root_;
    int compute_loop_level_ = -1;
    std::vector<loop_info_t> loops_;

    int outer_loop_size_;
    outer_loop_info_t outer_loop_;
};

struct compute_params_t {
    compute_params_t() = default;

    compute_params_t(int slm_bufs, int gmem_bufs, int slm_buf_size,
            int prefetch_bufs, int inner_loops_iters)
        : slm_bufs(slm_bufs)
        , gmem_bufs(gmem_bufs)
        , slm_buf_size(slm_buf_size)
        , prefetch_bufs(prefetch_bufs) {
        use_slm = (slm_buf_size > 0);
        use_prefetch = (prefetch_bufs > 0);
        ir_assert(!use_slm || !use_prefetch)
                << "Can't have both SLM buffering and prefetch enabled.";
        if (use_slm) {
            ir_assert(utils::one_of(slm_bufs, 1, 2, 3));
            ir_assert(utils::one_of(gmem_bufs, 1, 2));
            preload_bufs = slm_bufs;
            unroll = math::lcm(slm_bufs * gmem_bufs, inner_loops_iters);
        } else if (use_prefetch) {
            preload_bufs = prefetch_bufs;
            ir_assert(slm_bufs == 0);
            ir_assert(gmem_bufs == 0);
            unroll = math::lcm(prefetch_bufs, inner_loops_iters);
        } else {
            preload_bufs = 0;
            ir_assert(slm_bufs == 0);
            ir_assert(gmem_bufs == 0);
            unroll = inner_loops_iters;
        }
    }

    int slm_bufs;
    int gmem_bufs;
    int slm_buf_size;
    int prefetch_bufs;
    int preload_bufs;
    int unroll;

    bool use_slm;
    bool use_prefetch;
};

// Helper class to implement SLM buffering.
class compute_iterator_t {
public:
    compute_iterator_t(const compute_params_t &params,
            const compute_loop_nest_t &loop_nest)
        : params(params)
        , preload_loop_it(loop_nest.loops())
        , mul_loop_it(loop_nest.loops()) {

        int compute_iters = loop_nest.size();
        iters = compute_iters;
        ir_assert(iters >= 1) << "Empty loop is not expected.";

        iters += std::max(0, preload_bufs() - 1) + std::max(0, gmem_bufs() - 1);
        ramp_up_iters
                = std::max(1, preload_bufs() + std::max(0, gmem_bufs() - 1));
        ramp_down_iters = std::min(
                std::max(0, preload_bufs() - 1) + std::max(0, gmem_bufs() - 1),
                iters - ramp_up_iters);
        body_iters = iters - ramp_up_iters - ramp_down_iters;
        body_iters = utils::rnd_dn(body_iters, params.unroll);
        ramp_down_iters = iters - ramp_up_iters - body_iters;

        ir_assert(ramp_up_iters + body_iters + ramp_down_iters == iters);

        iter = 0;
        linear_id = 0;
        riter = iters - 1;
    }

    int unroll() const { return params.unroll; }

    int preload_bufs() const { return params.preload_bufs; }

    int slm_bufs() const { return params.slm_bufs; }

    int gmem_bufs() const { return params.gmem_bufs; }

    compute_iterator_t &operator++() {
        if (do_preload()) preload_loop_it.advance();
        if (do_mul()) mul_loop_it.advance();
        ++iter;
        ++linear_id;
        --riter;
        return *this;
    }

    void advance(int n) {
        if (n == 0) return;

        ir_assert(n % params.unroll == 0);
        ir_assert(iter + n <= iters);

        if (preload_bufs() > 0) ir_assert(do_preload());
        ir_assert(do_mul());

        iter += n;
        riter -= n;

        if (preload_bufs() > 0) preload_loop_it.advance(n);
        mul_loop_it.advance(n);
    }

    bool do_mul() const {
        return iter >= std::max(0, preload_bufs() - 1)
                + std::max(0, gmem_bufs() - 1);
    }

    bool is_first_mul() const {
        return iter
                == std::max(0, preload_bufs() - 1)
                + std::max(0, gmem_bufs() - 1);
    }
    bool is_last_mul() const { return riter == 0; }

    bool is_last_g2s_store() const {
        if (!do_g2s_store()) return false;
        return riter == slm_bufs() - 1;
    }

    bool is_last_preload() const {
        if (!do_preload()) return false;
        return riter == (preload_bufs() - 1) + std::max(0, gmem_bufs() - 1);
    }

    bool is_last_g2s_load() const {
        if (!do_g2s_load()) return false;
        return is_last_preload();
    }

    bool is_last_prefetch() const {
        if (!do_prefetch()) return false;
        return is_last_preload();
    }

    bool do_preload() const {
        if (preload_bufs() == 0) return false;
        return riter >= (preload_bufs() - 1) + std::max(0, gmem_bufs() - 1);
    }

    bool do_prefetch() const {
        if (!params.use_prefetch) return false;
        return do_preload();
    }

    bool do_g2s_load() const {
        if (!params.use_slm) return false;
        return do_preload();
    }

    bool do_g2s_store() const {
        if (!params.use_slm) return false;
        ir_assert(gmem_bufs() >= 1);
        return iter >= (gmem_bufs() - 1) && riter >= (slm_bufs() - 1);
    }

    int gmem_write_buf_index() const {
        ir_assert(do_g2s_load());
        return iter % gmem_bufs();
    }

    int gmem_read_buf_index() const {
        ir_assert(do_g2s_store());
        return (iter - (gmem_bufs() - 1)) % gmem_bufs();
    }

    int slm_read_offset_update() const {
        ir_assert(params.use_slm);
        ir_assert(do_mul());

        int slm_iter = iter - (gmem_bufs() - 1) - (slm_bufs() - 1);
        int cur_slm_idx = slm_iter % slm_bufs();
        int next_slm_idx = (slm_iter + 1) % slm_bufs();
        int ret = next_slm_idx * params.slm_buf_size
                - cur_slm_idx * params.slm_buf_size;
        return ret;
    }

    int slm_write_offset_update() const {
        ir_assert(params.use_slm);
        ir_assert(do_g2s_store());

        int slm_iter = iter - (gmem_bufs() - 1);
        int cur_slm_idx = slm_iter % slm_bufs();
        int next_slm_idx = (slm_iter + 1) % slm_bufs();
        int ret = next_slm_idx * params.slm_buf_size
                - cur_slm_idx * params.slm_buf_size;
        return ret;
    }

    compute_params_t params;
    multi_loop_iterator_t preload_loop_it;
    multi_loop_iterator_t mul_loop_it;

    // ramp_up_iters + body_iters + ramp_down_iters == iters
    int iters;
    int ramp_up_iters;
    int body_iters;
    int ramp_down_iters;

    // Invariant: iter + riter = iters - 1
    int iter;
    int riter;

    int linear_id;
};

// Basic LRU SBID allocator, tries to use the same SBIDs for the same GRF
// buffers.
class sbid_manager_t {
public:
    sbid_manager_t(ngen::HW hw = ngen::HW::Unknown)
        : sbid_count_(hw >= ngen::HW::XeHPC ? 32 : 16)
        , tuple_func_(builtin_t::make("tuple")) {}

    ngen_proxy::SBID get_sbid(const expr_t &buf, int index = 0) {
        auto key = tuple_func_.call({buf, expr_t(index)});

        int free_idx = -1;
        for (int i = 0; i < sbid_count_; i++) {
            auto &e = entries_[i];
            if (key.is_equal(e.key)) {
                e.time = cur_time_++;
                return ngen_proxy::SBID(i);
            }
            if (free_idx == -1 && e.key.is_empty()) free_idx = i;
        }

        // Not found but there is a free SBID.
        if (free_idx != -1) {
            entries_[free_idx] = {key, cur_time_++};
            return ngen_proxy::SBID(free_idx);
        }

        // Find the oldest SBID and use it.
        int old_idx = 0;
        int old_time = entries_[0].time;
        for (int i = 1; i < sbid_count_; i++) {
            if (entries_[i].time < old_time) {
                old_idx = i;
                old_time = entries_[i].time;
            }
        }

        entries_[old_idx] = entry_t({key, cur_time_++});
        return ngen_proxy::SBID(old_idx);
    }

private:
    struct entry_t {
        stmt_t key;
        int time;
    };

    static const int max_sbid_count = 32;
    std::array<entry_t, max_sbid_count> entries_;

    int sbid_count_ = 0;
    func_t tuple_func_;
    int cur_time_ = 0;
};

// Helper to assign SBIDs to IR function calls.
class sbid_assigner_t {
public:
    sbid_assigner_t(ngen::HW hw) : local_sbid_mgr_(hw) {}

    sbid_assigner_t(sbid_manager_t &external_sbid_mgr)
        : external_sbid_mgr_(&external_sbid_mgr) {}

    stmt_t assign(const stmt_t &stmt) {
        auto stmt_vec = flatten_statements(stmt);
        stmt_t ret = stmt;
        int prefetch_idx = 0;
        for (auto &_s : stmt_vec) {
            if (!_s.is<func_call_t>()) continue;
            auto s = _s;
            if (is_func_call<send_t>(s)) {
                auto &send = s.as<func_call_t>().func.as<send_t>();
                int idx = (send.is_prefetch() || send.is_prefetch_2d()
                                ? prefetch_idx++
                                : 0);
                auto sbid = get_sbid(send_t::arg_reg_buf(s), idx);
                s = update_call_with_sbid(s, sbid);
            } else if (is_func_call<dpas_t>(s)) {
                auto &c = s.as<func_call_t>();
                auto *mod_attr = c.attr.as_ptr<instruction_modifier_attr_t>();
                if (!c.func.as<dpas_t>().is_dp4a() && // dp4a-s do not need SBID
                        (!mod_attr || !mod_attr->mod.is_atomic)) {
                    // Last dpas in Atomic chain.
                    auto sbid = get_sbid(dpas_t::arg_src1(s));
                    s = update_call_with_sbid(s, sbid);
                }
            } else if (s.is<func_call_t>()) {
                auto &c = s.as<func_call_t>();
                if (c.func.is_equal(funcs::signal_func())
                        || c.func.is_equal(funcs::slm_fence_func())
                        || c.func.is_equal(funcs::barrier_func())) {
                    // Use 0 as the key for signals and SLM fences.
                    auto sbid = get_sbid(expr_t(0));
                    s = update_call_with_sbid(s, sbid);
                }
            } else {
                ir_error_not_expected() << s;
            }
            ret = substitute(ret, _s, s);
        }
        return ret;
    }

private:
    ngen_proxy::SBID get_sbid(const expr_t &ptr, int index = 0) {
        auto &sbid_mgr
                = (external_sbid_mgr_ ? *external_sbid_mgr_ : local_sbid_mgr_);
        return sbid_mgr.get_sbid(ptr, index);
    }

    static stmt_t update_call_with_sbid(
            const stmt_t &s, const ngen_proxy::SBID &sbid) {
        return instruction_modifier_attr_t::make(
                ngen_proxy::InstructionModifier().with_sbid(sbid))
                .apply_to(s);
    }

    sbid_manager_t local_sbid_mgr_;
    sbid_manager_t *external_sbid_mgr_ = nullptr;
};

// Work around due to limited scoping functionality in current generator
// Prepends all newly created var_t names with given prefix.
class var_prepender_t : public ir_mutator_t {
public:
    var_prepender_t(const std::string &prefix) : prefix_(prefix) {}
    object_t _mutate(const for_t &obj) override {
        auto new_obj = ir_mutator_t::_mutate(obj);
        auto new_var = var_t::make(
                obj.var.type(), prefix_ + obj.var.as<var_t>().name);
        new_obj = substitute(new_obj, obj.var, new_var);
        return new_obj;
    }
    object_t _mutate(const let_t &obj) override {
        auto new_obj = ir_mutator_t::_mutate(obj);
        auto new_var = var_t::make(
                obj.var.type(), prefix_ + obj.var.as<var_t>().name);
        new_obj = substitute(new_obj, obj.var, new_var);
        return new_obj;
    }

private:
    std::string prefix_;
};

object_t prepend_new_vars(const object_t &root, const std::string &prefix) {
    var_prepender_t mutator(prefix);
    return mutator.mutate(root);
}

// Perform pipelining operation. The goal is to transform
// the loop structure from:
//
// for i in range(init, bound):
//     A_block(i);
//     B_block(i);
//
// to the following
//
// for i in range(init, init + length):
//     A_block(i);
// for i in range(init, bound):
//     if (i < bound - length):
//         A_block(i + length);
//      B_block(i);
//
// Since A_block and B_block have to be independent to maintain correctness,
// this transform ignores the operations within the for_loop and relies on a
// correct substitution for A_block and B_block.

struct pipeline_ctx_t {
    pipeline_ctx_t(const stmt_t &prologue, const stmt_t &body)
        : prologue_(prologue), body_(body) {}
    stmt_t stmt() const { return prologue_.append(body_); }
    stmt_t prologue() { return prologue_; }
    stmt_t body() { return body_; }

private:
    stmt_t prologue_;
    stmt_t body_;
};

pipeline_ctx_t pipeline(
        int length, const loop_info_t &loop, stmt_t A_block, stmt_t B_block) {

    expr_t idx = loop.var;
    int bound = loop.bound();
    int init = loop.init();

    int pipe_len = std::min(init + length, bound);

    stmt_t prologue = prepend_new_vars(
            for_t::make(idx, init, pipe_len, A_block,
                    pipe_len <= loop.unroll() ? pipe_len : 1),
            "prefetch_");

    expr_t A_idx = idx + pipe_len;
    stmt_t body = if_t::make(
            idx < (bound - pipe_len), substitute(A_block, idx, A_idx));
    body = body.append(B_block);
    body = for_t::make(idx, init, bound, body, loop.unroll());

    return pipeline_ctx_t(prologue, body);
}

class prefetch_pipeliner_t {
public:
    prefetch_pipeliner_t(
            const stmt_t &root, const conv_config_t &cfg, ir_context_t &ir_ctx)
        : root_(root), cfg_(cfg), ir_ctx_(ir_ctx) {}
    stmt_t inject() {
        auto compute_loop_stmt
                = find_stmt_group(root_, stmt_label_t::compute_loop());
        if (!compute_loop_stmt.has_value()) return root_;
        auto compute_loop = compute_loop_stmt.value();
        auto loop_nest = compute_loop_nest_t(compute_loop, ir_ctx_);
        auto &loops = loop_nest.loops();

        // No loops to pipeline
        if (loops.size() == 0) return root_;
        auto &loop_body = loops[0].body();

        auto A_block_stmt
                = find_stmt_group(loop_body, stmt_label_t::prefetch());
        if (!A_block_stmt.has_value()) return root_;
        auto A_block = A_block_stmt.value();
        auto B_block = remove_stmt_group(loop_body, stmt_label_t::prefetch());
        size_t prefetch_count = 0;
        size_t max_nested_prefetch = 2;
        for (size_t i = 0; i < loops.size(); i++) {
            if (prefetch_count < max_nested_prefetch) {
                if (!contains_object(A_block, loops[i].var)) {
                    // No point in prefetching a constant in a loop
                    B_block = for_t::make(loops[i].var, loops[i].init(),
                            loops[i].bound(), B_block, loops[i].unroll());
                    continue;
                }

                auto next = pipeline(
                        cfg_.prefetch_bufs, loops[i], A_block, B_block);
                A_block = next.prologue();
                B_block = next.body();
                prefetch_count++;

            } else {
                B_block = for_t::make(loops[i].var, loops[i].init(),
                        loops[i].bound(), A_block.append(B_block),
                        loops[i].unroll());
                A_block = stmt_t();
            }
        }
        return substitute(root_, compute_loop, A_block.append(B_block));
    }

private:
    stmt_t root_;
    const conv_config_t &cfg_;
    ir_context_t &ir_ctx_;
};

stmt_t inject_prefetch_pipeline(
        const stmt_t &s, const conv_config_t &cfg, ir_context_t &ir_ctx) {
    auto ret = prefetch_pipeliner_t(s, cfg, ir_ctx).inject();
    trace_pass("inject_prefetch_pipeline", ret);
    return ret;
}

stmt_t create_reorder_stmt(const layout_t &src, const layout_t &dst,
        const expr_t &src_buf, const expr_t &dst_buf) {
    ir_assert(src.ndims() == dst.ndims()) << "Layouts are incompatible.";
    ir_assert(src.elems() == dst.elems()) << "Layouts are incompatible.";
    auto func = reorder_t::make(src, dst);
    return func.call({dst_buf, src_buf});
}

class simple_slm_buffering_injector_t {
public:
    simple_slm_buffering_injector_t(ngen::HW hw, const stmt_t &root,
            const conv_config_t &cfg, ir_context_t &ir_ctx, int ab_slm_size)
        : hw_(hw)
        , cfg_(cfg)
        , ir_ctx_(ir_ctx)
        , ab_slm_size_(ab_slm_size)
        , root_(root)
        , alloc_mgr_(root_)
        , step_(root)
        , loop_nest_(root, ir_ctx) {}

    stmt_t inject() {
        ir_assert(cfg_.gmem_bufs == 1) << "GRF buffering is not supported.";
        if (utils::one_of(cfg_.slm_bufs, 0, 1)) return root_;

        ir_assert(cfg_.use_a_slm == cfg_.use_b_slm)
                << "Mixed SLM/GMEM loads are not supported.";

        auto loop = step_.compute_loop();

        // SLM indices are allocated as follows:
        // slm_idx[0] -> slm_buf_store
        // slm_idx[1] -> slm_buf_compute
        // slm_idx[2] -> slm_counter
        auto slm_idx_buf
                = ir_ctx_.create_tmp_var(type_t::byte_ptr(), "slm_idx");
        int slm_idx_size = type_t::s32().size();

        auto slm_idx_load = [&](int off, int elems) {
            return load_t::make(
                    type_t::s32(elems), slm_idx_buf, slm_idx_size * off);
        };

        // Initialize slm_idx.
        int off = 0;
        auto store0 = store_t::make(slm_idx_buf, off, 0);
        off += slm_idx_size;

        auto store1 = store_t::make(slm_idx_buf, off, 1);
        off += slm_idx_size;

        auto store2 = store_t::make(
                slm_idx_buf, off, int_imm_t::make(0, type_t::s32()));

        auto slm_idx_init = store0.append(store1).append(store2);

        auto slm_idx_load2 = slm_idx_load(0, 2);
        auto slm_idx_load4 = slm_idx_load(0, 4);
        auto slm_idx_store = store_t::make(slm_idx_buf, 0,
                slm_idx_load4 + shuffle_t::make_broadcast(1, 4));

        // Update slm_idx.
        auto mask = (slm_idx_load2
                == shuffle_t::make_broadcast(cfg_.slm_bufs, 2));
        auto slm_idx_store_fix = store_t::make(slm_idx_buf, 0,
                shuffle_t::make_broadcast(int_imm_t::make(0, type_t::s32()), 2),
                store_t::default_stride, mask);

        auto slm_idx_update = slm_idx_store.append(slm_idx_store_fix);

        loop = slm_idx_init.append(loop);

        auto &g2s_store_orig = step_.g2s_store();
        auto &s2r_load = step_.s2r_load();
        auto &mul = step_.mul();

        auto g2s_store = g2s_store_orig;

        ir_assert(s2r_load.size() == mul.size());

        stmt_t s2r_mul;
        for (int i = 0; i < int(mul.size()); i++) {
            s2r_mul = s2r_mul.append(s2r_load[i]);
            loop = substitute(loop, s2r_load[i], stmt_t(), 1);
            s2r_mul = s2r_mul.append(mul[i]);
            loop = substitute(loop, mul[i], stmt_t(), 1);
        }

        loop = remove_synchronization(loop);

        object_eq_set_t<expr_t> mask_bufs;
        std::vector<stmt_t> masks;

        slm_zp_mask_extractor_t(masks, mask_bufs).visit(s2r_mul);
        if (!mask_bufs.empty())
            for (auto &m : masks)
                s2r_mul = substitute(s2r_mul, m, stmt_t());

        s2r_mul = sub_slm_bufs(s2r_mul, slm_idx_load(1, 1));
        g2s_store = sub_slm_bufs(g2s_store, slm_idx_load(0, 1));
        g2s_store = g2s_store.append(slm_idx_update);

        auto s2r_mul_body = s2r_mul;
        auto s2r_mul_tail = s2r_mul;
        auto slm_counter = slm_idx_load(2, 1);
        auto cond = (slm_counter >= cfg_.slm_bufs - 1);

        if (cfg_.slm_bufs == 2) {
            s2r_mul_body = if_t::make(cond, s2r_mul_body);
            g2s_store = g2s_store.append(funcs::barrier());
        } else {
            // In general we have to use SLM fence before signal to flush all
            // previous SLM stores. However any SLM load behaves as implicit
            // SLM fence for all previous SLM stores. This means we don't need
            // explicit SLM fence when we perform SLM load/multiplication
            // before signal.
            auto fence_signal = funcs::slm_fence().append(funcs::signal());
            s2r_mul_body = s2r_mul_body.append(funcs::signal());
            s2r_mul_body = if_t::make(cond, s2r_mul_body, fence_signal);
            s2r_mul_body = funcs::barrier_wait().append(s2r_mul_body);
        }

        alloc_updater_t alloc_updater;

        for (auto &mbuf : mask_bufs) {
            auto sz = alloc_mgr_.alloc_size(mbuf);
            alloc_updater.resize(mbuf, sz * cfg_.slm_bufs);
            for (auto &m : masks)
                m = substitute(m, mbuf, mbuf[sz * (cfg_.slm_bufs - 1)]);
            layout_t comp_layout(type_t::u8(), 0, std::vector<dim_t> {sz});
            for (int b = 1; b < cfg_.slm_bufs; b++) {
                auto reorder = create_reorder_stmt(comp_layout, comp_layout,
                        mbuf + b * sz, mbuf + (b - 1) * sz);
                s2r_mul_body = s2r_mul_body.append(reorder);
                if ((cfg_.slm_bufs == 3) && (b == 1))
                    s2r_mul_tail = s2r_mul_tail.append(reorder);
            }
        }
        if (!mask_bufs.empty()) {
            stmt_t all_masks;
            for (auto &m : masks)
                all_masks = all_masks.append(m);
            s2r_mul_body = all_masks.append(s2r_mul_body);
        }
        loop = substitute(
                loop, g2s_store_orig, s2r_mul_body.append(g2s_store), 1);

        if (cfg_.slm_bufs == 3) {
            // Emit initial signal, to match wait-signal pairs in the loop.
            loop = funcs::signal().append(loop);
        }

        // Complete the remaining iterations.
        int rem_iters = cfg_.slm_bufs - 1;
        int mul_start = std::max(0, rem_iters - loop_nest_.size());
        multi_loop_iterator_t multi(loop_nest_.loops());
        multi.advance(loop_nest_.size() - rem_iters + mul_start);
        for (int i = 0; i < rem_iters; i++) {
            if (cfg_.slm_bufs == 3) loop = loop.append(funcs::barrier_wait());
            if (i >= mul_start) {
                auto tmp_mul_tail = s2r_mul_tail;
                loop_nest_.for_each_loop_var([&](const expr_t &v) {
                    expr_t iter(multi.var_value(v));
                    tmp_mul_tail = substitute(tmp_mul_tail, v, iter);
                });
                // SLM load/multiplication works as implicit SLM fence.
                loop = loop.append(tmp_mul_tail);
                multi.advance();
            } else {
                loop = loop.append(funcs::slm_fence());
            }
            loop = loop.append(slm_idx_update);
            if (cfg_.slm_bufs == 3 && i + 1 < rem_iters)
                loop = loop.append(funcs::signal());
        }

        if (cfg_.assign_sbids) loop = sbid_assigner_t(hw_).assign(loop);

        const auto grf_size = ngen::GRF::bytes(hw_);
        loop = alloc_t::make(slm_idx_buf, grf_size, alloc_kind_t::grf, loop);

        auto slm_buffers = alloc_mgr_.find_buffers(alloc_kind_t::slm);
        ir_assert(slm_buffers.size() == 1);
        auto &slm_buf = slm_buffers[0];
        int non_ab_slm_size = alloc_mgr_.alloc_size(slm_buf) - ab_slm_size_;
        alloc_updater.resize(
                slm_buf, non_ab_slm_size + ab_slm_size_ * cfg_.slm_bufs);

        auto ret = substitute(root_, step_.compute_loop(), loop, 1);
        ret = alloc_updater.update(ret);
        return ret;
    }

    static stmt_t remove_synchronization(const stmt_t &s) {
        auto ret = s;
        for (auto &_c : find_objects<func_call_t>(s)) {
            auto &c = _c.as<func_call_t>();
            if (c.func.is_equal(funcs::signal_func())
                    || c.func.is_equal(funcs::slm_fence_func())
                    || c.func.is_equal(funcs::barrier_func())) {
                ret = substitute(ret, _c, stmt_t(), 1);
            }
        }
        return ret;
    }

    stmt_t sub_slm_bufs(const stmt_t &stmt, const expr_t &slm_idx) const {
        auto stmt_vec = flatten_statements(stmt);

        stmt_t ret = stmt;
        for (auto &s : stmt_vec) {
            if (!is_func_call<send_t>(s)) continue;

            auto &send = s.as<func_call_t>().func.as<send_t>();

            // This is not send to SLM, skip.
            if (!send.is_slm()) continue;

            auto new_args = s.as<func_call_t>().args;
            send_t::arg_mem_off(new_args) += ab_slm_size_ * slm_idx;
            auto new_send = send.call(new_args);
            ret = substitute(ret, s, new_send, 1);
        }

        return ret;
    }

    ngen::HW hw_;
    const conv_config_t &cfg_;
    ir_context_t &ir_ctx_;
    int ab_slm_size_;

    stmt_t root_;
    alloc_manager_t alloc_mgr_;
    compute_step_t step_;
    compute_loop_nest_t loop_nest_;
};

// Injects SLM buffering without unrolling based on the config.
stmt_t inject_simple_slm_buffering(ngen::HW hw, const stmt_t &s,
        const conv_config_t &cfg, ir_context_t &ir_ctx, int ab_slm_size) {
    auto ret = simple_slm_buffering_injector_t(hw, s, cfg, ir_ctx, ab_slm_size)
                       .inject();
    trace_pass("inject_simple_slm_buffering", ret);
    return ret;
}
class if_unroll_injector_t : public ir_visitor_t {
public:
    if_unroll_injector_t(
            const std::vector<std::pair<stmt_t, std::set<size_t>>> &if_stmts)
        : if_stmt_(stmt_t())
        , stmt_(stmt_t())
        , idx_(-1)
        , prev_matches_(0)
        , if_stmts_(if_stmts) {}

    void _visit(const stmt_group_t &obj) override {
        stmt_label_t label = obj.as<stmt_group_t>().label;
        int tmp_idx = -1;
        int matches = 0;
        stmt_t if_stmt = stmt_t();
        stmt_t body = stmt_t();
        if (label == stmt_label_t::g2s_store()) {
            body = body.append(funcs::barrier());
            body = body.append(obj);
            body = body.append(funcs::barrier());
        } else {
            body = body.append(obj);
        }
        bool pop_stack = true;
        for (int i = 0; i < (int)if_stmts_.size(); i++) {
            if (if_stmts_[i].second.count(label.get_hash()) > 0) {
                tmp_idx = i;
                matches++;
            } else {
                continue;
            }
            if (tmp_idx != idx_) {
                auto tmp_if = if_t::make(
                        if_stmts_[tmp_idx].first.as<if_t>().cond, stmt_t());
                bool insert_if = true;
                for (auto &s : if_stack)
                    if (tmp_if.as<if_t>().cond.is_equal(s.as<if_t>().cond))
                        insert_if = false;
                if (matches > 0) {
                    if (insert_if) {
                        for (int i = matches; i < prev_matches_ + 1; i++)
                            if_stack.pop_back();
                        if (if_stack.size() > 0) {
                            auto &top_if = if_stack.back();
                            if_stmt = top_if;
                            auto &ifs = if_stmt.as<if_t>();
                            ifs.body = ifs.body.append(tmp_if);

                        } else {
                            stmt_ = stmt_.append(tmp_if);
                        }
                        pop_stack = false;
                        if_stack.push_back(tmp_if);
                    }
                }
            }
        }
        if (pop_stack)
            for (int i = matches; i < prev_matches_; i++)
                if_stack.pop_back();
        if (tmp_idx > -1) {
            auto &ifs = if_stack.back().as<if_t>();
            ifs.body = ifs.body.append(body);
        }
        idx_ = tmp_idx;
        prev_matches_ = matches;
        if (matches == 0) stmt_ = stmt_.append(body);
    }

    stmt_t stmt() { return stmt_; }

private:
    stmt_t if_stmt_;
    stmt_t stmt_;
    int idx_;
    int prev_matches_;
    std::vector<stmt_t> if_stack;
    const std::vector<std::pair<stmt_t, std::set<size_t>>> &if_stmts_;
};

class unrolling_injector_t {
public:
    unrolling_injector_t(const stmt_t &root, const conv_config_t &cfg,
            ir_context_t &ir_ctx, int ab_slm_size)
        : cfg_(cfg)
        , ir_ctx_(ir_ctx)
        , ab_slm_size_(ab_slm_size)
        , root_(root)
        , alloc_mgr_(root_)
        , step_(root)
        , loop_nest_(root, ir_ctx) {
        int inner_iters = loop_nest_.inner_loops_size();
        params_ = compute_params_t(cfg_.slm_bufs, cfg_.gmem_bufs, ab_slm_size,
                cfg_.prefetch_bufs, inner_iters);
        if (params_.use_slm) {
            for (auto &b :
                    find_send_buffers(step_.g2s_load(), /*is_mem=*/false)) {
                g2s_reg_bufs_.emplace_back(b, alloc_mgr_.alloc_size(b));
            }
        }

        // Can't fuse top-level zero-out statement unless the compute loop is
        // top-level as well.
        fuse_zero_out_with_fma_ = (loop_nest_.compute_loop_level() == 0);
    }

    stmt_t inject() {
        compute_iterator_t it(params_, loop_nest_);
        stmt_t body;

        sbid_manager_t sbid_mgr(cfg_.hw());

        auto &outer_loop_info = loop_nest_.outer_loop_info();

        auto append_outer_post_inc = [&](const stmt_t &_s) {
            auto &mul = outer_loop_info.mul_post_inc_stmt();
            auto &preload = outer_loop_info.preload_post_inc_stmt();
            auto s = _s;
            if (it.mul_loop_it.is_outer_loop_end() && it.do_mul()) {
                s = s.append(mul);
            }
            if (it.preload_loop_it.is_outer_loop_end() && it.do_preload()) {
                s = s.append(preload);
            }
            return s;
        };

        // Ramp-up.
        for (int i = 0; i < it.ramp_up_iters; i++) {
            body = stmt_seq_t::make(body, create_iteration(it, sbid_mgr));
            body = append_outer_post_inc(body);
            ++it;
        }

        // Body.
        if (it.body_iters > 0) {
            int extent = it.body_iters / it.unroll();
            bool has_loop = (extent > 1);

            stmt_t loop_body;
            for (int i = 0; i < it.unroll(); i++) {
                loop_body = loop_body.append(create_iteration(
                        it, sbid_mgr, /*in_loop_body=*/has_loop));
                ir_assert(it.do_mul());
                loop_body = append_outer_post_inc(loop_body);
                ++it;
            }
            if (!has_loop) {
                body = body.append(loop_body);
            } else {
                ir_assert(extent > 0);
                auto for_var = ir_ctx_.create_tmp_var(type_t::s32(), "i");
                body = body.append(for_t::make(for_var, 0, extent, loop_body));
            }
            it.advance(it.body_iters - it.unroll());
        }

        // Ramp-down.
        for (int i = 0; i < it.ramp_down_iters; i++) {
            ir_assert(it.do_mul());
            body = body.append(create_iteration(it, sbid_mgr));
            body = append_outer_post_inc(body);
            ++it;
        }

        if (outer_loop_info.has_var_refs()) {
            body = outer_loop_info.init_stmt().append(body);
            body = outer_loop_info.inject_alloc_stmts(body);
        }

        // When compute loop is part of outer loop and SLM buffering is used
        // then synchronization is required between outer iterations.
        if (loop_nest_.compute_loop_level() != 0 && params_.use_slm) {
            body = funcs::barrier().append(body);
        }

        body = stmt_group_t::make(stmt_label_t::compute_loop(), body);
        auto ret = substitute(root_, step_.compute_loop(), body, 1);

        if (params_.use_slm) {
            alloc_updater_t alloc_updater;

            // Update buffer sizes.
            for (auto &b : g2s_reg_bufs_) {
                alloc_updater.resize(
                        b.buf, alloc_mgr_.alloc_size(b.buf) * cfg_.gmem_bufs);
            }

            auto slm_buffers = alloc_mgr_.find_buffers(alloc_kind_t::slm);
            if (!slm_buffers.empty()) {
                ir_assert(slm_buffers.size() == 1);

                auto &slm_buf = slm_buffers[0];
                int non_ab_slm_size
                        = alloc_mgr_.alloc_size(slm_buf) - ab_slm_size_;
                alloc_updater.resize(slm_buf,
                        non_ab_slm_size + ab_slm_size_ * cfg_.slm_bufs);
            }

            ret = alloc_updater.update(ret);
        }

        // Remove zero-out statement for C (handled by sub_fma_acc_with_zero).
        if (fuse_zero_out_with_fma_)
            ret = substitute(ret, step_.c_zero_out(), stmt_t(), 1);

        return ret;
    }

private:
    struct buffer_info_t {
        buffer_info_t(const expr_t &buf, int size) : buf(buf), size(size) {}

        expr_t buf;
        int size;
    };

    stmt_t create_iteration(const compute_iterator_t &it,
            sbid_manager_t &sbid_mgr, bool in_loop_body = false) const {
        auto g2s_load = step_.g2s_load();
        auto g2s_store = step_.g2s_store();
        auto prefetch = step_.prefetch();
        auto g2r_load = step_.g2r_load();
        auto s2r_load = step_.s2r_load();
        auto mul = step_.mul();
        auto lets = step_.inner_let_stmts();
        auto &outer_loop_info = loop_nest_.outer_loop_info();
        auto if_stmts = step_.if_stmts();

        loop_nest_.for_each_loop_var([&](const expr_t &v) {
            expr_t mul_var_value;
            expr_t preload_var_value;
            if (v.is_same(outer_loop_info.var) && in_loop_body
                    && outer_loop_info.has_var_refs()) {
                mul_var_value = outer_loop_info.mul_var_load();
                preload_var_value = outer_loop_info.preload_var_load();
            } else {
                mul_var_value = it.mul_loop_it.var_value(v);
                preload_var_value = it.preload_loop_it.var_value(v);
            }
            g2s_load = const_fold(substitute(g2s_load, v, preload_var_value));
            g2s_store = const_fold(substitute(g2s_store, v, preload_var_value));
            prefetch = const_fold(substitute(prefetch, v, preload_var_value));
            for (auto &m : mul) {
                m = const_fold(substitute(m, v, mul_var_value));
            }
            for (auto &s : g2r_load) {
                s = const_fold(substitute(s, v, mul_var_value));
            }
            for (auto &s : s2r_load) {
                if (count_object(s, v) > 0) ir_error_not_expected();
                s = const_fold(substitute(s, v, preload_var_value));
            }
            for (int i = 0; i < int(lets.size()); i++) {
                auto &let = lets[i];
                auto &orig_let = step_.inner_let_stmts()[i];
                expr_t var_value;
                bool is_preload_let = step_.is_preload_let(orig_let);
                bool is_mul_let = step_.is_mul_let(orig_let);
                if (is_preload_let && !is_mul_let) {
                    var_value = preload_var_value;
                } else if (is_mul_let && !is_preload_let) {
                    var_value = mul_var_value;
                } else {
                    ir_assert(count_object(let.as<let_t>().value, v) == 0)
                            << "Unexpected reference to variable " << v
                            << " from " << let;
                    continue;
                }
                let = const_fold(substitute(let, v, var_value));
            }
        });

        if (params_.use_slm) {
            g2s_load = sub_gmem_bufs(g2s_load, it, /*is_read=*/false);
            g2s_store = sub_gmem_bufs(g2s_store, it, /*is_read=*/true);

            g2s_store = sub_slm_bufs(g2s_store, it, /*is_read=*/false);
            for (auto &s : s2r_load) {
                s = sub_slm_bufs(s, it, /*is_read=*/true);
            }
        }

        if (it.is_first_mul() && fuse_zero_out_with_fma_) {
            for (auto &m : mul) {
                m = sub_fma_acc_with_zero(m);
            }
        }

        if (it.is_last_g2s_store())
            g2s_store = remove_post_inc_stores(g2s_store);
        if (it.is_last_g2s_load()) g2s_load = remove_post_inc_stores(g2s_load);
        if (it.is_last_prefetch()) prefetch = remove_post_inc_stores(prefetch);
        if (it.is_last_mul()) {
            for (auto &s : s2r_load)
                s = remove_post_inc_stores(s);
            for (auto &s : g2r_load)
                s = remove_post_inc_stores(s);
        }

        stmt_t iter_stmt;

        if (it.do_g2s_load()) iter_stmt = iter_stmt.append(g2s_load);

        if (it.do_g2s_store() && it.slm_bufs() == 1) {
            iter_stmt = iter_stmt.append(g2s_store);
        }

        if (it.do_prefetch()) iter_stmt = iter_stmt.append(prefetch);

        if (it.do_mul()) {
            for (size_t i = 0; i < mul.size(); i++) {
                iter_stmt = iter_stmt.append(g2r_load[i]);
                iter_stmt = iter_stmt.append(s2r_load[i]);
                iter_stmt = iter_stmt.append(mul[i]);
            }
        }
        if (it.do_g2s_store() && it.slm_bufs() >= 2) {
            iter_stmt = iter_stmt.append(g2s_store);
        }

        if_unroll_injector_t if_unroll(if_stmts);
        if_unroll.visit(iter_stmt);
        iter_stmt = if_unroll.stmt();

        if (cfg_.assign_sbids)
            iter_stmt = sbid_assigner_t(sbid_mgr).assign(iter_stmt);

        iter_stmt = inject_local_let(iter_stmt, lets, it.linear_id);

        return iter_stmt;
    }

    stmt_t sub_gmem_bufs(const stmt_t &stmt, const compute_iterator_t &it,
            bool is_read) const {
        if (it.slm_bufs() == 0) return stmt;
        if (is_read && !it.do_g2s_store()) return stmt;
        if (!is_read && !it.do_g2s_load()) return stmt;

        int buf_idx = (is_read ? it.gmem_read_buf_index()
                               : it.gmem_write_buf_index());
        if (buf_idx == 0) return stmt;

        auto ret = stmt;
        for (auto &b : g2s_reg_bufs_) {
            ret = substitute(ret, b.buf, b.buf[buf_idx * b.size]);
        }
        return ret;
    }

    stmt_t sub_slm_bufs(const stmt_t &stmt, const compute_iterator_t &it,
            bool is_read) const {
        if (it.slm_bufs() <= 1) return stmt;
        if (is_read && !it.do_mul()) return stmt;
        if (!is_read && !it.do_g2s_store()) return stmt;

        int upd = (is_read ? it.slm_read_offset_update()
                           : it.slm_write_offset_update());

        auto stmt_vec = flatten_statements(stmt);

        stmt_t ret = stmt;
        for (auto &s : stmt_vec) {
            auto *call = s.as_ptr<func_call_t>();
            if (!call) continue;
            auto *func = call->func.as_ptr<send_t>();
            if (!func) continue;

            auto &send = call->func.as<send_t>();
            auto &args = call->args;
            auto &mem_buf = send_t::arg_mem_buf(args);
            auto &header_buf = send_t::arg_mem_off(args);

            // This is not send to SLM, skip.
            if (!send.is_slm()) continue;

            // May have signed offset.
            auto store_obj = send.create_offset_store(
                    header_buf, mem_buf, upd, /*is_signed_offset=*/true);
            auto &store = store_obj.as<store_t>();
            expr_t old_value
                    = load_t::make(send.address_type(), store.buf, store.off);
            auto post_inc_store = store_t::make(
                    store.buf, store.off, old_value + store.value);
            ret = substitute(ret, s, stmt_seq_t::make(s, post_inc_store), 1);
        }

        return ret;
    }

    static stmt_t sub_fma_acc_with_zero(const stmt_t &stmt) {
        auto stmt_vec = flatten_statements(stmt);

        object_eq_set_t<expr_t> seen_dst;
        stmt_t ret = stmt;
        for (auto &s : stmt_vec) {
            if (is_func_call<dpas_t>(s) && !dpas_t::is_dp4a_call(s)) {
                auto &call = s.as<func_call_t>();

                auto &dst = dpas_t::arg_dst(s);
                auto src0 = expr_t(0); // Will be translated to null register.
                auto &src1 = dpas_t::arg_src1(s);
                auto &src2 = dpas_t::arg_src2(s);

                if (!seen_dst.insert(dst).second) continue;

                auto new_call = func_call_t::make(
                        call.func, {dst, src0, src1, src2}, call.attr);
                ret = substitute(ret, s, new_call, 1);
            } else if (is_func_call<mad_t>(s)) {
                auto &call = s.as<func_call_t>();

                auto &dst = mad_t::arg_dst(s);
                auto src0 = expr_t(0); // Will be translated to null register.
                auto &src1 = mad_t::arg_src1(s);
                auto &src2 = mad_t::arg_src2(s);

                if (!seen_dst.insert(dst).second) continue;

                auto new_call = func_call_t::make(
                        call.func, {dst, src0, src1, src2}, call.attr);
                ret = substitute(ret, s, new_call, 1);
            }
        }
        return ret;
    }

    // Returns memory buffers if is_mem is true and register buffers otherwise.
    static object_set_t<expr_t> find_send_buffers(
            const stmt_t &s, bool is_mem) {
        object_set_t<expr_t> ret;
        auto calls = find_objects<func_call_t>(s);
        for (auto &_c : calls) {
            auto &c = _c.as<func_call_t>();
            if (!c.func.is<send_t>()) continue;
            auto &buf = (is_mem ? send_t::arg_mem_buf(_c)
                                : send_t::arg_reg_buf(_c));
            ret.insert(buf.as<ptr_t>().base);
        }
        return ret;
    }

    static stmt_t inject_local_let(const stmt_t &_s,
            const std::vector<stmt_t> &enclosed_lets, int id) {
        auto s = _s;

        // Inject let statements from the innermost loop.
        for (auto &_let : enclosed_lets) {
            auto &let = _let.as<let_t>();
            s = let_t::make(let.var, let.value, s);
        }

        // Substitute variables to avoid clashing.
        auto lets = find_objects<let_t>(s);
        for (auto &_let : lets) {
            auto &let = _let.as<let_t>();
            auto &var = let.var.as<var_t>();
            auto local_var = var_t::make(
                    var.type, var.name + "_" + std::to_string(id));
            s = substitute(s, let.var, local_var);
        }
        return s;
    }

    static stmt_t remove_post_inc_stores(const stmt_t &_s) {
        auto stores = find_objects<store_t>(_s);
        auto s = _s;
        for (auto &_store : stores) {
            auto &store = _store.as<store_t>();
            if (!contains_object(store.value, store.buf)) continue;
            s = substitute(s, store, stmt_t());
        }
        return s;
    }

    const conv_config_t &cfg_;
    ir_context_t &ir_ctx_;
    int ab_slm_size_;

    stmt_t root_;
    alloc_manager_t alloc_mgr_;
    compute_step_t step_;
    compute_loop_nest_t loop_nest_;
    compute_params_t params_;

    std::vector<buffer_info_t> g2s_reg_bufs_; // For SLM buffering.
    bool fuse_zero_out_with_fma_ = false;
};

// Injects loop unrolling based on the config. Possible options:
// - Without preload (no SLM buffering, no prefetch)
// - With SLM buffering
// - With prefetch
stmt_t inject_unrolling(const stmt_t &s, const conv_config_t &cfg,
        ir_context_t &ir_ctx, int ab_slm_size) {
    auto ret = unrolling_injector_t(s, cfg, ir_ctx, ab_slm_size).inject();
    trace_pass("inject_unrolling", ret);
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

// Splits wide GRF stores otherwise unsupported in HW.
stmt_t split_wide_stores(ngen::HW hw, const stmt_t &s) {
    auto ret = store_splitter_t(hw).mutate(s);
    trace_pass("split_wide_stores", ret);
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

class expr_scalarizer_t : public ir_mutator_t {
public:
    expr_scalarizer_t(int elems, int idx,
            const object_map_t<expr_t, std::vector<expr_t>> &vec_vars)
        : elems_(elems), idx_(idx), vec_vars_(vec_vars) {}

    object_t _mutate(const cast_t &obj) override {
        if (obj.is_bool_vec_u16()) return obj;
        auto type = obj.type;
        auto expr = mutate(obj.expr);
        if (!type.is_scalar()) {
            ir_assert(type.elems() == elems_) << expr;
            type = type.scalar();
        }
        return cast_t::make(type, expr, obj.saturate);
    }

    object_t _mutate(const var_t &obj) override {
        if (obj.type.is_scalar()) return obj;

        auto it = vec_vars_.find(obj);
        ir_assert(it != vec_vars_.end()) << "Can't find variable: " << obj;
        ir_assert(int(it->second.size()) == elems_);
        return it->second[idx_];
    }

    object_t _mutate(const shuffle_t &obj) override {
        expr_t new_obj = ir_mutator_t::_mutate(obj);
        auto &shuffle = new_obj.as<shuffle_t>();
        ir_assert(shuffle.type.elems() == elems_) << new_obj;
        return new_obj[idx_];
    }

private:
    int elems_;
    int idx_;
    const object_map_t<expr_t, std::vector<expr_t>> &vec_vars_;
};

class overflow_fixer_t : public ir_mutator_t {
public:
    overflow_fixer_t(const constraint_set_t &cset) {
        for (auto &kv : cset.relations()) {
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
        return ir_mutator_t::_mutate(obj);
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

    overflow_bound_finder_t bound_finder_;
    object_map_t<expr_t, std::vector<expr_t>> vec_vars_;
    object_set_t<expr_t> vars_with_load_;
};

// Detects and fixes overflows of operations with 32-bit integers.
// Before (a * b can overflow):
//     c.u64 = u64(c_ptr) + a.s32 * b.s32
// After:
//     c.u64 = u64(c_ptr) + s64(a.s32) * b.s32
stmt_t fix_int32_overflow(const stmt_t &s, const constraint_set_t &cset) {
    auto ret = overflow_fixer_t(cset).mutate(s);
    trace_pass("fix_int32_overflow", ret);
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

stmt_t optimize_peephole(const stmt_t &s) {
    auto ret = peephole_optimizer_t().mutate(s);
    trace_pass("optimize_peephole", ret);
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

stmt_t optimize_barrier(const stmt_t &s) {
    auto ret = barrier_optimizer_t().mutate(s);
    trace_pass("optimize_barrier", ret);
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

// Injects broadcasts for scalar if conditions. Example:
// Before:
//     if (cond) { ... }
// After (for SIMD8):
//     if (bcast8(cond)) { ... }
stmt_t fixup_if_conditions(const stmt_t &s, const conv_config_t &cfg) {
    auto ret = if_condition_fixer_t(cfg.simd_size()).mutate(s);
    trace_pass("fixup_if_conditions", ret);
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

// Unrolls loops according to their unroll attribute.
// Before:
//     for (int i = 0; i < 2; i++) [unroll: 2] {
//         body(i);
//     }
// After:
//     body(0);
//     body(1);
stmt_t unroll_loops(const stmt_t &s, ir_context_t &ir_ctx) {
    auto ret = loop_unroller_t(ir_ctx).mutate(s);
    trace_pass("unroll_loops", ret);
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
        if (!is_mad && !is_dpas) return ir_mutator_t::_mutate(obj);

        auto dst_buf = ptr_base(obj.args[0]);
        auto src0_buf = ptr_base(obj.args[1]);
        auto src1_buf = ptr_base(obj.args[2]);
        auto src2_buf = ptr_base(obj.args[3]);

        // src0 may be null in some cases, skip it.
        if (!src0_buf.is_empty()) bufs_.insert(src0_buf);
        bufs_.insert(src1_buf);
        bufs_.insert(src2_buf);

        instructions_.insert(obj);

        return obj;
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
        for (auto &buf : bufs_) {
            buf_vec.push_back(buf);
            buf_sizes.push_back(all_buf_sizes_.at(buf));
        }
        attr_ = bank_conflict_attr_t::make(buf_vec, buf_sizes, instructions);
    }

    static expr_t ptr_base(const expr_t &e) {
        if (e.is<var_t>()) return e;
        auto *ptr = e.as_ptr<ptr_t>();
        if (ptr) return e.as<ptr_t>().base;
        return expr_t();
    }

    object_map_t<expr_t, int> all_buf_sizes_;
    object_eq_set_t<expr_t> bufs_;
    object_eq_set_t<stmt_t> instructions_;
    bool is_frozen = false;

    alloc_attr_t attr_;
};

// Injects an allocation attribute to store information about buffer usages in
// instructions. This information is used during nGEN lowering to avoid bank
// conflicts in allocated buffers.
stmt_t inject_bank_conflict_attribute(const stmt_t &s) {
    auto ret = bank_conflict_attribute_injector_t().mutate(s);
    trace_pass("inject_bank_conflict_attribute", ret);
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

// Converts dpas to dp4a.
stmt_t inject_dp4a(const stmt_t &s) {
    auto ret = dp4a_injector_t().mutate(s);
    trace_pass("inject_dp4a", ret);
    return ret;
}

stmt_t create_reduce_stmt(const layout_t &src, const layout_t &dst,
        const expr_t &src_buf, const expr_t &dst_buf, const tensor_t &_sub_tile,
        uint32_t reduction_mask, bool drop_dims = true) {
    auto sub_tile = _sub_tile;
    if (sub_tile.is_empty()) sub_tile = tensor_t(src.dims());
    ir_assert(src.ndims() == sub_tile.ndims());
    int ndims = src.ndims();

    // Align dst layout with src layout according to the mask if needed.
    layout_t dst_aligned;
    if (drop_dims) {
        std::vector<int> dst2src(dst.ndims());
        int dst_dim_idx = 0;
        for (int i = 0; i < ndims; i++) {
            if ((reduction_mask & (1 << i)) != 0) {
                dst2src[dst_dim_idx] = i;
                dst_dim_idx++;
            }
        }
        ir_assert(dst_dim_idx == dst.ndims()) << "Incompatible reduction mask.";

        auto dst_blocks = dst.blocks();
        for (auto &b : dst_blocks)
            b.dim_idx = dst2src[b.dim_idx];

        // Create final layout.
        dst_aligned = layout_t(dst.type(), ndims, dst.offset(), dst_blocks);
    } else {
        dst_aligned = dst;
    }

    std::vector<dim_t> dst_tile_dims = sub_tile.dims();
    std::vector<expr_t> dst_tile_start = sub_tile.start();
    for (int i = 0; i < ndims; i++) {
        if ((reduction_mask & (1 << i)) == 0) {
            dst_tile_dims[i] = 1;
            dst_tile_start[i] = expr_t(0);
            continue;
        }
    }
    dst_aligned = dst_aligned.map(tensor_t(dst_tile_dims, dst_tile_start));

    auto func = reduce_t::make(src, dst_aligned);
    return func.call({dst_buf, src_buf});
}

// Performs the following operation:
//     buf = alpha * buf + beta
stmt_t create_mul_add_stmt(ngen::HW hw, const expr_t &buf, int size,
        const type_t &type, float alpha, float beta) {
    if (alpha == 1 && beta == 0) return stmt_t();

    stmt_t ret;
    int step_bytes = 2 * ngen::GRF::bytes(hw);
    for (int i = 0; i < size; i += step_bytes) {
        auto elems = std::min(step_bytes, size - i) / type.size();
        auto e_alpha = shuffle_t::make_broadcast(alpha, elems);
        auto e_beta = shuffle_t::make_broadcast(beta, elems);
        auto e = load_t::make(type.with_elems(elems), buf, i);
        // Avoid extra IR expressions when not needed.
        if (alpha == 0)
            e = shuffle_t::make_broadcast(expr_t(0.0f), elems);
        else if (alpha != 1)
            e *= e_alpha;
        if (beta != 0) e += e_beta;
        ir_assert(e.type().scalar() == type);
        ret = ret.append(store_t::make(buf, i, e));
    }
    return ret;
}

stmt_t create_zero_out_stmt(ngen::HW hw, const expr_t &buf, int size) {
    return create_mul_add_stmt(hw, buf, size, type_t::f32(), 0, 0);
}

class slm_reduce_builder_t {
public:
    slm_reduce_builder_t() = default;

    slm_reduce_builder_t(const ngen::HW &hw, ir_context_t &ir_ctx,
            const constraint_set_t &cset, const grid_info_t &tg_grid,
            const expr_t &reg_buf, const layout_t &reg_layout,
            const tensor_t &thr_tile, int dim = 2)
        : hw_(hw)
        , tg_grid_(tg_grid)
        , reg_buf_(reg_buf)
        , reg_layout_(reg_layout)
        , thr_tile_(thr_tile)
        , dim_(dim) {
        ir_assert((dim_ >= 0) && (dim_ <= 2));
        ir_assert(tg_grid_.dim(dim_) > 1);

        tmp_reg_buf_ = ir_ctx.create_tmp_var(type_t::byte_ptr());
        slm_buf_ = ir_ctx.create_tmp_var(type_t::byte_ptr(), "reduce_slm");
        tg_ndims_ = (dim_ != 2) ? dim_ + 1 : tg_grid_.ndims();

        build(ir_ctx, cset);
    }

    bool is_empty() const { return reg_buf_.is_empty(); }

    const layout_t &reg_layout() const { return reg_layout_; }

    const tensor_t &thr_tile() const { return thr_tile_; }

    const stmt_t &store_stmt() const { return store_stmt_; }

    const stmt_t &load_stmt() const { return load_stmt_; }

    const std::vector<stmt_t> &allocs() const { return allocs_; }

    stmt_t stmt() const {
        stmt_t ret;
        ret = ret.append(funcs::barrier());
        ret = ret.append(store_stmt_);
        ret = ret.append(funcs::barrier());
        ret = ret.append(load_stmt_);
        ret = inject_alloc_stmts(ret, allocs_);
        return ret;
    }

private:
    void build(ir_context_t &ir_ctx, const constraint_set_t &cset) {
        int ndims = reg_layout_.ndims();

        // Create SLM layout to store all intermediate buffers from the thread
        // group.
        layout_t slm_layout(reg_layout_.type(), ndims + tg_ndims_,
                reg_layout_.offset(), reg_layout_.blocks());
        for (int i = tg_ndims_ - 1; i >= 0; i--) {
            slm_layout = slm_layout.add_outer_block(ndims + i, tg_grid_.dim(i));
        }

        slm_buf_size_ = slm_layout.size();

        // Write thread tile to SLM.
        std::vector<dim_t> write_dims = reg_layout_.dims();
        std::vector<expr_t> write_start(ndims + tg_ndims_, 0);
        write_dims.resize(ndims + tg_ndims_, 1);
        for (int i = tg_ndims_ - 1; i >= 0; i--) {
            write_start[ndims + i] = tg_grid_.idx(i);
        }
        auto write_tile = tensor_t(write_dims, write_start);
        auto write = make_access_builder(hw_, ir_ctx, cset,
                view_t(slm_layout.map(write_tile)), slm_buf_, reg_buf_,
                send_op_t::store, send_address_t::slm);
        store_stmt_ = write.stmt();

        auto &write_layout = write.reg_layout();
        ir_assert(write_layout == reg_layout_) << "Incompatible layouts.";

        // Redistribute the layout to read/reduce all k-axis tiles from every
        // thread.
        auto local_thr_tile = reg_layout_.split(tg_grid_.sub_grid({dim_}));
        reg_layout_ = reg_layout_.map(tensor_t(local_thr_tile.dims()));

        std::vector<dim_t> read_dims(ndims + tg_ndims_, 1);
        std::vector<expr_t> read_start(ndims + tg_ndims_);
        for (int i = 0; i < ndims; i++) {
            read_dims[i] = local_thr_tile(i);
            read_start[i] = local_thr_tile.start(i);
        }
        read_dims[ndims + dim_] = tg_grid_.dim(dim_);
        for (int i = 0; i < tg_ndims_; i++) {
            read_start[ndims + i] = (i == dim_) ? 0 : tg_grid_.idx(i);
        }
        tensor_t read_tile(read_dims, read_start);
        auto read = make_access_builder(hw_, ir_ctx, cset,
                view_t(slm_layout.map(read_tile)), slm_buf_, tmp_reg_buf_,
                send_op_t::load, send_address_t::slm);

        load_stmt_ = load_stmt_.append(
                create_zero_out_stmt(hw_, reg_buf_, reg_layout_.size()));
        load_stmt_ = load_stmt_.append(read.stmt());

        tmp_reg_buf_size_ = std::max(tmp_reg_buf_size_, read.reg_buf_size());

        auto read_layout = read.reg_layout();
        load_stmt_
                = load_stmt_.append(create_reduce_stmt(read_layout, reg_layout_,
                        tmp_reg_buf_, reg_buf_, tensor_t(), reduction_mask()));

        allocs_.push_back(
                alloc_t::make(slm_buf_, slm_buf_size_, alloc_kind_t::slm));
        allocs_.push_back(alloc_t::make(
                tmp_reg_buf_, tmp_reg_buf_size_, alloc_kind_t::grf));

        if (!thr_tile_.is_empty()) {
            thr_tile_ = thr_tile_.create_sub_tensor(local_thr_tile);
        }
    }

    uint32_t reduction_mask() const {
        uint32_t mask = 0xFFFFFFFF;
        for (int i = 0; i < tg_ndims_; i++) {
            int k_dim_idx = reg_layout_.ndims() + i;
            mask &= ~(1 << k_dim_idx);
        }
        return mask;
    }

    ngen::HW hw_;
    grid_info_t tg_grid_;

    expr_t reg_buf_;
    layout_t reg_layout_;
    tensor_t thr_tile_;

    int dim_;

    expr_t tmp_reg_buf_;
    int tmp_reg_buf_size_ = 0;

    expr_t slm_buf_;
    int slm_buf_size_ = 0;

    int tg_ndims_;

    stmt_t store_stmt_;
    stmt_t load_stmt_;

    std::vector<stmt_t> allocs_;
};

// Zero pads a register buffer of f32 type.
class zero_pad_builder_t {
public:
    zero_pad_builder_t() = default;

    zero_pad_builder_t(const constraint_set_t &cset,
            const view_t &full_mem_view, const view_t &mem_view)
        : cset_(&cset), full_mem_view_(full_mem_view), mem_view_(mem_view) {}

    bool is_empty() const { return mem_view_.is_empty(); }

    expr_t create_mask(const layout_t &reg_layout, const tensor_t &tile) const {
        ir_assert(!is_empty());
        auto layout = reg_layout.map(tile);
        auto view = mem_view_.create_sub_view(tile);
        mask_tensor_t mask_tensor(layout);
        std::vector<dim_t> args(layout.ndims());
        fill_mask_impl(mask_tensor, 0, args, view, layout);
        mask_tensor.simplify(*cset_);
        return mask_tensor.to_expr(tile.elems());
    }

    stmt_t build_stmt(const layout_t &reg_layout, const expr_t &reg_buf) const {
        ir_assert(mem_view_.nvdims() == reg_layout.ndims())
                << "Incompatible view/layout.";
        int max_step = 16; // Handle 16 elements at most in one step.
        auto base_tile = reg_layout.split_into_max_tile(
                max_step, /*is_dense_tile=*/true);
        stmt_t stmt;
        reg_layout.for_each_tile(
                base_tile, [&](const std::vector<dim_t> &start) {
                    tensor_t tile(base_tile.dims(), start);
                    int off = reg_layout(start) * reg_layout.type().size();
                    auto mask = create_mask(reg_layout, tile);
                    auto store = store_t::make(reg_buf, off,
                            shuffle_t::make_broadcast(0.0f, tile.elems()),
                            store_t::default_stride, -mask);
                    stmt = stmt.append(store);
                });
        return stmt;
    }

private:
    void fill_mask_impl(mask_tensor_t &mask_tensor, int idx,
            std::vector<dim_t> &args, const view_t &view,
            const layout_t &layout) const {
        if (idx == layout.ndims()) {
            std::vector<expr_t> vargs;
            for (int i = 0; i < layout.ndims(); i++)
                vargs.push_back(view.vstart(i) + args[i]);
            expr_t mask = full_mem_view_.vmask(vargs);
            auto off = layout.offset(args, /*ignore_offset=*/true);
            mask_tensor.set_mask(off, mask);
            return;
        }

        for (int i = 0; i < int(layout.dims()[idx]); i++) {
            args[idx] = i;
            fill_mask_impl(mask_tensor, idx + 1, args, view, layout);
        }
    }

    const constraint_set_t *cset_;

    view_t full_mem_view_;
    view_t mem_view_;

    stmt_t stmt_;
};

// Represents the state of a post-op tensor.
//
// There are three kinds of tensors:
// - C tensor converted to f32
//   - Never loaded or stored to global memory
// - Input tensor
//   - No store, only load
// - Output tensor
//   - No load, only store
//
// Post-op tensors that are both input/output are not expected/supported as
// they doesn't occur in convolution. Post-op tensors with global reduction
// (like lhs += rhs) are treated as output-only and handled via atomic stores.
//
// A post-op tensor optionally requires:
// - Conversion to f32 (post-ops are done in f32)
// - Reduction
//   - For output tensors with broadcast dimensions
// - Masking during post-ops
//   - When a post-op is not zero preserving
class post_op_tensor_t {
public:
    post_op_tensor_t(ngen::HW hw, ir_context_t &ir_ctx,
            const constraint_set_t &cset, const post_op_tensor_info_t &info)
        : hw_(hw), ir_ctx_(&ir_ctx), cset_(&cset), info_(info) {
        if (!mem_buf().is_empty()) {
            auto &type = mem_buf().type();
            if (!type.is_ptr()) {
                ir_assert(type.is_f32()) << "Expected f32: " << mem_buf();
                reg_buf_ = mem_buf();
                reg_layout_ = layout_t(
                        type, 0, std::vector<dim_t>(mem_view().nvdims(), 1));
            }
        }
    }

    const view_t &mem_view() const { return info_.view(); }

    const expr_t &mem_buf() const { return info_.buf(); }

    // Bitmask with broadcast information for the tensor:
    // - (mask() & (1 << idx)) == 0 -> idx is a brodcast dimension (equal to 1)
    // - (mask() & (1 << idx)) != 0 -> idx dimension matches the C dimension
    uint32_t mask() const { return info_.mask(); }

    // Placeholder variable to represent the tensor in post-op expressions.
    const expr_t &op_var() const { return info_.op_var(); }

    const layout_t &reg_layout() const { return reg_layout_; }

    const expr_t &reg_buf() const { return reg_buf_; }

    post_op_tensor_t create_sub_tensor(const tensor_t &_tile) const {
        auto ret = *this;
        auto tile = apply_mask(_tile);
        ret.info_ = ret.info_.create_sub_tensor(tile);
        if (!reg_layout_.is_empty()) {
            if (needs_reduction()) {
                tensor_t reduce_tile(_tile.dims(), tile.start());
                ret.reg_layout_ = ret.reg_layout_.map(reduce_tile);
            } else {
                ret.reg_layout_ = ret.reg_layout_.map(tile);
            }
        }
        ret.allocs_.clear();
        return ret;
    }

    bool needs_load() const {
        if (!info_.is_input()) return false;
        if (!mem_buf().type().is_ptr()) return false;
        return true;
    }

    bool needs_store() const { return info_.is_output(); }

    bool needs_masked_update() const { return info_.needs_masked_update(); }

    bool needs_f32_convert() const { return !mem_view().type().is_f32(); }

    bool needs_reduction() const {
        if (!info_.is_output()) return false;

        for (int i = 0; i < mem_view().nvdims(); i++) {
            if (is_broadcast_dim(i)) {
                if (reg_layout_.dims()[i] != 1) return true;
            }
        }
        return false;
    }

    bool is_broadcast_dim(int dim_idx) const {
        ir_assert(dim_idx >= 0 && dim_idx < mem_view().nvdims());
        return (mask() & (1 << dim_idx)) == 0;
    }

    int estimate_grf_consumption() const {
        int elems = int(mem_view().create_dense_vlayout().elems());

        int ret = 0;
        ret += elems * mem_view().type().size();
        if (needs_f32_convert()) ret += elems * type_t::f32().size();
        return ret;
    }

    void set_reg_layout(const layout_t &layout) { reg_layout_ = layout; }

    void set_reg_buf(const expr_t &buf) { reg_buf_ = buf; }

    void set_preload(bool value = true) { do_preload_ = value; }

    bool do_preload() const { return do_preload_; }

    tensor_t apply_mask(const tensor_t &tile) const {
        ir_assert(mem_view().nvdims() == tile.ndims());

        auto start = tile.start();
        auto dims = tile.dims();

        for (int i = 0; i < tile.ndims(); i++) {
            if (!is_broadcast_dim(i)) continue;
            start[i] = expr_t(0);
            dims[i] = 1;
        }
        return tensor_t(dims, start);
    }

    void init_output_buffer(const tensor_t &tile) {
        ir_assert(needs_store());

        ir_assert(reg_layout_.is_empty());
        ir_assert(reg_buf_.is_empty());

        reg_buf_ = make_tmp_reg_buffer();

        reg_layout_ = mem_view().create_dense_vlayout();
        reg_layout_ = reg_layout_.retype(type_t::f32());

        // If this is output and there are masked dimensions then this buffer
        // is computed via reduction. Extend layout to cover full masked_tile
        // and apply the final reduction after all tiles.
        auto masked_tile = apply_mask(tile);
        for (int i = 0; i < masked_tile.ndims(); i++) {
            if (masked_tile(i) >= tile(i)) continue;
            ir_assert(masked_tile(i) == 1) << "Unexpected output tensor shape.";
            reg_layout_ = reg_layout_.add_outer_block(i, tile(i));
        }
        register_buffer(reg_buf_, reg_layout_.size());
    }

    stmt_t build_load_stmt() {
        ir_assert(needs_load());
        ir_assert(reg_buf_.is_empty());

        reg_buf_ = make_tmp_reg_buffer();
        auto read = make_access_builder(hw_, *ir_ctx_, *cset_, mem_view(),
                mem_buf(), reg_buf_, send_op_t::load, send_address_t::a64);
        reg_layout_ = read.reg_layout();
        register_buffer(reg_buf_, read.reg_buf_size());
        return read.stmt();
    }

    stmt_t build_prefetch_stmt() const {
        ir_assert(needs_load());

        auto prefetch = make_access_builder(hw_, *ir_ctx_, *cset_, mem_view(),
                mem_buf(), expr_t(), send_op_t::prefetch, send_address_t::a64);
        return prefetch.stmt();
    }

    stmt_t build_convert_stmt() {
        if (!needs_load() || !needs_f32_convert()) return stmt_t();

        auto f32_buf = make_tmp_reg_buffer();
        auto f32_layout = reg_layout_.retype(type_t::f32()).make_dense();

        register_buffer(f32_buf, f32_layout.size());

        // Reorder to f32.
        auto ret = create_reorder_stmt(
                reg_layout_, f32_layout, reg_buf_, f32_buf);

        // Assign new f32 layout and buffer.
        reg_layout_ = f32_layout;
        reg_buf_ = f32_buf;

        return ret;
    }

    stmt_t build_zero_out_stmt() const {
        ir_assert(needs_store());
        return create_zero_out_stmt(hw_, reg_buf_, reg_layout_.size());
    }

    stmt_t build_reduce_stmt() {
        ir_assert(needs_store());

        stmt_t stmt;

        if (needs_reduction()) {
            auto reduced_layout = mem_view().create_dense_vlayout();
            ir_assert(reduced_layout.size() <= reg_layout_.size());

            stmt = stmt.append(
                    create_reduce_stmt(reg_layout_, reduced_layout, reg_buf_,
                            reg_buf_, tensor_t(), mask(), /*drop_dims=*/false));
            reg_layout_ = reduced_layout;
        }

        // Apply optional scaling.
        stmt = stmt.append(create_mul_add_stmt(hw_, reg_buf_,
                reg_layout_.size(), reg_layout_.type(), info_.scale(), 0));

        return stmt;
    }

    stmt_t build_slm_store_stmt(const grid_info_t &tg_grid) {
        ir_assert(needs_store());
        tensor_t tile(mem_view().vdims());
        slm_reduce_builder_ = slm_reduce_builder_t(
                hw_, *ir_ctx_, *cset_, tg_grid, reg_buf_, reg_layout_, tile, 1);
        return slm_reduce_builder_.store_stmt();
    }

    stmt_t build_slm_load_stmt() {
        ir_assert(needs_store());
        ir_assert(!slm_reduce_builder_.is_empty());

        reg_layout_ = slm_reduce_builder_.reg_layout();

        auto new_tile = slm_reduce_builder_.thr_tile();
        info_ = info_.create_sub_tensor(new_tile);

        auto &slm_allocs = slm_reduce_builder_.allocs();
        allocs_.insert(allocs_.end(), slm_allocs.begin(), slm_allocs.end());

        return slm_reduce_builder_.load_stmt();
    }

    stmt_t build_store_stmt() const {
        ir_assert(needs_store());

        auto write = make_access_builder(hw_, *ir_ctx_, *cset_, mem_view(),
                mem_buf(), reg_buf(), send_op_t::atomic_fadd,
                send_address_t::a64);
        ir_assert(write.reg_layout() == reg_layout());

        return write.stmt();
    }

    expr_t load_expr(const tensor_t &tile, int dim_idx) const {
        auto &type = reg_layout_.type();
        int elems = is_broadcast_dim(dim_idx) ? 1 : tile.elems();
        int off = reg_layout_.offset_in_bytes(expr_cast<dim_t>(tile.start()));
        auto ret = (reg_buf_.type().is_ptr()
                        ? load_t::make(type.with_elems(elems), reg_buf_, off)
                        : reg_buf_);
        if (elems != tile.elems())
            ret = shuffle_t::make_broadcast(ret, tile.elems());
        return ret;
    }

    stmt_t store_stmt(const tensor_t &tile, int dim_idx, const expr_t &_value,
            const expr_t &mask = expr_t()) const {
        auto value = _value;
        ir_assert(!is_broadcast_dim(dim_idx));
        ir_assert(value.type().elems() == tile.elems());
        // Add cast for booleans for comparison ops.
        if (value.type().is_bool()) {
            value = cast(value, reg_layout_.type().with_elems(tile.elems()));
        }
        int off = reg_layout_.offset_in_bytes(expr_cast<dim_t>(tile.start()));
        auto ret = store_t::make(
                reg_buf_, off, value, store_t::default_stride, mask);
        return ret;
    }

    const std::vector<stmt_t> &allocs() const { return allocs_; }

private:
    expr_t make_tmp_reg_buffer() {
        auto *var = mem_buf().as_ptr<var_t>();
        if (!var) {
            auto *ptr = mem_buf().as_ptr<ptr_t>();
            if (ptr) var = ptr->base.as_ptr<var_t>();
        }
        ir_assert(var) << "Can't extract variable from buffer: " << mem_buf();
        auto &name = var->name;
        return ir_ctx_->create_tmp_var(type_t::byte_ptr(), "tmp_" + name);
    }

    void register_buffer(const expr_t &buf, int size) {
        for (auto &_a : allocs_) {
            auto &a = _a.as<alloc_t>();
            if (a.buf.is_same(buf)) {
                if (size > a.size) {
                    _a = alloc_t::make(a.buf, a.size, a.kind, a.attrs);
                }
                return;
            }
        }
        allocs_.push_back(alloc_t::make(buf, size, alloc_kind_t::grf));
    }

    ngen::HW hw_;
    ir_context_t *ir_ctx_;
    const constraint_set_t *cset_;

    post_op_tensor_info_t info_;

    layout_t reg_layout_;
    expr_t reg_buf_;

    bool do_preload_ = false;

    std::vector<stmt_t> allocs_;

    slm_reduce_builder_t slm_reduce_builder_;
};

// Applies substitutions and broadcasts to generate the final post-op
// expression.
class post_op_bcast_mutator_t : public ir_mutator_t {
public:
    post_op_bcast_mutator_t(
            int elems, const object_map_t<object_t, object_t> &from2to)
        : elems_(elems), from2to_(from2to) {}

    object_t _mutate(const float_imm_t &obj) override {
        return make_bcast(obj);
    }

    object_t _mutate(const int_imm_t &obj) override {
        return make_bcast(float_imm_t::make(obj.value));
    }

    object_t _mutate(const var_t &obj) override {
        auto it = from2to_.find(obj);
        if (it != from2to_.end()) return make_bcast(it->second);

        ir_error_not_expected() << "Unknown variable.";
        return obj;
    }

private:
    object_t make_bcast(const expr_t &e) const {
        if (e.type().elems() == elems_) return e;
        ir_assert(e.type().elems() == 1);
        return shuffle_t::make_broadcast(e, elems_);
    }

    int elems_;
    object_map_t<object_t, object_t> from2to_;
};

// Builds statements to apply a post-op for a given tile.
class post_op_builder_t {
public:
    post_op_builder_t(ngen::HW hw, const post_op_t &post_op)
        : hw_(hw), post_op_(post_op) {}

    const post_op_t &post_op() const { return post_op_; }

    // Applies post-op for a single tile.
    stmt_t build_tile_stmt(const tensor_t &tile,
            const object_map_t<expr_t, post_op_tensor_t *> &args,
            const zero_pad_builder_t &zero_pad_builder) const {
        auto &lhs_tensor = *args.at(post_op_.lhs());
        if (!post_op_.eltwise().is_empty()) {
            // Apply eltwise post-op.
            ir_assert(post_op_.lhs().is_equal(post_op_.rhs()))
                    << "Only supported form is lhs = eltwise(lhs).";
            int lhs_size = lhs_tensor.reg_layout().size();
            int lhs_elems = lhs_size / int(sizeof(float));
            return post_op_.eltwise().call(
                    {expr_t(lhs_elems), lhs_tensor.reg_buf()});
        }

        int inner_dim_idx = -1;
        auto base_inner_tile = find_1d_tile(args, inner_dim_idx);
        auto inner_layout = lhs_tensor.reg_layout().map(base_inner_tile);
        ir_assert(inner_dim_idx != -1);

        // All post-ops are performed in f32.
        for (auto &kv : args) {
            ir_assert(kv.second->reg_layout().type().is_f32());
        }

        // Handle one inner tile at a time. Inner tile covers a single block
        // within a single dimension.
        stmt_t stmt;
        lhs_tensor.reg_layout().for_each_tile(
                base_inner_tile, [&](const std::vector<dim_t> &lhs_start) {
                    tensor_t inner_tile(base_inner_tile.dims(), lhs_start);
                    auto rhs_value = compute_post_op_expr(
                            post_op_.rhs(), inner_tile, inner_dim_idx, args);
                    auto &t = *args.at(post_op_.lhs());
                    expr_t store_mask;
                    if (lhs_tensor.needs_masked_update()) {
                        store_mask = zero_pad_builder.create_mask(
                                inner_layout, inner_tile);
                    }
                    auto inner_stmt = t.store_stmt(
                            inner_tile, inner_dim_idx, rhs_value, store_mask);
                    stmt = stmt.append(inner_stmt);
                });

        return stmt;
    }

private:
    // Returns a 1D tile corresponding to an instruction to partially apply the
    // post-op.
    tensor_t find_1d_tile(const object_map_t<expr_t, post_op_tensor_t *> &args,
            int &inner_dim_idx) const {
        auto &lhs_tensor = *args.at(post_op_.lhs());

        int ndims = lhs_tensor.mem_view().nvdims();

        ir_assert(!lhs_tensor.reg_layout().is_empty());
        ir_assert(!lhs_tensor.reg_layout().blocks().empty());
        auto &b0 = lhs_tensor.reg_layout().blocks()[0];
        ir_assert(dim_t(b0.stride) == 1);
        inner_dim_idx = b0.dim_idx;

        int inner_block = b0.block;
        int max_step = (hw_ >= ngen::HW::XeHPC ? 32 : 16);
        inner_block = std::max(8, math::gcd(inner_block, max_step));

        for (auto &kv : args) {
            auto &t = *kv.second;
            if (t.is_broadcast_dim(b0.dim_idx)) continue;

            auto &l = t.reg_layout();
            ir_assert(!l.is_empty());
            ir_assert(!l.blocks().empty());
            auto &lb0 = l.blocks()[0];
            ir_assert(lb0.dim_idx == b0.dim_idx);
            ir_assert(dim_t(lb0.stride) == 1);
            inner_block = math::gcd(int(lb0.block), inner_block);
        }

        std::vector<dim_t> dims(ndims, 1);
        dims[b0.dim_idx] = inner_block;

        return tensor_t(dims);
    }

    expr_t compute_post_op_expr(const expr_t &expr, const tensor_t &tile,
            int dim_idx,
            const object_map_t<expr_t, post_op_tensor_t *> &args) const {
        object_map_t<object_t, object_t> sub_map;
        for (auto &kv : args) {
            auto &t = *kv.second;
            auto te = t.load_expr(tile, dim_idx);
            sub_map.insert({t.op_var(), te});
        }
        post_op_bcast_mutator_t bcast_mutator(tile.elems(), sub_map);
        return bcast_mutator.mutate(expr);
    }

    ngen::HW hw_;
    post_op_t post_op_;
};

// Epilogue consists of the following steps after the main computation (C += A * B):
// - C GRF reorder to match the memory layout for global memory store
// - C conversion to f32 (if there are post-ops)
// - Applying post-ops (if there are any)
// - C conversion to the memory layout data type
// - C store to global memory
// - Reduction and storing output post-op tensors
//
// In general C tensor is updated/transformed following the C stages described
// below. Each C stage is associated with GRF buffer and its layout.
//   Multiplication ->
//     M_x -> [R_f32] -> [P0_f32] -> ... -> [Pn_f32] -> [Z_f32] -> S_y ->
//   GMEM
//
// Where:
// - x      is data type after multiplication
// - y      is destination data type
// - M_x    is the stage after multiplication
// - R_f32  is the stage after reordering from M_x to f32 (optional)
// - Pi_f32 is the stage after applying Pi post-op (optional)
// - Z_f32  is the stage after restoring zero padding (optional)
// - S_y    is the stage before storing C to global memory
class epilogue_builder_t {
public:
    epilogue_builder_t(const conv_config_t &cfg, ir_context_t &ir_ctx,
            const constraint_set_t &cset, const gemm_schedule_t &gemm_schedule,
            const post_op_context_t &post_op_ctx, const tensor_t &thr_tile,
            const view_t &c_mem_view, const layout_t &c_reg_layout,
            const expr_t &c_mem_buf, const expr_t &c_reg_buf, int tile_size,
            int preload_max_size, int post_op_blk)
        : cfg_(cfg)
        , ir_ctx_(ir_ctx)
        , cset_(cset)
        , gemm_schedule_(gemm_schedule)
        , post_op_ctx_(post_op_ctx)
        , c_mem_view_(c_mem_view)
        , c_mem_buf_(c_mem_buf)
        , tg_grid_(gemm_schedule.tg_grid())
        , tile_size_(tile_size)
        , preload_max_size_(preload_max_size)
        , post_op_blk_(post_op_blk) {

        int tensor_idx = 0;
        for (auto &po_tensor_info : post_op_ctx_.post_op_tensor_infos()) {
            post_op_tensor_t po_tensor(
                    cfg_.hw(), ir_ctx_, cset_, po_tensor_info);
            po_tensor = po_tensor.create_sub_tensor(thr_tile);
            if (po_tensor_info.buf().is_empty()) {
                // C tensor.
                ir_assert(c_po_idx_ == -1);
                c_po_idx_ = tensor_idx;
            }
            post_op_tensors_.push_back(po_tensor);
            tensor_idx++;
        }

        restore_zero_padding_ = post_op_ctx_.need_to_restore_zero_padding();

        for (auto &po : post_op_ctx_.post_ops()) {
            post_op_builders_.emplace_back(cfg.hw(), po);
        }

        // Estimate buffer sizes required to load the full tensor, do not do
        // preload if it requires too much GRF memory.
        int available_size = preload_max_size_;
        for (auto &t : post_op_tensors_) {
            if (!t.needs_load()) continue;
            int required_size = t.estimate_grf_consumption();
            if (required_size > available_size) continue;
            available_size -= required_size;
            t.set_preload();
        }

        build(c_reg_layout, c_reg_buf);
    }

    const stmt_t &stmt() const { return stmt_; }

private:
    void register_buffer(const expr_t &buf, int size) {
        buf_sizes_[buf] = std::max(buf_sizes_[buf], size);
    }

    expr_t make_c_tmp_buffer() const {
        return ir_ctx_.create_tmp_var(type_t::byte_ptr(), "c_tmp");
    }

    // Represents a GRF buffer and layout to store C tensor.
    struct c_stage_t {
        c_stage_t(const layout_t &layout, const expr_t &buf,
                const stmt_t &stmt = stmt_t())
            : layout(layout), buf(buf), stmt(stmt) {}

        void set_next(ngen::HW hw, ir_context_t &ir_ctx, c_stage_t *next,
                bool force_reorder) {
            if (!next) return;
            bool do_reorder
                    = !layout.is_equal(next->layout, /*compare_offset=*/false);
            if (force_reorder) do_reorder = true;
            if (do_reorder) {
                ir_assert(stmt.is_empty());
                // Generate reorder between stages.
                stmt = create_reorder_stmt(
                        layout, next->layout, buf, next->buf);
            } else {
                // Reuse the same GRF buffer for the next stage.
                int this_off = to_cpp<int>(layout.offset_in_bytes());
                int next_off = to_cpp<int>(next->layout.offset_in_bytes());
                ir_assert(next_off == 0);
                MAYBE_UNUSED(next_off);
                next->set_buf(buf[this_off]);
            }
        }

        void set_buf(const expr_t &buf) {
            // Replace old buffer if there is an assigned statement.
            if (!stmt.is_empty()) { stmt = substitute(stmt, this->buf, buf); }
            this->buf = buf;
        }

        const expr_t &buf_base() const {
            if (buf.is<var_t>()) return buf;
            return buf.as<ptr_t>().base;
        }

        int buf_size() const {
            ir_assert(buf.is_same(buf_base()))
                    << "Size must be queried from another stage.";
            return int(layout.size());
        }

        void prepend_stmt(const stmt_t &stmt) {
            this->stmt = stmt.append(this->stmt);
        }

        layout_t layout;
        expr_t buf;
        stmt_t stmt; // Statement to emit after the stage.
    };

    void build(const layout_t &c_reg_layout, const expr_t &c_reg_buf) {
        auto tmp_type = (post_op_builders_.empty() ? c_mem_view_.type()
                                                   : type_t::f32());
        int tmp_buf_elems = tile_size_ / tmp_type.size();
        auto base_tile = c_mem_view_.split_into_max_tile(
                tmp_buf_elems, /*is_dense=*/false);

        // Generate preload statements.
        for (auto &t : post_op_tensors_) {
            if (!t.do_preload()) continue;
            stmt_ = stmt_.append(t.build_load_stmt());
        }

        // Generate prefetch statements.
        if (cfg_.hw() >= ngen::HW::XeHPC) {
            for (auto &t : post_op_tensors_) {
                if (!t.needs_load()) continue;
                if (t.do_preload()) continue;
                stmt_ = stmt_.append(t.build_prefetch_stmt());
            }
        }

        // Generate f32 convert statements.
        for (auto &t : post_op_tensors_) {
            if (!t.do_preload()) continue;
            if (!t.needs_f32_convert()) continue;
            stmt_ = stmt_.append(t.build_convert_stmt());
        }

        // Initialize buffers for output post-op tensors.
        for (auto &t : post_op_tensors_) {
            if (!t.needs_store()) continue;
            t.init_output_buffer(base_tile);
        }

        // Generate zero-out statements for output post-op tensors.
        for (auto &t : post_op_tensors_) {
            if (!t.needs_store()) continue;
            stmt_ = stmt_.append(t.build_zero_out_stmt());
        }

        // Iterate by tiles and apply post-ops.
        c_mem_view_.for_each_tile(
                base_tile, [&](const std::vector<dim_t> &start) {
                    tensor_t tile(base_tile.dims(), start);
                    auto c_tile_layout = c_reg_layout.map(tile);
                    build_tile(tile, c_tile_layout, c_reg_buf);
                });

        // TODO: Generalize the condition. Iterate through output tensor masks
        // and ensure C is distributed accordingly in thread group.
        bool use_slm_reduction = (tg_grid_.dim(1) > 1);

        // Generate reduce and store statements for output post-op tensors.
        stmt_t thr_reduce_stmt;
        stmt_t slm_store_stmt;
        stmt_t slm_load_stmt;
        stmt_t mem_store_stmt;
        for (auto &t : post_op_tensors_) {
            if (!t.needs_store()) continue;

            thr_reduce_stmt = thr_reduce_stmt.append(t.build_reduce_stmt());
            if (use_slm_reduction) {
                auto store_stmt = t.build_slm_store_stmt(tg_grid_);
                auto load_stmt = t.build_slm_load_stmt();
                slm_store_stmt = slm_store_stmt.append(store_stmt);
                slm_load_stmt = slm_load_stmt.append(load_stmt);
            }
            mem_store_stmt = mem_store_stmt.append(t.build_store_stmt());
        }

        stmt_ = stmt_.append(thr_reduce_stmt);
        if (!slm_store_stmt.is_empty()) {
            stmt_ = stmt_.append(funcs::barrier());
            stmt_ = stmt_.append(slm_store_stmt);
            stmt_ = stmt_.append(funcs::barrier());
            stmt_ = stmt_.append(slm_load_stmt);
        }

        stmt_ = stmt_.append(mem_store_stmt);

        // Generate alloc statements for post-op tensors.
        std::vector<stmt_t> allocs;
        for (auto &t : post_op_tensors_) {
            auto t_allocs = t.allocs();
            allocs.insert(allocs.end(), t_allocs.begin(), t_allocs.end());
        }
        stmt_ = jit::inject_alloc_stmts(stmt_, allocs, /*put_innermost=*/true);
    }

    // Builds statements for a tile iterating through all post-ops.
    void build_tile(const tensor_t &tile, const layout_t &c_tile_layout,
            const expr_t &c_reg_buf) {
        auto c_mem_tile_view = c_mem_view_.create_sub_view(tile);
        auto tmp_reg_buf = make_c_tmp_buffer();

        bool create_zero_pad_builder = restore_zero_padding_;
        for (auto &t : post_op_tensors_) {
            if (t.needs_masked_update()) {
                create_zero_pad_builder = true;
                break;
            }
        }
        if (create_zero_pad_builder) {
            zero_pad_builder_ = zero_pad_builder_t(
                    cset_, post_op_ctx_.cp_view(), c_mem_tile_view);
        }

        // S_y -> GMEM.
        auto send_op = cfg_.do_atomic_update ? send_op_t::atomic_fadd
                                             : send_op_t::store;
        auto send_hint = get_send_hint(send_op, abc_kind_t::c, c_mem_tile_view,
                cfg_.use_2d_send, cfg_.simd_size(), gemm_schedule_);
        auto r2g = make_access_builder(cfg_.hw(), ir_ctx_, cset_,
                c_mem_tile_view, c_mem_buf_, tmp_reg_buf, send_op,
                send_address_t::a64, send_hint);

        // Initialize C stages.
        std::vector<c_stage_t> c_stages;

        auto c_f32_layout = r2g.reg_layout().retype(type_t::f32()).make_dense();
        bool with_post_ops = !post_op_builders_.empty();
        int npost_ops = int(post_op_builders_.size());

        int c_f32_stage_idx = -1;
        int c_zero_pad_stage_idx = -1;

        c_stages.emplace_back(c_tile_layout, c_reg_buf); // M_x
        if (with_post_ops) {
            c_f32_stage_idx = int(c_stages.size());
            c_stages.emplace_back(c_f32_layout, make_c_tmp_buffer()); // R_f32
        }
        if (restore_zero_padding_) {
            c_zero_pad_stage_idx = int(c_stages.size());
            c_stages.emplace_back(c_f32_layout, make_c_tmp_buffer()); // Z_f32
        }
        c_stages.emplace_back(r2g.reg_layout(), tmp_reg_buf, r2g.stmt()); // S_y

        int nstages = int(c_stages.size());
        bool is_dpasw = (cfg_.fma_kind == fma_kind_t::dpasw);

        // Generate reorders between C stages if needed.
        for (int i = 0; i < nstages; i++) {
            auto *next_stage = (i + 1 < nstages ? &c_stages[i + 1] : nullptr);
            // Always perform reorder when dpasw is used. This is to ensure
            // that C is properly restored and permuted after dpasw.
            c_stages[i].set_next(cfg_.hw(), ir_ctx_, next_stage,
                    /*force_reorder=*/i == 0 && is_dpasw);
        }

        // Restore zero padding if needed.
        if (c_zero_pad_stage_idx != -1) {
            auto &s = c_stages[c_zero_pad_stage_idx];
            s.prepend_stmt(zero_pad_builder_.build_stmt(s.layout, s.buf));
        }

        // Create sub-tensors for post-ops.
        std::vector<post_op_tensor_t> sub_po_tensors;
        for (auto &t : post_op_tensors_)
            sub_po_tensors.push_back(t.create_sub_tensor(tile));

        // Set C tensor layout and buffer to use for post-ops.
        if (c_f32_stage_idx != -1) {
            auto &s = c_stages[c_f32_stage_idx];
            sub_po_tensors[c_po_idx_].set_reg_layout(s.layout);
            sub_po_tensors[c_po_idx_].set_reg_buf(s.buf);
        }

        stmt_t tile_stmt;

        // Add C stage statements and post-op statements.
        for (int i = 0; i < nstages; i++) {
            if (with_post_ops && i == c_f32_stage_idx) {
                // Emit post-ops in blocks to reduce GRF consumption.
                for (int j = 0; j < npost_ops; j += post_op_blk_) {
                    int k_beg = j;
                    int k_end = std::min(npost_ops, j + post_op_blk_);
                    auto blk_stmt = build_post_op_block_stmt(
                            tile, sub_po_tensors, k_beg, k_end);
                    tile_stmt = tile_stmt.append(blk_stmt);
                }
            }
            tile_stmt = tile_stmt.append(c_stages[i].stmt);
        }

        // Generate alloc statements for C stage buffers.
        object_set_t<expr_t> seen;
        for (int i = 0; i < nstages; i++) {
            auto &s = c_stages[i];
            auto &buf = s.buf_base();
            auto ret = seen.insert(buf);
            if (i == 0 || !ret.second) continue;
            tile_stmt = alloc_t::make(
                    buf, s.buf_size(), alloc_kind_t::grf, tile_stmt);
        }

        stmt_ = stmt_.append(tile_stmt);
    }

    stmt_t build_post_op_block_stmt(const tensor_t &tile,
            std::vector<post_op_tensor_t> &sub_po_tensors, int po_beg,
            int po_end) const {
        // Collect post-op inputs/outputs.
        object_map_t<expr_t, post_op_tensor_t *> args;
        for (int i = po_beg; i < po_end; i++) {
            auto &po_builder = post_op_builders_[i];
            for (auto &t : sub_po_tensors) {
                if (po_builder.post_op().uses(t.op_var())) {
                    args.insert({t.op_var(), &t});
                }
            }
        }

        // Generate load and convert statements for the post-op.
        stmt_t load_stmt;
        stmt_t convert_stmt;
        for (auto &kv : args) {
            auto &t = *kv.second;
            if (!t.needs_load()) continue;
            if (t.do_preload()) continue;
            load_stmt = load_stmt.append(t.build_load_stmt());
            if (t.needs_f32_convert()) {
                convert_stmt = convert_stmt.append(t.build_convert_stmt());
            }
        }

        stmt_t stmt;
        stmt = stmt.append(load_stmt);
        stmt = stmt.append(convert_stmt);

        for (int i = po_beg; i < po_end; i++) {
            auto &po_builder = post_op_builders_[i];
            auto po_stmt
                    = po_builder.build_tile_stmt(tile, args, zero_pad_builder_);
            stmt = stmt.append(po_stmt);
        }

        // Generate alloc statements for post-op tensors.
        std::vector<stmt_t> allocs;
        for (auto &kv : args) {
            auto &t = *kv.second;
            if (!t.needs_load()) continue;
            if (t.do_preload()) continue;
            auto t_allocs = t.allocs();
            allocs.insert(allocs.end(), t_allocs.begin(), t_allocs.end());
        }
        stmt = jit::inject_alloc_stmts(stmt, allocs);

        return stmt;
    }

    const conv_config_t &cfg_;
    ir_context_t &ir_ctx_;
    const constraint_set_t &cset_;
    const gemm_schedule_t &gemm_schedule_;
    const post_op_context_t &post_op_ctx_;

    // C view in global memory.
    view_t c_mem_view_;
    expr_t c_mem_buf_;

    // C layout after the main loop.
    layout_t c_reg_layout_;
    expr_t c_reg_buf_;

    const grid_info_t &tg_grid_;

    bool restore_zero_padding_;

    zero_pad_builder_t zero_pad_builder_;

    // Tile size in bytes. The tile data type is:
    // - the destination data type without post-ops
    // - f32 with post-ops
    int tile_size_;
    int preload_max_size_;
    int post_op_blk_;

    std::vector<post_op_builder_t> post_op_builders_;
    std::vector<post_op_tensor_t> post_op_tensors_;
    int c_po_idx_ = -1;

    object_map_t<expr_t, int> buf_sizes_;

    stmt_t stmt_;
};

epilogue_builder_t create_epilogue_builder(const conv_config_t &cfg,
        ir_context_t &ir_ctx, const constraint_set_t &cset,
        const gemm_schedule_t &gemm_schedule,
        const post_op_context_t &post_op_ctx, const tensor_t &thr_tile,
        const view_t &c_mem_view, const layout_t &c_reg_layout,
        const expr_t &c_mem_buf, const expr_t &c_reg_buf) {
    // Tile size in bytes. All post-ops are applied to a single tile, then to
    // the next tile, etc.
    int tile_size = 512;
    // Max size of post-op tensor buffers to preload and reuse for all tiles.
    int preload_max_size = 512;
    // Block size to apply post-ops within tile. A post-op may have associated
    // loads/conversions, larger block size helps to have more latency hiding
    // across multiple post-ops.
    int post_op_blk = 8;

    int bufs = 0;
    for (auto &t : post_op_ctx.post_op_tensor_infos()) {
        if (t.is_input() && t.buf().type().is_ptr()) bufs++;
    }

    // Reduce GRF usage when there are too many post-op buffers to load.
    if (bufs > 8) {
        tile_size = 128;
    } else if (bufs > 4) {
        tile_size = 256;
    }

    ir_trace() << "Creating epilogue with parameters"
               << ": tile_size = " << tile_size
               << ", preload_max_size = " << preload_max_size
               << ", post_op_blk = " << post_op_blk << std::endl;
    epilogue_builder_t builder(cfg, ir_ctx, cset, gemm_schedule, post_op_ctx,
            thr_tile, c_mem_view, c_reg_layout, c_mem_buf, c_reg_buf, tile_size,
            preload_max_size, post_op_blk);
    return builder;
}

class multiply_builder_t {
public:
    multiply_builder_t() = default;

    multiply_builder_t(const conv_config_t &cfg,
            const bmnk_mapper_t &bmnk_mapper, const view_t &a_view,
            const view_t &b_view, const expr_t &a_buf, const expr_t &b_buf,
            const expr_t &c_buf)
        : hw_(cfg.hw())
        , simd_size_(cfg.simd_size())
        , bmnk_mapper_(bmnk_mapper)
        , a_view_(a_view)
        , b_view_(b_view)
        , a_buf_(a_buf)
        , b_buf_(b_buf)
        , c_buf_(c_buf) {
        switch (cfg.fma_kind) {
            case fma_kind_t::dp4a:
            case fma_kind_t::dpas:
            case fma_kind_t::dpasw:
                if (try_build_dpas()) return;
                break;
            case fma_kind_t::mad:
                if (try_build_mad()) return;
                break;
            default: ir_error_not_expected() << "Unknown FMA kind.";
        }

        ir_error_not_expected()
                << "Can't decompose into multiplication instructions. A view: "
                << a_view << ". B view: " << b_view;
    }

    const stmt_t &stmt() const { return stmt_; }

    const layout_t &c_layout() const { return c_layout_; }

    bool do_transpose() const { return do_transpose_; }

    std::string str() const {
        std::ostringstream oss;
        oss << "A view:    " << a_view_ << std::endl;
        oss << "B view:    " << b_view_ << std::endl;
        oss << "C layout:  " << c_layout_ << std::endl;
        oss << "Statement: " << std::endl << stmt_;
        return oss.str();
    }

private:
    struct loop_info_t {
        loop_info_t() = default;

        loop_info_t(const expr_t &var, bmnk_kind_t bmnk_kind, int dim)
            : var(var), bmnk_kind(bmnk_kind), dim(dim) {}

        expr_t var;
        bmnk_kind_t bmnk_kind;

        int dim;
        int a_idx = -1;
        int b_idx = -1;
        int c_idx = -1;
        int block = 1;
    };

    bool try_build_dpas() {
        ir_assert(a_view_.can_convert_to_vlayout())
                << "Views are not supported with dpas/dpasw.";
        ir_assert(b_view_.can_convert_to_vlayout())
                << "Views are not supported with dpas/dpasw.";

        auto a_layout = a_view_.create_vlayout();
        auto b_layout = b_view_.create_vlayout();

        bmnk_block_mapper_t from_bmnk_mapper(bmnk_mapper_);
        from_bmnk_mapper.push_blocks(abc_kind_t::a, a_layout.blocks());
        from_bmnk_mapper.push_blocks(abc_kind_t::b, b_layout.blocks());

        // Convert to MNK layouts.
        a_layout = bmnk_mapper_.map_to_bmnk(
                abc_kind_t::a, {bmnk_kind_t::m, bmnk_kind_t::k}, a_layout);
        b_layout = bmnk_mapper_.map_to_bmnk(
                abc_kind_t::b, {bmnk_kind_t::k, bmnk_kind_t::n}, b_layout);

        multiply_desc_t desc(a_layout, b_layout, /*force_c_upconvert=*/true);
        if (!dpas_t::matches_types(
                    hw_, desc.a_type(), desc.b_type(), desc.c_type()))
            return false;

        int sdepth = 8;
        int rcount = std::min(utils::rnd_up_pow2(desc.n()), 8);
        auto _dpas = dpas_t::make(/*is_dpasw=*/false, simd_size_, sdepth,
                rcount, desc.c_type(), desc.a_type(), desc.b_type());
        if (_dpas.as<dpas_t>().matches(desc)) {
            build_dpas(from_bmnk_mapper, _dpas.as<dpas_t>(), desc);
            return true;
        }

        // Try to transpose and flip: C += A * B -> C^T = B^T * A^T.
        rcount = std::min(utils::rnd_up_pow2(desc.m()), 8);
        desc = multiply_desc_t(
                b_layout.transpose(), a_layout.transpose(), true);
        _dpas = dpas_t::make(/*is_dpasw=*/false, /*exec_size=*/simd_size_,
                sdepth, rcount, desc.c_type(), desc.a_type(), desc.b_type());

        if (_dpas.as<dpas_t>().matches(desc)) {
            do_transpose_ = true;
            build_dpas(from_bmnk_mapper, _dpas.as<dpas_t>(), desc);
            return true;
        }
        return false;
    }

    void build_dpas(const bmnk_block_mapper_t &from_bmnk_mapper,
            const dpas_t &dpas, const multiply_desc_t &desc) {
        int m_blk = dpas.exec_size;
        int n_blk = dpas.rcount;
        int k_blk = dpas.sdepth * 4 / dpas.src1_type.size();

        c_layout_ = compute_dpas_c_layout(m_blk, n_blk, dpas.c_layout(), desc);

        expr_t a_buf = a_buf_;
        expr_t b_buf = b_buf_;
        if (do_transpose_) std::swap(a_buf, b_buf);

        for (int i_k = 0; i_k < desc.k(); i_k += k_blk) {
            for (int i_m = 0; i_m < desc.m(); i_m += m_blk) {
                for (int i_n = 0; i_n < desc.n(); i_n += n_blk) {
                    std::vector<int> a_args = {i_m, i_k};
                    std::vector<int> b_args = {i_k, i_n};
                    std::vector<int> c_args = {i_m, i_n};
                    auto a = a_buf[desc.a_layout()(a_args)
                            * desc.a_type().size()];
                    auto b = b_buf[desc.b_layout()(b_args)
                            * desc.b_type().size()];
                    auto c = c_buf_[c_layout_(c_args) * desc.c_type().size()];
                    stmt_ = stmt_.append(dpas(c, c, a, b));
                }
            }
        }

        // Transpose C layout back if needed.
        if (do_transpose_) c_layout_ = c_layout_.transpose();

        // Convert C layout back to problem notation.
        c_layout_ = from_bmnk_mapper.map_from_bmnk(
                abc_kind_t::c, {bmnk_kind_t::m, bmnk_kind_t::n}, c_layout_);
    }

    static layout_t compute_dpas_c_layout(int m_blk, int n_blk,
            const layout_t &blk_layout, const multiply_desc_t &desc) {
        auto c_layout = blk_layout;
        c_layout = c_layout.add_outer_block(1, desc.n() / n_blk);
        c_layout = c_layout.add_outer_block(0, desc.m() / m_blk);
        return c_layout;
    }

    bool try_build_mad() {
        auto loops = create_loop_nest();

        if (try_build_mad_kmn_block_by_n(loops)) return true;
        if (try_build_mad_kmn_block_by_b(loops)) return true;

        return false;
    }

    std::vector<loop_info_t> create_loop_nest() const {
        object_map_t<expr_t, loop_info_t> loops;
        for (auto *view : {&a_view_, &b_view_}) {
            abc_kind_t abc_kind
                    = (view == &a_view_ ? abc_kind_t::a : abc_kind_t::b);
            for (int i = 0; i < view->nvdims(); i++) {
                auto &var = bmnk_mapper_.var(abc_kind, i);
                int dim = int(view->vdims()[i]);
                if (dim == 1) continue;

                if (loops.count(var) > 0) continue;
                loops[var] = loop_info_t(var, bmnk_mapper_.bmnk_kind(var), dim);
            }
        }

        std::vector<loop_info_t> ret;
        for (auto &kv : loops) {
            auto &loop = kv.second;
            loop.a_idx = bmnk_mapper_.dim_idx(abc_kind_t::a, loop.var);
            loop.b_idx = bmnk_mapper_.dim_idx(abc_kind_t::b, loop.var);
            loop.c_idx = bmnk_mapper_.dim_idx(abc_kind_t::c, loop.var);
            ret.push_back(kv.second);
        }
        return ret;
    }

    // Order of loops: BKMN, block by N.
    bool try_build_mad_kmn_block_by_n(std::vector<loop_info_t> &_loops) {
        return try_build_mad_impl(_loops,
                {bmnk_kind_t::b, bmnk_kind_t::k, bmnk_kind_t::m,
                        bmnk_kind_t::n},
                bmnk_kind_t::n);
    }

    // Order of loops: BKMN, block by B.
    bool try_build_mad_kmn_block_by_b(std::vector<loop_info_t> &_loops) {
        return try_build_mad_impl(_loops,
                {bmnk_kind_t::b, bmnk_kind_t::k, bmnk_kind_t::m,
                        bmnk_kind_t::n},
                bmnk_kind_t::b);
    }

    bool try_build_mad_impl(std::vector<loop_info_t> &_loops,
            const std::vector<bmnk_kind_t> &loop_order,
            bmnk_kind_t block_bmnk_kind) {
        auto loops = _loops;
        int nloops = int(loops.size());
        std::sort(loops.begin(), loops.end(),
                [&](const loop_info_t &a, const loop_info_t &b) {
                    int a_key = ir_utils::find_index(loop_order, a.bmnk_kind);
                    int b_key = ir_utils::find_index(loop_order, b.bmnk_kind);
                    ir_assert(a_key != -1);
                    ir_assert(b_key != -1);
                    return a_key < b_key;
                });

        int block_idx = -1;
        for (int i = 0; i < nloops; i++) {
            auto &l = loops[i];
            if (l.bmnk_kind == block_bmnk_kind) {
                ir_assert(block_idx == -1) << "Can't block 2+ dimensions.";
                block_idx = i;
            }
        }

        // Couldn't find N dimension, try different blocking scheme.
        if (block_idx == -1) return false;

        auto &block_loop = loops[block_idx];

        int block = simd_size_;
        while (block >= 1) {
            if (block_loop.dim % block == 0) break;
            block /= 2;
        }

        ir_assert(block >= 1) << "Invalid block size.";
        block_loop.block = block;

        int a_stride = 0;
        int b_stride = 0;

        // Ensure that A tile is dense.
        if (block_loop.a_idx != -1) {
            std::vector<dim_t> tile_dims(a_view_.nvdims(), 1);
            tile_dims[block_loop.a_idx] = block;
            auto layout = a_view_.create_pseudo_vlayout();
            auto tile = layout.map(tensor_t(tile_dims));
            if (!is_1d_strided(tile)) return false;
            a_stride = tile.blocks()[0].stride;
        }

        // Ensure that B tile is dense.
        if (block_loop.b_idx != -1) {
            std::vector<dim_t> tile_dims(b_view_.nvdims(), 1);
            tile_dims[block_loop.b_idx] = block;
            auto layout = b_view_.create_pseudo_vlayout();
            auto tile = layout.map(tensor_t(tile_dims));
            if (!is_1d_strided(tile)) return false;
            b_stride = tile.blocks()[0].stride;
        }

        build_mad(loops, block_loop, a_stride, b_stride);
        return true;
    }

    static bool is_1d_strided(const layout_t &layout) {
        auto &blocks = layout.blocks();
        if (blocks.size() > 1) return false;
        return true;
    }

    void build_mad(const std::vector<loop_info_t> &loops,
            const loop_info_t &block_loop, int a_stride, int b_stride) {
        ir_assert(utils::one_of(
                block_loop.bmnk_kind, bmnk_kind_t::b, bmnk_kind_t::n))
                << "Unsupported blocking (expected blocking by B or N).";

        auto &a_type = a_view_.type();
        auto &b_type = b_view_.type();
        auto c_type = multiply_desc_t::get_c_type(a_type, b_type,
                /*force_c_upconvert=*/false);

        int block = block_loop.block;
        auto _mad = mad_t::make(
                c_type, block, a_type, a_stride, b_type, b_stride);
        auto &mad = _mad.as<mad_t>();

        c_layout_ = compute_mad_c_layout(c_type, loops, block_loop);

        int nloops = int(loops.size());
        std::vector<int> bounds(loops.size());
        for (int i = 0; i < nloops; i++) {
            bounds[i] = loops[i].dim / loops[i].block;
        }
        std::vector<int> a_idx(a_view_.nvdims());
        std::vector<int> b_idx(b_view_.nvdims());
        std::vector<int> c_idx(c_layout_.ndims());
        ir_utils::for_each(bounds, [&](const std::vector<int> &idx) {
            for (int i = 0; i < nloops; i++) {
                int full_idx = idx[i] * loops[i].block;
                auto &loop = loops[i];
                if (loop.a_idx != -1) a_idx[loop.a_idx] = full_idx;
                if (loop.b_idx != -1) b_idx[loop.b_idx] = full_idx;
                if (loop.c_idx != -1) c_idx[loop.c_idx] = full_idx;
            }
            int a_off = a_view_(a_idx) * a_type.size();
            int b_off = b_view_(b_idx) * b_type.size();
            int c_off = c_layout_(c_idx) * c_type.size();
            stmt_ = stmt_.append(mad(c_buf_[c_off], c_buf_[c_off],
                    a_buf_[a_off], b_buf_[b_off]));
        });
    }

    layout_t compute_mad_c_layout(const type_t &c_type,
            const std::vector<loop_info_t> &loops,
            const loop_info_t &block_loop) const {
        layout_t c_layout(c_type, bmnk_mapper_.ndims(abc_kind_t::c), 0, {});

        int c_dim_idx = bmnk_mapper_.dim_idx(abc_kind_t::c, block_loop.var);
        c_layout = c_layout.add_outer_block(c_dim_idx, block_loop.block);

        for (size_t i = 0; i < loops.size(); i++) {
            if (loops[i].bmnk_kind == bmnk_kind_t::k) continue;
            int dim_idx = bmnk_mapper_.dim_idx(abc_kind_t::c, loops[i].var);
            int bound = loops[i].dim / loops[i].block;
            c_layout = c_layout.add_outer_block(dim_idx, bound);
        }
        return c_layout;
    }

    ngen::HW hw_;
    int simd_size_;
    bmnk_mapper_t bmnk_mapper_;

    bool do_transpose_ = false;

    view_t a_view_;
    view_t b_view_;
    layout_t c_layout_;

    expr_t a_buf_;
    expr_t b_buf_;
    expr_t c_buf_;

    stmt_t stmt_;
};

layout_t get_fma_friendly_layout(abc_kind_t abc_kind, int simd_size,
        const layout_t &bmnk_layout, const type_t &a_type,
        const type_t &b_type) {
    bool is_a = (abc_kind == abc_kind_t::a);
    int mn_idx = (is_a ? 0 : 1);
    int k_idx = (is_a ? 1 : 0);

    dim_t mn_blk = bmnk_layout.dim(mn_idx);
    dim_t k_blk = bmnk_layout.dim(k_idx);

    // Cannot calculate correct r_count when !is_a, but rcount is effectively
    // ignored in that case as rcount mainly effects b_layout.
    int rcount = is_a && mn_blk < 8 ? utils::rnd_up_pow2(mn_blk) : 8;
    auto _dpas = dpas_t::make(/*is_dpasw=*/false, simd_size, /*sdepth=*/8,
            rcount, type_t::undef(), b_type, a_type);
    auto &dpas = _dpas.as<dpas_t>();

    auto dpas_layout = (is_a ? dpas.b_layout() : dpas.a_layout());
    dpas_layout = dpas_layout.transpose();

    ir_assert(k_blk % dpas_layout.dim(k_idx) == 0);

    dim_t dpas_mn_blk = dpas_layout.dim(mn_idx);
    dim_t dpas_k_blk = dpas_layout.dim(k_idx);
    dim_t k_outer = ir_utils::safe_divide(k_blk, dpas_k_blk);
    dim_t mn_outer = ir_utils::safe_divide(mn_blk, dpas_mn_blk);
    dpas_layout = dpas_layout.add_outer_block(k_idx, k_outer);
    dpas_layout = dpas_layout.add_outer_block(mn_idx, mn_outer);

    return dpas_layout;
}

layout_t convert_to_fma_friendly_type(const conv_config_t &cfg,
        abc_kind_t abc_kind, const layout_t &layout, const type_t &a_type,
        const type_t &b_type, bool *changed = nullptr) {
    if (changed) *changed = false;
    if (cfg.fma_kind != fma_kind_t::mad) return layout;

    // mad with s8/u8 is not supported, promote to strided s16.
    if (a_type.is_x8() && b_type.is_x8()) {
        if (changed) *changed = true;
        return layout.retype(type_t::s16()).make_strided(2);
    }

    // bf16 mixed mode mad requires src2 to be f32.
    if (abc_kind == abc_kind_t::b && a_type.is_bf16()) {
        if (changed) *changed = true;
        return layout.retype(type_t::f32()).make_dense();
    }

    // bf16 mixed mode mad requires src1 to be packed, src1 is broadcasted for
    // non-depthwise cases so it needs to be converted to f32.
    if (abc_kind == abc_kind_t::a && a_type.is_bf16() && !cfg.is_dw) {
        if (changed) *changed = true;
        return layout.retype(type_t::f32()).make_dense();
    }

    // Ensure the layout is dense to align regioning.
    if (!layout.is_dense()) {
        if (changed) *changed = true;
        return layout.make_dense();
    }

    return layout;
}

layout_t convert_to_fma_friendly_layout(const conv_config_t &cfg,
        abc_kind_t abc_kind, const bmnk_mapper_t &bmnk_mapper,
        const layout_t &layout, const type_t &a_type, const type_t &b_type,
        bool is_slm, bool *changed = nullptr) {
    if (changed) *changed = false;
    if (!cfg.allow_grf_reorder) return layout;

    // GRF reorder is only supported with dpas/dpasw.
    if (!cfg.is_dp_fma()) {
        if (is_slm) return layout;
        // mad may require type conversion, supported for GRF layouts only.
        return convert_to_fma_friendly_type(
                cfg, abc_kind, layout, a_type, b_type, changed);
    }

    std::vector<bmnk_kind_t> bmnk_kinds;
    if (abc_kind == abc_kind_t::a) {
        bmnk_kinds.push_back(bmnk_kind_t::m);
        bmnk_kinds.push_back(bmnk_kind_t::k);
    } else {
        bmnk_kinds.push_back(bmnk_kind_t::k);
        bmnk_kinds.push_back(bmnk_kind_t::n);
    }

    auto bmnk_layout = bmnk_mapper.map_to_bmnk(abc_kind, bmnk_kinds, layout);

    auto dpas_layout = get_fma_friendly_layout(
            abc_kind, cfg.simd_size(), bmnk_layout, a_type, b_type);
    if (dpas_layout == bmnk_layout) return layout;

    if (changed) *changed = true;

    bmnk_block_mapper_t from_bmnk_mapper(bmnk_mapper);
    from_bmnk_mapper.push_blocks(abc_kind, layout.blocks());

    auto fma_layout
            = from_bmnk_mapper.map_from_bmnk(abc_kind, bmnk_kinds, dpas_layout);
    fma_layout = fma_layout.make_dense();
    return fma_layout;
}

class b_reduce_context_t {
public:
    b_reduce_context_t(const conv_config_t &cfg)
        : cfg_(cfg), reduce_condition_(true) {
        if (cfg.do_b_reduction) b_reduced_reg_buf_ = make_buffer("b_reduced");
    }

    // Setters for B reduced memory buffer/view.
    void set_b_reduced_mem_buf(const expr_t &buf) { b_reduced_mem_buf_ = buf; }
    void set_b_reduced_view(const view_t &v) { b_reduced_view_ = v; }

    // Sets the condition to update B reduced output. Reduction is done across
    // K for B (KxN tensor) so M dimension should be checked before the update.
    void set_reduce_condition(const expr_t &cond) { reduce_condition_ = cond; }

    // Global memory buffer.
    const expr_t &b_reduced_mem_buf() const { return b_reduced_mem_buf_; }

    // Register buffer.
    const expr_t &b_reduced_reg_buf() const { return b_reduced_reg_buf_; }
    int b_reduced_size() const { return b_reduced_size_; }

    // Memory view.
    const view_t &b_reduced_thr_view() const { return b_reduced_thr_view_; }

    // Register layout.
    const layout_t &b_reduced_reg_layout() const {
        return b_reduced_reg_layout_;
    }

    void init_reduced_thr_view(
            const tensor_t &b_thr_tile, const expr_t &cond = expr_t()) {
        ir_assert(b_reduced_thr_view_.is_empty()) << "Can't initialize twice.";

        auto b_reduced_thr_tile = b_to_b_reduced_tile(b_thr_tile);
        b_reduced_thr_view_
                = b_reduced_view_.create_sub_view(b_reduced_thr_tile);
        b_reduced_reg_layout_ = b_reduced_thr_view_.create_dense_vlayout();
        b_reduced_size_ = b_reduced_reg_layout_.size();
        b_reduced_size_ = utils::rnd_up(b_reduced_size_, cfg_.grf_size());

        if (!cond.is_empty()) reduce_condition_ &= cond;
    }

    stmt_t create_reduce_stmt(const layout_t &b_layout, const expr_t &b_buf,
            const tensor_t &sub_tile = tensor_t()) {
        auto reduction_stmt
                = jit::create_reduce_stmt(b_layout, b_reduced_reg_layout_,
                        b_buf, b_reduced_reg_buf_, sub_tile, reduction_mask_);
        return reduction_stmt;
    }

    stmt_t create_store_stmt(
            ir_context_t &ir_ctx, const constraint_set_t &cset) const {
        auto r2g = make_access_builder(cfg_.hw(), ir_ctx, cset,
                b_reduced_thr_view_, b_reduced_mem_buf_, b_reduced_reg_buf_,
                send_op_t::atomic_fadd, send_address_t::a64);
        // TODO: Check that layouts match.
        auto ret = r2g.stmt();
        if (!reduce_condition_.is_empty()) {
            ret = if_t::make(reduce_condition_, ret);
        }
        return ret;
    }

private:
    tensor_t b_to_b_reduced_tile(const tensor_t &b_tile) const {
        std::vector<dim_t> dims;
        std::vector<expr_t> start;
        for (int i = 0; i < b_tile.ndims(); i++) {
            if ((reduction_mask_ & (1 << i)) != 0) {
                dims.push_back(b_tile(i));
                start.push_back(b_tile.start(i));
            }
        }
        return tensor_t(dims, start);
    }

    const conv_config_t &cfg_;

    expr_t reduce_condition_;

    expr_t b_reduced_mem_buf_;
    expr_t b_reduced_reg_buf_;

    view_t b_reduced_view_;
    view_t b_reduced_thr_view_;

    layout_t b_reduced_reg_layout_;
    int b_reduced_size_ = 0;

    uint32_t reduction_mask_ = (1 << 1) | (1 << 2);
};

class load_multiply_builder_t {
public:
    load_multiply_builder_t(const conv_config_t &cfg, ir_context_t &ir_ctx,
            const constraint_set_t &cset, const gemm_schedule_t &gemm_schedule,
            b_reduce_context_t &b_reduce_ctx, const expr_t &ap_buf,
            const expr_t &a_slm_buf, const expr_t &bp_buf,
            const expr_t &b_slm_buf, const view_t &ap_x_view,
            const view_t &bp_x_view, const kernel_info_t &kernel_info)
        : cfg_(cfg)
        , ir_ctx_(ir_ctx)
        , cset_(cset)
        , gemm_schedule_(gemm_schedule)
        , b_reduce_ctx_(b_reduce_ctx)
        , ap_buf_(ap_buf)
        , a_slm_buf_(a_slm_buf)
        , bp_buf_(bp_buf)
        , b_slm_buf_(b_slm_buf)
        , kernel_info_(kernel_info) {
        ir_assert(cfg_.a_sub_tiles == 1 || cfg_.b_sub_tiles == 1)
                << "At most one tensor can be tiled.";

        ab_tmp_buf_ = make_buffer("ab_tmp");
        a_buf_ = make_buffer("a");
        b_buf_ = make_buffer("b");
        c_buf_ = make_buffer("c");

        // Views to multiply by a thread.
        a_thr_view_ = ap_x_view.create_sub_view(gemm_schedule_.a_thr_tile());
        b_thr_view_ = bp_x_view.create_sub_view(gemm_schedule_.b_thr_tile());

        // Initialize view for reduced B.
        if (cfg_.do_b_reduction && !cfg_.use_b_slm) {
            b_reduce_ctx_.init_reduced_thr_view(
                    gemm_schedule_.b_thr_tile(/*is_relative=*/false));
        }

        // TODO: Specify loops over sub-tiles in the schedule, use unrolling.
        // Sub-tile indices.
        a_idx_ = ir_ctx_.create_tmp_var(type_t::s32(), "a_idx");
        b_idx_ = ir_ctx_.create_tmp_var(type_t::s32(), "b_idx");

        // Sub-tile views.
        a_i_view_ = create_sub_tile_view(abc_kind_t::a, a_thr_view_,
                cfg_.a_sub_tiles, a_idx_, bmnk_kind_t::m, &a_i_outer_blocks_,
                a_i_tile_);
        b_j_view_ = create_sub_tile_view(abc_kind_t::b, b_thr_view_,
                cfg_.b_sub_tiles, b_idx_, bmnk_kind_t::n, &b_j_outer_blocks_,
                b_j_tile_);

        build();
    }

    const std::vector<stmt_t> &allocs() const { return allocs_; }

    const stmt_t &load_mul_stmt() const { return load_mul_stmt_; }

    const expr_t &c_buf() const { return c_buf_; }

    const layout_t &c_reg_layout() const { return c_reg_layout_; }

private:
    struct sub_tile_info_t {
        bool is_loaded = false;
        view_t reg_view;
        int reg_buf_size;
    };

    view_t create_sub_tile_view(abc_kind_t abc_kind, const view_t &thr_view,
            int sub_tiles, const expr_t &idx, bmnk_kind_t bmnk_kind,
            std::vector<block_t> *outer_blocks, tensor_t &sub_tile) const {
        auto &bmnk_mapper = gemm_schedule_.bmnk_mapper();
        auto layout = thr_view.create_pseudo_vlayout();
        dim_t mn_dim = 1;
        for (auto &b : layout.blocks()) {
            auto b_bmnk_kind = bmnk_mapper.bmnk_kind(abc_kind, b.dim_idx);
            if (b_bmnk_kind == bmnk_kind) mn_dim *= b.block;
        }

        std::vector<dim_t> sub_tile_dims(thr_view.nvdims(), 1);
        dim_t mn_sub_tile_dim = ir_utils::safe_divide(mn_dim, dim_t(sub_tiles));
        for (auto &b : layout.blocks()) {
            auto b_bmnk_kind = bmnk_mapper.bmnk_kind(abc_kind, b.dim_idx);
            if (b_bmnk_kind == bmnk_kind) {
                if (mn_sub_tile_dim == 1) continue;
                dim_t next_block;
                if (mn_sub_tile_dim % b.block == 0) {
                    next_block = b.block;
                } else {
                    ir_assert(b.block % mn_sub_tile_dim == 0);
                    next_block = mn_sub_tile_dim;
                }
                sub_tile_dims[b.dim_idx] *= next_block;
                mn_sub_tile_dim /= next_block;
            } else {
                sub_tile_dims[b.dim_idx] *= b.block;
            }
        }
        grid_info_t grid({sub_tiles}, {idx});
        sub_tile = layout.split(tensor_t(sub_tile_dims), grid, outer_blocks);
        return thr_view.create_sub_view(sub_tile);
    }

    const type_t &a_type() const { return a_i_view_.type(); }
    const type_t &b_type() const { return b_j_view_.type(); }

    void build() {
        a_sub_tiles_.resize(cfg_.a_sub_tiles);
        b_sub_tiles_.resize(cfg_.b_sub_tiles);
        for (int i = 0; i < cfg_.a_sub_tiles; i++) {
            for (int j = 0; j < cfg_.b_sub_tiles; j++) {
                build_sub_tile(i, j);
            }
        }

        if (tmp_buf_size_ > 0) {
            register_buffer(ab_tmp_buf_, tmp_buf_size_, alloc_kind_t::grf);
        }

        // C layout in problem notation.
        auto c_layout = c_sub_tile_layout_;

        // Add outer blocks coming from A/B sub-tiles.
        auto &bmnk_mapper = gemm_schedule_.bmnk_mapper();
        for (auto &b : a_i_outer_blocks_) {
            auto &var = bmnk_mapper.var(abc_kind_t::a, b.dim_idx);
            int c_dim_idx = bmnk_mapper.dim_idx(abc_kind_t::c, var);
            c_layout = c_layout.add_outer_block(c_dim_idx, b.block);
        }
        for (auto &b : b_j_outer_blocks_) {
            auto &var = bmnk_mapper.var(abc_kind_t::b, b.dim_idx);
            int c_dim_idx = bmnk_mapper.dim_idx(abc_kind_t::c, var);
            c_layout = c_layout.add_outer_block(c_dim_idx, b.block);
        }

        c_reg_layout_ = c_layout;
    }

    expr_t vector2expr(const std::vector<expr_t> &expr,
            object_eq_map_t<expr_t, expr_t> &vars) {
        const size_t mask = 0x8000;
        auto hash = [mask](const binary_op_t &b) -> size_t {
            return size_t(b.op_kind) | ((b.b.is<int_imm_t>()) ? mask : 0UL);
        };
        if (expr.empty()) return expr_t();
        // Can only vectorize if the element count is a power of 2
        ir_assert(math::is_pow2(expr.size())) << "Cannot vectorize.";

        std::unordered_map<size_t, size_t> kind;
        for (const expr_t &e : expr)
            if (const auto *bin = e.as_ptr<binary_op_t>()) kind[hash(*bin)]++;
        if (!kind.empty()) {
            using k_type = decltype(kind)::value_type;
            auto k = std::max_element(kind.begin(), kind.end(),
                    [](k_type &a, k_type &b) { return a.second < b.second; });
            const auto k_raw = op_kind_t(k->first & (mask - 1));
            std::vector<expr_t> a, b;
            for (const expr_t &e : expr) {
                const auto *bin = e.as_ptr<binary_op_t>();
                if (bin && (hash(*bin) == k->first)) {
                    a.emplace_back(bin->a);
                    b.emplace_back(bin->b);
                } else {
                    const int is_mul = (k_raw == op_kind_t::_mul);
                    ir_assert(is_mul || (k_raw == op_kind_t::_add));
                    a.emplace_back(e);
                    b.emplace_back(is_mul);
                }
            }
            auto a_new = vector2expr(a, vars);
            auto b_new = vector2expr(b, vars);
            return binary_op_t::make(k_raw, a_new, b_new);
        }

        size_t num_ints = 0;
        for (const expr_t &e : expr)
            num_ints += e.is<int_imm_t>();
        ir_assert((num_ints == 0) || (num_ints == expr.size()));
        if (num_ints == expr.size()) {
            auto offs = shuffle_t::make(expr);
            if (offs.as<shuffle_t>().is_broadcast()) return offs;
            if (vars.find(offs) == vars.end()) {
                auto var = ir_ctx_.create_tmp_var(
                        type_t::s32(int(num_ints)), "zp_mask");
                vars.emplace(offs, var);
            }
            return vars[offs];
        }

        size_t num_bools = 0;
        for (const expr_t &e : expr)
            num_bools += e.is<bool_imm_t>();
        ir_assert((num_bools == 0) || (num_bools == expr.size()));
        if (num_bools == expr.size()) return shuffle_t::make(expr);

        ir_assert(expr.front().is<var_t>());
        for (const expr_t &e : expr)
            ir_assert(e.is_same(expr.front()));
        return shuffle_t::make_broadcast(expr.front(), int(expr.size()));
    }

    stmt_t maybe_add_src_zps(
            const view_t &a_view, const view_t &b_view, bool is_transposed) {
        if (!cfg_.zp_cfg.do_src_compensation) return stmt_t();
        stmt_t retn;
        const int m_blk = cfg_.simd_size();
        const int ic = utils::rnd_up_pow2(cfg_.ic);
        const bool is_runtime = cfg_.zp_cfg.is_runtime_src_zero_points;
        const bool is_scalar = cfg_.zp_cfg.is_common_src_zero_point;
        ir_assert(cfg_.fma_kind != fma_kind_t::mad);

        type_t s_type = b_view.type();
        type_t d_type = (s_type.is_s8()) ? type_t::s32() : type_t::u32();
        ir_assert(s_type.is_x8());

        const int vd_size = m_blk * ((is_scalar) ? 2 : 4);
        auto src_zp = ir_ctx_.create_tmp_var(type_t::byte_ptr());
        auto acc = src_zp[(vd_size - m_blk) * d_type.size()];
        register_buffer(src_zp, vd_size * d_type.size(), alloc_kind_t::grf);

        auto tile = gemm_schedule_.a_thr_tile(/*is_relative=*/false);
        auto a_thr_view = gemm_schedule_.a_view().create_sub_view(tile);
        auto a_layout = gemm_schedule_.bmnk_mapper().map_to_bmnk(abc_kind_t::a,
                {bmnk_kind_t::m, bmnk_kind_t::k}, a_view.create_vlayout());
        auto b_layout = gemm_schedule_.bmnk_mapper().map_to_bmnk(abc_kind_t::b,
                {bmnk_kind_t::k, bmnk_kind_t::n}, b_view.create_vlayout());
        if (is_transposed) {
            a_layout = a_layout.transpose();
            b_layout = b_layout.transpose();
            std::swap(a_layout, b_layout);
        }
        multiply_desc_t desc(a_layout, b_layout, true);

        bool need_masks = (cfg_.kd * cfg_.kh * cfg_.kw > 1)
                || ((cfg_.pd + 1) * (cfg_.ph + 1) * (cfg_.pw + 1) > 1);
        bool scalar_masks
                = (tile.dims()[3] * tile.dims()[4] * tile.dims()[5] == 1);
        auto zp_mask = ir_ctx_.create_tmp_var(type_t::byte_ptr(), "zp_mask");
        auto var_mask = ir_ctx_.create_tmp_var(type_t::_bool(m_blk));

        if (need_masks) {
            std::vector<dim_t> a_dims(a_thr_view.vvars().size(), 1);
            std::vector<dim_t> c_dims(c_sub_tile_layout_.ndims(), 1);
            ir_assert((a_dims.size() > 2) && (c_dims.size() > 2));
            a_dims[2] = c_dims[2] = std::min(m_blk, ic);

            std::vector<int> c_to_a(a_dims.size());
            for (size_t i = 0; i < c_to_a.size(); i++) {
                const auto &c_vvars = gemm_schedule_.c_view().vvars();
                for (size_t j = 0; j < c_vvars.size(); j++) {
                    if (a_thr_view.vvars()[i].is_equal(c_vvars[j])) {
                        c_to_a[i] = int(j + 1);
                        break;
                    }
                }
            }
            auto mask_tensor = a_thr_view.create_mask_tensor(cset_);
            mask_tensor_t masks(layout_t(type_t::_bool(), 0,
                    std::vector<dim_t> {desc.n() * desc.m() / m_blk}));
            c_sub_tile_layout_.for_each_tile(
                    tensor_t(c_dims), [&](const std::vector<dim_t> &start) {
                        std::vector<dim_t> a_start;
                        std::transform(c_to_a.begin(), c_to_a.end(),
                                std::back_inserter(a_start),
                                [&](int i) { return (i) ? start[i - 1] : 0; });
                        tensor_t a_tensor(a_dims, a_start);
                        masks.set_mask(c_sub_tile_layout_.offset(start) / m_blk,
                                mask_tensor.map(a_tensor).to_expr(1));
                    });

            ir_assert(math::is_pow2(masks.elems()));
            int ntrue = 0;
            for (int n = 0; n < desc.n(); n++)
                ntrue += masks.mask(n).is_equal(expr_t(true));
            if (ntrue == desc.n()) {
                need_masks = false;
            } else if (!scalar_masks) {
                std::vector<expr_t> exprs;
                object_eq_map_t<expr_t, expr_t> vars;
                ir_assert((desc.n() % m_blk == 0) || (m_blk % desc.n() == 0));

                // Here we assume two important things:
                // - C has exactly one N block like 4c16f8c (where f is ow)
                // - The innermost block is by M and it matches the SIMD size

                for (int n = 0; n < desc.n(); n += m_blk) {
                    std::vector<expr_t> e;
                    int ntrue = 0, nfalse = 0;
                    for (int m = 0; m < m_blk; m++) {
                        e.emplace_back(masks.mask((n + m) % masks.elems()));
                        if (e[m].is<bool_imm_t>()) {
                            ((e[m].as<bool_imm_t>().value) ? ntrue : nfalse)++;
                        }
                    }
                    ir_assert((ntrue == 0) || (ntrue + nfalse == m_blk));
                    if ((ntrue == 0) && (nfalse > 0) && (nfalse < m_blk)) {
                        auto nb = *std::find_if(e.begin(), e.end(),
                                [](expr_t &x) { return !x.is<bool_imm_t>(); });
                        for (int m = 0; m < m_blk; m++) {
                            e[m] = (e[m].is<bool_imm_t>())
                                    ? (nb & expr_t(false))
                                    : (e[m] & expr_t(true));
                        }
                    }
                    exprs.emplace_back(vector2expr(e, vars));
                }

                const int sz = m_blk * type_t::s16().size();
                const int total = (scalar_masks) ? 1 : int(exprs.size());
                register_buffer(zp_mask, total * sz, alloc_kind_t::grf);
                for (int i = 0; i < total; i++) {
                    auto store = store_t::make(zp_mask, i * sz,
                            -cast_t::make(type_t::s16(m_blk), exprs[i]));
                    retn = retn.append(store);
                }
                for (auto &v : vars)
                    retn = let_t::make(v.second, v.first, retn);
            } else { // scalar_masks == true
                register_buffer(
                        zp_mask, type_t::s16().size(), alloc_kind_t::grf);
                auto expr = -cast_t::make(type_t::s16(), masks.mask(0));
                retn = retn.append(store_t::make(zp_mask, 0, expr));
            }
        }

        const std::vector<dim_t> dims
                = {m_blk * ((is_scalar || (ic <= m_blk)) ? 1 : 2)};
        const bool tiny_ic = (ic <= dims[0]);
        const bool sc_ic = is_scalar || tiny_ic;
        expr_t offs = (!sc_ic) ? a_thr_view.vstart(2) * d_type.size() : 0;
        if (is_runtime && !sc_ic && !cfg_.do_pipeline_unroll
                && (cfg_.slm_bufs > 1)) {
            auto buf = ir_ctx_.create_tmp_var(type_t::byte_ptr(), "zp_mask");
            register_buffer(buf, type_t::u32().size(), alloc_kind_t::grf);
            retn = retn.append(store_t::make(buf, 0, offs));
            offs = load_t::make(type_t::u32(), buf, 0);
        }
        for (int i = (is_runtime) ? 0 : std::numeric_limits<int>::max();
                i < dims[0] * ((!sc_ic) ? 2 : 1); i += dims[0]) {
            const int b = i * d_type.size();
            view_t zpv(layout_t(d_type, 0, dims));
            auto acc = make_access_builder(cfg_.hw(), ir_ctx_, cset_, zpv,
                    kernel_info_.find_arg("src_zero_points")[offs + b],
                    src_zp[b], send_op_t::load, send_address_t::a64);
            retn = retn.append(acc.stmt());
        }
        if (is_scalar) {
            expr_t expr = (!is_runtime)
                    ? (cfg_.zp_cfg.common_src_zero_point & 0xFF) * 0x01010101
                    : cast_t::make(type_t::s8(4),
                            shuffle_t::make_broadcast(
                                    load_t::make(s_type, src_zp, 0), 4));
            retn = retn.append(store_t::make(src_zp, 0, expr));
        } else {
            auto ic_s = std::to_string(std::min(32, ic)) + "a";
            retn = retn.append(
                    create_reorder_stmt(layout_t(s_type, 0, ic_s + "4b"),
                            layout_t(s_type, 0, "4b" + ic_s), src_zp, src_zp));
        }
        auto _dp4a = dpas_t::make(
                /*is_dpasw=*/false, m_blk, 1, 1, d_type, d_type, d_type);
        auto &dp4a = _dp4a.as<dpas_t>();
        stmt_t loop;

        auto word2bool = [m_blk](expr_t &expr, int off) {
            auto load = load_t::make(type_t::u16(), expr, off);
            return cast_t::make(type_t::_bool(m_blk), load);
        };
        for (int i_m = 0; i_m < desc.m(); i_m += m_blk) {
            for (int i_k = 0; i_k < ((!tiny_ic) ? 32 / 4 : cfg_.kw); i_k++) {
                auto wei = b_buf_
                        + (i_m * (32 / 4) + i_k * m_blk) * d_type.size();
                loop = loop.append(dp4a(acc, (i_k) ? acc : 0, wei,
                        src_zp[(!sc_ic) ? i_k * d_type.size() : 0]));
            }
            for (int i_n = 0; i_n < desc.n(); i_n++) {
                const int off_n = i_m / m_blk * desc.n() + i_n;
                auto dst = c_buf_ + off_n * m_blk * d_type.size();
                auto mask = (!need_masks)
                        ? expr_t()
                        : (scalar_masks) ? var_mask
                                         : word2bool(zp_mask, i_n * 2);
                type_t vd(d_type.kind(), m_blk);
                auto sub = binary_op_t::make(op_kind_t::_sub,
                        load_t::make(vd, dst, 0), load_t::make(vd, acc, 0));
                loop = loop.append(store_t::make(
                        dst, 0, sub, store_t::default_stride, mask));
            }
        }
        return retn.append((need_masks && scalar_masks)
                        ? let_t::make(var_mask, word2bool(zp_mask, 0), loop)
                        : loop);
    }

    void build_sub_tile(int i, int j) {
        bool is_first = (i == 0 && j == 0);

        stmt_t ab_s2r_load;
        stmt_t ab_g2r_load;
        load_sub_tile(abc_kind_t::a, i, ab_s2r_load, ab_g2r_load);
        load_sub_tile(abc_kind_t::b, j, ab_s2r_load, ab_g2r_load);

        load_mul_stmt_ = load_mul_stmt_.append(
                stmt_group_t::make(stmt_label_t::g2r_load(i + j), ab_g2r_load));
        load_mul_stmt_ = load_mul_stmt_.append(
                stmt_group_t::make(stmt_label_t::s2r_load(i + j), ab_s2r_load));

        auto &a_i_view = a_sub_tiles_[i].reg_view;
        auto &b_j_view = b_sub_tiles_[j].reg_view;

        // Multiply C_i_j += A_i x B_j in GEMM notation.
        multiply_builder_t mul_builder(cfg_, gemm_schedule_.bmnk_mapper(),
                a_i_view, b_j_view, a_buf_, b_buf_, c_buf_[c_buf_off_]);
        c_sub_tile_layout_ = mul_builder.c_layout();

        auto zero_pts = maybe_add_src_zps(
                a_i_view, b_j_view, mul_builder.do_transpose());

        c_buf_off_ += c_sub_tile_layout_.size();
        ir_trace() << "Multiply (" << i << ", " << j << "):\n"
                   << mul_builder.str() << zero_pts.str() << std::endl;

        load_mul_stmt_ = load_mul_stmt_.append(stmt_group_t::make(
                stmt_label_t::mul(i + j), mul_builder.stmt().append(zero_pts)));

        if (!is_first) {
            ir_assert(mul_builder.c_layout() == c_sub_tile_layout_)
                    << "Sub-tile layouts must be equal.";
            return;
        }

        register_buffer(
                a_buf_, a_sub_tiles_[i].reg_buf_size, alloc_kind_t::grf);
        register_buffer(
                b_buf_, b_sub_tiles_[j].reg_buf_size, alloc_kind_t::grf);
    }

    // Loads A_i or B_j sub-tile.
    void load_sub_tile(abc_kind_t abc_kind, int i, stmt_t &ab_s2r_load,
            stmt_t &ab_g2r_load) {
        bool is_a = (abc_kind == abc_kind_t::a);
        auto &info = (is_a ? a_sub_tiles_[i] : b_sub_tiles_[i]);
        if (info.is_loaded) return;

        auto &bmnk_mapper = gemm_schedule_.bmnk_mapper();

        auto &x_view = (is_a ? a_i_view_ : b_j_view_);
        auto &x_tile = (is_a ? a_i_tile_ : b_j_tile_);
        auto &x_idx = (is_a ? a_idx_ : b_idx_);

        auto view = x_view.substitute(x_idx, i);
        auto tile = x_tile.substitute(x_idx, i);

        bool use_x_slm = (is_a ? cfg_.use_a_slm : cfg_.use_b_slm);
        auto &x_slm_buf = (is_a ? a_slm_buf_ : b_slm_buf_);
        auto &x_gmem_buf = (is_a ? ap_buf_ : bp_buf_);
        auto &x_buf = (use_x_slm ? x_slm_buf : x_gmem_buf);
        auto &x_reg_buf = (is_a ? a_buf_ : b_buf_);

        layout_t load_layout;
        view_t reg_view;
        stmt_t stmt;
        load_sub_tile_impl(abc_kind, i, view, x_buf, x_reg_buf, use_x_slm,
                load_layout, reg_view, stmt);

        auto reg_layout = load_layout;

        if (!is_a && cfg_.do_b_reduction && !cfg_.use_b_slm) {
            auto reduce_stmt = b_reduce_ctx_.create_reduce_stmt(
                    reg_layout, b_buf_, tile);
            stmt = stmt.append(reduce_stmt);
        }

        bool changed;
        auto fma_layout = convert_to_fma_friendly_layout(cfg_, abc_kind,
                bmnk_mapper, reg_layout, a_type(), b_type(), /*is_slm=*/false,
                &changed);

        if (changed) {
            if (fma_layout.type() != reg_layout.type()) {
                reg_view = reg_view.retype(fma_layout.type());
            }
            reg_layout = fma_layout;
            reg_view.set_tlayout(reg_layout);
            stmt = substitute(stmt, x_reg_buf, ab_tmp_buf_);
            stmt = stmt.append(create_reorder_stmt(
                    load_layout, reg_layout, ab_tmp_buf_, x_reg_buf));
            tmp_buf_size_ = std::max(tmp_buf_size_, int(load_layout.size()));
        }

        if (use_x_slm) {
            ab_s2r_load = ab_s2r_load.append(stmt);
        } else {
            ab_g2r_load = ab_g2r_load.append(stmt);
        }
        info.is_loaded = true;
        info.reg_view = reg_view;
        info.reg_buf_size = utils::rnd_up(reg_layout.size(), cfg_.grf_size());
    }

    void load_sub_tile_impl(abc_kind_t abc_kind, int sub_tile_idx,
            const view_t &_mem_view, const expr_t &buf, const expr_t &reg_buf,
            bool is_slm, layout_t &reg_layout, view_t &reg_view, stmt_t &stmt) {
        bool is_a = (abc_kind == abc_kind_t::a);

        view_t mem_view;
        bool load_buffered = false;

        // Using buffered view is enabled only when:
        // - Loading directly from global memory
        // - FMA kind is mad (dpas implementation is more strict and requires
        //   layouts, not views)
        // - Loading A tensor (A - activations for FWD/BWD_D where we may have
        //   overlapping when applying KW blocking )
        if (cfg_.use_ow_kw_grf_cache && !is_slm && is_a
                && cfg_.fma_kind == fma_kind_t::mad) {
            load_buffered
                    = _mem_view.try_create_buffer_view(mem_view, reg_view);
        }

        if (!load_buffered) mem_view = _mem_view;

        send_op_t send_op = send_op_t::load;
        auto send_hint = get_send_hint(send_op_t::load, abc_kind, mem_view,
                can_use_2d_send(), cfg_.simd_size(), gemm_schedule_);
        auto read = make_access_builder(cfg_.hw(), ir_ctx_, cset_, mem_view,
                buf, reg_buf, send_op,
                is_slm ? send_address_t::slm : send_address_t::a64, send_hint);
        ir_trace() << (is_a ? "A" : "B") << " GMEM/SLM to GRF load #"
                   << sub_tile_idx << ":\n"
                   << read.str() << std::endl;

        if (load_buffered) {
            reg_view.set_tlayout(read.reg_layout());
        } else {
            reg_view = view_t(read.reg_layout());
        }

        reg_layout = read.reg_layout();
        stmt = read.stmt();
    }

    bool can_use_2d_send() const {
        if (cfg_.use_2d_send) return true;
        if (cfg_.hw() < ngen::HW::XeHPC) return false;

        // Enable 2D load messages for bwd_w for 32n16c layout
        if (cfg_.is_bwd_w) {
            auto block_2d_ok = [&](const view_t &view, abc_kind_t abc_kind) {
                auto inner = view.tlayout().innermost_block_layout();
                if (view.type() != type_t::bf16()) return false;
                if (inner.blocks().size() != 2) return false;
                auto &bmnk_mapper = gemm_schedule_.bmnk_mapper();
                auto &b0 = inner.blocks()[0];
                auto &b1 = inner.blocks()[1];
                if (b0.block != 16 || b1.block != 32) return false;
                bool b1_is_k = (bmnk_mapper.bmnk_kind(abc_kind, b1.dim_idx)
                        == bmnk_kind_t::k);
                return b1_is_k;
            };
            bool a_ok = block_2d_ok(gemm_schedule_.a_view(), abc_kind_t::a);
            bool b_ok = block_2d_ok(gemm_schedule_.b_view(), abc_kind_t::b);
            return a_ok && b_ok;
        }
        return false;
    }

    void register_buffer(const stmt_t &alloc) {
        ir_assert(alloc.is<alloc_t>());
        allocs_.push_back(alloc);
    }

    void register_buffer(const expr_t &buf, int size, alloc_kind_t kind,
            const alloc_attr_t &attr = {}) {
        register_buffer(alloc_t::make(buf, size, kind, attr));
    }

    const conv_config_t &cfg_;
    ir_context_t ir_ctx_;
    const constraint_set_t &cset_;
    const gemm_schedule_t &gemm_schedule_;
    b_reduce_context_t &b_reduce_ctx_;

    expr_t ap_buf_;
    expr_t a_slm_buf_;

    expr_t bp_buf_;
    expr_t b_slm_buf_;

    layout_t c_reg_layout_;

    expr_t ab_tmp_buf_;
    expr_t a_buf_;
    expr_t b_buf_;
    expr_t c_buf_;

    int tmp_buf_size_ = 0;

    // Per-thread views to multiply.
    view_t a_thr_view_;
    view_t b_thr_view_;

    // Sub-tile indices.
    expr_t a_idx_;
    expr_t b_idx_;

    // Sub-tile views.
    view_t a_i_view_;
    view_t b_j_view_;

    tensor_t a_i_tile_;
    tensor_t b_j_tile_;

    std::vector<sub_tile_info_t> a_sub_tiles_;
    std::vector<sub_tile_info_t> b_sub_tiles_;

    std::vector<block_t> a_i_outer_blocks_;
    std::vector<block_t> b_j_outer_blocks_;

    std::vector<stmt_t> allocs_;

    stmt_t load_mul_stmt_;

    int c_buf_off_ = 0;
    layout_t c_sub_tile_layout_;

    const kernel_info_t &kernel_info_;
};

class compute_builder_t {
public:
    compute_builder_t(const conv_config_t &cfg, ir_context_t &ir_ctx,
            constraint_set_t &cset, const kernel_info_t &kernel_info)
        : cfg_(cfg)
        , ir_ctx_(ir_ctx)
        , cset_(cset)
        , b_reduce_ctx_(cfg)
        , g2s_ctx_(ir_ctx)
        , kernel_info_(kernel_info) {}

    int ab_slm_size() const { return ab_slm_size_; }

    const stmt_t &c_zero_out_stmt() const { return c_zero_out_stmt_; }
    const stmt_t &b_reduced_zero_out_stmt() const {
        return b_reduced_zero_out_stmt_;
    }

    stmt_t zero_out_stmt() const {
        stmt_t ret;
        ret = ret.append(c_zero_out_stmt());
        ret = ret.append(b_reduced_zero_out_stmt());
        return ret;
    }

    stmt_t iter_stmt() const {
        stmt_t stmt;
        bool use_prefetch = !prefetch_stmt_.is_empty();
        bool use_slm = !g2s_load_stmt_.is_empty();
        if (use_prefetch) {
            stmt = stmt.append(stmt_group_t::make(
                    stmt_label_t::prefetch(), prefetch_stmt_));
        } else if (use_slm) {
            stmt = stmt.append(stmt_group_t::make(
                    stmt_label_t::g2s_load(), g2s_load_stmt_));
            stmt = stmt.append(funcs::barrier());
            stmt = stmt.append(stmt_group_t::make(
                    stmt_label_t::g2s_store(), g2s_store_stmt_));
            stmt = stmt.append(funcs::barrier());
        }
        stmt = stmt.append(load_mul_stmt_);
        return stmt;
    }

    const stmt_t &c_store_stmt() const { return c_store_stmt_; }
    const stmt_t &b_reduced_store_stmt() const { return b_reduced_store_stmt_; }

    stmt_t inject_compute_alloc_stmts(const stmt_t &stmt) const {
        return jit::inject_alloc_stmts(stmt, compute_allocs_);
    }

    stmt_t inject_out_alloc_stmts(const stmt_t &stmt) const {
        return jit::inject_alloc_stmts(stmt, out_allocs_);
    }

    stmt_t inject_let_stmts(const stmt_t &stmt) const {
        return jit::inject_let_stmts(stmt, g2s_ctx_.grid_idx_lets);
    }

    void set_gemm_schedule(const gemm_schedule_t &gemm_schedule) {
        gemm_schedule_ = gemm_schedule;
    }

    // Setters for original AP/BP/CP buffers (P - problem notation).
    void set_ap_buf(const expr_t &buf) { ap_buf_ = buf; }
    void set_bp_buf(const expr_t &buf) { bp_buf_ = buf; }
    void set_cp_buf(const expr_t &buf) { cp_buf_ = buf; }
    void set_b_reduced_mem_buf(const expr_t &buf) {
        b_reduce_ctx_.set_b_reduced_mem_buf(buf);
    }

    void set_b_reduced_view(const view_t &v) {
        b_reduce_ctx_.set_b_reduced_view(v);
    }

    void set_post_op_context(const post_op_context_t &post_op_ctx) {
        post_op_ctx_ = post_op_ctx;
    }

    void set_reduce_condition(const expr_t &cond) {
        b_reduce_ctx_.set_reduce_condition(cond);
    }

    void build() {
        // Initialize SLM buffers.
        expr_t a_slm_buf = make_buffer("a_slm");
        expr_t b_slm_buf = make_buffer("b_slm");

        view_t ap_gmem_view = gemm_schedule_.a_tg_view();
        view_t bp_gmem_view = gemm_schedule_.b_tg_view();

        // Views to multiply by a thread group (either GMEM or SLM).
        view_t ap_x_view;
        view_t bp_x_view;
        prepare_gmem_to_slm("A", cfg_.use_a_slm, gemm_schedule_.a_tg_tile(),
                ap_gmem_view, ap_buf_, a_slm_buf, ap_x_view, g2s_ctx_);
        prepare_gmem_to_slm("B", cfg_.use_b_slm, gemm_schedule_.b_tg_tile(),
                bp_gmem_view, bp_buf_, b_slm_buf, bp_x_view, g2s_ctx_);
        prepare_prefetch("A", cfg_.use_prefetch, ap_gmem_view, ap_buf_);
        prepare_prefetch("B", cfg_.use_prefetch, bp_gmem_view, bp_buf_);

        if (ap_x_view.is_empty()) ap_x_view = ap_gmem_view;
        if (bp_x_view.is_empty()) bp_x_view = bp_gmem_view;

        for (auto &bi : g2s_ctx_.bufs) {
            register_compute_buffer(bi.buf, bi.size, alloc_kind_t::grf);
        }

        load_multiply_builder_t load_mul_builder(cfg_, ir_ctx_, cset_,
                gemm_schedule_, b_reduce_ctx_, ap_buf_, a_slm_buf, bp_buf_,
                b_slm_buf, ap_x_view, bp_x_view, kernel_info_);

        load_mul_stmt_ = load_mul_builder.load_mul_stmt();
        compute_allocs_.insert(compute_allocs_.end(),
                load_mul_builder.allocs().begin(),
                load_mul_builder.allocs().end());

        auto c_buf = load_mul_builder.c_buf();
        int c_size = load_mul_builder.c_reg_layout().size();
        register_out_buffer(c_buf, c_size, alloc_kind_t::grf);

        auto c_thr_reg_layout = load_mul_builder.c_reg_layout();
        auto thr_tile = gemm_schedule_.c_thr_tile(/*is_relative=*/false);

        if (gemm_schedule_.with_thread_group_k_slicing()) {
            slm_reduce_builder_t slm_reduce_builder(cfg_.hw(), ir_ctx_, cset_,
                    gemm_schedule_.tg_grid(), c_buf, c_thr_reg_layout,
                    thr_tile);
            c_store_stmt_ = c_store_stmt_.append(slm_reduce_builder.stmt());
            c_thr_reg_layout = slm_reduce_builder.reg_layout();
            thr_tile = slm_reduce_builder.thr_tile();
        }

        auto c_thr_mem_view = gemm_schedule_.c_view().create_sub_view(thr_tile);
        auto c_m2g = create_epilogue_builder(cfg_, ir_ctx_, cset_,
                gemm_schedule_, post_op_ctx_, thr_tile, c_thr_mem_view,
                c_thr_reg_layout, cp_buf_, c_buf);
        ir_trace() << "C GRF to GMEM store:\n" << c_m2g.stmt() << std::endl;

        c_zero_out_stmt_ = stmt_group_t::make(stmt_label_t::c_zero_out(),
                create_zero_out_stmt(cfg_.hw(), c_buf, c_size));
        c_store_stmt_ = c_store_stmt_.append(c_m2g.stmt());

        if (cfg_.do_b_reduction) {
            auto &ctx = b_reduce_ctx_;
            b_reduced_zero_out_stmt_ = create_zero_out_stmt(
                    cfg_.hw(), ctx.b_reduced_reg_buf(), ctx.b_reduced_size());
            b_reduced_store_stmt_ = ctx.create_store_stmt(ir_ctx_, cset_);
            register_out_buffer(ctx.b_reduced_reg_buf(), ctx.b_reduced_size(),
                    alloc_kind_t::grf);
        }

        // Replace DPAS by DPASW when applicable.
        if (cfg_.fma_kind == fma_kind_t::dpasw) {
            alloc_updater_t alloc_updater;
            inject_dpasw(cfg_.hw(), load_mul_stmt_, c_buf, c_store_stmt_,
                    alloc_updater, gemm_schedule_.tg_grid().idx(0));
            for (auto &a : compute_allocs_) {
                a = alloc_updater.update(a);
            }
            for (auto &a : out_allocs_) {
                a = alloc_updater.update(a);
            }
        }

        // Assign {Atomic} for DPAS(W) when applicable.
        load_mul_stmt_ = inject_atomic(load_mul_stmt_);
    }

private:
    struct buf_info_t {
        buf_info_t(const std::string &tag, const expr_t &buf)
            : tag(tag), buf(buf) {}

        std::string tag;
        expr_t buf;
        int size = 0;
    };

    struct g2s_context_t {
        g2s_context_t(ir_context_t &ir_ctx) : ir_ctx(ir_ctx) {}

        expr_t create_buf(const char *tag, bool force_reuse = false) {
            if (reuse_buffers || force_reuse) {
                for (auto &bi : bufs) {
                    if (bi.tag == tag) return bi.buf;
                }
            }
            auto buf = ir_ctx.create_tmp_var(type_t::byte_ptr(), tag);
            bufs.emplace_back(tag, buf);
            return buf;
        }

        void set_buf_size(const expr_t &buf, int size) {
            for (auto &bi : bufs) {
                if (bi.buf.is_same(buf)) bi.size = std::max(bi.size, size);
            }
        }

        expr_t create_tmp_grid_idx() {
            auto var = ir_ctx.create_tmp_var(type_t::s32(), "idx");
            tmp_grid_idxs.insert({var, expr_t()});
            return var;
        }

        void set_grid_idx_value(const expr_t &idx, const expr_t &value) {
            auto &old = tmp_grid_idxs[idx];
            ir_assert(old.is_empty());
            old = substitute_grid_idx_value(value);
        }

        expr_t substitute_grid_idx_value(const expr_t &_e) {
            auto e = _e;
            auto vars = find_unique_objects<var_t>(e);
            for (auto &v : vars) {
                auto it = tmp_grid_idxs.find(v);
                if (it == tmp_grid_idxs.end()) continue;
                e = substitute(e, v, it->second);
            }
            return e;
        }

        void register_grid(const grid_info_t &grid) {
            for (int i = 0; i < grid.ndims(); i++) {
                auto &idx = grid.idx(i);
                auto it = tmp_grid_idxs.find(idx);
                if (it == tmp_grid_idxs.end()) continue;
                grid_idx_lets.emplace_back(let_t::make(idx, it->second));
            }
        }

        ir_context_t &ir_ctx;
        grid_info_t prev_load_grid;
        bool reuse_buffers = false;
        std::vector<buf_info_t> bufs;

        object_map_t<expr_t, expr_t> tmp_grid_idxs;
        std::vector<stmt_t> grid_idx_lets;
    };

    void register_compute_buffer(const expr_t &buf, int size, alloc_kind_t kind,
            const alloc_attr_t &attr = {}) {
        compute_allocs_.push_back(alloc_t::make(buf, size, kind, attr));
    }

    void register_out_buffer(const expr_t &buf, int size, alloc_kind_t kind,
            const alloc_attr_t &attr = {}) {
        out_allocs_.push_back(alloc_t::make(buf, size, kind, attr));
    }

    // Handles GMEM to SLM load for A and B. Done in two steps:
    // 1. Load: GMEM -> GRF (temporary)
    // 2. Store: GRF (temporary) -> SLM
    void prepare_gmem_to_slm(const char *tag, bool use_x_slm,
            const tensor_t &tg_tile, const view_t &x_gmem_view,
            const expr_t &xp_buf, const expr_t &x_slm_buf, view_t &x_slm_view,
            g2s_context_t &g2s_ctx) {
        if (!use_x_slm) return;

        grid_info_t load_grid = gemm_schedule_.tg_grid();
        for (;;) {
            bool ok = prepare_gmem_to_slm_impl(tag, use_x_slm, tg_tile,
                    x_gmem_view, xp_buf, x_slm_buf, x_slm_view, load_grid,
                    g2s_ctx);
            if (ok) {
                g2s_ctx.prev_load_grid = load_grid;
                g2s_ctx.register_grid(load_grid);
                return;
            }

            // Reduce grid and try again.
            auto grid_idx = g2s_ctx.create_tmp_grid_idx();
            int dim_idx;
            expr_t grid_idx_value;
            auto new_load_grid
                    = load_grid.halven(grid_idx, dim_idx, grid_idx_value);
            if (new_load_grid.is_empty()) break;

            if (new_load_grid == g2s_ctx.prev_load_grid) {
                new_load_grid = load_grid.halven(
                        grid_idx, dim_idx, grid_idx_value, /*first=*/false);
                g2s_ctx.reuse_buffers = true;
            }
            g2s_ctx.set_grid_idx_value(grid_idx, grid_idx_value);

            cset_.add_constraint(grid_idx >= 0);
            cset_.add_constraint(grid_idx < new_load_grid.dim(dim_idx));

            load_grid = new_load_grid;
        }
        ir_error_not_expected() << "Can't create GMEM -> SLM loads/stores.";
    }

    bool prepare_gmem_to_slm_impl(const char *tag, bool use_x_slm,
            const tensor_t &tg_tile, const view_t &x_gmem_view,
            const expr_t &xp_buf, const expr_t &x_slm_buf, view_t &x_slm_view,
            const grid_info_t &load_grid, g2s_context_t &g2s_ctx) {
        bool is_a = (tag[0] == 'A');
        abc_kind_t ab_kind = (is_a ? abc_kind_t::a : abc_kind_t::b);

        auto xp_slm_layout = create_slm_layout(x_gmem_view, ab_kind, load_grid);

        auto grid_cond = load_grid.slice_condition();

        // Per-thread tile and view to load from GMEM and store to SLM.
        tensor_t thr_tile;
        view_t x_g2s_view;
        if (cfg_.allow_slm_tg_slicing) {
            x_g2s_view = x_gmem_view.split(load_grid, thr_tile);
        } else {
            thr_tile = xp_slm_layout.split(load_grid);
            x_g2s_view = x_gmem_view.create_sub_view(thr_tile);
        }

        auto slm_thr_layout = xp_slm_layout.map(thr_tile);

        // Ensure that each thread writes a dense region to SLM. If the layout
        // is not dense, return and try with smaller grid.
        if (!slm_thr_layout.is_dense()) return false;

        register_compute_buffer(
                x_slm_buf, xp_slm_layout.size(), alloc_kind_t::slm);
        ab_slm_size_ += xp_slm_layout.size();

        // Temporary GRF buffer.
        expr_t x_g2s_reg_buf = g2s_ctx.create_buf("g2s");

        // GMEM -> GRF load.
        auto x_read = make_access_builder(cfg_.hw(), ir_ctx_, cset_, x_g2s_view,
                xp_buf, x_g2s_reg_buf, send_op_t::load, send_address_t::a64);
        ir_trace() << tag << " GMEM to GRF load:\n"
                   << x_read.str() << std::endl;

        g2s_ctx.set_buf_size(x_g2s_reg_buf, x_read.reg_buf_size());

        auto load_stmt = x_read.stmt();
        if (!grid_cond.is_empty()) load_stmt = if_t::make(grid_cond, load_stmt);
        g2s_load_stmt_ = g2s_load_stmt_.append(load_stmt);

        // GRF -> SLM store.
        auto x_write = make_access_builder(cfg_.hw(), ir_ctx_, cset_,
                view_t(slm_thr_layout), x_slm_buf, x_g2s_reg_buf,
                send_op_t::store, send_address_t::slm);
        ir_trace() << tag << " GRF to SLM store:\n"
                   << x_write.str() << std::endl;
        auto store_stmt = x_write.stmt();

        auto &read_layout = x_read.reg_layout();
        auto &write_layout = x_write.reg_layout();
        if (read_layout != write_layout) {
            if (cfg_.allow_grf_reorder) {
                // Temporary GRF buffer.
                expr_t tmp_buf
                        = g2s_ctx.create_buf("g2s_tmp", /*force_reuse=*/true);
                auto reorder_stmt = create_reorder_stmt(
                        read_layout, write_layout, x_g2s_reg_buf, tmp_buf);
                g2s_ctx.set_buf_size(tmp_buf, x_write.reg_buf_size());
                store_stmt = substitute(store_stmt, x_g2s_reg_buf, tmp_buf);
                store_stmt = reorder_stmt.append(store_stmt);
            } else {
                ir_error_not_expected() << "Requested register layouts for "
                                        << tag << " do not match: "
                                        << "read: " << read_layout
                                        << ", write: " << write_layout;
            }
        }
        // Generate reduction statement for B.
        if (!is_a && cfg_.do_b_reduction) {
            auto absolute_thr_tile = tg_tile.create_sub_tensor(thr_tile);
            b_reduce_ctx_.init_reduced_thr_view(absolute_thr_tile, grid_cond);
            auto reduce_stmt = b_reduce_ctx_.create_reduce_stmt(
                    read_layout, x_g2s_reg_buf);
            store_stmt = reduce_stmt.append(store_stmt);
        }
        if (!grid_cond.is_empty())
            store_stmt = if_t::make(grid_cond, store_stmt);
        g2s_store_stmt_ = g2s_store_stmt_.append(store_stmt);

        x_slm_view = view_t(xp_slm_layout);

        return true;
    }

    void prepare_prefetch(const char *tag, bool use_prefetch,
            const view_t &x_gmem_view, const expr_t &xp_buf) {
        if (!use_prefetch) return;

        // Per-thread view to prefetch from GMEM.
        auto thr_view = x_gmem_view.split(gemm_schedule_.tg_grid());

        auto send_hint = get_send_hint(send_op_t::prefetch,
                (tag[0] == 'A') ? abc_kind_t::a : abc_kind_t::b, thr_view,
                cfg_.use_2d_send, cfg_.simd_size(), gemm_schedule_);

        // GMEM prefetch.
        auto x_prefetch = make_access_builder(cfg_.hw(), ir_ctx_, cset_,
                thr_view, xp_buf, expr_t(), send_op_t::prefetch,
                send_address_t::a64, send_hint);
        ir_trace() << tag << " GMEM prefetch:\n"
                   << x_prefetch.str() << std::endl;

        prefetch_stmt_ = prefetch_stmt_.append(x_prefetch.stmt());
    }

    layout_t create_slm_layout(const view_t &tg_view, abc_kind_t abc_kind,
            const grid_info_t &load_grid) const {
        auto layout = tg_view.create_dense_vlayout();
        auto &a_type = gemm_schedule_.a_view().type();
        auto &b_type = gemm_schedule_.b_view().type();
        auto ret = convert_to_fma_friendly_layout(cfg_, abc_kind,
                gemm_schedule_.bmnk_mapper(), layout, a_type, b_type,
                /*is_slm=*/true);
        if (cfg_.pad_slm) ret = pad_slm_layout(ret, load_grid);
        return ret;
    }

    // SLM has 65 dword-granularity banks (Xe_HP):
    //      banks:   [bank 0] [bank 1] [bank 2] ... [bank 0]
    // byte offsets: | 0      | 4      | 8      ... | 4 * 65
    // SLM reads don't have conflicts. During SLM writes each fused EU writes
    // 64 bytes (in total 128 bytes per clock). If there are repeating banks
    // between 128 bytes the write takes 2 clocks to complete.
    // Assume that every X-axis thread (across tg_dim[0]) writes the
    // corresponding outer block of the layout. The goal is to ensure that the
    // stride between outer blocks allows to avoid duplicated banks.
    layout_t pad_slm_layout(
            const layout_t &layout, const grid_info_t &load_grid) const {
        auto tg_dim0 = load_grid.dim(0);
        auto tg_dim1 = load_grid.dim(1);
        int type_size = layout.type().size();

        ir_assert(layout.elems() % tg_dim0 == 0) << layout;
        dim_t inner_block = layout.elems() / tg_dim0;

        ir_assert((inner_block * type_size) % tg_dim1 == 0) << layout;
        dim_t per_thr_bytes = (inner_block * type_size) / tg_dim1;

        std::vector<dim_t> multi_blocks = {inner_block, tg_dim0};
        auto l = layout.split_into_multi_blocks(multi_blocks);

        auto padded_blocks = l.blocks();
        dim_t stride = -1;
        dim_t remaining_elems = inner_block;
        bool past_inner_block = remaining_elems == 1;
        for (auto &b : padded_blocks) {
            if (past_inner_block) {
                if (stride == -1) {
                    dim_t stride_bytes = find_min_stride_without_conflicts(
                            per_thr_bytes, dim_t(b.stride) * type_size);
                    ir_assert(stride_bytes % type_size == 0);
                    stride = stride_bytes / type_size;
                }
                b.stride = stride;
                stride = b.stride * b.block;
                continue;
            }
            ir_assert(remaining_elems % b.block == 0);
            remaining_elems /= b.block;
            if (remaining_elems == 1) past_inner_block = true;
        }
        return layout_t(
                layout.type(), layout.ndims(), layout.offset(), padded_blocks);
    }

    dim_t find_min_stride_without_conflicts(
            dim_t inner_bytes, dim_t dense_stride_bytes) const {
        int write_step = 64;
        int stride_step = 16;
        dim_t stride_beg = dense_stride_bytes;
        dim_t stride_end = 2 * dense_stride_bytes;
        const int slm_banks = 65;
        for (dim_t s = stride_beg; s < stride_end; s += stride_step) {
            bool ok = true;
            for (dim_t off0 = 0; off0 < inner_bytes; off0 += write_step) {
                // Check banks for a single SLM write.
                bool found[slm_banks] = {false};
                for (dim_t off = off0; off < off0 + write_step;
                        off += sizeof(uint32_t)) {
                    int bank0 = (off / sizeof(uint32_t)) % slm_banks;
                    int bank1 = ((off + s) / sizeof(uint32_t)) % slm_banks;
                    if (found[bank0]) {
                        ok = false;
                        break;
                    }
                    found[bank0] = true;
                    if (found[bank1]) {
                        ok = false;
                        break;
                    }
                    found[bank1] = true;
                }
                if (ok) return s;
            }
        }

        ir_warning()
                << "Couldn't find stride without conflicts for SLM padding."
                << std::endl;

        return dense_stride_bytes;
    }

    const conv_config_t &cfg_;
    ir_context_t &ir_ctx_;
    constraint_set_t &cset_;
    post_op_context_t post_op_ctx_;
    b_reduce_context_t b_reduce_ctx_;

    g2s_context_t g2s_ctx_;

    gemm_schedule_t gemm_schedule_;

    expr_t ap_buf_;
    expr_t bp_buf_;
    expr_t cp_buf_;

    std::vector<stmt_t> compute_allocs_;
    std::vector<stmt_t> out_allocs_;
    int ab_slm_size_ = 0;

    stmt_t g2s_load_stmt_;
    stmt_t g2s_store_stmt_;
    stmt_t prefetch_stmt_;
    stmt_t load_mul_stmt_;

    stmt_t c_zero_out_stmt_;
    stmt_t c_store_stmt_;

    stmt_t b_reduced_zero_out_stmt_;
    stmt_t b_reduced_store_stmt_;

    const kernel_info_t &kernel_info_;
};

class compute_loop_label_injector_t : public ir_mutator_t {
public:
    object_t _mutate(const for_t &obj) override {
        if (injected_) return obj;

        bool found_continue = false;
        auto calls = find_objects<func_call_t>(obj);
        for (auto &_c : calls) {
            auto &c = _c.as<func_call_t>();
            if (c.func.is_equal(funcs::continue_func())) found_continue = true;
        }

        if (!found_continue) {
            injected_ = true;
            return stmt_group_t::make(stmt_label_t::compute_loop(), obj);
        }
        return ir_mutator_t::_mutate(obj);
    }

private:
    bool injected_ = false;
};

// Injects compute_loop statement label to the outermost loop that can be
// pipelined. If a loop contains a "continue" function call it can't be
// pipelined because of conditional flow.
stmt_t inject_compute_loop_label(const stmt_t &s) {
    return compute_loop_label_injector_t().mutate(s);
}

void init_kernel_grid(const std::array<int, 3> &kernel_grid_dims,
        const std::array<int, 3> &tg_grid_dims, int simd_size,
        constraint_set_t &cset, grid_info_t &kernel_grid, grid_info_t &tg_grid,
        std::array<expr_t, 3> &local_id) {
    int grid_ndims = 3;
    kernel_grid = grid_info_t(grid_ndims);
    tg_grid = grid_info_t(grid_ndims);
    for (int i = 0; i < grid_ndims; i++) {
        local_id[i]
                = var_t::make(type_t::u16(), "local_id" + std::to_string(i));
        kernel_grid.dim(i) = kernel_grid_dims[i];
        kernel_grid.idx(i)
                = var_t::make(type_t::s32(), "grid_idx" + std::to_string(i));
        tg_grid.dim(i) = tg_grid_dims[i];
        tg_grid.idx(i)
                = var_t::make(type_t::s32(), "tg_idx" + std::to_string(i));

        int local_id_bound = tg_grid_dims[i];
        if (i == 0) local_id_bound *= simd_size;
        cset.add_constraint(local_id[i] >= 0);
        cset.add_constraint(local_id[i] < local_id_bound);

        cset.add_constraint(kernel_grid.idx(i) >= 0);
        cset.add_constraint(kernel_grid.idx(i) < kernel_grid_dims[i]);
        cset.add_constraint(tg_grid.idx(i) >= 0);
        cset.add_constraint(tg_grid.idx(i) < tg_grid_dims[i]);
    }
}

void kernel_builder_t::build() {
    ir_context_t ir_ctx;
    constraint_set_t init_cset;

    init_kernel_grid(cfg_.kernel_grid_dim, cfg_.tg_grid_dim, cfg_.simd_size(),
            init_cset, kernel_grid_, tg_grid_, local_id_);

    gemm_schedule_t gemm_schedule(init_cset, kernel_grid_, tg_grid_);

    std::vector<stmt_t> init_stmts;
    int grid_ndims = 3;
    for (int i = 0; i < grid_ndims; i++) {
        auto value = local_id_[i];
        if (i == 0) value /= cfg_.simd_size();
        init_stmts.push_back(let_t::make(tg_grid_.idx(i), value));
    }

    // Initialize memory buffers.
    std::vector<stmt_t> inner_lets;

    view_t a_view;
    view_t b_view;
    view_t c_view;
    view_t bp_reduced_view;

    expr_t ap_buf;
    expr_t bp_buf;
    expr_t cp_buf;
    expr_t b_reduced_mem_buf;
    expr_t b_reduction_condition;

    if (cfg_.is_fwd) {
        init_fwd(gemm_schedule, a_view, b_view, c_view, ap_buf, bp_buf, cp_buf);
    } else if (cfg_.is_bwd_d) {
        init_bwd_d(
                gemm_schedule, a_view, b_view, c_view, ap_buf, bp_buf, cp_buf);
    } else if (cfg_.is_bwd_w) {
        init_bwd_w(gemm_schedule, a_view, b_view, c_view, bp_reduced_view,
                ap_buf, bp_buf, cp_buf, b_reduced_mem_buf,
                b_reduction_condition);
    } else {
        ir_error_not_expected();
    }

    gemm_schedule.finalize();

    post_op_context_t post_op_ctx(pd_, cfg_, gemm_schedule, kernel_info_);
    compute_builder_t cb(cfg_, ir_ctx, init_cset, kernel_info_);

    cb.set_gemm_schedule(gemm_schedule);
    cb.set_ap_buf(ap_buf);
    cb.set_bp_buf(bp_buf);
    cb.set_cp_buf(cp_buf);
    cb.set_b_reduced_mem_buf(b_reduced_mem_buf);
    cb.set_b_reduced_view(bp_reduced_view);
    cb.set_post_op_context(post_op_ctx);
    cb.set_reduce_condition(b_reduction_condition);

    cb.build();

    std::vector<stmt_t> allocs;
    for (int i = 0; i < kernel_info_.nargs(); i++) {
        auto &var = kernel_info_.arg_var(i);
        if (!var.type().is_ptr()) continue;
        allocs.push_back(alloc_t::make(var, 0, alloc_kind_t::global));
    }

    // Create IR statements.
    stmt_t loop_stmt = cb.iter_stmt();
    loop_stmt = gemm_schedule.create_loop_nest(loop_stmt);
    loop_stmt = inject_compute_loop_label(loop_stmt);
    loop_stmt = cb.inject_compute_alloc_stmts(loop_stmt);

    auto c_store_stmt
            = stmt_group_t::make(stmt_label_t::c_store(), cb.c_store_stmt());
    stmt_ = loop_stmt;
    stmt_ = stmt_seq_t::make(cb.zero_out_stmt(), stmt_);
    stmt_ = stmt_seq_t::make(stmt_, cb.b_reduced_store_stmt());
    stmt_ = stmt_seq_t::make(stmt_, c_store_stmt);

    stmt_ = cb.inject_out_alloc_stmts(stmt_);
    stmt_ = cb.inject_let_stmts(stmt_);

    stmt_ = gemm_schedule.create_bind_stmt(stmt_);
    stmt_ = inject_let_stmts(stmt_, init_stmts);
    stmt_ = inject_alloc_stmts(stmt_, allocs);

    stmt_ = inject_external_var_let(stmt_);
    stmt_ = merge_slm_buffers(stmt_);
    if (!cfg_.do_pipeline_unroll && (cfg_.use_a_slm || cfg_.use_b_slm)) {
        stmt_ = inject_simple_slm_buffering(
                cfg_.hw(), stmt_, cfg_, ir_ctx, cb.ab_slm_size());
    } else if (!cfg_.do_pipeline_unroll && cfg_.use_prefetch) {
        // Simplify to remove loops with only 1 iteration
        stmt_ = simplify_pass(stmt_, init_cset);
        stmt_ = inject_prefetch_pipeline(stmt_, cfg_, ir_ctx);
    }
    stmt_ = inject_slm_reorder(stmt_, cfg_, tg_grid_);
    stmt_ = lift_buffer_offsets_in_send(stmt_);
    stmt_ = simplify_pass(stmt_, init_cset);
    stmt_ = inject_send(stmt_, ir_ctx, init_cset);
    stmt_ = split_wide_stores(cfg_.hw(), stmt_);
    stmt_ = lift_alloc(stmt_, cfg_);
    stmt_ = lift_send_2d_header_store(stmt_);
    stmt_ = hoist_send_masks(stmt_, ir_ctx, stmt_label_t::c_store(), false);
    stmt_ = eliminate_common_subexprs(stmt_, ir_ctx);
    stmt_ = hoist_exprs(stmt_, ir_ctx);
    if (cfg_.do_pipeline_unroll) stmt_ = loop_strength_reduce(stmt_);
    stmt_ = optimize_alloc_let(stmt_);
    if (cfg_.do_pipeline_unroll) {
        stmt_ = update_loops_for_unrolling(stmt_, cfg_);
        stmt_ = inject_unrolling(stmt_, cfg_, ir_ctx, cb.ab_slm_size());
    }
    if (cfg_.hoist_masks_from_compute_loop) {
        stmt_ = hoist_send_masks(
                stmt_, ir_ctx, stmt_label_t::compute_loop(), true);
    }
    stmt_ = fixup_if_conditions(stmt_, cfg_);
    stmt_ = unroll_loops(stmt_, ir_ctx);
    stmt_ = simplify_pass(stmt_, init_cset);
    stmt_ = optimize_alloc_let(stmt_);
    if (cfg_.hoist_masks_from_compute_loop) {
        stmt_ = remove_spurious_send_mask_cast(stmt_);
    }
    stmt_ = fix_int32_overflow(stmt_, init_cset);
    stmt_ = optimize_peephole(stmt_);
    stmt_ = optimize_barrier(stmt_);
    if (cfg_.fma_kind == fma_kind_t::dp4a) stmt_ = inject_dp4a(stmt_);
    stmt_ = inject_bank_conflict_attribute(stmt_);
    stmt_ = stmt_group_t::make(stmt_label_t::kernel(), stmt_);

    ir_trace() << "Convolution kernel body:\n" << stmt_ << std::endl;
}

std::vector<int> reorder_kernel_builder_t::compute_blocks(
        const layout_t &src, const layout_t &dst, int &threads) {
    ir_assert(src.ndims() == dst.ndims());
    int ndims = src.ndims();
    std::vector<dim_t> dims(ndims);
    std::vector<dim_t> inner_dims(ndims);
    auto src_inner = src.innermost_block_layout();
    auto dst_inner = dst.innermost_block_layout();
    for (int i = 0; i < ndims; i++) {
        dims[i] = std::max(src.dim(i), dst.dim(i));
        inner_dims[i] = std::max(src_inner.dim(i), dst_inner.dim(i));
    }

    std::vector<int> candidate_dim_idxs;
    for (int i = 0; i < ndims; i++) {
        if (inner_dims[i] != 1) candidate_dim_idxs.push_back(i);
    }

    for (dim_t dim_limit : {16, 1}) {
        if (candidate_dim_idxs.size() >= 2) break;
        for (int i = 0; i < ndims; i++) {
            if (inner_dims[i] != 1 || dims[i] == 1) continue;
            if (dims[i] >= dim_limit) candidate_dim_idxs.push_back(i);
        }
    }

    auto get_block = [&](int dim_idx) {
        dim_t max_block = 16;
        dim_t dim = inner_dims[dim_idx];
        if (dim == 1) dim = dims[dim_idx];
        return (int)utils::max_div(dim, max_block);
    };

    dim_t max_total_block = 256;
    dim_t total_block = 1;
    std::vector<int> blocks(ndims, 1);
    for (int idx : candidate_dim_idxs) {
        int block = get_block(idx);
        if (total_block * block >= max_total_block) break;
        blocks[idx] = block;
        total_block *= block;
    }

    threads = 1;
    for (int i = 0; i < ndims; i++) {
        threads *= utils::div_up(dims[i], blocks[i]);
    }
    return blocks;
}

void reorder_kernel_builder_t::build() {
    ir_context_t ir_ctx;
    constraint_set_t init_cset;

    int ndims = src_layout_.ndims();
    std::vector<expr_t> vars;
    for (int i = 0; i < ndims; i++) {
        char letter = 'a' + i;
        vars.push_back(var_t::make(type_t::s32(), std::string(1, letter)));
    }

    int threads = 1;
    auto blocks = compute_blocks(src_layout_, dst_layout_, threads);

    ir_info() << "Reorder configuration:" << std::endl;
    ir_info() << "  Tile size:                  "
              << ir_utils::make_seq_print_helper(blocks, " x ") << std::endl;

    std::array<int, 3> kernel_grid_dims = {threads, 1, 1};
    std::array<int, 3> tg_grid_dims = {1, 1, 1};

    init_kernel_grid(kernel_grid_dims, tg_grid_dims, hw_cfg_.simd_size(),
            init_cset, kernel_grid_, tg_grid_, local_id_);

    auto &x = view_t::placeholder_var();

    std::vector<dim_t> vdims(ndims);
    for (int i = 0; i < ndims; i++) {
        vdims[i] = std::max(src_layout_.dim(i), dst_layout_.dim(i));
    }

    view_t src_view(vars, ndims);
    for (int i = 0; i < ndims; i++) {
        int dim = src_layout_.dim(i);
        src_view.set_vdim(vars[i], vdims[i]);
        expr_t mask(true);
        if (dim != vdims[i]) mask = x < dim;
        src_view.set_tdim(i, vars[i], mask);
    }
    src_view.set_tlayout(src_layout_);

    view_t dst_view(vars, ndims);
    for (int i = 0; i < ndims; i++) {
        int dim = dst_layout_.dim(i);
        dst_view.set_vdim(vars[i], vdims[i]);
        expr_t mask(true);
        if (dim != vdims[i]) mask = x < dim;
        dst_view.set_tdim(i, vars[i], mask);
    }
    dst_view.set_tlayout(dst_layout_);

    gemm_schedule_t schedule(init_cset, kernel_grid_, tg_grid_);

    schedule.set_view(src_view);
    schedule.set_view(dst_view);

    std::vector<expr_t> fused_idxs;
    int block_size = 1;
    for (int i = 0; i < ndims; i++) {
        if (blocks[i] == 1) {
            fused_idxs.push_back(vars[i]);
            continue;
        }
        block_size *= blocks[i];
        expr_t outer;
        expr_t inner;
        schedule.split(vars[i], blocks[i], outer, inner);
        schedule.tensorize(inner);
        fused_idxs.push_back(outer);
    }

    auto tg_idx = schedule.fuse(fused_idxs);

    schedule.bind(tg_idx, kernel_grid_.idx(0));

    schedule.finalize();

    auto thr_tile = schedule.thr_view_tile(src_view, /*is_relative=*/false);

    auto src_thr_view = src_view.create_sub_view(thr_tile);
    auto dst_thr_view = dst_view.create_sub_view(thr_tile);

    auto src_buf = kernel_info_.arg_var(0);
    auto dst_buf = kernel_info_.arg_var(1);

    auto reg_buf = ir_ctx.create_tmp_var(type_t::byte_ptr(), "reg");

    std::vector<stmt_t> allocs;
    for (int i = 0; i < kernel_info_.nargs(); i++) {
        auto &var = kernel_info_.arg_var(i);
        if (!var.type().is_ptr()) continue;
        allocs.push_back(alloc_t::make(var, 0, alloc_kind_t::global));
    }

    auto read
            = make_access_builder(hw_cfg_.hw(), ir_ctx, init_cset, src_thr_view,
                    src_buf, reg_buf, send_op_t::load, send_address_t::a64);
    auto read_stmt = read.stmt();

    auto write
            = make_access_builder(hw_cfg_.hw(), ir_ctx, init_cset, dst_thr_view,
                    dst_buf, reg_buf, send_op_t::store, send_address_t::a64);
    auto write_stmt = write.stmt();

    auto read_layout = read.reg_layout();
    auto write_layout = write.reg_layout();
    allocs.push_back(
            alloc_t::make(reg_buf, read_layout.size(), alloc_kind_t::grf));

    if (read_layout != write_layout) {
        auto tmp_buf = ir_ctx.create_tmp_var(type_t::byte_ptr(), "tmp");
        allocs.push_back(
                alloc_t::make(tmp_buf, write_layout.size(), alloc_kind_t::grf));

        auto reorder_stmt = create_reorder_stmt(
                read_layout, write_layout, reg_buf, tmp_buf);
        write_stmt = substitute(write_stmt, reg_buf, tmp_buf);
        write_stmt = reorder_stmt.append(write_stmt);
    }

    stmt_ = stmt_.append(read_stmt);
    stmt_ = stmt_.append(write_stmt);

    stmt_ = schedule.create_loop_nest(stmt_);
    stmt_ = schedule.create_bind_stmt(stmt_);
    stmt_ = inject_alloc_stmts(stmt_, allocs);
    stmt_ = inject_external_var_let(stmt_);

    stmt_ = simplify_pass(stmt_, init_cset);
    stmt_ = lift_buffer_offsets_in_send(stmt_);
    stmt_ = inject_send(stmt_, ir_ctx, init_cset);
    stmt_ = split_wide_stores(hw_cfg_.hw(), stmt_);
    stmt_ = simplify_pass(stmt_, init_cset);
    stmt_ = stmt_group_t::make(stmt_label_t::kernel(), stmt_);

    ir_trace() << "Reorder kernel body:\n" << stmt_ << std::endl;
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
