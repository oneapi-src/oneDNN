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

#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <compiler/ir/builder.hpp>
#include <compiler/ir/ir_module.hpp>
#include <compiler/jit/xbyak/ir/reg_allocation/reg_allocator.hpp>
#include <compiler/jit/xbyak/ir/transform/constant_optimizer.hpp>
#include <compiler/jit/xbyak/ir/xbyak_visitor.hpp>
#include <compiler/jit/xbyak/x86_64/abi_function_interface.hpp>

#include "call_transform.hpp"
#include "register_allocation.hpp"

using namespace sc::sc_xbyak::x86_64;

namespace sc {
namespace sc_xbyak {

SC_MODULE(xbyakjit.register_allocation)

void prepare_virtual_reg(reg_allocator_t *allocator, virtual_reg_t *virt_reg,
        const Xbyak::Reg &phy_reg, sc_data_type_t dtype) {
    if (phy_reg.isNone()) {
        virt_reg->set_type(get_virt_reg_type(dtype));
        virt_reg->set_unassigned();
        virt_reg->add_weight(virt_reg->range_weight());
    } else {
        auto &slots_map = allocator->slots_map();
        virt_reg->set_type(get_virt_reg_type(dtype));
        virt_reg->set_designated(slots_map.get_reg_index(phy_reg));
    }
    if (!virt_reg->live_range_.empty()) { allocator->enqueue(virt_reg); }
}

/* *
 * Pre-allocation pass: assign hints for each expr,
 * prepare physical regs and enqueue all virtual regs to be assigned.
 * */
class pre_allocation_t : public xbyak_visitor_t {
public:
    using xbyak_visitor_t::dispatch;
    using xbyak_visitor_t::visit;

    pre_allocation_t(reg_allocator_t *allocator,
            std::vector<live_range_t> &call_scopes,
            std::map<virtual_reg_t *, expr_c> &virt_reg_map)
        : allocator_(allocator)
        , call_scopes_(call_scopes)
        , virt_reg_map_(virt_reg_map)
        , is_local_scope_(false) {}

    func_c operator()(func_c v) { return dispatch(std::move(v)); }

    func_c dispatch(func_c v) override {
        func_t func = std::const_pointer_cast<func_base>(v);
        func_iface_ = cached_func_abi_interface(func);
        set_func_args_hint(v->params_);
        return xbyak_visitor_t::dispatch(std::move(v));
    }

    expr_c dispatch(expr_c v) override {
        // If need reg allocation
        if (v.isa<var>() || v.isa<tensor>()) {
            auto &live_range = GET_LIVE_RANGE(v);
            auto &virt_reg = GET_VIRTUAL_REG(v);
            auto &phy_reg = GET_PHYSICAL_REG(v);
            virt_reg_map_[&virt_reg] = v;
            if (is_local_scope_
                    && call_scopes_.back().encompasses(live_range)) {
                return v;
            }
            if (virt_reg.stat_ == virt_reg_stat::disabled) {
                prepare_virtual_reg(allocator_, &virt_reg, phy_reg, v->dtype_);
            }
            return v;
        }
        return xbyak_visitor_t::dispatch(std::move(v));
    }

    stmt_c visit(stmts_c v) override {
        if (TRANSFORMED_CALL(v)) {
            if (v->size() == 0) {
                GET_STMT_DATA(v).optimized_out_ = true;
                return v;
            }
            stmt_index_t beg = GET_STMT_INIT_INDEX(v);
            stmt_index_t end = GET_STMT_INDEX(v);
            call_scopes_.emplace_back(live_range_t(beg, end));
            is_local_scope_ = true;
        }
        auto ret = xbyak_visitor_t::visit(std::move(v));
        is_local_scope_ = false;
        return ret;
    }

    stmt_c visit(define_c v) override {
        if (GET_LIVE_RANGE(v->var_).empty()) {
            GET_STMT_DATA(v).optimized_out_ = true;
            return v;
        }
        if (v->var_.isa<tensor>() && !v->init_.defined()) {
            auto &virt_reg = GET_VIRTUAL_REG(v->var_);
            virt_reg.set_buffered();
            allocator_->spilled_virt_regs().insert(&virt_reg);
        }
        return xbyak_visitor_t::visit(std::move(v));
    }

    stmt_c visit(assign_c v) override {
        auto &live_range = GET_LIVE_RANGE(v->var_);
        if (v->var_.isa<var>() && live_range.empty()) {
            GET_STMT_DATA(v).optimized_out_ = true;
            return v;
        }
        if (v->value_.isa<call>()) {
            if (call_scopes_.back().encompasses(live_range)) {
                auto call_v = v->value_.static_as<call_c>();
                auto &callee_iface = *cached_call_abi_interface(call_v);
                GET_PHYSICAL_REG(v->var_)
                        = callee_iface.return_val_loc_.get_register();
            }
            // TODO(xxx): better way to check liveness cross call
            call_scopes_.back().end_ = GET_STMT_INDEX(v);
        }
        return xbyak_visitor_t::visit(std::move(v));
    }

    stmt_c visit(returns_c v) override {
        if (v->value_.defined()) { set_func_return_hint(v->value_); }
        return xbyak_visitor_t::visit(std::move(v));
    }

    expr_c visit(tensorptr_c v) override {
        // TODO(XXX): why this temp_data go missing?
        v->base_->temp_data() = xbyak_expr_data_t();
        return xbyak_visitor_t::visit(std::move(v));
    }

    expr_c visit(call_c v) override {
        auto &args = v->args_;
        auto &callee_iface = *cached_call_abi_interface(v);
        auto &slots_map = allocator_->slots_map();
        for (size_t i = 0; i < args.size(); ++i) {
            auto &param = args[i];
            auto &virt_para = GET_VIRTUAL_REG(param);
            auto &phy_para = GET_PHYSICAL_REG(param);
            auto &initial_loc = callee_iface.param_locs_[i];
            if (initial_loc.get_type()
                    == abi_value_location::tag_type::REGISTER) {
                phy_para = initial_loc.get_register();
            } else {
                virt_para.set_spilled();
            }
        }
        return xbyak_visitor_t::visit(std::move(v));
    }

    expr_c visit(indexing_c v) override {
        GET_VIRTUAL_REG(v).set_spilled();
        return xbyak_visitor_t::visit(std::move(v));
    }

    expr_c visit(func_addr_c v) override {
        GET_VIRTUAL_REG(v).set_spilled();
        return xbyak_visitor_t::visit(std::move(v));
    }

    expr_c visit(constant_c v) override {
        auto &virt_reg = GET_VIRTUAL_REG(v);
        virt_reg.set_type(get_virt_reg_type(v->dtype_));
        if (is_simd_data(v->dtype_) || FORCE_SIMD_ENCODE(v)) {
            virt_reg.set_spilled();
        }
        return v;
    }

private:
    reg_allocator_t *allocator_;
    std::vector<live_range_t> &call_scopes_;
    std::map<virtual_reg_t *, expr_c> &virt_reg_map_;

    bool is_local_scope_;
    abi_function_interface::ptr func_iface_;

    void set_func_args_hint(const std::vector<expr> &params) {
        auto &slots_map = allocator_->slots_map();
        for (size_t i = 0; i < params.size(); ++i) {
            auto &param = params[i];
            auto &virt_reg = GET_VIRTUAL_REG(param);
            auto &initial_loc = func_iface_->param_locs_[i];
            virt_reg.set_type(get_virt_reg_type(param->dtype_));
            if (initial_loc.get_type()
                    == abi_value_location::tag_type::REGISTER) {
                Xbyak::Reg src_reg = initial_loc.get_register();
                COMPILE_ASSERT(src_reg.isREG(64) || src_reg.isXMM(),
                        "Unhandled register kind: " << src_reg.toString());
                virt_reg.set_hint(virt_reg_hint::strong,
                        slots_map.get_reg_index(src_reg));
                if (i < 2) {
                    // arg is stream or modudle_data
                    virt_reg.spill_weight_ = spill_weight_const::initial;
                }
            }
        }
    }

    void set_func_return_hint(const expr &v) {
        auto &slots_map = allocator_->slots_map();
        auto &virt_reg = GET_VIRTUAL_REG(v);
        auto ret_reg = func_iface_->return_val_loc_.get_register();
        virt_reg.set_hint(
                virt_reg_hint::weak, slots_map.get_reg_index(ret_reg));
    }
};

/* *
 * Check address mode conflicts for each operation, resolve conflicts by
 * creating small live intervels and inserting load/store when nedded.
 * */
class spill_resolver_t : public xbyak_visitor_t {
public:
    using xbyak_visitor_t::dispatch;
    using xbyak_visitor_t::visit;

    spill_resolver_t(const live_range_t &spill_range,
            std::vector<virtual_reg_t *> &virtual_regs,
            std::set<virtual_reg_t *> &spilled_regs, bool alloc_local)
        : spill_range_(spill_range)
        , virtual_regs_(virtual_regs)
        , spilled_regs_(spilled_regs)
        , alloc_local_(alloc_local)
        , is_local_scope_(false) {}

    func_c operator()(func_c v) {
        return xbyak_visitor_t::dispatch(std::move(v));
    }

    stmt_c dispatch(stmt_c v) override {
        auto old_stmt_data = GET_STMT_DATA(v);
        cur_index_ = old_stmt_data.index_;
        // if current index out of spill_range, skip resolve
        if (spill_range_.defined_
                && (old_stmt_data.index_ < spill_range_.start_
                        || old_stmt_data.init_index_ > spill_range_.end_)) {
            return v;
        }
        // if current index in spill_range, try resolve
        auto ret = xbyak_visitor_t::dispatch(std::move(v));
        if (!ret->temp_data().isa<xbyak_stmt_data_t>()) {
            ret->temp_data() = old_stmt_data;
        }
        return ret;
    }

    stmt_c visit(stmts_c v) override {
        if (TRANSFORMED_CALL(v)) { is_local_scope_ = true; }
        std::vector<stmt> new_seq;
        for (auto &s : v->seq_) {
            // dispatch stmt
            auto ss = dispatch(s).remove_const();
            // remove optimized_out
            if (GET_STMT_DATA(ss).optimized_out_) { continue; }
            // insert before
            for (auto &&ts : insert_before_) {
                new_seq.push_back(std::move(ts));
            }
            insert_before_.clear();
            // insert middle
            new_seq.emplace_back(ss);
            // insert after
            for (auto &&ts : insert_after_) {
                new_seq.push_back(std::move(ts));
            }
            insert_after_.clear();
        }
        is_local_scope_ = false;
        return copy_attr(*v, make_stmt<stmts_node_t>(std::move(new_seq)));
    }

    stmt_c visit(assign_c v) override {
        resolve_dst_ = resolve_dst::none;
        dst_is_mem_ = is_spilled(v->var_);
        auto ret = xbyak_visitor_t::visit(std::move(v))
                           .remove_const()
                           .static_as<assign>();
        if (ret->value_.isa<call>()) { is_local_scope_ = false; }
        if (dst_is_mem_) {
            if (resolve_dst_ == resolve_dst::store) {
                return insert_store(std::move(ret), cur_index_);
            } else if (resolve_dst_ == resolve_dst::load_store) {
                return insert_load_store(std::move(ret), cur_index_);
            } else if (is_spilled(ret->value_)) {
                if (ret->var_.isa<var>()) {
                    return insert_store(std::move(ret), cur_index_);
                } else {
                    ret->value_
                            = insert_load(std::move(ret->value_), cur_index_);
                    return ret;
                }
            }
        }
        return ret;
    }

    // avoid dispatch into index dependent tensor
    expr_c visit(tensor_c v) override { return v; }

    // TODO(XXX): for_loop_c var spilled and begin/end/step
    // spilled(non-constant)
    stmt_c visit(for_loop_c v) override {
        // if var spilled, spilled begin/end/step must load
        return resolve_spill(std::move(v));
    }

    stmt_c visit(define_c v) override {
        // var = tensor_ptr, spilled var must be store from reg
        return resolve_spill(std::move(v));
    }

    expr_c visit(cast_c v) override {
        // var = cast(in), spilled var must be store from reg
        return resolve_spill(std::move(v));
    }

    expr_c visit(tensorptr_c v) override {
        // var = &ptr[idx], spilled var must be store from reg
        return resolve_spill(std::move(v));
    }

    expr_c visit(indexing_c v) override {
        // ptr[idx], spilled ptr and idx need to be load to reg
        return resolve_spill(std::move(v));
    }

    expr_c visit(xbyak_intrin_c v) override {
        // var = xbyak_intrin(args...), check intrin_format
        return resolve_spill(std::move(v));
    }

protected:
    stmt_c resolve_spill(for_loop_c v) {
        // TODO(XXX): temp_data missing
        auto stmt_data = v->get_temp_data();
        auto vv = xbyak_visitor_t::visit(std::move(v))
                          .remove_const()
                          .static_as<for_loop>();
        vv->temp_data() = stmt_data;
        bool loop_var_spilled = is_spilled(vv->var_);
        if (loop_var_spilled && is_spilled(vv->iter_begin_)) {
            auto begin_index = GET_STMT_INIT_INDEX(vv);
            vv->attr().set(attr_keys::load_loop_begin,
                    loop_var_load(vv->iter_begin_, begin_index));
        }
        if (loop_var_spilled && is_spilled(vv->iter_end_)) {
            auto cond_index = GET_STMT_INIT_INDEX(vv->body_) + 1;
            vv->attr().set(attr_keys::load_loop_end,
                    loop_var_load(vv->iter_end_, cond_index));
        }
        if (loop_var_spilled && is_spilled(vv->step_)) {
            auto step_index = GET_STMT_INDEX(vv->body_);
            vv->attr().set(attr_keys::load_loop_step,
                    loop_var_load(vv->step_, step_index));
        }
        return vv;
    }

    stmt_c resolve_spill(define_c v) {
        auto vv = xbyak_visitor_t::visit(std::move(v))
                          .remove_const()
                          .static_as<define>();
        if (vv->var_.isa<tensor>() && vv->init_.defined()
                && vv->init_.isa<tensorptr>() && is_spilled(vv->var_)) {
            vv->init_ = insert_load(std::move(vv->init_), cur_index_);
        }
        return vv;
    }

    expr_c resolve_spill(cast_c v) {
        // if operand_mem_sum({dst, in}) > 1
        // in allow mem, dst must be reg
        resolve_dst_ = resolve_dst::store;
        return v;
    }

    expr_c resolve_spill(tensorptr_c v) {
        resolve_dst_ = resolve_dst::store;
        return xbyak_visitor_t::visit(std::move(v));
    }

    expr_c resolve_spill(indexing_c x) {
        auto v = xbyak_visitor_t::visit(std::move(x))
                         .remove_const()
                         .static_as<indexing>();
        auto ptr = v->ptr_;
        auto idx = v->idx_.back();
        if (is_spilled(ptr)) {
            v->ptr_ = insert_load(std::move(ptr), cur_index_);
        }
        if (is_spilled(idx)) {
            v->idx_ = {insert_load(std::move(idx), cur_index_)};
        }
        return v;
    }

    expr_c resolve_spill(xbyak_intrin_c x) {
        auto v = xbyak_visitor_t::visit(std::move(x))
                         .remove_const()
                         .static_as<xbyak_intrin>();
        // resolve args load
        auto resolve = [&]() {
            size_t i = 0;
            for (auto &arg : v->args_) {
                if (is_spilled(arg)) { break; }
                i++;
            }
            v->args_[i] = insert_load(std::move(v->args_[i]), cur_index_);
            return v;
        };
        // resolve differnet format
        switch (v->format_) {
            case xbyak_intrin_format::undefined: {
                // no need to resolve
                resolve_dst_ = resolve_dst::none;
            } break;
            case xbyak_intrin_format::directed_all_reg: {
                // if operand_mem_sum({src, dst}) > 0
                // src must load to reg, dst must be reg
                resolve_dst_ = resolve_dst::store;
                if (spilled_args_sum(v->args_) > 0) { return resolve(); }
            } break;
            case xbyak_intrin_format::directed_end_mem: {
                // if operand_mem_sum({src, dst}) > 1
                // src allow 1 mem at the end of args, excluding imm
                // dst must be reg
                resolve_dst_ = resolve_dst::store;
                assert(v->args_.size() > 0);
                auto n = v->args_.size() - 1;
                if (n > 0 && is_imm(v->args_.back())) { n = n - 1; }
                if (spilled_args_sum(v->args_, n) > 0) { return resolve(); }
            } break;
            case xbyak_intrin_format::directed_dst_mem: {
                // if operand_mem_sum({src, dst}) > 1
                // when dst is mem, src must all load to reg
                // when dst is reg, src allow 1 mem
                resolve_dst_ = resolve_dst::none;
                if (dst_is_mem_ && spilled_args_sum(v->args_) > 0) {
                    return resolve();
                } else if (spilled_args_sum(v->args_) > 1) {
                    return resolve();
                }
            } break;
            case xbyak_intrin_format::directed_dst_reg: {
                // if operand_mem_sum({src, dst}) > 1
                // src allow 1 mem, dst must be reg
                resolve_dst_ = resolve_dst::store;
                if (spilled_args_sum(v->args_) > 1) { return resolve(); }
            } break;
            case xbyak_intrin_format::compound_dst_mem: {
                // if operand_mem_sum({src, dst}) > 1
                // when dst is mem, src must all load to reg
                // when dst is reg, src allow 1 mem
                resolve_dst_ = resolve_dst::none;
                if (dst_is_mem_ && spilled_args_sum(v->args_) > 0) {
                    return resolve();
                } else if (spilled_args_sum(v->args_) > 1) {
                    return resolve();
                }
            } break;
            case xbyak_intrin_format::compound_dst_reg: {
                // if operand_mem_sum({src, dst}) > 1
                // src allow 1 mem, dst must be reg
                resolve_dst_ = resolve_dst::load_store;
                if (spilled_args_sum(v->args_) > 1) { return resolve(); }
            } break;
        }
        // Special case of local tensor as all stack arg
        if (static_cast<xbyak_intrin_type>(v->type_)
                == xbyak_intrin_type::call_arg) {
            auto &arg = v->args_.back();
            auto &virt_reg = GET_VIRTUAL_REG(arg);
            if (dst_is_mem_ && virt_reg.buffered()) {
                auto node = builder::make_cast(sc_data_type_t::generic(), arg);
                v->args_ = {insert_load(std::move(node), cur_index_)};
                return v;
            } else if (dst_is_mem_ && const_exceed_32bit(arg)) {
                v->args_ = {insert_load(std::move(arg), cur_index_)};
                return v;
            }
        }
        return v;
    }

private:
    // index for load/store insert
    stmt_index_t get_index_load(stmt_index_t index) { return index - 1; }
    stmt_index_t get_index_store(stmt_index_t index) { return index + 1; }

    stmt loop_var_load(expr &old_expr, stmt_index_t index) {
        assert(old_expr.isa<var>());
        auto index_load = get_index_load(index);
        auto old_var = old_expr.static_as<var>();
        auto new_var = new_temp_var(
                old_var, "load_" + old_var->name_, index_load, index);
        // replace old loop var with new var
        old_expr = new_var;
        // return load assign stmt
        return new_temp_assign(new_var, old_var, index_load);
    }

    expr insert_load(expr old_expr, stmt_index_t index) {
        auto index_load = get_index_load(index);
        auto get_new_expr = [&]() {
            if (old_expr.isa<tensor>()) {
                auto old_ptr = old_expr.static_as<tensor>();
                auto new_ptr = new_temp_tensor(
                        old_ptr, "load_tensor_", index_load, index);
                return new_ptr;
            } else {
                auto old_var = old_expr.static_as<var>();
                auto new_var
                        = new_temp_var(old_var, "load_var_", index_load, index);
                return new_var;
            }
        };
        auto new_expr = get_new_expr();
        auto new_load
                = new_temp_assign(new_expr, std::move(old_expr), index_load);
        insert_before_.emplace_back(new_load);
        return new_expr;
    }

    stmt insert_store(assign vv, stmt_index_t index) {
        auto index_store = get_index_store(index);
        auto &var_range = GET_LIVE_RANGE(vv->var_);
        if (var_range.end_ < index_store) {
            SC_MODULE_WARN << "POTENTIAL ERROR, dead store to var: " << vv;
            var_range.update(index_store);
        }
        if (vv->var_.isa<tensor>()) {
            auto old_ptr = vv->var_.static_as<tensor>();
            auto new_ptr = new_temp_tensor(
                    old_ptr, "store_tensor_", index, index_store);
            auto new_store = new_temp_assign(old_ptr, new_ptr, index_store);
            insert_after_.emplace_back(new_store);
            vv->var_ = new_ptr;
        } else {
            auto old_var = vv->var_.static_as<var>();
            auto new_var
                    = new_temp_var(old_var, "store_var_", index, index_store);
            auto new_store = new_temp_assign(old_var, new_var, index_store);
            insert_after_.emplace_back(new_store);
            vv->var_ = new_var;
        }
        return vv;
    }

    stmt insert_load_store(assign vv, stmt_index_t index) {
        auto index_load = get_index_load(index);
        auto index_store = get_index_store(index);
        auto &var_range = GET_LIVE_RANGE(vv->var_);
        if (var_range.end_ < index_store) {
            SC_MODULE_WARN << "POTENTIAL ERROR, dead store to var: " << vv;
            var_range.update(index_store);
        }
        if (vv->var_.isa<tensor>()) {
            auto old_ptr = vv->var_.static_as<tensor>();
            auto new_ptr = new_temp_tensor(
                    old_ptr, "store_tensor_", index_load, index_store);
            auto new_load = new_temp_assign(new_ptr, old_ptr, index_load);
            insert_before_.emplace_back(new_load);
            auto new_store = new_temp_assign(old_ptr, new_ptr, index_store);
            insert_after_.emplace_back(new_store);
            vv->var_ = new_ptr;
        } else {
            auto old_var = vv->var_.static_as<var>();
            auto new_var = new_temp_var(
                    old_var, "store_var_", index_load, index_store);
            auto new_load = new_temp_assign(new_var, old_var, index_load);
            insert_before_.emplace_back(new_load);
            auto new_store = new_temp_assign(old_var, new_var, index_store);
            insert_after_.emplace_back(new_store);
            vv->var_ = new_var;
        }
        return vv;
    }

    expr new_temp_var(const expr &old_var, const std::string &prefix,
            stmt_index_t start, stmt_index_t end) {
        // new var
        auto new_var = builder::make_var(
                old_var->dtype_, prefix + std::to_string(temp_index_++));
        // set xbyak_expr_data_t
        new_var->temp_data() = xbyak_expr_data_t();
        // set virt_reg
        auto &new_virt_reg = GET_VIRTUAL_REG(new_var);
        new_virt_reg.type_ = get_virt_reg_type(new_var->dtype_);
        new_virt_reg.spill_weight_ = spill_weight_const::infinity;
        new_virt_reg.live_range_ = live_range_t(start, end);
        // add to new virtual_regs
        if (is_local_scope_ == alloc_local_) {
            new_virt_reg.set_unassigned();
            virtual_regs_.push_back(&new_virt_reg);
        }
        return new_var;
    }

    expr new_temp_tensor(const tensor &old_tsr, const std::string &prefix,
            stmt_index_t start, stmt_index_t end) {
        // new var
        auto new_tensor = builder::make_tensor(
                prefix + old_tsr->name_, old_tsr->dims_, old_tsr->elem_dtype_);
        // set xbyak_expr_data_t
        new_tensor->temp_data() = xbyak_expr_data_t();
        // set virt_reg
        auto &new_virt_reg = GET_VIRTUAL_REG(new_tensor);
        new_virt_reg.type_ = get_virt_reg_type(new_tensor->dtype_);
        new_virt_reg.spill_weight_ = spill_weight_const::infinity;
        new_virt_reg.live_range_ = live_range_t(start, end);
        // add to new virtual_regs
        if (is_local_scope_ == alloc_local_) {
            new_virt_reg.set_unassigned();
            virtual_regs_.push_back(&new_virt_reg);
        }
        return new_tensor;
    }

    stmt new_temp_assign(expr var, expr value, stmt_index_t index) {
        // new var
        auto new_assign
                = make_stmt<assign_node_t>(std::move(var), std::move(value));
        // set xbyak_stmt_data_t
        new_assign->temp_data() = xbyak_stmt_data_t(loop_depth());
        GET_STMT_INDEX(new_assign) = index;
        GET_STMT_INIT_INDEX(new_assign) = index;
        return new_assign;
    }

    int spilled_args_sum(const std::vector<expr> &args) {
        int sum = 0;
        for (auto &v : args) {
            sum += is_spilled(v) ? 1 : 0;
        }
        return sum;
    }

    int spilled_args_sum(const std::vector<expr> &args, size_t n) {
        int sum = 0;
        for (size_t i = 0; i < n; i++) {
            sum += is_spilled(args[i]) ? 1 : 0;
        }
        return sum;
    }

    bool is_imm(const expr &v) {
        return v.isa<constant>() && GET_VIRTUAL_REG(v).disabled();
    }

    bool is_spilled(const expr &v) {
        auto &virt_reg = GET_VIRTUAL_REG(v);
        if (alloc_local_) {
            return spilled_regs_.find(&virt_reg) != spilled_regs_.end();
        } else {
            return virt_reg.spilled();
        }
    }

    std::vector<stmt> insert_before_;
    std::vector<stmt> insert_after_;

    const live_range_t &spill_range_;
    std::vector<virtual_reg_t *> &virtual_regs_;
    std::set<virtual_reg_t *> &spilled_regs_;

    enum class resolve_dst {
        none,
        store,
        load_store,
    } resolve_dst_
            = resolve_dst::none;

    bool dst_is_mem_ = false;

    bool alloc_local_;
    bool is_local_scope_;

    stmt_index_t cur_index_ = 0;

    static int32_t temp_index_;
};

int32_t spill_resolver_t::temp_index_ = 0;

/* *
 * Prepare local allocation in call scpoe.
 * */
class call_scope_info_t : public xbyak_visitor_t {
public:
    using xbyak_visitor_t::dispatch;
    using xbyak_visitor_t::visit;

    call_scope_info_t(
            reg_allocator_t *allocator, const live_range_t &call_range)
        : allocator_(allocator), call_range_(call_range) {}

    func_c operator()(func_c v) {
        return xbyak_visitor_t::dispatch(std::move(v));
    }

    void swap_caller_save(std::vector<expr_c> &caller_save) {
        caller_save_->swap(caller_save);
    }

    void swap_local_spill(std::vector<expr_c> &local_spill) {
        local_spill_->swap(local_spill);
    }

    expr_c dispatch(expr_c v) override {
        // If need reg allocation
        if (v.isa<var>() || v.isa<tensor>()) {
            auto &virt_reg = GET_VIRTUAL_REG(v);
            auto &phy_reg = GET_PHYSICAL_REG(v);
            if (virt_reg.stat_ == virt_reg_stat::disabled) {
                prepare_virtual_reg(allocator_, &virt_reg, phy_reg, v->dtype_);
            }
            return v;
        }
        return xbyak_visitor_t::dispatch(std::move(v));
    }

    stmt_c dispatch(stmt_c v) override {
        auto stmt_data = GET_STMT_DATA(v);
        // if current index out of call_range, skip
        if (stmt_data.index_ < call_range_.start_
                || stmt_data.init_index_ > call_range_.end_) {
            return v;
        }
        return xbyak_visitor_t::dispatch(std::move(v));
    }

    stmt_c visit(stmts_c v) override {
        if (TRANSFORMED_CALL(v)) {
            auto vv = v.remove_const();
            local_spill_ = std::make_shared<std::vector<expr_c>>();
            caller_save_ = std::make_shared<std::vector<expr_c>>();
            vv->attr().set(attr_keys::local_spilled, local_spill_);
            vv->attr().set(attr_keys::caller_saved, caller_save_);
        }
        return xbyak_visitor_t::visit(std::move(v));
    }

private:
    reg_allocator_t *allocator_;
    const live_range_t &call_range_;

    std::shared_ptr<std::vector<expr_c>> caller_save_;
    std::shared_ptr<std::vector<expr_c>> local_spill_;
};

/* *
 * The actual allocator that run allocation pass for each func_c
 * */
class register_allocation_impl_t : public reg_allocator_t {
public:
    register_allocation_impl_t(const x86_64::target_profile_t &profile)
        : reg_allocator_t(profile) {}

    // Fuction pass for register_allocation_impl
    func_c operator()(func_c v) {
        func_ = pre_allocation(std::move(v));
        resolve_spill(live_range_t());
        run_allocator();
        set_global_spilled();
        for (auto &scope : call_scopes_) {
            allocate_call_scope(scope);
        }
        set_register_usage();
        return std::move(func_);
    }

    // Enqueue all virtual regs and prepare for spill insertion
    func_c pre_allocation(func_c v) {
        pre_allocation_t pre_allocation(this, call_scopes_, virt_reg_map_);
        return pre_allocation(std::move(v));
    }

    // Check intrin format and create load/store
    void resolve_spill_impl(const live_range_t &spill_range,
            std::vector<virtual_reg_t *> &virtual_regs,
            bool alloc_local) override {
        auto &spilled_regs = spilled_virt_regs();
        spill_resolver_t spill_resolver(
                spill_range, virtual_regs, spilled_regs, alloc_local);
        func_ = spill_resolver(std::move(func_));
    }

    void set_global_spilled() {
        func_t func = std::const_pointer_cast<func_base>(func_);
        func->attr().set(attr_keys::global_spilled, spilled_expr_vec());
    }

    void set_register_usage() {
        func_t func = std::const_pointer_cast<func_base>(func_);
        func->attr().set(
                attr_keys::register_usage, slots_array().utilized_slots());
    }

    void allocate_call_scope(const live_range_t &call_scope) {
        // Gather all caller saved expr
        auto caller_save = extract_caller_save(call_scope);
        // Prepare call scope for local allocation
        call_scope_info_t call_scope_info(this, call_scope);
        func_ = call_scope_info(std::move(func_));
        // run local allocation
        run_allocator(call_scope);
        // Set caller save and local spill
        auto local_spill = spilled_expr_vec();
        call_scope_info.swap_local_spill(local_spill);
        call_scope_info.swap_caller_save(caller_save);
    }

    std::vector<expr_c> extract_caller_save(const live_range_t &range) {
        // Go through all slots index, divide caller saved reg intervals
        auto divide_caller_save = [this](const virt_reg_type &reg_type,
                                          const live_range_t &range) {
            auto &slots_index = slots_map().get_slots_index(reg_type);
            auto &callee_save = slots_map().get_callee_save(reg_type);
            for (auto &i : slots_index) {
                if (callee_save.find(i) == callee_save.end()) {
                    auto virt_reg = slots_array().encompassing_reg(range, i);
                    if (virt_reg) { spill(virt_reg, range); }
                }
            }
        };
        divide_caller_save(virt_reg_type::gp_reg, range);
        divide_caller_save(virt_reg_type::fp_reg, range);
        if (require_resolve()) { resolve_spill(range, true); }
        // return all caller_saved
        return spilled_expr_vec();
    }

    std::vector<expr_c> spilled_expr_vec() {
        std::vector<expr_c> ret_vec;
        auto &spilled = spilled_virt_regs();
        ret_vec.reserve(spilled.size());
        for (auto &virt_reg : spilled) {
            ret_vec.push_back(virt_reg_map_[virt_reg]);
        }
        spilled.clear();
        return ret_vec;
    }

private:
    func_c func_;
    std::vector<live_range_t> call_scopes_;
    std::map<virtual_reg_t *, expr_c> virt_reg_map_;
};

func_c register_allocation_t::operator()(func_c v) {
    if (v->name_.find("_should_inline_") != std::string::npos) { return v; }
    register_allocation_impl_t reg_allocator(profile_);
    return reg_allocator(std::move(v));
}

register_allocation_t::register_allocation_t(
        const x86_64::target_profile_t &profile)
    : profile_(profile) {}

} // namespace sc_xbyak
} // namespace sc
