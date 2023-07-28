/*******************************************************************************
 * Copyright 2023 Intel Corporation
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
#include <algorithm>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "shared_include.hpp"

using namespace llvm;
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

void codegen_llvm_vis_t::generate_codeblock(
        const stmt_c &v, BasicBlock *current, BasicBlock *cont) {
    builder_.SetInsertPoint(current);
    dispatch(v);
    current = builder_.GetInsertBlock();
    if (current->empty() || !llvm::isa<llvm::ReturnInst>(current->back())) {
        builder_.CreateBr(cont);
    }
}

void codegen_llvm_vis_t::view(assign_c v) {
    auto val = generate_expr(v->value_);
    is_lvalue_mode_ = true;
    auto ptr = generate_expr(v->var_);
    if (v->var_->dtype_.lanes_ > 1) {
        ptr = builder_.CreatePointerCast(
                ptr, get_type(v->var_->dtype_)->getPointerTo());
    }
    if (v->var_->dtype_.is_pointer()
            && v->value_->dtype_.type_code_ != v->var_->dtype_.type_code_) {
        val = builder_.CreatePointerCast(val, get_type(v->var_->dtype_));
    }
    if (v->value_->dtype_.lanes_ > 1 && v->var_.isa<indexing>()) {
        // assigning to tensor
        if (v->var_.static_as<indexing>()->mask_.defined()) {
            Value *mask;
            auto v_index = v->var_.static_as<indexing>();
            auto bit_len = utils::get_sizeof_type(v_index->mask_->dtype_) * 8;
            if (v_index->dtype_.lanes_ != bit_len) {
                COMPILE_ASSERT(v_index->dtype_.lanes_ == 4,
                        "Currently only 8bit -> 4bit is supported.");
                mask = convert_mask(v_index->mask_, true);
            } else {
                mask = convert_mask(v_index->mask_);
            }
#if SC_LLVM_BACKEND > 10
            set_alias(builder_.CreateMaskedStore(
                              val, ptr, SC_LLVM_ALIGN(1), mask),
                    v->var_);
#else
            set_alias(builder_.CreateMaskedStore(val, ptr, 1, mask), v->var_);
#endif
        } else {
            set_alias(builder_.CreateAlignedStore(val, ptr, SC_LLVM_ALIGN(1)),
                    v->var_);
        }
    } else {
        set_alias(builder_.CreateStore(val, ptr), v->var_);
    }
}

void codegen_llvm_vis_t::view(if_else_c v) {
    auto cond = generate_expr(v->condition_);
    BasicBlock *tb = BasicBlock::Create(context_, "if_t", current_func_);
    BasicBlock *cb = BasicBlock::Create(context_, "if_cont", current_func_);
    BasicBlock *fb = v->else_case_.defined()
            ? BasicBlock::Create(context_, "if_f", current_func_)
            : cb;
    fb->moveBefore(cb);
    builder_.CreateCondBr(cond, tb, fb);
    generate_codeblock(v->then_case_, tb, cb);
    if (fb != cb) { generate_codeblock(v->else_case_, fb, cb); }
    builder_.SetInsertPoint(cb);
}

void codegen_llvm_vis_t::view(returns_c v) {
    if (v->value_.defined()) {
        builder_.CreateRet(generate_expr(v->value_));
    } else {
        builder_.CreateRetVoid();
    }
}

void codegen_llvm_vis_t::view(define_c v) {
    COMPILE_ASSERT(v->linkage_ != linkage::static_local
                    && v->linkage_ != linkage::private_global,
            "LLVM backend cannot handle non-local variable "
            "definitions");
    if (v->var_.isa<var>()) {
        auto thevar = v->var_.static_as<var>();
        if (thevar->attr_
                && thevar->attr_->has_key(attr_keys::module_global_offset)) {
            // if it is a global variable that is lowered to local
            auto &offset
                    = thevar->attr_->get_any(attr_keys::module_global_offset);
            Value *ptr;
            if (auto absptr = offset.get_or_null<void *>()) {
                ptr = builder_.CreateIntToPtr(
                        builder_.getInt64(reinterpret_cast<uint64_t>(*absptr)),
                        get_type(thevar->dtype_)->getPointerTo(),
                        thevar->name_);
            } else {
                auto module_ptr = current_func_->arg_begin() + 1;
                assert(module_ptr->getName() == "__module_data_arg");

                ptr = builder_.CreateGEP(builder_.getInt8Ty(), module_ptr,
                        builder_.getInt64(offset.get<size_t>()));
                ptr = builder_.CreatePointerCast(ptr,
                        get_type(thevar->dtype_)->getPointerTo(),
                        thevar->name_);
            }
            var_ptr_in_func_.insert(std::make_pair(thevar, ptr));
            set_dbg_info_for_local_var(v.get(), thevar->name_, ptr, false);
        } else {
            Value *init_v = nullptr;
            if (v->init_.defined()) { init_v = generate_expr(v->init_); }
            auto retv = define_var(thevar, init_v);
            set_dbg_info_for_local_var(v.get(), thevar->name_, retv, false);
        }
    } else if (v->var_.isa<tensor>()) {
        tensor t = v->var_.static_as<tensor>();
        if (auto alias_info = alias_info::get_alias_info(*t)) {
            auto alias_itr = alias_set_to_alias_scope_.find(alias_info);
            if (alias_itr != alias_set_to_alias_scope_.end()) {
                tsr_to_alias_scope_[t] = &alias_itr->second;
            }
        }
        // if it is a view of the rescheduled buffer/ local tensor on
        // heap
        if (v->init_.defined()) {
            Value *ptr = generate_expr(v->init_);
            ptr = builder_.CreatePointerCast(
                    ptr, get_type(t->elem_dtype_)->getPointerTo(), t->name_);
            var_ptr_in_func_.insert(std::make_pair(t, ptr));
            set_dbg_info_for_local_var(v.get(), t->name_, ptr, true);
            return;
        }

        // explicitly align tensor with cache line size, except that
        // tensor is a scalar or bytes size < 64.
        bool need_align = false;
        // check condition.
        if (t->dims_.size() == 1
                && get_const_as_int(t->dims_[0].checked_as<constant>()) == 1) {
            // it is a scalar
        } else {
            size_t shape = 1;
            for (auto &d : t->dims_) {
                shape *= get_const_as_int(d.checked_as<constant>());
            }
            size_t dtsize = utils::get_sizeof_etype(t->elem_dtype_.type_code_);
            // check bytes size
            if (shape * dtsize > 64) need_align = true;
        }
        auto ptr = builder_.CreateAlloca(get_type(t->elem_dtype_),
                generate_expr(t->dims_.front()), t->name_);
        // cache line alignment

        if (need_align) { ptr->setAlignment(SC_LLVM_ALIGN(64)); }

        var_ptr_in_func_.insert(std::make_pair(t, ptr));
        set_dbg_info_for_local_var(v.get(), t->name_, ptr, true);
    } else {
        assert(0 && "Bad var type");
    }
}

void codegen_llvm_vis_t::view(for_loop_c v) {
    COMPILE_ASSERT(v->kind_ == for_type::NORMAL,
            "LLVM backend can only handle normal for-loops");
    auto itr_v = define_var(v->var_, generate_expr(v->iter_begin_));

    if (ctx_->flags_.debug_info_) {
        auto pos = v->attr_->get_or_null<source_pos>("source_pos");
        if (pos) {
            set_dbg_info_for_local_var(pos, v->var_->dtype_,
                    v->var_.checked_as<var>()->name_, itr_v, false);
        }
    }

    BasicBlock *chk = BasicBlock::Create(context_, "for_check", current_func_);
    BasicBlock *body = BasicBlock::Create(context_, "for_body", current_func_);
    BasicBlock *cont = BasicBlock::Create(context_, "for_cont", current_func_);
    builder_.CreateBr(chk);
    {
        builder_.SetInsertPoint(chk);
        auto cate = get_type_category(v->var_->dtype_);
        auto end_v = generate_expr(v->iter_end_);
        auto itr_value = builder_.CreateLoad(get_type(v->var_->dtype_), itr_v);
        Value *cond;
        if (cate == CATE_INT) {
            cond = builder_.CreateICmpSLT(itr_value, end_v);
        } else {
            assert(cate == CATE_UINT);
            cond = builder_.CreateICmpULT(itr_value, end_v);
        }
        builder_.CreateCondBr(cond, body, cont);
    }
    {
        builder_.SetInsertPoint(body);
        dispatch(v->body_);
        if (body->empty() || !llvm::isa<llvm::ReturnInst>(body->back())) {
            auto step_v = generate_expr(v->step_);
            Value *itr_value
                    = builder_.CreateLoad(get_type(v->var_->dtype_), itr_v);
            itr_value = builder_.CreateAdd(itr_value, step_v);
            builder_.CreateStore(itr_value, itr_v);
            builder_.CreateBr(chk);
        }
    }
    cont->moveAfter(builder_.GetInsertBlock());
    builder_.SetInsertPoint(cont);
}
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
