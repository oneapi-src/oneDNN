/*******************************************************************************
 * Copyright 2020-2022 Intel Corporation
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

#include "codegen_llvm.hpp"
#include "precodegen_passes.hpp"
#include <compiler/ir/pass/printer.hpp>
#include <compiler/ir/transform/constant_fold.hpp>
#include <compiler/ir/transform/module_globals_resolve.hpp>
#include <compiler/ir/transform/pointer_alias_info.hpp>
#include <compiler/ir/viewer.hpp>
#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DIBuilder.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/MDBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/Support/Host.h>
#if SC_LLVM_BACKEND > 13
#include <llvm/MC/TargetRegistry.h>
#else
#include <llvm/Support/TargetRegistry.h>
#endif
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <unordered_set>
#include <util/any_map.hpp>
#include <util/scoped_timer.hpp>
#include <util/unique_file_name.hpp>

#if SC_LLVM_BACKEND > 8
#include <llvm/IR/IntrinsicsX86.h>
#endif

SC_MODULE(codegen.llvm);

#if SC_LLVM_BACKEND > 8
#define SC_LLVM_ALIGN(a) Align(a)
#else
#define SC_LLVM_ALIGN(a) (a)
#endif

using namespace llvm;
namespace sc {
std::unique_ptr<TargetMachine> get_llvm_target_machine(
        CodeGenOpt::Level optlevel = CodeGenOpt::Level::Default) {
    auto target_triple = sys::getProcessTriple();

    std::string err;
    auto target = TargetRegistry::lookupTarget(target_triple, err);
    if (!target) { throw std::runtime_error(err); }

    TargetOptions opt;
    auto reloc_model = Optional<Reloc::Model>(Reloc::Static);
    auto tm = target->createTargetMachine(target_triple, sys::getHostCPUName(),
            /*Features*/ "", opt, reloc_model, llvm::None, optlevel, true);
    return std::unique_ptr<TargetMachine>(tm);
}

static void print_helper(Value *v) {
    v->print(llvm::errs());
}
class codegen_llvm_vis_t : public ir_viewer_t {
public:
    context_ptr ctx_;
    LLVMContext &context_;
    IRBuilder<> builder_;
    std::unique_ptr<Module> module_;
    std::unique_ptr<DIBuilder> dbuilder_;
    DICompileUnit *dbg_cu_ = nullptr;
    std::vector<DIScope *> dbg_scopes_;
    Function *current_func_ = nullptr;
    Value *current_val_ = nullptr;
    // the **pointer** of local var in a function
    std::unordered_map<expr_c, Value *> var_ptr_in_func_;
    // tensor to <alias scope, noalias>
    std::unordered_map<expr_c, std::pair<MDNode *, MDNode *> *>
            tsr_to_alias_scope_;
    std::unordered_map<alias_info::alias_set_t *, std::pair<MDNode *, MDNode *>>
            alias_set_to_alias_scope_;
    std::unordered_map<std::string, Function *> name_to_func_;
    bool is_lvalue_mode_ = false;

    codegen_llvm_vis_t(const context_ptr &ctx, LLVMContext &context,
            const std::string &source_dir, const std::string &source_file_name)
        : ctx_(ctx)
        , context_(context)
        , builder_(context_)
        , module_(utils::make_unique<Module>("name", context_))
        , dbuilder_(utils::make_unique<DIBuilder>(*module_)) {
        static bool initialized = []() {
            // make sure LLVM native targets are initialized once and avoid race
            // condition
            InitializeNativeTarget();
            InitializeNativeTargetAsmParser();
            InitializeNativeTargetAsmPrinter();
            return true;
        }();
        SC_UNUSED(initialized);
        auto tm = get_llvm_target_machine();
        module_->setTargetTriple(tm->getTargetTriple().str());
        module_->setDataLayout(tm->createDataLayout());
        FastMathFlags fmflag;
        fmflag.setFast(true);
        fmflag.setAllowContract(false);
        builder_.setFastMathFlags(fmflag);
        if (ctx->flags_.debug_info_) {
            dbg_cu_ = dbuilder_->createCompileUnit(dwarf::DW_LANG_C,
                    dbuilder_->createFile(source_file_name, source_dir),
                    "oneDNN Graph Compiler", false, "", 0);

            if (!tm->getTargetTriple().isOSWindows()) {
                // Add the current debug info version into the module.
                module_->addModuleFlag(Module::Warning, "Debug Info Version",
                        DEBUG_METADATA_VERSION);

                // Darwin only supports dwarf2.
                if (tm->getTargetTriple().isOSDarwin())
                    module_->addModuleFlag(
                            llvm::Module::Warning, "Dwarf Version", 2);
            } else {
                module_->addModuleFlag(llvm::Module::Warning, "CodeView", 1);
            }
        }
    }

    using ir_viewer_t::dispatch;
    using ir_viewer_t::view;

    expr_c dispatch(expr_c v) override {
        emit_location(v.get());
        return ir_viewer_t::dispatch(v);
    }

    stmt_c dispatch(stmt_c v) override {
        emit_location(v.get());
        return ir_viewer_t::dispatch(v);
    }

    const std::string &get_node_name(const expr_c &c) {
        if (c.isa<var>()) { return c.static_as<var>()->name_; }
        return c.checked_as<tensor>()->name_;
    }

    FunctionType *create_func_type(const func_c &v) {
        std::vector<Type *> tys;
        for (auto &param : v->params_) {
            tys.push_back(get_type(param->dtype_));
        }
        FunctionType *FT
                = FunctionType::get(get_type(v->ret_type_), tys, false);
        return FT;
    }

    DISubroutineType *create_func_dtype(const func_c &v) {
        std::vector<Metadata *> tys {get_type_both(v->ret_type_).second};
        for (auto &param : v->params_) {
            tys.push_back(get_type_both(param->dtype_).second);
        }
        return dbuilder_->createSubroutineType(
                dbuilder_->getOrCreateTypeArray(tys));
    }

    Function *get_or_create_func(const func_c &v) {
        auto itr = name_to_func_.find(v->name_);
        if (itr != name_to_func_.end()) { return itr->second; }
        auto FT = create_func_type(v);
        bool is_private = v->attr_
                && v->attr_->get_or_else(function_attrs::private_, false);
        Function *F = Function::Create(FT,
                is_private ? Function::InternalLinkage
                           : Function::ExternalLinkage,
                v->name_, module_.get());
        for (size_t i = 0; i < v->params_.size(); i++) {
            (F->arg_begin() + i)
                    ->setName(get_node_name(v->params_[i]) + "_arg");
        }
        name_to_func_.insert(std::make_pair(v->name_, F));
        if (v->attr_ && v->attr_->get_or_else(function_attrs::pure, false)) {
            F->addFnAttr(llvm::Attribute::AttrKind::ReadNone);
            F->addFnAttr(llvm::Attribute::AttrKind::Speculatable);
        }
        if (v->attr_
                && v->attr_->get_or_else(function_attrs::no_alias, false)) {
            F->setReturnDoesNotAlias();
        }
        F->addFnAttr(llvm::Attribute::AttrKind::NoUnwind);
        return F;
    }

    Value *generate_expr(const expr_c &e) {
        dispatch(e);
        return current_val_;
    }

    std::unordered_map<uint64_t, std::pair<Type *, DIType *>> type_cache_;
    std::pair<Type *, DIType *> do_get_type(sc_data_type_t dtype) {
        Type *ty = nullptr;
        DIType *dty = nullptr;
        if (dtype.is_etype_pointer()
                && dtype.type_code_ != sc_data_etype::POINTER) {
            auto ret = do_get_type(dtype.get_pointer_element());
            return {ret.first->getPointerTo(),
                    dbuilder_->createPointerType(ret.second, 64)};
        }
        switch (dtype.type_code_) {
            case sc_data_etype::UNDEF:
                throw std::runtime_error("Unsupported dtype");
            case sc_data_etype::BF16:
                ty = builder_.getInt16Ty();
                dty = dbuilder_->createBasicType(
                        "bf16", 16, dwarf::DW_ATE_unsigned);
                break;
            case sc_data_etype::F16:
                ty = builder_.getHalfTy();
                dty = dbuilder_->createBasicType(
                        "f16", 16, dwarf::DW_ATE_unsigned);
                break;
            case sc_data_etype::U16:
                ty = builder_.getInt16Ty();
                dty = dbuilder_->createBasicType(
                        "u16", 16, dwarf::DW_ATE_unsigned);
                break;
            case sc_data_etype::F32:
                ty = builder_.getFloatTy();
                dty = dbuilder_->createBasicType(
                        "f32", 32, dwarf::DW_ATE_float);
                break;
            case sc_data_etype::S32:
                ty = builder_.getInt32Ty();
                dty = dbuilder_->createBasicType(
                        "s32", 32, dwarf::DW_ATE_signed);
                break;
            case sc_data_etype::U32:
                ty = builder_.getInt32Ty();
                dty = dbuilder_->createBasicType(
                        "u32", 32, dwarf::DW_ATE_unsigned);
                break;
            case sc_data_etype::S8:
                ty = builder_.getInt8Ty();
                dty = dbuilder_->createBasicType("s8", 8, dwarf::DW_ATE_signed);
                break;
            case sc_data_etype::U8:
                ty = builder_.getInt8Ty();
                dty = dbuilder_->createBasicType(
                        "u8", 8, dwarf::DW_ATE_unsigned);
                break;
            case sc_data_etype::INDEX:
            case sc_data_etype::GENERIC:
                ty = builder_.getInt64Ty();
                dty = dbuilder_->createBasicType(
                        "u64", 64, dwarf::DW_ATE_unsigned);
                break;
            case sc_data_etype::BOOLEAN:
                ty = builder_.getInt1Ty();
                dty = dbuilder_->createBasicType(
                        "bool", 1, dwarf::DW_ATE_unsigned);
                break;
            case sc_data_etype::VOID_T:
                ty = builder_.getVoidTy();
                dty = dbuilder_->createBasicType(
                        "void", 0, dwarf::DW_ATE_address);
                break;
            case sc_data_etype::POINTER:
                ty = builder_.getInt8PtrTy();
                dty = dbuilder_->createBasicType(
                        "pointer", 64, dwarf::DW_ATE_address);
                break;

            default: assert("Unreachable" && 0); break;
        }
        if (dtype.lanes_ > 1) {
#if SC_LLVM_BACKEND > 10
            ty = VectorType::get(ty, dtype.lanes_, false);
#else
            ty = VectorType::get(ty, dtype.lanes_);
#endif

            auto subscript = dbuilder_->getOrCreateSubrange(0, dtype.lanes_);
            llvm::DINodeArray subscriptarray
                    = dbuilder_->getOrCreateArray(subscript);
            dty = dbuilder_->createVectorType(
                    utils::get_sizeof_type(dtype) * 8, 8, dty, subscriptarray);
        }
        return {ty, dty};
    }

    std::pair<Type *, DIType *> get_type_both(sc_data_type_t dtype) {
        auto itr = type_cache_.find(dtype);
        if (itr != type_cache_.end()) { return itr->second; }
        auto ret = do_get_type(dtype);
        type_cache_.insert(std::make_pair(dtype, ret));
        return ret;
    }

    Type *get_type(sc_data_type_t dtype) { return get_type_both(dtype).first; }

    Value *get_defined_var_ptr(const expr_c &e) {
        auto itr = var_ptr_in_func_.find(e);
        assert(itr != var_ptr_in_func_.end());
        return itr->second;
    }

    Value *define_var(const expr_c &e, Value *initv) {
        auto ptr = builder_.CreateAlloca(
                get_type(e->dtype_), nullptr, e.checked_as<var>()->name_);
        if (initv) builder_.CreateStore(initv, ptr);
        var_ptr_in_func_.insert(std::make_pair(e, ptr));
        return ptr;
    }

    void set_dbg_info_for_func_arg(llvm::Value *v, DISubprogram *SP,
            DIFile *dunit, sc_data_type_t type, const std::string &name,
            int argidx, int lineno, bool need_ref) {
        if (!ctx_->flags_.debug_info_) { return; }
        auto types = get_type_both(type);
        auto dbgtype = types.second;
        if (need_ref) {
            auto tmp = builder_.CreateAlloca(v->getType());
            builder_.CreateStore(v, tmp);
            v = tmp;
        }
        // Create a debug descriptor for the variable.
        DILocalVariable *D = dbuilder_->createParameterVariable(
                SP, name, argidx, dunit, lineno, dbgtype, true);

        dbuilder_->insertDeclare(v, D, dbuilder_->createExpression(),
                DILocation::get(SP->getContext(), lineno, 0, SP),
                builder_.GetInsertBlock());
    }

    void emit_location(const node_base *p) {
        if (!ctx_->flags_.debug_info_) { return; }
        if (!p) { return builder_.SetCurrentDebugLocation(DebugLoc()); }
        if (p->attr_) {
            if (auto loc = p->attr_->get_or_null<source_pos>("source_pos")) {
                DIScope *Scope;
                if (dbg_scopes_.empty())
                    Scope = dbg_cu_;
                else
                    Scope = dbg_scopes_.back();
                builder_.SetCurrentDebugLocation(DILocation::get(
                        Scope->getContext(), loc->line_, loc->pos_, Scope));
            }
        }
    }

    func_c dispatch(func_c v) override {
        var_ptr_in_func_.clear();
        tsr_to_alias_scope_.clear();
        alias_set_to_alias_scope_.clear();
        if (utils::string_startswith(v->name_, "_should_inline_")) { return v; }
        if (!v->body_.defined()) { return v; }
        auto F = get_or_create_func(v);
        BasicBlock *BB = BasicBlock::Create(context_, "entry", F);
        builder_.SetInsertPoint(BB);

        unsigned LineNo = 1;
        unsigned ScopeLine = 1;
        DIFile *dunit = nullptr;
        DISubprogram *SP = nullptr;
        if (ctx_->flags_.debug_info_) {
            auto pos = v->attr_->get<source_pos>("source_pos");

            LineNo = pos.line_;
            ScopeLine = LineNo;
            dunit = dbuilder_->createFile(
                    dbg_cu_->getFilename(), dbg_cu_->getDirectory());

            SP = dbuilder_->createFunction(dunit, v->name_, StringRef(), dunit,
                    LineNo, create_func_dtype(v), ScopeLine,
                    DINode::FlagPrototyped, DISubprogram::SPFlagDefinition);
            F->setSubprogram(SP);

            // Push the current scope.
            dbg_scopes_.push_back(SP);

            // Unset the location for the prologue emission (leading
            // instructions with no location in a function are considered part
            // of the prologue and the debugger will run past them when breaking
            // on a function)
            emit_location(nullptr);
        }

        current_func_ = F;
        F->addFnAttr("no-frame-pointer-elim", "true");
        F->addFnAttr("frame-pointer", "all");
        bool has_alias = false;
        MDBuilder MDB(context_);
        MDNode *alias_domain = nullptr;
        auto add_alias_group = [&alias_domain, &v, &MDB, this](
                                       alias_info::alias_set_t *alias_set,
                                       const std::string &tsrname)
                -> std::pair<MDNode *, MDNode *> * {
            if (!alias_domain) {
                alias_domain = MDB.createAnonymousAliasScopeDomain(v->name_);
            }
            auto itr = alias_set_to_alias_scope_.find(alias_set);
            if (itr == alias_set_to_alias_scope_.end()) {
                MDNode *scope
                        = MDB.createAnonymousAliasScope(alias_domain, tsrname);
                auto &ret = (alias_set_to_alias_scope_[alias_set]
                        = std::make_pair(scope, nullptr));
                return &ret;
            }
            return &itr->second;
        };
        for (size_t i = 0; i < v->params_.size(); i++) {
            if (v->params_[i].isa<tensor>()) {
                auto ainfo = alias_info::get_alias_info(*v->params_[i]);
                if (ainfo && !ainfo->has_no_alias()) {
                    add_alias_group(ainfo->get_alias_set().get(),
                            v->params_[i].static_as<tensor>()->name_);
                } else {
                    F->addParamAttr(i, llvm::Attribute::AttrKind::NoAlias);
                }

                F->addParamAttr(i, llvm::Attribute::AttrKind::NoCapture);
                F->addParamAttr(i, llvm::Attribute::AttrKind::NonNull);
            }
        }
        if (v->attr_) {
            if (auto local_tsr_alias_set
                    = v->attr_->get_or_null<std::vector<
                              std::shared_ptr<alias_info::alias_set_t>>>(
                            "alias_sets")) {
                int cnt = 0;
                for (auto &aset : *local_tsr_alias_set) {
                    add_alias_group(aset.get(), std::to_string(cnt++));
                }
            }
        }

        // if has custom alias info, need to construct alias scope/noalias for
        // LLVM each alias scope are exclusive to each other
        if (alias_domain) {
            for (size_t i = 0; i < v->params_.size(); i++) {
                if (v->params_[i].isa<tensor>()) {
                    auto ainfo = alias_info::get_or_create_alias_info(
                            *v->params_[i]);
                    // need to add alias set for tensors with "NoAlias"
                    // attribute
                    auto llvm_alias
                            = add_alias_group(ainfo->get_alias_set().get(),
                                    v->params_[i].static_as<tensor>()->name_);
                    tsr_to_alias_scope_[v->params_[i]] = llvm_alias;
                }
            }
            // each alias scope is in noalias of others
            for (auto &kv : alias_set_to_alias_scope_) {
                auto cur_scope = kv.second.first;
                std::unordered_set<Metadata *> map_scopes;
                for (auto &kv2 : alias_set_to_alias_scope_) {
                    auto other_scope = kv2.second.first;
                    if (other_scope != cur_scope)
                        map_scopes.insert(other_scope);
                }
                SmallVector<Metadata *, 4> vec {
                        map_scopes.begin(), map_scopes.end()};
                // sort the map to make it stable for unit test
                std::sort(
                        vec.begin(), vec.end(), [](Metadata *&a, Metadata *&b) {
                            auto name_a = llvm::dyn_cast<MDNode>(a)
                                                  ->getOperand(2)
                                                  .get();
                            auto name_b = llvm::dyn_cast<MDNode>(b)
                                                  ->getOperand(2)
                                                  .get();
                            return llvm::dyn_cast<MDString>(name_a)->getString()
                                    < llvm::dyn_cast<MDString>(name_b)
                                              ->getString();
                        });
                // construct no-alias set, which is all other scopes
                kv.second.second = MDNode::get(context_, vec);
            }
        }
        bool is_low_level = v->attr_
                && v->attr_->get_or_else(function_attrs::low_level, false);
        // LLVM func args are SSA values and cannot be modified. We use alloca
        // to alloc modifiable slots for each params
        for (size_t i = 0; i < v->params_.size(); i++) {
            Value *arg = F->args().begin() + i;
            auto &p = v->params_[i];
            if (p.isa<var>()) {
                auto varnode = p.static_as<var>();
                if (is_low_level) {
                    auto varalloca = define_var(v->params_[i], arg);
                    set_dbg_info_for_func_arg(varalloca, SP, dunit, p->dtype_,
                            varnode->name_, i + 1, LineNo, false);
                } else {
                    switch (i) {
                        case 0:
                            assert(arg->getName() == "__stream_arg");
                            var_ptr_in_func_.insert(std::make_pair(p, arg));
                            set_dbg_info_for_func_arg(arg, SP, dunit, p->dtype_,
                                    varnode->name_, i + 1, LineNo, true);
                            break;
                        case 1:
                            assert(arg->getName() == "__module_data_arg");
                            var_ptr_in_func_.insert(std::make_pair(p, arg));
                            set_dbg_info_for_func_arg(arg, SP, dunit, p->dtype_,
                                    varnode->name_, i + 1, LineNo, true);
                            break;
                        default: {
                            auto varalloca = define_var(v->params_[i], arg);
                            set_dbg_info_for_func_arg(varalloca, SP, dunit,
                                    p->dtype_, varnode->name_, i + 1, LineNo,
                                    false);
                            break;
                        }
                    }
                }
            } else {
                assert(p.isa<tensor>());
                auto tnode = p.static_as<tensor>();
                var_ptr_in_func_.insert(std::make_pair(p, arg));
                set_dbg_info_for_func_arg(arg, SP, dunit, tnode->dtype_,
                        tnode->name_, i + 1, LineNo, true);
            }
        }
        dispatch(v->body_);
        if (builder_.GetInsertBlock()->empty()
                || !builder_.GetInsertBlock()->back().isTerminator()) {
            assert(v->ret_type_ == datatypes::void_t);
            builder_.CreateRetVoid();
        }
        if (ctx_->flags_.debug_info_) { dbg_scopes_.pop_back(); }
        return v;
    }

    void view(constant_c v) override {
        std::vector<Constant *> vals;
        vals.reserve(v->value_.size());
        auto cate = get_etype_category_nothrow(v->dtype_.type_code_);
        if (v->dtype_.type_code_ == sc_data_etype::BF16) { cate = CATE_UINT; }
        sc_data_type_t base_type = v->dtype_;
        base_type.lanes_ = 1;
        auto llvm_base_type = get_type(base_type);
        switch (cate) {
            case CATE_FLOAT: {
                for (auto &val : v->value_) {
                    vals.push_back(
                            ConstantFP::get(llvm_base_type, APFloat(val.f32)));
                }
            } break;
            case CATE_UINT:
            case CATE_INT: {
                bool is_signed = cate == CATE_INT;
                for (auto &val : v->value_) {
                    vals.push_back(ConstantInt::get(
                            llvm_base_type, val.u64, is_signed));
                }
            } break;
            default:
                COMPILE_ASSERT(v->dtype_ == datatypes::pointer,
                        "Unexpected type for LLVM. Expecting pointer.");
                for (auto &val : v->value_) {
                    vals.push_back(ConstantExpr::getIntToPtr(
                            builder_.getInt64(val.u64), llvm_base_type));
                }
                break;
        }
        if (vals.size() != v->dtype_.lanes_) {
            COMPILE_ASSERT(
                    vals.size() == 1, "Bad constant node. Expecting 1 value");
            // broadcast value
            current_val_ = builder_.CreateVectorSplat(
                    v->dtype_.lanes_, vals.front());
        } else {
            if (vals.size() != 1) {
                current_val_ = ConstantVector::get(vals);
            } else {
                current_val_ = vals.front();
            }
        }
    }
    void view(var_c v) override {
        auto ret = get_defined_var_ptr(v);
        bool is_special_params
                = (v->name_ == "__stream" || v->name_ == "__module_data");
        if (is_lvalue_mode_) {
            assert(!is_special_params);
            is_lvalue_mode_ = false;
            current_val_ = ret;
        } else {
            if (is_special_params) {
                current_val_ = ret;
            } else {
                current_val_ = builder_.CreateLoad(
                        ret->getType()->getPointerElementType(), ret,
                        v->name_ + "_v");
            }
        }
    }

    void view(cast_c v) override {
        auto cate_out = get_etype_category_nothrow(v->dtype_.type_code_);
        auto cate_in = get_etype_category_nothrow(v->in_->dtype_.type_code_);
        auto in_v = generate_expr(v->in_);
        auto outtype = get_type(v->dtype_);
        auto check_cate = [&v]() {
            COMPILE_ASSERT(v->dtype_ == datatypes::generic,
                    "Unexpected outtype " << v);
        };
        if (v->in_->dtype_.is_etype(sc_data_etype::F32)
                && v->dtype_.is_etype(sc_data_etype::BF16)) {
#if SC_LLVM_BACKEND > 10
            switch (v->in_->dtype_.lanes_) {
                case 1: {
                    Value *vec = builder_.CreateVectorSplat(4, in_v);
                    vec = builder_.CreateIntrinsic(
                            Intrinsic::x86_avx512bf16_mask_cvtneps2bf16_128, {},
                            {vec,
                                    UndefValue::get(
                                            get_type(sc_data_type_t::bf16(8))),
                                    /*mask*/
                                    builder_.CreateVectorSplat(
                                            4, builder_.getInt1(true))});
                    current_val_
                            = builder_.CreateExtractElement(vec, UINT64_C(0));
                } break;
                case 4:
                    current_val_ = builder_.CreateIntrinsic(
                            Intrinsic::x86_avx512bf16_mask_cvtneps2bf16_128, {},
                            {in_v,
                                    UndefValue::get(
                                            get_type(sc_data_type_t::bf16(8))),
                                    /*mask*/
                                    builder_.CreateVectorSplat(
                                            4, builder_.getInt1(true))});
                    current_val_ = builder_.CreateShuffleVector(
                            current_val_, {0, 1, 2, 3});
                    break;
                case 8:
                    current_val_ = builder_.CreateIntrinsic(
                            Intrinsic::x86_avx512bf16_cvtneps2bf16_256, {},
                            {in_v});
                    break;
                case 16:
                    current_val_ = builder_.CreateIntrinsic(
                            Intrinsic::x86_avx512bf16_cvtneps2bf16_512, {},
                            {in_v});
                    break;
                default:
                    std::stringstream ss;
                    ss << "Unsupport cast lanes " << v->in_->dtype_.lanes_;
                    throw std::runtime_error(ss.str());
            }
            return;
#else
            throw std::runtime_error("LLVM-8 cannot handle bf16");
#endif
        }
        switch (cate_in) {
            case CATE_FLOAT: {
                switch (cate_out) {
                    case CATE_FLOAT:
                        current_val_ = builder_.CreateFPCast(in_v, outtype);
                        break;
                    case CATE_INT:
                        current_val_ = builder_.CreateFPToSI(in_v, outtype);
                        break;
                    case CATE_UINT:
                        current_val_ = builder_.CreateFPToUI(in_v, outtype);
                        break;
                    case CATE_OTHER: {
                        check_cate();
                        auto bits = utils::get_sizeof_etype(
                                            v->in_->dtype_.type_code_)
                                * 8;
                        auto ret = builder_.CreateBitCast(
                                in_v, IntegerType::get(context_, bits));
                        current_val_ = builder_.CreateZExtOrBitCast(
                                ret, builder_.getInt64Ty());
                    } break;
                }
            } break;
            case CATE_INT: {
                switch (cate_out) {
                    case CATE_FLOAT:
                        current_val_ = builder_.CreateSIToFP(in_v, outtype);
                        break;
                    case CATE_INT:
                    case CATE_UINT:
                        current_val_
                                = builder_.CreateSExtOrTrunc(in_v, outtype);
                        break;
                    case CATE_OTHER: {
                        check_cate();
                        current_val_ = builder_.CreateZExtOrBitCast(
                                in_v, builder_.getInt64Ty());
                    } break;
                }
            } break;
            case CATE_UINT: {
                switch (cate_out) {
                    case CATE_FLOAT:
                        current_val_ = builder_.CreateUIToFP(in_v, outtype);
                        break;
                    case CATE_INT:
                    case CATE_UINT:
                        current_val_
                                = builder_.CreateZExtOrTrunc(in_v, outtype);
                        break;
                    case CATE_OTHER: {
                        check_cate();
                        current_val_ = builder_.CreateZExtOrBitCast(
                                in_v, builder_.getInt64Ty());
                    } break;
                }
            } break;
            case CATE_OTHER:
                if (v->in_->dtype_ == datatypes::generic) {
                    auto bits = module_->getDataLayout().getTypeAllocSizeInBits(
                            outtype);
                    auto ret = builder_.CreateTruncOrBitCast(
                            in_v, IntegerType::get(context_, bits));
                    switch (cate_out) {
                        case CATE_OTHER:
                            COMPILE_ASSERT(v->dtype_.is_pointer(),
                                    "Unexpected out type " << v);
                            current_val_
                                    = builder_.CreateIntToPtr(ret, outtype);
                            break;
                        case CATE_FLOAT:
                            current_val_ = builder_.CreateBitCast(ret, outtype);
                            break;
                        case CATE_INT:
                        case CATE_UINT: current_val_ = ret; break;
                    }
                } else {
                    COMPILE_ASSERT(v->in_->dtype_.is_pointer(),
                            "Unexpected in type " << v);
                    if (v->dtype_.is_pointer()) {
                        // pointer to pointer
                        current_val_
                                = builder_.CreatePointerCast(in_v, outtype);
                    } else {
                        // pointer to generic val
                        check_cate();
                        current_val_ = builder_.CreatePtrToInt(
                                in_v, builder_.getInt64Ty());
                    }
                }
                break;
        }
    }
    void generate_bin_op(const expr_c &v, const expr_c &l, const expr_c &r) {
        auto cate = get_etype_category_nothrow(l->dtype_.type_code_);
        auto lhs = generate_expr(l);
        auto rhs = generate_expr(r);
        COMPILE_ASSERT(cate != CATE_OTHER,
                "Cannot generate binary op for this type: " << v);
#define HANDLE_BIN_OP2(scname, intname, fpname) \
    case sc_expr_type::scname: \
        switch (cate) { \
            case CATE_INT: \
            case CATE_UINT: \
                current_val_ = builder_.Create##intname(lhs, rhs); \
                break; \
            case CATE_FLOAT: \
                current_val_ = builder_.Create##fpname(lhs, rhs); \
                break; \
            default: assert(0); \
        } \
        break
#define HANDLE_BIN_OP(scname, intname) \
    HANDLE_BIN_OP2(scname, intname, F##intname)
#define HANDLE_BIN_SIGNED_OP2(scname, intname1, intname2, fpname) \
    case sc_expr_type::scname: \
        switch (cate) { \
            case CATE_INT: \
                current_val_ \
                        = builder_.Create##intname1##S##intname2(lhs, rhs); \
                break; \
            case CATE_UINT: \
                current_val_ \
                        = builder_.Create##intname1##U##intname2(lhs, rhs); \
                break; \
            case CATE_FLOAT: \
                current_val_ = builder_.Create##fpname(lhs, rhs); \
                break; \
            default: assert(0); \
        } \
        break
#define HANDLE_BIN_SIGNED_OP(scname, intname) \
    HANDLE_BIN_SIGNED_OP2(scname, , intname, F##intname)
#define HANDLE_CMP_SIGNED_OP(scname, llvmname) \
    HANDLE_BIN_SIGNED_OP2(scname, ICmp, llvmname, FCmpO##llvmname)
        switch (v->node_type_) {
            HANDLE_BIN_OP(add, Add);
            HANDLE_BIN_OP(sub, Sub);
            HANDLE_BIN_OP(mul, Mul);
            HANDLE_BIN_SIGNED_OP(div, Div);
            HANDLE_BIN_SIGNED_OP(mod, Rem);
            HANDLE_BIN_OP2(cmp_eq, ICmpEQ, FCmpOEQ);
            HANDLE_BIN_OP2(cmp_ne, ICmpNE, FCmpONE);
            HANDLE_CMP_SIGNED_OP(cmp_lt, LT);
            HANDLE_CMP_SIGNED_OP(cmp_le, LE);
            HANDLE_CMP_SIGNED_OP(cmp_gt, GT);
            HANDLE_CMP_SIGNED_OP(cmp_ge, GE);
            case sc_expr_type::logic_and:
                current_val_ = builder_.CreateAnd(lhs, rhs);
                break;
            case sc_expr_type::logic_or:
                current_val_ = builder_.CreateOr(lhs, rhs);
                break;
            default: assert(0);
        }
    }

    void view(binary_c v) override { generate_bin_op(v, v->l_, v->r_); }
    void view(cmp_c v) override { generate_bin_op(v, v->l_, v->r_); }

    void view(logic_c v) override { generate_bin_op(v, v->l_, v->r_); }
    void view(logic_not_c v) override {
        current_val_ = builder_.CreateNot(generate_expr(v->in_));
    }
    void view(select_c v) override {
        auto l = generate_expr(v->l_);
        auto r = generate_expr(v->r_);
        auto cond = convert_mask(v->cond_);
        current_val_ = builder_.CreateSelect(cond, l, r);
    }

    Instruction *set_alias(Instruction *inst, const expr_c &tsr) {
        if (tsr.isa<indexing>()) {
            return set_alias(inst, tsr.static_as<indexing>()->ptr_);
        }
        auto itr = tsr_to_alias_scope_.find(tsr);
        if (itr != tsr_to_alias_scope_.end()) {
            SmallVector<Metadata *, 4> scope {itr->second->first};
            // alias.scope metadata.
            inst->setMetadata(LLVMContext::MD_alias_scope,
                    MDNode::concatenate(
                            inst->getMetadata(LLVMContext::MD_alias_scope),
                            MDNode::get(inst->getContext(), scope)));

            // noalias metadata.
            inst->setMetadata(LLVMContext::MD_noalias,
                    MDNode::concatenate(
                            inst->getMetadata(LLVMContext::MD_noalias),
                            itr->second->second));
        }
        return inst;
    }

    Value *set_alias(Value *inst, const expr_c &tsr) {
        set_alias(static_cast<Instruction *>(inst), tsr);
        return inst;
    }

    Value *convert_mask(const expr &in) {
        auto mask = generate_expr(in);
        auto &dtype = in->dtype_;
        if (dtype.lanes_ == 1 && !dtype.is_etype(sc_data_etype::BOOLEAN)) {
            auto ty_int1 = builder_.getInt1Ty();
            auto bit_len = utils::get_sizeof_type(dtype) * 8;
            auto mask_ty =
#if SC_LLVM_BACKEND > 10
                    VectorType::get(ty_int1, bit_len, false);
#else
                    VectorType::get(ty_int1, bit_len);
#endif
            mask = builder_.CreateBitCast(mask, mask_ty);
        }
        return mask;
    }

    void view(indexing_c v) override {
        bool is_lvalue_mode = is_lvalue_mode_;
        is_lvalue_mode_ = false;
        COMPILE_ASSERT(v->idx_.size() == 1, "Expecting 1D array: " << v);
        auto base = generate_expr(v->ptr_);
        auto ptr = builder_.CreateGEP(base->getType()->getPointerElementType(),
                base, generate_expr(v->idx_.front()));
        auto target_type = get_type(v->dtype_);
        if (target_type != ptr->getType()->getPointerElementType()) {
            // allow pointer to pointer
            assert(v->dtype_ == datatypes::pointer
                    || llvm::cast<VectorType>(*target_type).getElementType()
                            == ptr->getType()->getPointerElementType());
            ptr = builder_.CreatePointerCast(ptr, target_type->getPointerTo());
        }
        if (is_lvalue_mode) {
            current_val_ = ptr;
        } else {
            bool is_volatile = (v->ptr_->attr_
                    && v->ptr_->attr_->get_or_else("volatile", false));
            if (v->mask_.defined()) {
                auto *mask = convert_mask(v->mask_);
                auto znode = make_expr<constant_node>(0UL, v->dtype_);
                auto zero = generate_expr(znode);
#if SC_LLVM_BACKEND > 10
                current_val_ = set_alias(
                        builder_.CreateMaskedLoad(get_type(v->dtype_), ptr,
                                SC_LLVM_ALIGN(1), mask, zero),
                        v->ptr_);
#else
                current_val_ = set_alias(
                        builder_.CreateMaskedLoad(ptr, 1, mask, zero), v->ptr_);
#endif
            } else {
                if (v->dtype_.lanes_ > 1) {
                    current_val_ = set_alias(
                            builder_.CreateAlignedLoad(get_type(v->dtype_), ptr,
                                    SC_LLVM_ALIGN(1), is_volatile),
                            v->ptr_);
                } else {
                    current_val_
                            = set_alias(builder_.CreateLoad(get_type(v->dtype_),
                                                ptr, is_volatile),
                                    v->ptr_);
                }
            }
        }
    }
    void view(tensorptr_c v) override {
        is_lvalue_mode_ = true;
        current_val_ = generate_expr(v->base_);
    }

    Value *gen_vec_const(uint64_t elements, float f) {
        return builder_.CreateVectorSplat(
                elements, ConstantFP::get(builder_.getFloatTy(), APFloat(f)));
    }

    Value *call_unary_llvm_intrin(const intrin_call_c &v, type_category cate,
            Intrinsic::ID id, bool must_fp) {
        assert(v->args_.size() == 1);
        auto inval = generate_expr(v->args_[0]);
        if (must_fp) {
            COMPILE_ASSERT(
                    cate == CATE_FLOAT, "Bad type. Expecting float: " << v);
        }
        return builder_.CreateUnaryIntrinsic(id, inval);
    }

    Value *call_binary_llvm_intrin(const intrin_call_c &v, type_category cate,
            Intrinsic::ID id, bool must_fp) {
        assert(v->args_.size() == 2);
        auto inval1 = generate_expr(v->args_[0]);
        auto inval2 = generate_expr(v->args_[1]);
        if (must_fp) {
            COMPILE_ASSERT(
                    cate == CATE_FLOAT, "Bad type. Expecting float: " << v);
        }
        return builder_.CreateBinaryIntrinsic(id, inval1, inval2);
    }

    typedef Value *(llvm::IRBuilder<>::*llvm_binary_func)(
            Value *LHS, Value *RHS, const Twine &Name);
    Value *call_binary_llvm_normal(
            const intrin_call_c &v, llvm_binary_func op) {
        assert(v->args_.size() == 2);
        auto inval1 = generate_expr(v->args_[0]);
        auto inval2 = generate_expr(v->args_[1]);
        return (builder_.*op)(inval1, inval2, "");
    }

    Value *make_int_min_max(
            const intrin_call_c &v, bool ismin, type_category cate) {
        assert(v->args_.size() == 2);
        auto v1 = generate_expr(v->args_[0]);
        auto v2 = generate_expr(v->args_[1]);
        return make_int_min_max(v1, v2, ismin, cate);
    }

    Value *make_int_min_max(
            Value *v1, Value *v2, bool ismin, type_category cate) {
        // fix-me: use smax/smin for newer LLVM
        llvm::Value *(llvm::IRBuilder<>::*ptr)(
                llvm::Value * LHS, llvm::Value * RHS, const llvm::Twine &Name);
        if (ismin) {
            if (cate == CATE_INT) {
                ptr = &IRBuilder<>::CreateICmpSLE;
            } else {
                ptr = &IRBuilder<>::CreateICmpULE;
            }
        } else {
            if (cate == CATE_INT) {
                ptr = &IRBuilder<>::CreateICmpSGE;
            } else {
                ptr = &IRBuilder<>::CreateICmpUGE;
            }
        }
        return builder_.CreateSelect((builder_.*ptr)(v1, v2, ""), v1, v2);
    }

    Value *do_lower_saturated_cast(const intrin_call_c &v) {
        COMPILE_ASSERT(ctx_->machine_.cpu_flags_.fAVX512F,
                "lowered saturated_cast needs AVX512F");
        assert(v->args_.size() == 1);
        auto inval1 = generate_expr(v->args_[0]);
        auto intype = v->args_[0]->dtype_;
        auto out_llvm_ty = get_type(v->dtype_);
        auto ths = this;
        // the fast path for AVX512
        auto pmovus_db_512 = [ths, out_llvm_ty](Value *v, bool issigned) {
            Intrinsic::ID id = issigned
                    ? Intrinsic::x86_avx512_mask_pmovs_db_512
                    : Intrinsic::x86_avx512_mask_pmovus_db_512;
            return ths->builder_.CreateIntrinsic(id, {},
                    {v, UndefValue::get(out_llvm_ty),
                            ths->builder_.getInt16(0xffff)});
        };
        if (v->dtype_ == sc_data_type_t::s8(16)) {
            if (intype == sc_data_type_t::s32(16)) {
                return pmovus_db_512(inval1, true);
            }
        } else if (v->dtype_ == sc_data_type_t::u8(16)) {
            if (intype == sc_data_type_t::s32(16)) {
                return pmovus_db_512(inval1, false);
            }
        }
        COMPILE_ASSERT(false,
                "lowered saturated_cast cannot handle: "
                        << v << '(' << intype << "->" << v->dtype_ << ')');
    }

    /**
     * Implementing LLVM-x86 intrinsics
     * 1. first find the GCC/Clang built-in intrinsic name in
     * https://github.com/llvm/llvm-project/blob/main/clang/lib/Headers
     *    e.g. Goto definition of _mm512_cvtusepi32_epi8, you will get
     *    __builtin_ia32_pmovusdb512_mask
     * 2. Find the built-in function name in
     * https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/IR/IntrinsicsX86.td
     * 3. Now we have the intrinsic name in LLVM :
     *    x86_avx512_mask_pmovus_db_512
     * */

    void view(intrin_call_c v) override {
        auto cate = get_etype_category_nothrow(v->dtype_.type_code_);
#if SC_LLVM_BACKEND > 10
        using shuffle_idx_t = int;
#else
        using shuffle_idx_t = uint32_t;
#endif
        switch (v->type_) {
            case intrin_type::reinterpret: {
                assert(v->args_.size() == 1);
                auto inval = generate_expr(v->args_[0]);
                auto outty = get_type(v->dtype_);
                if (outty->isPointerTy()) {
                    auto src_cate
                            = get_type_category_nothrow(v->args_[0]->dtype_);
                    bool is_src_int
                            = src_cate == CATE_INT || src_cate == CATE_UINT;
                    if (is_src_int) {
                        current_val_ = builder_.CreateIntToPtr(inval, outty);
                    } else {
                        current_val_ = builder_.CreatePointerCast(inval, outty);
                    }
                } else if (inval->getType()->isPointerTy()) {
                    auto dst_cate = get_type_category_nothrow(v->dtype_);
                    bool is_dest_int
                            = dst_cate == CATE_INT || dst_cate == CATE_UINT;
                    COMPILE_ASSERT(is_dest_int,
                            "Expecting pointer to int for reinterpret");
                    current_val_ = builder_.CreatePtrToInt(inval, outty);
                } else {
                    current_val_ = builder_.CreateBitCast(inval, outty);
                }
            } break;
            case intrin_type::abs: {
                assert(v->args_.size() == 1);
                auto inval = generate_expr(v->args_[0]);

                std::string intrin_name;
                llvm::raw_string_ostream os(intrin_name);
                switch (cate) {
                    case CATE_FLOAT:
                        current_val_ = builder_.CreateUnaryIntrinsic(
                                Intrinsic::fabs, inval);
                        break;
                    case CATE_INT: {
                        auto znode = make_expr<constant_node>(
                                0UL, v->args_[0]->dtype_);
                        auto zero = generate_expr(znode);
                        auto sign = builder_.CreateICmpSGT(inval, zero);
                        auto neg = builder_.CreateSub(zero, inval);
                        current_val_ = builder_.CreateSelect(sign, inval, neg);
                    } break;
                    default: assert(0); break;
                }
            } break;
            case intrin_type::rsqrt: {
                current_val_ = call_unary_llvm_intrin(
                        v, cate, Intrinsic::sqrt, true);
                Value *ones
                        = ConstantFP::get(builder_.getFloatTy(), APFloat(1.0f));
                if (v->dtype_.lanes_ > 1) {
                    ones = builder_.CreateVectorSplat(v->dtype_.lanes_, ones);
                }
                current_val_ = builder_.CreateFDiv(ones, current_val_);

                // fix-me: (yijie) LLVM-8 does not correctly generate
                // x86_avx_rsqrt_ps_256. LLVM-13 not tested

                // COMPILE_ASSERT(v->dtype_ == sc_data_type_t::f32(8),
                //         "Expecting f32x8 for rsqrt, got " << v->dtype_);
                // current_val_ = call_unary_llvm_intrin(
                //         v, cate, Intrinsic::x86_avx_rsqrt_ps_256, true);
            } break;
            case intrin_type::int_and: {
                current_val_ = call_binary_llvm_normal(
                        v, &llvm::IRBuilder<>::CreateAnd);
            } break;
            case intrin_type::int_or: {
                current_val_ = call_binary_llvm_normal(
                        v, &llvm::IRBuilder<>::CreateOr);
            } break;
            case intrin_type::int_xor: {
                current_val_ = call_binary_llvm_normal(
                        v, &llvm::IRBuilder<>::CreateXor);
            } break;
            case intrin_type::round: {
                current_val_ = call_unary_llvm_intrin(
                        v, cate, Intrinsic::nearbyint, true);
            } break;
            case intrin_type::ceil: {
                current_val_ = call_unary_llvm_intrin(
                        v, cate, Intrinsic::ceil, true);
            } break;
            case intrin_type::floor: {
                current_val_ = call_unary_llvm_intrin(
                        v, cate, Intrinsic::floor, true);
            } break;
            case intrin_type::max: {
                if (cate == CATE_FLOAT) {
                    current_val_ = call_binary_llvm_intrin(
                            v, cate, Intrinsic::maxnum, true);
                } else {
                    current_val_ = make_int_min_max(v, false, cate);
                }
            } break;
            case intrin_type::min: {
                if (cate == CATE_FLOAT) {
                    current_val_ = call_binary_llvm_intrin(
                            v, cate, Intrinsic::minnum, true);
                } else {
                    current_val_ = make_int_min_max(v, true, cate);
                }
            } break;
            case intrin_type::shl: {
                assert(v->args_.size() == 2);
                auto inval1 = generate_expr(v->args_[0]);
                auto inval2 = generate_expr(v->args_[1]);
                COMPILE_ASSERT(cate == CATE_INT || cate == CATE_UINT,
                        "Bad type. Expecting int: " << v);
                current_val_ = builder_.CreateShl(inval1, inval2);
            } break;
            case intrin_type::shr: {
                assert(v->args_.size() == 2);
                auto inval1 = generate_expr(v->args_[0]);
                auto inval2 = generate_expr(v->args_[1]);
                if (cate == CATE_INT) {
                    current_val_ = builder_.CreateAShr(inval1, inval2);
                } else {
                    COMPILE_ASSERT(cate == CATE_UINT,
                            "Bad type. Expecting int: " << v);
                    current_val_ = builder_.CreateLShr(inval1, inval2);
                }
            } break;
            case intrin_type::broadcast: {
                assert(v->args_.size() == 1);
                auto inval1 = generate_expr(v->args_[0]);
                auto lanes = v->dtype_.lanes_;
                auto in_lanes = v->args_[0]->dtype_.lanes_;
                if (lanes != 1) {
                    if (in_lanes != 1) {
                        while (in_lanes < lanes) {
                            std::vector<shuffle_idx_t> array(in_lanes << 1);
                            for (uint16_t i = 0; i < (in_lanes << 1); i++) {
                                array[i] = i;
                            }
                            inval1 = builder_.CreateShuffleVector(
                                    inval1, inval1, array);
                            in_lanes = in_lanes << 1;
                        }
                        current_val_ = inval1;
                    } else {
                        current_val_
                                = builder_.CreateVectorSplat(lanes, inval1);
                    }
                } else {
                    current_val_ = inval1;
                }
            } break;
            case intrin_type::fmadd: {
                assert(v->args_.size() == 3);
                auto inval1 = generate_expr(v->args_[0]);
                auto inval2 = generate_expr(v->args_[1]);
                auto inval3 = generate_expr(v->args_[2]);
                auto ret = builder_.CreateIntrinsic(Intrinsic::fma,
                        {get_type(v->dtype_)}, {inval1, inval2, inval3});
                ret->setFastMathFlags(builder_.getFastMathFlags());
                current_val_ = ret;
            } break;
            case intrin_type::reduce_add:
            case intrin_type::reduce_mul:
            case intrin_type::reduce_max:
            case intrin_type::reduce_min: {
                Intrinsic::ID cur_intrinsic = Intrinsic::not_intrinsic;
#if SC_LLVM_BACKEND > 10
#define LLVM_INTRINSIC_EXP_V2(name) Intrinsic::vector_reduce_##name
#define LLVM_INTRINSIC_EXP LLVM_INTRINSIC_EXP_V2
#elif SC_LLVM_BACKEND > 8
#define LLVM_INTRINSIC_EXP_V2(name) \
    Intrinsic::experimental_vector_reduce_v2_##name
#define LLVM_INTRINSIC_EXP(name) Intrinsic::experimental_vector_reduce_##name
#else
#define LLVM_INTRINSIC_EXP_V2(name) Intrinsic::experimental_vector_reduce_##name
#define LLVM_INTRINSIC_EXP LLVM_INTRINSIC_EXP_V2
#endif
                if (v->type_ == intrin_type::reduce_add) {
                    if (cate == CATE_FLOAT) {
                        cur_intrinsic = LLVM_INTRINSIC_EXP_V2(fadd);
                    } else {
                        cur_intrinsic = LLVM_INTRINSIC_EXP(add);
                    }
                } else if (v->type_ == intrin_type::reduce_mul) {
                    if (cate == CATE_FLOAT) {
                        cur_intrinsic = LLVM_INTRINSIC_EXP_V2(fmul);
                    } else {
                        cur_intrinsic = LLVM_INTRINSIC_EXP(mul);
                    }
                } else if (v->type_ == intrin_type::reduce_max) {
                    if (cate == CATE_FLOAT) {
                        cur_intrinsic = LLVM_INTRINSIC_EXP(fmax);
                    } else if (cate == CATE_INT) {
                        cur_intrinsic = LLVM_INTRINSIC_EXP(smax);
                    } else if (cate == CATE_UINT) {
                        cur_intrinsic = LLVM_INTRINSIC_EXP(umax);
                    }
                } else if (v->type_ == intrin_type::reduce_min) {
                    if (cate == CATE_FLOAT) {
                        cur_intrinsic = LLVM_INTRINSIC_EXP(fmin);
                    } else if (cate == CATE_INT) {
                        cur_intrinsic = LLVM_INTRINSIC_EXP(smin);
                    } else if (cate == CATE_UINT) {
                        cur_intrinsic = LLVM_INTRINSIC_EXP(umin);
                    }
                }
                assert(v->args_.size() == 1);
                auto inval = generate_expr(v->args_[0]);
                if ((v->type_ == intrin_type::reduce_add
                            || v->type_ == intrin_type::reduce_mul)
                        && cate == CATE_FLOAT) {
                    current_val_ = builder_.CreateIntrinsic(
#if SC_LLVM_BACKEND > 10
                            cur_intrinsic, {inval->getType()},
#elif SC_LLVM_BACKEND > 8
                            cur_intrinsic, {inval->getType()},
#else
                            cur_intrinsic,
                            {get_type(v->dtype_), get_type(v->dtype_),
                                    inval->getType()},
#endif
                            {ConstantFP::get(
                                     get_type(v->dtype_), APFloat(0.0f)),
                                    inval});
                    llvm::cast<CallInst>(*current_val_)
                            .setFastMathFlags(builder_.getFastMathFlags());
                } else {
                    current_val_ = builder_.CreateIntrinsic(
#if SC_LLVM_BACKEND > 10
                            cur_intrinsic, {inval->getType()},
#else
                            cur_intrinsic,
                            {get_type(v->dtype_), inval->getType()},
#endif
                            {inval});
                }
            } break;
            case intrin_type::saturated_cast: {
                current_val_ = do_lower_saturated_cast(v);
            } break;
            case intrin_type::round_and_cast: {
                assert(v->args_.size() == 1);
                auto inval1 = generate_expr(v->args_[0]);
                COMPILE_ASSERT(v->dtype_.type_code_ == sc_data_etype::S32
                                && v->args_[0]->dtype_.type_code_
                                        == sc_data_etype::F32,
                        "LLVM backend has not yet support round_and_cast like "
                        "this: " << v);
                switch (v->dtype_.lanes_) {
                    case 1:
                        current_val_ = builder_.CreateFPToSI(
                                builder_.CreateUnaryIntrinsic(
                                        Intrinsic::round, inval1),
                                builder_.getInt32Ty());
                        break;
                    case 4:
                        current_val_ = builder_.CreateIntrinsic(
                                Intrinsic::x86_sse2_cvtps2dq, {}, inval1);
                        break;
                    case 8:
                        current_val_ = builder_.CreateIntrinsic(
                                Intrinsic::x86_avx_cvt_ps2dq_256, {}, inval1);
                        break;
                    case 16:
                        COMPILE_ASSERT(ctx_->machine_.cpu_flags_.fAVX512F,
                                "round_and_cast of 16 floats needs AVX512");
                        current_val_ = builder_.CreateIntrinsic(
                                Intrinsic::x86_avx512_mask_cvtps2dq_512, {},
                                {inval1, UndefValue::get(get_type(v->dtype_)),
                                        /*mask*/ builder_.getInt16(0xffff),
                                        /*rounding mode =
                                           _MM_FROUND_CUR_DIRECTION    0x04*/
                                        builder_.getInt32(0x04)});
                        break;
                    default:
                        COMPILE_ASSERT(false,
                                "LLVM backend has not yet support "
                                "round_and_cast with lanes = "
                                        << v->dtype_.lanes_);
                        break;
                }
            } break;
            case intrin_type::permutex2var: {
                assert(v->args_.size() == 3);
                auto inval1 = generate_expr(v->args_[0]);
                auto inval2 = generate_expr(v->args_[1]);
                auto inval3 = generate_expr(v->args_[2]);
                switch (v->args_[0]->dtype_.type_code_) {
                    case sc_data_etype::F32:
                        current_val_ = builder_.CreateIntrinsic(
                                Intrinsic::x86_avx512_vpermi2var_ps_128, {},
                                {inval1, inval2, inval3});
                        break;
                    case sc_data_etype::U8:
                    default:
                        current_val_ = builder_.CreateIntrinsic(
                                Intrinsic::x86_avx512_vpermi2var_qi_128, {},
                                {inval1, inval2, inval3});
                        break;
                }
            } break;
            case intrin_type::unpack_high:
            case intrin_type::unpack_low: {
                assert(v->args_.size() == 2);
                COMPILE_ASSERT(v->dtype_.is_etype(sc_data_etype::BF16)
                                || v->dtype_.is_etype(sc_data_etype::U16)
                                || v->dtype_.is_etype(sc_data_etype::F32),
                        "Expecting u16/bf16/f32 for unpack: " << v);
                auto inval1 = generate_expr(v->args_[0]);
                auto inval2 = generate_expr(v->args_[1]);
                auto elem_bits = v->intrin_attrs_->get<int>("elem_bits");
                std::vector<shuffle_idx_t> hi_array, lo_array;
                if (v->dtype_.lanes_ == 8) {
                    // todo: currently only support f32
                    assert(elem_bits == 32);
                    hi_array = std::vector<shuffle_idx_t> {
                            2, 10, 2 + 1, 10 + 1, 6, 14, 6 + 1, 14 + 1};
                    lo_array = std::vector<shuffle_idx_t> {
                            0, 8, 0 + 1, 8 + 1, 4, 12, 4 + 1, 12 + 1};
                } else {
                    assert(v->dtype_.lanes_ == 32);
                    switch (elem_bits) {
                        case 16:
                            hi_array = std::vector<shuffle_idx_t> {4, 36, 4 + 1,
                                    36 + 1, 4 + 2, 36 + 2, 4 + 3, 36 + 3, 12,
                                    44, 12 + 1, 44 + 1, 12 + 2, 44 + 2, 12 + 3,
                                    44 + 3, 20, 52, 20 + 1, 52 + 1, 20 + 2,
                                    52 + 2, 20 + 3, 52 + 3, 28, 60, 28 + 1,
                                    60 + 1, 28 + 2, 60 + 2, 28 + 3, 60 + 3};
                            lo_array.resize(32);
                            std::transform(hi_array.begin(), hi_array.end(),
                                    lo_array.begin(),
                                    [](shuffle_idx_t x) { return x - 4; });
                            break;
                        case 32:
                            hi_array = std::vector<shuffle_idx_t> {4, 4 + 1, 36,
                                    36 + 1, 4 + 2, 4 + 3, 36 + 2, 36 + 3, 12,
                                    12 + 1, 44, 44 + 1, 12 + 2, 12 + 3, 44 + 2,
                                    44 + 3, 20, 20 + 1, 52, 52 + 1, 20 + 2,
                                    20 + 3, 52 + 2, 52 + 3, 28, 28 + 1, 60,
                                    60 + 1, 28 + 2, 28 + 3, 60 + 2, 60 + 3};
                            lo_array.resize(32);
                            std::transform(hi_array.begin(), hi_array.end(),
                                    lo_array.begin(),
                                    [](shuffle_idx_t x) { return x - 4; });
                            break;
                        case 64:
                            hi_array = std::vector<shuffle_idx_t> {4, 4 + 1,
                                    4 + 2, 4 + 3, 36, 36 + 1, 36 + 2, 36 + 3,
                                    12, 12 + 1, 12 + 2, 12 + 3, 44, 44 + 1,
                                    44 + 2, 44 + 3, 20, 20 + 1, 20 + 2, 20 + 3,
                                    52, 52 + 1, 52 + 2, 52 + 3, 28, 28 + 1,
                                    28 + 2, 28 + 3, 60, 60 + 1, 60 + 2, 60 + 3};
                            lo_array.resize(32);
                            std::transform(hi_array.begin(), hi_array.end(),
                                    lo_array.begin(),
                                    [](shuffle_idx_t x) { return x - 4; });
                            break;
                    }
                }
                ArrayRef<shuffle_idx_t> arr
                        = v->type_ == intrin_type::unpack_high ? hi_array
                                                               : lo_array;
                current_val_
                        = builder_.CreateShuffleVector(inval1, inval2, arr);
            } break;
            case intrin_type::shuffle: {
                assert(v->args_.size() == 2);
                COMPILE_ASSERT(v->dtype_.lanes_ == 8,
                        "Expecting 8-lane for shuffle: " << v);
                auto inval1 = generate_expr(v->args_[0]);
                auto inval2 = generate_expr(v->args_[1]);
                auto imm8 = v->intrin_attrs_->get<int>("shuffle_imm");
                shuffle_idx_t array[8];
                for (int i = 0; i <= 6; i += 2) {
                    auto arr_idx = i / 2;
                    auto val_idx = (imm8 >> i) % 4;
                    if (i >= 4) {
                        array[arr_idx] = val_idx + 8;
                        array[arr_idx + 4] = val_idx + 8 + 4;
                    } else {
                        array[arr_idx] = val_idx;
                        array[arr_idx + 4] = val_idx + 4;
                    }
                }
                current_val_
                        = builder_.CreateShuffleVector(inval1, inval2, array);
            } break;
            case intrin_type::permute: {
                assert(v->args_.size() == 2);
                COMPILE_ASSERT(v->dtype_.lanes_ == 8,
                        "Expecting 8-lane for permute: " << v);
                auto inval1 = generate_expr(v->args_[0]);
                auto inval2 = generate_expr(v->args_[1]);
                auto imm8 = v->intrin_attrs_->get<int>("permute_imm");
                shuffle_idx_t array[8];
                auto low_idx = 0, high_idx = 0;
                switch (imm8 % 4) {
                    case 0: break;
                    case 1: low_idx += 4; break;
                    case 2: low_idx += 8; break;
                    case 3: low_idx += 12; break;
                    default: break;
                }
                array[0] = low_idx;
                array[1] = low_idx + 1;
                array[2] = low_idx + 2;
                array[3] = low_idx + 3;
                switch ((imm8 >> 4) % 4) {
                    case 0: break;
                    case 1: high_idx += 4; break;
                    case 2: high_idx += 8; break;
                    case 3: high_idx += 12; break;
                    default: break;
                }
                array[4] = high_idx;
                array[5] = high_idx + 1;
                array[6] = high_idx + 2;
                array[7] = high_idx + 3;
                current_val_
                        = builder_.CreateShuffleVector(inval1, inval2, array);
            } break;

            case intrin_type::prefetch: {
                assert(v->args_.size() == 1);
                auto locality = v->intrin_attrs_->get<int>("locality");
                assert(locality <= 3 && locality >= 0
                        && "bad locality for prefetch");
                auto inval1 = generate_expr(v->args_[0]);
                current_val_ = builder_.CreateIntrinsic(Intrinsic::prefetch,
#if SC_LLVM_BACKEND > 8
                        {builder_.getInt8PtrTy()},
#else
                        {},
#endif
                        {builder_.CreatePointerCast(
                                 inval1, builder_.getInt8PtrTy()),
                                /*rw*/ builder_.getInt32(0),
                                /*locality*/ builder_.getInt32(3 - locality),
                                /*type:i/d*/ builder_.getInt32(1)});
            } break;
            default: {
                std::stringstream ss;
                ss << "Intrinsics not implemented ";
                v->to_string(ss);
                throw std::runtime_error(ss.str());
            }
        }
    }
    void view(func_addr_c v) override {
        current_val_ = builder_.CreatePointerCast(
                get_or_create_func(v->func_), builder_.getInt8PtrTy());
    }
    void view(call_c v) override {
        std::vector<Value *> args;
        auto the_func = std::dynamic_pointer_cast<func_base>(v->func_);
        Value *ll_func;
        FunctionType *ft;
        if (the_func) {
            ll_func = get_or_create_func(the_func);
            ft = &llvm::cast<FunctionType>(
                    *ll_func->getType()->getPointerElementType());
        } else {
            auto the_expr = std::dynamic_pointer_cast<expr_base>(v->func_);
            assert(the_expr);
            auto proto_func
                    = the_expr->attr().get_or_else("prototype", func_t());
            COMPILE_ASSERT(
                    proto_func, "Call node expects an expr with prototype");
            ft = create_func_type(proto_func);
            ll_func = generate_expr(expr_c(the_expr));
            ll_func = builder_.CreatePointerCast(ll_func, ft->getPointerTo());
        }
        for (size_t i = 0; i < v->args_.size(); i++) {
            auto &val = v->args_[i];
            auto ll_value = generate_expr(val);
            auto target_type = *(ft->param_begin() + i);
            if (ll_value->getType() != target_type) {
                COMPILE_ASSERT(target_type == builder_.getInt8PtrTy(),
                        "LLVM can only handle autocast to pointer");
                ll_value = builder_.CreatePointerCast(ll_value, target_type);
            }
            args.push_back(ll_value);
        }
        current_val_ = builder_.CreateCall(ft, ll_func, args);
    }
    void view(tensor_c v) override { current_val_ = get_defined_var_ptr(v); }

    // void view(stmts_c v) override;
    // void view(evaluate_c v) override;

    void generate_codeblock(
            const stmt_c &v, BasicBlock *current, BasicBlock *cont) {
        builder_.SetInsertPoint(current);
        dispatch(v);
        current = builder_.GetInsertBlock();
        if (current->empty() || !llvm::isa<llvm::ReturnInst>(current->back())) {
            builder_.CreateBr(cont);
        }
    }

    void view(assign_c v) override {
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
                auto *mask = convert_mask(v->var_.static_as<indexing>()->mask_);
#if SC_LLVM_BACKEND > 10
                set_alias(builder_.CreateMaskedStore(
                                  val, ptr, SC_LLVM_ALIGN(1), mask),
                        v->var_);
#else
                set_alias(
                        builder_.CreateMaskedStore(val, ptr, 1, mask), v->var_);
#endif
            } else {
                set_alias(
                        builder_.CreateAlignedStore(val, ptr, SC_LLVM_ALIGN(1)),
                        v->var_);
            }
        } else {
            set_alias(builder_.CreateStore(val, ptr), v->var_);
        }
    }

    void view(if_else_c v) override {
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

    void view(returns_c v) override {
        if (v->value_.defined()) {
            builder_.CreateRet(generate_expr(v->value_));
        } else {
            builder_.CreateRetVoid();
        }
    }

    void set_dbg_info_for_local_var(const source_pos *pos, sc_data_type_t type,
            const std::string &name, Value *llvm_value, bool need_ref) {
        if (!ctx_->flags_.debug_info_) { return; }
        auto types = get_type_both(type);
        auto dbgtype = types.second;
        if (need_ref) {
            auto tmp = builder_.CreateAlloca(llvm_value->getType());
            builder_.CreateStore(llvm_value, tmp);
            llvm_value = tmp;
        }
        // Create a debug descriptor for the variable.
        DILocalVariable *D = dbuilder_->createAutoVariable(dbg_scopes_.back(),
                name, dbg_cu_->getFile(), pos->line_, dbgtype, true);

        dbuilder_->insertDeclare(llvm_value, D, dbuilder_->createExpression(),
                DILocation::get(dbg_scopes_.back()->getContext(), pos->line_,
                        pos->pos_, dbg_scopes_.back()),
                builder_.GetInsertBlock());
    }

    void set_dbg_info_for_local_var(const define_node_t *v,
            const std::string &name, Value *llvm_value, bool need_ref) {
        if (!ctx_->flags_.debug_info_) { return; }
        auto pos = v->attr_->get_or_null<source_pos>("source_pos");
        if (pos) {
            set_dbg_info_for_local_var(
                    pos, v->var_->dtype_, name, llvm_value, need_ref);
        }
    }

    void view(define_c v) override {
        COMPILE_ASSERT(v->linkage_ != linkage::static_local
                        && v->linkage_ != linkage::private_global,
                "LLVM backend cannot handle non-local variable definitions");
        if (v->var_.isa<var>()) {
            auto thevar = v->var_.static_as<var>();
            if (thevar->attr_
                    && thevar->attr_->has_key(
                            attr_keys::module_global_offset)) {
                // if it is a global variable that is lowered to local
                size_t offset = thevar->attr_->get<size_t>(
                        attr_keys::module_global_offset);
                Argument *module_ptr = current_func_->arg_begin() + 1;
                assert(module_ptr->getName() == "__module_data_arg");
                auto ptr = builder_.CreateGEP(
                        module_ptr->getType()->getPointerElementType(),
                        module_ptr, builder_.getInt64(offset));
                ptr = builder_.CreatePointerCast(ptr,
                        get_type(thevar->dtype_)->getPointerTo(),
                        thevar->name_);
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
                auto alias_itr = alias_set_to_alias_scope_.find(
                        alias_info->get_alias_set().get());
                if (alias_itr != alias_set_to_alias_scope_.end()) {
                    tsr_to_alias_scope_[t] = &alias_itr->second;
                }
            }
            // if it is a view of the rescheduled buffer/ local tensor on heap
            if (v->init_.defined()) {
                Value *ptr = generate_expr(v->init_);
                ptr = builder_.CreatePointerCast(ptr,
                        get_type(t->elem_dtype_)->getPointerTo(), t->name_);
                var_ptr_in_func_.insert(std::make_pair(t, ptr));
                set_dbg_info_for_local_var(v.get(), t->name_, ptr, true);
                return;
            }

            // explicitly align tensor with cache line size, except that
            // tensor is a scalar or bytes size < 64.
            bool need_align = false;
            // check condition.
            if (t->dims_.size() == 1
                    && get_const_as_int(t->dims_[0].checked_as<constant>())
                            == 1) {
                // it is a scalar
            } else {
                size_t shape = 1;
                for (auto &d : t->dims_) {
                    shape *= get_const_as_int(d.checked_as<constant>());
                }
                size_t dtsize
                        = utils::get_sizeof_etype(t->elem_dtype_.type_code_);
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

    void view(for_loop_c v) override {
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

        BasicBlock *chk
                = BasicBlock::Create(context_, "for_check", current_func_);
        BasicBlock *body
                = BasicBlock::Create(context_, "for_body", current_func_);
        BasicBlock *cont
                = BasicBlock::Create(context_, "for_cont", current_func_);
        builder_.CreateBr(chk);
        {
            builder_.SetInsertPoint(chk);
            auto cate = get_type_category(v->var_->dtype_);
            auto end_v = generate_expr(v->iter_end_);
            auto itr_value = builder_.CreateLoad(
                    itr_v->getType()->getPointerElementType(), itr_v);
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
                Value *itr_value = builder_.CreateLoad(
                        itr_v->getType()->getPointerElementType(), itr_v);
                itr_value = builder_.CreateAdd(itr_value, step_v);
                builder_.CreateStore(itr_value, itr_v);
                builder_.CreateBr(chk);
            }
        }
        cont->moveAfter(builder_.GetInsertBlock());
        builder_.SetInsertPoint(cont);
    }
};

static std::string dump_module_to_string(Module *m) {
    std::string ret;
    raw_string_ostream os(ret);
    os << *m;
    return ret;
}

const_ir_module_ptr llvm_generator_pass::operator()(const_ir_module_ptr f) {
    auto passes = get_default_precodegen_passes(f->ctx_, gen_wrapper_);
    auto mod = run_precodegen_passes(passes, f);
    std::string unique_name;
    const auto &tmpdir = utils::compiler_configs_t::get().temp_dir_;
    if (f->ctx_->flags_.debug_info_) {
        std::string file_name;
        file_name = "llvm_jit-" + utils::get_unique_name_for_file() + ".gcir";
        std::string unique_name = tmpdir + "/" + file_name;
        std::ofstream ofs(unique_name);
        out_source_path_ = unique_name;
        print_ir_and_annotate_source_pos(*mod, ofs);
    } else {
        out_source_path_ = "";
    }

    codegen_llvm_vis_t vis {f->ctx_, llvm_ctx_, tmpdir, out_source_path_};
    auto timer = SC_SCOPED_TIMER_INFO("pass.time.llvm_generator_pass", "");
    for (auto &funct : mod->get_contents()) {
        vis.dispatch(funct);
    }
    if (f->ctx_->flags_.debug_info_) { vis.dbuilder_->finalize(); }
    out_module_ = std::move(vis.module_);
    SC_MODULE_INFO << dump_module_to_string(out_module_.get());
    return mod;
}

} // namespace sc
