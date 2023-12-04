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
#include <util/compiler_macros.hpp>

// g++-7 + LLVM16 has an unknown bug, which will generate broken TargetMachine
// we use C-API instead
#if SC_GNUC_VERSION_LT(8) && SC_LLVM_BACKEND >= 16
#include <llvm-c/TargetMachine.h>
#define WORKAROUND_LLVM_TM
#endif

using namespace llvm;
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

static void print_helper(Value *v) {
    v->print(llvm::errs());
}

#define WHEN_ARCH_NAME(...) __VA_ARGS__
#define ARCH_GraniteRapids \
    case 0xae: \
    case 0xad:
#define ARCH_EmeraldRapids case 0xcf:
#define ARCH_SapphireRapids case 0x8f:
#define END_ARCH() break;

#define WHEN_LLVM_VER_GE(V, VALUE) \
    if (SC_LLVM_BACKEND >= (V)) { return VALUE; }
#define WHEN_LLVM_VER_RANGE(V1, V2, VALUE) \
    if (SC_LLVM_BACKEND >= (V1) && SC_LLVM_BACKEND <= (V2)) { return VALUE; }

// Handle the case that the CPU is known to us, but not to current version of
// LLVM. We help LLVM to better fallback to a closer generation of CPU known to
// it.
static const char *handle_cpu_name(runtime::cpu_flags_t &flags) {
    if (flags.family != 6) { return nullptr; }
    constexpr const char *sapphirerapids = "sapphirerapids";
    constexpr const char *icelake_server = "icelake-server";
    constexpr const char *let_llvm_handle = nullptr;
    switch (flags.model) {
        // clang-format off
        WHEN_ARCH_NAME(ARCH_GraniteRapids ARCH_EmeraldRapids)
            WHEN_LLVM_VER_GE(16, let_llvm_handle)
            WHEN_LLVM_VER_RANGE(12, 15, sapphirerapids)
            WHEN_LLVM_VER_RANGE(9, 11, icelake_server)
        END_ARCH()
        WHEN_ARCH_NAME(ARCH_SapphireRapids)
            WHEN_LLVM_VER_GE(12, let_llvm_handle)
            WHEN_LLVM_VER_RANGE(9, 11, icelake_server)
        END_ARCH()
        // clang-format on
    }
    // default, let llvm decide
    return let_llvm_handle;
}

std::unique_ptr<TargetMachine> get_llvm_target_machine(
        CodeGenOpt::Level optlevel) {
    auto target_triple = sys::getProcessTriple();

    std::string err;
    auto target = TargetRegistry::lookupTarget(target_triple, err);
    if (!target) { throw std::runtime_error(err); }
    TargetOptions opt;
#if SC_LLVM_BACKEND > 15
    auto the_none = std::nullopt;
#else
    auto the_none = llvm::None;
#endif

    auto reloc_model = Optional<Reloc::Model>(Reloc::Static);
    auto inferred_cpu_name
            = handle_cpu_name(runtime::get_runtime_target_machine().cpu_flags_);
    auto host_cpu
            = inferred_cpu_name ? inferred_cpu_name : sys::getHostCPUName();
    if (inferred_cpu_name) {
        SC_MODULE_WARN << "Your CPU is not recognized by LLVM. Falling back to "
                       << inferred_cpu_name;
    }
    llvm::StringMap<bool> feature_map;
    sys::getHostCPUFeatures(feature_map);
    llvm::SubtargetFeatures f;
    for (auto &feature : feature_map) {
        f.AddFeature(feature.first(), feature.second);
    }
    std::string features = f.getString();
#ifdef WORKAROUND_LLVM_TM
    SC_UNUSED(the_none);
    SC_UNUSED(reloc_model);
    auto tm = (TargetMachine *)LLVMCreateTargetMachine((LLVMTargetRef)target,
            target_triple.c_str(), host_cpu.data(), features.c_str(),
            LLVMCodeGenLevelDefault, LLVMRelocStatic, LLVMCodeModelJITDefault);
#else
    auto tm = target->createTargetMachine(target_triple, host_cpu, features,
            opt, reloc_model, the_none, optlevel, true);
#endif
    return std::unique_ptr<TargetMachine>(tm);
}

std::unique_ptr<TargetMachine> get_llvm_target_machine(
        CodeGenOpt::Level optlevel = CodeGenOpt::Level::Default);

codegen_llvm_vis_t::codegen_llvm_vis_t(const context_ptr &ctx,
        LLVMContext &context, const std::string &source_dir,
        const std::string &source_file_name)
    : ctx_(ctx)
    , context_(context)
    , builder_(context_)
    , module_(utils::make_unique<Module>("name", context_))
    , dbuilder_(utils::make_unique<DIBuilder>(*module_)) {
    static bool initialized = []() {
        // make sure LLVM native targets are initialized once and avoid
        // race condition
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
    // some optimization in FastMath may cause accuracy loss, which needs
    // further investigation in the future
    fmflag.setFast(false);
    // keep FMA optimization on
    fmflag.setAllowContract(true);
    // turn on following options for performance
    fmflag.setAllowReassoc(true);
    fmflag.setNoNaNs(true);
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

expr_c codegen_llvm_vis_t::dispatch(expr_c v) {
    emit_location(v.get());
    return ir_viewer_t::dispatch(v);
}

stmt_c codegen_llvm_vis_t::dispatch(stmt_c v) {
    emit_location(v.get());
    return ir_viewer_t::dispatch(v);
}

const std::string &codegen_llvm_vis_t::get_node_name(const expr_c &c) {
    if (c.isa<var>()) { return c.static_as<var>()->name_; }
    return c.checked_as<tensor>()->name_;
}

FunctionType *codegen_llvm_vis_t::create_func_type(const func_c &v) {
    std::vector<Type *> tys;
    for (auto &param : v->params_) {
        tys.push_back(get_type(param->dtype_));
    }
    FunctionType *FT = FunctionType::get(get_type(v->ret_type_), tys, false);
    return FT;
}

DISubroutineType *codegen_llvm_vis_t::create_func_dtype(const func_c &v) {
    std::vector<Metadata *> tys {get_type_both(v->ret_type_).second};
    for (auto &param : v->params_) {
        tys.push_back(get_type_both(param->dtype_).second);
    }
    return dbuilder_->createSubroutineType(
            dbuilder_->getOrCreateTypeArray(tys));
}

Function *codegen_llvm_vis_t::get_or_create_func(const func_c &v) {
    auto itr = name_to_func_.find(v->name_);
    if (itr != name_to_func_.end()) { return itr->second; }
    auto FT = create_func_type(v);
    bool is_private = v->attr_
            && v->attr_->get_or_else(function_attrs::private_, false);
    Function *F = Function::Create(FT,
            is_private ? Function::InternalLinkage : Function::ExternalLinkage,
            v->name_, module_.get());
    assert(FT == F->getFunctionType());
    for (size_t i = 0; i < v->params_.size(); i++) {
        (F->arg_begin() + i)->setName(get_node_name(v->params_[i]) + "_arg");
    }
    name_to_func_.insert(std::make_pair(v->name_, F));
    if (v->attr_ && v->attr_->get_or_else(function_attrs::pure, false)) {
#if SC_LLVM_BACKEND < 16
        F->addFnAttr(llvm::Attribute::AttrKind::ReadNone);
#else
        F->addFnAttr(llvm::Attribute::getWithMemoryEffects(
                context_, llvm::MemoryEffects::none()));
#endif
#if SC_LLVM_BACKEND > 10
        F->addFnAttr(llvm::Attribute::AttrKind::Speculatable);
#endif
    }
    if (v->attr_ && v->attr_->get_or_else(function_attrs::no_alias, false)) {
        F->setReturnDoesNotAlias();
    }
    F->addFnAttr(llvm::Attribute::AttrKind::NoUnwind);
    return F;
}

codegen_llvm_vis_t::~codegen_llvm_vis_t() = default;

Value *codegen_llvm_vis_t::generate_expr(const expr_c &e) {
    dispatch(e);
    return current_val_;
}
std::pair<Type *, DIType *> codegen_llvm_vis_t::do_get_type(
        sc_data_type_t dtype) {
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
            dty = dbuilder_->createBasicType("f16", 16, dwarf::DW_ATE_unsigned);
            break;
        case sc_data_etype::U16:
            ty = builder_.getInt16Ty();
            dty = dbuilder_->createBasicType("u16", 16, dwarf::DW_ATE_unsigned);
            break;
        case sc_data_etype::F32:
            ty = builder_.getFloatTy();
            dty = dbuilder_->createBasicType("f32", 32, dwarf::DW_ATE_float);
            break;
        case sc_data_etype::S32:
            ty = builder_.getInt32Ty();
            dty = dbuilder_->createBasicType("s32", 32, dwarf::DW_ATE_signed);
            break;
        case sc_data_etype::U32:
            ty = builder_.getInt32Ty();
            dty = dbuilder_->createBasicType("u32", 32, dwarf::DW_ATE_unsigned);
            break;
        case sc_data_etype::S8:
            ty = builder_.getInt8Ty();
            dty = dbuilder_->createBasicType("s8", 8, dwarf::DW_ATE_signed);
            break;
        case sc_data_etype::U8:
            ty = builder_.getInt8Ty();
            dty = dbuilder_->createBasicType("u8", 8, dwarf::DW_ATE_unsigned);
            break;
        case sc_data_etype::INDEX:
        case sc_data_etype::GENERIC:
            ty = builder_.getInt64Ty();
            dty = dbuilder_->createBasicType("u64", 64, dwarf::DW_ATE_unsigned);
            break;
        case sc_data_etype::BOOLEAN:
            ty = builder_.getInt1Ty();
            dty = dbuilder_->createBasicType("bool", 1, dwarf::DW_ATE_unsigned);
            break;
        case sc_data_etype::VOID_T:
            ty = builder_.getVoidTy();
            dty = dbuilder_->createBasicType("void", 0, dwarf::DW_ATE_address);
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

std::pair<Type *, DIType *> codegen_llvm_vis_t::get_type_both(
        sc_data_type_t dtype) {
    auto itr = type_cache_.find(dtype);
    if (itr != type_cache_.end()) { return itr->second; }
    auto ret = do_get_type(dtype);
    type_cache_.insert(std::make_pair(dtype, ret));
    return ret;
}

Type *codegen_llvm_vis_t::get_type(sc_data_type_t dtype) {
    return get_type_both(dtype).first;
}

Value *codegen_llvm_vis_t::get_defined_var_ptr(const expr_c &e) {
    auto itr = var_ptr_in_func_.find(e);
    assert(itr != var_ptr_in_func_.end());
    return itr->second;
}

Value *codegen_llvm_vis_t::define_var(const expr_c &e, Value *initv) {
    auto ptr = builder_.CreateAlloca(
            get_type(e->dtype_), nullptr, e.checked_as<var>()->name_);
    if (initv) builder_.CreateStore(initv, ptr);
    var_ptr_in_func_.insert(std::make_pair(e, ptr));
    return ptr;
}

void codegen_llvm_vis_t::set_dbg_info_for_func_arg(llvm::Value *v,
        DISubprogram *SP, DIFile *dunit, sc_data_type_t type,
        const std::string &name, int argidx, int lineno, bool need_ref) {
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

void codegen_llvm_vis_t::emit_location(const node_base *p) {
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

void codegen_llvm_vis_t::prepare_alias_metadata(
        const std::vector<alias_info::tensor_alias_identity_t *> &tensors,
        MDNode *alias_domain, MDBuilder &MDB) {
    std::unordered_set<std::shared_ptr<alias_info::alias_set_t>> cliques;
    int64_t new_tensor_id = -1;
    for (auto &aid : tensors) {
        if (aid->alias_cliques_.empty()) {
            auto new_clique = std::make_shared<alias_info::alias_set_t>();
            aid->add_to_clique(new_clique);
            new_clique->id_ = new_tensor_id;
            new_tensor_id--;
        }
        cliques.insert(aid->alias_cliques_.begin(), aid->alias_cliques_.end());
    }
    std::vector<std::shared_ptr<alias_info::alias_set_t>> cliques_sorted {
            cliques.begin(), cliques.end()};
    std::sort(cliques_sorted.begin(), cliques_sorted.end(),
            [](const std::shared_ptr<alias_info::alias_set_t> &v1,
                    const std::shared_ptr<alias_info::alias_set_t> &v2) {
                return v1->id_ < v2->id_;
            });
    std::unordered_map<alias_info::alias_set_t *, MDNode *> clique_to_MD;
    for (auto &clique : cliques_sorted) {
        MDNode *scope = MDB.createAnonymousAliasScope(
                alias_domain, std::to_string(clique->id_));
        clique_to_MD[clique.get()] = scope;
    }
    for (auto &aid : tensors) {
        auto shared_aid = aid->shared_from_this();
        vec_metadata alias_scope;
        vec_metadata noalias_scope;
        for (auto &clique : cliques_sorted) {
            if (clique->set_.has(shared_aid)) {
                alias_scope.emplace_back(clique_to_MD[clique.get()]);
            } else {
                noalias_scope.emplace_back(clique_to_MD[clique.get()]);
            }
        }
        alias_set_to_alias_scope_[aid] = {MDNode::get(context_, alias_scope),
                MDNode::get(context_, noalias_scope)};
    }
}

func_c codegen_llvm_vis_t::dispatch(func_c v) {
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
                LineNo, create_func_dtype(v), ScopeLine, DINode::FlagPrototyped,
                DISubprogram::SPFlagDefinition);
        F->setSubprogram(SP);

        // Push the current scope.
        dbg_scopes_.push_back(SP);

        // Unset the location for the prologue emission (leading
        // instructions with no location in a function are considered
        // part of the prologue and the debugger will run past them when
        // breaking on a function)
        emit_location(nullptr);
    }

    current_func_ = F;
    F->addFnAttr("no-frame-pointer-elim", "true");
    F->addFnAttr("frame-pointer", "all");
    bool has_alias = false;

    for (size_t i = 0; i < v->params_.size(); i++) {
        if (v->params_[i].isa<tensor>()) {
            auto ainfo = alias_info::get_alias_info(*v->params_[i]);
            if (ainfo && !ainfo->has_no_alias()) {
                has_alias = true;
            } else {
                F->addParamAttr(i, llvm::Attribute::AttrKind::NoAlias);
            }

            F->addParamAttr(i, llvm::Attribute::AttrKind::NoCapture);
            F->addParamAttr(i, llvm::Attribute::AttrKind::NonNull);
        }
    }
    using alias_id_vec
            = std::vector<std::shared_ptr<alias_info::tensor_alias_identity_t>>;
    std::vector<alias_info::tensor_alias_identity_t *> alias_ids;
    if (v->attr_) {
        if (auto local_tsr_alias_set
                = v->attr_->get_or_null<alias_id_vec>("alias_sets")) {
            has_alias = true;
            alias_ids.reserve(local_tsr_alias_set->size());
            for (auto &v : *local_tsr_alias_set) {
                alias_ids.push_back(v.get());
            }
        }
    }

    // if has custom alias info, need to construct alias scope/noalias
    // for LLVM each alias scope are exclusive to each other
    if (has_alias) {
        MDBuilder MDB(context_);
        MDNode *alias_domain = MDB.createAnonymousAliasScopeDomain(v->name_);

        for (size_t i = 0; i < v->params_.size(); i++) {
            if (v->params_[i].isa<tensor>()) {
                auto ainfo
                        = alias_info::get_or_create_alias_info(*v->params_[i]);
                alias_ids.push_back(ainfo.get());
            }
        }
        prepare_alias_metadata(alias_ids, alias_domain, MDB);
        for (size_t i = 0; i < v->params_.size(); i++) {
            if (v->params_[i].isa<tensor>()) {
                auto ainfo
                        = alias_info::get_or_create_alias_info(*v->params_[i]);
                tsr_to_alias_scope_[v->params_[i]]
                        = &(alias_set_to_alias_scope_[ainfo.get()]);
            }
        }
    }
    bool is_low_level = v->attr_
            && v->attr_->get_or_else(function_attrs::low_level, false);
    // LLVM func args are SSA values and cannot be modified. We use
    // alloca to alloc modifiable slots for each params
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

Instruction *codegen_llvm_vis_t::set_alias(
        Instruction *inst, const expr_c &tsr) {
    if (tsr.isa<indexing>()) {
        return set_alias(inst, tsr.static_as<indexing>()->ptr_);
    }
    auto itr = tsr_to_alias_scope_.find(tsr);
    if (itr != tsr_to_alias_scope_.end()) {
        // alias.scope metadata.
        inst->setMetadata(LLVMContext::MD_alias_scope,
                MDNode::concatenate(
                        inst->getMetadata(LLVMContext::MD_alias_scope),
                        itr->second->first));

        // noalias metadata.
        inst->setMetadata(LLVMContext::MD_noalias,
                MDNode::concatenate(inst->getMetadata(LLVMContext::MD_noalias),
                        itr->second->second));
    }
    return inst;
}

Value *codegen_llvm_vis_t::set_alias(Value *inst, const expr_c &tsr) {
    set_alias(static_cast<Instruction *>(inst), tsr);
    return inst;
}

Value *codegen_llvm_vis_t::convert_mask(const expr &in, const bool is_int4) {
    // true means mask must have 4 bits. Otherwise, it is always false.
    auto mask = generate_expr(in);
    auto &dtype = in->dtype_;
    if (dtype.lanes_ == 1 && !dtype.is_etype(sc_data_etype::BOOLEAN)) {
        auto ty_int1 = builder_.getInt1Ty();
        auto bit_len = is_int4 ? 4 : utils::get_sizeof_type(dtype) * 8;
        auto mask_ty =
#if SC_LLVM_BACKEND > 10
                VectorType::get(ty_int1, bit_len, false);
#else
                VectorType::get(ty_int1, bit_len);
#endif
        if (is_int4) {
            mask = builder_.CreateTrunc(mask, builder_.getIntNTy(4));
        }
        mask = builder_.CreateBitCast(mask, mask_ty);
    }
    return mask;
}

void codegen_llvm_vis_t::set_dbg_info_for_local_var(const source_pos *pos,
        sc_data_type_t type, const std::string &name, Value *llvm_value,
        bool need_ref) {
    if (!ctx_->flags_.debug_info_) { return; }
    auto types = get_type_both(type);
    auto dbgtype = types.second;
    if (need_ref) {
        auto tmp = builder_.CreateAlloca(llvm_value->getType());
        builder_.CreateStore(llvm_value, tmp);
        llvm_value = tmp;
    }
    // Create a debug descriptor for the variable.
    DILocalVariable *D = dbuilder_->createAutoVariable(dbg_scopes_.back(), name,
            dbg_cu_->getFile(), pos->line_, dbgtype, true);

    dbuilder_->insertDeclare(llvm_value, D, dbuilder_->createExpression(),
            DILocation::get(dbg_scopes_.back()->getContext(), pos->line_,
                    pos->pos_, dbg_scopes_.back()),
            builder_.GetInsertBlock());
}

void codegen_llvm_vis_t::set_dbg_info_for_local_var(const define_node_t *v,
        const std::string &name, Value *llvm_value, bool need_ref) {
    if (!ctx_->flags_.debug_info_) { return; }
    auto pos = v->attr_->get_or_null<source_pos>("source_pos");
    if (pos) {
        set_dbg_info_for_local_var(
                pos, v->var_->dtype_, name, llvm_value, need_ref);
    }
}

Value *codegen_llvm_vis_t::gen_vec_const(uint64_t elements, float f) {
    return builder_.CreateVectorSplat(
            elements, ConstantFP::get(builder_.getFloatTy(), APFloat(f)));
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
