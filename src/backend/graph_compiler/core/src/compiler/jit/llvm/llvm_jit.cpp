/*******************************************************************************
 * Copyright 2020-2021 Intel Corporation
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
#include <memory>
#include <string>
#include <utility>

#include "llvm_jit.hpp"
#include "llvm_jit_resolver.hpp"
#include <compiler/codegen/codegen_llvm.hpp>
#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/MCJIT.h>
#include <llvm/ExecutionEngine/SectionMemoryManager.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/DynamicLibrary.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/IPO/AlwaysInliner.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include <runtime/config.hpp>
#include <runtime/runtime.hpp>

namespace sc {

static void optimize_llvm_module(llvm::TargetMachine *tm, llvm::Module *module,
        llvm::CodeGenOpt::Level llvm_opt) {
    llvm::PassManagerBuilder passbuilder;
    passbuilder.OptLevel = llvm_opt;
    passbuilder.SizeLevel = 0;
    passbuilder.LoopVectorize = true;
    passbuilder.SLPVectorize = true;
    llvm::legacy::PassManager MPM;
    llvm::legacy::FunctionPassManager FPM(module);
    tm->adjustPassManager(passbuilder);

    passbuilder.Inliner = llvm::createFunctionInliningPass(
            static_cast<unsigned int>(llvm_opt), 0, false);

    MPM.add(new llvm::TargetLibraryInfoWrapperPass(tm->getTargetTriple()));
    MPM.add(llvm::createTargetTransformInfoWrapperPass(
            tm->getTargetIRAnalysis()));
    FPM.add(llvm::createTargetTransformInfoWrapperPass(
            tm->getTargetIRAnalysis()));

    passbuilder.populateFunctionPassManager(FPM);
    passbuilder.populateModulePassManager(MPM);

    passbuilder.populateLTOPassManager(MPM);

    FPM.doInitialization();
    for (llvm::Function &F : *module)
        FPM.run(F);
    FPM.doFinalization();
    MPM.run(*module);
}
std::unique_ptr<llvm::TargetMachine> get_llvm_target_machine(
        llvm::CodeGenOpt::Level optlevel);

std::shared_ptr<jit_module> llvm_jit::make_jit_module(
        const_ir_module_ptr module, bool generate_wrapper) {
    auto llvm_ctx = utils::make_unique<llvm::LLVMContext>();
    std::unique_ptr<llvm::Module> llvmmod;
    llvm_generator_pass gen {*llvm_ctx, llvmmod, generate_wrapper};
    auto new_mod = gen(module);

    auto &attr_table = *new_mod->attr_.get<std::shared_ptr<statics_table_t>>(
            ir_module_t::attr_key_t::MODULE_DATA_BUFFERS);

    std::string err;
    llvm::raw_string_ostream ss(err);
    if (llvm::verifyModule(*llvmmod, &ss)) {
        ss << "Module:\n" << *llvmmod;
        throw std::runtime_error("LLVM module verify error: " + err);
    }

    auto opt = std::min(opt_level_, 3U);
    auto llvm_opt = static_cast<llvm::CodeGenOpt::Level>(opt);

    llvm::Module *mod_ptr = llvmmod.get();
    auto tm = get_llvm_target_machine(llvm_opt).release();
    optimize_llvm_module(tm, mod_ptr, llvm_opt);
    auto engine = llvm::EngineBuilder(std::move(llvmmod))
                          .setErrorStr(&err)
                          .setOptLevel(llvm_opt)
                          .setEngineKind(llvm::EngineKind::JIT)
                          .setMCJITMemoryManager(
                                  utils::make_unique<sc_llvm_jit_resolver>())
                          .create(tm);
    if (!engine) {
        throw std::runtime_error("LLVM EngineBuilder error: " + err);
    }
    engine->finalizeObject();
    typedef void (*init_func_t)(void *ctx, void *mod);
    auto init_func = reinterpret_cast<init_func_t>(
            engine->getPointerToNamedFunction("__sc_init__", false));
    if (init_func) { init_func(nullptr, attr_table.data_.data_); }
    return std::make_shared<llvm_jit_module>(
            std::unique_ptr<llvm::ExecutionEngine>(engine), std::move(llvm_ctx),
            std::move(attr_table));
}

llvm_jit_module::llvm_jit_module(std::unique_ptr<llvm::ExecutionEngine> engine,
        std::unique_ptr<llvm::LLVMContext> llvm_ctx, statics_table_t &&globals)
    : llvm_ctx_(std::move(llvm_ctx))
    , engine_(std::move(engine))
    , globals_(std::move(globals)) {}

llvm_jit_module::~llvm_jit_module() = default;

void *llvm_jit_module::get_address_of_symbol(const std::string &name) {
    void *global_var = globals_.get_or_null(name);
    if (global_var) { return global_var; }

    return engine_->getPointerToNamedFunction(name, false);
}
std::shared_ptr<jit_function_t> llvm_jit_module::get_function(
        const std::string &name) {
    void *fun = engine_->getPointerToNamedFunction(name, false);
    void *wrapper
            = engine_->getPointerToNamedFunction(name + "_0wrapper", false);
    if (fun || wrapper) {
        if (runtime_config_t::get().execution_verbose_) {
            return std::make_shared<llvm_jit_function>(
                    shared_from_this(), fun, wrapper, name);
        } else {
            return std::make_shared<llvm_jit_function>(
                    shared_from_this(), fun, wrapper);
        }
    } else {
        return nullptr;
    }
}

// todo: merge with other JIT engines
void llvm_jit_function::call_generic(
        runtime::stream_t *stream, generic_val *args) const {
    assert(wrapper_ && "Trying to call 'call_generic' on a jit funciton with no wrapper."); // NOLINT
    using functype = void (*)(runtime::stream_t *, void *, generic_val *);
    functype f = reinterpret_cast<functype>(wrapper_);
    if (runtime_config_t::get().execution_verbose_) {
        using namespace std::chrono;
        auto start = steady_clock::now();
        f(stream, module_->globals_.data_.data_, args);
        auto stop = steady_clock::now();
        double duration
                = static_cast<double>(
                          duration_cast<nanoseconds>(stop - start).count())
                / 1e6;
        std::cout << "Entry point: " << fname_ << ". Time elapsed: " << duration
                  << " ms" << std::endl;
    } else {
        f(stream, module_->globals_.data_.data_, args);
    }
}

void llvm_jit_function::call_generic(
        runtime::stream_t *stream, void *module_data, generic_val *args) const {
    assert(wrapper_ && "Trying to call 'call_generic' on a jit funciton with no wrapper."); // NOLINT

    using functype = void (*)(runtime::stream_t *, void *, generic_val *);
    functype f = reinterpret_cast<functype>(wrapper_);
    f(stream, module_->globals_.data_.data_, args);
}

} // namespace sc
