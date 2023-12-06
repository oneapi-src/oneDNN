/*******************************************************************************
 * Copyright 2020-2023 Intel Corporation
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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_CFAKE_CFAKE_JIT_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_CFAKE_CFAKE_JIT_HPP

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <compiler/ir/sc_function.hpp>
#include <compiler/jit/jit.hpp>
#include <runtime/generic_val.hpp>
#include <runtime/target_machine.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

class cfake_jit;
class SC_INTERNAL_API cfake_jit_module_code_t : public jit_module_code {
    friend cfake_jit;
    void *module_;
    std::string path_;
    std::string src_path_;
    cfake_jit_module_code_t(void *module, const std::string &src_path,
            const std::string &path, bool has_generic_wrapper,
            thread_pool_mode_t managed_thread_pool)
        : jit_module_code(managed_thread_pool)
        , module_(module)
        , path_(path)
        , src_path_(src_path) {}
    cfake_jit_module_code_t(cfake_jit_module_code_t &&other) = delete;
    cfake_jit_module_code_t(const cfake_jit_module_code_t &other) = delete;

public:
    ~cfake_jit_module_code_t() override;
    std::vector<std::string> get_temp_filenames() const override {
        return {path_, src_path_};
    }

    void *get_address_of_symbol(const std::string &name) override;
    void *get_function(const std::string &name, void *&wrapperfunc) override;
};

struct c_generator_optional_out_t;

class SC_INTERNAL_API cfake_jit : public jit_engine_t {
public:
    cfake_jit(context_ptr ctx = get_default_context())
        : jit_engine_t(std::move(ctx)) {
        opt_level_ = context_->flags_.backend_opt_level_;
        debug_info_ = opt_level_ <= 1 || context_->flags_.debug_info_;
    }
    unsigned opt_level_;
    bool debug_info_;

    statics_table_t codegen_to_cpp(std::ostream &os,
            const_ir_module_ptr &new_mod, const const_ir_module_ptr &module,
            bool generate_wrapper);
    statics_table_t codegen_to_cpp(std::ostream &os,
            const_ir_module_ptr &new_mod, const const_ir_module_ptr &module,
            bool generate_wrapper, thread_pool_mode_t &out_managed_thread_pool,
            c_generator_optional_out_t *optional_out = nullptr);
    std::shared_ptr<jit_module> make_jit_module(
            const_ir_module_ptr module, bool generate_wrapper) override;
    std::shared_ptr<jit_module> make_jit_module(const std::string &inpath,
            const std::string &outpath, statics_table_t &&globals,
            bool has_generic_wrapper,
            thread_pool_mode_t managed_thread_pool) const;
    static const runtime::cpu_flags_t &get_compiler_flags();
    static std::string &get_compiler_command();
    static void set_target_machine(runtime::target_machine_t &tm);
};

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
