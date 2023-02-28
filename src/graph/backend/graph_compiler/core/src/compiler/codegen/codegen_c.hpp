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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_CODEGEN_CODEGEN_C_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_CODEGEN_CODEGEN_C_HPP

#include <ostream>

#include <ostream>
#include "../ir/ir_module.hpp"
#include "../ir/util_module_passes.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

struct c_generator_optional_out_t {
    std::ostream *offline_source_;
    std::ostream *header_source_;
    std::ostream *data_source_;
};

class SC_INTERNAL_API c_generator_pass_t : public module_pass_t {
private:
    std::ostream &source_;
    context_ptr context_;
    bool gen_wrapper_;
    sequential_module_pass_t pre_passes_;
    c_generator_optional_out_t *optional_out_;

public:
    const_ir_module_ptr operator()(const_ir_module_ptr f) override;
    /**
     * Generates a single function to the stream
     * */
    void operator()(func_t f);

    c_generator_pass_t(std::ostream &source, const context_ptr &ctx,
            bool gen_wrapper,
            c_generator_optional_out_t *optional_out = nullptr);
};

/**
 * Creates a pass pipeline. It finds all dependent functions of the function.
 * For each function, it runs legalization and optimization. Then runs
 * codegen-to-c on it. Finally outputs the C++ source code to `os`. In itself,
 * the pipeline remembers all functions that has already been generated to
 * prevent multi-defining the functions
 *
 * @param os the stream to output the generated source code
 * @param ctx the context
 * @param gen_wrapper if true, generates a function "NAME_wrapper" which has
 *      type erased prototype
 * @param optional_out the optional output for AOT offline mode codegen
 * */
SC_INTERNAL_API c_generator_pass_t create_c_generator(std::ostream &os,
        const context_ptr &ctx, bool gen_wrapper = false,
        c_generator_optional_out_t *optional_out = nullptr);
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
