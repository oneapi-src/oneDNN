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

#include <utility>
#include <vector>
#include <compiler/jit/xbyak/sc_xbyak_jit_generator.hpp>
#include <compiler/jit/xbyak/xbyak_jit_module.hpp>
#include <runtime/config.hpp>

#ifdef _MSC_VER
#define posix_memalign(p, a, s) \
    (((*(p)) = _aligned_malloc((s), (a))), *(p) ? 0 : errno)
#endif

SC_MODULE(xbyakjit.xbyak_jit_module)

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace sc_xbyak {

xbyak_jit_module::xbyak_jit_module(
        std::shared_ptr<sc_xbyak_jit_generator> jit_output,
        statics_table_t &&globals, bool managed_thread_pool)
    : jit_module(std::move(globals), managed_thread_pool)
    , jit_output_(std::move(jit_output)) {}

void *xbyak_jit_module::get_address_of_symbol(const std::string &name) {
    void *global_var = globals_.get_or_null(name);
    if (global_var) { return global_var; }
    return jit_output_->get_func_address(name);
}

std::shared_ptr<jit_function_t> xbyak_jit_module::get_function(
        const std::string &name) {
    void *fun = jit_output_->get_func_address(name);
    void *wrapper = jit_output_->get_func_address(name + "_0wrapper");
    if (fun || wrapper) {
        if (runtime_config_t::get().execution_verbose_) {
            return general_jit_function_t::make(shared_from_this(), fun,
                    wrapper, name, managed_thread_pool_);
        } else {
            return general_jit_function_t::make(shared_from_this(), fun,
                    wrapper, std::string(), managed_thread_pool_);
        }
    } else {
        return nullptr;
    }
}

} // namespace sc_xbyak
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
