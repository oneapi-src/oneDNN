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

#include "util_module_passes.hpp"
#include <chrono>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "pass_manager.hpp"
#include "visitor.hpp"
#include <util/scoped_timer.hpp>
#include <util/utils.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
sequential_module_pass_t::sequential_module_pass_t(
        std::vector<module_pass_ptr> &&passes)
    : passes_(std::move(passes)) {}

sequential_module_pass_t::sequential_module_pass_t(
        sequential_module_pass_t &&other)
    : passes_(std::move(other.passes_)) {}

const_ir_module_ptr sequential_module_pass_t::operator()(
        const_ir_module_ptr f) {
    bool need_print_time = utils::compiler_configs_t::get().print_pass_time_;
    bool need_result = utils::compiler_configs_t::get().print_pass_result_;
    for (auto &p : passes_) {
        auto timer = utils::create_scoped_timer(
                need_print_time, [&p](utils::time_duration dur) {
                    auto diff = std::chrono::duration_cast<
                            std::chrono::microseconds>(dur)
                                        .count();
                    std::string mod_name = std::string("pass.time.")
                            + get_pass_name(p.get());
                    SC_MODULE_INFO2(mod_name.c_str())
                            << "The pass took " << diff << "us";
                });
        f = (*p)(f);
        if (need_result) {
            std::string mod_name
                    = std::string("pass.debug.") + get_pass_name(p.get());
            SC_MODULE_INFO2(mod_name.c_str()) << f;
        }
    }
    return f;
}

const char *module_function_pass_t::get_name() const {
    return impl_->get_name();
}

#ifndef NDEBUG
void module_function_pass_t::get_dependency_info(
        tir_pass_dependency_t &out) const {
    impl_->get_dependency_info(out);
}
#endif

module_function_pass_t::module_function_pass_t(function_pass_ptr impl)
    : impl_(std::move(impl)) {}

const_ir_module_ptr module_function_pass_t::operator()(const_ir_module_ptr m) {
    auto ret = m->copy();
    ret->run_pass(*impl_);
    return ret;
}
const_ir_module_ptr dispatch_module_on_visitor(
        ir_visitor_t *pass, const const_ir_module_ptr &f) {
    auto ret = std::make_shared<ir_module_t>(*f);
    for (auto &gv : ret->get_module_vars()) {
        gv = pass->visit(gv).checked_as<define>();
    }
    for (auto &funct : ret->get_contents()) {
        funct = std::const_pointer_cast<func_base>(pass->dispatch(funct));
    }
    return ret;
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
