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

#include "util_module_passes.hpp"
#include <memory>
#include <utility>
#include <vector>

namespace sc {
sequential_module_pass_t::sequential_module_pass_t(
        std::vector<module_pass_ptr> &&passes)
    : passes_(std::move(passes)) {}

sequential_module_pass_t::sequential_module_pass_t(
        sequential_module_pass_t &&other)
    : passes_(std::move(other.passes_)) {}

const_ir_module_ptr sequential_module_pass_t::operator()(
        const_ir_module_ptr f) {
    for (auto &p : passes_) {
        f = (*p)(f);
    }
    return f;
}

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

} // namespace sc
