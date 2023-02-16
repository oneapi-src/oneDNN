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

#include "tir_pos_trace.hpp"
#include <compiler/ir/pass/printer.hpp>
#include <util/utils.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

std::string tir_pos_tracer::to_string() const {
    if (cur_func_ && cur_node_ && utils::compiler_configs_t::get().diagnose_) {
        std::stringstream ss;
        ss << '\n';
        auto func = dynamic_cast<const func_base *>(cur_func_);
        print_ir_and_annotate_position_in_source(
                func->shared_from_this(), cur_node_, ss);
        return ss.str();
    }
    return std::string();
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
