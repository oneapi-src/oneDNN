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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_PASS_MANAGER_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_PASS_MANAGER_HPP

#include <memory>
#include <stdint.h>
#include <vector>
#include <compiler/config/context.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
class module_pass_t;
#ifndef NDEBUG
void validate_pass_order(const context_ptr &ctx,
        const std::vector<std::unique_ptr<module_pass_t>> &passes,
        bool gen_wrapper);
#else
#define validate_pass_order(ctx, passes, gen_wrapper)
#endif
const char *get_pass_name(module_pass_t *pass);
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
