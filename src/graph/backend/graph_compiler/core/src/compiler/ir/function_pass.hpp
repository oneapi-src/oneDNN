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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_FUNCTION_PASS_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_FUNCTION_PASS_HPP

#include <memory>
#include <vector>
#include "pass_info_macros.hpp"
#include "sc_function.hpp"
#include <util/utils.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
struct tir_pass_dependency_t;

/**
 * The base abstruct class of all function passes
 * */
class function_pass_t {
public:
    virtual func_c operator()(func_c f) = 0;
    virtual ~function_pass_t() = default;
    virtual const char *get_name() const { return nullptr; }
#ifndef NDEBUG
    virtual void get_dependency_info(tir_pass_dependency_t &out) const;
#endif
};

using function_pass_ptr = std::unique_ptr<function_pass_t>;
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
