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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_PASS_FUNC_DEPENDENCY_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_PASS_FUNC_DEPENDENCY_HPP

#include <vector>
#include "../function_pass.hpp"
#include "../sc_function.hpp"
#include <unordered_set>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

/**
 * Walks through the function's body and find the functions that the function
 * may call.
 *
 * @param dep the output result array of all functions that this function may
 *      call. Different function (of different pointers) occurs once in the
 *      resulting array. The functions are pushed into the resulting array in
 *      the order that they are called in the function
 * */
class func_dependency_finder_t : public function_pass_t {
    std::vector<func_t> &dep_;

public:
    func_c operator()(func_c f) override;
    stmt_c operator()(stmt_c f);
    /**
     * @param f the stmt to walk through
     * @param func in & out, the function already met before. Will insert new
     * dependencies after calling this function
     * */
    stmt_c operator()(stmt_c f, std::unordered_set<func_t> &funcs);
    /**
     * @param f the func_t to walk through
     * @param func in & out, the function already met before. Will insert new
     * dependencies after calling this function
     * */
    func_c operator()(func_c f, std::unordered_set<func_t> &funcs);
    func_dependency_finder_t(std::vector<func_t> &dep) : dep_(dep) {}
};

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
