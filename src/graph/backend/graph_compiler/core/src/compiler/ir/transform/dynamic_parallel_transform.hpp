/*******************************************************************************
 * Copyright 2023 Intel Corporation
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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_DYNAMIC_PARALLEL_TRANSFORM_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_DYNAMIC_PARALLEL_TRANSFORM_HPP

#include <compiler/ir/module_pass.hpp>
#include <compiler/ir/sc_function.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

/**
 * Split nested parallel-for loops into "scopes" and runtime library calls
 * A "scope" is a basic block for parallel-for body, without any other nested
 * parallel loops.
 * The parallel body:
 * pfor() {
 *  A()
 *  pfor() {
 *      C()
 *  }
 *  B()
 * }
 * .. will have 3 scopes containing A(), B(), C() respectively.
 * This pass also handles dynamic buffer management, if a buffer is shared by
 * multiple scopes. A more complicated example:
 *
 * buffer B
 * pfor(i) {
 *  use(i)
 *  buffer A
 *  pfor(j) {
 *      use(i,j)
 *      pfor(k) {
 *          use(A,B)
 *          use(i,j,k)
 *      }
 *  }
 * }
 *
 * ... will be transformed to
 * buffer B
 * submit(func1)
 * func1 = (i,B) {
 *  use(i)
 *  buffer A = shared_malloc();
 *  submit(func2, A, B)
 * }
 * func2 = (i,j,A,B) {
 *      use(i,j)
 *      submit(func3, A, B)
 * }
 * func3 = (i,j,k,A,B) {
 *          use(A,B)
 *          use(i,j,k)
 * }
 */
class dynamic_parallel_transform_t : public module_pass_t {
public:
    // transform all parallel-for loops and disregard whether the parallel-for
    // is in dynamic mode
    bool always_transform_;
    dynamic_parallel_transform_t(bool always_transform)
        : always_transform_ {always_transform} {}
    const_ir_module_ptr operator()(const_ir_module_ptr f) override;
    SC_DECL_PASS_INFO_FUNC();
};

func_t get_dyn_threadpool_shared_buffer_func();
func_t get_dyn_threadpool_loop_end_func();
func_t get_dyn_threadpool_submit_func();
func_t get_dyn_threadpool_run_func();
func_t get_dyn_threadpool_destroy_func();
func_t get_dyn_threadpool_init_func();

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
