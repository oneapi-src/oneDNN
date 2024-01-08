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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_CPU_CLOSURIZE_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_CPU_CLOSURIZE_HPP

#include <compiler/ir/module_pass.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
/**
 * Base class for closurizer. Different targets should extend the
 * class to make target-specific parallel functions and parallel calls. Replaces
 * parallel for nodes with call_parallel nodes. Also moves the bodies of the
 * parallel for nodes to new closure functions. The closure functions have the
 * interface like `void closure_name(uint64_t i, T1 capture1, T2 capture2,
 * ...);`, where `i` is the loop variable, and captureX is the captured
 * variables that may be used in the body of the original for-loop.
 *
 * The target specific subclass should at least override `make_closure_func` and
 * `make_parallel_call`. See comments below.
 *
 * */

/**
 * Replaces parallel for nodes with call_parallel nodes for CPU
 * @see closurize_impl_t
 * This pass will also generate a wrapper function for each closure on CPU
 * backend, which has signature `void closure_name_wrapper(uint64_t i,
 * generic_val* args);`. The wrapper will extract the arguments from `args` and
 * call the closure.
 *
 *  the for-loop will finally replaced by:
 *  {
 *     tensor argbuf: [generic * X]
 *     argbuf[0] = capture_0;
 *     ...
 *     argbuf[X-1] = capture_X_1;
 *     parallel_call(closure_wrapper_func, argbuf, parallel_attr = {...})
 *  }
 * The backend should lower parallel_call into a call to `sc_parallel_call_cpu`
 * */
class closurizer_cpu_t : public module_pass_t {
public:
    bool single_core_;
    closurizer_cpu_t(bool single_core) : single_core_(single_core) {}
    const_ir_module_ptr operator()(const_ir_module_ptr m) override;
    SC_DECL_PASS_INFO_FUNC();
};

func_t get_parallel_call_with_env_func(bool managed);
func_c remove_parallel_on_func(const func_c &f);
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
