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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_CPU_LOCAL_TENSOR_LOWER_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_CPU_LOCAL_TENSOR_LOWER_HPP

#include "../../function_pass.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

/**
 * Insert allocation/deallocation calls for local tensor. The alloc/dealloc
 * follows "first-alloc-last-dealloc" order to let the memory allocator use a
 * stack for easier and faster implementation. This pass should be placed after
 * module_globals_resolver_t
 * */
class local_tensor_lowering_cpu_t : public function_pass_t {
public:
    // the threshold in bytes. If a local tensor is larger than the threshold,
    // we should allocate it on heap (using alloc function). Otherwise, we
    // should keep the tensor untouched (the backend should lower it a buffer on
    // stack)
    size_t size_threshold_;
    func_c operator()(func_c m) override;
    local_tensor_lowering_cpu_t(size_t size_threshold)
        : size_threshold_(size_threshold) {}
    SC_DECL_PASS_INFO_FUNC();
};

func_t get_cpu_temp_malloc_func(bool is_thread_local);

func_t get_cpu_temp_free_func(bool is_thread_local);
func_t get_acquire_const_cache_func();
func_t get_release_const_cache_func();

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
