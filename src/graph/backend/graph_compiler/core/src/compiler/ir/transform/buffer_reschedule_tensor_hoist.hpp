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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_BUFFER_RESCHEDULE_TENSOR_HOIST_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_BUFFER_RESCHEDULE_TENSOR_HOIST_HPP

#include <utility>
#include "../function_pass.hpp"
#include <compiler/config/context.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

/**
 * buffer rescheduling and tensor hoisting recursively
 * before buffer_schedule pass and nested_parallel_flatten pass
 * */
class buffer_rescheduling_tensor_hoisting_t : public function_pass_t {
public:
    // params for buffer rescheduling
    context_ptr ctx_;
    bool eliminate_dead_writes_;
    bool do_inplace_opt_;

    buffer_rescheduling_tensor_hoisting_t(context_ptr ctx,
            bool eliminate_dead_writes, bool do_inplace_opt = false)
        : ctx_(std::move(ctx))
        , eliminate_dead_writes_(eliminate_dead_writes)
        , do_inplace_opt_(do_inplace_opt) {}

    func_c operator()(func_c f) override;

    SC_DECL_PASS_INFO_FUNC();
};

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
