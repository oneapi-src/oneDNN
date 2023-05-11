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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_TENSOR_INPLACE_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_TENSOR_INPLACE_HPP

#include <utility>
#include "tensor_inplace_info.hpp"
#include <compiler/config/context.hpp>
#include <compiler/ir/module_pass.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

/**
 * Buffer inplace optimization in main entry function based on "inplace_hint"
 * attr of the functions. It will decide the specific tensor to reuse for each
 * function arg called by main entry. For each output tensor of function, this
 * pass then narrows down the "inplace_hint" to a single selected candidate.
 * This pass also sets the pointer alias info to help index2var and dead write
 * elim pass to correctly handle inplaced tensor. Note that this pass will only
 * modify the attrs of IR and will not change the IR itself. The real tensor
 * inplace and allocation happens in buffer_schedule pass.
 * */
class tensor_inplace_t : public module_pass_t {
public:
    context_ptr ctx_;
    tensor_inplace_t(const context_ptr &ctx) : ctx_(std::move(ctx)) {}
    const_ir_module_ptr operator()(const_ir_module_ptr f) override;
    SC_DECL_PASS_INFO_FUNC();
};

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
