/*******************************************************************************
 * Copyright 2022 Intel Corporation
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

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_TENSOR_INPLACE_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_TENSOR_INPLACE_HPP

#include <utility>
#include "tensor_inplace_info.hpp"
#include <compiler/config/context.hpp>
#include <compiler/ir/module_pass.hpp>

namespace sc {

/**
 * Buffer inplace optimization in main entry function based on "inplace_hint"
 * attr of the functions. It will also schedule local buffers of the main
 * function.
 * */
class tensor_inplace_t : public module_pass_t {
public:
    context_ptr ctx_;
    tensor_inplace_t(const context_ptr &ctx) : ctx_(std::move(ctx)) {}
    const_ir_module_ptr operator()(const_ir_module_ptr f) override;
};

} // namespace sc

#endif
