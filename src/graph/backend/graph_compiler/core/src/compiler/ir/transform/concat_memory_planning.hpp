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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_CONCAT_MEMORY_PLANNING_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_CONCAT_MEMORY_PLANNING_HPP

#include <compiler/ir/module_pass.hpp>
#include <compiler/ir/sc_function.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

namespace concat_optim_attr_keys {
constexpr const char *graph_memory_offset = "memory_offset";
constexpr const char *pass_memory_offset = "memory_offset";
constexpr const char *graph_memory_offset_to = "memory_offset_to";
constexpr const char *pass_memory_offset_to = "memory_offset_to";
constexpr const char *is_final_concat = "is_final_concat";
constexpr const char *is_standalone_concat = "is_standalone_concat";
} // namespace concat_optim_attr_keys

/*
All inputs and output of concat share the strides_ vector. The input tensor
of concat is a tensorptr on output tensor of concat. The output tensor of
concat is dense, and its's strides_ is straightforward. The input tensor is
strided, but we do not need to set its strides_, because the strides_ of a
tensorptr is from the strides_ of its base. But different inputs have
different offsets and we need to set them.
*/

/*
For the following graph:
    op0   op1   op2
      \    |    /
       \   |   /
         concat
           |
           |
          op3
There are four tensors: output of op0, output of op1, output of op2, output of
concat. After optimization, there is only one tensor (and only one buffer):
output of concat. The output of op0, output of op1 and output of op2
are offset-and-strided tensorptrs from the output of concat.
In GraphIR, we use graph_concat_memory_planning pass to set the strides and
offsets to the output of op0, output of op1 and output of op2.
In TensorIR, we only allocate the output buffer of concat.
*/

class concat_memory_planning_t : public module_pass_t {
public:
    const_ir_module_ptr operator()(const_ir_module_ptr f) override;
    SC_DECL_PASS_INFO_FUNC();
};

bool is_standalone_concat_call(call_c &v);

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
