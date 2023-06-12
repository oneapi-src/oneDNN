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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_DYN_TSR_TRANSFORM_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_DYN_TSR_TRANSFORM_HPP
#include "../module_pass.hpp"
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace attr_keys {
constexpr const char *plain_dims = "pass.plain_dims";
constexpr const char *always_trans = "pass.always_trans";
constexpr const char *forbid_trans = "pass.forbid_trans";
} // namespace attr_keys
/**
 * Do dynamic tensor transform, tensor_node=>dynamic_tensor(void pointer in IR,
 * actually runtime struct). Extract and define vars from tensor when enter the
 * function. This pass only acts on tensors who has dynamic shape(var_node)
 * */
class dyn_tensor_transformer_t : public module_pass_t {
public:
    func_c operator()(func_c f);
    const_ir_module_ptr operator()(const_ir_module_ptr f) override;
    SC_DECL_PASS_INFO_FUNC();
};
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
