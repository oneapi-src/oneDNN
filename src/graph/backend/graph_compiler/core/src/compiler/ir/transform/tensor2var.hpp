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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_TENSOR2VAR_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_TENSOR2VAR_HPP

#include <vector>
#include "../function_pass.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

namespace attr_keys {
// bool. applied on tensor node
constexpr const char *no_tensor2var = "no_tensor2var";
// bool. applied on tensor node. Throw an error if the tensor cannot be
// transformed to var
constexpr const char *must_tensor2var = "must_tensor2var";
} // namespace attr_keys

/**
 * Replace small local tensors with vars. If all indexing nodes on a local
 * tensor have constant indices and the SIMD length are the same, we will
 * replace the tensor with a (list of) local var(s).
 * */
class tensor2var_t : public function_pass_t {
public:
    func_c operator()(func_c f) override;
    SC_DECL_PASS_INFO_FUNC();
};

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
