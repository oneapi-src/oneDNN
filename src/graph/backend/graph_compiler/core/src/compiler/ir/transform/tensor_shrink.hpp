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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_TENSOR_SHRINK_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_TENSOR_SHRINK_HPP

#include <vector>
#include "../module_pass.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

namespace tensor_shrinker_attrs {
constexpr const char *should_shrink = "should_shrink";
constexpr const char *may_shrink = "may_shrink";
constexpr const char *no_shrink = "no_shrink";
constexpr const char *tensor_for_placerholder = "tsr4placeholder";
} // namespace tensor_shrinker_attrs

/**
 * Shrinks large tensors into small ones if the access pattern is limited in a
 * range. This pass depends on the attr "should_shrink" marked on tensors, which
 * is manually added by users or automatically added by fusion manager. The
 * "should_shrink" attr of a tensor should be mapped to an array of expr, say,
 * `base` and another array of expr for shape. The original tensor should be
 * replaced by a shrinked tensor with shape as the shape given in
 * "should_shrink" attr. The accesses on original tensor `A[idx]` should be
 * mapped to the accesses on the shrinked tensor `shrinked_A[idx - base]`
 * */
class tensor_shrinker_t : public module_pass_t {
public:
    struct shrink_info_t {
        std::vector<expr> base_;
        std::vector<expr> shape_;
        // the placeholder for the location of the new definition position of
        // the tenosr. Can be null, indicating the tensor does not need to be
        // moved
        stmts move_def_;
    };
    const_ir_module_ptr operator()(const_ir_module_ptr f) override;
    func_c operator()(func_c f);
    stmt_c operator()(stmt_c f);
    SC_DECL_PASS_INFO_FUNC();
};

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
