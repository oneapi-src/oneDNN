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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_POINTER_ALIAS_INFO_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_POINTER_ALIAS_INFO_HPP

#include <memory>
#include <vector>
#include <unordered_set>
#include <util/weakptr_utils.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

namespace attr_keys {
constexpr const char *pointer_alias = "pointer_alias";
}

class expr_base;

namespace alias_info {
struct tensor_alias_identity_t;
using tensor_alias_indentity_ptr = std::weak_ptr<tensor_alias_identity_t>;
struct alias_set_t;

// the struct to identify a tensor. We don't use "expr" or "tensor" pointers
// because in passes they will be copied/remade with a different address and the
// alias info will lose track of them
// the tensors that may alias are stored in cliques (a graph theory concepts).
// Tensors in a clique are alias of each other. A tensor may alias to multiple
// cliques. The union of clique sets should be the set of alias tensor of this
// tensor. The union of clique sets that the tensor does not belong to should be
// the "noalias" set of this tensor.
struct tensor_alias_identity_t
    : std::enable_shared_from_this<tensor_alias_identity_t> {
    // push the clique to alias_cliques_ and add this tensor to the clique
    void add_to_clique(const std::shared_ptr<alias_set_t> &v);
    bool has_no_alias() const;
    tensor_alias_identity_t() = default;
    bool is_alias_of(tensor_alias_identity_t *) const;
    std::vector<std::shared_ptr<alias_set_t>> alias_cliques_;
};

// a set of tensors. tensors in the same set may have pointer alias
struct alias_set_t {
    utils::weakptr_hashset_t<tensor_alias_identity_t> set_;
    // the identifier for sorting to make the resulting IR stable
    int64_t id_;

    // copy the alias set, also add the new set to all its tensor_alias_identity
    std::shared_ptr<alias_set_t> copy() const;
};

std::shared_ptr<tensor_alias_identity_t> get_or_create_alias_info(expr_base &v);
tensor_alias_identity_t *get_alias_info(const expr_base &v);

} // namespace alias_info

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
