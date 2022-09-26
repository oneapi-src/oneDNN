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

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_POINTER_ALIAS_INFO_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_POINTER_ALIAS_INFO_HPP

#include <memory>
#include <unordered_set>
#include <util/weakptr_utils.hpp>

namespace sc {

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
struct tensor_alias_identity_t
    : std::enable_shared_from_this<tensor_alias_identity_t> {
    void add_alias(const std::shared_ptr<tensor_alias_identity_t> &v);
    bool has_no_alias() const;
    tensor_alias_identity_t() = default;
    const std::shared_ptr<alias_set_t> &get_alias_set();

private:
    std::shared_ptr<alias_set_t> alias_set_;
};

// a set of tensors. tensors in the same set may have pointer alias
struct alias_set_t {
    utils::weakptr_hashset_t<tensor_alias_identity_t> set_;
};

std::shared_ptr<tensor_alias_identity_t> get_or_create_alias_info(expr_base &v);
tensor_alias_identity_t *get_alias_info(const expr_base &v);

} // namespace alias_info

} // namespace sc

#endif
