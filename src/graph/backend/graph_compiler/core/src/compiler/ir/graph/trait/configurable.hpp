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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_TRAIT_CONFIGURABLE_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_TRAIT_CONFIGURABLE_HPP

#include <memory>
#include <vector>
#include <compiler/ir/graph/traits.hpp>
#include <unordered_map>
#include <util/general_object.hpp>
#include <util/reflection.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
using config_ptr_vec = std::vector<reflection::shared_general_object_t>;
using impl_kind_map = std::unordered_map<std::vector<uint64_t>, int>;
namespace op_traits {
struct configurable_t : public virtual op_base_trait_t {
    virtual reflection::shared_general_object_t get_config() = 0;

    virtual void set_config(const reflection::shared_general_object_t &config)
            = 0;
    virtual reflection::shared_general_object_t get_default_config(
            context_ptr ctx)
            = 0;
    // Get config space in dynamic shape case, similar with get_config_space but
    // due to the number limit of kernels, the space is shrinked from static
    // config space. Return the vector of config ptr, index is the impl kind.
    virtual config_ptr_vec get_dynamic_config_candidates(const context_ptr &ctx)
            = 0;
    // Convert the input config space vector to a unordered map for query in
    // runtime. This function accepts the configs from output of
    // `get_dynamic_config_candidates`. We need the intermediate
    // result(config_ptr_vec) so we split the implement into this two functions.
    virtual impl_kind_map convert_config_candidates_to_impl_map(
            const config_ptr_vec &configs)
            = 0;
};

} // namespace op_traits
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
