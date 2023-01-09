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

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_TRAIT_CONFIGURABLE_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_TRAIT_CONFIGURABLE_HPP

#include <memory>
#include <compiler/ir/graph/traits.hpp>
#include <util/general_object.hpp>

namespace sc {
namespace op_traits {
struct configurable_t : public virtual op_base_trait_t {
    virtual reflection::shared_general_object_t get_config() = 0;

    virtual void set_config(const reflection::shared_general_object_t &config)
            = 0;
    virtual reflection::shared_general_object_t get_default_config(
            context_ptr ctx)
            = 0;
};

} // namespace op_traits
} // namespace sc

#endif
