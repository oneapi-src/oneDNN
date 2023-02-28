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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_VISITABLE_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_VISITABLE_HPP

#include "visitor.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
template <typename T, typename Base>
node_ptr<Base, Base> visitable_t<T, Base>::visited_by(ir_visitor_base_t *vis) {
    using ptr_ty = node_ptr<T, Base>;
    return vis->visit_impl(static_cast<T *>(this)
                                   ->node_ptr_from_this()
                                   .template static_as<ptr_ty>());
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
