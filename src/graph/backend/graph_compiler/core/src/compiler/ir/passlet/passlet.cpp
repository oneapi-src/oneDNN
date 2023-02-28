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
#include "passlet.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace passlet {

#define SC_PASSLET_IMPL_METHODS_IMPL(node_type, ...) \
    void passlet_t::view(const node_type##_c &v, pass_phase phase) {}

FOR_EACH_EXPR_IR_TYPE(SC_PASSLET_IMPL_METHODS_IMPL)
FOR_EACH_STMT_IR_TYPE(SC_PASSLET_IMPL_METHODS_IMPL)
FOR_EACH_BASE_EXPR_IR_TYPE(SC_PASSLET_IMPL_METHODS_IMPL)
SC_PASSLET_IMPL_METHODS_IMPL(func)
SC_PASSLET_IMPL_METHODS_IMPL(expr)
SC_PASSLET_IMPL_METHODS_IMPL(stmt)

} // namespace passlet
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
