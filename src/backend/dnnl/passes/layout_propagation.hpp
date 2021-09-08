/*******************************************************************************
 * Copyright 2021 Intel Corporation
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
#ifndef BACKEND_DNNL_PASSES_LAYOUT_PROPAGATION_HPP
#define BACKEND_DNNL_PASSES_LAYOUT_PROPAGATION_HPP

#include <memory>
#include <vector>

#include "dnnl.hpp"

#include "interface/c_types_map.hpp"

#include "backend/dnnl/passes/lower_down.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

impl::status_t layout_propagation(
        std::vector<std::shared_ptr<impl::op_t>> &subgraph,
        const dnnl::engine &p_engine, primitive_attr_mgr &prm_attr_mgr,
        pd_cache_t &pd_cache);

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
