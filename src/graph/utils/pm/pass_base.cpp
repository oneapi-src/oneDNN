/*******************************************************************************
* Copyright 2023-2025 Intel Corporation
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

#include "graph/utils/pm/pass_base.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace pass {

template <>
pass_base_t &pass_base_t::set_attr<FCreatePattern>(
        const std::string &attr_name, // NOLINT(*)
        const FCreatePattern &func) {
    Pattern pgraph = std::make_shared<pb_graph_t>();
    // create pattern graph by calling FCreatePattern func
    func(pgraph);
    return this->set_attr<Pattern>("Pattern", pgraph);
}

} // namespace pass
} // namespace graph
} // namespace impl
} // namespace dnnl
