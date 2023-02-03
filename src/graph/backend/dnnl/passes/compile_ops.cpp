/*******************************************************************************
 * Copyright 2021-2023 Intel Corporation
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

#include <memory>
#include <vector>
#include <unordered_map>

#include "graph/interface/c_types_map.hpp"
#include "graph/interface/value.hpp"

#include "graph/backend/dnnl/internal_attrs.hpp"
#include "graph/backend/dnnl/op_executable.hpp"
#include "graph/backend/dnnl/passes/compile_ops.hpp"

#include "oneapi/dnnl/dnnl.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {
using op_ptr = std::shared_ptr<op_t>;

/// After the lower down, infer shape, infer type and layout propagation passes,
/// each op in the subgraph will has complete attributes and each edge will have
/// complete shape/dtype/layout information. We can create executable for these
/// ops.
status_t compile_ops(std::shared_ptr<subgraph_t> &sg) {
    auto &mgr = sg->fusion_info_mgr_;
    const auto &p_engine = *(sg->p_engine_);
    auto &pd_cache = sg->pd_cache_;

    return topo_order_visit(sg->get_output_ops(), [&](op_t *op) {
        const op_schema_t *opm
                = op_schema_registry_t::get_op_schema(op->get_kind());
        if (!opm) {
            assertm(false, "no schema for current op");
            return status::invalid_graph_op;
        }

        if (!opm->has_additional_item("executable_creator")) {
            assertm(false, "no executable creator in this op schema");
            return status::invalid_graph_op;
        }

        auto cur_op = op->shared_from_this();
        auto creator = opm->get_additional_item<executable_creator_func>(
                "executable_creator");
        std::shared_ptr<op_executable_t> exec
                = creator(cur_op, p_engine, mgr, pd_cache);

        if (!exec) {
            assertm(false, "unimplemented op, can't compile it");
            return status::unimplemented;
        }

        sg->execs_.emplace_back(exec);
        sg->is_constant_.push_back(op->has_attr(op_attr::is_constant)
                && op->get_attr<bool>(op_attr::is_constant));
        return status::success;
    });
}

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
