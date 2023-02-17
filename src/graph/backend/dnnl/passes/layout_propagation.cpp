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
#include <string>
#include <vector>

#include "oneapi/dnnl/dnnl.hpp"

#include "graph/interface/c_types_map.hpp"
#include "graph/interface/value.hpp"

#include "graph/backend/dnnl/common.hpp"
#include "graph/backend/dnnl/layout_propagator.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {
using op_t = op_t;
using op_ptr = std::shared_ptr<op_t>;
using value_ptr = std::shared_ptr<value_t>;
using ltw = logical_tensor_wrapper_t;

bool need_prop_once_more(const std::shared_ptr<subgraph_t> &sg) {
    for (const auto &cur_op : sg->get_ops()) {
        for (const auto &in : cur_op->get_input_values()) {
            if (ltw(in->get_logical_tensor()).layout_type()
                    == layout_type::any) {
                return true;
            }
        }
        for (const auto &out : cur_op->get_output_values()) {
            if (ltw(out->get_logical_tensor()).layout_type()
                    == layout_type::any) {
                return true;
            }
        }
    }
    return false;
}

void force_partition_output_plain_layout(std::shared_ptr<subgraph_t> &sg) {
    const auto &p_engine = *(sg->p_engine_);
    auto &mgr = sg->fusion_info_mgr_;
    auto &pd_cache = sg->pd_cache_;

    subgraph_rewriter_t rewriter(sg);
    for (const auto &out_op : sg->get_output_ops()) {
        auto out_op_ptr = out_op->shared_from_this();
        const auto &out_vals = out_op_ptr->get_output_values();
        for (size_t i = 0; i < out_vals.size(); ++i) {
            const auto lt = out_vals[i]->get_logical_tensor();
            if (lt.id != std::numeric_limits<size_t>::max()
                    && lt.layout_type != layout_type::strided) {
                auto ori_mem_desc = make_dnnl_memory_desc(lt);
                if (!is_plain(ori_mem_desc)) {
                    auto expect_mem_desc = to_nxc_format(ori_mem_desc);
                    const auto strides = expect_mem_desc.get_strides();
                    out_vals[i]->set_strides(strides);
                    insert_reorder_after(out_op_ptr, i, ori_mem_desc, p_engine,
                            mgr, pd_cache, rewriter);
                }
            }
        }
    }

    rewriter.run();
}

/// This function is used to chooses optimal layout for computation bound op and
/// propagate the chosen optimal layout and given in/outputs layout in the
/// subgraph.
///
/// The workflow of layout propagation is:
///
/// Step1: propagate layout info according to the topological order
/// Step2: when comes to compute bound ops like Convolution/MatMul, it will
///     always use *any* format to create pd. And corresponding layout
///     propagation function will decide if insert a reorder based on comparison
///     result between input/output layout and queried optimal layout
/// Step3: the following internal ops (permute/squeeze) will also be responsible
///     for deciding if insert a reorder before the op.
/// Step4: at the most cases the layout propagation should be done only once
///
/// \note The layout propagation function for each op should be bidirectional to
/// support propagating layout both from inputs to outputs and from outputs to
/// inputs.
status_t layout_propagation(std::shared_ptr<subgraph_t> &sg) {
    const auto &p_engine = *(sg->p_engine_);
    auto &mgr = sg->fusion_info_mgr_;
    auto &pd_cache = sg->pd_cache_;

    status_t ret;
    std::unordered_set<op_t *> visited;
    do {
        subgraph_rewriter_t rewriter(sg);
        ret = topo_order_visit(sg->get_output_ops(), [&](op_t *op) {
            if (visited.count(op)) return status::success;

            const op_schema_t *opm
                    = op_schema_registry_t::get_op_schema(op->get_kind());
            if (!opm) {
                assertm(false, "no schema for current op");
                return status::invalid_graph_op;
            }

            if (!opm->has_additional_item("layout_propagator")) {
                assertm(false, "no layout propagator in this op schema");
                return status::invalid_graph_op;
            }

            auto cur_op = op->shared_from_this();
            auto propagator = opm->get_additional_item<layout_propagator_func>(
                    "layout_propagator");
            status_t status
                    = propagator(cur_op, p_engine, mgr, pd_cache, rewriter);

            visited.insert(op);
            return status;
        });

        if (ret != status::success) return ret;

        rewriter.run();
    } while (need_prop_once_more(sg));

    // Add check for the layout type of partition outputs to make partition
    // always output public layouts: abcd or acdb. If non-strided output, we
    // need insert a reorder to convert to public acdb layout. Currently,
    // deconvolution primitive still chooses blocked layout for best
    // performance.
    if (!mgr.get_use_blocked_layout()) force_partition_output_plain_layout(sg);

    // fill layout information for subgraph's inputs
    for (size_t i = 0; i < sg->ins_.size(); i++) {
        for (auto in_val : sg->get_input_values()) {
            auto lt = in_val->get_logical_tensor();
            if (lt.id == sg->ins_[i].id) {
                auto md = make_dnnl_memory_desc(lt);
                auto status = fill_layout_info(&(sg->ins_[i]), md);
                if (status != status::success) return status;
            }
        }
    }

    // fill layout information for subgraph's outputs
    for (size_t i = 0; i < sg->outs_.size(); i++) {
        for (auto out_val : sg->get_output_values()) {
            auto lt = out_val->get_logical_tensor();
            if (lt.id == sg->outs_[i].id) {
                auto md = make_dnnl_memory_desc(lt);
                auto status = fill_layout_info(&(sg->outs_[i]), md);
                if (status != status::success) return status;
            }
        }
    }

    return status::success;
}

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
