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

#include "interface/c_types_map.hpp"

#include "backend/dnnl/subgraph/passes.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {
using op_t = impl::op_t;
using ltw = impl::logical_tensor_wrapper;

/// This function is used to infer dtype for the internal edges of a subgraph.
/// The dtype of entire subgraph's in/outputs should be given before infer type.
/// The workflow of infer shape pass is:
///     Step1: check if the entire subgraph's all in/outputs have valid dtype.
///     Step2: visit each op in topological order, infer each op's inputs or
///     outputs type
/// \note
/// The infer type for each op should be bidirectional to support both inferring
/// outputs dtype according to inputs and inferring inputs dtype according to
/// outputs. Because inferring type for some ops is impossible. We have to skip
/// these ops and their outputs dtype should be determined by the consumers.
/// Take the following transformed INT8_Conv pattern as example:
///     (u8) \  / (s8)
///        dnnl_conv
///            | (unknown)
///         convert
///            | (u8)
/// Users specify the pattern's inputs to be u8/s8 and the outputs to be u8;
/// According to the u8/s8 inputs, we can't deduce dnnl_conv's output dtype; We
/// have to deduce it according to convert op's output dtype.
impl::status_t infer_type(impl::graph_t &subgraph) {
    // Check inputs dtype
    for (impl::value_t *in : subgraph.get_input_values()) {
        impl::logical_tensor_t lt = in->get_logical_tensor();
        if (ltw(lt).data_type() == impl::data_type::undef)
            return impl::status::invalid_type;
    }

    // Check outputs dtype
    for (impl::value_t *out : subgraph.get_output_values()) {
        impl::logical_tensor_t lt = out->get_logical_tensor();
        if (ltw(lt).data_type() == impl::data_type::undef)
            return impl::status::invalid_type;
    }

    impl::topo_order_visit(subgraph.get_output_ops(), [](impl::op_t *op) {
        if (op->get_kind() == op_kind::mul_scales) {
            op->get_output_value(0)->set_data_type(impl::data_type::f32);
        } else if (op->get_kind() == op_kind::add_zps) {
            //This op should be fused, can't infer type for it
            return impl::status::invalid_graph;
        } else if (op->get_kind() == op_kind::permute
                || op->get_kind() == op_kind::Reorder
                || op->get_kind() == op_kind::to_group
                || op->get_kind() == op_kind::expand) {
            auto in_lt = op->get_input_value(0)->get_logical_tensor();
            auto out_lt = op->get_output_value(0)->get_logical_tensor();
            if (out_lt.data_type == impl::data_type::undef) {
                op->get_output_value(0)->set_data_type(in_lt.data_type);
            } else if (in_lt.data_type == impl::data_type::undef) {
                op->get_input_value(0)->set_data_type(out_lt.data_type);
            }
        } else {
        }
        return impl::status::success;
    });
    return impl::status::success;
}

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl
