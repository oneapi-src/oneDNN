/*******************************************************************************
 * Copyright 2021-2022 Intel Corporation
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

#include <algorithm>
#include <chrono>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "graph/interface/shape_infer.hpp"
#include "graph/interface/value.hpp"
#include "graph/utils/debug.hpp"

#include "graph/backend/dnnl/common.hpp"
#include "graph/backend/dnnl/dnnl_backend.hpp"
#include "graph/backend/dnnl/internal_attrs.hpp"
#include "graph/backend/dnnl/passes/utils.hpp"
#include "graph/backend/dnnl/utils.hpp"

#include "oneapi/dnnl/dnnl.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {
using op_t = op_t;
using op_ptr = std::shared_ptr<op_t>;
using value_ptr = std::shared_ptr<value_t>;
using ltw = logical_tensor_wrapper_t;

// this function fuse a op to its successor.
// you should guarantee that the op has only one successor
//
//   in_val
//     |
//   next_op         in_val
//     |      --->     |
//   base_op         base_op
//     |               |
//   out_val         out_val
void fuse_op_to_successor(op_t *op, std::vector<op_ptr> &subgraph) {
    assertm(op->num_inputs() == 1, "this op should have only one input value.");
    value_ptr in_val = op->get_input_value(0);
    in_val->remove_consumer(*op, 0);

    value_ptr out_val = op->get_output_value(0);
    auto consumers = out_val->get_consumers();
    assertm(!consumers.empty() && consumers.size() == 1,
            "this op has zero consumer or more than one consumers.");

    op_t &successor = consumers[0].get_op();
    size_t offset = consumers[0].get_offset();
    in_val->add_consumer(successor, offset);
    successor.connect_input(offset, in_val);

    auto pos = std::find_if(subgraph.begin(), subgraph.end(),
            [op](const op_ptr &tmp) { return op == tmp.get(); });
    if (pos != subgraph.end()) subgraph.erase(pos);
}

//   in_val                  in_val     in_val2
//     |                         \       /
//   base_op  in_val2             base_op
//      \       /       --->         |
//       next_op                  out_val
//          |
//       out_val
void fuse_op_to_predecessor(
        op_t *op, std::vector<op_ptr> &subgraph, size_t in_offset) {
    value_ptr in_val = op->get_input_value(in_offset);
    value_ptr out_val = op->get_output_value(0);

    op_t &predecessor = in_val->get_producer();
    size_t offset = in_val->get_offset();
    predecessor.connect_output(offset, out_val);

    for (size_t i = 0; i < op->num_inputs(); i++) {
        value_ptr tmp = op->get_input_value(i);
        if (tmp == in_val) { continue; }

        tmp->remove_consumer(*op, i);
        tmp->add_consumer(predecessor, predecessor.num_inputs());
        predecessor.add_input(tmp);
    }

    auto pos = std::find_if(subgraph.begin(), subgraph.end(),
            [op](const op_ptr &tmp) { return op == tmp.get(); });
    if (pos != subgraph.end()) subgraph.erase(pos);
}

//   in_val          in_val
//     |               |
//     |     ->    inserted_op
//     |               |
//     |             new_val
//     |               |
//  base_op         base_op
void insert_op_before(op_ptr &inserted_op, op_ptr &base_op, size_t offset) {
    return insert_op_before(inserted_op.get(), base_op.get(), offset);
}

void insert_op_before(op_t *inserted_op, op_t *base_op, size_t offset) {
    value_ptr in_val = base_op->get_input_value(offset);
    in_val->remove_consumer(*base_op, offset);
    in_val->add_consumer(*inserted_op, inserted_op->num_inputs());
    inserted_op->add_input(in_val);

    logical_tensor_t new_lt = empty_logical_tensor_with_default_id();
    auto new_val = std::make_shared<value_t>(*inserted_op, 0, new_lt, true);
    auto in_dtype = in_val->get_logical_tensor().data_type;
    new_val->set_data_type(in_dtype);

    inserted_op->add_output(new_val);

    new_val->add_consumer(*base_op, offset);
    base_op->connect_input(offset, new_val);
}

void insert_op_before(op_t *inserted_op, op_t *base_op, size_t base_offset,
        size_t inserted_offset) {
    value_ptr in_val = base_op->get_input_value(base_offset);
    in_val->remove_consumer(*base_op, base_offset);
    inserted_op->connect_input(inserted_offset, in_val);

    logical_tensor_t new_lt = empty_logical_tensor_with_default_id();
    auto new_val = std::make_shared<value_t>(*inserted_op, 0, new_lt, true);
    auto in_dtype = in_val->get_logical_tensor().data_type;
    new_val->set_data_type(in_dtype);

    inserted_op->add_output(new_val);

    new_val->add_consumer(*base_op, base_offset);
    base_op->connect_input(base_offset, new_val);
}

//   base_op         base_op
//     |               |
//     |             new_val
//     |               |
//     |     ->    inserted_op
//     |               |
//  out_val         out_value
void insert_op_after(op_ptr &inserted_op, op_ptr &base_op, size_t offset) {
    return insert_op_after(inserted_op.get(), base_op.get(), offset);
}

void insert_op_after(op_t *inserted_op, op_t *base_op, size_t offset) {
    value_ptr out_val = base_op->get_output_value(offset);
    inserted_op->add_output(out_val);

    logical_tensor_t new_lt = empty_logical_tensor_with_default_id();
    auto new_val = std::make_shared<value_t>(*base_op, 0, new_lt, true);
    auto out_type = out_val->get_logical_tensor().data_type;
    new_val->set_data_type(out_type);

    base_op->connect_output(offset, new_val);

    new_val->add_consumer(*inserted_op, inserted_op->num_inputs());
    inserted_op->add_input(new_val);
}

void insert_op_after(op_t *inserted_op, op_t *base_op, size_t output_offset,
        size_t input_offset) {
    value_ptr out_val = base_op->get_output_value(output_offset);
    inserted_op->add_output(out_val);

    logical_tensor_t new_lt = empty_logical_tensor_with_default_id();
    auto new_val = std::make_shared<value_t>(*base_op, 0, new_lt, true);
    auto out_type = out_val->get_logical_tensor().data_type;
    new_val->set_data_type(out_type);

    base_op->connect_output(output_offset, new_val);

    new_val->add_consumer(*inserted_op, input_offset);
    inserted_op->connect_input(input_offset, new_val);
}

status_t set_given_inputs_outputs(std::shared_ptr<subgraph_t> &sg,
        const std::vector<logical_tensor_t> &inputs,
        const std::vector<logical_tensor_t> &outputs) {
    sg->ins_ = inputs;
    sg->outs_ = outputs;

    // set the inputs's layout to subgraph's inputs value
    auto graph_in_vals = sg->get_input_values();
    auto graph_out_vals = sg->get_output_values();

    auto func = [](std::vector<value_t *> &edges,
                        const std::vector<logical_tensor_t> &givens,
                        bool check_given, bool must_have_shape) {
        for (auto &edge : edges) {
            size_t edge_id = edge->get_logical_tensor().id;

            // partition in/outs should not have default id. There must be some
            // errors in previous graph transformation stage
            if (edge_id == std::numeric_limits<size_t>::max())
                return status::invalid_graph;

            bool found = false;
            for (const auto &given : givens) {
                if (edge_id == given.id) {
                    if (check_given) {
                        logical_tensor_wrapper_t given_ltw(given);
                        // check given lts
                        bool valid = !given_ltw.is_data_type_undef()
                                && !given_ltw.is_layout_type_undef();
                        if (must_have_shape) {
                            valid = valid && !given_ltw.is_empty();
                            // ndims=0 means the tensor is a scalar, we don't
                            // need to check its shape
                            if (given_ltw.ndims() > 0) {
                                for (auto dim : given_ltw.vdims()) {
                                    valid = valid && dim != -1;
                                }
                            }
                        }
                        if (!valid) return status::invalid_arguments;
                    }

                    edge->set_logical_tensor(given);
                    found = true;
                    break;
                }
            }

            if (!found) return status::invalid_arguments;
        }
        return status::success;
    };

    status_t ret;
    ret = func(graph_in_vals, inputs, true, true);
    if (ret != status::success) return ret;

    ret = func(graph_out_vals, outputs, true, false);
    return ret;
}

status_t set_given_inputs_outputs(std::vector<op_ptr> &subgraph,
        const std::vector<logical_tensor_t> &inputs,
        const std::vector<logical_tensor_t> &outputs) {
    auto sg = std::make_shared<subgraph_t>(subgraph);
    return set_given_inputs_outputs(sg, inputs, outputs);
}

void set_all_layout_to_any(std::vector<op_ptr> &subgraph) {
    for (auto &cur_op : subgraph) {
        for (const auto &val : cur_op->get_input_values()) {
            val->set_layout_type(layout_type::any);
        }

        for (const auto &val : cur_op->get_output_values()) {
            val->set_layout_type(layout_type::any);
        }
    }
}

// Constant property should be set by users from API level, this function is
// just a workaround at this moment.
void set_weight_bias_constant(std::vector<op_ptr> &subgraph) {
    for (auto &op : subgraph) {
        if (!(op->get_kind() == op_kind::dnnl_matmul
                    || op->get_kind() == op_kind::dnnl_convolution))
            continue;

        // set weight to be constant
        op->get_input_value(1)->set_property(property_type::constant);

        // set bias to be constant
        if (op->has_attr(op_attr::with_bias)
                && op->get_attr<bool>(op_attr::with_bias)) {
            op->get_input_value(2)->set_property(property_type::constant);
        }
    }
}

std::string kind2str(op_kind_t kind) {
    // 0: Abs, ..., N: LastSymbol, 0x1234: any, ...
    const size_t k = static_cast<size_t>(kind);
    const size_t l = static_cast<size_t>(graph::op_kind::LastSymbol);

    if (k <= l) {
        return op_t::kind2str(kind);
    } else {
        return dnnl_impl::op_kind::internal_op_strings.at(k
                - static_cast<size_t>(op_kind::kDNNL_INTERNAL_OP_STARTER) - 1);
    }
}

#ifdef DNNL_ENABLE_GRAPH_DUMP
namespace {
std::string layout2str(const dnnl::memory::desc &md) {
    std::string str;

    if (md.dims().empty()) return "";

    // format tag
    if (md.data.format_kind == dnnl_blocked) {
        std::string blk_tag;

        int ndims = md.data.ndims;
        auto &blk = md.data.format_desc.blocking;

        dnnl_dims_t blocks = {0};
        std::fill(blocks, blocks + ndims, 1);
        for (int iblk = 0; iblk < blk.inner_nblks; ++iblk)
            blocks[blk.inner_idxs[iblk]] *= blk.inner_blks[iblk];

        char dim_chars[DNNL_MAX_NDIMS + 1] = {'\0'};

        dims_t ou_blocks = {0};
        std::copy(md.data.padded_dims, md.data.padded_dims + ndims, ou_blocks);

        bool plain = true;
        for (int d = 0; d < ndims; ++d) {
            dim_chars[d] = static_cast<char>((blocks[d] == 1 ? 'a' : 'A') + d);
            if (blocks[d] != 1) plain = false;
            ou_blocks[d] /= blocks[d];
        }

        dnnl_dims_t strides = {0};
        std::copy(blk.strides, blk.strides + ndims, strides);

        utils::simultaneous_sort(strides, ou_blocks, dim_chars, ndims,
                [](dim_t a, dim_t b) { return b - a; });

        blk_tag = std::string(dim_chars);

        if (!plain) {
            for (int iblk = 0; iblk < blk.inner_nblks; ++iblk) {
                blk_tag += std::to_string(blk.inner_blks[iblk])
                        + static_cast<char>('a' + blk.inner_idxs[iblk]);
            }
        }

        str += blk_tag;
    } else if (md.data.format_kind == dnnl_format_kind_any) {
        str += "any";
    } else if (md.data.format_kind == dnnl_format_kind_undef) {
        str += "undef";
    }

    return str;
}

std::string property2str(property_type_t ptype) {
    std::string str;
    switch (ptype) {
        case property_type::undef: str = "undef"; break;
        case property_type::variable: str = "variable"; break;
        case property_type::constant: str = "constant"; break;
        default: break;
    }
    return str;
}
} // namespace
#endif

status_t subgraph_visualizer_t::run(const std::shared_ptr<subgraph_t> &sg,
        const std::string &name_suffix, bool is_layout_sensitive,
        bool is_memory_sensitive) {
#ifdef DNNL_ENABLE_GRAPH_DUMP
    if (!enabled_) return status::success;

    std::ofstream out;

    std::string backend_name = dnnl_backend::get_singleton().get_name();
    std::string partition_name = "partition_" + std::to_string(partition_id_);
    std::string index_str = std::to_string(index_++);
    const std::string &pass_name = name_suffix;

    // file_name: (backend_name)_partition_(id)_(index)_(pass_name).dot
    std::string file_name = backend_name + "_" + partition_name + "_"
            + index_str + "_" + pass_name + ".dot";
    std::cout << "visualize partition subgraph to a dot file: " << file_name
              << std::endl;

    // ID or address when ID is not available
    auto get_op_identifier = [](op_t *op) {
        if (op->get_id() != op_t::DEFAULT_ID) return op->get_id();
        return reinterpret_cast<size_t>(op);
    };

    out.open(file_name);
    out << "digraph G {\n";
    impl::status_t ret = topo_order_visit(sg->get_output_ops(), [&](op_t *op) {
        const auto &cur_op_name = kind2str(op->get_kind());
        const size_t cur_op_id = get_op_identifier(op);
        if (op->num_inputs() > 0) {
            for (size_t i = 0; i < op->num_inputs(); ++i) {
                auto input_value = op->get_input_value(i);
                if (input_value->has_producer()) {
                    op_t *input_op = &(input_value->get_producer());
                    const auto &input_op_name = kind2str(input_op->get_kind());
                    const size_t input_op_id = get_op_identifier(input_op);
                    out << "\"" << input_op_name << "_" << input_op_id
                        << "\" -> \"" << cur_op_name << "_" << cur_op_id
                        << "\";\n";
                }
            }
        } else {
            out << "\"" << cur_op_name << "_" << cur_op_id << "\"[label=\""
                << cur_op_name << "_" << cur_op_id << "\"];\n";
        }
        return status::success;
    });

    if (ret != impl::status::success) return ret;

    // value str: (data_type):(logical tensor id):(layout type):(dims):(layout
    // desc):(property):(mem_info)
    auto val2str = [this, is_layout_sensitive, is_memory_sensitive](
                           const value_t *val) {
        auto dims2str = [](const dims &dims) {
            if (dims.empty()) return std::string("");

            std::string str;
            str += std::to_string(dims[0]);
            for (size_t d = 1; d < dims.size(); ++d)
                str += ("x" + std::to_string(dims[d]));
            return str;
        };

        auto lt = val->get_logical_tensor();
        auto ltw = logical_tensor_wrapper_t(lt);
        std::string str
                = std::string(graph::utils::data_type2str(ltw.data_type()))
                + ":"
                + ((ltw.id() < std::numeric_limits<size_t>::max())
                                ? std::to_string(ltw.id())
                                : "def")
                + ":"
                + std::string(graph::utils::layout_type2str(ltw.layout_type()))
                + ":" + std::to_string(ltw.ndims()) + ":"
                + dims2str(ltw.ndims() < 0 ? std::vector<dim_t>() : ltw.vdims())
                + ":"
                + (is_layout_sensitive ? layout2str(make_dnnl_memory_desc(lt))
                                       : "")
                + ":" + property2str(ltw.property_type()) + ":"
                + (is_memory_sensitive ? this->mem_info_func_(val) : "");
        return str;
    };

    // dump inputs/outputs info
    // in(no)_(lt str) or out(no)_(lt str)
    ret = topo_order_visit(sg->get_output_ops(), [&](op_t *op) {
        const auto &op_name = kind2str(op->get_kind());
        const size_t op_id = get_op_identifier(op);
        out << "\"" << op_name << "_" << op_id << "\"[label=\"" << op_name
            << "_" << op_id;

        for (size_t i = 0; i < op->num_inputs(); i++) {
            out << "\\n"
                << "in" << std::to_string(i) << "_"
                << val2str(op->get_input_value(i).get());
        }

        for (size_t i = 0; i < op->num_outputs(); i++) {
            out << "\\n"
                << "out" << std::to_string(i) << "_"
                << val2str(op->get_output_value(i).get());
        }

        out << "\"];\n";
        return status::success;
    });

    if (ret != impl::status::success) return ret;

    out << "}\n";
    out.close();
#else
    UNUSED(sg);
    UNUSED(name_suffix);
    UNUSED(is_layout_sensitive);
    UNUSED(is_memory_sensitive);
#endif

    return status::success;
}

status_t subgraph_validator_t::run(const std::shared_ptr<subgraph_t> &sg) {
    auto ret = topo_order_visit(sg->get_output_ops(), [&](op_t *op) {
        // TODO(qun) Call each op's validator
        const op_schema_t *opm
                = op_schema_registry_t::get_op_schema(op->get_kind());
        if (!opm) { return status::invalid_graph_op; }

        // ops in this list need to be refined further, we should lower them to
        // internal ops or improve the definition
        const static std::set<op_kind_t> ops_need_refine = {
                // dnnl internal ops
                op_kind::squeeze,
                op_kind::expand,
                op_kind::to_group,
                op_kind::from_group,
                op_kind::permute,
                // frontend ops that need to be lower to dnnl internal ops.
                // but now, we reuse these frontend ops and set some new attrs
                // for them, which makes them unwell defined
                graph::op_kind::StaticReshape,
                graph::op_kind::StaticTranspose,
        };

        // Validate
        // TODO(xxx) Skip unwell defined ops for now. we need to validate
        // all ops after refactor done
        if (ops_need_refine.count(op->get_kind()) == 0) {
            if (!opm->verify(op, false)) {
                assertm(false, "schema verify failed");
                return status::invalid_graph_op;
            }

            // Not allow undefined attributes
            const auto &expected_attrs = opm->get_attrs();
            const auto &actual_attrs = op->get_attributes();
            for (const auto &elem : actual_attrs) {
                // The matched_pattern attr is added by pattern matcher, we skip
                // it. The with_sum attr will be removed later, we skip it.
                bool skip = elem.first == op_attr::matched
                        || elem.first == op_attr::with_sum;
                if (!skip && expected_attrs.count(elem.first) == 0) {
#ifndef NDEBUG
                    if (op_t::attr2str(elem.first).compare("undefined_attr")) {
                        DEBUG_PRINT_ERROR("common attribute "
                                + op_t::attr2str(elem.first) + " in op "
                                + op->get_name() + " is not defined");
                    } else {
                        DEBUG_PRINT_ERROR("internal attribute "
                                + op_attr::internal_attr2str(elem.first)
                                + " in op " + op->get_name()
                                + " is not defined");
                    }
#endif
                    return status::invalid_graph_op;
                }
            }
        }

        // Additional verifications
        if (op->get_kind() == op_kind::dnnl_convolution) {
            bool canonicalized = op->has_attr(op_attr::canonicalized)
                    && op->get_attr<bool>(op_attr::canonicalized);
            if (canonicalized) {
                auto data_fmt = op->get_attr<std::string>(op_attr::data_format);
                auto filter_fmt
                        = op->get_attr<std::string>(op_attr::filter_format);
                auto groups = op->get_attr<int64_t>(op_attr::groups);
                bool ok = data_fmt == "NCX" && filter_fmt == "OIX"
                        && groups == 1;
                if (!ok) {
                    DEBUG_PRINT_ERROR("data_format:" + data_fmt + ";"
                            + "filter_format:" + filter_fmt + ";"
                            + "groups:" + std::to_string(groups));
                    assertm(false, "additional verify failed");
                    return status::invalid_graph_op;
                }
            }
        } else {
            // TODO(qun)
        }

        // Check shape and type (each pass should be shape/type consistent in
        // static shape scenarios)
        const auto &in_vals = op->get_input_values();
        for (size_t i = 0; i < in_vals.size(); i++) {
            // dnnl_pool_bwd's index 1 and index 2 input are optional
            if (op->get_kind() == dnnl_impl::op_kind::dnnl_pool_bwd
                    && (i == 1 || i == 2))
                continue;

            auto lt = in_vals[i]->get_logical_tensor();
            logical_tensor_wrapper_t ltw(lt);
            if (ltw.is_shape_unknown()) { return status::invalid_shape; }
            if (ltw.data_type() == graph::data_type::undef) {
                return status::invalid_data_type;
            }
        }

        // only check the first output now
        const auto &out_val = op->get_output_value(0);
        auto lt = out_val->get_logical_tensor();
        logical_tensor_wrapper_t ltw(lt);

        if (ltw.is_shape_unknown()) { return status::invalid_shape; }
        if (ltw.data_type() == graph::data_type::undef) {
            return status::invalid_data_type;
        }

        return status::success;
    });
    return ret;
}

void replace_op(op_ptr &org_op, op_ptr &new_op) {
    for (size_t i = 0; i < org_op->num_inputs(); i++) {
        auto in_val = org_op->get_input_value(i);
        in_val->remove_consumer(*org_op, i);
        in_val->add_consumer(*new_op, new_op->num_inputs());
        new_op->add_input(in_val);
    }
    for (size_t i = 0; i < org_op->num_outputs(); i++) {
        auto out_val = org_op->get_output_value(i);
        new_op->add_output(out_val);
    }
}

void merge_common_eltwise_attrs(
        std::shared_ptr<op_t> &org_op, std::shared_ptr<op_t> &new_op) {
    if (org_op->has_attr(op_attr::alpha)) {
        new_op->set_attr<float>(
                op_attr::alpha, org_op->get_attr<float>(op_attr::alpha));
    } else if (org_op->has_attr(op_attr::min)) {
        new_op->set_attr<float>(
                op_attr::alpha, org_op->get_attr<float>(op_attr::min));
    } else {
        new_op->set_attr<float>(op_attr::alpha, 0);
    }

    if (org_op->has_attr(op_attr::beta)) {
        new_op->set_attr<float>(
                op_attr::beta, org_op->get_attr<float>(op_attr::beta));
    } else if (org_op->has_attr(op_attr::max)) {
        new_op->set_attr<float>(
                op_attr::beta, org_op->get_attr<float>(op_attr::max));
    } else {
        new_op->set_attr<float>(op_attr::beta, 0);
    }
}

std::vector<value_t *> get_constant_block_output_values(
        const std::vector<op_ptr> &subgraph) {
    using ltw = logical_tensor_wrapper_t;
    std::vector<value_t *> ret;
    auto func = [&](op_t *op) {
        auto out_vals = op->get_output_values();
        for (auto &val : out_vals) {
            if (!ltw(val->get_logical_tensor()).is_constant()) continue;
            // if a constant value feed into a consumer whose output is not
            // constant, then the value is the final output of a constant block
            auto consumers = val->get_consumers();
            for (auto &csm : consumers) {
                // A consumer is not constant
                if (!csm.get_op().get_attr<bool>(op_attr::is_constant)) {
                    ret.emplace_back(val.get());
                    break;
                }
            }
        }
        return status::success;
    };
    impl::status_t status
            = topo_order_visit(graph_t(subgraph).get_output_ops(), func);
    if (status != impl::status::success) return {};
    return ret;
}

status_t infer_shape(std::shared_ptr<subgraph_t> &sg) {
    auto ret = sg->infer_shape();
    if (ret != status::success) return ret;

    // Fill the inferred shape and strides to subgraph's outputs
    for (size_t i = 0; i < sg->outs_.size(); i++) {
        for (auto val : sg->get_output_values()) {
            auto lt = val->get_logical_tensor();
            if (lt.id == sg->outs_[i].id) {
                auto inferred_shape = ltw(lt).vdims();
                set_shape_and_strides(sg->outs_[i], inferred_shape);
            }
        }
    }

    return ret;
}

subgraph_t::subgraph_t(const std::vector<op_ptr> &ops, const dnnl::engine &eng,
        bool reset_layout)
    : graph_t(ops), p_engine_(&eng) {
    if (reset_layout) { set_all_layout_to_any(get_mutable_ops()); }
}

subgraph_t::subgraph_t(const std::vector<op_ptr> &ops, const dnnl::engine &eng,
        fpmath_mode_t fpm_mode, bool reset_layout)
    : graph_t(ops, static_cast<engine_kind_t>(eng.get_kind()), fpm_mode)
    , p_engine_(&eng)
    , fusion_info_mgr_(fpm_mode) {
    if (reset_layout) { set_all_layout_to_any(get_mutable_ops()); }
}

subgraph_t::subgraph_t(const std::vector<op_ptr> &ops, bool reset_layout)
    : graph_t(ops), p_engine_(nullptr) {
    if (reset_layout) { set_all_layout_to_any(get_mutable_ops()); }
}

const std::map<op_kind_t, dnnl::algorithm> &get_binary_alg_map() {
    static const std::map<op_kind_t, dnnl::algorithm> &binary_alg_map
            = {{graph::op_kind::Add, dnnl::algorithm::binary_add},
                    {graph::op_kind::Multiply, dnnl::algorithm::binary_mul},
                    {graph::op_kind::Divide, dnnl::algorithm::binary_div},
                    {graph::op_kind::Minimum, dnnl::algorithm::binary_min},
                    {graph::op_kind::Maximum, dnnl::algorithm::binary_max},
                    {graph::op_kind::Subtract, dnnl::algorithm::binary_sub},
                    {graph::op_kind::BiasAdd, dnnl::algorithm::binary_add}};
    return binary_alg_map;
}

bool binary_doable(
        const std::vector<dim_t> &shape_0, const std::vector<dim_t> &shape_1) {
    const int ndims_0 = static_cast<int>(shape_0.size());
    const int ndims_1 = static_cast<int>(shape_1.size());
    const int small = ndims_0 < ndims_1 ? ndims_0 : ndims_1;
    for (int i = 1; i <= small; ++i) {
        bool match = shape_0[ndims_0 - i] == shape_1[ndims_1 - i]
                || shape_0[ndims_0 - i] == 1 || shape_1[ndims_1 - i] == 1;
        if (!match) return false;
    }
    return true;
}

static bool post_binary_fusible_impl(const op_t *base_op,
        const std::vector<dim_t> &fused_shape,
        const std::vector<dim_t> &other_shape) {
    assertm(fused_shape.size() == other_shape.size(),
            "must have same ndims, pls run binary_canonicalization pass first");
    // full tensor and per tensor broadcasted
    if (fused_shape == other_shape
            || std::all_of(other_shape.begin(), other_shape.end(),
                    [](dim_t i) { return i == 1; }))
        return true;

    // any broadcasted for 4d tensor MatMul
    int32_t output_ndims = static_cast<int32_t>(fused_shape.size());
    if (base_op->get_kind() == op_kind::dnnl_matmul && output_ndims == 4) {
        for (int32_t i = output_ndims - 1; i >= 0; i--) {
            if (other_shape[i] == 1) continue;
            if (fused_shape[i] != other_shape[i]) { return false; }
        }
        return true;
    }

    // per channel broadcasted
    const auto is_not_one = [](dim_t d) { return d != 1; };
    const auto n_not_broadcastable
            = std::count_if(other_shape.begin(), other_shape.end(), is_not_one);
    if (n_not_broadcastable != 1) return false;
    const auto c_axis_it
            = std::find_if(other_shape.begin(), other_shape.end(), is_not_one);
    const auto c_axis = static_cast<size_t>(
            std::distance(other_shape.begin(), c_axis_it));
    if (other_shape[c_axis] != fused_shape[c_axis]) return false;
    if (base_op->has_attr(op_attr::data_format)) {
        const auto data_fmt
                = base_op->get_attr<std::string>(op_attr::data_format);
        int32_t orig_c_axis = data_fmt == "NCX" ? 1 : output_ndims - 1;
        return c_axis == static_cast<size_t>(orig_c_axis);
    }

    return true;
}

std::pair<bool, std::pair<size_t, int64_t>> shuffle_fusible(
        const op_t *reshape0, op_t *reshape1, op_t *transpose) {
    using result_t = std::pair<bool, std::pair<size_t, int64_t>>;
    const result_t dflt_res {false, {0, 0}};

    const logical_tensor_t src_port
            = reshape0->get_input_value(0)->get_logical_tensor();
    const logical_tensor_t dst_port
            = reshape1->get_output_value(0)->get_logical_tensor();
    const auto src_lt_shape = ltw(src_port).vdims();
    const auto dst_lt_shape = ltw(dst_port).vdims();
    const auto attr_shape = reshape0->get_attr<dims>(op_attr::shape);
    const auto tp_order = transpose->get_attr<dims>(op_attr::order);

    if (src_lt_shape != dst_lt_shape) return dflt_res;
    if (src_lt_shape.size() + 1 != attr_shape.size()) return dflt_res;

    size_t last_unmatched_pos = tp_order.size();
    size_t matched_pos = 0;
    for (size_t i = 0; i < tp_order.size(); ++i) {
        if (tp_order[i] == static_cast<dim>(i))
            ++matched_pos;
        else
            last_unmatched_pos = i;
    }

    // more or less than two positions were swapped
    if (matched_pos != tp_order.size() - 2) return dflt_res;
    // all positions were matched
    if (last_unmatched_pos == tp_order.size()) return dflt_res;
    // transposition not on consecutive positions
    if (last_unmatched_pos
            != static_cast<size_t>(tp_order[last_unmatched_pos - 1]))
        return dflt_res;

    const size_t g_pos = last_unmatched_pos;
    const size_t c_over_g_pos = g_pos - 1;
    const int64_t groups = attr_shape[g_pos];
    auto mod_attr_shape = attr_shape;
    mod_attr_shape[c_over_g_pos] *= groups;
    mod_attr_shape.erase(mod_attr_shape.begin() + g_pos);

    if (src_lt_shape != mod_attr_shape) return dflt_res;

    return {true, {c_over_g_pos, groups}};
}

bool post_binary_fusible(const op_t *base_op, const op_t *bin_op) {
    auto fused_out = base_op->get_output_values()[0];
    auto consumers = fused_out->get_consumers();
    if (consumers.size() != 1) return false;

    size_t fused_in_off = consumers[0].get_offset();
    auto fused_in = bin_op->get_input_value(fused_in_off)->get_logical_tensor();
    auto other_in
            = bin_op->get_input_value(1 - fused_in_off)->get_logical_tensor();
    return post_binary_fusible_impl(
            base_op, ltw(fused_in).vdims(), ltw(other_in).vdims());
}

bool post_depthwise_conv_fusible(
        const op_t *base_conv_op, const op_t *post_conv_op) {
    using spatial_dims_t = std::vector<int64_t>;
    using oix_dims_t = std::tuple<int64_t, int64_t, spatial_dims_t>;
    const auto extract_dims_as_oix = [](const op_t *op) -> oix_dims_t {
        const size_t wei_offset = 1;
        const auto wei_dims
                = ltw(op->get_input_value(wei_offset)->get_logical_tensor())
                          .vdims();
        const auto wei_format = (op->has_attr(op_attr::filter_format))
                ? op->get_attr<std::string>(op_attr::filter_format)
                : "XIO";
        const size_t ndims = wei_dims.size();
        const int64_t o
                = (wei_format == "OIX") ? wei_dims[0] : wei_dims[ndims - 1];
        const int64_t i
                = (wei_format == "OIX") ? wei_dims[1] : wei_dims[ndims - 2];
        const auto spatial_dims = (wei_format == "OIX")
                ? spatial_dims_t(wei_dims.begin() + 2, wei_dims.end())
                : spatial_dims_t(wei_dims.begin(), wei_dims.end() - 2);

        return std::make_tuple(o, i, spatial_dims);
    };
    const auto all_equal_to = [](const dims &ds, const int64_t val) -> bool {
        return std::all_of(ds.begin(), ds.end(),
                [val](const int64_t d) { return d == val; });
    };

    spatial_dims_t conv_spatial;
    std::tie(std::ignore, std::ignore, conv_spatial)
            = extract_dims_as_oix(base_conv_op);

    int64_t dw_o = 0;
    int64_t dw_i = 0;
    spatial_dims_t dw_spatial;
    std::tie(dw_o, dw_i, dw_spatial) = extract_dims_as_oix(post_conv_op);

    // only 2D conv is supported
    const size_t expected_spatial_ndims = 2;
    if (conv_spatial.size() != expected_spatial_ndims
            || dw_spatial.size() != expected_spatial_ndims)
        return false;

    // base conv has to be 1x1 conv
    if (!all_equal_to(conv_spatial, 1)) return false;

    // post conv has to be 3x3 conv
    if (!all_equal_to(dw_spatial, 3)) return false;

    // other post conv requirements
    if (post_conv_op->has_attr(op_attr::auto_pad)
            && post_conv_op->get_attr<std::string>(op_attr::auto_pad) != "None")
        return false;
    if (!post_conv_op->has_attr(op_attr::groups)) return false;

    const auto groups = post_conv_op->get_attr<int64_t>(op_attr::groups);
    if (!(groups == dw_o && dw_o == groups * dw_i)) return false;

    const auto strides = post_conv_op->get_attr<dims>(op_attr::strides);
    if (!(all_equal_to(strides, 1) || all_equal_to(strides, 2))) return false;

    const auto pads_begin = post_conv_op->get_attr<dims>(op_attr::pads_begin);
    if (!all_equal_to(pads_begin, 1)) return false;

    const auto pads_end = post_conv_op->get_attr<dims>(op_attr::pads_end);
    if (!(all_equal_to(pads_end, 0) || all_equal_to(pads_end, 1))) return false;

    return true;
}

const std::unordered_map<op_kind_t, std::unordered_set<op_kind_t>> &
get_post_ops_fusible_map() {
    using namespace graph::op_kind;
    using namespace dnnl_impl::op_kind;
    static const std::unordered_map<op_kind_t, std::unordered_set<op_kind_t>>
            fusible_map = {
                    {dnnl_convolution,
                            {dnnl_eltwise, dnnl_binary, dnnl_convolution}},
                    {dnnl_convtranspose, {dnnl_eltwise, dnnl_binary}},
                    {dnnl_matmul, {dnnl_eltwise, dnnl_binary}},
                    {dnnl_pool, {dnnl_binary}},
                    {dnnl_eltwise, {dnnl_binary}},
                    {dnnl_binary, {dnnl_eltwise, dnnl_binary}},
                    // bn
                    {dnnl_batchnorm, {dnnl_eltwise}},
                    // reduction
                    {dnnl_reduction, {dnnl_eltwise, dnnl_binary}},
                    // resample
                    {dnnl_resampling, {dnnl_eltwise, dnnl_binary}},
                    {dnnl_reorder, {dnnl_binary}},
            };
    return fusible_map;
}

// data_format = NXC:
// (1, 2, 3, 4); (4) is doable
// data_format = NCX, channel broadcast = false:
// (1, 2, 3, 4); (4) is doable
// data_format = NCX, channel broadcast = true:
// (1, 2, 3, 4); (2) is doable

// src      wei
// (3, 4); (3, 4) is doable
// (1, 4); (3, 4) is not doable
// (3, 4); (1, 4) is doable
// (3, 4, 5); (4, 5) is doable
// (3, 4, 5); (1, 5) is doable
// (3, 4, 5); (2, 4, 5) is NOT doable
bool prelu_doable(const std::vector<dim_t> &src_dims,
        const std::vector<dim_t> &wei_dims, const std::string &data_format,
        const bool per_channel_broadcast) {
    const int src_ndims = static_cast<int>(src_dims.size());
    const int wei_ndims = static_cast<int>(wei_dims.size());
    // src ndims should be equal or greater than wei ndims
    if (src_ndims < wei_ndims) return false;

    bool doable = false;
    if (wei_ndims == 1) {
        if (!per_channel_broadcast || src_ndims == wei_ndims) {
            // if no broadcast to channel or src_ndims == 1
            // then wei dim should be equal to last src dim,
            // or equal to 1.
            doable = src_dims[src_ndims - 1] == wei_dims[0] || wei_dims[0] == 1;
        } else {
            // if broadcast to channel,
            // then src channel dim should be equal to wei dim
            const int channel_dim_num
                    = data_format == "NCX" ? 1 : src_ndims - 1;
            doable = src_dims[channel_dim_num] == wei_dims[0];
        }
    } else {
        for (int i = 1; i <= wei_ndims; ++i) {
            // Weights are broadcastable to src when:
            // 1) they are equal on the same ndims,
            // 2) one of them is 1,
            // 3) In the case when weights have fewer dimensions,
            //    1s are added to the front and then 1) and 2) must be met.
            doable = src_dims[src_ndims - i] == wei_dims[wei_ndims - i]
                    || wei_dims[wei_ndims - i] == 1;
            if (!doable) break;
        }
    }
    return doable;
}

value_ptr insert_empty_scratchpad(op_ptr &op) {
    logical_tensor_t lt = empty_logical_tensor_with_default_id();
    value_ptr scratchpad_val
            = std::make_shared<value_t>(*op, op->num_outputs(), lt);
    op->add_output(scratchpad_val);
    scratchpad_val->set_data_type(graph::data_type::u8);
    return scratchpad_val;
}

bool is_typecast(const op_t *op) {
    bool is_typecast = op->get_kind() == dnnl_impl::op_kind::dnnl_reorder
            && !op->get_attr<bool>(op_attr::change_layout)
            && (!op->has_attr(op_attr::qtype)
                    || op->get_attr<std::string>(op_attr::qtype)
                            == "per_tensor")
            && (!op->has_attr(op_attr::axis)
                    || op->get_attr<int64_t>(op_attr::axis) == -1)
            && !op->has_attr(op_attr::scales) && !op->has_attr(op_attr::src_zps)
            && !op->has_attr(op_attr::dst_zps)
            && (!op->has_attr(op_attr::with_runtime_scales)
                    || !op->get_attr<bool>(op_attr::with_runtime_scales))
            && (!op->has_attr(op_attr::with_runtime_src_zps)
                    || !op->get_attr<bool>(op_attr::with_runtime_src_zps))
            && (!op->has_attr(op_attr::with_runtime_dst_zps)
                    || !op->get_attr<bool>(op_attr::with_runtime_dst_zps))
            && op->get_input_value(0)->get_logical_tensor().data_type
                    != op->get_output_value(0)->get_logical_tensor().data_type;
    return is_typecast;
}

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
