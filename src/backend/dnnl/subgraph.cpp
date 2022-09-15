/*******************************************************************************
 * Copyright 2022 Intel Corporation
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

#include "interface/shape_infer.hpp"
#include "interface/value.hpp"
#include "utils/debug.hpp"

#include "backend/dnnl/common.hpp"
#include "backend/dnnl/dnnl_backend.hpp"
#include "backend/dnnl/internal_attrs.hpp"
#include "backend/dnnl/subgraph.hpp"
#include "backend/dnnl/utils.hpp"

#include "dnnl.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {
using op_t = impl::op_t;
using op_ptr = std::shared_ptr<impl::op_t>;
using value_ptr = std::shared_ptr<impl::value_t>;
using ltw = impl::logical_tensor_wrapper_t;

subgraph_t::subgraph_t(const std::vector<op_ptr> &ops, const dnnl::engine &eng,
        bool reset_layout)
    : impl::graph_t(ops), p_engine_(&eng) {
    if (reset_layout) { set_all_layout_to_any(get_mutable_ops()); }
}

subgraph_t::subgraph_t(const std::vector<op_ptr> &ops, const dnnl::engine &eng,
        impl::fpmath_mode_t fpm_mode, bool reset_layout)
    : impl::graph_t(
            ops, static_cast<impl::engine_kind_t>(eng.get_kind()), fpm_mode)
    , p_engine_(&eng)
    , fusion_info_mgr_(fpm_mode) {
    if (reset_layout) { set_all_layout_to_any(get_mutable_ops()); }
}

subgraph_t::subgraph_t(const std::vector<op_ptr> &ops, bool reset_layout)
    : impl::graph_t(ops), p_engine_(nullptr) {
    if (reset_layout) { set_all_layout_to_any(get_mutable_ops()); }
}

std::string kind2str(op_kind_t kind) {
    // 0: Abs, ..., N: LastSymbol, 0x1234: any, ...
    const size_t k = static_cast<size_t>(kind);
    const size_t l
            = static_cast<size_t>(dnnl::graph::impl::op_kind::LastSymbol);

    if (k <= l) {
        return op_t::kind2str(kind);
    } else {
        return impl::dnnl_impl::op_kind::internal_op_strings.at(k
                - static_cast<size_t>(op_kind::kDNNL_INTERNAL_OP_STARTER) - 1);
    }
}

#ifdef DNNL_GRAPH_ENABLE_DUMP
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

std::string property2str(impl::property_type_t ptype) {
    std::string str;
    switch (ptype) {
        case impl::property_type::undef: str = "undef"; break;
        case impl::property_type::variable: str = "variable"; break;
        case impl::property_type::constant: str = "constant"; break;
        default: break;
    }
    return str;
}
} // namespace
#endif

status_t subgraph_visualizer_t::run(const std::shared_ptr<subgraph_t> &sg,
        const std::string &name_suffix, bool is_layout_sensitive,
        bool is_memory_sensitive) {
#ifdef DNNL_GRAPH_ENABLE_DUMP
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
        auto dims2str = [](const impl::dims &dims) {
            if (dims.empty()) return std::string("");

            std::string str;
            str += std::to_string(dims[0]);
            for (int d = 1; d < dims.size(); ++d)
                str += ("x" + std::to_string(dims[d]));
            return str;
        };

        auto lt = val->get_logical_tensor();
        auto ltw = impl::logical_tensor_wrapper_t(lt);
        std::string str
                = std::string(impl::utils::data_type2str(ltw.data_type())) + ":"
                + ((ltw.id() < std::numeric_limits<size_t>::max())
                                ? std::to_string(ltw.id())
                                : "def")
                + ":"
                + std::string(impl::utils::layout_type2str(ltw.layout_type()))
                + ":" + std::to_string(ltw.ndims()) + ":"
                + dims2str(ltw.ndims() < 0 ? std::vector<impl::dim_t>()
                                           : ltw.vdims())
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
        const impl::op_schema_t *opm
                = impl::op_schema_registry_t::get_op_schema(op->get_kind());
        if (!opm) { return impl::status::invalid_op; }

        // Validate
        if (!opm->verify(op, false)) {
            assertm(false, "schema verify failed");
            return impl::status::invalid_op;
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
                if (impl::op_t::attr2str(elem.first)
                                .compare("undefined_attr")) {
                    DEBUG_PRINT_ERROR("common attribute "
                            + impl::op_t::attr2str(elem.first) + " in op "
                            + op->get_name() + " is not defined");
                } else {
                    DEBUG_PRINT_ERROR("internal attribute "
                            + op_attr::internal_attr2str(elem.first) + " in op "
                            + op->get_name() + " is not defined");
                }
#endif
                return impl::status::invalid_op;
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
                    return status::invalid_op;
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
            impl::logical_tensor_wrapper_t ltw(lt);
            if (ltw.is_shape_unknown()) { return status::invalid_shape; }
            if (ltw.data_type() == impl::data_type::undef) {
                return status::invalid_data_type;
            }
        }

        // only check the first output now
        const auto &out_val = op->get_output_value(0);
        auto lt = out_val->get_logical_tensor();
        impl::logical_tensor_wrapper_t ltw(lt);

        if (ltw.is_shape_unknown()) { return status::invalid_shape; }
        if (ltw.data_type() == impl::data_type::undef) {
            return status::invalid_data_type;
        }

        return status::success;
    });
    return ret;
}

void subgraph_rewriter_t::run() {
    std::vector<op_ptr> &mutable_ops = subgraph_->get_mutable_ops();

    // first remove and then insert to minimize the memory re-allocation
    for (const auto &op : to_be_removed_ops_) {
        auto pos = std::find_if(mutable_ops.begin(), mutable_ops.end(),
                [op](const op_ptr &tmp) { return op.get() == tmp.get(); });
        if (pos != mutable_ops.end()) mutable_ops.erase(pos);
    }

    for (const auto &op : to_be_inserted_ops_) {
        mutable_ops.emplace_back(op);
    }

    to_be_removed_ops_.clear();
    to_be_inserted_ops_.clear();
}

subgraph_rewriter_t::~subgraph_rewriter_t() {
    run();
}

void subgraph_rewriter_t::fuse_op_to_successor(const op_ptr &op) {
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

    to_remove(op);
}

void subgraph_rewriter_t::fuse_op_to_predecessor(const op_ptr &op, size_t i) {
    value_ptr in_val = op->get_input_value(i);
    value_ptr out_val = op->get_output_value(0);

    op_t &predecessor = in_val->get_producer();
    size_t offset = in_val->get_offset();
    predecessor.connect_output(offset, out_val);

    for (size_t iter = 0; iter < op->num_inputs(); iter++) {
        value_ptr tmp = op->get_input_value(iter);
        if (tmp == in_val) { continue; }

        tmp->remove_consumer(*op, iter);
        tmp->add_consumer(predecessor, predecessor.num_inputs());
        predecessor.add_input(tmp);
    }

    to_remove(op);
}

void subgraph_rewriter_t::insert_op_before(const op_ptr &inserted_op,
        const op_ptr &base_op, size_t i, size_t j, size_t k) {
    // make sure the base_op is not to be removed
    if (is_to_be_removed(base_op)) {
        assertm(false, "the base op is to be removed");
        return;
    }

    value_ptr in_val = base_op->get_input_value(i);
    in_val->remove_consumer(*base_op, i);
    if (j == std::numeric_limits<size_t>::max()) {
        j = inserted_op->num_inputs();
    }
    inserted_op->connect_input(j, in_val);

    impl::logical_tensor_t new_lt
            = impl::empty_logical_tensor_with_default_id();
    auto new_val = std::make_shared<value_t>(*inserted_op, 0, new_lt, true);
    auto in_dtype = in_val->get_logical_tensor().data_type;
    new_val->set_data_type(in_dtype);

    if (k == std::numeric_limits<size_t>::max()) {
        k = inserted_op->num_outputs();
    }
    inserted_op->connect_output(k, new_val);

    new_val->add_consumer(*base_op, i);
    base_op->connect_input(i, new_val);

    to_insert(inserted_op);
}

void subgraph_rewriter_t::insert_op_after(const op_ptr &inserted_op,
        const op_ptr &base_op, size_t i, size_t j, size_t k) {
    // make sure the base_op is not to be removed
    if (is_to_be_removed(base_op)) {
        assertm(false, "the base op is to be removed");
        return;
    }

    value_ptr out_val = base_op->get_output_value(i);
    if (k == std::numeric_limits<size_t>::max()) {
        k = inserted_op->num_outputs();
    }
    inserted_op->connect_output(k, out_val);

    impl::logical_tensor_t new_lt
            = impl::empty_logical_tensor_with_default_id();
    auto new_val = std::make_shared<value_t>(*base_op, 0, new_lt, true);
    auto out_type = out_val->get_logical_tensor().data_type;
    new_val->set_data_type(out_type);

    base_op->connect_output(i, new_val);

    if (j == std::numeric_limits<size_t>::max()) {
        j = inserted_op->num_inputs();
    }
    new_val->add_consumer(*inserted_op, j);
    inserted_op->connect_input(j, new_val);

    to_insert(inserted_op);
}

void subgraph_rewriter_t::replace_op(
        const op_ptr &org_op, const op_ptr &new_op) {
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

    to_insert(new_op);
    to_remove(org_op);
}

bool subgraph_rewriter_t::is_to_be_removed(
        const std::shared_ptr<impl::op_t> &op) const {
    auto pos
            = std::find_if(to_be_removed_ops_.begin(), to_be_removed_ops_.end(),
                    [&](const op_ptr &tmp) { return op.get() == tmp.get(); });
    return pos != to_be_removed_ops_.end();
}

void subgraph_rewriter_t::swap_neighboring_si_ops(
        const std::shared_ptr<impl::op_t> &producer,
        const std::shared_ptr<impl::op_t> &consumer) {
    assertm(producer->num_inputs() == 1,
            "only support swap single input operators.");
    assertm(consumer->num_inputs() == 1,
            "only support swap single input operators.");
    assertm(consumer->get_input_value(0)->has_producer()
                    && consumer->get_input_value(0)->get_offset() == 0
                    && consumer->get_input_op(0) == producer.get(),
            "only support swap neighboring operators.");
    auto producer_src = producer->get_input_value(0);
    auto producer_dst = producer->get_output_value(0);

    producer_src->remove_consumer(*producer, 0);
    consumer->connect_input(0, producer_src);

    auto consumer_dst = consumer->get_output_value(0);
    producer->connect_output(0, consumer_dst);

    impl::logical_tensor_t new_lt
            = impl::empty_logical_tensor_with_default_id();
    auto new_val = std::make_shared<value_t>(*consumer, 0, new_lt, true);
    new_val->set_data_type(consumer_dst->get_logical_tensor().data_type);
    consumer->connect_output(0, new_val);
    producer->connect_input(0, new_val);
}

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl
