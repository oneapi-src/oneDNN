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
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "interface/c_types_map.hpp"

#include "insert_ops.hpp"
#include "utils.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {
using op_t = impl::op_t;
using op_ptr = std::shared_ptr<impl::op_t>;

// TODO(xxx): extend to support other ops
static bool need_insert_reorder(op_kind_t kind) {
    std::set<op_kind_t> ops {op_kind::dnnl_convolution, op_kind::Convolution,
            op_kind::MatMul, op_kind::MaxPool};
    return ops.count(kind) != 0;
}

// TODO(xxx): extend to support other ops
// for those ops with data_format/filter_format attributes
static bool need_insert_permute(op_kind_t kind) {
    std::set<op_kind_t> ops {
            op_kind::dnnl_convolution, op_kind::Convolution, op_kind::MaxPool};
    return ops.count(kind) != 0;
}

// TODO(xxx): extend to support other ops
// for those ops whose input's format must be defined, such as pool, eltwise,...
static bool require_input_format(op_kind_t kind) {
    std::set<op_kind_t> ops {op_kind::MaxPool};
    return ops.count(kind) != 0;
}

void insert_reorder(std::vector<op_ptr> &subgraph) {
    std::vector<op_ptr> to_be_inserted_ops;
    for (auto &cur_op : subgraph) {
        if (!need_insert_reorder(cur_op->get_kind())) continue;

        bool with_bias = cur_op->has_attr("with_bias")
                ? cur_op->get_attr<bool>("with_bias")
                : false;
        size_t in_bound = with_bias ? 3 : 2;

        for (size_t i = 0; i < cur_op->num_outputs(); i++) {
            op_ptr reorder_op = std::make_shared<impl::op_t>(op_kind::convert);
            insert_op_after(reorder_op, cur_op, i);
            to_be_inserted_ops.emplace_back(reorder_op);
        }

        // for those primitive whose input's format must be defined (not any),
        // we don't need to insert reorder
        if (require_input_format(cur_op->get_kind())) continue;

        for (size_t i = 0; i < cur_op->num_inputs(); i++) {
            if (i >= in_bound) break;

            op_ptr reorder_op = std::make_shared<impl::op_t>(op_kind::convert);
            insert_op_before(reorder_op, cur_op, i);
            to_be_inserted_ops.emplace_back(reorder_op);
        }
    }

    for (const auto &op : to_be_inserted_ops)
        subgraph.emplace_back(std::move(op));
}

void insert_permute(std::vector<op_ptr> &subgraph) {
    std::vector<op_ptr> to_be_inserted_ops;
    for (auto &cur_op : subgraph) {
        if (!need_insert_permute(cur_op->get_kind())) continue;

        bool need_permute_0 = cur_op->has_attr("data_format")
                ? (cur_op->get_attr<std::string>("data_format") == "NXC")
                : false;
        bool need_permute_1 = cur_op->has_attr("filter_format")
                ? (cur_op->get_attr<std::string>("filter_format") == "XIO")
                : false;
        bool need_permute_sum_1 = cur_op->has_attr("with_sum")
                ? (cur_op->get_attr<bool>("with_sum") && need_permute_0)
                : false;

        if (!(need_permute_0 || need_permute_1 || need_permute_sum_1)) continue;

        for (size_t i = 0; i < cur_op->num_inputs(); i++) {
            op_ptr perm_op = std::make_shared<impl::op_t>(op_kind::permute);
            perm_op->set_attr<std::string>("permute_kind", "permute");
            if (i == 0 && need_permute_0) {
                perm_op->set_attr<std::string>("from_format", "NXC");
                perm_op->set_attr<std::string>("to_format", "NCX");
            } else if (i == 1 && need_permute_1) {
                perm_op->set_attr<std::string>("from_format", "XIO");
                perm_op->set_attr<std::string>("to_format", "OIX");
            } else if (i == cur_op->num_inputs() - 1 && need_permute_sum_1) {
                perm_op->set_attr<std::string>("from_format", "NXC");
                perm_op->set_attr<std::string>("to_format", "NCX");
            } else {
                continue;
            }

            insert_op_before(perm_op, cur_op, i);
            to_be_inserted_ops.emplace_back(perm_op);
        }

        // remove the attrs in cur_op to avoid re-permute
        cur_op->set_attr<std::string>("data_format", "NCX");
        cur_op->set_attr<std::string>("filter_format", "OIX");

        if (need_permute_0)
            cur_op->set_attr<std::string>("output_format", "NXC");
    }
    for (const auto &op : to_be_inserted_ops)
        subgraph.emplace_back(std::move(op));
}

void insert_to_group_for_conv(std::vector<op_ptr> &subgraph) {
    std::vector<op_ptr> to_be_inserted_ops;
    for (auto &cur_op : subgraph) {
        if (cur_op->get_kind() != op_kind::dnnl_convolution) continue;

        auto groups = cur_op->get_attr<int64_t>("groups");
        if (groups <= 1) continue;

        op_ptr to_group_op = std::make_shared<impl::op_t>(op_kind::to_group);
        to_group_op->set_attr<int64_t>("groups", groups);

        insert_op_before(to_group_op, cur_op, 1);
        to_be_inserted_ops.emplace_back(to_group_op);
    }
    for (const auto &op : to_be_inserted_ops)
        subgraph.emplace_back(std::move(op));
}

void insert_transpose_for_matmul(std::vector<op_ptr> &subgraph) {
    std::vector<op_ptr> to_be_inserted_ops;
    for (auto &cur_op : subgraph) {
        if (cur_op->get_kind() != op_kind::MatMul) continue;

        std::vector<bool> trans_flag(2);
        if (cur_op->has_attr("transpose_a"))
            trans_flag[0] = cur_op->get_attr<bool>("transpose_a");
        if (cur_op->has_attr("transpose_b"))
            trans_flag[1] = cur_op->get_attr<bool>("transpose_b");
        if (!(trans_flag[0] || trans_flag[1])) continue;

        for (size_t i = 0; i < trans_flag.size(); ++i) {
            // skip if transpose flag is false or the input's ndim is 1
            // otherwise, we need do do expand
            if (!trans_flag[i]
                    || cur_op->get_input_value(i)->get_logical_tensor().ndims
                            <= 1)
                continue;
            op_ptr transpose_op = std::make_shared<op_t>(op_kind::permute);
            transpose_op->set_attr<std::string>("permute_kind", "transpose");
            insert_op_before(transpose_op, cur_op, i);
            to_be_inserted_ops.emplace_back(transpose_op);
        }
        // remove attr to avoid re-transpose during shape inference
        cur_op->set_attr<bool>("transpose_a", false);
        cur_op->set_attr<bool>("transpose_b", false);
    }
    for (const auto &op : to_be_inserted_ops)
        subgraph.emplace_back(std::move(op));
}

void insert_expand_for_matmul(std::vector<op_ptr> &subgraph) {
    std::vector<op_ptr> to_be_inserted_ops;
    for (auto &cur_op : subgraph) {
        if (cur_op->get_kind() != op_kind::MatMul) continue;

        std::vector<op_ptr> expand_ops;
        expand_ops.reserve(cur_op->num_inputs());

        int32_t new_src_ndims, new_wei_ndims;
        // FIXME(wuxun): if the producer op is transpose, the ndims is unknown
        // and should be derived from producer's producer
        if (cur_op->get_input_value(0)->has_producer()) {
            auto &src_producer = cur_op->get_input_value(0)->get_producer();
            if (src_producer.get_kind() == op_kind::permute) {
                new_src_ndims = src_producer.get_input_value(0)
                                        ->get_logical_tensor()
                                        .ndims;
            } else {
                new_src_ndims = cur_op->get_input_value(0)
                                        ->get_logical_tensor()
                                        .ndims;
            }
        } else {
            new_src_ndims
                    = cur_op->get_input_value(0)->get_logical_tensor().ndims;
        }

        if (cur_op->get_input_value(1)->has_producer()) {
            auto &wei_producer = cur_op->get_input_value(1)->get_producer();
            if (wei_producer.get_kind() == op_kind::permute) {
                new_wei_ndims = wei_producer.get_input_value(0)
                                        ->get_logical_tensor()
                                        .ndims;
            } else {
                new_wei_ndims = cur_op->get_input_value(1)
                                        ->get_logical_tensor()
                                        .ndims;
            }
        } else {
            new_wei_ndims
                    = cur_op->get_input_value(1)->get_logical_tensor().ndims;
        }

        std::vector<int32_t> ori_ndims {new_src_ndims, new_wei_ndims};
        int32_t dst_ndims
                = cur_op->get_output_value(0)->get_logical_tensor().ndims;
        for (size_t i = 0; i < cur_op->num_inputs(); ++i) {
            auto expand_op = std::make_shared<op_t>(op_kind::expand);
            if (i < 2) { // src and weight
                auto ndims = ori_ndims[i];
                if (ndims != 1) {
                    expand_ops.emplace_back(expand_op);
                    continue;
                }
                // 1D -> 2D
                if (i == 0) {
                    expand_op->set_attr<std::string>("insert_1dim", "before");
                    new_src_ndims = ndims + 1;
                } else if (i == 1) {
                    expand_op->set_attr<std::string>("insert_1dim", "after");
                    new_wei_ndims = ndims + 1;
                }
            } else { // bias
                // expand bias to dst ndims if they are mis-matched
                if (cur_op->get_input_value(i)->get_logical_tensor().ndims
                        != dst_ndims)
                    expand_op->set_attr<int64_t>("expand_to", dst_ndims);
            }
            expand_ops.emplace_back(expand_op);
        }

        for (size_t i = 0; i < expand_ops.size(); ++i) {
            if (i == 0 && new_src_ndims < new_wei_ndims) {
                expand_ops[i]->set_attr<int64_t>("expand_to", new_wei_ndims);
            } else if (i == 1 && new_wei_ndims < new_src_ndims) {
                expand_ops[i]->set_attr<int64_t>("expand_to", new_src_ndims);
            }

            insert_op_before(expand_ops[i], cur_op, i);
            to_be_inserted_ops.emplace_back(expand_ops[i]);
        }
    }
    for (const auto &op : to_be_inserted_ops)
        subgraph.emplace_back(std::move(op));
}

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl
