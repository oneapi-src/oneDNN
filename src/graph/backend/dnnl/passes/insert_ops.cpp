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
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "graph/interface/c_types_map.hpp"

#include "graph/backend/dnnl/common.hpp"
#include "graph/backend/dnnl/internal_attrs.hpp"
#include "graph/backend/dnnl/passes/insert_ops.hpp"
#include "graph/backend/dnnl/passes/utils.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {
using op_t = op_t;
using op_ptr = std::shared_ptr<op_t>;
using value_ptr = std::shared_ptr<value_t>;

// TODO(xxx): extend to support other ops
// for those ops with data_format/filter_format attributes
static bool need_insert_permute(op_kind_t kind) {
    static const std::set<op_kind_t> ops {op_kind::dnnl_convolution,
            op_kind::dnnl_conv_depthwise, op_kind::dnnl_convtranspose,
            op_kind::dnnl_convtranspose_bwd_data, op_kind::dnnl_pool,
            op_kind::dnnl_batchnorm, op_kind::dnnl_prelu,
            op_kind::dnnl_prelu_bwd, op_kind::dnnl_resampling,
            op_kind::dnnl_resampling_bwd};
    return ops.count(kind) != 0;
}

status_t insert_permute(std::shared_ptr<subgraph_t> &sg) {
    auto &subgraph = sg->get_mutable_ops();
    std::vector<op_ptr> to_be_inserted_ops;
    std::vector<op_ptr> to_be_removed_ops;

    // insert permute for convolution/convtranspose op
    auto insert_permute_for_conv_or_deconv = [&](op_ptr &conv_op) -> bool {
        const bool need_permute_0 = conv_op->has_attr(op_attr::data_format)
                ? (conv_op->get_attr<std::string>(op_attr::data_format)
                        == "NXC")
                : false;
        const bool need_permute_1 = conv_op->has_attr(op_attr::filter_format)
                ? (conv_op->get_attr<std::string>(op_attr::filter_format)
                        == "XIO")
                : false;
        // conv + depthwise case
        const bool need_permute_2 = conv_op->has_attr(op_attr::dw_filter_format)
                ? (conv_op->get_attr<std::string>(op_attr::dw_filter_format)
                        == "XIO")
                : false;

        if (!(need_permute_0 || need_permute_1 || need_permute_2)) return false;

        for (size_t i = 0; i < conv_op->num_inputs(); ++i) {
            op_ptr perm_op = std::make_shared<op_t>(op_kind::permute);
            perm_op->set_attr<std::string>(op_attr::permute_kind, "permute");

            if (i == 0) {
                if (!need_permute_0) continue;
                perm_op->set_attr<std::string>(op_attr::from_format, "NXC");
                perm_op->set_attr<std::string>(op_attr::to_format, "NCX");
            } else if (i == 1) {
                if (!need_permute_1) continue;
                perm_op->set_attr<std::string>(op_attr::from_format, "XIO");
                perm_op->set_attr<std::string>(op_attr::to_format, "OIX");
            } else if (i == 2) {
                // skip for bias input
                if (conv_op->get_attr<bool>(op_attr::with_bias)) continue;
                if (need_permute_2) {
                    perm_op->set_attr<std::string>(op_attr::from_format, "XIO");
                    perm_op->set_attr<std::string>(op_attr::to_format, "OIX");
                } else if (need_permute_0
                        && conv_op->get_kind()
                                != op_kind::dnnl_conv_depthwise) {
                    // this input is also the input of post binary ops
                    perm_op->set_attr<std::string>(op_attr::from_format, "NXC");
                    perm_op->set_attr<std::string>(op_attr::to_format, "NCX");
                } else {
                    continue;
                }
            } else {
                // if not set data_format as NXC, no need to permute for other
                // inputs of post binary ops
                if (!need_permute_0) continue;
                perm_op->set_attr<std::string>(op_attr::from_format, "NXC");
                perm_op->set_attr<std::string>(op_attr::to_format, "NCX");
            }

            insert_op_before(perm_op, conv_op, i);
            to_be_inserted_ops.emplace_back(perm_op);
        }

        // remove the attrs in cur_op to avoid re-permute
        conv_op->set_attr<std::string>(op_attr::data_format, "NCX");
        conv_op->set_attr<std::string>(op_attr::filter_format, "OIX");
        // conv + depthwise case
        if (need_permute_2)
            conv_op->set_attr<std::string>(op_attr::dw_filter_format, "OIX");

        return need_permute_0;
    };

    // insert permute for those ops only requiring data_format attribute
    // (e.g pool, batchnorm, interpolate)
    auto insert_permute_for_op_only_require_data_format
            = [&](op_ptr &op) -> bool {
        const bool need_permute_0 = op->has_attr(op_attr::data_format)
                ? (op->get_attr<std::string>(op_attr::data_format) == "NXC")
                : false;
        if (!need_permute_0) return false;

        size_t num_post_binary_ops = 0;
        const auto &mgr = sg->fusion_info_mgr_;
        if (op->has_attr(op_attr::fusion_info_key)
                && op->get_attr<int64_t>(op_attr::fusion_info_key) != -1) {
            int64_t key = op->get_attr<int64_t>(op_attr::fusion_info_key);
            const auto &pops = mgr.get_info(key).get_post_ops();
            for (size_t n = 0; n < pops.size(); ++n) {
                if (pops[n]->get_op()->get_kind() == op_kind::dnnl_binary)
                    num_post_binary_ops++;
            }
        }

        for (size_t i = 0; i < op->num_inputs(); ++i) {
            // Skip for those non-data input and non-post-binary inputs,
            // If PReLU/PReLUBackprop data format is NXC, we need to permute all
            // inputs.
            if (i > 0 && i < op->num_inputs() - num_post_binary_ops
                    && op->get_kind() != op_kind::dnnl_prelu
                    && op->get_kind() != op_kind::dnnl_prelu_bwd
                    && op->get_kind() != op_kind::dnnl_resampling_bwd)
                continue;
            // Skip optional non-data input for resampling backward op
            if (i > 1 && op->get_kind() == op_kind::dnnl_resampling_bwd)
                continue;

            op_ptr perm_op = std::make_shared<op_t>(op_kind::permute);
            perm_op->set_attr<std::string>(op_attr::permute_kind, "permute");
            perm_op->set_attr<std::string>(op_attr::from_format, "NXC");
            perm_op->set_attr<std::string>(op_attr::to_format, "NCX");
            insert_op_before(perm_op, op, i);
            to_be_inserted_ops.emplace_back(perm_op);
        }
        // remove the attrs in cur_op to avoid re-permute
        op->set_attr<std::string>(op_attr::data_format, "NCX");
        return true;
    };

    for (auto &cur_op : subgraph) {
        if (!need_insert_permute(cur_op->get_kind())) continue;

        bool require_output_permute = false;
        if (cur_op->get_kind() == op_kind::dnnl_convolution
                || cur_op->get_kind() == op_kind::dnnl_conv_depthwise
                || cur_op->get_kind() == op_kind::dnnl_convtranspose
                || cur_op->get_kind() == op_kind::dnnl_convtranspose_bwd_data) {
            require_output_permute = insert_permute_for_conv_or_deconv(cur_op);
        } else {
            require_output_permute
                    = insert_permute_for_op_only_require_data_format(cur_op);
        }

        // permute output back to NXC
        if (require_output_permute) {
            op_ptr perm_op = std::make_shared<op_t>(op_kind::permute);
            perm_op->set_attr<std::string>(op_attr::permute_kind, "permute");
            perm_op->set_attr<std::string>(op_attr::from_format, "NCX");
            perm_op->set_attr<std::string>(op_attr::to_format, "NXC");
            insert_op_after(perm_op, cur_op, 0);
            to_be_inserted_ops.emplace_back(perm_op);

            // Insert permute after prelu bprop second output
            if (cur_op->get_kind() == op_kind::dnnl_prelu_bwd) {
                op_ptr perm_op_1 = std::make_shared<op_t>(op_kind::permute);
                perm_op_1->set_attr<std::string>(
                        op_attr::permute_kind, "permute");
                perm_op_1->set_attr<std::string>(op_attr::from_format, "NCX");
                perm_op_1->set_attr<std::string>(op_attr::to_format, "NXC");
                insert_op_after(perm_op_1, cur_op, 1);
                to_be_inserted_ops.emplace_back(perm_op_1);
            }
        }
    }
    for (const auto &op : to_be_inserted_ops)
        subgraph.emplace_back(op);
    for (const auto &op : to_be_removed_ops) {
        auto pos = std::find_if(subgraph.begin(), subgraph.end(),
                [op](const op_ptr &tmp) { return op.get() == tmp.get(); });
        if (pos != subgraph.end()) subgraph.erase(pos);
    }
    return infer_shape(sg);
}

status_t insert_permute_for_shuffle(std::shared_ptr<subgraph_t> &sg) {
    // optimization for NXC case - it will help to hit an optimized kernel
    auto &subgraph = sg->get_mutable_ops();
    std::vector<op_ptr> to_be_inserted_ops;
    for (auto &cur_op : subgraph) {
        if (cur_op->get_kind() != op_kind::dnnl_shuffle) continue;

        logical_tensor_t src_lt
                = cur_op->get_input_value(0)->get_logical_tensor();
        const logical_tensor_wrapper_t src(src_lt);
        const auto axis = cur_op->get_attr<int64_t>(op_attr::axis);
        const auto known_strides
                = (src.is_strided()) ? !src.is_stride_unknown() : false;
        const bool need_permute = axis == src.ndims() - 1 && known_strides
                && src.vstrides() == get_ncx_strides(src.vdims());
        if (!need_permute) continue;

        const int64_t new_axis = 1;
        cur_op->set_attr(op_attr::axis, new_axis);
        op_ptr perm_src_op = std::make_shared<op_t>(op_kind::permute);
        perm_src_op->set_attr<std::string>(op_attr::permute_kind, "permute");
        perm_src_op->set_attr<std::string>(op_attr::from_format, "NXC");
        perm_src_op->set_attr<std::string>(op_attr::to_format, "NCX");
        insert_op_before(perm_src_op, cur_op, 0);
        to_be_inserted_ops.emplace_back(perm_src_op);

        // permute output back to NXC
        op_ptr perm_dst_op = std::make_shared<op_t>(op_kind::permute);
        perm_dst_op->set_attr<std::string>(op_attr::permute_kind, "permute");
        perm_dst_op->set_attr<std::string>(op_attr::from_format, "NCX");
        perm_dst_op->set_attr<std::string>(op_attr::to_format, "NXC");
        insert_op_after(perm_dst_op, cur_op, 0);
        to_be_inserted_ops.emplace_back(perm_dst_op);
    }

    for (const auto &op : to_be_inserted_ops)
        subgraph.emplace_back(op);

    return infer_shape(sg);
}

status_t insert_to_group_for_conv_or_deconv(std::shared_ptr<subgraph_t> &sg) {
    auto &subgraph = sg->get_mutable_ops();
    std::vector<op_ptr> to_be_inserted_ops;
    std::vector<op_ptr> to_be_removed_ops;

    auto insert_to_group
            = [&to_be_inserted_ops](op_ptr &op, op_attr_t attr_name,
                      const size_t offset) -> bool {
        auto groups = op->get_attr<int64_t>(attr_name);
        if (groups <= 1) {
            op->set_attr<bool>(op_attr::canonicalized, true);
            return false;
        }

        op_ptr to_group_op = std::make_shared<op_t>(op_kind::to_group);
        to_group_op->set_attr<int64_t>(op_attr::groups, groups);

        op->set_attr<bool>(op_attr::canonicalized, true);
        op->set_attr<int64_t>(op_attr::groups, 1);

        insert_op_before(to_group_op, op, offset);
        to_be_inserted_ops.emplace_back(to_group_op);

        if (op->get_kind() == op_kind::dnnl_convtranspose
                || op->get_kind() == op_kind::dnnl_convtranspose_bwd_data)
            to_group_op->set_attr<bool>(op_attr::is_convtranspose, true);

        return true;
    };

    for (auto &cur_op : subgraph) {
        if (cur_op->get_kind() != op_kind::dnnl_convolution
                && cur_op->get_kind() != op_kind::dnnl_convtranspose
                && cur_op->get_kind() != op_kind::dnnl_convtranspose_bwd_data
                && cur_op->get_kind() != op_kind::dnnl_conv_depthwise)
            continue;

        if (cur_op->get_kind() == op_kind::dnnl_conv_depthwise) {
            const auto inserted
                    = insert_to_group(cur_op, op_attr::dw_groups, 2);
            if (!inserted) continue;
        }

        const auto inserted = insert_to_group(cur_op, op_attr::groups, 1);
        if (!inserted) continue;
    }
    for (const auto &op : to_be_inserted_ops)
        subgraph.emplace_back(op);
    for (const auto &op : to_be_removed_ops) {
        auto pos = std::find_if(subgraph.begin(), subgraph.end(),
                [op](const op_ptr &tmp) { return op.get() == tmp.get(); });
        if (pos != subgraph.end()) subgraph.erase(pos);
    }
    return infer_shape(sg);
}

status_t insert_to_group_for_reorder(std::shared_ptr<subgraph_t> &sg) {
    auto &subgraph = sg->get_mutable_ops();
    std::vector<op_ptr> to_be_inserted_ops;
    for (auto &cur_op : subgraph) {
        if (cur_op->get_kind() != op_kind::dnnl_reorder) continue;
        auto in_md = make_dnnl_memory_desc(
                cur_op->get_input_value(0)->get_logical_tensor());
        auto out_md = make_dnnl_memory_desc(
                cur_op->get_output_value(0)->get_logical_tensor());
        if (in_md.data.ndims == out_md.data.ndims) {
            // no group
            return status::success;
        } else if (in_md.data.ndims == out_md.data.ndims + 1) {
            // reorder's input has blocked format with group
            // while output has plain format, perhaps for
            // backward path. No such case for now, disable
            return status::unimplemented;
        } else if (in_md.data.ndims + 1 == out_md.data.ndims) {
            // reorder's input has plain format while output
            // has blocked format with group, typically for
            // weight prepacking
            auto group = out_md.data.dims[0];
            if (group * out_md.data.dims[1] != in_md.data.dims[0])
                return status::invalid_shape;

            // insert to_group op
            op_ptr to_group_op = std::make_shared<op_t>(op_kind::to_group);
            to_group_op->set_attr<int64_t>(op_attr::groups, group);

            insert_op_before(to_group_op, cur_op, 0);
            to_be_inserted_ops.emplace_back(to_group_op);
        } else {
            // illegal shape
            return status::invalid_shape;
        }
    }

    for (const auto &op : to_be_inserted_ops)
        subgraph.emplace_back(op);
    return status::success;
}

status_t insert_transpose_for_matmul(std::shared_ptr<subgraph_t> &sg) {
    auto &subgraph = sg->get_mutable_ops();
    std::vector<op_ptr> to_be_inserted_ops;
    for (auto &cur_op : subgraph) {
        if (cur_op->get_kind() != op_kind::dnnl_matmul) continue;

        std::vector<bool> trans_flag(2);
        if (cur_op->has_attr(op_attr::transpose_a))
            trans_flag[0] = cur_op->get_attr<bool>(op_attr::transpose_a);
        if (cur_op->has_attr(op_attr::transpose_b))
            trans_flag[1] = cur_op->get_attr<bool>(op_attr::transpose_b);
        if (!(trans_flag[0] || trans_flag[1])) continue;

        for (size_t i = 0; i < trans_flag.size(); ++i) {
            // skip if transpose flag is false or the input's ndim is 1
            // otherwise, we need do do expand
            if (!trans_flag[i]
                    || cur_op->get_input_value(i)->get_logical_tensor().ndims
                            <= 1)
                continue;
            op_ptr transpose_op = std::make_shared<op_t>(op_kind::permute);
            transpose_op->set_attr<std::string>(
                    op_attr::permute_kind, "transpose");
            insert_op_before(transpose_op, cur_op, i);
            to_be_inserted_ops.emplace_back(transpose_op);
        }
        // remove attr to avoid re-transpose during shape inference
        cur_op->set_attr<bool>(op_attr::transpose_a, false);
        cur_op->set_attr<bool>(op_attr::transpose_b, false);
    }
    for (const auto &op : to_be_inserted_ops)
        subgraph.emplace_back(op);
    return infer_shape(sg);
}

status_t insert_reshape_for_ndx2d_matmul(std::shared_ptr<subgraph_t> &sg) {
    auto &subgraph = sg->get_mutable_ops();
    auto &mgr = sg->fusion_info_mgr_;

    std::vector<op_ptr> to_be_inserted_ops;
    for (auto &cur_op : subgraph) {
        if (cur_op->get_kind() != op_kind::dnnl_matmul) continue;
        // skip due to dnnl cannot reshape such kind of strided memory desc
        if (cur_op->get_input_value(0)->has_producer()
                && cur_op->get_input_value(0)->get_producer().get_kind()
                        == op_kind::permute) {
            continue;
        }

        const bool with_bias = cur_op->has_attr(op_attr::with_bias)
                ? cur_op->get_attr<bool>(op_attr::with_bias)
                : false;
        const size_t expected = with_bias ? 4 : 3;
        // TODO(xx): handle multiple post-binary case
        if (cur_op->num_inputs() > expected) { continue; }

        int32_t src_ndims
                = cur_op->get_input_value(0)->get_logical_tensor().ndims;
        int32_t wei_ndims
                = cur_op->get_input_value(1)->get_logical_tensor().ndims;
        if (wei_ndims != 2 || src_ndims <= 2) continue;

        auto src_dims = logical_tensor_wrapper_t(
                cur_op->get_input_value(0)->get_logical_tensor())
                                .vdims();
        dims expected_dims {-1, src_dims.back()};
        auto reshape_op = std::make_shared<op_t>(graph::op_kind::StaticReshape);
        reshape_op->set_attr<bool>(op_attr::special_zero, false);
        reshape_op->set_attr<std::vector<int64_t>>(
                op_attr::shape, expected_dims);
        to_be_inserted_ops.emplace_back(reshape_op);
        insert_op_before(reshape_op, cur_op, 0);

        dims expected_dims2(src_dims);
        expected_dims2[expected_dims2.size() - 1] = 0;
        auto reshape_op2
                = std::make_shared<op_t>(graph::op_kind::StaticReshape);
        reshape_op2->set_attr<bool>(op_attr::special_zero, true);
        reshape_op2->set_attr<std::vector<int64_t>>(
                op_attr::shape, expected_dims2);
        to_be_inserted_ops.emplace_back(reshape_op2);
        insert_op_after(reshape_op2, cur_op, 0);

        if (!with_bias && cur_op->num_inputs() == 3) {
            auto post_src_dims = logical_tensor_wrapper_t(
                    cur_op->get_input_value(2)->get_logical_tensor())
                                         .vdims();
            dims expected_dims3 {-1, post_src_dims.back()};
            auto reshape_op3
                    = std::make_shared<op_t>(graph::op_kind::StaticReshape);
            reshape_op3->set_attr<bool>(op_attr::special_zero, false);
            reshape_op3->set_attr<std::vector<int64_t>>(
                    op_attr::shape, expected_dims3);
            to_be_inserted_ops.emplace_back(reshape_op3);
            insert_op_before(reshape_op3, cur_op, 2);
        } else if (with_bias && cur_op->num_inputs() == 4) {
            auto post_src_dims = logical_tensor_wrapper_t(
                    cur_op->get_input_value(3)->get_logical_tensor())
                                         .vdims();
            dims expected_dims3 {-1, post_src_dims.back()};
            auto reshape_op3
                    = std::make_shared<op_t>(graph::op_kind::StaticReshape);
            reshape_op3->set_attr<bool>(op_attr::special_zero, false);
            reshape_op3->set_attr<std::vector<int64_t>>(
                    op_attr::shape, expected_dims3);
            to_be_inserted_ops.emplace_back(reshape_op3);
            insert_op_before(reshape_op3, cur_op, 3);
        }

        // update the axis
        if (cur_op->has_attr(op_attr::fusion_info_key)
                && cur_op->get_attr<int64_t>(op_attr::fusion_info_key) != -1) {
            int64_t key = cur_op->get_attr<int64_t>(op_attr::fusion_info_key);
            fusion_info_t &fusion_info = mgr.get_mutable_info(key);
            op_t *oscales_op = fusion_info.get_mutable_output_scales();
            if (oscales_op
                    && oscales_op->get_attr<std::string>(op_attr::qtype)
                            == "per_channel") {
                oscales_op->set_attr<int64_t>(op_attr::axis, 1); // the 2nd dim;
            }
        }
    }
    for (const auto &op : to_be_inserted_ops)
        subgraph.emplace_back(op);
    return infer_shape(sg);
}

status_t insert_expand_and_squeeze_for_matmul(std::shared_ptr<subgraph_t> &sg) {
    auto &subgraph = sg->get_mutable_ops();
    std::vector<op_ptr> to_be_inserted_ops;
    for (auto &cur_op : subgraph) {
        if (cur_op->get_kind() != op_kind::dnnl_matmul) continue;

        std::vector<op_ptr> expand_ops;
        expand_ops.reserve(cur_op->num_inputs());

        int32_t new_src_ndims
                = cur_op->get_input_value(0)->get_logical_tensor().ndims;
        int32_t new_wei_ndims
                = cur_op->get_input_value(1)->get_logical_tensor().ndims;
        assertm(new_src_ndims >= 1 && new_wei_ndims >= 1, "invalid dims");

        std::vector<int32_t> ori_ndims {new_src_ndims, new_wei_ndims};
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
                    expand_op->set_attr<std::string>(
                            op_attr::insert_1dim, "before");
                    new_src_ndims = ndims + 1;
                } else if (i == 1) {
                    expand_op->set_attr<std::string>(
                            op_attr::insert_1dim, "after");
                    new_wei_ndims = ndims + 1;
                }
            } else { // bias
                // expand bias to dst ndims if they are mis-matched
                int64_t tgt_ndims = std::max(new_src_ndims, new_wei_ndims);
                if (cur_op->get_input_value(i)->get_logical_tensor().ndims
                        != tgt_ndims)
                    expand_op->set_attr<int64_t>(op_attr::expand_to, tgt_ndims);
            }
            expand_ops.emplace_back(expand_op);
        }

        std::vector<int64_t> squeeze_dims = {};
        for (size_t i = 0; i < expand_ops.size(); ++i) {
            if (i == 0 && new_src_ndims < new_wei_ndims) {
                expand_ops[i]->set_attr<int64_t>(
                        op_attr::expand_to, new_wei_ndims);
            } else if (i == 1 && new_wei_ndims < new_src_ndims) {
                expand_ops[i]->set_attr<int64_t>(
                        op_attr::expand_to, new_src_ndims);
            }

            // insert expand ops
            if ((expand_ops[i]->has_attr(op_attr::insert_1dim)
                        && expand_ops[i]->get_attr<std::string>(
                                   op_attr::insert_1dim)
                                != "none")
                    || (expand_ops[i]->has_attr(op_attr::expand_to)
                            && expand_ops[i]->get_attr<int64_t>(
                                       op_attr::expand_to)
                                    != -1)) {
                insert_op_before(expand_ops[i], cur_op, i);
                to_be_inserted_ops.emplace_back(expand_ops[i]);
            }

            // decide squeeze dims
            if (expand_ops[i]->has_attr(op_attr::insert_1dim)
                    && expand_ops[i]->get_attr<std::string>(
                               op_attr::insert_1dim)
                            != "none") {
                // -2 means the second rightmost dimension, -1 means the
                // rightmost dimension
                int64_t dim_to_be_squeezed
                        = expand_ops[i]->get_attr<std::string>(
                                  op_attr::insert_1dim)
                                == "before"
                        ? static_cast<int64_t>(-2)
                        : static_cast<int64_t>(-1);
                squeeze_dims.push_back(dim_to_be_squeezed);
            }
        }

        // insert squeeze ops
        if (!squeeze_dims.empty()) {
            op_ptr squeeze_op = std::make_shared<op_t>(op_kind::squeeze);
            squeeze_op->set_attr<std::vector<int64_t>>(
                    op_attr::axes, squeeze_dims);
            insert_op_after(squeeze_op, cur_op, 0);
            to_be_inserted_ops.emplace_back(squeeze_op);
        }
    }
    for (const auto &op : to_be_inserted_ops)
        subgraph.emplace_back(op);
    return infer_shape(sg);
}

status_t insert_u8_to_s8_for_matmul(std::shared_ptr<subgraph_t> &sg) {
    auto &subgraph = sg->get_mutable_ops();
    auto &mgr = sg->fusion_info_mgr_;

    std::vector<op_ptr> to_be_inserted_ops;
    for (auto &cur_op : subgraph) {
        if (cur_op->get_kind() != op_kind::dnnl_matmul) continue;

        int32_t new_src0_dtype
                = cur_op->get_input_value(0)->get_logical_tensor().data_type;
        int32_t new_src1_dtype
                = cur_op->get_input_value(1)->get_logical_tensor().data_type;
        if (!graph::utils::one_of(
                    new_src0_dtype, graph::data_type::u8, graph::data_type::s8)
                || new_src1_dtype != graph::data_type::u8)
            continue;

        int64_t key = -1;
        if (cur_op->get_attr<int64_t>(op_attr::fusion_info_key) != -1) {
            key = cur_op->get_attr<int64_t>(op_attr::fusion_info_key);
        } else {
            key = mgr.init_info();
            cur_op->set_attr<int64_t>(op_attr::fusion_info_key, key);
        }
        fusion_info_t &fusion_info = mgr.get_mutable_info(key);
        op_t *wei_zps_op = fusion_info.get_mutable_zero_points(
                true, /*the wei indice*/ 1);
        if (wei_zps_op) { // already fused zps, update the zps
            std::vector<int64_t> current_zp
                    = wei_zps_op->get_attr<std::vector<int64_t>>(op_attr::zps);
            if (current_zp.size() != 1) continue;
            // the equivalent transformation: mm(src_u8, wei_u8) -> mm(src_u8,
            // wei_u8 - 128 + 128) -> mm(src_u8, wei_s8 + 128), which wei_s8 =
            // wei_u8 - 128
            std::vector<int64_t> adjusted_zp {current_zp[0] + 128};
            wei_zps_op->set_attr<std::vector<int64_t>>(
                    op_attr::zps, adjusted_zp);
        } else { // fuse a 128 zps
            std::vector<int64_t> zp {128};
            auto zps_op = std::make_shared<op_t>(op_kind::dnnl_add_zps);
            zps_op->set_attr<std::string>(op_attr::qtype, "per_tensor");
            zps_op->set_attr<int64_t>(op_attr::axis, 0);
            zps_op->set_attr<std::vector<int64_t>>(op_attr::zps, zp);
            fusion_info.set_zero_points(zps_op, true, /*the wei indice*/ 1);
        }

        op_ptr u8_to_s8_op = std::make_shared<op_t>(op_kind::dnnl_reorder);
        u8_to_s8_op->set_attr<std::vector<int64_t>>(
                op_attr::dst_zps, std::vector<int64_t> {-128});
        insert_op_before(u8_to_s8_op, cur_op, 1);
        u8_to_s8_op->get_output_value(0)->set_data_type(graph::data_type::s8);
        insert_empty_scratchpad(u8_to_s8_op);
        to_be_inserted_ops.emplace_back(u8_to_s8_op);
    }
    for (const auto &op : to_be_inserted_ops)
        subgraph.emplace_back(op);

    return infer_shape(sg);
}

status_t insert_expand_for_prelu(std::shared_ptr<subgraph_t> &sg) {
    using ltw = logical_tensor_wrapper_t;

    std::vector<op_ptr> to_be_inserted_ops;
    std::vector<op_ptr> to_be_removed_ops;

    auto &subgraph = sg->get_mutable_ops();

    for (auto &cur_op : subgraph) {
        if (cur_op->get_kind() != op_kind::dnnl_prelu) continue;

        // check doable
        auto src_lt = cur_op->get_input_value(0)->get_logical_tensor();
        auto wei_lt = cur_op->get_input_value(1)->get_logical_tensor();
        const std::string data_format
                = cur_op->get_attr<std::string>(op_attr::data_format);
        const bool per_channel_broadcast
                = cur_op->get_attr<bool>(op_attr::per_channel_broadcast);

        if (!prelu_doable(ltw(src_lt).vdims(), ltw(wei_lt).vdims(), data_format,
                    per_channel_broadcast)) {
            return status::invalid_shape;
        }
        // insert expand op
        int32_t src_ndims = src_lt.ndims;
        int32_t wei_ndims = wei_lt.ndims;
        // we only broadcast wei dims
        if (wei_ndims != src_ndims) {
            auto expand_op = std::make_shared<op_t>(op_kind::expand);
            expand_op->set_attr<int64_t>(op_attr::expand_to, src_ndims);
            // insert op before weights, which are the second input
            int wei_id = 1;
            insert_op_before(expand_op, cur_op, wei_id);
            to_be_inserted_ops.emplace_back(expand_op);

            if (data_format == "NCX" && src_ndims > 2
                    && per_channel_broadcast) {
                // If data format is NCX we need to permute expanded weights,
                // which are in format NXC.
                op_ptr perm_op = std::make_shared<op_t>(op_kind::permute);
                perm_op->set_attr<std::string>(
                        op_attr::permute_kind, "permute");
                perm_op->set_attr<std::string>(op_attr::from_format, "NXC");
                perm_op->set_attr<std::string>(op_attr::to_format, "NCX");
                insert_op_after(perm_op, expand_op, 0);
                to_be_inserted_ops.emplace_back(perm_op);
            }
        }
    }

    for (const auto &op : to_be_inserted_ops) {
        subgraph.emplace_back(op);
    }

    for (const auto &op : to_be_removed_ops) {
        auto pos = std::find_if(subgraph.begin(), subgraph.end(),
                [op](const op_ptr &tmp) { return op.get() == tmp.get(); });
        if (pos != subgraph.end()) subgraph.erase(pos);
    }

    return infer_shape(sg);
}

status_t insert_expand_and_squeeze_for_prelu_bwd(
        std::shared_ptr<subgraph_t> &sg) {
    using ltw = logical_tensor_wrapper_t;

    std::vector<op_ptr> to_be_inserted_ops;
    std::vector<op_ptr> to_be_removed_ops;

    auto &subgraph = sg->get_mutable_ops();

    for (auto &cur_op : subgraph) {
        if (cur_op->get_kind() != op_kind::dnnl_prelu_bwd) continue;

        // check doable
        auto src_lt = cur_op->get_input_value(0)->get_logical_tensor();
        auto wei_lt = cur_op->get_input_value(1)->get_logical_tensor();
        const auto wei_vdims = ltw(wei_lt).vdims();
        const std::string data_format
                = cur_op->get_attr<std::string>(op_attr::data_format);

        // In backward pass if the slope is one-dimensional
        // and its dims[0] != 1, then per channel broadcast will be performed.
        const bool per_channel_broadcast
                = wei_vdims.size() == 1 && wei_vdims[0] != 1;

        if (!prelu_doable(ltw(src_lt).vdims(), wei_vdims, data_format,
                    per_channel_broadcast)) {
            return status::invalid_shape;
        }
        // insert expand op
        int32_t src_ndims = src_lt.ndims;
        int32_t wei_ndims = wei_lt.ndims;
        // we only broadcast wei dims
        if (wei_ndims != src_ndims) {
            auto expand_op = std::make_shared<op_t>(op_kind::expand);
            expand_op->set_attr<int64_t>(op_attr::expand_to, src_ndims);
            // insert expand before weights, which are the second input
            int wei_id = 1;
            insert_op_before(expand_op, cur_op, wei_id);
            to_be_inserted_ops.emplace_back(expand_op);

            auto expand_op_diff_wei = std::make_shared<op_t>(op_kind::expand);
            expand_op_diff_wei->set_attr<int64_t>(
                    op_attr::expand_to, src_ndims);
            // insert expand before diff_weights, which are the second output
            insert_op_after(expand_op_diff_wei, cur_op, wei_id);
            to_be_inserted_ops.emplace_back(expand_op_diff_wei);

            if (data_format == "NCX" && src_ndims > 2
                    && per_channel_broadcast) {
                // If data format is NCX we need to permute expanded weights,
                // which are in format NXC.
                op_ptr perm_op = std::make_shared<op_t>(op_kind::permute);
                perm_op->set_attr<std::string>(
                        op_attr::permute_kind, "permute");
                perm_op->set_attr<std::string>(op_attr::from_format, "NXC");
                perm_op->set_attr<std::string>(op_attr::to_format, "NCX");
                insert_op_after(perm_op, expand_op, 0);
                to_be_inserted_ops.emplace_back(perm_op);
            }

            std::vector<int64_t> squeeze_dims = {};
            if (wei_vdims.size() == 1) {
                squeeze_dims.resize(src_ndims - 1);
                squeeze_dims[0] = 0;
                if (data_format == "NCX") {
                    std::iota(squeeze_dims.begin() + 1, squeeze_dims.end(), 2);
                } else {
                    // data_format == "NXC"
                    std::iota(squeeze_dims.begin() + 1, squeeze_dims.end(), 1);
                }
            } else {
                squeeze_dims.resize(1);
                squeeze_dims[0] = 0;
            }

            op_ptr squeeze_op = std::make_shared<op_t>(op_kind::squeeze);
            squeeze_op->set_attr<std::vector<int64_t>>(
                    op_attr::axes, squeeze_dims);
            // Insert squeeze after diff weights, so that its dimensions
            // have their original shape.
            insert_op_after(squeeze_op, expand_op_diff_wei, 0);
            to_be_inserted_ops.emplace_back(squeeze_op);
        }
    }

    for (const auto &op : to_be_inserted_ops) {
        subgraph.emplace_back(op);
    }

    for (const auto &op : to_be_removed_ops) {
        auto pos = std::find_if(subgraph.begin(), subgraph.end(),
                [op](const op_ptr &tmp) { return op.get() == tmp.get(); });
        if (pos != subgraph.end()) subgraph.erase(pos);
    }

    return infer_shape(sg);
}

status_t insert_expand_and_squeeze_for_reduction(
        std::shared_ptr<subgraph_t> &sg) {
    auto &subgraph = sg->get_mutable_ops();
    std::vector<op_ptr> to_be_inserted_ops;

    for (auto &cur_op : subgraph) {
        if (cur_op->get_kind() != op_kind::dnnl_reduction) continue;

        const auto keep_dims = cur_op->get_attr<bool>(op_attr::keep_dims);
        if (keep_dims) continue;

        const auto axes = cur_op->get_attr<std::vector<int64_t>>(op_attr::axes);

        // go through successor OPs until reach the end of subgraph or OP
        // which is not supported as a reductions post-op
        op_t *cur_op_ptr = cur_op.get();
        while (!cur_op_ptr->get_output_value(0)->get_consumers().empty()) {
            value_ptr connector = cur_op_ptr->get_output_value(0);
            op_t &post_op = connector->get_consumers()[0].get_op();
            if (post_op.get_kind() != op_kind::dnnl_binary
                    && post_op.get_kind() != op_kind::dnnl_eltwise)
                break;

            size_t src1_offset
                    = (post_op.get_input_value(0).get() == connector.get()) ? 1
                                                                            : 0;
            // insert expand OP before binary's src1 input
            if (post_op.get_kind() == op_kind::dnnl_binary) {
                if (!post_binary_fusible(cur_op.get(), &post_op)) break;
                op_ptr expand_op = std::make_shared<op_t>(op_kind::expand);
                expand_op->set_attr<std::vector<int64_t>>(op_attr::axes, axes);
                insert_op_before(expand_op.get(), &post_op, src1_offset);
                to_be_inserted_ops.emplace_back(expand_op);
            }

            // set fresh value for cur_op_ptr output (which is post-op input
            // value), so later correct shape will be inferred
            logical_tensor_t new_lt = empty_logical_tensor_with_default_id();
            auto new_val
                    = std::make_shared<value_t>(*cur_op_ptr, 0, new_lt, true);
            new_val->set_data_type(cur_op_ptr->get_input_value(0)
                                           ->get_logical_tensor()
                                           .data_type);
            cur_op_ptr->connect_output(0, new_val);
            post_op.connect_input(1 - src1_offset, new_val);

            cur_op_ptr = &post_op;
        }

        op_ptr squeeze_op = std::make_shared<op_t>(op_kind::squeeze);
        squeeze_op->set_attr<std::vector<int64_t>>(op_attr::axes, axes);
        // insert squeeze op after reduction or after its last post-op
        insert_op_after(squeeze_op.get(), cur_op_ptr, 0);
        to_be_inserted_ops.emplace_back(squeeze_op);

        // set to true, as squeeze will be handled by separate op
        cur_op->set_attr(op_attr::keep_dims, true);
    }

    for (const auto &op : to_be_inserted_ops)
        subgraph.emplace_back(op);

    return infer_shape(sg);
}

status_t insert_maxpool_forward(std::shared_ptr<subgraph_t> &sg) {
    auto &subgraph = sg->get_mutable_ops();
    std::vector<op_ptr> to_be_inserted_ops;
    std::vector<op_ptr> to_be_removed_ops;
    for (auto &cur_op : subgraph) {
        if (cur_op->get_kind() != graph::op_kind::MaxPoolBackprop) continue;

        // For MaxPoolBackprop op, we get diff_src (the output) shape from it's
        // src input. While dnnl_pool_bwd op didn't define src input (because
        // it's not used in primitive computation, and AvgPoolBackprop also
        // didn't have src input), we must transfer the shape info from the src
        // input to dnnl_pool_bwd op's op_attr::input_shape attribute. So, we
        // need to check that the src input must have shape info.
        auto src_lt = cur_op->get_input_value(0)->get_logical_tensor();
        logical_tensor_wrapper_t src_ltw(src_lt);
        if (src_ltw.is_shape_unknown()) {
            DEBUG_PRINT_ERROR(
                    "MaxPoolBackprop op's src input must have valid shape");
            return status::invalid_shape;
        }

        op_ptr maxpool_bwd = std::make_shared<op_t>(op_kind::dnnl_pool_bwd);
        maxpool_bwd->merge_attributes(cur_op->get_attributes());
        maxpool_bwd->set_attr<std::string>(op_attr::kind, "maxpool");
        maxpool_bwd->set_attr<std::vector<int64_t>>(
                op_attr::input_shape, src_ltw.vdims());

        // connect diff_dst
        auto diff_dst_value = cur_op->get_input_value(1);
        diff_dst_value->remove_consumer(*cur_op, 1);
        diff_dst_value->add_consumer(*maxpool_bwd, 0);
        maxpool_bwd->add_input(diff_dst_value);

        if (cur_op->num_inputs() > 2) {
            // with indices. we can use the indices directly instead of
            // re-computing it
            auto indices_value = cur_op->get_input_value(2);
            indices_value->remove_consumer(*cur_op, 2);
            indices_value->add_consumer(*maxpool_bwd, 1);
            maxpool_bwd->add_input(indices_value);
        } else {
            // no indices. we need to insert a maxpool fwd to re-compute the
            // indices from src
            op_ptr maxpool_fwd = std::make_shared<op_t>(op_kind::dnnl_pool);
            maxpool_fwd->merge_attributes(cur_op->get_attributes());
            maxpool_fwd->set_attr<std::string>(op_attr::kind, "maxpool");

            // connect src value to fwd op
            auto src_value = cur_op->get_input_value(0);
            src_value->remove_consumer(*cur_op, 0);
            src_value->add_consumer(*maxpool_fwd, 0);
            maxpool_fwd->add_input(src_value);

            // create dst value for fwd op
            // this might be an extra end edge since no consumers
            logical_tensor_t maxpool_fwd_dst
                    = empty_logical_tensor_with_default_id();
            maxpool_fwd_dst.data_type
                    = src_value->get_logical_tensor().data_type;
            value_ptr maxpool_fwd_dst_value = std::make_shared<value_t>(
                    *maxpool_fwd, 0, maxpool_fwd_dst);
            maxpool_fwd->add_output(maxpool_fwd_dst_value);

            // create scratchpad value for fwd op
            insert_empty_scratchpad(maxpool_fwd);

            // create ws value for fwd op
            logical_tensor_t maxpool_fwd_ws
                    = empty_logical_tensor_with_default_id();
            value_ptr maxpool_fwd_ws_value = std::make_shared<value_t>(
                    *maxpool_fwd, 2, maxpool_fwd_ws);
            maxpool_fwd->add_output(maxpool_fwd_ws_value);

            // connect forward op's ws value to bwd op
            maxpool_fwd_ws_value->add_consumer(*maxpool_bwd, 1);
            maxpool_bwd->add_input(maxpool_fwd_ws_value);

            to_be_inserted_ops.emplace_back(maxpool_fwd);
        }

        // connect the forward src as the dnnl_pool_bwd op's 3rd input (used
        // to store the logical tensor which will be converted to a md to
        // create the forward hint)
        auto src_value = cur_op->get_input_value(0);
        src_value->remove_consumer(*cur_op, 0);
        src_value->add_consumer(*maxpool_bwd, 2);
        maxpool_bwd->add_input(src_value);

        // connect diff_src
        auto diff_src_value = cur_op->get_output_value(0);
        maxpool_bwd->add_output(diff_src_value);

        // connect scratchpad
        insert_empty_scratchpad(maxpool_bwd);

        to_be_inserted_ops.emplace_back(maxpool_bwd);
        to_be_removed_ops.emplace_back(cur_op);
    }

    for (const auto &op : to_be_inserted_ops)
        subgraph.emplace_back(op);
    for (const auto &op : to_be_removed_ops) {
        auto pos = std::find_if(subgraph.begin(), subgraph.end(),
                [op](const op_ptr &tmp) { return op.get() == tmp.get(); });
        if (pos != subgraph.end()) subgraph.erase(pos);
    }
    return infer_shape(sg);
}

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
