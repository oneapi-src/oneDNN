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
#include <algorithm>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>

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

status_t insert_permute_for_conv_or_deconv(std::shared_ptr<subgraph_t> &sg) {
    subgraph_rewriter_t rewriter(sg);
    const auto &mgr = sg->fusion_info_mgr_;

    for (auto &op : sg->get_ops()) {
        if (!(op->get_kind() == op_kind::dnnl_convolution
                    || op->get_kind() == op_kind::dnnl_convtranspose
                    || op->get_kind() == op_kind::dnnl_convtranspose_bwd_data))
            continue;

        const bool need_permute_src = op->has_attr(op_attr::data_format)
                && op->get_attr<std::string>(op_attr::data_format) == "NXC";
        const bool need_permute_wei = op->has_attr(op_attr::weights_format)
                && op->get_attr<std::string>(op_attr::weights_format) != "OIX";
        // conv + depthwise case
        bool need_permute_post_dw_conv_wei = false;
        const op_t *post_dw_conv = nullptr;
        fusion_info_t fusion_info;
        if (op->has_attr(op_attr::fusion_info_key)
                && op->get_attr<int64_t>(op_attr::fusion_info_key) != -1) {
            int64_t key = op->get_attr<int64_t>(op_attr::fusion_info_key);
            fusion_info = mgr.get_info(key);
        }
        if (fusion_info.has_post_dw_conv()) {
            post_dw_conv = fusion_info.get_post_dw_conv()->get_op();
            need_permute_post_dw_conv_wei = post_dw_conv->get_attr<std::string>(
                                                    op_attr::weights_format)
                    == "XIO";
        }

        size_t num_post_binary_ops = 0;
        size_t dw_conv_index = 2;
        bool with_runtime_dst_scales = false;
        bool with_runtime_dst_points = false;
        const auto &pops = fusion_info.get_post_ops();
        for (size_t n = 0; n < pops.size(); ++n) {
            if (pops[n]->get_op()->get_kind() == op_kind::dnnl_binary)
                num_post_binary_ops++;
        }

        if (fusion_info.with_runtime_scales(true, 0)) { dw_conv_index += 1; }
        if (fusion_info.with_runtime_scales(true, 1)) { dw_conv_index += 1; }
        if (fusion_info.with_runtime_zero_points(true, 0)) {
            dw_conv_index += 1;
        }
        if (fusion_info.with_runtime_zero_points(true, 1)) {
            dw_conv_index += 1;
        }
        if (fusion_info.with_runtime_scales(false, 0)) {
            with_runtime_dst_scales = true;
        }
        if (fusion_info.with_runtime_zero_points(false, 0)) {
            with_runtime_dst_points = true;
        }

        for (size_t i = 0; i < op->num_inputs() - with_runtime_dst_scales
                        - with_runtime_dst_points;
                ++i) {
            auto val = op->get_input_value(i);
            auto ndims = val->get_logical_tensor().ndims;

            std::vector<int64_t> perm;
            if (i == 0 && need_permute_src) {
                // optionally permute src
                perm = get_permutation(ndims, "NXC", "NCX");
            } else if (i == 1 && need_permute_wei) {
                // optionally permute weight
                std::string filter_format
                        = op->get_attr<std::string>(op_attr::weights_format);
                perm = get_permutation(ndims, filter_format, "OIX");
            } else if (i == dw_conv_index && need_permute_post_dw_conv_wei) {
                perm = get_permutation(ndims, "XIO", "OIX");
            } else if (i >= op->num_inputs() - num_post_binary_ops
                                    - with_runtime_dst_scales
                                    - with_runtime_dst_points
                    && need_permute_src) {
                // optionally permute post-binary/post-sum inputs
                perm = get_permutation(ndims, "NXC", "NCX");
            }

            if (!perm.empty()) {
                op_ptr perm_op = std::make_shared<op_t>(op_kind::dnnl_permute);
                perm_op->set_attr<std::vector<int64_t>>(
                        op_attr::permutation, perm);
                rewriter.insert_op_before(perm_op, op, i);
            }
        }

        // remove the attrs in cur_op to avoid re-permute
        op->set_attr<std::string>(op_attr::data_format, "NCX");
        op->set_attr<std::string>(op_attr::weights_format, "OIX");
        if (need_permute_post_dw_conv_wei)
            const_cast<op_t *>(post_dw_conv)
                    ->set_attr<std::string>(op_attr::weights_format, "OIX");

        // permute output back to NXC
        if (need_permute_src) {
            auto ndims = op->get_output_value(0)->get_logical_tensor().ndims;
            auto perm = get_permutation(ndims, "NCX", "NXC");
            op_ptr perm_op = std::make_shared<op_t>(op_kind::dnnl_permute);
            perm_op->set_attr<std::vector<int64_t>>(op_attr::permutation, perm);
            rewriter.insert_op_after(perm_op, op, 0);
        }
    }

    rewriter.run();
    return infer_shape(sg);
}

using io_indices_t = std::vector<size_t>;
std::unordered_map<op_kind_t, std::pair<io_indices_t, io_indices_t>>
        io_idx_to_permute = {
                {op_kind::dnnl_batchnorm, {{0}, {0}}},
                {op_kind::dnnl_prelu, {{0, 1}, {0}}},
                {op_kind::dnnl_prelu_bwd, {{0, 1, 2}, {0, 1}}},
                {op_kind::dnnl_resampling, {{0}, {0}}},
                {op_kind::dnnl_resampling_bwd, {{0, 1}, {0}}},
};

// insert permute for those ops only requiring data_format attribute
// (e.g batchnorm, interpolate)
status_t insert_permute_for_op_only_require_data_format(
        std::shared_ptr<subgraph_t> &sg) {
    subgraph_rewriter_t rewriter(sg);
    const auto &mgr = sg->fusion_info_mgr_;

    for (auto &op : sg->get_ops()) {
        if (!io_idx_to_permute.count(op->get_kind())) continue;

        const bool need_permute = op->has_attr(op_attr::data_format)
                && op->get_attr<std::string>(op_attr::data_format) == "NXC";
        if (!need_permute) continue;

        io_indices_t in_indices = io_idx_to_permute.at(op->get_kind()).first;
        io_indices_t out_indices = io_idx_to_permute.at(op->get_kind()).second;

        // permute explicitly defined inputs
        for (auto idx : in_indices) {
            auto ndims = op->get_input_value(idx)->get_logical_tensor().ndims;
            auto perm = get_permutation(ndims, "NXC", "NCX");
            op_ptr perm_op = std::make_shared<op_t>(op_kind::dnnl_permute);
            perm_op->set_attr<std::vector<int64_t>>(op_attr::permutation, perm);
            rewriter.insert_op_before(perm_op, op, idx);
        }

        fusion_info_t fusion_info;
        if (op->has_attr(op_attr::fusion_info_key)
                && op->get_attr<int64_t>(op_attr::fusion_info_key) != -1) {
            int64_t key = op->get_attr<int64_t>(op_attr::fusion_info_key);
            fusion_info = mgr.get_info(key);
        }

        // permute extra inputs for fused post-binary
        const auto &pops = fusion_info.get_post_ops();
        for (size_t n = 0; n < pops.size(); ++n) {
            if (!pops[n]->is_post_binary() && !pops[n]->is_post_sum()) continue;
            const size_t idx = pops[n]->get_unfused_input_indices()[0];

            auto ndims = op->get_input_value(idx)->get_logical_tensor().ndims;
            auto perm = get_permutation(ndims, "NXC", "NCX");
            op_ptr perm_op = std::make_shared<op_t>(op_kind::dnnl_permute);
            perm_op->set_attr<std::vector<int64_t>>(op_attr::permutation, perm);
            rewriter.insert_op_before(perm_op, op, idx);
        }

        // permute explicitly defined output back to NXC
        for (auto idx : out_indices) {
            auto ndims = op->get_output_value(idx)->get_logical_tensor().ndims;
            auto perm = get_permutation(ndims, "NCX", "NXC");
            op_ptr perm_op = std::make_shared<op_t>(op_kind::dnnl_permute);
            perm_op->set_attr<std::vector<int64_t>>(op_attr::permutation, perm);
            rewriter.insert_op_after(perm_op, op, idx);
        }

        // remove the attrs in cur_op to avoid re-permute
        op->set_attr<std::string>(op_attr::data_format, "NCX");
    }

    rewriter.run();
    return infer_shape(sg);
}

status_t insert_permute_for_shuffle(std::shared_ptr<subgraph_t> &sg) {
    // optimization for NXC case - it will help to hit an optimized kernel
    subgraph_rewriter_t rewriter(sg);

    for (auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() != op_kind::dnnl_shuffle) continue;

        logical_tensor_t src_lt
                = cur_op->get_input_value(0)->get_logical_tensor();
        logical_tensor_t dst_lt
                = cur_op->get_output_value(0)->get_logical_tensor();
        const logical_tensor_wrapper_t src(src_lt), dst(dst_lt);
        const auto axis = cur_op->get_attr<int64_t>(op_attr::axis);
        const auto known_strides
                = (src.is_strided()) ? !src.is_stride_unknown() : false;
        const bool need_permute = axis == src.ndims() - 1 && known_strides
                && src.vstrides() == get_ncx_strides(src.vdims());
        if (!need_permute) continue;

        const int64_t new_axis = 1;
        cur_op->set_attr(op_attr::axis, new_axis);
        op_ptr perm_src_op = std::make_shared<op_t>(op_kind::dnnl_permute);
        auto src_perm = get_permutation(src.ndims(), "NXC", "NCX");
        perm_src_op->set_attr<std::vector<int64_t>>(
                op_attr::permutation, src_perm);
        rewriter.insert_op_before(perm_src_op, cur_op, 0);

        // permute output back to NXC
        op_ptr perm_dst_op = std::make_shared<op_t>(op_kind::dnnl_permute);
        auto dst_perm = get_permutation(dst.ndims(), "NCX", "NXC");
        perm_dst_op->set_attr<std::vector<int64_t>>(
                op_attr::permutation, dst_perm);
        rewriter.insert_op_after(perm_dst_op, cur_op, 0);
    }

    rewriter.run();
    return infer_shape(sg);
}

status_t insert_to_group_for_conv_or_deconv(std::shared_ptr<subgraph_t> &sg) {
    subgraph_rewriter_t rewriter(sg);
    const auto &mgr = sg->fusion_info_mgr_;

    auto insert_to_group = [&](const op_ptr &op, int64_t groups,
                                   const size_t offset) -> bool {
        if (groups <= 1) {
            op->set_attr<bool>(op_attr::canonicalized, true);
            return false;
        }

        op_ptr to_group_op = std::make_shared<op_t>(op_kind::dnnl_to_group);
        to_group_op->set_attr<int64_t>(op_attr::groups, groups);

        op->set_attr<bool>(op_attr::canonicalized, true);
        op->set_attr<int64_t>(op_attr::groups, 1);

        rewriter.insert_op_before(to_group_op, op, offset);

        if (op->get_kind() == op_kind::dnnl_convtranspose
                || op->get_kind() == op_kind::dnnl_convtranspose_bwd_data)
            to_group_op->set_attr<bool>(op_attr::is_convtranspose, true);

        return true;
    };

    for (auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() != op_kind::dnnl_convolution
                && cur_op->get_kind() != op_kind::dnnl_convtranspose
                && cur_op->get_kind() != op_kind::dnnl_convtranspose_bwd_data)
            continue;

        fusion_info_t fusion_info;
        if (cur_op->has_attr(op_attr::fusion_info_key)
                && cur_op->get_attr<int64_t>(op_attr::fusion_info_key) != -1) {
            int64_t key = cur_op->get_attr<int64_t>(op_attr::fusion_info_key);
            fusion_info = mgr.get_info(key);
        }

        if (fusion_info.has_post_dw_conv()) {
            const auto &dw_conv = fusion_info.get_post_dw_conv()->get_op();
            auto dw_conv_groups = dw_conv->get_attr<int64_t>(op_attr::groups);
            const auto inserted = insert_to_group(cur_op, dw_conv_groups, 2);
            if (!inserted) continue;
        }

        auto groups = cur_op->get_attr<int64_t>(op_attr::groups);
        const auto inserted = insert_to_group(cur_op, groups, 1);
        if (!inserted) continue;
    }

    rewriter.run();
    return infer_shape(sg);
}

status_t insert_to_group_for_reorder(std::shared_ptr<subgraph_t> &sg) {
    subgraph_rewriter_t rewriter(sg);

    for (auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() != op_kind::dnnl_reorder) continue;
        auto in_md = make_dnnl_memory_desc(
                cur_op->get_input_value(0)->get_logical_tensor());
        auto out_md = make_dnnl_memory_desc(
                cur_op->get_output_value(0)->get_logical_tensor());
        if (in_md.get_ndims() == out_md.get_ndims()) {
            // no group
            return status::success;
        } else if (in_md.get_ndims() == out_md.get_ndims() + 1) {
            // reorder's input has blocked format with group
            // while output has plain format, perhaps for
            // backward path. No such case for now, disable
            return status::unimplemented;
        } else if (in_md.get_ndims() + 1 == out_md.get_ndims()) {
            // reorder's input has plain format while output
            // has blocked format with group, typically for
            // weight prepacking
            auto group = out_md.get_dims()[0];
            if (group * out_md.get_dims()[1] != in_md.get_dims()[0])
                return status::invalid_shape;

            // insert to_group op
            op_ptr to_group_op = std::make_shared<op_t>(op_kind::dnnl_to_group);
            to_group_op->set_attr<int64_t>(op_attr::groups, group);

            rewriter.insert_op_before(to_group_op, cur_op, 0);
        } else {
            // illegal shape
            return status::invalid_shape;
        }
    }

    rewriter.run();
    return status::success;
}

status_t insert_permute_for_matmul(std::shared_ptr<subgraph_t> &sg) {
    auto &mgr = sg->fusion_info_mgr_;
    subgraph_rewriter_t rewriter(sg);

    for (auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() != op_kind::dnnl_matmul) continue;

        std::vector<bool> trans_flag(2);
        trans_flag[0] = cur_op->has_attr(op_attr::transpose_a)
                && cur_op->get_attr<bool>(op_attr::transpose_a);
        trans_flag[1] = cur_op->has_attr(op_attr::transpose_b)
                && cur_op->get_attr<bool>(op_attr::transpose_b);
        if (!(trans_flag[0] || trans_flag[1])) continue;

        for (size_t i = 0; i < trans_flag.size(); ++i) {
            auto ndims = cur_op->get_input_value(i)->get_logical_tensor().ndims;
            // skip if transpose flag is false or the input's ndim is 1
            if (!trans_flag[i] || ndims <= 1) continue;
            op_ptr permute_op = std::make_shared<op_t>(op_kind::dnnl_permute);
            auto perm = get_last_two_dims_permutation(ndims);
            permute_op->set_attr<std::vector<int64_t>>(
                    op_attr::permutation, perm);
            rewriter.insert_op_before(permute_op, cur_op, i);
            if (cur_op->has_attr(op_attr::fusion_info_key)
                    && cur_op->get_attr<int64_t>(op_attr::fusion_info_key)
                            != -1) {
                int64_t key
                        = cur_op->get_attr<int64_t>(op_attr::fusion_info_key);
                fusion_info_t &fusion_info = mgr.get_mutable_info(key);
                op_t *scales_op = fusion_info.get_mutable_scales(true, i);
                if (scales_op
                        && scales_op->get_attr<std::string>(op_attr::qtype)
                                == "per_channel") {
                    scales_op->set_attr<int64_t>(op_attr::axis, ndims - 1);
                }
            }
        }
        // remove attr to avoid re-transpose during shape inference
        cur_op->set_attr<bool>(op_attr::transpose_a, false);
        cur_op->set_attr<bool>(op_attr::transpose_b, false);
    }

    rewriter.run();
    return infer_shape(sg);
}

status_t insert_reshape_for_ndx2d_matmul(std::shared_ptr<subgraph_t> &sg) {
    auto &mgr = sg->fusion_info_mgr_;

    subgraph_rewriter_t rewriter(sg);

    for (auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() != op_kind::dnnl_matmul) continue;
        // skip due to dnnl cannot reshape such kind of strided memory desc
        if (cur_op->get_input_value(0)->has_producer()
                && cur_op->get_input_value(0)->get_producer().get_kind()
                        == op_kind::dnnl_permute) {
            continue;
        }

        int32_t src_ndims
                = cur_op->get_input_value(0)->get_logical_tensor().ndims;
        int32_t wei_ndims
                = cur_op->get_input_value(1)->get_logical_tensor().ndims;
        if (wei_ndims != 2 || src_ndims <= 2) continue;

        auto src_dims = logical_tensor_wrapper_t(
                cur_op->get_input_value(0)->get_logical_tensor())
                                .vdims();
        dims expected_dims {-1, src_dims.back()};
        auto reshape_op = std::make_shared<op_t>(op_kind::dnnl_reshape);
        reshape_op->set_attr<bool>(op_attr::special_zero, false);
        reshape_op->set_attr<std::vector<int64_t>>(
                op_attr::shape, expected_dims);

        rewriter.insert_op_before(reshape_op, cur_op, 0);

        dims expected_dims2(src_dims);
        expected_dims2[expected_dims2.size() - 1] = 0;
        auto reshape_op2 = std::make_shared<op_t>(op_kind::dnnl_reshape);
        reshape_op2->set_attr<bool>(op_attr::special_zero, true);
        reshape_op2->set_attr<std::vector<int64_t>>(
                op_attr::shape, expected_dims2);
        rewriter.insert_op_after(reshape_op2, cur_op, 0);

        if (cur_op->has_attr(op_attr::fusion_info_key)
                && cur_op->get_attr<int64_t>(op_attr::fusion_info_key) != -1) {
            int64_t key = cur_op->get_attr<int64_t>(op_attr::fusion_info_key);
            fusion_info_t &fusion_info = mgr.get_mutable_info(key);
            const auto &pops = fusion_info.get_post_ops();
            for (size_t i = 0; i < pops.size(); i++) {
                if (!pops[i]->is_post_binary() && !pops[i]->is_post_sum())
                    continue;
                const size_t offset = pops[i]->get_unfused_input_indices()[0];
                auto post_src_dims = logical_tensor_wrapper_t(
                        cur_op->get_input_value(offset)->get_logical_tensor())
                                             .vdims();
                dims expected_dims3 {-1, post_src_dims.back()};
                auto reshape_op3
                        = std::make_shared<op_t>(op_kind::dnnl_reshape);
                reshape_op3->set_attr<bool>(op_attr::special_zero, false);
                reshape_op3->set_attr<std::vector<int64_t>>(
                        op_attr::shape, expected_dims3);
                rewriter.insert_op_before(reshape_op3, cur_op, offset);
            }
            // update weight axis
            op_t *scales_op = fusion_info.get_mutable_scales(true, 1);
            if (scales_op
                    && scales_op->get_attr<std::string>(op_attr::qtype)
                            == "per_channel") {
                scales_op->set_attr<int64_t>(op_attr::axis, 1);
            }
        }
    }

    rewriter.run();
    return infer_shape(sg);
}

status_t insert_unsqueeze_and_squeeze_for_matmul(
        std::shared_ptr<subgraph_t> &sg) {
    subgraph_rewriter_t rewriter(sg);
    auto &mgr = sg->fusion_info_mgr_;

    for (auto &op : sg->get_ops()) {
        if (op->get_kind() != op_kind::dnnl_matmul) continue;

        int32_t src_ndims = op->get_input_value(0)->get_logical_tensor().ndims;
        int32_t wei_ndims = op->get_input_value(1)->get_logical_tensor().ndims;
        assertm(src_ndims >= 1 && wei_ndims >= 1, "invalid dims");

        int32_t unsqueezed_dst_ndims
                = std::max(std::max(src_ndims, wei_ndims), 2);

        std::vector<int64_t> squeeze_axes;
        for (size_t i = 0; i < op->num_inputs(); i++) {
            int32_t ndims = op->get_input_value(i)->get_logical_tensor().ndims;
            std::vector<int64_t> axes;
            if (i == 0 && ndims == 1) {
                // 1D src: [K] -> [1, K]
                axes.emplace_back(-2);
                squeeze_axes.emplace_back(-2);
            } else if (i == 1 && ndims == 1) {
                // 1D weight: [K] -> [K, 1]
                axes.emplace_back(-1);
                squeeze_axes.emplace_back(-1);
            }
            // skip unsqueeze runtime scales
            if (op->get_input_value(i)->has_producer()
                    && impl::utils::one_of(
                            op->get_input_value(i)->get_producer().get_kind(),
                            op_kind::dnnl_constant_scales,
                            op_kind::dnnl_constant_zps))
                continue;

            size_t batch_dim_num = unsqueezed_dst_ndims - axes.size() - ndims;
            for (size_t b = 0; b < batch_dim_num; b++) {
                axes.emplace_back(b);
            }

            if (!axes.empty()) {
                auto unsqueeze_op
                        = std::make_shared<op_t>(op_kind::dnnl_unsqueeze);
                unsqueeze_op->set_attr<std::vector<int64_t>>(
                        op_attr::axes, axes);
                rewriter.insert_op_before(unsqueeze_op, op, i);
            }

            // update the axis
            if (i == 1 && op->has_attr(op_attr::fusion_info_key)
                    && op->get_attr<int64_t>(op_attr::fusion_info_key) != -1) {
                int64_t key = op->get_attr<int64_t>(op_attr::fusion_info_key);
                fusion_info_t &fusion_info = mgr.get_mutable_info(key);
                op_t *scales_op = fusion_info.get_mutable_scales(true, 1);
                if (scales_op
                        && scales_op->get_attr<std::string>(op_attr::qtype)
                                == "per_channel") {
                    scales_op->set_attr<int64_t>(op_attr::axis,
                            unsqueezed_dst_ndims - 1); // the 2nd dim;
                }
            }
        }

        // squeeze dst
        if (!squeeze_axes.empty()) {
            auto squeeze_op = std::make_shared<op_t>(op_kind::dnnl_squeeze);
            squeeze_op->set_attr<std::vector<int64_t>>(
                    op_attr::axes, squeeze_axes);
            rewriter.insert_op_after(squeeze_op, op, 0);
        }
    }

    rewriter.run();
    return infer_shape(sg);
}

impl::status_t insert_runtime_u8_to_s8_for_matmul(
        std::shared_ptr<subgraph_t> &sg) {
    auto &mgr = sg->fusion_info_mgr_;
    subgraph_rewriter_t rewriter(sg);
    for (auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() != op_kind::dnnl_matmul) continue;

        int32_t new_src0_dtype
                = cur_op->get_input_value(0)->get_logical_tensor().data_type;
        int32_t new_src1_dtype
                = cur_op->get_input_value(1)->get_logical_tensor().data_type;
        if (!impl::utils::one_of(
                    new_src0_dtype, impl::data_type::u8, impl::data_type::s8)
                || new_src1_dtype != impl::data_type::u8)
            continue;

        bool with_bias = cur_op->has_attr(op_attr::with_bias)
                && cur_op->get_attr<bool>(op_attr::with_bias);
        const bool has_runtime_src_scales
                = with_runtime_scales(cur_op, mgr, true, 0);
        const bool has_runtime_wei_scales
                = with_runtime_scales(cur_op, mgr, true, 1);
        const bool has_runtime_src_zps = with_runtime_zps(cur_op, mgr, true, 0);
        const bool has_runtime_wei_zps = with_runtime_zps(cur_op, mgr, true, 1);

        size_t index = 2;
        if (with_bias) index += 1;
        if (has_runtime_src_scales) index += 1;
        if (has_runtime_wei_scales) index += 1;
        if (has_runtime_src_zps) index += 1;
        if (has_runtime_wei_zps) {
            if (cur_op->get_input_value(index)->has_producer()
                    && cur_op->get_input_value(index)->get_producer().get_kind()
                            == op_kind::dnnl_constant_zps) {
                auto &const_ops
                        = cur_op->get_input_value(index)->get_producer();
                std::vector<int64_t> current_zp
                        = const_ops.get_attr<std::vector<int64_t>>(
                                op_attr::zps);
                if (current_zp.size() != 1) continue;
                // the equivalent transformation: mm(src_u8, wei_u8) ->
                // mm(src_u8, wei_u8 - 128 + 128) -> mm(src_u8, wei_s8 + 128),
                // which wei_s8 = wei_u8 - 128
                std::vector<int64_t> adjusted_zp {current_zp[0] - 128};
                const_ops.set_attr<std::vector<int64_t>>(
                        op_attr::zps, adjusted_zp);
            } else {
                // add a binary add here.
            }
        } else {
            assertm(cur_op->num_inputs() == index,
                    "only support insert input at the end of inputs");
            std::vector<int64_t> zp {-128};
            auto zps_op = std::make_shared<op_t>(op_kind::dnnl_add_zps);
            zps_op->set_attr<std::string>(op_attr::qtype, "per_tensor");
            zps_op->set_attr<int64_t>(op_attr::axis, 0);
            zps_op->set_attr(op_attr::with_runtime_zps, true);
            op_ptr const_data_op;
            const_data_op = std::make_shared<op_t>(op_kind::dnnl_constant_zps);
            const_data_op->set_attr(op_attr::zps, zp);
            std::vector<int64_t> dst_shape(1, zp.size());
            const_data_op->set_attr(op_attr::shape, dst_shape);
            logical_tensor_t const_data_dst_lt
                    = empty_logical_tensor_with_default_id();
            auto const_data_dst_value = std::make_shared<value_t>(
                    *const_data_op, 0, const_data_dst_lt, true);
            const_data_dst_value->set_data_type(graph::data_type::s32);
            const_data_dst_value->set_layout_type(layout_type::strided);
            const_data_dst_value->set_strides({1});
            const_data_op->add_output(const_data_dst_value);

            int64_t key = -1;
            if (cur_op->has_attr(op_attr::fusion_info_key)) {
                key = cur_op->get_attr<int64_t>(op_attr::fusion_info_key);
            } else {
                key = mgr.init_info();
                cur_op->set_attr<int64_t>(op_attr::fusion_info_key, key);
            }

            fusion_info_t &fusion_info = mgr.get_mutable_info(key);
            fusion_info.set_zero_points(zps_op->shared_from_this(), true, 1);

            // connect add_zp and constant data
            cur_op->add_input(const_data_dst_value);
            const_data_dst_value->add_consumer(*cur_op, cur_op->num_inputs());
            rewriter.to_insert(const_data_op);
        }

        op_ptr u8_to_s8_op = std::make_shared<op_t>(op_kind::dnnl_reorder);

        // make zps as a constant input
        op_ptr const_zps_op;
        const_zps_op = std::make_shared<op_t>(op_kind::dnnl_constant_zps);
        const_zps_op->set_attr(op_attr::zps, std::vector<int64_t> {-128});
        const_zps_op->set_attr(op_attr::shape, std::vector<int64_t> {1});
        logical_tensor_t const_zps_dst_lt
                = empty_logical_tensor_with_default_id();
        auto const_zps_dst_value = std::make_shared<value_t>(
                *const_zps_op, 0, const_zps_dst_lt, true);
        const_zps_dst_value->set_data_type(graph::data_type::s32);
        const_zps_dst_value->set_layout_type(layout_type::strided);
        const_zps_dst_value->set_strides({1});
        const_zps_op->add_output(const_zps_dst_value);
        u8_to_s8_op->set_attr(op_attr::with_runtime_dst_zps, true);
        rewriter.insert_op_before(u8_to_s8_op, cur_op, 1);
        u8_to_s8_op->connect_input(1, const_zps_dst_value);
        u8_to_s8_op->get_output_value(0)->set_data_type(graph::data_type::s8);
        insert_empty_scratchpad(u8_to_s8_op);
        rewriter.to_insert(const_zps_op);
    }

    rewriter.run();
    return infer_shape(sg);
}

status_t insert_u8_to_s8_for_matmul(std::shared_ptr<subgraph_t> &sg) {
    auto &mgr = sg->fusion_info_mgr_;

    subgraph_rewriter_t rewriter(sg);

    for (auto &cur_op : sg->get_ops()) {
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
        if (cur_op->has_attr(op_attr::fusion_info_key)
                && cur_op->get_attr<int64_t>(op_attr::fusion_info_key) != -1) {
            key = cur_op->get_attr<int64_t>(op_attr::fusion_info_key);
        } else {
            key = mgr.init_info();
            cur_op->set_attr<int64_t>(op_attr::fusion_info_key, key);
        }
        fusion_info_t &fusion_info = mgr.get_mutable_info(key);
        op_t *wei_zps_op = fusion_info.get_mutable_zero_points(
                true, /*the wei index*/ 1);
        if (wei_zps_op) { // already fused zps, update the zps
            std::vector<int64_t> current_zp
                    = wei_zps_op->get_attr<std::vector<int64_t>>(op_attr::zps);
            if (current_zp.size() != 1) continue;
            // the equivalent transformation: mm(src_u8, wei_u8) -> mm(src_u8,
            // wei_u8 - 128 + 128) -> mm(src_u8, wei_s8 + 128), which wei_s8 =
            // wei_u8 - 128. since the weight zps is actually subtracted in
            // primitive side, so the wei_s8 + 128 should be then converted to
            // wei_s8 - (-128)
            std::vector<int64_t> adjusted_zp {current_zp[0] - 128};
            wei_zps_op->set_attr<std::vector<int64_t>>(
                    op_attr::zps, adjusted_zp);
        } else { // fuse a 128 zps
            std::vector<int64_t> zp {-128};
            auto zps_op = std::make_shared<op_t>(op_kind::dnnl_add_zps);
            zps_op->set_attr<std::string>(op_attr::qtype, "per_tensor");
            zps_op->set_attr<int64_t>(op_attr::axis, 0);
            zps_op->set_attr<std::vector<int64_t>>(op_attr::zps, zp);
            fusion_info.set_zero_points(zps_op, true, /*the wei index*/ 1);
        }

        op_ptr u8_to_s8_op = std::make_shared<op_t>(op_kind::dnnl_reorder);
        u8_to_s8_op->set_attr<std::vector<int64_t>>(
                op_attr::dst_zps, std::vector<int64_t> {-128});
        rewriter.insert_op_before(u8_to_s8_op, cur_op, 1);
        u8_to_s8_op->get_output_value(0)->set_data_type(graph::data_type::s8);
        insert_empty_scratchpad(u8_to_s8_op);
    }

    rewriter.run();
    return infer_shape(sg);
}

status_t insert_unsqueeze_for_prelu(std::shared_ptr<subgraph_t> &sg) {
    using ltw = logical_tensor_wrapper_t;

    subgraph_rewriter_t rewriter(sg);

    for (auto &cur_op : sg->get_ops()) {
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
        // insert unsqueeze op
        int32_t src_ndims = src_lt.ndims;
        int32_t wei_ndims = wei_lt.ndims;
        // we only broadcast wei dims
        if (wei_ndims != src_ndims) {
            std::vector<int64_t> axes(src_ndims - wei_ndims);
            std::iota(axes.begin(), axes.end(), 0);

            // Only for NCX format per_channel broadcast PReLU, we need to
            // unsqueeze the 1D weight to [1, C, 1, 1]
            const bool channel_first
                    = data_format == "NCX" && per_channel_broadcast;
            if (channel_first && axes.size() >= 2) {
                axes.erase(axes.begin() + 1);
                axes.emplace_back(-1);
            }

            auto unsqueeze_op = std::make_shared<op_t>(op_kind::dnnl_unsqueeze);
            unsqueeze_op->set_attr<std::vector<int64_t>>(op_attr::axes, axes);
            int wei_id = 1; // weight is the second input
            rewriter.insert_op_before(unsqueeze_op, cur_op, wei_id);
        }
    }

    rewriter.run();
    return infer_shape(sg);
}

status_t insert_unsqueeze_and_squeeze_for_prelu_bwd(
        std::shared_ptr<subgraph_t> &sg) {
    using ltw = logical_tensor_wrapper_t;

    subgraph_rewriter_t rewriter(sg);

    for (auto &cur_op : sg->get_ops()) {
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
        // insert unsqueeze op
        int32_t src_ndims = src_lt.ndims;
        int32_t wei_ndims = wei_lt.ndims;
        // we only broadcast wei dims
        if (wei_ndims != src_ndims) {
            std::vector<int64_t> axes(src_ndims - wei_ndims);
            std::iota(axes.begin(), axes.end(), 0);

            // Only for NCX format per_channel broadcast PReLU, we need to
            // unsqueeze the 1D weight to [1, C, 1, 1]
            const bool channel_first
                    = data_format == "NCX" && per_channel_broadcast;
            if (channel_first && axes.size() >= 2) {
                axes.erase(axes.begin() + 1);
                axes.emplace_back(-1);
            }

            auto unsqueeze_op = std::make_shared<op_t>(op_kind::dnnl_unsqueeze);
            unsqueeze_op->set_attr<std::vector<int64_t>>(op_attr::axes, axes);
            int wei_id = 1; // weight is the second input
            rewriter.insert_op_before(unsqueeze_op, cur_op, wei_id);

            // the squeeze is exactly the inverse of unsqueeze, so they use same
            // axes
            std::vector<int64_t> squeeze_axes = axes;
            op_ptr squeeze_op = std::make_shared<op_t>(op_kind::dnnl_squeeze);
            squeeze_op->set_attr<std::vector<int64_t>>(
                    op_attr::axes, squeeze_axes);
            // Insert squeeze after diff weights, so that its dimensions
            // have their original shape.
            rewriter.insert_op_after(squeeze_op, cur_op, 1);
        }
    }

    rewriter.run();
    return infer_shape(sg);
}

status_t insert_unsqueeze_and_squeeze_for_reduction(
        std::shared_ptr<subgraph_t> &sg) {
    subgraph_rewriter_t rewriter(sg);

    for (auto &cur_op : sg->get_ops()) {
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
            // insert unsqueeze op before binary's src1 input
            if (post_op.get_kind() == op_kind::dnnl_binary) {
                if (!post_binary_fusible(cur_op.get(), &post_op)) break;
                op_ptr unsqueeze_op
                        = std::make_shared<op_t>(op_kind::dnnl_unsqueeze);
                unsqueeze_op->set_attr<std::vector<int64_t>>(
                        op_attr::axes, axes);
                rewriter.insert_op_before(
                        unsqueeze_op, post_op.shared_from_this(), src1_offset);
            }

            // clear the existing shape for cur_op_ptr output (which is post-op
            // input value), so later correct shape will be inferred
            cur_op_ptr->get_output_value(0)->set_ndims(-1);
            cur_op_ptr = &post_op;
        }

        op_ptr squeeze_op = std::make_shared<op_t>(op_kind::dnnl_squeeze);
        squeeze_op->set_attr<std::vector<int64_t>>(op_attr::axes, axes);
        // insert squeeze op after reduction or after its last post-op
        rewriter.insert_op_after(squeeze_op, cur_op_ptr->shared_from_this(), 0);

        // set to true, as squeeze will be handled by separate op
        cur_op->set_attr(op_attr::keep_dims, true);
    }

    rewriter.run();
    return infer_shape(sg);
}

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
