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
#include <string>
#include <vector>

#include "dnnl.hpp"

#include "interface/c_types_map.hpp"
#include "interface/value.hpp"

#include "backend/dnnl/common.hpp"
#include "backend/dnnl/passes/op_executable.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {
using op_t = impl::op_t;
using op_ptr = std::shared_ptr<impl::op_t>;
using value_ptr = std::shared_ptr<impl::value_t>;
using ltw = impl::logical_tensor_wrapper_t;

static value_ptr insert_scratchpad(op_ptr &op) {
    logical_tensor_t lt = impl::empty_logical_tensor_with_default_id();
    value_ptr scratchpad_val
            = std::make_shared<value_t>(*op, op->num_outputs(), lt);
    op->add_output(scratchpad_val);
    return scratchpad_val;
}

static value_ptr insert_workspace(op_ptr &op) {
    logical_tensor_t lt = impl::empty_logical_tensor_with_default_id();
    value_ptr workspace_val
            = std::make_shared<value_t>(*op, op->num_outputs(), lt);
    op->add_output(workspace_val);
    return workspace_val;
}

static bool layout_propagation_for_conv(op_ptr &op,
        const dnnl::engine &p_engine, primitive_attr_mgr_t &prm_attr_mgr,
        pd_cache_t &pd_cache) {
    std::shared_ptr<impl::value_t> src, wei, bias, dst;
    src = op->get_input_value(0);
    wei = op->get_input_value(1);
    dst = op->get_output_value(0);
    if (op->has_attr("with_bias") && op->get_attr<bool>("with_bias")) {
        bias = op->get_input_value(2);
    }

    assertm(ltw(src->get_logical_tensor()).is_any()
                    && ltw(wei->get_logical_tensor()).is_any()
                    && ltw(dst->get_logical_tensor()).is_any(),
            "conv's src, weight, dst should be any layout_type");

    auto pd = create_conv_pd(op, p_engine, prm_attr_mgr, pd_cache);

    fill_layout_info(src, pd.src_desc());
    fill_layout_info(wei, pd.weights_desc());
    if (op->has_attr("with_bias") && op->get_attr<bool>("with_bias")) {
        fill_layout_info(bias, pd.bias_desc());
    }

    fill_layout_info(dst, pd.dst_desc());

    // fill scratchpads dimensions and data type to scratchpad value_t
    auto scratchpad_val = insert_scratchpad(op);
    const memory::desc scratchpad_desc = pd.scratchpad_desc();
    const memory::dims dims = scratchpad_desc.dims();
    scratchpad_val->set_dims(dims);
    scratchpad_val->set_data_type(
            static_cast<impl::data_type_t>(scratchpad_desc.data_type()));

    // make scratchpad as conv's last output
    fill_layout_info(scratchpad_val, scratchpad_desc);

    return true;
}

static bool layout_propagation_for_deconv(op_ptr &op,
        const dnnl::engine &p_engine, primitive_attr_mgr_t &prm_attr_mgr,
        pd_cache_t &pd_cache) {
    std::shared_ptr<impl::value_t> src, wei, bias, dst;
    src = op->get_input_value(0);
    wei = op->get_input_value(1);
    dst = op->get_output_value(0);
    if (op->has_attr("with_bias") && op->get_attr<bool>("with_bias")) {
        bias = op->get_input_value(2);
    }

    assertm(ltw(src->get_logical_tensor()).is_any()
                    && ltw(wei->get_logical_tensor()).is_any()
                    && ltw(dst->get_logical_tensor()).is_any(),
            "conv's src, weight, dst should be any layout_type");

    auto pd = create_deconv_pd(op, p_engine, prm_attr_mgr, pd_cache);

    fill_layout_info(src, pd.src_desc());
    fill_layout_info(wei, pd.weights_desc());
    if (op->has_attr("with_bias") && op->get_attr<bool>("with_bias")) {
        fill_layout_info(bias, pd.bias_desc());
    }

    fill_layout_info(dst, pd.dst_desc());

    // fill scratchpads dimensions and data type to scratchpad value_t
    auto scratchpad_val = insert_scratchpad(op);
    const memory::desc scratchpad_desc = pd.scratchpad_desc();
    const memory::dims dims = scratchpad_desc.dims();
    scratchpad_val->set_dims(dims);
    scratchpad_val->set_data_type(
            static_cast<impl::data_type_t>(scratchpad_desc.data_type()));

    // make scratchpad as conv's last output
    fill_layout_info(scratchpad_val, scratchpad_desc);

    return true;
}

static void get_expected_input_layout(
        std::shared_ptr<impl::value_t> &input_value) {
    if (!ltw(input_value->get_logical_tensor()).is_constant()
            && input_value->has_producer()
            && input_value->get_producer().get_kind()
                    == impl::op_kind::Reorder) {
        op_t &producer = input_value->get_producer();
        std::shared_ptr<value_t> producer_input_value
                = producer.get_input_value(0);
        logical_tensor_t producer_input_lt
                = producer_input_value->get_logical_tensor();
        if (producer_input_lt.layout_type != layout_type::any
                && producer_input_lt.layout_type != layout_type::undef) {
            impl::logical_tensor_t temp_lt
                    = producer_input_value->get_logical_tensor();
            // replace the input with the input of reorder but keep the id
            temp_lt.id = input_value->get_logical_tensor().id;
            input_value->set_logical_tensor(temp_lt);
        }
    }
}

static bool layout_propagation_for_matmul(op_ptr &op,
        const dnnl::engine &p_engine, primitive_attr_mgr_t &prm_attr_mgr,
        pd_cache_t &pd_cache) {
    std::shared_ptr<impl::value_t> src, wei, bias, dst;
    src = op->get_input_value(0);
    // get reorder's input value as matmul's input value if not constant
    get_expected_input_layout(src);
    wei = op->get_input_value(1);
    get_expected_input_layout(wei);

    dst = op->get_output_value(0);
    if (op->has_attr("with_bias") && op->get_attr<bool>("with_bias")) {
        bias = op->get_input_value(2);
        get_expected_input_layout(bias);
    }

    auto pd = create_matmul_pd(op, p_engine, prm_attr_mgr, pd_cache);

    fill_layout_info(src, pd.src_desc());
    fill_layout_info(wei, pd.weights_desc());
    if (op->has_attr("with_bias") && op->get_attr<bool>("with_bias"))
        fill_layout_info(bias, pd.bias_desc());
    fill_layout_info(dst, pd.dst_desc());

    // fill scratchpads dimensions and data type to scratchpad value_t
    auto scratchpad_val = insert_scratchpad(op);
    const memory::desc scratchpad_desc = pd.scratchpad_desc();
    const memory::dims dims = scratchpad_desc.dims();
    scratchpad_val->set_dims(dims);
    scratchpad_val->set_data_type(
            static_cast<impl::data_type_t>(scratchpad_desc.data_type()));

    // make scratchpad as matmul's last output
    fill_layout_info(scratchpad_val, scratchpad_desc);

    return false;
}

static bool layout_propagation_for_pool(op_ptr &op,
        const dnnl::engine &p_engine, primitive_attr_mgr_t &prm_attr_mgr,
        pd_cache_t &pd_cache) {
    bool changed = true;
    value_ptr src = op->get_input_value(0);
    value_ptr dst = op->get_output_value(0);

    // for pooling primitive, src's format must be defined. dst's format should
    // be any to obtain better performance
    if (!ltw(src->get_logical_tensor()).is_any()
            && ltw(dst->get_logical_tensor()).is_any()) {
        auto pd = create_pool_pd(op, p_engine, prm_attr_mgr, pd_cache);

        fill_layout_info(src, pd.src_desc());
        fill_layout_info(dst, pd.dst_desc());

        // make scratchpad as pool's last output
        value_ptr scratchpad_val = insert_scratchpad(op);
        fill_layout_info(scratchpad_val, pd.scratchpad_desc());
        // if pooling's prop_kind id forward_training or backward
        if (op->has_attr("is_training") && op->get_attr<bool>("is_training")) {
            value_ptr workspace_val = insert_workspace(op);
            fill_layout_info(workspace_val, pd.workspace_desc());
        }
    } else {
        changed = false;
    }

    return changed;
}

static bool layout_propagation_for_permute(op_ptr &op) {
    bool changed = true;

    std::shared_ptr<impl::value_t> src, dst;
    src = op->get_input_value(0);
    dst = op->get_output_value(0);

    auto in_lt = src->get_logical_tensor();
    auto out_lt = dst->get_logical_tensor();

    if (!ltw(in_lt).is_any() && ltw(out_lt).is_any()) {
        dnnl::memory::desc in_md = make_dnnl_memory_desc(in_lt);
        dnnl::memory::desc out_md;

        auto permute_kind = op->get_attr<std::string>("permute_kind");
        if (permute_kind == "transpose") {
            // transpose the right-most two dims
            out_md = permute_last_two_dims(in_md);
        } else {
            auto from_format = op->get_attr<std::string>("from_format");
            auto to_format = op->get_attr<std::string>("to_format");
            if (from_format == "NCX" && to_format == "NXC") {
                out_md = permute_NCX2NXC(in_md);
            } else if (from_format == "NXC" && to_format == "NCX") {
                out_md = permute_NXC2NCX(in_md);
            } else if (from_format == "XIO" && to_format == "OIX") {
                out_md = permute_XIO2OIX(in_md);
            } else {
                assertm(false, "not a supported permutation");
            }
        }

        fill_layout_info(dst, out_md);
    } else if (!ltw(out_lt).is_any() && ltw(in_lt).is_any()) {
        dnnl::memory::desc out_md = make_dnnl_memory_desc(out_lt);
        dnnl::memory::desc in_md;

        auto permute_kind = op->get_attr<std::string>("permute_kind");
        if (permute_kind == "transpose") {
            // transpose the right-most two dims
            in_md = permute_last_two_dims(out_md);
        } else {
            auto from_format = op->get_attr<std::string>("from_format");
            auto to_format = op->get_attr<std::string>("to_format");
            if (from_format == "NCX" && to_format == "NXC") {
                in_md = permute_NXC2NCX(out_md);
            } else if (from_format == "NXC" && to_format == "NCX") {
                in_md = permute_NCX2NXC(out_md);
            } else {
                assertm(false, "not a supported permutation");
            }
        }

        fill_layout_info(src, in_md);
    } else {
        changed = false;
    }

    return changed;
}

static bool layout_propagation_for_to_group(op_ptr &op) {
    bool changed = true;
    std::shared_ptr<impl::value_t> src, dst;
    src = op->get_input_value(0);
    dst = op->get_output_value(0);
    auto in_lt = src->get_logical_tensor();
    auto out_lt = dst->get_logical_tensor();

    if (!ltw(in_lt).is_any() && ltw(out_lt).is_any()) {
        dnnl::memory::desc in_md = make_dnnl_memory_desc(in_lt);
        dnnl::memory::desc out_md;
        auto groups = op->get_attr<int64_t>("groups");
        if (op->has_attr("is_convtranspose")
                && op->get_attr<bool>("is_convtranspose")) {
            auto permuted_weight = transpose(in_md, 0, 1);
            auto permuted_group_weight = to_grouped(permuted_weight, groups);
            out_md = transpose(permuted_group_weight, 1, 2);
        } else {
            out_md = to_grouped(in_md, groups);
        }
        fill_layout_info(dst, out_md);
    } else {
        changed = false;
    }

    return changed;
}

static bool layout_propagation_for_expand(op_ptr &op) {
    bool changed = true;
    std::shared_ptr<impl::value_t> src, dst;
    src = op->get_input_value(0);
    dst = op->get_output_value(0);
    auto in_lt = src->get_logical_tensor();
    auto out_lt = dst->get_logical_tensor();

    if (!ltw(in_lt).is_any() && ltw(out_lt).is_any()) {
        dnnl::memory::desc in_md = make_dnnl_memory_desc(in_lt);
        dnnl::memory::desc out_md = in_md;

        if (op->has_attr("insert_1dim")) {
            auto insert_1dim = op->get_attr<std::string>("insert_1dim");
            if (insert_1dim == "before") {
                out_md = in_md.reshape({1, in_md.dims()[0]});
            } else if (insert_1dim == "after") {
                out_md = in_md.reshape({in_md.dims()[0], 1});
            }
        }

        if (op->has_attr("expand_to")) {
            auto expand_to_ndims = op->get_attr<int64_t>("expand_to");
            if (expand_to_ndims != -1) {
                out_md = expand(out_md, static_cast<int>(expand_to_ndims));
            }
        }

        fill_layout_info(dst, out_md);

    } else {
        changed = false;
    }

    return changed;
}

static bool layout_propagation_for_squeeze(op_ptr &op) {
    bool changed = true;
    std::shared_ptr<impl::value_t> src, dst;
    src = op->get_input_value(0);
    dst = op->get_output_value(0);
    auto in_lt = src->get_logical_tensor();
    auto out_lt = dst->get_logical_tensor();

    if ((ltw(in_lt).is_strided() || ltw(in_lt).is_opaque())
            && (!ltw(out_lt).is_strided() && !ltw(out_lt).is_opaque())) {
        dnnl::memory::desc in_md = make_dnnl_memory_desc(in_lt);
        dnnl::memory::desc out_md = in_md;

        if (op->has_attr("axes")) {
            auto axes = op->get_attr<std::vector<int64_t>>("axes");
            auto in_ndim = in_md.dims().size();

            if (axes.empty() || axes.size() == 2) {
                // the output is a scalar or no need to squeeze,
                // just skip for layout propagation
            } else {
                assertm(axes.size() == 1,
                        "the size of axes in squeeze op "
                        "should be 1.");
                auto reshaped_dims = in_md.dims();
                if (axes.back() == -1) {
                    reshaped_dims.erase(reshaped_dims.begin() + in_ndim - 1);
                } else if (axes.back() == -2) {
                    reshaped_dims.erase(reshaped_dims.begin() + in_ndim - 2);
                }
                out_md = in_md.reshape(reshaped_dims);
            }
        }

        fill_layout_info(dst, out_md);
    } else {
        changed = false;
    }

    return changed;
}

static bool layout_propagation_for_reorder(op_ptr &op) {
    bool changed = true;
    std::shared_ptr<impl::value_t> src, dst;
    src = op->get_input_value(0);
    dst = op->get_output_value(0);
    auto in_lt = src->get_logical_tensor();
    auto out_lt = dst->get_logical_tensor();

    if (!ltw(in_lt).is_any() && ltw(out_lt).is_any()) {
        auto out_md = make_dnnl_memory_desc(in_lt);
        out_md.data.data_type
                = static_cast<dnnl_data_type_t>(ltw(out_lt).data_type());
        fill_layout_info(dst, out_md);
    } else if (!ltw(out_lt).is_any() && ltw(in_lt).is_any()) {
        auto in_md = make_dnnl_memory_desc(out_lt);
        in_md.data.data_type
                = static_cast<dnnl_data_type_t>(ltw(in_lt).data_type());
        fill_layout_info(src, in_md);
    } else {
        changed = false;
    }
    return changed;
}

static bool layout_propagation_for_mul_scales(op_ptr &op) {
    return layout_propagation_for_reorder(op);
}

static bool layout_propagation_for_bn_folding(
        op_ptr &op, const dnnl::engine &p_engine) {
    bool changed = false;

    for (size_t i = 0; i < op->num_outputs(); i++) {
        auto in_lt = op->get_input_value(i)->get_logical_tensor();
        auto out_lt = op->get_output_value(i)->get_logical_tensor();
        if (!ltw(in_lt).is_any() && ltw(out_lt).is_any()) {
            dnnl::memory::desc in_md = make_dnnl_memory_desc(in_lt);
            auto dst = op->get_output_value(i);
            fill_layout_info(dst, in_md);
            changed = true;
        }
    }

    auto prm = std::make_shared<bn_folding_t>(op, p_engine);
    // make scratchpad as bn_folding's last inputs
    auto val = insert_scratchpad(op);
    fill_layout_info(
            val, dynamic_cast<bn_folding_t *>(prm.get())->scratchpad_desc());

    return changed;
}

static bool layout_propagation_for_conv_bwd_data(op_ptr &op,
        const dnnl::engine &p_engine, primitive_attr_mgr_t &prm_attr_mgr,
        pd_cache_t &pd_cache) {
    std::shared_ptr<impl::value_t> diff_dst, wei, bias, diff_src;
    diff_dst = op->get_input_value(0);
    wei = op->get_input_value(1);
    diff_src = op->get_output_value(0);

    assertm(ltw(diff_dst->get_logical_tensor()).is_any()
                    && ltw(wei->get_logical_tensor()).is_any()
                    && ltw(diff_src->get_logical_tensor()).is_any(),
            "conv_bwd_data's diff_dst, weight, diff_src should be any "
            "layout_type");

    auto pd = create_conv_bwd_data_pd(op, p_engine, prm_attr_mgr, pd_cache);

    fill_layout_info(diff_dst, pd.diff_dst_desc());
    fill_layout_info(wei, pd.weights_desc());
    fill_layout_info(diff_src, pd.diff_src_desc());

    // fill scratchpads dimensions and data type to scratchpad value_t
    auto scratchpad_val = insert_scratchpad(op);
    const memory::desc scratchpad_desc = pd.scratchpad_desc();
    const memory::dims dims = scratchpad_desc.dims();
    scratchpad_val->set_dims(dims);
    scratchpad_val->set_data_type(
            static_cast<impl::data_type_t>(scratchpad_desc.data_type()));

    // make scratchpad as conv's last output
    fill_layout_info(scratchpad_val, scratchpad_desc);

    return true;
}

static void remove_unnecessary_reorder(std::vector<op_ptr> &subgraph) {
    std::vector<op_t *> fuse_to_precursor;
    std::vector<op_t *> fuse_to_successor;
    for (auto &cur_op : subgraph) {
        if (cur_op->get_kind() != impl::op_kind::Reorder) continue;

        auto in_lt = cur_op->get_input_value(0)->get_logical_tensor();
        auto out_lt = cur_op->get_output_value(0)->get_logical_tensor();

        auto in_md = make_dnnl_memory_desc(in_lt);
        auto out_md = make_dnnl_memory_desc(out_lt);
        if (in_md == out_md) {
            if (out_lt.id != std::numeric_limits<size_t>::max()) {
                // the out_lt is given by user, it should be reserved
                fuse_to_precursor.emplace_back(cur_op.get());
            } else {
                // the in_lt is given by user, it should be reserved
                fuse_to_successor.emplace_back(cur_op.get());
            }
        }
    }

    for (auto &op : fuse_to_precursor) {
        fuse_op_to_predecessor(op, subgraph);
    }

    for (auto &op : fuse_to_successor) {
        fuse_op_to_successor(op, subgraph);
    }
}

/// This function is used to chooses optimal layout for computation bound op and
/// propagate the chosen optimal layout and given in/outputs layout in the
/// subgraph.
///
/// The workflow of layout propagation is:
/// Step1: choose the optimal in/outputs layout for computation bound op. We
/// need to create dnnl primitive descriptor in this step, so we need to do
/// infer shape and infer type first.
/// Step2: propagate layout for other ops except for reorder op by calling each
/// op's layout propagation function
/// Step3: propagate layout for reorder op by calling reorder op's layout
/// propagation function
/// Step4: go back to Step2 until the graph is not changed
/// Step5: remove unnecessary reorder op, whose input and output have same
/// layout
///
/// \note The layout propagation function for each op should be bidirectional to
/// support propagating layout both from inputs to outputs and from outputs to
/// inputs. See the following figure for example:
impl::status_t layout_propagation(std::vector<op_ptr> &subgraph,
        const dnnl::engine &p_engine, primitive_attr_mgr_t &prm_attr_mgr,
        pd_cache_t &pd_cache) {
    auto need_prop = [&](op_t *op) {
        for (const auto &in : op->get_input_values()) {
            if (ltw(in->get_logical_tensor()).layout_type()
                    == layout_type::any) {
                return true;
            }
        }
        for (const auto &out : op->get_output_values()) {
            if (ltw(out->get_logical_tensor()).layout_type()
                    == layout_type::any) {
                return true;
            }
        }
        return false;
    };
    // lambda function to do layout propagation for non-computation-intensive
    // and non-layout-reorder ops. If no layout is changed, this function will
    // return false.
    auto layout_propagation_func = [&](std::vector<op_ptr> &subgraph) -> bool {
        bool changed = false;
        for (auto &cur_op : subgraph) {
            if (!need_prop(cur_op.get())) continue;

            if (cur_op->get_kind() == impl::op_kind::Convolution
                    || cur_op->get_kind() == op_kind::dnnl_convolution
                    || cur_op->get_kind() == impl::op_kind::MatMul
                    || cur_op->get_kind() == op_kind::dnnl_conv_bwd_data)
                continue;

            if (cur_op->get_kind() == impl::op_kind::Reorder
                    && cur_op->has_attr("change_layout")
                    && cur_op->get_attr<bool>("change_layout"))
                continue;

            if (cur_op->get_kind() == impl::op_kind::MaxPool
                    || cur_op->get_kind() == impl::op_kind::AvgPool
                    || cur_op->get_kind() == op_kind::dnnl_pool) {
                changed = layout_propagation_for_pool(
                                  cur_op, p_engine, prm_attr_mgr, pd_cache)
                        || changed;
            } else if (cur_op->get_kind() == op_kind::permute) {
                changed = layout_propagation_for_permute(cur_op) || changed;
            } else if (cur_op->get_kind() == op_kind::mul_scales) {
                changed = layout_propagation_for_mul_scales(cur_op) || changed;
            } else if (cur_op->get_kind() == op_kind::to_group) {
                changed = layout_propagation_for_to_group(cur_op) || changed;
            } else if (cur_op->get_kind() == op_kind::expand) {
                changed = layout_propagation_for_expand(cur_op) || changed;
            } else if (cur_op->get_kind() == impl::op_kind::Reorder
                    || cur_op->get_kind() == op_kind::dnnl_u8_to_s8) {
                changed = layout_propagation_for_reorder(cur_op) || changed;
            } else if (cur_op->get_kind() == op_kind::squeeze) {
                changed = layout_propagation_for_squeeze(cur_op) || changed;
            } else if (cur_op->get_kind() == op_kind::dnnl_bn_folding) {
                changed |= layout_propagation_for_bn_folding(cur_op, p_engine);
            } else {
                assertm(false,
                        "none layout propagation function for current op");
            }
        }
        return changed;
    };

    // we need to choose optimal layout for computation-intensive ops first
    for (auto &cur_op : subgraph) {
        if (!need_prop(cur_op.get())) continue;

        if (cur_op->get_kind() == impl::op_kind::Convolution
                || cur_op->get_kind() == op_kind::dnnl_convolution) {
            layout_propagation_for_conv(
                    cur_op, p_engine, prm_attr_mgr, pd_cache);
        } else if (cur_op->get_kind() == op_kind::dnnl_conv_bwd_data) {
            layout_propagation_for_conv_bwd_data(
                    cur_op, p_engine, prm_attr_mgr, pd_cache);
        } else if (cur_op->get_kind() == impl::op_kind::ConvTranspose
                || cur_op->get_kind() == op_kind::dnnl_convtranspose) {
            layout_propagation_for_deconv(
                    cur_op, p_engine, prm_attr_mgr, pd_cache);
        } else {
            // do nothing
        }
    }

    // then, we can propagate the layout from conv and in/outputs to other
    // values
    int cnt = 0;
    const int max_num_limit = static_cast<int>(subgraph.size());

    bool changed = true;
    do {
        changed = layout_propagation_func(subgraph);

        // layout propagation for matmul
        for (auto &cur_op : subgraph) {
            if (cur_op->get_kind() != impl::op_kind::MatMul) continue;
            if (!need_prop(cur_op.get())) continue;
            changed = layout_propagation_for_matmul(
                              cur_op, p_engine, prm_attr_mgr, pd_cache)
                    || changed;
        }
        // layout propagation for layout reorder op
        for (auto &cur_op : subgraph) {
            if (cur_op->get_kind() != impl::op_kind::Reorder
                    || (cur_op->has_attr("change_layout")
                            && !cur_op->get_attr<bool>("change_layout")))
                continue;
            if (!need_prop(cur_op.get())) continue;
            changed = layout_propagation_for_reorder(cur_op) || changed;
        }
        cnt++;
    } while (changed && cnt <= max_num_limit);

    assertm(cnt <= max_num_limit, "Failed to propagate layout for all ops");
    if (cnt > max_num_limit) return impl::status::unsupported;

    remove_unnecessary_reorder(subgraph);

    return impl::status::success;
}

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl
