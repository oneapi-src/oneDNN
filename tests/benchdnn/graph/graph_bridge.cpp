/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "graph_bridge.hpp"

namespace graph {
const std::unordered_set<std::string> &get_special_backward_op_kind_set() {
    static const std::unordered_set<std::string> set_ = {
            // bnorm backward
            "BatchNormTrainingBackward",
            // eltwise backward
            "AbsBackward",
            "ClampBackward",
            "EluBackward",
            "GELUBackward",
            "HardSigmoidBackward",
            "HardSwishBackward",
            "MishBackward",
            "ReLUBackward",
            "SigmoidBackward",
            "SoftPlusBackward",
            "SqrtBackward",
            "TanhBackward",
            // pool backward
            "AvgPoolBackward",
            "MaxPoolBackward",
    };
    return set_;
}

const std::unordered_map<size_t, data_kind_t> &get_dnnl_arg_2_data_kind_map() {
    static const std::unordered_map<size_t, data_kind_t> map_ {
            {DNNL_ARG_SRC, SRC},
            {DNNL_ARG_WEIGHTS_0, WEI},
            {DNNL_ARG_DIFF_WEIGHTS_0, WEI},
            {DNNL_ARG_BIAS, BIA},
            {DNNL_ARG_DIFF_BIAS, BIA},
            {DNNL_ARG_DST, DST},
            {DNNL_ARG_DIFF_SRC_0, DST},
            {DNNL_ARG_SRC_1, SRC_1},
            {DNNL_ARG_MEAN, MEAN},
            {DNNL_ARG_VARIANCE, VAR},
            {DNNL_ARG_SCALE, SC},
            {DNNL_ARG_DIFF_SCALE, SC},
            {DNNL_ARG_SHIFT, SH},
            {DNNL_ARG_DIFF_SHIFT, SH},
    };
    return map_;
}

int input_md_process(dnn_mem_map_t &mems, const deserialized_op &base_op_ref,
        const bool is_init_stage, res_t *res) {

    // Mapping from the op kind to a vector that indicates which input logical
    // tensor needs memory permutation
    static const std::unordered_map<std::string, std::vector<int>>
            op_2_input_offset_for_NXC_permute = {
                    {"AvgPool", {0}},
                    {"AvgPoolBackward", {0}},
                    {"BatchNormInference", {0}},
                    {"BatchNormForwardTraining", {0}},
                    {"BiasAddBackward", {0}},
                    {"Interpolate", {0}},
                    {"MaxPool", {0}},
                    {"Convolution", {0}},
                    {"ConvolutionBackwardData", {0}},
                    {"ConvTranspose", {0}},
                    {"ConvTransposeBackwardData", {0}},
                    {"BatchNormTrainingBackward", {0, 1}},
                    {"BiasAdd", {0, 1}},
                    {"InterpolateBackward", {0, 1}},
                    {"MaxPoolBackward", {0, 1}},
                    {"ConvolutionBackwardWeights", {0, 1}},
                    {"ConvTransposeBackwardWeights", {0, 1}},
                    {"PReLU", {0, 1}},
                    {"PReLUBackward", {0, 1, 2}},
            };

    using permute_func_t = dnnl::memory::desc (*)(const dnnl::memory::desc &);
    using reshape_func_t
            = dnnl::memory::desc (*)(const dnnl::memory::desc &, int64_t, bool);
    permute_func_t permute_NCX_func
            = is_init_stage ? permute_NCX2NXC : permute_NXC2NCX;
    permute_func_t permute_OIX_func;
    static const std::unordered_map<std::string, permute_func_t>
            str_2_permute_OIX_func = {
                    {"permute_OIX2XOI", permute_OIX2XOI},
                    {"permute_OIX2XIO", permute_OIX2XIO},
                    {"permute_OIX2IOX", permute_OIX2IOX},
                    {"permute_XOI2OIX", permute_XOI2OIX},
                    {"permute_XIO2OIX", permute_XIO2OIX},
                    {"permute_IOX2OIX", permute_IOX2OIX},
            };
    reshape_func_t reshape_group_func
            = is_init_stage ? reshape_GOIX2OIX : reshape_OIX2GOIX;

    const auto &op_kind = base_op_ref.kind_;
    // For primitive init stage, permute data from NCX to NXC if the op has
    // data_format = NXC
    if (base_op_ref.has_NXC_format()) {
        for (auto offset : op_2_input_offset_for_NXC_permute.at(op_kind)) {
            int prim_arg_name = get_prim_arg_name_from_graph_op_input_offset(
                    opstr2kind(op_kind), offset);
            if (prim_arg_name == -1) return FAIL;
            permute_md(mems[prim_arg_name], permute_NCX_func);
        }
    }

    // Exchange last 2 dims for matmul op on graph side if transpose=true
    bool t_a = false, t_b = false;
    bool has_t_a = base_op_ref.get_attr_bool(t_a, "transpose_a");
    bool has_t_b = base_op_ref.get_attr_bool(t_b, "transpose_b");
    if (has_t_a && t_a) {
        int prim_arg_name = get_prim_arg_name_from_graph_op_input_offset(
                opstr2kind(op_kind), 0);
        if (prim_arg_name == -1) return FAIL;
        auto permutation
                = get_transpose_permutation_vec(mems[prim_arg_name].ndims());
        permute_md(mems[prim_arg_name], permutation);
    }
    if (has_t_b && t_b) {
        int prim_arg_name = get_prim_arg_name_from_graph_op_input_offset(
                opstr2kind(op_kind), 1);
        if (prim_arg_name == -1) return FAIL;
        auto permutation
                = get_transpose_permutation_vec(mems[prim_arg_name].ndims());
        permute_md(mems[prim_arg_name], permutation);
    }

    // If init_stage:
    //     first reshape weight from GOIX to OIX, then permute weight from OIX
    //     to XIO/XOI/IOX if filter_format = XIO/XOI/IOX
    // If execute_stage:
    //     first permute weight from XIO/XOI/IOX to OIX if
    //     filter_format = XIO/XOI/IOX then reshape weight from OIX to GOIX
    int64_t groups;
    bool has_group = base_op_ref.get_attr_s64(groups, "groups");
    std::string f_fmt;
    bool has_f_fmt = base_op_ref.get_attr_string(f_fmt, "weights_format");
    if (op_kind == "Convolution" || op_kind == "ConvolutionBackwardData"
            || op_kind == "ConvTranspose"
            || op_kind == "ConvTransposeBackwardData") {
        int prim_arg_name = get_prim_arg_name_from_graph_op_input_offset(
                opstr2kind(op_kind), 1);
        if (prim_arg_name == -1) return FAIL;
        bool is_convtranspose = (op_kind == "ConvTranspose"
                || op_kind == "ConvTransposeBackwardData");
        if (is_init_stage) { // Init stage
            if (has_group && groups > 1) {
                reshape_md(mems[prim_arg_name], reshape_group_func, groups,
                        is_convtranspose);
            }
            if (has_f_fmt && f_fmt != "OIX") {
                permute_OIX_func
                        = str_2_permute_OIX_func.at("permute_OIX2" + f_fmt);
                permute_md(mems[prim_arg_name], permute_OIX_func);
            }
        } else { // Execute_stage
            if (has_f_fmt && f_fmt != "OIX") {
                permute_OIX_func = str_2_permute_OIX_func.at(
                        "permute_" + f_fmt + "2OIX");
                permute_md(mems[prim_arg_name], permute_OIX_func);
            }
            if (has_group && groups > 1) {
                reshape_md(mems[prim_arg_name], reshape_group_func, groups,
                        is_convtranspose);
            }
        }
    }

    return OK;
}

int output_md_process(
        dnn_mem_map_t &mems, const deserialized_op &base_op_ref, res_t *res) {

    using permute_func_t = dnnl::memory::desc (*)(const dnnl::memory::desc &);
    permute_func_t permute_OIX_func;
    static const std::unordered_map<std::string, permute_func_t>
            str_2_permute_OIX_func = {
                    {"permute_OIX2XOI", permute_OIX2XOI},
                    {"permute_OIX2XIO", permute_OIX2XIO},
                    {"permute_OIX2IOX", permute_OIX2IOX},
            };

    const auto &op_kind = base_op_ref.kind_;
    // Permute result from NCX to NXC if data_format = NXC
    std::string f_fmt, d_fmt;
    bool has_d_fmt = base_op_ref.get_attr_string(d_fmt, "data_format");
    bool has_f_fmt = base_op_ref.get_attr_string(f_fmt, "weights_format");
    if (has_d_fmt && d_fmt == "NXC") {
        if (op_kind != "ConvolutionBackwardWeights"
                && op_kind != "ConvTransposeBackwardWeights") {
            int prim_arg_name = get_prim_arg_name_from_graph_op_output_offset(
                    opstr2kind(op_kind), 0);
            if (prim_arg_name == -1) return FAIL;
            permute_md(mems[prim_arg_name], permute_NCX2NXC);
        }
        if (op_kind == "PReLUBackward") {
            int prim_arg_name = get_prim_arg_name_from_graph_op_output_offset(
                    opstr2kind(op_kind), 1);
            if (prim_arg_name == -1) return FAIL;
            permute_md(mems[prim_arg_name], permute_NCX2NXC);
        }
    }

    // Reshape output from GOIX to OIX
    int64_t groups;
    bool has_group = base_op_ref.get_attr_s64(groups, "groups");
    if (has_group && groups > 1) {
        int prim_arg_name = get_prim_arg_name_from_graph_op_output_offset(
                opstr2kind(op_kind), 0);
        if (prim_arg_name == -1) return FAIL;
        if (op_kind == "ConvTransposeBackwardWeights") {
            reshape_md(mems[prim_arg_name], reshape_GOIX2OIX, groups,
                    /* is_convtranspose = */ true);
        } else if (op_kind == "ConvolutionBackwardWeights") {
            reshape_md(mems[prim_arg_name], reshape_GOIX2OIX, groups,
                    /* is_convtranspose = */ false);
        }
    }

    // Permute result from OIX to XIO/XOI/IOX if filter_format = XIO/XOI/IOX
    if (has_f_fmt && f_fmt != "OIX") {
        if (op_kind == "ConvolutionBackwardWeights"
                || op_kind == "ConvTransposeBackwardWeights") {
            int prim_arg_name = get_prim_arg_name_from_graph_op_output_offset(
                    opstr2kind(op_kind), 0);
            if (prim_arg_name == -1) return FAIL;
            permute_OIX_func
                    = str_2_permute_OIX_func.at("permute_OIX2" + f_fmt);
            permute_md(mems[prim_arg_name], permute_OIX_func);
        }
    }

    // Permute output md according to the order (StaticTranspose case)
    std::vector<int64_t> order;
    bool has_order = base_op_ref.get_attr_s64_vector(order, "order");
    if (has_order) {
        int prim_arg_name = get_prim_arg_name_from_graph_op_output_offset(
                opstr2kind(op_kind), 0);
        if (prim_arg_name == -1) return FAIL;
        permute_md(mems[prim_arg_name], order);
    }

    return OK;
}

} // namespace graph
