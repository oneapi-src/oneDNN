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

#include "interface/op_def_constraint.hpp"

#include "graph/unit/unit_test_common.hpp"
#include "graph/unit/utils.hpp"

#include "gtest/gtest.h"

using namespace dnnl::impl::graph::op_kind;
using namespace dnnl::impl::graph::data_type;
namespace graph = dnnl::impl::graph;
namespace utils = dnnl::graph::tests::unit::utils;

struct dnnl_graph_data_type_params_t {
    graph::op_kind_t op_name;
    size_t input_nums;
    size_t output_nums;
    graph::data_type_t T1;
    graph::data_type_t T2;
    graph::op_def_constraint_fn fn;
    bool result;
};

class data_type_check_t
    : public ::testing::TestWithParam<dnnl_graph_data_type_params_t> {
public:
    void TestDataTypeCheck() {
        auto params = ::testing::TestWithParam<
                dnnl_graph_data_type_params_t>::GetParam();

        graph::op_t op(params.op_name);
        if (params.op_name != TypeCast) {
            op.set_attr<float>(graph::op_attr::epsilon, 0.0);
            op.set_attr<std::string>(graph::op_attr::data_format, "NCX");
        }
        // prepare logical tensor
        size_t idx = 0;
        for (size_t i = 0; i < params.input_nums; i++) {
            graph::logical_tensor_t lt
                    = utils::logical_tensor_init(idx, params.T2);
            if (i == 0
                    || (i == 1
                            && params.op_name == BatchNormTrainingBackward)) {
                lt.data_type = params.T1;
            }
            op.add_input(lt);
            ++idx;
        }

        for (size_t i = 0; i < params.output_nums; i++) {
            graph::logical_tensor_t lt
                    = utils::logical_tensor_init(idx, params.T2);
            if (i == 0 && params.op_name != TypeCast) {
                lt.data_type = params.T1;
            }
            op.add_output(lt);
            ++idx;
        }

        ASSERT_EQ(params.fn(&op), params.result);
    }
};

TEST_P(data_type_check_t, TestDataTypeCheck) {
    TestDataTypeCheck();
}

INSTANTIATE_TEST_SUITE_P(OpDefConstraint, data_type_check_t,
        ::testing::Values(
                // test function of CheckBatchNormDataType
                dnnl_graph_data_type_params_t {BatchNormTrainingBackward, 5, 1,
                        f32, f32, graph::check_bn_data_type, true},
                dnnl_graph_data_type_params_t {BatchNormTrainingBackward, 5, 1,
                        bf16, f32, graph::check_bn_data_type, true},
                dnnl_graph_data_type_params_t {BatchNormTrainingBackward, 5, 1,
                        f32, bf16, graph::check_bn_data_type, false},
                dnnl_graph_data_type_params_t {BatchNormTrainingBackward, 5, 1,
                        bf16, bf16, graph::check_bn_data_type, true},
                // test function of CheckBatchNormDataType
                dnnl_graph_data_type_params_t {BatchNormForwardTraining, 5, 5,
                        f32, f32, graph::check_bn_data_type, true},
                dnnl_graph_data_type_params_t {BatchNormForwardTraining, 5, 5,
                        bf16, f32, graph::check_bn_data_type, true},
                dnnl_graph_data_type_params_t {BatchNormForwardTraining, 5, 5,
                        f32, bf16, graph::check_bn_data_type, false},
                dnnl_graph_data_type_params_t {BatchNormForwardTraining, 5, 5,
                        bf16, bf16, graph::check_bn_data_type, true},
                // test function of CheckBatchNormDataType
                dnnl_graph_data_type_params_t {BatchNormInference, 5, 1, f32,
                        f32, graph::check_bn_data_type, true},
                dnnl_graph_data_type_params_t {BatchNormInference, 5, 1, bf16,
                        f32, graph::check_bn_data_type, true},
                dnnl_graph_data_type_params_t {BatchNormInference, 5, 1, f32,
                        bf16, graph::check_bn_data_type, false},
                dnnl_graph_data_type_params_t {BatchNormInference, 5, 1, bf16,
                        bf16, graph::check_bn_data_type, true},
                // test function of CheckTypeCastDataType
                dnnl_graph_data_type_params_t {TypeCast, 1, 1, f32, bf16,
                        graph::check_typecast_data_type, true},
                dnnl_graph_data_type_params_t {TypeCast, 1, 1, f32, f16,
                        graph::check_typecast_data_type, true},
                dnnl_graph_data_type_params_t {TypeCast, 1, 1, bf16, f32,
                        graph::check_typecast_data_type, true},
                dnnl_graph_data_type_params_t {TypeCast, 1, 1, f16, f32,
                        graph::check_typecast_data_type, true},
                dnnl_graph_data_type_params_t {TypeCast, 1, 1, bf16, bf16,
                        graph::check_typecast_data_type, false},
                dnnl_graph_data_type_params_t {TypeCast, 1, 1, f16, f16,
                        graph::check_typecast_data_type, false},
                dnnl_graph_data_type_params_t {TypeCast, 1, 1, f32, f32,
                        graph::check_typecast_data_type, false}));

struct dnnl_graph_ln_params_t {
    graph::op_kind_t op_name;
    graph::data_type_t T1;
    graph::data_type_t T2;
    bool keep_stats;
    bool use_affine;
    bool add_var;
    graph::op_def_constraint_fn fn;
    bool result;
};

class layer_norm_all_check_t
    : public ::testing::TestWithParam<dnnl_graph_ln_params_t> {
public:
    void TestLayerNormAllCheck() {
        auto params
                = ::testing::TestWithParam<dnnl_graph_ln_params_t>::GetParam();

        graph::op_t op(params.op_name);

        op.set_attr<float>(graph::op_attr::epsilon, 0);
        graph::logical_tensor_t lt_src
                = utils::logical_tensor_init(0, {1, 3, 2}, params.T1);
        graph::logical_tensor_t lt_dst
                = utils::logical_tensor_init(6, {1, 3, 2}, params.T1);
        op.add_input(lt_src);
        op.add_output(lt_dst);

        if (params.op_name == LayerNormBackward) {
            for (size_t i = 1; i < 6; i++) {
                graph::logical_tensor_t lt
                        = utils::logical_tensor_init(i, {1, 3}, params.T2);
                op.add_input(lt);
            }
        }

        op.set_attr(graph::op_attr::keep_stats, params.keep_stats);
        op.set_attr(graph::op_attr::use_affine, params.use_affine);
        if (params.add_var) {
            graph::logical_tensor_t lt_out_1
                    = utils::logical_tensor_init(7, {1, 3}, params.T2);
            graph::logical_tensor_t lt_out_2
                    = utils::logical_tensor_init(8, {1, 3}, params.T2);
            op.add_output(lt_out_1);
            op.add_output(lt_out_2);
        }

        ASSERT_EQ(params.fn(&op), params.result);
    }
};

TEST_P(layer_norm_all_check_t, TestLayerNormAllCheck) {
    TestLayerNormAllCheck();
}

INSTANTIATE_TEST_SUITE_P(OpDefConstraint, layer_norm_all_check_t,
        ::testing::Values(
                // test function of CheckLayerNormDataType
                dnnl_graph_ln_params_t {LayerNorm, f32, f32, true, false, true,
                        graph::check_ln_data_type, true},
                dnnl_graph_ln_params_t {LayerNorm, bf16, f32, true, false, true,
                        graph::check_ln_data_type, true},
                dnnl_graph_ln_params_t {LayerNorm, f32, bf16, true, false, true,
                        graph::check_ln_data_type, false},
                dnnl_graph_ln_params_t {LayerNorm, bf16, bf16, true, false,
                        true, graph::check_ln_data_type, true},
                // test function of CheckLayerNormFwdOutputsNum
                dnnl_graph_ln_params_t {LayerNorm, f32, f32, true, false, true,
                        graph::check_ln_fwd_outputs_num, true},
                dnnl_graph_ln_params_t {LayerNorm, f32, f32, true, false, false,
                        graph::check_ln_fwd_outputs_num, false},
                dnnl_graph_ln_params_t {LayerNorm, f32, bf16, false, false,
                        true, graph::check_ln_fwd_outputs_num, true},
                dnnl_graph_ln_params_t {LayerNorm, f32, bf16, false, false,
                        false, graph::check_ln_fwd_outputs_num, true},
                // test function of CheckLayerNormBwdUseAffine
                dnnl_graph_ln_params_t {LayerNormBackward, f32, f32, true, true,
                        true, graph::check_ln_bwd_use_affine, true},
                dnnl_graph_ln_params_t {LayerNormBackward, f32, f32, true, true,
                        false, graph::check_ln_bwd_use_affine, false},
                dnnl_graph_ln_params_t {LayerNormBackward, f32, bf16, true,
                        false, true, graph::check_ln_bwd_use_affine, true},
                dnnl_graph_ln_params_t {LayerNormBackward, f32, bf16, true,
                        false, false, graph::check_ln_bwd_use_affine, true}));

struct dnnl_graph_shape_params_t {
    graph::op_kind_t op_name;
    size_t input_nums;
    std::vector<graph::data_type_t> inputs_dtype;
    size_t output_nums;
    std::vector<graph::op_attr_t> attr_name;
    std::vector<std::vector<int64_t>> attr_shapes;
    graph::op_def_constraint_fn fn;
    bool keep_dims;
    bool result;
};

class shape_check_t
    : public ::testing::TestWithParam<dnnl_graph_shape_params_t> {
public:
    void TestShapeCheck() {
        auto params = ::testing::TestWithParam<
                dnnl_graph_shape_params_t>::GetParam();

        graph::op_t op(params.op_name);

        size_t idx = 0;
        for (size_t i = 0; i < params.input_nums; i++) {
            graph::logical_tensor_t lt
                    = utils::logical_tensor_init(idx, params.inputs_dtype[i]);
            op.add_input(lt);
            ++idx;
        }

        for (size_t i = 0; i < params.output_nums; i++) {
            graph::logical_tensor_t lt = utils::logical_tensor_init(idx, f32);
            op.add_output(lt);
            ++idx;
        }

        if (params.keep_dims) { op.set_attr(graph::op_attr::keep_dims, true); }

        size_t nums_attr = params.attr_name.size();
        for (size_t i = 0; i < nums_attr; i++) {
            if (params.attr_name[i] == graph::op_attr::scales) {
                size_t scales_dims = params.attr_shapes[i].size();
                size_t value_scales = params.attr_shapes[i][0];
                std::vector<float> new_scales(scales_dims, (float)value_scales);
                op.set_attr(graph::op_attr::scales, new_scales);
            } else {
                op.set_attr(params.attr_name[i], params.attr_shapes[i]);
            }
        }
        ASSERT_EQ(params.fn(&op), params.result);
    }
};

TEST_P(shape_check_t, TestShapeCheck) {
    TestShapeCheck();
}

INSTANTIATE_TEST_SUITE_P(OpDefConstraint, shape_check_t,
        ::testing::Values(
                // test function of CheckAvgPoolBwdInputShape
                dnnl_graph_shape_params_t {AvgPoolBackward, 1, {f32}, 1, {}, {},
                        graph::check_avgpool_bwd_input_shape, false, false},
                dnnl_graph_shape_params_t {AvgPoolBackward, 1, {f32}, 1,
                        {graph::op_attr::src_shape}, {{1, 1, 6, 6}},
                        graph::check_avgpool_bwd_input_shape, false, true},
                dnnl_graph_shape_params_t {AvgPoolBackward, 2, {f32, s32}, 1,
                        {}, {}, graph::check_avgpool_bwd_input_shape, false,
                        true},
                dnnl_graph_shape_params_t {AvgPoolBackward, 2, {f32, s32}, 1,
                        {graph::op_attr::src_shape}, {{1, 1, 6, 6}},
                        graph::check_avgpool_bwd_input_shape, false, true},
                // test function of CheckConvBwdDataOutputShape
                dnnl_graph_shape_params_t {ConvolutionBackwardData, 2,
                        {f32, f32}, 1, {}, {},
                        graph::check_conv_bwd_data_output_shape, false, false},
                dnnl_graph_shape_params_t {ConvolutionBackwardData, 2,
                        {f32, f32}, 1, {graph::op_attr::dst_shape},
                        {{1, 1, 6, 6}}, graph::check_conv_bwd_data_output_shape,
                        false, true},
                dnnl_graph_shape_params_t {ConvolutionBackwardData, 3,
                        {f32, f32, s32}, 1, {}, {},
                        graph::check_conv_bwd_data_output_shape, false, true},
                dnnl_graph_shape_params_t {ConvolutionBackwardData, 3,
                        {f32, f32, s32}, 1, {graph::op_attr::dst_shape},
                        {{1, 1, 6, 6}}, graph::check_conv_bwd_data_output_shape,
                        false, true},
                // test function of CheckConvBwdWeightsWeightsShape
                dnnl_graph_shape_params_t {ConvolutionBackwardWeights, 2,
                        {f32, f32}, 1, {}, {},
                        graph::check_conv_bwd_weights_weights_shape, false,
                        false},
                dnnl_graph_shape_params_t {ConvolutionBackwardWeights, 2,
                        {f32, f32}, 1, {graph::op_attr::weights_shape},
                        {{1, 1, 3, 3}},
                        graph::check_conv_bwd_weights_weights_shape, false,
                        true},
                dnnl_graph_shape_params_t {ConvolutionBackwardWeights, 3,
                        {f32, f32, s32}, 1, {}, {},
                        graph::check_conv_bwd_weights_weights_shape, false,
                        true},
                dnnl_graph_shape_params_t {ConvolutionBackwardWeights, 3,
                        {f32, f32, s32}, 1, {graph::op_attr::weights_shape},
                        {{1, 1, 3, 3}},
                        graph::check_conv_bwd_weights_weights_shape, false,
                        true},
                // test function of CheckInterpolateSizesScales
                dnnl_graph_shape_params_t {Interpolate, 1, {f32}, 1, {}, {},
                        graph::check_interpolate_sizes_scales, false, false},
                dnnl_graph_shape_params_t {Interpolate, 1, {f32}, 1,
                        {graph::op_attr::sizes}, {{2, 2}},
                        graph::check_interpolate_sizes_scales, false, true},
                dnnl_graph_shape_params_t {Interpolate, 1, {f32}, 1,
                        {graph::op_attr::scales}, {{2, 2}},
                        graph::check_interpolate_sizes_scales, false, true},
                dnnl_graph_shape_params_t {Interpolate, 1, {f32}, 1,
                        {graph::op_attr::sizes, graph::op_attr::scales},
                        {{2, 2}, {2, 2}}, graph::check_interpolate_sizes_scales,
                        false, false},
                // test function of CheckReduceAxes
                // including Reduce: L1/L2/Max/Mean/Min/Prod/Sum
                // here we take ReduceL2 as an example
                dnnl_graph_shape_params_t {ReduceL2, 1, {f32}, 1, {}, {},
                        graph::check_reduce_axes, true, false},
                dnnl_graph_shape_params_t {ReduceL2, 1, {f32}, 1,
                        {graph::op_attr::axes}, {{1}}, graph::check_reduce_axes,
                        true, true},
                dnnl_graph_shape_params_t {ReduceL2, 2, {f32, s32}, 1, {}, {},
                        graph::check_reduce_axes, true, true},
                dnnl_graph_shape_params_t {ReduceL2, 2, {f32, s32}, 1,
                        {graph::op_attr::axes}, {{1}}, graph::check_reduce_axes,
                        true, false}));

// including Quantize/Dequantize/DynamicQuantize/DynamicDequantize
struct dnnl_graph_quant_params_t {
    graph::op_kind_t op_name;
    int64_t scales;
    int64_t zps;
    std::string qtype;
    graph::op_def_constraint_fn fn;
    bool result;
};

class quant_check_t
    : public ::testing::TestWithParam<dnnl_graph_quant_params_t> {
public:
    void TestQuantCheck() {
        auto params = ::testing::TestWithParam<
                dnnl_graph_quant_params_t>::GetParam();

        graph::op_t op(params.op_name);

        graph::logical_tensor_t lt_in
                = utils::logical_tensor_init(0, {1, 3, 4, 4}, f32);
        graph::logical_tensor_t lt_out
                = utils::logical_tensor_init(2, {1, 3, 4, 4}, s8);
        op.add_input(lt_in);
        op.add_output(lt_out);

        op.set_attr<std::string>(graph::op_attr::qtype, params.qtype);
        op.set_attr<int64_t>(graph::op_attr::axis, 2);
        if (params.op_name == Quantize || params.op_name == Dequantize) {
            std::vector<float> attr_scales(params.scales, 1.f);
            op.set_attr(graph::op_attr::scales, attr_scales);
            std::vector<int64_t> attr_zps(params.zps, 1);
            op.set_attr(graph::op_attr::zps, attr_zps);
        } else {
            graph::logical_tensor_t lt_scales
                    = utils::logical_tensor_init(1, {params.scales}, f32);
            graph::logical_tensor_t lt_zps
                    = utils::logical_tensor_init(2, {params.zps}, s8);
            op.add_input(lt_scales);
            op.add_input(lt_zps);
        }

        ASSERT_EQ(params.fn(&op), params.result);
    }
};

TEST_P(quant_check_t, TestQuantCheck) {
    TestQuantCheck();
}

INSTANTIATE_TEST_SUITE_P(OpDefConstraint, quant_check_t,
        ::testing::Values(
                // test function of CheckQuantDequantScalesZps
                dnnl_graph_quant_params_t {Quantize, 4, 4, "per_channel",
                        graph::check_quant_dequant_scales_zps, true},
                dnnl_graph_quant_params_t {Quantize, 3, 4, "per_channel",
                        graph::check_quant_dequant_scales_zps, false},
                dnnl_graph_quant_params_t {Quantize, 1, 1, "per_tensor",
                        graph::check_quant_dequant_scales_zps, true},
                dnnl_graph_quant_params_t {Quantize, 1, 2, "per_tensor",
                        graph::check_quant_dequant_scales_zps, false},
                dnnl_graph_quant_params_t {Dequantize, 4, 4, "per_channel",
                        graph::check_quant_dequant_scales_zps, true},
                dnnl_graph_quant_params_t {Dequantize, 3, 4, "per_channel",
                        graph::check_quant_dequant_scales_zps, false},
                dnnl_graph_quant_params_t {Dequantize, 1, 1, "per_tensor",
                        graph::check_quant_dequant_scales_zps, true},
                dnnl_graph_quant_params_t {Dequantize, 1, 2, "per_tensor",
                        graph::check_quant_dequant_scales_zps, false},
                // test function of CheckDynQuantDequantScalesZps
                dnnl_graph_quant_params_t {DynamicQuantize, 4, 4, "per_channel",
                        graph::check_dyn_quant_dequant_scales_zps, true},
                dnnl_graph_quant_params_t {DynamicQuantize, 3, 4, "per_channel",
                        graph::check_dyn_quant_dequant_scales_zps, false},
                dnnl_graph_quant_params_t {DynamicQuantize, 1, 1, "per_tensor",
                        graph::check_dyn_quant_dequant_scales_zps, true},
                dnnl_graph_quant_params_t {DynamicQuantize, 1, 2, "per_tensor",
                        graph::check_dyn_quant_dequant_scales_zps, false},
                dnnl_graph_quant_params_t {DynamicDequantize, 4, 4,
                        "per_channel",
                        graph::check_dyn_quant_dequant_scales_zps, true},
                dnnl_graph_quant_params_t {DynamicDequantize, 3, 4,
                        "per_channel",
                        graph::check_dyn_quant_dequant_scales_zps, false},
                dnnl_graph_quant_params_t {DynamicDequantize, 1, 1,
                        "per_tensor", graph::check_dyn_quant_dequant_scales_zps,
                        true},
                dnnl_graph_quant_params_t {DynamicDequantize, 1, 2,
                        "per_tensor", graph::check_dyn_quant_dequant_scales_zps,
                        false}));
