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

#include "graph/unit/unit_test_common.hpp"
#include "graph/unit/utils.hpp"
#include "interface/c_types_map.hpp"
#include "interface/op_def_constraint.hpp"

#include "gtest/gtest.h"

using namespace dnnl::impl::graph::op_kind;
using namespace dnnl::impl::graph::data_type;
namespace graph = dnnl::impl::graph;
namespace utils = dnnl::graph::tests::unit::utils;
using dims = graph::dims;

struct dnnl_graph_pads_value_params_t {
    graph::op_kind_t op_name;
    dims pads_begin;
    dims pads_end;
    graph::op_def_constraint_fn fn;
    bool result;
};
class pads_value_check_t
    : public ::testing::TestWithParam<dnnl_graph_pads_value_params_t> {
public:
    void TestPadsValueCheck() {
        const auto &params = ::testing::TestWithParam<
                dnnl_graph_pads_value_params_t>::GetParam();
        graph::op_t op(params.op_name);
        op.set_attr<dims>(graph::op_attr::pads_begin, params.pads_begin);
        op.set_attr<dims>(graph::op_attr::pads_end, params.pads_end);
        ASSERT_EQ(params.fn(&op), params.result);
    }
};

TEST_P(pads_value_check_t, TestPadsValueCheck) {
    TestPadsValueCheck();
}

INSTANTIATE_TEST_SUITE_P(test_interface_op_def_constraint, pads_value_check_t,
        ::testing::Values(
                // test function to check padding values
                dnnl_graph_pads_value_params_t {Convolution, dims {-1},
                        dims {1}, graph::check_pads, false},
                dnnl_graph_pads_value_params_t {Convolution, dims {1},
                        dims {-1}, graph::check_pads, false},
                dnnl_graph_pads_value_params_t {Convolution, dims {-1},
                        dims {-1}, graph::check_pads, false},
                dnnl_graph_pads_value_params_t {Convolution, dims {0, 0},
                        dims {0, 0}, graph::check_pads, true},
                dnnl_graph_pads_value_params_t {Convolution, dims {-1, 1},
                        dims {1, 1}, graph::check_pads, false},
                dnnl_graph_pads_value_params_t {Convolution, dims {1, 1},
                        dims {-1, 1}, graph::check_pads, false},
                dnnl_graph_pads_value_params_t {Convolution, dims {1, 1},
                        dims {1, 1}, graph::check_pads, true},
                dnnl_graph_pads_value_params_t {ConvolutionBackwardData,
                        dims {-1}, dims {1}, graph::check_pads, false},
                dnnl_graph_pads_value_params_t {ConvolutionBackwardData,
                        dims {1}, dims {-1}, graph::check_pads, false},
                dnnl_graph_pads_value_params_t {ConvolutionBackwardData,
                        dims {-1}, dims {-1}, graph::check_pads, false},
                dnnl_graph_pads_value_params_t {ConvolutionBackwardData,
                        dims {0, 0}, dims {0, 0}, graph::check_pads, true},
                dnnl_graph_pads_value_params_t {ConvolutionBackwardData,
                        dims {-1, 1}, dims {1, 1}, graph::check_pads, false},
                dnnl_graph_pads_value_params_t {ConvolutionBackwardData,
                        dims {1, 1}, dims {-1, 1}, graph::check_pads, false},
                dnnl_graph_pads_value_params_t {ConvolutionBackwardData,
                        dims {1, 1}, dims {1, 1}, graph::check_pads, true},
                dnnl_graph_pads_value_params_t {ConvolutionBackwardWeights,
                        dims {-1}, dims {1}, graph::check_pads, false},
                dnnl_graph_pads_value_params_t {ConvolutionBackwardWeights,
                        dims {1}, dims {-1}, graph::check_pads, false},
                dnnl_graph_pads_value_params_t {ConvolutionBackwardWeights,
                        dims {-1}, dims {-1}, graph::check_pads, false},
                dnnl_graph_pads_value_params_t {ConvolutionBackwardWeights,
                        dims {0, 0}, dims {0, 0}, graph::check_pads, true},
                dnnl_graph_pads_value_params_t {ConvolutionBackwardWeights,
                        dims {-1, 1}, dims {1, 1}, graph::check_pads, false},
                dnnl_graph_pads_value_params_t {ConvolutionBackwardWeights,
                        dims {1, 1}, dims {-1, 1}, graph::check_pads, false},
                dnnl_graph_pads_value_params_t {ConvolutionBackwardWeights,
                        dims {1, 1}, dims {1, 1}, graph::check_pads, true},
                dnnl_graph_pads_value_params_t {ConvTranspose, dims {-1},
                        dims {1}, graph::check_pads, false},
                dnnl_graph_pads_value_params_t {ConvTranspose, dims {1},
                        dims {-1}, graph::check_pads, false},
                dnnl_graph_pads_value_params_t {ConvTranspose, dims {-1},
                        dims {-1}, graph::check_pads, false},
                dnnl_graph_pads_value_params_t {ConvTranspose, dims {0, 0},
                        dims {0, 0}, graph::check_pads, true},
                dnnl_graph_pads_value_params_t {ConvTranspose, dims {-1, 1},
                        dims {1, 1}, graph::check_pads, false},
                dnnl_graph_pads_value_params_t {ConvTranspose, dims {1, 1},
                        dims {-1, 1}, graph::check_pads, false},
                dnnl_graph_pads_value_params_t {ConvTranspose, dims {1, 1},
                        dims {1, 1}, graph::check_pads, true},
                dnnl_graph_pads_value_params_t {ConvTransposeBackwardData,
                        dims {-1}, dims {1}, graph::check_pads, false},
                dnnl_graph_pads_value_params_t {ConvTransposeBackwardData,
                        dims {1}, dims {-1}, graph::check_pads, false},
                dnnl_graph_pads_value_params_t {ConvTransposeBackwardData,
                        dims {-1}, dims {-1}, graph::check_pads, false},
                dnnl_graph_pads_value_params_t {ConvTransposeBackwardData,
                        dims {0, 0}, dims {0, 0}, graph::check_pads, true},
                dnnl_graph_pads_value_params_t {ConvTransposeBackwardData,
                        dims {-1, 1}, dims {1, 1}, graph::check_pads, false},
                dnnl_graph_pads_value_params_t {ConvTransposeBackwardData,
                        dims {1, 1}, dims {-1, 1}, graph::check_pads, false},
                dnnl_graph_pads_value_params_t {ConvTransposeBackwardData,
                        dims {1, 1}, dims {1, 1}, graph::check_pads, true},
                dnnl_graph_pads_value_params_t {ConvTransposeBackwardWeights,
                        dims {-1}, dims {1}, graph::check_pads, false},
                dnnl_graph_pads_value_params_t {ConvTransposeBackwardWeights,
                        dims {1}, dims {-1}, graph::check_pads, false},
                dnnl_graph_pads_value_params_t {ConvTransposeBackwardWeights,
                        dims {-1}, dims {-1}, graph::check_pads, false},
                dnnl_graph_pads_value_params_t {ConvTransposeBackwardWeights,
                        dims {0, 0}, dims {0, 0}, graph::check_pads, true},
                dnnl_graph_pads_value_params_t {ConvTransposeBackwardWeights,
                        dims {-1, 1}, dims {1, 1}, graph::check_pads, false},
                dnnl_graph_pads_value_params_t {ConvTransposeBackwardWeights,
                        dims {1, 1}, dims {-1, 1}, graph::check_pads, false},
                dnnl_graph_pads_value_params_t {ConvTransposeBackwardWeights,
                        dims {1, 1}, dims {1, 1}, graph::check_pads, true},
                dnnl_graph_pads_value_params_t {
                        AvgPool, dims {-1}, dims {1}, graph::check_pads, false},
                dnnl_graph_pads_value_params_t {
                        AvgPool, dims {1}, dims {-1}, graph::check_pads, false},
                dnnl_graph_pads_value_params_t {AvgPool, dims {-1}, dims {-1},
                        graph::check_pads, false},
                dnnl_graph_pads_value_params_t {AvgPool, dims {0, 0},
                        dims {0, 0}, graph::check_pads, true},
                dnnl_graph_pads_value_params_t {AvgPool, dims {-1, 1},
                        dims {1, 1}, graph::check_pads, false},
                dnnl_graph_pads_value_params_t {AvgPool, dims {1, 1},
                        dims {-1, 1}, graph::check_pads, false},
                dnnl_graph_pads_value_params_t {AvgPool, dims {1, 1},
                        dims {1, 1}, graph::check_pads, true},
                dnnl_graph_pads_value_params_t {AvgPoolBackward, dims {-1},
                        dims {1}, graph::check_pads, false},
                dnnl_graph_pads_value_params_t {AvgPoolBackward, dims {1},
                        dims {-1}, graph::check_pads, false},
                dnnl_graph_pads_value_params_t {AvgPoolBackward, dims {-1},
                        dims {-1}, graph::check_pads, false},
                dnnl_graph_pads_value_params_t {AvgPoolBackward, dims {0, 0},
                        dims {0, 0}, graph::check_pads, true},
                dnnl_graph_pads_value_params_t {AvgPoolBackward, dims {-1, 1},
                        dims {1, 1}, graph::check_pads, false},
                dnnl_graph_pads_value_params_t {AvgPoolBackward, dims {1, 1},
                        dims {-1, 1}, graph::check_pads, false},
                dnnl_graph_pads_value_params_t {AvgPoolBackward, dims {1, 1},
                        dims {1, 1}, graph::check_pads, true},
                dnnl_graph_pads_value_params_t {MaxPoolBackward, dims {-1},
                        dims {1}, graph::check_pads, false},
                dnnl_graph_pads_value_params_t {MaxPoolBackward, dims {1},
                        dims {-1}, graph::check_pads, false},
                dnnl_graph_pads_value_params_t {MaxPoolBackward, dims {-1},
                        dims {-1}, graph::check_pads, false},
                dnnl_graph_pads_value_params_t {MaxPoolBackward, dims {0, 0},
                        dims {0, 0}, graph::check_pads, true},
                dnnl_graph_pads_value_params_t {MaxPoolBackward, dims {-1, 1},
                        dims {1, 1}, graph::check_pads, false},
                dnnl_graph_pads_value_params_t {MaxPoolBackward, dims {1, 1},
                        dims {-1, 1}, graph::check_pads, false},
                dnnl_graph_pads_value_params_t {MaxPoolBackward, dims {1, 1},
                        dims {1, 1}, graph::check_pads, true},
                dnnl_graph_pads_value_params_t {
                        MaxPool, dims {-1}, dims {1}, graph::check_pads, false},
                dnnl_graph_pads_value_params_t {
                        MaxPool, dims {1}, dims {-1}, graph::check_pads, false},
                dnnl_graph_pads_value_params_t {MaxPool, dims {-1}, dims {-1},
                        graph::check_pads, false},
                dnnl_graph_pads_value_params_t {MaxPool, dims {0, 0},
                        dims {0, 0}, graph::check_pads, true},
                dnnl_graph_pads_value_params_t {MaxPool, dims {-1, 1},
                        dims {1, 1}, graph::check_pads, false},
                dnnl_graph_pads_value_params_t {MaxPool, dims {1, 1},
                        dims {-1, 1}, graph::check_pads, false},
                dnnl_graph_pads_value_params_t {MaxPool, dims {1, 1},
                        dims {1, 1}, graph::check_pads, true}));

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
        const auto &params = ::testing::TestWithParam<
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

INSTANTIATE_TEST_SUITE_P(test_interface_op_def_constraint, data_type_check_t,
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
        const auto &params
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

INSTANTIATE_TEST_SUITE_P(test_interface_op_def_constraint,
        layer_norm_all_check_t,
        ::testing::Values(
                // test function of CheckLayerNormDataType
                dnnl_graph_ln_params_t {LayerNorm, f32, f32, true, false, true,
                        graph::check_norm_data_type, true},
                dnnl_graph_ln_params_t {LayerNorm, bf16, f32, true, false, true,
                        graph::check_norm_data_type, true},
                dnnl_graph_ln_params_t {LayerNorm, f32, bf16, true, false, true,
                        graph::check_norm_data_type, false},
                dnnl_graph_ln_params_t {LayerNorm, bf16, bf16, true, false,
                        true, graph::check_norm_data_type, true},
                // test function of CheckLayerNormFwdOutputsNum
                dnnl_graph_ln_params_t {LayerNorm, f32, f32, true, false, true,
                        graph::check_ln_gn_fwd_outputs_num, true},
                dnnl_graph_ln_params_t {LayerNorm, f32, f32, true, false, false,
                        graph::check_ln_gn_fwd_outputs_num, false},
                dnnl_graph_ln_params_t {LayerNorm, f32, bf16, false, false,
                        true, graph::check_ln_gn_fwd_outputs_num, true},
                dnnl_graph_ln_params_t {LayerNorm, f32, bf16, false, false,
                        false, graph::check_ln_gn_fwd_outputs_num, true},
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
        const auto &params = ::testing::TestWithParam<
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

INSTANTIATE_TEST_SUITE_P(test_interface_op_def_constraint, shape_check_t,
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

// including Quantize/Dequantize
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
        const auto &params = ::testing::TestWithParam<
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
        std::vector<float> attr_scales(params.scales, 1.f);
        op.set_attr(graph::op_attr::scales, attr_scales);
        std::vector<int64_t> attr_zps(params.zps, 1);
        op.set_attr(graph::op_attr::zps, attr_zps);

        ASSERT_EQ(params.fn(&op), params.result);
    }
};

TEST_P(quant_check_t, TestQuantCheck) {
    TestQuantCheck();
}

INSTANTIATE_TEST_SUITE_P(test_interface_op_def_constraint, quant_check_t,
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
                        graph::check_quant_dequant_scales_zps, false}));

namespace {
// Builder for DynamicQuantize/DynamicDequantize constraint checks. Attributes
// are only set when requested so that defaults are exercised as well.
struct dynq_t {
    dynq_t(const dims &src, const dims &scales) : src_(src), scales_(scales) {}

    dynq_t &kind(graph::op_kind_t k) {
        kind_ = k;
        return *this;
    }
    dynq_t &mask(int64_t m) {
        mask_ = m;
        with_mask_ = true;
        return *this;
    }
    dynq_t &qtype(const std::string &q) {
        qtype_ = q;
        return *this;
    }
    dynq_t &axis(int64_t a) {
        axis_ = a;
        with_axis_ = true;
        return *this;
    }
    dynq_t &group(const dims &g) {
        group_ = g;
        return *this;
    }
    dynq_t &zps(const dims &z) {
        zps_ = z;
        with_zps_ = true;
        return *this;
    }
    dynq_t &unknown_src_rank() {
        unknown_src_rank_ = true;
        return *this;
    }
    // Use f8_e4m3 as the quantized data type.
    dynq_t &f8() {
        f8_ = true;
        return *this;
    }

    bool check() const {
        const bool is_quant = kind_ == DynamicQuantize;
        const auto qdt = f8_ ? f8_e4m3 : s8;
        const auto src_dt = is_quant ? f32 : qdt;
        const auto dst_dt = is_quant ? qdt : f32;
        graph::op_t op(kind_);
        op.add_input(unknown_src_rank_
                        ? utils::logical_tensor_init(0, src_dt)
                        : utils::logical_tensor_init(0, src_, src_dt));
        op.add_input(utils::logical_tensor_init(1, scales_, f32));
        if (with_zps_) op.add_input(utils::logical_tensor_init(2, zps_, s8));
        op.add_output(utils::logical_tensor_init(3, src_, dst_dt));
        if (with_mask_) op.set_attr<int64_t>(graph::op_attr::mask, mask_);
        if (!qtype_.empty())
            op.set_attr<std::string>(graph::op_attr::qtype, qtype_);
        if (with_axis_) op.set_attr<int64_t>(graph::op_attr::axis, axis_);
        if (!group_.empty())
            op.set_attr<dims>(graph::op_attr::group_shape, group_);
        return graph::check_dyn_quant_dequant_scales_zps(&op);
    }

private:
    graph::op_kind_t kind_ = DynamicDequantize;
    dims src_, scales_, zps_, group_;
    std::string qtype_;
    int64_t mask_ = 0, axis_ = 1;
    bool with_mask_ = false, with_axis_ = false, with_zps_ = false;
    bool unknown_src_rank_ = false, f8_ = false;
};
} // namespace

TEST(test_interface_op_def_constraint, DynQuantSpecAttributeValidation) {
    const dims src = {2, 3, 4, 5};
    // mask
    EXPECT_FALSE(dynq_t(src, {2}).mask(-1).check());
    EXPECT_FALSE(dynq_t(src, {2}).mask(1 << 4).check());
    EXPECT_FALSE(dynq_t(src, {3}).mask(1 << 1).qtype("per_channel").check());
    EXPECT_FALSE(dynq_t(src, {3}).mask(1 << 1).qtype("per_group").check());
    EXPECT_TRUE(dynq_t(src, {3}).mask(1 << 1).qtype("per_tensor").check());
    EXPECT_TRUE(dynq_t(src, {3}).mask(1 << 1).axis(1).check());
    EXPECT_FALSE(dynq_t(src, {3}).mask(1 << 1).axis(2).check());
    // qtype / axis
    EXPECT_FALSE(dynq_t(src, {1}).qtype("per_foo").check());
    EXPECT_FALSE(dynq_t(src, {5}).qtype("per_channel").axis(4).check());
    EXPECT_FALSE(dynq_t(src, {2}).qtype("per_channel").axis(-5).check());
    EXPECT_TRUE(dynq_t(src, {5}).qtype("per_channel").axis(-1).check());
    // group_shape
    for (auto kind : {DynamicQuantize, DynamicDequantize})
        EXPECT_FALSE(
                dynq_t({2, 64}, {2, 2}).kind(kind).qtype("per_group").check());
    EXPECT_FALSE(dynq_t({2, 64}, {2, 2}).group({1, 32}).check());
    EXPECT_FALSE(dynq_t({2, 64}, {2, 2})
                         .qtype("per_channel")
                         .group({1, 32})
                         .check());
    EXPECT_FALSE(
            dynq_t({2, 64}, {2, 2}).qtype("per_group").group({32}).check());
    EXPECT_FALSE(dynq_t({2, 64}, {2, 2}).mask(3).group({32}).check());
    EXPECT_FALSE(
            dynq_t({2, 64}, {2, 2}).qtype("per_group").group({1, 0}).check());
    EXPECT_FALSE(
            dynq_t({2, 64}, {2, 2}).qtype("per_group").group({1, 24}).check());
    EXPECT_FALSE(
            dynq_t({2, 64}, {2, 1}).qtype("per_group").group({1, 128}).check());
}

TEST(test_interface_op_def_constraint, DynQuantSpecUnknownShapesSkipped) {
    EXPECT_TRUE(dynq_t({}, {7}).unknown_src_rank().mask(3).check());
    EXPECT_TRUE(dynq_t({2, 3}, {DNNL_GRAPH_UNKNOWN_DIM}).mask(3).check());
    EXPECT_TRUE(dynq_t({2, 3}, {3})
                        .mask(1 << 1)
                        .zps({DNNL_GRAPH_UNKNOWN_DIM})
                        .check());
}

// per_tensor: one element in any rank whose dims are all 1.
TEST(test_interface_op_def_constraint, DynQuantSpecPerTensor) {
    const dims src = {2, 3, 4, 5};
    for (auto kind : {DynamicQuantize, DynamicDequantize}) {
        for (const auto &s : std::vector<dims> {
                     {}, {1}, {1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1, 1}}) {
            EXPECT_TRUE(dynq_t(src, s).kind(kind).check());
            EXPECT_TRUE(dynq_t(src, s).kind(kind).qtype("per_tensor").check());
            EXPECT_TRUE(dynq_t(src, s).kind(kind).mask(0).check());
        }
        for (const auto &s : std::vector<dims> {{2}, {1, 2}, {2, 1, 1, 1}}) {
            EXPECT_FALSE(dynq_t(src, s).kind(kind).check());
            EXPECT_FALSE(dynq_t(src, s).kind(kind).mask(0).check());
        }
    }
}

// per_channel by qtype/axis: implied mask is 1 << axis.
TEST(test_interface_op_def_constraint, DynQuantSpecPerChannel) {
    const dims src = {2, 3, 4, 5};
    for (auto kind : {DynamicQuantize, DynamicDequantize}) {
        auto pc = [&](const dims &s, int64_t axis) {
            return dynq_t(src, s).kind(kind).qtype("per_channel").axis(axis);
        };
        EXPECT_TRUE(pc({4}, 2).check());
        EXPECT_TRUE(pc({1, 1, 4, 1}, 2).check());
        EXPECT_TRUE(pc({5}, -1).check());
        EXPECT_TRUE(dynq_t(src, {3}).kind(kind).qtype("per_channel").check());
        EXPECT_FALSE(pc({3}, 2).check());
        EXPECT_FALSE(pc({4, 1}, 2).check());
        EXPECT_FALSE(pc({1, 1, 1, 4}, 2).check());
        EXPECT_FALSE(pc({1, 4, 1, 1}, 2).check());
        // size-1 channel axis
        auto pc1 = [&](const dims &s) {
            return dynq_t({2, 1, 4}, s)
                    .kind(kind)
                    .qtype("per_channel")
                    .axis(1)
                    .check();
        };
        EXPECT_TRUE(pc1({1}));
        EXPECT_TRUE(pc1({1, 1, 1}));
        EXPECT_FALSE(pc1({2}));
    }
}

// mask: packed scales have one dim per set bit, including masked dims of size
// 1; full-rank scales have 1 on unmasked dims.
TEST(test_interface_op_def_constraint, DynQuantSpecMask) {
    for (auto kind : {DynamicQuantize, DynamicDequantize}) {
        auto m = [&](const dims &src, int64_t mask, const dims &s) {
            return dynq_t(src, s).kind(kind).mask(mask).check();
        };
        // per-head
        EXPECT_TRUE(m({2, 3, 4, 5}, 3, {2, 3}));
        EXPECT_TRUE(m({2, 3, 4, 5}, 3, {2, 3, 1, 1}));
        EXPECT_FALSE(m({2, 3, 4, 5}, 3, {2, 4}));
        EXPECT_FALSE(m({2, 3, 4, 5}, 3, {2, 3, 4}));
        // single bit
        EXPECT_TRUE(m({2, 3, 4, 5}, 1 << 2, {4}));
        EXPECT_TRUE(m({2, 3, 4, 5}, 1 << 2, {1, 1, 4, 1}));
        EXPECT_FALSE(m({2, 3, 4, 5}, 1 << 2, {3}));
        // all bits set: packed == full-rank
        EXPECT_TRUE(m({2, 3, 4, 5}, 15, {2, 3, 4, 5}));
        EXPECT_FALSE(m({2, 3, 4, 5}, 15, {3, 4, 5}));
        // non-contiguous bits
        EXPECT_TRUE(m({2, 3, 4, 5}, 0b0101, {2, 4}));
        EXPECT_TRUE(m({2, 3, 4, 5}, 0b0101, {2, 1, 4, 1}));
        EXPECT_FALSE(m({2, 3, 4, 5}, 0b0101, {4, 2}));
        EXPECT_FALSE(m({2, 3, 4, 5}, 0b0101, {2, 3, 4, 1}));
        EXPECT_FALSE(m({2, 3, 4, 5}, 0b0101, {8}));
        // unmasked dims of a full-rank scales must be 1
        EXPECT_FALSE(m({2, 3, 4, 5}, 3, {2, 3, 4, 1}));
        // masked dims of size 1 stay in the packed layout
        EXPECT_TRUE(m({1, 3, 4, 5}, 3, {1, 3}));
        EXPECT_TRUE(m({1, 3, 4, 5}, 3, {1, 3, 1, 1}));
        EXPECT_FALSE(m({1, 3, 4, 5}, 3, {3}));
        EXPECT_TRUE(m({2, 1, 4, 5}, 3, {2, 1}));
        EXPECT_TRUE(m({1, 3, 4, 5}, 0b0101, {1, 4}));
        EXPECT_FALSE(m({1, 3, 4, 5}, 0b0101, {4}));
        EXPECT_TRUE(m({2, 1, 4}, 0b010, {1}));
        EXPECT_TRUE(m({2, 1, 4}, 0b010, {1, 1, 1}));
        EXPECT_FALSE(m({2, 1, 4}, 0b010, {}));
        EXPECT_TRUE(m({1, 1, 4, 5}, 3, {1, 1}));
        EXPECT_TRUE(m({1, 1, 4, 5}, 3, {1, 1, 1, 1}));
        EXPECT_FALSE(m({1, 1, 4, 5}, 3, {1}));
        EXPECT_FALSE(m({1, 1, 4, 5}, 3, {}));
        // 5D GQA
        EXPECT_TRUE(m({1, 2, 4, 8, 16}, 7, {1, 2, 4}));
        EXPECT_TRUE(m({2, 1, 4, 8, 16}, 7, {2, 1, 4}));
        EXPECT_TRUE(m({2, 2, 1, 8, 16}, 7, {2, 2, 1}));
        EXPECT_TRUE(m({2, 2, 1, 8, 16}, 3, {2, 2}));
        EXPECT_FALSE(m({2, 2, 1, 8, 16}, 7, {2, 2}));
    }
}

// per_group quantization by qtype + group_shape.
TEST(test_interface_op_def_constraint, DynDequantSpecPerGroupByQtype) {
    auto pg = [](const dims &src, const dims &group, const dims &s) {
        return dynq_t(src, s).qtype("per_group").group(group).check();
    };
    EXPECT_TRUE(pg({2, 64}, {1, 32}, {2, 2}));
    EXPECT_FALSE(pg({2, 64}, {1, 32}, {4}));
    EXPECT_FALSE(pg({2, 64}, {1, 32}, {2}));
    EXPECT_FALSE(pg({2, 64}, {1, 32}, {2, 4}));
    // broadcast dim 0
    EXPECT_TRUE(pg({2, 64}, {2, 32}, {1, 2}));
    EXPECT_FALSE(pg({2, 64}, {2, 32}, {2}));
    // size-1 src dim
    EXPECT_TRUE(pg({1, 64}, {1, 32}, {1, 2}));
    EXPECT_FALSE(pg({1, 64}, {1, 32}, {2}));
    // single group: still full-rank
    EXPECT_TRUE(pg({2, 64}, {2, 64}, {1, 1}));
    EXPECT_FALSE(pg({2, 64}, {2, 64}, {1}));
    EXPECT_FALSE(pg({2, 64}, {2, 64}, {}));
    // library limitation: rank >= 2
    EXPECT_FALSE(pg({64}, {32}, {2}));
    // compressed KV cache
    EXPECT_TRUE(pg({1, 32, 128, 32}, {1, 1, 32, 1}, {1, 32, 4, 32}));
    EXPECT_FALSE(pg({1, 32, 128, 32}, {1, 1, 32, 1}, {32, 4, 32}));
    // zps must match the scales shape
    EXPECT_TRUE(dynq_t({2, 64}, {2, 2})
                        .qtype("per_group")
                        .group({1, 32})
                        .zps({2, 2})
                        .check());
    EXPECT_FALSE(dynq_t({2, 64}, {2, 2})
                         .qtype("per_group")
                         .group({1, 32})
                         .zps({1})
                         .check());
}

// per_group quantization by mask + group_shape.
TEST(test_interface_op_def_constraint, DynDequantSpecMaskWithGroup) {
    auto mg = [](const dims &src, int64_t mask, const dims &group,
                      const dims &s) {
        return dynq_t(src, s).mask(mask).group(group).check();
    };
    EXPECT_TRUE(mg({2, 64}, 3, {1, 32}, {2, 2}));
    EXPECT_FALSE(mg({2, 64}, 3, {1, 32}, {2, 4}));
    EXPECT_FALSE(mg({2, 64}, 3, {1, 32}, {4}));
    // size-1 masked dim
    EXPECT_TRUE(mg({1, 64}, 3, {1, 32}, {1, 2}));
    EXPECT_FALSE(mg({1, 64}, 3, {1, 32}, {2}));
    EXPECT_TRUE(mg({4, 1, 64}, 0b110, {1, 1, 32}, {1, 2}));
    EXPECT_FALSE(mg({4, 1, 64}, 0b110, {1, 1, 32}, {2}));
    // unmasked leading dim, group_shape 1
    EXPECT_TRUE(mg({4, 8, 64}, 0b110, {1, 1, 32}, {8, 2}));
    EXPECT_TRUE(mg({4, 8, 64}, 0b110, {1, 1, 32}, {1, 8, 2}));
    EXPECT_FALSE(mg({4, 8, 64}, 0b110, {1, 1, 32}, {4, 8, 2}));
    // unmasked leading dim, group_shape == src
    EXPECT_TRUE(mg({4, 8, 64}, 0b110, {4, 1, 32}, {8, 2}));
    EXPECT_TRUE(mg({4, 8, 64}, 0b110, {4, 1, 32}, {1, 8, 2}));
    // unmasked leading dim, group_shape neither 1 nor src
    EXPECT_FALSE(mg({4, 8, 64}, 0b110, {2, 1, 32}, {8, 2}));
    EXPECT_FALSE(mg({4, 8, 64}, 0b110, {2, 1, 32}, {2, 8, 2}));
    // grouped masked leading dim
    EXPECT_TRUE(mg({4, 8, 64}, 0b110, {1, 2, 32}, {4, 2}));
    EXPECT_TRUE(mg({4, 8, 64}, 0b110, {1, 2, 32}, {1, 4, 2}));
    // library limitation: the last two dims must be masked, rank >= 2
    EXPECT_FALSE(mg({4, 8, 64}, 0b011, {1, 2, 1}, {4, 4}));
    EXPECT_FALSE(mg({4, 8, 64}, 0b101, {1, 1, 32}, {4, 2}));
    EXPECT_FALSE(mg({64}, 1, {32}, {2}));
    // zps follows the scales shape
    EXPECT_TRUE(dynq_t({4, 8, 64}, {8, 2})
                        .mask(0b110)
                        .group({1, 1, 32})
                        .zps({8, 2})
                        .check());
    EXPECT_FALSE(dynq_t({4, 8, 64}, {8, 2})
                         .mask(0b110)
                         .group({1, 1, 32})
                         .zps({1, 8, 2})
                         .check());
}

// zps (zero points) support for dynamic quantization
TEST(test_interface_op_def_constraint, DynQuantSpecZps) {
    for (auto kind : {DynamicQuantize, DynamicDequantize}) {
        auto z = [&](const dims &s, const dims &zp) {
            return dynq_t({2, 3, 4, 5}, s).kind(kind).mask(3).zps(zp).check();
        };
        EXPECT_TRUE(z({2, 3}, {2, 3}));
        EXPECT_TRUE(z({2, 3, 1, 1}, {2, 3, 1, 1}));
        EXPECT_FALSE(z({2, 3}, {2, 3, 1, 1}));
        EXPECT_FALSE(z({2, 3}, {3, 2}));
        // scalar zps broadcasts when not grouped
        EXPECT_TRUE(z({2, 3}, {1}));
        EXPECT_TRUE(z({2, 3}, {}));
        EXPECT_FALSE(z({2, 3}, {2}));
        auto zc = [&](const dims &zp) {
            return dynq_t({2, 3, 4, 5}, {4})
                    .kind(kind)
                    .qtype("per_channel")
                    .axis(2)
                    .zps(zp)
                    .check();
        };
        EXPECT_TRUE(zc({4}));
        EXPECT_FALSE(zc({3}));
        EXPECT_TRUE(dynq_t({2, 3}, {1}).kind(kind).zps({1}).check());
        EXPECT_FALSE(dynq_t({2, 3}, {1}).kind(kind).zps({2}).check());
        // f8 does not support zps, checked even with unknown scales shape
        EXPECT_TRUE(dynq_t({2, 3}, {1}).kind(kind).f8().check());
        EXPECT_FALSE(dynq_t({2, 3}, {1}).kind(kind).f8().zps({1}).check());
        EXPECT_FALSE(dynq_t({2, 3}, {DNNL_GRAPH_UNKNOWN_DIM})
                             .kind(kind)
                             .f8()
                             .zps({1})
                             .check());
    }
}
