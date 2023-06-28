/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
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

#include <functional>
#include <random>

#include "interface/c_types_map.hpp"

#include "gtest/gtest.h"

#include "graph/unit/backend/dnnl/dnnl_test_common.hpp"
#include "graph/unit/backend/dnnl/ref_func.hpp"
#include "graph/unit/unit_test_common.hpp"
#include "graph/unit/utils.hpp"

namespace graph = dnnl::impl::graph;
namespace utils = dnnl::graph::tests::unit::utils;

template <typename attr_data_t = float>
static inline void test_eltwise_common(test::vector<float> &src,
        test::vector<float> &ref_dst, dnnl::impl::graph::dims &dims,
        const graph::op_kind_t op_kind, const std::string &op_name,
        const std::map<dnnl::impl::graph::op_attr_t, attr_data_t> &attrs_data
        = {},
        bool is_constant_input = false) {
    graph::engine_t *eng = get_engine();
    graph::op_t op(op_kind, op_name);

    test::vector<float> dst(src.size(), 0.0);

    for (const auto &attr_data : attrs_data) {
        op.set_attr<attr_data_t>(attr_data.first, attr_data.second);
    }

    graph::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, dims, graph::data_type::f32);
    if (is_constant_input) src_lt.property = graph::property_type::constant;
    graph::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, dims, graph::data_type::f32, graph::layout_type::any);

    op.add_input(src_lt);
    op.add_output(dst_lt);

    graph::graph_t g(eng->kind());
    g.add_op(&op);
    g.finalize();

    std::string pass_name = op_name + "_pass";

    graph::pass::pass_base_ptr apass = get_pass(pass_name);
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {&src_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&dst_lt};

    p.compile(&cp, inputs, outputs, eng);

    graph::logical_tensor_t lt;
    cp.query_logical_tensor(dst_lt.id, &lt);

    ASSERT_EQ(lt.layout_type, graph::layout_type::strided);

    graph::tensor_t src_ts(src_lt, eng, src.data());
    graph::tensor_t dst_ts(dst_lt, eng, dst.data());

    graph::stream_t *strm = get_stream();

    cp.execute(strm, {src_ts}, {dst_ts});
    strm->wait();

    if (op_kind == graph::op_kind::Log || op_kind == graph::op_kind::GELU
            || op_kind == graph::op_kind::SoftPlus) {
        for (size_t i = 0; i < src.size(); ++i) {
            ASSERT_NEAR(dst[i], ref_dst[i], 1e-5);
        }
    } else {
        for (size_t i = 0; i < src.size(); ++i) {
            ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
        }
    }

    //test with same data and stream to see
    //if the memory cache runs correctly
    test::vector<float> dst2(src.size(), 0.0);

    graph::tensor_t src_ts2(src_lt, eng, src.data());
    graph::tensor_t dst_ts2(dst_lt, eng, dst2.data());

    cp.execute(strm, {src_ts2}, {dst_ts2});
    strm->wait();
    if (!is_constant_input) return;
    // for constant input case, we will check the results of second iteration to
    // make sure it's properly handled.
    if (op_kind == graph::op_kind::Log || op_kind == graph::op_kind::GELU
            || op_kind == graph::op_kind::SoftPlus) {
        for (size_t i = 0; i < src.size(); ++i) {
            ASSERT_NEAR(dst2[i], ref_dst[i], 1e-5);
        }
    } else {
        for (size_t i = 0; i < src.size(); ++i) {
            ASSERT_FLOAT_EQ(dst2[i], ref_dst[i]);
        }
    }
}

TEST(Execute, Abs) {
    test::vector<float> src {-2.0, -1.5, -1.0, -0.5, 0.0, 3.5};
    test::vector<float> ref_dst {2.0, 1.5, 1.0, 0.5, 0.0, 3.5};

    dnnl::impl::graph::dims dims {1, 2, 3};

    test_eltwise_common(src, ref_dst, dims, graph::op_kind::Abs, "abs");
}

TEST(Execute, AbsConstantInput) {
    test::vector<float> src {-2.0, -1.5, -1.0, -0.5, 0.0, 3.5};
    test::vector<float> ref_dst {2.0, 1.5, 1.0, 0.5, 0.0, 3.5};

    dnnl::impl::graph::dims dims {1, 2, 3};

    test_eltwise_common(
            src, ref_dst, dims, graph::op_kind::Abs, "abs", {}, true);
}

TEST(Execute, Round) {
    test::vector<float> src {1.1f, -2.3f, 4.7f, 1.0f, 0.0f, -8.34f};
    test::vector<float> ref_dst = round_func(src);

    dnnl::impl::graph::dims dims {1, 2, 3};

    test_eltwise_common(src, ref_dst, dims, graph::op_kind::Round, "round");
}

TEST(Execute, Clamp) {
    test::vector<float> src {-2.0, -1.5, -1.0, -0.5, 0.0, 3.5};
    test::vector<float> ref_dst {-1.0, -1.0, -1.0, -0.5, 0.0, 2.0};

    graph::dims dims {1, 2, 3};
    const std::map<graph::op_attr_t, float> attrs_data {
            {graph::op_attr::min, -1.f}, {graph::op_attr::max, 2.f}};

    test_eltwise_common(
            src, ref_dst, dims, graph::op_kind::Clamp, "clamp", attrs_data);
}

template <typename attr_data_t = float>
static inline void test_eltwise_bwd_common(
        const std::pair<test::vector<float>, bool> &fwd_data_pair,
        const test::vector<float> &diff_dst_data,
        const test::vector<float> &ref_diff_src, dnnl::impl::graph::dims &dims,
        const graph::op_kind_t op_kind, const std::string &op_name,
        const dnnl::impl::graph::dims &strides_fwd,
        const dnnl::impl::graph::dims &strides_diff_src,
        const dnnl::impl::graph::dims &strides_diff_dst,
        const std::map<dnnl::impl::graph::op_attr_t, attr_data_t> &attrs_data
        = {}) {
    static const std::set<graph::op_kind_t> with_support_for_use_dst {
            graph::op_kind::EluBackward,
            graph::op_kind::ClampBackward,
            graph::op_kind::ReLUBackward,
            graph::op_kind::SigmoidBackward,
            graph::op_kind::SqrtBackward,
            graph::op_kind::TanhBackward,
    };
    const test::vector<float> &fwd_data = fwd_data_pair.first;
    const bool is_fwd_data_src = fwd_data_pair.second;

    graph::op_t op(op_kind, op_name);
    for (const auto &attr_data : attrs_data) {
        op.set_attr<attr_data_t>(attr_data.first, attr_data.second);
    }
    if (with_support_for_use_dst.count(op_kind)) {
        op.set_attr<bool>(graph::op_attr::use_dst, !is_fwd_data_src);
    }

    test::vector<float> diff_src_data(ref_diff_src.size(), 0.0f);

    graph::logical_tensor_t fwd_data_lt = utils::logical_tensor_init(
            0, dims, strides_fwd, graph::data_type::f32);
    graph::logical_tensor_t diff_src_lt = utils::logical_tensor_init(
            1, dims, strides_diff_src, graph::data_type::f32);
    graph::logical_tensor_t diff_dst_lt = utils::logical_tensor_init(
            2, dims, strides_diff_dst, graph::data_type::f32);

    op.add_input(fwd_data_lt);
    op.add_input(diff_dst_lt);
    op.add_output(diff_src_lt);

    graph::engine_t *eng = get_engine();
    graph::graph_t g(eng->kind());
    g.add_op(&op);
    g.finalize();

    std::string pass_name = op_name + "_pass";
    graph::pass::pass_base_ptr apass = get_pass(pass_name);
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);
    std::vector<const graph::logical_tensor_t *> inputs {
            &fwd_data_lt, &diff_dst_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&diff_src_lt};
    p.compile(&cp, inputs, outputs, eng);

    graph::tensor_t fwd_data_ts(
            fwd_data_lt, eng, const_cast<float *>(fwd_data.data()));
    graph::tensor_t diff_dst_ts(
            diff_dst_lt, eng, const_cast<float *>(diff_dst_data.data()));
    graph::tensor_t diff_src_ts(diff_src_lt, eng, diff_src_data.data());

    std::vector<graph::tensor_t> input_ts {fwd_data_ts, diff_dst_ts};
    std::vector<graph::tensor_t> output_ts {diff_src_ts};
    graph::stream_t *strm = get_stream();
    cp.execute(strm, input_ts, output_ts);
    strm->wait();

    for (size_t i = 0; i < diff_src_data.size(); ++i) {
        ASSERT_NEAR(diff_src_data[i], ref_diff_src[i], 1e-5);
    }
}

TEST(Execute, ClampBackward) {
    test::vector<float> src {0.f, -1.f, 0.0723652f, -0.0364869f, 40.f, -50.f,
            0.188521f, -0.729739f, 88.371f};
    test::vector<float> dst {0.f, -1.f, 0.0723652f, -0.0364869f, 2.f, -1.f,
            0.188521f, -0.729739f, 2.f};
    test::vector<float> diff_dst {3.f, -7.f, 0.0194608f, -0.0559478f, 70.f, 0.f,
            0.754086f, -0.218955f, 88.5838f};
    test::vector<float> ref_diff_src {3.f, -0.f, 0.0194608f, -0.0559478f, 0.f,
            0.f, 0.754086f, -0.218955f, 0.f};

    graph::dims dims {1, 3, 3};
    graph::dims strides {9, 3, 1};
    const std::map<graph::op_attr_t, float> attrs_data {
            {graph::op_attr::min, -1.f}, {graph::op_attr::max, 2.f}};

    test_eltwise_bwd_common({src, true}, diff_dst, ref_diff_src, dims,
            graph::op_kind::ClampBackward, "clamp_bw", strides, strides,
            strides, attrs_data);
    test_eltwise_bwd_common({dst, false}, diff_dst, ref_diff_src, dims,
            graph::op_kind::ClampBackward, "clamp_bw", strides, strides,
            strides, attrs_data);
}

TEST(Execute, Elu) {
    test::vector<float> src {-2.0, -1.5, -1.0, -0.5, 0.0, 3.5};
    test::vector<float> ref_dst;

    for (size_t i = 0; i < src.size(); ++i) {
        float temp;
        if (src[i] > 0) {
            temp = src[i];
        } else {
            temp = static_cast<float>(exp(src[i]) - 1);
        }
        ref_dst.push_back(temp);
    }

    graph::dims dims {1, 2, 3};
    const std::map<graph::op_attr_t, float> attrs_data {
            {graph::op_attr::alpha, 1.f}};

    test_eltwise_common(
            src, ref_dst, dims, graph::op_kind::Elu, "elu", attrs_data);
}

TEST(Execute, EluBackward) {
    test::vector<float> src {0.f, -1.f, 0.0723652f, -0.0364869f, 40.f, -50.f,
            0.188521f, -0.729739f, 88.371f};
    test::vector<float> dst {0.f, -0.632121f, 0.0723652f, -0.0358293f, 40.f,
            -1.f, 0.188521f, -0.517965f, 88.371f};
    test::vector<float> diff_dst {3.f, -7.f, 0.0194608f, -0.0559478f, 70.f, 0.f,
            0.754086f, -0.218955f, 88.5838f};
    test::vector<float> ref_diff_src {3.f, -2.57516f, 0.0194608f, -0.0539432f,
            70.f, 0.f, 0.754086f, -0.105544f, 88.5838f};

    graph::dims dims {1, 3, 3};
    graph::dims strides {9, 3, 1};
    const std::map<graph::op_attr_t, float> attrs_data {
            {graph::op_attr::alpha, 1.f}};

    test_eltwise_bwd_common({src, true}, diff_dst, ref_diff_src, dims,
            graph::op_kind::EluBackward, "elu_bw", strides, strides, strides,
            attrs_data);
    test_eltwise_bwd_common({dst, false}, diff_dst, ref_diff_src, dims,
            graph::op_kind::EluBackward, "elu_bw", strides, strides, strides,
            attrs_data);
}

TEST(Execute, Exp) {
    test::vector<float> src {-2.0, -1.5, -1.0, -0.5, 0.0, 3.5};
    test::vector<float> ref_dst;
    for (size_t i = 0; i < src.size(); ++i) {
        float temp = static_cast<float>(exp(src[i]));
        ref_dst.push_back(temp);
    }

    dnnl::impl::graph::dims dims {1, 2, 3};

    test_eltwise_common(src, ref_dst, dims, graph::op_kind::Exp, "exp");
}

TEST(Execute, Gelu) {
    test::vector<float> src {
            -2.f, -1.5f, -1.f, -0.5f, 0.f, 0.5f, 1.f, 1.5f, 2.f};
    test::vector<float> ref_dst {-0.0455001f, -0.10021085f, -0.15865527f,
            -0.15426877f, 0.f, 0.3457312f, 0.84134465f, 1.399789f, 1.9544998f};

    dnnl::impl::graph::dims dims {1, 3, 3};

    test_eltwise_common(src, ref_dst, dims, graph::op_kind::GELU, "gelu");
}

TEST(Execute, GeluBackward) {
    test::vector<float> src {0.f, -1.f, 0.0723652f, -0.0364869f, 40.f, -50.f,
            0.188521f, -0.729739f, 88.371f};
    test::vector<float> diff_dst {3.f, -7.f, 0.0194608f, -0.0559478f, 70.f, 0.f,
            0.754086f, -0.218955f, 88.5838f};
    test::vector<float> ref_diff_src {1.5f, 0.583208f, 0.0108521f, -0.0263458f,
            70.f, 0.f, 0.489138f, -0.00212478f, 88.5838f};

    dnnl::impl::graph::dims dims {1, 3, 3};
    graph::dims strides {9, 3, 1};
    test_eltwise_bwd_common({src, true}, diff_dst, ref_diff_src, dims,
            graph::op_kind::GELUBackward, "gelu_bw", strides, strides, strides);
}

TEST(Execute, HardSigmoid) {
    float a = 1.0f / 6;
    float b = 0.5f;
    test::vector<float> src {-3.1f, -3.f, -1.5f, 0.f, 0.6f, 3.f, 3.1f, 100.f};
    test::vector<float> ref_dst = hardsigmoid_func(src, a, b);

    graph::dims dims {1, 2, 4};
    std::map<graph::op_attr_t, float> attrs_data {
            {graph::op_attr::alpha, a},
            {graph::op_attr::beta, b},
    };

    test_eltwise_common(src, ref_dst, dims, graph::op_kind::HardSigmoid,
            "hardsigmoid", attrs_data);
}

TEST(Execute, HardSigmoidBackward) {
    float a = 1.0f / 6;
    float b = 0.5f;
    const test::vector<float> src {
            -3.1f, -3.f, -1.5f, 0.f, 0.6f, 3.f, 3.1f, 100.f};
    const test::vector<float> diff_dst(8, 0.5f);
    test::vector<float> ref_diff_src
            = hardsigmoidbackward_func(src, diff_dst, a, b);

    graph::dims dims {1, 2, 4};
    graph::dims strides {8, 4, 1};
    std::map<graph::op_attr_t, float> attrs_data {
            {graph::op_attr::alpha, a},
            {graph::op_attr::beta, b},
    };

    test_eltwise_bwd_common({src, false}, diff_dst, ref_diff_src, dims,
            graph::op_kind::HardSigmoidBackward, "hardsigmoid_bw", strides,
            strides, strides, attrs_data);
}

TEST(Execute, HardSwish) {
    test::vector<float> src {7.f, -4.f, 0.0723652095f, -0.0364869386f, 60.f,
            -20.f, 0.188521415f, -0.729738772f, 88.3709564f};
    test::vector<float> ref_dst {7.f, 0.f, 0.0370553955f, -0.0180215873f, 60.f,
            0.f, 0.100184090f, -0.276116282f, 88.3709564f};
    dnnl::impl::graph::dims dims {1, 3, 3};

    test_eltwise_common(
            src, ref_dst, dims, graph::op_kind::HardSwish, "hardswish");
}

TEST(Execute, HardSwishBackward) {
    test::vector<float> src {7.f, -4.f, 0.0723652095f, -0.0364869386f, 60.f,
            -20.f, 0.188521415f, -0.729738772f, 88.3709564f};
    test::vector<float> diff_dst {9.f, -10.f, 0.0194608141f, -0.0559477545f,
            70.f, -20.f, 0.754085660f, -0.218955040f, 88.5838242f};
    test::vector<float> ref_diff_src {9.f, 0.f, 0.0101998346f, -0.0272934232f,
            70.f, 0.f, 0.424429923f, -0.0562175252f, 88.5838242f};

    dnnl::impl::graph::dims dims {1, 3, 3};
    graph::dims strides {9, 3, 1};
    test_eltwise_bwd_common({src, true}, diff_dst, ref_diff_src, dims,
            graph::op_kind::HardSwishBackward, "hardswish_bw", strides, strides,
            strides);
}

TEST(Execute, Relu) {
    test::vector<float> src {
            -2.f, -1.5f, -1.f, -0.5f, 0.f, 0.5f, 1.f, 1.5f, 2.f};
    test::vector<float> ref_dst {0.f, 0.f, 0.f, 0.f, 0.f, 0.5f, 1.f, 1.5f, 2.f};

    graph::dims dims {1, 3, 3};

    test_eltwise_common(src, ref_dst, dims, graph::op_kind::ReLU, "relu");
}

TEST(Execute, ReluBackward) {
    test::vector<float> src {0.f, -1.f, 0.0723652f, -0.0364869f, 40.f, -50.f,
            0.188521f, -0.729739f, 88.371f};
    test::vector<float> dst {
            0.f, -0.f, 0.0723652f, -0.f, 40.f, -0.f, 0.188521f, -0.f, 88.371f};
    test::vector<float> diff_dst {3.f, -7.f, 0.0194608f, -0.0559478f, 70.f, 0.f,
            0.754086f, -0.218955f, 88.5838f};
    test::vector<float> ref_diff_src {
            0.f, -0.f, 0.0194608f, -0.f, 70.f, 0.f, 0.754086f, -0.f, 88.5838f};

    dnnl::impl::graph::dims dims {1, 3, 3};
    graph::dims strides {9, 3, 1};
    test_eltwise_bwd_common({src, true}, diff_dst, ref_diff_src, dims,
            graph::op_kind::ReLUBackward, "relu_bw", strides, strides, strides);
    test_eltwise_bwd_common({dst, false}, diff_dst, ref_diff_src, dims,
            graph::op_kind::ReLUBackward, "relu_bw", strides, strides, strides);
    graph::dims strides_diff {9, 1, 3};
    diff_dst = {3.f, -0.0559478f, 0.754086f, -7.f, 70.f, -0.218955f, 0.0194608f,
            0.f, 88.5838f};
    test_eltwise_bwd_common({src, true}, diff_dst, ref_diff_src, dims,
            graph::op_kind::ReLUBackward, "relu_bw", strides, strides,
            strides_diff);
    test_eltwise_bwd_common({dst, false}, diff_dst, ref_diff_src, dims,
            graph::op_kind::ReLUBackward, "relu_bw", strides, strides,
            strides_diff);
    ref_diff_src = {
            0.f, -0.f, 0.754086f, -0.f, 70.f, 0.f, 0.0194608f, -0.f, 88.5838f};
    test_eltwise_bwd_common({src, true}, diff_dst, ref_diff_src, dims,
            graph::op_kind::ReLUBackward, "relu_bw", strides, strides_diff,
            strides_diff);
    test_eltwise_bwd_common({dst, false}, diff_dst, ref_diff_src, dims,
            graph::op_kind::ReLUBackward, "relu_bw", strides, strides_diff,
            strides_diff);
}

TEST(Execute, Sigmoid) {
    test::vector<float> src {-5.f, -2.5f, -1.f, 0.3f, 0.f, 1.2f};
    test::vector<float> ref_dst = sigmoid_func(src);

    dnnl::impl::graph::dims dims {1, 2, 3};

    test_eltwise_common(src, ref_dst, dims, graph::op_kind::Sigmoid, "sigmoid");
}

TEST(Execute, SigmoidBackward) {
    test::vector<float> src {0.f, -1.f, 0.0723652f, -0.0364869f, 40.f, -50.f,
            0.188521f, -0.729739f, 88.371f};
    test::vector<float> dst {0.5f, 0.268941f, 0.518083f, 0.490879f, 1.f,
            1.92875e-22f, 0.546991f, 0.325252f, 1.f};
    test::vector<float> diff_dst {3.f, -7.f, 0.0194608f, -0.0559478f, 70.f, 0.f,
            0.754086f, -0.218955f, 88.5838f};
    test::vector<float> ref_diff_src {0.75f, -1.37628f, 0.00485884f,
            -0.0139823f, 0.f, 0.f, 0.186856f, -0.0480526f, 0.f};

    dnnl::impl::graph::dims dims {1, 3, 3};
    graph::dims strides {9, 3, 1};
    test_eltwise_bwd_common({src, true}, diff_dst, ref_diff_src, dims,
            graph::op_kind::SigmoidBackward, "sigmoid_bw", strides, strides,
            strides);
    test_eltwise_bwd_common({dst, false}, diff_dst, ref_diff_src, dims,
            graph::op_kind::SigmoidBackward, "sigmoid_bw", strides, strides,
            strides);
}

TEST(Execute, SoftPlus) {
    test::vector<float> src {0.f, -1.f, 0.0723652f, -0.0364869f, 40.f, -25.f,
            0.188521f, -0.729739f, 44.317f};
    test::vector<float> ref_dst_case1 = {-0.693147f, -1.31326f, -0.657619f,
            -0.711557f, -0.f, -25.f, -0.603322f, -1.12315f, -0.f};
    test::vector<float> ref_dst_case2 = {0.693147f, 0.313262f, 0.729984f,
            0.67507f, 40.f, 0.f, 0.791844f, 0.393416f, 44.317f};
    test::vector<float> ref_dst_case3 = {-0.346574f, -1.06346f, -0.311699f,
            -0.36515f, -0.f, -25.f, -0.261146f, -0.834203f, -0.f};
    test::vector<float> ref_dst_case4 = {0.346574f, 0.063464f, 0.384064f,
            0.328663f, 40.f, 0.f, 0.449667f, 0.104465f, 44.317f};

    dnnl::impl::graph::dims dims {1, 3, 3};
    const std::map<graph::op_attr_t, float> attrs_data_case1 {
            {graph::op_attr::beta, -1.f}};
    const std::map<graph::op_attr_t, float> attrs_data_case2 {
            {graph::op_attr::beta, 1.f}};
    const std::map<graph::op_attr_t, float> attrs_data_case3 {
            {graph::op_attr::beta, -2.f}};
    const std::map<graph::op_attr_t, float> attrs_data_case4 {
            {graph::op_attr::beta, 2.f}};

    test_eltwise_common(src, ref_dst_case1, dims, graph::op_kind::SoftPlus,
            "softplus", attrs_data_case1);
    test_eltwise_common(src, ref_dst_case2, dims, graph::op_kind::SoftPlus,
            "softplus", attrs_data_case2);
    test_eltwise_common(src, ref_dst_case3, dims, graph::op_kind::SoftPlus,
            "softplus", attrs_data_case3);
    test_eltwise_common(src, ref_dst_case4, dims, graph::op_kind::SoftPlus,
            "softplus", attrs_data_case4);
}

TEST(Execute, SoftPlusBackward) {
    test::vector<float> src {0.f, -1.f, 0.0723652f, -0.0364869f, 40.f, -50.f,
            0.188521f, -0.729739f, 88.371f};
    test::vector<float> diff_dst {3.f, -7.f, 0.0194608f, -0.0559478f, 70.f, 0.f,
            0.754086f, -0.218955f, 88.5838f};
    test::vector<float> ref_diff_src_case1 {1.5f, -5.11741f, 0.00937849f,
            -0.0284842f, 2.97385e-16f, 0.f, 0.341607f, -0.147739f, 0.f};
    test::vector<float> ref_diff_src_case2 {1.5f, -1.88259f, 0.0100823f,
            -0.0274636f, 70.f, 0.f, 0.412478f, -0.0712156f, 88.5838f};
    test::vector<float> ref_diff_src_case3 {1.5f, -6.16558f, 0.00902748f,
            -0.0289941f, 1.2634e-33f, 0.f, 0.306793f, -0.177672f, 0.f};
    test::vector<float> ref_diff_src_case4 {1.5f, -0.83442f, 0.0104333f,
            -0.0269537f, 70.f, 0.f, 0.447293f, -0.0412833f, 88.5838f};

    dnnl::impl::graph::dims dims {1, 3, 3};
    graph::dims strides {9, 3, 1};
    const std::map<graph::op_attr_t, float> attrs_data_case1 {
            {graph::op_attr::beta, -1.f}};
    const std::map<graph::op_attr_t, float> attrs_data_case2 {
            {graph::op_attr::beta, 1.f}};
    const std::map<graph::op_attr_t, float> attrs_data_case3 {
            {graph::op_attr::beta, -2.f}};
    const std::map<graph::op_attr_t, float> attrs_data_case4 {
            {graph::op_attr::beta, 2.f}};

    test_eltwise_bwd_common({src, true}, diff_dst, ref_diff_src_case1, dims,
            graph::op_kind::SoftPlusBackward, "softplus_bw", strides, strides,
            strides, attrs_data_case1);
    test_eltwise_bwd_common({src, true}, diff_dst, ref_diff_src_case2, dims,
            graph::op_kind::SoftPlusBackward, "softplus_bw", strides, strides,
            strides, attrs_data_case2);
    test_eltwise_bwd_common({src, true}, diff_dst, ref_diff_src_case3, dims,
            graph::op_kind::SoftPlusBackward, "softplus_bw", strides, strides,
            strides, attrs_data_case3);
    test_eltwise_bwd_common({src, true}, diff_dst, ref_diff_src_case4, dims,
            graph::op_kind::SoftPlusBackward, "softplus_bw", strides, strides,
            strides, attrs_data_case4);
}

TEST(Execute, Sqrt) {
    test::vector<float> src {2.f, 1.5f, 1.0f, 0.5f, 0.f, 3.5f};
    test::vector<float> ref_dst = sqrt_func(src);

    dnnl::impl::graph::dims dims {1, 2, 3};

    test_eltwise_common(src, ref_dst, dims, graph::op_kind::Sqrt, "sqrt");
}

TEST(Execute, SqrtBackward) {
    test::vector<float> src {1.f, 0.0262164f, 30.f, 0.424328f, 88.1054f,
            22.5865f, 44.2676f, 0.0748657f, 0.0429739f};
    test::vector<float> dst {1.f, 0.161915f, 5.47723f, 0.651405f, 9.38645f,
            4.75252f, 6.65339f, 0.273616f, 0.207301f};
    test::vector<float> diff_dst {4.f, 0.0748657f, 10.f, 0.597313f, 88.1216f,
            22.446f, 44.7703f, 0.0294627f, 0.0618955f};
    test::vector<float> ref_diff_src {2.f, 0.231188f, 0.912871f, 0.458481f,
            4.69409f, 2.36148f, 3.36447f, 0.0538395f, 0.149289f};

    dnnl::impl::graph::dims dims {1, 3, 3};
    graph::dims strides {9, 3, 1};
    test_eltwise_bwd_common({src, true}, diff_dst, ref_diff_src, dims,
            graph::op_kind::SqrtBackward, "sqrt_bw", strides, strides, strides);
    test_eltwise_bwd_common({dst, false}, diff_dst, ref_diff_src, dims,
            graph::op_kind::SqrtBackward, "sqrt_bw", strides, strides, strides);
}

TEST(Execute, Square) {
    test::vector<float> src {-2.f, -1.5f, -1.f, -0.5f, 0.f, 3.5f};
    test::vector<float> ref_dst;
    for (size_t i = 0; i < src.size(); ++i) {
        float temp = src[i] * src[i];
        ref_dst.push_back(temp);
    }

    dnnl::impl::graph::dims dims {1, 2, 3};

    test_eltwise_common(src, ref_dst, dims, graph::op_kind::Square, "square");
}

TEST(Execute, Log) {
    test::vector<float> src {2.f, 1.5f, 1.f, 0.5f, 0.8f, 3.5f};
    test::vector<float> ref_dst;
    for (size_t i = 0; i < src.size(); ++i) {
        float temp = static_cast<float>(log(src[i]));
        ref_dst.push_back(temp);
    }

    dnnl::impl::graph::dims dims {1, 2, 3};

    test_eltwise_common(src, ref_dst, dims, graph::op_kind::Log, "log");
}

TEST(Execute, Tanh) {
    test::vector<float> src {-2.f, -1.5f, -1.f, -0.5f, 0.f, 3.5f};
    test::vector<float> ref_dst = tanh_func(src);

    dnnl::impl::graph::dims dims {1, 2, 3};

    test_eltwise_common(src, ref_dst, dims, graph::op_kind::Tanh, "tanh");
}

TEST(Execute, TanhBackward) {
    test::vector<float> src {0.f, -1.f, 0.0723652f, -0.0364869f, 40.f, -50.f,
            0.188521f, -0.729739f, 88.371f};
    test::vector<float> dst {0.f, -0.761594f, 0.0722392f, -0.0364708f, 1.f,
            -1.f, 0.186319f, -0.622906f, 1.f};
    test::vector<float> diff_dst {3.f, -7.f, 0.0194608f, -0.0559478f, 70.f, 0.f,
            0.754086f, -0.218955f, 88.5838f};
    test::vector<float> ref_diff_src {3.f, -2.93982f, 0.0193593f, -0.0558733f,
            0.f, 0.f, 0.727908f, -0.133998f, 0.f};

    dnnl::impl::graph::dims dims {1, 3, 3};
    graph::dims strides {9, 3, 1};
    test_eltwise_bwd_common({src, true}, diff_dst, ref_diff_src, dims,
            graph::op_kind::TanhBackward, "tanh_bw", strides, strides, strides);
    test_eltwise_bwd_common({dst, false}, diff_dst, ref_diff_src, dims,
            graph::op_kind::TanhBackward, "tanh_bw", strides, strides, strides);
}

TEST(ExecuteSubgraphFp32, Shuffle) {
    graph::engine_t *engine = get_engine();
    graph::stream_t *strm = get_stream();

    const std::vector<std::pair<size_t, test::vector<float>>> configs {
            {1,
                    {1.f, 1.f, 2.f, 2.f, 5.f, 5.f, 6.f, 6.f, 3.f, 3.f, 4.f, 4.f,
                            7.f, 7.f, 8.f, 8.f}},
            {3,
                    {1.f, 2.f, 1.f, 2.f, 3.f, 4.f, 3.f, 4.f, 5.f, 6.f, 5.f, 6.f,
                            7.f, 8.f, 7.f, 8.f}}};
    const std::vector<int64_t> base_shape {1, 2, 2, 2};
    const int64_t g = 2;

    for (const auto &config : configs) {
        const auto axis = config.first;
        const auto &expected_data = config.second;
        auto configured_shape = base_shape;
        configured_shape[axis] = 4;

        std::vector<int64_t> reshape0_src_shape = configured_shape;
        std::vector<int64_t> reshape1_dst_shape = configured_shape;

        std::vector<int64_t> reshape0_dst_shape = configured_shape;
        reshape0_dst_shape[axis] /= g;
        reshape0_dst_shape.insert(reshape0_dst_shape.begin() + axis + 1, g);

        std::vector<int64_t> transpose_dst_shape = reshape0_dst_shape;
        std::swap(transpose_dst_shape[axis], transpose_dst_shape[axis + 1]);

        std::vector<int64_t> order(transpose_dst_shape.size());
        std::iota(order.begin(), order.end(), 0);
        std::swap(order[axis], order[axis + 1]);

        test::vector<float> src_data {1.f, 1.f, 2.f, 2.f, 3.f, 3.f, 4.f, 4.f,
                5.f, 5.f, 6.f, 6.f, 7.f, 7.f, 8.f, 8.f};

        graph::op_t reshape0 {0, graph::op_kind::StaticReshape, "reshape0"};
        reshape0.set_attr(graph::op_attr::shape, reshape0_dst_shape);
        reshape0.set_attr(graph::op_attr::special_zero, false);

        graph::op_t transpose {1, graph::op_kind::StaticTranspose, "transpose"};
        transpose.set_attr(graph::op_attr::order, order);

        graph::op_t reshape1 {2, graph::op_kind::StaticReshape, "reshape1"};
        reshape1.set_attr(graph::op_attr::shape, reshape1_dst_shape);
        reshape1.set_attr(graph::op_attr::special_zero, false);

        graph::logical_tensor_t reshape0_src = utils::logical_tensor_init(
                0, reshape0_src_shape, graph::data_type::f32);
        graph::logical_tensor_t reshape0_dst = utils::logical_tensor_init(
                1, reshape0_dst_shape, graph::data_type::f32);
        graph::logical_tensor_t transpose_dst = utils::logical_tensor_init(
                2, transpose_dst_shape, graph::data_type::f32);
        graph::logical_tensor_t reshape1_dst = utils::logical_tensor_init(
                3, reshape1_dst_shape, graph::data_type::f32);

        reshape0.add_input(reshape0_src);
        reshape0.add_output(reshape0_dst);

        transpose.add_input(reshape0_dst);
        transpose.add_output(transpose_dst);

        reshape1.add_input(transpose_dst);
        reshape1.add_output(reshape1_dst);

        graph::graph_t agraph(engine->kind());
        agraph.add_op(&reshape0);
        agraph.add_op(&transpose);
        agraph.add_op(&reshape1);
        agraph.finalize();

        graph::pass::pass_base_ptr apass = get_pass("shuffle_fusion");
        apass->run(agraph);
        ASSERT_EQ(agraph.get_num_partitions(), 1U);
        auto part = agraph.get_partitions()[0];

        graph::partition_t p;
        p.init(part);

        graph::compiled_partition_t cp(p);

        std::vector<const graph::logical_tensor_t *> lt_ins {&reshape0_src};
        std::vector<const graph::logical_tensor_t *> lt_outs {&reshape1_dst};

        p.compile(&cp, lt_ins, lt_outs, engine);

        test::vector<float> dst_data(product(reshape1_dst_shape));
        graph::tensor_t reshape0_src_ts(reshape0_src, engine, src_data.data());
        graph::tensor_t reshape1_dst_ts(reshape1_dst, engine, dst_data.data());

        cp.execute(strm, {reshape0_src_ts}, {reshape1_dst_ts});
        strm->wait();

        for (size_t i = 0; i < expected_data.size(); ++i) {
            ASSERT_FLOAT_EQ(expected_data[i], dst_data[i]);
        }
    }
}

TEST(ExecuteSubgraphFp32, ReciprocalMul) {
    graph::engine_t *eng = get_engine();

    test::vector<float> src0_data {1.f, 2.f, 3.f, 10.f, 20.f, 30.f};
    test::vector<float> src1_data {2.f, 2.f, 2.f, 5.f, 5.f, 5.f};
    test::vector<float> ref_dst_data {0.5f, 1.f, 1.5f, 2.f, 4.f, 6.f};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    graph::op_t reciprocal_op {0, graph::op_kind::Reciprocal, "reciprocal"};

    graph::op_t mul_op {1, graph::op_kind::Multiply, "mul"};

    // prepare logical tensor
    graph::logical_tensor_t src0
            = utils::logical_tensor_init(0, {1, 2, 3}, graph::data_type::f32);
    graph::logical_tensor_t src1
            = utils::logical_tensor_init(1, {1, 2, 3}, graph::data_type::f32);
    graph::logical_tensor_t reciprocal_dst = utils::logical_tensor_init(
            3, {1, 2, 3}, graph::data_type::f32, graph::layout_type::any);
    graph::logical_tensor_t mul_dst = utils::logical_tensor_init(
            4, {1, 2, 3}, graph::data_type::f32, graph::layout_type::any);

    reciprocal_op.add_input(src1);
    reciprocal_op.add_output(reciprocal_dst);
    mul_op.add_input(src0);
    mul_op.add_input(reciprocal_dst);
    mul_op.add_output(mul_dst);

    graph::graph_t g(eng->kind());
    ASSERT_EQ(g.add_op(&reciprocal_op), graph::status::success);
    ASSERT_EQ(g.add_op(&mul_op), graph::status::success);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("reciprocal_multiply_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];
    graph::partition_t p;
    p.init(part);

    // compile
    std::vector<const graph::logical_tensor_t *> inputs {&src0, &src1};
    std::vector<const graph::logical_tensor_t *> outputs {&mul_dst};

    graph::compiled_partition_t cp(p);
    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);

    graph::logical_tensor_t lt;
    cp.query_logical_tensor(mul_dst.id, &lt);
    ASSERT_EQ(lt.layout_type, graph::layout_type::strided);

    graph::tensor_t src0_ts(src0, eng, src0_data.data());
    graph::tensor_t src1_ts(src1, eng, src1_data.data());
    graph::tensor_t dst_ts(lt, eng, dst_data.data());

    graph::stream_t *strm = get_stream();
    ASSERT_EQ(cp.execute(strm, {src0_ts, src1_ts}, {dst_ts}),
            graph::status::success);
    strm->wait();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(Execute, Sum) {
    /*
        input0  input1
          \       /
             Add    input2
               \      /
                  Add    input3
                    \     /
                      Add     input4
                        \      /
                           Add
    */
    graph::engine_t *engine = get_engine();

    std::vector<int64_t> input_dims {1, 3, 3};

    graph::op_t add0 {0, graph::op_kind::Add, "add_0"};
    add0.set_attr<std::string>(graph::op_attr::auto_broadcast, "none");
    graph::logical_tensor_t input0
            = utils::logical_tensor_init(0, input_dims, graph::data_type::f32);
    graph::logical_tensor_t input1
            = utils::logical_tensor_init(1, input_dims, graph::data_type::f32);
    graph::logical_tensor_t output0
            = utils::logical_tensor_init(2, input_dims, graph::data_type::f32);

    graph::op_t add1 {1, graph::op_kind::Add, "add_1"};
    add1.set_attr<std::string>(graph::op_attr::auto_broadcast, "none");
    graph::logical_tensor_t input2
            = utils::logical_tensor_init(3, input_dims, graph::data_type::f32);
    graph::logical_tensor_t output1
            = utils::logical_tensor_init(4, input_dims, graph::data_type::f32);

    graph::op_t add2 {2, graph::op_kind::Add, "add_2"};
    add2.set_attr<std::string>(graph::op_attr::auto_broadcast, "none");
    graph::logical_tensor_t input3
            = utils::logical_tensor_init(5, input_dims, graph::data_type::f32);
    graph::logical_tensor_t output2
            = utils::logical_tensor_init(6, input_dims, graph::data_type::f32);

    graph::op_t add3 {3, graph::op_kind::Add, "add_3"};
    add3.set_attr<std::string>(graph::op_attr::auto_broadcast, "none");
    graph::logical_tensor_t input4
            = utils::logical_tensor_init(7, input_dims, graph::data_type::f32);
    graph::logical_tensor_t output3
            = utils::logical_tensor_init(8, input_dims, graph::data_type::f32);

    add0.add_input(input0);
    add0.add_input(input1);
    add0.add_output(output0);

    add1.add_input(output0);
    add1.add_input(input2);
    add1.add_output(output1);

    add2.add_input(output1);
    add2.add_input(input3);
    add2.add_output(output2);

    add3.add_input(output2);
    add3.add_input(input4);
    add3.add_output(output3);

    graph::graph_t agraph(engine->kind());

    ASSERT_EQ(agraph.add_op(&add0), graph::status::success);
    ASSERT_EQ(agraph.add_op(&add1), graph::status::success);
    ASSERT_EQ(agraph.add_op(&add2), graph::status::success);
    ASSERT_EQ(agraph.add_op(&add3), graph::status::success);
    agraph.finalize();
    graph::pass::pass_base_ptr apass = get_pass("sum_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    auto part = agraph.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {
            &input0, &input1, &input2, &input3, &input4};
    std::vector<const graph::logical_tensor_t *> outputs {&output3};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, engine), graph::status::success);

    test::vector<float> input0_data(product(input_dims), 1);
    test::vector<float> input1_data(product(input_dims), 1);
    test::vector<float> input2_data(product(input_dims), 1);
    test::vector<float> input3_data(product(input_dims), 1);
    test::vector<float> input4_data(product(input_dims), 1);
    test::vector<float> output_data(product(input_dims), 0);

    graph::tensor_t input0_ts(input0, engine, input0_data.data());
    graph::tensor_t input1_ts(input1, engine, input1_data.data());
    graph::tensor_t input2_ts(input2, engine, input2_data.data());
    graph::tensor_t input3_ts(input3, engine, input3_data.data());
    graph::tensor_t input4_ts(input4, engine, input4_data.data());
    graph::tensor_t output_ts(output3, engine, output_data.data());

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {input0_ts, input1_ts, input2_ts, input3_ts, input4_ts},
            {output_ts});
    strm->wait();

    for (const auto v : output_data) {
        ASSERT_FLOAT_EQ(v, 5.f);
    }
}

TEST(Compile, EltwiseGetInplacePair) {
    // TODO(qun): re-enable this test once library and bridge align the inplace
    // logic
    SKIP_IF(true, "library and bridge have different inplace logic");

    graph::engine_t *eng = get_engine();

    graph::op_t eltwise_op(graph::op_kind::Tanh);

    // prepare logical tensor
    graph::logical_tensor_t src
            = utils::logical_tensor_init(0, {2, 3}, graph::data_type::f32);
    graph::logical_tensor_t dst
            = utils::logical_tensor_init(1, {2, 3}, graph::data_type::f32);

    eltwise_op.add_input(src);
    eltwise_op.add_output(dst);

    graph::graph_t g(eng->kind());
    g.add_op(&eltwise_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("tanh_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];
    graph::partition_t p;
    p.init(part);

    // compile
    std::vector<const graph::logical_tensor_t *> inputs {&src};
    std::vector<const graph::logical_tensor_t *> outputs {&dst};

    graph::compiled_partition_t cp(p);
    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);

    auto pairs = cp.get_inplace_pairs();
    ASSERT_EQ(pairs.size(), 1U);
    ASSERT_EQ(pairs[0].input_id, 0U);
    ASSERT_EQ(pairs[0].output_id, 1U);
}

struct eltwise_binary_params_t {
    graph::op_kind_t eltwise_kind;
    graph::op_kind_t binary_kind;
    std::vector<graph::dim_t> binary_src_shape;
    bool swap;
};

class eltwise_binary_t
    : public ::testing::TestWithParam<eltwise_binary_params_t> {
public:
    void TestEltwiseBinary() {
        const auto params
                = ::testing::TestWithParam<eltwise_binary_params_t>::GetParam();
        graph::engine_t *eng = get_engine();

        std::vector<graph::dim_t> binary_src_shape = params.binary_src_shape;
        test::vector<float> src {-2.f, -1.5f, 1.f, 0.5f};
        test::vector<float> binary_src(product(binary_src_shape));
        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
        std::generate(binary_src.begin(), binary_src.end(),
                [&]() { return f32_distribution(generator); });
        test::vector<float> dst {0.f, 0.f, 0.f, 0.f};

        graph::op_t eltwise_op(0, params.eltwise_kind, "elt");
        if (params.eltwise_kind == graph::op_kind::Elu) {
            eltwise_op.set_attr<float>(graph::op_attr::alpha, 1.f);
        } else if (params.eltwise_kind == graph::op_kind::Clamp) {
            eltwise_op.set_attr<float>(graph::op_attr::min, -1.f);
            eltwise_op.set_attr<float>(graph::op_attr::max, 2.f);
        }
        graph::op_t binary_op(1, params.binary_kind, "binary");

        graph::logical_tensor_t eltwise_src_lt = utils::logical_tensor_init(
                0, {1, 1, 2, 2}, graph::data_type::f32);
        graph::logical_tensor_t eltwise_dst_lt = utils::logical_tensor_init(1,
                {1, 1, 2, 2}, graph::data_type::f32, graph::layout_type::any);
        graph::logical_tensor_t binary_src_lt = utils::logical_tensor_init(
                2, binary_src_shape, graph::data_type::f32);
        graph::logical_tensor_t binary_dst_lt = utils::logical_tensor_init(3,
                {1, 1, 2, 2}, graph::data_type::f32, graph::layout_type::any);

        eltwise_op.add_input(eltwise_src_lt);
        eltwise_op.add_output(eltwise_dst_lt);
        if (params.swap) {
            binary_op.add_input(binary_src_lt);
            binary_op.add_input(eltwise_dst_lt);

        } else {
            binary_op.add_input(eltwise_dst_lt);
            binary_op.add_input(binary_src_lt);
        }
        binary_op.add_output(binary_dst_lt);

        graph::graph_t g(eng->kind());
        g.add_op(&eltwise_op);
        g.add_op(&binary_op);
        g.finalize();

        graph::pass::pass_base_ptr apass = get_pass("eltwise_binary_fusion");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        // compile
        graph::partition_t p;
        p.init(part);

        graph::compiled_partition_t cp(p);

        std::vector<const graph::logical_tensor_t *> inputs {
                &eltwise_src_lt, &binary_src_lt};

        std::vector<const graph::logical_tensor_t *> outputs {&binary_dst_lt};

        ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), graph::status::success);

        graph::logical_tensor_t lt;
        cp.query_logical_tensor(binary_dst_lt.id, &lt);

        ASSERT_EQ(lt.layout_type, graph::layout_type::strided);

        graph::tensor_t src_ts(eltwise_src_lt, eng, src.data());
        graph::tensor_t binary_src_ts(binary_src_lt, eng, binary_src.data());
        graph::tensor_t binary_dst_ts(binary_dst_lt, eng, dst.data());

        graph::stream_t *strm = get_stream();

        ASSERT_EQ(cp.execute(strm, {src_ts, binary_src_ts}, {binary_dst_ts}),
                graph::status::success);
        strm->wait();
    }
};

TEST_P(eltwise_binary_t, TestEltwiseBinary) {
    TestEltwiseBinary();
}

INSTANTIATE_TEST_SUITE_P(Execute, eltwise_binary_t,
        ::testing::Values(
                // with broadcast add and no swap inputs
                eltwise_binary_params_t {
                        graph::op_kind::ReLU, graph::op_kind::Add, {1}, false},
                // with broadcast add and swap inputs
                eltwise_binary_params_t {
                        graph::op_kind::ReLU, graph::op_kind::Add, {1}, true},
                // no broadcast add and no swap inputs
                eltwise_binary_params_t {graph::op_kind::ReLU,
                        graph::op_kind::Add, {1, 1, 2, 2}, false},
                // no broadcast add and swap inputs
                eltwise_binary_params_t {graph::op_kind::ReLU,
                        graph::op_kind::Add, {1, 1, 2, 2}, true},
                eltwise_binary_params_t {graph::op_kind::Clamp,
                        graph::op_kind::Multiply, {1}, false},
                eltwise_binary_params_t {graph::op_kind::Round,
                        graph::op_kind::Maximum, {1, 1, 2, 2}, false},
                eltwise_binary_params_t {graph::op_kind::Sigmoid,
                        graph::op_kind::Divide, {1}, false},
                eltwise_binary_params_t {graph::op_kind::SoftPlus,
                        graph::op_kind::Minimum, {1, 1, 2, 2}, false},
                eltwise_binary_params_t {graph::op_kind::Exp,
                        graph::op_kind::Minimum, {1, 1, 2, 2}, false},
                eltwise_binary_params_t {graph::op_kind::Elu,
                        graph::op_kind::Subtract, {1, 1, 2, 2}, false},
                eltwise_binary_params_t {graph::op_kind::Sqrt,
                        graph::op_kind::Minimum, {1}, false},
                eltwise_binary_params_t {graph::op_kind::HardSwish,
                        graph::op_kind::Minimum, {1}, false},
                eltwise_binary_params_t {
                        graph::op_kind::Tanh, graph::op_kind::Add, {1}, true}));
