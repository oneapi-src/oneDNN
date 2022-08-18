/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

#include "graph/unit/backend/dnnl/dnnl_test_common.hpp"
#include "graph/unit/backend/dnnl/ref_func.hpp"

template <typename attr_data_t = float>
static inline void test_eltwise_common(test::vector<float> &src,
        test::vector<float> &ref_dst, dnnl::impl::graph::dims &dims,
        const dnnl_graph_op_kind_t op_kind, const std::string &op_name,
        const std::map<dnnl::impl::graph::op_attr_t, attr_data_t> &attrs_data
        = {}) {
    impl::engine_t *eng = get_engine();
    impl::op_t op(op_kind, op_name);

    test::vector<float> dst(src.size(), 0.0);

    for (const auto &attr_data : attrs_data) {
        op.set_attr<attr_data_t>(attr_data.first, attr_data.second);
    }

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, dims, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, dims, impl::data_type::f32, impl::layout_type::any);

    op.add_input(src_lt);
    op.add_output(dst_lt);

    impl::graph_t g(eng->kind());
    g.add_op(&op);
    g.finalize();

    std::string pass_name = op_name + "_pass";

    impl::pass::pass_base_ptr apass = get_pass(pass_name);
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};

    p.compile(&cp, inputs, outputs, eng);

    impl::logical_tensor_t lt;
    cp.query_logical_tensor(dst_lt.id, &lt);

    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, eng, src.data());
    impl::tensor_t dst_ts(dst_lt, eng, dst.data());

    impl::stream_t *strm = get_stream();

    cp.execute(strm, {src_ts}, {dst_ts});
    strm->wait();

    if (op_kind == impl::op_kind::Log || op_kind == impl::op_kind::GELU
            || op_kind == impl::op_kind::SoftPlus) {
        for (size_t i = 0; i < src.size(); ++i) {
            ASSERT_NEAR(dst[i], ref_dst[i], 1e-5);
        }
    } else {
        for (size_t i = 0; i < src.size(); ++i) {
            ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
        }
    }
}

TEST(Execute, Abs) {
    test::vector<float> src {-2.0, -1.5, -1.0, -0.5, 0.0, 3.5};
    test::vector<float> ref_dst {2.0, 1.5, 1.0, 0.5, 0.0, 3.5};

    dnnl::impl::graph::dims dims {1, 2, 3};

    test_eltwise_common(src, ref_dst, dims, impl::op_kind::Abs, "abs");
}

TEST(Execute, Round) {
    test::vector<float> src {1.1f, -2.3f, 4.7f, 1.0f, 0.0f, -8.34f};
    test::vector<float> ref_dst = round_func(src);

    dnnl::impl::graph::dims dims {1, 2, 3};

    test_eltwise_common(src, ref_dst, dims, impl::op_kind::Round, "round");
}

TEST(Execute, Clamp) {
    test::vector<float> src {-2.0, -1.5, -1.0, -0.5, 0.0, 3.5};
    test::vector<float> ref_dst {-1.0, -1.0, -1.0, -0.5, 0.0, 2.0};

    impl::dims dims {1, 2, 3};
    const std::map<impl::op_attr_t, float> attrs_data {
            {impl::op_attr::min, -1.f}, {impl::op_attr::max, 2.f}};

    test_eltwise_common(
            src, ref_dst, dims, impl::op_kind::Clamp, "clamp", attrs_data);
}

template <typename attr_data_t = float>
static inline void test_eltwise_bwd_common(
        const std::pair<test::vector<float>, bool> &fwd_data_pair,
        const test::vector<float> &diff_dst_data,
        const test::vector<float> &ref_diff_src, dnnl::impl::graph::dims &dims,
        const dnnl_graph_op_kind_t op_kind, const std::string &op_name,
        const std::map<dnnl::impl::graph::op_attr_t, attr_data_t> &attrs_data
        = {}) {
    static const std::set<dnnl_graph_op_kind_t> with_support_for_use_dst {
            impl::op_kind::EluBackprop, impl::op_kind::ClampBackprop,
            impl::op_kind::ReLUBackprop, impl::op_kind::SigmoidBackprop,
            impl::op_kind::SqrtBackprop, impl::op_kind::TanhBackprop};
    const test::vector<float> &fwd_data = fwd_data_pair.first;
    const bool is_fwd_data_src = fwd_data_pair.second;

    impl::op_t op(op_kind, op_name);
    for (const auto &attr_data : attrs_data) {
        op.set_attr<attr_data_t>(attr_data.first, attr_data.second);
    }
    if (with_support_for_use_dst.count(op_kind)) {
        op.set_attr<bool>(impl::op_attr::use_dst, !is_fwd_data_src);
    }

    test::vector<float> diff_src_data(ref_diff_src.size(), 0.0f);

    impl::logical_tensor_t fwd_data_lt
            = utils::logical_tensor_init(0, dims, impl::data_type::f32);
    impl::logical_tensor_t diff_src_lt = utils::logical_tensor_init(
            1, dims, impl::data_type::f32, impl::layout_type::any);
    impl::logical_tensor_t diff_dst_lt
            = utils::logical_tensor_init(2, dims, impl::data_type::f32);

    op.add_input(fwd_data_lt);
    op.add_input(diff_dst_lt);
    op.add_output(diff_src_lt);

    impl::engine_t *eng = get_engine();
    impl::graph_t g(eng->kind());
    g.add_op(&op);
    g.finalize();

    std::string pass_name = op_name + "_pass";
    impl::pass::pass_base_ptr apass = get_pass(pass_name);
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);
    std::vector<const impl::logical_tensor_t *> inputs {
            &fwd_data_lt, &diff_dst_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&diff_src_lt};
    p.compile(&cp, inputs, outputs, eng);

    impl::tensor_t fwd_data_ts(
            fwd_data_lt, eng, const_cast<float *>(fwd_data.data()));
    impl::tensor_t diff_dst_ts(
            diff_dst_lt, eng, const_cast<float *>(diff_dst_data.data()));
    impl::tensor_t diff_src_ts(diff_src_lt, eng, diff_src_data.data());

    std::vector<impl::tensor_t> input_ts {fwd_data_ts, diff_dst_ts};
    std::vector<impl::tensor_t> output_ts {diff_src_ts};
    impl::stream_t *strm = get_stream();
    cp.execute(strm, input_ts, output_ts);
    strm->wait();

    for (size_t i = 0; i < diff_src_data.size(); ++i) {
        ASSERT_NEAR(diff_src_data[i], ref_diff_src[i], 1e-5);
    }
}

TEST(Execute, ClampBackward) {
    test::vector<float> src {
            0, -1, 0.0723652, -0.0364869, 40, -50, 0.188521, -0.729739, 88.371};
    test::vector<float> dst {
            0, -1, 0.0723652, -0.0364869, 2, -1, 0.188521, -0.729739, 2};
    test::vector<float> diff_dst {
            3, -7, 0.0194608, -0.0559478, 70, 0, 0.754086, -0.218955, 88.5838};
    test::vector<float> ref_diff_src {
            3, -0, 0.0194608, -0.0559478, 0, 0, 0.754086, -0.218955, 0};

    impl::dims dims {1, 3, 3};
    const std::map<impl::op_attr_t, float> attrs_data {
            {impl::op_attr::min, -1.f}, {impl::op_attr::max, 2.f}};

    test_eltwise_bwd_common({src, true}, diff_dst, ref_diff_src, dims,
            impl::op_kind::ClampBackprop, "clamp_bw", attrs_data);
    test_eltwise_bwd_common({dst, false}, diff_dst, ref_diff_src, dims,
            impl::op_kind::ClampBackprop, "clamp_bw", attrs_data);
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

    impl::dims dims {1, 2, 3};
    const std::map<impl::op_attr_t, float> attrs_data {
            {impl::op_attr::alpha, 1.f}};

    test_eltwise_common(
            src, ref_dst, dims, impl::op_kind::Elu, "elu", attrs_data);
}

TEST(Execute, EluBackward) {
    test::vector<float> src {
            0, -1, 0.0723652, -0.0364869, 40, -50, 0.188521, -0.729739, 88.371};
    test::vector<float> dst {0, -0.632121, 0.0723652, -0.0358293, 40, -1,
            0.188521, -0.517965, 88.371};
    test::vector<float> diff_dst {
            3, -7, 0.0194608, -0.0559478, 70, 0, 0.754086, -0.218955, 88.5838};
    test::vector<float> ref_diff_src {3, -2.57516, 0.0194608, -0.0539432, 70, 0,
            0.754086, -0.105544, 88.5838};

    dnnl::impl::graph::dims dims {1, 3, 3};
    const std::map<impl::op_attr_t, float> attrs_data {
            {impl::op_attr::alpha, 1.f}};

    test_eltwise_bwd_common({src, true}, diff_dst, ref_diff_src, dims,
            impl::op_kind::EluBackprop, "elu_bw", attrs_data);
    test_eltwise_bwd_common({dst, false}, diff_dst, ref_diff_src, dims,
            impl::op_kind::EluBackprop, "elu_bw", attrs_data);
}

TEST(Execute, Exp) {
    test::vector<float> src {-2.0, -1.5, -1.0, -0.5, 0.0, 3.5};
    test::vector<float> ref_dst;
    for (size_t i = 0; i < src.size(); ++i) {
        float temp = static_cast<float>(exp(src[i]));
        ref_dst.push_back(temp);
    }

    dnnl::impl::graph::dims dims {1, 2, 3};

    test_eltwise_common(src, ref_dst, dims, impl::op_kind::Exp, "exp");
}

TEST(Execute, Gelu) {
    test::vector<float> src {-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0};
    test::vector<float> ref_dst {-0.0455001f, -0.10021085f, -0.15865527f,
            -0.15426877f, 0.f, 0.3457312f, 0.84134465f, 1.399789f, 1.9544998f};

    dnnl::impl::graph::dims dims {1, 3, 3};

    test_eltwise_common(src, ref_dst, dims, impl::op_kind::GELU, "gelu");
}

TEST(Execute, GeluBackward) {
    test::vector<float> src {
            0, -1, 0.0723652, -0.0364869, 40, -50, 0.188521, -0.729739, 88.371};
    test::vector<float> diff_dst {
            3, -7, 0.0194608, -0.0559478, 70, 0, 0.754086, -0.218955, 88.5838};
    test::vector<float> ref_diff_src {1.5, 0.583208, 0.0108521, -0.0263458, 70,
            0, 0.489138, -0.00212478, 88.5838};

    dnnl::impl::graph::dims dims {1, 3, 3};

    test_eltwise_bwd_common({src, true}, diff_dst, ref_diff_src, dims,
            impl::op_kind::GELUBackprop, "gelu_bw");
}

TEST(Execute, HardSwish) {
    test::vector<float> src {7, -4, 0.0723652095, -0.0364869386, 60, -20,
            0.188521415, -0.729738772, 88.3709564};
    test::vector<float> ref_dst {7, 0, 0.0370553955, -0.0180215873, 60, 0,
            0.100184090, -0.276116282, 88.3709564};
    dnnl::impl::graph::dims dims {1, 3, 3};

    test_eltwise_common(
            src, ref_dst, dims, impl::op_kind::HardSwish, "hardswish");
}

TEST(Execute, HardSwishBackward) {
    test::vector<float> src {7, -4, 0.0723652095, -0.0364869386, 60, -20,
            0.188521415, -0.729738772, 88.3709564};
    test::vector<float> diff_dst {9, -10, 0.0194608141, -0.0559477545, 70, -20,
            0.754085660, -0.218955040, 88.5838242};
    test::vector<float> ref_diff_src {9, 0, 0.0101998346, -0.0272934232, 70, 0,
            0.424429923, -0.0562175252, 88.5838242};

    dnnl::impl::graph::dims dims {1, 3, 3};

    test_eltwise_bwd_common({src, true}, diff_dst, ref_diff_src, dims,
            impl::op_kind::HardSwishBackprop, "hardswish_bw");
}

TEST(Execute, Relu) {
    test::vector<float> src {-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0};
    test::vector<float> ref_dst {0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.5, 2.0};

    impl::dims dims {1, 3, 3};

    test_eltwise_common(src, ref_dst, dims, impl::op_kind::ReLU, "relu");
}

TEST(Execute, ReluBackward) {
    test::vector<float> src {
            0, -1, 0.0723652, -0.0364869, 40, -50, 0.188521, -0.729739, 88.371};
    test::vector<float> dst {
            0, -0, 0.0723652, -0, 40, -0, 0.188521, -0, 88.371};
    test::vector<float> diff_dst {
            3, -7, 0.0194608, -0.0559478, 70, 0, 0.754086, -0.218955, 88.5838};
    test::vector<float> ref_diff_src {
            0, -0, 0.0194608, -0, 70, 0, 0.754086, -0, 88.5838};

    dnnl::impl::graph::dims dims {1, 3, 3};

    test_eltwise_bwd_common({src, true}, diff_dst, ref_diff_src, dims,
            impl::op_kind::ReLUBackprop, "relu_bw");
    test_eltwise_bwd_common({dst, false}, diff_dst, ref_diff_src, dims,
            impl::op_kind::ReLUBackprop, "relu_bw");
}

TEST(Execute, Sigmoid) {
    test::vector<float> src {-5.0, -2.5, -1.0, 0.3, 0.0, 1.2};
    test::vector<float> ref_dst = sigmoid_func(src);

    dnnl::impl::graph::dims dims {1, 2, 3};

    test_eltwise_common(src, ref_dst, dims, impl::op_kind::Sigmoid, "sigmoid");
}

TEST(Execute, SigmoidBackward) {
    test::vector<float> src {
            0, -1, 0.0723652, -0.0364869, 40, -50, 0.188521, -0.729739, 88.371};
    test::vector<float> dst {0.5, 0.268941, 0.518083, 0.490879, 1, 1.92875e-22,
            0.546991, 0.325252, 1};
    test::vector<float> diff_dst {
            3, -7, 0.0194608, -0.0559478, 70, 0, 0.754086, -0.218955, 88.5838};
    test::vector<float> ref_diff_src {0.75, -1.37628, 0.00485884, -0.0139823, 0,
            0, 0.186856, -0.0480526, 0};

    dnnl::impl::graph::dims dims {1, 3, 3};

    test_eltwise_bwd_common({src, true}, diff_dst, ref_diff_src, dims,
            impl::op_kind::SigmoidBackprop, "sigmoid_bw");
    test_eltwise_bwd_common({dst, false}, diff_dst, ref_diff_src, dims,
            impl::op_kind::SigmoidBackprop, "sigmoid_bw");
}

TEST(Execute, SoftPlus) {
    test::vector<float> src {
            0, -1, 0.0723652, -0.0364869, 40, -25, 0.188521, -0.729739, 44.317};
    test::vector<float> ref_dst_case1 = {-0.693147, -1.31326, -0.657619,
            -0.711557, -0, -25, -0.603322, -1.12315, -0};
    test::vector<float> ref_dst_case2 = {0.693147, 0.313262, 0.729984, 0.67507,
            40, 0, 0.791844, 0.393416, 44.317};
    test::vector<float> ref_dst_case3 = {-0.346574, -1.06346, -0.311699,
            -0.36515, -0, -25, -0.261146, -0.834203, -0};
    test::vector<float> ref_dst_case4 = {0.346574, 0.063464, 0.384064, 0.328663,
            40, 0, 0.449667, 0.104465, 44.317};

    dnnl::impl::graph::dims dims {1, 3, 3};
    const std::map<impl::op_attr_t, int64_t> attrs_data_case1 {
            {impl::op_attr::beta, -1}};
    const std::map<impl::op_attr_t, int64_t> attrs_data_case2 {
            {impl::op_attr::beta, 1}};
    const std::map<impl::op_attr_t, int64_t> attrs_data_case3 {
            {impl::op_attr::beta, -2}};
    const std::map<impl::op_attr_t, int64_t> attrs_data_case4 {
            {impl::op_attr::beta, 2}};

    test_eltwise_common(src, ref_dst_case1, dims, impl::op_kind::SoftPlus,
            "softplus", attrs_data_case1);
    test_eltwise_common(src, ref_dst_case2, dims, impl::op_kind::SoftPlus,
            "softplus", attrs_data_case2);
    test_eltwise_common(src, ref_dst_case3, dims, impl::op_kind::SoftPlus,
            "softplus", attrs_data_case3);
    test_eltwise_common(src, ref_dst_case4, dims, impl::op_kind::SoftPlus,
            "softplus", attrs_data_case4);
}

TEST(Execute, SoftPlusBackward) {
    test::vector<float> src {
            0, -1, 0.0723652, -0.0364869, 40, -50, 0.188521, -0.729739, 88.371};
    test::vector<float> diff_dst {
            3, -7, 0.0194608, -0.0559478, 70, 0, 0.754086, -0.218955, 88.5838};
    test::vector<float> ref_diff_src_case1 {1.5, -5.11741, 0.00937849,
            -0.0284842, 2.97385e-16, 0, 0.341607, -0.147739, 0};
    test::vector<float> ref_diff_src_case2 {1.5, -1.88259, 0.0100823,
            -0.0274636, 70, 0, 0.412478, -0.0712156, 88.5838};
    test::vector<float> ref_diff_src_case3 {1.5, -6.16558, 0.00902748,
            -0.0289941, 1.2634e-33, 0, 0.306793, -0.177672, 0};
    test::vector<float> ref_diff_src_case4 {1.5, -0.83442, 0.0104333,
            -0.0269537, 70, 0, 0.447293, -0.0412833, 88.5838};

    dnnl::impl::graph::dims dims {1, 3, 3};
    const std::map<impl::op_attr_t, int64_t> attrs_data_case1 {
            {impl::op_attr::beta, -1}};
    const std::map<impl::op_attr_t, int64_t> attrs_data_case2 {
            {impl::op_attr::beta, 1}};
    const std::map<impl::op_attr_t, int64_t> attrs_data_case3 {
            {impl::op_attr::beta, -2}};
    const std::map<impl::op_attr_t, int64_t> attrs_data_case4 {
            {impl::op_attr::beta, 2}};

    test_eltwise_bwd_common({src, true}, diff_dst, ref_diff_src_case1, dims,
            impl::op_kind::SoftPlusBackprop, "softplus_bw", attrs_data_case1);
    test_eltwise_bwd_common({src, true}, diff_dst, ref_diff_src_case2, dims,
            impl::op_kind::SoftPlusBackprop, "softplus_bw", attrs_data_case2);
    test_eltwise_bwd_common({src, true}, diff_dst, ref_diff_src_case3, dims,
            impl::op_kind::SoftPlusBackprop, "softplus_bw", attrs_data_case3);
    test_eltwise_bwd_common({src, true}, diff_dst, ref_diff_src_case4, dims,
            impl::op_kind::SoftPlusBackprop, "softplus_bw", attrs_data_case4);
}

TEST(Execute, Sqrt) {
    test::vector<float> src {2.0, 1.5, 1.0, 0.5, 0.0, 3.5};
    test::vector<float> ref_dst = sqrt_func(src);

    dnnl::impl::graph::dims dims {1, 2, 3};

    test_eltwise_common(src, ref_dst, dims, impl::op_kind::Sqrt, "sqrt");
}

TEST(Execute, SqrtBackward) {
    test::vector<float> src {1, 0.0262164, 30, 0.424328, 88.1054, 22.5865,
            44.2676, 0.0748657, 0.0429739};
    test::vector<float> dst {1, 0.161915, 5.47723, 0.651405, 9.38645, 4.75252,
            6.65339, 0.273616, 0.207301};
    test::vector<float> diff_dst {4, 0.0748657, 10, 0.597313, 88.1216, 22.446,
            44.7703, 0.0294627, 0.0618955};
    test::vector<float> ref_diff_src {2, 0.231188, 0.912871, 0.458481, 4.69409,
            2.36148, 3.36447, 0.0538395, 0.149289};

    dnnl::impl::graph::dims dims {1, 3, 3};

    test_eltwise_bwd_common({src, true}, diff_dst, ref_diff_src, dims,
            impl::op_kind::SqrtBackprop, "sqrt_bw");
    test_eltwise_bwd_common({dst, false}, diff_dst, ref_diff_src, dims,
            impl::op_kind::SqrtBackprop, "sqrt_bw");
}

TEST(Execute, Square) {
    test::vector<float> src {-2.0, -1.5, -1.0, -0.5, 0.0, 3.5};
    test::vector<float> ref_dst;
    for (size_t i = 0; i < src.size(); ++i) {
        float temp = src[i] * src[i];
        ref_dst.push_back(temp);
    }

    dnnl::impl::graph::dims dims {1, 2, 3};

    test_eltwise_common(src, ref_dst, dims, impl::op_kind::Square, "square");
}

TEST(Execute, Log) {
    test::vector<float> src {2.f, 1.5f, 1.f, 0.5f, 0.8f, 3.5f};
    test::vector<float> ref_dst;
    for (size_t i = 0; i < src.size(); ++i) {
        float temp = static_cast<float>(log(src[i]));
        ref_dst.push_back(temp);
    }

    dnnl::impl::graph::dims dims {1, 2, 3};

    test_eltwise_common(src, ref_dst, dims, impl::op_kind::Log, "log");
}

TEST(Execute, Tanh) {
    test::vector<float> src {-2.0, -1.5, -1.0, -0.5, 0.0, 3.5};
    test::vector<float> ref_dst = tanh_func(src);

    dnnl::impl::graph::dims dims {1, 2, 3};

    test_eltwise_common(src, ref_dst, dims, impl::op_kind::Tanh, "tanh");
}

TEST(Execute, TanhBackward) {
    test::vector<float> src {
            0, -1, 0.0723652, -0.0364869, 40, -50, 0.188521, -0.729739, 88.371};
    test::vector<float> dst {
            0, -0.761594, 0.0722392, -0.0364708, 1, -1, 0.186319, -0.622906, 1};
    test::vector<float> diff_dst {
            3, -7, 0.0194608, -0.0559478, 70, 0, 0.754086, -0.218955, 88.5838};
    test::vector<float> ref_diff_src {
            3, -2.93982, 0.0193593, -0.0558733, 0, 0, 0.727908, -0.133998, 0};

    dnnl::impl::graph::dims dims {1, 3, 3};

    test_eltwise_bwd_common({src, true}, diff_dst, ref_diff_src, dims,
            impl::op_kind::TanhBackprop, "tanh_bw");
    test_eltwise_bwd_common({dst, false}, diff_dst, ref_diff_src, dims,
            impl::op_kind::TanhBackprop, "tanh_bw");
}

TEST(ExecuteSubgraphFp32, Shuffle) {
    impl::engine_t *engine = get_engine();
    impl::stream_t *strm = get_stream();

    const std::vector<std::pair<size_t, test::vector<float>>> configs {
            {1,
                    {1., 1., 2., 2., 5., 5., 6., 6., 3., 3., 4., 4., 7., 7., 8.,
                            8.}},
            {3,
                    {1., 2., 1., 2., 3., 4., 3., 4., 5., 6., 5., 6., 7., 8., 7.,
                            8.}}};
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

        test::vector<float> src_data {
                1., 1., 2., 2., 3., 3., 4., 4., 5., 5., 6., 6., 7., 7., 8., 8.};

        impl::op_t reshape0 {0, impl::op_kind::StaticReshape, "reshape0"};
        reshape0.set_attr(impl::op_attr::shape, reshape0_dst_shape);
        reshape0.set_attr(impl::op_attr::special_zero, false);

        impl::op_t transpose {1, impl::op_kind::StaticTranspose, "transpose"};
        transpose.set_attr(impl::op_attr::order, order);

        impl::op_t reshape1 {2, impl::op_kind::StaticReshape, "reshape1"};
        reshape1.set_attr(impl::op_attr::shape, reshape1_dst_shape);
        reshape1.set_attr(impl::op_attr::special_zero, false);

        impl::logical_tensor_t reshape0_src = utils::logical_tensor_init(
                0, reshape0_src_shape, impl::data_type::f32);
        impl::logical_tensor_t reshape0_dst = utils::logical_tensor_init(
                1, reshape0_dst_shape, impl::data_type::f32);
        impl::logical_tensor_t transpose_dst = utils::logical_tensor_init(
                2, transpose_dst_shape, impl::data_type::f32);
        impl::logical_tensor_t reshape1_dst = utils::logical_tensor_init(
                3, reshape1_dst_shape, impl::data_type::f32);

        reshape0.add_input(reshape0_src);
        reshape0.add_output(reshape0_dst);

        transpose.add_input(reshape0_dst);
        transpose.add_output(transpose_dst);

        reshape1.add_input(transpose_dst);
        reshape1.add_output(reshape1_dst);

        impl::graph_t agraph(engine->kind());
        agraph.add_op(&reshape0);
        agraph.add_op(&transpose);
        agraph.add_op(&reshape1);
        agraph.finalize();

        impl::pass::pass_base_ptr apass = get_pass("shuffle_fusion");
        apass->run(agraph);
        ASSERT_EQ(agraph.get_num_partitions(), 1U);
        auto part = agraph.get_partitions()[0];

        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> lt_ins {&reshape0_src};
        std::vector<const impl::logical_tensor_t *> lt_outs {&reshape1_dst};

        p.compile(&cp, lt_ins, lt_outs, engine);

        test::vector<float> dst_data(product(reshape1_dst_shape));
        impl::tensor_t reshape0_src_ts(reshape0_src, engine, src_data.data());
        impl::tensor_t reshape1_dst_ts(reshape1_dst, engine, dst_data.data());

        cp.execute(strm, {reshape0_src_ts}, {reshape1_dst_ts});
        strm->wait();

        for (size_t i = 0; i < expected_data.size(); ++i) {
            ASSERT_FLOAT_EQ(expected_data[i], dst_data[i]);
        }
    }
}

TEST(ExecuteSubgraphFp32, ReciprocalMul) {
    impl::engine_t *eng = get_engine();

    test::vector<float> src0_data {1.0, 2.0, 3.0, 10.0, 20.0, 30.0};
    test::vector<float> src1_data {2.0, 2.0, 2.0, 5.0, 5.0, 5.0};
    test::vector<float> ref_dst_data {0.5, 1.0, 1.5, 2.0, 4.0, 6.0};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    impl::op_t reciprocal_op {0, impl::op_kind::Reciprocal, "reciprocal"};

    impl::op_t mul_op {1, impl::op_kind::Multiply, "mul"};

    // prepare logical tensor
    impl::logical_tensor_t src0
            = utils::logical_tensor_init(0, {1, 2, 3}, impl::data_type::f32);
    impl::logical_tensor_t src1
            = utils::logical_tensor_init(1, {1, 2, 3}, impl::data_type::f32);
    impl::logical_tensor_t reciprocal_dst = utils::logical_tensor_init(
            3, {1, 2, 3}, impl::data_type::f32, impl::layout_type::any);
    impl::logical_tensor_t mul_dst = utils::logical_tensor_init(
            4, {1, 2, 3}, impl::data_type::f32, impl::layout_type::any);

    reciprocal_op.add_input(src1);
    reciprocal_op.add_output(reciprocal_dst);
    mul_op.add_input(src0);
    mul_op.add_input(reciprocal_dst);
    mul_op.add_output(mul_dst);

    impl::graph_t g(eng->kind());
    ASSERT_EQ(g.add_op(&reciprocal_op), impl::status::success);
    ASSERT_EQ(g.add_op(&mul_op), impl::status::success);
    g.finalize();

    impl::pass::pass_base_ptr apass = get_pass("reciprocal_multiply_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];
    impl::partition_t p;
    p.init(part);

    // compile
    std::vector<const impl::logical_tensor_t *> inputs {&src0, &src1};
    std::vector<const impl::logical_tensor_t *> outputs {&mul_dst};

    impl::compiled_partition_t cp(p);
    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), impl::status::success);

    impl::logical_tensor_t lt;
    cp.query_logical_tensor(mul_dst.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src0_ts(src0, eng, src0_data.data());
    impl::tensor_t src1_ts(src1, eng, src1_data.data());
    impl::tensor_t dst_ts(lt, eng, dst_data.data());

    impl::stream_t *strm = get_stream();
    ASSERT_EQ(cp.execute(strm, {src0_ts, src1_ts}, {dst_ts}),
            impl::status::success);
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
    impl::engine_t *engine = get_engine();

    std::vector<int64_t> input_dims {1, 3, 3};

    impl::op_t add0 {0, impl::op_kind::Add, "add_0"};
    add0.set_attr<std::string>(impl::op_attr::auto_broadcast, "none");
    impl::logical_tensor_t input0
            = utils::logical_tensor_init(0, input_dims, impl::data_type::f32);
    impl::logical_tensor_t input1
            = utils::logical_tensor_init(1, input_dims, impl::data_type::f32);
    impl::logical_tensor_t output0
            = utils::logical_tensor_init(2, input_dims, impl::data_type::f32);

    impl::op_t add1 {1, impl::op_kind::Add, "add_1"};
    add1.set_attr<std::string>(impl::op_attr::auto_broadcast, "none");
    impl::logical_tensor_t input2
            = utils::logical_tensor_init(3, input_dims, impl::data_type::f32);
    impl::logical_tensor_t output1
            = utils::logical_tensor_init(4, input_dims, impl::data_type::f32);

    impl::op_t add2 {2, impl::op_kind::Add, "add_2"};
    add2.set_attr<std::string>(impl::op_attr::auto_broadcast, "none");
    impl::logical_tensor_t input3
            = utils::logical_tensor_init(5, input_dims, impl::data_type::f32);
    impl::logical_tensor_t output2
            = utils::logical_tensor_init(6, input_dims, impl::data_type::f32);

    impl::op_t add3 {3, impl::op_kind::Add, "add_3"};
    add3.set_attr<std::string>(impl::op_attr::auto_broadcast, "none");
    impl::logical_tensor_t input4
            = utils::logical_tensor_init(7, input_dims, impl::data_type::f32);
    impl::logical_tensor_t output3
            = utils::logical_tensor_init(8, input_dims, impl::data_type::f32);

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

    impl::graph_t agraph(engine->kind());

    ASSERT_EQ(agraph.add_op(&add0), impl::status::success);
    ASSERT_EQ(agraph.add_op(&add1), impl::status::success);
    ASSERT_EQ(agraph.add_op(&add2), impl::status::success);
    ASSERT_EQ(agraph.add_op(&add3), impl::status::success);
    agraph.finalize();
    impl::pass::pass_base_ptr apass = get_pass("sum_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1U);
    auto part = agraph.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {
            &input0, &input1, &input2, &input3, &input4};
    std::vector<const impl::logical_tensor_t *> outputs {&output3};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, engine), impl::status::success);

    test::vector<float> input0_data(product(input_dims), 1);
    test::vector<float> input1_data(product(input_dims), 1);
    test::vector<float> input2_data(product(input_dims), 1);
    test::vector<float> input3_data(product(input_dims), 1);
    test::vector<float> input4_data(product(input_dims), 1);
    test::vector<float> output_data(product(input_dims), 0);

    impl::tensor_t input0_ts(input0, engine, input0_data.data());
    impl::tensor_t input1_ts(input1, engine, input1_data.data());
    impl::tensor_t input2_ts(input2, engine, input2_data.data());
    impl::tensor_t input3_ts(input3, engine, input3_data.data());
    impl::tensor_t input4_ts(input4, engine, input4_data.data());
    impl::tensor_t output_ts(output3, engine, output_data.data());

    impl::stream_t *strm = get_stream();
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

    impl::engine_t *eng = get_engine();

    impl::op_t eltwise_op(impl::op_kind::Tanh);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {2, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst
            = utils::logical_tensor_init(1, {2, 3}, impl::data_type::f32);

    eltwise_op.add_input(src);
    eltwise_op.add_output(dst);

    impl::graph_t g(eng->kind());
    g.add_op(&eltwise_op);
    g.finalize();

    impl::pass::pass_base_ptr apass = get_pass("tanh_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];
    impl::partition_t p;
    p.init(part);

    // compile
    std::vector<const impl::logical_tensor_t *> inputs {&src};
    std::vector<const impl::logical_tensor_t *> outputs {&dst};

    impl::compiled_partition_t cp(p);
    ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), impl::status::success);

    auto pairs = cp.get_inplace_pairs();
    ASSERT_EQ(pairs.size(), 1U);
    ASSERT_EQ(pairs[0].input_id, 0U);
    ASSERT_EQ(pairs[0].output_id, 1U);
}

struct eltwise_binary_params_t {
    impl::op_kind_t eltwise_kind;
    impl::op_kind_t binary_kind;
    std::vector<impl::dim_t> binary_src_shape;
    bool swap;
};

class eltwise_binary_t
    : public ::testing::TestWithParam<eltwise_binary_params_t> {
public:
    void TestEltwiseBinary() {
        const auto params
                = ::testing::TestWithParam<eltwise_binary_params_t>::GetParam();
        impl::engine_t *eng = get_engine();

        std::vector<impl::dim_t> binary_src_shape = params.binary_src_shape;
        test::vector<float> src {-2.0, -1.5, 1.0, 0.5};
        test::vector<float> binary_src(product(binary_src_shape));
        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
        std::generate(binary_src.begin(), binary_src.end(),
                [&]() { return f32_distribution(generator); });
        test::vector<float> dst {0.0, 0.0, 0.0, 0.0};

        impl::op_t eltwise_op(0, params.eltwise_kind, "elt");
        if (params.eltwise_kind == impl::op_kind::Elu) {
            eltwise_op.set_attr<float>(impl::op_attr::alpha, 1.f);
        } else if (params.eltwise_kind == impl::op_kind::Clamp) {
            eltwise_op.set_attr<float>(impl::op_attr::min, -1.f);
            eltwise_op.set_attr<float>(impl::op_attr::max, 2.f);
        }
        impl::op_t binary_op(1, params.binary_kind, "binary");

        impl::logical_tensor_t eltwise_src_lt = utils::logical_tensor_init(
                0, {1, 1, 2, 2}, impl::data_type::f32);
        impl::logical_tensor_t eltwise_dst_lt = utils::logical_tensor_init(
                1, {1, 1, 2, 2}, impl::data_type::f32, impl::layout_type::any);
        impl::logical_tensor_t binary_src_lt = utils::logical_tensor_init(
                2, binary_src_shape, impl::data_type::f32);
        impl::logical_tensor_t binary_dst_lt = utils::logical_tensor_init(
                3, {1, 1, 2, 2}, impl::data_type::f32, impl::layout_type::any);

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

        impl::graph_t g(eng->kind());
        g.add_op(&eltwise_op);
        g.add_op(&binary_op);
        g.finalize();

        impl::pass::pass_base_ptr apass = get_pass("eltwise_binary_fusion");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        // compile
        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> inputs {
                &eltwise_src_lt, &binary_src_lt};

        std::vector<const impl::logical_tensor_t *> outputs {&binary_dst_lt};

        ASSERT_EQ(p.compile(&cp, inputs, outputs, eng), impl::status::success);

        impl::logical_tensor_t lt;
        cp.query_logical_tensor(binary_dst_lt.id, &lt);

        ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

        impl::tensor_t src_ts(eltwise_src_lt, eng, src.data());
        impl::tensor_t binary_src_ts(binary_src_lt, eng, binary_src.data());
        impl::tensor_t binary_dst_ts(binary_dst_lt, eng, dst.data());

        impl::stream_t *strm = get_stream();

        ASSERT_EQ(cp.execute(strm, {src_ts, binary_src_ts}, {binary_dst_ts}),
                impl::status::success);
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
                        impl::op_kind::ReLU, impl::op_kind::Add, {1}, false},
                // with broadcast add and swap inputs
                eltwise_binary_params_t {
                        impl::op_kind::ReLU, impl::op_kind::Add, {1}, true},
                // no broadcast add and no swap inputs
                eltwise_binary_params_t {impl::op_kind::ReLU,
                        impl::op_kind::Add, {1, 1, 2, 2}, false},
                // no broadcast add and swap inputs
                eltwise_binary_params_t {impl::op_kind::ReLU,
                        impl::op_kind::Add, {1, 1, 2, 2}, true},
                eltwise_binary_params_t {impl::op_kind::Clamp,
                        impl::op_kind::Multiply, {1}, false},
                eltwise_binary_params_t {impl::op_kind::Round,
                        impl::op_kind::Maximum, {1, 1, 2, 2}, false},
                eltwise_binary_params_t {impl::op_kind::Sigmoid,
                        impl::op_kind::Divide, {1}, false},
                eltwise_binary_params_t {impl::op_kind::SoftPlus,
                        impl::op_kind::Minimum, {1, 1, 2, 2}, false},
                eltwise_binary_params_t {impl::op_kind::Exp,
                        impl::op_kind::Minimum, {1, 1, 2, 2}, false},
                eltwise_binary_params_t {impl::op_kind::Elu,
                        impl::op_kind::Subtract, {1, 1, 2, 2}, false},
                eltwise_binary_params_t {impl::op_kind::Sqrt,
                        impl::op_kind::Minimum, {1}, false},
                eltwise_binary_params_t {impl::op_kind::HardSwish,
                        impl::op_kind::Minimum, {1}, false},
                eltwise_binary_params_t {
                        impl::op_kind::Tanh, impl::op_kind::Add, {1}, true}));
