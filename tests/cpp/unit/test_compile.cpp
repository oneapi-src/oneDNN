/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include <cmath>
#include <functional>
#include <limits>
#include <random>
#include <string>

#include <gtest/gtest.h>

#include "interface/backend.hpp"

#include "backend/dnnl/common.hpp"
#include "backend/dnnl/dnnl_backend.hpp"
#include "backend/dnnl/dnnl_partition_impl.hpp"
#include "backend/dnnl/kernels/batchnorm.hpp"
#include "backend/dnnl/kernels/binary.hpp"
#include "backend/dnnl/kernels/concat.hpp"
#include "backend/dnnl/kernels/conv.hpp"
#include "backend/dnnl/kernels/eltwise.hpp"
#include "backend/dnnl/kernels/inner_product.hpp"
#include "backend/dnnl/kernels/layernorm.hpp"
#include "backend/dnnl/kernels/matmul.hpp"
#include "backend/dnnl/kernels/pool.hpp"
#include "backend/dnnl/kernels/quantize.hpp"
#include "backend/dnnl/kernels/reorder.hpp"
#include "backend/dnnl/kernels/softmax.hpp"
#include "backend/dnnl/passes/utils.hpp"

#include "unit_test_common.hpp"

#include "dnnl.h"
#include "utils.hpp"

namespace impl = dnnl::graph::impl;
namespace dnnl_impl = impl::dnnl_impl;
namespace utils = dnnl::graph::tests::unit::utils;

static dnnl_impl::kernel_registry &get_dnnl_kernel_registry() {
    return dnnl_impl::dnnl_backend::get_singleton().get_kernel_registry();
}

namespace {
dnnl::graph::impl::pass::pass_base_ptr get_pass(const std::string &pass_name) {
    auto &backend_ptr
            = dnnl::graph::impl::dnnl_impl::dnnl_backend::get_singleton();
    auto pm = dnnl::graph::impl::pass::pass_manager(
            backend_ptr.get_pass_registry());
    auto &passes = pm.get_passes();
    auto find = std::find_if(passes.begin(), passes.end(),
            [&pass_name](const dnnl::graph::impl::pass::pass_base_ptr &p)
                    -> bool { return p->get_pass_name() == pass_name; });

    return *find;
}

// This function run the unfused graph op by op. The output tensor of the whole
// graph should also be strided so that we can read and check the results. The
// given graph will be deep copied first so that all the changes inside the
// function are not visible outside.
impl::status_t run_graph(impl::graph_t &agraph,
        const std::vector<impl::tensor_t> &g_in_ts,
        const std::vector<impl::tensor_t> &g_out_ts, impl::engine_t &eng,
        impl::stream_t &strm) {
    using ltw = impl::logical_tensor_wrapper;
    impl::status_t ret;
    impl::graph_t copied(agraph);
    auto ops = copied.get_ops();

    // force each tensor to be strided
    for (auto &op : ops) {
        for (auto val : op->get_input_values()) {
            val->set_layout_type(impl::layout_type::strided);
        }

        for (auto val : op->get_output_values()) {
            val->set_layout_type(impl::layout_type::strided);
        }
    }

    // set the given in/outputs to the graph
    std::vector<impl::logical_tensor_t> g_in_lts, g_out_lts;
    for (auto &in_t : g_in_ts) {
        g_in_lts.emplace_back(in_t.get_logical_tensor());
    }
    for (auto &out_t : g_out_ts) {
        g_out_lts.emplace_back(out_t.get_logical_tensor());
    }
    ret = dnnl_impl::set_given_inputs_outputs(ops, g_in_lts, g_out_lts);
    if (ret != impl::status::success) return ret;

    // do shape inference
    ret = copied.infer_shape();
    if (ret != impl::status::success) return ret;

    // used to hold the temporary buffers
    std::unordered_map<size_t, test::vector<char>> temp_data;

    // compile and execute each op in topo order
    return impl::topo_order_visit(copied.get_output_ops(), [&](impl::op_t *op) {
        // construct a single op partition without running pass
        auto part = std::make_shared<dnnl_impl::dnnl_partition_impl_t>(
                eng.kind());
        part->add_op(op->shared_from_this());
        part->init(op);
        impl::partition_t p;
        p.init(part);

        // prepare logical tensors
        std::vector<impl::logical_tensor_t> in_lts, out_lts;
        std::vector<const impl::logical_tensor_t *> in_lt_ptrs, out_lt_ptrs;
        in_lts.reserve(op->num_inputs());
        in_lt_ptrs.reserve(op->num_inputs());
        out_lts.reserve(op->num_outputs());
        out_lt_ptrs.reserve(op->num_outputs());

        for (auto &in : op->get_input_values()) {
            in_lts.emplace_back(in->get_logical_tensor());
            in_lt_ptrs.emplace_back(&in_lts.back());
        }

        for (auto &out : op->get_output_values()) {
            out_lts.emplace_back(out->get_logical_tensor());
            out_lt_ptrs.emplace_back(&out_lts.back());
        }

        // compile
        impl::compiled_partition_t cp(p);
        ret = p.compile(&cp, in_lt_ptrs, out_lt_ptrs, &eng);
        if (ret != impl::status::success) return ret;

        // update the layout info in output values
        for (auto &out_val : op->get_output_values()) {
            impl::logical_tensor_t compiled_lt;
            cp.query_logical_tensor(
                    out_val->get_logical_tensor().id, &compiled_lt);
            out_val->set_logical_tensor(compiled_lt);
        }

        // prepare tensors
        std::vector<impl::tensor_t> in_ts, out_ts;
        for (auto &in_val : op->get_input_values()) {
            auto in_lt = in_val->get_logical_tensor();
            auto pos = std::find_if(g_in_ts.begin(), g_in_ts.end(),
                    [&](const impl::tensor_t &t) {
                        return in_lt.id == t.get_logical_tensor().id;
                    });
            if (pos != g_in_ts.end()) {
                in_ts.emplace_back(*pos);
                continue;
            }
            if (temp_data.find(in_lt.id) == temp_data.end()) {
                temp_data.insert(
                        {in_lt.id, test::vector<char>(ltw(in_lt).size(), 0)});
            }
            in_ts.emplace_back(in_lt, &eng, temp_data.at(in_lt.id).data());
        }
        for (auto &out_val : op->get_output_values()) {
            auto out_lt = out_val->get_logical_tensor();
            auto pos = std::find_if(g_out_ts.begin(), g_out_ts.end(),
                    [&](const impl::tensor_t &t) {
                        return out_lt.id == t.get_logical_tensor().id;
                    });
            if (pos != g_out_ts.end()) {
                out_ts.emplace_back(*pos);
                continue;
            }
            if (temp_data.find(out_lt.id) == temp_data.end()) {
                temp_data.insert(
                        {out_lt.id, test::vector<char>(ltw(out_lt).size(), 0)});
            }
            out_ts.emplace_back(out_lt, &eng, temp_data.at(out_lt.id).data());
        }

        // execute
        ret = cp.execute(&strm, in_ts, out_ts);
        if (ret != impl::status::success) return ret;

        strm.wait();
        return impl::status::success;
    });
}
} // namespace

TEST(operator_compile, convolution_compile_fp32) {
    using dims = impl::dnnl_impl::dims;

    impl::engine_t &engine = get_engine();

    impl::op_t conv_op(impl::op_kind::Convolution);
    conv_op.set_attr<dims>("strides", dims {1, 1});
    conv_op.set_attr<dims>("dilations", dims {1, 1});
    conv_op.set_attr<dims>("pads_begin", dims {0, 0});
    conv_op.set_attr<dims>("pads_end", dims {0, 0});
    conv_op.set_attr<int64_t>("groups", 1);
    conv_op.set_attr<std::string>("data_format", "NCX");
    conv_op.set_attr<std::string>("filter_format", "OIX");

    // prepare logical tensor
    impl::logical_tensor_t src = utils::logical_tensor_init(
            0, {8, 3, 224, 224}, impl::data_type::f32);
    impl::logical_tensor_t weight = utils::logical_tensor_init(
            1, {16, 3, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst = utils::logical_tensor_init(
            2, {8, 16, 222, 222}, impl::data_type::f32, impl::layout_type::any);

    conv_op.add_input(src);
    conv_op.add_input(weight);
    conv_op.add_output(dst);

    impl::graph_t g;
    g.add_op(&conv_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("conv_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src, &weight};
    std::vector<const impl::logical_tensor_t *> outputs {&dst};

    p.compile(&cp, inputs, outputs, &engine);

    impl::logical_tensor_t lt;
    cp.query_logical_tensor(dst.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::opaque);
}

TEST(operator_compile, convolution_backward_data_compile_fp32) {
    using dims = dnnl::graph::impl::dnnl_impl::dims;

    impl::engine_t &eng = get_engine();

    impl::op_t conv_op(impl::op_kind::ConvolutionBackpropData);
    conv_op.set_attr<dims>("strides", dims {1, 1});
    conv_op.set_attr<dims>("dilations", dims {1, 1});
    conv_op.set_attr<dims>("pads_begin", dims {0, 0});
    conv_op.set_attr<dims>("pads_end", dims {0, 0});
    // according to spec, group should be greater than 0
    conv_op.set_attr<int64_t>("groups", 1);
    conv_op.set_attr<std::string>("data_format", "NCX");
    conv_op.set_attr<std::string>("filter_format", "OIX");

    // prepare logical tensor
    impl::logical_tensor_t diff_src = utils::logical_tensor_init(
            0, {8, 3, 224, 224}, impl::data_type::f32);
    impl::logical_tensor_t weights = utils::logical_tensor_init(
            1, {16, 3, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t diff_dst = utils::logical_tensor_init(
            2, {8, 16, 222, 222}, impl::data_type::f32);

    conv_op.add_input(diff_dst);
    conv_op.add_input(weights);
    conv_op.add_output(diff_src);

    impl::graph_t g;
    g.add_op(&conv_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("conv_data_bw_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);
    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&diff_dst, &weights};
    std::vector<const impl::logical_tensor_t *> outputs {&diff_src};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

    impl::logical_tensor_t lt;
    cp.query_logical_tensor(diff_src.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);
}

TEST(operator_compile, convolution_backward_weights_compile_fp32) {
    using dims = dnnl::graph::impl::dnnl_impl::dims;

    impl::engine_t &eng = get_engine();

    impl::op_t conv_op(impl::dnnl_impl::op_kind::conv_bwd_f_biasadd_bwd);
    conv_op.set_attr<dims>("strides", dims {1, 1});
    conv_op.set_attr<dims>("dilations", dims {1, 1});
    conv_op.set_attr<dims>("pads_begin", dims {0, 0});
    conv_op.set_attr<dims>("pads_end", dims {0, 0});
    conv_op.set_attr<int64_t>("groups", 0);
    conv_op.set_attr<std::string>("data_format", "NCX");
    conv_op.set_attr<std::string>("filter_format", "OIX");

    // prepare logical tensor
    impl::logical_tensor_t src = utils::logical_tensor_init(
            0, {8, 3, 224, 224}, impl::data_type::f32);
    impl::logical_tensor_t diff_weights = utils::logical_tensor_init(
            2, {16, 3, 3, 3}, impl::data_type::f32, impl::layout_type::any);
    impl::logical_tensor_t diff_bias = utils::logical_tensor_init(
            3, dims {16}, impl::data_type::f32, impl::layout_type::any);
    impl::logical_tensor_t diff_dst = utils::logical_tensor_init(
            4, {8, 16, 222, 222}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> inputs {src, diff_dst};
    std::vector<impl::logical_tensor_t> outputs {diff_weights, diff_bias};

    size_t operator_num = get_dnnl_kernel_registry().get_register_kernels_num();
    ASSERT_NE(operator_num, 0);

    auto conv_bwd_kernel = get_dnnl_kernel_registry().create_kernel(conv_op);
    ASSERT_TRUE(conv_bwd_kernel);

    conv_bwd_kernel->compile(&conv_op, &eng, inputs, outputs);

    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);
    ASSERT_EQ(outputs[1].layout_type, impl::layout_type::opaque);
}

TEST(operator_compile, convtranspose_compile_fp32) {
    using dims = impl::dnnl_impl::dims;

    impl::engine_t &eng = get_engine();

    impl::op_t convtranspose_op(impl::op_kind::ConvTranspose);
    convtranspose_op.set_attr<dims>("strides", dims {1, 1});
    convtranspose_op.set_attr<dims>("dilations", dims {1, 1});
    convtranspose_op.set_attr<dims>("pads_begin", dims {0, 0});
    convtranspose_op.set_attr<dims>("pads_end", dims {0, 0});
    convtranspose_op.set_attr<int64_t>("groups", 1);
    convtranspose_op.set_attr<std::string>("data_format", "NCX");
    convtranspose_op.set_attr<std::string>("filter_format", "OIX");

    // prepare logical tensor
    impl::logical_tensor_t src = utils::logical_tensor_init(
            0, {8, 3, 222, 222}, impl::data_type::f32);
    impl::logical_tensor_t weight = utils::logical_tensor_init(
            1, {16, 3, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst = utils::logical_tensor_init(
            2, {8, 16, 224, 224}, impl::data_type::f32, impl::layout_type::any);

    convtranspose_op.add_input(src);
    convtranspose_op.add_input(weight);
    convtranspose_op.add_output(dst);

    impl::graph_t g;
    g.add_op(&convtranspose_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("convtranspose_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);
    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src, &weight};
    std::vector<const impl::logical_tensor_t *> outputs {&dst};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);
    impl::logical_tensor_t lt;
    cp.query_logical_tensor(dst.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::opaque);
}

void ref_batchnorm_fwd(impl::dim_t mb, impl::dim_t ic, impl::dim_t ih,
        impl::dim_t iw, impl::tensor_t *src, impl::tensor_t *dst,
        impl::tensor_t *scale, impl::tensor_t *shift,
        impl::tensor_t *mean = nullptr, impl::tensor_t *variance = nullptr,
        float epsilon = 0.001f,
        std::function<float(const float)> activation = nullptr,
        bool channel_last = false) {
    if (!activation) activation = [](const float v) { return v; };

    float *mean_ptr = nullptr;
    float *variance_ptr = nullptr;

    size_t mb_ = static_cast<size_t>(mb);
    size_t ic_ = static_cast<size_t>(ic);
    size_t ih_ = static_cast<size_t>(ih);
    size_t iw_ = static_cast<size_t>(iw);

    std::unique_ptr<float[]> mean_data;
    std::unique_ptr<float[]> variance_data;
    if (!mean) {
        mean_data.reset(new float[static_cast<size_t>(ic)]);
        mean_ptr = mean_data.get();
    } else {
        mean_ptr = mean->get_data_handle<float>();
    }
    if (!variance) {
        variance_data.reset(new float[static_cast<size_t>(ic)]);
        variance_ptr = variance_data.get();
    } else {
        variance_ptr = variance->get_data_handle<float>();
    }

    float *src_ptr = src->get_data_handle<float>();

    // derives ref mean
    for (size_t channel = 0; channel < ic_; ++channel) {
        mean_ptr[channel] = 0.f;
        for (size_t batch = 0; batch < mb_; ++batch) {
            for (size_t h = 0; h < ih_; ++h) {
                for (size_t w = 0; w < iw_; ++w) {
                    size_t idx = channel_last
                            ? channel + w * ic_ + h * iw_ * ic_
                                    + batch * ih_ * iw_ * ic_
                            : w + h * iw_ + channel * ih_ * iw_
                                    + batch * ic_ * ih_ * iw_;
                    mean_ptr[channel] += src_ptr[idx];
                }
            }
        }
        mean_ptr[channel] /= static_cast<float>(mb_ * ih_ * iw_);
    }
    // derives ref variance
    for (size_t channel = 0; channel < ic_; ++channel) {
        variance_ptr[channel] = 0.f;
        for (size_t batch = 0; batch < mb_; ++batch) {
            for (size_t h = 0; h < ih_; ++h) {
                for (size_t w = 0; w < iw_; ++w) {
                    size_t idx = channel_last
                            ? channel + w * ic_ + h * iw_ * ic_
                                    + batch * ih_ * iw_ * ic_
                            : w + h * iw_ + channel * ih_ * iw_
                                    + batch * ic_ * ih_ * iw_;
                    float variance_tmp = src_ptr[idx] - mean_ptr[channel];
                    variance_ptr[channel] += variance_tmp * variance_tmp;
                }
            }
        }
        variance_ptr[channel] /= static_cast<float>(mb * ih * iw);
    }

    float *dst_ptr = dst->get_data_handle<float>();
    float *scale_ptr = scale->get_data_handle<float>();
    float *shift_ptr = shift->get_data_handle<float>();
    for (size_t batch = 0; batch < mb_; ++batch) {
        for (size_t channel = 0; channel < ic_; ++channel) {
            for (size_t h = 0; h < ih_; ++h) {
                for (size_t w = 0; w < iw_; ++w) {
                    size_t idx = channel_last
                            ? channel + w * ic_ + h * iw_ * ic_
                                    + batch * ih_ * iw_ * ic_
                            : w + h * iw_ + channel * ih_ * iw_
                                    + batch * ic_ * ih_ * iw_;
                    dst_ptr[idx] = activation(scale_ptr[channel]
                                    * (src_ptr[idx] - mean_ptr[channel])
                                    / sqrtf(variance_ptr[channel] + epsilon)
                            + shift_ptr[channel]);
                }
            }
        }
    }
}

struct dnnl_graph_test_batchnorm_params {
    impl::op_kind_t op_kind;
    impl::dim_t N;
    impl::dim_t IC;
    impl::dim_t IH;
    impl::dim_t IW;
    float epsilon;
    std::string data_format;
    std::string backend_name;
    impl::data_type_t data_type;
};

class test_batchnorm_compile
    : public ::testing::TestWithParam<dnnl_graph_test_batchnorm_params> {
public:
    void TestBatchnorm(bool dims_in_order = true) {
        using dims = dnnl::graph::impl::dnnl_impl::dims;

        auto params = ::testing::TestWithParam<
                dnnl_graph_test_batchnorm_params>::GetParam();

        impl::op_t batchnorm_op(params.op_kind);
        batchnorm_op.set_attr("epsilon", params.epsilon);
        batchnorm_op.set_attr("data_format", params.data_format);

        impl::engine_t &engine = get_engine();

        // Tensor dimensions.
        const impl::dim_t N = params.N, // batch size
                IC = params.IC, // channels
                IH = params.IH, // tensor height
                IW = params.IW; // tensor width
        // Source (src) and destination (dst) tensors dimensions.
        dims src_dims;

        // |                     | == NCX?   | == NXC?   |
        // |---------------------|-----------|-----------|
        // |in-order (true)      | NCHW      | NHWC      |
        // |out-of-order (false) | NHWC      | HCHW      |
        if ((params.data_format == "NCX") ^ dims_in_order)
            src_dims = {N, IH, IW, IC};
        else
            src_dims = {N, IC, IH, IW};
        // Scale/shift tensor dimensions.
        dims scale_dims = {IC};
        dims shift_dims = {IC};

        // prepare logical tensor
        impl::logical_tensor_t src
                = utils::logical_tensor_init(0, src_dims, impl::data_type::f32);
        impl::logical_tensor_t scale = utils::logical_tensor_init(
                1, scale_dims, impl::data_type::f32);
        impl::logical_tensor_t shift = utils::logical_tensor_init(
                2, shift_dims, impl::data_type::f32);
        impl::logical_tensor_t dst
                = utils::logical_tensor_init(3, src_dims, impl::data_type::f32);

        // Allocate buffers.
        test::vector<float> src_data(static_cast<size_t>(N * IC * IH * IW));
        test::vector<float> scale_data(static_cast<size_t>(IC));
        test::vector<float> shift_data(static_cast<size_t>(IC));
        test::vector<float> dst_data(src_data.size(), 0.0);
        test::vector<float> ref_dst_data(src_data.size(), 0.0);
        // Initialize src.
        std::generate(src_data.begin(), src_data.end(), []() {
            static int i = 0;
            return std::cos(static_cast<float>(i++) / 10.f);
        });
        // Initialize scale.
        std::generate(scale_data.begin(), scale_data.end(), []() {
            static int i = 0;
            return std::sin(static_cast<float>(i++) * 2.f);
        });
        // Initialize shift.
        std::generate(shift_data.begin(), shift_data.end(), []() {
            static int i = 0;
            return std::tan(i++);
        });

        auto &op_factory = get_dnnl_kernel_registry();
        auto bn_kernel = op_factory.create_kernel(batchnorm_op);
        ASSERT_TRUE(bn_kernel);

        std::vector<impl::logical_tensor_t> inputs {src, scale, shift};
        std::vector<impl::logical_tensor_t> outputs {dst};

        // compile the bn operator
        bn_kernel->compile(&batchnorm_op, &engine, inputs, outputs);

        ASSERT_EQ(outputs[0].layout_type, impl::layout_type::strided);

        impl::tensor_t src_ts(src, &engine, src_data.data());
        impl::tensor_t scale_ts(scale, &engine, scale_data.data());
        impl::tensor_t shift_ts(shift, &engine, shift_data.data());
        impl::tensor_t dst_ts(dst, &engine, dst_data.data());
        impl::tensor_t ref_dst_ts(dst, &engine, ref_dst_data.data());

        impl::stream_t &strm = get_stream();

        if (dims_in_order) {
            bn_kernel->execute(&batchnorm_op, &strm,
                    {src_ts, scale_ts, shift_ts}, {dst_ts});
            strm.wait();

            std::function<float(const float)> activation = nullptr;
            if (params.op_kind == impl::dnnl_impl::op_kind::bn_relu) {
                activation = [](const float v) { return v < 0.f ? 0.f : v; };
            }
            ref_batchnorm_fwd(N, IC, IH, IW, &src_ts, &ref_dst_ts, &scale_ts,
                    &shift_ts, nullptr, nullptr, params.epsilon, activation,
                    params.data_format == "NXC");

            for (size_t i = 0; i < dst_data.size(); ++i) {
                ASSERT_NEAR(dst_data[i], ref_dst_data[i], 1.e-6f);
            }
        } else {
            EXPECT_THROW(bn_kernel->execute(&batchnorm_op, &strm,
                                 {src_ts, scale_ts, shift_ts}, {dst_ts}),
                    std::runtime_error);
            strm.wait();
        };
    }

    void Test() {
        TestBatchnorm(true);
        TestBatchnorm(false);
    }
};

TEST_P(test_batchnorm_compile, TestBatchnormCompile) {
    Test();
}

INSTANTIATE_TEST_SUITE_P(TestBatchnormCompile, test_batchnorm_compile,
        ::testing::Values(
                dnnl_graph_test_batchnorm_params {
                        impl::op_kind::BatchNormInference, 3, 3, 2, 2, 0.001f,
                        "NCX", "dnnl", impl::data_type::f32},
                dnnl_graph_test_batchnorm_params {
                        impl::op_kind::BatchNormInference, 3, 3, 2, 2, 0.001f,
                        "NXC", "dnnl", impl::data_type::f32},
                dnnl_graph_test_batchnorm_params {
                        impl::dnnl_impl::op_kind::bn_relu, 3, 3, 2, 2, 0.001f,
                        "NCX", "dnnl", impl::data_type::f32},
                dnnl_graph_test_batchnorm_params {
                        impl::dnnl_impl::op_kind::bn_relu, 3, 3, 2, 2, 0.001f,
                        "NXC", "dnnl", impl::data_type::f32}));

TEST(operator_compile, bn_compile_bwd_fp32) {
    using dims = dnnl::graph::impl::dnnl_impl::dims;

    impl::op_t bn_op(impl::op_kind::BatchNormTrainingBackprop);
    bn_op.set_attr<float>("epsilon", 0.0);
    bn_op.set_attr<std::string>("data_format", "NCX");

    impl::engine_t &engine = get_engine();

    // Tensor dimensions.
    const impl::dim_t N = 1, // batch size
            IC = 1, // channels
            IH = 2, // tensor height
            IW = 2; // tensor width
    // Source (src) and destination (dst) tensors dimensions.
    dims src_dims = {N, IC, IH, IW};
    // Scale/shift tensor dimensions.
    dims scale_dims = {IC};
    dims shift_dims = {IC};
    dims mean_dims = {IC};
    dims varience_dims = {IC};

    // prepare logical tensor
    impl::logical_tensor_t src = utils::logical_tensor_init(
            0, src_dims, impl::data_type::f32, impl::layout_type::strided);
    impl::logical_tensor_t scale = utils::logical_tensor_init(
            1, scale_dims, impl::data_type::f32, impl::layout_type::strided);
    impl::logical_tensor_t mean = utils::logical_tensor_init(
            2, mean_dims, impl::data_type::f32, impl::layout_type::strided);
    impl::logical_tensor_t varience = utils::logical_tensor_init(
            3, varience_dims, impl::data_type::f32, impl::layout_type::strided);
    impl::logical_tensor_t diff_dst = utils::logical_tensor_init(
            4, src_dims, impl::data_type::f32, impl::layout_type::strided);
    impl::logical_tensor_t diff_src = utils::logical_tensor_init(
            5, src_dims, impl::data_type::f32, impl::layout_type::strided);
    impl::logical_tensor_t diff_scale = utils::logical_tensor_init(
            6, scale_dims, impl::data_type::f32, impl::layout_type::strided);
    impl::logical_tensor_t diff_shift = utils::logical_tensor_init(
            7, shift_dims, impl::data_type::f32, impl::layout_type::strided);

    // Allocate buffers.
    test::vector<float> src_data {1.0, 2.0, 3.0, 4.0};
    test::vector<float> scale_data {2.0, 0.0, 0.0, 0.0};
    test::vector<float> mean_data {2.5, 0.0, 0.0, 0.0};
    test::vector<float> varience_data {1.25, 0.0, 0.0, 0.0};
    test::vector<float> diff_dst_data {0.1f, 0.1f, 0.2f, 0.2f};
    test::vector<float> diff_src_data(src_data.size(), 0.0);
    test::vector<float> diff_scale_data(scale_data.size(), 0.0);
    test::vector<float> diff_shift_data(scale_data.size(), 0.0);

    // expected results
    test::vector<float> ref_diff_src_data {
            0.17888544f, 0.17888544f, 0.35777088f, 0.35777088f};
    test::vector<float> ref_diff_scale_data {0.178885f, 0.0, 0.0, 0.0};
    test::vector<float> ref_diff_shift_data {0.6f, 0.0, 0.0, 0.0};

    auto &op_factory = get_dnnl_kernel_registry();
    auto bn_kernel = op_factory.create_kernel(bn_op);
    ASSERT_TRUE(bn_kernel);

    std::vector<impl::logical_tensor_t> inputs {
            src, diff_dst, scale, mean, varience};
    std::vector<impl::logical_tensor_t> outputs {
            diff_src, diff_scale, diff_shift};

    // compile the bn operator
    bn_kernel->compile(&bn_op, &engine, inputs, outputs);

    impl::tensor_t src_ts(src, &engine, src_data.data());
    impl::tensor_t scale_ts(scale, &engine, scale_data.data());
    impl::tensor_t mean_ts(mean, &engine, mean_data.data());
    impl::tensor_t varience_ts(varience, &engine, varience_data.data());
    impl::tensor_t diff_dst_ts(diff_dst, &engine, diff_dst_data.data());
    impl::tensor_t diff_src_ts(diff_src, &engine, diff_src_data.data());
    impl::tensor_t diff_scale_ts(diff_scale, &engine, diff_scale_data.data());
    impl::tensor_t diff_shift_ts(diff_shift, &engine, diff_shift_data.data());

    impl::stream_t &strm = get_stream();
    bn_kernel->execute(&bn_op, &strm,
            {src_ts, diff_dst_ts, scale_ts, mean_ts, varience_ts},
            {diff_src_ts, diff_scale_ts, diff_shift_ts});
    strm.wait();
    for (size_t i = 0; i < diff_src_data.size(); ++i) {
        auto absolute_error
                = std::fabs(diff_src_data[i] - ref_diff_src_data[i]);
        ASSERT_TRUE(absolute_error < 0.001);
    }
    for (size_t i = 0; i < diff_scale_data.size(); ++i) {
        auto absolute_error
                = std::fabs(diff_scale_data[i] - ref_diff_scale_data[i]);
        ASSERT_TRUE(absolute_error < 0.001);
    }
    for (size_t i = 0; i < diff_shift_data.size(); ++i) {
        auto absolute_error
                = std::fabs(diff_shift_data[i] - ref_diff_shift_data[i]);
        ASSERT_TRUE(absolute_error < 0.001);
    }
}

struct dnnl_graph_test_concat_params {
    std::vector<impl::dim_t> src0_shape;
    std::vector<impl::dim_t> src1_shape;
    std::vector<impl::dim_t> dst_shape;
    test::vector<float> src0_data;
    test::vector<float> src1_data;
    test::vector<float> ref_dst;
    int64_t axis;
    bool is_nhwc;
};

class test_concat_compile
    : public ::testing::TestWithParam<dnnl_graph_test_concat_params> {
public:
    void TestConcat() {
        const auto params = ::testing::TestWithParam<
                dnnl_graph_test_concat_params>::GetParam();

        const std::vector<impl::dim_t> &src0_dims = params.src0_shape;
        const std::vector<impl::dim_t> &src1_dims = params.src1_shape;
        const std::vector<impl::dim_t> &dst_dims = params.dst_shape;
        test::vector<float> src0_data = params.src0_data;
        test::vector<float> src1_data = params.src1_data;
        test::vector<float> dst_data(params.ref_dst.size(), 0.);
        const auto &axis = params.axis;

        impl::op_t concat_op(impl::op_kind::Concat);
        concat_op.set_attr<int64_t>("axis", axis);

        auto &op_factory = get_dnnl_kernel_registry();
        const auto concat_kernel = op_factory.create_kernel(concat_op);
        ASSERT_TRUE(concat_kernel);

        impl::logical_tensor_t src0_lt = utils::logical_tensor_init(
                0, src0_dims, impl::data_type::f32, impl::layout_type::strided);

        impl::logical_tensor_t src1_lt;
        if (!params.is_nhwc) {
            src1_lt = utils::logical_tensor_init(1, src1_dims,
                    impl::data_type::f32, impl::layout_type::strided);
        } else {
            auto permuted_strides = utils::compute_dense_strides(src1_dims);
            // convert strides to nhwc format
            permuted_strides.insert(
                    permuted_strides.begin() + 1, permuted_strides.back());
            permuted_strides.pop_back();
            src1_lt = utils::logical_tensor_init(
                    1, src1_dims, permuted_strides, impl::data_type::f32);
        }
        impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
                2, dst_dims, impl::data_type::f32, impl::layout_type::strided);

        const auto &eng = get_engine();
        concat_kernel->compile(&concat_op, &eng, {src0_lt, src1_lt}, {dst_lt});

        impl::tensor_t src0_ts(src0_lt, &eng, src0_data.data());
        impl::tensor_t src1_ts(src1_lt, &eng, src1_data.data());
        impl::tensor_t dst_ts(dst_lt, &eng, dst_data.data());

        auto &strm = get_stream();

        concat_kernel->execute(&concat_op, &strm, {src0_ts, src1_ts}, {dst_ts});
        strm.wait();

        for (size_t i = 0; i < dst_data.size(); ++i) {
            ASSERT_FLOAT_EQ(params.ref_dst[i], dst_data[i]);
        }
    }
};

TEST_P(test_concat_compile, TestConcatCompile) {
    TestConcat();
}

INSTANTIATE_TEST_SUITE_P(TestConcatCompile, test_concat_compile,
        ::testing::Values(
                // 2D, axis = 0
                dnnl_graph_test_concat_params {{1, 2}, {1, 2}, {2, 2}, {1., 1.},
                        {2., 2.}, {1., 1., 2., 2.}, 0, false},
                // 2D, axis = 1
                dnnl_graph_test_concat_params {{1, 2}, {1, 2}, {1, 4}, {1., 1.},
                        {2., 2.}, {1., 1., 2., 2.}, 1, false},
                // 3D, axis = 0
                dnnl_graph_test_concat_params {{1, 2, 2}, {1, 2, 2}, {2, 2, 2},
                        {1., 1., 2., 2.}, {3., 3., 4., 4.},
                        {1., 1., 2., 2., 3., 3., 4., 4.}, 0, false},
                // 3D, axis = 1
                dnnl_graph_test_concat_params {{1, 2, 2}, {1, 2, 2}, {1, 4, 2},
                        {1., 1., 2., 2.}, {3., 3., 4., 4.},
                        {1., 1., 2., 2., 3., 3., 4., 4.}, 1, false},
                // 3D, axis = 2
                dnnl_graph_test_concat_params {{1, 2, 2}, {1, 2, 2}, {1, 2, 4},
                        {1., 1., 2., 2.}, {3., 3., 4., 4.},
                        {1., 1., 3., 3., 2., 2., 4., 4.}, 2, false},
                // 4D, axis = 0
                dnnl_graph_test_concat_params {{1, 2, 2, 2}, {1, 2, 2, 2},
                        {2, 2, 2, 2}, {1., 1., 2., 2., 3., 3., 4., 4.},
                        {5., 5., 6., 6., 7., 7., 8., 8.},
                        {1., 1., 2., 2., 3., 3., 4., 4., 5., 5., 6., 6., 7., 7.,
                                8., 8.},
                        0, false},
                // 4D, axis = 0
                dnnl_graph_test_concat_params {{1, 2, 2, 2}, {1, 2, 2, 2},
                        {2, 2, 2, 2}, {1., 1., 2., 2., 3., 3., 4., 4.},
                        {5., 5., 6., 6., 7., 7., 8., 8.},
                        {1., 1., 2., 2., 3., 3., 4., 4., 5., 6., 7., 8., 5., 6.,
                                7., 8.},
                        0, true},
                // 4D, axis = 1
                dnnl_graph_test_concat_params {{1, 2, 2, 2}, {1, 2, 2, 2},
                        {1, 4, 2, 2}, {1., 1., 2., 2., 3., 3., 4., 4.},
                        {5., 5., 6., 6., 7., 7., 8., 8.},
                        {1., 1., 2., 2., 3., 3., 4., 4., 5., 5., 6., 6., 7., 7.,
                                8., 8.},
                        1, false},
                // 4D, axis = 2
                dnnl_graph_test_concat_params {{1, 2, 2, 2}, {1, 2, 2, 2},
                        {1, 2, 4, 2}, {1., 1., 2., 2., 3., 3., 4., 4.},
                        {5., 5., 6., 6., 7., 7., 8., 8.},
                        {1., 1., 2., 2., 5., 5., 6., 6., 3., 3., 4., 4., 7., 7.,
                                8., 8.},
                        2, false},
                // 4D, axis = 3
                dnnl_graph_test_concat_params {{1, 2, 2, 2}, {1, 2, 2, 2},
                        {1, 2, 2, 4}, {1., 1., 2., 2., 3., 3., 4., 4.},
                        {5., 5., 6., 6., 7., 7., 8., 8.},
                        {1., 1., 5., 5., 2., 2., 6., 6., 3., 3., 7., 7., 4., 4.,
                                8., 8.},
                        3, false},
                // 4D, axis = -1
                dnnl_graph_test_concat_params {{1, 2, 2, 2}, {1, 2, 2, 2},
                        {1, 2, 2, 4}, {1., 1., 2., 2., 3., 3., 4., 4.},
                        {5., 5., 6., 6., 7., 7., 8., 8.},
                        {1., 1., 5., 5., 2., 2., 6., 6., 3., 3., 7., 7., 4., 4.,
                                8., 8.},
                        -1, false}));

TEST(operator_kernel, relu) {
    test::vector<float> src {-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0};
    test::vector<float> ref_dst {0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.5, 2.0};
    test::vector<float> dst(src.size(), 0.0);

    impl::op_t relu_op(impl::op_kind::ReLU);

    auto &op_factory = get_dnnl_kernel_registry();
    auto relu_kernel = op_factory.create_kernel(relu_op);
    ASSERT_TRUE(relu_kernel);

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(1, {1, 3, 3}, impl::data_type::f32);

    impl::engine_t &eng = get_engine();
    // compile the relu operator
    relu_kernel->compile(&relu_op, &eng, {src_lt}, {dst_lt});

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    relu_kernel->execute(&relu_op, &strm, {src_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < src.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, relu_backward) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src {-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0};
    test::vector<float> diff_dst {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    test::vector<float> ref_diff_src {
            0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 6.0, 7.0, 8.0};
    test::vector<float> diff_src(src.size(), 0.0);

    impl::op_t relu_op(impl::op_kind::ReLUBackprop);

    auto &op_factory = get_dnnl_kernel_registry();
    auto relu_kernel = op_factory.create_kernel(relu_op);
    ASSERT_TRUE(relu_kernel);

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t diff_src_lt
            = utils::logical_tensor_init(1, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t diff_dst_lt
            = utils::logical_tensor_init(2, {1, 3, 3}, impl::data_type::f32);

    // compile the relu backward operator
    relu_kernel->compile(&relu_op, &eng, {diff_dst_lt, src_lt}, {diff_src_lt});

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t diff_src_ts(diff_src_lt, &eng, diff_src.data());
    impl::tensor_t diff_dst_ts(diff_dst_lt, &eng, diff_dst.data());

    impl::stream_t &strm = get_stream();
    relu_kernel->execute(&relu_op, &strm, {diff_dst_ts, src_ts}, {diff_src_ts});
    strm.wait();
    for (size_t i = 0; i < diff_src.size(); ++i) {
        ASSERT_FLOAT_EQ(diff_src[i], ref_diff_src[i]);
    }
}

TEST(operator_kernel, gelu) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src {-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0};
    test::vector<float> ref_dst {-0.0455001f, -0.10021085f, -0.15865527f,
            -0.15426877f, 0.f, 0.3457312f, 0.84134465f, 1.399789f, 1.9544998f};
    test::vector<float> dst(src.size(), 0.0);

    impl::op_t gelu_op(impl::op_kind::GELU);

    auto &op_factory = get_dnnl_kernel_registry();
    auto gelu_kernel = op_factory.create_kernel(gelu_op);
    ASSERT_TRUE(gelu_kernel);

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(1, {1, 3, 3}, impl::data_type::f32);

    gelu_kernel->compile(&gelu_op, &eng, {src_lt}, {dst_lt});

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    gelu_kernel->execute(&gelu_op, &strm, {src_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < src.size(); ++i) {
        ASSERT_NEAR(dst[i], ref_dst[i], 1.e-6f);
    }
}

TEST(operator_kernel, gelu_backward) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src {-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0};
    test::vector<float> diff_dst {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    test::vector<float> ref_diff_src {0.0f, -0.1274692f, -0.1666309f,
            0.3975146f, 2.0f, 4.3374756f, 6.4998928f, 7.8922843f, 8.6818544f};
    test::vector<float> diff_src(src.size(), 0.0);

    impl::op_t gelu_op(impl::op_kind::GELUBackprop);

    auto &op_factory = get_dnnl_kernel_registry();
    auto gelu_kernel = op_factory.create_kernel(gelu_op);
    ASSERT_TRUE(gelu_kernel);

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t diff_src_lt
            = utils::logical_tensor_init(1, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t diff_dst_lt
            = utils::logical_tensor_init(2, {1, 3, 3}, impl::data_type::f32);

    // compile the helu backward operator
    gelu_kernel->compile(&gelu_op, &eng, {diff_dst_lt, src_lt}, {diff_src_lt});

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t diff_src_ts(diff_src_lt, &eng, diff_src.data());
    impl::tensor_t diff_dst_ts(diff_dst_lt, &eng, diff_dst.data());

    impl::stream_t &strm = get_stream();
    gelu_kernel->execute(&gelu_op, &strm, {diff_dst_ts, src_ts}, {diff_src_ts});
    strm.wait();
    for (size_t i = 0; i < diff_src.size(); ++i) {
        ASSERT_FLOAT_EQ(diff_src[i], ref_diff_src[i]);
    }
}

TEST(operator_kernel, add) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src0 {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    test::vector<float> src1 {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> ref_dst {3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0};
    test::vector<float> dst(src0.size(), 0.0);

    impl::op_t add_op(impl::op_kind::Add);

    auto &op_factory = get_dnnl_kernel_registry();
    auto add_kernel = op_factory.create_kernel(add_op);
    ASSERT_TRUE(add_kernel);

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            2, {1, 3, 3}, impl::data_type::f32, impl::layout_type::any);
    // compile the add operator
    std::vector<impl::logical_tensor_t> inputs {src0_lt, src1_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};
    add_kernel->compile(&add_op, &eng, inputs, outputs);

    ASSERT_EQ(dst_lt.layout_type, impl::layout_type::any);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src0_ts(src0_lt, &eng, src0.data());
    impl::tensor_t src1_ts(src1_lt, &eng, src1.data());
    impl::tensor_t dst_ts(outputs[0], &eng, dst.data());

    impl::stream_t &strm = get_stream();
    add_kernel->execute(&add_op, &strm, {src0_ts, src1_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }

    // the second iteration
    test::vector<float> src0_2nd {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    test::vector<float> src1_2nd {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    test::vector<float> ref_dst_2nd {
            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> dst_2nd(src0_2nd.size(), 0.0);

    impl::tensor_t src0_2nd_ts(src0_lt, &eng, src0_2nd.data());
    impl::tensor_t src1_2nd_ts(src1_lt, &eng, src1_2nd.data());
    impl::tensor_t dst_2nd_ts(outputs[0], &eng, dst_2nd.data());
    add_kernel->execute(
            &add_op, &strm, {src0_2nd_ts, src1_2nd_ts}, {dst_2nd_ts});
    strm.wait();

    for (size_t i = 0; i < src0_2nd.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_2nd[i], ref_dst_2nd[i]);
    }
}

TEST(operator_kernel, different_format_add) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src0 {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    test::vector<float> src1 {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    test::vector<float> ref_dst {2.0, 5.0, 8.0, 3.0, 6.0, 9.0, 4.0, 7.0, 10.0};
    test::vector<float> dst(src0.size(), 0.0);

    impl::op_t add_op(impl::op_kind::Add);

    auto &op_factory = get_dnnl_kernel_registry();
    auto add_kernel = op_factory.create_kernel(add_op);
    ASSERT_TRUE(add_kernel);

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt = utils::logical_tensor_init(
            1, {1, 1, 3, 3}, {3, 3, 1, 3}, impl::data_type::f32); // abdc format
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {1, 1, 3, 3}, impl::data_type::f32);
    // compile the add operator
    add_kernel->compile(&add_op, &eng, {src0_lt, src1_lt}, {dst_lt});

    impl::tensor_t src0_ts(src0_lt, &eng, src0.data());
    impl::tensor_t src1_ts(src1_lt, &eng, src1.data());
    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    add_kernel->execute(&add_op, &strm, {src0_ts, src1_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, broadcast_add) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src0 {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    test::vector<float> src1 {
            2.0, 2.0, 2.0}; // bianary op's src1 support broadcast
    test::vector<float> ref_dst {3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0};
    test::vector<float> dst(src0.size(), 0.0);

    impl::op_t broadcast_add_op(impl::op_kind::Add);

    auto &op_factory = get_dnnl_kernel_registry();
    auto broadcast_add_kernel = op_factory.create_kernel(broadcast_add_op);
    ASSERT_TRUE(broadcast_add_kernel);

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 2, 3, 3}, impl::data_type::f32);
    // src1 will first be unsequeeze to {1,1,1,3} and then broadcast
    // to {1,2,3,3}
    impl::logical_tensor_t src1_lt = utils::logical_tensor_init(
            1, std::vector<impl::dim_t> {3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {1, 2, 3, 3}, impl::data_type::f32);

    // compile the add operator
    broadcast_add_kernel->compile(
            &broadcast_add_op, &eng, {src0_lt, src1_lt}, {dst_lt});

    impl::tensor_t src0_ts(src0_lt, &eng, src0.data());
    impl::tensor_t src1_ts(src1_lt, &eng, src1.data());
    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    broadcast_add_kernel->execute(
            &broadcast_add_op, &strm, {src0_ts, src1_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }

    // the second iteration
    test::vector<float> src0_2nd {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    test::vector<float> src1_2nd {1.0, 1.0, 1.0};
    test::vector<float> ref_dst_2nd {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> dst_2nd(src0_2nd.size(), 0.0);

    impl::tensor_t src0_2nd_ts(src0_lt, &eng, src0_2nd.data());
    impl::tensor_t src1_2nd_ts(src1_lt, &eng, src1_2nd.data());
    impl::tensor_t dst_2nd_ts(dst_lt, &eng, dst_2nd.data());
    broadcast_add_kernel->execute(
            &broadcast_add_op, &strm, {src0_2nd_ts, src1_2nd_ts}, {dst_2nd_ts});
    strm.wait();

    for (size_t i = 0; i < src0_2nd.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_2nd[i], ref_dst_2nd[i]);
    }
}

TEST(operator_kernel, multidirectional_broadcast_add_BA) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src0 {1.0, 1.0, 1.0};
    test::vector<float> src1 {2.0, 2.0, 2.0};
    test::vector<float> ref_dst {3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0};
    test::vector<float> dst(ref_dst.size(), 0.0);

    impl::op_t broadcast_add_op(impl::op_kind::Add);

    auto &op_factory = get_dnnl_kernel_registry();
    auto broadcast_add_kernel = op_factory.create_kernel(broadcast_add_op);
    ASSERT_TRUE(broadcast_add_kernel);

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {3}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt = utils::logical_tensor_init(
            1, std::vector<impl::dim_t> {3, 1}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {3, 3}, impl::data_type::f32);

    // compile the add operator
    broadcast_add_kernel->compile(
            &broadcast_add_op, &eng, {src0_lt, src1_lt}, {dst_lt});

    impl::tensor_t src0_ts(src0_lt, &eng, src0.data());
    impl::tensor_t src1_ts(src1_lt, &eng, src1.data());
    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    broadcast_add_kernel->execute(
            &broadcast_add_op, &strm, {src0_ts, src1_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < ref_dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }

    // the second iteration
    test::vector<float> src0_2nd {1.0, 1.0, 1.0};
    test::vector<float> src1_2nd {1.0, 1.0, 1.0};
    test::vector<float> ref_dst_2nd {
            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> dst_2nd(ref_dst_2nd.size(), 0.0);

    impl::tensor_t src0_2nd_ts(src0_lt, &eng, src0_2nd.data());
    impl::tensor_t src1_2nd_ts(src1_lt, &eng, src1_2nd.data());
    impl::tensor_t dst_2nd_ts(dst_lt, &eng, dst_2nd.data());
    broadcast_add_kernel->execute(
            &broadcast_add_op, &strm, {src0_2nd_ts, src1_2nd_ts}, {dst_2nd_ts});
    strm.wait();

    for (size_t i = 0; i < ref_dst_2nd.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_2nd[i], ref_dst_2nd[i]);
    }
}

TEST(operator_kernel, multidirectional_broadcast_add_AB) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src0 {1.0, 1.0, 1.0};
    test::vector<float> src1 {2.0, 2.0, 2.0};
    test::vector<float> ref_dst {3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0};
    test::vector<float> dst(ref_dst.size(), 0.0);

    impl::op_t broadcast_add_op(impl::op_kind::Add);

    auto &op_factory = get_dnnl_kernel_registry();
    auto broadcast_add_kernel = op_factory.create_kernel(broadcast_add_op);
    ASSERT_TRUE(broadcast_add_kernel);

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {3, 1}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt = utils::logical_tensor_init(
            1, std::vector<impl::dim_t> {3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {3, 3}, impl::data_type::f32);

    // compile the add operator
    broadcast_add_kernel->compile(
            &broadcast_add_op, &eng, {src0_lt, src1_lt}, {dst_lt});

    impl::tensor_t src0_ts(src0_lt, &eng, src0.data());
    impl::tensor_t src1_ts(src1_lt, &eng, src1.data());
    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    broadcast_add_kernel->execute(
            &broadcast_add_op, &strm, {src0_ts, src1_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < ref_dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, multidirectional_broadcast_add_2x1x4_1x3x1) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src0(8, 1.0);
    test::vector<float> src1(3, 2.0);
    test::vector<float> ref_dst(24, 3.0);
    test::vector<float> dst(ref_dst.size(), 0.0);

    impl::op_t broadcast_add_op(impl::op_kind::Add);

    auto &op_factory = get_dnnl_kernel_registry();
    auto broadcast_add_kernel = op_factory.create_kernel(broadcast_add_op);
    ASSERT_TRUE(broadcast_add_kernel);

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {2, 1, 4}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt = utils::logical_tensor_init(
            1, std::vector<impl::dim_t> {1, 3, 1}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {2, 3, 4}, impl::data_type::f32);

    // compile the add operator
    broadcast_add_kernel->compile(
            &broadcast_add_op, &eng, {src0_lt, src1_lt}, {dst_lt});

    impl::tensor_t src0_ts(src0_lt, &eng, src0.data());
    impl::tensor_t src1_ts(src1_lt, &eng, src1.data());
    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    broadcast_add_kernel->execute(
            &broadcast_add_op, &strm, {src0_ts, src1_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < ref_dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, multidirectional_broadcast_add_2x1x1_3x4) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src0(2, 1.0);
    test::vector<float> src1(12, 2.0);
    test::vector<float> ref_dst(24, 3.0);
    test::vector<float> dst(ref_dst.size(), 0.0);

    impl::op_t broadcast_add_op(impl::op_kind::Add);

    auto &op_factory = get_dnnl_kernel_registry();
    auto broadcast_add_kernel = op_factory.create_kernel(broadcast_add_op);
    ASSERT_TRUE(broadcast_add_kernel);

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {2, 1, 1}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt = utils::logical_tensor_init(
            1, std::vector<impl::dim_t> {3, 4}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {2, 3, 4}, impl::data_type::f32);

    // compile the add operator
    broadcast_add_kernel->compile(
            &broadcast_add_op, &eng, {src0_lt, src1_lt}, {dst_lt});

    impl::tensor_t src0_ts(src0_lt, &eng, src0.data());
    impl::tensor_t src1_ts(src1_lt, &eng, src1.data());
    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    broadcast_add_kernel->execute(
            &broadcast_add_op, &strm, {src0_ts, src1_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < ref_dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, add_shape_mismatch_0) {
    impl::engine_t &eng = get_engine();

    impl::op_t add_op(impl::op_kind::Add);

    auto &op_factory = get_dnnl_kernel_registry();
    auto add_kernel = op_factory.create_kernel(add_op);
    ASSERT_TRUE(add_kernel);

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {8, 4, 256}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {2, 2, 512}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {8, 4, 256}, impl::data_type::f32);

    // compile the add operator
    auto ret = add_kernel->compile(&add_op, &eng, {src0_lt, src1_lt}, {dst_lt});
    ASSERT_EQ(ret, impl::status::invalid_shape);
}

TEST(operator_kernel, add_shape_mismatch_1) {
    impl::engine_t &eng = get_engine();

    impl::op_t add_op(impl::op_kind::Add);

    auto &op_factory = get_dnnl_kernel_registry();
    auto add_kernel = op_factory.create_kernel(add_op);
    ASSERT_TRUE(add_kernel);

    impl::logical_tensor_t src0_lt = utils::logical_tensor_init(
            0, {8, 15, 5, 7}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            2, {8, 15, 5, 7}, impl::data_type::f32);

    // compile the add operator
    auto ret = add_kernel->compile(&add_op, &eng, {src0_lt, src1_lt}, {dst_lt});
    ASSERT_EQ(ret, impl::status::success);
}

TEST(operator_kernel, add_shape_mismatch_2) {
    impl::engine_t &eng = get_engine();

    impl::op_t add_op(impl::op_kind::Add);

    auto &op_factory = get_dnnl_kernel_registry();
    auto add_kernel = op_factory.create_kernel(add_op);
    ASSERT_TRUE(add_kernel);

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt = utils::logical_tensor_init(
            1, {8, 15, 5, 7}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            2, {8, 15, 5, 7}, impl::data_type::f32);

    // compile the add operator
    auto ret = add_kernel->compile(&add_op, &eng, {src0_lt, src1_lt}, {dst_lt});
    ASSERT_EQ(ret, impl::status::success);
}

TEST(operator_kernel, reversed_different_format_broadcast_add) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src1 {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    test::vector<float> src0 {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    test::vector<float> ref_dst {2.0, 5.0, 8.0, 3.0, 6.0, 9.0, 4.0, 7.0, 10.0,
            2.0, 5.0, 8.0, 3.0, 6.0, 9.0, 4.0, 7.0, 10.0};
    test::vector<float> dst(src1.size(), 0.0);

    impl::op_t broadcast_add_op(impl::op_kind::Add);

    auto &op_factory = get_dnnl_kernel_registry();
    auto broadcast_add_kernel = op_factory.create_kernel(broadcast_add_op);
    ASSERT_TRUE(broadcast_add_kernel);

    // we reverse the order of src0 and src1
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(0, {1, 2, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t src0_lt = utils::logical_tensor_init(
            1, {3, 3}, {1, 3}, impl::data_type::f32); // ba format
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {1, 2, 3, 3}, impl::data_type::f32);

    // compile the add operator
    broadcast_add_kernel->compile(
            &broadcast_add_op, &eng, {src0_lt, src1_lt}, {dst_lt});

    impl::tensor_t src0_ts(src0_lt, &eng, src0.data());
    impl::tensor_t src1_ts(src1_lt, &eng, src1.data());
    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    broadcast_add_kernel->execute(
            &broadcast_add_op, &strm, {src0_ts, src1_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, bias_add) {
    impl::engine_t &eng = get_engine();
    auto &op_factory = get_dnnl_kernel_registry();

    test::vector<float> src {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    test::vector<float> bias {1.0, 2.0};
    test::vector<float> ref_dst1 {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0};
    test::vector<float> ref_dst2 {2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0,
            3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0};
    test::vector<float> dst(src.size(), 0.0);

    std::vector<std::vector<impl::dim_t>> src_shapes {
            {1, 2, 3, 3}, {1, 3, 3, 2}};
    std::vector<std::vector<impl::dim_t>> dst_shapes {
            {1, 2, 3, 3}, {1, 3, 3, 2}};
    std::vector<impl::dim_t> bias_shape {2};
    std::vector<std::string> data_formats {"NCX", "NXC"};

    for (size_t i = 0; i < data_formats.size(); i++) {
        impl::op_t bias_add_op(impl::op_kind::BiasAdd);
        bias_add_op.set_attr<std::string>("data_format", data_formats[i]);

        auto bias_add_kernel = op_factory.create_kernel(bias_add_op);
        ASSERT_TRUE(bias_add_kernel);

        impl::logical_tensor_t src_lt = utils::logical_tensor_init(
                0, src_shapes[i], impl::data_type::f32);
        impl::logical_tensor_t bias_lt = utils::logical_tensor_init(
                1, bias_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
                2, dst_shapes[i], impl::data_type::f32);

        // compile the add operator
        bias_add_kernel->compile(
                &bias_add_op, &eng, {src_lt, bias_lt}, {dst_lt});

        impl::tensor_t src_ts(src_lt, &eng, src.data());
        impl::tensor_t bias_ts(bias_lt, &eng, bias.data());
        impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

        impl::stream_t &strm = get_stream();
        bias_add_kernel->execute(
                &bias_add_op, &strm, {src_ts, bias_ts}, {dst_ts});
        strm.wait();

        if (i == 0) {
            for (size_t i = 0; i < src.size(); ++i) {
                ASSERT_FLOAT_EQ(dst[i], ref_dst1[i]);
            }
        } else {
            for (size_t i = 0; i < src.size(); ++i) {
                ASSERT_FLOAT_EQ(dst[i], ref_dst2[i]);
            }
        }
    }
}

TEST(operator_kernel, add_mul) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src0 {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    test::vector<float> src1 {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> post_src {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> ref_dst {6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0};
    test::vector<float> dst(src0.size(), 0.0);

    impl::op_t add_op(impl::dnnl_impl::op_kind::add_multiply);

    auto &op_factory = get_dnnl_kernel_registry();
    auto add_kernel = op_factory.create_kernel(add_op);
    ASSERT_TRUE(add_kernel);

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t post_src_lt
            = utils::logical_tensor_init(2, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            3, {1, 3, 3}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src0_lt, src1_lt, post_src_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};
    add_kernel->compile(&add_op, &eng, inputs, outputs);

    impl::tensor_t src0_ts(src0_lt, &eng, src0.data());
    impl::tensor_t src1_ts(src1_lt, &eng, src1.data());
    impl::tensor_t post_src_ts(post_src_lt, &eng, post_src.data());
    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    add_kernel->execute(
            &add_op, &strm, {src0_ts, src1_ts, post_src_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, add_mul_post_src_as_nxc) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src0 {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    test::vector<float> src1 {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> post_src {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    test::vector<float> ref_dst {
            3.0, 12.0, 21.0, 6.0, 15.0, 24.0, 9.0, 18.0, 27.0};
    test::vector<float> dst(src0.size(), 0.0);

    impl::op_t add_op(impl::dnnl_impl::op_kind::add_multiply);

    auto &op_factory = get_dnnl_kernel_registry();
    auto add_kernel = op_factory.create_kernel(add_op);
    ASSERT_TRUE(add_kernel);

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t post_src_lt = utils::logical_tensor_init(
            2, {1, 3, 3}, {9, 1, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            3, {1, 3, 3}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src0_lt, src1_lt, post_src_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};
    add_kernel->compile(&add_op, &eng, inputs, outputs);

    impl::tensor_t src0_ts(src0_lt, &eng, src0.data());
    impl::tensor_t src1_ts(src1_lt, &eng, src1.data());
    impl::tensor_t post_src_ts(post_src_lt, &eng, post_src.data());
    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    add_kernel->execute(
            &add_op, &strm, {src0_ts, src1_ts, post_src_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, add_relu) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src0 {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    test::vector<float> src1 {-2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> ref_dst {0.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0};
    test::vector<float> dst(src0.size(), 0.0);

    impl::op_t add_op(impl::dnnl_impl::op_kind::add_relu);

    auto &op_factory = get_dnnl_kernel_registry();
    auto add_kernel = op_factory.create_kernel(add_op);
    ASSERT_TRUE(add_kernel);

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            2, {1, 3, 3}, impl::data_type::f32, impl::layout_type::any);
    // compile the add operator
    std::vector<impl::logical_tensor_t> inputs {src0_lt, src1_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};
    add_kernel->compile(&add_op, &eng, inputs, outputs);

    ASSERT_EQ(dst_lt.layout_type, impl::layout_type::any);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src0_ts(src0_lt, &eng, src0.data());
    impl::tensor_t src1_ts(src1_lt, &eng, src1.data());
    impl::tensor_t dst_ts(outputs[0], &eng, dst.data());

    impl::stream_t &strm = get_stream();
    add_kernel->execute(&add_op, &strm, {src0_ts, src1_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, add_sigmoid) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src0 {2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0};
    test::vector<float> src1 {
            -2.0, -1.0, -2.0, -1.0, -2.0, -1.0, -2.0, -1.0, -2.0};
    test::vector<float> ref_dst {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
    test::vector<float> dst(src0.size(), 0.0);

    impl::op_t add_op(impl::dnnl_impl::op_kind::add_sigmoid);

    auto &op_factory = get_dnnl_kernel_registry();
    auto add_kernel = op_factory.create_kernel(add_op);
    ASSERT_TRUE(add_kernel);

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            2, {1, 3, 3}, impl::data_type::f32, impl::layout_type::any);
    // compile the add operator
    std::vector<impl::logical_tensor_t> inputs {src0_lt, src1_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};
    add_kernel->compile(&add_op, &eng, inputs, outputs);

    ASSERT_EQ(dst_lt.layout_type, impl::layout_type::any);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src0_ts(src0_lt, &eng, src0.data());
    impl::tensor_t src1_ts(src1_lt, &eng, src1.data());
    impl::tensor_t dst_ts(outputs[0], &eng, dst.data());

    impl::stream_t &strm = get_stream();
    add_kernel->execute(&add_op, &strm, {src0_ts, src1_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, mul) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src0 {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> src1 {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> ref_dst {4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0};
    test::vector<float> dst(src0.size(), 0.0);

    impl::op_t mul_op(impl::op_kind::Multiply);

    auto &op_factory = get_dnnl_kernel_registry();
    auto mul_kernel = op_factory.create_kernel(mul_op);
    ASSERT_TRUE(mul_kernel);

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {1, 3, 3}, impl::data_type::f32);

    // compile the add operator
    mul_kernel->compile(&mul_op, &eng, {src0_lt, src1_lt}, {dst_lt});

    impl::tensor_t src0_ts(src0_lt, &eng, src0.data());
    impl::tensor_t src1_ts(src1_lt, &eng, src1.data());
    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    mul_kernel->execute(&mul_op, &strm, {src0_ts, src1_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, mul_relu) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src0 {-2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> src1 {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> ref_dst {0.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0};
    test::vector<float> dst(src0.size(), 0.0);

    impl::op_t mul_op(impl::dnnl_impl::op_kind::multiply_relu);

    auto &op_factory = get_dnnl_kernel_registry();
    auto mul_kernel = op_factory.create_kernel(mul_op);
    ASSERT_TRUE(mul_kernel);

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {1, 3, 3}, impl::data_type::f32);

    // compile the add operator
    mul_kernel->compile(&mul_op, &eng, {src0_lt, src1_lt}, {dst_lt});

    impl::tensor_t src0_ts(src0_lt, &eng, src0.data());
    impl::tensor_t src1_ts(src1_lt, &eng, src1.data());
    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    mul_kernel->execute(&mul_op, &strm, {src0_ts, src1_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, mul_sigmoid) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src0 {2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0};
    test::vector<float> src1 {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    test::vector<float> ref_dst {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
    test::vector<float> dst(src0.size(), 0.0);

    impl::op_t mul_op(impl::dnnl_impl::op_kind::multiply_sigmoid);

    auto &op_factory = get_dnnl_kernel_registry();
    auto mul_kernel = op_factory.create_kernel(mul_op);
    ASSERT_TRUE(mul_kernel);

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {1, 3, 3}, impl::data_type::f32);

    // compile the add operator
    mul_kernel->compile(&mul_op, &eng, {src0_lt, src1_lt}, {dst_lt});

    impl::tensor_t src0_ts(src0_lt, &eng, src0.data());
    impl::tensor_t src1_ts(src1_lt, &eng, src1.data());
    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    mul_kernel->execute(&mul_op, &strm, {src0_ts, src1_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, mul_add) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src0 {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> src1 {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> post_src {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> ref_dst {6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0};
    test::vector<float> dst(src0.size(), 0.0);

    impl::op_t mul_op(impl::dnnl_impl::op_kind::multiply_add);

    auto &op_factory = get_dnnl_kernel_registry();
    auto mul_kernel = op_factory.create_kernel(mul_op);
    ASSERT_TRUE(mul_kernel);

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t post_src_lt
            = utils::logical_tensor_init(2, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(3, {1, 3, 3}, impl::data_type::f32);

    // compile the add operator
    mul_kernel->compile(
            &mul_op, &eng, {src0_lt, src1_lt, post_src_lt}, {dst_lt});

    impl::tensor_t src0_ts(src0_lt, &eng, src0.data());
    impl::tensor_t src1_ts(src1_lt, &eng, src1.data());
    impl::tensor_t post_src_ts(post_src_lt, &eng, post_src.data());
    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    mul_kernel->execute(
            &mul_op, &strm, {src0_ts, src1_ts, post_src_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }

    mul_kernel->execute(
            &mul_op, &strm, {src0_ts, src1_ts, post_src_ts}, {post_src_ts});
    strm.wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(post_src[i], ref_dst[i]);
    }
}

TEST(operator_kernel, min) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src0 {2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0};
    test::vector<float> src1 {1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0};
    test::vector<float> ref_dst {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    test::vector<float> dst(src0.size(), 0.0);

    impl::op_t max_op(impl::op_kind::Minimum);

    auto &op_factory = get_dnnl_kernel_registry();
    auto max_kernel = op_factory.create_kernel(max_op);
    ASSERT_TRUE(max_kernel);

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {1, 3, 3}, impl::data_type::f32);

    // compile the add operator
    max_kernel->compile(&max_op, &eng, {src0_lt, src1_lt}, {dst_lt});

    impl::tensor_t src0_ts(src0_lt, &eng, src0.data());
    impl::tensor_t src1_ts(src1_lt, &eng, src1.data());
    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    max_kernel->execute(&max_op, &strm, {src0_ts, src1_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, min_relu) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src0 {2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0};
    test::vector<float> src1 {-1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0};
    test::vector<float> ref_dst {0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    test::vector<float> dst(src0.size(), 0.0);

    impl::op_t max_op(impl::dnnl_impl::op_kind::minimum_relu);

    auto &op_factory = get_dnnl_kernel_registry();
    auto max_kernel = op_factory.create_kernel(max_op);
    ASSERT_TRUE(max_kernel);

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {1, 3, 3}, impl::data_type::f32);

    // compile the add operator
    max_kernel->compile(&max_op, &eng, {src0_lt, src1_lt}, {dst_lt});

    impl::tensor_t src0_ts(src0_lt, &eng, src0.data());
    impl::tensor_t src1_ts(src1_lt, &eng, src1.data());
    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    max_kernel->execute(&max_op, &strm, {src0_ts, src1_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, min_sigmoid) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src0 {2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0};
    test::vector<float> src1 {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    test::vector<float> ref_dst {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
    test::vector<float> dst(src0.size(), 0.0);

    impl::op_t max_op(impl::dnnl_impl::op_kind::minimum_sigmoid);

    auto &op_factory = get_dnnl_kernel_registry();
    auto max_kernel = op_factory.create_kernel(max_op);
    ASSERT_TRUE(max_kernel);

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {1, 3, 3}, impl::data_type::f32);

    // compile the add operator
    max_kernel->compile(&max_op, &eng, {src0_lt, src1_lt}, {dst_lt});

    impl::tensor_t src0_ts(src0_lt, &eng, src0.data());
    impl::tensor_t src1_ts(src1_lt, &eng, src1.data());
    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    max_kernel->execute(&max_op, &strm, {src0_ts, src1_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, min_add) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src0 {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    test::vector<float> src1 {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> post_src {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> ref_dst {3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0};
    test::vector<float> dst(src0.size(), 0.0);

    impl::op_t min_op(impl::dnnl_impl::op_kind::minimum_add);

    auto &op_factory = get_dnnl_kernel_registry();
    auto min_kernel = op_factory.create_kernel(min_op);
    ASSERT_TRUE(min_kernel);

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t post_src_lt
            = utils::logical_tensor_init(2, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(3, {1, 3, 3}, impl::data_type::f32);

    // compile the add operator
    min_kernel->compile(
            &min_op, &eng, {src0_lt, src1_lt, post_src_lt}, {dst_lt});

    impl::tensor_t src0_ts(src0_lt, &eng, src0.data());
    impl::tensor_t src1_ts(src1_lt, &eng, src1.data());
    impl::tensor_t post_src_ts(post_src_lt, &eng, post_src.data());
    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    min_kernel->execute(
            &min_op, &strm, {src0_ts, src1_ts, post_src_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }

    min_kernel->execute(
            &min_op, &strm, {src0_ts, src1_ts, post_src_ts}, {post_src_ts});
    strm.wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(post_src[i], ref_dst[i]);
    }
}

TEST(operator_kernel, max) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src0 {2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0};
    test::vector<float> src1 {1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0};
    test::vector<float> ref_dst {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> dst(src0.size(), 0.0);

    impl::op_t min_op(impl::op_kind::Maximum);

    auto &op_factory = get_dnnl_kernel_registry();
    auto min_kernel = op_factory.create_kernel(min_op);
    ASSERT_TRUE(min_kernel);

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {1, 3, 3}, impl::data_type::f32);

    // compile the add operator
    min_kernel->compile(&min_op, &eng, {src0_lt, src1_lt}, {dst_lt});

    impl::tensor_t src0_ts(src0_lt, &eng, src0.data());
    impl::tensor_t src1_ts(src1_lt, &eng, src1.data());
    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    min_kernel->execute(&min_op, &strm, {src0_ts, src1_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, max_relu) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src0 {-2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0};
    test::vector<float> src1 {-1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0};
    test::vector<float> ref_dst {0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> dst(src0.size(), 0.0);

    impl::op_t min_op(impl::dnnl_impl::op_kind::maximum_relu);

    auto &op_factory = get_dnnl_kernel_registry();
    auto min_kernel = op_factory.create_kernel(min_op);
    ASSERT_TRUE(min_kernel);

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {1, 3, 3}, impl::data_type::f32);

    // compile the add operator
    min_kernel->compile(&min_op, &eng, {src0_lt, src1_lt}, {dst_lt});

    impl::tensor_t src0_ts(src0_lt, &eng, src0.data());
    impl::tensor_t src1_ts(src1_lt, &eng, src1.data());
    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    min_kernel->execute(&min_op, &strm, {src0_ts, src1_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, max_sigmoid) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src0 {
            -2.0, -1.0, -2.0, -1.0, -2.0, -1.0, -2.0, -1.0, -2.0};
    test::vector<float> src1 {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    test::vector<float> ref_dst {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
    test::vector<float> dst(src0.size(), 0.0);

    impl::op_t min_op(impl::dnnl_impl::op_kind::maximum_sigmoid);

    auto &op_factory = get_dnnl_kernel_registry();
    auto min_kernel = op_factory.create_kernel(min_op);
    ASSERT_TRUE(min_kernel);

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {1, 3, 3}, impl::data_type::f32);

    // compile the add operator
    min_kernel->compile(&min_op, &eng, {src0_lt, src1_lt}, {dst_lt});

    impl::tensor_t src0_ts(src0_lt, &eng, src0.data());
    impl::tensor_t src1_ts(src1_lt, &eng, src1.data());
    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    min_kernel->execute(&min_op, &strm, {src0_ts, src1_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, max_add) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src0 {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    test::vector<float> src1 {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> post_src {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> ref_dst {4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0};
    test::vector<float> dst(src0.size(), 0.0);

    impl::op_t max_op(impl::dnnl_impl::op_kind::maximum_add);

    auto &op_factory = get_dnnl_kernel_registry();
    auto max_kernel = op_factory.create_kernel(max_op);
    ASSERT_TRUE(max_kernel);

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t post_src_lt
            = utils::logical_tensor_init(2, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(3, {1, 3, 3}, impl::data_type::f32);

    // compile the add operator
    max_kernel->compile(
            &max_op, &eng, {src0_lt, src1_lt, post_src_lt}, {dst_lt});

    impl::tensor_t src0_ts(src0_lt, &eng, src0.data());
    impl::tensor_t src1_ts(src1_lt, &eng, src1.data());
    impl::tensor_t post_src_ts(post_src_lt, &eng, post_src.data());
    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    max_kernel->execute(
            &max_op, &strm, {src0_ts, src1_ts, post_src_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }

    max_kernel->execute(
            &max_op, &strm, {src0_ts, src1_ts, post_src_ts}, {post_src_ts});
    strm.wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(post_src[i], ref_dst[i]);
    }
}

TEST(operator_kernel, matmul_compile_fwd_fp32) {
    impl::op_t matmul_op(impl::op_kind::MatMul);
    matmul_op.set_attr<bool>("transpose_b", true);
    impl::engine_t &eng = get_engine();

    test::vector<float> src_data {-2.0, -1.5};
    test::vector<float> weight_data {-2.0, -1.5};
    test::vector<float> bias_data {3.75};
    test::vector<float> ref_dst_data {10};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {1, 2}, impl::data_type::f32);
    impl::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2}, impl::data_type::f32);
    impl::logical_tensor_t bias
            = utils::logical_tensor_init(2, {1}, impl::data_type::f32);
    impl::logical_tensor_t dst
            = utils::logical_tensor_init(3, {1}, impl::data_type::f32);

    matmul_op.add_input(src);
    matmul_op.add_input(weight);
    matmul_op.add_input(bias);
    matmul_op.add_output(dst);

    impl::graph_t g;
    g.add_op(&matmul_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("matmul_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src, &weight, &bias};
    std::vector<const impl::logical_tensor_t *> outputs {&dst};

    p.compile(&cp, inputs, outputs, &eng);

    impl::tensor_t src_ts(src, &eng, src_data.data());
    impl::tensor_t weight_ts(weight, &eng, weight_data.data());
    impl::tensor_t bias_ts(bias, &eng, bias_data.data());
    impl::tensor_t dst_ts(dst, &eng, dst_data.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src_ts, weight_ts, bias_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }

    // test with second iteration to see
    // if memory cache works correctly
    test::vector<float> src_data2 {-2.0, -1.5};
    test::vector<float> weight_data2 {-1.0, -1.0};
    test::vector<float> bias_data2 {1.5};
    test::vector<float> ref_dst_data2 {5};
    test::vector<float> dst_data2(ref_dst_data2.size(), 0.0);

    impl::tensor_t src_ts2(src, &eng, src_data2.data());
    impl::tensor_t weight_ts2(weight, &eng, weight_data2.data());
    impl::tensor_t bias_ts2(bias, &eng, bias_data2.data());
    impl::tensor_t dst_ts2(dst, &eng, dst_data2.data());

    cp.execute(&strm, {src_ts2, weight_ts2, bias_ts2}, {dst_ts2});
    strm.wait();
    for (size_t i = 0; i < ref_dst_data2.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data2[i], ref_dst_data2[i]);
    }
}

TEST(operator_kernel, matmul_compile_fwd_f16f16f16) {
    impl::op_t matmul_op(impl::op_kind::MatMul);

    impl::engine_t &eng = get_engine();
    SKIP_IF(eng.kind() != impl::engine_kind::gpu,
            "Skip fp16 test for non-GPU device.");

    // in real use case, the data type should be half
    test::vector<float> src_data {-2.0, -1.5, -1.0, -0.5, 0.0, 0.5};
    test::vector<float> weight_data {
            -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, -2.0, -1.5};
    test::vector<float> dst_data(12, 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {2}, impl::data_type::f16);
    impl::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2}, impl::data_type::f16);
    impl::logical_tensor_t dst
            = utils::logical_tensor_init(2, {1, 1}, impl::data_type::f16);

    matmul_op.add_input(src);
    matmul_op.add_input(weight);
    matmul_op.add_output(dst);

    impl::graph_t g;
    g.add_op(&matmul_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("matmul_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src, &weight};
    std::vector<const impl::logical_tensor_t *> outputs {&dst};

    p.compile(&cp, inputs, outputs, &eng);

    impl::tensor_t src_ts(src, &eng, src_data.data());
    impl::tensor_t weight_ts(weight, &eng, weight_data.data());
    impl::tensor_t dst_ts(dst, &eng, dst_data.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src_ts, weight_ts}, {dst_ts});
    strm.wait();
}

TEST(operator_kernel, matmul_compile_fwd_bf16bf16bf16) {
    impl::op_t matmul_op(impl::op_kind::MatMul);

    impl::engine_t &eng = get_engine();
    static auto isa = dnnl_get_effective_cpu_isa();
    SKIP_IF(isa < dnnl_cpu_isa_avx512_core
                    && eng.kind() == impl::engine_kind::cpu,
            "Skip bf16 examples for systems that do not support avx512_core.");

    // in real use case, the data type should be half
    test::vector<float> src_data {-2.0, -1.5, -1.0, -0.5, 0.0, 0.5};
    test::vector<float> weight_data {
            -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, -2.0, -1.5};
    test::vector<float> dst_data(12, 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {1, 2}, impl::data_type::bf16);
    impl::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2}, impl::data_type::bf16);
    impl::logical_tensor_t dst
            = utils::logical_tensor_init(2, {1}, impl::data_type::bf16);

    matmul_op.add_input(src);
    matmul_op.add_input(weight);
    matmul_op.add_output(dst);

    impl::graph_t g;
    g.add_op(&matmul_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("matmul_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src, &weight};
    std::vector<const impl::logical_tensor_t *> outputs {&dst};

    p.compile(&cp, inputs, outputs, &eng);

    impl::tensor_t src_ts(src, &eng, src_data.data());
    impl::tensor_t weight_ts(weight, &eng, weight_data.data());
    impl::tensor_t dst_ts(dst, &eng, dst_data.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src_ts, weight_ts}, {dst_ts});
    strm.wait();
}

static size_t product(std::vector<int64_t> &in) {
    if (in.empty()) return 0;
    int64_t prod = std::accumulate(in.begin(), in.end(),
            static_cast<int64_t>(1), std::multiplies<int64_t>());
    return static_cast<size_t>(prod);
}

TEST(operator_kernel, matmul_ndx1d) {
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    std::vector<std::vector<int64_t>> weight_shapes {{16}, {16, 1}};
    std::vector<std::vector<int64_t>> src_shapes {
            {4, 2, 16}, {6, 4, 2, 16}, {8, 6, 4, 2, 16}};
    std::vector<std::vector<int64_t>> dst_shapes {{4, 2}, {6, 4, 2},
            {8, 6, 4, 2}, {4, 2, 1}, {6, 4, 2, 1}, {8, 6, 4, 2, 1}};

    // Initialize
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0f, 1.0f);

    for (size_t w_idx = 0; w_idx < weight_shapes.size(); ++w_idx) {
        for (size_t idx = 0; idx < src_shapes.size(); idx++) {
            auto src_shape = src_shapes[idx];
            auto dst_shape = dst_shapes[idx + w_idx * src_shapes.size()];

            test::vector<float> src_data(product(src_shape));
            test::vector<float> weight_data(product(weight_shapes[w_idx]));
            test::vector<float> dst_data(product(dst_shape));
            test::vector<float> ref_dst_data(product(dst_shape), 0.0);

            std::generate(src_data.begin(), src_data.end(),
                    [&]() { return distribution(generator); });
            std::generate(weight_data.begin(), weight_data.end(),
                    [&]() { return distribution(generator); });

            // prepare logical tensor
            impl::logical_tensor_t src = utils::logical_tensor_init(
                    0, src_shape, impl::data_type::f32);
            impl::logical_tensor_t weight = utils::logical_tensor_init(
                    1, weight_shapes[w_idx], impl::data_type::f32);
            impl::logical_tensor_t dst = utils::logical_tensor_init(
                    2, dst_shape, impl::data_type::f32);

            impl::op_t matmul_op(impl::op_kind::MatMul);
            matmul_op.add_input(src);
            matmul_op.add_input(weight);
            matmul_op.add_output(dst);

            impl::graph_t g;
            g.add_op(&matmul_op);
            g.build_graph();

            impl::pass::pass_base_ptr apass = get_pass("matmul_pass");
            apass->run(g);
            ASSERT_EQ(g.get_num_partitions(), 1);
            auto part = g.get_partitions()[0];

            // compile
            impl::partition_t p;
            p.init(part);

            impl::compiled_partition_t cp(p);

            std::vector<const impl::logical_tensor_t *> inputs {&src, &weight};
            std::vector<const impl::logical_tensor_t *> outputs {&dst};

            p.compile(&cp, inputs, outputs, &engine);

            impl::tensor_t src_ts(src, &engine, src_data.data());
            impl::tensor_t weight_ts(weight, &engine, weight_data.data());
            impl::tensor_t dst_ts(dst, &engine, dst_data.data());

            cp.execute(&strm, {src_ts, weight_ts}, {dst_ts});
            strm.wait();

            // compute the ref results
            size_t M = product(src_shape)
                    / static_cast<size_t>(src_shape.back());
            size_t K = static_cast<size_t>(src_shape.back());
            size_t N = weight_shapes[w_idx].size() == 1
                    ? static_cast<size_t>(1)
                    : static_cast<size_t>(weight_shapes[w_idx].back());
            // size_t N = static_cast<size_t>(1);
            for (size_t m = 0; m < M; m++) {
                for (size_t n = 0; n < N; n++) {
                    for (size_t k = 0; k < K; k++)
                        ref_dst_data[m * N + n]
                                += src_data[m * K + k] * weight_data[k * N + n];
                }
            }

            // FIXME(qun) the max error will be bigger than 1e-6
            for (size_t i = 0; i < ref_dst_data.size(); ++i) {
                ASSERT_NEAR(dst_data[i], ref_dst_data[i], 3e-6f);
            }
        }
    }
}

TEST(operator_kernel, matmul_1dxnd) {
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    std::vector<std::vector<int64_t>> src_shapes {{16}, {1, 16}};
    std::vector<std::vector<int64_t>> weight_shapes {
            {4, 16, 2}, {6, 4, 16, 2}, {8, 6, 4, 16, 2}};
    std::vector<std::vector<int64_t>> dst_shapes {{4, 2}, {6, 4, 2},
            {8, 6, 4, 2}, {4, 1, 2}, {6, 4, 1, 2}, {8, 6, 4, 1, 2}};

    // Initialize
    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    for (size_t idx = 0; idx < src_shapes.size(); ++idx) {
        for (size_t w_idx = 0; w_idx < weight_shapes.size(); w_idx++) {
            auto src_shape = src_shapes[idx];
            auto dst_shape = dst_shapes[w_idx + idx * weight_shapes.size()];

            test::vector<float> src_data(product(src_shape));
            test::vector<float> weight_data(product(weight_shapes[w_idx]));
            test::vector<float> dst_data(product(dst_shape));
            test::vector<float> ref_dst_data(product(dst_shape), 0.0);

            std::generate(src_data.begin(), src_data.end(),
                    [&]() { return distribution(generator); });
            std::generate(weight_data.begin(), weight_data.end(),
                    [&]() { return distribution(generator); });

            // prepare logical tensor
            impl::logical_tensor_t src = utils::logical_tensor_init(
                    0, src_shape, impl::data_type::f32);
            impl::logical_tensor_t weight = utils::logical_tensor_init(
                    1, weight_shapes[w_idx], impl::data_type::f32);
            impl::logical_tensor_t dst = utils::logical_tensor_init(
                    2, dst_shape, impl::data_type::f32);

            impl::op_t matmul_op(impl::op_kind::MatMul);
            matmul_op.add_input(src);
            matmul_op.add_input(weight);
            matmul_op.add_output(dst);

            impl::graph_t g;
            g.add_op(&matmul_op);
            g.build_graph();

            impl::pass::pass_base_ptr apass = get_pass("matmul_pass");
            apass->run(g);
            ASSERT_EQ(g.get_num_partitions(), 1);
            auto part = g.get_partitions()[0];

            // compile
            impl::partition_t p;
            p.init(part);

            impl::compiled_partition_t cp(p);

            std::vector<const impl::logical_tensor_t *> inputs {&src, &weight};
            std::vector<const impl::logical_tensor_t *> outputs {&dst};

            p.compile(&cp, inputs, outputs, &engine);

            impl::tensor_t src_ts(src, &engine, src_data.data());
            impl::tensor_t weight_ts(weight, &engine, weight_data.data());
            impl::tensor_t dst_ts(dst, &engine, dst_data.data());

            cp.execute(&strm, {src_ts, weight_ts}, {dst_ts});
            strm.wait();

            // compute the ref results
            size_t M = product(dst_shape)
                    / static_cast<size_t>(dst_shape.back());
            size_t K = static_cast<size_t>(src_shape.back());
            size_t N = weight_shapes[w_idx].back();
            for (size_t m = 0; m < M; m++) {
                for (size_t n = 0; n < N; n++) {
                    for (size_t k = 0; k < K; k++)
                        ref_dst_data[m * N + n] += src_data[k]
                                * weight_data[m * K * N + k * N + n];
                }
            }

            // FIXME(qun) the max error will be bigger than 1e-6
            for (size_t i = 0; i < ref_dst_data.size(); ++i) {
                ASSERT_NEAR(dst_data[i], ref_dst_data[i], 3e-6f);
            }
        }
    }
}

TEST(operator_kernel, matmul_3dx3d) {
    impl::op_t matmul_op(impl::op_kind::MatMul);
    impl::engine_t &eng = get_engine();

    test::vector<float> src_data {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    test::vector<float> weight_data {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    test::vector<float> ref_dst_data {2.0, 2.0, 2.0, 2.0};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {4, 1, 2}, impl::data_type::f32);
    impl::logical_tensor_t weight
            = utils::logical_tensor_init(1, {4, 2, 1}, impl::data_type::f32);
    impl::logical_tensor_t dst
            = utils::logical_tensor_init(2, {4, 1, 1}, impl::data_type::f32);

    matmul_op.add_input(src);
    matmul_op.add_input(weight);
    matmul_op.add_output(dst);

    impl::graph_t g;
    g.add_op(&matmul_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("matmul_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src, &weight};
    std::vector<const impl::logical_tensor_t *> outputs {&dst};

    p.compile(&cp, inputs, outputs, &eng);

    impl::tensor_t src_ts(src, &eng, src_data.data());
    impl::tensor_t weight_ts(weight, &eng, weight_data.data());
    impl::tensor_t dst_ts(dst, &eng, dst_data.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src_ts, weight_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(operator_kernel, matmul_relu_fusion) {
    impl::op_t matmul_op(0, impl::op_kind::MatMul, "matmul_op");
    impl::op_t relu_op(1, impl::op_kind::ReLU, "relu_op");

    impl::engine_t &eng = get_engine();

    test::vector<float> src_data {-2.0, -1.5};
    test::vector<float> weight_data {2.0, 1.5};
    test::vector<float> ref_dst_data {0.0};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {1, 2}, impl::data_type::f32);
    impl::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2, 1}, impl::data_type::f32);
    impl::logical_tensor_t dst
            = utils::logical_tensor_init(2, {1, 1}, impl::data_type::f32);
    impl::logical_tensor_t relu_dst
            = utils::logical_tensor_init(3, {1, 1}, impl::data_type::f32);

    matmul_op.add_input(src);
    matmul_op.add_input(weight);
    matmul_op.add_output(dst);
    relu_op.add_input(dst);
    relu_op.add_output(relu_dst);

    impl::graph_t g;
    ASSERT_EQ(g.add_op(&matmul_op), impl::status::success);
    ASSERT_EQ(g.add_op(&relu_op), impl::status::success);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("matmul_relu_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src, &weight};
    std::vector<const impl::logical_tensor_t *> outputs {&relu_dst};

    p.compile(&cp, inputs, outputs, &eng);

    impl::tensor_t src_ts(src, &eng, src_data.data());
    impl::tensor_t weight_ts(weight, &eng, weight_data.data());
    impl::tensor_t dst_ts(relu_dst, &eng, dst_data.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src_ts, weight_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(operator_kernel, matmul_bias_fusion) {
    impl::op_t matmul_op(impl::op_kind::MatMul);

    impl::engine_t &eng = get_engine();

    test::vector<float> src_data {-2.0, -1.5, 3.0, 0.5};
    test::vector<float> weight_data {-2.0, -1.5, 1.0, 1.0};
    test::vector<float> bias_data {1.0};
    test::vector<float> ref_dst_data {7.25, 4.5};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {2, 1, 2}, impl::data_type::f32);
    impl::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2, 2, 1}, impl::data_type::f32);
    impl::logical_tensor_t bias
            = utils::logical_tensor_init(2, {1}, impl::data_type::f32);
    impl::logical_tensor_t dst
            = utils::logical_tensor_init(3, {2, 1, 1}, impl::data_type::f32);

    matmul_op.add_input(src);
    matmul_op.add_input(weight);
    matmul_op.add_input(bias);
    matmul_op.add_output(dst);

    impl::graph_t g;
    g.add_op(&matmul_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("matmul_bias_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src, &weight, &bias};
    std::vector<const impl::logical_tensor_t *> outputs {&dst};

    p.compile(&cp, inputs, outputs, &eng);

    impl::tensor_t src_ts(src, &eng, src_data.data());
    impl::tensor_t weight_ts(weight, &eng, weight_data.data());
    impl::tensor_t bias_ts(bias, &eng, bias_data.data());
    impl::tensor_t dst_ts(dst, &eng, dst_data.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src_ts, weight_ts, bias_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(operator_kernel, matmul_sum_fusion_broadcast_1d) {
    impl::op_t matmul_op(0, impl::op_kind::MatMul, "matmul_op");
    impl::op_t add_op(1, impl::op_kind::Add, "add_op");

    impl::engine_t &engine = get_engine();

    test::vector<float> src_data {-2.0, -1.5, 3.0, 0.5};
    test::vector<float> weight_data {-2.0, -1.5, 1.0, 1.0};
    test::vector<float> add_src1_data {1.0, 1.0};
    test::vector<float> ref_dst_data {5.0, 4.0, 4.0, 3.25, 4.0, 4.0, 1.5, 1.5};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {2, 2, 1}, impl::data_type::f32);
    impl::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2, 1, 2}, impl::data_type::f32);
    impl::logical_tensor_t add_src1
            = utils::logical_tensor_init(2, {1}, impl::data_type::f32);
    impl::logical_tensor_t dst
            = utils::logical_tensor_init(3, {2, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t add_dst
            = utils::logical_tensor_init(4, {2, 2, 2}, impl::data_type::f32);

    matmul_op.add_input(src);
    matmul_op.add_input(weight);
    matmul_op.add_output(dst);
    add_op.add_input(dst);
    add_op.add_input(add_src1);
    add_op.add_output(add_dst);

    impl::graph_t g;
    ASSERT_EQ(g.add_op(&matmul_op), impl::status::success);
    ASSERT_EQ(g.add_op(&add_op), impl::status::success);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("matmul_sum_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {
            &src, &weight, &add_src1};
    std::vector<const impl::logical_tensor_t *> outputs {&add_dst};

    p.compile(&cp, inputs, outputs, &engine);

    impl::tensor_t src_ts(src, &engine, src_data.data());
    impl::tensor_t weight_ts(weight, &engine, weight_data.data());
    impl::tensor_t add_src1_ts(add_src1, &engine, add_src1_data.data());
    impl::tensor_t dst_ts(add_dst, &engine, dst_data.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src_ts, weight_ts, add_src1_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(operator_kernel, matmul_sum_fusion) {
    impl::op_t matmul_op(0, impl::op_kind::MatMul, "matmul_op");
    impl::op_t add_op(1, impl::op_kind::Add, "add_op");

    test::vector<float> src_data {-2.0, -1.5, 3.0, 0.5};
    test::vector<float> weight_data {-2.0, -1.5, 1.0, 1.0};
    test::vector<float> add_src1_data {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    test::vector<float> ref_dst_data {5.0, 4.0, 4.0, 3.25, 4.0, 4.0, 1.5, 1.5};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {2, 2, 1}, impl::data_type::f32);
    impl::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2, 1, 2}, impl::data_type::f32);
    impl::logical_tensor_t add_src1
            = utils::logical_tensor_init(2, {2, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t dst
            = utils::logical_tensor_init(3, {2, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t add_dst
            = utils::logical_tensor_init(4, {2, 2, 2}, impl::data_type::f32);

    matmul_op.add_input(src);
    matmul_op.add_input(weight);
    matmul_op.add_output(dst);
    add_op.add_input(dst);
    add_op.add_input(add_src1);
    add_op.add_output(add_dst);

    impl::graph_t g;
    ASSERT_EQ(g.add_op(&matmul_op), impl::status::success);
    ASSERT_EQ(g.add_op(&add_op), impl::status::success);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("matmul_sum_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {
            &src, &weight, &add_src1};
    std::vector<const impl::logical_tensor_t *> outputs {&add_dst};

    impl::engine_t &engine = get_engine();

    p.compile(&cp, inputs, outputs, &engine);

    impl::tensor_t src_ts(src, &engine, src_data.data());
    impl::tensor_t weight_ts(weight, &engine, weight_data.data());
    impl::tensor_t add_src1_ts(add_src1, &engine, add_src1_data.data());
    impl::tensor_t dst_ts(add_dst, &engine, dst_data.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src_ts, weight_ts, add_src1_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(operator_kernel, matmul_sum_gelu_fusion) {
    impl::op_t matmul_op(0, impl::op_kind::MatMul, "matmul_op");
    impl::op_t add_op(1, impl::op_kind::Add, "add_op");
    impl::op_t gelu_op(2, impl::op_kind::GELU, "gelu_op");

    impl::engine_t &eng = get_engine();

    test::vector<float> src_data {1.0, 1.0, 1.0, 1.0};
    test::vector<float> weight_data {0.5, 0.5, 0.5, 0.5};
    test::vector<float> other_data {1.0, 1.0};
    test::vector<float> ref_dst_data {1.9544998f, 1.9544998f};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {2, 1, 2}, impl::data_type::f32);
    impl::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2, 2, 1}, impl::data_type::f32);
    impl::logical_tensor_t other
            = utils::logical_tensor_init(2, {1}, impl::data_type::f32);
    impl::logical_tensor_t dst
            = utils::logical_tensor_init(3, {2, 1, 1}, impl::data_type::f32);
    impl::logical_tensor_t add_dst
            = utils::logical_tensor_init(4, {2, 1, 1}, impl::data_type::f32);
    impl::logical_tensor_t gelu_dst
            = utils::logical_tensor_init(5, {2, 1, 1}, impl::data_type::f32);

    matmul_op.add_input(src);
    matmul_op.add_input(weight);
    matmul_op.add_output(dst);
    add_op.add_input(dst);
    add_op.add_input(other);
    add_op.add_output(add_dst);
    gelu_op.add_input(add_dst);
    gelu_op.add_output(gelu_dst);

    impl::graph_t g;
    ASSERT_EQ(g.add_op(&matmul_op), impl::status::success);
    ASSERT_EQ(g.add_op(&add_op), impl::status::success);
    ASSERT_EQ(g.add_op(&gelu_op), impl::status::success);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("matmul_sum_gelu_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src, &weight, &other};
    std::vector<const impl::logical_tensor_t *> outputs {&gelu_dst};

    p.compile(&cp, inputs, outputs, &eng);

    impl::tensor_t src_ts(src, &eng, src_data.data());
    impl::tensor_t weight_ts(weight, &eng, weight_data.data());
    impl::tensor_t other_ts(other, &eng, other_data.data());
    impl::tensor_t dst_ts(gelu_dst, &eng, dst_data.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src_ts, weight_ts, other_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(operator_kernel, matmul_sum_relu_fusion) {
    impl::op_t matmul_op(0, impl::op_kind::MatMul, "matmul_op");
    impl::op_t add_op(1, impl::op_kind::Add, "add_op");
    impl::op_t relu_op(2, impl::op_kind::ReLU, "relu_op");

    impl::engine_t &engine = get_engine();

    test::vector<float> src_data {1.0, 2.0, 3.0, 4.0};
    test::vector<float> weight_data {-1.0, -2.0, 3.0, 4.0};
    test::vector<float> other_data {2.5};
    test::vector<float> ref_dst_data {0, 27.5f};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {2, 1, 2}, impl::data_type::f32);
    impl::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2, 2, 1}, impl::data_type::f32);
    impl::logical_tensor_t other
            = utils::logical_tensor_init(2, {1}, impl::data_type::f32);
    impl::logical_tensor_t dst
            = utils::logical_tensor_init(3, {2, 1, 1}, impl::data_type::f32);
    impl::logical_tensor_t add_dst
            = utils::logical_tensor_init(4, {2, 1, 1}, impl::data_type::f32);
    impl::logical_tensor_t relu_dst
            = utils::logical_tensor_init(5, {2, 1, 1}, impl::data_type::f32);

    matmul_op.add_input(src);
    matmul_op.add_input(weight);
    matmul_op.add_output(dst);
    add_op.add_input(dst);
    add_op.add_input(other);
    add_op.add_output(add_dst);
    relu_op.add_input(add_dst);
    relu_op.add_output(relu_dst);

    impl::graph_t g;
    ASSERT_EQ(g.add_op(&matmul_op), impl::status::success);
    ASSERT_EQ(g.add_op(&add_op), impl::status::success);
    ASSERT_EQ(g.add_op(&relu_op), impl::status::success);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("matmul_sum_relu_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src, &weight, &other};
    std::vector<const impl::logical_tensor_t *> outputs {&relu_dst};

    p.compile(&cp, inputs, outputs, &engine);

    impl::tensor_t src_ts(src, &engine, src_data.data());
    impl::tensor_t weight_ts(weight, &engine, weight_data.data());
    impl::tensor_t other_ts(other, &engine, other_data.data());
    impl::tensor_t dst_ts(relu_dst, &engine, dst_data.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src_ts, weight_ts, other_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(operator_kernel, matmul_bias_relu_fusion) {
    impl::op_t matmul_op(0, impl::op_kind::MatMul, "matmul_op");
    impl::op_t relu_op(1, impl::op_kind::ReLU, "relu_op");
    impl::engine_t &engine = get_engine();

    test::vector<float> src_data {-2.0, -1.5};
    test::vector<float> weight_data {2.0, -1.5};
    test::vector<float> bias_data {1.0};
    test::vector<float> ref_dst_data {0.0};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {1, 2}, impl::data_type::f32);
    impl::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2, 1}, impl::data_type::f32);
    impl::logical_tensor_t bias
            = utils::logical_tensor_init(2, {1}, impl::data_type::f32);
    impl::logical_tensor_t dst
            = utils::logical_tensor_init(3, {1, 1}, impl::data_type::f32);
    impl::logical_tensor_t relu_dst
            = utils::logical_tensor_init(5, {1, 1}, impl::data_type::f32);

    matmul_op.add_input(src);
    matmul_op.add_input(weight);
    matmul_op.add_input(bias);
    matmul_op.add_output(dst);
    relu_op.add_input(dst);
    relu_op.add_output(relu_dst);

    impl::graph_t g;
    ASSERT_EQ(g.add_op(&matmul_op), impl::status::success);
    ASSERT_EQ(g.add_op(&relu_op), impl::status::success);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("matmul_bias_relu_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src, &weight, &bias};
    std::vector<const impl::logical_tensor_t *> outputs {&relu_dst};

    p.compile(&cp, inputs, outputs, &engine);

    impl::tensor_t src_ts(src, &engine, src_data.data());
    impl::tensor_t weight_ts(weight, &engine, weight_data.data());
    impl::tensor_t bias_ts(bias, &engine, bias_data.data());
    impl::tensor_t dst_ts(relu_dst, &engine, dst_data.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src_ts, weight_ts, bias_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(operator_kernel, matmul_bias_gelu_fusion) {
    impl::op_t matmul_op(0, impl::op_kind::MatMul, "matmul_op");
    impl::op_t gelu_op(1, impl::op_kind::GELU, "gelu_op");
    impl::engine_t &engine = get_engine();

    test::vector<float> src_data {1.0, 1.0, 1.0, 1.0};
    test::vector<float> weight_data {0.5, 0.5, 0.5, 0.5};
    test::vector<float> bias_data {1.0};
    test::vector<float> ref_dst_data {1.9544998f, 1.9544998f};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {2, 1, 2}, impl::data_type::f32);
    impl::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2, 2, 1}, impl::data_type::f32);
    impl::logical_tensor_t bias
            = utils::logical_tensor_init(2, {1}, impl::data_type::f32);
    impl::logical_tensor_t dst
            = utils::logical_tensor_init(3, {2, 1, 1}, impl::data_type::f32);
    impl::logical_tensor_t gelu_dst
            = utils::logical_tensor_init(5, {2, 1, 1}, impl::data_type::f32);

    matmul_op.add_input(src);
    matmul_op.add_input(weight);
    matmul_op.add_input(bias);
    matmul_op.add_output(dst);
    gelu_op.add_input(dst);
    gelu_op.add_output(gelu_dst);

    impl::graph_t g;
    ASSERT_EQ(g.add_op(&matmul_op), impl::status::success);
    ASSERT_EQ(g.add_op(&gelu_op), impl::status::success);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("matmul_bias_gelu_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src, &weight, &bias};
    std::vector<const impl::logical_tensor_t *> outputs {&gelu_dst};

    p.compile(&cp, inputs, outputs, &engine);

    impl::tensor_t src_ts(src, &engine, src_data.data());
    impl::tensor_t weight_ts(weight, &engine, weight_data.data());
    impl::tensor_t bias_ts(bias, &engine, bias_data.data());
    impl::tensor_t dst_ts(gelu_dst, &engine, dst_data.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src_ts, weight_ts, bias_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(operator_kernel, matmul_bias_relu6_fusion) {
    impl::op_t matmul_op(0, impl::op_kind::MatMul, "matmul_op");
    impl::op_t hardtanh_op(1, impl::op_kind::HardTanh, "hardtanh_op");
    hardtanh_op.set_attr<float>("min", 0.0);
    hardtanh_op.set_attr<float>("max", 6.0);

    impl::engine_t &engine = get_engine();

    test::vector<float> src_data {-2.0, -1.5};
    test::vector<float> weight_data {-2.0, -1.5};
    test::vector<float> bias_data {1.0};
    test::vector<float> ref_dst_data {6.0};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {1, 2}, impl::data_type::f32);
    impl::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2, 1}, impl::data_type::f32);
    impl::logical_tensor_t bias
            = utils::logical_tensor_init(2, {1, 1}, impl::data_type::f32);
    impl::logical_tensor_t dst
            = utils::logical_tensor_init(3, {1, 1}, impl::data_type::f32);
    impl::logical_tensor_t hardtanh_dst
            = utils::logical_tensor_init(5, {1, 1}, impl::data_type::f32);

    matmul_op.add_input(src);
    matmul_op.add_input(weight);
    matmul_op.add_input(bias);
    matmul_op.add_output(dst);
    hardtanh_op.add_input(dst);
    hardtanh_op.add_output(hardtanh_dst);

    impl::graph_t g;
    ASSERT_EQ(g.add_op(&matmul_op), impl::status::success);
    ASSERT_EQ(g.add_op(&hardtanh_op), impl::status::success);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("matmul_bias_relu6_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src, &weight, &bias};
    std::vector<const impl::logical_tensor_t *> outputs {&hardtanh_dst};

    p.compile(&cp, inputs, outputs, &engine);

    impl::tensor_t src_ts(src, &engine, src_data.data());
    impl::tensor_t weight_ts(weight, &engine, weight_data.data());
    impl::tensor_t bias_ts(bias, &engine, bias_data.data());
    impl::tensor_t dst_ts(hardtanh_dst, &engine, dst_data.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src_ts, weight_ts, bias_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(operator_kernel, matmul_bias_hardtanh_fusion) {
    impl::op_t matmul_op(0, impl::op_kind::MatMul, "matmul_op");
    impl::op_t hardtanh_op(1, impl::op_kind::HardTanh, "hardtanh_op");
    hardtanh_op.set_attr<float>("min", -3.0);
    hardtanh_op.set_attr<float>("max", 3.0);

    impl::engine_t &engine = get_engine();

    test::vector<float> src_data {-2.0, -1.5};
    test::vector<float> weight_data {-2.0, -1.5};
    test::vector<float> bias_data {1.0};
    test::vector<float> ref_dst_data {3.0};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {1, 2}, impl::data_type::f32);
    impl::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2, 1}, impl::data_type::f32);
    impl::logical_tensor_t bias
            = utils::logical_tensor_init(2, {1, 1}, impl::data_type::f32);
    impl::logical_tensor_t dst
            = utils::logical_tensor_init(3, {1, 1}, impl::data_type::f32);
    impl::logical_tensor_t hardtanh_dst
            = utils::logical_tensor_init(5, {1, 1}, impl::data_type::f32);

    matmul_op.add_input(src);
    matmul_op.add_input(weight);
    matmul_op.add_input(bias);
    matmul_op.add_output(dst);
    hardtanh_op.add_input(dst);
    hardtanh_op.add_output(hardtanh_dst);

    impl::graph_t g;
    ASSERT_EQ(g.add_op(&matmul_op), impl::status::success);
    ASSERT_EQ(g.add_op(&hardtanh_op), impl::status::success);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("matmul_bias_hardtanh_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src, &weight, &bias};
    std::vector<const impl::logical_tensor_t *> outputs {&hardtanh_dst};

    p.compile(&cp, inputs, outputs, &engine);

    impl::tensor_t src_ts(src, &engine, src_data.data());
    impl::tensor_t weight_ts(weight, &engine, weight_data.data());
    impl::tensor_t bias_ts(bias, &engine, bias_data.data());
    impl::tensor_t dst_ts(hardtanh_dst, &engine, dst_data.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src_ts, weight_ts, bias_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(operator_kernel, matmul_bias_elu_fusion) {
    impl::op_t matmul_op(0, impl::op_kind::MatMul, "matmul_op");
    impl::op_t elu_op(1, impl::op_kind::Elu, "elu_op");
    elu_op.set_attr<float>("alpha", 1.f);

    impl::engine_t &engine = get_engine();

    test::vector<float> src_data {-2.0, -1.5};
    test::vector<float> weight_data {2.0, -1.5};
    test::vector<float> bias_data {1.0};
    test::vector<float> ref_dst_data {-0.75};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);
    ref_dst_data[0] = static_cast<float>(exp(-0.75) - 1);
    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {1, 2}, impl::data_type::f32);
    impl::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2, 1}, impl::data_type::f32);
    impl::logical_tensor_t bias
            = utils::logical_tensor_init(2, {1, 1}, impl::data_type::f32);
    impl::logical_tensor_t dst
            = utils::logical_tensor_init(3, {1, 1}, impl::data_type::f32);
    impl::logical_tensor_t elu_dst
            = utils::logical_tensor_init(5, {1, 1}, impl::data_type::f32);

    matmul_op.add_input(src);
    matmul_op.add_input(weight);
    matmul_op.add_input(bias);
    matmul_op.add_output(dst);
    elu_op.add_input(dst);
    elu_op.add_output(elu_dst);

    impl::graph_t g;
    ASSERT_EQ(g.add_op(&matmul_op), impl::status::success);
    ASSERT_EQ(g.add_op(&elu_op), impl::status::success);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("matmul_bias_elu_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src, &weight, &bias};
    std::vector<const impl::logical_tensor_t *> outputs {&elu_dst};

    p.compile(&cp, inputs, outputs, &engine);

    impl::tensor_t src_ts(src, &engine, src_data.data());
    impl::tensor_t weight_ts(weight, &engine, weight_data.data());
    impl::tensor_t bias_ts(bias, &engine, bias_data.data());
    impl::tensor_t dst_ts(elu_dst, &engine, dst_data.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src_ts, weight_ts, bias_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(operator_kernel, matmul_bias_sigmoid_fusion) {
    impl::op_t matmul_op(0, impl::op_kind::MatMul, "matmul_op");
    impl::op_t sigmoid_op(1, impl::op_kind::Sigmoid, "sigmoid_op");
    impl::engine_t &engine = get_engine();

    test::vector<float> src_data {-2.0, -1.5};
    test::vector<float> weight_data {2.0, -1.5};
    test::vector<float> bias_data {1.0};
    test::vector<float> ref_dst_data {-0.75};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);
    ref_dst_data[0] = static_cast<float>(1 / (exp(-ref_dst_data[0]) + 1));
    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {1, 2}, impl::data_type::f32);
    impl::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2, 1}, impl::data_type::f32);
    impl::logical_tensor_t bias
            = utils::logical_tensor_init(2, {1, 1}, impl::data_type::f32);
    impl::logical_tensor_t dst = utils::logical_tensor_init(
            3, {1, 1}, impl::data_type::f32, impl::layout_type::any);
    impl::logical_tensor_t sigmoid_dst
            = utils::logical_tensor_init(5, {1, 1}, impl::data_type::f32);

    matmul_op.add_input(src);
    matmul_op.add_input(weight);
    matmul_op.add_input(bias);
    matmul_op.add_output(dst);
    sigmoid_op.add_input(dst);
    sigmoid_op.add_output(sigmoid_dst);

    impl::graph_t g;
    ASSERT_EQ(g.add_op(&matmul_op), impl::status::success);
    ASSERT_EQ(g.add_op(&sigmoid_op), impl::status::success);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("matmul_bias_sigmoid_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src, &weight, &bias};
    std::vector<const impl::logical_tensor_t *> outputs {&sigmoid_dst};

    p.compile(&cp, inputs, outputs, &engine);

    impl::tensor_t src_ts(src, &engine, src_data.data());
    impl::tensor_t weight_ts(weight, &engine, weight_data.data());
    impl::tensor_t bias_ts(bias, &engine, bias_data.data());
    impl::tensor_t dst_ts(sigmoid_dst, &engine, dst_data.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src_ts, weight_ts, bias_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(operator_kernel, matmul_bias_add_fusion) {
    impl::op_t matmul_op(0, impl::op_kind::MatMul, "matmul_op");
    impl::op_t add_op(1, impl::op_kind::Add, "add_op");
    impl::engine_t &engine = get_engine();

    test::vector<float> src_data {-2.0, -1.5};
    test::vector<float> weight_data {2.0, -1.5};
    test::vector<float> bias_data {1.0};
    test::vector<float> post_src_data {-2.0};
    test::vector<float> ref_dst_data {-2.75};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {1, 2}, impl::data_type::f32);
    impl::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2, 1}, impl::data_type::f32);
    impl::logical_tensor_t bias
            = utils::logical_tensor_init(2, {1, 1}, impl::data_type::f32);
    impl::logical_tensor_t post_src
            = utils::logical_tensor_init(3, {1, 1}, impl::data_type::f32);
    impl::logical_tensor_t dst = utils::logical_tensor_init(
            4, {1, 1}, impl::data_type::f32, impl::layout_type::any);
    impl::logical_tensor_t add_dst
            = utils::logical_tensor_init(5, {1, 1}, impl::data_type::f32);

    matmul_op.add_input(src);
    matmul_op.add_input(weight);
    matmul_op.add_input(bias);
    matmul_op.add_output(dst);
    add_op.add_input(dst);
    add_op.add_input(post_src);
    add_op.add_output(add_dst);

    impl::graph_t g;
    ASSERT_EQ(g.add_op(&matmul_op), impl::status::success);
    ASSERT_EQ(g.add_op(&add_op), impl::status::success);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("matmul_bias_sum_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {
            &src, &weight, &bias, &post_src};
    std::vector<const impl::logical_tensor_t *> outputs {&add_dst};

    p.compile(&cp, inputs, outputs, &engine);

    impl::tensor_t src_ts(src, &engine, src_data.data());
    impl::tensor_t weight_ts(weight, &engine, weight_data.data());
    impl::tensor_t bias_ts(bias, &engine, bias_data.data());
    impl::tensor_t post_src_ts(post_src, &engine, post_src_data.data());
    impl::tensor_t dst_ts(add_dst, &engine, dst_data.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src_ts, weight_ts, bias_ts, post_src_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(operator_kernel, matmul_bias_per_tensor_broadcast_add_fusion) {
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    test::vector<float> src_data {-2.0, -1.5};
    test::vector<float> weight_data {2.0, -1.5};
    test::vector<float> bias_data {1.0, 2.0};
    test::vector<float> post_src_data {-2.0};
    test::vector<float> ref_dst_data {-5.0, 3.0, -4.0, 2.25};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    std::vector<impl::dims> post_src_shapes = {{1}, {1, 1}};

    for (auto &post_src_shape : post_src_shapes) {
        impl::op_t matmul_op(0, impl::op_kind::MatMul, "matmul_op");
        impl::op_t add_op(1, impl::op_kind::Add, "add_op");

        // prepare logical tensor
        impl::logical_tensor_t src
                = utils::logical_tensor_init(0, {2, 1}, impl::data_type::f32);
        impl::logical_tensor_t weight
                = utils::logical_tensor_init(1, {1, 2}, impl::data_type::f32);
        impl::logical_tensor_t bias
                = utils::logical_tensor_init(2, {1, 2}, impl::data_type::f32);
        impl::logical_tensor_t post_src = utils::logical_tensor_init(
                3, post_src_shape, impl::data_type::f32);
        impl::logical_tensor_t dst
                = utils::logical_tensor_init(4, {2, 2}, impl::data_type::f32);
        impl::logical_tensor_t add_dst
                = utils::logical_tensor_init(5, {2, 2}, impl::data_type::f32);

        matmul_op.add_input(src);
        matmul_op.add_input(weight);
        matmul_op.add_input(bias);
        matmul_op.add_output(dst);
        add_op.add_input(dst);
        add_op.add_input(post_src);
        add_op.add_output(add_dst);

        impl::graph_t g;
        ASSERT_EQ(g.add_op(&matmul_op), impl::status::success);
        ASSERT_EQ(g.add_op(&add_op), impl::status::success);
        g.build_graph();

        impl::pass::pass_base_ptr apass = get_pass("matmul_bias_sum_fusion");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1);
        auto part = g.get_partitions()[0];

        // compile
        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> inputs {
                &src, &weight, &bias, &post_src};
        std::vector<const impl::logical_tensor_t *> outputs {&add_dst};

        p.compile(&cp, inputs, outputs, &engine);

        impl::tensor_t src_ts(src, &engine, src_data.data());
        impl::tensor_t weight_ts(weight, &engine, weight_data.data());
        impl::tensor_t bias_ts(bias, &engine, bias_data.data());
        impl::tensor_t post_src_ts(post_src, &engine, post_src_data.data());
        impl::tensor_t dst_ts(add_dst, &engine, dst_data.data());

        cp.execute(&strm, {src_ts, weight_ts, bias_ts, post_src_ts}, {dst_ts});
        strm.wait();
        for (size_t i = 0; i < ref_dst_data.size(); ++i) {
            ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
        }
    }
}

TEST(operator_kernel, matmul_bias_per_channel_broadcast_add_fusion) {
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    test::vector<float> src_data {-2.0, -1.5};
    test::vector<float> weight_data {2.0, -1.5};
    test::vector<float> bias_data {1.0, 2.0};
    test::vector<float> post_src_data {-2.0, -1.0};
    test::vector<float> ref_dst_data {-5.0, 4.0, -4.0, 3.25};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    std::vector<impl::dims> post_src_shapes = {{2}, {1, 2}};

    for (auto &post_src_shape : post_src_shapes) {
        impl::op_t matmul_op(0, impl::op_kind::MatMul, "matmul_op");
        impl::op_t add_op(1, impl::op_kind::Add, "add_op");
        // prepare logical tensor
        impl::logical_tensor_t src
                = utils::logical_tensor_init(0, {2, 1}, impl::data_type::f32);
        impl::logical_tensor_t weight
                = utils::logical_tensor_init(1, {1, 2}, impl::data_type::f32);
        impl::logical_tensor_t bias
                = utils::logical_tensor_init(2, {1, 2}, impl::data_type::f32);
        impl::logical_tensor_t post_src = utils::logical_tensor_init(
                3, post_src_shape, impl::data_type::f32);
        impl::logical_tensor_t dst
                = utils::logical_tensor_init(4, {2, 2}, impl::data_type::f32);
        impl::logical_tensor_t add_dst
                = utils::logical_tensor_init(5, {2, 2}, impl::data_type::f32);

        matmul_op.add_input(src);
        matmul_op.add_input(weight);
        matmul_op.add_input(bias);
        matmul_op.add_output(dst);
        add_op.add_input(dst);
        add_op.add_input(post_src);
        add_op.add_output(add_dst);

        impl::graph_t g;
        ASSERT_EQ(g.add_op(&matmul_op), impl::status::success);
        ASSERT_EQ(g.add_op(&add_op), impl::status::success);
        g.build_graph();

        impl::pass::pass_base_ptr apass = get_pass("matmul_bias_sum_fusion");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1);
        auto part = g.get_partitions()[0];

        // compile
        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> inputs {
                &src, &weight, &bias, &post_src};
        std::vector<const impl::logical_tensor_t *> outputs {&add_dst};

        p.compile(&cp, inputs, outputs, &engine);

        impl::tensor_t src_ts(src, &engine, src_data.data());
        impl::tensor_t weight_ts(weight, &engine, weight_data.data());
        impl::tensor_t bias_ts(bias, &engine, bias_data.data());
        impl::tensor_t post_src_ts(post_src, &engine, post_src_data.data());
        impl::tensor_t dst_ts(add_dst, &engine, dst_data.data());

        cp.execute(&strm, {src_ts, weight_ts, bias_ts, post_src_ts}, {dst_ts});
        strm.wait();
        for (size_t i = 0; i < ref_dst_data.size(); ++i) {
            ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
        }
    }
}

TEST(operator_kernel, matmul_bias_unsupported_broadcast_add_fusion) {
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    std::vector<impl::dims> post_src_shapes = {{3}, {1, 3}, {2, 1}};

    for (auto &post_src_shape : post_src_shapes) {
        impl::op_t matmul_op(0, impl::op_kind::MatMul, "matmul_op");
        impl::op_t add_op(1, impl::op_kind::Add, "add_op");
        // prepare logical tensor
        impl::logical_tensor_t src
                = utils::logical_tensor_init(0, {2, 1}, impl::data_type::f32);
        impl::logical_tensor_t weight
                = utils::logical_tensor_init(1, {1, 2}, impl::data_type::f32);
        impl::logical_tensor_t bias
                = utils::logical_tensor_init(2, {1, 2}, impl::data_type::f32);
        impl::logical_tensor_t post_src = utils::logical_tensor_init(
                3, post_src_shape, impl::data_type::f32);
        impl::logical_tensor_t dst
                = utils::logical_tensor_init(4, {2, 2}, impl::data_type::f32);
        impl::logical_tensor_t add_dst
                = utils::logical_tensor_init(5, {2, 2}, impl::data_type::f32);

        matmul_op.add_input(src);
        matmul_op.add_input(weight);
        matmul_op.add_input(bias);
        matmul_op.add_output(dst);
        add_op.add_input(dst);
        add_op.add_input(post_src);
        add_op.add_output(add_dst);

        impl::graph_t g;
        ASSERT_EQ(g.add_op(&matmul_op), impl::status::success);
        ASSERT_EQ(g.add_op(&add_op), impl::status::success);
        g.build_graph();

        impl::pass::pass_base_ptr apass = get_pass("matmul_bias_sum_fusion");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1);
        auto part = g.get_partitions()[0];

        // compile
        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> inputs {
                &src, &weight, &bias, &post_src};
        std::vector<const impl::logical_tensor_t *> outputs {&add_dst};

        ASSERT_EQ(p.compile(&cp, inputs, outputs, &engine),
                impl::status::compile_fail);
    }
}

TEST(operator_kernel, matmul_bias_add_relu_fusion) {
    impl::op_t matmul_op(0, impl::op_kind::MatMul, "matmul_op");
    impl::op_t add_op(1, impl::op_kind::Add, "add_op");
    impl::op_t relu_op(2, impl::op_kind::ReLU, "relu_op");
    impl::engine_t &engine = get_engine();

    test::vector<float> src_data {-2.0, -1.5};
    test::vector<float> weight_data {2.0, -1.5};
    test::vector<float> bias_data {1.0};
    test::vector<float> post_src_data {-2.0};
    test::vector<float> ref_dst_data {0.0};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {1, 2}, impl::data_type::f32);
    impl::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2, 1}, impl::data_type::f32);
    impl::logical_tensor_t bias
            = utils::logical_tensor_init(2, {1, 1}, impl::data_type::f32);
    impl::logical_tensor_t post_src
            = utils::logical_tensor_init(3, {1, 1}, impl::data_type::f32);
    impl::logical_tensor_t dst = utils::logical_tensor_init(
            4, {1, 1}, impl::data_type::f32, impl::layout_type::any);
    impl::logical_tensor_t add_dst
            = utils::logical_tensor_init(5, {1, 1}, impl::data_type::f32);
    impl::logical_tensor_t relu_dst
            = utils::logical_tensor_init(6, {1, 1}, impl::data_type::f32);

    matmul_op.add_input(src);
    matmul_op.add_input(weight);
    matmul_op.add_input(bias);
    matmul_op.add_output(dst);
    add_op.add_input(dst);
    add_op.add_input(post_src);
    add_op.add_output(add_dst);
    relu_op.add_input(add_dst);
    relu_op.add_output(relu_dst);

    impl::graph_t g;
    ASSERT_EQ(g.add_op(&matmul_op), impl::status::success);
    ASSERT_EQ(g.add_op(&add_op), impl::status::success);
    ASSERT_EQ(g.add_op(&relu_op), impl::status::success);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("matmul_bias_sum_relu_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {
            &src, &weight, &bias, &post_src};
    std::vector<const impl::logical_tensor_t *> outputs {&relu_dst};

    p.compile(&cp, inputs, outputs, &engine);

    impl::tensor_t src_ts(src, &engine, src_data.data());
    impl::tensor_t weight_ts(weight, &engine, weight_data.data());
    impl::tensor_t bias_ts(bias, &engine, bias_data.data());
    impl::tensor_t post_src_ts(post_src, &engine, post_src_data.data());
    impl::tensor_t dst_ts(relu_dst, &engine, dst_data.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src_ts, weight_ts, bias_ts, post_src_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(operator_kernel, max_pool) {
    using dims = impl::dnnl_impl::dims;
    impl::engine_t &eng = get_engine();

    test::vector<float> src {-2.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> ref_dst {-0.5, 2.0, 3.0, 4.0};
    test::vector<float> dst(ref_dst.size(), 0.0);

    impl::op_t max_pool_op(impl::op_kind::MaxPool);
    max_pool_op.set_attr<dims>("strides", {2, 2});
    max_pool_op.set_attr<dims>("kernel", {2, 2});
    max_pool_op.set_attr<dims>("pads_begin", {0, 0});
    max_pool_op.set_attr<dims>("pads_end", {0, 0});
    max_pool_op.set_attr<std::string>("data_format", "NCX");
    max_pool_op.set_attr<dims>("dilations", {1, 1});

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {1, 1, 2, 2}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto max_pool_kernel = op_factory.create_kernel(max_pool_op);

    max_pool_kernel->compile(&max_pool_op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t dst_ts(outputs[0], &eng, dst.data());

    impl::stream_t &strm = get_stream();
    max_pool_kernel->execute(&max_pool_op, &strm, {src_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, max_pool_with_opaque_input) {
    // dequantize - maxpool
    using dims = impl::dnnl_impl::dims;
    impl::engine_t &eng = get_engine();

    // prepare ops
    impl::op_t dequantize(impl::op_kind::Dequantize);
    dequantize.set_attr<std::vector<float>>("scales", {0.1f});
    dequantize.set_attr<std::vector<int64_t>>("zps", {10});
    dequantize.set_attr<std::string>("qtype", "per_tensor");
    dequantize.set_attr<int64_t>("axis", 0);

    impl::op_t maxpool(impl::op_kind::MaxPool);
    maxpool.set_attr<dims>("strides", {2, 2});
    maxpool.set_attr<dims>("kernel", {2, 2});
    maxpool.set_attr<dims>("pads_begin", {0, 0});
    maxpool.set_attr<dims>("pads_end", {0, 0});
    maxpool.set_attr<std::string>("data_format", "NXC");
    maxpool.set_attr<dims>("dilations", {1, 1});

    // prepare input/output logical tensor
    impl::logical_tensor_t dq_src_lt = utils::logical_tensor_init(
            0, {1, 2, 2, 1}, impl::data_type::u8, impl::layout_type::strided);
    impl::logical_tensor_t dq_dst_lt = utils::logical_tensor_init(
            1, {1, 2, 2, 1}, impl::data_type::f32, impl::layout_type::any);
    impl::logical_tensor_t mp_dst_lt = utils::logical_tensor_init(
            2, {1, 1, 1, 1}, impl::data_type::f32, impl::layout_type::any);

    auto &op_factory = get_dnnl_kernel_registry();
    auto op_dq = op_factory.create_kernel(dequantize);
    auto op_mp = op_factory.create_kernel(maxpool);

    std::vector<impl::logical_tensor_t> dq_in = {dq_src_lt};
    std::vector<impl::logical_tensor_t> dq_out = {dq_dst_lt};
    op_dq->compile(&dequantize, &eng, dq_in, dq_out);
    ASSERT_EQ(dq_out[0].layout_type, impl::layout_type::opaque);

    std::vector<impl::logical_tensor_t> mp_out = {mp_dst_lt};
    op_mp->compile(&maxpool, &eng, dq_out, mp_out);
    ASSERT_EQ(mp_out[0].layout_type, impl::layout_type::opaque);
}

TEST(operator_kernel, avg_pool_exclude_pad) {
    using dims = impl::dnnl_impl::dims;
    impl::engine_t &eng = get_engine();

    test::vector<float> src {-2.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            3.0, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> ref_dst {
            -2.0, 0.25, 0.5, 0.75, 0.5, 0.75, 3.0, -1.5, 4.0};
    test::vector<float> dst(ref_dst.size(), 0.0);

    impl::op_t avg_pool_op(impl::op_kind::AvgPool);
    avg_pool_op.set_attr<dims>("strides", {2, 2});
    avg_pool_op.set_attr<dims>("kernel", {2, 2});
    avg_pool_op.set_attr<dims>("pads_begin", {1, 1});
    avg_pool_op.set_attr<dims>("pads_end", {1, 1});
    avg_pool_op.set_attr<bool>("exclude_pad", true);
    avg_pool_op.set_attr<std::string>("data_format", "NCX");

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {1, 1, 3, 3}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto avg_pool_kernel = op_factory.create_kernel(avg_pool_op);

    avg_pool_kernel->compile(&avg_pool_op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t dst_ts(outputs[0], &eng, dst.data());

    impl::stream_t &strm = get_stream();
    avg_pool_kernel->execute(&avg_pool_op, &strm, {src_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, avg_pool_include_pad) {
    using dims = impl::dnnl_impl::dims;
    impl::engine_t &eng = get_engine();

    test::vector<float> src {-2.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            3.0, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> ref_dst {
            -0.5, 0.125, 0.125, 0.375, 0.5, 0.375, 0.75, -0.75, 1.0};
    test::vector<float> dst(ref_dst.size(), 0.0);

    impl::op_t avg_pool_op(impl::op_kind::AvgPool);
    avg_pool_op.set_attr<dims>("strides", {2, 2});
    avg_pool_op.set_attr<dims>("kernel", {2, 2});
    avg_pool_op.set_attr<dims>("pads_begin", {1, 1});
    avg_pool_op.set_attr<dims>("pads_end", {1, 1});
    avg_pool_op.set_attr<std::string>("data_format", "NCX");
    avg_pool_op.set_attr<bool>("exclude_pad", false);

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {1, 1, 3, 3}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto avg_pool_kernel = op_factory.create_kernel(avg_pool_op);

    avg_pool_kernel->compile(&avg_pool_op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t dst_ts(outputs[0], &eng, dst.data());

    impl::stream_t &strm = get_stream();
    avg_pool_kernel->execute(&avg_pool_op, &strm, {src_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_compile, Convolution_NCX_OIX) {
    using dims = impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> ref_dst {-1.0, 2.5, 5.0, 1.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};
    impl::op_t conv_op(impl::op_kind::Convolution);
    conv_op.set_attr<dims>("strides", dims {1, 1});
    conv_op.set_attr<dims>("dilations", dims {1, 1});
    conv_op.set_attr<dims>("pads_begin", dims {0, 0});
    conv_op.set_attr<dims>("pads_end", dims {0, 0});
    conv_op.set_attr<int64_t>("groups", 1);
    conv_op.set_attr<std::string>("data_format", "NCX");
    conv_op.set_attr<std::string>("filter_format", "OIX");

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt
            = utils::logical_tensor_init(1, {1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {1, 1, 2, 2}, impl::data_type::f32);

    conv_op.add_input(src_lt);
    conv_op.add_input(weight_lt);
    conv_op.add_output(dst_lt);

    impl::graph_t g;
    g.add_op(&conv_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("conv_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src_lt, &weight_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};

    p.compile(&cp, inputs, outputs, &eng);

    impl::logical_tensor_t lt;
    cp.query_logical_tensor(dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t weight_ts(weight_lt, &eng, weight.data());
    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src_ts, weight_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_compile, Convtranspose_with_groups) {
    using dims = impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {1.0, 2.0, 3.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
            0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0};
    test::vector<float> ref_dst {1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0};
    test::vector<float> dst(ref_dst.size(), 0);
    impl::op_t convtranspose_op(impl::op_kind::ConvTranspose);
    convtranspose_op.set_attr<dims>("strides", dims {1, 1});
    convtranspose_op.set_attr<dims>("dilations", dims {1, 1});
    convtranspose_op.set_attr<dims>("pads_begin", dims {0, 0});
    convtranspose_op.set_attr<dims>("pads_end", dims {0, 0});
    convtranspose_op.set_attr<int64_t>("groups", 2);
    convtranspose_op.set_attr<std::string>("data_format", "NCX");
    convtranspose_op.set_attr<std::string>("filter_format", "OIX");

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 4, 1, 1}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt
            = utils::logical_tensor_init(1, {4, 4, 1, 1}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {1, 8, 1, 1}, impl::data_type::f32);

    convtranspose_op.add_input(src_lt);
    convtranspose_op.add_input(weight_lt);
    convtranspose_op.add_output(dst_lt);

    impl::graph_t g;
    g.add_op(&convtranspose_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("convtranspose_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src_lt, &weight_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};

    p.compile(&cp, inputs, outputs, &eng);
    ASSERT_EQ(dst_lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t weight_ts(weight_lt, &eng, weight.data());
    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src_ts, weight_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

struct dnnl_graph_test_convtranspose_params {
    std::string data_format;
    std::string filter_format;
    std::vector<int64_t> src_shape;
    std::vector<int64_t> weight_shape;
    bool with_bias;
    std::vector<int64_t> bias_shape;
    std::vector<int64_t> dst_shape;
    impl::dnnl_impl::dims strides;
    impl::dnnl_impl::dims dilations;
    impl::dnnl_impl::dims pads_begin;
    impl::dnnl_impl::dims pads_end;
    test::vector<float> src;
    test::vector<float> weight;
    test::vector<float> bias;
    test::vector<float> ref_dst;
};

class test_convtranspose_compile
    : public ::testing::TestWithParam<dnnl_graph_test_convtranspose_params> {
public:
    void TestConvtranspose() {
        using dims = impl::dnnl_impl::dims;

        auto params = ::testing::TestWithParam<
                dnnl_graph_test_convtranspose_params>::GetParam();

        // default engine kind is cpu.
        impl::engine_t &eng = get_engine();
        test::vector<float> dst(params.ref_dst.size(), 0.0);
        impl::op_t convtranspose_op(impl::op_kind::ConvTranspose);
        convtranspose_op.set_attr<dims>("strides", params.strides);
        convtranspose_op.set_attr<dims>("dilations", params.dilations);
        convtranspose_op.set_attr<dims>("pads_begin", params.pads_begin);
        convtranspose_op.set_attr<dims>("pads_end", params.pads_end);
        convtranspose_op.set_attr<int64_t>("groups", 1);
        convtranspose_op.set_attr<std::string>(
                "data_format", params.data_format);
        convtranspose_op.set_attr<std::string>(
                "filter_format", params.filter_format);

        // prepare logical tensor
        impl::logical_tensor_t src_lt = utils::logical_tensor_init(
                0, params.src_shape, impl::data_type::f32);
        impl::logical_tensor_t weight_lt = utils::logical_tensor_init(
                1, params.weight_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
                2, params.dst_shape, impl::data_type::f32);
        impl::logical_tensor_t bias_lt
                = utils::logical_tensor_init(3, impl::data_type::f32);

        convtranspose_op.add_input(src_lt);
        convtranspose_op.add_input(weight_lt);
        if (params.with_bias) {
            bias_lt = utils::logical_tensor_init(
                    3, params.bias_shape, impl::data_type::f32);
            convtranspose_op.add_input(bias_lt);
        }
        convtranspose_op.add_output(dst_lt);

        impl::graph_t g;
        g.add_op(&convtranspose_op);
        g.build_graph();

        impl::pass::pass_base_ptr apass = params.with_bias
                ? get_pass("convtranspose_bias_fusion")
                : get_pass("convtranspose_pass");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1);
        auto part = g.get_partitions()[0];

        // compile
        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> inputs {
                &src_lt, &weight_lt};
        if (params.with_bias) inputs.push_back(&bias_lt);
        std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};

        p.compile(&cp, inputs, outputs, &eng);
        ASSERT_EQ(dst_lt.layout_type, impl::layout_type::strided);

        impl::tensor_t src_ts(src_lt, &eng, params.src.data());
        impl::tensor_t weight_ts(weight_lt, &eng, params.weight.data());
        impl::tensor_t dst_ts(dst_lt, &eng, dst.data());
        impl::tensor_t bias_ts;
        if (params.with_bias)
            bias_ts = impl::tensor_t(bias_lt, &eng, params.bias.data());

        impl::stream_t &strm = get_stream();
        if (params.with_bias)
            cp.execute(&strm, {src_ts, weight_ts, bias_ts}, {dst_ts});
        else
            cp.execute(&strm, {src_ts, weight_ts}, {dst_ts});
        strm.wait();
        for (size_t i = 0; i < dst.size(); ++i) {
            ASSERT_FLOAT_EQ(dst[i], params.ref_dst[i]);
        }
    }
};

TEST_P(test_convtranspose_compile, TestConvtranspose_compile) {
    TestConvtranspose();
}

INSTANTIATE_TEST_SUITE_P(TestConvtranspose_compile, test_convtranspose_compile,
        ::testing::Values(
                dnnl_graph_test_convtranspose_params {"NXC", "OIX",
                        {1, 2, 2, 1}, {1, 1, 3, 3}, false, {1}, {1, 4, 4, 1},
                        {1, 1}, {1, 1}, {0, 0}, {0, 0}, {-1.0, 2.5, 5.0, 1.5},
                        {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0}, {0.0},
                        {-1.0, 2.5, -1.0, 2.5, 5.0, 0.5, 7.5, 1.5, -1.0, 7.5,
                                0.5, 2.5, 5.0, 1.5, 5.0, 1.5}},
                dnnl_graph_test_convtranspose_params {"NCX", "OIX",
                        {1, 1, 2, 2}, {1, 1, 3, 3}, true, {1}, {1, 1, 4, 4},
                        {1, 1}, {1, 1}, {0, 0}, {0, 0}, {-1.0, 2.5, 5.0, 1.5},
                        {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0}, {1.0},
                        {0.0, 3.5, 0.0, 3.5, 6.0, 1.5, 8.5, 2.5, 0.0, 8.5, 1.5,
                                3.5, 6.0, 2.5, 6.0, 2.5}},
                dnnl_graph_test_convtranspose_params {"NXC", "XIO",
                        {1, 2, 2, 1}, {3, 3, 1, 1}, false, {1}, {1, 4, 4, 1},
                        {1, 1}, {1, 1}, {0, 0}, {0, 0}, {-1.0, 2.5, 5.0, 1.5},
                        {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0}, {1.0},
                        {-1.0, 2.5, -1.0, 2.5, 5.0, 0.5, 7.5, 1.5, -1.0, 7.5,
                                0.5, 2.5, 5.0, 1.5, 5.0, 1.5}},
                dnnl_graph_test_convtranspose_params {"NCX", "XIO",
                        {1, 1, 2, 2}, {3, 3, 1, 1}, true, {1}, {1, 1, 4, 4},
                        {1, 1}, {1, 1}, {0, 0}, {0, 0}, {-1.0, 2.5, 5.0, 1.5},
                        {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0}, {1.0},
                        {0.0, 3.5, 0.0, 3.5, 6.0, 1.5, 8.5, 2.5, 0.0, 8.5, 1.5,
                                3.5, 6.0, 2.5, 6.0, 2.5}},
                dnnl_graph_test_convtranspose_params {"NXC", "OIX",
                        {1, 1, 2, 2, 1}, {1, 1, 1, 3, 3}, false, {1},
                        {1, 1, 4, 4, 1}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0},
                        {0, 0, 0}, {-1.0, 2.5, 5.0, 1.5},
                        {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0}, {1.0},
                        {-1.0, 2.5, -1.0, 2.5, 5.0, 0.5, 7.5, 1.5, -1.0, 7.5,
                                0.5, 2.5, 5.0, 1.5, 5.0, 1.5}},
                dnnl_graph_test_convtranspose_params {"NCX", "OIX",
                        {1, 1, 1, 2, 2}, {1, 1, 1, 3, 3}, false, {1},
                        {1, 1, 1, 4, 4}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0},
                        {0, 0, 0}, {-1.0, 2.5, 5.0, 1.5},
                        {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0}, {1.0},
                        {-1.0, 2.5, -1.0, 2.5, 5.0, 0.5, 7.5, 1.5, -1.0, 7.5,
                                0.5, 2.5, 5.0, 1.5, 5.0, 1.5}},
                dnnl_graph_test_convtranspose_params {"NXC", "XIO",
                        {1, 1, 2, 2, 1}, {1, 3, 3, 1, 1}, false, {1},
                        {1, 1, 4, 4, 1}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0},
                        {0, 0, 0}, {-1.0, 2.5, 5.0, 1.5},
                        {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0}, {1.0},
                        {-1.0, 2.5, -1.0, 2.5, 5.0, 0.5, 7.5, 1.5, -1.0, 7.5,
                                0.5, 2.5, 5.0, 1.5, 5.0, 1.5}},
                dnnl_graph_test_convtranspose_params {"NCX", "XIO",
                        {1, 1, 1, 2, 2}, {1, 3, 3, 1, 1}, true, {1},
                        {1, 1, 1, 4, 4}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0},
                        {0, 0, 0}, {-1.0, 2.5, 5.0, 1.5},
                        {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0}, {1.0},
                        {0.0, 3.5, 0.0, 3.5, 6.0, 1.5, 8.5, 2.5, 0.0, 8.5, 1.5,
                                3.5, 6.0, 2.5, 6.0, 2.5}}));

TEST(operator_compile, Convolution3D_NCX_OIX) {
    using dims = std::vector<int64_t>;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> ref_dst {-1.0, 2.5, 5.0, 1.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};
    impl::op_t conv_op(impl::op_kind::Convolution);
    conv_op.set_attr<dims>("strides", dims {1, 1, 1});
    conv_op.set_attr<dims>("dilations", dims {1, 1, 1});
    conv_op.set_attr<dims>("pads_begin", dims {0, 0, 0});
    conv_op.set_attr<dims>("pads_end", dims {0, 0, 0});
    conv_op.set_attr<int64_t>("groups", 1);
    conv_op.set_attr<std::string>("data_format", "NCX");
    conv_op.set_attr<std::string>("filter_format", "OIX");

    // prepare logical tensor
    impl::logical_tensor_t src_lt = utils::logical_tensor_init(
            0, {1, 1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt = utils::logical_tensor_init(
            1, {1, 1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            2, {1, 1, 1, 2, 2}, impl::data_type::f32);

    conv_op.add_input(src_lt);
    conv_op.add_input(weight_lt);
    conv_op.add_output(dst_lt);

    impl::graph_t g;
    g.add_op(&conv_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("conv_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src_lt, &weight_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};

    p.compile(&cp, inputs, outputs, &eng);

    impl::logical_tensor_t lt;
    cp.query_logical_tensor(dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t weight_ts(weight_lt, &eng, weight.data());
    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src_ts, weight_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_compile, Convolution_NCX_XIO) {
    using dims = impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> ref_dst {-1.0, 2.5, 5.0, 1.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};
    impl::op_t conv_op(impl::op_kind::Convolution);
    conv_op.set_attr<dims>("strides", dims {1, 1});
    conv_op.set_attr<dims>("dilations", dims {1, 1});
    conv_op.set_attr<dims>("pads_begin", dims {0, 0});
    conv_op.set_attr<dims>("pads_end", dims {0, 0});
    conv_op.set_attr<int64_t>("groups", 1);
    conv_op.set_attr<std::string>("data_format", "NCX");
    conv_op.set_attr<std::string>("filter_format", "XIO");

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt
            = utils::logical_tensor_init(1, {3, 3, 1, 1}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {1, 1, 2, 2}, impl::data_type::f32);

    conv_op.add_input(src_lt);
    conv_op.add_input(weight_lt);
    conv_op.add_output(dst_lt);

    impl::graph_t g;
    g.add_op(&conv_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("conv_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src_lt, &weight_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};

    p.compile(&cp, inputs, outputs, &eng);

    impl::logical_tensor_t lt;
    cp.query_logical_tensor(dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t weight_ts(weight_lt, &eng, weight.data());
    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src_ts, weight_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_compile, Convolution3D_NCX_XIO) {
    using dims = impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> ref_dst {-1.0, 2.5, 5.0, 1.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};
    impl::op_t conv_op(impl::op_kind::Convolution);
    conv_op.set_attr<dims>("strides", dims {1, 1, 1});
    conv_op.set_attr<dims>("dilations", dims {1, 1, 1});
    conv_op.set_attr<dims>("pads_begin", dims {0, 0, 0});
    conv_op.set_attr<dims>("pads_end", dims {0, 0, 0});
    conv_op.set_attr<int64_t>("groups", 1);
    conv_op.set_attr<std::string>("data_format", "NCX");
    conv_op.set_attr<std::string>("filter_format", "XIO");

    // prepare logical tensor
    impl::logical_tensor_t src_lt = utils::logical_tensor_init(
            0, {1, 1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt = utils::logical_tensor_init(
            1, {1, 3, 3, 1, 1}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            2, {1, 1, 1, 2, 2}, impl::data_type::f32);

    conv_op.add_input(src_lt);
    conv_op.add_input(weight_lt);
    conv_op.add_output(dst_lt);

    impl::graph_t g;
    g.add_op(&conv_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("conv_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src_lt, &weight_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};

    p.compile(&cp, inputs, outputs, &eng);

    impl::logical_tensor_t lt;
    cp.query_logical_tensor(dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t weight_ts(weight_lt, &eng, weight.data());
    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src_ts, weight_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_compile, Convolution_NXC_XIO) {
    using dims = impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {
            -3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0, 2.5, -1.0, 0};
    test::vector<float> weight {
            1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0};
    test::vector<float> ref_dst {0.5};
    test::vector<float> dst {0.0};
    impl::op_t conv_op(impl::op_kind::Convolution);
    conv_op.set_attr<dims>("strides", dims {1, 1});
    conv_op.set_attr<dims>("dilations", dims {1, 1});
    conv_op.set_attr<dims>("pads_begin", dims {0, 0});
    conv_op.set_attr<dims>("pads_end", dims {0, 0});
    conv_op.set_attr<int64_t>("groups", 1);
    conv_op.set_attr<std::string>("data_format", "NXC");
    conv_op.set_attr<std::string>("filter_format", "XIO");

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 3, 4, 1}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt
            = utils::logical_tensor_init(1, {3, 4, 1, 1}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {1, 1, 1, 1}, impl::data_type::f32);

    conv_op.add_input(src_lt);
    conv_op.add_input(weight_lt);
    conv_op.add_output(dst_lt);

    impl::graph_t g;
    g.add_op(&conv_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("conv_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src_lt, &weight_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};

    p.compile(&cp, inputs, outputs, &eng);

    impl::logical_tensor_t lt;
    cp.query_logical_tensor(dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t weight_ts(weight_lt, &eng, weight.data());
    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src_ts, weight_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_compile, Convolution3D_NXC_XIO) {
    using dims = impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {
            -3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0, 2.5, -1.0, 0};
    test::vector<float> weight {
            1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0};
    test::vector<float> ref_dst {0.5};
    test::vector<float> dst {0.0};
    impl::op_t conv_op(impl::op_kind::Convolution);
    conv_op.set_attr<dims>("strides", dims {1, 1, 1});
    conv_op.set_attr<dims>("dilations", dims {1, 1, 1});
    conv_op.set_attr<dims>("pads_begin", dims {0, 0, 0});
    conv_op.set_attr<dims>("pads_end", dims {0, 0, 0});
    conv_op.set_attr<int64_t>("groups", 1);
    conv_op.set_attr<std::string>("data_format", "NXC");
    conv_op.set_attr<std::string>("filter_format", "XIO");

    // prepare logical tensor
    impl::logical_tensor_t src_lt = utils::logical_tensor_init(
            0, {1, 1, 3, 4, 1}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt = utils::logical_tensor_init(
            1, {1, 3, 4, 1, 1}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            2, {1, 1, 1, 1, 1}, impl::data_type::f32);

    conv_op.add_input(src_lt);
    conv_op.add_input(weight_lt);
    conv_op.add_output(dst_lt);

    impl::graph_t g;
    g.add_op(&conv_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("conv_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src_lt, &weight_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};

    p.compile(&cp, inputs, outputs, &eng);

    impl::logical_tensor_t lt;
    cp.query_logical_tensor(dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t weight_ts(weight_lt, &eng, weight.data());
    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src_ts, weight_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_compile, Convolution_NXC_OIX) {
    using dims = impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> ref_dst {-1.0, 2.5, 5.0, 1.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};
    impl::op_t conv_op(impl::op_kind::Convolution);
    conv_op.set_attr<dims>("strides", dims {1, 1});
    conv_op.set_attr<dims>("dilations", dims {1, 1});
    conv_op.set_attr<dims>("pads_begin", dims {0, 0});
    conv_op.set_attr<dims>("pads_end", dims {0, 0});
    conv_op.set_attr<int64_t>("groups", 1);
    conv_op.set_attr<std::string>("data_format", "NXC");
    conv_op.set_attr<std::string>("filter_format", "OIX");

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 4, 4, 1}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt
            = utils::logical_tensor_init(1, {1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {1, 2, 2, 1}, impl::data_type::f32);

    conv_op.add_input(src_lt);
    conv_op.add_input(weight_lt);
    conv_op.add_output(dst_lt);

    impl::graph_t g;
    g.add_op(&conv_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("conv_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src_lt, &weight_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};

    p.compile(&cp, inputs, outputs, &eng);

    impl::logical_tensor_t lt;
    cp.query_logical_tensor(dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t weight_ts(weight_lt, &eng, weight.data());
    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src_ts, weight_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_compile, Convolution3D_NXC_OIX) {
    using dims = impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> ref_dst {-1.0, 2.5, 5.0, 1.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};
    impl::op_t conv_op(impl::op_kind::Convolution);
    conv_op.set_attr<dims>("strides", dims {1, 1, 1});
    conv_op.set_attr<dims>("dilations", dims {1, 1, 1});
    conv_op.set_attr<dims>("pads_begin", dims {0, 0, 0});
    conv_op.set_attr<dims>("pads_end", dims {0, 0, 0});
    conv_op.set_attr<int64_t>("groups", 1);
    conv_op.set_attr<std::string>("data_format", "NXC");
    conv_op.set_attr<std::string>("filter_format", "OIX");

    // prepare logical tensor
    impl::logical_tensor_t src_lt = utils::logical_tensor_init(
            0, {1, 1, 4, 4, 1}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt = utils::logical_tensor_init(
            1, {1, 1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            2, {1, 1, 2, 2, 1}, impl::data_type::f32);

    conv_op.add_input(src_lt);
    conv_op.add_input(weight_lt);
    conv_op.add_output(dst_lt);

    impl::graph_t g;
    g.add_op(&conv_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("conv_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src_lt, &weight_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};

    p.compile(&cp, inputs, outputs, &eng);

    impl::logical_tensor_t lt;
    cp.query_logical_tensor(dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t weight_ts(weight_lt, &eng, weight.data());
    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src_ts, weight_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_compile, ConvolutionF16F16F16) {
    using dims = impl::dnnl_impl::dims;

    impl::engine_t &eng = get_engine();
    SKIP_IF(eng.kind() != impl::engine_kind::gpu,
            "Skip fp16 test for non-GPU device.");
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> ref_dst {-1.0, 2.5, 5.0, 1.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};
    impl::op_t conv_op(impl::op_kind::Convolution);
    conv_op.set_attr<dims>("strides", dims {1, 1});
    conv_op.set_attr<dims>("dilations", dims {1, 1});
    conv_op.set_attr<dims>("pads_begin", dims {0, 0});
    conv_op.set_attr<dims>("pads_end", dims {0, 0});
    conv_op.set_attr<int64_t>("groups", 1);
    conv_op.set_attr<std::string>("data_format", "NCX");
    conv_op.set_attr<std::string>("filter_format", "OIX");

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f16);
    impl::logical_tensor_t weight_lt
            = utils::logical_tensor_init(1, {1, 1, 3, 3}, impl::data_type::f16);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {1, 1, 2, 2}, impl::data_type::f16);

    conv_op.add_input(src_lt);
    conv_op.add_input(weight_lt);
    conv_op.add_output(dst_lt);

    impl::graph_t g;
    g.add_op(&conv_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("conv_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src_lt, &weight_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};

    p.compile(&cp, inputs, outputs, &eng);

    impl::logical_tensor_t lt;
    cp.query_logical_tensor(dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t weight_ts(weight_lt, &eng, weight.data());
    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src_ts, weight_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_compile, ConvolutionBF16BF16BF16) {
    using dims = impl::dnnl_impl::dims;

    impl::engine_t &eng = get_engine();

    static auto isa = dnnl_get_effective_cpu_isa();
    SKIP_IF(isa < dnnl_cpu_isa_avx512_core
                    && eng.kind() == impl::engine_kind::cpu,
            "Skip bf16 examples for systems that do not support avx512_core.");

    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> ref_dst {-1.0, 2.5, 5.0, 1.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};
    impl::op_t conv_op(impl::op_kind::Convolution);
    conv_op.set_attr<dims>("strides", dims {1, 1});
    conv_op.set_attr<dims>("dilations", dims {1, 1});
    conv_op.set_attr<dims>("pads_begin", dims {0, 0});
    conv_op.set_attr<dims>("pads_end", dims {0, 0});
    conv_op.set_attr<int64_t>("groups", 1);
    conv_op.set_attr<std::string>("data_format", "NCX");
    conv_op.set_attr<std::string>("filter_format", "OIX");

    // prepare logical tensor
    impl::logical_tensor_t src_lt = utils::logical_tensor_init(
            0, {1, 1, 4, 4}, impl::data_type::bf16);
    impl::logical_tensor_t weight_lt = utils::logical_tensor_init(
            1, {1, 1, 3, 3}, impl::data_type::bf16);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            2, {1, 1, 2, 2}, impl::data_type::bf16);

    conv_op.add_input(src_lt);
    conv_op.add_input(weight_lt);
    conv_op.add_output(dst_lt);

    impl::graph_t g;
    g.add_op(&conv_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("conv_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src_lt, &weight_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};

    p.compile(&cp, inputs, outputs, &eng);

    impl::logical_tensor_t lt;
    cp.query_logical_tensor(dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t weight_ts(weight_lt, &eng, weight.data());
    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src_ts, weight_ts}, {dst_ts});
    strm.wait();
}

TEST(operator_compile, ConvolutionBF16BF16F32) {
    using dims = impl::dnnl_impl::dims;

    impl::engine_t &eng = get_engine();
    static auto isa = dnnl_get_effective_cpu_isa();
    SKIP_IF(isa < dnnl_cpu_isa_avx512_core
                    && eng.kind() == impl::engine_kind::cpu,
            "Skip bf16 examples for systems that do not support avx512_core.");

    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> ref_dst {-1.0, 2.5, 5.0, 1.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};
    impl::op_t conv_op(impl::op_kind::Convolution);
    conv_op.set_attr<dims>("strides", dims {1, 1});
    conv_op.set_attr<dims>("dilations", dims {1, 1});
    conv_op.set_attr<dims>("pads_begin", dims {0, 0});
    conv_op.set_attr<dims>("pads_end", dims {0, 0});
    conv_op.set_attr<int64_t>("groups", 1);
    conv_op.set_attr<std::string>("data_format", "NCX");
    conv_op.set_attr<std::string>("filter_format", "OIX");

    // prepare logical tensor
    impl::logical_tensor_t src_lt = utils::logical_tensor_init(
            0, {1, 1, 4, 4}, impl::data_type::bf16);
    impl::logical_tensor_t weight_lt = utils::logical_tensor_init(
            1, {1, 1, 3, 3}, impl::data_type::bf16);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {1, 1, 2, 2}, impl::data_type::f32);

    conv_op.add_input(src_lt);
    conv_op.add_input(weight_lt);
    conv_op.add_output(dst_lt);

    impl::graph_t g;
    g.add_op(&conv_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("conv_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src_lt, &weight_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};

    p.compile(&cp, inputs, outputs, &eng);

    impl::logical_tensor_t lt;
    cp.query_logical_tensor(dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t weight_ts(weight_lt, &eng, weight.data());
    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src_ts, weight_ts}, {dst_ts});
    strm.wait();
}

TEST(operator_kernel, group_convolution) {
    using dims = impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();

    impl::op_t conv_op(impl::op_kind::Convolution);
    conv_op.set_attr<dims>("strides", dims {1, 1});
    conv_op.set_attr<dims>("dilations", dims {1, 1});
    conv_op.set_attr<dims>("pads_begin", dims {0, 0});
    conv_op.set_attr<dims>("pads_end", dims {0, 0});
    conv_op.set_attr<int64_t>("groups", 4);
    conv_op.set_attr<std::string>("data_format", "NCX");
    conv_op.set_attr<std::string>("filter_format", "OIX");

    // prepare logical tensor
    impl::logical_tensor_t src_lt = utils::logical_tensor_init(
            0, {8, 32, 16, 16}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt = utils::logical_tensor_init(
            1, {32, 8, 1, 1}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            2, {8, 32, 16, 16}, impl::data_type::f32);

    conv_op.add_input(src_lt);
    conv_op.add_input(weight_lt);
    conv_op.add_output(dst_lt);

    impl::graph_t g;
    g.add_op(&conv_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("conv_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src_lt, &weight_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};

    p.compile(&cp, inputs, outputs, &eng);

    impl::logical_tensor_t lt;
    cp.query_logical_tensor(dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    test::vector<float> src(8 * 32 * 16 * 16, 1);
    test::vector<float> weight(32 * 8 * 1 * 1, 1);
    test::vector<float> dst(8 * 32 * 16 * 16, 1);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t weight_ts(weight_lt, &eng, weight.data());
    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src_ts, weight_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], 8);
    }
}

TEST(operator_kernel, ConvolutionBackpropData) {
    using dims = dnnl::graph::impl::dnnl_impl::dims;

    impl::engine_t &eng = get_engine();

    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> ref_diff_src {0.0, 1.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0,
            0.0, 3.0, 3.0, 1.0, 2.0, 3.0, 2.0, 3.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> diff_dst {0.0, 1.0, 2.0, 3.0};
    test::vector<float> diff_src(src.size(), 0.0);

    impl::op_t conv_op(impl::op_kind::ConvolutionBackpropData);
    conv_op.set_attr<dims>("strides", dims {1, 1});
    conv_op.set_attr<dims>("dilations", dims {1, 1});
    conv_op.set_attr<dims>("pads_begin", dims {0, 0});
    conv_op.set_attr<dims>("pads_end", dims {0, 0});
    conv_op.set_attr<int64_t>("groups", 1);
    conv_op.set_attr<std::string>("data_format", "NCX");
    conv_op.set_attr<std::string>("filter_format", "OIX");

    // prepare logical tensor
    impl::logical_tensor_t diff_src_lt
            = utils::logical_tensor_init(1, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt
            = utils::logical_tensor_init(2, {1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t diff_dst_lt
            = utils::logical_tensor_init(3, {1, 1, 2, 2}, impl::data_type::f32);

    conv_op.add_input(diff_dst_lt);
    conv_op.add_input(weight_lt);
    conv_op.add_output(diff_src_lt);

    impl::graph_t g;
    g.add_op(&conv_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("conv_data_bw_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);
    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {
            &diff_dst_lt, &weight_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&diff_src_lt};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

    impl::logical_tensor_t lt;
    cp.query_logical_tensor(diff_src_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    impl::tensor_t weight_ts(weight_lt, &eng, weight.data());
    impl::tensor_t diff_dst_ts(diff_dst_lt, &eng, diff_dst.data());
    impl::tensor_t diff_src_ts(diff_src_lt, &eng, diff_src.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {diff_dst_ts, weight_ts}, {diff_src_ts});
    strm.wait();
    for (size_t i = 0; i < diff_src.size(); ++i) {
        ASSERT_FLOAT_EQ(diff_src[i], ref_diff_src[i]);
    }
}

TEST(operator_kernel, conv_bwd_weight_bias) {
    using dims = dnnl::graph::impl::dnnl_impl::dims;

    impl::engine_t &eng = get_engine();

    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> diff_dst {0.0, 1.0, 2.0, 3.0};
    test::vector<float> ref_diff_weights {
            -5.5, 3.0, 7.0, 10.5, 3.0, -0.5, 2.5, -8.0, 10.0};
    test::vector<float> ref_diff_bias {6.0};
    test::vector<float> diff_weights(ref_diff_weights.size(), 0.0);
    test::vector<float> diff_bias(ref_diff_bias.size(), 0.0);

    impl::op_t conv_op(impl::dnnl_impl::op_kind::conv_bwd_f_biasadd_bwd);
    conv_op.set_attr<dims>("strides", dims {1, 1});
    conv_op.set_attr<dims>("dilations", dims {1, 1});
    conv_op.set_attr<dims>("pads_begin", dims {0, 0});
    conv_op.set_attr<dims>("pads_end", dims {0, 0});
    conv_op.set_attr<int64_t>("groups", 0);
    conv_op.set_attr<std::string>("data_format", "NCX");
    conv_op.set_attr<std::string>("filter_format", "OIX");

    // prepare input/output logical_tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t diff_weights_lt
            = utils::logical_tensor_init(1, {1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t diff_bias_lt
            = utils::logical_tensor_init(2, dims {1}, impl::data_type::f32);
    impl::logical_tensor_t diff_dst_lt
            = utils::logical_tensor_init(3, {1, 1, 2, 2}, impl::data_type::f32);

    auto &op_factory = get_dnnl_kernel_registry();
    auto conv_bwd_kernel = op_factory.create_kernel(conv_op);

    conv_bwd_kernel->compile(&conv_op, &eng, {src_lt, diff_dst_lt},
            {diff_weights_lt, diff_bias_lt});
    ASSERT_EQ(diff_weights_lt.layout_type, impl::layout_type::strided);
    ASSERT_EQ(diff_bias_lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t diff_dst_ts(diff_dst_lt, &eng, diff_dst.data());
    impl::tensor_t diff_weights_ts(diff_weights_lt, &eng, diff_weights.data());
    impl::tensor_t diff_bias_ts(diff_bias_lt, &eng, diff_bias.data());

    impl::stream_t &strm = get_stream();
    conv_bwd_kernel->execute(&conv_op, &strm, {src_ts, diff_dst_ts},
            {diff_weights_ts, diff_bias_ts});
    strm.wait();
    for (size_t i = 0; i < diff_weights.size(); ++i) {
        ASSERT_FLOAT_EQ(diff_weights[i], ref_diff_weights[i]);
    }
    for (size_t i = 0; i < diff_bias.size(); ++i) {
        ASSERT_FLOAT_EQ(diff_bias[i], ref_diff_bias[i]);
    }
}

TEST(operator_compile, conv_relu_unfused) {
    using dims = impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    impl::stream_t &strm = get_stream();

    impl::op_t conv_op(0, impl::op_kind::Convolution, "conv");
    conv_op.set_attr<dims>("strides", dims {1, 1});
    conv_op.set_attr<dims>("dilations", dims {1, 1});
    conv_op.set_attr<dims>("pads_begin", dims {0, 0});
    conv_op.set_attr<dims>("pads_end", dims {0, 0});
    conv_op.set_attr<int64_t>("groups", 1);
    conv_op.set_attr<std::string>("data_format", "NCX");
    conv_op.set_attr<std::string>("filter_format", "OIX");

    impl::op_t relu_op(1, impl::op_kind::ReLU, "relu");

    // prepare logical tensor
    impl::logical_tensor_t conv_src_lt = utils::logical_tensor_init(
            0, {8, 32, 16, 16}, impl::data_type::f32);
    impl::logical_tensor_t conv_weight_lt = utils::logical_tensor_init(
            1, {32, 32, 1, 1}, impl::data_type::f32);
    impl::logical_tensor_t conv_dst_lt = utils::logical_tensor_init(
            2, {8, 32, 16, 16}, impl::data_type::f32, impl::layout_type::any);
    impl::logical_tensor_t relu_dst_lt = utils::logical_tensor_init(
            3, {8, 32, 16, 16}, impl::data_type::f32);

    conv_op.add_input(conv_src_lt);
    conv_op.add_input(conv_weight_lt);
    conv_op.add_output(conv_dst_lt);
    relu_op.add_input(conv_dst_lt);
    relu_op.add_output(relu_dst_lt);

    impl::graph_t g;
    g.add_op(&conv_op);
    g.add_op(&relu_op);
    g.build_graph();

    test::vector<float> conv_src(8 * 32 * 16 * 16, 1);
    test::vector<float> conv_weight(32 * 32 * 1 * 1, 1);
    test::vector<float> relu_dst(8 * 32 * 16 * 16, 1);

    impl::tensor_t conv_src_ts(conv_src_lt, &eng, conv_src.data());
    impl::tensor_t conv_weight_ts(conv_weight_lt, &eng, conv_weight.data());
    impl::tensor_t relu_dst_ts(relu_dst_lt, &eng, relu_dst.data());

    // run unfused graph to compute the reference
    ASSERT_EQ(run_graph(g, {conv_src_ts, conv_weight_ts}, {relu_dst_ts}, eng,
                      strm),
            impl::status::success);

    for (size_t i = 0; i < relu_dst.size(); ++i) {
        ASSERT_FLOAT_EQ(relu_dst[i], 32);
    }
}

TEST(operator_compile, convolution_bn_fp32) {
    using dims = impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    impl::stream_t &strm = get_stream();

    impl::op_t conv_op(0, impl::op_kind::Convolution, "conv");
    conv_op.set_attr<dims>("strides", dims {1, 1});
    conv_op.set_attr<dims>("dilations", dims {1, 1});
    conv_op.set_attr<dims>("pads_begin", dims {0, 0});
    conv_op.set_attr<dims>("pads_end", dims {0, 0});
    conv_op.set_attr<int64_t>("groups", 1);
    conv_op.set_attr<std::string>("data_format", "NCX");
    conv_op.set_attr<std::string>("filter_format", "OIX");

    impl::op_t bn_op(1, impl::op_kind::BatchNormInference, "bn");
    bn_op.set_attr<std::string>("data_format", "NCX");
    bn_op.set_attr("epsilon", 1e-6f);

    // prepare logical tensor
    impl::logical_tensor_t conv_src_lt = utils::logical_tensor_init(
            0, {8, 32, 16, 16}, impl::data_type::f32);
    impl::logical_tensor_t conv_weight_lt = utils::logical_tensor_init(
            1, {32, 32, 1, 1}, impl::data_type::f32);
    impl::logical_tensor_t conv_dst_lt = utils::logical_tensor_init(
            2, {8, 32, 16, 16}, impl::data_type::f32);
    impl::logical_tensor_t gamma_lt
            = utils::logical_tensor_init(3, {32}, impl::data_type::f32);
    impl::logical_tensor_t beta_lt
            = utils::logical_tensor_init(4, {32}, impl::data_type::f32);
    impl::logical_tensor_t scale_lt
            = utils::logical_tensor_init(5, {32}, impl::data_type::f32);
    impl::logical_tensor_t shift_lt
            = utils::logical_tensor_init(6, {32}, impl::data_type::f32);
    impl::logical_tensor_t bn_dst_lt = utils::logical_tensor_init(
            7, {8, 32, 16, 16}, impl::data_type::f32);

    test::vector<float> conv_src(8 * 32 * 16 * 16);
    test::vector<float> conv_weight(32 * 32 * 1 * 1);
    test::vector<float> bn_gamma(32);
    test::vector<float> bn_beta(32);
    test::vector<float> bn_scale(32);
    test::vector<float> bn_shift(32);
    test::vector<float> bn_dst(8 * 32 * 16 * 16);

    // Initialize
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0f, 0.1f);

    std::generate(conv_src.begin(), conv_src.end(),
            [&]() { return distribution(generator); });
    std::generate(conv_weight.begin(), conv_weight.end(),
            [&]() { return distribution(generator); });
    std::generate(bn_gamma.begin(), bn_gamma.end(),
            [&]() { return distribution(generator); });
    std::generate(bn_beta.begin(), bn_beta.end(),
            [&]() { return distribution(generator); });
    std::generate(bn_scale.begin(), bn_scale.end(),
            [&]() { return distribution(generator); });
    std::generate(bn_shift.begin(), bn_shift.end(),
            [&]() { return distribution(generator); });

    impl::tensor_t conv_src_ts(conv_src_lt, &eng, conv_src.data());
    impl::tensor_t conv_weight_ts(conv_weight_lt, &eng, conv_weight.data());
    impl::tensor_t bn_gamma_ts(gamma_lt, &eng, bn_gamma.data());
    impl::tensor_t bn_beta_ts(beta_lt, &eng, bn_beta.data());
    impl::tensor_t bn_scale_ts(scale_lt, &eng, bn_scale.data());
    impl::tensor_t bn_shift_ts(shift_lt, &eng, bn_shift.data());
    impl::tensor_t bn_dst_ts(bn_dst_lt, &eng, bn_dst.data());

    conv_op.add_input(conv_src_lt);
    conv_op.add_input(conv_weight_lt);
    conv_op.add_output(conv_dst_lt);
    bn_op.add_input(conv_dst_lt);
    bn_op.add_input(gamma_lt);
    bn_op.add_input(beta_lt);
    bn_op.add_input(scale_lt);
    bn_op.add_input(shift_lt);
    bn_op.add_output(bn_dst_lt);

    impl::graph_t g;
    g.add_op(&conv_op);
    g.add_op(&bn_op);
    g.build_graph();

    // run unfused graph to compute the reference
    ASSERT_EQ(run_graph(g,
                      {conv_src_ts, conv_weight_ts, bn_gamma_ts, bn_beta_ts,
                              bn_scale_ts, bn_shift_ts},
                      {bn_dst_ts}, eng, strm),
            impl::status::success);

    // run fusion partition
    impl::pass::pass_base_ptr apass = get_pass("conv_bn_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&conv_src_lt,
            &conv_weight_lt, &gamma_lt, &beta_lt, &scale_lt, &shift_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&bn_dst_lt};

    p.compile(&cp, inputs, outputs, &eng);

    test::vector<float> convbn_dst(8 * 32 * 16 * 16, 0.0);
    impl::tensor_t convbn_dst_ts(bn_dst_lt, &eng, convbn_dst.data());

    cp.execute(&strm,
            {conv_src_ts, conv_weight_ts, bn_gamma_ts, bn_beta_ts, bn_scale_ts,
                    bn_shift_ts},
            {convbn_dst_ts});
    strm.wait();

    float max_diff = 0;
    for (size_t i = 0; i < bn_dst.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(bn_dst[i] - convbn_dst[i]));
    }
    ASSERT_LT(max_diff, 1e-6f);
}

TEST(operator_kernel, conv_add) {
    using dims = impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> post_src {1.0, 2.0, 3.0, 4.0};
    test::vector<float> ref_dst {0.0, 4.5, 8.0, 5.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};

    std::vector<bool> swaps {false, true};

    for (auto swap : swaps) {
        impl::op_t conv_op(1, impl::op_kind::Convolution, "Convolution");
        conv_op.set_attr<dims>("strides", dims {1, 1});
        conv_op.set_attr<dims>("dilations", dims {1, 1});
        conv_op.set_attr<dims>("pads_begin", dims {0, 0});
        conv_op.set_attr<dims>("pads_end", dims {0, 0});
        conv_op.set_attr<int64_t>("groups", 1);
        conv_op.set_attr<std::string>("data_format", "NCX");
        conv_op.set_attr<std::string>("filter_format", "OIX");
        impl::op_t add_op(2, impl::op_kind::Add, "Add");

        // prepare logical tensor
        impl::logical_tensor_t src_lt = utils::logical_tensor_init(
                0, {1, 1, 4, 4}, impl::data_type::f32);
        impl::logical_tensor_t weight_lt = utils::logical_tensor_init(
                1, {1, 1, 3, 3}, impl::data_type::f32);
        impl::logical_tensor_t post_src_lt = utils::logical_tensor_init(
                2, {1, 1, 2, 2}, impl::data_type::f32);
        impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
                3, {1, 1, 2, 2}, impl::data_type::f32);
        impl::logical_tensor_t add_dst_lt = utils::logical_tensor_init(
                4, {1, 1, 2, 2}, impl::data_type::f32);

        impl::op_t in_op(0, impl::op_kind::Wildcard, "Wildcard");

        in_op.add_output(post_src_lt);
        conv_op.add_input(src_lt);
        conv_op.add_input(weight_lt);
        conv_op.add_output(dst_lt);
        if (swap) {
            add_op.add_input(post_src_lt);
            add_op.add_input(dst_lt);
        } else {
            add_op.add_input(dst_lt);
            add_op.add_input(post_src_lt);
        }
        add_op.add_output(add_dst_lt);

        impl::graph_t g;
        g.add_op(&in_op);
        g.add_op(&conv_op);
        g.add_op(&add_op);
        g.build_graph();

        impl::pass::pass_base_ptr apass = get_pass("conv_sum_fusion");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1);
        auto part = g.get_partitions()[0];

        // compile
        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> inputs {
                &src_lt, &weight_lt, &post_src_lt};
        std::vector<const impl::logical_tensor_t *> outputs {&add_dst_lt};

        p.compile(&cp, inputs, outputs, &eng);

        impl::logical_tensor_t lt;
        cp.query_logical_tensor(add_dst_lt.id, &lt);
        ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

        impl::tensor_t src_ts(src_lt, &eng, src.data());
        impl::tensor_t weight_ts(weight_lt, &eng, weight.data());
        impl::tensor_t post_src_ts(post_src_lt, &eng, post_src.data());
        impl::tensor_t add_dst_ts(add_dst_lt, &eng, dst.data());

        impl::stream_t &strm = get_stream();
        cp.execute(&strm, {src_ts, weight_ts, post_src_ts}, {add_dst_ts});
        strm.wait();
        for (size_t i = 0; i < dst.size(); ++i) {
            ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
        }
    }
}

TEST(operator_kernel, conv_per_tensor_broadcast_add) {
    using dims = impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> post_src {3.0};
    test::vector<float> ref_dst {2.0, 5.5, 8.0, 4.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};
    impl::op_t conv_op(1, impl::op_kind::Convolution, "Convolution");
    conv_op.set_attr<dims>("strides", {1, 1});
    conv_op.set_attr<dims>("dilations", {1, 1});
    conv_op.set_attr<dims>("pads_begin", {0, 0});
    conv_op.set_attr<dims>("pads_end", {0, 0});
    conv_op.set_attr<int64_t>("groups", 1);
    conv_op.set_attr<std::string>("data_format", "NCX");
    conv_op.set_attr<std::string>("filter_format", "OIX");
    impl::op_t add_op(2, impl::op_kind::Add, "Add");

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt
            = utils::logical_tensor_init(1, {1, 1, 3, 3}, impl::data_type::f32);
    // post_src will first be unsequeeze to {1,1,1,1} and then broadcast
    // to {1,1,2,2}
    impl::logical_tensor_t post_src_lt
            = utils::logical_tensor_init(2, {1}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(3, {1, 1, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t add_dst_lt
            = utils::logical_tensor_init(4, {1, 1, 2, 2}, impl::data_type::f32);

    impl::op_t in_op(0, impl::op_kind::Wildcard, "Wildcard");

    in_op.add_output(post_src_lt);
    conv_op.add_input(src_lt);
    conv_op.add_input(weight_lt);
    conv_op.add_output(dst_lt);
    add_op.add_input(dst_lt);
    add_op.add_input(post_src_lt);
    add_op.add_output(add_dst_lt);

    impl::graph_t g;
    g.add_op(&in_op);
    g.add_op(&conv_op);
    g.add_op(&add_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("conv_sum_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {
            &src_lt, &weight_lt, &post_src_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&add_dst_lt};

    p.compile(&cp, inputs, outputs, &eng);
    impl::logical_tensor_t lt;
    cp.query_logical_tensor(add_dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t weight_ts(weight_lt, &eng, weight.data());
    impl::tensor_t post_src_ts(post_src_lt, &eng, post_src.data());
    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src_ts, weight_ts, post_src_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, conv_expanded_per_tensor_broadcast_add) {
    using dims = impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> post_src {3.0};
    test::vector<float> ref_dst {2.0, 5.5, 8.0, 4.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};
    impl::op_t conv_op(1, impl::op_kind::Convolution, "Convolution");
    conv_op.set_attr<dims>("strides", {1, 1});
    conv_op.set_attr<dims>("dilations", {1, 1});
    conv_op.set_attr<dims>("pads_begin", {0, 0});
    conv_op.set_attr<dims>("pads_end", {0, 0});
    conv_op.set_attr<int64_t>("groups", 1);
    conv_op.set_attr<std::string>("data_format", "NCX");
    conv_op.set_attr<std::string>("filter_format", "OIX");
    impl::op_t add_op(2, impl::op_kind::Add, "Add");

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt
            = utils::logical_tensor_init(1, {1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t post_src_lt
            = utils::logical_tensor_init(2, {1, 1, 1, 1}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(3, {1, 1, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t add_dst_lt
            = utils::logical_tensor_init(4, {1, 1, 2, 2}, impl::data_type::f32);

    impl::op_t in_op(0, impl::op_kind::Wildcard, "Wildcard");

    in_op.add_output(post_src_lt);
    conv_op.add_input(src_lt);
    conv_op.add_input(weight_lt);
    conv_op.add_output(dst_lt);
    add_op.add_input(dst_lt);
    add_op.add_input(post_src_lt);
    add_op.add_output(add_dst_lt);

    impl::graph_t g;
    g.add_op(&in_op);
    g.add_op(&conv_op);
    g.add_op(&add_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("conv_sum_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {
            &src_lt, &weight_lt, &post_src_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&add_dst_lt};

    p.compile(&cp, inputs, outputs, &eng);
    impl::logical_tensor_t lt;
    cp.query_logical_tensor(add_dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t weight_ts(weight_lt, &eng, weight.data());
    impl::tensor_t post_src_ts(post_src_lt, &eng, post_src.data());
    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src_ts, weight_ts, post_src_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, conv_per_channel_broadcast_add) {
    using dims = impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
            1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> post_src {3.0, 3.0};
    test::vector<float> ref_dst {2.0, 5.5, 8.0, 4.5, 2.0, 5.5, 8.0, 4.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    impl::op_t conv_op(1, impl::op_kind::Convolution, "Convolution");
    conv_op.set_attr<dims>("strides", {1, 1});
    conv_op.set_attr<dims>("dilations", {1, 1});
    conv_op.set_attr<dims>("pads_begin", {0, 0});
    conv_op.set_attr<dims>("pads_end", {0, 0});
    conv_op.set_attr<int64_t>("groups", 1);
    conv_op.set_attr<std::string>("data_format", "NCX");
    conv_op.set_attr<std::string>("filter_format", "OIX");
    impl::op_t add_op(2, impl::op_kind::Add, "Add");

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt
            = utils::logical_tensor_init(1, {2, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t post_src_lt
            = utils::logical_tensor_init(2, {1, 2, 1, 1}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(3, {1, 2, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t add_dst_lt
            = utils::logical_tensor_init(4, {1, 2, 2, 2}, impl::data_type::f32);

    impl::op_t in_op(0, impl::op_kind::Wildcard, "Wildcard");

    in_op.add_output(post_src_lt);
    conv_op.add_input(src_lt);
    conv_op.add_input(weight_lt);
    conv_op.add_output(dst_lt);
    add_op.add_input(dst_lt);
    add_op.add_input(post_src_lt);
    add_op.add_output(add_dst_lt);

    impl::graph_t g;
    g.add_op(&in_op);
    g.add_op(&conv_op);
    g.add_op(&add_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("conv_sum_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {
            &src_lt, &weight_lt, &post_src_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&add_dst_lt};

    p.compile(&cp, inputs, outputs, &eng);
    impl::logical_tensor_t lt;
    cp.query_logical_tensor(add_dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t weight_ts(weight_lt, &eng, weight.data());
    impl::tensor_t post_src_ts(post_src_lt, &eng, post_src.data());
    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src_ts, weight_ts, post_src_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, conv_nxc_per_channel_broadcast_add) {
    using dims = impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0,
            1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0};
    test::vector<float> post_src {3.0, 3.0};
    test::vector<float> ref_dst {2.0, 2.0, 5.5, 5.5, 8.0, 8.0, 4.5, 4.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    impl::op_t conv_op(1, impl::op_kind::Convolution, "Convolution");
    conv_op.set_attr<dims>("strides", {1, 1});
    conv_op.set_attr<dims>("dilations", {1, 1});
    conv_op.set_attr<dims>("pads_begin", {0, 0});
    conv_op.set_attr<dims>("pads_end", {0, 0});
    conv_op.set_attr<int64_t>("groups", 1);
    conv_op.set_attr<std::string>("data_format", "NXC");
    conv_op.set_attr<std::string>("filter_format", "XIO");
    impl::op_t add_op(2, impl::op_kind::Add, "Add");

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 4, 4, 1}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt
            = utils::logical_tensor_init(1, {3, 3, 1, 2}, impl::data_type::f32);
    impl::logical_tensor_t post_src_lt
            = utils::logical_tensor_init(2, {1, 1, 1, 2}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(3, {1, 2, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t add_dst_lt
            = utils::logical_tensor_init(4, {1, 2, 2, 2}, impl::data_type::f32);

    impl::op_t in_op(0, impl::op_kind::Wildcard, "Wildcard");

    in_op.add_output(post_src_lt);
    conv_op.add_input(src_lt);
    conv_op.add_input(weight_lt);
    conv_op.add_output(dst_lt);
    add_op.add_input(dst_lt);
    add_op.add_input(post_src_lt);
    add_op.add_output(add_dst_lt);

    impl::graph_t g;
    g.add_op(&in_op);
    g.add_op(&conv_op);
    g.add_op(&add_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("conv_sum_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {
            &src_lt, &weight_lt, &post_src_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&add_dst_lt};

    p.compile(&cp, inputs, outputs, &eng);
    impl::logical_tensor_t lt;
    cp.query_logical_tensor(add_dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t weight_ts(weight_lt, &eng, weight.data());
    impl::tensor_t post_src_ts(post_src_lt, &eng, post_src.data());
    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src_ts, weight_ts, post_src_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, conv_unsupported_broadcast_add) {
    using dims = impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> post_src {3.0, 3.0};
    impl::op_t conv_op(1, impl::op_kind::Convolution, "Convolution");
    conv_op.set_attr<dims>("strides", {1, 1});
    conv_op.set_attr<dims>("dilations", {1, 1});
    conv_op.set_attr<dims>("pads_begin", {0, 0});
    conv_op.set_attr<dims>("pads_end", {0, 0});
    conv_op.set_attr<int64_t>("groups", 1);
    conv_op.set_attr<std::string>("data_format", "NCX");
    conv_op.set_attr<std::string>("filter_format", "OIX");
    impl::op_t add_op(2, impl::op_kind::Add, "Add");

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt
            = utils::logical_tensor_init(1, {1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t post_src_lt
            = utils::logical_tensor_init(2, {2}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(3, {1, 1, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t add_dst_lt
            = utils::logical_tensor_init(4, {1, 1, 2, 2}, impl::data_type::f32);

    impl::op_t in_op(0, impl::op_kind::Wildcard, "Wildcard");

    in_op.add_output(post_src_lt);
    conv_op.add_input(src_lt);
    conv_op.add_input(weight_lt);
    conv_op.add_output(dst_lt);
    add_op.add_input(dst_lt);
    add_op.add_input(post_src_lt);
    add_op.add_output(add_dst_lt);

    impl::graph_t g;
    g.add_op(&in_op);
    g.add_op(&conv_op);
    g.add_op(&add_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("conv_sum_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {
            &src_lt, &weight_lt, &post_src_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&add_dst_lt};

    ASSERT_NE(p.compile(&cp, inputs, outputs, &eng), impl::status::success);
}

TEST(operator_kernel, conv_add_relu) {
    using dims = impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> post_src {-1.0, -2.0, -3.0, -4.0};
    test::vector<float> ref_dst {0.0, 0.5, 2.0, 0.0};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};
    impl::op_t in_op(0, impl::op_kind::Wildcard, "Wildcard");

    impl::op_t conv_op(1, impl::op_kind::Convolution, "Convolution");

    conv_op.set_attr<dims>("strides", {1, 1});
    conv_op.set_attr<dims>("dilations", {1, 1});
    conv_op.set_attr<dims>("pads_begin", {0, 0});
    conv_op.set_attr<dims>("pads_end", {0, 0});
    conv_op.set_attr<int64_t>("groups", 1);
    conv_op.set_attr<std::string>("data_format", "NCX");
    conv_op.set_attr<std::string>("filter_format", "OIX");
    impl::op_t add_op(2, impl::op_kind::Add, "Add");
    impl::op_t relu_op(3, impl::op_kind::ReLU, "ReLU");

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt
            = utils::logical_tensor_init(1, {1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t post_lt
            = utils::logical_tensor_init(2, {1, 1, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(3, {1, 1, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t add_dst_lt
            = utils::logical_tensor_init(4, {1, 1, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t relu_dst_lt
            = utils::logical_tensor_init(5, {1, 1, 2, 2}, impl::data_type::f32);

    in_op.add_output(post_lt);
    conv_op.add_input(src_lt);
    conv_op.add_input(weight_lt);
    conv_op.add_output(dst_lt);
    add_op.add_input(dst_lt);
    add_op.add_input(post_lt);
    add_op.add_output(add_dst_lt);
    relu_op.add_input(add_dst_lt);
    relu_op.add_output(relu_dst_lt);

    impl::graph_t g;
    g.add_op(&in_op);
    g.add_op(&conv_op);
    g.add_op(&add_op);
    g.add_op(&relu_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("conv_sum_relu_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {
            &src_lt, &weight_lt, &post_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&relu_dst_lt};

    p.compile(&cp, inputs, outputs, &eng);

    impl::logical_tensor_t lt;
    cp.query_logical_tensor(relu_dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t weight_ts(weight_lt, &eng, weight.data());
    impl::tensor_t post_src_ts(post_lt, &eng, post_src.data());
    impl::tensor_t relu_dst_ts(relu_dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src_ts, weight_ts, post_src_ts}, {relu_dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, abs) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src {-2.0, -1.5, -1.0, -0.5, 0.0, 3.5};
    test::vector<float> ref_dst {2.0, 1.5, 1.0, 0.5, 0.0, 3.5};
    test::vector<float> dst(src.size(), 0.0);

    impl::op_t op(impl::op_kind::Abs);

    auto &op_factory = get_dnnl_kernel_registry();
    auto this_kernel = op_factory.create_kernel(op);
    ASSERT_TRUE(this_kernel);

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {1, 2, 3}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    // compile the relu operator
    this_kernel->compile(&op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t dst_ts(outputs[0], &eng, dst.data());

    impl::stream_t &strm = get_stream();
    this_kernel->execute(&op, &strm, {src_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, elu) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src {-2.0, -1.5, -1.0, -0.5, 0.0, 3.5};
    test::vector<float> ref_dst;
    test::vector<float> dst(src.size(), 0.0);

    impl::op_t op(impl::op_kind::Elu);
    op.set_attr<float>("alpha", 1.f);
    auto &op_factory = get_dnnl_kernel_registry();
    auto this_kernel = op_factory.create_kernel(op);
    ASSERT_TRUE(this_kernel);

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {1, 2, 3}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    // compile the relu operator
    this_kernel->compile(&op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t dst_ts(outputs[0], &eng, dst.data());

    impl::stream_t &strm = get_stream();
    this_kernel->execute(&op, &strm, {src_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src.size(); ++i) {
        float temp;
        if (src[i] > 0) {
            temp = src[i];
        } else {
            temp = static_cast<float>(exp(src[i]) - 1);
        }
        ref_dst.push_back(temp);
    }

    for (size_t i = 0; i < src.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, exp) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src {-2.0, -1.5, -1.0, -0.5, 0.0, 3.5};
    test::vector<float> ref_dst;
    test::vector<float> dst(src.size(), 0.0);

    impl::op_t op(impl::op_kind::Exp);

    auto &op_factory = get_dnnl_kernel_registry();
    auto this_kernel = op_factory.create_kernel(op);
    ASSERT_TRUE(this_kernel);

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {1, 2, 3}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    // compile the relu operator
    this_kernel->compile(&op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t dst_ts(outputs[0], &eng, dst.data());

    impl::stream_t &strm = get_stream();
    this_kernel->execute(&op, &strm, {src_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src.size(); ++i) {
        float temp = static_cast<float>(exp(src[i]));
        ref_dst.push_back(temp);
    }

    for (size_t i = 0; i < src.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, hardtanh) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src {-2.0, -1.5, -1.0, -0.5, 0.0, 3.5};
    test::vector<float> ref_dst {-1.0, -1.0, -1.0, -0.5, 0.0, 2.0};
    test::vector<float> dst(src.size(), 0.0);

    impl::op_t op(impl::op_kind::HardTanh);
    op.set_attr<float>("max", 2.f);
    op.set_attr<float>("min", -1.f);
    auto &op_factory = get_dnnl_kernel_registry();
    auto this_kernel = op_factory.create_kernel(op);
    ASSERT_TRUE(this_kernel);

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {1, 2, 3}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    // compile the relu operator
    this_kernel->compile(&op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t dst_ts(outputs[0], &eng, dst.data());

    impl::stream_t &strm = get_stream();
    this_kernel->execute(&op, &strm, {src_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, sqrt) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src {2.0, 1.5, 1.0, 0.5, 0.0, 3.5};
    test::vector<float> ref_dst;
    test::vector<float> dst(src.size(), 0.0);

    impl::op_t op(impl::op_kind::Sqrt);

    auto &op_factory = get_dnnl_kernel_registry();
    auto this_kernel = op_factory.create_kernel(op);
    ASSERT_TRUE(this_kernel);

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {1, 2, 3}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    // compile the relu operator
    this_kernel->compile(&op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t dst_ts(outputs[0], &eng, dst.data());

    impl::stream_t &strm = get_stream();
    this_kernel->execute(&op, &strm, {src_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src.size(); ++i) {
        float temp = static_cast<float>(sqrt(src[i]));
        ref_dst.push_back(temp);
    }

    for (size_t i = 0; i < src.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, square) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src {-2.0, -1.5, -1.0, -0.5, 0.0, 3.5};
    test::vector<float> ref_dst;
    test::vector<float> dst(src.size(), 0.0);

    impl::op_t op(impl::op_kind::Square);

    auto &op_factory = get_dnnl_kernel_registry();
    auto this_kernel = op_factory.create_kernel(op);
    ASSERT_TRUE(this_kernel);

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {1, 2, 3}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    // compile the relu operator
    this_kernel->compile(&op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t dst_ts(outputs[0], &eng, dst.data());

    impl::stream_t &strm = get_stream();
    this_kernel->execute(&op, &strm, {src_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src.size(); ++i) {
        float temp = src[i] * src[i];
        ref_dst.push_back(temp);
    }

    for (size_t i = 0; i < src.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, pow) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src {-2.0, -1.5, -1.0, -0.5, 0.0, 3.5};
    test::vector<float> ref_dst;
    test::vector<float> dst(src.size(), 0.0);

    impl::op_t op(impl::op_kind::Pow);
    op.set_attr<float>("alpha", 1.f);
    op.set_attr<float>("beta", 3.f);
    auto &op_factory = get_dnnl_kernel_registry();
    auto this_kernel = op_factory.create_kernel(op);
    ASSERT_TRUE(this_kernel);

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {1, 2, 3}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    // compile the relu operator
    this_kernel->compile(&op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t dst_ts(outputs[0], &eng, dst.data());

    impl::stream_t &strm = get_stream();
    this_kernel->execute(&op, &strm, {src_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src.size(); ++i) {
        float temp = static_cast<float>(pow(src[i], 3.0));
        ref_dst.push_back(temp);
    }

    for (size_t i = 0; i < src.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, log) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src {2.f, 1.5f, 1.f, 0.5f, 0.8f, 3.5f};
    test::vector<float> ref_dst;
    test::vector<float> dst(src.size(), 0.0);

    impl::op_t op(impl::op_kind::Log);

    auto &op_factory = get_dnnl_kernel_registry();
    auto this_kernel = op_factory.create_kernel(op);
    ASSERT_TRUE(this_kernel);

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {1, 2, 3}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    // compile the relu operator
    this_kernel->compile(&op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t dst_ts(outputs[0], &eng, dst.data());

    impl::stream_t &strm = get_stream();
    this_kernel->execute(&op, &strm, {src_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src.size(); ++i) {
        float temp = static_cast<float>(log(src[i]));
        ref_dst.push_back(temp);
    }

    for (size_t i = 0; i < src.size(); ++i) {
        ASSERT_TRUE(std::fabs(dst[i] - ref_dst[i]) < 0.00001);
    }
}

TEST(operator_kernel, tanh) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src {-2.0, -1.5, -1.0, -0.5, 0.0, 3.5};
    test::vector<float> ref_dst;
    test::vector<float> dst(src.size(), 0.0);

    impl::op_t op(impl::op_kind::Tanh);

    auto &op_factory = get_dnnl_kernel_registry();
    auto this_kernel = op_factory.create_kernel(op);
    ASSERT_TRUE(this_kernel);

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {1, 2, 3}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    // compile the relu operator
    this_kernel->compile(&op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t dst_ts(outputs[0], &eng, dst.data());

    impl::stream_t &strm = get_stream();
    this_kernel->execute(&op, &strm, {src_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src.size(); ++i) {
        float temp = static_cast<float>(
                (exp(src[i]) - exp(-src[i])) / (exp(src[i]) + exp(-src[i])));
        ref_dst.push_back(temp);
    }

    for (size_t i = 0; i < src.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

struct eltwise_param {
    std::string pass_name;
    test::vector<float> bias;
    test::vector<float> ref_dst;
    impl::op_kind_t op_kind;
    std::string op_name;
    std::vector<std::pair<std::string, float>> attrs;
};

test::vector<float> sigmoid_func(const test::vector<float> &ref_dst) {
    test::vector<float> out;
    for (auto &rdst : ref_dst) {
        out.emplace_back(static_cast<float>(1 / (exp(-rdst) + 1)));
    }
    return out;
}

test::vector<float> tanh_func(const test::vector<float> &ref_dst) {
    test::vector<float> out;
    for (auto &rdst : ref_dst) {
        out.emplace_back(static_cast<float>(
                (exp(rdst) - exp(-rdst)) / (exp(rdst) + exp(-rdst))));
    }
    return out;
}

test::vector<float> sqrt_func(const test::vector<float> &ref_dst) {
    test::vector<float> out;
    for (auto &rdst : ref_dst) {
        out.emplace_back(static_cast<float>(sqrt(rdst)));
    }
    return out;
}

TEST(operator_compile, conv_bias_eltwise) {
    using dims = dnnl::graph::impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0.0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};

    std::vector<eltwise_param> params1 = {
            eltwise_param {"conv_bias_abs_fusion", {-1.0}, {2.0, 1.5, 4.0, 0.5},
                    impl::op_kind::Abs, "Abs", {}},
            eltwise_param {"conv_bias_elu_fusion", {-1.0},
                    {static_cast<float>(exp(-2) - 1), 1.5, 4.0, 0.5},
                    impl::op_kind::Elu, "Elu", {{"alpha", 1.f}}},
            eltwise_param {"conv_bias_hardtanh_fusion", {-1.0},
                    {0.0, 1.5, 3.0, 0.5}, impl::op_kind::HardTanh, "HardTanh",
                    {{"min", 0.f}, {"max", 3.f}}},
            eltwise_param {"conv_bias_sigmoid_fusion", {-1.0},
                    sigmoid_func({-2.0, 1.5, 4.0, 0.5}), impl::op_kind::Sigmoid,
                    "Sigmoid", {}},
            eltwise_param {"conv_bias_square_fusion", {-1.0},
                    {4.0, 2.25, 16.0, 0.25}, impl::op_kind::Square, "Square",
                    {}},
            eltwise_param {"conv_bias_tanh_fusion", {-1.0},
                    tanh_func({-2.0, 1.5, 4.0, 0.5}), impl::op_kind::Tanh,
                    "Tanh", {}},
            eltwise_param {"conv_bias_sqrt_fusion", {1.0},
                    sqrt_func({0.0, 3.5, 6.0, 2.5}), impl::op_kind::Sqrt,
                    "Sqrt", {}},
    };

    for (auto &param : params1) {
        impl::op_t conv_op(1, impl::op_kind::Convolution, "Convolution");
        conv_op.set_attr<dims>("strides", {1, 1});
        conv_op.set_attr<dims>("dilations", {1, 1});
        conv_op.set_attr<dims>("pads_begin", {0, 0});
        conv_op.set_attr<dims>("pads_end", {0, 0});
        conv_op.set_attr<int64_t>("groups", 1);
        conv_op.set_attr<std::string>("data_format", "NCX");
        conv_op.set_attr<std::string>("filter_format", "OIX");
        impl::op_t eltwise_op(2, param.op_kind, param.op_name);
        for (auto &attr : param.attrs) {
            eltwise_op.set_attr<float>(attr.first, attr.second);
        }

        // prepare logical tensor
        impl::logical_tensor_t src_lt = utils::logical_tensor_init(
                0, {1, 1, 4, 4}, impl::data_type::f32);
        impl::logical_tensor_t weight_lt = utils::logical_tensor_init(
                1, {1, 1, 3, 3}, impl::data_type::f32);
        impl::logical_tensor_t bias_lt
                = utils::logical_tensor_init(2, dims {1}, impl::data_type::f32);
        impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
                4, {1, 1, 2, 2}, impl::data_type::f32);
        impl::logical_tensor_t eltwise_dst_lt = utils::logical_tensor_init(
                5, {1, 1, 2, 2}, impl::data_type::f32);

        conv_op.add_input(src_lt);
        conv_op.add_input(weight_lt);
        conv_op.add_input(bias_lt);
        conv_op.add_output(dst_lt);
        eltwise_op.add_input(dst_lt);
        eltwise_op.add_output(eltwise_dst_lt);

        impl::graph_t g;
        g.add_op(&conv_op);
        g.add_op(&eltwise_op);
        g.build_graph();

        impl::pass::pass_base_ptr apass = get_pass(param.pass_name);
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1);
        auto part = g.get_partitions()[0];

        // compile
        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> inputs {
                &src_lt, &weight_lt, &bias_lt};
        std::vector<const impl::logical_tensor_t *> outputs {&eltwise_dst_lt};

        p.compile(&cp, inputs, outputs, &eng);

        impl::logical_tensor_t lt;
        cp.query_logical_tensor(eltwise_dst_lt.id, &lt);
        ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

        impl::tensor_t src_ts(src_lt, &eng, src.data());
        impl::tensor_t weight_ts(weight_lt, &eng, weight.data());
        impl::tensor_t bias_ts(bias_lt, &eng, param.bias.data());
        impl::tensor_t eltwise_dst_ts(eltwise_dst_lt, &eng, dst.data());

        impl::stream_t &strm = get_stream();
        cp.execute(&strm, {src_ts, weight_ts, bias_ts}, {eltwise_dst_ts});
        strm.wait();
        for (size_t i = 0; i < dst.size(); ++i) {
            ASSERT_FLOAT_EQ(dst[i], param.ref_dst[i]);
        }
    }
}

TEST(operator_compile, conv_bias_add_eltwise) {
    using dims = dnnl::graph::impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0.0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> post_src {-2.0, 1.0, -1.0, 0.0};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};

    std::vector<eltwise_param> params2 = {
            eltwise_param {"conv_bias_sum_elu_fusion", {-1.0},
                    {static_cast<float>(exp(-4.0) - 1), 2.5, 3.0, 0.5},
                    impl::op_kind::Elu, "Elu", {{"alpha", 1.f}}},
            eltwise_param {"conv_bias_sum_relu6_fusion", {3.0},
                    {0.0, 6.f, 6.f, 4.5}, impl::op_kind::HardTanh, "ReLU6",
                    {{"min", 0.f}, {"max", 6.f}}},
    };

    for (auto &param : params2) {
        impl::op_t in_op(0, impl::op_kind::Wildcard, "Wildcard");

        impl::op_t conv_op(1, impl::op_kind::Convolution, "Convolution");

        conv_op.set_attr<dims>("strides", {1, 1});
        conv_op.set_attr<dims>("dilations", {1, 1});
        conv_op.set_attr<dims>("pads_begin", {0, 0});
        conv_op.set_attr<dims>("pads_end", {0, 0});
        conv_op.set_attr<int64_t>("groups", 1);
        conv_op.set_attr<std::string>("data_format", "NCX");
        conv_op.set_attr<std::string>("filter_format", "OIX");

        impl::op_t add_op(2, impl::op_kind::Add, "Add");
        impl::op_t eltwise_op(3, param.op_kind, param.op_name);
        for (auto &attr : param.attrs) {
            eltwise_op.set_attr<float>(attr.first, attr.second);
        }

        // prepare logical tensor
        impl::logical_tensor_t src_lt = utils::logical_tensor_init(
                0, {1, 1, 4, 4}, impl::data_type::f32);
        impl::logical_tensor_t weight_lt = utils::logical_tensor_init(
                1, {1, 1, 3, 3}, impl::data_type::f32);
        impl::logical_tensor_t bias_lt
                = utils::logical_tensor_init(2, dims {1}, impl::data_type::f32);
        impl::logical_tensor_t post_lt = utils::logical_tensor_init(
                3, {1, 1, 2, 2}, impl::data_type::f32);
        impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
                4, {1, 1, 2, 2}, impl::data_type::f32);
        impl::logical_tensor_t add_dst_lt = utils::logical_tensor_init(
                5, {1, 1, 2, 2}, impl::data_type::f32);
        impl::logical_tensor_t eltwise_dst_lt = utils::logical_tensor_init(
                6, {1, 1, 2, 2}, impl::data_type::f32);

        in_op.add_output(post_lt);
        conv_op.add_input(src_lt);
        conv_op.add_input(weight_lt);
        conv_op.add_input(bias_lt);
        conv_op.add_output(dst_lt);
        add_op.add_input(dst_lt);
        add_op.add_input(post_lt);
        add_op.add_output(add_dst_lt);
        eltwise_op.add_input(add_dst_lt);
        eltwise_op.add_output(eltwise_dst_lt);

        impl::graph_t g;
        g.add_op(&in_op);
        g.add_op(&conv_op);
        g.add_op(&add_op);
        g.add_op(&eltwise_op);
        g.build_graph();

        impl::pass::pass_base_ptr apass = get_pass(param.pass_name);
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1);
        auto part = g.get_partitions()[0];

        // compile
        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> inputs {
                &src_lt, &weight_lt, &bias_lt, &post_lt};
        std::vector<const impl::logical_tensor_t *> outputs {&eltwise_dst_lt};

        p.compile(&cp, inputs, outputs, &eng);

        impl::logical_tensor_t lt;
        cp.query_logical_tensor(eltwise_dst_lt.id, &lt);
        ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

        impl::tensor_t src_ts(src_lt, &eng, src.data());
        impl::tensor_t weight_ts(weight_lt, &eng, weight.data());
        impl::tensor_t bias_ts(bias_lt, &eng, param.bias.data());
        impl::tensor_t post_src_ts(post_lt, &eng, post_src.data());
        impl::tensor_t eltwise_dst_ts(eltwise_dst_lt, &eng, dst.data());

        impl::stream_t &strm = get_stream();
        cp.execute(&strm, {src_ts, weight_ts, bias_ts, post_src_ts},
                {eltwise_dst_ts});
        strm.wait();
        for (size_t i = 0; i < dst.size(); ++i) {
            ASSERT_FLOAT_EQ(dst[i], param.ref_dst[i]);
        }
    }
}

TEST(operator_compile, conv_add_eltwise) {
    using dims = dnnl::graph::impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0.0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> post_src {-2.0, 1.0, -1.0, 0.0};
    test::vector<float> ref_dst {-3.0, 3.5, 4.0, 1.5};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};

    std::vector<eltwise_param> params = {
            eltwise_param {"conv_sum_elu_fusion", {0.0},
                    {static_cast<float>(exp(-3.0) - 1), 3.5, 4.0, 1.5},
                    impl::op_kind::Elu, "Elu", {{"alpha", 1.f}}},
            eltwise_param {"conv_sum_relu6_fusion", {0.0}, {0.0, 3.5, 4.f, 1.5},
                    impl::op_kind::HardTanh, "ReLU6",
                    {{"min", 0.f}, {"max", 6.f}}},
    };

    for (auto &param : params) {
        impl::op_t in_op(0, impl::op_kind::Wildcard, "Wildcard");
        impl::op_t conv_op(1, impl::op_kind::Convolution, "Convolution");
        conv_op.set_attr<dims>("strides", {1, 1});
        conv_op.set_attr<dims>("dilations", {1, 1});
        conv_op.set_attr<dims>("pads_begin", {0, 0});
        conv_op.set_attr<dims>("pads_end", {0, 0});
        conv_op.set_attr<int64_t>("groups", 1);
        conv_op.set_attr<std::string>("data_format", "NCX");
        conv_op.set_attr<std::string>("filter_format", "OIX");
        impl::op_t add_op(2, impl::op_kind::Add, "Add");
        impl::op_t eltwise_op(3, param.op_kind, param.op_name);
        for (auto &attr : param.attrs) {
            eltwise_op.set_attr<float>(attr.first, attr.second);
        }

        // prepare logical tensor
        impl::logical_tensor_t src_lt = utils::logical_tensor_init(
                0, {1, 1, 4, 4}, impl::data_type::f32);
        impl::logical_tensor_t weight_lt = utils::logical_tensor_init(
                1, {1, 1, 3, 3}, impl::data_type::f32);
        impl::logical_tensor_t post_lt = utils::logical_tensor_init(
                2, {1, 1, 2, 2}, impl::data_type::f32);
        impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
                3, {1, 1, 2, 2}, impl::data_type::f32);
        impl::logical_tensor_t add_dst_lt = utils::logical_tensor_init(
                4, {1, 1, 2, 2}, impl::data_type::f32);
        impl::logical_tensor_t eltwise_dst_lt = utils::logical_tensor_init(
                5, {1, 1, 2, 2}, impl::data_type::f32);

        in_op.add_output(post_lt);
        conv_op.add_input(src_lt);
        conv_op.add_input(weight_lt);
        conv_op.add_output(dst_lt);
        add_op.add_input(dst_lt);
        add_op.add_input(post_lt);
        add_op.add_output(add_dst_lt);
        eltwise_op.add_input(add_dst_lt);
        eltwise_op.add_output(eltwise_dst_lt);

        impl::graph_t g;
        g.add_op(&in_op);
        g.add_op(&conv_op);
        g.add_op(&add_op);
        g.add_op(&eltwise_op);
        g.build_graph();

        impl::pass::pass_base_ptr apass = get_pass(param.pass_name);
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1);
        auto part = g.get_partitions()[0];

        // compile
        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> inputs {
                &src_lt, &weight_lt, &post_lt};
        std::vector<const impl::logical_tensor_t *> outputs {&eltwise_dst_lt};

        p.compile(&cp, inputs, outputs, &eng);

        impl::logical_tensor_t lt;
        cp.query_logical_tensor(eltwise_dst_lt.id, &lt);
        ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

        impl::tensor_t src_ts(src_lt, &eng, src.data());
        impl::tensor_t weight_ts(weight_lt, &eng, weight.data());
        impl::tensor_t post_src_ts(post_lt, &eng, post_src.data());
        impl::tensor_t eltwise_dst_ts(eltwise_dst_lt, &eng, dst.data());

        impl::stream_t &strm = get_stream();
        cp.execute(&strm, {src_ts, weight_ts, post_src_ts}, {eltwise_dst_ts});
        strm.wait();
        for (size_t i = 0; i < dst.size(); ++i) {
            ASSERT_FLOAT_EQ(dst[i], param.ref_dst[i]);
        }
    }
}

TEST(operator_kernel, softmax) {
    impl::op_t softmax_op(impl::op_kind::SoftMax);
    impl::engine_t &eng = get_engine();

    softmax_op.set_attr<int64_t>("axis", 0);

    test::vector<float> src_data {0.0, 1.0, 2.0, 0.0, 1.0, 2.0};
    test::vector<float> ref_dst_data {0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {2, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst = utils::logical_tensor_init(
            1, {2, 3}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src};
    std::vector<impl::logical_tensor_t> outputs {dst};

    auto &op_factory = get_dnnl_kernel_registry();
    auto softmax_kernel = op_factory.create_kernel(softmax_op);

    softmax_kernel->compile(&softmax_op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src, &eng, src_data.data());
    impl::tensor_t dst_ts(outputs[0], &eng, dst_data.data());

    impl::stream_t &strm = get_stream();
    softmax_kernel->execute(&softmax_op, &strm, {src_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(operator_kernel, softmax_with_last_dim) {
    impl::op_t softmax_op(impl::op_kind::SoftMax);
    impl::engine_t &eng = get_engine();

    softmax_op.set_attr<int64_t>("axis", -1);

    test::vector<float> src_data {3.0, 3.0, 1.0, 1.0};
    test::vector<float> ref_dst_data {0.5, 0.5, 0.5, 0.5};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {2, 2}, impl::data_type::f32);
    impl::logical_tensor_t dst = utils::logical_tensor_init(
            1, {2, 2}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src};
    std::vector<impl::logical_tensor_t> outputs {dst};

    auto &op_factory = get_dnnl_kernel_registry();
    auto softmax_kernel = op_factory.create_kernel(softmax_op);

    softmax_kernel->compile(&softmax_op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src, &eng, src_data.data());
    impl::tensor_t dst_ts(outputs[0], &eng, dst_data.data());

    impl::stream_t &strm = get_stream();
    softmax_kernel->execute(&softmax_op, &strm, {src_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(operator_kernel, softmax_backward) {
    impl::op_t softmax_op(impl::op_kind::SoftMaxBackprop);
    impl::engine_t &eng = get_engine();

    softmax_op.set_attr<int64_t>("axis", 1);

    test::vector<float> dst {0.5, 0.5, 0.5, 0.5};
    test::vector<float> diff_dst {1.0, 5.0, 2.0, 4.0};
    test::vector<float> ref_diff_src {-1, 1, -0.5, 0.5};
    test::vector<float> diff_src(ref_diff_src.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(1, {2, 2}, impl::data_type::f32);
    impl::logical_tensor_t diff_dst_lt
            = utils::logical_tensor_init(2, {2, 2}, impl::data_type::f32);
    impl::logical_tensor_t diff_src_lt = utils::logical_tensor_init(
            3, {2, 2}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {diff_dst_lt, dst_lt};
    std::vector<impl::logical_tensor_t> outputs {diff_src_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto softmax_kernel = op_factory.create_kernel(softmax_op);

    softmax_kernel->compile(&softmax_op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());
    impl::tensor_t diff_dst_ts(diff_dst_lt, &eng, diff_dst.data());
    impl::tensor_t diff_src_ts(outputs[0], &eng, diff_src.data());

    impl::stream_t &strm = get_stream();
    softmax_kernel->execute(
            &softmax_op, &strm, {diff_dst_ts, dst_ts}, {diff_src_ts});
    strm.wait();
    for (size_t i = 0; i < ref_diff_src.size(); ++i) {
        ASSERT_FLOAT_EQ(diff_src[i], ref_diff_src[i]);
    }
}

TEST(operator_kernel, logsoftmax) {
    impl::op_t logsoftmax_op(impl::op_kind::LogSoftmax);
    impl::engine_t &eng = get_engine();

    logsoftmax_op.set_attr<int64_t>("axis", 0);

    test::vector<float> src_data {0.0, 1.0, 2.0, 0.0, 1.0, 2.0};
    test::vector<float> ref_dst_data {-0.6931472, -0.6931472, -0.6931472,
            -0.6931472, -0.6931472, -0.6931472};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {2, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst = utils::logical_tensor_init(
            1, {2, 3}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src};
    std::vector<impl::logical_tensor_t> outputs {dst};

    auto &op_factory = get_dnnl_kernel_registry();
    auto softmax_kernel = op_factory.create_kernel(logsoftmax_op);

    softmax_kernel->compile(&logsoftmax_op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src, &eng, src_data.data());
    impl::tensor_t dst_ts(outputs[0], &eng, dst_data.data());

    impl::stream_t &strm = get_stream();
    softmax_kernel->execute(&logsoftmax_op, &strm, {src_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(operator_kernel, logsoftmax_backward) {
    impl::op_t logsoftmax_op(impl::op_kind::LogSoftmaxBackprop);
    impl::engine_t &eng = get_engine();

    logsoftmax_op.set_attr<int64_t>("axis", 1);

    test::vector<float> dst {-0.6931472, -0.6931472, -0.6931472, -0.6931472};
    test::vector<float> diff_dst {1.0, 5.0, 2.0, 4.0};
    test::vector<float> ref_diff_src {-2, 2, -1, 1};
    test::vector<float> diff_src(ref_diff_src.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(1, {2, 2}, impl::data_type::f32);
    impl::logical_tensor_t diff_dst_lt
            = utils::logical_tensor_init(2, {2, 2}, impl::data_type::f32);
    impl::logical_tensor_t diff_src_lt = utils::logical_tensor_init(
            3, {2, 2}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {diff_dst_lt, dst_lt};
    std::vector<impl::logical_tensor_t> outputs {diff_src_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto softmax_kernel = op_factory.create_kernel(logsoftmax_op);

    softmax_kernel->compile(&logsoftmax_op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());
    impl::tensor_t diff_dst_ts(diff_dst_lt, &eng, diff_dst.data());
    impl::tensor_t diff_src_ts(outputs[0], &eng, diff_src.data());

    impl::stream_t &strm = get_stream();
    softmax_kernel->execute(
            &logsoftmax_op, &strm, {diff_dst_ts, dst_ts}, {diff_src_ts});
    strm.wait();
    for (size_t i = 0; i < ref_diff_src.size(); ++i) {
        ASSERT_FLOAT_EQ(diff_src[i], ref_diff_src[i]);
    }
}

TEST(operator_kernel, avg_pool_backward_exclude_pad) {
    using dims = dnnl::graph::impl::dnnl_impl::dims;
    impl::engine_t &eng = get_engine();

    test::vector<float> src {-2.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            3.0, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> diff_src(src.size(), 0.0);
    test::vector<float> ref_diff_src {-1.0, 1.5, 1.5, 10.0, 2.0, 4.0, 4.0, 4.0,
            2.0, 4.0, 4.0, 4.0, 12.0, -2.5, -2.5, -3.0};
    test::vector<float> diff_dst {
            -1.0, 3.0, 10.0, 4.0, 16.0, 8.0, 12.0, -5.0, -3.0};

    impl::op_t avg_pool_bwd_op(impl::op_kind::AvgPoolBackprop);
    avg_pool_bwd_op.set_attr<dims>("strides", {2, 2});
    avg_pool_bwd_op.set_attr<dims>("kernel", {2, 2});
    avg_pool_bwd_op.set_attr<dims>("pads_begin", {1, 1});
    avg_pool_bwd_op.set_attr<dims>("pads_end", {1, 1});
    avg_pool_bwd_op.set_attr<bool>("exclude_pad", true);

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t diff_src_lt = utils::logical_tensor_init(
            0, {1, 1, 4, 4}, impl::data_type::f32, impl::layout_type::any);
    impl::logical_tensor_t diff_dst_lt
            = utils::logical_tensor_init(1, {1, 1, 3, 3}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> inputs {src_lt, diff_dst_lt};
    std::vector<impl::logical_tensor_t> outputs {diff_src_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto avg_pool_bwd_kernel = op_factory.create_kernel(avg_pool_bwd_op);

    avg_pool_bwd_kernel->compile(&avg_pool_bwd_op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t diff_dst_ts(diff_dst_lt, &eng, diff_dst.data());
    impl::tensor_t diff_src_ts(outputs[0], &eng, diff_src.data());

    impl::stream_t &strm = get_stream();
    avg_pool_bwd_kernel->execute(
            &avg_pool_bwd_op, &strm, {src_ts, diff_dst_ts}, {diff_src_ts});
    strm.wait();
    for (size_t i = 0; i < diff_src.size(); ++i) {
        ASSERT_FLOAT_EQ(diff_src[i], ref_diff_src[i]);
    }
}

TEST(operator_kernel, avg_pool_backward_include_pad) {
    using dims = dnnl::graph::impl::dnnl_impl::dims;
    impl::engine_t &eng = get_engine();

    test::vector<float> src {-2.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            3.0, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> diff_src(src.size(), 0.0);
    test::vector<float> ref_diff_src {-0.25, 0.75, 0.75, 2.5, 1.0, 4.0, 4.0,
            2.0, 1.0, 4.0, 4.0, 2.0, 3.0, -1.25, -1.25, -3.0 / 4};
    test::vector<float> diff_dst {
            -1.0, 3.0, 10.0, 4.0, 16.0, 8.0, 12.0, -5.0, -3.0};

    impl::op_t avg_pool_bwd_op(impl::op_kind::AvgPoolBackprop);
    avg_pool_bwd_op.set_attr<dims>("strides", {2, 2});
    avg_pool_bwd_op.set_attr<dims>("kernel", {2, 2});
    avg_pool_bwd_op.set_attr<dims>("pads_begin", {1, 1});
    avg_pool_bwd_op.set_attr<dims>("pads_end", {1, 1});
    avg_pool_bwd_op.set_attr<bool>("exclude_pad", false);

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t diff_src_lt = utils::logical_tensor_init(
            0, {1, 1, 4, 4}, impl::data_type::f32, impl::layout_type::any);
    impl::logical_tensor_t diff_dst_lt
            = utils::logical_tensor_init(1, {1, 1, 3, 3}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> inputs {src_lt, diff_dst_lt};
    std::vector<impl::logical_tensor_t> outputs {diff_src_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto avg_pool_bwd_kernel = op_factory.create_kernel(avg_pool_bwd_op);

    avg_pool_bwd_kernel->compile(&avg_pool_bwd_op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t diff_dst_ts(diff_dst_lt, &eng, diff_dst.data());
    impl::tensor_t diff_src_ts(outputs[0], &eng, diff_src.data());

    impl::stream_t &strm = get_stream();
    avg_pool_bwd_kernel->execute(
            &avg_pool_bwd_op, &strm, {src_ts, diff_dst_ts}, {diff_src_ts});
    strm.wait();
    for (size_t i = 0; i < diff_src.size(); ++i) {
        ASSERT_FLOAT_EQ(diff_src[i], ref_diff_src[i]);
    }
}

TEST(operator_kernel, max_pool_backward_with_incides) {
    using dims = dnnl::graph::impl::dnnl_impl::dims;
    impl::engine_t &eng = get_engine();

    test::vector<float> src {-2.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            3.0, -1.0, 0.0, 1.0, -2.0, -1.0, 4.0};
    test::vector<float> diff_src(src.size(), 0.0);
    test::vector<float> ref_diff_src {0.0, 0.0, 16.0, 0.0, 4.0, 0.0, 0.0, 0.0,
            0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.0};

    test::vector<float> diff_dst {4.0, 16.0, 8.0, 12.0};
#if DNNL_GRAPH_WITH_SYCL
    test::vector<int32_t> indices {2, 0, 1, 3};
#else
    test::vector<uint8_t> indices {2, 0, 1, 3};
#endif

    impl::op_t max_pool_bwd_op(impl::op_kind::MaxPoolBackprop);

    max_pool_bwd_op.set_attr<dims>("strides", dims {2, 2});
    max_pool_bwd_op.set_attr<dims>("kernel", dims {2, 2});
    max_pool_bwd_op.set_attr<dims>("pads_begin", dims {0, 0});
    max_pool_bwd_op.set_attr<dims>("pads_end", dims {0, 0});
    max_pool_bwd_op.set_attr<dims>("dilations", {1, 1});

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t diff_src_lt = utils::logical_tensor_init(
            1, {1, 1, 4, 4}, impl::data_type::f32, impl::layout_type::any);
    impl::logical_tensor_t diff_dst_lt
            = utils::logical_tensor_init(3, {1, 1, 2, 2}, impl::data_type::f32);
#if DNNL_GRAPH_WITH_SYCL
    impl::logical_tensor_t indices_lt
            = utils::logical_tensor_init(4, {1, 1, 2, 2}, impl::data_type::s32);
#else
    impl::logical_tensor_t indices_lt
            = utils::logical_tensor_init(4, {1, 1, 2, 2}, impl::data_type::u8);
#endif

    auto &op_factory = get_dnnl_kernel_registry();
    auto max_pool_bwd_kernel = op_factory.create_kernel(max_pool_bwd_op);
    std::vector<impl::logical_tensor_t> outputs {diff_src_lt};
    max_pool_bwd_kernel->compile(
            &max_pool_bwd_op, &eng, {src_lt, diff_dst_lt}, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t diff_dst_ts(diff_dst_lt, &eng, diff_dst.data());
    impl::tensor_t diff_src_ts(outputs[0], &eng, diff_src.data());
    impl::tensor_t indices_ts(indices_lt, &eng, indices.data());

    impl::stream_t &strm = get_stream();
    max_pool_bwd_kernel->execute(&max_pool_bwd_op, &strm,
            {src_ts, indices_ts, diff_dst_ts}, {diff_src_ts});
    strm.wait();

    for (size_t i = 0; i < diff_src.size(); ++i) {
        ASSERT_FLOAT_EQ(diff_src[i], ref_diff_src[i]);
    }
}

TEST(operator_kernel, max_pool_backward_without_incides) {
    using dims = dnnl::graph::impl::dnnl_impl::dims;
    impl::engine_t &eng = get_engine();

    test::vector<float> src {-2.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            3.0, -1.0, 0.0, 1.0, -2.0, -1.0, 4.0};
    test::vector<float> diff_src(src.size(), 0.0);
    test::vector<float> ref_diff_src {0.0, 0.0, 16.0, 0.0, 4.0, 0.0, 0.0, 0.0,
            0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.0};

    test::vector<float> diff_dst {4.0, 16.0, 8.0, 12.0};

    impl::op_t max_pool_bwd_op(impl::op_kind::MaxPoolBackprop);

    max_pool_bwd_op.set_attr<dims>("strides", dims {2, 2});
    max_pool_bwd_op.set_attr<dims>("kernel", dims {2, 2});
    max_pool_bwd_op.set_attr<dims>("pads_begin", dims {0, 0});
    max_pool_bwd_op.set_attr<dims>("pads_end", dims {0, 0});
    max_pool_bwd_op.set_attr<dims>("dilations", {1, 1});

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t diff_src_lt = utils::logical_tensor_init(
            1, {1, 1, 4, 4}, impl::data_type::f32, impl::layout_type::any);
    impl::logical_tensor_t diff_dst_lt
            = utils::logical_tensor_init(3, {1, 1, 2, 2}, impl::data_type::f32);

    std::vector<impl::logical_tensor_t> outputs {diff_src_lt};
    auto &op_factory = get_dnnl_kernel_registry();
    auto max_pool_bwd_kernel = op_factory.create_kernel(max_pool_bwd_op);
    max_pool_bwd_kernel->compile(
            &max_pool_bwd_op, &eng, {src_lt, diff_dst_lt}, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t diff_dst_ts(diff_dst_lt, &eng, diff_dst.data());
    impl::tensor_t diff_src_ts(outputs[0], &eng, diff_src.data());

    impl::stream_t &strm = get_stream();
    max_pool_bwd_kernel->execute(
            &max_pool_bwd_op, &strm, {src_ts, diff_dst_ts}, {diff_src_ts});
    strm.wait();

    for (size_t i = 0; i < diff_src.size(); ++i) {
        ASSERT_FLOAT_EQ(diff_src[i], ref_diff_src[i]);
    }
}

TEST(operator_kernel, layernorm_training) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src {2.0, 4.0, 5.0, 2.0, 3.0, 5.0};
    test::vector<float> scale {1.0, 2.0};
    test::vector<float> shift {0.0, 1.0};
    test::vector<float> ref_dst {-1.0, 3.0, 1.0, -1.0, -1.0, 3.0};
    test::vector<float> ref_mean {3.0, 3.5, 4.0};
    test::vector<float> ref_var {1.0, 2.25, 1.0};
    test::vector<float> dst(src.size(), 0.0);
    test::vector<float> mean(ref_mean.size(), 0.0);
    test::vector<float> var(ref_var.size(), 0.0);

    impl::op_t op(impl::op_kind::LayerNorm);

    op.set_attr<float>("epsilon", 0);

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 3, 2}, impl::data_type::f32);
    impl::logical_tensor_t scale_lt
            = utils::logical_tensor_init(1, {2}, impl::data_type::f32);
    impl::logical_tensor_t shift_lt
            = utils::logical_tensor_init(2, {2}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            3, {1, 3, 2}, impl::data_type::f32, impl::layout_type::any);
    impl::logical_tensor_t mean_lt = utils::logical_tensor_init(
            4, {1, 3}, impl::data_type::f32, impl::layout_type::any);
    impl::logical_tensor_t variance_lt = utils::logical_tensor_init(
            5, {1, 3}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src_lt, scale_lt, shift_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt, mean_lt, variance_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto kernel = op_factory.create_kernel(op);

    kernel->compile(&op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);
    ASSERT_EQ(outputs[1].layout_type, impl::layout_type::opaque);
    ASSERT_EQ(outputs[2].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t scale_ts(scale_lt, &eng, scale.data());
    impl::tensor_t shift_ts(shift_lt, &eng, shift.data());
    impl::tensor_t dst_ts(outputs[0], &eng, dst.data());
    impl::tensor_t mean_ts(outputs[1], &eng, mean.data());
    impl::tensor_t var_ts(outputs[2], &eng, var.data());

    impl::stream_t &strm = get_stream();
    kernel->execute(&op, &strm, {src_ts, scale_ts, shift_ts},
            {dst_ts, mean_ts, var_ts});
    strm.wait();

    for (size_t i = 0; i < ref_dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }

    for (size_t i = 0; i < ref_mean.size(); ++i) {
        ASSERT_FLOAT_EQ(mean[i], ref_mean[i]);
        ASSERT_FLOAT_EQ(var[i], ref_var[i]);
    }
}

TEST(operator_kernel, layernorm_inference) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src {2.0, 4.0, 3.0, 5.5, 5.0, 4.0, 1.0, 2.5};
    test::vector<float> scale {1.0, 2.0};
    test::vector<float> shift {0.0, 1.0};
    test::vector<float> ref_dst {-1.0, 3.0, -1.0, 3.0, 1.0, -1.0, -1.0, 3.0};
    test::vector<float> dst(src.size(), 0.0);

    impl::op_t op(impl::op_kind::LayerNorm);

    op.set_attr<float>("epsilon", 0);
    op.set_attr<bool>("keep_stats", false); //inference

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {2, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t scale_lt
            = utils::logical_tensor_init(1, {2}, impl::data_type::f32);
    impl::logical_tensor_t shift_lt
            = utils::logical_tensor_init(2, {2}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            3, {2, 2, 2}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src_lt, scale_lt, shift_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto kernel = op_factory.create_kernel(op);

    kernel->compile(&op, &eng, inputs, outputs);

    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t scale_ts(scale_lt, &eng, scale.data());
    impl::tensor_t shift_ts(shift_lt, &eng, shift.data());
    impl::tensor_t dst_ts(outputs[0], &eng, dst.data());

    impl::stream_t &strm = get_stream();
    kernel->execute(&op, &strm, {src_ts, scale_ts, shift_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < ref_dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, layernorm_inference_without_scale_shift) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src {2.0, 4.0, 3.0, 5.5, 5.0, 4.0, 1.0, 2.5};
    test::vector<float> ref_dst {-1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0};
    test::vector<float> dst(src.size(), 0.0);

    impl::op_t op(impl::op_kind::LayerNorm);

    op.set_attr<float>("epsilon", 0);
    op.set_attr<bool>("keep_stats", false); //inference
    op.set_attr<bool>("use_affine", false);

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {2, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            3, {2, 2, 2}, impl::data_type::f32, impl::layout_type::any);

    std::vector<impl::logical_tensor_t> inputs {src_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    auto &op_factory = get_dnnl_kernel_registry();
    auto kernel = op_factory.create_kernel(op);

    kernel->compile(&op, &eng, inputs, outputs);

    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t dst_ts(outputs[0], &eng, dst.data());

    impl::stream_t &strm = get_stream();
    kernel->execute(&op, &strm, {src_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < ref_dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, reorder_data) {
    impl::engine_t &engine = get_engine();

    test::vector<float> src {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    test::vector<float> ref_dst {1.0, 4.0, 2.0, 5.0, 3.0, 6.0};
    test::vector<float> dst(src.size(), 10);

    impl::op_t reorder_op(impl::op_kind::Reorder);

    // prepare input/output logical tensor
    // [[1, 2, 3],
    //  [4, 5, 6]]
    impl::logical_tensor_t src_lt = utils::logical_tensor_init(
            0, {2, 3}, {3, 1}, impl::data_type::f32);
    // [[1, 4],
    //  [2, 5],
    //  [3, 6]]
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {2, 3}, {1, 2}, impl::data_type::f32);

    auto &op_factory = get_dnnl_kernel_registry();
    auto kernel = op_factory.create_kernel(reorder_op);

    kernel->compile(&reorder_op, &engine, {src_lt}, {dst_lt});

    impl::stream_t &stream = get_stream();
    impl::tensor_t src_ts(src_lt, &engine, src.data());
    impl::tensor_t dst_ts(dst_lt, &engine, dst.data());

    kernel->execute(&reorder_op, &stream, {src_ts}, {dst_ts});
    stream.wait();
    for (size_t i = 0; i < src.size(); ++i) {
        ASSERT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_compile, reorder_negative_test) {
    impl::engine_t &engine = get_engine();

    impl::op_t reorder_op(impl::op_kind::Reorder);
    auto &op_factory = get_dnnl_kernel_registry();
    auto kernel = op_factory.create_kernel(reorder_op);

    impl::logical_tensor_t src_lt = utils::logical_tensor_init(
            0, {2, 3}, {3, 1}, impl::data_type::f32);

    // failure case: different shape (dims)
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {1, 2, 3}, {1, 1, 2}, impl::data_type::f32);

    ASSERT_EQ(kernel->compile(&reorder_op, &engine, {src_lt}, {dst_lt}),
            impl::status::compile_fail);
}

TEST(operator_kernel, typecast) {
    impl::engine_t &engine = get_engine();

    test::vector<float> f32_val {
            12.5234537, 0, -32.6735142, -1, -2.8765223, 66.66};
    test::vector<float> ref_val {12.5, 0, -32.75, -1, -2.875, 66.5};
    test::vector<uint16_t> bf16_val(f32_val.size(), 10);

    impl::op_t typecast_op(impl::op_kind::TypeCast);

    // prepare input/output logical tensor
    // shape (2, 3), stride (3, 1), plain format
    impl::logical_tensor_t f32_lt = utils::logical_tensor_init(
            0, {2, 3}, {3, 1}, impl::data_type::f32);
    impl::logical_tensor_t bf16_lt = utils::logical_tensor_init(
            1, {2, 3}, {3, 1}, impl::data_type::bf16);

    auto &op_factory = get_dnnl_kernel_registry();
    auto kernel = op_factory.create_kernel(typecast_op);

    // f32 --> bf16
    kernel->compile(&typecast_op, &engine, {f32_lt}, {bf16_lt});

    impl::stream_t &stream = get_stream();
    impl::tensor_t f32_ts(f32_lt, &engine, f32_val.data());
    impl::tensor_t bf16_ts(bf16_lt, &engine, bf16_val.data());

    kernel->execute(&typecast_op, &stream, {f32_ts}, {bf16_ts});

    // bf16 --> f32
    kernel->compile(&typecast_op, &engine, {bf16_lt}, {f32_lt});
    kernel->execute(&typecast_op, &stream, {bf16_ts}, {f32_ts});

    stream.wait();
    for (size_t i = 0; i < f32_val.size(); ++i) {
        ASSERT_EQ(f32_val[i], ref_val[i]);
    }
}

TEST(operator_compile, typecast_negative_test) {
    impl::engine_t &engine = get_engine();

    impl::op_t typecast_op(impl::op_kind::TypeCast);
    auto &op_factory = get_dnnl_kernel_registry();
    auto kernel = op_factory.create_kernel(typecast_op);

    impl::logical_tensor_t src_lt = utils::logical_tensor_init(
            0, {2, 3}, {3, 1}, impl::data_type::f32);

    // failure case : different shape (dims)
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {1, 2, 3}, {1, 1, 2}, impl::data_type::f32);

    ASSERT_EQ(kernel->compile(&typecast_op, &engine, {src_lt}, {dst_lt}),
            impl::status::compile_fail);

    impl::logical_tensor_t src_bf16_lt = utils::logical_tensor_init(
            0, {2, 3}, {3, 1}, impl::data_type::bf16);

    // failure case : unsupported type bf16->f16
    impl::logical_tensor_t dst_f16_lt = utils::logical_tensor_init(
            1, {1, 2, 3}, {1, 1, 2}, impl::data_type::f16);

    ASSERT_EQ(
            kernel->compile(&typecast_op, &engine, {src_bf16_lt}, {dst_f16_lt}),
            impl::status::compile_fail);

    impl::logical_tensor_t src_f16_lt = utils::logical_tensor_init(
            0, {2, 3}, {3, 1}, impl::data_type::f16);

    // failure case : unsupported type f16->bf16
    impl::logical_tensor_t dst_bf16_lt = utils::logical_tensor_init(
            1, {1, 2, 3}, {1, 1, 2}, impl::data_type::bf16);

    ASSERT_EQ(
            kernel->compile(&typecast_op, &engine, {src_f16_lt}, {dst_bf16_lt}),
            impl::status::compile_fail);
}

TEST(operator_kernel, quantize_per_tensor) {
    impl::engine_t &engine = get_engine();
    if (engine.kind() == impl::engine_kind::gpu) return;

    impl::op_t quantize(impl::op_kind::Quantize);
    quantize.set_attr<std::vector<float>>("scales", {0.1f});
    quantize.set_attr<std::vector<int64_t>>("zps", {10});
    quantize.set_attr<std::string>("qtype", "per_tensor");
    quantize.set_attr<int64_t>("axis", 0);

    test::vector<float> src {-1.0, 0.0, 1.0, 2.0};
    test::vector<uint8_t> dst(src.size(), 0);

    // int8 = f32 / scales + zero_points
    test::vector<uint8_t> ref_dst {0, 10, 20, 30};

    // prepare input/output logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(1, {1, 2, 2}, impl::data_type::u8);

    auto &op_factory = get_dnnl_kernel_registry();
    auto op = op_factory.create_kernel(quantize);

    op->compile(&quantize, &engine, {src_lt}, {dst_lt});

    impl::stream_t &stream = get_stream();
    impl::tensor_t src_ts(src_lt, &engine, src.data());
    impl::tensor_t dst_ts(dst_lt, &engine, dst.data());

    op->execute(&quantize, &stream, {src_ts}, {dst_ts});
    stream.wait();
    for (size_t i = 0; i < src.size(); ++i) {
        ASSERT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, quantize_per_tensor_any) {
    impl::engine_t &engine = get_engine();
    if (engine.kind() == impl::engine_kind::gpu) return;

    impl::op_t quantize(impl::op_kind::Quantize);
    quantize.set_attr<std::vector<float>>("scales", {0.1f});
    quantize.set_attr<std::vector<int64_t>>("zps", {10});
    quantize.set_attr<std::string>("qtype", "per_tensor");
    quantize.set_attr<int64_t>("axis", 0);

    test::vector<float> src {-1.0, 0.0, 1.0, 2.0};
    test::vector<uint8_t> dst(src.size(), 0);

    // int8 = f32 / scales + zero_points
    test::vector<uint8_t> ref_dst {0, 10, 20, 30};

    // prepare input/output logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {1, 2, 2}, impl::data_type::u8, impl::layout_type::any);

    auto &op_factory = get_dnnl_kernel_registry();
    auto op = op_factory.create_kernel(quantize);

    op->compile(&quantize, &engine, {src_lt}, {dst_lt});

    impl::stream_t &stream = get_stream();
    impl::tensor_t src_ts(src_lt, &engine, src.data());
    impl::tensor_t dst_ts(dst_lt, &engine, dst.data());

    op->execute(&quantize, &stream, {src_ts}, {dst_ts});
    stream.wait();
    for (size_t i = 0; i < src.size(); ++i) {
        ASSERT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, quantize_per_channel_symmetric) {
    impl::engine_t &engine = get_engine();
    if (engine.kind() == impl::engine_kind::gpu) return;

    impl::op_t quantize(impl::op_kind::Quantize);
    quantize.set_attr<std::vector<float>>("scales", {0.1f, 0.2f});
    quantize.set_attr<std::vector<int64_t>>("zps", {0, 0});
    quantize.set_attr<std::string>("qtype", "per_channel");
    quantize.set_attr<int64_t>("axis", 1);

    test::vector<float> src {-1.0, 0.0, 1.0, 2.0};
    test::vector<int8_t> dst(src.size(), 0);

    // int8 = f32 / scales
    test::vector<int8_t> ref_dst {-10, 0, 5, 10};

    // prepare input/output logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(1, {1, 2, 2}, impl::data_type::s8);

    auto &op_factory = get_dnnl_kernel_registry();
    auto op = op_factory.create_kernel(quantize);

    op->compile(&quantize, &engine, {src_lt}, {dst_lt});

    impl::stream_t &stream = get_stream();
    impl::tensor_t src_ts(src_lt, &engine, src.data());
    impl::tensor_t dst_ts(dst_lt, &engine, dst.data());

    op->execute(&quantize, &stream, {src_ts}, {dst_ts});
    stream.wait();
    for (size_t i = 0; i < src.size(); ++i) {
        ASSERT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, dequantize_per_tensor) {
    impl::engine_t &engine = get_engine();
    if (engine.kind() == impl::engine_kind::gpu) return;

    impl::op_t dequantize(impl::op_kind::Dequantize);
    dequantize.set_attr<std::vector<float>>("scales", {0.1f});
    dequantize.set_attr<std::vector<int64_t>>("zps", {10});
    dequantize.set_attr<std::string>("qtype", "per_tensor");
    dequantize.set_attr<int64_t>("axis", 0);

    test::vector<uint8_t> src {0, 10, 20, 30};
    test::vector<float> dst(src.size(), 0);

    // f32 = scales * (int8 - zero_points)
    test::vector<float> ref_dst {-1.0, 0.0, 1.0, 2.0};

    // prepare input/output logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 2}, impl::data_type::u8);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(1, {1, 2, 2}, impl::data_type::f32);

    auto &op_factory = get_dnnl_kernel_registry();
    auto op = op_factory.create_kernel(dequantize);

    op->compile(&dequantize, &engine, {src_lt}, {dst_lt});

    impl::stream_t &stream = get_stream();
    impl::tensor_t src_ts(src_lt, &engine, src.data());
    impl::tensor_t dst_ts(dst_lt, &engine, dst.data());

    op->execute(&dequantize, &stream, {src_ts}, {dst_ts});
    stream.wait();
    for (size_t i = 0; i < src.size(); ++i) {
        ASSERT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, dequantize_per_tensor_any) {
    impl::engine_t &engine = get_engine();
    if (engine.kind() == impl::engine_kind::gpu) return;

    impl::op_t dequantize(impl::op_kind::Dequantize);
    dequantize.set_attr<std::vector<float>>("scales", {0.1f});
    dequantize.set_attr<std::vector<int64_t>>("zps", {10});
    dequantize.set_attr<std::string>("qtype", "per_tensor");
    dequantize.set_attr<int64_t>("axis", 0);

    test::vector<uint8_t> src {0, 10, 20, 30};
    test::vector<float> dst(src.size(), 0);

    // f32 = scales * (int8 - zero_points)
    test::vector<float> ref_dst {-1.0, 0.0, 1.0, 2.0};

    // prepare input/output logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 2}, impl::data_type::u8);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {1, 2, 2}, impl::data_type::f32, impl::layout_type::any);

    auto &op_factory = get_dnnl_kernel_registry();
    auto op = op_factory.create_kernel(dequantize);

    op->compile(&dequantize, &engine, {src_lt}, {dst_lt});

    impl::stream_t &stream = get_stream();
    impl::tensor_t src_ts(src_lt, &engine, src.data());
    impl::tensor_t dst_ts(dst_lt, &engine, dst.data());

    op->execute(&dequantize, &stream, {src_ts}, {dst_ts});
    stream.wait();
    for (size_t i = 0; i < src.size(); ++i) {
        ASSERT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, dequantize_per_channel_symmetric) {
    impl::engine_t &engine = get_engine();
    if (engine.kind() == impl::engine_kind::gpu) return;

    impl::op_t dequantize(impl::op_kind::Dequantize);
    dequantize.set_attr<std::vector<float>>("scales", {0.1f, 0.2f});
    dequantize.set_attr<std::vector<int64_t>>("zps", {0, 0});
    dequantize.set_attr<std::string>("qtype", "per_channel");
    dequantize.set_attr<int64_t>("axis", 1);

    test::vector<int8_t> src {0, 10, 20, 30};
    test::vector<float> dst(src.size(), 0);

    // f32 = scales * int8
    test::vector<float> ref_dst {0.0, 1, 4, 6};

    // prepare input/output logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 2}, impl::data_type::s8);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(1, {1, 2, 2}, impl::data_type::f32);

    auto &op_factory = get_dnnl_kernel_registry();
    auto op = op_factory.create_kernel(dequantize);

    op->compile(&dequantize, &engine, {src_lt}, {dst_lt});

    impl::stream_t &stream = get_stream();
    impl::tensor_t src_ts(src_lt, &engine, src.data());
    impl::tensor_t dst_ts(dst_lt, &engine, dst.data());

    op->execute(&dequantize, &stream, {src_ts}, {dst_ts});
    stream.wait();
    for (size_t i = 0; i < src.size(); ++i) {
        ASSERT_EQ(dst[i], ref_dst[i]);
    }
}

template <typename T>
static bool allclose(const test::vector<T> &a, const test::vector<T> &b,
        float rtol, float atol) {
    if (a.size() != b.size()) return false;
    bool flag = true;
    for (size_t i = 0; i < a.size(); i++) {
        if (std::abs(static_cast<float>(a[i]) - static_cast<float>(b[i]))
                > (atol + rtol * std::abs(static_cast<float>(b[i])))) {
            flag = false;
            break;
        }
    }
    return flag;
}

// FIXME(qun) If the atol and rtol in the following cases are too small, then
// these cases may can't pass when using AVX2 or AVX512, because of the issue
// described in: https://oneapi-src.github.io/oneDNN/dev_guide_int8_computations.html

#define for_ for
#define SET_Q_DQ_DATA_ATTR(q_dq_data) \
    q_dq_data.set_attr<std::string>("qtype", "per_tensor"); \
    q_dq_data.set_attr<std::vector<int64_t>>("zps", {zp_src}); \
    q_dq_data.set_attr<std::vector<float>>("scales", {scale_src}); \
    q_dq_data.set_attr<int64_t>("axis", 0);

#define SET_Q_DQ_WEIGHT_ATTR(q_dq_weight) \
    q_dq_weight.set_attr<std::string>("qtype", wei_qtype); \
    q_dq_weight.set_attr<std::vector<int64_t>>("zps", zp_wei); \
    q_dq_weight.set_attr<std::vector<float>>("scales", scale_wei); \
    q_dq_weight.set_attr<int64_t>("axis", 0);

#define SET_CONV_ATTR(conv, nd) \
    conv.set_attr<dims>("strides", dims(nd, 1)); \
    conv.set_attr<dims>("dilations", dims(nd, 1)); \
    conv.set_attr<dims>("pads_begin", dims(nd, 0)); \
    conv.set_attr<dims>("pads_end", dims(nd, 0)); \
    conv.set_attr<int64_t>("groups", g); \
    conv.set_attr<std::string>("data_format", "NCX"); \
    conv.set_attr<std::string>("filter_format", "OIX");

#define SET_Q_DQ_OUT_ATTR(q_dq_out) \
    q_dq_out.set_attr<std::string>("qtype", "per_tensor"); \
    q_dq_out.set_attr<std::vector<int64_t>>("zps", {zp_out}); \
    q_dq_out.set_attr<std::vector<float>>("scales", {scale_out}); \
    q_dq_out.set_attr<int64_t>("axis", 0);

TEST(int8_subgraph_mode, int8_conv1d_conv2d_conv3d) {
    using dims = impl::dnnl_impl::dims;

    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    std::vector<size_t> nds = {1, 2, 3};
    std::vector<int64_t> groups = {1, 4};
    std::vector<bool> with_biases = {true, false};
    std::vector<std::string> weight_qtypes = {"per_tensor", "per_channel"};
    std::vector<std::string> src_qtypes = {"symmetric", "asymmetric"};

    if (engine.kind() == impl::engine_kind::gpu) return;
    static auto isa = dnnl_get_effective_cpu_isa();

    for_(const auto &nd : nds)
    for_(const auto &g : groups)
    for_(const auto &with_bias : with_biases)
    for_(const auto &src_qtype : src_qtypes)
    for (const auto &wei_qtype : weight_qtypes) {
        if (isa < dnnl_cpu_isa_avx512_core_vnni && src_qtype == "asymmetric")
            continue;

        // prepare data
        int64_t in_channel = 8, out_channel = 8;
        int64_t kernel_size = 3;
        std::vector<int64_t> src_shape = nd == 1
                ? std::vector<int64_t> {1, in_channel, 12}
                : nd == 2 ? std::vector<int64_t> {1, in_channel, 12, 12}
                          : std::vector<int64_t> {1, in_channel, 12, 12, 12};
        std::vector<int64_t> weight_shape = nd == 1
                ? std::vector<int64_t> {out_channel, in_channel / g,
                        kernel_size}
                : nd == 2 ? std::vector<int64_t> {out_channel, in_channel / g,
                          kernel_size, kernel_size}
                          : std::vector<int64_t> {out_channel, in_channel / g,
                                  kernel_size, kernel_size, kernel_size};
        std::vector<int64_t> bias_shape {out_channel};
        std::vector<int64_t> dst_shape = nd == 1
                ? std::vector<int64_t> {1, out_channel, 10}
                : nd == 2 ? std::vector<int64_t> {1, out_channel, 10, 10}
                          : std::vector<int64_t> {1, out_channel, 10, 10, 10};

        test::vector<uint8_t> src_u8_data(product(src_shape));
        test::vector<int8_t> weight_s8_data(product(weight_shape));
        size_t bias_size = with_bias ? product(bias_shape) : 0;
        test::vector<float> bias_data(bias_size);
        test::vector<int8_t> case1_out_data(product(dst_shape));
        test::vector<int8_t> case2_out_data(product(dst_shape));

        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> u8_distribution(0.0f, 255.0f);
        std::uniform_real_distribution<float> s8_distribution(-127.0f, 128.0f);
        std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
        std::generate(src_u8_data.begin(), src_u8_data.end(), [&]() {
            return static_cast<uint8_t>(u8_distribution(generator));
        });
        std::generate(weight_s8_data.begin(), weight_s8_data.end(), [&]() {
            return static_cast<int8_t>(s8_distribution(generator));
        });
        if (with_bias) {
            std::generate(bias_data.begin(), bias_data.end(),
                    [&]() { return f32_distribution(generator); });
        }

        float scale_src = 1 / 255.f; // map to 0~255
        float scale_out = 1;
        int64_t zp_src = src_qtype == "symmetric" ? 0 : 128;
        int64_t zp_out = 78;

        size_t scale_size = wei_qtype == "per_tensor" ? 1 : out_channel;
        std::vector<float> scale_wei(scale_size, 1 / 127.f);
        std::vector<int64_t> zp_wei(scale_size, 0);

        impl::op_t dqdata_node(1, impl::op_kind::Dequantize, "dqdata_node");
        SET_Q_DQ_DATA_ATTR(dqdata_node)

        impl::op_t dqweight_node(3, impl::op_kind::Dequantize, "dqweight_node");
        SET_Q_DQ_WEIGHT_ATTR(dqweight_node)

        impl::op_t conv_node(4, impl::op_kind::Convolution, "conv_node");
        SET_CONV_ATTR(conv_node, nd)

        impl::op_t qout_node(5, impl::op_kind::Quantize, "qout_node");
        SET_Q_DQ_OUT_ATTR(qout_node)

        impl::logical_tensor_t src_u8
                = utils::logical_tensor_init(1, src_shape, impl::data_type::u8);
        impl::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                2, src_shape, impl::data_type::f32);
        impl::logical_tensor_t weight_s8 = utils::logical_tensor_init(
                4, weight_shape, impl::data_type::s8);
        impl::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
                5, weight_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_f32 = utils::logical_tensor_init(
                7, dst_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_s8
                = utils::logical_tensor_init(8, dst_shape, impl::data_type::s8);
        impl::logical_tensor_t bias_f32;
        if (with_bias) {
            bias_f32 = utils::logical_tensor_init(
                    6, bias_shape, impl::data_type::f32);
        }

        dqdata_node.add_input(src_u8);
        dqdata_node.add_output(src_f32_dq);

        dqweight_node.add_input(weight_s8);
        dqweight_node.add_output(weight_f32_dq);

        conv_node.add_input(src_f32_dq);
        conv_node.add_input(weight_f32_dq);
        if (with_bias) conv_node.add_input(bias_f32);
        conv_node.add_output(dst_f32);

        qout_node.add_input(dst_f32);
        qout_node.add_output(dst_s8);

        impl::graph_t g;
        g.add_op(&dqdata_node);
        g.add_op(&dqweight_node);
        g.add_op(&conv_node);
        g.add_op(&qout_node);
        g.build_graph();

        impl::tensor_t src_u8_ts(src_u8, &engine, src_u8_data.data());
        impl::tensor_t weight_s8_ts(weight_s8, &engine, weight_s8_data.data());
        impl::tensor_t bias_f32_ts;
        if (with_bias) {
            bias_f32_ts = impl::tensor_t(bias_f32, &engine, bias_data.data());
        }
        impl::tensor_t dst_s8_ts(dst_s8, &engine, case1_out_data.data());
        impl::tensor_t dst_s8_case2_ts(dst_s8, &engine, case2_out_data.data());

        // -------------------------case 1----------------------------------
        ASSERT_EQ(run_graph(g, {src_u8_ts, weight_s8_ts, bias_f32_ts},
                          {dst_s8_ts}, engine, strm),
                impl::status::success);

        // -------------------------case 2----------------------------------
        impl::pass::pass_base_ptr apass = with_bias
                ? get_pass("int8_conv_bias_fusion")
                : get_pass("int8_conv_fusion");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1);
        auto part = g.get_partitions()[0];

        // compile
        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> lt_ins;
        if (with_bias)
            lt_ins = {&src_u8, &weight_s8, &bias_f32};
        else
            lt_ins = {&src_u8, &weight_s8};
        std::vector<const impl::logical_tensor_t *> lt_outs {&dst_s8};

        p.compile(&cp, lt_ins, lt_outs, &engine);

        if (with_bias)
            cp.execute(&strm, {src_u8_ts, weight_s8_ts, bias_f32_ts},
                    {dst_s8_case2_ts});
        else
            cp.execute(&strm, {src_u8_ts, weight_s8_ts}, {dst_s8_case2_ts});
        strm.wait();

        if (isa < dnnl_cpu_isa_avx512_core_vnni)
            ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.1f,
                    /*atol*/ 1.f));
        else
            ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.01f,
                    /*atol*/ 1.f));
    }
}

TEST(int8_subgraph_mode, int8_convtranspose_1d_2d_3d) {
    using dims = impl::dnnl_impl::dims;

    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    std::vector<size_t> nds = {1, 2, 3};
    std::vector<int64_t> groups = {1, 4};
    std::vector<bool> with_biases = {true, false};
    std::vector<std::string> weight_qtypes = {"per_tensor", "per_channel"};
    std::vector<std::string> src_qtypes = {"symmetric", "asymmetric"};

    if (engine.kind() == impl::engine_kind::gpu) return;
    static auto isa = dnnl_get_effective_cpu_isa();

    for_(const auto &nd : nds)
    for_(const auto &g : groups)
    for_(const auto &with_bias : with_biases)
    for_(const auto &src_qtype : src_qtypes)
    for (const auto &wei_qtype : weight_qtypes) {
        if (isa < dnnl_cpu_isa_avx512_core_vnni && src_qtype == "asymmetric")
            continue;

        // prepare data
        int64_t in_channel = 8, out_channel = 8;
        int64_t kernel_size = 3;
        std::vector<int64_t> src_shape = nd == 1
                ? std::vector<int64_t> {1, in_channel, 12}
                : nd == 2 ? std::vector<int64_t> {1, in_channel, 12, 12}
                          : std::vector<int64_t> {1, in_channel, 12, 12, 12};
        std::vector<int64_t> weight_shape = nd == 1
                ? std::vector<int64_t> {out_channel / g, in_channel,
                        kernel_size}
                : nd == 2 ? std::vector<int64_t> {out_channel / g, in_channel,
                          kernel_size, kernel_size}
                          : std::vector<int64_t> {out_channel / g, in_channel,
                                  kernel_size, kernel_size, kernel_size};
        std::vector<int64_t> bias_shape {out_channel};
        std::vector<int64_t> dst_shape = nd == 1
                ? std::vector<int64_t> {1, out_channel, 14}
                : nd == 2 ? std::vector<int64_t> {1, out_channel, 14, 14}
                          : std::vector<int64_t> {1, out_channel, 14, 14, 14};

        test::vector<uint8_t> src_u8_data(product(src_shape));
        test::vector<int8_t> weight_s8_data(product(weight_shape));
        size_t bias_size = with_bias ? product(bias_shape) : 0;
        test::vector<float> bias_data(bias_size);
        test::vector<int8_t> case1_out_data(product(dst_shape));
        test::vector<int8_t> case2_out_data(product(dst_shape));

        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> u8_distribution(0.0f, 255.0f);
        std::uniform_real_distribution<float> s8_distribution(-127.0f, 128.0f);
        std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
        std::generate(src_u8_data.begin(), src_u8_data.end(), [&]() {
            return static_cast<uint8_t>(u8_distribution(generator));
        });
        std::generate(weight_s8_data.begin(), weight_s8_data.end(), [&]() {
            return static_cast<int8_t>(s8_distribution(generator));
        });
        if (with_bias) {
            std::generate(bias_data.begin(), bias_data.end(),
                    [&]() { return f32_distribution(generator); });
        }

        float scale_src = 1 / 255.f; // map to 0~255
        float scale_out = 1;
        int64_t zp_src = src_qtype == "symmetric" ? 0 : 128;
        int64_t zp_out = 78;

        size_t scale_size = wei_qtype == "per_tensor" ? 1 : (out_channel / g);
        std::vector<float> scale_wei(scale_size, 1 / 127.f);
        std::vector<int64_t> zp_wei(scale_size, 0);

        impl::op_t dqdata_node(1, impl::op_kind::Dequantize, "dqdata_node");
        SET_Q_DQ_DATA_ATTR(dqdata_node)

        impl::op_t dqweight_node(3, impl::op_kind::Dequantize, "dqweight_node");
        SET_Q_DQ_WEIGHT_ATTR(dqweight_node)

        impl::op_t convtranspose_node(
                4, impl::op_kind::ConvTranspose, "convtranspose_node");
        SET_CONV_ATTR(convtranspose_node, nd)

        impl::op_t qout_node(5, impl::op_kind::Quantize, "qout_node");
        SET_Q_DQ_OUT_ATTR(qout_node)

        impl::logical_tensor_t src_u8
                = utils::logical_tensor_init(1, src_shape, impl::data_type::u8);
        impl::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                2, src_shape, impl::data_type::f32);
        impl::logical_tensor_t weight_s8 = utils::logical_tensor_init(
                4, weight_shape, impl::data_type::s8);
        impl::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
                5, weight_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_f32 = utils::logical_tensor_init(
                7, dst_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_s8
                = utils::logical_tensor_init(8, dst_shape, impl::data_type::s8);
        impl::logical_tensor_t bias_f32;
        if (with_bias) {
            bias_f32 = utils::logical_tensor_init(
                    6, bias_shape, impl::data_type::f32);
        }

        dqdata_node.add_input(src_u8);
        dqdata_node.add_output(src_f32_dq);

        dqweight_node.add_input(weight_s8);
        dqweight_node.add_output(weight_f32_dq);

        convtranspose_node.add_input(src_f32_dq);
        convtranspose_node.add_input(weight_f32_dq);
        if (with_bias) convtranspose_node.add_input(bias_f32);
        convtranspose_node.add_output(dst_f32);

        qout_node.add_input(dst_f32);
        qout_node.add_output(dst_s8);

        impl::graph_t g;
        g.add_op(&dqdata_node);
        g.add_op(&dqweight_node);
        g.add_op(&convtranspose_node);
        g.add_op(&qout_node);
        g.build_graph();

        impl::tensor_t src_u8_ts(src_u8, &engine, src_u8_data.data());
        impl::tensor_t weight_s8_ts(weight_s8, &engine, weight_s8_data.data());
        impl::tensor_t bias_f32_ts;
        if (with_bias) {
            bias_f32_ts = impl::tensor_t(bias_f32, &engine, bias_data.data());
        }
        impl::tensor_t dst_s8_ts(dst_s8, &engine, case1_out_data.data());
        impl::tensor_t dst_s8_case2_ts(dst_s8, &engine, case2_out_data.data());

        // -------------------------case 1----------------------------------
        ASSERT_EQ(run_graph(g, {src_u8_ts, weight_s8_ts, bias_f32_ts},
                          {dst_s8_ts}, engine, strm),
                impl::status::success);

        // -------------------------case 2----------------------------------
        impl::pass::pass_base_ptr apass = with_bias
                ? get_pass("int8_convtranspose_bias_fusion")
                : get_pass("int8_convtranspose_fusion");
        ASSERT_TRUE(apass != nullptr);
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1);
        auto part = g.get_partitions()[0];

        // compile
        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> lt_ins;
        if (with_bias)
            lt_ins = {&src_u8, &weight_s8, &bias_f32};
        else
            lt_ins = {&src_u8, &weight_s8};
        std::vector<const impl::logical_tensor_t *> lt_outs {&dst_s8};

        p.compile(&cp, lt_ins, lt_outs, &engine);

        if (with_bias)
            cp.execute(&strm, {src_u8_ts, weight_s8_ts, bias_f32_ts},
                    {dst_s8_case2_ts});
        else
            cp.execute(&strm, {src_u8_ts, weight_s8_ts}, {dst_s8_case2_ts});
        strm.wait();

        if (isa < dnnl_cpu_isa_avx512_core_vnni)
            ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.1f,
                    /*atol*/ 1.f));
        else
            ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.01f,
                    /*atol*/ 1.f));
    }
}

TEST(int8_subgraph_mode, int8_conv2d_relu) {
    using dims = impl::dnnl_impl::dims;

    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    std::vector<int64_t> groups = {1, 4};
    std::vector<bool> with_biases = {true, false};
    std::vector<std::string> weight_qtypes = {"per_tensor", "per_channel"};

    if (engine.kind() == impl::engine_kind::gpu) return;
    static auto isa = dnnl_get_effective_cpu_isa();

    for_(const auto &g : groups)
    for_(const auto &with_bias : with_biases)
    for (const auto &wei_qtype : weight_qtypes) {
        // prepare data
        int64_t in_channel = 8, out_channel = 8;
        int64_t kernel_size = 3;
        std::vector<int64_t> src_shape {1, in_channel, 112, 112};
        std::vector<int64_t> weight_shape {
                out_channel, in_channel / g, kernel_size, kernel_size};
        std::vector<int64_t> bias_shape {out_channel};
        std::vector<int64_t> dst_shape {1, out_channel, 110, 110};

        test::vector<uint8_t> src_u8_data(product(src_shape));
        test::vector<int8_t> weight_s8_data(product(weight_shape));
        size_t bias_size = with_bias ? product(bias_shape) : 0;
        test::vector<float> bias_data(bias_size);
        test::vector<int8_t> case1_out_data(product(dst_shape));
        test::vector<int8_t> case2_out_data(product(dst_shape));

        // random generate src, weight and bias data random seed = 7
        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> u8_distribution(0.0f, 255.0f);
        std::uniform_real_distribution<float> s8_distribution(-127.0f, 128.0f);
        std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
        std::generate(src_u8_data.begin(), src_u8_data.end(), [&]() {
            return static_cast<uint8_t>(u8_distribution(generator));
        });
        std::generate(weight_s8_data.begin(), weight_s8_data.end(), [&]() {
            return static_cast<int8_t>(s8_distribution(generator));
        });
        if (with_bias) {
            std::generate(bias_data.begin(), bias_data.end(),
                    [&]() { return f32_distribution(generator); });
        }

        float scale_src = 1 / 255.f; // map to 0~255
        float scale_out = 1;
        int64_t zp_src = 0;
        int64_t zp_out = 78;

        size_t scale_size = wei_qtype == "per_tensor" ? 1 : out_channel;
        std::vector<float> scale_wei(scale_size, 1 / 127.f);
        std::vector<int64_t> zp_wei(scale_size, 0);

        impl::op_t dqdata_node(1, impl::op_kind::Dequantize, "dqdata_node");
        SET_Q_DQ_DATA_ATTR(dqdata_node)

        impl::op_t dqweight_node(3, impl::op_kind::Dequantize, "dqweight_node");
        SET_Q_DQ_WEIGHT_ATTR(dqweight_node)

        impl::op_t conv_node(4, impl::op_kind::Convolution, "conv_node");
        SET_CONV_ATTR(conv_node, 2)

        impl::op_t relu_node(5, impl::op_kind::ReLU, "relu_node");

        impl::op_t qout_node(6, impl::op_kind::Quantize, "qout_node");
        SET_Q_DQ_OUT_ATTR(qout_node)

        // prepare logical tensor
        impl::logical_tensor_t src_u8
                = utils::logical_tensor_init(1, src_shape, impl::data_type::u8);
        impl::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                2, src_shape, impl::data_type::f32);
        impl::logical_tensor_t weight_s8 = utils::logical_tensor_init(
                4, weight_shape, impl::data_type::s8);
        impl::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
                5, weight_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_f32 = utils::logical_tensor_init(
                7, dst_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_relu_f32 = utils::logical_tensor_init(
                8, dst_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_s8
                = utils::logical_tensor_init(9, dst_shape, impl::data_type::s8);
        impl::logical_tensor_t bias_f32;
        if (with_bias) {
            bias_f32 = utils::logical_tensor_init(
                    6, bias_shape, impl::data_type::f32);
        }

        dqdata_node.add_input(src_u8);
        dqdata_node.add_output(src_f32_dq);

        dqweight_node.add_input(weight_s8);
        dqweight_node.add_output(weight_f32_dq);

        conv_node.add_input(src_f32_dq);
        conv_node.add_input(weight_f32_dq);
        if (with_bias) conv_node.add_input(bias_f32);
        conv_node.add_output(dst_f32);

        relu_node.add_input(dst_f32);
        relu_node.add_output(dst_relu_f32);

        qout_node.add_input(dst_relu_f32);
        qout_node.add_output(dst_s8);

        impl::graph_t g;
        g.add_op(&dqdata_node);
        g.add_op(&dqweight_node);
        g.add_op(&conv_node);
        g.add_op(&relu_node);
        g.add_op(&qout_node);
        g.build_graph();

        impl::tensor_t src_u8_ts(src_u8, &engine, src_u8_data.data());
        impl::tensor_t weight_s8_ts(weight_s8, &engine, weight_s8_data.data());
        impl::tensor_t bias_f32_ts;
        if (with_bias) {
            bias_f32_ts = impl::tensor_t(bias_f32, &engine, bias_data.data());
        }
        impl::tensor_t dst_s8_ts(dst_s8, &engine, case1_out_data.data());
        impl::tensor_t dst_s8_case2_ts(dst_s8, &engine, case2_out_data.data());

        // -------------------------case 1----------------------------------
        ASSERT_EQ(run_graph(g, {src_u8_ts, weight_s8_ts, bias_f32_ts},
                          {dst_s8_ts}, engine, strm),
                impl::status::success);

        // -------------------------case 2----------------------------------
        impl::pass::pass_base_ptr apass = with_bias
                ? get_pass("int8_conv_bias_relu_fusion")
                : get_pass("int8_conv_relu_fusion");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1);
        auto part = g.get_partitions()[0];

        // compile
        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> lt_ins;
        if (with_bias)
            lt_ins = {&src_u8, &weight_s8, &bias_f32};
        else
            lt_ins = {&src_u8, &weight_s8};
        std::vector<const impl::logical_tensor_t *> lt_outs {&dst_s8};

        p.compile(&cp, lt_ins, lt_outs, &engine);

        if (with_bias)
            cp.execute(&strm, {src_u8_ts, weight_s8_ts, bias_f32_ts},
                    {dst_s8_case2_ts});
        else
            cp.execute(&strm, {src_u8_ts, weight_s8_ts}, {dst_s8_case2_ts});
        strm.wait();

        if (isa < dnnl_cpu_isa_avx512_core_vnni)
            ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.1f,
                    /*atol*/ 1.f));
        else
            ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.01f,
                    /*atol*/ 1.f));
    }
}

TEST(int8_subgraph_mode, int8_conv2d_sum_relu) {
    using dims = impl::dnnl_impl::dims;

    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    if (engine.kind() == impl::engine_kind::gpu) return;
    static auto isa = dnnl_get_effective_cpu_isa();

    std::vector<int64_t> groups = {1, 4};
    std::vector<bool> with_biases = {true, false};
    // swap add's two inputs
    std::vector<bool> swaps = {true, false};
    std::vector<std::string> weight_qtypes = {"per_tensor", "per_channel"};
    std::vector<std::string> other_qtypes = {"symmetric", "asymmetric"};

    for_(const auto &g : groups)
    for_(const auto &with_bias : with_biases)
    for_(const auto &swap : swaps)
    for_(const auto &other_qtype : other_qtypes)
    for (const auto &wei_qtype : weight_qtypes) {
        // prepare data
        int64_t in_channel = 8, out_channel = 8;
        int64_t kernel_size = 3;
        std::vector<int64_t> src_shape {1, in_channel, 112, 112};
        std::vector<int64_t> weight_shape {
                out_channel, in_channel / g, kernel_size, kernel_size};
        std::vector<int64_t> bias_shape {out_channel};
        std::vector<int64_t> dst_shape {1, out_channel, 110, 110};

        test::vector<uint8_t> src_u8_data(product(src_shape));
        test::vector<int8_t> weight_s8_data(product(weight_shape));
        test::vector<int8_t> other_s8_data(product(dst_shape));
        size_t bias_size = with_bias ? product(bias_shape) : 0;
        test::vector<float> bias_data(bias_size);
        test::vector<int8_t> case1_out_data(product(dst_shape));
        test::vector<int8_t> case2_out_data(product(dst_shape));

        // random generate src, weight and bias data random seed = 7
        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> u8_distribution(0.0f, 255.0f);
        std::uniform_real_distribution<float> s8_distribution(-127.0f, 128.0f);
        std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
        std::generate(src_u8_data.begin(), src_u8_data.end(), [&]() {
            return static_cast<uint8_t>(u8_distribution(generator));
        });
        std::generate(weight_s8_data.begin(), weight_s8_data.end(), [&]() {
            return static_cast<int8_t>(s8_distribution(generator));
        });
        std::generate(other_s8_data.begin(), other_s8_data.end(), [&]() {
            return static_cast<int8_t>(s8_distribution(generator));
        });
        if (with_bias) {
            std::generate(bias_data.begin(), bias_data.end(),
                    [&]() { return f32_distribution(generator); });
        }

        float scale_src = 1 / 255.f; // map to 0~255
        float scale_other = 1 / 127.f;
        float scale_out = 1;
        int64_t zp_src = 0;
        int64_t zp_other = other_qtype == "symmetric" ? 0 : 128;
        int64_t zp_out = 78;

        size_t scale_size = wei_qtype == "per_tensor" ? 1 : out_channel;

        std::vector<float> scale_wei(scale_size, 1 / 127.f);
        std::vector<int64_t> zp_wei(scale_size, 0);

        impl::op_t dqdata_node(1, impl::op_kind::Dequantize, "dqdata_node");
        SET_Q_DQ_DATA_ATTR(dqdata_node)

        impl::op_t dqweight_node(3, impl::op_kind::Dequantize, "dqweight_node");
        SET_Q_DQ_WEIGHT_ATTR(dqweight_node)

        impl::op_t conv_node(4, impl::op_kind::Convolution, "conv_node");
        SET_CONV_ATTR(conv_node, 2)

        impl::op_t relu_node(5, impl::op_kind::ReLU, "relu_node");

        impl::op_t qout_node(6, impl::op_kind::Quantize, "qout_node");
        SET_Q_DQ_OUT_ATTR(qout_node)

        impl::op_t dqother_node(8, impl::op_kind::Dequantize, "dqother_node");
        dqother_node.set_attr<std::string>("qtype", "per_tensor");
        dqother_node.set_attr<std::vector<int64_t>>("zps", {zp_other});
        dqother_node.set_attr<std::vector<float>>("scales", {scale_other});
        dqother_node.set_attr<int64_t>("axis", 0);

        impl::op_t add_node(9, impl::op_kind::Add, "add_node");

        // prepare logical tensor
        auto src_u8 = utils::logical_tensor_init(1, impl::data_type::u8);
        auto src_f32_dq = utils::logical_tensor_init(2, impl::data_type::f32);
        auto weight_s8 = utils::logical_tensor_init(4, impl::data_type::s8);
        auto weight_f32_dq
                = utils::logical_tensor_init(5, impl::data_type::f32);
        auto dst_f32 = utils::logical_tensor_init(7, impl::data_type::f32);
        auto dst_relu_f32 = utils::logical_tensor_init(8, impl::data_type::f32);
        auto dst_s8 = utils::logical_tensor_init(9, impl::data_type::s8);
        auto other_s8 = utils::logical_tensor_init(11, impl::data_type::s8);
        auto other_f32_dq
                = utils::logical_tensor_init(12, impl::data_type::f32);
        auto dst_add_f32 = utils::logical_tensor_init(13, impl::data_type::f32);
        impl::logical_tensor_t bias_f32;
        if (with_bias) {
            bias_f32 = utils::logical_tensor_init(6, impl::data_type::f32);
        }

        dqdata_node.add_input(src_u8);
        dqdata_node.add_output(src_f32_dq);

        dqweight_node.add_input(weight_s8);
        dqweight_node.add_output(weight_f32_dq);

        conv_node.add_input(src_f32_dq);
        conv_node.add_input(weight_f32_dq);
        if (with_bias) conv_node.add_input(bias_f32);
        conv_node.add_output(dst_f32);

        dqother_node.add_input(other_s8);
        dqother_node.add_output(other_f32_dq);
        if (swap) {
            add_node.add_input(dst_f32);
            add_node.add_input(other_f32_dq);
        } else {
            add_node.add_input(other_f32_dq);
            add_node.add_input(dst_f32);
        }
        add_node.add_output(dst_add_f32);

        relu_node.add_input(dst_add_f32);
        relu_node.add_output(dst_relu_f32);

        qout_node.add_input(dst_relu_f32);
        qout_node.add_output(dst_s8);

        impl::graph_t g;
        g.add_op(&dqdata_node);
        g.add_op(&dqweight_node);
        g.add_op(&conv_node);
        g.add_op(&dqother_node);
        g.add_op(&add_node);
        g.add_op(&relu_node);
        g.add_op(&qout_node);
        g.build_graph();

        // prepare in/out with full shape
        src_u8 = utils::logical_tensor_init(1, src_shape, impl::data_type::u8);
        weight_s8 = utils::logical_tensor_init(
                4, weight_shape, impl::data_type::s8);
        other_s8 = utils::logical_tensor_init(
                11, dst_shape, impl::data_type::s8);
        if (with_bias) {
            bias_f32 = utils::logical_tensor_init(
                    6, bias_shape, impl::data_type::f32);
        }
        dst_s8 = utils::logical_tensor_init(9, dst_shape, impl::data_type::s8);

        impl::tensor_t src_u8_ts(src_u8, &engine, src_u8_data.data());
        impl::tensor_t weight_s8_ts(weight_s8, &engine, weight_s8_data.data());
        impl::tensor_t other_s8_ts(other_s8, &engine, other_s8_data.data());
        impl::tensor_t bias_f32_ts;
        if (with_bias) {
            bias_f32_ts = impl::tensor_t(bias_f32, &engine, bias_data.data());
        }
        impl::tensor_t dst_s8_ts(dst_s8, &engine, case1_out_data.data());
        impl::tensor_t dst_s8_case2_ts(dst_s8, &engine, case2_out_data.data());

        // -------------------------case 1----------------------------------
        ASSERT_EQ(run_graph(g,
                          {src_u8_ts, weight_s8_ts, bias_f32_ts, other_s8_ts},
                          {dst_s8_ts}, engine, strm),
                impl::status::success);

        // -------------------------case 2----------------------------------
        impl::pass::pass_base_ptr apass = with_bias
                ? get_pass("int8_conv_bias_add_relu_fusion")
                : get_pass("int8_conv_add_relu_fusion");

        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1);
        auto part = g.get_partitions()[0];

        // compile
        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> lt_ins;
        if (with_bias)
            lt_ins = {&src_u8, &weight_s8, &bias_f32, &other_s8};
        else
            lt_ins = {&src_u8, &weight_s8, &other_s8};
        std::vector<const impl::logical_tensor_t *> lt_outs {&dst_s8};

        p.compile(&cp, lt_ins, lt_outs, &engine);

        if (with_bias)
            cp.execute(&strm,
                    {src_u8_ts, weight_s8_ts, bias_f32_ts, other_s8_ts},
                    {dst_s8_case2_ts});
        else
            cp.execute(&strm, {src_u8_ts, weight_s8_ts, other_s8_ts},
                    {dst_s8_case2_ts});
        strm.wait();

        if (isa < dnnl_cpu_isa_avx512_core_vnni)
            ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.1f,
                    /*atol*/ 1.f));
        else
            ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.01f,
                    /*atol*/ 1.f));
    }
}

TEST(int8_subgraph_mode, int8_conv2d_sum_relu_NXC) {
    using dims = impl::dnnl_impl::dims;

    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    if (engine.kind() == impl::engine_kind::gpu) return;
    static auto isa = dnnl_get_effective_cpu_isa();

    std::vector<int64_t> groups = {1, 4};
    std::vector<bool> with_biases = {true, false};
    std::vector<std::string> weight_qtypes = {"per_tensor", "per_channel"};

    for_(const auto &g : groups)
    for_(const auto &with_bias : with_biases)
    for (const auto &wei_qtype : weight_qtypes) {
        // prepare fp32 data
        int64_t in_channel = 8, out_channel = 8;
        int64_t kernel_size = 3;
        std::vector<int64_t> src_shape {1, 12, 12, in_channel};
        std::vector<int64_t> weight_shape {
                kernel_size, kernel_size, in_channel / g, out_channel};
        std::vector<int64_t> bias_shape {out_channel};
        std::vector<int64_t> dst_shape {1, 10, 10, out_channel};

        test::vector<uint8_t> src_u8_data(product(src_shape));
        test::vector<int8_t> weight_s8_data(product(weight_shape));
        test::vector<int8_t> other_s8_data(product(dst_shape));
        size_t bias_size = with_bias ? product(bias_shape) : 0;
        test::vector<float> bias_data(bias_size);
        test::vector<int8_t> case1_out_data(product(dst_shape));
        test::vector<int8_t> case2_out_data(product(dst_shape));

        // random generate src, weight and bias data random seed = 7
        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> u8_distribution(0.0f, 255.0f);
        std::uniform_real_distribution<float> s8_distribution(-127.0f, 128.0f);
        std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
        std::generate(src_u8_data.begin(), src_u8_data.end(), [&]() {
            return static_cast<uint8_t>(u8_distribution(generator));
        });
        std::generate(weight_s8_data.begin(), weight_s8_data.end(), [&]() {
            return static_cast<int8_t>(s8_distribution(generator));
        });
        std::generate(other_s8_data.begin(), other_s8_data.end(), [&]() {
            return static_cast<int8_t>(s8_distribution(generator));
        });
        if (with_bias) {
            std::generate(bias_data.begin(), bias_data.end(),
                    [&]() { return f32_distribution(generator); });
        }

        float scale_src = 1 / 255.f; // map to 0~255
        float scale_other = 1 / 127.f;
        float scale_out = 1;
        int64_t zp_src = 0;
        int64_t zp_other = 0;
        int64_t zp_out = 78;

        size_t scale_size = wei_qtype == "per_tensor" ? 1 : out_channel;

        std::vector<float> scale_wei(scale_size, 1 / 127.f);
        std::vector<int64_t> zp_wei(scale_size, 0);

        impl::op_t dqdata_node(1, impl::op_kind::Dequantize, "dqdata_node");
        SET_Q_DQ_DATA_ATTR(dqdata_node)

        impl::op_t dqweight_node(3, impl::op_kind::Dequantize, "dqweight_node");
        SET_Q_DQ_WEIGHT_ATTR(dqweight_node)
        dqweight_node.set_attr<int64_t>(
                "axis", wei_qtype == "per_tensor" ? 0 : 3);

        impl::op_t conv_node(4, impl::op_kind::Convolution, "conv_node");
        SET_CONV_ATTR(conv_node, 2)
        conv_node.set_attr<std::string>("data_format", "NXC");
        conv_node.set_attr<std::string>("filter_format", "XIO");

        impl::op_t relu_node(5, impl::op_kind::ReLU, "relu_node");

        impl::op_t qout_node(6, impl::op_kind::Quantize, "qout_node");
        SET_Q_DQ_OUT_ATTR(qout_node)

        impl::op_t dqother_node(8, impl::op_kind::Dequantize, "dqother_node");
        dqother_node.set_attr<std::string>("qtype", "per_tensor");
        dqother_node.set_attr<std::vector<int64_t>>("zps", {zp_other});
        dqother_node.set_attr<std::vector<float>>("scales", {scale_other});
        dqother_node.set_attr<int64_t>("axis", 0);

        impl::op_t add_node(9, impl::op_kind::Add, "add_node");

        // prepare logical tensor
        auto src_u8 = utils::logical_tensor_init(1, impl::data_type::u8);
        auto src_f32_dq = utils::logical_tensor_init(2, impl::data_type::f32);
        auto weight_s8 = utils::logical_tensor_init(4, impl::data_type::s8);
        auto weight_f32_dq
                = utils::logical_tensor_init(5, impl::data_type::f32);
        auto dst_f32 = utils::logical_tensor_init(7, impl::data_type::f32);
        auto dst_relu_f32 = utils::logical_tensor_init(8, impl::data_type::f32);
        auto dst_s8 = utils::logical_tensor_init(9, impl::data_type::s8);
        auto other_s8 = utils::logical_tensor_init(11, impl::data_type::s8);
        auto other_f32_dq
                = utils::logical_tensor_init(12, impl::data_type::f32);
        auto dst_add_f32 = utils::logical_tensor_init(13, impl::data_type::f32);
        impl::logical_tensor_t bias_f32;
        if (with_bias) {
            bias_f32 = utils::logical_tensor_init(6, impl::data_type::f32);
        }

        // -------------------------case 2----------------------------------
        dqdata_node.add_input(src_u8);
        dqdata_node.add_output(src_f32_dq);

        dqweight_node.add_input(weight_s8);
        dqweight_node.add_output(weight_f32_dq);

        conv_node.add_input(src_f32_dq);
        conv_node.add_input(weight_f32_dq);
        if (with_bias) conv_node.add_input(bias_f32);
        conv_node.add_output(dst_f32);

        dqother_node.add_input(other_s8);
        dqother_node.add_output(other_f32_dq);

        add_node.add_input(dst_f32);
        add_node.add_input(other_f32_dq);
        add_node.add_output(dst_add_f32);

        relu_node.add_input(dst_add_f32);
        relu_node.add_output(dst_relu_f32);

        qout_node.add_input(dst_relu_f32);
        qout_node.add_output(dst_s8);

        impl::graph_t g;
        g.add_op(&dqdata_node);
        g.add_op(&dqweight_node);
        g.add_op(&conv_node);
        g.add_op(&dqother_node);
        g.add_op(&add_node);
        g.add_op(&relu_node);
        g.add_op(&qout_node);
        g.build_graph();

        // prepare in/out with full shape
        src_u8 = utils::logical_tensor_init(1, src_shape, impl::data_type::u8);
        weight_s8 = utils::logical_tensor_init(
                4, weight_shape, impl::data_type::s8);
        other_s8 = utils::logical_tensor_init(
                11, dst_shape, impl::data_type::s8);
        if (with_bias) {
            bias_f32 = utils::logical_tensor_init(
                    6, bias_shape, impl::data_type::f32);
        }
        dst_s8 = utils::logical_tensor_init(9, dst_shape, impl::data_type::s8);

        impl::tensor_t src_u8_ts(src_u8, &engine, src_u8_data.data());
        impl::tensor_t weight_s8_ts(weight_s8, &engine, weight_s8_data.data());
        impl::tensor_t other_s8_ts(other_s8, &engine, other_s8_data.data());
        impl::tensor_t bias_f32_ts;
        if (with_bias) {
            bias_f32_ts = impl::tensor_t(bias_f32, &engine, bias_data.data());
        }
        impl::tensor_t dst_s8_ts(dst_s8, &engine, case1_out_data.data());
        impl::tensor_t dst_s8_case2_ts(dst_s8, &engine, case2_out_data.data());

        // -------------------------case 1----------------------------------
        ASSERT_EQ(run_graph(g,
                          {src_u8_ts, weight_s8_ts, bias_f32_ts, other_s8_ts},
                          {dst_s8_ts}, engine, strm),
                impl::status::success);

        // -------------------------case 2----------------------------------
        impl::pass::pass_base_ptr apass = with_bias
                ? get_pass("int8_conv_bias_add_relu_fusion")
                : get_pass("int8_conv_add_relu_fusion");

        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1);
        auto part = g.get_partitions()[0];

        // compile
        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> lt_ins;
        if (with_bias)
            lt_ins = {&src_u8, &weight_s8, &bias_f32, &other_s8};
        else
            lt_ins = {&src_u8, &weight_s8, &other_s8};
        std::vector<const impl::logical_tensor_t *> lt_outs {&dst_s8};

        p.compile(&cp, lt_ins, lt_outs, &engine);

        if (with_bias)
            cp.execute(&strm,
                    {src_u8_ts, weight_s8_ts, bias_f32_ts, other_s8_ts},
                    {dst_s8_case2_ts});
        else
            cp.execute(&strm, {src_u8_ts, weight_s8_ts, other_s8_ts},
                    {dst_s8_case2_ts});
        strm.wait();

        if (isa < dnnl_cpu_isa_avx512_core_vnni)
            ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.1f,
                    /*atol*/ 1.f));
        else
            ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.01f,
                    /*atol*/ 1.f));
    }
}

TEST(int8_subgraph_mode, x8s8f32_conv1d_conv2d_conv3d) {
    using dims = impl::dnnl_impl::dims;

    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    std::vector<size_t> nds = {1, 2, 3};
    std::vector<int64_t> groups = {1, 4};
    std::vector<bool> with_biases = {true, false};
    std::vector<std::string> weight_qtypes = {"per_tensor", "per_channel"};
    std::vector<std::string> src_qtypes = {"symmetric", "asymmetric"};

    if (engine.kind() == impl::engine_kind::gpu) return;
    static auto isa = dnnl_get_effective_cpu_isa();
    SKIP_IF(isa < dnnl_cpu_isa_avx512_core_vnni
                    && engine.kind() == impl::engine_kind::cpu,
            "Skip the test for systems that do not support svx512_core_vnni.");

    for_(const auto &nd : nds)
    for_(const auto &g : groups)
    for_(const auto &with_bias : with_biases)
    for_(const auto &src_qtype : src_qtypes)
    for (const auto &wei_qtype : weight_qtypes) {
        if (isa < dnnl_cpu_isa_avx512_core_vnni && src_qtype == "asymmetric")
            continue;

        // prepare fp32 data
        int64_t in_channel = 8, out_channel = 8;
        int64_t kernel_size = 3;
        std::vector<int64_t> src_shape = nd == 1
                ? std::vector<int64_t> {1, in_channel, 12}
                : nd == 2 ? std::vector<int64_t> {1, in_channel, 12, 12}
                          : std::vector<int64_t> {1, in_channel, 12, 12, 12};
        std::vector<int64_t> weight_shape = nd == 1
                ? std::vector<int64_t> {out_channel, in_channel / g,
                        kernel_size}
                : nd == 2 ? std::vector<int64_t> {out_channel, in_channel / g,
                          kernel_size, kernel_size}
                          : std::vector<int64_t> {out_channel, in_channel / g,
                                  kernel_size, kernel_size, kernel_size};
        std::vector<int64_t> bias_shape {out_channel};
        std::vector<int64_t> dst_shape = nd == 1
                ? std::vector<int64_t> {1, out_channel, 10}
                : nd == 2 ? std::vector<int64_t> {1, out_channel, 10, 10}
                          : std::vector<int64_t> {1, out_channel, 10, 10, 10};

        test::vector<uint8_t> src_u8_data(product(src_shape));
        test::vector<int8_t> weight_s8_data(product(weight_shape));
        size_t bias_size = with_bias ? product(bias_shape) : 0;
        test::vector<float> bias_data(bias_size);

        // random generate src, weight and bias data random seed = 7
        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> u8_distribution(0.0f, 255.0f);
        std::uniform_real_distribution<float> s8_distribution(-127.0f, 128.0f);
        std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
        std::generate(src_u8_data.begin(), src_u8_data.end(), [&]() {
            return static_cast<uint8_t>(u8_distribution(generator));
        });
        std::generate(weight_s8_data.begin(), weight_s8_data.end(), [&]() {
            return static_cast<int8_t>(s8_distribution(generator));
        });
        if (with_bias) {
            std::generate(bias_data.begin(), bias_data.end(),
                    [&]() { return f32_distribution(generator); });
        }

        float scale_src = 1 / 255.f; // map to 0~255
        float scale_out = 1;
        int64_t zp_src = src_qtype == "symmetric" ? 0 : 128;
        int64_t zp_out = 78;

        size_t scale_size = wei_qtype == "per_tensor" ? 1 : out_channel;
        std::vector<float> scale_wei(scale_size, 1 / 127.f);
        std::vector<int64_t> zp_wei(scale_size, 0);

        impl::op_t dqdata_node(1, impl::op_kind::Dequantize, "dqdata_node");
        SET_Q_DQ_DATA_ATTR(dqdata_node)
        impl::op_t dqweight_node(3, impl::op_kind::Dequantize, "dqweight_node");
        SET_Q_DQ_WEIGHT_ATTR(dqweight_node)
        impl::op_t conv_node(4, impl::op_kind::Convolution, "conv_node");
        SET_CONV_ATTR(conv_node, nd)

        // prepare logical tensor
        impl::logical_tensor_t src_u8
                = utils::logical_tensor_init(1, src_shape, impl::data_type::u8);
        impl::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                2, src_shape, impl::data_type::f32);
        impl::logical_tensor_t weight_s8 = utils::logical_tensor_init(
                4, weight_shape, impl::data_type::s8);
        impl::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
                5, weight_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_f32 = utils::logical_tensor_init(
                7, dst_shape, impl::data_type::f32);
        impl::logical_tensor_t bias_f32;
        if (with_bias) {
            bias_f32 = utils::logical_tensor_init(
                    6, bias_shape, impl::data_type::f32);
        }

        dqdata_node.add_input(src_u8);
        dqdata_node.add_output(src_f32_dq);

        dqweight_node.add_input(weight_s8);
        dqweight_node.add_output(weight_f32_dq);

        conv_node.add_input(src_f32_dq);
        conv_node.add_input(weight_f32_dq);
        if (with_bias) conv_node.add_input(bias_f32);
        conv_node.add_output(dst_f32);

        impl::graph_t g;
        g.add_op(&dqdata_node);
        g.add_op(&dqweight_node);
        g.add_op(&conv_node);
        g.build_graph();

        impl::tensor_t src_u8_ts(src_u8, &engine, src_u8_data.data());
        impl::tensor_t weight_s8_ts(weight_s8, &engine, weight_s8_data.data());
        impl::tensor_t bias_f32_ts;
        if (with_bias) {
            bias_f32_ts = impl::tensor_t(bias_f32, &engine, bias_data.data());
        }

        // -------------------------case 1----------------------------------
        test::vector<float> case1_out_data(product(dst_shape));
        impl::tensor_t dst_f32_ts(dst_f32, &engine, case1_out_data.data());

        ASSERT_EQ(run_graph(g, {src_u8_ts, weight_s8_ts, bias_f32_ts},
                          {dst_f32_ts}, engine, strm),
                impl::status::success);

        // -------------------------case 2----------------------------------
        impl::pass::pass_base_ptr apass = with_bias
                ? get_pass("x8s8f32_conv_bias_fusion")
                : get_pass("x8s8f32_conv_fusion");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1);
        auto part = g.get_partitions()[0];

        // compile
        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> lt_ins;
        if (with_bias)
            lt_ins = {&src_u8, &weight_s8, &bias_f32};
        else
            lt_ins = {&src_u8, &weight_s8};
        std::vector<const impl::logical_tensor_t *> lt_outs {&dst_f32};

        p.compile(&cp, lt_ins, lt_outs, &engine);

        test::vector<float> case2_out_data(product(dst_shape));
        impl::tensor_t dst_f32_case2_ts(
                dst_f32, &engine, case2_out_data.data());
        if (with_bias)
            cp.execute(&strm, {src_u8_ts, weight_s8_ts, bias_f32_ts},
                    {dst_f32_case2_ts});
        else
            cp.execute(&strm, {src_u8_ts, weight_s8_ts}, {dst_f32_case2_ts});
        strm.wait();

        if (isa < dnnl_cpu_isa_avx512_core_vnni)
            ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.1f,
                    /*atol*/ 1.f));
        else
            ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.01f,
                    /*atol*/ 1.f));
    }
}

TEST(int8_subgraph_mode, x8s8f32_conv2d_relu) {
    using dims = impl::dnnl_impl::dims;

    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    std::vector<int64_t> groups = {1, 4};
    std::vector<bool> with_biases = {true, false};
    std::vector<std::string> weight_qtypes = {"per_tensor", "per_channel"};

    if (engine.kind() == impl::engine_kind::gpu) return;
    static auto isa = dnnl_get_effective_cpu_isa();
    SKIP_IF(isa < dnnl_cpu_isa_avx512_core_vnni,
            "Skip the test for systems that do not support "
            "avx512_core_vnni.");

    for_(const auto &g : groups)
    for_(const auto &with_bias : with_biases)
    for (const auto &wei_qtype : weight_qtypes) {
        // prepare data
        int64_t in_channel = 8, out_channel = 8;
        int64_t kernel_size = 3;
        std::vector<int64_t> src_shape {1, in_channel, 112, 112};
        std::vector<int64_t> weight_shape {
                out_channel, in_channel / g, kernel_size, kernel_size};
        std::vector<int64_t> bias_shape {out_channel};
        std::vector<int64_t> dst_shape {1, out_channel, 110, 110};

        test::vector<uint8_t> src_u8_data(product(src_shape));
        test::vector<int8_t> weight_s8_data(product(weight_shape));
        size_t bias_size = with_bias ? product(bias_shape) : 0;
        test::vector<float> bias_data(bias_size);
        test::vector<float> case1_out_data(product(dst_shape));
        test::vector<float> case2_out_data(product(dst_shape));

        // random generate src, weight and bias data random seed = 7
        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> u8_distribution(0.0f, 255.0f);
        std::uniform_real_distribution<float> s8_distribution(-127.0f, 128.0f);
        std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
        std::generate(src_u8_data.begin(), src_u8_data.end(), [&]() {
            return static_cast<uint8_t>(u8_distribution(generator));
        });
        std::generate(weight_s8_data.begin(), weight_s8_data.end(), [&]() {
            return static_cast<int8_t>(s8_distribution(generator));
        });
        if (with_bias) {
            std::generate(bias_data.begin(), bias_data.end(),
                    [&]() { return f32_distribution(generator); });
        }

        float scale_src = 1 / 255.f; // map to 0~255
        float scale_out = 1;
        int64_t zp_src = 0;
        int64_t zp_out = 78;

        size_t scale_size = wei_qtype == "per_tensor" ? 1 : out_channel;
        std::vector<float> scale_wei(scale_size, 1 / 127.f);
        std::vector<int64_t> zp_wei(scale_size, 0);

        impl::op_t dqdata_node(1, impl::op_kind::Dequantize, "dqdata_node");
        SET_Q_DQ_DATA_ATTR(dqdata_node)

        impl::op_t dqweight_node(3, impl::op_kind::Dequantize, "dqweight_node");
        SET_Q_DQ_WEIGHT_ATTR(dqweight_node)

        impl::op_t conv_node(4, impl::op_kind::Convolution, "conv_node");
        SET_CONV_ATTR(conv_node, 2)

        impl::op_t relu_node(5, impl::op_kind::ReLU, "relu_node");

        // prepare logical tensor
        impl::logical_tensor_t src_u8
                = utils::logical_tensor_init(1, src_shape, impl::data_type::u8);
        impl::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                2, src_shape, impl::data_type::f32);
        impl::logical_tensor_t weight_s8 = utils::logical_tensor_init(
                4, weight_shape, impl::data_type::s8);
        impl::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
                5, weight_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_f32 = utils::logical_tensor_init(
                7, dst_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_relu_f32 = utils::logical_tensor_init(
                8, dst_shape, impl::data_type::f32);
        impl::logical_tensor_t bias_f32;
        if (with_bias) {
            bias_f32 = utils::logical_tensor_init(
                    6, bias_shape, impl::data_type::f32);
        }

        dqdata_node.add_input(src_u8);
        dqdata_node.add_output(src_f32_dq);

        dqweight_node.add_input(weight_s8);
        dqweight_node.add_output(weight_f32_dq);

        conv_node.add_input(src_f32_dq);
        conv_node.add_input(weight_f32_dq);
        if (with_bias) conv_node.add_input(bias_f32);
        conv_node.add_output(dst_f32);

        relu_node.add_input(dst_f32);
        relu_node.add_output(dst_relu_f32);

        impl::graph_t g;
        g.add_op(&dqdata_node);
        g.add_op(&dqweight_node);
        g.add_op(&conv_node);
        g.add_op(&relu_node);
        g.build_graph();

        impl::tensor_t src_u8_ts(src_u8, &engine, src_u8_data.data());
        impl::tensor_t weight_s8_ts(weight_s8, &engine, weight_s8_data.data());
        impl::tensor_t bias_f32_ts;
        if (with_bias) {
            bias_f32_ts = impl::tensor_t(bias_f32, &engine, bias_data.data());
        }
        impl::tensor_t dst_relu_f32_ts(
                dst_relu_f32, &engine, case1_out_data.data());
        impl::tensor_t dst_f32_case2_ts(
                dst_relu_f32, &engine, case2_out_data.data());

        // -------------------------case 1----------------------------------
        ASSERT_EQ(run_graph(g, {src_u8_ts, weight_s8_ts, bias_f32_ts},
                          {dst_relu_f32_ts}, engine, strm),
                impl::status::success);

        // -------------------------case 2----------------------------------
        impl::pass::pass_base_ptr apass = with_bias
                ? get_pass("x8s8f32_conv_bias_relu_fusion")
                : get_pass("x8s8f32_conv_relu_fusion");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1);
        auto part = g.get_partitions()[0];

        // compile
        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> lt_ins;
        if (with_bias)
            lt_ins = {&src_u8, &weight_s8, &bias_f32};
        else
            lt_ins = {&src_u8, &weight_s8};
        std::vector<const impl::logical_tensor_t *> lt_outs {&dst_relu_f32};

        p.compile(&cp, lt_ins, lt_outs, &engine);

        if (with_bias)
            cp.execute(&strm, {src_u8_ts, weight_s8_ts, bias_f32_ts},
                    {dst_f32_case2_ts});
        else
            cp.execute(&strm, {src_u8_ts, weight_s8_ts}, {dst_f32_case2_ts});
        strm.wait();

        if (isa < dnnl_cpu_isa_avx512_core_vnni)
            ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.1f,
                    /*atol*/ 1.f));
        else
            ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.01f,
                    /*atol*/ 1.f));
    }
}

TEST(int8_subgraph_mode, x8s8f32_conv2d_sum_relu) {
    using dims = impl::dnnl_impl::dims;

    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    if (engine.kind() == impl::engine_kind::gpu) return;
    static auto isa = dnnl_get_effective_cpu_isa();
    SKIP_IF(isa < dnnl_cpu_isa_avx512_core_vnni,
            "Skip the test for systems that do not support "
            "avx512_core_vnni.");

    std::vector<int64_t> groups = {1, 4};
    std::vector<bool> with_biases = {true, false};
    std::vector<std::string> weight_qtypes = {"per_tensor", "per_channel"};

    for_(const auto &g : groups)
    for_(const auto &with_bias : with_biases)
    for (const auto &wei_qtype : weight_qtypes) {
        // prepare fp32 data
        int64_t in_channel = 8, out_channel = 8;
        int64_t kernel_size = 3;
        std::vector<int64_t> src_shape {1, in_channel, 112, 112};
        std::vector<int64_t> weight_shape {
                out_channel, in_channel / g, kernel_size, kernel_size};
        std::vector<int64_t> bias_shape {out_channel};
        std::vector<int64_t> dst_shape {1, out_channel, 110, 110};

        test::vector<uint8_t> src_u8_data(product(src_shape));
        test::vector<int8_t> weight_s8_data(product(weight_shape));
        test::vector<int8_t> other_s8_data(product(dst_shape));
        size_t bias_size = with_bias ? product(bias_shape) : 0;
        test::vector<float> bias_data(bias_size);
        test::vector<float> case1_out_data(product(dst_shape));
        test::vector<float> case2_out_data(product(dst_shape));

        // random generate src, weight and bias data random seed = 7
        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> u8_distribution(0.0f, 255.0f);
        std::uniform_real_distribution<float> s8_distribution(-127.0f, 128.0f);
        std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
        std::generate(src_u8_data.begin(), src_u8_data.end(), [&]() {
            return static_cast<uint8_t>(u8_distribution(generator));
        });
        std::generate(weight_s8_data.begin(), weight_s8_data.end(), [&]() {
            return static_cast<int8_t>(s8_distribution(generator));
        });
        std::generate(other_s8_data.begin(), other_s8_data.end(), [&]() {
            return static_cast<int8_t>(s8_distribution(generator));
        });
        if (with_bias) {
            std::generate(bias_data.begin(), bias_data.end(),
                    [&]() { return f32_distribution(generator); });
        }

        float scale_src = 1 / 255.f; // map to 0~255
        float scale_other = 1 / 127.f;
        float scale_out = 1;
        int64_t zp_src = 0;
        int64_t zp_other = 0;
        int64_t zp_out = 78;

        size_t scale_size = wei_qtype == "per_tensor" ? 1 : out_channel;

        std::vector<float> scale_wei(scale_size, 1 / 127.f);
        std::vector<int64_t> zp_wei(scale_size, 0);

        impl::op_t dqdata_node(1, impl::op_kind::Dequantize, "dqdata_node");
        SET_Q_DQ_DATA_ATTR(dqdata_node)

        impl::op_t dqweight_node(3, impl::op_kind::Dequantize, "dqweight_node");
        SET_Q_DQ_WEIGHT_ATTR(dqweight_node)

        impl::op_t conv_node(4, impl::op_kind::Convolution, "conv_node");
        SET_CONV_ATTR(conv_node, 2)

        impl::op_t relu_node(5, impl::op_kind::ReLU, "relu_node");

        impl::op_t dqother_node(8, impl::op_kind::Dequantize, "dqother_node");
        dqother_node.set_attr<std::string>("qtype", "per_tensor");
        dqother_node.set_attr<std::vector<int64_t>>("zps", {zp_other});
        dqother_node.set_attr<std::vector<float>>("scales", {scale_other});
        dqother_node.set_attr<int64_t>("axis", 0);

        impl::op_t add_node(9, impl::op_kind::Add, "add_node");

        // prepare logical tensor
        auto src_u8 = utils::logical_tensor_init(1, impl::data_type::u8);
        auto src_f32_dq = utils::logical_tensor_init(2, impl::data_type::f32);
        auto weight_s8 = utils::logical_tensor_init(4, impl::data_type::s8);
        auto weight_f32_dq
                = utils::logical_tensor_init(5, impl::data_type::f32);
        auto dst_f32 = utils::logical_tensor_init(7, impl::data_type::f32);
        auto dst_relu_f32 = utils::logical_tensor_init(8, impl::data_type::f32);
        auto other_s8 = utils::logical_tensor_init(11, impl::data_type::s8);
        auto other_f32_dq
                = utils::logical_tensor_init(12, impl::data_type::f32);
        auto dst_add_f32 = utils::logical_tensor_init(13, impl::data_type::f32);
        impl::logical_tensor_t bias_f32;
        if (with_bias) {
            bias_f32 = utils::logical_tensor_init(6, impl::data_type::f32);
        }

        dqdata_node.add_input(src_u8);
        dqdata_node.add_output(src_f32_dq);

        dqweight_node.add_input(weight_s8);
        dqweight_node.add_output(weight_f32_dq);

        conv_node.add_input(src_f32_dq);
        conv_node.add_input(weight_f32_dq);
        if (with_bias) conv_node.add_input(bias_f32);
        conv_node.add_output(dst_f32);

        dqother_node.add_input(other_s8);
        dqother_node.add_output(other_f32_dq);

        add_node.add_input(dst_f32);
        add_node.add_input(other_f32_dq);
        add_node.add_output(dst_add_f32);

        relu_node.add_input(dst_add_f32);
        relu_node.add_output(dst_relu_f32);

        impl::graph_t g;
        g.add_op(&dqdata_node);
        g.add_op(&dqweight_node);
        g.add_op(&conv_node);
        g.add_op(&dqother_node);
        g.add_op(&add_node);
        g.add_op(&relu_node);
        g.build_graph();

        // prepare in/out with full shape
        src_u8 = utils::logical_tensor_init(1, src_shape, impl::data_type::u8);
        weight_s8 = utils::logical_tensor_init(
                4, weight_shape, impl::data_type::s8);
        other_s8 = utils::logical_tensor_init(
                11, dst_shape, impl::data_type::s8);
        if (with_bias) {
            bias_f32 = utils::logical_tensor_init(
                    6, bias_shape, impl::data_type::f32);
        }
        dst_relu_f32 = utils::logical_tensor_init(
                8, dst_shape, impl::data_type::f32);

        impl::tensor_t src_u8_ts(src_u8, &engine, src_u8_data.data());
        impl::tensor_t weight_s8_ts(weight_s8, &engine, weight_s8_data.data());
        impl::tensor_t other_s8_ts(other_s8, &engine, other_s8_data.data());
        impl::tensor_t bias_f32_ts;
        if (with_bias) {
            bias_f32_ts = impl::tensor_t(bias_f32, &engine, bias_data.data());
        }
        impl::tensor_t dst_relu_f32_ts(
                dst_relu_f32, &engine, case1_out_data.data());
        impl::tensor_t dst_f32_case2_ts(
                dst_relu_f32, &engine, case2_out_data.data());

        // -------------------------case 1----------------------------------
        ASSERT_EQ(run_graph(g,
                          {src_u8_ts, weight_s8_ts, bias_f32_ts, other_s8_ts},
                          {dst_relu_f32_ts}, engine, strm),
                impl::status::success);

        // -------------------------case 2----------------------------------
        impl::pass::pass_base_ptr apass = with_bias
                ? get_pass("x8s8f32_conv_bias_add_relu_fusion")
                : get_pass("x8s8f32_conv_add_relu_fusion");

        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1);
        auto part = g.get_partitions()[0];

        // compile
        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> lt_ins;
        if (with_bias)
            lt_ins = {&src_u8, &weight_s8, &bias_f32, &other_s8};
        else
            lt_ins = {&src_u8, &weight_s8, &other_s8};
        std::vector<const impl::logical_tensor_t *> lt_outs {&dst_relu_f32};

        p.compile(&cp, lt_ins, lt_outs, &engine);

        if (with_bias)
            cp.execute(&strm,
                    {src_u8_ts, weight_s8_ts, bias_f32_ts, other_s8_ts},
                    {dst_f32_case2_ts});
        else
            cp.execute(&strm, {src_u8_ts, weight_s8_ts, other_s8_ts},
                    {dst_f32_case2_ts});
        strm.wait();

        if (isa < dnnl_cpu_isa_avx512_core_vnni)
            ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.1f,
                    /*atol*/ 1.f));
        else
            ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.01f,
                    /*atol*/ 1.f));
    }
}

TEST(int8_subgraph_mode, x8s8f32_conv2d_sum_relu_NXC) {
    using dims = impl::dnnl_impl::dims;

    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    if (engine.kind() == impl::engine_kind::gpu) return;
    static auto isa = dnnl_get_effective_cpu_isa();
    SKIP_IF(isa < dnnl_cpu_isa_avx512_core_vnni,
            "Skip the test for systems that do not support "
            "avx512_core_vnni.");

    std::vector<int64_t> groups = {1, 4};
    std::vector<bool> with_biases = {true, false};
    std::vector<std::string> weight_qtypes = {"per_tensor", "per_channel"};

    for_(const auto &g : groups)
    for_(const auto &with_bias : with_biases)
    for (const auto &wei_qtype : weight_qtypes) {
        // prepare fp32 data
        int64_t in_channel = 8, out_channel = 8;
        int64_t kernel_size = 3;
        std::vector<int64_t> src_shape {1, 12, 12, in_channel};
        std::vector<int64_t> weight_shape {
                kernel_size, kernel_size, in_channel / g, out_channel};
        std::vector<int64_t> bias_shape {out_channel};
        std::vector<int64_t> dst_shape {1, 10, 10, out_channel};

        test::vector<uint8_t> src_u8_data(product(src_shape));
        test::vector<int8_t> weight_s8_data(product(weight_shape));
        test::vector<int8_t> other_s8_data(product(dst_shape));
        size_t bias_size = with_bias ? product(bias_shape) : 0;
        test::vector<float> bias_data(bias_size);
        test::vector<float> case1_out_data(product(dst_shape));
        test::vector<float> case2_out_data(product(dst_shape));

        // random generate src, weight and bias data random seed = 7
        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> u8_distribution(0.0f, 255.0f);
        std::uniform_real_distribution<float> s8_distribution(-127.0f, 128.0f);
        std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
        std::generate(src_u8_data.begin(), src_u8_data.end(), [&]() {
            return static_cast<uint8_t>(u8_distribution(generator));
        });
        std::generate(weight_s8_data.begin(), weight_s8_data.end(), [&]() {
            return static_cast<int8_t>(s8_distribution(generator));
        });
        std::generate(other_s8_data.begin(), other_s8_data.end(), [&]() {
            return static_cast<int8_t>(s8_distribution(generator));
        });
        if (with_bias) {
            std::generate(bias_data.begin(), bias_data.end(),
                    [&]() { return f32_distribution(generator); });
        }

        float scale_src = 1 / 255.f; // map to 0~255
        float scale_other = 1 / 127.f;
        float scale_out = 1;
        int64_t zp_src = 0;
        int64_t zp_other = 0;
        int64_t zp_out = 78;

        size_t scale_size = wei_qtype == "per_tensor" ? 1 : out_channel;

        std::vector<float> scale_wei(scale_size, 1 / 127.f);
        std::vector<int64_t> zp_wei(scale_size, 0);

        impl::op_t dqdata_node(1, impl::op_kind::Dequantize, "dqdata_node");
        SET_Q_DQ_DATA_ATTR(dqdata_node)

        impl::op_t dqweight_node(3, impl::op_kind::Dequantize, "dqweight_node");
        SET_Q_DQ_WEIGHT_ATTR(dqweight_node)
        dqweight_node.set_attr<int64_t>(
                "axis", wei_qtype == "per_tensor" ? 0 : 3);

        impl::op_t conv_node(4, impl::op_kind::Convolution, "conv_node");
        SET_CONV_ATTR(conv_node, 2)
        conv_node.set_attr<std::string>("data_format", "NXC");
        conv_node.set_attr<std::string>("filter_format", "XIO");

        impl::op_t relu_node(5, impl::op_kind::ReLU, "relu_node");

        impl::op_t dqother_node(8, impl::op_kind::Dequantize, "dqother_node");
        dqother_node.set_attr<std::string>("qtype", "per_tensor");
        dqother_node.set_attr<std::vector<int64_t>>("zps", {zp_other});
        dqother_node.set_attr<std::vector<float>>("scales", {scale_other});
        dqother_node.set_attr<int64_t>("axis", 0);

        impl::op_t add_node(9, impl::op_kind::Add, "add_node");

        // prepare logical tensor
        auto src_u8 = utils::logical_tensor_init(1, impl::data_type::u8);
        auto src_f32_dq = utils::logical_tensor_init(2, impl::data_type::f32);
        auto weight_s8 = utils::logical_tensor_init(4, impl::data_type::s8);
        auto weight_f32_dq
                = utils::logical_tensor_init(5, impl::data_type::f32);
        auto dst_f32 = utils::logical_tensor_init(7, impl::data_type::f32);
        auto dst_relu_f32 = utils::logical_tensor_init(8, impl::data_type::f32);
        auto other_s8 = utils::logical_tensor_init(11, impl::data_type::s8);
        auto other_f32_dq
                = utils::logical_tensor_init(12, impl::data_type::f32);
        auto dst_add_f32 = utils::logical_tensor_init(13, impl::data_type::f32);
        impl::logical_tensor_t bias_f32;
        if (with_bias) {
            bias_f32 = utils::logical_tensor_init(6, impl::data_type::f32);
        }

        dqdata_node.add_input(src_u8);
        dqdata_node.add_output(src_f32_dq);

        dqweight_node.add_input(weight_s8);
        dqweight_node.add_output(weight_f32_dq);

        conv_node.add_input(src_f32_dq);
        conv_node.add_input(weight_f32_dq);
        if (with_bias) conv_node.add_input(bias_f32);
        conv_node.add_output(dst_f32);

        dqother_node.add_input(other_s8);
        dqother_node.add_output(other_f32_dq);

        add_node.add_input(dst_f32);
        add_node.add_input(other_f32_dq);
        add_node.add_output(dst_add_f32);

        relu_node.add_input(dst_add_f32);
        relu_node.add_output(dst_relu_f32);

        impl::graph_t g;
        g.add_op(&dqdata_node);
        g.add_op(&dqweight_node);
        g.add_op(&conv_node);
        g.add_op(&dqother_node);
        g.add_op(&add_node);
        g.add_op(&relu_node);
        g.build_graph();

        // prepare in/out with full shape
        src_u8 = utils::logical_tensor_init(1, src_shape, impl::data_type::u8);
        weight_s8 = utils::logical_tensor_init(
                4, weight_shape, impl::data_type::s8);
        other_s8 = utils::logical_tensor_init(
                11, dst_shape, impl::data_type::s8);
        if (with_bias) {
            bias_f32 = utils::logical_tensor_init(
                    6, bias_shape, impl::data_type::f32);
        }
        dst_relu_f32 = utils::logical_tensor_init(
                8, dst_shape, impl::data_type::f32);

        impl::tensor_t src_u8_ts(src_u8, &engine, src_u8_data.data());
        impl::tensor_t weight_s8_ts(weight_s8, &engine, weight_s8_data.data());
        impl::tensor_t other_s8_ts(other_s8, &engine, other_s8_data.data());
        impl::tensor_t bias_f32_ts;
        if (with_bias) {
            bias_f32_ts = impl::tensor_t(bias_f32, &engine, bias_data.data());
        }
        impl::tensor_t dst_relu_f32_ts(
                dst_relu_f32, &engine, case1_out_data.data());
        impl::tensor_t dst_f32_case2_ts(
                dst_relu_f32, &engine, case2_out_data.data());

        // -------------------------case 1----------------------------------
        ASSERT_EQ(run_graph(g,
                          {src_u8_ts, weight_s8_ts, bias_f32_ts, other_s8_ts},
                          {dst_relu_f32_ts}, engine, strm),
                impl::status::success);

        // -------------------------case 2----------------------------------
        impl::pass::pass_base_ptr apass = with_bias
                ? get_pass("x8s8f32_conv_bias_add_relu_fusion")
                : get_pass("x8s8f32_conv_add_relu_fusion");

        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1);
        auto part = g.get_partitions()[0];

        // compile
        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> lt_ins;
        if (with_bias)
            lt_ins = {&src_u8, &weight_s8, &bias_f32, &other_s8};
        else
            lt_ins = {&src_u8, &weight_s8, &other_s8};
        std::vector<const impl::logical_tensor_t *> lt_outs {&dst_relu_f32};

        p.compile(&cp, lt_ins, lt_outs, &engine);

        if (with_bias)
            cp.execute(&strm,
                    {src_u8_ts, weight_s8_ts, bias_f32_ts, other_s8_ts},
                    {dst_f32_case2_ts});
        else
            cp.execute(&strm, {src_u8_ts, weight_s8_ts, other_s8_ts},
                    {dst_f32_case2_ts});
        strm.wait();

        if (isa < dnnl_cpu_isa_avx512_core_vnni)
            ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.1f,
                    /*atol*/ 1.f));
        else
            ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.01f,
                    /*atol*/ 1.f));
    }
}

TEST(int8_subgraph_mode, int8_matmul_ndx2d) {
    // compare results between:
    // case 1: [quantize] - [dequantize] - [fp32_matmul] - [quantize]
    // case 2: [quantize] - [int8_matmul]
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    if (engine.kind() == impl::engine_kind::gpu) return;

    std::vector<std::string> qtypes {"per_tensor", "per_channel"};
    std::vector<std::vector<int64_t>> src_shapes {
            {3, 3, 3, 8, 4}, {3, 3, 8, 4}, {3, 8, 4}, {8, 4}, {4}};
    std::vector<std::vector<int64_t>> weight_shapes {{4, 2}};
    std::vector<std::vector<int64_t>> dst_shapes {
            {3, 3, 3, 8, 2}, {3, 3, 8, 2}, {3, 8, 2}, {8, 2}, {2}};
    for_(const auto &qtype : qtypes)
    for_(size_t i = 0; i < src_shapes.size(); ++i)
    for (size_t j = 0; j < weight_shapes.size(); ++j) {
        // prepare fp32 data
        std::vector<int64_t> src_shape = src_shapes[i];
        std::vector<int64_t> weight_shape = weight_shapes[j];
        std::vector<int64_t> bias_shape {2};
        std::vector<int64_t> dst_shape = dst_shapes[i];

        test::vector<uint8_t> src_data(product(src_shape));
        test::vector<int8_t> weight_data(product(weight_shape));
        test::vector<float> bias_data(product(bias_shape));

        // random generate src, weight and bias data random seed = 7
        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
        std::uniform_real_distribution<float> u8_distribution(0.0f, 255.0f);
        std::uniform_real_distribution<float> s8_distribution(-127.0f, 128.0f);
        std::generate(src_data.begin(), src_data.end(), [&]() {
            return static_cast<uint8_t>(u8_distribution(generator));
        });
        std::generate(weight_data.begin(), weight_data.end(), [&]() {
            return static_cast<int8_t>(s8_distribution(generator));
        });
        std::generate(bias_data.begin(), bias_data.end(),
                [&]() { return f32_distribution(generator); });
        float scale_src = 1 / 255.f; // map to 0~255
        float scale_out = 1;
        int64_t zp_src = 0;
        int64_t zp_out = 78;

        size_t scales_wei_sizes = qtype == "per_tensor" ? 1 : dst_shape.back();
        std::vector<float> scale_wei(scales_wei_sizes, 1 / 127.f);
        std::vector<int64_t> zp_wei(scales_wei_sizes, 0);

        impl::op_t dqdata_op(1, impl::op_kind::Dequantize, "dqdata_op");
        dqdata_op.set_attr<std::string>("qtype", "per_tensor");
        dqdata_op.set_attr<std::vector<int64_t>>("zps", {zp_src});
        dqdata_op.set_attr<std::vector<float>>("scales", {scale_src});
        dqdata_op.set_attr<int64_t>("axis", 0);

        impl::op_t dqweight_op(2, impl::op_kind::Dequantize, "dqweight_op");
        dqweight_op.set_attr<std::string>("qtype", qtype);
        dqweight_op.set_attr<std::vector<int64_t>>("zps", zp_wei);
        dqweight_op.set_attr<std::vector<float>>("scales", scale_wei);
        dqweight_op.set_attr<int64_t>("axis", 1);

        impl::op_t matmul_op(3, impl::op_kind::MatMul, "matmul_op");
        matmul_op.set_attr<bool>("transpose_a", false);
        matmul_op.set_attr<bool>("transpose_b", false);

        impl::op_t qout_op(4, impl::op_kind::Quantize, "qout_op");
        qout_op.set_attr<std::string>("qtype", "per_tensor");
        qout_op.set_attr<std::vector<int64_t>>("zps", {zp_out});
        qout_op.set_attr<std::vector<float>>("scales", {scale_out});
        qout_op.set_attr<int64_t>("axis", 0);

        // prepare logical tensor
        impl::logical_tensor_t src_u8
                = utils::logical_tensor_init(1, src_shape, impl::data_type::u8);
        impl::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                2, src_shape, impl::data_type::f32);
        impl::logical_tensor_t weight_s8 = utils::logical_tensor_init(
                4, weight_shape, impl::data_type::s8);
        impl::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
                5, weight_shape, impl::data_type::f32);
        impl::logical_tensor_t bias_f32 = utils::logical_tensor_init(
                6, bias_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_f32 = utils::logical_tensor_init(
                7, dst_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_s8
                = utils::logical_tensor_init(8, dst_shape, impl::data_type::s8);

        dqdata_op.add_input(src_u8);
        dqdata_op.add_output(src_f32_dq);

        dqweight_op.add_input(weight_s8);
        dqweight_op.add_output(weight_f32_dq);

        matmul_op.add_input(src_f32_dq);
        matmul_op.add_input(weight_f32_dq);
        matmul_op.add_input(bias_f32);
        matmul_op.add_output(dst_f32);

        qout_op.add_input(dst_f32);
        qout_op.add_output(dst_s8);

        impl::graph_t g;
        g.add_op(&dqdata_op);
        g.add_op(&dqweight_op);
        g.add_op(&matmul_op);
        g.add_op(&qout_op);
        g.build_graph();

        impl::tensor_t src_u8_ts(src_u8, &engine, src_data.data());
        impl::tensor_t weight_s8_ts(weight_s8, &engine, weight_data.data());
        impl::tensor_t bias_f32_ts(bias_f32, &engine, bias_data.data());

        // -------------------------case 1----------------------------------
        test::vector<int8_t> case1_out_data(product(dst_shape));
        impl::tensor_t dst_s8_ts(dst_s8, &engine, case1_out_data.data());
        ASSERT_EQ(run_graph(g, {src_u8_ts, weight_s8_ts, bias_f32_ts},
                          {dst_s8_ts}, engine, strm),
                impl::status::success);
        // -------------------------case 2----------------------------------
        impl::pass::pass_base_ptr apass = get_pass("int8_matmul_bias_fusion");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1);
        auto part = g.get_partitions()[0];

        // compile
        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> lt_ins {
                &src_u8, &weight_s8, &bias_f32};
        std::vector<const impl::logical_tensor_t *> lt_outs {&dst_s8};

        p.compile(&cp, lt_ins, lt_outs, &engine);

        test::vector<int8_t> case2_out_data(product(dst_shape));
        impl::tensor_t dst_s8_case2_ts(dst_s8, &engine, case2_out_data.data());
        cp.execute(&strm, {src_u8_ts, weight_s8_ts, bias_f32_ts},
                {dst_s8_case2_ts});
        strm.wait();

        static auto isa = dnnl_get_effective_cpu_isa();
        if (isa < dnnl_cpu_isa_avx512_core_vnni)
            ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.1f,
                    /*atol*/ 1.f));
        else
            ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.01f,
                    /*atol*/ 1.f));
    }
}

TEST(int8_subgraph_mode, int8_matmul_ndx1d) {
    // compare results between: case 1: [quantize] - [dequantize] -
    // [fp32_matmul] - [quantize] case 2: [quantize] - [int8_matmul]
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    if (engine.kind() == impl::engine_kind::gpu) return;

    std::vector<std::vector<int64_t>> src_shapes {{3, 8, 4}, {8, 4}};
    std::vector<std::vector<int64_t>> weight_shapes {{4, 1}, {4}};
    std::vector<std::vector<int64_t>> dst_shapes {
            {3, 8, 1}, {8, 1}, {3, 8}, {8}};

    for (size_t i = 0; i < weight_shapes.size(); ++i) {
        for (size_t j = 0; j < src_shapes.size(); ++j) {
            // prepare fp32 data
            std::vector<int64_t> src_shape = src_shapes[j];
            std::vector<int64_t> weight_shape = weight_shapes[i];
            std::vector<int64_t> dst_shape
                    = dst_shapes[j + i * src_shapes.size()];

            test::vector<uint8_t> src_data(product(src_shape));
            test::vector<int8_t> weight_data(product(weight_shape));

            // random generate src, weight and bias data random seed = 7
            std::default_random_engine generator(7);
            std::uniform_real_distribution<float> u8_distribution(0.0f, 255.0f);
            std::uniform_real_distribution<float> s8_distribution(
                    -127.0f, 128.0f);
            std::generate(src_data.begin(), src_data.end(), [&]() {
                return static_cast<uint8_t>(u8_distribution(generator));
            });
            std::generate(weight_data.begin(), weight_data.end(), [&]() {
                return static_cast<int8_t>(s8_distribution(generator));
            });
            float scale_src = 1 / 255.f; // map to 0~255
            float scale_wei = 1 / 127.f;
            float scale_out = 1;
            int64_t zp_src = 0;
            int64_t zp_wei = 0;
            int64_t zp_out = 78;

            impl::op_t dqdata_op(1, impl::op_kind::Dequantize, "dqdata_op");
            dqdata_op.set_attr<std::string>("qtype", "per_tensor");
            dqdata_op.set_attr<std::vector<int64_t>>("zps", {zp_src});
            dqdata_op.set_attr<std::vector<float>>("scales", {scale_src});
            dqdata_op.set_attr<int64_t>("axis", 0);

            impl::op_t dqweight_op(2, impl::op_kind::Dequantize, "dqweight_op");
            dqweight_op.set_attr<std::string>("qtype", "per_tensor");
            dqweight_op.set_attr<std::vector<int64_t>>("zps", {zp_wei});
            dqweight_op.set_attr<std::vector<float>>("scales", {scale_wei});
            dqweight_op.set_attr<int64_t>("axis", 0);

            impl::op_t matmul_op(3, impl::op_kind::MatMul, "matmul_op");
            matmul_op.set_attr<bool>("transpose_a", false);
            matmul_op.set_attr<bool>("transpose_b", false);

            impl::op_t qout_op(4, impl::op_kind::Quantize, "qout_op");
            qout_op.set_attr<std::string>("qtype", "per_tensor");
            qout_op.set_attr<std::vector<int64_t>>("zps", {zp_out});
            qout_op.set_attr<std::vector<float>>("scales", {scale_out});
            qout_op.set_attr<int64_t>("axis", 0);

            // prepare logical tensor
            impl::logical_tensor_t src_u8 = utils::logical_tensor_init(
                    1, src_shape, impl::data_type::u8);
            impl::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                    2, src_shape, impl::data_type::f32);
            impl::logical_tensor_t weight_s8 = utils::logical_tensor_init(
                    4, weight_shape, impl::data_type::s8);
            impl::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
                    5, weight_shape, impl::data_type::f32);
            impl::logical_tensor_t dst_f32 = utils::logical_tensor_init(
                    7, dst_shape, impl::data_type::f32);
            impl::logical_tensor_t dst_s8 = utils::logical_tensor_init(
                    8, dst_shape, impl::data_type::s8);

            dqdata_op.add_input(src_u8);
            dqdata_op.add_output(src_f32_dq);

            dqweight_op.add_input(weight_s8);
            dqweight_op.add_output(weight_f32_dq);

            matmul_op.add_input(src_f32_dq);
            matmul_op.add_input(weight_f32_dq);
            matmul_op.add_output(dst_f32);

            qout_op.add_input(dst_f32);
            qout_op.add_output(dst_s8);

            impl::graph_t g;
            g.add_op(&dqdata_op);
            g.add_op(&dqweight_op);
            g.add_op(&matmul_op);
            g.add_op(&qout_op);
            g.build_graph();

            impl::tensor_t src_u8_ts(src_u8, &engine, src_data.data());
            impl::tensor_t weight_s8_ts(weight_s8, &engine, weight_data.data());
            // -------------------------case 1----------------------------------
            test::vector<int8_t> case1_out_data(product(dst_shape));
            impl::tensor_t dst_s8_case1_ts(
                    dst_s8, &engine, case1_out_data.data());
            ASSERT_EQ(run_graph(g, {src_u8_ts, weight_s8_ts}, {dst_s8_case1_ts},
                              engine, strm),
                    impl::status::success);

            // -------------------------case 2----------------------------------
            impl::pass::pass_base_ptr apass = get_pass("int8_matmul_fusion");
            apass->run(g);
            ASSERT_EQ(g.get_num_partitions(), 1);
            auto part = g.get_partitions()[0];

            // compile
            impl::partition_t p;
            p.init(part);

            impl::compiled_partition_t cp(p);

            std::vector<const impl::logical_tensor_t *> lt_ins {
                    &src_u8, &weight_s8};
            std::vector<const impl::logical_tensor_t *> lt_outs {&dst_s8};

            p.compile(&cp, lt_ins, lt_outs, &engine);

            test::vector<int8_t> case2_out_data(product(dst_shape));
            impl::tensor_t dst_s8_case2_ts(
                    dst_s8, &engine, case2_out_data.data());
            cp.execute(&strm, {src_u8_ts, weight_s8_ts}, {dst_s8_case2_ts});
            strm.wait();

            static auto isa = dnnl_get_effective_cpu_isa();
            if (isa < dnnl_cpu_isa_avx512_core_vnni)
                ASSERT_TRUE(
                        allclose(case1_out_data, case2_out_data, /*rtol*/ 0.1f,
                                /*atol*/ 1.f));
            else
                ASSERT_TRUE(
                        allclose(case1_out_data, case2_out_data, /*rtol*/ 0.01f,
                                /*atol*/ 1.f));
        }
    }
}

TEST(int8_subgraph_mode, int8_matmul_ndx2d_with_transpose) {
    // compare results between:
    // case 1: [quantize] - [dequantize] - [fp32_matmul] - [quantize]
    // case 2: [quantize] - [int8_matmul]
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    if (engine.kind() == impl::engine_kind::gpu) return;

    std::vector<std::vector<int64_t>> src_shapes {
            {3, 3, 3, 8, 4}, {3, 3, 8, 4}, {3, 8, 4}, {8, 4}, {4}};
    std::vector<std::vector<int64_t>> weight_shapes {{2, 4}};
    std::vector<std::vector<int64_t>> dst_shapes {
            {3, 3, 3, 8, 2}, {3, 3, 8, 2}, {3, 8, 2}, {8, 2}, {2}};

    for (size_t i = 0; i < src_shapes.size(); ++i) {
        for (size_t j = 0; j < weight_shapes.size(); ++j) {
            // prepare fp32 data
            std::vector<int64_t> src_shape = src_shapes[i];
            std::vector<int64_t> weight_shape = weight_shapes[j];
            std::vector<int64_t> bias_shape {1};
            std::vector<int64_t> dst_shape = dst_shapes[i];

            test::vector<uint8_t> src_data(product(src_shape));
            test::vector<int8_t> weight_data(product(weight_shape));
            test::vector<float> bias_data(product(bias_shape));

            // random generate src, weight and bias data random seed = 7
            std::default_random_engine generator(7);
            std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
            std::uniform_real_distribution<float> u8_distribution(0.0f, 255.0f);
            std::uniform_real_distribution<float> s8_distribution(
                    -127.0f, 128.0f);
            std::generate(src_data.begin(), src_data.end(), [&]() {
                return static_cast<uint8_t>(u8_distribution(generator));
            });
            std::generate(weight_data.begin(), weight_data.end(), [&]() {
                return static_cast<int8_t>(s8_distribution(generator));
            });
            std::generate(bias_data.begin(), bias_data.end(),
                    [&]() { return f32_distribution(generator); });
            float scale_src = 1 / 255.f; // map to 0~255
            float scale_wei = 1 / 127.f;
            float scale_out = 1;
            int64_t zp_src = 0;
            int64_t zp_wei = 0;
            int64_t zp_out = 78;

            // -------------------------case 1----------------------------------
            impl::op_t dqdata_op(1, impl::op_kind::Dequantize, "dqdata_op");
            dqdata_op.set_attr<std::string>("qtype", "per_tensor");
            dqdata_op.set_attr<std::vector<int64_t>>("zps", {zp_src});
            dqdata_op.set_attr<std::vector<float>>("scales", {scale_src});
            dqdata_op.set_attr<int64_t>("axis", 0);

            impl::op_t dqweight_op(2, impl::op_kind::Dequantize, "dqweight_op");
            dqweight_op.set_attr<std::string>("qtype", "per_tensor");
            dqweight_op.set_attr<std::vector<int64_t>>("zps", {zp_wei});
            dqweight_op.set_attr<std::vector<float>>("scales", {scale_wei});
            dqweight_op.set_attr<int64_t>("axis", 0);

            impl::op_t matmul_op(3, impl::op_kind::MatMul, "matmul_op");
            matmul_op.set_attr<bool>("transpose_a", false);
            matmul_op.set_attr<bool>("transpose_b", true);

            impl::op_t qout_op(4, impl::op_kind::Quantize, "qout_op");
            qout_op.set_attr<std::string>("qtype", "per_tensor");
            qout_op.set_attr<std::vector<int64_t>>("zps", {zp_out});
            qout_op.set_attr<std::vector<float>>("scales", {scale_out});
            qout_op.set_attr<int64_t>("axis", 0);

            // prepare logical tensor
            impl::logical_tensor_t src_u8 = utils::logical_tensor_init(
                    1, src_shape, impl::data_type::u8);
            impl::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                    2, src_shape, impl::data_type::f32);
            impl::logical_tensor_t weight_s8 = utils::logical_tensor_init(
                    4, weight_shape, impl::data_type::s8);
            impl::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
                    5, weight_shape, impl::data_type::f32);
            impl::logical_tensor_t bias_f32 = utils::logical_tensor_init(
                    6, bias_shape, impl::data_type::f32);
            impl::logical_tensor_t dst_f32 = utils::logical_tensor_init(
                    7, dst_shape, impl::data_type::f32);
            impl::logical_tensor_t dst_s8 = utils::logical_tensor_init(
                    8, dst_shape, impl::data_type::s8);

            dqdata_op.add_input(src_u8);
            dqdata_op.add_output(src_f32_dq);

            dqweight_op.add_input(weight_s8);
            dqweight_op.add_output(weight_f32_dq);

            matmul_op.add_input(src_f32_dq);
            matmul_op.add_input(weight_f32_dq);
            matmul_op.add_input(bias_f32);
            matmul_op.add_output(dst_f32);

            qout_op.add_input(dst_f32);
            qout_op.add_output(dst_s8);

            impl::graph_t g;
            g.add_op(&dqdata_op);
            g.add_op(&dqweight_op);
            g.add_op(&matmul_op);
            g.add_op(&qout_op);
            g.build_graph();

            impl::tensor_t src_u8_ts(src_u8, &engine, src_data.data());
            impl::tensor_t weight_s8_ts(weight_s8, &engine, weight_data.data());
            impl::tensor_t bias_f32_ts
                    = impl::tensor_t(bias_f32, &engine, bias_data.data());
            // -------------------------case 1----------------------------------
            test::vector<int8_t> case1_out_data(product(dst_shape));
            impl::tensor_t dst_s8_ts(dst_s8, &engine, case1_out_data.data());
            ASSERT_EQ(run_graph(g, {src_u8_ts, weight_s8_ts, bias_f32_ts},
                              {dst_s8_ts}, engine, strm),
                    impl::status::success);

            // -------------------------case 2----------------------------------
            impl::pass::pass_base_ptr apass
                    = get_pass("int8_matmul_bias_fusion");
            apass->run(g);
            ASSERT_EQ(g.get_num_partitions(), 1);
            auto part = g.get_partitions()[0];

            // compile
            impl::partition_t p;
            p.init(part);

            impl::compiled_partition_t cp(p);

            std::vector<const impl::logical_tensor_t *> lt_ins {
                    &src_u8, &weight_s8, &bias_f32};
            std::vector<const impl::logical_tensor_t *> lt_outs {&dst_s8};

            p.compile(&cp, lt_ins, lt_outs, &engine);

            test::vector<int8_t> case2_out_data(product(dst_shape));
            impl::tensor_t dst_s8_case2_ts(
                    dst_s8, &engine, case2_out_data.data());
            cp.execute(&strm, {src_u8_ts, weight_s8_ts, bias_f32_ts},
                    {dst_s8_case2_ts});
            strm.wait();

            static auto isa = dnnl_get_effective_cpu_isa();
            if (isa < dnnl_cpu_isa_avx512_core_vnni)
                ASSERT_TRUE(
                        allclose(case1_out_data, case2_out_data, /*rtol*/ 0.1f,
                                /*atol*/ 1.f));
            else
                ASSERT_TRUE(
                        allclose(case1_out_data, case2_out_data, /*rtol*/ 0.01f,
                                /*atol*/ 1.f));
        }
    }
}

TEST(int8_subgraph_mode, int8_matmul_relu_fusion) {
    // compare results between:
    // case 1: [quantize] - [dequantize] - [fp32_matmul] - [relu] - [quantize]
    // case 2: [quantize] - [int8_matmul]
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    if (engine.kind() == impl::engine_kind::gpu) return;

    // prepare fp32 data
    std::vector<int64_t> src_shape {8, 6};
    std::vector<int64_t> weight_shape {6, 4};
    std::vector<int64_t> dst_shape {8, 4};

    test::vector<uint8_t> src_data(product(src_shape));
    test::vector<int8_t> weight_data(product(weight_shape));

    // random generate src, weight and bias data random seed = 7
    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> u8_distribution(0.0f, 255.0f);
    std::uniform_real_distribution<float> s8_distribution(-127.0f, 128.0f);
    std::generate(src_data.begin(), src_data.end(),
            [&]() { return static_cast<uint8_t>(u8_distribution(generator)); });
    std::generate(weight_data.begin(), weight_data.end(),
            [&]() { return static_cast<int8_t>(s8_distribution(generator)); });
    float scale_src = 1 / 255.f; // map to 0~255
    float scale_wei = 1 / 127.f;
    float scale_out = 1;
    int64_t zp_src = 0;
    int64_t zp_wei = 0;
    int64_t zp_out = 78;

    // -------------------------case 1----------------------------------
    impl::op_t dqdata_op(1, impl::op_kind::Dequantize, "dqdata_op");
    dqdata_op.set_attr<std::string>("qtype", "per_tensor");
    dqdata_op.set_attr<std::vector<int64_t>>("zps", {zp_src});
    dqdata_op.set_attr<std::vector<float>>("scales", {scale_src});
    dqdata_op.set_attr<int64_t>("axis", 0);

    impl::op_t dqweight_op(2, impl::op_kind::Dequantize, "dqweight_op");
    dqweight_op.set_attr<std::string>("qtype", "per_tensor");
    dqweight_op.set_attr<std::vector<int64_t>>("zps", {zp_wei});
    dqweight_op.set_attr<std::vector<float>>("scales", {scale_wei});
    dqweight_op.set_attr<int64_t>("axis", 0);

    impl::op_t matmul_op(3, impl::op_kind::MatMul, "matmul_op");
    matmul_op.set_attr<bool>("transpose_a", false);
    matmul_op.set_attr<bool>("transpose_b", false);

    impl::op_t relu_op(4, impl::op_kind::ReLU, "relu_op");

    impl::op_t qout_op(5, impl::op_kind::Quantize, "qout_op");
    qout_op.set_attr<std::string>("qtype", "per_tensor");
    qout_op.set_attr<std::vector<int64_t>>("zps", {zp_out});
    qout_op.set_attr<std::vector<float>>("scales", {scale_out});
    qout_op.set_attr<int64_t>("axis", 0);

    // prepare logical tensor
    impl::logical_tensor_t src_u8
            = utils::logical_tensor_init(1, src_shape, impl::data_type::u8);
    impl::logical_tensor_t src_f32_dq
            = utils::logical_tensor_init(2, src_shape, impl::data_type::f32);
    impl::logical_tensor_t weight_s8
            = utils::logical_tensor_init(4, weight_shape, impl::data_type::s8);
    impl::logical_tensor_t weight_f32_dq
            = utils::logical_tensor_init(5, weight_shape, impl::data_type::f32);
    impl::logical_tensor_t dst_f32
            = utils::logical_tensor_init(7, dst_shape, impl::data_type::f32);
    impl::logical_tensor_t dst_relu_f32
            = utils::logical_tensor_init(8, dst_shape, impl::data_type::f32);
    impl::logical_tensor_t dst_s8
            = utils::logical_tensor_init(9, dst_shape, impl::data_type::s8);

    dqdata_op.add_input(src_u8);
    dqdata_op.add_output(src_f32_dq);

    dqweight_op.add_input(weight_s8);
    dqweight_op.add_output(weight_f32_dq);

    matmul_op.add_input(src_f32_dq);
    matmul_op.add_input(weight_f32_dq);
    matmul_op.add_output(dst_f32);

    relu_op.add_input(dst_f32);
    relu_op.add_output(dst_relu_f32);

    qout_op.add_input(dst_relu_f32);
    qout_op.add_output(dst_s8);

    impl::graph_t g;
    g.add_op(&dqdata_op);
    g.add_op(&dqweight_op);
    g.add_op(&matmul_op);
    g.add_op(&relu_op);
    g.add_op(&qout_op);
    g.build_graph();

    impl::tensor_t src_u8_ts(src_u8, &engine, src_data.data());
    impl::tensor_t weight_s8_ts(weight_s8, &engine, weight_data.data());
    // -------------------------case 1----------------------------------
    test::vector<int8_t> case1_out_data(product(dst_shape));
    impl::tensor_t dst_s8_ts(dst_s8, &engine, case1_out_data.data());
    ASSERT_EQ(
            run_graph(g, {src_u8_ts, weight_s8_ts}, {dst_s8_ts}, engine, strm),
            impl::status::success);
    // -------------------------case 2----------------------------------
    impl::pass::pass_base_ptr apass = get_pass("int8_matmul_relu_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];
    ASSERT_TRUE(part != nullptr);

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> lt_ins {&src_u8, &weight_s8};
    std::vector<const impl::logical_tensor_t *> lt_outs {&dst_s8};

    p.compile(&cp, lt_ins, lt_outs, &engine);

    test::vector<int8_t> case2_out_data(product(dst_shape));
    impl::tensor_t dst_s8_case2_ts(dst_s8, &engine, case2_out_data.data());
    cp.execute(&strm, {src_u8_ts, weight_s8_ts}, {dst_s8_case2_ts});
    strm.wait();

    static auto isa = dnnl_get_effective_cpu_isa();
    if (isa < dnnl_cpu_isa_avx512_core_vnni)
        ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.1f,
                /*atol*/ 1.f));
    else
        ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.01f,
                /*atol*/ 1.f));
}

TEST(operator_kernel, interpolate_forkward_nearest) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src {-2.0, -1.5, -1.0, -0.5};
    test::vector<float> dst {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    test::vector<float> ref_dst {
            -2.f, -1.5f, -1.5f, -1.f, -0.5f, -0.5f, -1.f, -0.5f, -0.5f};

    impl::op_t op(impl::op_kind::Interpolate);
    op.set_attr<std::string>("mode", "nearest");
    auto &op_factory = get_dnnl_kernel_registry();
    auto kernel = op_factory.create_kernel(op);
    ASSERT_TRUE(kernel);

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {1, 1, 3, 3}, impl::data_type::f32, impl::layout_type::any);

    // compile the relu backward operator
    std::vector<impl::logical_tensor_t> inputs {src_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    kernel->compile(&op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t dst_ts(outputs[0], &eng, dst.data());

    impl::stream_t &strm = get_stream();
    kernel->execute(&op, &strm, {src_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, interpolate_forkward_linear) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src {-2.0, -1.5, -1.0, -0.5};
    test::vector<float> dst {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    test::vector<float> ref_dst {
            -2.f, -1.75f, -1.5f, -1.5f, -1.25f, -1.f, -1.f, -0.75f, -0.5f};

    impl::op_t op(impl::op_kind::Interpolate);
    op.set_attr<std::string>("mode", "linear");
    auto &op_factory = get_dnnl_kernel_registry();
    auto kernel = op_factory.create_kernel(op);
    ASSERT_TRUE(kernel);

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {1, 1, 3, 3}, impl::data_type::f32, impl::layout_type::any);

    // compile the relu backward operator
    std::vector<impl::logical_tensor_t> inputs {src_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    kernel->compile(&op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t dst_ts(outputs[0], &eng, dst.data());

    impl::stream_t &strm = get_stream();
    kernel->execute(&op, &strm, {src_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, interpolate_backward_nearest) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src {-2.0, -1.5, -1.0, -0.5};
    test::vector<float> diff_dst {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    test::vector<float> diff_src {0.0, 0.0, 0.0, 0.0};
    test::vector<float> ref_diff_src {0.f, 3.f, 9.f, 24.f};

    impl::op_t op(impl::op_kind::InterpolateBackprop);
    op.set_attr<std::string>("mode", "nearest");
    auto &op_factory = get_dnnl_kernel_registry();
    auto kernel = op_factory.create_kernel(op);
    ASSERT_TRUE(kernel);

    impl::logical_tensor_t diff_dst_lt
            = utils::logical_tensor_init(0, {1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(1, {1, 1, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t diff_src_lt = utils::logical_tensor_init(
            2, {1, 1, 2, 2}, impl::data_type::f32, impl::layout_type::any);

    // compile the relu backward operator
    std::vector<impl::logical_tensor_t> inputs {src_lt, diff_dst_lt};
    std::vector<impl::logical_tensor_t> outputs {diff_src_lt};

    kernel->compile(&op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t diff_dst_ts(diff_dst_lt, &eng, diff_dst.data());
    impl::tensor_t diff_src_ts(outputs[0], &eng, diff_src.data());

    impl::stream_t &strm = get_stream();
    kernel->execute(&op, &strm, {src_ts, diff_dst_ts}, {diff_src_ts});
    strm.wait();
    for (size_t i = 0; i < diff_src.size(); ++i) {
        ASSERT_FLOAT_EQ(diff_src[i], ref_diff_src[i]);
    }
}

TEST(operator_kernel, interpolate_backward_linear) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src {-2.0, -1.5, -1.0, -0.5};
    test::vector<float> diff_dst {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    test::vector<float> diff_src {0.0, 0.0, 0.0, 0.0};
    test::vector<float> ref_diff_src {3.f, 6.f, 12.f, 15.f};

    impl::op_t op(impl::op_kind::InterpolateBackprop);
    op.set_attr<std::string>("mode", "linear");
    auto &op_factory = get_dnnl_kernel_registry();
    auto kernel = op_factory.create_kernel(op);
    ASSERT_TRUE(kernel);

    impl::logical_tensor_t diff_dst_lt
            = utils::logical_tensor_init(0, {1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(1, {1, 1, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t diff_src_lt = utils::logical_tensor_init(
            2, {1, 1, 2, 2}, impl::data_type::f32, impl::layout_type::any);

    // compile the relu backward operator
    std::vector<impl::logical_tensor_t> inputs {src_lt, diff_dst_lt};
    std::vector<impl::logical_tensor_t> outputs {diff_src_lt};

    kernel->compile(&op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t diff_dst_ts(diff_dst_lt, &eng, diff_dst.data());
    impl::tensor_t diff_src_ts(outputs[0], &eng, diff_src.data());

    impl::stream_t &strm = get_stream();
    kernel->execute(&op, &strm, {src_ts, diff_dst_ts}, {diff_src_ts});
    strm.wait();
    for (size_t i = 0; i < diff_src.size(); ++i) {
        ASSERT_FLOAT_EQ(diff_src[i], ref_diff_src[i]);
    }
}

TEST(int8_subgraph_mode, int8_matmul_bias_sum_ndx2d) {
    // compare results between:
    // case 1: [quantize] - [dequantize] - [fp32_matmul] - [quantize]
    // case 2: [quantize] - [int8_matmul]
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    if (engine.kind() == impl::engine_kind::gpu) return;

    std::vector<std::string> qtypes {"per_tensor", "per_channel"};
    std::vector<std::string> other_qtypes = {"symmetric", "asymmetric"};
    std::vector<std::vector<int64_t>> src_shapes {
            {3, 3, 3, 8, 4}, {3, 3, 8, 4}, {3, 8, 4}, {8, 4}, {4}};
    std::vector<std::vector<int64_t>> weight_shapes {{4, 2}};
    std::vector<std::vector<int64_t>> dst_shapes {
            {3, 3, 3, 8, 2}, {3, 3, 8, 2}, {3, 8, 2}, {8, 2}, {2}};
    for_(const auto &qtype : qtypes)
    for_(size_t i = 0; i < src_shapes.size(); ++i)
    for_(const auto &other_qtype : other_qtypes)
    for (size_t j = 0; j < weight_shapes.size(); ++j) {
        // prepare fp32 data
        std::vector<int64_t> src_shape = src_shapes[i];
        std::vector<int64_t> weight_shape = weight_shapes[j];
        std::vector<int64_t> bias_shape {2};
        std::vector<int64_t> dst_shape = dst_shapes[i];

        test::vector<uint8_t> src_data(product(src_shape));
        test::vector<int8_t> weight_data(product(weight_shape));
        test::vector<float> bias_data(product(bias_shape));
        test::vector<int8_t> other_data(product(dst_shape));

        // random generate src, weight and bias data
        // random seed = 7
        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
        std::uniform_real_distribution<float> u8_distribution(0.0f, 255.0f);
        std::uniform_real_distribution<float> s8_distribution(-127.0f, 128.0f);
        std::generate(src_data.begin(), src_data.end(), [&]() {
            return static_cast<uint8_t>(u8_distribution(generator));
        });
        std::generate(weight_data.begin(), weight_data.end(), [&]() {
            return static_cast<int8_t>(s8_distribution(generator));
        });
        std::generate(bias_data.begin(), bias_data.end(),
                [&]() { return f32_distribution(generator); });
        std::generate(other_data.begin(), other_data.end(), [&]() {
            return static_cast<int8_t>(s8_distribution(generator));
        });
        float scale_src = 1 / 255.f; // map to 0~255
        float scale_other = 1 / 127.f;
        float scale_out = 1;
        int64_t zp_src = 0;
        int64_t zp_other = other_qtype == "symmetric" ? 0 : 128;
        int64_t zp_out = 78;

        size_t scales_wei_sizes = qtype == "per_tensor" ? 1 : dst_shape.back();
        std::vector<float> scale_wei(scales_wei_sizes, 1 / 127.f);
        std::vector<int64_t> zp_wei(scales_wei_sizes, 0);

        impl::op_t dqdata_op(1, impl::op_kind::Dequantize, "dqdata_op");
        dqdata_op.set_attr<std::string>("qtype", "per_tensor");
        dqdata_op.set_attr<std::vector<int64_t>>("zps", {zp_src});
        dqdata_op.set_attr<std::vector<float>>("scales", {scale_src});
        dqdata_op.set_attr<int64_t>("axis", 0);

        impl::op_t dqweight_op(2, impl::op_kind::Dequantize, "dqweight_op");
        dqweight_op.set_attr<std::string>("qtype", qtype);
        dqweight_op.set_attr<std::vector<int64_t>>("zps", zp_wei);
        dqweight_op.set_attr<std::vector<float>>("scales", scale_wei);
        dqweight_op.set_attr<int64_t>("axis", 1);

        impl::op_t matmul_op(3, impl::op_kind::MatMul, "matmul_op");
        matmul_op.set_attr<bool>("transpose_a", false);
        matmul_op.set_attr<bool>("transpose_b", false);

        impl::op_t qout_op(4, impl::op_kind::Quantize, "qout_op");
        qout_op.set_attr<std::string>("qtype", "per_tensor");
        qout_op.set_attr<std::vector<int64_t>>("zps", {zp_out});
        qout_op.set_attr<std::vector<float>>("scales", {scale_out});
        qout_op.set_attr<int64_t>("axis", 0);

        impl::op_t dqother_op(5, impl::op_kind::Dequantize, "dqother_op");
        dqother_op.set_attr<std::string>("qtype", "per_tensor");
        dqother_op.set_attr<std::vector<int64_t>>("zps", {zp_other});
        dqother_op.set_attr<std::vector<float>>("scales", {scale_other});
        dqother_op.set_attr<int64_t>("axis", 0);

        impl::op_t add_op(6, impl::op_kind::Add, "add_op");

        // prepare logical tensor
        impl::logical_tensor_t src_u8
                = utils::logical_tensor_init(1, src_shape, impl::data_type::u8);
        impl::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                2, src_shape, impl::data_type::f32);
        impl::logical_tensor_t weight_s8 = utils::logical_tensor_init(
                4, weight_shape, impl::data_type::s8);
        impl::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
                5, weight_shape, impl::data_type::f32);
        impl::logical_tensor_t bias_f32 = utils::logical_tensor_init(
                6, bias_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_f32 = utils::logical_tensor_init(
                7, dst_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_s8
                = utils::logical_tensor_init(8, dst_shape, impl::data_type::s8);
        impl::logical_tensor_t other_s8 = utils::logical_tensor_init(
                10, dst_shape, impl::data_type::s8);
        impl::logical_tensor_t other_f32_dq = utils::logical_tensor_init(
                11, dst_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_add_f32 = utils::logical_tensor_init(
                12, dst_shape, impl::data_type::f32);

        dqdata_op.add_input(src_u8);
        dqdata_op.add_output(src_f32_dq);

        dqweight_op.add_input(weight_s8);
        dqweight_op.add_output(weight_f32_dq);

        matmul_op.add_input(src_f32_dq);
        matmul_op.add_input(weight_f32_dq);
        matmul_op.add_input(bias_f32);
        matmul_op.add_output(dst_f32);

        dqother_op.add_input(other_s8);
        dqother_op.add_output(other_f32_dq);

        add_op.add_input(dst_f32);
        add_op.add_input(other_f32_dq);
        add_op.add_output(dst_add_f32);

        qout_op.add_input(dst_add_f32);
        qout_op.add_output(dst_s8);

        impl::graph_t g;
        g.add_op(&dqdata_op);
        g.add_op(&dqweight_op);
        g.add_op(&matmul_op);
        g.add_op(&dqother_op);
        g.add_op(&add_op);
        g.add_op(&qout_op);
        g.build_graph();

        impl::tensor_t src_u8_ts(src_u8, &engine, src_data.data());
        impl::tensor_t weight_s8_ts(weight_s8, &engine, weight_data.data());
        impl::tensor_t bias_f32_ts(bias_f32, &engine, bias_data.data());
        impl::tensor_t other_s8_ts(other_s8, &engine, other_data.data());
        // -------------------------case 1----------------------------------
        test::vector<int8_t> case1_out_data(product(dst_shape));
        impl::tensor_t dst_s8_ts(dst_s8, &engine, case1_out_data.data());
        ASSERT_EQ(run_graph(g,
                          {src_u8_ts, weight_s8_ts, bias_f32_ts, other_s8_ts},
                          {dst_s8_ts}, engine, strm),
                impl::status::success);
        // -------------------------case 2----------------------------------
        impl::pass::pass_base_ptr apass
                = get_pass("int8_matmul_bias_add_fusion");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1);
        auto part = g.get_partitions()[0];

        // compile
        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> lt_ins {
                &src_u8, &weight_s8, &bias_f32, &other_s8};
        std::vector<const impl::logical_tensor_t *> lt_outs {&dst_s8};

        p.compile(&cp, lt_ins, lt_outs, &engine);

        test::vector<int8_t> case2_out_data(product(dst_shape));
        impl::tensor_t dst_s8_case2_ts(dst_s8, &engine, case2_out_data.data());
        cp.execute(&strm, {src_u8_ts, weight_s8_ts, bias_f32_ts, other_s8_ts},
                {dst_s8_case2_ts});
        strm.wait();

        static auto isa = dnnl_get_effective_cpu_isa();
        if (isa < dnnl_cpu_isa_avx512_core_vnni)
            ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.1f,
                    /*atol*/ 1.f));
        else
            ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.01f,
                    /*atol*/ 1.f));
    }
}

TEST(int8_subgraph_mode, x8s8f32_matmul_bias_sum_ndx2d) {
    // compare results between:
    // case 1: [quantize] - [dequantize] - [fp32_matmul]
    // case 2: [quantize] - [int8_matmul]
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    if (engine.kind() == impl::engine_kind::gpu) return;

    std::vector<std::string> qtypes {"per_tensor", "per_channel"};
    std::vector<std::vector<int64_t>> src_shapes {
            {3, 3, 3, 8, 4}, {3, 3, 8, 4}, {3, 8, 4}, {8, 4}, {4}};
    std::vector<std::vector<int64_t>> weight_shapes {{4, 2}};
    std::vector<std::vector<int64_t>> dst_shapes {
            {3, 3, 3, 8, 2}, {3, 3, 8, 2}, {3, 8, 2}, {8, 2}, {2}};
    for_(const auto &qtype : qtypes)
    for_(size_t i = 0; i < src_shapes.size(); ++i)
    for (size_t j = 0; j < weight_shapes.size(); ++j) {
        // prepare fp32 data
        std::vector<int64_t> src_shape = src_shapes[i];
        std::vector<int64_t> weight_shape = weight_shapes[j];
        std::vector<int64_t> bias_shape {2};
        std::vector<int64_t> dst_shape = dst_shapes[i];

        test::vector<uint8_t> src_data(product(src_shape));
        test::vector<int8_t> weight_data(product(weight_shape));
        test::vector<float> bias_data(product(bias_shape));
        test::vector<int8_t> other_data(product(dst_shape));

        // random generate src, weight and bias data
        // random seed = 7
        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
        std::uniform_real_distribution<float> u8_distribution(0.0f, 255.0f);
        std::uniform_real_distribution<float> s8_distribution(-127.0f, 128.0f);
        std::generate(src_data.begin(), src_data.end(), [&]() {
            return static_cast<uint8_t>(u8_distribution(generator));
        });
        std::generate(weight_data.begin(), weight_data.end(), [&]() {
            return static_cast<int8_t>(s8_distribution(generator));
        });
        std::generate(bias_data.begin(), bias_data.end(),
                [&]() { return f32_distribution(generator); });
        std::generate(other_data.begin(), other_data.end(), [&]() {
            return static_cast<int8_t>(s8_distribution(generator));
        });
        float scale_src = 1 / 255.f; // map to 0~255
        float scale_other = 1 / 127.f;
        float scale_out = 1;
        int64_t zp_src = 0;
        int64_t zp_other = 0;
        int64_t zp_out = 78;

        size_t scales_wei_sizes = qtype == "per_tensor" ? 1 : dst_shape.back();
        std::vector<float> scale_wei(scales_wei_sizes, 1 / 127.f);
        std::vector<int64_t> zp_wei(scales_wei_sizes, 0);

        impl::op_t dqdata_op(1, impl::op_kind::Dequantize, "dqdata_op");
        dqdata_op.set_attr<std::string>("qtype", "per_tensor");
        dqdata_op.set_attr<std::vector<int64_t>>("zps", {zp_src});
        dqdata_op.set_attr<std::vector<float>>("scales", {scale_src});
        dqdata_op.set_attr<int64_t>("axis", 0);

        impl::op_t dqweight_op(2, impl::op_kind::Dequantize, "dqweight_op");
        dqweight_op.set_attr<std::string>("qtype", qtype);
        dqweight_op.set_attr<std::vector<int64_t>>("zps", zp_wei);
        dqweight_op.set_attr<std::vector<float>>("scales", scale_wei);
        dqweight_op.set_attr<int64_t>("axis", 1);

        impl::op_t matmul_op(3, impl::op_kind::MatMul, "matmul_op");
        matmul_op.set_attr<bool>("transpose_a", false);
        matmul_op.set_attr<bool>("transpose_b", false);

        impl::op_t dqother_op(4, impl::op_kind::Dequantize, "dqother_op");
        dqother_op.set_attr<std::string>("qtype", "per_tensor");
        dqother_op.set_attr<std::vector<int64_t>>("zps", {zp_other});
        dqother_op.set_attr<std::vector<float>>("scales", {scale_other});
        dqother_op.set_attr<int64_t>("axis", 0);

        impl::op_t add_op(5, impl::op_kind::Add, "add_op");

        // prepare logical tensor
        impl::logical_tensor_t src_u8
                = utils::logical_tensor_init(1, src_shape, impl::data_type::u8);
        impl::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                2, src_shape, impl::data_type::f32);
        impl::logical_tensor_t weight_s8 = utils::logical_tensor_init(
                4, weight_shape, impl::data_type::s8);
        impl::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
                5, weight_shape, impl::data_type::f32);
        impl::logical_tensor_t bias_f32 = utils::logical_tensor_init(
                6, bias_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_f32 = utils::logical_tensor_init(
                7, dst_shape, impl::data_type::f32);
        impl::logical_tensor_t other_s8
                = utils::logical_tensor_init(9, dst_shape, impl::data_type::s8);
        impl::logical_tensor_t other_f32_dq = utils::logical_tensor_init(
                10, dst_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_add_f32 = utils::logical_tensor_init(
                11, dst_shape, impl::data_type::f32);

        dqdata_op.add_input(src_u8);
        dqdata_op.add_output(src_f32_dq);

        dqweight_op.add_input(weight_s8);
        dqweight_op.add_output(weight_f32_dq);

        matmul_op.add_input(src_f32_dq);
        matmul_op.add_input(weight_f32_dq);
        matmul_op.add_input(bias_f32);
        matmul_op.add_output(dst_f32);

        dqother_op.add_input(other_s8);
        dqother_op.add_output(other_f32_dq);

        add_op.add_input(dst_f32);
        add_op.add_input(other_f32_dq);
        add_op.add_output(dst_add_f32);

        impl::graph_t g;
        g.add_op(&dqdata_op);
        g.add_op(&dqweight_op);
        g.add_op(&matmul_op);
        g.add_op(&dqother_op);
        g.add_op(&add_op);
        g.build_graph();

        impl::tensor_t src_u8_ts(src_u8, &engine, src_data.data());
        impl::tensor_t weight_s8_ts(weight_s8, &engine, weight_data.data());
        impl::tensor_t bias_f32_ts(bias_f32, &engine, bias_data.data());
        impl::tensor_t other_s8_ts(other_s8, &engine, other_data.data());
        // -------------------------case 1----------------------------------
        test::vector<float> case1_out_data(product(dst_shape));
        impl::tensor_t dst_f32_ts(dst_add_f32, &engine, case1_out_data.data());
        ASSERT_EQ(run_graph(g,
                          {src_u8_ts, weight_s8_ts, bias_f32_ts, other_s8_ts},
                          {dst_f32_ts}, engine, strm),
                impl::status::success);
        // -------------------------case 2----------------------------------
        impl::pass::pass_base_ptr apass
                = get_pass("x8s8f32_matmul_bias_add_fusion");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1);
        auto part = g.get_partitions()[0];

        // compile
        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> lt_ins {
                &src_u8, &weight_s8, &bias_f32, &other_s8};
        std::vector<const impl::logical_tensor_t *> lt_outs {&dst_add_f32};

        p.compile(&cp, lt_ins, lt_outs, &engine);

        test::vector<float> case2_out_data(product(dst_shape));
        impl::tensor_t dst_f32_case2_ts(
                dst_add_f32, &engine, case2_out_data.data());
        cp.execute(&strm, {src_u8_ts, weight_s8_ts, bias_f32_ts, other_s8_ts},
                {dst_f32_case2_ts});
        strm.wait();

        static auto isa = dnnl_get_effective_cpu_isa();
        if (isa < dnnl_cpu_isa_avx512_core_vnni)
            ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.1f,
                    /*atol*/ 1.f));
        else
            ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.01f,
                    /*atol*/ 1.f));
    }
}

TEST(int8_subgraph_mode, x8s8f32_matmul_bias_ndx2d) {
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    if (engine.kind() == impl::engine_kind::gpu) return;

    std::vector<std::string> qtypes {"per_tensor", "per_channel"};
    std::vector<std::vector<int64_t>> src_shapes {
            {3, 3, 3, 8, 4}, {3, 3, 8, 4}, {3, 8, 4}, {8, 4}, {4}};
    std::vector<std::vector<int64_t>> weight_shapes {{4, 2}};
    std::vector<std::vector<int64_t>> dst_shapes {
            {3, 3, 3, 8, 2}, {3, 3, 8, 2}, {3, 8, 2}, {8, 2}, {2}};
    for_(const auto &qtype : qtypes)
    for_(size_t i = 0; i < src_shapes.size(); ++i)
    for (size_t j = 0; j < weight_shapes.size(); ++j) {
        // prepare fp32 data
        std::vector<int64_t> src_shape = src_shapes[i];
        std::vector<int64_t> weight_shape = weight_shapes[j];
        std::vector<int64_t> bias_shape {2};
        std::vector<int64_t> dst_shape = dst_shapes[i];

        test::vector<uint8_t> src_data(product(src_shape));
        test::vector<int8_t> weight_data(product(weight_shape));
        test::vector<float> bias_data(product(bias_shape));

        // random generate src, weight and bias data
        // random seed = 7
        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
        std::uniform_real_distribution<float> u8_distribution(0.0f, 255.0f);
        std::uniform_real_distribution<float> s8_distribution(-127.0f, 128.0f);
        std::generate(src_data.begin(), src_data.end(), [&]() {
            return static_cast<uint8_t>(u8_distribution(generator));
        });
        std::generate(weight_data.begin(), weight_data.end(), [&]() {
            return static_cast<int8_t>(s8_distribution(generator));
        });
        std::generate(bias_data.begin(), bias_data.end(),
                [&]() { return f32_distribution(generator); });
        float scale_src = 1 / 255.f; // map to 0~255
        float scale_other = 1 / 127.f;
        float scale_out = 1;
        int64_t zp_src = 0;
        int64_t zp_other = 0;
        int64_t zp_out = 78;

        size_t scales_wei_sizes = qtype == "per_tensor" ? 1 : dst_shape.back();
        std::vector<float> scale_wei(scales_wei_sizes, 1 / 127.f);
        std::vector<int64_t> zp_wei(scales_wei_sizes, 0);

        impl::op_t dqdata_op(1, impl::op_kind::Dequantize, "dqdata_op");
        dqdata_op.set_attr<std::string>("qtype", "per_tensor");
        dqdata_op.set_attr<std::vector<int64_t>>("zps", {zp_src});
        dqdata_op.set_attr<std::vector<float>>("scales", {scale_src});
        dqdata_op.set_attr<int64_t>("axis", 0);

        impl::op_t dqweight_op(2, impl::op_kind::Dequantize, "dqweight_op");
        dqweight_op.set_attr<std::string>("qtype", qtype);
        dqweight_op.set_attr<std::vector<int64_t>>("zps", zp_wei);
        dqweight_op.set_attr<std::vector<float>>("scales", scale_wei);
        dqweight_op.set_attr<int64_t>("axis", 1);

        impl::op_t matmul_op(3, impl::op_kind::MatMul, "matmul_op");
        matmul_op.set_attr<bool>("transpose_a", false);
        matmul_op.set_attr<bool>("transpose_b", false);

        // prepare logical tensor
        impl::logical_tensor_t src_u8
                = utils::logical_tensor_init(1, src_shape, impl::data_type::u8);
        impl::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                2, src_shape, impl::data_type::f32);
        impl::logical_tensor_t weight_s8 = utils::logical_tensor_init(
                4, weight_shape, impl::data_type::s8);
        impl::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
                5, weight_shape, impl::data_type::f32);
        impl::logical_tensor_t bias_f32 = utils::logical_tensor_init(
                6, bias_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_f32 = utils::logical_tensor_init(
                7, dst_shape, impl::data_type::f32);

        dqdata_op.add_input(src_u8);
        dqdata_op.add_output(src_f32_dq);

        dqweight_op.add_input(weight_s8);
        dqweight_op.add_output(weight_f32_dq);

        matmul_op.add_input(src_f32_dq);
        matmul_op.add_input(weight_f32_dq);
        matmul_op.add_input(bias_f32);
        matmul_op.add_output(dst_f32);

        impl::graph_t g;
        g.add_op(&dqdata_op);
        g.add_op(&dqweight_op);
        g.add_op(&matmul_op);
        g.build_graph();

        impl::tensor_t src_u8_ts(src_u8, &engine, src_data.data());
        impl::tensor_t weight_s8_ts(weight_s8, &engine, weight_data.data());
        impl::tensor_t bias_f32_ts(bias_f32, &engine, bias_data.data());
        // -------------------------case 1----------------------------------
        test::vector<float> case1_out_data(product(dst_shape));
        impl::tensor_t dst_f32_ts(dst_f32, &engine, case1_out_data.data());
        ASSERT_EQ(run_graph(g, {src_u8_ts, weight_s8_ts, bias_f32_ts},
                          {dst_f32_ts}, engine, strm),
                impl::status::success);
        // -------------------------case 2----------------------------------
        impl::pass::pass_base_ptr apass
                = get_pass("x8s8f32_matmul_bias_fusion");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1);
        auto part = g.get_partitions()[0];

        // compile
        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> lt_ins {
                &src_u8, &weight_s8, &bias_f32};
        std::vector<const impl::logical_tensor_t *> lt_outs {&dst_f32};

        p.compile(&cp, lt_ins, lt_outs, &engine);

        test::vector<float> case2_out_data(product(dst_shape));
        impl::tensor_t dst_f32_case2_ts(
                dst_f32, &engine, case2_out_data.data());
        cp.execute(&strm, {src_u8_ts, weight_s8_ts, bias_f32_ts},
                {dst_f32_case2_ts});
        strm.wait();

        static auto isa = dnnl_get_effective_cpu_isa();
        if (isa < dnnl_cpu_isa_avx512_core_vnni)
            ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.1f,
                    /*atol*/ 1.f));
        else
            ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.01f,
                    /*atol*/ 1.f));
    }
}

TEST(int8_subgraph_mode, x8s8f32_matmul_ndx2d) {
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    if (engine.kind() == impl::engine_kind::gpu) return;

    std::vector<std::string> qtypes {"per_tensor", "per_channel"};
    std::vector<std::vector<int64_t>> src_shapes {
            {3, 3, 3, 8, 4}, {3, 3, 8, 4}, {3, 8, 4}, {8, 4}, {4}};
    std::vector<std::vector<int64_t>> weight_shapes {{4, 2}};
    std::vector<std::vector<int64_t>> dst_shapes {
            {3, 3, 3, 8, 2}, {3, 3, 8, 2}, {3, 8, 2}, {8, 2}, {2}};
    for_(const auto &qtype : qtypes)
    for_(size_t i = 0; i < src_shapes.size(); ++i)
    for (size_t j = 0; j < weight_shapes.size(); ++j) {
        // prepare fp32 data
        std::vector<int64_t> src_shape = src_shapes[i];
        std::vector<int64_t> weight_shape = weight_shapes[j];
        std::vector<int64_t> dst_shape = dst_shapes[i];

        test::vector<uint8_t> src_data(product(src_shape));
        test::vector<int8_t> weight_data(product(weight_shape));

        // random generate src, weight and bias data
        // random seed = 7
        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> u8_distribution(0.0f, 255.0f);
        std::uniform_real_distribution<float> s8_distribution(-127.0f, 128.0f);
        std::generate(src_data.begin(), src_data.end(), [&]() {
            return static_cast<uint8_t>(u8_distribution(generator));
        });
        std::generate(weight_data.begin(), weight_data.end(), [&]() {
            return static_cast<int8_t>(s8_distribution(generator));
        });
        float scale_src = 1 / 255.f; // map to 0~255
        float scale_other = 1 / 127.f;
        float scale_out = 1;
        int64_t zp_src = 0;
        int64_t zp_other = 0;
        int64_t zp_out = 78;

        size_t scales_wei_sizes = qtype == "per_tensor" ? 1 : dst_shape.back();
        std::vector<float> scale_wei(scales_wei_sizes, 1 / 127.f);
        std::vector<int64_t> zp_wei(scales_wei_sizes, 0);

        impl::op_t dqdata_op(1, impl::op_kind::Dequantize, "dqdata_op");
        dqdata_op.set_attr<std::string>("qtype", "per_tensor");
        dqdata_op.set_attr<std::vector<int64_t>>("zps", {zp_src});
        dqdata_op.set_attr<std::vector<float>>("scales", {scale_src});
        dqdata_op.set_attr<int64_t>("axis", 0);

        impl::op_t dqweight_op(2, impl::op_kind::Dequantize, "dqweight_op");
        dqweight_op.set_attr<std::string>("qtype", qtype);
        dqweight_op.set_attr<std::vector<int64_t>>("zps", zp_wei);
        dqweight_op.set_attr<std::vector<float>>("scales", scale_wei);
        dqweight_op.set_attr<int64_t>("axis", 1);

        impl::op_t matmul_op(3, impl::op_kind::MatMul, "matmul_op");
        matmul_op.set_attr<bool>("transpose_a", false);
        matmul_op.set_attr<bool>("transpose_b", false);

        // prepare logical tensor
        impl::logical_tensor_t src_u8
                = utils::logical_tensor_init(1, src_shape, impl::data_type::u8);
        impl::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                2, src_shape, impl::data_type::f32);
        impl::logical_tensor_t weight_s8 = utils::logical_tensor_init(
                4, weight_shape, impl::data_type::s8);
        impl::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
                5, weight_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_f32 = utils::logical_tensor_init(
                7, dst_shape, impl::data_type::f32);

        dqdata_op.add_input(src_u8);
        dqdata_op.add_output(src_f32_dq);

        dqweight_op.add_input(weight_s8);
        dqweight_op.add_output(weight_f32_dq);

        matmul_op.add_input(src_f32_dq);
        matmul_op.add_input(weight_f32_dq);
        matmul_op.add_output(dst_f32);

        impl::graph_t g;
        g.add_op(&dqdata_op);
        g.add_op(&dqweight_op);
        g.add_op(&matmul_op);
        g.build_graph();

        impl::tensor_t src_u8_ts(src_u8, &engine, src_data.data());
        impl::tensor_t weight_s8_ts(weight_s8, &engine, weight_data.data());
        // -------------------------case 1----------------------------------
        test::vector<float> case1_out_data(product(dst_shape));
        impl::tensor_t dst_f32_ts(dst_f32, &engine, case1_out_data.data());
        ASSERT_EQ(run_graph(g, {src_u8_ts, weight_s8_ts}, {dst_f32_ts}, engine,
                          strm),
                impl::status::success);
        // -------------------------case 2----------------------------------
        impl::pass::pass_base_ptr apass = get_pass("x8x8f32_matmul_fusion");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1);
        auto part = g.get_partitions()[0];

        // compile
        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> lt_ins {
                &src_u8, &weight_s8};
        std::vector<const impl::logical_tensor_t *> lt_outs {&dst_f32};

        p.compile(&cp, lt_ins, lt_outs, &engine);

        test::vector<float> case2_out_data(product(dst_shape));
        impl::tensor_t dst_f32_case2_ts(
                dst_f32, &engine, case2_out_data.data());
        cp.execute(&strm, {src_u8_ts, weight_s8_ts}, {dst_f32_case2_ts});
        strm.wait();

        static auto isa = dnnl_get_effective_cpu_isa();
        if (isa < dnnl_cpu_isa_avx512_core_vnni)
            ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.1f,
                    /*atol*/ 1.f));
        else
            ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.01f,
                    /*atol*/ 1.f));
    }
}

TEST(int8_subgraph_mode, x8s8f32_matmul_bias_gelu_ndx2d) {
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    if (engine.kind() == impl::engine_kind::gpu) return;

    std::vector<std::string> qtypes {"per_tensor", "per_channel"};
    std::vector<std::vector<int64_t>> src_shapes {
            {3, 3, 3, 8, 4}, {3, 3, 8, 4}, {3, 8, 4}, {8, 4}, {4}};
    std::vector<std::vector<int64_t>> weight_shapes {{4, 2}};
    std::vector<std::vector<int64_t>> dst_shapes {
            {3, 3, 3, 8, 2}, {3, 3, 8, 2}, {3, 8, 2}, {8, 2}, {2}};
    for_(const auto &qtype : qtypes)
    for_(size_t i = 0; i < src_shapes.size(); ++i)
    for (size_t j = 0; j < weight_shapes.size(); ++j) {
        // prepare fp32 data
        std::vector<int64_t> src_shape = src_shapes[i];
        std::vector<int64_t> weight_shape = weight_shapes[j];
        std::vector<int64_t> bias_shape {2};
        std::vector<int64_t> dst_shape = dst_shapes[i];

        test::vector<uint8_t> src_data(product(src_shape));
        test::vector<int8_t> weight_data(product(weight_shape));
        test::vector<float> bias_data(product(bias_shape));

        // random generate src, weight and bias data
        // random seed = 7
        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
        std::uniform_real_distribution<float> u8_distribution(0.0f, 255.0f);
        std::uniform_real_distribution<float> s8_distribution(-127.0f, 128.0f);
        std::generate(src_data.begin(), src_data.end(), [&]() {
            return static_cast<uint8_t>(u8_distribution(generator));
        });
        std::generate(weight_data.begin(), weight_data.end(), [&]() {
            return static_cast<int8_t>(s8_distribution(generator));
        });
        std::generate(bias_data.begin(), bias_data.end(),
                [&]() { return f32_distribution(generator); });
        float scale_src = 1 / 255.f; // map to 0~255
        float scale_other = 1 / 127.f;
        float scale_out = 1;
        int64_t zp_src = 0;
        int64_t zp_other = 0;
        int64_t zp_out = 78;

        size_t scales_wei_sizes = qtype == "per_tensor" ? 1 : dst_shape.back();
        std::vector<float> scale_wei(scales_wei_sizes, 1 / 127.f);
        std::vector<int64_t> zp_wei(scales_wei_sizes, 0);

        impl::op_t dqdata_op(1, impl::op_kind::Dequantize, "dqdata_op");
        dqdata_op.set_attr<std::string>("qtype", "per_tensor");
        dqdata_op.set_attr<std::vector<int64_t>>("zps", {zp_src});
        dqdata_op.set_attr<std::vector<float>>("scales", {scale_src});
        dqdata_op.set_attr<int64_t>("axis", 0);

        impl::op_t dqweight_op(2, impl::op_kind::Dequantize, "dqweight_op");
        dqweight_op.set_attr<std::string>("qtype", qtype);
        dqweight_op.set_attr<std::vector<int64_t>>("zps", zp_wei);
        dqweight_op.set_attr<std::vector<float>>("scales", scale_wei);
        dqweight_op.set_attr<int64_t>("axis", 1);

        impl::op_t matmul_op(3, impl::op_kind::MatMul, "matmul_op");
        matmul_op.set_attr<bool>("transpose_a", false);
        matmul_op.set_attr<bool>("transpose_b", false);

        impl::op_t gelu_op(4, impl::op_kind::GELU, "gelu_op");

        // prepare logical tensor
        impl::logical_tensor_t src_u8
                = utils::logical_tensor_init(1, src_shape, impl::data_type::u8);
        impl::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                2, src_shape, impl::data_type::f32);
        impl::logical_tensor_t weight_s8 = utils::logical_tensor_init(
                4, weight_shape, impl::data_type::s8);
        impl::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
                5, weight_shape, impl::data_type::f32);
        impl::logical_tensor_t bias_f32 = utils::logical_tensor_init(
                6, bias_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_f32 = utils::logical_tensor_init(
                7, dst_shape, impl::data_type::f32);
        impl::logical_tensor_t gelu_f32 = utils::logical_tensor_init(
                8, dst_shape, impl::data_type::f32);

        dqdata_op.add_input(src_u8);
        dqdata_op.add_output(src_f32_dq);

        dqweight_op.add_input(weight_s8);
        dqweight_op.add_output(weight_f32_dq);

        matmul_op.add_input(src_f32_dq);
        matmul_op.add_input(weight_f32_dq);
        matmul_op.add_input(bias_f32);
        matmul_op.add_output(dst_f32);

        gelu_op.add_input(dst_f32);
        gelu_op.add_output(gelu_f32);

        impl::graph_t g;
        g.add_op(&dqdata_op);
        g.add_op(&dqweight_op);
        g.add_op(&matmul_op);
        g.add_op(&gelu_op);
        g.build_graph();

        impl::tensor_t src_u8_ts(src_u8, &engine, src_data.data());
        impl::tensor_t weight_s8_ts(weight_s8, &engine, weight_data.data());
        impl::tensor_t bias_f32_ts(bias_f32, &engine, bias_data.data());
        // -------------------------case 1----------------------------------
        test::vector<float> case1_out_data(product(dst_shape));
        impl::tensor_t dst_f32_ts(gelu_f32, &engine, case1_out_data.data());
        ASSERT_EQ(run_graph(g, {src_u8_ts, weight_s8_ts, bias_f32_ts},
                          {dst_f32_ts}, engine, strm),
                impl::status::success);
        // -------------------------case 2----------------------------------
        impl::pass::pass_base_ptr apass
                = get_pass("x8s8f32_matmul_bias_gelu_fusion");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1);
        auto part = g.get_partitions()[0];

        // compile
        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> lt_ins {
                &src_u8, &weight_s8, &bias_f32};
        std::vector<const impl::logical_tensor_t *> lt_outs {&gelu_f32};

        p.compile(&cp, lt_ins, lt_outs, &engine);

        test::vector<float> case2_out_data(product(dst_shape));
        impl::tensor_t dst_f32_case2_ts(
                gelu_f32, &engine, case2_out_data.data());
        cp.execute(&strm, {src_u8_ts, weight_s8_ts, bias_f32_ts},
                {dst_f32_case2_ts});
        strm.wait();

        static auto isa = dnnl_get_effective_cpu_isa();
        if (isa < dnnl_cpu_isa_avx512_core_vnni)
            ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.1f,
                    /*atol*/ 1.f));
        else
            ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.01f,
                    /*atol*/ 1.f));
    }
}

TEST(int8_subgraph_mode, int8_conv2d_sum_relu_get_inplace_pair) {
    using dims = impl::dnnl_impl::dims;

    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    if (engine.kind() == impl::engine_kind::gpu) return;

    std::vector<int64_t> groups = {1, 4};
    std::vector<std::string> weight_qtypes = {"per_tensor", "per_channel"};
    std::vector<std::string> other_qtypes = {"symmetric", "asymmetric"};

    for_(const auto &g : groups)
    for_(const auto &other_qtype : other_qtypes)
    for (const auto &wei_qtype : weight_qtypes) {
        int64_t in_channel = 8, out_channel = 8;
        int64_t kernel_size = 3;
        std::vector<int64_t> src_shape {1, in_channel, 112, 112};
        std::vector<int64_t> weight_shape {
                out_channel, in_channel / g, kernel_size, kernel_size};
        std::vector<int64_t> dst_shape {1, out_channel, 110, 110};

        float scale_src = 1 / 255.f; // map to 0~255
        float scale_other = 1 / 127.f;
        float scale_out = 1;
        int64_t zp_src = 0;
        int64_t zp_other = other_qtype == "symmetric" ? 0 : 128;
        int64_t zp_out = 78;

        size_t scale_size = wei_qtype == "per_tensor" ? 1 : out_channel;

        std::vector<float> scale_wei(scale_size, 1 / 127.f);
        std::vector<int64_t> zp_wei(scale_size, 0);

        impl::op_t dqdata_node(1, impl::op_kind::Dequantize, "dqdata_node");
        SET_Q_DQ_DATA_ATTR(dqdata_node)

        impl::op_t dqweight_node(3, impl::op_kind::Dequantize, "dqweight_node");
        SET_Q_DQ_WEIGHT_ATTR(dqweight_node)

        impl::op_t conv_node(4, impl::op_kind::Convolution, "conv_node");
        SET_CONV_ATTR(conv_node, 2)

        impl::op_t relu_node(5, impl::op_kind::ReLU, "relu_node");

        impl::op_t qout_node(6, impl::op_kind::Quantize, "qout_node");
        SET_Q_DQ_OUT_ATTR(qout_node)

        impl::op_t dqother_node(8, impl::op_kind::Dequantize, "dqother_node");
        dqother_node.set_attr<std::string>("qtype", "per_tensor");
        dqother_node.set_attr<std::vector<int64_t>>("zps", {zp_other});
        dqother_node.set_attr<std::vector<float>>("scales", {scale_other});
        dqother_node.set_attr<int64_t>("axis", 0);

        impl::op_t add_node(9, impl::op_kind::Add, "add_node");

        impl::op_t dqdata_node2(10, impl::op_kind::Dequantize, "dqdata_node");
        SET_Q_DQ_DATA_ATTR(dqdata_node2)

        impl::op_t dqweight_node2(
                11, impl::op_kind::Dequantize, "dqweight_node");
        SET_Q_DQ_WEIGHT_ATTR(dqweight_node2)

        impl::op_t conv_node2(12, impl::op_kind::Convolution, "conv_node");
        SET_CONV_ATTR(conv_node2, 2)

        impl::op_t relu_node2(13, impl::op_kind::ReLU, "relu_node");

        impl::op_t qout_node2(14, impl::op_kind::Quantize, "qout_node");
        SET_Q_DQ_OUT_ATTR(qout_node2)

        // prepare logical tensor
        impl::logical_tensor_t src_u8
                = utils::logical_tensor_init(1, src_shape, impl::data_type::u8);
        impl::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                2, src_shape, impl::data_type::f32);
        impl::logical_tensor_t weight_s8 = utils::logical_tensor_init(
                4, weight_shape, impl::data_type::s8);
        impl::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
                5, weight_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_f32 = utils::logical_tensor_init(
                7, dst_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_relu_f32 = utils::logical_tensor_init(
                8, dst_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_s8
                = utils::logical_tensor_init(9, dst_shape, impl::data_type::s8);
        impl::logical_tensor_t other_f32_dq = utils::logical_tensor_init(
                12, dst_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_add_f32 = utils::logical_tensor_init(
                13, dst_shape, impl::data_type::f32);

        impl::logical_tensor_t src_u8_2 = utils::logical_tensor_init(
                14, src_shape, impl::data_type::u8);
        impl::logical_tensor_t src_f32_dq_2 = utils::logical_tensor_init(
                15, src_shape, impl::data_type::f32);
        impl::logical_tensor_t weight_s8_2 = utils::logical_tensor_init(
                16, weight_shape, impl::data_type::s8);
        impl::logical_tensor_t weight_f32_dq_2 = utils::logical_tensor_init(
                17, weight_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_f32_2 = utils::logical_tensor_init(
                18, dst_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_s8_2 = utils::logical_tensor_init(
                19, dst_shape, impl::data_type::s8);

        dqdata_node.add_input(src_u8);
        dqdata_node.add_output(src_f32_dq);

        dqweight_node.add_input(weight_s8);
        dqweight_node.add_output(weight_f32_dq);

        conv_node.add_input(src_f32_dq);
        conv_node.add_input(weight_f32_dq);
        conv_node.add_output(dst_f32);

        dqother_node.add_input(dst_s8_2);
        dqother_node.add_output(other_f32_dq);

        add_node.add_input(dst_f32);
        add_node.add_input(other_f32_dq);
        add_node.add_output(dst_add_f32);

        relu_node.add_input(dst_add_f32);
        relu_node.add_output(dst_relu_f32);

        qout_node.add_input(dst_relu_f32);
        qout_node.add_output(dst_s8);

        dqdata_node2.add_input(src_u8_2);
        dqdata_node2.add_output(src_f32_dq_2);

        dqweight_node2.add_input(weight_s8_2);
        dqweight_node2.add_output(weight_f32_dq_2);

        conv_node2.add_input(src_f32_dq_2);
        conv_node2.add_input(weight_f32_dq_2);
        conv_node2.add_output(dst_f32_2);

        qout_node2.add_input(dst_f32_2);
        qout_node2.add_output(dst_s8_2);

        impl::graph_t g;
        g.add_op(&dqdata_node);
        g.add_op(&dqweight_node);
        g.add_op(&conv_node);
        g.add_op(&dqother_node);
        g.add_op(&add_node);
        g.add_op(&relu_node);
        g.add_op(&qout_node);
        g.add_op(&dqdata_node2);
        g.add_op(&dqweight_node2);
        g.add_op(&conv_node2);
        g.add_op(&qout_node2);
        g.build_graph();

        impl::pass::pass_base_ptr apass1 = get_pass("int8_conv_fusion");
        impl::pass::pass_base_ptr apass2
                = get_pass("int8_conv_add_relu_fusion");

        apass1->run(g);
        apass2->run(g);
        ASSERT_EQ(g.get_num_partitions(), 2);
        auto part1 = g.get_partitions()[0]; // int8_conv
        auto part2 = g.get_partitions()[1]; // int8_conv_sum_elu

        // compile
        impl::partition_t p1, p2;
        p1.init(part1);
        p2.init(part2);

        impl::compiled_partition_t cp1(p1);
        impl::compiled_partition_t cp2(p2);

        std::vector<const impl::logical_tensor_t *> lt_ins1;
        lt_ins1 = {&src_u8_2, &weight_s8_2};

        dst_s8_2.layout_type = impl::layout_type::any;
        std::vector<const impl::logical_tensor_t *> lt_outs1 {&dst_s8_2};

        p1.compile(&cp1, lt_ins1, lt_outs1, &engine);

        cp1.query_logical_tensor(dst_s8_2.id, &dst_s8_2);

        std::vector<const impl::logical_tensor_t *> lt_ins;
        lt_ins = {&src_u8, &weight_s8, &dst_s8_2};

        dst_s8.layout_type = impl::layout_type::any;
        std::vector<const impl::logical_tensor_t *> lt_outs {&dst_s8};

        p2.compile(&cp2, lt_ins, lt_outs, &engine);

        std::vector<impl::inplace_pair_t> inplace_pairs
                = cp2.get_inplace_pairs();

        ASSERT_EQ(inplace_pairs.size(), 1);
        ASSERT_EQ(inplace_pairs[0].input, dst_s8_2.id);
        ASSERT_EQ(inplace_pairs[0].output, dst_s8.id);
    }
}

TEST(operator_kernel, relu_add_fusion) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src {-2.0, -1.5, 1.0, 0.5};
    test::vector<float> post_src {2.0};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};
    test::vector<float> ref_dst {2.0, 2.0, 3.0, 2.5};

    impl::op_t op(impl::dnnl_impl::op_kind::relu_add);
    auto &op_factory = get_dnnl_kernel_registry();
    auto kernel = op_factory.create_kernel(op);
    ASSERT_TRUE(kernel);

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t post_src_lt
            = utils::logical_tensor_init(1, {1}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            2, {1, 1, 2, 2}, impl::data_type::f32, impl::layout_type::any);

    // compile the relu backward operator
    std::vector<impl::logical_tensor_t> inputs {src_lt, post_src_lt};
    std::vector<impl::logical_tensor_t> outputs {dst_lt};

    kernel->compile(&op, &eng, inputs, outputs);
    ASSERT_EQ(outputs[0].layout_type, impl::layout_type::opaque);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t post_src_ts(post_src_lt, &eng, post_src.data());
    impl::tensor_t dst_ts(outputs[0], &eng, dst.data());

    impl::stream_t &strm = get_stream();
    kernel->execute(&op, &strm, {src_ts, post_src_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, avgpool_add) {
    using dims = impl::dnnl_impl::dims;

    impl::engine_t &eng = get_engine();
    impl::stream_t &strm = get_stream();

    std::vector<std::string> data_formats {"NCX", "NXC"};
    std::vector<bool> with_channel_broadcast_flags {true, false};
    std::vector<impl::data_type_t> data_types {
            impl::data_type::f32, impl::data_type::bf16};

    for_(const auto dt : data_types)
    for_(const auto &data_format : data_formats)
    for (const auto c_broadcast : with_channel_broadcast_flags) {
        static auto isa = dnnl_get_effective_cpu_isa();
        if (dt == impl::data_type::bf16 && isa < dnnl_cpu_isa_avx512_core
                && eng.kind() == impl::engine_kind::cpu) {
            continue;
        }

        std::vector<int64_t> src_shape {3, 3, 4, 4, 4};
        std::vector<int64_t> dst_shape {3, 3, 2, 2, 2};
        const size_t spatial_size = src_shape.size() - 2;
        std::vector<int64_t> post_src_shape {1, 1, 1, 1, 1};

        if (c_broadcast) { post_src_shape[1] = src_shape[1]; }
        if (data_format == "NXC") {
            src_shape.emplace_back(src_shape[1]);
            src_shape.erase(src_shape.begin() + 1);
            dst_shape.emplace_back(dst_shape[1]);
            dst_shape.erase(dst_shape.begin() + 1);
            post_src_shape.emplace_back(post_src_shape[1]);
            post_src_shape.erase(post_src_shape.begin() + 1);
        }

        std::vector<float> src(product(src_shape), 4.0);
        std::vector<float> dst(product(dst_shape), 0.0);
        std::vector<float> post_src(product(post_src_shape), 2.0);

        impl::op_t avgpool_op(impl::dnnl_impl::op_kind::avgpool_add);
        avgpool_op.set_attr<dims>("strides", dims(spatial_size, 2));
        avgpool_op.set_attr<dims>("kernel", dims(spatial_size, 2));
        avgpool_op.set_attr<dims>("pads_begin", dims(spatial_size, 0));
        avgpool_op.set_attr<dims>("pads_end", dims(spatial_size, 0));
        avgpool_op.set_attr<std::string>("data_format", data_format);
        avgpool_op.set_attr<bool>("exclude_pad", false);

        auto &op_factory = get_dnnl_kernel_registry();
        auto avgpool_kernel = op_factory.create_kernel(avgpool_op);
        ASSERT_TRUE(avgpool_kernel);

        impl::logical_tensor_t src_lt
                = utils::logical_tensor_init(0, src_shape, dt);
        impl::logical_tensor_t dst_lt
                = utils::logical_tensor_init(1, dst_shape, dt);
        impl::logical_tensor_t post_src_lt
                = utils::logical_tensor_init(2, post_src_shape, dt);

        std::vector<impl::logical_tensor_t> inputs {src_lt, post_src_lt};
        std::vector<impl::logical_tensor_t> outputs {dst_lt};
        avgpool_kernel->compile(&avgpool_op, &eng, inputs, outputs);

        impl::tensor_t src_ts(src_lt, &eng, src.data());
        impl::tensor_t dst_ts(dst_lt, &eng, dst.data());
        impl::tensor_t post_src_ts(post_src_lt, &eng, post_src.data());

        impl::stream_t &strm = get_stream();
        avgpool_kernel->execute(
                &avgpool_op, &strm, {src_ts, post_src_ts}, {dst_ts});
        strm.wait();
    }
}

TEST(operator_kernel, maxpool_add) {
    using dims = impl::dnnl_impl::dims;

    impl::engine_t &eng = get_engine();
    impl::stream_t &strm = get_stream();

    std::vector<std::string> data_formats {"NCX", "NXC"};
    std::vector<bool> with_channel_broadcast_flags {true, false};
    std::vector<impl::data_type_t> data_types {
            impl::data_type::f32, impl::data_type::bf16};

    for_(const auto dt : data_types)
    for_(const auto &data_format : data_formats)
    for (const auto c_broadcast : with_channel_broadcast_flags) {
        static auto isa = dnnl_get_effective_cpu_isa();
        if (dt == impl::data_type::bf16 && isa < dnnl_cpu_isa_avx512_core
                && eng.kind() == impl::engine_kind::cpu) {
            continue;
        }

        std::vector<int64_t> src_shape {3, 3, 4, 4, 4};
        std::vector<int64_t> dst_shape {3, 3, 2, 2, 2};
        const size_t spatial_size = src_shape.size() - 2;
        std::vector<int64_t> post_src_shape {1, 1, 1, 1, 1};

        if (c_broadcast) { post_src_shape[1] = src_shape[1]; }
        if (data_format == "NXC") {
            src_shape.emplace_back(src_shape[1]);
            src_shape.erase(src_shape.begin() + 1);
            dst_shape.emplace_back(dst_shape[1]);
            dst_shape.erase(dst_shape.begin() + 1);
            post_src_shape.emplace_back(post_src_shape[1]);
            post_src_shape.erase(post_src_shape.begin() + 1);
        }

        std::vector<float> src(product(src_shape), 4.0);
        std::vector<float> dst(product(dst_shape), 0.0);
        std::vector<float> post_src(product(post_src_shape), 2.0);

        impl::op_t maxpool_op(impl::dnnl_impl::op_kind::maxpool_add);
        maxpool_op.set_attr<dims>("strides", dims(spatial_size, 2));
        maxpool_op.set_attr<dims>("kernel", dims(spatial_size, 2));
        maxpool_op.set_attr<dims>("pads_begin", dims(spatial_size, 0));
        maxpool_op.set_attr<dims>("pads_end", dims(spatial_size, 0));
        maxpool_op.set_attr<std::string>("data_format", data_format);
        maxpool_op.set_attr<dims>("dilations", dims(spatial_size, 1));

        auto &op_factory = get_dnnl_kernel_registry();
        auto maxpool_kernel = op_factory.create_kernel(maxpool_op);
        ASSERT_TRUE(maxpool_kernel);

        impl::logical_tensor_t src_lt
                = utils::logical_tensor_init(0, src_shape, dt);
        impl::logical_tensor_t dst_lt
                = utils::logical_tensor_init(1, dst_shape, dt);
        impl::logical_tensor_t post_src_lt
                = utils::logical_tensor_init(2, post_src_shape, dt);

        std::vector<impl::logical_tensor_t> inputs {src_lt, post_src_lt};
        std::vector<impl::logical_tensor_t> outputs {dst_lt};
        maxpool_kernel->compile(&maxpool_op, &eng, inputs, outputs);

        impl::tensor_t src_ts(src_lt, &eng, src.data());
        impl::tensor_t dst_ts(dst_lt, &eng, dst.data());
        impl::tensor_t post_src_ts(post_src_lt, &eng, post_src.data());

        impl::stream_t &strm = get_stream();
        maxpool_kernel->execute(
                &maxpool_op, &strm, {src_ts, post_src_ts}, {dst_ts});
        strm.wait();
    }
}

TEST(int8_subgraph_mode, int8_maxpool) {
    // compare results between:
    // case 1: [quantize] - [dequantize] - [fp32_maxpool] - [quantize]
    // case 2: [quantize] - [int8_maxpool]
    using dims = impl::dnnl_impl::dims;
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    if (engine.kind() == impl::engine_kind::gpu) return;

    std::vector<std::string> data_formats {"NCX", "NXC"};
    std::vector<std::vector<int64_t>> src_shapes {
            {3, 3, 4, 4, 4}, {3, 3, 4, 4}, {3, 3, 4}};
    std::vector<std::vector<int64_t>> dst_shapes {
            {3, 3, 2, 2, 2}, {3, 3, 2, 2}, {3, 3, 2}};
    for_(const auto &data_format : data_formats)
    for (size_t i = 0; i < src_shapes.size(); ++i) {
        // prepare fp32 data
        std::vector<int64_t> src_shape = src_shapes[i];
        std::vector<int64_t> dst_shape = dst_shapes[i];

        if (data_format == "NXC") {
            src_shape.emplace_back(src_shape[1]);
            src_shape.erase(src_shape.begin() + 1);
            dst_shape.emplace_back(dst_shape[1]);
            dst_shape.erase(dst_shape.begin() + 1);
        }

        test::vector<float> src_data(product(src_shape));
        test::vector<float> other_data(product(dst_shape));

        // random generate src, weight and bias data
        // random seed = 7
        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
        std::generate(src_data.begin(), src_data.end(),
                [&]() { return distribution(generator); });
        float scale_src = 1 / 127.f;
        float scale_out = 1 / 127.f;
        int64_t zp_src = 0;
        int64_t zp_out = 0;

        // -------------------------case 1----------------------------------
        impl::op_t qdata_op(0, impl::op_kind::Quantize, "qdata_op");
        qdata_op.set_attr<std::string>("qtype", "per_tensor");
        qdata_op.set_attr<std::vector<int64_t>>("zps", {zp_src});
        qdata_op.set_attr<std::vector<float>>("scales", {scale_src});
        qdata_op.set_attr<int64_t>("axis", 0);

        impl::op_t dqdata_op(1, impl::op_kind::Dequantize, "dqdata_op");
        dqdata_op.set_attr<std::string>("qtype", "per_tensor");
        dqdata_op.set_attr<std::vector<int64_t>>("zps", {zp_src});
        dqdata_op.set_attr<std::vector<float>>("scales", {scale_src});
        dqdata_op.set_attr<int64_t>("axis", 0);

        impl::op_t maxpool_op(2, impl::op_kind::MaxPool, "maxpool_op");
        size_t spatial_size = src_shape.size() - 2;
        maxpool_op.set_attr<dims>("strides", dims(spatial_size, 2));
        maxpool_op.set_attr<dims>("kernel", dims(spatial_size, 2));
        maxpool_op.set_attr<dims>("pads_begin", dims(spatial_size, 0));
        maxpool_op.set_attr<dims>("pads_end", dims(spatial_size, 0));
        maxpool_op.set_attr<std::string>("data_format", data_format);
        maxpool_op.set_attr<dims>("dilations", dims(spatial_size, 1));

        impl::op_t qout_op(3, impl::op_kind::Quantize, "qout_op");
        qout_op.set_attr<std::string>("qtype", "per_tensor");
        qout_op.set_attr<std::vector<int64_t>>("zps", {zp_out});
        qout_op.set_attr<std::vector<float>>("scales", {scale_out});
        qout_op.set_attr<int64_t>("axis", 0);

        // create kernels
        auto kernel_qdata = get_dnnl_kernel_registry().create_kernel(qdata_op);
        auto kernel_dqdata
                = get_dnnl_kernel_registry().create_kernel(dqdata_op);
        auto kernel_maxpool
                = get_dnnl_kernel_registry().create_kernel(maxpool_op);
        auto kernel_qout = get_dnnl_kernel_registry().create_kernel(qout_op);

        // prepare logical tensor
        impl::logical_tensor_t src_f32 = utils::logical_tensor_init(
                0, src_shape, impl::data_type::f32);
        impl::logical_tensor_t src_u8
                = utils::logical_tensor_init(1, src_shape, impl::data_type::u8);
        impl::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                2, src_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_f32 = utils::logical_tensor_init(
                3, dst_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_u8
                = utils::logical_tensor_init(4, dst_shape, impl::data_type::u8);

        // compile
        kernel_qdata->compile(&qdata_op, &engine, {src_f32}, {src_u8});
        kernel_dqdata->compile(&dqdata_op, &engine, {src_u8}, {src_f32_dq});
        kernel_maxpool->compile(&maxpool_op, &engine, {src_f32_dq}, {dst_f32});
        kernel_qout->compile(&qout_op, &engine, {dst_f32}, {dst_u8});

        // execute
        test::vector<uint8_t> src_u8_data(product(src_shape));
        impl::tensor_t src_f32_ts(src_f32, &engine, src_data.data());
        impl::tensor_t src_u8_ts(src_u8, &engine, src_u8_data.data());
        kernel_qdata->execute(&qdata_op, &strm, {src_f32_ts}, {src_u8_ts});
        strm.wait();

        test::vector<float> src_f32_dq_data(product(src_shape));
        impl::tensor_t src_f32_dq_ts(
                src_f32_dq, &engine, src_f32_dq_data.data());
        kernel_dqdata->execute(&dqdata_op, &strm, {src_u8_ts}, {src_f32_dq_ts});
        strm.wait();

        test::vector<float> out_f32_data(product(dst_shape));
        impl::tensor_t dst_f32_ts(dst_f32, &engine, out_f32_data.data());
        kernel_maxpool->execute(
                &maxpool_op, &strm, {src_f32_dq_ts}, {dst_f32_ts});
        strm.wait();

        test::vector<int8_t> case1_out_data(product(dst_shape));
        impl::tensor_t dst_u8_ts(dst_u8, &engine, case1_out_data.data());
        kernel_qout->execute(&qout_op, &strm, {dst_f32_ts}, {dst_u8_ts});
        strm.wait();

        // -------------------------case 2----------------------------------
        dqdata_op.add_input(src_u8);
        dqdata_op.add_output(src_f32_dq);

        maxpool_op.add_input(src_f32_dq);
        maxpool_op.add_output(dst_f32);

        qout_op.add_input(dst_f32);
        qout_op.add_output(dst_u8);

        impl::graph_t g;
        g.add_op(&dqdata_op);
        g.add_op(&maxpool_op);
        g.add_op(&qout_op);
        g.build_graph();

        impl::pass::pass_base_ptr apass = get_pass("int8_maxpool_fusion");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1);
        auto part = g.get_partitions()[0];

        // compile
        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> lt_ins {&src_u8};
        std::vector<const impl::logical_tensor_t *> lt_outs {&dst_u8};

        p.compile(&cp, lt_ins, lt_outs, &engine);

        test::vector<int8_t> case2_out_data(product(dst_shape));
        impl::tensor_t dst_u8_case2_ts(dst_u8, &engine, case2_out_data.data());
        cp.execute(&strm, {src_u8_ts}, {dst_u8_case2_ts});
        strm.wait();

        static auto isa = dnnl_get_effective_cpu_isa();
        if (isa < dnnl_cpu_isa_avx512_core_vnni)
            ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.1f,
                    /*atol*/ 1.f));
        else
            ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.01f,
                    /*atol*/ 1.f));
    }
}

TEST(int8_subgraph_mode, int8_avgpool) {
    // compare results between:
    // case 1: [quantize] - [dequantize] - [fp32_avgpool] - [quantize]
    // case 2: [quantize] - [int8_avgpool]
    using dims = impl::dnnl_impl::dims;
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    if (engine.kind() == impl::engine_kind::gpu) return;

    std::vector<std::string> data_formats {"NCX", "NXC"};
    std::vector<std::vector<int64_t>> src_shapes {
            {3, 3, 4, 4, 4}, {3, 3, 4, 4}, {3, 3, 4}};
    std::vector<std::vector<int64_t>> dst_shapes {
            {3, 3, 2, 2, 2}, {3, 3, 2, 2}, {3, 3, 2}};
    for_(const auto &data_format : data_formats)
    for (size_t i = 0; i < src_shapes.size(); ++i) {
        // prepare fp32 data
        std::vector<int64_t> src_shape = src_shapes[i];
        std::vector<int64_t> dst_shape = dst_shapes[i];

        if (data_format == "NXC") {
            src_shape.emplace_back(src_shape[1]);
            src_shape.erase(src_shape.begin() + 1);
            dst_shape.emplace_back(dst_shape[1]);
            dst_shape.erase(dst_shape.begin() + 1);
        }

        test::vector<float> src_data(product(src_shape));
        test::vector<float> other_data(product(dst_shape));

        // random generate src, weight and bias data
        // random seed = 7
        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
        std::generate(src_data.begin(), src_data.end(),
                [&]() { return distribution(generator); });
        float scale_src = 1 / 127.f;
        float scale_out = 1 / 127.f;
        int64_t zp_src = 0;
        int64_t zp_out = 0;

        // -------------------------case 1----------------------------------
        impl::op_t qdata_op(0, impl::op_kind::Quantize, "qdata_op");
        qdata_op.set_attr<std::string>("qtype", "per_tensor");
        qdata_op.set_attr<std::vector<int64_t>>("zps", {zp_src});
        qdata_op.set_attr<std::vector<float>>("scales", {scale_src});
        qdata_op.set_attr<int64_t>("axis", 0);

        impl::op_t dqdata_op(1, impl::op_kind::Dequantize, "dqdata_op");
        dqdata_op.set_attr<std::string>("qtype", "per_tensor");
        dqdata_op.set_attr<std::vector<int64_t>>("zps", {zp_src});
        dqdata_op.set_attr<std::vector<float>>("scales", {scale_src});
        dqdata_op.set_attr<int64_t>("axis", 0);

        impl::op_t avgpool_op(2, impl::op_kind::AvgPool, "avgpool_op");
        size_t spatial_size = src_shape.size() - 2;
        avgpool_op.set_attr<dims>("strides", dims(spatial_size, 2));
        avgpool_op.set_attr<dims>("kernel", dims(spatial_size, 2));
        avgpool_op.set_attr<dims>("pads_begin", dims(spatial_size, 0));
        avgpool_op.set_attr<dims>("pads_end", dims(spatial_size, 0));
        avgpool_op.set_attr<std::string>("data_format", data_format);
        avgpool_op.set_attr<bool>("exclude_pad", false);

        impl::op_t qout_op(3, impl::op_kind::Quantize, "qout_op");
        qout_op.set_attr<std::string>("qtype", "per_tensor");
        qout_op.set_attr<std::vector<int64_t>>("zps", {zp_out});
        qout_op.set_attr<std::vector<float>>("scales", {scale_out});
        qout_op.set_attr<int64_t>("axis", 0);

        // create kernels
        auto kernel_qdata = get_dnnl_kernel_registry().create_kernel(qdata_op);
        auto kernel_dqdata
                = get_dnnl_kernel_registry().create_kernel(dqdata_op);
        auto kernel_avgpool
                = get_dnnl_kernel_registry().create_kernel(avgpool_op);
        auto kernel_qout = get_dnnl_kernel_registry().create_kernel(qout_op);

        // prepare logical tensor
        impl::logical_tensor_t src_f32 = utils::logical_tensor_init(
                0, src_shape, impl::data_type::f32);
        impl::logical_tensor_t src_u8
                = utils::logical_tensor_init(1, src_shape, impl::data_type::u8);
        impl::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                2, src_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_f32 = utils::logical_tensor_init(
                3, dst_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_u8
                = utils::logical_tensor_init(4, dst_shape, impl::data_type::u8);

        // compile
        kernel_qdata->compile(&qdata_op, &engine, {src_f32}, {src_u8});
        kernel_dqdata->compile(&dqdata_op, &engine, {src_u8}, {src_f32_dq});
        kernel_avgpool->compile(&avgpool_op, &engine, {src_f32_dq}, {dst_f32});
        kernel_qout->compile(&qout_op, &engine, {dst_f32}, {dst_u8});

        // execute
        test::vector<uint8_t> src_u8_data(product(src_shape));
        impl::tensor_t src_f32_ts(src_f32, &engine, src_data.data());
        impl::tensor_t src_u8_ts(src_u8, &engine, src_u8_data.data());
        kernel_qdata->execute(&qdata_op, &strm, {src_f32_ts}, {src_u8_ts});
        strm.wait();

        test::vector<float> src_f32_dq_data(product(src_shape));
        impl::tensor_t src_f32_dq_ts(
                src_f32_dq, &engine, src_f32_dq_data.data());
        kernel_dqdata->execute(&dqdata_op, &strm, {src_u8_ts}, {src_f32_dq_ts});
        strm.wait();

        test::vector<float> out_f32_data(product(dst_shape));
        impl::tensor_t dst_f32_ts(dst_f32, &engine, out_f32_data.data());
        kernel_avgpool->execute(
                &avgpool_op, &strm, {src_f32_dq_ts}, {dst_f32_ts});
        strm.wait();

        test::vector<int8_t> case1_out_data(product(dst_shape));
        impl::tensor_t dst_u8_ts(dst_u8, &engine, case1_out_data.data());
        kernel_qout->execute(&qout_op, &strm, {dst_f32_ts}, {dst_u8_ts});
        strm.wait();

        // -------------------------case 2----------------------------------
        dqdata_op.add_input(src_u8);
        dqdata_op.add_output(src_f32_dq);

        avgpool_op.add_input(src_f32_dq);
        avgpool_op.add_output(dst_f32);

        qout_op.add_input(dst_f32);
        qout_op.add_output(dst_u8);

        impl::graph_t g;
        g.add_op(&dqdata_op);
        g.add_op(&avgpool_op);
        g.add_op(&qout_op);
        g.build_graph();

        impl::pass::pass_base_ptr apass = get_pass("int8_avgpool_fusion");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1);
        auto part = g.get_partitions()[0];

        // compile
        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> lt_ins {&src_u8};
        std::vector<const impl::logical_tensor_t *> lt_outs {&dst_u8};

        p.compile(&cp, lt_ins, lt_outs, &engine);

        test::vector<int8_t> case2_out_data(product(dst_shape));
        impl::tensor_t dst_u8_case2_ts(dst_u8, &engine, case2_out_data.data());
        cp.execute(&strm, {src_u8_ts}, {dst_u8_case2_ts});
        strm.wait();

        static auto isa = dnnl_get_effective_cpu_isa();
        if (isa < dnnl_cpu_isa_avx512_core_vnni)
            ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.1f,
                    /*atol*/ 1.f));
        else
            ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.01f,
                    /*atol*/ 1.f));
    }
}

TEST(operator_compile, matmul_add_get_inplace_pair) {
    using dims = impl::dnnl_impl::dims;

    impl::graph_t agraph;
    impl::op_t mm {0, impl::op_kind::MatMul, "matmul"};
    impl::op_t add {1, impl::op_kind::Add, "add"};
    impl::op_t mm2 {2, impl::op_kind::MatMul, "matmul2"};

    impl::logical_tensor_t lt_mm_src = utils::logical_tensor_init(
            1, {1, 16, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t lt_mm_weight
            = utils::logical_tensor_init(2, {4, 4}, impl::data_type::f32);
    impl::logical_tensor_t lt_mm_out = utils::logical_tensor_init(
            3, {1, 16, 4, 4}, impl::data_type::f32, impl::layout_type::any);
    impl::logical_tensor_t lt_mm_src2 = utils::logical_tensor_init(
            4, {1, 16, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t lt_mm_weight2
            = utils::logical_tensor_init(5, {4, 4}, impl::data_type::f32);
    impl::logical_tensor_t lt_mm_out2 = utils::logical_tensor_init(
            6, {1, 16, 4, 4}, impl::data_type::f32, impl::layout_type::any);
    impl::logical_tensor_t lt_add_out = utils::logical_tensor_init(
            7, {1, 16, 4, 4}, impl::data_type::f32, impl::layout_type::any);
    mm.add_input(lt_mm_src);
    mm.add_input(lt_mm_weight);
    mm.add_output(lt_mm_out);
    mm2.add_input(lt_mm_src2);
    mm2.add_input(lt_mm_weight2);
    mm2.add_output(lt_mm_out2);
    add.add_input(lt_mm_out);
    add.add_input(lt_mm_out2);
    add.add_output(lt_add_out);

    ASSERT_EQ(agraph.add_op(&mm), impl::status::success);
    ASSERT_EQ(agraph.add_op(&mm2), impl::status::success);
    ASSERT_EQ(agraph.add_op(&add), impl::status::success);
    agraph.build_graph();
    ASSERT_EQ(agraph.num_ops(), 3);

    impl::pass::pass_base_ptr apass1 = get_pass("matmul_sum_fusion");
    impl::pass::pass_base_ptr apass2 = get_pass("matmul_pass");

    apass1->run(agraph);
    apass2->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 2);
    auto part1 = agraph.get_partitions()[0]; // matmul_add
    auto part2 = agraph.get_partitions()[1]; // matmul
    impl::partition_t p1, p2;
    p1.init(part1);
    p2.init(part2);

    impl::compiled_partition_t cp1(p1), cp2(p2);
    impl::engine_t &eng = get_engine();

    // compile matmul partition
    std::vector<const impl::logical_tensor_t *> inputs2 {
            &lt_mm_src2, &lt_mm_weight2};
    std::vector<const impl::logical_tensor_t *> outputs2 {&lt_mm_out2};
    p2.compile(&cp2, inputs2, outputs2, &eng);
    cp2.query_logical_tensor(lt_mm_out2.id, &lt_mm_out2);

    // compile matmul_add partition
    std::vector<const impl::logical_tensor_t *> inputs1 {
            &lt_mm_src, &lt_mm_weight, &lt_mm_out2};
    std::vector<const impl::logical_tensor_t *> outputs1 {&lt_add_out};
    p1.compile(&cp1, inputs1, outputs1, &eng);

    std::vector<impl::inplace_pair_t> inplace_pairs = cp1.get_inplace_pairs();
    ASSERT_EQ(inplace_pairs.size(), 1);
    ASSERT_EQ(inplace_pairs[0].input, lt_mm_out2.id);
    ASSERT_EQ(inplace_pairs[0].output, lt_add_out.id);
}

TEST(int8_subgraph_mode, int8_quant_wei_conv2d_sum_relu) {
    using dims = impl::dnnl_impl::dims;

    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    if (engine.kind() == impl::engine_kind::gpu) return;

    std::vector<int64_t> groups = {1, 4};
    std::vector<bool> with_biases = {true, false};
    std::vector<std::string> weight_qtypes = {"per_tensor", "per_channel"};

    for_(const auto &g : groups)
    for_(const auto &with_bias : with_biases)
    for (const auto &wei_qtype : weight_qtypes) {
        // prepare fp32 data
        int64_t in_channel = 8, out_channel = 8;
        int64_t kernel_size = 3;
        std::vector<int64_t> src_shape {1, in_channel, 112, 112};
        std::vector<int64_t> weight_shape {
                out_channel, in_channel / g, kernel_size, kernel_size};
        std::vector<int64_t> bias_shape {out_channel};
        std::vector<int64_t> dst_shape {1, out_channel, 110, 110};

        test::vector<uint8_t> src_u8_data(product(src_shape));
        test::vector<float> weight_f32_data(product(weight_shape));
        test::vector<int8_t> other_s8_data(product(dst_shape));
        size_t bias_size = with_bias ? product(bias_shape) : 0;
        test::vector<float> bias_data(bias_size);
        test::vector<int8_t> case1_out_data(product(dst_shape));
        test::vector<int8_t> case2_out_data(product(dst_shape));

        // random generate src, weight and bias data
        // random seed = 7
        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> u8_distribution(0.0f, 255.0f);
        std::uniform_real_distribution<float> s8_distribution(-127.0f, 128.0f);
        std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
        std::generate(src_u8_data.begin(), src_u8_data.end(), [&]() {
            return static_cast<uint8_t>(u8_distribution(generator));
        });
        std::generate(weight_f32_data.begin(), weight_f32_data.end(),
                [&]() { return f32_distribution(generator); });
        std::generate(other_s8_data.begin(), other_s8_data.end(), [&]() {
            return static_cast<int8_t>(s8_distribution(generator));
        });
        if (with_bias) {
            std::generate(bias_data.begin(), bias_data.end(),
                    [&]() { return f32_distribution(generator); });
        }

        float scale_src = 1 / 255.f; // map to 0~255
        float scale_other = 1 / 127.f;
        float scale_out = 1;
        int64_t zp_src = 0;
        int64_t zp_other = 0;
        int64_t zp_out = 78;

        size_t scale_size = wei_qtype == "per_tensor" ? 1 : out_channel;

        std::vector<float> scale_wei(scale_size, 1 / 127.f);
        std::vector<int64_t> zp_wei(scale_size, 0);

        impl::op_t dqdata_node(1, impl::op_kind::Dequantize, "dqdata_node");
        SET_Q_DQ_DATA_ATTR(dqdata_node)

        impl::op_t qweight_node(2, impl::op_kind::Quantize, "qweight_node");
        SET_Q_DQ_WEIGHT_ATTR(qweight_node)

        impl::op_t dqweight_node(3, impl::op_kind::Dequantize, "dqweight_node");
        SET_Q_DQ_WEIGHT_ATTR(dqweight_node)

        impl::op_t conv_node(4, impl::op_kind::Convolution, "conv_node");
        SET_CONV_ATTR(conv_node, 2)

        impl::op_t relu_node(5, impl::op_kind::ReLU, "relu_node");

        impl::op_t qout_node(6, impl::op_kind::Quantize, "qout_node");
        SET_Q_DQ_OUT_ATTR(qout_node)

        impl::op_t dqother_node(8, impl::op_kind::Dequantize, "dqother_node");
        dqother_node.set_attr<std::string>("qtype", "per_tensor");
        dqother_node.set_attr<std::vector<int64_t>>("zps", {zp_other});
        dqother_node.set_attr<std::vector<float>>("scales", {scale_other});
        dqother_node.set_attr<int64_t>("axis", 0);

        impl::op_t add_node(9, impl::op_kind::Add, "add_node");

        // prepare logical tensor
        auto src_u8 = utils::logical_tensor_init(1, impl::data_type::u8);
        auto src_f32_dq = utils::logical_tensor_init(2, impl::data_type::f32);
        auto weight_f32 = utils::logical_tensor_init(3, impl::data_type::f32);
        auto weight_s8 = utils::logical_tensor_init(4, impl::data_type::s8);
        auto weight_f32_dq
                = utils::logical_tensor_init(5, impl::data_type::f32);
        auto dst_f32 = utils::logical_tensor_init(7, impl::data_type::f32);
        auto dst_relu_f32 = utils::logical_tensor_init(8, impl::data_type::f32);
        auto dst_s8 = utils::logical_tensor_init(9, impl::data_type::s8);
        auto other_s8 = utils::logical_tensor_init(11, impl::data_type::s8);
        auto other_f32_dq
                = utils::logical_tensor_init(12, impl::data_type::f32);
        auto dst_add_f32 = utils::logical_tensor_init(13, impl::data_type::f32);
        impl::logical_tensor_t bias_f32;
        if (with_bias) {
            bias_f32 = utils::logical_tensor_init(6, impl::data_type::f32);
        }

        dqdata_node.add_input(src_u8);
        dqdata_node.add_output(src_f32_dq);

        qweight_node.add_input(weight_f32);
        qweight_node.add_output(weight_s8);

        dqweight_node.add_input(weight_s8);
        dqweight_node.add_output(weight_f32_dq);

        conv_node.add_input(src_f32_dq);
        conv_node.add_input(weight_f32_dq);
        if (with_bias) conv_node.add_input(bias_f32);
        conv_node.add_output(dst_f32);

        dqother_node.add_input(other_s8);
        dqother_node.add_output(other_f32_dq);

        add_node.add_input(dst_f32);
        add_node.add_input(other_f32_dq);
        add_node.add_output(dst_add_f32);

        relu_node.add_input(dst_add_f32);
        relu_node.add_output(dst_relu_f32);

        qout_node.add_input(dst_relu_f32);
        qout_node.add_output(dst_s8);

        impl::graph_t g;
        g.add_op(&dqdata_node);
        g.add_op(&qweight_node);
        g.add_op(&dqweight_node);
        g.add_op(&conv_node);
        g.add_op(&dqother_node);
        g.add_op(&add_node);
        g.add_op(&relu_node);
        g.add_op(&qout_node);
        g.build_graph();

        // prepare in/out with full shape
        src_u8 = utils::logical_tensor_init(1, src_shape, impl::data_type::u8);
        weight_f32 = utils::logical_tensor_init(
                3, weight_shape, impl::data_type::f32);
        // set weight to be constant
        weight_f32.property = impl::property_type::constant;
        dst_s8 = utils::logical_tensor_init(9, dst_shape, impl::data_type::s8);
        other_s8 = utils::logical_tensor_init(
                11, dst_shape, impl::data_type::s8);
        if (with_bias) {
            bias_f32 = utils::logical_tensor_init(
                    6, bias_shape, impl::data_type::f32);
            // set bias to be constant
            bias_f32.property = impl::property_type::constant;
        }

        impl::tensor_t src_u8_ts(src_u8, &engine, src_u8_data.data());
        impl::tensor_t weight_f32_ts(
                weight_f32, &engine, weight_f32_data.data());
        impl::tensor_t other_s8_ts(other_s8, &engine, other_s8_data.data());
        impl::tensor_t bias_f32_ts;
        if (with_bias) {
            bias_f32_ts = impl::tensor_t(bias_f32, &engine, bias_data.data());
        }
        impl::tensor_t dst_s8_ts(dst_s8, &engine, case1_out_data.data());
        impl::tensor_t dst_s8_case2_ts(dst_s8, &engine, case2_out_data.data());

        // -------------------------case 1----------------------------------
        ASSERT_EQ(run_graph(g,
                          {src_u8_ts, weight_f32_ts, bias_f32_ts, other_s8_ts},
                          {dst_s8_ts}, engine, strm),
                impl::status::success);

        // -------------------------case 2----------------------------------
        impl::pass::pass_base_ptr apass = with_bias
                ? get_pass("int8_quant_wei_conv_bias_add_relu_fusion")
                : get_pass("int8_quant_wei_conv_add_relu_fusion");

        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1);
        auto part = g.get_partitions()[0];

        // compile
        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> lt_ins;
        if (with_bias)
            lt_ins = {&src_u8, &weight_f32, &bias_f32, &other_s8};
        else
            lt_ins = {&src_u8, &weight_f32, &other_s8};
        std::vector<const impl::logical_tensor_t *> lt_outs {&dst_s8};

        impl::status_t ret = p.compile(&cp, lt_ins, lt_outs, &engine);
        ASSERT_EQ(ret, impl::status::success);

        // single thread
        for (size_t iter = 0; iter < 5; iter++) {
            if (with_bias)
                cp.execute(&strm,
                        {src_u8_ts, weight_f32_ts, bias_f32_ts, other_s8_ts},
                        {dst_s8_case2_ts});
            else
                cp.execute(&strm, {src_u8_ts, weight_f32_ts, other_s8_ts},
                        {dst_s8_case2_ts});
            strm.wait();

            static auto isa = dnnl_get_effective_cpu_isa();
            if (isa < dnnl_cpu_isa_avx512_core_vnni)
                ASSERT_TRUE(
                        allclose(case1_out_data, case2_out_data, /*rtol*/ 0.1f,
                                /*atol*/ 1.f));
            else
                ASSERT_TRUE(
                        allclose(case1_out_data, case2_out_data, /*rtol*/ 0.01f,
                                /*atol*/ 1.f));
        }
    }
}

TEST(int8_subgraph_mode, int8_quant_wei_matmul_bias_sum_ndx2d) {
    // compare results between:
    // case 1: [quantize] - [dequantize] - [fp32_matmul] - [quantize]
    // case 2: [quantize] - [int8_matmul]
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    if (engine.kind() == impl::engine_kind::gpu) return;

    std::vector<std::string> qtypes {"per_tensor", "per_channel"};
    std::vector<std::string> weight_qtypes = {"symmetric", "asymmetric"};
    std::vector<std::vector<int64_t>> src_shapes {
            {3, 3, 3, 8, 4}, {3, 3, 8, 4}, {3, 8, 4}, {8, 4}, {4}};
    std::vector<std::vector<int64_t>> weight_shapes {{4, 2}};
    std::vector<std::vector<int64_t>> dst_shapes {
            {3, 3, 3, 8, 2}, {3, 3, 8, 2}, {3, 8, 2}, {8, 2}, {2}};
    for_(const auto &qtype : qtypes)
    for_(const auto &wei_qtype : weight_qtypes)
    for_(size_t i = 0; i < src_shapes.size(); ++i)
    for (size_t j = 0; j < weight_shapes.size(); ++j) {
        // prepare fp32 data
        std::vector<int64_t> src_shape = src_shapes[i];
        std::vector<int64_t> weight_shape = weight_shapes[j];
        std::vector<int64_t> bias_shape {2};
        std::vector<int64_t> dst_shape = dst_shapes[i];

        test::vector<uint8_t> src_data(product(src_shape));
        test::vector<float> weight_data(product(weight_shape));
        test::vector<float> bias_data(product(bias_shape));
        test::vector<int8_t> other_data(product(dst_shape));

        // random generate src, weight and bias data
        // random seed = 7
        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> u8_distribution(0.0f, 255.0f);
        std::uniform_real_distribution<float> s8_distribution(-127.0f, 128.0f);
        std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
        std::generate(src_data.begin(), src_data.end(), [&]() {
            return static_cast<uint8_t>(u8_distribution(generator));
        });
        std::generate(weight_data.begin(), weight_data.end(),
                [&]() { return f32_distribution(generator); });
        std::generate(bias_data.begin(), bias_data.end(),
                [&]() { return f32_distribution(generator); });
        std::generate(other_data.begin(), other_data.end(), [&]() {
            return static_cast<int8_t>(s8_distribution(generator));
        });
        float scale_src = 1 / 255.f; // map to 0~255
        float scale_other = 1 / 127.f;
        float scale_out = 1;
        int64_t zp_src = 90; // src is asymmetric
        int64_t zp_other = 0;
        int64_t zp_out = 78;

        auto generate_zps = [&]() {
            // backend integration doesn't support per_channel asym quant now.
            if (qtype == "per_channel" || wei_qtype == "symmetric")
                return 0;
            else {
                return 78;
            }
        };

        size_t scales_wei_sizes = qtype == "per_tensor" ? 1 : dst_shape.back();
        std::vector<float> scale_wei(scales_wei_sizes, 1 / 127.f);
        std::vector<int64_t> zp_wei(scales_wei_sizes, generate_zps());

        impl::op_t dqdata_op(1, impl::op_kind::Dequantize, "dqdata_op");
        dqdata_op.set_attr<std::string>("qtype", "per_tensor");
        dqdata_op.set_attr<std::vector<int64_t>>("zps", {zp_src});
        dqdata_op.set_attr<std::vector<float>>("scales", {scale_src});
        dqdata_op.set_attr<int64_t>("axis", 0);

        impl::op_t qweight_op(2, impl::op_kind::Quantize, "qweight_op");
        qweight_op.set_attr<std::string>("qtype", qtype);
        qweight_op.set_attr<std::vector<int64_t>>("zps", zp_wei);
        qweight_op.set_attr<std::vector<float>>("scales", scale_wei);
        qweight_op.set_attr<int64_t>("axis", 1);

        impl::op_t dqweight_op(3, impl::op_kind::Dequantize, "dqweight_op");
        dqweight_op.set_attr<std::string>("qtype", qtype);
        dqweight_op.set_attr<std::vector<int64_t>>("zps", zp_wei);
        dqweight_op.set_attr<std::vector<float>>("scales", scale_wei);
        dqweight_op.set_attr<int64_t>("axis", 1);

        impl::op_t matmul_op(4, impl::op_kind::MatMul, "matmul_op");
        matmul_op.set_attr<bool>("transpose_a", false);
        matmul_op.set_attr<bool>("transpose_b", false);

        impl::op_t qout_op(5, impl::op_kind::Quantize, "qout_op");
        qout_op.set_attr<std::string>("qtype", "per_tensor");
        qout_op.set_attr<std::vector<int64_t>>("zps", {zp_out});
        qout_op.set_attr<std::vector<float>>("scales", {scale_out});
        qout_op.set_attr<int64_t>("axis", 0);

        impl::op_t dqother_op(6, impl::op_kind::Dequantize, "dqother_op");
        dqother_op.set_attr<std::string>("qtype", "per_tensor");
        dqother_op.set_attr<std::vector<int64_t>>("zps", {zp_other});
        dqother_op.set_attr<std::vector<float>>("scales", {scale_other});
        dqother_op.set_attr<int64_t>("axis", 0);

        impl::op_t add_op(7, impl::op_kind::Add, "add_op");

        // The logical tensor in graph stage should only have valid id and dtype
        auto src_u8 = utils::logical_tensor_init(1, impl::data_type::u8);
        auto src_f32_dq = utils::logical_tensor_init(2, impl::data_type::f32);
        auto weight_f32 = utils::logical_tensor_init(3, impl::data_type::f32);
        auto weight_s8 = utils::logical_tensor_init(4, impl::data_type::s8);
        auto weight_f32_dq
                = utils::logical_tensor_init(5, impl::data_type::f32);
        auto bias_f32 = utils::logical_tensor_init(6, impl::data_type::f32);
        auto dst_f32 = utils::logical_tensor_init(
                7, dst_shape, impl::data_type::f32);
        auto dst_s8
                = utils::logical_tensor_init(9, dst_shape, impl::data_type::s8);
        auto other_s8 = utils::logical_tensor_init(
                11, dst_shape, impl::data_type::s8);
        auto other_f32_dq = utils::logical_tensor_init(
                12, dst_shape, impl::data_type::f32);
        auto dst_add_f32 = utils::logical_tensor_init(
                13, dst_shape, impl::data_type::f32);

        dqdata_op.add_input(src_u8);
        dqdata_op.add_output(src_f32_dq);

        qweight_op.add_input(weight_f32);
        qweight_op.add_output(weight_s8);

        dqweight_op.add_input(weight_s8);
        dqweight_op.add_output(weight_f32_dq);

        matmul_op.add_input(src_f32_dq);
        matmul_op.add_input(weight_f32_dq);
        matmul_op.add_input(bias_f32);
        matmul_op.add_output(dst_f32);

        dqother_op.add_input(other_s8);
        dqother_op.add_output(other_f32_dq);

        add_op.add_input(dst_f32);
        add_op.add_input(other_f32_dq);
        add_op.add_output(dst_add_f32);

        qout_op.add_input(dst_add_f32);
        qout_op.add_output(dst_s8);

        impl::graph_t g;
        g.add_op(&dqdata_op);
        g.add_op(&qweight_op);
        g.add_op(&dqweight_op);
        g.add_op(&matmul_op);
        g.add_op(&dqother_op);
        g.add_op(&add_op);
        g.add_op(&qout_op);
        g.build_graph();

        // prepare in/out full shape
        src_u8 = utils::logical_tensor_init(1, src_shape, impl::data_type::u8);
        weight_f32 = utils::logical_tensor_init(
                3, weight_shape, impl::data_type::f32);
        // set weight to be constant
        weight_f32.property = impl::property_type::constant;
        dst_s8 = utils::logical_tensor_init(9, dst_shape, impl::data_type::s8);
        other_s8 = utils::logical_tensor_init(
                11, dst_shape, impl::data_type::s8);
        bias_f32 = utils::logical_tensor_init(
                6, bias_shape, impl::data_type::f32);
        // set bias to be constant
        bias_f32.property = impl::property_type::constant;

        impl::tensor_t src_u8_ts(src_u8, &engine, src_data.data());
        impl::tensor_t weight_f32_ts(weight_f32, &engine, weight_data.data());
        impl::tensor_t other_s8_ts(other_s8, &engine, other_data.data());
        impl::tensor_t bias_f32_ts(bias_f32, &engine, bias_data.data());
        // -------------------------case 1----------------------------------
        test::vector<int8_t> case1_out_data(product(dst_shape));
        impl::tensor_t dst_s8_ts(dst_s8, &engine, case1_out_data.data());
        ASSERT_EQ(run_graph(g,
                          {src_u8_ts, weight_f32_ts, bias_f32_ts, other_s8_ts},
                          {dst_s8_ts}, engine, strm),
                impl::status::success);

        // -------------------------case 2----------------------------------
        impl::pass::pass_base_ptr apass
                = get_pass("int8_quant_wei_matmul_bias_add_fusion");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1);
        auto part = g.get_partitions()[0];

        // compile
        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> lt_ins {
                &src_u8, &weight_f32, &bias_f32, &other_s8};
        std::vector<const impl::logical_tensor_t *> lt_outs {&dst_s8};

        p.compile(&cp, lt_ins, lt_outs, &engine);

        test::vector<int8_t> case2_out_data(product(dst_shape));
        impl::tensor_t dst_s8_case2_ts(dst_s8, &engine, case2_out_data.data());
        for (size_t iter = 0; iter < 5; iter++) {
            cp.execute(&strm,
                    {src_u8_ts, weight_f32_ts, bias_f32_ts, other_s8_ts},
                    {dst_s8_case2_ts});
            strm.wait();

            static auto isa = dnnl_get_effective_cpu_isa();
            if (isa < dnnl_cpu_isa_avx512_core_vnni)
                ASSERT_TRUE(
                        allclose(case1_out_data, case2_out_data, /*rtol*/ 0.1f,
                                /*atol*/ 1.f));
            else
                ASSERT_TRUE(
                        allclose(case1_out_data, case2_out_data, /*rtol*/ 0.01f,
                                /*atol*/ 1.f));
        }
    }
}

TEST(int8_subgraph_mode, int8_quant_wei_matmul_bias_ndx2d_with_transpose) {
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    if (engine.kind() == impl::engine_kind::gpu) return;

    std::vector<std::string> qtypes {"per_tensor", "per_channel"};
    std::vector<std::vector<int64_t>> src_shapes {
            {3, 3, 3, 8, 4}, {3, 3, 8, 4}, {3, 8, 4}, {8, 4}, {4}};
    std::vector<std::vector<int64_t>> weight_shapes {{2, 4}};
    std::vector<std::vector<int64_t>> dst_shapes {
            {3, 3, 3, 8, 2}, {3, 3, 8, 2}, {3, 8, 2}, {8, 2}, {2}};

    for_(const auto &qtype : qtypes)
    for_(size_t i = 0; i < src_shapes.size(); ++i)
    for (size_t j = 0; j < weight_shapes.size(); ++j) {
        // prepare fp32 data
        std::vector<int64_t> src_shape = src_shapes[i];
        std::vector<int64_t> weight_shape = weight_shapes[j];
        std::vector<int64_t> bias_shape {2};
        std::vector<int64_t> dst_shape = dst_shapes[i];

        test::vector<uint8_t> src_data(product(src_shape));
        test::vector<float> weight_data(product(weight_shape));
        test::vector<float> bias_data(product(bias_shape));
        test::vector<int8_t> other_data(product(dst_shape));

        // random generate src, weight and bias data
        // random seed = 7
        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> u8_distribution(0.0f, 255.0f);
        std::uniform_real_distribution<float> s8_distribution(-127.0f, 128.0f);
        std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
        std::generate(src_data.begin(), src_data.end(), [&]() {
            return static_cast<uint8_t>(u8_distribution(generator));
        });
        std::generate(weight_data.begin(), weight_data.end(),
                [&]() { return f32_distribution(generator); });
        std::generate(bias_data.begin(), bias_data.end(),
                [&]() { return f32_distribution(generator); });
        std::generate(other_data.begin(), other_data.end(), [&]() {
            return static_cast<int8_t>(s8_distribution(generator));
        });
        float scale_src = 1 / 255.f; // map to 0~255
        float scale_other = 1 / 127.f;
        float scale_out = 1;
        int64_t zp_src = 0;
        int64_t zp_other = 0;
        int64_t zp_out = 78;

        size_t scales_wei_sizes = qtype == "per_tensor" ? 1 : dst_shape.back();
        std::vector<float> scale_wei(scales_wei_sizes, 1 / 127.f);
        std::vector<int64_t> zp_wei(scales_wei_sizes, 0);

        // -------------------------case 1----------------------------------
        impl::op_t dqdata_op(1, impl::op_kind::Dequantize, "dqdata_op");
        dqdata_op.set_attr<std::string>("qtype", "per_tensor");
        dqdata_op.set_attr<std::vector<int64_t>>("zps", {zp_src});
        dqdata_op.set_attr<std::vector<float>>("scales", {scale_src});
        dqdata_op.set_attr<int64_t>("axis", 0);

        impl::op_t qweight_op(2, impl::op_kind::Quantize, "qweight_op");
        qweight_op.set_attr<std::string>("qtype", qtype);
        qweight_op.set_attr<std::vector<int64_t>>("zps", zp_wei);
        qweight_op.set_attr<std::vector<float>>("scales", scale_wei);
        qweight_op.set_attr<int64_t>("axis", 0);

        impl::op_t dqweight_op(3, impl::op_kind::Dequantize, "dqweight_op");
        dqweight_op.set_attr<std::string>("qtype", qtype);
        dqweight_op.set_attr<std::vector<int64_t>>("zps", zp_wei);
        dqweight_op.set_attr<std::vector<float>>("scales", scale_wei);
        dqweight_op.set_attr<int64_t>("axis", 0);

        impl::op_t matmul_op(4, impl::op_kind::MatMul, "matmul_op");
        matmul_op.set_attr<bool>("transpose_a", false);
        matmul_op.set_attr<bool>("transpose_b", true);

        impl::op_t qout_op(5, impl::op_kind::Quantize, "qout_op");
        qout_op.set_attr<std::string>("qtype", "per_tensor");
        qout_op.set_attr<std::vector<int64_t>>("zps", {zp_out});
        qout_op.set_attr<std::vector<float>>("scales", {scale_out});
        qout_op.set_attr<int64_t>("axis", 0);

        // prepare logical tensor
        // The logical tensor in graph stage should only have valid id and dtype
        auto src_u8 = utils::logical_tensor_init(1, impl::data_type::u8);
        auto src_f32_dq = utils::logical_tensor_init(2, impl::data_type::f32);
        auto weight_f32 = utils::logical_tensor_init(3, impl::data_type::f32);
        auto weight_s8 = utils::logical_tensor_init(4, impl::data_type::s8);
        auto weight_f32_dq
                = utils::logical_tensor_init(5, impl::data_type::f32);
        auto dst_f32 = utils::logical_tensor_init(
                7, dst_shape, impl::data_type::f32);
        auto dst_s8
                = utils::logical_tensor_init(9, dst_shape, impl::data_type::s8);
        auto bias_f32 = utils::logical_tensor_init(6, impl::data_type::f32);

        dqdata_op.add_input(src_u8);
        dqdata_op.add_output(src_f32_dq);

        qweight_op.add_input(weight_f32);
        qweight_op.add_output(weight_s8);

        dqweight_op.add_input(weight_s8);
        dqweight_op.add_output(weight_f32_dq);

        matmul_op.add_input(src_f32_dq);
        matmul_op.add_input(weight_f32_dq);
        matmul_op.add_input(bias_f32);
        matmul_op.add_output(dst_f32);

        qout_op.add_input(dst_f32);
        qout_op.add_output(dst_s8);

        impl::graph_t g;
        g.add_op(&dqdata_op);
        g.add_op(&qweight_op);
        g.add_op(&dqweight_op);
        g.add_op(&matmul_op);
        g.add_op(&qout_op);
        g.build_graph();

        src_u8 = utils::logical_tensor_init(1, src_shape, impl::data_type::u8);
        weight_f32 = utils::logical_tensor_init(
                3, weight_shape, impl::data_type::f32);
        // set weight to be constant
        weight_f32.property = impl::property_type::constant;
        dst_s8 = utils::logical_tensor_init(9, dst_shape, impl::data_type::s8);
        bias_f32 = utils::logical_tensor_init(
                6, bias_shape, impl::data_type::f32);
        // set bias to be constant
        bias_f32.property = impl::property_type::constant;

        impl::tensor_t src_u8_ts(src_u8, &engine, src_data.data());
        impl::tensor_t weight_f32_ts(weight_f32, &engine, weight_data.data());
        impl::tensor_t bias_f32_ts(bias_f32, &engine, bias_data.data());
        // -------------------------case 1----------------------------------
        test::vector<int8_t> case1_out_data(product(dst_shape));
        impl::tensor_t dst_s8_ts(dst_s8, &engine, case1_out_data.data());
        ASSERT_EQ(run_graph(g, {src_u8_ts, weight_f32_ts, bias_f32_ts},
                          {dst_s8_ts}, engine, strm),
                impl::status::success);
        // -------------------------case 2----------------------------------
        impl::pass::pass_base_ptr apass
                = get_pass("int8_quant_wei_matmul_bias_fusion");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1);
        auto part = g.get_partitions()[0];

        // compile
        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> lt_ins {
                &src_u8, &weight_f32, &bias_f32};
        std::vector<const impl::logical_tensor_t *> lt_outs {&dst_s8};

        p.compile(&cp, lt_ins, lt_outs, &engine);

        test::vector<int8_t> case2_out_data(product(dst_shape));
        impl::tensor_t dst_s8_case2_ts(dst_s8, &engine, case2_out_data.data());
        for (size_t iter = 0; iter < 1; iter++) {
            cp.execute(&strm, {src_u8_ts, weight_f32_ts, bias_f32_ts},
                    {dst_s8_case2_ts});
            strm.wait();

            static auto isa = dnnl_get_effective_cpu_isa();
            if (isa < dnnl_cpu_isa_avx512_core_vnni)
                ASSERT_TRUE(
                        allclose(case1_out_data, case2_out_data, /*rtol*/ 0.1f,
                                /*atol*/ 1.f));
            else
                ASSERT_TRUE(
                        allclose(case1_out_data, case2_out_data, /*rtol*/ 0.01f,
                                /*atol*/ 1.f));
        }
    }
}

TEST(int8_subgraph_mode, int8_quant_wei_matmul_bias_relu_ndx2d) {
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    if (engine.kind() == impl::engine_kind::gpu) return;

    std::vector<bool> with_bias_types {true, false};
    std::vector<std::string> qtypes {"per_tensor", "per_channel"};
    std::vector<std::vector<int64_t>> src_shapes {
            {3, 3, 3, 8, 4}, {3, 3, 8, 4}, {3, 8, 4}, {8, 4}, {4}};
    std::vector<std::vector<int64_t>> weight_shapes {{4, 2}};
    std::vector<std::vector<int64_t>> dst_shapes {
            {3, 3, 3, 8, 2}, {3, 3, 8, 2}, {3, 8, 2}, {8, 2}, {2}};
    for_(const auto with_bias : with_bias_types)
    for_(const auto &qtype : qtypes)
    for_(size_t i = 0; i < src_shapes.size(); ++i)
    for (size_t j = 0; j < weight_shapes.size(); ++j) {
        // prepare fp32 data
        std::vector<int64_t> src_shape = src_shapes[i];
        std::vector<int64_t> weight_shape = weight_shapes[j];
        std::vector<int64_t> bias_shape {2};
        std::vector<int64_t> dst_shape = dst_shapes[i];

        test::vector<uint8_t> src_data(product(src_shape));
        test::vector<float> weight_data(product(weight_shape));
        test::vector<float> bias_data(product(bias_shape));

        // random generate src, weight and bias data
        // random seed = 7
        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> u8_distribution(0.0f, 255.0f);
        std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
        std::generate(src_data.begin(), src_data.end(), [&]() {
            return static_cast<uint8_t>(u8_distribution(generator));
        });
        std::generate(weight_data.begin(), weight_data.end(),
                [&]() { return f32_distribution(generator); });
        std::generate(bias_data.begin(), bias_data.end(),
                [&]() { return f32_distribution(generator); });
        float scale_src = 1 / 255.f; // map to 0~255
        float scale_out = 1;
        int64_t zp_src = 0;
        int64_t zp_out = 78;

        size_t scales_wei_sizes = qtype == "per_tensor" ? 1 : dst_shape.back();
        std::vector<float> scale_wei(scales_wei_sizes, 1 / 127.f);
        std::vector<int64_t> zp_wei(scales_wei_sizes, 0);

        // -------------------------case 1----------------------------------
        impl::op_t dqdata_op(1, impl::op_kind::Dequantize, "dqdata_op");
        dqdata_op.set_attr<std::string>("qtype", "per_tensor");
        dqdata_op.set_attr<std::vector<int64_t>>("zps", {zp_src});
        dqdata_op.set_attr<std::vector<float>>("scales", {scale_src});
        dqdata_op.set_attr<int64_t>("axis", 0);

        impl::op_t qweight_op(2, impl::op_kind::Quantize, "qweight_op");
        qweight_op.set_attr<std::string>("qtype", qtype);
        qweight_op.set_attr<std::vector<int64_t>>("zps", zp_wei);
        qweight_op.set_attr<std::vector<float>>("scales", scale_wei);
        qweight_op.set_attr<int64_t>("axis", 1);

        impl::op_t dqweight_op(3, impl::op_kind::Dequantize, "dqweight_op");
        dqweight_op.set_attr<std::string>("qtype", qtype);
        dqweight_op.set_attr<std::vector<int64_t>>("zps", zp_wei);
        dqweight_op.set_attr<std::vector<float>>("scales", scale_wei);
        dqweight_op.set_attr<int64_t>("axis", 1);

        impl::op_t matmul_op(4, impl::op_kind::MatMul, "matmul_op");
        matmul_op.set_attr<bool>("transpose_a", false);
        matmul_op.set_attr<bool>("transpose_b", false);

        impl::op_t relu_op(5, impl::op_kind::ReLU, "relu_op");

        impl::op_t qout_op(6, impl::op_kind::Quantize, "qout_op");
        qout_op.set_attr<std::string>("qtype", "per_tensor");
        qout_op.set_attr<std::vector<int64_t>>("zps", {zp_out});
        qout_op.set_attr<std::vector<float>>("scales", {scale_out});
        qout_op.set_attr<int64_t>("axis", 0);

        // prepare logical tensor
        impl::logical_tensor_t src_u8
                = utils::logical_tensor_init(1, src_shape, impl::data_type::u8);
        impl::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                2, src_shape, impl::data_type::f32);
        impl::logical_tensor_t weight_f32 = utils::logical_tensor_init(
                3, weight_shape, impl::data_type::f32);
        impl::logical_tensor_t weight_s8 = utils::logical_tensor_init(
                4, weight_shape, impl::data_type::s8);
        impl::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
                5, weight_shape, impl::data_type::f32);
        impl::logical_tensor_t bias_f32 = utils::logical_tensor_init(
                6, bias_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_f32 = utils::logical_tensor_init(
                7, dst_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_s8
                = utils::logical_tensor_init(8, dst_shape, impl::data_type::s8);
        impl::logical_tensor_t dst_relu_f32 = utils::logical_tensor_init(
                9, dst_shape, impl::data_type::f32);

        dqdata_op.add_input(src_u8);
        dqdata_op.add_output(src_f32_dq);

        qweight_op.add_input(weight_f32);
        qweight_op.add_output(weight_s8);

        dqweight_op.add_input(weight_s8);
        dqweight_op.add_output(weight_f32_dq);

        matmul_op.add_input(src_f32_dq);
        matmul_op.add_input(weight_f32_dq);
        if (with_bias) matmul_op.add_input(bias_f32);
        matmul_op.add_output(dst_f32);

        relu_op.add_input(dst_f32);
        relu_op.add_output(dst_relu_f32);

        qout_op.add_input(dst_relu_f32);
        qout_op.add_output(dst_s8);

        impl::graph_t g;
        g.add_op(&dqdata_op);
        g.add_op(&qweight_op);
        g.add_op(&dqweight_op);
        g.add_op(&matmul_op);
        g.add_op(&relu_op);
        g.add_op(&qout_op);
        g.build_graph();

        impl::tensor_t src_u8_ts(src_u8, &engine, src_data.data());
        impl::tensor_t weight_f32_ts(weight_f32, &engine, weight_data.data());
        impl::tensor_t bias_f32_ts(bias_f32, &engine, bias_data.data());
        // -------------------------case 1----------------------------------
        test::vector<int8_t> case1_out_data(product(dst_shape));
        impl::tensor_t dst_s8_ts(dst_s8, &engine, case1_out_data.data());
        ASSERT_EQ(run_graph(g, {src_u8_ts, weight_f32_ts, bias_f32_ts},
                          {dst_s8_ts}, engine, strm),
                impl::status::success);

        // -------------------------case 2----------------------------------
        impl::pass::pass_base_ptr apass = nullptr;
        if (with_bias)
            apass = get_pass("int8_quant_wei_matmul_bias_relu_fusion");
        else
            apass = get_pass("int8_quant_wei_matmul_relu_fusion");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1);
        auto part = g.get_partitions()[0];

        // compile
        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> lt_ins {
                &src_u8, &weight_f32, &bias_f32};
        if (!with_bias) lt_ins.pop_back();
        std::vector<const impl::logical_tensor_t *> lt_outs {&dst_s8};

        p.compile(&cp, lt_ins, lt_outs, &engine);

        test::vector<int8_t> case2_out_data(product(dst_shape));
        impl::tensor_t dst_s8_case2_ts(dst_s8, &engine, case2_out_data.data());
        for (size_t iter = 0; iter < 5; iter++) {
            if (with_bias) {
                cp.execute(&strm, {src_u8_ts, weight_f32_ts, bias_f32_ts},
                        {dst_s8_case2_ts});
            } else {
                cp.execute(
                        &strm, {src_u8_ts, weight_f32_ts}, {dst_s8_case2_ts});
            }
            strm.wait();

            static auto isa = dnnl_get_effective_cpu_isa();
            if (isa < dnnl_cpu_isa_avx512_core_vnni)
                ASSERT_TRUE(
                        allclose(case1_out_data, case2_out_data, /*rtol*/ 0.1f,
                                /*atol*/ 1.f));
            else
                ASSERT_TRUE(
                        allclose(case1_out_data, case2_out_data, /*rtol*/ 0.01f,
                                /*atol*/ 1.f));
        }
    }
}

TEST(int8_subgraph_mode, int8_matmul_2dx3d_with_transpose) {
    // compare results between:
    // case 1: [quantize] - [dequantize] - [fp32_matmul] - [quantize]
    // case 2: [quantize] - [int8_matmul]
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    if (engine.kind() == impl::engine_kind::gpu) return;

    std::vector<std::vector<int64_t>> src_shapes {{8, 4}};
    std::vector<std::vector<int64_t>> weight_shapes {{8, 2, 4}};
    std::vector<std::vector<int64_t>> dst_shapes {{8, 8, 2}};

    for_(size_t i = 0; i < src_shapes.size(); ++i)
    for (size_t j = 0; j < weight_shapes.size(); ++j) {
        // prepare fp32 data
        std::vector<int64_t> src_shape = src_shapes[i];
        std::vector<int64_t> weight_shape = weight_shapes[j];
        std::vector<int64_t> bias_shape {1};
        std::vector<int64_t> dst_shape = dst_shapes[i];

        test::vector<uint8_t> src_data(product(src_shape));
        test::vector<int8_t> weight_data(product(weight_shape));
        test::vector<float> bias_data(product(bias_shape));

        // random generate src, weight and bias data random seed = 7
        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
        std::uniform_real_distribution<float> int8_distribution(0.0f, 100.0f);
        std::generate(src_data.begin(), src_data.end(), [&]() {
            return static_cast<uint8_t>(int8_distribution(generator));
        });
        std::generate(weight_data.begin(), weight_data.end(), [&]() {
            return static_cast<int8_t>(int8_distribution(generator));
        });
        std::generate(bias_data.begin(), bias_data.end(),
                [&]() { return f32_distribution(generator); });
        float scale_src = 1 / 255.f;
        float scale_wei = 1 / 127.f;
        float scale_out = 1;
        int64_t zp_src = 0;
        int64_t zp_wei = 0;
        int64_t zp_out = 78;

        impl::op_t qdata_op(0, impl::op_kind::Quantize, "qdata_op");
        qdata_op.set_attr<std::string>("qtype", "per_tensor");
        qdata_op.set_attr<std::vector<int64_t>>("zps", {zp_src});
        qdata_op.set_attr<std::vector<float>>("scales", {scale_src});
        qdata_op.set_attr<int64_t>("axis", 0);

        impl::op_t dqdata_op(1, impl::op_kind::Dequantize, "dqdata_op");
        dqdata_op.set_attr<std::string>("qtype", "per_tensor");
        dqdata_op.set_attr<std::vector<int64_t>>("zps", {zp_src});
        dqdata_op.set_attr<std::vector<float>>("scales", {scale_src});
        dqdata_op.set_attr<int64_t>("axis", 0);

        impl::op_t qweight_op(2, impl::op_kind::Quantize, "qweight_op");
        qweight_op.set_attr<std::string>("qtype", "per_tensor");
        qweight_op.set_attr<std::vector<int64_t>>("zps", {zp_wei});
        qweight_op.set_attr<std::vector<float>>("scales", {scale_wei});
        qweight_op.set_attr<int64_t>("axis", 0);

        impl::op_t dqweight_op(3, impl::op_kind::Dequantize, "dqweight_op");
        dqweight_op.set_attr<std::string>("qtype", "per_tensor");
        dqweight_op.set_attr<std::vector<int64_t>>("zps", {zp_wei});
        dqweight_op.set_attr<std::vector<float>>("scales", {scale_wei});
        dqweight_op.set_attr<int64_t>("axis", 0);

        impl::op_t matmul_op(4, impl::op_kind::MatMul, "matmul_op");
        matmul_op.set_attr<bool>("transpose_a", false);
        matmul_op.set_attr<bool>("transpose_b", true);

        impl::op_t qout_op(5, impl::op_kind::Quantize, "qout_op");
        qout_op.set_attr<std::string>("qtype", "per_tensor");
        qout_op.set_attr<std::vector<int64_t>>("zps", {zp_out});
        qout_op.set_attr<std::vector<float>>("scales", {scale_out});
        qout_op.set_attr<int64_t>("axis", 0);

        // prepare logical tensor
        impl::logical_tensor_t src_u8
                = utils::logical_tensor_init(1, src_shape, impl::data_type::u8);
        impl::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                2, src_shape, impl::data_type::f32);
        impl::logical_tensor_t weight_s8 = utils::logical_tensor_init(
                4, weight_shape, impl::data_type::s8);
        impl::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
                5, weight_shape, impl::data_type::f32);
        impl::logical_tensor_t bias_f32 = utils::logical_tensor_init(
                6, bias_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_f32 = utils::logical_tensor_init(
                7, dst_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_s8
                = utils::logical_tensor_init(8, dst_shape, impl::data_type::s8);

        dqdata_op.add_input(src_u8);
        dqdata_op.add_output(src_f32_dq);

        dqweight_op.add_input(weight_s8);
        dqweight_op.add_output(weight_f32_dq);

        matmul_op.add_input(src_f32_dq);
        matmul_op.add_input(weight_f32_dq);
        matmul_op.add_input(bias_f32);
        matmul_op.add_output(dst_f32);

        qout_op.add_input(dst_f32);
        qout_op.add_output(dst_s8);

        impl::graph_t g;
        g.add_op(&dqdata_op);
        g.add_op(&dqweight_op);
        g.add_op(&matmul_op);
        g.add_op(&qout_op);
        g.build_graph();

        impl::pass::pass_base_ptr apass = get_pass("int8_matmul_bias_fusion");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1);
        auto part = g.get_partitions()[0];

        // compile
        impl::partition_t p;
        p.init(part);
        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> lt_ins {
                &src_u8, &weight_s8, &bias_f32};
        std::vector<const impl::logical_tensor_t *> lt_outs {&dst_s8};

        p.compile(&cp, lt_ins, lt_outs, &engine);

        // execute
        impl::tensor_t src_u8_ts(src_u8, &engine, src_data.data());
        impl::tensor_t weight_s8_ts(weight_s8, &engine, weight_data.data());
        impl::tensor_t bias_f32_ts(bias_f32, &engine, bias_data.data());
        test::vector<int8_t> dst_data(product(dst_shape));
        impl::tensor_t dst_s8_ts(dst_s8, &engine, dst_data.data());
        cp.execute(&strm, {src_u8_ts, weight_s8_ts, bias_f32_ts}, {dst_s8_ts});
        strm.wait();
    }
}

TEST(int8_subgraph_mode, int8_matmul_bias_sum_get_inplace_pair) {
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    if (engine.kind() == impl::engine_kind::gpu) return;

    std::vector<std::string> qtypes {"per_tensor", "per_channel"};
    std::vector<std::string> other_qtypes = {"symmetric", "asymmetric"};
    std::vector<std::vector<int64_t>> src_shapes {
            {3, 3, 3, 8, 4}, {3, 3, 8, 4}, {3, 8, 4}, {8, 4}, {4}};
    std::vector<std::vector<int64_t>> weight_shapes {{4, 2}};
    std::vector<std::vector<int64_t>> dst_shapes {
            {3, 3, 3, 8, 2}, {3, 3, 8, 2}, {3, 8, 2}, {8, 2}, {2}};
    for_(const auto &qtype : qtypes)
    for_(size_t i = 0; i < src_shapes.size(); ++i)
    for_(const auto &other_qtype : other_qtypes)
    for (size_t j = 0; j < weight_shapes.size(); ++j) {
        // prepare fp32 data
        std::vector<int64_t> src_shape = src_shapes[i];
        std::vector<int64_t> weight_shape = weight_shapes[j];
        std::vector<int64_t> bias_shape {2};
        std::vector<int64_t> dst_shape = dst_shapes[i];

        float scale_src = 1 / 255.f; // map to 0~255
        float scale_other = 1 / 127.f;
        float scale_out = 1;
        int64_t zp_src = 0;
        int64_t zp_other = other_qtype == "symmetric" ? 0 : 128;
        int64_t zp_out = 78;

        size_t scales_wei_sizes = qtype == "per_tensor" ? 1 : dst_shape.back();
        std::vector<float> scale_wei(scales_wei_sizes, 1 / 127.f);
        std::vector<int64_t> zp_wei(scales_wei_sizes, 0);

        impl::op_t dqdata_op(1, impl::op_kind::Dequantize, "dqdata_op");
        dqdata_op.set_attr<std::string>("qtype", "per_tensor");
        dqdata_op.set_attr<std::vector<int64_t>>("zps", {zp_src});
        dqdata_op.set_attr<std::vector<float>>("scales", {scale_src});
        dqdata_op.set_attr<int64_t>("axis", 0);

        impl::op_t dqweight_op(3, impl::op_kind::Dequantize, "dqweight_op");
        dqweight_op.set_attr<std::string>("qtype", qtype);
        dqweight_op.set_attr<std::vector<int64_t>>("zps", zp_wei);
        dqweight_op.set_attr<std::vector<float>>("scales", scale_wei);
        dqweight_op.set_attr<int64_t>("axis", 1);

        impl::op_t matmul_op(4, impl::op_kind::MatMul, "matmul_op");
        matmul_op.set_attr<bool>("transpose_a", false);
        matmul_op.set_attr<bool>("transpose_b", false);

        impl::op_t add_op(8, impl::op_kind::Add, "add_op");

        impl::op_t qout_op(5, impl::op_kind::Quantize, "qout_op");
        qout_op.set_attr<std::string>("qtype", "per_tensor");
        qout_op.set_attr<std::vector<int64_t>>("zps", {zp_out});
        qout_op.set_attr<std::vector<float>>("scales", {scale_out});
        qout_op.set_attr<int64_t>("axis", 0);

        impl::op_t dqdata_op2(9, impl::op_kind::Dequantize, "dqdata_op");
        dqdata_op2.set_attr<std::string>("qtype", "per_tensor");
        dqdata_op2.set_attr<std::vector<int64_t>>("zps", {zp_src});
        dqdata_op2.set_attr<std::vector<float>>("scales", {scale_src});
        dqdata_op2.set_attr<int64_t>("axis", 0);

        impl::op_t dqweight_op2(10, impl::op_kind::Dequantize, "dqweight_op");
        dqweight_op2.set_attr<std::string>("qtype", qtype);
        dqweight_op2.set_attr<std::vector<int64_t>>("zps", zp_wei);
        dqweight_op2.set_attr<std::vector<float>>("scales", scale_wei);
        dqweight_op2.set_attr<int64_t>("axis", 1);

        impl::op_t matmul_op2(11, impl::op_kind::MatMul, "matmul_op");
        matmul_op2.set_attr<bool>("transpose_a", false);
        matmul_op2.set_attr<bool>("transpose_b", false);

        impl::op_t qout_op2(6, impl::op_kind::Quantize, "qother_op");
        qout_op2.set_attr<std::string>("qtype", "per_tensor");
        qout_op2.set_attr<std::vector<int64_t>>("zps", {zp_other});
        qout_op2.set_attr<std::vector<float>>("scales", {scale_other});
        qout_op2.set_attr<int64_t>("axis", 0);

        impl::op_t dqout_o2p(7, impl::op_kind::Dequantize, "dqother_op");
        dqout_o2p.set_attr<std::string>("qtype", "per_tensor");
        dqout_o2p.set_attr<std::vector<int64_t>>("zps", {zp_other});
        dqout_o2p.set_attr<std::vector<float>>("scales", {scale_other});
        dqout_o2p.set_attr<int64_t>("axis", 0);

        // prepare logical tensor
        impl::logical_tensor_t src_u8
                = utils::logical_tensor_init(1, src_shape, impl::data_type::u8);
        impl::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                2, src_shape, impl::data_type::f32);
        impl::logical_tensor_t weight_s8 = utils::logical_tensor_init(
                4, weight_shape, impl::data_type::s8);
        impl::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
                5, weight_shape, impl::data_type::f32);
        impl::logical_tensor_t bias_f32 = utils::logical_tensor_init(
                6, bias_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_f32 = utils::logical_tensor_init(
                7, dst_shape, impl::data_type::f32);

        impl::logical_tensor_t src_u8_2 = utils::logical_tensor_init(
                21, src_shape, impl::data_type::u8);
        impl::logical_tensor_t src_f32_dq_2 = utils::logical_tensor_init(
                22, src_shape, impl::data_type::f32);
        impl::logical_tensor_t weight_s8_2 = utils::logical_tensor_init(
                24, weight_shape, impl::data_type::s8);
        impl::logical_tensor_t weight_f32_dq_2 = utils::logical_tensor_init(
                25, weight_shape, impl::data_type::f32);
        impl::logical_tensor_t bias_f32_2 = utils::logical_tensor_init(
                26, bias_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_f32_2 = utils::logical_tensor_init(
                27, dst_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_s8_2 = utils::logical_tensor_init(
                28, dst_shape, impl::data_type::s8);
        impl::logical_tensor_t dst_f32_dq_2 = utils::logical_tensor_init(
                29, dst_shape, impl::data_type::f32);

        impl::logical_tensor_t dst_add_f32 = utils::logical_tensor_init(
                12, dst_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_s8
                = utils::logical_tensor_init(8, dst_shape, impl::data_type::s8);

        dqdata_op2.add_input(src_u8_2);
        dqdata_op2.add_output(src_f32_dq_2);
        dqweight_op2.add_input(weight_s8_2);
        dqweight_op2.add_output(weight_f32_dq_2);
        matmul_op2.add_input(src_f32_dq_2);
        matmul_op2.add_input(weight_f32_dq_2);
        matmul_op2.add_input(bias_f32_2);
        matmul_op2.add_output(dst_f32_2);
        qout_op2.add_input(dst_f32_2);
        qout_op2.add_output(dst_s8_2);
        dqout_o2p.add_input(dst_s8_2);
        dqout_o2p.add_output(dst_f32_dq_2);

        dqdata_op.add_input(src_u8);
        dqdata_op.add_output(src_f32_dq);
        dqweight_op.add_input(weight_s8);
        dqweight_op.add_output(weight_f32_dq);
        matmul_op.add_input(src_f32_dq);
        matmul_op.add_input(weight_f32_dq);
        matmul_op.add_input(bias_f32);
        matmul_op.add_output(dst_f32);
        add_op.add_input(dst_f32);
        add_op.add_input(dst_f32_dq_2);
        add_op.add_output(dst_add_f32);

        qout_op.add_input(dst_add_f32);
        qout_op.add_output(dst_s8);

        impl::graph_t g;
        g.add_op(&dqdata_op2);
        g.add_op(&dqweight_op2);
        g.add_op(&matmul_op2);
        g.add_op(&qout_op2);
        g.add_op(&dqout_o2p);

        g.add_op(&dqdata_op);
        g.add_op(&dqweight_op);
        g.add_op(&matmul_op);
        g.add_op(&add_op);
        g.add_op(&qout_op);
        g.build_graph();

        impl::pass::pass_base_ptr apass1 = get_pass("int8_matmul_bias_fusion");
        impl::pass::pass_base_ptr apass2
                = get_pass("int8_matmul_bias_add_fusion");
        apass1->run(g);
        apass2->run(g);
        ASSERT_EQ(g.get_num_partitions(), 2);
        auto part1 = g.get_partitions()[0]; // int8_mamtul_bias
        auto part2 = g.get_partitions()[1]; // int8_matmul_bias_sum

        // compile
        impl::partition_t p1, p2;
        p1.init(part1);
        p2.init(part2);

        impl::compiled_partition_t cp1(p1);
        impl::compiled_partition_t cp2(p2);

        dst_s8_2.layout_type = impl::layout_type::any;
        std::vector<const impl::logical_tensor_t *> lt_ins1 {
                &src_u8_2, &weight_s8_2, &bias_f32_2};
        std::vector<const impl::logical_tensor_t *> lt_outs1 {&dst_s8_2};
        p1.compile(&cp1, lt_ins1, lt_outs1, &engine);

        cp1.query_logical_tensor(dst_s8_2.id, &dst_s8_2);

        dst_s8.layout_type = impl::layout_type::any;
        std::vector<const impl::logical_tensor_t *> lt_ins2 {
                &src_u8, &weight_s8, &bias_f32, &dst_s8_2};
        std::vector<const impl::logical_tensor_t *> lt_outs2 {&dst_s8};
        p2.compile(&cp2, lt_ins2, lt_outs2, &engine);

        std::vector<impl::inplace_pair_t> inplace_pairs
                = cp2.get_inplace_pairs();

        ASSERT_EQ(inplace_pairs.size(), 1);
        ASSERT_EQ(inplace_pairs[0].input, dst_s8_2.id);
        ASSERT_EQ(inplace_pairs[0].output, dst_s8.id);
    }
}

TEST(operator_kernel, mul_add_per_tensor_broadcast) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src0 {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> src1 {2.0};
    test::vector<float> post_src {2.0};
    test::vector<float> ref_dst {6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0};
    test::vector<float> dst(src0.size(), 0.0);

    impl::op_t mul_op(impl::dnnl_impl::op_kind::multiply_add);

    auto &op_factory = get_dnnl_kernel_registry();
    auto mul_kernel = op_factory.create_kernel(mul_op);
    ASSERT_TRUE(mul_kernel);

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1}, impl::data_type::f32);
    impl::logical_tensor_t post_src_lt
            = utils::logical_tensor_init(2, {1}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(3, {1, 1, 3, 3}, impl::data_type::f32);

    // compile the add operator
    mul_kernel->compile(
            &mul_op, &eng, {src0_lt, src1_lt, post_src_lt}, {dst_lt});

    impl::tensor_t src0_ts(src0_lt, &eng, src0.data());
    impl::tensor_t src1_ts(src1_lt, &eng, src1.data());
    impl::tensor_t post_src_ts(post_src_lt, &eng, post_src.data());
    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    mul_kernel->execute(
            &mul_op, &strm, {src0_ts, src1_ts, post_src_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, mul_add_per_hw_broadcast) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src0(18, 2.0);
    test::vector<float> src1(1, 2.0);
    test::vector<float> post_src(6, 2.0);
    test::vector<float> ref_dst(18, 6.0);
    test::vector<float> dst(src0.size(), 0.0);

    impl::op_t mul_op(impl::dnnl_impl::op_kind::multiply_add);

    auto &op_factory = get_dnnl_kernel_registry();
    auto mul_kernel = op_factory.create_kernel(mul_op);
    ASSERT_TRUE(mul_kernel);

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 3, 2, 3}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1}, impl::data_type::f32);
    impl::logical_tensor_t post_src_lt
            = utils::logical_tensor_init(2, {2, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(3, {1, 3, 2, 3}, impl::data_type::f32);

    // compile the add operator
    mul_kernel->compile(
            &mul_op, &eng, {src0_lt, src1_lt, post_src_lt}, {dst_lt});

    impl::tensor_t src0_ts(src0_lt, &eng, src0.data());
    impl::tensor_t src1_ts(src1_lt, &eng, src1.data());
    impl::tensor_t post_src_ts(post_src_lt, &eng, post_src.data());
    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    mul_kernel->execute(
            &mul_op, &strm, {src0_ts, src1_ts, post_src_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(operator_kernel, mul_add_per_channel_broadcast) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src0 {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> src1 {2.0};
    test::vector<float> post_src {2.0, 2.0, 2.0};
    test::vector<float> ref_dst {6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0};
    test::vector<float> dst(src0.size(), 0.0);

    impl::op_t mul_op(impl::dnnl_impl::op_kind::multiply_add);

    auto &op_factory = get_dnnl_kernel_registry();
    auto mul_kernel = op_factory.create_kernel(mul_op);
    ASSERT_TRUE(mul_kernel);

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 3, 1, 3}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1}, impl::data_type::f32);
    impl::logical_tensor_t post_src_lt
            = utils::logical_tensor_init(2, {3, 1, 1}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(3, {1, 3, 1, 3}, impl::data_type::f32);

    // compile the add operator
    mul_kernel->compile(
            &mul_op, &eng, {src0_lt, src1_lt, post_src_lt}, {dst_lt});

    impl::tensor_t src0_ts(src0_lt, &eng, src0.data());
    impl::tensor_t src1_ts(src1_lt, &eng, src1.data());
    impl::tensor_t post_src_ts(post_src_lt, &eng, post_src.data());
    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    mul_kernel->execute(
            &mul_op, &strm, {src0_ts, src1_ts, post_src_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(int8_subgraph_mode, u8u8f32_bmm) {
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    if (engine.kind() == impl::engine_kind::gpu) return;
    std::string qtype = "per_tensor";
    // prepare fp32 data
    std::vector<int64_t> src_shape = {1, 4, 16, 8};
    std::vector<int64_t> weight_shape = {1, 4, 8, 16};
    std::vector<int64_t> dst_shape = {1, 4, 16, 16};

    test::vector<uint8_t> src_data(product(src_shape));
    test::vector<uint8_t> weight_data(product(weight_shape));

    // random generate src, weight data
    // random seed = 7
    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> u8_distribution(0.0f, 255.0f);
    std::generate(src_data.begin(), src_data.end(),
            [&]() { return static_cast<uint8_t>(u8_distribution(generator)); });
    std::generate(weight_data.begin(), weight_data.end(),
            [&]() { return static_cast<uint8_t>(u8_distribution(generator)); });
    float scale_src = 1 / 255.f; // map to 0~255
    int64_t zp_src = 110;

    size_t scales_wei_sizes = qtype == "per_tensor" ? 1 : dst_shape.back();
    std::vector<float> scale_wei(scales_wei_sizes, 1 / 255.f);
    std::vector<int64_t> zp_wei(scales_wei_sizes, 114);

    // -------------------------case 1----------------------------------
    impl::op_t dqdata_op(1, impl::op_kind::Dequantize, "dqdata_op");
    dqdata_op.set_attr<std::string>("qtype", "per_tensor");
    dqdata_op.set_attr<std::vector<int64_t>>("zps", {zp_src});
    dqdata_op.set_attr<std::vector<float>>("scales", {scale_src});
    dqdata_op.set_attr<int64_t>("axis", 0);

    impl::op_t dqweight_op(2, impl::op_kind::Dequantize, "dqweight_op");
    dqweight_op.set_attr<std::string>("qtype", qtype);
    dqweight_op.set_attr<std::vector<int64_t>>("zps", zp_wei);
    dqweight_op.set_attr<std::vector<float>>("scales", scale_wei);
    dqweight_op.set_attr<int64_t>("axis", 1);

    impl::op_t matmul_op(3, impl::op_kind::MatMul, "matmul_op");
    matmul_op.set_attr<bool>("transpose_a", false);
    matmul_op.set_attr<bool>("transpose_b", false);

    // prepare logical tensor
    impl::logical_tensor_t src_u8
            = utils::logical_tensor_init(1, src_shape, impl::data_type::u8);
    impl::logical_tensor_t src_f32_dq
            = utils::logical_tensor_init(2, src_shape, impl::data_type::f32);
    impl::logical_tensor_t weight_u8
            = utils::logical_tensor_init(4, weight_shape, impl::data_type::u8);
    impl::logical_tensor_t weight_f32_dq
            = utils::logical_tensor_init(5, weight_shape, impl::data_type::f32);
    impl::logical_tensor_t dst_f32
            = utils::logical_tensor_init(7, dst_shape, impl::data_type::f32);

    dqdata_op.add_input(src_u8);
    dqdata_op.add_output(src_f32_dq);

    dqweight_op.add_input(weight_u8);
    dqweight_op.add_output(weight_f32_dq);

    matmul_op.add_input(src_f32_dq);
    matmul_op.add_input(weight_f32_dq);
    matmul_op.add_output(dst_f32);

    impl::graph_t g;
    g.add_op(&dqdata_op);
    g.add_op(&dqweight_op);
    g.add_op(&matmul_op);
    g.build_graph();

    impl::tensor_t src_u8_ts(src_u8, &engine, src_data.data());
    impl::tensor_t weight_u8_ts(weight_u8, &engine, weight_data.data());
    // -------------------------case 1----------------------------------
    test::vector<float> case1_out_data(product(dst_shape));
    impl::tensor_t dst_f32_ts(dst_f32, &engine, case1_out_data.data());
    ASSERT_EQ(
            run_graph(g, {src_u8_ts, weight_u8_ts}, {dst_f32_ts}, engine, strm),
            impl::status::success);
    // -------------------------case 2----------------------------------
    impl::pass::pass_base_ptr apass = get_pass("x8x8f32_matmul_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> lt_ins {&src_u8, &weight_u8};
    std::vector<const impl::logical_tensor_t *> lt_outs {&dst_f32};

    p.compile(&cp, lt_ins, lt_outs, &engine);

    test::vector<float> case2_out_data(product(dst_shape));
    impl::tensor_t dst_f32_case2_ts(dst_f32, &engine, case2_out_data.data());
    cp.execute(&strm, {src_u8_ts, weight_u8_ts}, {dst_f32_case2_ts});
    strm.wait();

    static auto isa = dnnl_get_effective_cpu_isa();
    if (isa >= dnnl_cpu_isa_avx512_core_vnni)
        ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.01f,
                /*atol*/ 1.f));
}

TEST(int8_subgraph_mode, u8u8f32_bmm_div) {
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    if (engine.kind() == impl::engine_kind::gpu) return;
    std::string qtype = "per_tensor";
    // prepare fp32 data
    std::vector<int64_t> src_shape = {1, 4, 16, 8};
    std::vector<int64_t> weight_shape = {1, 4, 8, 16};
    std::vector<int64_t> dst_shape = {1, 4, 16, 16};

    test::vector<uint8_t> src_data(product(src_shape));
    test::vector<uint8_t> weight_data(product(weight_shape));

    // random generate src, weight data
    // random seed = 7
    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> u8_distribution(0.0f, 255.0f);
    std::generate(src_data.begin(), src_data.end(),
            [&]() { return static_cast<uint8_t>(u8_distribution(generator)); });
    std::generate(weight_data.begin(), weight_data.end(),
            [&]() { return static_cast<uint8_t>(u8_distribution(generator)); });
    test::vector<float> div_src1_data {0.5f};
    float scale_src = 1 / 255.f; // map to 0~255
    int64_t zp_src = 110;

    size_t scales_wei_sizes = qtype == "per_tensor" ? 1 : dst_shape.back();
    std::vector<float> scale_wei(scales_wei_sizes, 1 / 255.f);
    std::vector<int64_t> zp_wei(scales_wei_sizes, 114);

    // -------------------------case 1----------------------------------
    impl::op_t dqdata_op(1, impl::op_kind::Dequantize, "dqdata_op");
    dqdata_op.set_attr<std::string>("qtype", "per_tensor");
    dqdata_op.set_attr<std::vector<int64_t>>("zps", {zp_src});
    dqdata_op.set_attr<std::vector<float>>("scales", {scale_src});
    dqdata_op.set_attr<int64_t>("axis", 0);

    impl::op_t dqweight_op(2, impl::op_kind::Dequantize, "dqweight_op");
    dqweight_op.set_attr<std::string>("qtype", qtype);
    dqweight_op.set_attr<std::vector<int64_t>>("zps", zp_wei);
    dqweight_op.set_attr<std::vector<float>>("scales", scale_wei);
    dqweight_op.set_attr<int64_t>("axis", 1);

    impl::op_t matmul_op(3, impl::op_kind::MatMul, "matmul_op");
    matmul_op.set_attr<bool>("transpose_a", false);
    matmul_op.set_attr<bool>("transpose_b", false);

    impl::op_t binary_op(5, impl::op_kind::Divide, "binary_div");
    binary_op.set_attr<string>("auto_broadcast", "numpy");

    // prepare logical tensor
    impl::logical_tensor_t src_u8
            = utils::logical_tensor_init(1, src_shape, impl::data_type::u8);
    impl::logical_tensor_t src_f32_dq
            = utils::logical_tensor_init(2, src_shape, impl::data_type::f32);
    impl::logical_tensor_t weight_u8
            = utils::logical_tensor_init(4, weight_shape, impl::data_type::u8);
    impl::logical_tensor_t weight_f32_dq
            = utils::logical_tensor_init(5, weight_shape, impl::data_type::f32);
    impl::logical_tensor_t dst_f32
            = utils::logical_tensor_init(7, dst_shape, impl::data_type::f32);
    impl::logical_tensor_t div_src1
            = utils::logical_tensor_init(8, {1}, impl::data_type::f32);
    impl::logical_tensor_t div_f32
            = utils::logical_tensor_init(9, dst_shape, impl::data_type::f32);

    dqdata_op.add_input(src_u8);
    dqdata_op.add_output(src_f32_dq);

    dqweight_op.add_input(weight_u8);
    dqweight_op.add_output(weight_f32_dq);

    matmul_op.add_input(src_f32_dq);
    matmul_op.add_input(weight_f32_dq);
    matmul_op.add_output(dst_f32);

    binary_op.add_input(dst_f32);
    binary_op.add_input(div_src1);
    binary_op.add_output(div_f32);

    impl::graph_t g;
    g.add_op(&dqdata_op);
    g.add_op(&dqweight_op);
    g.add_op(&matmul_op);
    g.add_op(&binary_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("x8x8f32_matmul_div_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> lt_ins {
            &src_u8, &weight_u8, &div_src1};
    std::vector<const impl::logical_tensor_t *> lt_outs {&div_f32};

    p.compile(&cp, lt_ins, lt_outs, &engine);

    impl::tensor_t src_u8_ts(src_u8, &engine, src_data.data());
    impl::tensor_t weight_u8_ts(weight_u8, &engine, weight_data.data());
    impl::tensor_t div_src1_ts(div_src1, &engine, div_src1_data.data());
    test::vector<float> case2_out_data(product(dst_shape));
    impl::tensor_t dst_f32_case2_ts(div_f32, &engine, case2_out_data.data());
    cp.execute(
            &strm, {src_u8_ts, weight_u8_ts, div_src1_ts}, {dst_f32_case2_ts});
    strm.wait();
}

TEST(int8_subgraph_mode, x8x8bf16_bmm_div) {
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    if (engine.kind() == impl::engine_kind::gpu) return;
    std::vector<std::string> dtypes = {"uint8", "int8"};

    std::vector<int64_t> src_shape = {1, 4, 16, 8};
    std::vector<int64_t> weight_shape = {1, 4, 8, 16};
    std::vector<int64_t> dst_shape = {1, 4, 16, 16};

    test::vector<uint8_t> src_data(product(src_shape));
    test::vector<uint8_t> weight_data(product(weight_shape));

    for (auto &src_dtype : dtypes) {
        for (auto &weight_dtype : dtypes) {
            auto src_lt_dtype = (src_dtype == "uint8") ? impl::data_type::u8
                                                       : impl::data_type::s8;
            auto weight_lt_dtype = (weight_dtype == "uint8")
                    ? impl::data_type::u8
                    : impl::data_type::s8;
            float src_range = (src_dtype == "uint8") ? 255.f : 127.f;
            float weight_range = (weight_dtype == "uint8") ? 255.f : 127.f;
            // random generate src, weight data
            // random seed = 7
            std::default_random_engine generator(7);
            std::uniform_real_distribution<float> src_distribution(
                    0.0f, src_range);
            std::uniform_real_distribution<float> weight_distribution(
                    0.0f, weight_range);
            std::generate(src_data.begin(), src_data.end(),
                    [&]() { return src_distribution(generator); });
            std::generate(weight_data.begin(), weight_data.end(),
                    [&]() { return weight_distribution(generator); });
            float scale_src = 1 / src_range;
            float scale_wei = 1 / weight_range;
            int64_t zp_src = 110;
            int64_t zp_wei = 114;

            impl::op_t dqdata_op(0, impl::op_kind::Dequantize, "dqdata_op");
            dqdata_op.set_attr<std::string>("qtype", "per_tensor");
            dqdata_op.set_attr<std::vector<int64_t>>("zps", {zp_src});
            dqdata_op.set_attr<std::vector<float>>("scales", {scale_src});
            dqdata_op.set_attr<int64_t>("axis", 0);

            impl::op_t dqweight_op(1, impl::op_kind::Dequantize, "dqweight_op");
            dqweight_op.set_attr<std::string>("qtype", "per_tensor");
            dqweight_op.set_attr<std::vector<int64_t>>("zps", {zp_wei});
            dqweight_op.set_attr<std::vector<float>>("scales", {scale_wei});
            dqweight_op.set_attr<int64_t>("axis", 1);

            impl::op_t tcdata_op {2, impl::op_kind::TypeCast, "typecast_data"};
            impl::op_t tcweight_op {
                    3, impl::op_kind::TypeCast, "typecast_weight"};

            impl::op_t matmul_op(4, impl::op_kind::MatMul, "matmul_op");
            matmul_op.set_attr<bool>("transpose_a", false);
            matmul_op.set_attr<bool>("transpose_b", false);

            impl::op_t binary_op(5, impl::op_kind::Divide, "binary_div");
            binary_op.set_attr<string>("auto_broadcast", "numpy");

            // prepare logical tensor
            impl::logical_tensor_t src
                    = utils::logical_tensor_init(0, src_shape, src_lt_dtype);
            impl::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                    1, src_shape, impl::data_type::f32);
            impl::logical_tensor_t src_bf16 = utils::logical_tensor_init(
                    2, src_shape, impl::data_type::bf16);
            impl::logical_tensor_t weight = utils::logical_tensor_init(
                    3, weight_shape, weight_lt_dtype);
            impl::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
                    4, weight_shape, impl::data_type::f32);
            impl::logical_tensor_t weight_bf16 = utils::logical_tensor_init(
                    5, weight_shape, impl::data_type::bf16);
            impl::logical_tensor_t dst_bf16 = utils::logical_tensor_init(
                    6, dst_shape, impl::data_type::bf16);
            impl::logical_tensor_t div_src1
                    = utils::logical_tensor_init(7, {1}, impl::data_type::f32);
            impl::logical_tensor_t div_bf16 = utils::logical_tensor_init(
                    8, dst_shape, impl::data_type::bf16);

            dqdata_op.add_input(src);
            dqdata_op.add_output(src_f32_dq);

            dqweight_op.add_input(weight);
            dqweight_op.add_output(weight_f32_dq);

            tcdata_op.add_input(src_f32_dq);
            tcdata_op.add_output(src_bf16);

            tcweight_op.add_input(weight_f32_dq);
            tcweight_op.add_output(weight_bf16);

            matmul_op.add_input(src_bf16);
            matmul_op.add_input(weight_bf16);
            matmul_op.add_output(dst_bf16);

            binary_op.add_input(dst_bf16);
            binary_op.add_input(div_src1);
            binary_op.add_output(div_bf16);

            impl::graph_t g;
            g.add_op(&dqdata_op);
            g.add_op(&dqweight_op);
            g.add_op(&matmul_op);
            g.add_op(&tcdata_op);
            g.add_op(&tcweight_op);
            g.add_op(&binary_op);
            g.build_graph();

            impl::pass::pass_base_ptr apass
                    = get_pass("x8x8bf16_matmul_div_fusion");
            apass->run(g);
            ASSERT_EQ(g.get_num_partitions(), 1);
            auto part = g.get_partitions()[0];

            // compile
            impl::partition_t p;
            p.init(part);

            impl::compiled_partition_t cp(p);

            std::vector<const impl::logical_tensor_t *> lt_ins {
                    &src, &weight, &div_src1};
            std::vector<const impl::logical_tensor_t *> lt_outs {&div_bf16};

            p.compile(&cp, lt_ins, lt_outs, &engine);

            test::vector<float> div_src1_data(1);
            test::vector<float> dst_data(product(dst_shape));
            impl::tensor_t src_ts(src, &engine, src_data.data());
            impl::tensor_t weight_ts(weight, &engine, weight_data.data());
            impl::tensor_t div_src1_ts(div_src1, &engine, div_src1_data.data());
            impl::tensor_t dst_ts(div_bf16, &engine, dst_data.data());
            cp.execute(&strm, {src_ts, weight_ts, div_src1_ts}, {dst_ts});
            strm.wait();
        }
    }
}

TEST(int8_subgraph_mode, u8s8bf16_matmul_bias) {
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    if (engine.kind() == impl::engine_kind::gpu) return;
    std::string qtype = "per_channel";
    std::vector<int64_t> src_shape = {1, 8, 16};
    std::vector<int64_t> weight_shape = {8, 16};
    std::vector<int64_t> bias_shape = {8};
    std::vector<int64_t> dst_shape = {1, 8, 8};

    test::vector<uint8_t> src_data(product(src_shape));
    test::vector<int8_t> weight_data(product(weight_shape));
    test::vector<float> bias_data(product(bias_shape));

    // random generate src, weight data
    // random seed = 7
    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> distribution(0.0f, 255.0f);
    std::generate(src_data.begin(), src_data.end(),
            [&]() { return distribution(generator); });
    std::uniform_real_distribution<float> distribution2(-127.0f, 127.0f);
    std::generate(weight_data.begin(), weight_data.end(),
            [&]() { return distribution2(generator); });
    std::uniform_real_distribution<float> distribution3(0.0f, 20.0f);
    std::generate(bias_data.begin(), bias_data.end(),
            [&]() { return distribution3(generator); });
    float scale_src = 1 / 255.f; // map to 0~255
    int64_t zp_src = 110;

    size_t scales_wei_sizes = qtype == "per_tensor" ? 1 : dst_shape.back();
    std::vector<float> scale_wei(scales_wei_sizes, 1 / 127.f);
    std::vector<int64_t> zp_wei(scales_wei_sizes, 0);

    impl::op_t dqdata_op(0, impl::op_kind::Dequantize, "dqdata_op");
    dqdata_op.set_attr<std::string>("qtype", "per_tensor");
    dqdata_op.set_attr<std::vector<int64_t>>("zps", {zp_src});
    dqdata_op.set_attr<std::vector<float>>("scales", {scale_src});
    dqdata_op.set_attr<int64_t>("axis", 0);

    impl::op_t dqweight_op(1, impl::op_kind::Dequantize, "dqweight_op");
    dqweight_op.set_attr<std::string>("qtype", qtype);
    dqweight_op.set_attr<std::vector<int64_t>>("zps", zp_wei);
    dqweight_op.set_attr<std::vector<float>>("scales", scale_wei);
    dqweight_op.set_attr<int64_t>("axis", 1);

    impl::op_t tcdata_op {2, impl::op_kind::TypeCast, "typecast_data"};
    impl::op_t tcweight_op {3, impl::op_kind::TypeCast, "typecast_weight"};

    impl::op_t matmul_op(4, impl::op_kind::MatMul, "matmul_op");
    matmul_op.set_attr<bool>("transpose_a", false);
    matmul_op.set_attr<bool>("transpose_b", true);

    // prepare logical tensor
    impl::logical_tensor_t src_u8
            = utils::logical_tensor_init(0, src_shape, impl::data_type::u8);
    impl::logical_tensor_t src_f32_dq
            = utils::logical_tensor_init(1, src_shape, impl::data_type::f32);
    impl::logical_tensor_t src_bf16
            = utils::logical_tensor_init(2, src_shape, impl::data_type::bf16);
    impl::logical_tensor_t weight_s8
            = utils::logical_tensor_init(3, weight_shape, impl::data_type::s8);
    impl::logical_tensor_t weight_f32_dq
            = utils::logical_tensor_init(4, weight_shape, impl::data_type::f32);
    impl::logical_tensor_t weight_bf16 = utils::logical_tensor_init(
            5, weight_shape, impl::data_type::bf16);
    impl::logical_tensor_t bias_f32
            = utils::logical_tensor_init(6, bias_shape, impl::data_type::f32);
    impl::logical_tensor_t dst_bf16
            = utils::logical_tensor_init(7, dst_shape, impl::data_type::bf16);

    dqdata_op.add_input(src_u8);
    dqdata_op.add_output(src_f32_dq);

    dqweight_op.add_input(weight_s8);
    dqweight_op.add_output(weight_f32_dq);

    tcdata_op.add_input(src_f32_dq);
    tcdata_op.add_output(src_bf16);

    tcweight_op.add_input(weight_f32_dq);
    tcweight_op.add_output(weight_bf16);

    matmul_op.add_input(src_bf16);
    matmul_op.add_input(weight_bf16);
    matmul_op.add_input(bias_f32);
    matmul_op.add_output(dst_bf16);

    impl::graph_t g;
    g.add_op(&dqdata_op);
    g.add_op(&dqweight_op);
    g.add_op(&matmul_op);
    g.add_op(&tcdata_op);
    g.add_op(&tcweight_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("x8s8bf16_matmul_bias_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> lt_ins {
            &src_u8, &weight_s8, &bias_f32};
    std::vector<const impl::logical_tensor_t *> lt_outs {&dst_bf16};

    p.compile(&cp, lt_ins, lt_outs, &engine);

    test::vector<float> dst_data(product(dst_shape));
    impl::tensor_t src_u8_ts(src_u8, &engine, src_data.data());
    impl::tensor_t weight_s8_ts(weight_s8, &engine, weight_data.data());
    impl::tensor_t bias_f32_ts(bias_f32, &engine, bias_data.data());
    impl::tensor_t dst_ts(dst_bf16, &engine, dst_data.data());
    cp.execute(&strm, {src_u8_ts, weight_s8_ts, bias_f32_ts}, {dst_ts});
    strm.wait();
}

TEST(int8_subgraph_mode, u8s8bf16_matmul_bias_add) {
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    if (engine.kind() == impl::engine_kind::gpu) return;
    std::string qtype = "per_channel";
    std::vector<int64_t> src_shape = {1, 8, 16};
    std::vector<int64_t> weight_shape = {8, 16};
    std::vector<int64_t> bias_shape = {8};
    std::vector<int64_t> dst_shape = {1, 8, 8};

    test::vector<uint8_t> src_data(product(src_shape));
    test::vector<int8_t> weight_data(product(weight_shape));
    test::vector<uint8_t> other_data(product(dst_shape));
    test::vector<float> bias_data(product(bias_shape));

    // random generate src, weight data
    // random seed = 7
    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> distribution(0.0f, 255.0f);
    std::generate(src_data.begin(), src_data.end(),
            [&]() { return distribution(generator); });
    std::generate(other_data.begin(), other_data.end(),
            [&]() { return distribution(generator); });
    std::uniform_real_distribution<float> distribution2(-127.0f, 127.0f);
    std::generate(weight_data.begin(), weight_data.end(),
            [&]() { return distribution2(generator); });
    std::uniform_real_distribution<float> distribution3(0.0f, 20.0f);
    std::generate(bias_data.begin(), bias_data.end(),
            [&]() { return distribution3(generator); });
    float scale_src = 1 / 255.f; // map to 0~255
    int64_t zp_src = 110;

    size_t scales_wei_sizes = qtype == "per_tensor" ? 1 : dst_shape.back();
    std::vector<float> scale_wei(scales_wei_sizes, 1 / 127.f);
    std::vector<int64_t> zp_wei(scales_wei_sizes, 0);

    impl::op_t dqdata_op(0, impl::op_kind::Dequantize, "dqdata_op");
    dqdata_op.set_attr<std::string>("qtype", "per_tensor");
    dqdata_op.set_attr<std::vector<int64_t>>("zps", {zp_src});
    dqdata_op.set_attr<std::vector<float>>("scales", {scale_src});
    dqdata_op.set_attr<int64_t>("axis", 0);

    impl::op_t dqweight_op(1, impl::op_kind::Dequantize, "dqweight_op");
    dqweight_op.set_attr<std::string>("qtype", qtype);
    dqweight_op.set_attr<std::vector<int64_t>>("zps", zp_wei);
    dqweight_op.set_attr<std::vector<float>>("scales", scale_wei);
    dqweight_op.set_attr<int64_t>("axis", 1);

    impl::op_t dqother_op(2, impl::op_kind::Dequantize, "dqother_op");
    dqother_op.set_attr<std::string>("qtype", qtype);
    dqother_op.set_attr<std::vector<int64_t>>("zps", {zp_src});
    dqother_op.set_attr<std::vector<float>>("scales", {scale_src});
    dqother_op.set_attr<int64_t>("axis", 1);

    impl::op_t tcdata_op {3, impl::op_kind::TypeCast, "typecast_data"};
    impl::op_t tcweight_op {4, impl::op_kind::TypeCast, "typecast_weight"};
    impl::op_t tcother_op {5, impl::op_kind::TypeCast, "typecast_other"};

    impl::op_t matmul_op(6, impl::op_kind::MatMul, "matmul_op");
    matmul_op.set_attr<bool>("transpose_a", false);
    matmul_op.set_attr<bool>("transpose_b", true);

    impl::op_t add_op(7, impl::op_kind::Add, "add_op");

    // prepare logical tensor
    impl::logical_tensor_t src_u8
            = utils::logical_tensor_init(0, src_shape, impl::data_type::u8);
    impl::logical_tensor_t src_f32_dq
            = utils::logical_tensor_init(1, src_shape, impl::data_type::f32);
    impl::logical_tensor_t src_bf16
            = utils::logical_tensor_init(2, src_shape, impl::data_type::bf16);
    impl::logical_tensor_t weight_s8
            = utils::logical_tensor_init(3, weight_shape, impl::data_type::s8);
    impl::logical_tensor_t weight_f32_dq
            = utils::logical_tensor_init(4, weight_shape, impl::data_type::f32);
    impl::logical_tensor_t weight_bf16 = utils::logical_tensor_init(
            5, weight_shape, impl::data_type::bf16);
    impl::logical_tensor_t bias_f32
            = utils::logical_tensor_init(6, bias_shape, impl::data_type::f32);
    impl::logical_tensor_t other_u8
            = utils::logical_tensor_init(7, dst_shape, impl::data_type::u8);
    impl::logical_tensor_t other_f32_dq
            = utils::logical_tensor_init(8, dst_shape, impl::data_type::f32);
    impl::logical_tensor_t other_bf16
            = utils::logical_tensor_init(9, dst_shape, impl::data_type::bf16);
    impl::logical_tensor_t matmul_bf16
            = utils::logical_tensor_init(10, dst_shape, impl::data_type::bf16);
    impl::logical_tensor_t dst_bf16
            = utils::logical_tensor_init(11, dst_shape, impl::data_type::bf16);

    dqdata_op.add_input(src_u8);
    dqdata_op.add_output(src_f32_dq);

    dqweight_op.add_input(weight_s8);
    dqweight_op.add_output(weight_f32_dq);

    dqother_op.add_input(other_u8);
    dqother_op.add_output(other_f32_dq);

    tcdata_op.add_input(src_f32_dq);
    tcdata_op.add_output(src_bf16);

    tcweight_op.add_input(weight_f32_dq);
    tcweight_op.add_output(weight_bf16);

    tcother_op.add_input(other_f32_dq);
    tcother_op.add_output(other_bf16);

    matmul_op.add_input(src_bf16);
    matmul_op.add_input(weight_bf16);
    matmul_op.add_input(bias_f32);
    matmul_op.add_output(matmul_bf16);

    add_op.add_input(other_bf16);
    add_op.add_input(matmul_bf16);
    add_op.add_output(dst_bf16);

    impl::graph_t g;
    g.add_op(&dqdata_op);
    g.add_op(&dqweight_op);
    g.add_op(&dqother_op);
    g.add_op(&matmul_op);
    g.add_op(&tcdata_op);
    g.add_op(&tcweight_op);
    g.add_op(&tcother_op);
    g.add_op(&add_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass
            = get_pass("x8s8bf16_matmul_bias_add_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> lt_ins {
            &src_u8, &weight_s8, &bias_f32, &other_u8};
    std::vector<const impl::logical_tensor_t *> lt_outs {&dst_bf16};

    p.compile(&cp, lt_ins, lt_outs, &engine);

    test::vector<float> dst_data(product(dst_shape));
    impl::tensor_t src_u8_ts(src_u8, &engine, src_data.data());
    impl::tensor_t weight_s8_ts(weight_s8, &engine, weight_data.data());
    impl::tensor_t bias_f32_ts(bias_f32, &engine, bias_data.data());
    impl::tensor_t other_u8_ts(other_u8, &engine, other_data.data());
    impl::tensor_t dst_ts(dst_bf16, &engine, dst_data.data());
    cp.execute(&strm, {src_u8_ts, weight_s8_ts, bias_f32_ts, other_u8_ts},
            {dst_ts});
    strm.wait();
}

TEST(int8_subgraph_mode, u8s8u8_mix_bf16_matmul_bias_gelu) {
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    if (engine.kind() == impl::engine_kind::gpu) return;
    std::string qtype = "per_channel";
    std::vector<int64_t> src_shape = {1, 8, 16};
    std::vector<int64_t> weight_shape = {8, 16};
    std::vector<int64_t> bias_shape = {8};
    std::vector<int64_t> dst_shape = {1, 8, 8};

    test::vector<uint8_t> src_data(product(src_shape));
    test::vector<int8_t> weight_data(product(weight_shape));
    test::vector<float> bias_data(product(bias_shape));

    // random generate src, weight data
    // random seed = 7
    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> distribution(0.0f, 255.0f);
    std::generate(src_data.begin(), src_data.end(),
            [&]() { return distribution(generator); });
    std::uniform_real_distribution<float> distribution2(-127.0f, 127.0f);
    std::generate(weight_data.begin(), weight_data.end(),
            [&]() { return distribution2(generator); });
    std::uniform_real_distribution<float> distribution3(0.0f, 20.0f);
    std::generate(bias_data.begin(), bias_data.end(),
            [&]() { return distribution3(generator); });
    float scale_src = 1 / 255.f; // map to 0~255
    int64_t zp_src = 110;

    size_t scales_wei_sizes = qtype == "per_tensor" ? 1 : dst_shape.back();
    std::vector<float> scale_wei(scales_wei_sizes, 1 / 127.f);
    std::vector<int64_t> zp_wei(scales_wei_sizes, 0);

    impl::op_t dqdata_op(0, impl::op_kind::Dequantize, "dqdata_op");
    dqdata_op.set_attr<std::string>("qtype", "per_tensor");
    dqdata_op.set_attr<std::vector<int64_t>>("zps", {zp_src});
    dqdata_op.set_attr<std::vector<float>>("scales", {scale_src});
    dqdata_op.set_attr<int64_t>("axis", 0);

    impl::op_t dqweight_op(1, impl::op_kind::Dequantize, "dqweight_op");
    dqweight_op.set_attr<std::string>("qtype", qtype);
    dqweight_op.set_attr<std::vector<int64_t>>("zps", zp_wei);
    dqweight_op.set_attr<std::vector<float>>("scales", scale_wei);
    dqweight_op.set_attr<int64_t>("axis", 1);

    impl::op_t tcdata_op {2, impl::op_kind::TypeCast, "typecast_data"};
    impl::op_t tcweight_op {3, impl::op_kind::TypeCast, "typecast_weight"};

    impl::op_t matmul_op(4, impl::op_kind::MatMul, "matmul_op");
    matmul_op.set_attr<bool>("transpose_a", false);
    matmul_op.set_attr<bool>("transpose_b", true);

    impl::op_t gelu_op {5, impl::op_kind::GELU, "gelu_op"};
    impl::op_t tcgelu_op {6, impl::op_kind::TypeCast, "typecast_gelu"};

    impl::op_t qout_op(7, impl::op_kind::Quantize, "qdout_op");
    qout_op.set_attr<std::string>("qtype", qtype);
    qout_op.set_attr<std::vector<int64_t>>("zps", {zp_src});
    qout_op.set_attr<std::vector<float>>("scales", {scale_src});
    qout_op.set_attr<int64_t>("axis", 1);

    // prepare logical tensor
    impl::logical_tensor_t src_u8
            = utils::logical_tensor_init(0, src_shape, impl::data_type::u8);
    impl::logical_tensor_t src_f32_dq
            = utils::logical_tensor_init(1, src_shape, impl::data_type::f32);
    impl::logical_tensor_t src_bf16
            = utils::logical_tensor_init(2, src_shape, impl::data_type::bf16);
    impl::logical_tensor_t weight_s8
            = utils::logical_tensor_init(3, weight_shape, impl::data_type::s8);
    impl::logical_tensor_t weight_f32_dq
            = utils::logical_tensor_init(4, weight_shape, impl::data_type::f32);
    impl::logical_tensor_t weight_bf16 = utils::logical_tensor_init(
            5, weight_shape, impl::data_type::bf16);
    impl::logical_tensor_t bias_f32
            = utils::logical_tensor_init(6, bias_shape, impl::data_type::f32);
    impl::logical_tensor_t matmul_bf16
            = utils::logical_tensor_init(7, dst_shape, impl::data_type::bf16);
    impl::logical_tensor_t gelu_bf16
            = utils::logical_tensor_init(8, dst_shape, impl::data_type::bf16);
    impl::logical_tensor_t gelu_f32
            = utils::logical_tensor_init(9, dst_shape, impl::data_type::f32);
    impl::logical_tensor_t dst_u8
            = utils::logical_tensor_init(10, dst_shape, impl::data_type::u8);

    dqdata_op.add_input(src_u8);
    dqdata_op.add_output(src_f32_dq);

    dqweight_op.add_input(weight_s8);
    dqweight_op.add_output(weight_f32_dq);

    tcdata_op.add_input(src_f32_dq);
    tcdata_op.add_output(src_bf16);

    tcweight_op.add_input(weight_f32_dq);
    tcweight_op.add_output(weight_bf16);

    matmul_op.add_input(src_bf16);
    matmul_op.add_input(weight_bf16);
    matmul_op.add_input(bias_f32);
    matmul_op.add_output(matmul_bf16);

    gelu_op.add_input(matmul_bf16);
    gelu_op.add_output(gelu_bf16);

    tcgelu_op.add_input(gelu_bf16);
    tcgelu_op.add_output(gelu_f32);

    qout_op.add_input(gelu_f32);
    qout_op.add_output(dst_u8);

    impl::graph_t g;
    g.add_op(&dqdata_op);
    g.add_op(&dqweight_op);
    g.add_op(&matmul_op);
    g.add_op(&tcdata_op);
    g.add_op(&tcweight_op);
    g.add_op(&gelu_op);
    g.add_op(&tcgelu_op);
    g.add_op(&qout_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("int8_matmul_bias_gelu_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> lt_ins {
            &src_u8, &weight_s8, &bias_f32};
    std::vector<const impl::logical_tensor_t *> lt_outs {&dst_u8};

    p.compile(&cp, lt_ins, lt_outs, &engine);

    test::vector<uint8_t> dst_data(product(dst_shape));
    impl::tensor_t src_u8_ts(src_u8, &engine, src_data.data());
    impl::tensor_t weight_s8_ts(weight_s8, &engine, weight_data.data());
    impl::tensor_t bias_f32_ts(bias_f32, &engine, bias_data.data());
    impl::tensor_t dst_ts(dst_u8, &engine, dst_data.data());
    cp.execute(&strm, {src_u8_ts, weight_s8_ts, bias_f32_ts}, {dst_ts});
    strm.wait();
}
