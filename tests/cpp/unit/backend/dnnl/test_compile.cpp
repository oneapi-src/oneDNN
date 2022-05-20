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

#include <cmath>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <tuple>

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
#include "backend/dnnl/kernels/layernorm.hpp"
#include "backend/dnnl/kernels/matmul.hpp"
#include "backend/dnnl/kernels/pool.hpp"
#include "backend/dnnl/kernels/quantize.hpp"
#include "backend/dnnl/kernels/reorder.hpp"
#include "backend/dnnl/kernels/softmax.hpp"
#include "backend/dnnl/passes/utils.hpp"

#include "cpp/unit/unit_test_common.hpp"

// #include "dnnl.h"
#include "cpp/unit/utils.hpp"

namespace impl = dnnl::graph::impl;
namespace dnnl_impl = impl::dnnl_impl;
namespace utils = dnnl::graph::tests::unit::utils;

static dnnl_impl::kernel_registry_t &get_dnnl_kernel_registry() {
    return dnnl_impl::dnnl_backend::get_singleton().get_kernel_registry();
}

namespace {
dnnl::graph::impl::pass::pass_base_ptr get_pass(const std::string &pass_name) {
    auto &backend_ptr
            = dnnl::graph::impl::dnnl_impl::dnnl_backend::get_singleton();
    auto pm = dnnl::graph::impl::pass::pass_manager_t(
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
    using ltw = impl::logical_tensor_wrapper_t;
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
} // namespace

TEST(Compile, ConvolutionFp32) {
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

    impl::graph_t g(engine.kind());
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
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);
}

TEST(Compile, ConvolutionBackpropDataFp32) {
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
    conv_op.set_attr<dims>("output_shape", dims {8, 3, 224, 224});

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

    impl::graph_t g(eng.kind());
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

TEST(Compile, ConvolutionBackpropFilterFp32) {
    using dims = dnnl::graph::impl::dnnl_impl::dims;

    impl::engine_t &eng = get_engine();

    impl::op_t conv_op(impl::op_kind::ConvolutionBackpropFilters);
    conv_op.set_attr<dims>("strides", dims {1, 1});
    conv_op.set_attr<dims>("dilations", dims {1, 1});
    conv_op.set_attr<dims>("pads_begin", dims {0, 0});
    conv_op.set_attr<dims>("pads_end", dims {0, 0});
    conv_op.set_attr<std::string>("data_format", "NXC");
    conv_op.set_attr<std::string>("filter_format", "XIO");
    conv_op.set_attr<dims>("filter_shape", dims {3, 3, 64, 64});

    // prepare logical tensor
    impl::logical_tensor_t src = utils::logical_tensor_init(
            0, {1, 224, 224, 64}, impl::data_type::f32);
    impl::logical_tensor_t diff_dst = utils::logical_tensor_init(
            1, {1, 222, 222, 64}, impl::data_type::f32);
    impl::logical_tensor_t diff_weight = utils::logical_tensor_init(
            2, {3, 3, 64, 64}, impl::data_type::f32, impl::layout_type::any);

    conv_op.add_input(src);
    conv_op.add_input(diff_dst);
    conv_op.add_output(diff_weight);

    impl::graph_t g(eng.kind());
    g.add_op(&conv_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("conv_filter_bw_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);
    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src, &diff_dst};
    std::vector<const impl::logical_tensor_t *> outputs {&diff_weight};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);
}

TEST(Compile, ConvolutionBackpropFiltersWithGroupsAndFiltersAnyLayout) {
    using dims = impl::dnnl_impl::dims;

    const dims src_dims {2, 4, 2};
    const dims diff_dst_dims {2, 4, 2};
    const dims diff_wei_dims {4, 2, 3};

    const dims strides {1};
    const dims pads_begin {1};
    const dims pads_end {1};
    const dims dilations {1};
    const int64_t groups {2};
    const std::string auto_pad {"None"};
    const std::string data_format {"NCX"};
    const std::string filter_format {"OIX"};

    impl::op_t conv_op(impl::op_kind::ConvolutionBackpropFilters);
    conv_op.set_attr("strides", strides)
            .set_attr("pads_begin", pads_begin)
            .set_attr("pads_end", pads_end)
            .set_attr("dilations", dilations)
            .set_attr("groups", groups)
            .set_attr("auto_pad", auto_pad)
            .set_attr("data_format", data_format)
            .set_attr("filter_format", filter_format);

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, src_dims, impl::data_type::f32);
    impl::logical_tensor_t diff_dst_lt = utils::logical_tensor_init(
            1, diff_dst_dims, impl::data_type::f32);
    impl::logical_tensor_t diff_wei_lt = utils::logical_tensor_init(
            2, diff_wei_dims, impl::data_type::f32, impl::layout_type::any);

    conv_op.add_input(src_lt);
    conv_op.add_input(diff_dst_lt);
    conv_op.add_output(diff_wei_lt);

    impl::engine_t &eng = get_engine();
    impl::graph_t g(eng.kind());
    g.add_op(&conv_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("conv_filter_bw_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src_lt, &diff_dst_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&diff_wei_lt};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

    impl::logical_tensor_t lt;
    cp.query_logical_tensor(diff_wei_lt.id, &lt);
    // if layout queried from the primitive will make descriptor impossible
    // to reshape (with groups -> no groups), we make it strided (via reorder)
    ASSERT_TRUE(lt.layout_type == impl::layout_type::opaque
            || lt.layout_type == impl::layout_type::strided);
}

TEST(Compile, ConvtransposeFp32) {
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

    impl::graph_t g(eng.kind());
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
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);
}

void ref_batchnorm_fwd(impl::dim_t mb, impl::dim_t ic, impl::dim_t ih,
        impl::dim_t iw, impl::tensor_t *src, impl::tensor_t *dst,
        impl::tensor_t *scale, impl::tensor_t *shift,
        impl::tensor_t *mean = nullptr, impl::tensor_t *variance = nullptr,
        impl::tensor_t *running_mean = nullptr,
        impl::tensor_t *running_variance = nullptr,
        impl::tensor_t *batch_mean = nullptr,
        impl::tensor_t *batch_variance = nullptr, float epsilon = 0.001f,
        float momentum = 0.1f,
        std::function<float(const float)> activation = nullptr,
        bool channel_last = false, bool is_training = false) {
    if (!activation) activation = [](const float v) { return v; };

    size_t mb_ = static_cast<size_t>(mb);
    size_t ic_ = static_cast<size_t>(ic);
    size_t ih_ = static_cast<size_t>(ih);
    size_t iw_ = static_cast<size_t>(iw);

    float *src_ptr = src->get_data_handle<float>();
    float *dst_ptr = dst->get_data_handle<float>();

    if (is_training) {
        float *mean_ptr = batch_mean->get_data_handle<float>();
        float *variance_ptr = batch_variance->get_data_handle<float>();
        float *scale_ptr = scale->get_data_handle<float>();
        float *shift_ptr = shift->get_data_handle<float>();
        float *est_mean_ptr = mean->get_data_handle<float>();
        float *est_variance_ptr = variance->get_data_handle<float>();
        float *running_mean_ptr = running_mean->get_data_handle<float>();
        float *running_variance_ptr
                = running_variance->get_data_handle<float>();
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
        for (int k = 0; k < ic_; k++) {
            running_mean_ptr[k]
                    = momentum * est_mean_ptr[k] + (1 - momentum) * mean_ptr[k];
            running_variance_ptr[k] = momentum * est_variance_ptr[k]
                    + (1 - momentum) * variance_ptr[k];
        }
    } else {
        float *mean_ptr = nullptr;
        float *variance_ptr = nullptr;
        float *scale_ptr = scale->get_data_handle<float>();
        float *shift_ptr = shift->get_data_handle<float>();
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
}

struct dnnl_graph_test_batchnorm_params {
    impl::op_kind_t op_kind;
    bool with_relu;
    impl::dim_t N;
    impl::dim_t IC;
    impl::dim_t IH;
    impl::dim_t IW;
    float epsilon;
    std::string data_format;
    std::string backend_name;
    impl::data_type_t data_type;
};

class BatchNorm4D
    : public ::testing::TestWithParam<dnnl_graph_test_batchnorm_params> {
public:
    void TestBatchnorm(bool dims_in_order = true) {
        using dims = dnnl::graph::impl::dnnl_impl::dims;

        auto params = ::testing::TestWithParam<
                dnnl_graph_test_batchnorm_params>::GetParam();

        impl::op_t batchnorm_op(0, params.op_kind, "batchnorm");
        batchnorm_op.set_attr("epsilon", params.epsilon);
        batchnorm_op.set_attr("data_format", params.data_format);
        impl::op_t relu_op(1, impl::op_kind::ReLU, "relu");

        bool is_training
                = params.op_kind == impl::op_kind::BatchNormForwardTraining;
        const float momentum = 0.1f;
        if (is_training) batchnorm_op.set_attr("momentum", momentum);

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
        // if ((params.data_format == "NCX") ^ dims_in_order)
        if (params.data_format == "NXC")
            src_dims = {N, IH, IW, IC};
        else
            src_dims = {N, IC, IH, IW};
        // Scale/shift tensor dimensions.
        dims scale_dims = {IC};
        dims shift_dims = {IC};
        dims mean_dims = {IC};
        dims variance_dims = {IC};

        // prepare logical tensor
        impl::logical_tensor_t src
                = utils::logical_tensor_init(0, src_dims, impl::data_type::f32);
        impl::logical_tensor_t scale = utils::logical_tensor_init(
                1, scale_dims, impl::data_type::f32);
        impl::logical_tensor_t shift = utils::logical_tensor_init(
                2, shift_dims, impl::data_type::f32);
        impl::logical_tensor_t mean = utils::logical_tensor_init(
                3, mean_dims, impl::data_type::f32);
        impl::logical_tensor_t variance = utils::logical_tensor_init(
                4, variance_dims, impl::data_type::f32);
        impl::logical_tensor_t running_mean = utils::logical_tensor_init(
                5, mean_dims, impl::data_type::f32);
        impl::logical_tensor_t running_variance = utils::logical_tensor_init(
                6, variance_dims, impl::data_type::f32);
        impl::logical_tensor_t batch_mean = utils::logical_tensor_init(
                7, mean_dims, impl::data_type::f32);
        impl::logical_tensor_t batch_variance = utils::logical_tensor_init(
                8, variance_dims, impl::data_type::f32);
        impl::logical_tensor_t dst
                = utils::logical_tensor_init(9, src_dims, impl::data_type::f32);
        impl::logical_tensor_t relu_dst = utils::logical_tensor_init(
                10, src_dims, impl::data_type::f32);

        batchnorm_op.add_input(src);
        batchnorm_op.add_input(scale);
        batchnorm_op.add_input(shift);
        batchnorm_op.add_input(mean);
        batchnorm_op.add_input(variance);
        batchnorm_op.add_output(dst);
        if (is_training) {
            batchnorm_op.add_output(running_mean);
            batchnorm_op.add_output(running_variance);
            batchnorm_op.add_output(batch_mean);
            batchnorm_op.add_output(batch_variance);
        }

        relu_op.add_input(dst);
        relu_op.add_output(relu_dst);

        impl::graph_t g(engine.kind());
        g.add_op(&batchnorm_op);
        if (params.with_relu) g.add_op(&relu_op);
        g.build_graph();

        impl::pass::pass_base_ptr apass = params.with_relu
                ? get_pass("bn_relu_fusion")
                : (is_training ? get_pass("bn_fw_train_pass")
                               : get_pass("bn_pass"));
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1);
        auto part = g.get_partitions()[0];

        // compile
        impl::partition_t p;
        p.init(part);
        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> inputs {
                &src, &scale, &shift, &mean, &variance};
        std::vector<const impl::logical_tensor_t *> outputs {&dst};
        if (params.with_relu) outputs[0] = &relu_dst;
        if (is_training) {
            outputs.emplace_back(&running_mean);
            outputs.emplace_back(&running_variance);
            outputs.emplace_back(&batch_mean);
            outputs.emplace_back(&batch_variance);
        }

        ASSERT_EQ(p.compile(&cp, inputs, outputs, &engine),
                impl::status::success);

        // Allocate buffers.
        test::vector<float> src_data(static_cast<size_t>(N * IC * IH * IW));
        test::vector<float> scale_data(static_cast<size_t>(IC));
        test::vector<float> shift_data(static_cast<size_t>(IC));
        test::vector<float> mean_data(static_cast<size_t>(IC));
        test::vector<float> variance_data(static_cast<size_t>(IC));
        test::vector<float> running_mean_data(static_cast<size_t>(IC));
        test::vector<float> running_variance_data(static_cast<size_t>(IC));
        test::vector<float> batch_mean_data(static_cast<size_t>(IC));
        test::vector<float> batch_variance_data(static_cast<size_t>(IC));
        test::vector<float> dst_data(src_data.size(), 0.0);
        test::vector<float> ref_dst_data(src_data.size(), 0.0);
        test::vector<float> ref_running_mean_data(
                running_mean_data.size(), 0.0);
        test::vector<float> ref_running_variance_data(
                running_variance_data.size(), 0.0);
        test::vector<float> ref_batch_mean_data(batch_mean_data.size(), 0.0);
        test::vector<float> ref_batch_variance_data(
                batch_variance_data.size(), 0.0);
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
        if (!is_training) {
            // Initialize mean.
            std::generate(mean_data.begin(), mean_data.end(), []() {
                static int i = 0;
                return std::sin(static_cast<float>(i++) * 2.f);
            });
            // Initialize variance.
            std::generate(variance_data.begin(), variance_data.end(), []() {
                static int i = 0;
                return static_cast<float>(i++);
            });
        } else {
            // Initialize mean.
            std::generate(
                    running_mean_data.begin(), running_mean_data.end(), []() {
                        static int i = 0;
                        return std::sin(static_cast<float>(i++) * 2.f);
                    });
            // Initialize variance.
            std::generate(running_variance_data.begin(),
                    running_variance_data.end(), []() {
                        static int i = 0;
                        return static_cast<float>(i++);
                    });
        }

        impl::tensor_t src_ts(src, &engine, src_data.data());
        impl::tensor_t scale_ts(scale, &engine, scale_data.data());
        impl::tensor_t shift_ts(shift, &engine, shift_data.data());
        impl::tensor_t mean_ts(mean, &engine, mean_data.data());
        impl::tensor_t variance_ts(variance, &engine, variance_data.data());
        impl::tensor_t running_mean_ts(
                running_mean, &engine, running_mean_data.data());
        impl::tensor_t running_variance_ts(
                running_variance, &engine, running_variance_data.data());
        impl::tensor_t batch_mean_ts(
                batch_mean, &engine, batch_mean_data.data());
        impl::tensor_t batch_variance_ts(
                batch_variance, &engine, batch_variance_data.data());
        impl::tensor_t dst_ts = params.with_relu
                ? impl::tensor_t(relu_dst, &engine, dst_data.data())
                : impl::tensor_t(dst, &engine, dst_data.data());
        impl::tensor_t ref_dst_ts = params.with_relu
                ? impl::tensor_t(relu_dst, &engine, ref_dst_data.data())
                : impl::tensor_t(dst, &engine, ref_dst_data.data());
        impl::tensor_t ref_running_mean_ts(
                running_mean, &engine, ref_running_mean_data.data());
        impl::tensor_t ref_running_variance_ts(
                running_variance, &engine, ref_running_variance_data.data());
        impl::tensor_t ref_batch_mean_ts(
                batch_mean, &engine, ref_batch_mean_data.data());
        impl::tensor_t ref_batch_variance_ts(
                batch_variance, &engine, ref_batch_variance_data.data());

        impl::stream_t &strm = get_stream();

        if (!is_training) {
            cp.execute(&strm,
                    {src_ts, scale_ts, shift_ts, mean_ts, variance_ts},
                    {dst_ts});
        } else {
            cp.execute(&strm,
                    {src_ts, scale_ts, shift_ts, mean_ts, variance_ts},
                    {dst_ts, running_mean_ts, running_variance_ts,
                            batch_mean_ts, batch_variance_ts});
        }

        strm.wait();

        std::function<float(const float)> activation = nullptr;
        if (params.with_relu) {
            activation = [](const float v) { return v < 0.f ? 0.f : v; };
        }
        if (!is_training) {
            ref_batchnorm_fwd(N, IC, IH, IW, &src_ts, &ref_dst_ts, &scale_ts,
                    &shift_ts, &mean_ts, &variance_ts, nullptr, nullptr,
                    nullptr, nullptr, params.epsilon, momentum, activation,
                    params.data_format == "NXC",
                    params.op_kind == impl::op_kind::BatchNormForwardTraining);
        } else {
            ref_batchnorm_fwd(N, IC, IH, IW, &src_ts, &ref_dst_ts, &scale_ts,
                    &shift_ts, &mean_ts, &variance_ts, &ref_running_mean_ts,
                    &ref_running_variance_ts, &ref_batch_mean_ts,
                    &ref_batch_variance_ts, params.epsilon, momentum,
                    activation, params.data_format == "NXC",
                    params.op_kind == impl::op_kind::BatchNormForwardTraining);
            for (size_t i = 0; i < IC; ++i) {
                ASSERT_NEAR(
                        running_mean_data[i], ref_running_mean_data[i], 1.e-6f);
                ASSERT_NEAR(running_variance_data[i],
                        ref_running_variance_data[i], 1.e-6f);
                ASSERT_NEAR(batch_mean_data[i], ref_batch_mean_data[i], 1.e-6f);
                ASSERT_NEAR(batch_variance_data[i], ref_batch_variance_data[i],
                        1.e-6f);
            }
        }
        for (size_t i = 0; i < dst_data.size(); ++i) {
            ASSERT_NEAR(dst_data[i], ref_dst_data[i], 1.e-6f);
        }
    }

    void Test() {
        TestBatchnorm(true);
        TestBatchnorm(false);
    }
};

TEST_P(BatchNorm4D, TestBatchnorm) {
    Test();
}

INSTANTIATE_TEST_SUITE_P(Execute, BatchNorm4D,
        ::testing::Values(
                dnnl_graph_test_batchnorm_params {
                        impl::op_kind::BatchNormInference, false, 3, 3, 2, 2,
                        0.001f, "NCX", "dnnl", impl::data_type::f32},
                dnnl_graph_test_batchnorm_params {
                        impl::op_kind::BatchNormInference, true, 3, 3, 2, 2,
                        0.001f, "NCX", "dnnl", impl::data_type::f32},
                dnnl_graph_test_batchnorm_params {
                        impl::op_kind::BatchNormInference, false, 3, 3, 2, 2,
                        0.001f, "NXC", "dnnl", impl::data_type::f32},
                dnnl_graph_test_batchnorm_params {
                        impl::op_kind::BatchNormForwardTraining, false, 3, 3, 2,
                        2, 0.001f, "NCX", "dnnl", impl::data_type::f32},
                dnnl_graph_test_batchnorm_params {
                        impl::op_kind::BatchNormForwardTraining, false, 3, 3, 2,
                        2, 0.001f, "NXC", "dnnl", impl::data_type::f32}));

TEST(Compile, BatchNormBackpropFp32) {
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
    test::vector<float> src_data {1.0f, 2.0f, 3.0f, 4.0f};
    test::vector<float> scale_data {2.0f};
    test::vector<float> mean_data {2.5f};
    test::vector<float> varience_data {1.25f};
    test::vector<float> diff_dst_data {0.1f, 0.1f, 0.2f, 0.2f};
    test::vector<float> diff_src_data(src_data.size(), 0.0);
    test::vector<float> diff_scale_data(scale_data.size(), 0.0);
    test::vector<float> diff_shift_data(scale_data.size(), 0.0);

    // expected results
    test::vector<float> ref_diff_src_data {
            0.01788854f, -0.05366563f, 0.05366563f, -0.01788854f};
    test::vector<float> ref_diff_scale_data {0.17888544f};
    test::vector<float> ref_diff_shift_data {0.6f};

    auto &op_factory = get_dnnl_kernel_registry();
    auto bn_kernel = op_factory.create_kernel(bn_op);
    ASSERT_TRUE(bn_kernel);

    bn_op.add_input(src);
    bn_op.add_input(diff_dst);
    bn_op.add_input(scale);
    bn_op.add_input(mean);
    bn_op.add_input(varience);
    bn_op.add_output(diff_src);
    bn_op.add_output(diff_scale);
    bn_op.add_output(diff_shift);

    impl::graph_t g(engine.kind());
    g.add_op(&bn_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("bn_bw_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);
    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {
            &src, &diff_dst, &scale, &mean, &varience};
    std::vector<const impl::logical_tensor_t *> outputs {
            &diff_src, &diff_scale, &diff_shift};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, &engine), impl::status::success);

    impl::tensor_t src_ts(src, &engine, src_data.data());
    impl::tensor_t scale_ts(scale, &engine, scale_data.data());
    impl::tensor_t mean_ts(mean, &engine, mean_data.data());
    impl::tensor_t variance_ts(varience, &engine, varience_data.data());
    impl::tensor_t diff_dst_ts(diff_dst, &engine, diff_dst_data.data());
    impl::tensor_t diff_src_ts(diff_src, &engine, diff_src_data.data());
    impl::tensor_t diff_scale_ts(diff_scale, &engine, diff_scale_data.data());
    impl::tensor_t diff_shift_ts(diff_shift, &engine, diff_shift_data.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src_ts, diff_dst_ts, scale_ts, mean_ts, variance_ts},
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

class Concat : public ::testing::TestWithParam<dnnl_graph_test_concat_params> {
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

        impl::op_t concat_op(impl::op_kind::Concat, "concat");
        concat_op.set_attr<int64_t>("axis", axis);

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

        concat_op.add_input(src0_lt);
        concat_op.add_input(src1_lt);
        concat_op.add_output(dst_lt);

        auto &eng = get_engine();
        impl::graph_t g(eng.kind());
        g.add_op(&concat_op);
        g.build_graph();

        impl::pass::pass_base_ptr apass = get_pass("concat_pass");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1);
        auto part = g.get_partitions()[0];

        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> inputs {&src0_lt, &src1_lt};
        std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};

        p.compile(&cp, inputs, outputs, &eng);

        impl::tensor_t src0_ts(src0_lt, &eng, src0_data.data());
        impl::tensor_t src1_ts(src1_lt, &eng, src1_data.data());
        impl::tensor_t dst_ts(dst_lt, &eng, dst_data.data());

        auto &strm = get_stream();
        cp.execute(&strm, {src0_ts, src1_ts}, {dst_ts});
        strm.wait();

        for (size_t i = 0; i < dst_data.size(); ++i) {
            ASSERT_FLOAT_EQ(params.ref_dst[i], dst_data[i]);
        }
    }
};

TEST_P(Concat, TestConcat) {
    TestConcat();
}

INSTANTIATE_TEST_SUITE_P(Execute, Concat,
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

TEST(Compile, ConcatWithMoreInputs) {
    size_t num_inputs = 64;
    const std::vector<impl::dim_t> src_dims = {1, 2, 2, 2};
    const std::vector<impl::dim_t> dst_dims
            = {1, 2, 2, 2 * static_cast<impl::dim_t>(num_inputs)};
    test::vector<float> dst_data(dst_dims.size(), 0.);
    const auto axis = 3;

    impl::op_t concat_op(impl::op_kind::Concat, "concat");
    concat_op.set_attr<int64_t>("axis", axis);

    std::vector<impl::logical_tensor_t> input_lts;
    for (size_t i = 0; i < num_inputs; ++i) {
        impl::logical_tensor_t src_lt = utils::logical_tensor_init(
                i, src_dims, impl::data_type::f32, impl::layout_type::strided);
        input_lts.push_back(src_lt);
        concat_op.add_input(input_lts[i]);
    }
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(num_inputs,
            dst_dims, impl::data_type::f32, impl::layout_type::strided);
    concat_op.add_output(dst_lt);

    auto &eng = get_engine();
    impl::graph_t g(eng.kind());
    g.add_op(&concat_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("concat_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs;
    for (size_t i = 0; i < num_inputs; ++i) {
        inputs.push_back(&input_lts[i]);
    }
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};
    auto ret = p.compile(&cp, inputs, outputs, &eng);
    ASSERT_EQ(ret, impl::status::success);
}

TEST(Compile, ConcatWithMoreInputsFail) {
    size_t num_inputs = 65;
    const std::vector<impl::dim_t> src_dims = {1, 2, 2, 2};
    const std::vector<impl::dim_t> dst_dims
            = {1, 2, 2, 2 * static_cast<impl::dim_t>(num_inputs)};
    test::vector<float> dst_data(dst_dims.size(), 0.);
    const auto axis = 3;

    impl::op_t concat_op(impl::op_kind::Concat, "concat");
    concat_op.set_attr<int64_t>("axis", axis);

    std::vector<impl::logical_tensor_t> input_lts;
    for (size_t i = 0; i < num_inputs; ++i) {
        impl::logical_tensor_t src_lt = utils::logical_tensor_init(
                i, src_dims, impl::data_type::f32, impl::layout_type::strided);
        input_lts.push_back(src_lt);
        concat_op.add_input(input_lts[i]);
    }
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(num_inputs,
            dst_dims, impl::data_type::f32, impl::layout_type::strided);
    concat_op.add_output(dst_lt);

    auto &eng = get_engine();
    impl::graph_t g(eng.kind());
    ASSERT_EQ(g.add_op(&concat_op), impl::status::invalid_op);
}

TEST(ExecuteSubgraphInt8, Concat) {
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    const int64_t channels = 2;
    const int64_t n_inputs = 3;
    std::vector<int64_t> in_shape {2, channels, 4, 4};
    std::vector<int64_t> out_shape {2, channels * n_inputs, 4, 4};

    std::vector<float> scales = {5.f / 127.f};
    std::vector<int64_t> zps = {0};

    test::vector<int8_t> src0_s8_data(utils::product(in_shape));
    test::vector<int8_t> src1_s8_data(utils::product(in_shape));
    test::vector<int8_t> src2_s8_data(utils::product(in_shape));
    test::vector<int8_t> case1_dst_s8_data(utils::product(out_shape));
    test::vector<int8_t> case2_dst_s8_data(utils::product(out_shape));

    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> s8_distribution(-127.0f, 128.0f);
    std::generate(src0_s8_data.begin(), src0_s8_data.end(),
            [&]() { return static_cast<uint8_t>(s8_distribution(generator)); });
    std::generate(src1_s8_data.begin(), src1_s8_data.end(),
            [&]() { return static_cast<uint8_t>(s8_distribution(generator)); });
    std::generate(src2_s8_data.begin(), src2_s8_data.end(),
            [&]() { return static_cast<uint8_t>(s8_distribution(generator)); });

    impl::op_t dq0_op(0, impl::op_kind::Dequantize, "dq0_op");
    impl::op_t dq1_op(1, impl::op_kind::Dequantize, "dq1_op");
    impl::op_t dq2_op(2, impl::op_kind::Dequantize, "dq2_op");
    for (auto dq_op : {&dq0_op, &dq1_op, &dq2_op}) {
        dq_op->set_attr<std::string>("qtype", "per_tensor");
        dq_op->set_attr<std::vector<int64_t>>("zps", zps);
        dq_op->set_attr<std::vector<float>>("scales", scales);
    }

    impl::op_t concat_op(3, impl::op_kind::Concat, "concat_op");
    concat_op.set_attr<int64_t>("axis", 1);

    impl::op_t q_op(4, impl::op_kind::Quantize, "q_op");
    q_op.set_attr<std::string>("qtype", "per_tensor");
    q_op.set_attr<std::vector<int64_t>>("zps", zps);
    q_op.set_attr<std::vector<float>>("scales", scales);

    auto src0_s8 = utils::logical_tensor_init(0, in_shape, impl::data_type::s8);
    auto src0_f32_dq
            = utils::logical_tensor_init(1, in_shape, impl::data_type::f32);

    auto src1_s8 = utils::logical_tensor_init(2, in_shape, impl::data_type::s8);
    auto src1_f32_dq
            = utils::logical_tensor_init(3, in_shape, impl::data_type::f32);

    auto src2_s8 = utils::logical_tensor_init(4, in_shape, impl::data_type::s8);
    auto src2_f32_dq
            = utils::logical_tensor_init(5, in_shape, impl::data_type::f32);

    auto dst_f32
            = utils::logical_tensor_init(6, out_shape, impl::data_type::f32);
    auto dst_s8_q
            = utils::logical_tensor_init(7, out_shape, impl::data_type::s8);

    dq0_op.add_input(src0_s8);
    dq0_op.add_output(src0_f32_dq);

    dq1_op.add_input(src1_s8);
    dq1_op.add_output(src1_f32_dq);

    dq2_op.add_input(src2_s8);
    dq2_op.add_output(src2_f32_dq);

    concat_op.add_input(src0_f32_dq);
    concat_op.add_input(src1_f32_dq);
    concat_op.add_input(src2_f32_dq);
    concat_op.add_output(dst_f32);

    q_op.add_input(dst_f32);
    q_op.add_output(dst_s8_q);

    impl::graph_t g(engine.kind());
    g.add_op(&dq0_op);
    g.add_op(&dq1_op);
    g.add_op(&dq2_op);
    g.add_op(&concat_op);
    g.add_op(&q_op);
    g.build_graph();

    impl::tensor_t src0_s8_ts(src0_s8, &engine, src0_s8_data.data());
    impl::tensor_t src1_s8_ts(src1_s8, &engine, src1_s8_data.data());
    impl::tensor_t src2_s8_ts(src2_s8, &engine, src2_s8_data.data());
    impl::tensor_t case1_dst_s8_ts(dst_s8_q, &engine, case1_dst_s8_data.data());
    impl::tensor_t case2_dst_s8_ts(dst_s8_q, &engine, case2_dst_s8_data.data());

    // -------------------------case 1----------------------------------
    ASSERT_EQ(run_graph(g, {src0_s8_ts, src1_s8_ts, src2_s8_ts},
                      {case1_dst_s8_ts}, engine, strm),
            impl::status::success);

    // -------------------------case 2----------------------------------
    impl::pass::pass_base_ptr apass = get_pass("int8_concat_fusion");
    apass->run(g);

    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);
    std::vector<const impl::logical_tensor_t *> lt_ins {
            &src0_s8, &src1_s8, &src2_s8};
    std::vector<const impl::logical_tensor_t *> lt_outs {&dst_s8_q};
    p.compile(&cp, lt_ins, lt_outs, &engine);

    cp.execute(&strm, {src0_s8_ts, src1_s8_ts, src2_s8_ts}, {case2_dst_s8_ts});
    strm.wait();

    ASSERT_TRUE(allclose(case1_dst_s8_data, case2_dst_s8_data,
            /*rtol*/ 0.01f,
            /*atol*/ 1.f));
}

TEST(Execute, Add) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src0 {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    test::vector<float> src1 {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> ref_dst {3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0};
    test::vector<float> dst(src0.size(), 0.0);

    impl::op_t add_op(0, impl::op_kind::Add, "add");

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            2, {1, 3, 3}, impl::data_type::f32, impl::layout_type::any);

    add_op.add_input(src0_lt);
    add_op.add_input(src1_lt);
    add_op.add_output(dst_lt);

    impl::graph_t g(eng.kind());
    g.add_op(&add_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("sum_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src0_lt, &src1_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};

    p.compile(&cp, inputs, outputs, &eng);

    impl::logical_tensor_t compiled_dst_lt;
    cp.query_logical_tensor(dst_lt.id, &compiled_dst_lt);

    ASSERT_EQ(dst_lt.layout_type, impl::layout_type::any);
    ASSERT_EQ(compiled_dst_lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src0_ts(src0_lt, &eng, src0.data());
    impl::tensor_t src1_ts(src1_lt, &eng, src1.data());
    impl::tensor_t dst_ts(compiled_dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src0_ts, src1_ts}, {dst_ts});
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
    impl::tensor_t dst_2nd_ts(compiled_dst_lt, &eng, dst_2nd.data());
    cp.execute(&strm, {src0_2nd_ts, src1_2nd_ts}, {dst_2nd_ts});
    strm.wait();

    for (size_t i = 0; i < src0_2nd.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_2nd[i], ref_dst_2nd[i]);
    }
}

TEST(Execute, AddWithDifferentFormat) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src0 {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    test::vector<float> src1 {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    test::vector<float> ref_dst {2.0, 5.0, 8.0, 3.0, 6.0, 9.0, 4.0, 7.0, 10.0};
    test::vector<float> dst(src0.size(), 0.0);

    impl::op_t add_op(0, impl::op_kind::Add, "add");

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt = utils::logical_tensor_init(
            1, {1, 1, 3, 3}, {3, 3, 1, 3}, impl::data_type::f32); // abdc format
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {1, 1, 3, 3}, impl::data_type::f32);

    add_op.add_input(src0_lt);
    add_op.add_input(src1_lt);
    add_op.add_output(dst_lt);

    impl::graph_t g(eng.kind());
    g.add_op(&add_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("sum_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src0_lt, &src1_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};

    p.compile(&cp, inputs, outputs, &eng);

    impl::logical_tensor_t compiled_dst_lt;
    cp.query_logical_tensor(dst_lt.id, &compiled_dst_lt);

    impl::tensor_t src0_ts(src0_lt, &eng, src0.data());
    impl::tensor_t src1_ts(src1_lt, &eng, src1.data());
    impl::tensor_t dst_ts(compiled_dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src0_ts, src1_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, BroadcastAdd) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src0 {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    test::vector<float> src1 {
            2.0, 2.0, 2.0}; // bianary op's src1 support broadcast
    test::vector<float> ref_dst {3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0};
    test::vector<float> dst(src0.size(), 0.0);

    impl::op_t add_op(0, impl::op_kind::Add, "add");

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 2, 3, 3}, impl::data_type::f32);
    // src1 will first be unsequeeze to {1,1,1,3} and then broadcast
    // to {1,2,3,3}
    impl::logical_tensor_t src1_lt = utils::logical_tensor_init(
            1, std::vector<impl::dim_t> {3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {1, 2, 3, 3}, impl::data_type::f32);

    add_op.add_input(src0_lt);
    add_op.add_input(src1_lt);
    add_op.add_output(dst_lt);

    impl::graph_t g(eng.kind());
    g.add_op(&add_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("sum_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src0_lt, &src1_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};

    p.compile(&cp, inputs, outputs, &eng);

    impl::logical_tensor_t compiled_dst_lt;
    cp.query_logical_tensor(dst_lt.id, &compiled_dst_lt);

    impl::tensor_t src0_ts(src0_lt, &eng, src0.data());
    impl::tensor_t src1_ts(src1_lt, &eng, src1.data());
    impl::tensor_t dst_ts(compiled_dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src0_ts, src1_ts}, {dst_ts});
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
    impl::tensor_t dst_2nd_ts(compiled_dst_lt, &eng, dst_2nd.data());
    cp.execute(&strm, {src0_2nd_ts, src1_2nd_ts}, {dst_2nd_ts});
    strm.wait();

    for (size_t i = 0; i < src0_2nd.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_2nd[i], ref_dst_2nd[i]);
    }
}

TEST(Execute, SwapBroadcastAdd) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src0 {1.0, 1.0, 1.0, 1.0};
    test::vector<float> src1 {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> ref_dst {3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0};
    test::vector<float> dst(src1.size(), 0.0);

    impl::op_t add_op(0, impl::op_kind::Add, "add");

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 1, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1, 2, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {1, 2, 2, 2}, impl::data_type::f32);

    add_op.add_input(src0_lt);
    add_op.add_input(src1_lt);
    add_op.add_output(dst_lt);

    impl::graph_t g(eng.kind());
    g.add_op(&add_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("sum_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src0_lt, &src1_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};

    p.compile(&cp, inputs, outputs, &eng);

    impl::logical_tensor_t compiled_dst_lt;
    cp.query_logical_tensor(dst_lt.id, &compiled_dst_lt);

    impl::tensor_t src0_ts(src0_lt, &eng, src0.data());
    impl::tensor_t src1_ts(src1_lt, &eng, src1.data());
    impl::tensor_t dst_ts(compiled_dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src0_ts, src1_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src1.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }

    // inplace
    auto inplace_pair = cp.get_inplace_pairs();
    ASSERT_EQ(inplace_pair[0].input, src1_lt.id);
    ASSERT_EQ(inplace_pair[0].output, dst_lt.id);

    impl::tensor_t dst_ts2(compiled_dst_lt, &eng, src1.data());
    cp.execute(&strm, {src0_ts, src1_ts}, {dst_ts2});
    strm.wait();

    for (size_t i = 0; i < src1.size(); ++i) {
        ASSERT_FLOAT_EQ(src1[i], ref_dst[i]);
    }
}

TEST(Execute, MultidirectionalBroadcastAddBA) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src0 {1.0, 1.0, 1.0};
    test::vector<float> src1 {2.0, 2.0, 2.0};
    test::vector<float> ref_dst {3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0};
    test::vector<float> dst(ref_dst.size(), 0.0);

    impl::op_t add_op(0, impl::op_kind::Add, "add");

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {3}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt = utils::logical_tensor_init(
            1, std::vector<impl::dim_t> {3, 1}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {3, 3}, impl::data_type::f32);

    add_op.add_input(src0_lt);
    add_op.add_input(src1_lt);
    add_op.add_output(dst_lt);

    impl::graph_t g(eng.kind());
    g.add_op(&add_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("sum_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src0_lt, &src1_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};

    p.compile(&cp, inputs, outputs, &eng);

    impl::logical_tensor_t compiled_dst_lt;
    cp.query_logical_tensor(dst_lt.id, &compiled_dst_lt);

    impl::tensor_t src0_ts(src0_lt, &eng, src0.data());
    impl::tensor_t src1_ts(src1_lt, &eng, src1.data());
    impl::tensor_t dst_ts(compiled_dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src0_ts, src1_ts}, {dst_ts});
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
    impl::tensor_t dst_2nd_ts(compiled_dst_lt, &eng, dst_2nd.data());
    cp.execute(&strm, {src0_2nd_ts, src1_2nd_ts}, {dst_2nd_ts});
    strm.wait();

    for (size_t i = 0; i < ref_dst_2nd.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_2nd[i], ref_dst_2nd[i]);
    }
}

TEST(Execute, multidirectionalbBroadcastAddAB) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src0 {1.0, 1.0, 1.0};
    test::vector<float> src1 {2.0, 2.0, 2.0};
    test::vector<float> ref_dst {3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0};
    test::vector<float> dst(ref_dst.size(), 0.0);

    impl::op_t add_op(0, impl::op_kind::Add, "add");

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {3, 1}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt = utils::logical_tensor_init(
            1, std::vector<impl::dim_t> {3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {3, 3}, impl::data_type::f32);

    add_op.add_input(src0_lt);
    add_op.add_input(src1_lt);
    add_op.add_output(dst_lt);

    impl::graph_t g(eng.kind());
    g.add_op(&add_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("sum_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src0_lt, &src1_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};

    p.compile(&cp, inputs, outputs, &eng);

    impl::logical_tensor_t compiled_dst_lt;
    cp.query_logical_tensor(dst_lt.id, &compiled_dst_lt);

    impl::tensor_t src0_ts(src0_lt, &eng, src0.data());
    impl::tensor_t src1_ts(src1_lt, &eng, src1.data());
    impl::tensor_t dst_ts(compiled_dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src0_ts, src1_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < ref_dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, MultidirectionalBroadcastAdd) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src0(8, 1.0);
    test::vector<float> src1(3, 2.0);
    test::vector<float> ref_dst(24, 3.0);
    test::vector<float> dst(ref_dst.size(), 0.0);

    impl::op_t add_op(0, impl::op_kind::Add, "add");

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {2, 1, 4}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt = utils::logical_tensor_init(
            1, std::vector<impl::dim_t> {1, 3, 1}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {2, 3, 4}, impl::data_type::f32);

    add_op.add_input(src0_lt);
    add_op.add_input(src1_lt);
    add_op.add_output(dst_lt);

    impl::graph_t g(eng.kind());
    g.add_op(&add_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("sum_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src0_lt, &src1_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};

    p.compile(&cp, inputs, outputs, &eng);

    impl::logical_tensor_t compiled_dst_lt;
    cp.query_logical_tensor(dst_lt.id, &compiled_dst_lt);

    impl::tensor_t src0_ts(src0_lt, &eng, src0.data());
    impl::tensor_t src1_ts(src1_lt, &eng, src1.data());
    impl::tensor_t dst_ts(compiled_dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src0_ts, src1_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < ref_dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, MultidirectionalBroadcastAddExpandDim) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src0(2, 1.0);
    test::vector<float> src1(12, 2.0);
    test::vector<float> ref_dst(24, 3.0);
    test::vector<float> dst(ref_dst.size(), 0.0);

    impl::op_t add_op(0, impl::op_kind::Add, "add");

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {2, 1, 1}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt = utils::logical_tensor_init(
            1, std::vector<impl::dim_t> {3, 4}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {2, 3, 4}, impl::data_type::f32);

    add_op.add_input(src0_lt);
    add_op.add_input(src1_lt);
    add_op.add_output(dst_lt);

    impl::graph_t g(eng.kind());
    g.add_op(&add_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("sum_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src0_lt, &src1_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};

    p.compile(&cp, inputs, outputs, &eng);

    impl::logical_tensor_t compiled_dst_lt;
    cp.query_logical_tensor(dst_lt.id, &compiled_dst_lt);

    impl::tensor_t src0_ts(src0_lt, &eng, src0.data());
    impl::tensor_t src1_ts(src1_lt, &eng, src1.data());
    impl::tensor_t dst_ts(compiled_dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src0_ts, src1_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < ref_dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Compile, AddShapeMismatchCase0) {
    impl::engine_t &eng = get_engine();

    impl::op_t add_op(0, impl::op_kind::Add, "add");

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {8, 4, 256}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {2, 2, 512}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {8, 4, 256}, impl::data_type::f32);

    add_op.add_input(src0_lt);
    add_op.add_input(src1_lt);
    add_op.add_output(dst_lt);

    impl::graph_t g(eng.kind());
    g.add_op(&add_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("sum_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);
    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src0_lt, &src1_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};
    auto ret = p.compile(&cp, inputs, outputs, &eng);

    // compile the add operator
    ASSERT_EQ(ret, impl::status::invalid_shape);
}

TEST(Compile, AddShapeMismatch1) {
    impl::engine_t &eng = get_engine();

    impl::op_t add_op(impl::op_kind::Add);

    impl::logical_tensor_t src0_lt = utils::logical_tensor_init(
            0, {8, 15, 5, 7}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            2, {8, 15, 5, 7}, impl::data_type::f32);

    add_op.add_input(src0_lt);
    add_op.add_input(src1_lt);
    add_op.add_output(dst_lt);

    impl::graph_t g(eng.kind());
    g.add_op(&add_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("sum_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);
    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src0_lt, &src1_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};
    auto ret = p.compile(&cp, inputs, outputs, &eng);

    // compile the add operator
    ASSERT_EQ(ret, impl::status::success);
}

TEST(Compile, AddShapeMismatch2) {
    impl::engine_t &eng = get_engine();

    impl::op_t add_op(impl::op_kind::Add);

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt = utils::logical_tensor_init(
            1, {8, 15, 5, 7}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            2, {8, 15, 5, 7}, impl::data_type::f32);

    // compile the add operator
    add_op.add_input(src0_lt);
    add_op.add_input(src1_lt);
    add_op.add_output(dst_lt);

    impl::graph_t g(eng.kind());
    g.add_op(&add_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("sum_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);
    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src0_lt, &src1_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};
    auto ret = p.compile(&cp, inputs, outputs, &eng);

    ASSERT_EQ(ret, impl::status::success);
}

TEST(Execute, ReversedDifferentFormatBroadcastAdd) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src1 {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    test::vector<float> src0 {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    test::vector<float> ref_dst {2.0, 5.0, 8.0, 3.0, 6.0, 9.0, 4.0, 7.0, 10.0,
            2.0, 5.0, 8.0, 3.0, 6.0, 9.0, 4.0, 7.0, 10.0};
    test::vector<float> dst(src1.size(), 0.0);

    impl::op_t add_op(0, impl::op_kind::Add, "add");

    // we reverse the order of src0 and src1
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(0, {1, 2, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t src0_lt = utils::logical_tensor_init(
            1, {3, 3}, {1, 3}, impl::data_type::f32); // ba format
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {1, 2, 3, 3}, impl::data_type::f32);

    add_op.add_input(src0_lt);
    add_op.add_input(src1_lt);
    add_op.add_output(dst_lt);

    impl::graph_t g(eng.kind());
    g.add_op(&add_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("sum_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);
    std::vector<const impl::logical_tensor_t *> inputs {&src0_lt, &src1_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};
    p.compile(&cp, inputs, outputs, &eng);

    impl::logical_tensor_t compiled_dst_lt;
    cp.query_logical_tensor(dst_lt.id, &compiled_dst_lt);

    impl::tensor_t src0_ts(src0_lt, &eng, src0.data());
    impl::tensor_t src1_ts(src1_lt, &eng, src1.data());
    impl::tensor_t dst_ts(compiled_dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src0_ts, src1_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, BiasAdd) {
    impl::engine_t &eng = get_engine();

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

        impl::logical_tensor_t src_lt = utils::logical_tensor_init(
                0, src_shapes[i], impl::data_type::f32);
        impl::logical_tensor_t bias_lt = utils::logical_tensor_init(
                1, bias_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
                2, dst_shapes[i], impl::data_type::f32);

        bias_add_op.add_input(src_lt);
        bias_add_op.add_input(bias_lt);
        bias_add_op.add_output(dst_lt);

        impl::graph_t g(eng.kind());
        ASSERT_EQ(g.add_op(&bias_add_op), impl::status::success);
        g.build_graph();

        impl::pass::pass_base_ptr apass = get_pass("bias_add_pass");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1);
        auto part = g.get_partitions()[0];

        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);
        std::vector<const impl::logical_tensor_t *> inputs {&src_lt, &bias_lt};
        std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};
        p.compile(&cp, inputs, outputs, &eng);

        impl::tensor_t src_ts(src_lt, &eng, src.data());
        impl::tensor_t bias_ts(bias_lt, &eng, bias.data());
        impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

        impl::stream_t &strm = get_stream();
        cp.execute(&strm, {src_ts, bias_ts}, {dst_ts});
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

TEST(Execute, AddMul) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src0 {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    test::vector<float> src1 {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> post_src {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> ref_dst {6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0};
    test::vector<float> dst(src0.size(), 0.0);

    impl::op_t add_op(0, impl::op_kind::Add, "add");
    impl::op_t mul_op(1, impl::op_kind::Multiply, "mul");

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t add_dst_lt = utils::logical_tensor_init(
            2, {1, 3, 3}, impl::data_type::f32, impl::layout_type::any);
    impl::logical_tensor_t post_src_lt
            = utils::logical_tensor_init(3, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            4, {1, 3, 3}, impl::data_type::f32, impl::layout_type::any);

    add_op.add_input(src0_lt);
    add_op.add_input(src1_lt);
    add_op.add_output(add_dst_lt);
    mul_op.add_input(add_dst_lt);
    mul_op.add_input(post_src_lt);
    mul_op.add_output(dst_lt);

    impl::graph_t g(eng.kind());
    g.add_op(&add_op);
    g.add_op(&mul_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("binary_post_ops_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {
            &src0_lt, &src1_lt, &post_src_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};
    p.compile(&cp, inputs, outputs, &eng);

    impl::logical_tensor_t compiled_dst_lt;
    cp.query_logical_tensor(dst_lt.id, &compiled_dst_lt);

    impl::tensor_t src0_ts(src0_lt, &eng, src0.data());
    impl::tensor_t src1_ts(src1_lt, &eng, src1.data());
    impl::tensor_t post_src_ts(post_src_lt, &eng, post_src.data());
    impl::tensor_t dst_ts(compiled_dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src0_ts, src1_ts, post_src_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, AddMulPostSrcAsNxc) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src0 {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    test::vector<float> src1 {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> post_src {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    test::vector<float> ref_dst {
            3.0, 12.0, 21.0, 6.0, 15.0, 24.0, 9.0, 18.0, 27.0};
    test::vector<float> dst(src0.size(), 0.0);

    impl::op_t add_op(0, impl::op_kind::Add, "add");
    impl::op_t mul_op(1, impl::op_kind::Multiply, "mul");

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t add_dst_lt = utils::logical_tensor_init(
            2, {1, 3, 3}, impl::data_type::f32, impl::layout_type::any);
    impl::logical_tensor_t post_src_lt = utils::logical_tensor_init(
            3, {1, 3, 3}, {9, 1, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            4, {1, 3, 3}, impl::data_type::f32, impl::layout_type::any);

    add_op.add_input(src0_lt);
    add_op.add_input(src1_lt);
    add_op.add_output(add_dst_lt);
    mul_op.add_input(add_dst_lt);
    mul_op.add_input(post_src_lt);
    mul_op.add_output(dst_lt);

    impl::graph_t g(eng.kind());
    g.add_op(&add_op);
    g.add_op(&mul_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("binary_post_ops_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);
    std::vector<const impl::logical_tensor_t *> inputs {
            &src0_lt, &src1_lt, &post_src_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};
    p.compile(&cp, inputs, outputs, &eng);

    impl::logical_tensor_t compiled_dst_lt;
    cp.query_logical_tensor(dst_lt.id, &compiled_dst_lt);

    impl::tensor_t src0_ts(src0_lt, &eng, src0.data());
    impl::tensor_t src1_ts(src1_lt, &eng, src1.data());
    impl::tensor_t post_src_ts(post_src_lt, &eng, post_src.data());
    impl::tensor_t dst_ts(compiled_dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src0_ts, src1_ts, post_src_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, AddRelu) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src0 {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    test::vector<float> src1 {-2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> ref_dst {0.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0};
    test::vector<float> dst(src0.size(), 0.0);

    impl::op_t add_op(0, impl::op_kind::Add, "add");
    impl::op_t relu_op(1, impl::op_kind::ReLU, "relu");

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t add_dst_lt = utils::logical_tensor_init(
            2, {1, 3, 3}, impl::data_type::f32, impl::layout_type::any);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            3, {1, 3, 3}, impl::data_type::f32, impl::layout_type::any);

    add_op.add_input(src0_lt);
    add_op.add_input(src1_lt);
    add_op.add_output(add_dst_lt);
    relu_op.add_input(add_dst_lt);
    relu_op.add_output(dst_lt);

    impl::graph_t g(eng.kind());
    g.add_op(&add_op);
    g.add_op(&relu_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("binary_post_ops_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src0_lt, &src1_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

    impl::logical_tensor_t compiled_dst_lt;
    cp.query_logical_tensor(dst_lt.id, &compiled_dst_lt);

    ASSERT_EQ(dst_lt.layout_type, impl::layout_type::any);
    ASSERT_EQ(compiled_dst_lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src0_ts(src0_lt, &eng, src0.data());
    impl::tensor_t src1_ts(src1_lt, &eng, src1.data());
    impl::tensor_t dst_ts(compiled_dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src0_ts, src1_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, AddSigmoid) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src0 {2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0};
    test::vector<float> src1 {
            -2.0, -1.0, -2.0, -1.0, -2.0, -1.0, -2.0, -1.0, -2.0};
    test::vector<float> ref_dst {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
    test::vector<float> dst(src0.size(), 0.0);

    impl::op_t add_op(0, impl::op_kind::Add, "add");
    impl::op_t sigmoid_op(1, impl::op_kind::Sigmoid, "sigmoid");

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t add_dst_lt = utils::logical_tensor_init(
            2, {1, 3, 3}, impl::data_type::f32, impl::layout_type::any);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            3, {1, 3, 3}, impl::data_type::f32, impl::layout_type::any);

    add_op.add_input(src0_lt);
    add_op.add_input(src1_lt);
    add_op.add_output(add_dst_lt);
    sigmoid_op.add_input(add_dst_lt);
    sigmoid_op.add_output(dst_lt);

    impl::graph_t g(eng.kind());
    g.add_op(&add_op);
    g.add_op(&sigmoid_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("binary_post_ops_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);
    std::vector<const impl::logical_tensor_t *> inputs {&src0_lt, &src1_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

    impl::logical_tensor_t compiled_dst_lt;
    cp.query_logical_tensor(dst_lt.id, &compiled_dst_lt);

    ASSERT_EQ(dst_lt.layout_type, impl::layout_type::any);
    ASSERT_EQ(compiled_dst_lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src0_ts(src0_lt, &eng, src0.data());
    impl::tensor_t src1_ts(src1_lt, &eng, src1.data());
    impl::tensor_t dst_ts(compiled_dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src0_ts, src1_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, BinaryOp) {
    impl::engine_t &eng = get_engine();

    std::vector<impl::op_kind_t> op_kinds
            = {impl::op_kind::Multiply, impl::op_kind::Minimum,
                    impl::op_kind::Maximum, impl::op_kind::Divide,
                    impl::op_kind::Subtract, impl::op_kind::SquaredDifference};
    std::vector<std::string> pass_names = {"mul_pass", "min_pass", "max_pass",
            "div_pass", "sub_pass", "squareddifference_pass"};

    test::vector<float> src0 {2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0};
    test::vector<float> src1 {3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0};
    test::vector<float> dst(src0.size(), 0.0);

    auto src0_lt
            = utils::logical_tensor_init(0, {1, 3, 3}, impl::data_type::f32);
    auto src1_lt
            = utils::logical_tensor_init(1, {1, 3, 3}, impl::data_type::f32);
    auto dst_lt
            = utils::logical_tensor_init(2, {1, 3, 3}, impl::data_type::f32);

    for (size_t i = 0; i < op_kinds.size(); i++) {
        impl::op_t binary_op(op_kinds[i]);

        binary_op.add_input(src0_lt);
        binary_op.add_input(src1_lt);
        binary_op.add_output(dst_lt);

        impl::graph_t g(eng.kind());
        g.add_op(&binary_op);
        g.build_graph();

        impl::pass::pass_base_ptr apass = get_pass(pass_names[i]);
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1);
        auto part = g.get_partitions()[0];

        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> inputs {&src0_lt, &src1_lt};
        std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};

        ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

        impl::tensor_t src0_ts(src0_lt, &eng, src0.data());
        impl::tensor_t src1_ts(src1_lt, &eng, src1.data());
        impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

        impl::stream_t &strm = get_stream();
        cp.execute(&strm, {src0_ts, src1_ts}, {dst_ts});
        strm.wait();
    }
}

TEST(Execute, MulEltwise) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src0 {2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0};
    test::vector<float> src1 {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    test::vector<float> dst(src0.size(), 0.0);

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t mul_dst_lt
            = utils::logical_tensor_init(2, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(3, {1, 3, 3}, impl::data_type::f32);

    std::vector<impl::op_kind_t> eltwise_ops
            = {impl::op_kind::ReLU, impl::op_kind::Sigmoid};

    for (size_t i = 0; i < eltwise_ops.size(); i++) {
        impl::op_t mul_op(0, impl::op_kind::Multiply, "mul");
        impl::op_t eltwise_op(1, eltwise_ops[i], "eltwise");

        mul_op.add_input(src0_lt);
        mul_op.add_input(src1_lt);
        mul_op.add_output(mul_dst_lt);
        eltwise_op.add_input(mul_dst_lt);
        eltwise_op.add_output(dst_lt);

        impl::graph_t g(eng.kind());
        g.add_op(&mul_op);
        g.add_op(&eltwise_op);
        g.build_graph();

        impl::pass::pass_base_ptr apass = get_pass("binary_post_ops_fusion");
        ASSERT_NE(apass.get(), nullptr);
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1);
        auto part = g.get_partitions()[0];

        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);
        std::vector<const impl::logical_tensor_t *> inputs {&src0_lt, &src1_lt};
        std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};
        ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

        impl::tensor_t src0_ts(src0_lt, &eng, src0.data());
        impl::tensor_t src1_ts(src1_lt, &eng, src1.data());
        impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

        impl::stream_t &strm = get_stream();
        cp.execute(&strm, {src0_ts, src1_ts}, {dst_ts});
        strm.wait();
    }
}

TEST(Execute, BinaryOpAddFusion) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src0 {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> src1 {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> post_src {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> dst(src0.size(), 0.0);

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t tmp_dst_lt
            = utils::logical_tensor_init(2, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t post_src_lt
            = utils::logical_tensor_init(3, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(4, {1, 3, 3}, impl::data_type::f32);

    std::vector<impl::op_kind_t> bin_ops = {impl::op_kind::Multiply,
            impl::op_kind::Maximum, impl::op_kind::Minimum};

    for (size_t i = 0; i < bin_ops.size(); i++) {
        impl::op_t bin_op(0, bin_ops[i], "bin");
        impl::op_t add_op(1, impl::op_kind::Add, "add");

        bin_op.add_input(src0_lt);
        bin_op.add_input(src1_lt);
        bin_op.add_output(tmp_dst_lt);
        add_op.add_input(tmp_dst_lt);
        add_op.add_input(post_src_lt);
        add_op.add_output(dst_lt);

        impl::graph_t g(eng.kind());
        g.add_op(&bin_op);
        g.add_op(&add_op);
        g.build_graph();

        impl::pass::pass_base_ptr apass = get_pass("binary_post_ops_fusion");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1);
        auto part = g.get_partitions()[0];

        impl::partition_t p;
        p.init(part);
        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> inputs {
                &src0_lt, &src1_lt, &post_src_lt};
        std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};
        p.compile(&cp, inputs, outputs, &eng);

        impl::tensor_t src0_ts(src0_lt, &eng, src0.data());
        impl::tensor_t src1_ts(src1_lt, &eng, src1.data());
        impl::tensor_t post_src_ts(post_src_lt, &eng, post_src.data());
        impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

        impl::stream_t &strm = get_stream();
        cp.execute(&strm, {src0_ts, src1_ts, post_src_ts}, {dst_ts});
        strm.wait();
    }
}

TEST(Execute, BinarySub) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src0 {2.0};
    test::vector<float> src1 {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    test::vector<float> dst(src1.size(), 0.0);
    test::vector<float> ref {1.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0};

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {2, 1, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {2, 1, 2, 2}, impl::data_type::f32);

    impl::op_t bin_op(0, impl::op_kind::Subtract, "bin");

    bin_op.add_input(src0_lt);
    bin_op.add_input(src1_lt);
    bin_op.add_output(dst_lt);

    impl::graph_t g(eng.kind());
    g.add_op(&bin_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("sub_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);
    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src0_lt, &src1_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};
    p.compile(&cp, inputs, outputs, &eng);

    impl::tensor_t src0_ts(src0_lt, &eng, src0.data());
    impl::tensor_t src1_ts(src1_lt, &eng, src1.data());
    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src0_ts, src1_ts}, {dst_ts});
    strm.wait();
    for (size_t i = 0; i < ref.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref[i]);
    }
}

TEST(Execute, MinEltwise) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src0 {2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0};
    test::vector<float> src1 {-1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0};
    test::vector<float> dst(src0.size(), 0.0);

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t min_dst_lt
            = utils::logical_tensor_init(2, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(3, {1, 3, 3}, impl::data_type::f32);

    std::vector<impl::op_kind_t> eltwise_ops
            = {impl::op_kind::ReLU, impl::op_kind::Sigmoid};

    for (size_t i = 0; i < eltwise_ops.size(); i++) {
        impl::op_t min_op(0, impl::op_kind::Minimum, "min");
        impl::op_t eltwise_op(1, eltwise_ops[i], "eltwise");

        min_op.add_input(src0_lt);
        min_op.add_input(src1_lt);
        min_op.add_output(min_dst_lt);
        eltwise_op.add_input(min_dst_lt);
        eltwise_op.add_output(dst_lt);

        impl::graph_t g(eng.kind());
        g.add_op(&min_op);
        g.add_op(&eltwise_op);
        g.build_graph();

        impl::pass::pass_base_ptr apass = get_pass("binary_post_ops_fusion");
        ASSERT_NE(apass.get(), nullptr);
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1);
        auto part = g.get_partitions()[0];

        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);
        std::vector<const impl::logical_tensor_t *> inputs {&src0_lt, &src1_lt};
        std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};
        p.compile(&cp, inputs, outputs, &eng);

        impl::tensor_t src0_ts(src0_lt, &eng, src0.data());
        impl::tensor_t src1_ts(src1_lt, &eng, src1.data());
        impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

        impl::stream_t &strm = get_stream();
        cp.execute(&strm, {src0_ts, src1_ts}, {dst_ts});
        strm.wait();
    }
}

TEST(Execute, MaxEltwise) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src0 {-2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0};
    test::vector<float> src1 {-1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0};
    test::vector<float> ref_dst {0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> dst(src0.size(), 0.0);

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t max_dst_lt
            = utils::logical_tensor_init(2, {1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(3, {1, 3, 3}, impl::data_type::f32);

    std::vector<impl::op_kind_t> eltwise_ops
            = {impl::op_kind::ReLU, impl::op_kind::Sigmoid};

    for (size_t i = 0; i < eltwise_ops.size(); i++) {
        impl::op_t max_op(0, impl::op_kind::Maximum, "max");
        impl::op_t eltwise_op(1, eltwise_ops[i], "eltwise");

        max_op.add_input(src0_lt);
        max_op.add_input(src1_lt);
        max_op.add_output(max_dst_lt);
        eltwise_op.add_input(max_dst_lt);
        eltwise_op.add_output(dst_lt);

        impl::graph_t g(eng.kind());
        g.add_op(&max_op);
        g.add_op(&eltwise_op);
        g.build_graph();

        impl::pass::pass_base_ptr apass = get_pass("binary_post_ops_fusion");
        ASSERT_NE(apass.get(), nullptr);
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1);
        auto part = g.get_partitions()[0];

        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);
        std::vector<const impl::logical_tensor_t *> inputs {&src0_lt, &src1_lt};
        std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};
        p.compile(&cp, inputs, outputs, &eng);

        impl::tensor_t src0_ts(src0_lt, &eng, src0.data());
        impl::tensor_t src1_ts(src1_lt, &eng, src1.data());
        impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

        impl::stream_t &strm = get_stream();
        cp.execute(&strm, {src0_ts, src1_ts}, {dst_ts});
        strm.wait();
    }
}

TEST(Execute, MatmulFp32) {
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

    impl::graph_t g(eng.kind());
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

TEST(Execute, MatmulF16F16F16) {
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
            = utils::logical_tensor_init(0, {1, 2}, impl::data_type::f16);
    impl::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2, 1}, impl::data_type::f16);
    impl::logical_tensor_t dst
            = utils::logical_tensor_init(2, {1, 1}, impl::data_type::f16);

    matmul_op.add_input(src);
    matmul_op.add_input(weight);
    matmul_op.add_output(dst);

    impl::graph_t g(eng.kind());
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

TEST(Execute, MatmulBf16Bf16Bf16) {
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

    impl::graph_t g(eng.kind());
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

TEST(Execute, MatmulNdx1d) {
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

            impl::graph_t g(engine.kind());
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

TEST(Execute, Matmul1dxNd) {
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

            impl::graph_t g(engine.kind());
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

TEST(Execute, Matmul3dx3d) {
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

    impl::graph_t g(eng.kind());
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

TEST(Execute, MatmulReluFusion) {
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

    impl::graph_t g(eng.kind());
    ASSERT_EQ(g.add_op(&matmul_op), impl::status::success);
    ASSERT_EQ(g.add_op(&relu_op), impl::status::success);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("matmul_post_ops_chain_fusion");
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

TEST(Execute, MatmulBiasFusion) {
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

    impl::graph_t g(eng.kind());
    g.add_op(&matmul_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass
            = get_pass("matmul_bias_post_ops_chain_fusion");
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

TEST(Execute, MatmulSumBroadcast1d) {
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

    impl::graph_t g(engine.kind());
    ASSERT_EQ(g.add_op(&matmul_op), impl::status::success);
    ASSERT_EQ(g.add_op(&add_op), impl::status::success);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("matmul_post_ops_chain_fusion");
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

TEST(Execute, MatmulSumFusion) {
    impl::engine_t &engine = get_engine();

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

    impl::graph_t g(engine.kind());
    ASSERT_EQ(g.add_op(&matmul_op), impl::status::success);
    ASSERT_EQ(g.add_op(&add_op), impl::status::success);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("matmul_post_ops_chain_fusion");
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

TEST(Execute, MatmulSumGeluFusion) {
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

    impl::graph_t g(eng.kind());
    ASSERT_EQ(g.add_op(&matmul_op), impl::status::success);
    ASSERT_EQ(g.add_op(&add_op), impl::status::success);
    ASSERT_EQ(g.add_op(&gelu_op), impl::status::success);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("matmul_post_ops_chain_fusion");
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

TEST(Execute, MatmulSumReluFusion) {
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

    impl::graph_t g(engine.kind());
    ASSERT_EQ(g.add_op(&matmul_op), impl::status::success);
    ASSERT_EQ(g.add_op(&add_op), impl::status::success);
    ASSERT_EQ(g.add_op(&relu_op), impl::status::success);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("matmul_post_ops_chain_fusion");
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

TEST(Execute, MatmulBiasReluFusion) {
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

    impl::graph_t g(engine.kind());
    ASSERT_EQ(g.add_op(&matmul_op), impl::status::success);
    ASSERT_EQ(g.add_op(&relu_op), impl::status::success);
    g.build_graph();

    impl::pass::pass_base_ptr apass
            = get_pass("matmul_bias_post_ops_chain_fusion");
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

TEST(Execute, MatmulBiasGeluFusion) {
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

    impl::graph_t g(engine.kind());
    ASSERT_EQ(g.add_op(&matmul_op), impl::status::success);
    ASSERT_EQ(g.add_op(&gelu_op), impl::status::success);
    g.build_graph();

    impl::pass::pass_base_ptr apass
            = get_pass("matmul_bias_post_ops_chain_fusion");
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

TEST(Execute, MatmulBiasRelu6Fusion) {
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

    impl::graph_t g(engine.kind());
    ASSERT_EQ(g.add_op(&matmul_op), impl::status::success);
    ASSERT_EQ(g.add_op(&hardtanh_op), impl::status::success);
    g.build_graph();

    impl::pass::pass_base_ptr apass
            = get_pass("matmul_bias_post_ops_chain_fusion");
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

TEST(Execute, MatmulBiasHardtanhFusion) {
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

    impl::graph_t g(engine.kind());
    ASSERT_EQ(g.add_op(&matmul_op), impl::status::success);
    ASSERT_EQ(g.add_op(&hardtanh_op), impl::status::success);
    g.build_graph();

    impl::pass::pass_base_ptr apass
            = get_pass("matmul_bias_post_ops_chain_fusion");
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

TEST(Execute, MatmulBiasEluFusion) {
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

    impl::graph_t g(engine.kind());
    ASSERT_EQ(g.add_op(&matmul_op), impl::status::success);
    ASSERT_EQ(g.add_op(&elu_op), impl::status::success);
    g.build_graph();

    impl::pass::pass_base_ptr apass
            = get_pass("matmul_bias_post_ops_chain_fusion");
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

TEST(Execute, MatmulBiasSigmoidFusion) {
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

    impl::graph_t g(engine.kind());
    ASSERT_EQ(g.add_op(&matmul_op), impl::status::success);
    ASSERT_EQ(g.add_op(&sigmoid_op), impl::status::success);
    g.build_graph();

    impl::pass::pass_base_ptr apass
            = get_pass("matmul_bias_post_ops_chain_fusion");
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

TEST(Execute, MatmulBiasAddFusion) {
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

    impl::graph_t g(engine.kind());
    ASSERT_EQ(g.add_op(&matmul_op), impl::status::success);
    ASSERT_EQ(g.add_op(&add_op), impl::status::success);
    g.build_graph();

    impl::pass::pass_base_ptr apass
            = get_pass("matmul_bias_post_ops_chain_fusion");
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

TEST(Execute, MatmulBiasAddPerTensorBroadcast) {
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

        impl::graph_t g(engine.kind());
        ASSERT_EQ(g.add_op(&matmul_op), impl::status::success);
        ASSERT_EQ(g.add_op(&add_op), impl::status::success);
        g.build_graph();

        impl::pass::pass_base_ptr apass
                = get_pass("matmul_bias_post_ops_chain_fusion");
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

TEST(Execute, MatmulBiasAddPerChannelBroadcast) {
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

        impl::graph_t g(engine.kind());
        ASSERT_EQ(g.add_op(&matmul_op), impl::status::success);
        ASSERT_EQ(g.add_op(&add_op), impl::status::success);
        g.build_graph();

        impl::pass::pass_base_ptr apass
                = get_pass("matmul_bias_post_ops_chain_fusion");
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

TEST(Compile, MatmulBiasAddUnsupportedBroadcast) {
    impl::engine_t &engine = get_engine();

    std::vector<impl::dims> post_src_shapes = {{3}, {1, 3}};

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

        impl::graph_t g(engine.kind());
        ASSERT_EQ(g.add_op(&matmul_op), impl::status::success);
        ASSERT_EQ(g.add_op(&add_op), impl::status::success);
        g.build_graph();

        impl::pass::pass_base_ptr apass
                = get_pass("matmul_bias_post_ops_chain_fusion");
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
                impl::status::invalid_shape);
    }
}

TEST(Execute, MatmulBiasAddReluFusion) {
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

    impl::graph_t g(engine.kind());
    ASSERT_EQ(g.add_op(&matmul_op), impl::status::success);
    ASSERT_EQ(g.add_op(&add_op), impl::status::success);
    ASSERT_EQ(g.add_op(&relu_op), impl::status::success);
    g.build_graph();

    impl::pass::pass_base_ptr apass
            = get_pass("matmul_bias_post_ops_chain_fusion");
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

TEST(Execute, MatmulDivFusion) {
    impl::engine_t &engine = get_engine();

    impl::op_t matmul_op(0, impl::op_kind::MatMul, "matmul_op");
    impl::op_t div_op(1, impl::op_kind::Divide, "div_op");

    test::vector<float> src_data {-2.0, -1.5, 3.0, 0.5};
    test::vector<float> weight_data {-2.0, -1.5, 1.0, 1.0};
    test::vector<float> div_src1_data {-1.0};
    test::vector<float> ref_dst_data {
            -4.0, -3.0, -3.0, -2.25, -3.0, -3.0, -0.5, -0.5};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {2, 2, 1}, impl::data_type::f32);
    impl::logical_tensor_t weight
            = utils::logical_tensor_init(1, {2, 1, 2}, impl::data_type::f32);
    impl::logical_tensor_t div_src1
            = utils::logical_tensor_init(2, {1}, impl::data_type::f32);
    impl::logical_tensor_t dst
            = utils::logical_tensor_init(3, {2, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t div_dst
            = utils::logical_tensor_init(4, {2, 2, 2}, impl::data_type::f32);

    matmul_op.add_input(src);
    matmul_op.add_input(weight);
    matmul_op.add_output(dst);
    div_op.add_input(dst);
    div_op.add_input(div_src1);
    div_op.add_output(div_dst);

    impl::graph_t g(engine.kind());
    ASSERT_EQ(g.add_op(&matmul_op), impl::status::success);
    ASSERT_EQ(g.add_op(&div_op), impl::status::success);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("matmul_post_ops_chain_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {
            &src, &weight, &div_src1};
    std::vector<const impl::logical_tensor_t *> outputs {&div_dst};

    p.compile(&cp, inputs, outputs, &engine);

    impl::tensor_t src_ts(src, &engine, src_data.data());
    impl::tensor_t weight_ts(weight, &engine, weight_data.data());
    impl::tensor_t div_src1_ts(div_src1, &engine, div_src1_data.data());
    impl::tensor_t dst_ts(div_dst, &engine, dst_data.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src_ts, weight_ts, div_src1_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(Execute, MatmulDivAddFusion) {
    impl::engine_t &engine = get_engine();

    impl::op_t matmul_op(0, impl::op_kind::MatMul, "matmul_op");
    impl::op_t div_op(1, impl::op_kind::Divide, "div_op");
    impl::op_t add_op(2, impl::op_kind::Add, "add_op");

    test::vector<float> src_data {-2.0, -1.5, 3.0, 0.5};
    test::vector<float> weight_data {-2.0, -1.5, 1.0, 1.0};
    test::vector<float> div_src1_data {-1.0};
    test::vector<float> add_src1_data {1.0, 2.0};
    test::vector<float> ref_dst_data {
            -3.0, -1.0, -2.0, -0.25, -2.0, -1.0, 0.5, 1.5};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {1, 2, 2, 1}, impl::data_type::f32);
    impl::logical_tensor_t weight
            = utils::logical_tensor_init(1, {1, 2, 1, 2}, impl::data_type::f32);
    impl::logical_tensor_t div_src1
            = utils::logical_tensor_init(2, {1}, impl::data_type::f32);
    impl::logical_tensor_t dst
            = utils::logical_tensor_init(3, {1, 2, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t div_dst
            = utils::logical_tensor_init(4, {1, 2, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t add_src1
            = utils::logical_tensor_init(5, {1, 1, 1, 2}, impl::data_type::f32);
    impl::logical_tensor_t add_dst
            = utils::logical_tensor_init(6, {1, 2, 2, 2}, impl::data_type::f32);

    matmul_op.add_input(src);
    matmul_op.add_input(weight);
    matmul_op.add_output(dst);
    div_op.add_input(dst);
    div_op.add_input(div_src1);
    div_op.add_output(div_dst);
    add_op.add_input(div_dst);
    add_op.add_input(add_src1);
    add_op.add_output(add_dst);

    impl::graph_t g(engine.kind());
    ASSERT_EQ(g.add_op(&matmul_op), impl::status::success);
    ASSERT_EQ(g.add_op(&div_op), impl::status::success);
    ASSERT_EQ(g.add_op(&add_op), impl::status::success);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("matmul_post_ops_chain_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {
            &src, &weight, &div_src1, &add_src1};
    std::vector<const impl::logical_tensor_t *> outputs {&add_dst};

    p.compile(&cp, inputs, outputs, &engine);

    impl::tensor_t src_ts(src, &engine, src_data.data());
    impl::tensor_t weight_ts(weight, &engine, weight_data.data());
    impl::tensor_t div_src1_ts(div_src1, &engine, div_src1_data.data());
    impl::tensor_t add_src1_ts(add_src1, &engine, add_src1_data.data());
    impl::tensor_t dst_ts(add_dst, &engine, dst_data.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src_ts, weight_ts, div_src1_ts, add_src1_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(Execute, MatmulSwapBinaryMulAddFusion) {
    impl::engine_t &engine = get_engine();

    impl::op_t matmul_op(0, impl::op_kind::MatMul, "matmul_op");
    impl::op_t div_op(1, impl::op_kind::Multiply, "div_op");
    impl::op_t add_op(2, impl::op_kind::Add, "add_op");

    std::vector<int64_t> src_shape {2, 2, 2, 2};
    std::vector<int64_t> div_shape {1};
    std::vector<int64_t> add_shape {2, 1, 2, 2};

    test::vector<float> src_data(product(src_shape));
    test::vector<float> weight_data(product(src_shape));
    test::vector<float> div_src1_data(product(div_shape));
    test::vector<float> add_src1_data(product(add_shape));
    test::vector<float> dst_data(product(src_shape), 0.0);

    // random generate src, weight and bias data random seed = 7
    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
    std::generate(src_data.begin(), src_data.end(), [&]() {
        return static_cast<uint8_t>(f32_distribution(generator));
    });
    std::generate(weight_data.begin(), weight_data.end(),
            [&]() { return static_cast<int8_t>(f32_distribution(generator)); });
    std::generate(div_src1_data.begin(), div_src1_data.end(),
            [&]() { return static_cast<int8_t>(f32_distribution(generator)); });
    std::generate(add_src1_data.begin(), add_src1_data.end(),
            [&]() { return static_cast<int8_t>(f32_distribution(generator)); });

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, src_shape, impl::data_type::f32);
    impl::logical_tensor_t weight
            = utils::logical_tensor_init(1, src_shape, impl::data_type::f32);
    impl::logical_tensor_t div_src1
            = utils::logical_tensor_init(2, div_shape, impl::data_type::f32);
    impl::logical_tensor_t dst
            = utils::logical_tensor_init(3, src_shape, impl::data_type::f32);
    impl::logical_tensor_t div_dst
            = utils::logical_tensor_init(4, src_shape, impl::data_type::f32);
    impl::logical_tensor_t add_src1
            = utils::logical_tensor_init(5, add_shape, impl::data_type::f32);
    impl::logical_tensor_t add_dst
            = utils::logical_tensor_init(6, src_shape, impl::data_type::f32);

    matmul_op.add_input(src);
    matmul_op.add_input(weight);
    matmul_op.add_output(dst);
    div_op.add_input(dst);
    div_op.add_input(div_src1);
    div_op.add_output(div_dst);
    add_op.add_input(add_src1);
    add_op.add_input(div_dst);
    add_op.add_output(add_dst);

    impl::graph_t g(engine.kind());
    ASSERT_EQ(g.add_op(&matmul_op), impl::status::success);
    ASSERT_EQ(g.add_op(&div_op), impl::status::success);
    ASSERT_EQ(g.add_op(&add_op), impl::status::success);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("matmul_post_ops_chain_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {
            &src, &weight, &div_src1, &add_src1};
    std::vector<const impl::logical_tensor_t *> outputs {&add_dst};

    p.compile(&cp, inputs, outputs, &engine);

    impl::tensor_t src_ts(src, &engine, src_data.data());
    impl::tensor_t weight_ts(weight, &engine, weight_data.data());
    impl::tensor_t div_src1_ts(div_src1, &engine, div_src1_data.data());
    impl::tensor_t add_src1_ts(add_src1, &engine, add_src1_data.data());
    impl::tensor_t dst_ts(add_dst, &engine, dst_data.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src_ts, weight_ts, div_src1_ts, add_src1_ts}, {dst_ts});
    strm.wait();
}

TEST(Execute, MaxPool) {
    using dims = impl::dnnl_impl::dims;
    impl::engine_t &eng = get_engine();

    test::vector<float> src {-2.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> ref_dst {-0.5, 2.0, 3.0, 4.0};
    test::vector<float> dst(ref_dst.size(), 0.0);

    impl::op_t max_pool_op(0, impl::op_kind::MaxPool, "maxpool");
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

    max_pool_op.add_input(src_lt);
    max_pool_op.add_output(dst_lt);

    impl::graph_t g(eng.kind());
    g.add_op(&max_pool_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("max_pool_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};

    p.compile(&cp, inputs, outputs, &eng);

    impl::logical_tensor_t lt;
    cp.query_logical_tensor(dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, MaxPoolWithOpaqueInput) {
    // dequantize - maxpool
    using dims = impl::dnnl_impl::dims;
    impl::engine_t &eng = get_engine();

    SKIP_IF(eng.kind() == impl::engine_kind::gpu,
            "Skip for GPU - not supported yet.");

    // prepare ops
    impl::op_t dequantize(0, impl::op_kind::Dequantize, "dq");
    dequantize.set_attr<std::vector<float>>("scales", {0.1f});
    dequantize.set_attr<std::vector<int64_t>>("zps", {10});
    dequantize.set_attr<std::string>("qtype", "per_tensor");
    dequantize.set_attr<int64_t>("axis", 0);

    impl::op_t maxpool(1, impl::op_kind::MaxPool, "maxpool");
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

    dequantize.add_input(dq_src_lt);
    dequantize.add_output(dq_dst_lt);
    maxpool.add_input(dq_dst_lt);
    maxpool.add_output(mp_dst_lt);

    impl::graph_t g(eng.kind());
    g.add_op(&dequantize);
    g.add_op(&maxpool);
    g.build_graph();

    impl::pass::pass_base_ptr apass1 = get_pass("dequant_pass");
    impl::pass::pass_base_ptr apass2 = get_pass("max_pool_pass");
    apass1->run(g);
    apass2->run(g);
    ASSERT_EQ(g.get_num_partitions(), 2);
    auto dq_part = g.get_partitions()[0];
    auto mp_part = g.get_partitions()[1];

    // compile
    impl::partition_t dq_p, mp_p;
    dq_p.init(dq_part);
    mp_p.init(mp_part);

    impl::compiled_partition_t dq_cp(dq_p);
    impl::compiled_partition_t mp_cp(mp_p);

    std::vector<const impl::logical_tensor_t *> dq_inputs {&dq_src_lt};
    std::vector<const impl::logical_tensor_t *> dq_outputs {&dq_dst_lt};
    ASSERT_EQ(dq_p.compile(&dq_cp, dq_inputs, dq_outputs, &eng),
            impl::status::success);

    impl::logical_tensor_t lt;
    dq_cp.query_logical_tensor(dq_dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    std::vector<const impl::logical_tensor_t *> mp_inputs {&lt};
    std::vector<const impl::logical_tensor_t *> mp_outputs {&mp_dst_lt};
    ASSERT_EQ(mp_p.compile(&mp_cp, mp_inputs, mp_outputs, &eng),
            impl::status::success);

    mp_cp.query_logical_tensor(mp_dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);
}

TEST(Execute, AvgPoolExcludePad) {
    using dims = impl::dnnl_impl::dims;
    impl::engine_t &eng = get_engine();

    test::vector<float> src {-2.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            3.0, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> ref_dst {
            -2.0, 0.25, 0.5, 0.75, 0.5, 0.75, 3.0, -1.5, 4.0};
    test::vector<float> dst(ref_dst.size(), 0.0);

    impl::op_t avg_pool_op(0, impl::op_kind::AvgPool, "avgpool");
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

    avg_pool_op.add_input(src_lt);
    avg_pool_op.add_output(dst_lt);

    impl::graph_t g(eng.kind());
    g.add_op(&avg_pool_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("avg_pool_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};

    p.compile(&cp, inputs, outputs, &eng);

    impl::logical_tensor_t lt;
    cp.query_logical_tensor(dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, AvgPoolIncludePad) {
    using dims = impl::dnnl_impl::dims;
    impl::engine_t &eng = get_engine();

    test::vector<float> src {-2.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            3.0, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> ref_dst {
            -0.5, 0.125, 0.125, 0.375, 0.5, 0.375, 0.75, -0.75, 1.0};
    test::vector<float> dst(ref_dst.size(), 0.0);

    impl::op_t avg_pool_op(0, impl::op_kind::AvgPool, "avgpool");
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

    avg_pool_op.add_input(src_lt);
    avg_pool_op.add_output(dst_lt);

    impl::graph_t g(eng.kind());
    g.add_op(&avg_pool_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("avg_pool_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};

    p.compile(&cp, inputs, outputs, &eng);

    impl::logical_tensor_t lt;
    cp.query_logical_tensor(dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, ConvolutionNcxOix) {
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

    impl::graph_t g(eng.kind());
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

TEST(Execute, ConvtransposeWithGroups) {
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

    impl::graph_t g(eng.kind());
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

class Convtranspose4D5D
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

        impl::graph_t g(eng.kind());
        g.add_op(&convtranspose_op);
        g.build_graph();

        impl::pass::pass_base_ptr apass
                = get_pass("convtranspose_post_ops_fusion");
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

TEST_P(Convtranspose4D5D, TestConvtranspose) {
    TestConvtranspose();
}

INSTANTIATE_TEST_SUITE_P(Execute, Convtranspose4D5D,
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

struct dnnl_graph_test_convtranspose_bwd_params {
    std::vector<int64_t> src_dims;
    std::vector<int64_t> wei_dims;
    std::vector<int64_t> dst_dims;
    test::vector<float> src0_data;
    test::vector<float> src1_data;
    test::vector<float> ref_dst_data;
    impl::dnnl_impl::dims strides;
    impl::dnnl_impl::dims pads_begin;
    impl::dnnl_impl::dims pads_end;
    impl::dnnl_impl::dims dilations;
    std::string data_format;
    std::string filter_format;
};

class ConvTransposeBackpropData
    : public ::testing::TestWithParam<
              dnnl_graph_test_convtranspose_bwd_params> {
public:
    void TestConvTransposeBackpropData() {
        using dims = impl::dnnl_impl::dims;

        auto params = ::testing::TestWithParam<
                dnnl_graph_test_convtranspose_bwd_params>::GetParam();

        // default engine kind is cpu.
        impl::engine_t &eng = get_engine();
        test::vector<float> diff_dst = params.src0_data;
        test::vector<float> weight = params.src1_data;
        test::vector<float> ref_diff_src = params.ref_dst_data;
        test::vector<float> diff_src(ref_diff_src.size(), 0.0);
        impl::op_t deconv_op(impl::op_kind::ConvTransposeBackpropData);
        deconv_op.set_attr<dims>("strides", params.strides);
        deconv_op.set_attr<dims>("dilations", params.dilations);
        deconv_op.set_attr<dims>("pads_begin", params.pads_begin);
        deconv_op.set_attr<dims>("pads_end", params.pads_end);
        deconv_op.set_attr<int64_t>("groups", 1);
        deconv_op.set_attr<std::string>("data_format", params.data_format);
        deconv_op.set_attr<std::string>("filter_format", params.filter_format);

        // prepare logical tensor
        impl::logical_tensor_t diff_dst_lt = utils::logical_tensor_init(
                0, params.dst_dims, impl::data_type::f32);
        impl::logical_tensor_t weight_lt = utils::logical_tensor_init(
                1, params.wei_dims, impl::data_type::f32);
        impl::logical_tensor_t diff_src_lt = utils::logical_tensor_init(
                2, params.src_dims, impl::data_type::f32);

        deconv_op.add_input(diff_dst_lt);
        deconv_op.add_input(weight_lt);
        deconv_op.add_output(diff_src_lt);

        impl::graph_t g(eng.kind());
        g.add_op(&deconv_op);
        g.build_graph();

        impl::pass::pass_base_ptr apass
                = get_pass("convtranspose_data_bwd_pass");
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

        impl::tensor_t diff_dst_ts(diff_dst_lt, &eng, diff_dst.data());
        impl::tensor_t weight_ts(weight_lt, &eng, weight.data());
        impl::tensor_t diff_src_ts(diff_src_lt, &eng, diff_src.data());

        impl::stream_t &strm = get_stream();
        cp.execute(&strm, {diff_dst_ts, weight_ts}, {diff_src_ts});
        strm.wait();
        for (size_t i = 0; i < diff_src.size(); ++i) {
            ASSERT_FLOAT_EQ(diff_src[i], ref_diff_src[i]);
        }
    }
};

TEST_P(ConvTransposeBackpropData, TestConvTransposeBackpropData) {
    TestConvTransposeBackpropData();
}

INSTANTIATE_TEST_SUITE_P(Execute, ConvTransposeBackpropData,
        ::testing::Values(
                // NCX, OIX
                dnnl_graph_test_convtranspose_bwd_params {{1, 1, 2, 2},
                        {1, 1, 3, 3}, {1, 1, 4, 4},
                        {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0, 2.5,
                                -1.0, 0, 3.0, -2.0, -1.0, 4.0},
                        {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0},
                        {-1.0, 2.5, 5.0, 1.5}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                        "NCX", "OIX"},
                // 3d, NCX, IOX
                dnnl_graph_test_convtranspose_bwd_params {{1, 1, 1, 2, 2},
                        {1, 1, 1, 3, 3}, {1, 1, 1, 4, 4},
                        {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0, 2.5,
                                -1.0, 0, 3.0, -2.0, -1.0, 4.0},
                        {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0},
                        {-1.0, 2.5, 5.0, 1.5}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0},
                        {1, 1, 1}, "NCX", "OIX"},
                // NCX, XIO
                dnnl_graph_test_convtranspose_bwd_params {{1, 1, 2, 2},
                        {3, 3, 1, 1}, {1, 1, 4, 4},
                        {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0, 2.5,
                                -1.0, 0, 3.0, -2.0, -1.0, 4.0},
                        {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0},
                        {-1.0, 2.5, 5.0, 1.5}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                        "NCX", "XIO"},
                // 3d, NCX, XIO
                dnnl_graph_test_convtranspose_bwd_params {{1, 1, 1, 2, 2},
                        {1, 3, 3, 1, 1}, {1, 1, 1, 4, 4},
                        {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0, 2.5,
                                -1.0, 0, 3.0, -2.0, -1.0, 4.0},
                        {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0},
                        {-1.0, 2.5, 5.0, 1.5}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0},
                        {1, 1, 1}, "NCX", "XIO"},
                // NXC, XIO
                dnnl_graph_test_convtranspose_bwd_params {{1, 1, 1, 1},
                        {3, 4, 1, 1}, {1, 3, 4, 1},
                        {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0, 2.5,
                                -1.0, 0},
                        {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
                                0.0},
                        {0.5}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, "NXC", "XIO"},
                // 3d, NXC, XIO
                dnnl_graph_test_convtranspose_bwd_params {{1, 1, 1, 1, 1},
                        {1, 3, 4, 1, 1}, {1, 1, 3, 4, 1},
                        {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0, 2.5,
                                -1.0, 0},
                        {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
                                0.0},
                        {0.5}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}, {1, 1, 1},
                        "NXC", "XIO"},
                // NXC, OIX
                dnnl_graph_test_convtranspose_bwd_params {{1, 2, 2, 1},
                        {1, 1, 3, 3}, {1, 4, 4, 1},
                        {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0, 2.5,
                                -1.0, 0, 3.0, -2.0, -1.0, 4.0},
                        {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0},
                        {-1.0, 2.5, 5.0, 1.5}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                        "NXC", "OIX"},
                // 3d, NXC, OIX
                dnnl_graph_test_convtranspose_bwd_params {{1, 1, 2, 2, 1},
                        {1, 1, 1, 3, 3}, {1, 1, 4, 4, 1},
                        {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0, 2.5,
                                -1.0, 0, 3.0, -2.0, -1.0, 4.0},
                        {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0},
                        {-1.0, 2.5, 5.0, 1.5}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0},
                        {1, 1, 1}, "NXC", "OIX"}));

class ConvTransposeBackpropFilters
    : public ::testing::TestWithParam<
              dnnl_graph_test_convtranspose_bwd_params> {
public:
    void TestConvTransposeBackpropFilters() {
        using dims = impl::dnnl_impl::dims;

        auto params = ::testing::TestWithParam<
                dnnl_graph_test_convtranspose_bwd_params>::GetParam();

        impl::engine_t &eng = get_engine();
        test::vector<float> src = params.src0_data;
        test::vector<float> diff_dst = params.src1_data;
        test::vector<float> ref_diff_wei = params.ref_dst_data;
        test::vector<float> diff_wei(ref_diff_wei.size(), 0.0);
        impl::op_t deconv_op(impl::op_kind::ConvTransposeBackpropFilters);
        deconv_op.set_attr<dims>("strides", params.strides);
        deconv_op.set_attr<dims>("dilations", params.dilations);
        deconv_op.set_attr<dims>("pads_begin", params.pads_begin);
        deconv_op.set_attr<dims>("pads_end", params.pads_end);
        deconv_op.set_attr<dims>("filter_shape", params.wei_dims);
        deconv_op.set_attr<int64_t>("groups", 1);
        deconv_op.set_attr<std::string>("data_format", params.data_format);
        deconv_op.set_attr<std::string>("filter_format", params.filter_format);

        impl::logical_tensor_t src_lt = utils::logical_tensor_init(
                0, params.src_dims, impl::data_type::f32);
        impl::logical_tensor_t diff_dst_lt = utils::logical_tensor_init(
                1, params.dst_dims, impl::data_type::f32);
        impl::logical_tensor_t diff_wei_lt = utils::logical_tensor_init(
                2, params.wei_dims, impl::data_type::f32);

        deconv_op.add_input(src_lt);
        deconv_op.add_input(diff_dst_lt);
        deconv_op.add_output(diff_wei_lt);

        impl::graph_t g(eng.kind());
        g.add_op(&deconv_op);
        g.build_graph();

        impl::pass::pass_base_ptr apass
                = get_pass("convtranspose_filter_bwd_pass");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1);
        auto part = g.get_partitions()[0];

        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> inputs {
                &src_lt, &diff_dst_lt};
        std::vector<const impl::logical_tensor_t *> outputs {&diff_wei_lt};

        ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

        impl::logical_tensor_t lt;
        cp.query_logical_tensor(diff_wei_lt.id, &lt);
        ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

        impl::tensor_t src_ts(src_lt, &eng, src.data());
        impl::tensor_t diff_dst_ts(diff_dst_lt, &eng, diff_dst.data());
        impl::tensor_t diff_wei_ts(diff_wei_lt, &eng, diff_wei.data());

        impl::stream_t &strm = get_stream();
        cp.execute(&strm, {src_ts, diff_dst_ts}, {diff_wei_ts});
        strm.wait();
        for (size_t i = 0; i < diff_wei.size(); ++i) {
            ASSERT_FLOAT_EQ(diff_wei[i], ref_diff_wei[i]);
        }
    }
};

TEST_P(ConvTransposeBackpropFilters, TestConvTransposeBackpropFilters) {
    TestConvTransposeBackpropFilters();
}

INSTANTIATE_TEST_SUITE_P(Execute, ConvTransposeBackpropFilters,
        ::testing::Values(
                dnnl_graph_test_convtranspose_bwd_params {{1, 1, 2, 2},
                        {1, 1, 3, 3}, {1, 1, 4, 4}, {-1.0, 2.5, 5, 1.5},
                        {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0, 2.5,
                                -1.0, 0.0, 3.0, -2.0, -1.0, 4.0},
                        {-4.75, 3.0, 6.5, 11.75, 14.5, -2.25, 16.25, -16.5,
                                2.0},
                        {1, 1}, {0, 0}, {0, 0}, {1, 1}, "NCX", "OIX"},
                dnnl_graph_test_convtranspose_bwd_params {{1, 1, 1, 2, 2},
                        {1, 1, 1, 3, 3}, {1, 1, 1, 4, 4}, {-1.0, 2.5, 5, 1.5},
                        {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0, 2.5,
                                -1.0, 0.0, 3.0, -2.0, -1.0, 4.0},
                        {-4.75, 3.0, 6.5, 11.75, 14.5, -2.25, 16.25, -16.5,
                                2.0},
                        {1, 1, 1}, {0, 0, 0}, {0, 0, 0}, {1, 1, 1}, "NCX",
                        "OIX"},
                dnnl_graph_test_convtranspose_bwd_params {{1, 2, 2, 1},
                        {3, 3, 1, 1}, {1, 4, 4, 1}, {-1.0, 2.5, 5, 1.5},
                        {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0, 2.5,
                                -1.0, 0.0, 3.0, -2.0, -1.0, 4.0},
                        {-4.75, 3.0, 6.5, 11.75, 14.5, -2.25, 16.25, -16.5,
                                2.0},
                        {1, 1}, {0, 0}, {0, 0}, {1, 1}, "NXC", "XIO"},
                dnnl_graph_test_convtranspose_bwd_params {{1, 1, 2, 2, 1},
                        {1, 3, 3, 1, 1}, {1, 1, 4, 4, 1}, {-1.0, 2.5, 5, 1.5},
                        {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0, 2.5,
                                -1.0, 0.0, 3.0, -2.0, -1.0, 4.0},
                        {-4.75, 3.0, 6.5, 11.75, 14.5, -2.25, 16.25, -16.5,
                                2.0},
                        {1, 1, 1}, {0, 0, 0}, {0, 0, 0}, {1, 1, 1}, "NXC",
                        "XIO"}));

TEST(Compile, ConvTransposeBackpropFiltersWithGroupsAndFiltersAnyLayout) {
    using dims = impl::dnnl_impl::dims;

    const dims src_dims {2, 4, 2};
    const dims diff_dst_dims {2, 4, 2};
    const dims diff_wei_dims {2, 4, 3};

    const dims strides {1};
    const dims pads_begin {1};
    const dims pads_end {1};
    const dims dilations {1};
    const int64_t groups {2};
    const std::string auto_pad {"None"};
    const std::string data_format {"NCX"};
    const std::string filter_format {"OIX"};

    impl::op_t deconv_op(impl::op_kind::ConvTransposeBackpropFilters);
    deconv_op.set_attr("strides", strides)
            .set_attr("pads_begin", pads_begin)
            .set_attr("pads_end", pads_end)
            .set_attr("dilations", dilations)
            .set_attr("groups", groups)
            .set_attr("auto_pad", auto_pad)
            .set_attr("data_format", data_format)
            .set_attr("filter_format", filter_format);

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, src_dims, impl::data_type::f32);
    impl::logical_tensor_t diff_dst_lt = utils::logical_tensor_init(
            1, diff_dst_dims, impl::data_type::f32);
    impl::logical_tensor_t diff_wei_lt = utils::logical_tensor_init(
            2, diff_wei_dims, impl::data_type::f32, impl::layout_type::any);

    deconv_op.add_input(src_lt);
    deconv_op.add_input(diff_dst_lt);
    deconv_op.add_output(diff_wei_lt);

    impl::engine_t &eng = get_engine();
    impl::graph_t g(eng.kind());
    g.add_op(&deconv_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("convtranspose_filter_bwd_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src_lt, &diff_dst_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&diff_wei_lt};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

    impl::logical_tensor_t lt;
    cp.query_logical_tensor(diff_wei_lt.id, &lt);
    // if layout queried from the primitive will make descriptor impossible
    // to reshape (with groups -> no groups), we make it strided (via reorder)
    ASSERT_TRUE(lt.layout_type == impl::layout_type::opaque
            || lt.layout_type == impl::layout_type::strided);
}

TEST(Execute, Convolution3DNcxOix) {
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

    impl::graph_t g(eng.kind());
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

TEST(Execute, ConvolutionNcxXio) {
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

    impl::graph_t g(eng.kind());
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

TEST(Execute, Convolution3DNcxXio) {
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

    impl::graph_t g(eng.kind());
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

TEST(Execute, ConvolutionNxcXio) {
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

    impl::graph_t g(eng.kind());
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

TEST(Execute, Convolution3DNxcXio) {
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

    impl::graph_t g(eng.kind());
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

TEST(Execute, ConvolutionNxcOix) {
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

    impl::graph_t g(eng.kind());
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

TEST(Execute, Convolution3DNxcOix) {
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

    impl::graph_t g(eng.kind());
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

TEST(Execute, ConvolutionF16F16F16) {
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

    impl::graph_t g(eng.kind());
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

TEST(Execute, ConvolutionBf16Bf16Bf16) {
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

    impl::graph_t g(eng.kind());
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

TEST(Compile, ConvAddSharedInputs) {
    /*      /\  /
           / Conv
           \  /
           Add
    */
    using dims = impl::dnnl_impl::dims;

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt
            = utils::logical_tensor_init(1, {1, 1, 1, 1}, impl::data_type::f32);
    impl::logical_tensor_t conv_dst_lt
            = utils::logical_tensor_init(2, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t add_dst_lt
            = utils::logical_tensor_init(3, {1, 1, 4, 4}, impl::data_type::f32);

    // create op conv
    impl::op_t conv_op(0, impl::op_kind::Convolution, "Convolution");
    conv_op.set_attr<dims>("strides", dims {1, 1});
    conv_op.set_attr<dims>("dilations", dims {1, 1});
    conv_op.set_attr<dims>("pads_begin", dims {0, 0});
    conv_op.set_attr<dims>("pads_end", dims {0, 0});
    conv_op.set_attr<int64_t>("groups", 1);
    conv_op.set_attr<std::string>("data_format", "NCX");
    conv_op.set_attr<std::string>("filter_format", "OIX");
    conv_op.add_input(src_lt);
    conv_op.add_input(weight_lt);
    conv_op.add_output(conv_dst_lt);
    // create op add
    impl::op_t add_op(1, impl::op_kind::Add, "Add");
    add_op.add_input(conv_dst_lt);
    add_op.add_input(src_lt);
    add_op.add_output(add_dst_lt);
    // build graph
    impl::engine_t &eng = get_engine();
    impl::graph_t g(eng.kind());
    g.add_op(&conv_op);
    g.add_op(&add_op);
    g.build_graph();

    // run pass
    impl::pass::pass_base_ptr apass = get_pass("conv_post_ops_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile conv+add partition
    impl::partition_t p;
    p.init(part);
    impl::compiled_partition_t cp(p);
    std::vector<const impl::logical_tensor_t *> inputs {
            &src_lt, &weight_lt, &src_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&add_dst_lt};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

    // check inplace pairs
    std::vector<impl::inplace_pair_t> inplace_pairs = cp.get_inplace_pairs();
    ASSERT_EQ(inplace_pairs.size(), 0);
}

TEST(Compile, ConvAddInplace) {
    /*      \  /
             Conv
           \  /
           Add
    */
    using dims = impl::dnnl_impl::dims;

    // TODO(qun): re-enable this test once library and bridge align the inplace
    // logic
    SKIP_IF(true, "library and bridge have different inplace logic");

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t weight_lt
            = utils::logical_tensor_init(1, {1, 1, 1, 1}, impl::data_type::f32);
    impl::logical_tensor_t conv_dst_lt
            = utils::logical_tensor_init(2, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t add_src_lt
            = utils::logical_tensor_init(3, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t add_dst_lt
            = utils::logical_tensor_init(4, {1, 1, 4, 4}, impl::data_type::f32);

    // create op conv
    impl::op_t conv_op(0, impl::op_kind::Convolution, "Convolution");
    conv_op.set_attr<dims>("strides", dims {1, 1});
    conv_op.set_attr<dims>("dilations", dims {1, 1});
    conv_op.set_attr<dims>("pads_begin", dims {0, 0});
    conv_op.set_attr<dims>("pads_end", dims {0, 0});
    conv_op.set_attr<int64_t>("groups", 1);
    conv_op.set_attr<std::string>("data_format", "NCX");
    conv_op.set_attr<std::string>("filter_format", "OIX");
    conv_op.add_input(src_lt);
    conv_op.add_input(weight_lt);
    conv_op.add_output(conv_dst_lt);
    // create op add
    impl::op_t add_op(1, impl::op_kind::Add, "Add");
    add_op.add_input(add_src_lt);
    add_op.add_input(conv_dst_lt);
    add_op.add_output(add_dst_lt);
    // build graph
    impl::engine_t &eng = get_engine();
    impl::graph_t g(eng.kind());
    g.add_op(&conv_op);
    g.add_op(&add_op);
    g.build_graph();

    // run pass
    impl::pass::pass_base_ptr apass = get_pass("conv_post_ops_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile conv+add partition
    impl::partition_t p;
    p.init(part);
    impl::compiled_partition_t cp(p);
    // arbitrary order of inputs
    std::vector<const impl::logical_tensor_t *> inputs {
            &add_src_lt, &weight_lt, &src_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&add_dst_lt};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

    // check inplace pairs
    std::vector<impl::inplace_pair_t> inplace_pairs = cp.get_inplace_pairs();
    ASSERT_EQ(inplace_pairs.size(), 1);
    ASSERT_EQ(inplace_pairs[0].input, add_src_lt.id);
    ASSERT_EQ(inplace_pairs[0].output, add_dst_lt.id);
}

TEST(Execute, GroupConvolution) {
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

    impl::graph_t g(eng.kind());
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

TEST(Execute, ConvolutionBackpropData) {
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
    conv_op.set_attr<dims>("output_shape", dims {1, 1, 4, 4});

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

    impl::graph_t g(eng.kind());
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

TEST(Execute, ConvReluUnfused) {
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

    impl::graph_t g(eng.kind());
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

TEST(Execute, ConvolutionBnFp32) {
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

    impl::graph_t g(eng.kind());
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
    impl::pass::pass_base_ptr apass = get_pass("conv_post_ops_fusion");
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

TEST(Compile, ConvBnSharedInputs) {
    // bn has shared gamma/beta/mean/var
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
    impl::logical_tensor_t shared_lt
            = utils::logical_tensor_init(3, {32}, impl::data_type::f32);
    impl::logical_tensor_t bn_dst_lt = utils::logical_tensor_init(
            7, {8, 32, 16, 16}, impl::data_type::f32);

    test::vector<float> conv_src(8 * 32 * 16 * 16);
    test::vector<float> conv_weight(32 * 32 * 1 * 1);
    test::vector<float> bn_shared_input(32);
    test::vector<float> bn_dst(8 * 32 * 16 * 16);

    // Initialize
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0f, 0.1f);

    std::generate(conv_src.begin(), conv_src.end(),
            [&]() { return distribution(generator); });
    std::generate(conv_weight.begin(), conv_weight.end(),
            [&]() { return distribution(generator); });
    std::generate(bn_shared_input.begin(), bn_shared_input.end(),
            [&]() { return distribution(generator); });

    impl::tensor_t conv_src_ts(conv_src_lt, &eng, conv_src.data());
    impl::tensor_t conv_weight_ts(conv_weight_lt, &eng, conv_weight.data());
    impl::tensor_t bn_shared_input_ts(shared_lt, &eng, bn_shared_input.data());
    impl::tensor_t bn_dst_ts(bn_dst_lt, &eng, bn_dst.data());

    conv_op.add_input(conv_src_lt);
    conv_op.add_input(conv_weight_lt);
    conv_op.add_output(conv_dst_lt);
    bn_op.add_input(conv_dst_lt);
    bn_op.add_input(shared_lt);
    bn_op.add_input(shared_lt);
    bn_op.add_input(shared_lt);
    bn_op.add_input(shared_lt);
    bn_op.add_output(bn_dst_lt);

    impl::graph_t g(eng.kind());
    g.add_op(&conv_op);
    g.add_op(&bn_op);
    g.build_graph();

    // run unfused graph to compute the reference
    ASSERT_EQ(run_graph(g,
                      {conv_src_ts, conv_weight_ts, bn_shared_input_ts,
                              bn_shared_input_ts, bn_shared_input_ts,
                              bn_shared_input_ts},
                      {bn_dst_ts}, eng, strm),
            impl::status::success);

    // run fusion partition
    impl::pass::pass_base_ptr apass = get_pass("conv_post_ops_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&conv_src_lt,
            &conv_weight_lt, &shared_lt, &shared_lt, &shared_lt, &shared_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&bn_dst_lt};

    p.compile(&cp, inputs, outputs, &eng);

    test::vector<float> convbn_dst(8 * 32 * 16 * 16, 0.0);
    impl::tensor_t convbn_dst_ts(bn_dst_lt, &eng, convbn_dst.data());

    cp.execute(&strm,
            {conv_src_ts, conv_weight_ts, bn_shared_input_ts,
                    bn_shared_input_ts, bn_shared_input_ts, bn_shared_input_ts},
            {convbn_dst_ts});
    strm.wait();
    ASSERT_TRUE(allclose(bn_dst, convbn_dst, /*rtol*/ 0.1f, 1e-6f));
}

TEST(Execute, ConvAdd) {
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

        impl::graph_t g(eng.kind());
        g.add_op(&in_op);
        g.add_op(&conv_op);
        g.add_op(&add_op);
        g.build_graph();

        impl::pass::pass_base_ptr apass = get_pass("conv_post_ops_fusion");
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

TEST(Execute, ConvAddPerTensorBroadcast) {
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

    impl::graph_t g(eng.kind());
    g.add_op(&in_op);
    g.add_op(&conv_op);
    g.add_op(&add_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("conv_post_ops_fusion");
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

TEST(Execute, ConvAddExpandedPerTensorBroadcast) {
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

    impl::graph_t g(eng.kind());
    g.add_op(&in_op);
    g.add_op(&conv_op);
    g.add_op(&add_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("conv_post_ops_fusion");
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

TEST(Execute, ConvAddPerChannelBroadcast) {
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

    impl::graph_t g(eng.kind());
    g.add_op(&in_op);
    g.add_op(&conv_op);
    g.add_op(&add_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("conv_post_ops_fusion");
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

TEST(Execute, ConvAddPerChannelBroadcastNxc) {
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

    impl::graph_t g(eng.kind());
    g.add_op(&in_op);
    g.add_op(&conv_op);
    g.add_op(&add_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("conv_post_ops_fusion");
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

TEST(Compile, ConvAddBroadcast) {
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

    impl::graph_t g(eng.kind());
    g.add_op(&in_op);
    g.add_op(&conv_op);
    g.add_op(&add_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("conv_post_ops_fusion");
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

    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);
}

TEST(Execute, ConvAddRelu) {
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

    impl::graph_t g(eng.kind());
    g.add_op(&in_op);
    g.add_op(&conv_op);
    g.add_op(&add_op);
    g.add_op(&relu_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("conv_post_ops_fusion");
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

TEST(Execute, ConvMultiplePostOps) {
    using dims = impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> bias {1.0};
    test::vector<float> mul_other {2.0, 2.0, 2.0, 2.0};
    test::vector<float> sum_other {-1.0, -2.0, -3.0, -4.0};
    test::vector<float> add_other {1.0};
    test::vector<float> ref_dst {0.0, 6.0, 10.0, 2.0};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};

    impl::op_t conv_op(1, impl::op_kind::Convolution, "Convolution");
    conv_op.set_attr<dims>("strides", {1, 1});
    conv_op.set_attr<dims>("dilations", {1, 1});
    conv_op.set_attr<dims>("pads_begin", {0, 0});
    conv_op.set_attr<dims>("pads_end", {0, 0});
    conv_op.set_attr<int64_t>("groups", 1);
    std::string data_format = "NXC";
    std::string filter_format = "XIO";
    conv_op.set_attr<std::string>("data_format", data_format);
    conv_op.set_attr<std::string>("filter_format", filter_format);

    impl::op_t mul_op(2, impl::op_kind::Multiply, "Mul");
    impl::op_t sum_op(3, impl::op_kind::Add, "Sum");
    impl::op_t add_op(4, impl::op_kind::Add, "Add");

    std::vector<int64_t> src_dims {1, 1, 4, 4};
    std::vector<int64_t> weight_dims {1, 1, 3, 3};
    std::vector<int64_t> dst_dims {1, 1, 2, 2};
    if (data_format == "NXC") {
        src_dims = {1, 4, 4, 1};
        dst_dims = {1, 2, 2, 1};
    }
    if (filter_format == "XIO") weight_dims = {3, 3, 1, 1};

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, src_dims, impl::data_type::f32);
    impl::logical_tensor_t weight_lt
            = utils::logical_tensor_init(1, weight_dims, impl::data_type::f32);
    impl::logical_tensor_t bias_lt
            = utils::logical_tensor_init(2, {1}, impl::data_type::f32);
    impl::logical_tensor_t mul_other_lt
            = utils::logical_tensor_init(3, dst_dims, impl::data_type::f32);
    impl::logical_tensor_t mul_dst_lt
            = utils::logical_tensor_init(4, dst_dims, impl::data_type::f32);
    impl::logical_tensor_t sum_other_lt
            = utils::logical_tensor_init(5, dst_dims, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(6, dst_dims, impl::data_type::f32);
    impl::logical_tensor_t sum_dst_lt
            = utils::logical_tensor_init(7, dst_dims, impl::data_type::f32);
    impl::logical_tensor_t add_other_lt
            = utils::logical_tensor_init(8, {1}, impl::data_type::f32);
    impl::logical_tensor_t add_dst_lt
            = utils::logical_tensor_init(9, dst_dims, impl::data_type::f32);

    conv_op.add_input(src_lt);
    conv_op.add_input(weight_lt);
    conv_op.add_input(bias_lt);
    conv_op.add_output(dst_lt);
    mul_op.add_input(dst_lt);
    mul_op.add_input(mul_other_lt);
    mul_op.add_output(mul_dst_lt);
    sum_op.add_input(mul_dst_lt);
    sum_op.add_input(sum_other_lt);
    sum_op.add_output(sum_dst_lt);
    add_op.add_input(sum_dst_lt);
    add_op.add_input(add_other_lt);
    add_op.add_output(add_dst_lt);

    impl::graph_t g(eng.kind());
    g.add_op(&conv_op);
    g.add_op(&mul_op);
    g.add_op(&sum_op);
    g.add_op(&add_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("conv_bias_post_ops_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src_lt, &weight_lt,
            &bias_lt, &mul_other_lt, &sum_other_lt, &add_other_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&add_dst_lt};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

    std::vector<impl::inplace_pair_t> inplace_pairs = cp.get_inplace_pairs();
    ASSERT_EQ(inplace_pairs.size(), 1);
    ASSERT_EQ(inplace_pairs[0].input, sum_other_lt.id);
    ASSERT_EQ(inplace_pairs[0].output, add_dst_lt.id);

    impl::logical_tensor_t lt;
    cp.query_logical_tensor(add_dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t weight_ts(weight_lt, &eng, weight.data());
    impl::tensor_t bias_ts(bias_lt, &eng, bias.data());
    impl::tensor_t mul_other_ts(mul_other_lt, &eng, mul_other.data());
    impl::tensor_t sum_other_ts(sum_other_lt, &eng, sum_other.data());
    impl::tensor_t add_other_ts(add_other_lt, &eng, add_other.data());
    impl::tensor_t add_dst_ts(add_dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm,
            {src_ts, weight_ts, bias_ts, mul_other_ts, sum_other_ts,
                    add_other_ts},
            {add_dst_ts});
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

struct dnnl_graph_test_convtranspose_add_params {
    std::vector<impl::dim_t> add_src_shape;
    test::vector<float> add_src_data;
    test::vector<float> ref_dst;
    bool swap;
    bool with_bias;
};

class test_convtranspose_add_compile
    : public ::testing::TestWithParam<
              dnnl_graph_test_convtranspose_add_params> {
public:
    void TestConvTransposeAdd() {
        const auto params = ::testing::TestWithParam<
                dnnl_graph_test_convtranspose_add_params>::GetParam();
        using dims = impl::dnnl_impl::dims;

        impl::engine_t &eng = get_engine();
        test::vector<float> src_data {-1.0, 2.5, 5.0, 1.5};
        test::vector<float> weight_data {
                1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
        test::vector<float> bias_data = {1.0};
        test::vector<float> add_src_data = params.add_src_data;
        test::vector<float> ref_dst_data = params.ref_dst;
        test::vector<float> dst_data(ref_dst_data.size(), 0.0);

        impl::op_t convtranspose_op(
                0, impl::op_kind::ConvTranspose, "ConvTranspose");
        convtranspose_op.set_attr<dims>("strides", dims {1, 1});
        convtranspose_op.set_attr<dims>("dilations", dims {1, 1});
        convtranspose_op.set_attr<dims>("pads_begin", dims {0, 0});
        convtranspose_op.set_attr<dims>("pads_end", dims {0, 0});
        convtranspose_op.set_attr<int64_t>("groups", 1);
        convtranspose_op.set_attr<std::string>("data_format", "NXC");
        convtranspose_op.set_attr<std::string>("filter_format", "OIX");
        impl::op_t add_op(1, impl::op_kind::Add, "Add");

        // prepare logical tensor
        impl::logical_tensor_t src_lt = utils::logical_tensor_init(
                0, {1, 2, 2, 1}, impl::data_type::f32);
        impl::logical_tensor_t weight_lt = utils::logical_tensor_init(
                1, {1, 1, 3, 3}, impl::data_type::f32);
        impl::logical_tensor_t bias_lt
                = utils::logical_tensor_init(2, {1}, impl::data_type::f32);
        impl::logical_tensor_t add_src_lt = utils::logical_tensor_init(
                3, params.add_src_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
                4, {1, 4, 4, 1}, impl::data_type::f32, impl::layout_type::any);
        impl::logical_tensor_t add_dst_lt = utils::logical_tensor_init(
                5, {1, 4, 4, 1}, impl::data_type::f32);

        convtranspose_op.add_input(src_lt);
        convtranspose_op.add_input(weight_lt);
        if (params.with_bias) { convtranspose_op.add_input(bias_lt); }
        convtranspose_op.add_output(dst_lt);
        if (params.swap) {
            add_op.add_input(add_src_lt);
            add_op.add_input(dst_lt);
        } else {
            add_op.add_input(dst_lt);
            add_op.add_input(add_src_lt);
        }
        add_op.add_output(add_dst_lt);

        impl::graph_t g(eng.kind());
        g.add_op(&convtranspose_op);
        g.add_op(&add_op);
        g.build_graph();

        impl::pass::pass_base_ptr apass
                = get_pass("convtranspose_post_ops_fusion");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1);
        auto part = g.get_partitions()[0];

        // compile
        impl::partition_t p;
        p.init(part);
        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> inputs {
                &src_lt, &weight_lt};
        if (params.with_bias) { inputs.push_back(&bias_lt); }
        inputs.push_back(&add_src_lt);
        std::vector<const impl::logical_tensor_t *> outputs {&add_dst_lt};

        ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);
        impl::logical_tensor_t lt;
        cp.query_logical_tensor(add_dst_lt.id, &lt);
        ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

        impl::tensor_t src_ts(src_lt, &eng, src_data.data());
        impl::tensor_t weight_ts(weight_lt, &eng, weight_data.data());
        impl::tensor_t bias_ts;
        if (params.with_bias)
            bias_ts = impl::tensor_t(bias_lt, &eng, bias_data.data());
        impl::tensor_t add_src_ts(add_src_lt, &eng, add_src_data.data());
        impl::tensor_t add_dst_ts(add_dst_lt, &eng, dst_data.data());

        impl::stream_t &strm = get_stream();
        if (params.with_bias)
            cp.execute(&strm, {src_ts, weight_ts, bias_ts, add_src_ts},
                    {add_dst_ts});
        else
            cp.execute(&strm, {src_ts, weight_ts, add_src_ts}, {add_dst_ts});
        strm.wait();

        for (size_t i = 0; i < dst_data.size(); ++i) {
            ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
        }
    }
};

TEST_P(test_convtranspose_add_compile, TestConvTransposeAddCompile) {
    TestConvTransposeAdd();
}

INSTANTIATE_TEST_SUITE_P(TestConvTransposeAddCompile,
        test_convtranspose_add_compile,
        ::testing::Values(
                // with broadcast add, no swap inputs, without bias
                dnnl_graph_test_convtranspose_add_params {{1, 1, 1, 1}, {1.0},
                        {0.0, 3.5, 0.0, 3.5, 6.0, 1.5, 8.5, 2.5, 0.0, 8.5, 1.5,
                                3.5, 6.0, 2.5, 6.0, 2.5},
                        false, false},
                // with broadcast add, swap inputs, without bias
                dnnl_graph_test_convtranspose_add_params {{1, 1, 1, 1}, {1.0},
                        {0.0, 3.5, 0.0, 3.5, 6.0, 1.5, 8.5, 2.5, 0.0, 8.5, 1.5,
                                3.5, 6.0, 2.5, 6.0, 2.5},
                        true, false},
                // no broadcast add (sum), no swap inputs, without bias
                dnnl_graph_test_convtranspose_add_params {{1, 4, 4, 1},
                        {3.0, 3.0, 3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0,
                                2.0, 1.0, 1.0, 1.0, 1.0},
                        {2.0, 5.5, 2.0, 5.5, 6.0, 1.5, 8.5, 2.5, 1.0, 9.5, 2.5,
                                4.5, 6.0, 2.5, 6.0, 2.5},
                        false, false},
                // no broadcast add (sum), swap inputs, without bias
                dnnl_graph_test_convtranspose_add_params {{1, 4, 4, 1},
                        {3.0, 3.0, 3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0,
                                2.0, 1.0, 1.0, 1.0, 1.0},
                        {2.0, 5.5, 2.0, 5.5, 6.0, 1.5, 8.5, 2.5, 1.0, 9.5, 2.5,
                                4.5, 6.0, 2.5, 6.0, 2.5},
                        true, false},
                // with broadcast add, no swap inputs, with bias
                dnnl_graph_test_convtranspose_add_params {{1, 1, 1, 1}, {1.0},
                        {1.0, 4.5, 1.0, 4.5, 7.0, 2.5, 9.5, 3.5, 1.0, 9.5, 2.5,
                                4.5, 7.0, 3.5, 7.0, 3.5},
                        false, true},
                // with broadcast add, swap inputs, with bias
                dnnl_graph_test_convtranspose_add_params {{1, 1, 1, 1}, {1.0},
                        {1.0, 4.5, 1.0, 4.5, 7.0, 2.5, 9.5, 3.5, 1.0, 9.5, 2.5,
                                4.5, 7.0, 3.5, 7.0, 3.5},
                        true, true},
                // no broadcast add (sum), no swap inputs, with bias
                dnnl_graph_test_convtranspose_add_params {{1, 4, 4, 1},
                        {3.0, 3.0, 3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0,
                                2.0, 1.0, 1.0, 1.0, 1.0},
                        {3.0, 6.5, 3.0, 6.5, 7.0, 2.5, 9.5, 3.5, 2.0, 10.5, 3.5,
                                5.5, 7.0, 3.5, 7.0, 3.5},
                        false, true},
                // no broadcast add (sum), swap inputs, with bias
                dnnl_graph_test_convtranspose_add_params {{1, 4, 4, 1},
                        {3.0, 3.0, 3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0,
                                2.0, 1.0, 1.0, 1.0, 1.0},
                        {3.0, 6.5, 3.0, 6.5, 7.0, 2.5, 9.5, 3.5, 2.0, 10.5, 3.5,
                                5.5, 7.0, 3.5, 7.0, 3.5},
                        true, true}));

TEST(operator_kernel, convtranspose_relu) {
    using dims = impl::dnnl_impl::dims;

    std::vector<bool> with_biases = {false, true};

    for (auto with_bias : with_biases) {
        impl::engine_t &eng = get_engine();
        test::vector<float> src_data {-1.0, 2.5, 5.0, 1.5};
        test::vector<float> weight_data {
                1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
        test::vector<float> bias_data = {2.0};
        test::vector<float> ref_dst_data = with_bias
                ? test::vector<float> {1.0, 4.5, 1.0, 4.5, 7.0, 2.5, 9.5, 3.5,
                        1.0, 9.5, 2.5, 4.5, 7.0, 3.5, 7.0, 3.5}
                : test::vector<float> {0.0, 2.5, 0.0, 2.5, 5.0, 0.5, 7.5, 1.5,
                        0.0, 7.5, 0.5, 2.5, 5.0, 1.5, 5.0, 1.5};
        test::vector<float> dst_data(ref_dst_data.size(), 0.0);

        impl::op_t convtranspose_op(
                0, impl::op_kind::ConvTranspose, "ConvTranspose");
        convtranspose_op.set_attr<dims>("strides", dims {1, 1});
        convtranspose_op.set_attr<dims>("dilations", dims {1, 1});
        convtranspose_op.set_attr<dims>("pads_begin", dims {0, 0});
        convtranspose_op.set_attr<dims>("pads_end", dims {0, 0});
        convtranspose_op.set_attr<int64_t>("groups", 1);
        convtranspose_op.set_attr<std::string>("data_format", "NXC");
        convtranspose_op.set_attr<std::string>("filter_format", "XIO");
        impl::op_t relu_op(1, impl::op_kind::ReLU, "ReLU");

        // prepare logical tensor
        impl::logical_tensor_t src_lt = utils::logical_tensor_init(
                0, {1, 2, 2, 1}, impl::data_type::f32);
        impl::logical_tensor_t weight_lt = utils::logical_tensor_init(
                1, {3, 3, 1, 1}, impl::data_type::f32);
        impl::logical_tensor_t bias_lt
                = utils::logical_tensor_init(2, {1}, impl::data_type::f32);
        impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
                4, {1, 4, 4, 1}, impl::data_type::f32, impl::layout_type::any);
        impl::logical_tensor_t relu_dst_lt = utils::logical_tensor_init(
                5, {1, 4, 4, 1}, impl::data_type::f32);

        convtranspose_op.add_input(src_lt);
        convtranspose_op.add_input(weight_lt);
        if (with_bias) { convtranspose_op.add_input(bias_lt); }
        convtranspose_op.add_output(dst_lt);

        relu_op.add_input(dst_lt);
        relu_op.add_output(relu_dst_lt);

        impl::graph_t g(eng.kind());
        g.add_op(&convtranspose_op);
        g.add_op(&relu_op);
        g.build_graph();

        impl::pass::pass_base_ptr apass
                = get_pass("convtranspose_post_ops_fusion");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1);
        auto part = g.get_partitions()[0];

        // compile
        impl::partition_t p;
        p.init(part);
        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> inputs {
                &src_lt, &weight_lt};
        if (with_bias) { inputs.push_back(&bias_lt); }
        std::vector<const impl::logical_tensor_t *> outputs {&relu_dst_lt};

        ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);
        impl::logical_tensor_t lt;
        cp.query_logical_tensor(relu_dst_lt.id, &lt);
        ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

        impl::tensor_t src_ts(src_lt, &eng, src_data.data());
        impl::tensor_t weight_ts(weight_lt, &eng, weight_data.data());
        impl::tensor_t bias_ts;
        if (with_bias)
            bias_ts = impl::tensor_t(bias_lt, &eng, bias_data.data());
        impl::tensor_t relu_dst_ts(relu_dst_lt, &eng, dst_data.data());

        impl::stream_t &strm = get_stream();
        if (with_bias)
            cp.execute(&strm, {src_ts, weight_ts, bias_ts}, {relu_dst_ts});
        else
            cp.execute(&strm, {src_ts, weight_ts}, {relu_dst_ts});
        strm.wait();

        for (size_t i = 0; i < dst_data.size(); ++i) {
            ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
        }
    }
}

TEST(operator_kernel, convtranspose_swish) {
    using dims = impl::dnnl_impl::dims;

    std::vector<bool> with_biases = {false, true};

    for (auto with_bias : with_biases) {
        impl::engine_t &eng = get_engine();
        test::vector<float> src_data {-1.0, 2.5, 5.0, 1.5};
        test::vector<float> weight_data {
                1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
        test::vector<float> bias_data = {2.0};
        test::vector<float> dst_data(16, 0.0);

        impl::op_t convtranspose_op(
                0, impl::op_kind::ConvTranspose, "ConvTranspose");
        convtranspose_op.set_attr<dims>("strides", dims {1, 1});
        convtranspose_op.set_attr<dims>("dilations", dims {1, 1});
        convtranspose_op.set_attr<dims>("pads_begin", dims {0, 0});
        convtranspose_op.set_attr<dims>("pads_end", dims {0, 0});
        convtranspose_op.set_attr<int64_t>("groups", 1);
        convtranspose_op.set_attr<std::string>("data_format", "NXC");
        convtranspose_op.set_attr<std::string>("filter_format", "XIO");
        impl::op_t sigmoid_op(1, impl::op_kind::Sigmoid, "Sigmoid");
        impl::op_t multiply_op(2, impl::op_kind::Multiply, "Multiply");

        // prepare logical tensor
        impl::logical_tensor_t src_lt = utils::logical_tensor_init(
                0, {1, 2, 2, 1}, impl::data_type::f32);
        impl::logical_tensor_t weight_lt = utils::logical_tensor_init(
                1, {3, 3, 1, 1}, impl::data_type::f32);
        impl::logical_tensor_t bias_lt
                = utils::logical_tensor_init(2, {1}, impl::data_type::f32);
        impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
                3, {1, 4, 4, 1}, impl::data_type::f32, impl::layout_type::any);
        impl::logical_tensor_t sigmoid_dst_lt = utils::logical_tensor_init(
                4, {1, 4, 4, 1}, impl::data_type::f32);
        impl::logical_tensor_t multiply_dst_lt = utils::logical_tensor_init(
                5, {1, 4, 4, 1}, impl::data_type::f32);

        convtranspose_op.add_input(src_lt);
        convtranspose_op.add_input(weight_lt);
        if (with_bias) { convtranspose_op.add_input(bias_lt); }
        convtranspose_op.add_output(dst_lt);

        sigmoid_op.add_input(dst_lt);
        sigmoid_op.add_output(sigmoid_dst_lt);
        multiply_op.add_input(dst_lt);
        multiply_op.add_input(sigmoid_dst_lt);
        multiply_op.add_output(multiply_dst_lt);

        impl::graph_t g(eng.kind());
        g.add_op(&convtranspose_op);
        g.add_op(&sigmoid_op);
        g.add_op(&multiply_op);
        g.build_graph();

        impl::pass::pass_base_ptr apass
                = get_pass("convtranspose_post_ops_fusion");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1);
        auto part = g.get_partitions()[0];

        // compile
        impl::partition_t p;
        p.init(part);
        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> inputs {
                &src_lt, &weight_lt};
        if (with_bias) { inputs.push_back(&bias_lt); }
        std::vector<const impl::logical_tensor_t *> outputs {&multiply_dst_lt};

        ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);
        impl::logical_tensor_t lt;
        cp.query_logical_tensor(multiply_dst_lt.id, &lt);
        ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

        impl::tensor_t src_ts(src_lt, &eng, src_data.data());
        impl::tensor_t weight_ts(weight_lt, &eng, weight_data.data());
        impl::tensor_t bias_ts;
        if (with_bias)
            bias_ts = impl::tensor_t(bias_lt, &eng, bias_data.data());
        impl::tensor_t multiply_dst_ts(multiply_dst_lt, &eng, dst_data.data());

        impl::stream_t &strm = get_stream();
        if (with_bias)
            cp.execute(&strm, {src_ts, weight_ts, bias_ts}, {multiply_dst_ts});
        else
            cp.execute(&strm, {src_ts, weight_ts}, {multiply_dst_ts});
        strm.wait();
    }
}

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

test::vector<float> round_func(const test::vector<float> &ref_dst) {
    test::vector<float> out;
    for (auto &rdst : ref_dst) {
        out.emplace_back(static_cast<float>(round(rdst)));
    }
    return out;
}

template <typename attr_data_t = float>
void test_eltwise_common(test::vector<float> &src, test::vector<float> &ref_dst,
        dnnl::graph::impl::dims &dims, const dnnl_graph_op_kind_t op_kind,
        const std::string &op_name,
        const std::map<std::string, attr_data_t> &attrs_data = {}) {
    impl::engine_t &eng = get_engine();
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

    impl::graph_t g(eng.kind());
    g.add_op(&op);
    g.build_graph();

    std::string pass_name = op_name + "_pass";

    impl::pass::pass_base_ptr apass = get_pass(pass_name);
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};

    p.compile(&cp, inputs, outputs, &eng);

    impl::logical_tensor_t lt;
    cp.query_logical_tensor(dst_lt.id, &lt);

    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();

    cp.execute(&strm, {src_ts}, {dst_ts});
    strm.wait();

    if (op_kind == impl::op_kind::Log || op_kind == impl::op_kind::GELU
            || op_kind == impl::op_kind::SoftPlus) {
        for (size_t i = 0; i < src.size(); ++i) {
            ASSERT_TRUE(std::fabs(dst[i] - ref_dst[i]) < 0.00001);
        }
    } else {
        for (size_t i = 0; i < src.size(); ++i) {
            ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
        }
    }
}

template <typename attr_data_t = float>
void test_eltwise_bwd_common(
        const std::pair<test::vector<float>, bool> &fwd_data_pair,
        const test::vector<float> &diff_dst_data,
        const test::vector<float> &ref_diff_src, dnnl::graph::impl::dims &dims,
        const dnnl_graph_op_kind_t op_kind, const std::string &op_name,
        const std::map<std::string, attr_data_t> &attrs_data = {}) {
    static const std::set<dnnl_graph_op_kind_t> with_support_for_use_dst {
            impl::op_kind::EluBackprop, impl::op_kind::HardTanhBackprop,
            impl::op_kind::ReLUBackprop, impl::op_kind::SigmoidBackprop,
            impl::op_kind::SqrtBackprop, impl::op_kind::TanhBackprop};
    const test::vector<float> &fwd_data = fwd_data_pair.first;
    const bool is_fwd_data_src = fwd_data_pair.second;

    impl::op_t op(op_kind, op_name);
    for (const auto &attr_data : attrs_data) {
        op.set_attr<attr_data_t>(attr_data.first, attr_data.second);
    }
    if (with_support_for_use_dst.count(op_kind)) {
        op.set_attr<bool>("use_dst", !is_fwd_data_src);
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

    impl::engine_t &eng = get_engine();
    impl::graph_t g(eng.kind());
    g.add_op(&op);
    g.build_graph();

    std::string pass_name = op_name + "_pass";
    impl::pass::pass_base_ptr apass = get_pass(pass_name);
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);
    std::vector<const impl::logical_tensor_t *> inputs {
            &fwd_data_lt, &diff_dst_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&diff_src_lt};
    p.compile(&cp, inputs, outputs, &eng);

    impl::tensor_t fwd_data_ts(
            fwd_data_lt, &eng, const_cast<float *>(fwd_data.data()));
    impl::tensor_t diff_dst_ts(
            diff_dst_lt, &eng, const_cast<float *>(diff_dst_data.data()));
    impl::tensor_t diff_src_ts(diff_src_lt, &eng, diff_src_data.data());

    std::vector<impl::tensor_t> input_ts {fwd_data_ts, diff_dst_ts};
    std::vector<impl::tensor_t> output_ts {diff_src_ts};
    impl::stream_t &strm = get_stream();
    cp.execute(&strm, input_ts, output_ts);
    strm.wait();

    for (size_t i = 0; i < diff_src_data.size(); ++i) {
        ASSERT_TRUE(std::fabs(diff_src_data[i] - ref_diff_src[i]) < 0.00001);
    }
}

TEST(Execute, Abs) {
    test::vector<float> src {-2.0, -1.5, -1.0, -0.5, 0.0, 3.5};
    test::vector<float> ref_dst {2.0, 1.5, 1.0, 0.5, 0.0, 3.5};

    dnnl::graph::impl::dims dims {1, 2, 3};

    test_eltwise_common(src, ref_dst, dims, impl::op_kind::Abs, "abs");
}

TEST(Execute, Round) {
    test::vector<float> src {1.1f, -2.3f, 4.7f, 1.0f, 0.0f, -8.34f};
    test::vector<float> ref_dst = round_func(src);

    dnnl::graph::impl::dims dims {1, 2, 3};

    test_eltwise_common(src, ref_dst, dims, impl::op_kind::Round, "round");
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

    dnnl::graph::impl::dims dims {1, 2, 3};
    const std::map<std::string, float> attrs_data {{"alpha", 1.f}};

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

    dnnl::graph::impl::dims dims {1, 3, 3};
    const std::map<std::string, float> attrs_data {{"alpha", 1.f}};

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

    dnnl::graph::impl::dims dims {1, 2, 3};

    test_eltwise_common(src, ref_dst, dims, impl::op_kind::Exp, "exp");
}

TEST(Execute, Gelu) {
    test::vector<float> src {-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0};
    test::vector<float> ref_dst {-0.0455001f, -0.10021085f, -0.15865527f,
            -0.15426877f, 0.f, 0.3457312f, 0.84134465f, 1.399789f, 1.9544998f};

    dnnl::graph::impl::dims dims {1, 3, 3};

    test_eltwise_common(src, ref_dst, dims, impl::op_kind::GELU, "gelu");
}

TEST(Execute, GeluBackward) {
    test::vector<float> src {
            0, -1, 0.0723652, -0.0364869, 40, -50, 0.188521, -0.729739, 88.371};
    test::vector<float> diff_dst {
            3, -7, 0.0194608, -0.0559478, 70, 0, 0.754086, -0.218955, 88.5838};
    test::vector<float> ref_diff_src {1.5, 0.583208, 0.0108521, -0.0263458, 70,
            0, 0.489138, -0.00212478, 88.5838};

    dnnl::graph::impl::dims dims {1, 3, 3};

    test_eltwise_bwd_common({src, true}, diff_dst, ref_diff_src, dims,
            impl::op_kind::GELUBackprop, "gelu_bw");
}

TEST(Execute, HardSwish) {
    test::vector<float> src {7, -4, 0.0723652095, -0.0364869386, 60, -20,
            0.188521415, -0.729738772, 88.3709564};
    test::vector<float> ref_dst {7, 0, 0.0370553955, -0.0180215873, 60, 0,
            0.100184090, -0.276116282, 88.3709564};
    dnnl::graph::impl::dims dims {1, 3, 3};

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

    dnnl::graph::impl::dims dims {1, 3, 3};

    test_eltwise_bwd_common({src, true}, diff_dst, ref_diff_src, dims,
            impl::op_kind::HardSwishBackprop, "hardswish_bw");
}

TEST(Execute, Hardtanh) {
    test::vector<float> src {-2.0, -1.5, -1.0, -0.5, 0.0, 3.5};
    test::vector<float> ref_dst {-1.0, -1.0, -1.0, -0.5, 0.0, 2.0};

    dnnl::graph::impl::dims dims {1, 2, 3};
    const std::map<std::string, float> attrs_data {{"min", -1.f}, {"max", 2.f}};

    test_eltwise_common(src, ref_dst, dims, impl::op_kind::HardTanh, "hardtanh",
            attrs_data);
}

TEST(Execute, HardtanhBackward) {
    test::vector<float> src {
            0, -1, 0.0723652, -0.0364869, 40, -50, 0.188521, -0.729739, 88.371};
    test::vector<float> dst {
            0, -1, 0.0723652, -0.0364869, 2, -1, 0.188521, -0.729739, 2};
    test::vector<float> diff_dst {
            3, -7, 0.0194608, -0.0559478, 70, 0, 0.754086, -0.218955, 88.5838};
    test::vector<float> ref_diff_src {
            3, -0, 0.0194608, -0.0559478, 0, 0, 0.754086, -0.218955, 0};

    dnnl::graph::impl::dims dims {1, 3, 3};
    const std::map<std::string, float> attrs_data {{"min", -1.f}, {"max", 2.f}};

    test_eltwise_bwd_common({src, true}, diff_dst, ref_diff_src, dims,
            impl::op_kind::HardTanhBackprop, "hardtanh_bw", attrs_data);
    test_eltwise_bwd_common({dst, false}, diff_dst, ref_diff_src, dims,
            impl::op_kind::HardTanhBackprop, "hardtanh_bw", attrs_data);
}

TEST(Execute, Relu) {
    test::vector<float> src {-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0};
    test::vector<float> ref_dst {0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.5, 2.0};

    dnnl::graph::impl::dims dims {1, 3, 3};

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

    dnnl::graph::impl::dims dims {1, 3, 3};

    test_eltwise_bwd_common({src, true}, diff_dst, ref_diff_src, dims,
            impl::op_kind::ReLUBackprop, "relu_bw");
    test_eltwise_bwd_common({dst, false}, diff_dst, ref_diff_src, dims,
            impl::op_kind::ReLUBackprop, "relu_bw");
}

TEST(Execute, Sigmoid) {
    test::vector<float> src {-5.0, -2.5, -1.0, 0.3, 0.0, 1.2};
    test::vector<float> ref_dst = sigmoid_func(src);

    dnnl::graph::impl::dims dims {1, 2, 3};

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

    dnnl::graph::impl::dims dims {1, 3, 3};

    test_eltwise_bwd_common({src, true}, diff_dst, ref_diff_src, dims,
            impl::op_kind::SigmoidBackprop, "sigmoid_bw");
    test_eltwise_bwd_common({dst, false}, diff_dst, ref_diff_src, dims,
            impl::op_kind::SigmoidBackprop, "sigmoid_bw");
}

TEST(Execute, SoftPlus) {
    test::vector<float> src {
            0, -1, 0.0723652, -0.0364869, 40, -50, 0.188521, -0.729739, 88.371};
    test::vector<float> ref_dst_case1 = {-0.693147, -1.31326, -0.657619,
            -0.711557, -0, -50, -0.603322, -1.12315, -0};
    test::vector<float> ref_dst_case2 = {0.693147, 0.313262, 0.729984, 0.67507,
            40, 0, 0.791844, 0.393416, 88.371};

    dnnl::graph::impl::dims dims {1, 3, 3};
    const std::map<std::string, int64_t> attrs_data_case1 {{"beta", -1}};
    const std::map<std::string, int64_t> attrs_data_case2 {{"beta", 1}};

    test_eltwise_common(src, ref_dst_case1, dims, impl::op_kind::SoftPlus,
            "softplus", attrs_data_case1);
    test_eltwise_common(src, ref_dst_case2, dims, impl::op_kind::SoftPlus,
            "softplus", attrs_data_case2);
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

    dnnl::graph::impl::dims dims {1, 3, 3};
    const std::map<std::string, int64_t> attrs_data_case1 {{"beta", -1}};
    const std::map<std::string, int64_t> attrs_data_case2 {{"beta", 1}};

    test_eltwise_bwd_common({src, true}, diff_dst, ref_diff_src_case1, dims,
            impl::op_kind::SoftPlusBackprop, "softplus_bw", attrs_data_case1);
    test_eltwise_bwd_common({src, true}, diff_dst, ref_diff_src_case2, dims,
            impl::op_kind::SoftPlusBackprop, "softplus_bw", attrs_data_case2);
}

TEST(Execute, Sqrt) {
    test::vector<float> src {2.0, 1.5, 1.0, 0.5, 0.0, 3.5};
    test::vector<float> ref_dst = sqrt_func(src);

    dnnl::graph::impl::dims dims {1, 2, 3};

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

    dnnl::graph::impl::dims dims {1, 3, 3};

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

    dnnl::graph::impl::dims dims {1, 2, 3};

    test_eltwise_common(src, ref_dst, dims, impl::op_kind::Square, "square");
}

TEST(Execute, Log) {
    test::vector<float> src {2.f, 1.5f, 1.f, 0.5f, 0.8f, 3.5f};
    test::vector<float> ref_dst;
    for (size_t i = 0; i < src.size(); ++i) {
        float temp = static_cast<float>(log(src[i]));
        ref_dst.push_back(temp);
    }

    dnnl::graph::impl::dims dims {1, 2, 3};

    test_eltwise_common(src, ref_dst, dims, impl::op_kind::Log, "log");
}

TEST(Execute, Tanh) {
    test::vector<float> src {-2.0, -1.5, -1.0, -0.5, 0.0, 3.5};
    test::vector<float> ref_dst = tanh_func(src);

    dnnl::graph::impl::dims dims {1, 2, 3};

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

    dnnl::graph::impl::dims dims {1, 3, 3};

    test_eltwise_bwd_common({src, true}, diff_dst, ref_diff_src, dims,
            impl::op_kind::TanhBackprop, "tanh_bw");
    test_eltwise_bwd_common({dst, false}, diff_dst, ref_diff_src, dims,
            impl::op_kind::TanhBackprop, "tanh_bw");
}

struct eltwise_param {
    std::string pass_name;
    test::vector<float> bias;
    test::vector<float> ref_dst;
    impl::op_kind_t op_kind;
    std::string op_name;
    std::vector<std::pair<std::string, float>> attrs;
};

TEST(Execute, ConvBiasEltwise) {
    using dims = dnnl::graph::impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0.0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};

    std::vector<eltwise_param> params1 = {
            eltwise_param {"conv_bias_post_ops_fusion", {-1.0},
                    {2.0, 1.5, 4.0, 0.5}, impl::op_kind::Abs, "Abs", {}},
            eltwise_param {"conv_bias_post_ops_fusion", {-1.0},
                    {static_cast<float>(exp(-2) - 1), 1.5, 4.0, 0.5},
                    impl::op_kind::Elu, "Elu", {{"alpha", 1.f}}},
            eltwise_param {"conv_bias_post_ops_fusion", {-1.0},
                    {-0.04f, 1.5f, 4.0f, 0.5f}, impl::op_kind::LeakyReLU,
                    "LeakyReLU", {{"alpha", 0.02f}}},
            eltwise_param {"conv_bias_post_ops_fusion", {-1.0},
                    {0.0, 1.5, 3.0, 0.5}, impl::op_kind::HardTanh, "HardTanh",
                    {{"min", 0.f}, {"max", 3.f}}},
            eltwise_param {"conv_bias_post_ops_fusion", {-1.0},
                    sigmoid_func({-2.0, 1.5, 4.0, 0.5}), impl::op_kind::Sigmoid,
                    "Sigmoid", {}},
            eltwise_param {"conv_bias_post_ops_fusion", {-1.0},
                    {4.0, 2.25, 16.0, 0.25}, impl::op_kind::Square, "Square",
                    {}},
            eltwise_param {"conv_bias_post_ops_fusion", {-1.0},
                    tanh_func({-2.0, 1.5, 4.0, 0.5}), impl::op_kind::Tanh,
                    "Tanh", {}},
            eltwise_param {"conv_bias_post_ops_fusion", {1.0},
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

        impl::graph_t g(eng.kind());
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

TEST(Execute, ConvBiasAddEltwise) {
    using dims = dnnl::graph::impl::dnnl_impl::dims;

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();
    test::vector<float> src {-3.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            2.5, -1.0, 0.0, 3.0, -2.0, -1.0, 4.0};
    test::vector<float> weight {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    test::vector<float> post_src {-2.0, 1.0, -1.0, 0.0};
    test::vector<float> dst {0.0, 0.0, 0.0, 0.0};

    std::vector<eltwise_param> params2 = {
            eltwise_param {"conv_bias_post_ops_fusion", {-1.0},
                    {static_cast<float>(exp(-4.0) - 1), 2.5, 3.0, 0.5},
                    impl::op_kind::Elu, "Elu", {{"alpha", 1.f}}},
            eltwise_param {"conv_bias_post_ops_fusion", {-1.0},
                    {-0.08f, 2.5f, 3.0f, 0.5f}, impl::op_kind::LeakyReLU,
                    "LeakyReLU", {{"alpha", 0.02f}}},
            eltwise_param {"conv_bias_post_ops_fusion", {3.0},
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

        impl::graph_t g(eng.kind());
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

TEST(Execute, ConvAddEltwise) {
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
            eltwise_param {"conv_post_ops_fusion", {0.0},
                    {static_cast<float>(exp(-3.0) - 1), 3.5, 4.0, 1.5},
                    impl::op_kind::Elu, "Elu", {{"alpha", 1.f}}},
            eltwise_param {"conv_post_ops_fusion", {0.0},
                    {-0.06f, 3.5f, 4.0f, 1.5f}, impl::op_kind::LeakyReLU,
                    "LeakyReLU", {{"alpha", 0.02f}}},
            eltwise_param {"conv_post_ops_fusion", {0.0}, {0.0, 3.5, 4.f, 1.5},
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

        impl::graph_t g(eng.kind());
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

TEST(ExecuteSubgraphFp32, ConvDepthwise) {
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    SKIP_IF(engine.kind() == impl::engine_kind::gpu,
            "Skip for GPU - not supported yet.");

    // N, IC, IH, IW
    std::vector<int64_t> conv_src_shape {4, 4, 4, 4};
    // OC, IC/G, KH, KW
    std::vector<int64_t> conv_wei_shape {4, 4, 1, 1};
    // N, OC, OH, OW
    std::vector<int64_t> conv_dst_shape {4, 4, 4, 4};
    // OC, IC/G, KH, KW
    std::vector<int64_t> dw_wei_shape {4, 1, 3, 3};
    // N, OC, OH, OW
    std::vector<int64_t> dw_dst_shape {4, 4, 2, 2};

    std::string dw_type {"k3s2p1"};

    test::vector<float> src_data(product(conv_src_shape));
    test::vector<float> wei_data(product(conv_wei_shape));
    test::vector<float> dw_wei_data(product(dw_wei_shape));

    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
    std::generate(src_data.begin(), src_data.end(),
            [&]() { return f32_distribution(generator); });
    std::generate(wei_data.begin(), wei_data.end(),
            [&]() { return f32_distribution(generator); });
    std::generate(dw_wei_data.begin(), dw_wei_data.end(),
            [&]() { return f32_distribution(generator); });

    impl::op_t conv {0, impl::op_kind::Convolution, "conv"};
    utils::set_conv_dw_base_op_attr(conv);

    impl::op_t depthwise {1, impl::op_kind::Convolution, "depthwise"};
    utils::set_conv_dw_post_op_attr(depthwise, dw_type);

    impl::logical_tensor_t conv_src = utils::logical_tensor_init(
            0, conv_src_shape, impl::data_type::f32);
    impl::logical_tensor_t conv_wei = utils::logical_tensor_init(
            1, conv_wei_shape, impl::data_type::f32);
    impl::logical_tensor_t conv_dst = utils::logical_tensor_init(
            2, conv_dst_shape, impl::data_type::f32);

    impl::logical_tensor_t dw_wei
            = utils::logical_tensor_init(3, dw_wei_shape, impl::data_type::f32);
    impl::logical_tensor_t dw_dst
            = utils::logical_tensor_init(4, dw_dst_shape, impl::data_type::f32);

    conv.add_input(conv_src);
    conv.add_input(conv_wei);
    conv.add_output(conv_dst);

    depthwise.add_input(conv_dst);
    depthwise.add_input(dw_wei);
    depthwise.add_output(dw_dst);

    impl::graph_t g(engine.kind());
    g.add_op(&conv);
    g.add_op(&depthwise);
    g.build_graph();

    impl::tensor_t conv_src_ts(conv_src, &engine, src_data.data());
    impl::tensor_t conv_wei_ts(conv_wei, &engine, wei_data.data());
    impl::tensor_t dw_wei_ts(dw_wei, &engine, dw_wei_data.data());

    // -------------------------case 1----------------------------------
    test::vector<float> case1_out_data(product(dw_dst_shape));
    impl::tensor_t dw_dst_ts(dw_dst, &engine, case1_out_data.data());

    ASSERT_EQ(run_graph(g, {conv_src_ts, conv_wei_ts, dw_wei_ts}, {dw_dst_ts},
                      engine, strm),
            impl::status::success);

    // -------------------------case 2----------------------------------
    impl::pass::pass_base_ptr apass = get_pass("conv_depthwise_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> lt_ins {
            &conv_src, &conv_wei, &dw_wei};
    std::vector<const impl::logical_tensor_t *> lt_outs {&dw_dst};

    ASSERT_EQ(p.compile(&cp, lt_ins, lt_outs, &engine), impl::status::success);

    test::vector<float> case2_out_data(product(dw_dst_shape));
    impl::tensor_t dw_dst_ts2(dw_dst, &engine, case2_out_data.data());

    cp.execute(&strm, {conv_src_ts, conv_wei_ts, dw_wei_ts}, {dw_dst_ts2});
    strm.wait();

    for (size_t i = 0; i < case1_out_data.size(); ++i) {
        ASSERT_FLOAT_EQ(case1_out_data[i], case2_out_data[i]);
    }
}

TEST(ExecuteSubgraphFp32, Shuffle) {
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

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
        reshape0.set_attr("shape", reshape0_dst_shape);
        reshape0.set_attr("special_zero", false);

        impl::op_t transpose {1, impl::op_kind::StaticTranspose, "transpose"};
        transpose.set_attr("order", order);

        impl::op_t reshape1 {2, impl::op_kind::StaticReshape, "reshape1"};
        reshape1.set_attr("shape", reshape1_dst_shape);
        reshape1.set_attr("special_zero", false);

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

        impl::graph_t agraph(engine.kind());
        agraph.add_op(&reshape0);
        agraph.add_op(&transpose);
        agraph.add_op(&reshape1);
        agraph.build_graph();

        impl::pass::pass_base_ptr apass = get_pass("shuffle_fusion");
        apass->run(agraph);
        ASSERT_EQ(agraph.get_num_partitions(), 1);
        auto part = agraph.get_partitions()[0];

        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> lt_ins {&reshape0_src};
        std::vector<const impl::logical_tensor_t *> lt_outs {&reshape1_dst};

        p.compile(&cp, lt_ins, lt_outs, &engine);

        test::vector<float> dst_data(product(reshape1_dst_shape));
        impl::tensor_t reshape0_src_ts(reshape0_src, &engine, src_data.data());
        impl::tensor_t reshape1_dst_ts(reshape1_dst, &engine, dst_data.data());

        cp.execute(&strm, {reshape0_src_ts}, {reshape1_dst_ts});
        strm.wait();

        for (size_t i = 0; i < expected_data.size(); ++i) {
            ASSERT_FLOAT_EQ(expected_data[i], dst_data[i]);
        }
    }
}

TEST(ExecuteSubgraphFp32, ReciprocalMul) {
    impl::engine_t &eng = get_engine();

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

    impl::graph_t g(eng.kind());
    ASSERT_EQ(g.add_op(&reciprocal_op), impl::status::success);
    ASSERT_EQ(g.add_op(&mul_op), impl::status::success);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("reciprocal_multiply_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];
    impl::partition_t p;
    p.init(part);

    // compile
    std::vector<const impl::logical_tensor_t *> inputs {&src0, &src1};
    std::vector<const impl::logical_tensor_t *> outputs {&mul_dst};

    impl::compiled_partition_t cp(p);
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

    impl::logical_tensor_t lt;
    cp.query_logical_tensor(mul_dst.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src0_ts(src0, &eng, src0_data.data());
    impl::tensor_t src1_ts(src1, &eng, src1_data.data());
    impl::tensor_t dst_ts(lt, &eng, dst_data.data());

    impl::stream_t &strm = get_stream();
    ASSERT_EQ(cp.execute(&strm, {src0_ts, src1_ts}, {dst_ts}),
            impl::status::success);
    strm.wait();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(Execute, Softmax) {
    impl::engine_t &eng = get_engine();

    impl::op_t softmax_op(impl::op_kind::SoftMax);
    softmax_op.set_attr<int64_t>("axis", 0);

    test::vector<float> src_data {0.0, 1.0, 2.0, 0.0, 1.0, 2.0};
    test::vector<float> ref_dst_data {0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {2, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst = utils::logical_tensor_init(
            1, {2, 3}, impl::data_type::f32, impl::layout_type::any);

    softmax_op.add_input(src);
    softmax_op.add_output(dst);

    impl::graph_t g(eng.kind());
    g.add_op(&softmax_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("softmax_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];
    impl::partition_t p;
    p.init(part);

    // compile
    std::vector<const impl::logical_tensor_t *> inputs {&src};
    std::vector<const impl::logical_tensor_t *> outputs {&dst};

    impl::compiled_partition_t cp(p);
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

    impl::logical_tensor_t lt;
    cp.query_logical_tensor(dst.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src, &eng, src_data.data());
    impl::tensor_t dst_ts(lt, &eng, dst_data.data());

    impl::stream_t &strm = get_stream();
    ASSERT_EQ(cp.execute(&strm, {src_ts}, {dst_ts}), impl::status::success);
    strm.wait();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(Execute, SoftmaxWithLastDim) {
    impl::engine_t &eng = get_engine();

    impl::op_t softmax_op(impl::op_kind::SoftMax);
    softmax_op.set_attr<int64_t>("axis", -1);

    test::vector<float> src_data {3.0, 3.0, 1.0, 1.0};
    test::vector<float> ref_dst_data {0.5, 0.5, 0.5, 0.5};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {2, 2}, impl::data_type::f32);
    impl::logical_tensor_t dst = utils::logical_tensor_init(
            1, {2, 2}, impl::data_type::f32, impl::layout_type::any);

    softmax_op.add_input(src);
    softmax_op.add_output(dst);

    impl::graph_t g(eng.kind());
    g.add_op(&softmax_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("softmax_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];
    impl::partition_t p;
    p.init(part);

    // compile
    std::vector<const impl::logical_tensor_t *> inputs {&src};
    std::vector<const impl::logical_tensor_t *> outputs {&dst};

    impl::compiled_partition_t cp(p);
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

    impl::logical_tensor_t lt;
    cp.query_logical_tensor(dst.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src, &eng, src_data.data());
    impl::tensor_t dst_ts(lt, &eng, dst_data.data());

    impl::stream_t &strm = get_stream();
    ASSERT_EQ(cp.execute(&strm, {src_ts}, {dst_ts}), impl::status::success);
    strm.wait();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

void test_softmax_bwd_common(const impl::op_kind_t op_kind,
        test::vector<float> &dst, test::vector<float> &diff_dst,
        test::vector<float> &ref_diff_src, const impl::dims &dims,
        const bool plain_grad) {
    impl::op_t softmax_bwd_op(op_kind);
    impl::engine_t &eng = get_engine();

    softmax_bwd_op.set_attr<int64_t>("axis", 1);

    test::vector<float> diff_src(ref_diff_src.size(), 0.0);

    impl::logical_tensor_t diff_dst_lt;
    // prepare logical tensor
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(0, dims, impl::data_type::f32);
    if (plain_grad) {
        diff_dst_lt = utils::logical_tensor_init(1, dims, impl::data_type::f32);
    } else {
        diff_dst_lt = utils::logical_tensor_init(
                1, dims, impl::data_type::f32, impl::layout_type::any);
    }
    impl::logical_tensor_t diff_src_lt = utils::logical_tensor_init(
            2, impl::data_type::f32, impl::layout_type::any);

    softmax_bwd_op.add_input(diff_dst_lt);
    softmax_bwd_op.add_input(dst_lt);
    softmax_bwd_op.add_output(diff_src_lt);

    impl::graph_t g(eng.kind());
    g.add_op(&softmax_bwd_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = op_kind == impl::op_kind::SoftMaxBackprop
            ? get_pass("softmax_bwd_pass")
            : get_pass("logsoftmax_bwd_pass");

    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];
    impl::partition_t p;
    p.init(part);

    std::vector<const impl::logical_tensor_t *> inputs {&diff_dst_lt, &dst_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&diff_src_lt};

    impl::compiled_partition_t cp(p);
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

    impl::logical_tensor_t q_lt;
    cp.query_logical_tensor(diff_src_lt.id, &q_lt);
    ASSERT_EQ(q_lt.layout_type, impl::layout_type::strided);

    for (auto i = 0; i < dst_lt.ndims; i++)
        ASSERT_EQ(q_lt.dims[i], dst_lt.dims[i]);

    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());
    impl::tensor_t diff_dst_ts(diff_dst_lt, &eng, diff_dst.data());
    impl::tensor_t diff_src_ts(q_lt, &eng, diff_src.data());

    impl::stream_t &strm = get_stream();
    ASSERT_EQ(cp.execute(&strm, {diff_dst_ts, dst_ts}, {diff_src_ts}),
            impl::status::success);
    strm.wait();
    for (size_t i = 0; i < ref_diff_src.size(); ++i) {
        ASSERT_FLOAT_EQ(diff_src[i], ref_diff_src[i]);
    }
}

TEST(Execute, SoftMaxBackward) {
    test::vector<float> dst {0.5, 0.5, 0.5, 0.5};
    test::vector<float> diff_dst {1.0, 5.0, 2.0, 4.0};
    test::vector<float> ref_diff_src {-1, 1, -0.5, 0.5};

    const impl::dims dims {2, 2};

    const bool plain_grad = false;

    test_softmax_bwd_common(impl::op_kind::SoftMaxBackprop, dst, diff_dst,
            ref_diff_src, dims, plain_grad);
}

TEST(Execute, SoftMaxBackwardPlainGrad) {
    test::vector<float> dst {0.5, 0.5, 0.5, 0.5};
    test::vector<float> diff_dst {1.0, 5.0, 2.0, 4.0};
    test::vector<float> ref_diff_src {-1, 1, -0.5, 0.5};

    const impl::dims dims {2, 2};

    const bool plain_grad = true;

    test_softmax_bwd_common(impl::op_kind::SoftMaxBackprop, dst, diff_dst,
            ref_diff_src, dims, plain_grad);
}

void test_softmax_bwd_get_inplace_pair_common(impl::op_kind_t op_kind) {
    impl::engine_t &eng = get_engine();
    impl::op_t softmax_bwd_op(op_kind);

    softmax_bwd_op.set_attr<int64_t>("axis", 1);

    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(0, {2, 2}, impl::data_type::f32);
    impl::logical_tensor_t diff_dst_lt
            = utils::logical_tensor_init(1, {2, 2}, impl::data_type::f32);
    impl::logical_tensor_t diff_src_lt
            = utils::logical_tensor_init(2, {2, 2}, impl::data_type::f32);

    softmax_bwd_op.add_input(diff_dst_lt);
    softmax_bwd_op.add_input(dst_lt);
    softmax_bwd_op.add_output(diff_src_lt);

    impl::graph_t g(eng.kind());
    g.add_op(&softmax_bwd_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = op_kind == impl::op_kind::SoftMaxBackprop
            ? get_pass("softmax_bwd_pass")
            : get_pass("logsoftmax_bwd_pass");
    apass->run(g);

    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    // compile reorder_add partition
    std::vector<const impl::logical_tensor_t *> inputs {&diff_dst_lt, &dst_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&diff_src_lt};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

    std::vector<impl::inplace_pair_t> inplace_pairs = cp.get_inplace_pairs();
    ASSERT_EQ(inplace_pairs.size(), 1);
    ASSERT_EQ(inplace_pairs[0].input, diff_dst_lt.id);
    ASSERT_EQ(inplace_pairs[0].output, diff_src_lt.id);
}

TEST(Compile, SoftMaxBackwardGetInplacePair) {
    test_softmax_bwd_get_inplace_pair_common(impl::op_kind::SoftMaxBackprop);
}

TEST(Execute, LogSoftmax) {
    impl::engine_t &eng = get_engine();

    impl::op_t logsoftmax_op(impl::op_kind::LogSoftmax);
    logsoftmax_op.set_attr<int64_t>("axis", 0);

    test::vector<float> src_data {0.0, 1.0, 2.0, 0.0, 1.0, 2.0};
    test::vector<float> ref_dst_data {-0.6931472f, -0.6931472f, -0.6931472f,
            -0.6931472f, -0.6931472f, -0.6931472f};
    test::vector<float> dst_data(ref_dst_data.size(), 0.0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {2, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst = utils::logical_tensor_init(
            1, {2, 3}, impl::data_type::f32, impl::layout_type::any);

    logsoftmax_op.add_input(src);
    logsoftmax_op.add_output(dst);

    impl::graph_t g(eng.kind());
    g.add_op(&logsoftmax_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("logsoftmax_pass");
    ASSERT_TRUE(apass);
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];
    impl::partition_t p;
    p.init(part);

    // compile
    std::vector<const impl::logical_tensor_t *> inputs {&src};
    std::vector<const impl::logical_tensor_t *> outputs {&dst};

    impl::compiled_partition_t cp(p);
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

    impl::logical_tensor_t lt;
    cp.query_logical_tensor(dst.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src, &eng, src_data.data());
    impl::tensor_t dst_ts(lt, &eng, dst_data.data());

    impl::stream_t &strm = get_stream();
    ASSERT_EQ(cp.execute(&strm, {src_ts}, {dst_ts}), impl::status::success);
    strm.wait();
    for (size_t i = 0; i < ref_dst_data.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_data[i], ref_dst_data[i]);
    }
}

TEST(Execute, LogSoftmaxBackward) {
    test::vector<float> dst {
            -0.6931472f, -0.6931472f, -0.6931472f, -0.6931472f};
    test::vector<float> diff_dst {1.0, 5.0, 2.0, 4.0};
    test::vector<float> ref_diff_src {-2, 2, -1, 1};

    const impl::dims dims {2, 2};

    const bool plain_grad = false;

    test_softmax_bwd_common(impl::op_kind::LogSoftmaxBackprop, dst, diff_dst,
            ref_diff_src, dims, plain_grad);
}

TEST(Execute, LogSoftmaxBackwardPlainGrad) {
    test::vector<float> dst {
            -0.6931472f, -0.6931472f, -0.6931472f, -0.6931472f};
    test::vector<float> diff_dst {1.0, 5.0, 2.0, 4.0};
    test::vector<float> ref_diff_src {-2, 2, -1, 1};

    const impl::dims dims {2, 2};

    const bool plain_grad = true;

    test_softmax_bwd_common(impl::op_kind::LogSoftmaxBackprop, dst, diff_dst,
            ref_diff_src, dims, plain_grad);
}

TEST(Compile, LogSoftmaxBackwardGetInplacePair) {
    test_softmax_bwd_get_inplace_pair_common(impl::op_kind::LogSoftmaxBackprop);
}

TEST(Execute, AvgPoolBackwardExcludePad) {
    using dims = dnnl::graph::impl::dnnl_impl::dims;
    impl::engine_t &eng = get_engine();

    test::vector<float> ref_diff_src {-1.0, 1.5, 1.5, 10.0, 2.0, 4.0, 4.0, 4.0,
            2.0, 4.0, 4.0, 4.0, 12.0, -2.5, -2.5, -3.0};
    test::vector<float> diff_dst {
            -1.0, 3.0, 10.0, 4.0, 16.0, 8.0, 12.0, -5.0, -3.0};
    test::vector<float> diff_src(ref_diff_src.size(), 0.0);

    impl::op_t avg_pool_bwd_op(impl::op_kind::AvgPoolBackprop);
    avg_pool_bwd_op.set_attr<dims>("strides", {2, 2});
    avg_pool_bwd_op.set_attr<dims>("kernel", {2, 2});
    avg_pool_bwd_op.set_attr<dims>("pads_begin", {1, 1});
    avg_pool_bwd_op.set_attr<dims>("pads_end", {1, 1});
    avg_pool_bwd_op.set_attr<bool>("exclude_pad", true);
    avg_pool_bwd_op.set_attr<std::string>("data_format", "NCX");
    avg_pool_bwd_op.set_attr<std::vector<int64_t>>(
            "input_shape", std::vector<int64_t> {1, 1, 4, 4});

    // prepare logical tensor
    impl::logical_tensor_t diff_dst_lt
            = utils::logical_tensor_init(1, {1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t diff_src_lt = utils::logical_tensor_init(
            2, {1, 1, 4, 4}, impl::data_type::f32, impl::layout_type::any);

    avg_pool_bwd_op.add_input(diff_dst_lt);
    avg_pool_bwd_op.add_output(diff_src_lt);
    impl::graph_t g(eng.kind());
    ASSERT_EQ(g.add_op(&avg_pool_bwd_op), impl::status::success);
    g.build_graph();
    impl::pass::pass_base_ptr apass = get_pass("avg_pool_bw_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);
    impl::compiled_partition_t cp(p);
    std::vector<const impl::logical_tensor_t *> inputs {&diff_dst_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&diff_src_lt};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);
    impl::logical_tensor_t lt;
    cp.query_logical_tensor(diff_src_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    impl::tensor_t diff_dst_ts(diff_dst_lt, &eng, diff_dst.data());
    impl::tensor_t diff_src_ts(lt, &eng, diff_src.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {diff_dst_ts}, {diff_src_ts});
    strm.wait();
    for (size_t i = 0; i < diff_src.size(); ++i) {
        ASSERT_FLOAT_EQ(diff_src[i], ref_diff_src[i]);
    }
}

TEST(Execute, AvgPoolBackwardIncludePad) {
    using dims = dnnl::graph::impl::dnnl_impl::dims;
    impl::engine_t &eng = get_engine();

    test::vector<float> ref_diff_src {-0.25, 0.75, 0.75, 2.5, 1.0, 4.0, 4.0,
            2.0, 1.0, 4.0, 4.0, 2.0, 3.0, -1.25, -1.25, -3.0 / 4};
    test::vector<float> diff_dst {
            -1.0, 3.0, 10.0, 4.0, 16.0, 8.0, 12.0, -5.0, -3.0};
    test::vector<float> diff_src(ref_diff_src.size(), 0.0);

    impl::op_t avg_pool_bwd_op(impl::op_kind::AvgPoolBackprop);
    avg_pool_bwd_op.set_attr<dims>("strides", {2, 2});
    avg_pool_bwd_op.set_attr<dims>("kernel", {2, 2});
    avg_pool_bwd_op.set_attr<dims>("pads_begin", {1, 1});
    avg_pool_bwd_op.set_attr<dims>("pads_end", {1, 1});
    avg_pool_bwd_op.set_attr<bool>("exclude_pad", false);
    avg_pool_bwd_op.set_attr<std::string>("data_format", "NCX");
    avg_pool_bwd_op.set_attr<std::vector<int64_t>>(
            "input_shape", std::vector<int64_t> {1, 1, 4, 4});

    // prepare logical tensor
    impl::logical_tensor_t diff_dst_lt
            = utils::logical_tensor_init(1, {1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t diff_src_lt = utils::logical_tensor_init(
            2, {1, 1, 4, 4}, impl::data_type::f32, impl::layout_type::any);

    avg_pool_bwd_op.add_input(diff_dst_lt);
    avg_pool_bwd_op.add_output(diff_src_lt);
    impl::graph_t g(eng.kind());
    ASSERT_EQ(g.add_op(&avg_pool_bwd_op), impl::status::success);
    g.build_graph();
    impl::pass::pass_base_ptr apass = get_pass("avg_pool_bw_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);
    impl::compiled_partition_t cp(p);
    std::vector<const impl::logical_tensor_t *> inputs {&diff_dst_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&diff_src_lt};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);
    impl::logical_tensor_t lt;
    cp.query_logical_tensor(diff_src_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    impl::tensor_t diff_dst_ts(diff_dst_lt, &eng, diff_dst.data());
    impl::tensor_t diff_src_ts(lt, &eng, diff_src.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {diff_dst_ts}, {diff_src_ts});
    strm.wait();
    for (size_t i = 0; i < diff_src.size(); ++i) {
        ASSERT_FLOAT_EQ(diff_src[i], ref_diff_src[i]);
    }
}

TEST(Execute, MaxPoolBackwardWithIncides) {
    using dims = dnnl::graph::impl::dnnl_impl::dims;
    impl::engine_t &eng = get_engine();

    test::vector<float> src {-2.0, -1.5, 2.0, 0.5, -0.5, -1.0, 1.0, 1.5, 2.0,
            3.0, -1.0, 0.0, 1.0, -2.0, -1.0, 4.0};
    test::vector<float> diff_src(src.size(), 0.0);
    test::vector<float> ref_diff_src {0.0, 0.0, 16.0, 0.0, 4.0, 0.0, 0.0, 0.0,
            0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.0};

    test::vector<float> diff_dst {4.0, 16.0, 8.0, 12.0};

    void *indices_data = nullptr;
    test::vector<int32_t> s32_indices;
    test::vector<uint8_t> u8_indices {2, 0, 1, 3};
    if (get_test_engine_kind() == impl::engine_kind::cpu) {
        u8_indices = test::vector<uint8_t> {2, 0, 1, 3};
        indices_data = u8_indices.data();

    } else {
        s32_indices = test::vector<int32_t> {2, 0, 1, 3};
        indices_data = s32_indices.data();
    }

    impl::op_t max_pool_bwd_op(impl::op_kind::MaxPoolBackprop);

    max_pool_bwd_op.set_attr<dims>("strides", dims {2, 2});
    max_pool_bwd_op.set_attr<dims>("kernel", dims {2, 2});
    max_pool_bwd_op.set_attr<dims>("pads_begin", dims {0, 0});
    max_pool_bwd_op.set_attr<dims>("pads_end", dims {0, 0});
    max_pool_bwd_op.set_attr<dims>("dilations", {1, 1});
    max_pool_bwd_op.set_attr<std::string>("data_format", "NCX");

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t diff_src_lt = utils::logical_tensor_init(
            1, {1, 1, 4, 4}, impl::data_type::f32, impl::layout_type::any);
    impl::logical_tensor_t diff_dst_lt
            = utils::logical_tensor_init(3, {1, 1, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t indices_lt;
    if (get_test_engine_kind() == impl::engine_kind::cpu) {
        indices_lt = utils::logical_tensor_init(
                4, {1, 1, 2, 2}, impl::data_type::u8);
    } else {
        indices_lt = utils::logical_tensor_init(
                4, {1, 1, 2, 2}, impl::data_type::s32);
    }

    max_pool_bwd_op.add_input(src_lt);
    max_pool_bwd_op.add_input(diff_dst_lt);
    max_pool_bwd_op.add_input(indices_lt);
    max_pool_bwd_op.add_output(diff_src_lt);
    impl::graph_t g(eng.kind());
    ASSERT_EQ(g.add_op(&max_pool_bwd_op), impl::status::success);
    g.build_graph();
    impl::pass::pass_base_ptr apass = get_pass("max_pool_bw_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);
    impl::compiled_partition_t cp(p);
    std::vector<const impl::logical_tensor_t *> inputs {
            &src_lt, &diff_dst_lt, &indices_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&diff_src_lt};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);
    impl::logical_tensor_t lt;
    cp.query_logical_tensor(diff_src_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t diff_dst_ts(diff_dst_lt, &eng, diff_dst.data());
    impl::tensor_t diff_src_ts(lt, &eng, diff_src.data());
    impl::tensor_t indices_ts(indices_lt, &eng, indices_data);

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src_ts, diff_dst_ts, indices_ts}, {diff_src_ts});
    strm.wait();

    for (size_t i = 0; i < diff_src.size(); ++i) {
        ASSERT_FLOAT_EQ(diff_src[i], ref_diff_src[i]);
    }
}

TEST(Execute, MaxPoolBackwardWithoutIncides) {
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
    max_pool_bwd_op.set_attr<std::string>("data_format", "NCX");

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 4, 4}, impl::data_type::f32);
    impl::logical_tensor_t diff_src_lt = utils::logical_tensor_init(
            1, {1, 1, 4, 4}, impl::data_type::f32, impl::layout_type::any);
    impl::logical_tensor_t diff_dst_lt
            = utils::logical_tensor_init(3, {1, 1, 2, 2}, impl::data_type::f32);

    max_pool_bwd_op.add_input(src_lt);
    max_pool_bwd_op.add_input(diff_dst_lt);
    max_pool_bwd_op.add_output(diff_src_lt);
    impl::graph_t g(eng.kind());
    ASSERT_EQ(g.add_op(&max_pool_bwd_op), impl::status::success);
    g.build_graph();
    impl::pass::pass_base_ptr apass = get_pass("max_pool_bw_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);
    impl::compiled_partition_t cp(p);
    std::vector<const impl::logical_tensor_t *> inputs {&src_lt, &diff_dst_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&diff_src_lt};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);
    impl::logical_tensor_t lt;
    cp.query_logical_tensor(diff_src_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t diff_dst_ts(diff_dst_lt, &eng, diff_dst.data());
    impl::tensor_t diff_src_ts(lt, &eng, diff_src.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src_ts, diff_dst_ts}, {diff_src_ts});
    strm.wait();

    for (size_t i = 0; i < diff_src.size(); ++i) {
        ASSERT_FLOAT_EQ(diff_src[i], ref_diff_src[i]);
    }
}

TEST(Execute, MaxPoolBackwardWithoutIncidesPlainGrad) {
    using dims = dnnl::graph::impl::dnnl_impl::dims;
    impl::engine_t &eng = get_engine();

    impl::op_t max_pool_bwd_op(impl::op_kind::MaxPoolBackprop);

    max_pool_bwd_op.set_attr<dims>("strides", dims {2, 2});
    max_pool_bwd_op.set_attr<dims>("kernel", dims {2, 2});
    max_pool_bwd_op.set_attr<dims>("pads_begin", dims {0, 0});
    max_pool_bwd_op.set_attr<dims>("pads_end", dims {0, 0});
    max_pool_bwd_op.set_attr<dims>("dilations", {1, 1});
    max_pool_bwd_op.set_attr<std::string>("data_format", "NCX");

    dims input_dims = {1, 8, 4, 4};
    dims input_stride = {128, 1, 32, 8};
    dims output_dims = {1, 8, 2, 2};
    // prepare logical tensor
    impl::logical_tensor_t src_lt = utils::logical_tensor_init(
            0, input_dims, input_stride, impl::data_type::f32);
    impl::logical_tensor_t diff_src_lt = utils::logical_tensor_init(
            1, input_dims, impl::data_type::f32, impl::layout_type::any);
    impl::logical_tensor_t diff_dst_lt
            = utils::logical_tensor_init(3, output_dims, impl::data_type::f32);

    max_pool_bwd_op.add_input(src_lt);
    max_pool_bwd_op.add_input(diff_dst_lt);
    max_pool_bwd_op.add_output(diff_src_lt);
    impl::graph_t g(eng.kind());
    ASSERT_EQ(g.add_op(&max_pool_bwd_op), impl::status::success);
    g.build_graph();
    impl::pass::pass_base_ptr apass = get_pass("max_pool_bw_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);
    impl::compiled_partition_t cp(p);
    std::vector<const impl::logical_tensor_t *> inputs {&src_lt, &diff_dst_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&diff_src_lt};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);
    impl::logical_tensor_t lt;
    cp.query_logical_tensor(diff_src_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    test::vector<float> src(product(input_dims), 1);
    test::vector<float> diff_dst(product(output_dims), 1);
    test::vector<float> diff_src(product(input_dims), 1);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t diff_dst_ts(diff_dst_lt, &eng, diff_dst.data());
    impl::tensor_t diff_src_ts(lt, &eng, diff_src.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src_ts, diff_dst_ts}, {diff_src_ts});
    strm.wait();
}

TEST(Execute, LayernormTraining) {
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

    impl::op_t layernorm_op(impl::op_kind::LayerNorm);

    layernorm_op.set_attr<float>("epsilon", 0);

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 3, 2}, impl::data_type::f32);
    impl::logical_tensor_t scale_lt
            = utils::logical_tensor_init(1, {2}, impl::data_type::f32);
    impl::logical_tensor_t shift_lt
            = utils::logical_tensor_init(2, {2}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(3, {1, 3, 2}, impl::data_type::f32);
    impl::logical_tensor_t mean_lt
            = utils::logical_tensor_init(4, {1, 3}, impl::data_type::f32);
    impl::logical_tensor_t variance_lt
            = utils::logical_tensor_init(5, {1, 3}, impl::data_type::f32);

    impl::engine_t &engine = get_engine();
    impl::graph_t g(engine.kind());

    layernorm_op.add_input(src_lt);
    layernorm_op.add_input(scale_lt);
    layernorm_op.add_input(shift_lt);
    layernorm_op.add_output(dst_lt);
    layernorm_op.add_output(mean_lt);
    layernorm_op.add_output(variance_lt);

    ASSERT_EQ(g.add_op(&layernorm_op), impl::status::success);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("ln_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);
    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {
            &src_lt, &scale_lt, &shift_lt};
    std::vector<const impl::logical_tensor_t *> outputs {
            &dst_lt, &mean_lt, &variance_lt};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, &engine), impl::status::success);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t scale_ts(scale_lt, &eng, scale.data());
    impl::tensor_t shift_ts(shift_lt, &eng, shift.data());
    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());
    impl::tensor_t mean_ts(mean_lt, &eng, mean.data());
    impl::tensor_t var_ts(variance_lt, &eng, var.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src_ts, scale_ts, shift_ts}, {dst_ts, mean_ts, var_ts});
    strm.wait();

    for (size_t i = 0; i < ref_dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }

    for (size_t i = 0; i < ref_mean.size(); ++i) {
        ASSERT_FLOAT_EQ(mean[i], ref_mean[i]);
        ASSERT_FLOAT_EQ(var[i], ref_var[i]);
    }
}

TEST(Execute, LayernormInference) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src {2.0, 4.0, 3.0, 5.5, 5.0, 4.0, 1.0, 2.5};
    test::vector<float> scale {1.0, 2.0};
    test::vector<float> shift {0.0, 1.0};
    test::vector<float> ref_dst {-1.0, 3.0, -1.0, 3.0, 1.0, -1.0, -1.0, 3.0};
    test::vector<float> dst(src.size(), 0.0);

    impl::op_t layernorm_op(impl::op_kind::LayerNorm);

    layernorm_op.set_attr<float>("epsilon", 0);
    layernorm_op.set_attr<bool>("keep_stats", false); //inference

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {2, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t scale_lt
            = utils::logical_tensor_init(1, {2}, impl::data_type::f32);
    impl::logical_tensor_t shift_lt
            = utils::logical_tensor_init(2, {2}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(3, {2, 2, 2}, impl::data_type::f32);

    impl::engine_t &engine = get_engine();
    impl::graph_t g(engine.kind());

    layernorm_op.add_input(src_lt);
    layernorm_op.add_input(scale_lt);
    layernorm_op.add_input(shift_lt);
    layernorm_op.add_output(dst_lt);

    ASSERT_EQ(g.add_op(&layernorm_op), impl::status::success);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("ln_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);
    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {
            &src_lt, &scale_lt, &shift_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, &engine), impl::status::success);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t scale_ts(scale_lt, &eng, scale.data());
    impl::tensor_t shift_ts(shift_lt, &eng, shift.data());
    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src_ts, scale_ts, shift_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < ref_dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, LayernormInferenceWithoutScaleShift) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src {2.0, 4.0, 3.0, 5.5, 5.0, 4.0, 1.0, 2.5};
    test::vector<float> ref_dst {-1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0};
    test::vector<float> dst(src.size(), 0.0);

    impl::op_t layernorm_op(impl::op_kind::LayerNorm);

    layernorm_op.set_attr<float>("epsilon", 0);
    layernorm_op.set_attr<bool>("keep_stats", false); //inference
    layernorm_op.set_attr<bool>("use_affine", false);

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {2, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(3, {2, 2, 2}, impl::data_type::f32);

    impl::engine_t &engine = get_engine();
    impl::graph_t g(engine.kind());

    layernorm_op.add_input(src_lt);
    layernorm_op.add_output(dst_lt);

    ASSERT_EQ(g.add_op(&layernorm_op), impl::status::success);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("ln_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);
    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, &engine), impl::status::success);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < ref_dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, LayerNormBackpropFp32) {
    using dims = impl::dnnl_impl::dims;

    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    const impl::dim_t T = 2;
    const impl::dim_t N = 2;
    const impl::dim_t C = 2;
    const dims data_dims {T, N, C};
    const dims diff_data_dims {T, N, C};
    const dims stats_dims {T, N};
    const dims scaleshift_dims {C};

    // inputs data
    test::vector<float> src_data {
            1.0f, -1.0f, 2.0f, 0.0f, 1.0f, -1.0f, 2.0f, 0.0f};
    test::vector<float> diff_dst_data {
            12.0f, -2.0f, 6.0f, -1.0f, 3.0f, -0.5f, 1.5f, -0.25f};
    test::vector<float> diff_dst_data_no_affine {
            1.5f, -0.5f, 0.75f, -0.25f, 0.375f, -0.125f, 0.1875f, -0.0625f};
    test::vector<float> mean_data {0.0f, 1.0f, 0.0f, 1.0f};
    test::vector<float> var_data {3.9375f, 0.9375f, 0.1875f, 3.9375f};
    test::vector<float> scale_data {0.125f, 0.25f};
    test::vector<float> shift_data {0.001953125f, 0.00390625f};

    // outputs data
    test::vector<float> diff_src_data(src_data.size());
    test::vector<float> diff_scale_data(scale_data.size());
    test::vector<float> diff_shift_data(shift_data.size());

    // expected outputs data
    test::vector<float> ref_diff_src_data {
            0.375f, -0.375f, 0.0f, 0.0f, -1.5f, 1.5f, 0.046875f, -0.046875f};
    test::vector<float> ref_diff_scale_data {18.75f, 3.125f};
    test::vector<float> ref_diff_shift_data {22.5f, -3.75f};

    const float epsilon {0.0625f};

    const std::vector<bool> use_affine_flags {true, false};
    for (const auto use_affine : use_affine_flags) {
        impl::op_t ln_bwd_op(impl::op_kind::LayerNormBackprop);
        ln_bwd_op.set_attr<float>("epsilon", epsilon);
        ln_bwd_op.set_attr<bool>("use_affine", use_affine);

        impl::logical_tensor_t src = utils::logical_tensor_init(
                0, data_dims, impl::data_type::f32, impl::layout_type::strided);
        impl::logical_tensor_t diff_dst
                = utils::logical_tensor_init(1, diff_data_dims,
                        impl::data_type::f32, impl::layout_type::strided);
        impl::logical_tensor_t mean = utils::logical_tensor_init(2, stats_dims,
                impl::data_type::f32, impl::layout_type::strided);
        impl::logical_tensor_t var = utils::logical_tensor_init(3, stats_dims,
                impl::data_type::f32, impl::layout_type::strided);
        impl::logical_tensor_t scale
                = utils::logical_tensor_init(4, scaleshift_dims,
                        impl::data_type::f32, impl::layout_type::strided);
        impl::logical_tensor_t shift
                = utils::logical_tensor_init(5, scaleshift_dims,
                        impl::data_type::f32, impl::layout_type::strided);
        impl::logical_tensor_t diff_src
                = utils::logical_tensor_init(6, diff_data_dims,
                        impl::data_type::f32, impl::layout_type::strided);
        impl::logical_tensor_t diff_scale
                = utils::logical_tensor_init(7, scaleshift_dims,
                        impl::data_type::f32, impl::layout_type::strided);
        impl::logical_tensor_t diff_shift
                = utils::logical_tensor_init(8, scaleshift_dims,
                        impl::data_type::f32, impl::layout_type::strided);

        auto &op_factory = get_dnnl_kernel_registry();
        auto ln_bwd_kernel = op_factory.create_kernel(ln_bwd_op);
        ASSERT_TRUE(ln_bwd_kernel);

        ln_bwd_op.add_input(src);
        ln_bwd_op.add_input(diff_dst);
        ln_bwd_op.add_input(mean);
        ln_bwd_op.add_input(var);
        ln_bwd_op.add_output(diff_src);
        if (use_affine) {
            ln_bwd_op.add_input(scale);
            ln_bwd_op.add_input(shift);
            ln_bwd_op.add_output(diff_scale);
            ln_bwd_op.add_output(diff_shift);
        }

        impl::graph_t g(engine.kind());
        g.add_op(&ln_bwd_op);
        g.build_graph();

        impl::pass::pass_base_ptr apass = get_pass("ln_bw_pass");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1);
        auto part = g.get_partitions()[0];

        impl::partition_t p;
        p.init(part);
        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> inputs {
                &src, &diff_dst, &mean, &var};
        std::vector<const impl::logical_tensor_t *> outputs {&diff_src};
        if (use_affine) {
            inputs.push_back(&scale);
            inputs.push_back(&shift);
            outputs.push_back(&diff_scale);
            outputs.push_back(&diff_shift);
        }

        ASSERT_EQ(p.compile(&cp, inputs, outputs, &engine),
                impl::status::success);

        auto inplace_pairs = cp.get_inplace_pairs();
        ASSERT_EQ(inplace_pairs.size(), 1);
        ASSERT_EQ(inplace_pairs[0].input, diff_dst.id);
        ASSERT_EQ(inplace_pairs[0].output, diff_src.id);

        impl::tensor_t src_ts(src, &engine, src_data.data());
        impl::tensor_t diff_dst_ts(diff_dst, &engine,
                use_affine ? diff_dst_data.data()
                           : diff_dst_data_no_affine.data());
        impl::tensor_t mean_ts(mean, &engine, mean_data.data());
        impl::tensor_t var_ts(var, &engine, var_data.data());
        impl::tensor_t scale_ts(scale, &engine, scale_data.data());
        impl::tensor_t shift_ts(shift, &engine, shift_data.data());
        impl::tensor_t diff_src_ts(diff_src, &engine, diff_src_data.data());
        impl::tensor_t diff_scale_ts(
                diff_scale, &engine, diff_scale_data.data());
        impl::tensor_t diff_shift_ts(
                diff_shift, &engine, diff_shift_data.data());

        std::vector<impl::tensor_t> inputs_ts {
                src_ts, diff_dst_ts, mean_ts, var_ts};
        std::vector<impl::tensor_t> outputs_ts {diff_src_ts};
        if (use_affine) {
            inputs_ts.push_back(scale_ts);
            inputs_ts.push_back(shift_ts);
            outputs_ts.push_back(diff_scale_ts);
            outputs_ts.push_back(diff_shift_ts);
        }

        cp.execute(&strm, inputs_ts, outputs_ts);
        strm.wait();

        const float abs_err {0.001f};
        for (size_t i = 0; i < diff_src_data.size(); ++i) {
            ASSERT_NEAR(ref_diff_src_data[i], diff_src_data[i], abs_err);
        }
        if (use_affine) {
            for (size_t i = 0; i < diff_scale_data.size(); ++i) {
                ASSERT_NEAR(
                        ref_diff_scale_data[i], diff_scale_data[i], abs_err);
            }
            for (size_t i = 0; i < diff_shift_data.size(); ++i) {
                ASSERT_NEAR(
                        ref_diff_shift_data[i], diff_shift_data[i], abs_err);
            }
        }
    }
}

TEST(Execute, ReorderData) {
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
    reorder_op.add_input(src_lt);
    reorder_op.add_output(dst_lt);

    impl::graph_t g(engine.kind());
    g.add_op(&reorder_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("reorder_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);
    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &engine), impl::status::success);

    impl::stream_t &stream = get_stream();
    impl::tensor_t src_ts(src_lt, &engine, src.data());
    impl::tensor_t dst_ts(dst_lt, &engine, dst.data());

    cp.execute(&stream, {src_ts}, {dst_ts});
    stream.wait();
    for (size_t i = 0; i < src.size(); ++i) {
        ASSERT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, Int8Reorder) {
    /*
        dequant
        |
        reorder
        |
        quant
    */
    impl::engine_t &engine = get_engine();

    test::vector<uint8_t> int8_src {1, 2, 3, 4, 5, 6};
    test::vector<uint8_t> int8_dst(int8_src.size(), 0);

    // int8 = f32 / scales + zero_points
    test::vector<uint8_t> ref_dst {9, 36, 18, 45, 27, 54};

    impl::graph_t agraph(engine.kind());
    std::vector<int64_t> zps1 = {0};
    std::vector<int64_t> zps2 = {0};
    std::vector<float> scales1 = {3.f};
    std::vector<float> scales2 = {1 / 3.f};
    impl::op_t dequant {0, impl::op_kind::Dequantize, "dequant"};
    dequant.set_attr("scales", scales1);
    dequant.set_attr("zps", zps1);
    impl::op_t reorder {1, impl::op_kind::Reorder, "reorder"};
    impl::op_t quant {2, impl::op_kind::Quantize, "quant"};
    quant.set_attr("scales", scales2);
    quant.set_attr("zps", zps2);

    impl::logical_tensor_t int8_src_lt = utils::logical_tensor_init(
            0, {2, 3}, {3, 1}, impl::data_type::u8);
    impl::logical_tensor_t src_lt = utils::logical_tensor_init(
            1, {2, 3}, {3, 1}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            2, {2, 3}, {1, 2}, impl::data_type::f32);
    impl::logical_tensor_t int8_dst_lt = utils::logical_tensor_init(
            3, {2, 3}, {1, 2}, impl::data_type::u8);
    dequant.add_input(int8_src_lt);
    dequant.add_output(src_lt);
    reorder.add_input(src_lt);
    reorder.add_output(dst_lt);
    quant.add_input(dst_lt);
    quant.add_output(int8_dst_lt);

    ASSERT_EQ(agraph.add_op(&dequant), impl::status::success);
    ASSERT_EQ(agraph.add_op(&reorder), impl::status::success);
    ASSERT_EQ(agraph.add_op(&quant), impl::status::success);

    ASSERT_EQ(agraph.build_graph(), impl::status::success);

    impl::pass::pass_base_ptr apass = get_pass("int8_reorder_fusion");
    apass->run(agraph);
    ASSERT_EQ(agraph.get_num_partitions(), 1);
    auto part = agraph.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);
    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&int8_src_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&int8_dst_lt};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &engine), impl::status::success);

    impl::stream_t &stream = get_stream();
    impl::tensor_t src_ts(int8_src_lt, &engine, int8_src.data());
    impl::tensor_t dst_ts(int8_dst_lt, &engine, int8_dst.data());

    cp.execute(&stream, {src_ts}, {dst_ts});
    stream.wait();
    for (size_t i = 0; i < int8_src.size(); ++i) {
        ASSERT_EQ(int8_dst[i], ref_dst[i]);
    }
}

TEST(Compile, ReorderNegativeInput) {
    impl::engine_t &engine = get_engine();

    impl::op_t reorder_op(impl::op_kind::Reorder);

    impl::logical_tensor_t src_lt = utils::logical_tensor_init(
            0, {2, 3}, {3, 1}, impl::data_type::f32);
    // failure case: different shape (dims)
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {1, 2, 3}, {1, 1, 2}, impl::data_type::f32);

    reorder_op.add_input(src_lt);
    reorder_op.add_output(dst_lt);

    impl::graph_t g(engine.kind());
    g.add_op(&reorder_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("reorder_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);
    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &engine),
            impl::status::invalid_shape);
}

TEST(Execute, ReorderDataBf16) {
    impl::engine_t &engine = get_engine();

    test::vector<float> src {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    test::vector<float> ref_dst {1.0, 4.0, 2.0, 5.0, 3.0, 6.0};
    test::vector<float> dst(src.size(), 10);

    impl::op_t reorder_op(impl::op_kind::Reorder);

    // prepare input/output logical tensor
    // [[1, 2, 3],
    //  [4, 5, 6]]
    impl::logical_tensor_t src_lt = utils::logical_tensor_init(
            0, {2, 3}, {3, 1}, impl::data_type::bf16);
    // [[1, 4],
    //  [2, 5],
    //  [3, 6]]
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {2, 3}, {1, 2}, impl::data_type::bf16);
    reorder_op.add_input(src_lt);
    reorder_op.add_output(dst_lt);

    impl::graph_t g(engine.kind());
    g.add_op(&reorder_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("reorder_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);
    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &engine), impl::status::success);

    impl::stream_t &stream = get_stream();
    impl::tensor_t src_ts(src_lt, &engine, src.data());
    impl::tensor_t dst_ts(dst_lt, &engine, dst.data());

    cp.execute(&stream, {src_ts}, {dst_ts});
    stream.wait();
}

TEST(Execute, Typecast) {
    impl::engine_t &engine = get_engine();

    test::vector<float> f32_val {
            12.5234537f, 0.f, -32.6735142f, -1.f, -2.8765223f, 66.66f};
    test::vector<uint16_t> bf16_val(f32_val.size(), 10);

    impl::op_t typecast_op(impl::op_kind::TypeCast);

    // prepare input/output logical tensor
    // shape (2, 3), stride (3, 1), plain format
    impl::logical_tensor_t f32_lt = utils::logical_tensor_init(
            0, {2, 3}, {3, 1}, impl::data_type::f32);
    impl::logical_tensor_t bf16_lt = utils::logical_tensor_init(
            1, {2, 3}, {3, 1}, impl::data_type::bf16);

    typecast_op.add_input(f32_lt);
    typecast_op.add_output(bf16_lt);

    impl::graph_t g(engine.kind());
    g.add_op(&typecast_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("typecast_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);
    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&f32_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&bf16_lt};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, &engine), impl::status::success);

    impl::stream_t &stream = get_stream();
    impl::tensor_t f32_ts(f32_lt, &engine, f32_val.data());
    impl::tensor_t bf16_ts(bf16_lt, &engine, bf16_val.data());

    // f32 --> bf16
    ASSERT_EQ(cp.execute(&stream, {f32_ts}, {bf16_ts}), impl::status::success);
    stream.wait();
}

TEST(Compile, TypecastNegativeInput) {
    impl::engine_t &engine = get_engine();

    impl::op_t typecast_op(impl::op_kind::TypeCast);
    impl::logical_tensor_t src_lt = utils::logical_tensor_init(
            0, {2, 3}, {3, 1}, impl::data_type::f32);
    // failure case : different shape (dims)
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {1, 2, 3}, {1, 1, 2}, impl::data_type::bf16);
    typecast_op.add_input(src_lt);
    typecast_op.add_output(dst_lt);

    impl::graph_t g(engine.kind());
    g.add_op(&typecast_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("typecast_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);
    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, &engine),
            impl::status::invalid_shape);
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
    impl::engine_t &engine = get_engine();

    std::vector<int64_t> input_dims {1, 3, 3};

    impl::op_t add0 {0, impl::op_kind::Add, "add_0"};
    add0.set_attr<std::string>("auto_broadcast", "none");
    impl::logical_tensor_t input0
            = utils::logical_tensor_init(0, input_dims, impl::data_type::f32);
    impl::logical_tensor_t input1
            = utils::logical_tensor_init(1, input_dims, impl::data_type::f32);
    impl::logical_tensor_t output0
            = utils::logical_tensor_init(2, input_dims, impl::data_type::f32);

    impl::op_t add1 {1, impl::op_kind::Add, "add_1"};
    add1.set_attr<std::string>("auto_broadcast", "none");
    impl::logical_tensor_t input2
            = utils::logical_tensor_init(3, input_dims, impl::data_type::f32);
    impl::logical_tensor_t output1
            = utils::logical_tensor_init(4, input_dims, impl::data_type::f32);

    impl::op_t add2 {2, impl::op_kind::Add, "add_2"};
    add2.set_attr<std::string>("auto_broadcast", "none");
    impl::logical_tensor_t input3
            = utils::logical_tensor_init(5, input_dims, impl::data_type::f32);
    impl::logical_tensor_t output2
            = utils::logical_tensor_init(6, input_dims, impl::data_type::f32);

    impl::op_t add3 {3, impl::op_kind::Add, "add_3"};
    add3.set_attr<std::string>("auto_broadcast", "none");
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

    impl::graph_t agraph(engine.kind());

    ASSERT_EQ(agraph.add_op(&add0), impl::status::success);
    ASSERT_EQ(agraph.add_op(&add1), impl::status::success);
    ASSERT_EQ(agraph.add_op(&add2), impl::status::success);
    ASSERT_EQ(agraph.add_op(&add3), impl::status::success);
    agraph.build_graph();
    impl::pass::pass_base_ptr apass = get_pass("sum_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1);
    auto part = agraph.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {
            &input0, &input1, &input2, &input3, &input4};
    std::vector<const impl::logical_tensor_t *> outputs {&output3};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &engine), impl::status::success);

    test::vector<float> input0_data(product(input_dims), 1);
    test::vector<float> input1_data(product(input_dims), 1);
    test::vector<float> input2_data(product(input_dims), 1);
    test::vector<float> input3_data(product(input_dims), 1);
    test::vector<float> input4_data(product(input_dims), 1);
    test::vector<float> output_data(product(input_dims), 0);

    impl::tensor_t input0_ts(input0, &engine, input0_data.data());
    impl::tensor_t input1_ts(input1, &engine, input1_data.data());
    impl::tensor_t input2_ts(input2, &engine, input2_data.data());
    impl::tensor_t input3_ts(input3, &engine, input3_data.data());
    impl::tensor_t input4_ts(input4, &engine, input4_data.data());
    impl::tensor_t output_ts(output3, &engine, output_data.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {input0_ts, input1_ts, input2_ts, input3_ts, input4_ts},
            {output_ts});
    strm.wait();

    for (const auto v : output_data) {
        ASSERT_FLOAT_EQ(v, 5.f);
    }
}

TEST(Execute, QuantizePerTensor) {
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

    quantize.add_input(src_lt);
    quantize.add_output(dst_lt);

    impl::graph_t g(engine.kind());
    g.add_op(&quantize);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("quant_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);
    std::vector<const impl::logical_tensor_t *> inputs {&src_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &engine), impl::status::success);

    impl::logical_tensor_t lt;
    cp.query_logical_tensor(dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, &engine, src.data());
    impl::tensor_t dst_ts(dst_lt, &engine, dst.data());

    impl::stream_t &strm = get_stream();
    ASSERT_EQ(cp.execute(&strm, {src_ts}, {dst_ts}), impl::status::success);
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, QuantizePerTensorAnyLayout) {
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

    quantize.add_input(src_lt);
    quantize.add_output(dst_lt);

    impl::graph_t g(engine.kind());
    g.add_op(&quantize);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("quant_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);
    std::vector<const impl::logical_tensor_t *> inputs {&src_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &engine), impl::status::success);

    impl::logical_tensor_t lt;
    cp.query_logical_tensor(dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, &engine, src.data());
    impl::tensor_t dst_ts(dst_lt, &engine, dst.data());

    impl::stream_t &strm = get_stream();
    ASSERT_EQ(cp.execute(&strm, {src_ts}, {dst_ts}), impl::status::success);
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, QuantizePerChannelSymmetric) {
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

    quantize.add_input(src_lt);
    quantize.add_output(dst_lt);

    impl::graph_t g(engine.kind());
    g.add_op(&quantize);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("quant_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);
    std::vector<const impl::logical_tensor_t *> inputs {&src_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &engine), impl::status::success);

    impl::logical_tensor_t lt;
    cp.query_logical_tensor(dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, &engine, src.data());
    impl::tensor_t dst_ts(dst_lt, &engine, dst.data());

    impl::stream_t &strm = get_stream();
    ASSERT_EQ(cp.execute(&strm, {src_ts}, {dst_ts}), impl::status::success);
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, TypecastQuantize) {
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();
    if (engine.kind() == impl::engine_kind::gpu) return;

    std::vector<int64_t> src_shape = {1, 8, 16};
    test::vector<float> src_data(product(src_shape));
    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
    std::generate(src_data.begin(), src_data.end(),
            [&]() { return f32_distribution(generator); });
    impl::op_t typecast(0, impl::op_kind::TypeCast, "typecast");
    impl::op_t quantize(1, impl::op_kind::Quantize, "quantize");
    quantize.set_attr<std::vector<float>>("scales", {0.1f});
    quantize.set_attr<std::vector<int64_t>>("zps", {10});
    quantize.set_attr<std::string>("qtype", "per_tensor");
    quantize.set_attr<int64_t>("axis", 0);

    impl::logical_tensor_t src_bf16
            = utils::logical_tensor_init(0, src_shape, impl::data_type::bf16);
    impl::logical_tensor_t src_f32
            = utils::logical_tensor_init(1, src_shape, impl::data_type::f32);
    impl::logical_tensor_t dst_int8
            = utils::logical_tensor_init(2, src_shape, impl::data_type::u8);

    typecast.add_input(src_bf16);
    typecast.add_output(src_f32);

    quantize.add_input(src_f32);
    quantize.add_output(dst_int8);

    impl::graph_t g(engine.kind());
    ASSERT_EQ(g.add_op(&typecast), impl::status::success);
    ASSERT_EQ(g.add_op(&quantize), impl::status::success);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("typecast_quantize_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> lt_ins {&src_bf16};
    std::vector<const impl::logical_tensor_t *> lt_outs {&dst_int8};

    p.compile(&cp, lt_ins, lt_outs, &engine);

    test::vector<int8_t> dst_data(product(src_shape));
    impl::tensor_t src_ts(src_bf16, &engine, src_data.data());
    impl::tensor_t dst_ts(dst_int8, &engine, dst_data.data());
    cp.execute(&strm, {src_ts}, {dst_ts});
    strm.wait();
}

TEST(Execute, DequantizePerTensor) {
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

    dequantize.add_input(src_lt);
    dequantize.add_output(dst_lt);

    impl::graph_t g(engine.kind());
    g.add_op(&dequantize);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("dequant_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);
    std::vector<const impl::logical_tensor_t *> inputs {&src_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &engine), impl::status::success);

    impl::logical_tensor_t lt;
    cp.query_logical_tensor(dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, &engine, src.data());
    impl::tensor_t dst_ts(dst_lt, &engine, dst.data());

    impl::stream_t &strm = get_stream();
    ASSERT_EQ(cp.execute(&strm, {src_ts}, {dst_ts}), impl::status::success);
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, DequantizePerTensorAnyLayout) {
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

    dequantize.add_input(src_lt);
    dequantize.add_output(dst_lt);

    impl::graph_t g(engine.kind());
    g.add_op(&dequantize);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("dequant_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);
    std::vector<const impl::logical_tensor_t *> inputs {&src_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &engine), impl::status::success);

    impl::logical_tensor_t lt;
    cp.query_logical_tensor(dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, &engine, src.data());
    impl::tensor_t dst_ts(dst_lt, &engine, dst.data());

    impl::stream_t &strm = get_stream();
    ASSERT_EQ(cp.execute(&strm, {src_ts}, {dst_ts}), impl::status::success);
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, DequantizePerChannelSymmetric) {
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

    dequantize.add_input(src_lt);
    dequantize.add_output(dst_lt);

    impl::graph_t g(engine.kind());
    g.add_op(&dequantize);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("dequant_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);
    std::vector<const impl::logical_tensor_t *> inputs {&src_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &engine), impl::status::success);

    impl::logical_tensor_t lt;
    cp.query_logical_tensor(dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, &engine, src.data());
    impl::tensor_t dst_ts(dst_lt, &engine, dst.data());

    impl::stream_t &strm = get_stream();
    ASSERT_EQ(cp.execute(&strm, {src_ts}, {dst_ts}), impl::status::success);
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
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

struct dnnl_graph_test_reduce_params {
    test::vector<float> ref_dst_data;
    impl::op_kind_t op_kind;
};

class test_reduce_compile
    : public ::testing::TestWithParam<dnnl_graph_test_reduce_params> {
public:
    void TestReduce() {
        const auto apply_keep_dims_attr
                = [](const std::vector<int64_t> &shape,
                          const std::vector<int64_t> &axes,
                          const bool keep_dims) {
                      if (keep_dims) return shape;
                      std::vector<size_t> excluded_axes;
                      for (const auto axis : axes) {
                          excluded_axes.push_back(static_cast<size_t>(
                                  (axis < 0) ? shape.size() + axis : axis));
                      }
                      std::vector<int64_t> new_shape;
                      for (size_t i = 0; i < shape.size(); ++i) {
                          const auto excluded = std::find(excluded_axes.begin(),
                                                        excluded_axes.end(), i)
                                  != excluded_axes.end();
                          if (!excluded) new_shape.push_back(shape[i]);
                      }
                      return new_shape;
                  };

        const auto params = ::testing::TestWithParam<
                dnnl_graph_test_reduce_params>::GetParam();

        impl::engine_t &engine = get_engine();
        impl::stream_t &strm = get_stream();

        std::vector<int64_t> reduce_src_shape {2, 2, 2, 2};
        std::vector<int64_t> reduce_dst_shape {2, 1, 2, 1};
        std::vector<bool> keep_dims_vals {true, false};
        std::vector<int64_t> axes {-3, 3};
        test::vector<float> src_data = {-1.75, -1.5, -1.25, -1.0, -0.75, -0.5,
                -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0};

        for (auto keep_dims : keep_dims_vals) {
            auto new_reduce_dst_shape
                    = apply_keep_dims_attr(reduce_dst_shape, axes, keep_dims);

            impl::op_t reduce {0, params.op_kind, "reduce"};
            reduce.set_attr<std::vector<int64_t>>("axes", axes);
            reduce.set_attr<bool>("keep_dims", keep_dims);

            impl::logical_tensor_t reduce_src = utils::logical_tensor_init(
                    0, reduce_src_shape, impl::data_type::f32);
            impl::logical_tensor_t reduce_dst = utils::logical_tensor_init(
                    1, new_reduce_dst_shape, impl::data_type::f32);

            reduce.add_input(reduce_src);
            reduce.add_output(reduce_dst);

            impl::graph_t g(engine.kind());
            g.add_op(&reduce);
            g.build_graph();

            impl::pass::pass_base_ptr apass = get_pass("reduce_pass");
            apass->run(g);
            ASSERT_EQ(g.get_num_partitions(), 1);
            auto part = g.get_partitions()[0];

            impl::partition_t p;
            p.init(part);

            impl::compiled_partition_t cp(p);

            std::vector<const impl::logical_tensor_t *> lt_ins {&reduce_src};
            std::vector<const impl::logical_tensor_t *> lt_outs {&reduce_dst};

            p.compile(&cp, lt_ins, lt_outs, &engine);

            test::vector<float> dst_data(product(new_reduce_dst_shape));
            impl::tensor_t reduce_src_ts(reduce_src, &engine, src_data.data());
            impl::tensor_t reduce_dst_ts(reduce_dst, &engine, dst_data.data());

            cp.execute(&strm, {reduce_src_ts}, {reduce_dst_ts});
            strm.wait();

            for (size_t i = 0; i < dst_data.size(); ++i) {
                ASSERT_FLOAT_EQ(dst_data[i], params.ref_dst_data[i]);
            }
        }
    }
};

TEST_P(test_reduce_compile, TestReduceCompile) {
    TestReduce();
}

INSTANTIATE_TEST_SUITE_P(TestReduceCompile, test_reduce_compile,
        ::testing::Values(
                dnnl_graph_test_reduce_params {
                        {4.5f, 2.5f, 3.5f, 5.5f}, impl::op_kind::ReduceL1},
                dnnl_graph_test_reduce_params {
                        {2.47487378f, 1.62018514f, 2.03100967f, 2.93683505f},
                        impl::op_kind::ReduceL2},
                dnnl_graph_test_reduce_params {
                        {-0.5f, 0.0f, 1.5f, 2.0f}, impl::op_kind::ReduceMax},
                dnnl_graph_test_reduce_params {
                        {-1.125f, -0.625f, 0.875f, 1.375f},
                        impl::op_kind::ReduceMean},
                dnnl_graph_test_reduce_params {{-1.75f, -1.25f, 0.25f, 0.75f},
                        impl::op_kind::ReduceMin},
                dnnl_graph_test_reduce_params {
                        {0.984375f, 0.0f, 0.234375f, 2.625f},
                        impl::op_kind::ReduceProd},
                dnnl_graph_test_reduce_params {
                        {-4.5f, -2.5f, 3.5f, 5.5f}, impl::op_kind::ReduceSum}));

TEST(ExecuteSubgraphFp32, ReduceAdd) {
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    std::vector<int64_t> reduce_src_shape {2, 2, 2, 2};
    std::vector<int64_t> base_reduce_dst_shape {1, 2, 2, 1};
    std::vector<int64_t> base_add_src1_shape {1, 2, 1, 1};
    std::vector<int64_t> axes {0, 3};

    test::vector<float> src_data(product(reduce_src_shape));

    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
    std::generate(src_data.begin(), src_data.end(),
            [&]() { return f32_distribution(generator); });

    std::vector<bool> keep_dims_infos {true, false};
    std::vector<bool> with_sum_infos {true, false};

    const std::vector<impl::op_kind_t> op_infos {impl::op_kind::ReduceL1,
            impl::op_kind::ReduceL2, impl::op_kind::ReduceMax,
            impl::op_kind::ReduceMean, impl::op_kind::ReduceMin,
            impl::op_kind::ReduceProd, impl::op_kind::ReduceSum};

    for_(bool keep_dims : keep_dims_infos)
    for_(bool with_sum : with_sum_infos)
    for (auto &akind : op_infos) {
        std::vector<int64_t> reduce_dst_shape = base_reduce_dst_shape;
        std::vector<int64_t> add_src1_shape = base_add_src1_shape;
        if (with_sum) { add_src1_shape[2] *= 2; }
        if (!keep_dims) {
            reduce_dst_shape.erase(reduce_dst_shape.begin());
            reduce_dst_shape.erase(reduce_dst_shape.end() - 1);
            add_src1_shape.erase(add_src1_shape.begin());
            add_src1_shape.erase(add_src1_shape.end() - 1);
        }

        test::vector<float> src1_data(product(add_src1_shape));
        std::generate(src1_data.begin(), src1_data.end(),
                [&]() { return f32_distribution(generator); });

        impl::op_t reduce {0, akind, "reduce"};
        reduce.set_attr("keep_dims", keep_dims);
        reduce.set_attr("axes", axes);

        impl::op_t add {1, impl::op_kind::Add, "add"};

        impl::logical_tensor_t reduce_src = utils::logical_tensor_init(
                0, reduce_src_shape, impl::data_type::f32);
        impl::logical_tensor_t reduce_dst = utils::logical_tensor_init(
                1, reduce_dst_shape, impl::data_type::f32);

        impl::logical_tensor_t add_src1 = utils::logical_tensor_init(
                2, add_src1_shape, impl::data_type::f32);
        impl::logical_tensor_t add_dst = utils::logical_tensor_init(
                3, reduce_dst_shape, impl::data_type::f32);

        reduce.add_input(reduce_src);
        reduce.add_output(reduce_dst);

        add.add_input(reduce_dst);
        add.add_input(add_src1);
        add.add_output(add_dst);

        impl::graph_t g(engine.kind());
        g.add_op(&reduce);
        g.add_op(&add);
        g.build_graph();

        impl::tensor_t reduce_src_ts(reduce_src, &engine, src_data.data());
        impl::tensor_t add_src1_ts(add_src1, &engine, src1_data.data());

        // -------------------------case 1----------------------------------
        test::vector<float> case1_out_data(product(reduce_dst_shape));
        impl::tensor_t add_dst_ts(add_dst, &engine, case1_out_data.data());

        ASSERT_EQ(run_graph(g, {reduce_src_ts, add_src1_ts}, {add_dst_ts},
                          engine, strm),
                impl::status::success);

        // -------------------------case 2----------------------------------
        impl::pass::pass_base_ptr apass = get_pass("reduction_post_ops_fusion");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1);
        auto part = g.get_partitions()[0];

        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> lt_ins {
                &reduce_src, &add_src1};
        std::vector<const impl::logical_tensor_t *> lt_outs {&add_dst};

        p.compile(&cp, lt_ins, lt_outs, &engine);

        test::vector<float> case2_out_data(product(reduce_dst_shape));
        impl::tensor_t add_dst_ts2(add_dst, &engine, case2_out_data.data());

        cp.execute(&strm, {reduce_src_ts, add_src1_ts}, {add_dst_ts2});
        strm.wait();

        for (size_t i = 0; i < case1_out_data.size(); ++i) {
            ASSERT_FLOAT_EQ(case1_out_data[i], case2_out_data[i]);
        }
    }
}

TEST(ExecuteSubgraphFp32, ReduceRelu) {
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    std::vector<int64_t> reduce_src_shape {2, 2, 2, 2};
    std::vector<int64_t> base_reduce_dst_shape {1, 2, 2, 1};
    std::vector<int64_t> axes {0, 3};

    test::vector<float> src_data(product(reduce_src_shape));

    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
    std::generate(src_data.begin(), src_data.end(),
            [&]() { return f32_distribution(generator); });

    std::vector<bool> keep_dims_infos {true, false};
    const std::vector<impl::op_kind_t> op_infos {impl::op_kind::ReduceL1,
            impl::op_kind::ReduceL2, impl::op_kind::ReduceMax,
            impl::op_kind::ReduceMean, impl::op_kind::ReduceMin,
            impl::op_kind::ReduceProd, impl::op_kind::ReduceSum};

    for_(bool keep_dims : keep_dims_infos)
    for (auto &akind : op_infos) {
        std::vector<int64_t> reduce_dst_shape = base_reduce_dst_shape;
        if (!keep_dims) {
            reduce_dst_shape.erase(reduce_dst_shape.begin());
            reduce_dst_shape.erase(reduce_dst_shape.end() - 1);
        }

        impl::op_t reduce {0, akind, "reduce"};
        reduce.set_attr("keep_dims", keep_dims);
        reduce.set_attr("axes", axes);

        impl::op_t relu {1, impl::op_kind::ReLU, "relu"};

        impl::logical_tensor_t reduce_src = utils::logical_tensor_init(
                0, reduce_src_shape, impl::data_type::f32);
        impl::logical_tensor_t reduce_dst = utils::logical_tensor_init(
                1, reduce_dst_shape, impl::data_type::f32);

        impl::logical_tensor_t relu_dst = utils::logical_tensor_init(
                3, reduce_dst_shape, impl::data_type::f32);

        reduce.add_input(reduce_src);
        reduce.add_output(reduce_dst);

        relu.add_input(reduce_dst);
        relu.add_output(relu_dst);

        impl::graph_t g(engine.kind());
        g.add_op(&reduce);
        g.add_op(&relu);
        g.build_graph();

        impl::tensor_t reduce_src_ts(reduce_src, &engine, src_data.data());

        // -------------------------case 1----------------------------------
        test::vector<float> case1_out_data(product(reduce_dst_shape));
        impl::tensor_t relu_dst_ts(relu_dst, &engine, case1_out_data.data());

        ASSERT_EQ(run_graph(g, {reduce_src_ts}, {relu_dst_ts}, engine, strm),
                impl::status::success);

        // -------------------------case 2----------------------------------
        impl::pass::pass_base_ptr apass = get_pass("reduction_post_ops_fusion");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1);
        auto part = g.get_partitions()[0];

        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> lt_ins {&reduce_src};
        std::vector<const impl::logical_tensor_t *> lt_outs {&relu_dst};

        p.compile(&cp, lt_ins, lt_outs, &engine);

        test::vector<float> case2_out_data(product(reduce_dst_shape));
        impl::tensor_t relu_dst_ts2(relu_dst, &engine, case2_out_data.data());

        cp.execute(&strm, {reduce_src_ts}, {relu_dst_ts2});
        strm.wait();

        for (size_t i = 0; i < case1_out_data.size(); ++i) {
            ASSERT_FLOAT_EQ(case1_out_data[i], case2_out_data[i]);
        }
    }
}

TEST(ExecuteSubgraphFp32, ReduceSwish) {
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    std::vector<int64_t> reduce_src_shape {2, 2, 2, 2};
    std::vector<int64_t> base_reduce_dst_shape {1, 2, 2, 1};
    std::vector<int64_t> axes {0, 3};

    test::vector<float> src_data(product(reduce_src_shape));

    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
    std::generate(src_data.begin(), src_data.end(),
            [&]() { return f32_distribution(generator); });

    std::vector<bool> keep_dims_infos {true, false};
    const std::vector<impl::op_kind_t> op_infos {impl::op_kind::ReduceL1,
            impl::op_kind::ReduceL2, impl::op_kind::ReduceMax,
            impl::op_kind::ReduceMean, impl::op_kind::ReduceMin,
            impl::op_kind::ReduceProd, impl::op_kind::ReduceSum};

    for_(bool keep_dims : keep_dims_infos)
    for (auto &akind : op_infos) {
        std::vector<int64_t> reduce_dst_shape = base_reduce_dst_shape;
        if (!keep_dims) {
            reduce_dst_shape.erase(reduce_dst_shape.begin());
            reduce_dst_shape.erase(reduce_dst_shape.end() - 1);
        }

        impl::op_t reduce {0, akind, "reduce"};
        reduce.set_attr("keep_dims", keep_dims);
        reduce.set_attr("axes", axes);

        impl::op_t sigmoid {1, impl::op_kind::Sigmoid, "sigmoid"};
        impl::op_t multiply {2, impl::op_kind::Multiply, "multiply"};

        impl::logical_tensor_t reduce_src = utils::logical_tensor_init(
                0, reduce_src_shape, impl::data_type::f32);
        impl::logical_tensor_t reduce_dst = utils::logical_tensor_init(
                1, reduce_dst_shape, impl::data_type::f32);
        impl::logical_tensor_t sigmoid_dst = utils::logical_tensor_init(
                2, reduce_dst_shape, impl::data_type::f32);
        impl::logical_tensor_t mul_dst = utils::logical_tensor_init(
                3, reduce_dst_shape, impl::data_type::f32);

        reduce.add_input(reduce_src);
        reduce.add_output(reduce_dst);
        sigmoid.add_input(reduce_dst);
        sigmoid.add_output(sigmoid_dst);
        multiply.add_input(reduce_dst);
        multiply.add_input(sigmoid_dst);
        multiply.add_output(mul_dst);

        impl::graph_t g(engine.kind());
        g.add_op(&reduce);
        g.add_op(&sigmoid);
        g.add_op(&multiply);
        g.build_graph();

        impl::tensor_t reduce_src_ts(reduce_src, &engine, src_data.data());

        // -------------------------case 1----------------------------------
        test::vector<float> case1_out_data(product(reduce_dst_shape));
        impl::tensor_t mul_dst_ts(mul_dst, &engine, case1_out_data.data());

        ASSERT_EQ(run_graph(g, {reduce_src_ts}, {mul_dst_ts}, engine, strm),
                impl::status::success);

        // -------------------------case 2----------------------------------
        impl::pass::pass_base_ptr apass = get_pass("reduction_post_ops_fusion");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1);
        auto part = g.get_partitions()[0];

        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> lt_ins {&reduce_src};
        std::vector<const impl::logical_tensor_t *> lt_outs {&mul_dst};

        p.compile(&cp, lt_ins, lt_outs, &engine);

        test::vector<float> case2_out_data(product(reduce_dst_shape));
        impl::tensor_t mul_dst_ts2(mul_dst, &engine, case2_out_data.data());

        cp.execute(&strm, {reduce_src_ts}, {mul_dst_ts2});
        strm.wait();

        for (size_t i = 0; i < case1_out_data.size(); ++i) {
            ASSERT_FLOAT_EQ(case1_out_data[i], case2_out_data[i]);
        }
    }
}

TEST(ExecuteSubgraphFp32, BinarySwish) {
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    std::vector<int64_t> binary_src_shape {2, 2, 2, 2};
    std::vector<int64_t> binary_dst_shape {2, 2, 2, 2};

    test::vector<float> src0_data(product(binary_src_shape));
    test::vector<float> src1_data(product(binary_src_shape));

    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> f32_distribution(0.0f, 1.0f);
    std::generate(src0_data.begin(), src0_data.end(),
            [&]() { return f32_distribution(generator); });
    std::generate(src1_data.begin(), src1_data.end(),
            [&]() { return f32_distribution(generator); });

    const std::vector<impl::op_kind_t> op_infos {impl::op_kind::Add,
            impl::op_kind::Divide, impl::op_kind::Maximum,
            impl::op_kind::Minimum, impl::op_kind::Multiply,
            impl::op_kind::Subtract};

    for (auto &akind : op_infos) {
        impl::op_t binary {0, akind, "binary"};

        impl::op_t sigmoid {1, impl::op_kind::Sigmoid, "sigmoid"};
        impl::op_t multiply {2, impl::op_kind::Multiply, "multiply"};

        impl::logical_tensor_t binary_src0 = utils::logical_tensor_init(
                0, binary_src_shape, impl::data_type::f32);

        impl::logical_tensor_t binary_src1 = utils::logical_tensor_init(
                1, binary_src_shape, impl::data_type::f32);

        impl::logical_tensor_t binary_dst = utils::logical_tensor_init(
                2, binary_dst_shape, impl::data_type::f32);
        impl::logical_tensor_t sigmoid_dst = utils::logical_tensor_init(
                3, binary_dst_shape, impl::data_type::f32);
        impl::logical_tensor_t mul_dst = utils::logical_tensor_init(
                4, binary_dst_shape, impl::data_type::f32);

        binary.add_input(binary_src0);
        binary.add_input(binary_src1);
        binary.add_output(binary_dst);
        sigmoid.add_input(binary_dst);
        sigmoid.add_output(sigmoid_dst);
        multiply.add_input(binary_dst);
        multiply.add_input(sigmoid_dst);
        multiply.add_output(mul_dst);

        impl::graph_t g(engine.kind());
        g.add_op(&binary);
        g.add_op(&sigmoid);
        g.add_op(&multiply);
        g.build_graph();

        impl::tensor_t binary_src0_ts(binary_src0, &engine, src0_data.data());
        impl::tensor_t binary_src1_ts(binary_src1, &engine, src1_data.data());

        // -------------------------case 1----------------------------------
        test::vector<float> case1_out_data(product(binary_dst_shape));
        impl::tensor_t mul_dst_ts(mul_dst, &engine, case1_out_data.data());

        ASSERT_EQ(run_graph(g, {binary_src0_ts, binary_src1_ts}, {mul_dst_ts},
                          engine, strm),
                impl::status::success);

        // -------------------------case 2----------------------------------
        impl::pass::pass_base_ptr apass = get_pass("binary_post_ops_fusion");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1);
        auto part = g.get_partitions()[0];

        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> lt_ins {
                &binary_src0, &binary_src1};
        std::vector<const impl::logical_tensor_t *> lt_outs {&mul_dst};

        p.compile(&cp, lt_ins, lt_outs, &engine);

        test::vector<float> case2_out_data(product(binary_dst_shape));
        impl::tensor_t mul_dst_ts2(mul_dst, &engine, case2_out_data.data());

        cp.execute(&strm, {binary_src0_ts, binary_src1_ts}, {mul_dst_ts2});
        strm.wait();

        for (size_t i = 0; i < case1_out_data.size(); ++i) {
            ASSERT_FLOAT_EQ(case1_out_data[i], case2_out_data[i]);
        }
    }
}

TEST(ExecuteSubgraphInt8, Conv1dConv2dConv3d) {
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
    for_(const auto with_bias : with_biases)
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

        impl::graph_t g(engine.kind());
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
        impl::pass::pass_base_ptr apass = get_pass("int8_conv_post_ops_fusion");
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

        ASSERT_EQ(p.compile(&cp, lt_ins, lt_outs, &engine),
                impl::status::success);

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

TEST(ExecuteSubgraphInt8, ConvTranspose1d2d3d) {
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
    for_(const auto with_bias : with_biases)
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

        impl::graph_t g(engine.kind());
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
        impl::pass::pass_base_ptr apass
                = get_pass("int8_convtranspose_post_ops_fusion");
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

TEST(ExecuteSubgraphInt8, ConvTranspose1d2d3dEltwise) {
    using dims = impl::dnnl_impl::dims;

    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    const std::vector<dnnl_graph_op_kind_t> eltwise_kinds = {impl::op_kind::Abs,
            impl::op_kind::Elu, impl::op_kind::Exp, impl::op_kind::GELU,
            impl::op_kind::HardTanh, impl::op_kind::HardSwish,
            impl::op_kind::Log, impl::op_kind::ReLU, impl::op_kind::Round,
            impl::op_kind::Sigmoid, impl::op_kind::Tanh};

    std::vector<size_t> nds = {1, 2, 3};
    std::vector<int64_t> groups = {1, 4};
    std::vector<bool> with_biases = {true, false};
    std::vector<std::string> weight_qtypes = {"per_tensor", "per_channel"};
    std::vector<std::string> src_qtypes = {"symmetric", "asymmetric"};

    if (engine.kind() == impl::engine_kind::gpu) return;
    static auto isa = dnnl_get_effective_cpu_isa();

    for_(const auto &eltwise_kind : eltwise_kinds)
    for_(const auto &nd : nds)
    for_(const auto &g : groups)
    for_(const auto with_bias : with_biases)
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
        std::uniform_real_distribution<float> u8_distribution(1.0f, 25.0f);
        std::uniform_real_distribution<float> s8_distribution(-1.0f, 25.0f);
        std::uniform_real_distribution<float> f32_distribution(1.0f, 25.0f);
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
        int64_t zp_src = src_qtype == "symmetric" ? 0 : -4;
        int64_t zp_out = 78;

        size_t scale_size = wei_qtype == "per_tensor" ? 1 : (out_channel / g);
        std::vector<float> scale_wei(scale_size, 1 / 127.f);
        std::vector<int64_t> zp_wei(scale_size, 0);

        impl::op_t dqdata_node(0, impl::op_kind::Dequantize, "dqdata_node");
        SET_Q_DQ_DATA_ATTR(dqdata_node)

        impl::op_t dqweight_node(1, impl::op_kind::Dequantize, "dqweight_node");
        SET_Q_DQ_WEIGHT_ATTR(dqweight_node)

        impl::op_t convtranspose_node(
                2, impl::op_kind::ConvTranspose, "convtranspose_node");
        SET_CONV_ATTR(convtranspose_node, nd)

        impl::op_t eltwise_node(3, eltwise_kind, "eltwise_node");

        if (eltwise_kind == impl::op_kind::Elu) {
            eltwise_node.set_attr<float>("alpha", 1.f);
        } else if (eltwise_kind == impl::op_kind::HardTanh) {
            eltwise_node.set_attr<float>("min", -1.f);
            eltwise_node.set_attr<float>("max", 2.f);
        }

        impl::op_t qout_node(4, impl::op_kind::Quantize, "qout_node");
        SET_Q_DQ_OUT_ATTR(qout_node)

        impl::logical_tensor_t src_u8
                = utils::logical_tensor_init(0, src_shape, impl::data_type::u8);
        impl::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                1, src_shape, impl::data_type::f32);
        impl::logical_tensor_t weight_s8 = utils::logical_tensor_init(
                2, weight_shape, impl::data_type::s8);
        impl::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
                3, weight_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_f32 = utils::logical_tensor_init(
                5, dst_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_eltwise_f32 = utils::logical_tensor_init(
                6, dst_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_s8
                = utils::logical_tensor_init(7, dst_shape, impl::data_type::s8);
        impl::logical_tensor_t bias_f32;
        if (with_bias) {
            bias_f32 = utils::logical_tensor_init(
                    4, bias_shape, impl::data_type::f32);
        }

        dqdata_node.add_input(src_u8);
        dqdata_node.add_output(src_f32_dq);

        dqweight_node.add_input(weight_s8);
        dqweight_node.add_output(weight_f32_dq);

        convtranspose_node.add_input(src_f32_dq);
        convtranspose_node.add_input(weight_f32_dq);
        if (with_bias) convtranspose_node.add_input(bias_f32);
        convtranspose_node.add_output(dst_f32);

        eltwise_node.add_input(dst_f32);
        eltwise_node.add_output(dst_eltwise_f32);

        qout_node.add_input(dst_eltwise_f32);
        qout_node.add_output(dst_s8);

        impl::graph_t graph(engine.kind());
        graph.add_op(&dqdata_node);
        graph.add_op(&dqweight_node);
        graph.add_op(&convtranspose_node);
        graph.add_op(&eltwise_node);
        graph.add_op(&qout_node);
        graph.build_graph();

        impl::tensor_t src_u8_ts(src_u8, &engine, src_u8_data.data());
        impl::tensor_t weight_s8_ts(weight_s8, &engine, weight_s8_data.data());
        impl::tensor_t bias_f32_ts;
        if (with_bias) {
            bias_f32_ts = impl::tensor_t(bias_f32, &engine, bias_data.data());
        }
        impl::tensor_t dst_s8_ts(dst_s8, &engine, case1_out_data.data());
        impl::tensor_t dst_s8_case2_ts(dst_s8, &engine, case2_out_data.data());

        // -------------------------case 1----------------------------------
        ASSERT_EQ(run_graph(graph, {src_u8_ts, weight_s8_ts, bias_f32_ts},
                          {dst_s8_ts}, engine, strm),
                impl::status::success);

        // -------------------------case 2----------------------------------
        impl::pass::pass_base_ptr apass
                = get_pass("int8_convtranspose_post_ops_fusion");
        ASSERT_TRUE(apass != nullptr);
        apass->run(graph);
        ASSERT_EQ(graph.get_num_partitions(), 1);
        auto part = graph.get_partitions()[0];

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

TEST(ExecuteSubgraphInt8, X8X8F32ConvTranspose1d2d3dEltwise) {
    using dims = impl::dnnl_impl::dims;

    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    const std::vector<dnnl_graph_op_kind_t> eltwise_kinds = {impl::op_kind::Abs,
            impl::op_kind::Elu, impl::op_kind::Exp, impl::op_kind::GELU,
            impl::op_kind::HardTanh, impl::op_kind::HardSwish,
            impl::op_kind::Log, impl::op_kind::ReLU, impl::op_kind::Round,
            impl::op_kind::Sigmoid, impl::op_kind::Tanh};

    std::vector<size_t> nds = {1, 2, 3};
    std::vector<int64_t> groups = {1, 4};
    std::vector<bool> with_biases = {true, false};
    std::vector<std::string> weight_qtypes = {"per_tensor", "per_channel"};
    std::vector<std::string> src_qtypes = {"symmetric", "asymmetric"};

    if (engine.kind() == impl::engine_kind::gpu) return;
    static auto isa = dnnl_get_effective_cpu_isa();

    for_(const auto &eltwise_kind : eltwise_kinds)
    for_(const auto &nd : nds)
    for_(const auto &g : groups)
    for_(const auto with_bias : with_biases)
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
        test::vector<float> case1_out_data(product(dst_shape));
        test::vector<float> case2_out_data(product(dst_shape));

        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> u8_distribution(1.0f, 25.0f);
        std::uniform_real_distribution<float> s8_distribution(-1.0f, 25.0f);
        std::uniform_real_distribution<float> f32_distribution(1.0f, 25.0f);
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
        int64_t zp_src = src_qtype == "symmetric" ? 0 : -4;

        size_t scale_size = wei_qtype == "per_tensor" ? 1 : (out_channel / g);
        std::vector<float> scale_wei(scale_size, 1 / 127.f);
        std::vector<int64_t> zp_wei(scale_size, 0);

        impl::op_t dqdata_node(0, impl::op_kind::Dequantize, "dqdata_node");
        SET_Q_DQ_DATA_ATTR(dqdata_node)

        impl::op_t dqweight_node(1, impl::op_kind::Dequantize, "dqweight_node");
        SET_Q_DQ_WEIGHT_ATTR(dqweight_node)

        impl::op_t convtranspose_node(
                2, impl::op_kind::ConvTranspose, "convtranspose_node");
        SET_CONV_ATTR(convtranspose_node, nd)

        impl::op_t eltwise_node(3, eltwise_kind, "eltwise_node");

        if (eltwise_kind == impl::op_kind::Elu) {
            eltwise_node.set_attr<float>("alpha", 1.f);
        } else if (eltwise_kind == impl::op_kind::HardTanh) {
            eltwise_node.set_attr<float>("min", -1.f);
            eltwise_node.set_attr<float>("max", 2.f);
        }

        impl::logical_tensor_t src_u8
                = utils::logical_tensor_init(0, src_shape, impl::data_type::u8);
        impl::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                1, src_shape, impl::data_type::f32);
        impl::logical_tensor_t weight_s8 = utils::logical_tensor_init(
                2, weight_shape, impl::data_type::s8);
        impl::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
                3, weight_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_f32 = utils::logical_tensor_init(
                5, dst_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_eltwise_f32 = utils::logical_tensor_init(
                6, dst_shape, impl::data_type::f32);
        impl::logical_tensor_t bias_f32;
        if (with_bias) {
            bias_f32 = utils::logical_tensor_init(
                    4, bias_shape, impl::data_type::f32);
        }

        dqdata_node.add_input(src_u8);
        dqdata_node.add_output(src_f32_dq);

        dqweight_node.add_input(weight_s8);
        dqweight_node.add_output(weight_f32_dq);

        convtranspose_node.add_input(src_f32_dq);
        convtranspose_node.add_input(weight_f32_dq);
        if (with_bias) convtranspose_node.add_input(bias_f32);
        convtranspose_node.add_output(dst_f32);

        eltwise_node.add_input(dst_f32);
        eltwise_node.add_output(dst_eltwise_f32);

        impl::graph_t graph(engine.kind());
        graph.add_op(&dqdata_node);
        graph.add_op(&dqweight_node);
        graph.add_op(&convtranspose_node);
        graph.add_op(&eltwise_node);
        // graph.add_op(&qout_node);
        graph.build_graph();

        impl::tensor_t src_u8_ts(src_u8, &engine, src_u8_data.data());
        impl::tensor_t weight_s8_ts(weight_s8, &engine, weight_s8_data.data());
        impl::tensor_t bias_f32_ts;
        if (with_bias) {
            bias_f32_ts = impl::tensor_t(bias_f32, &engine, bias_data.data());
        }
        impl::tensor_t dst_f32_ts(
                dst_eltwise_f32, &engine, case1_out_data.data());
        impl::tensor_t dst_f32_case2_ts(
                dst_eltwise_f32, &engine, case2_out_data.data());

        // -------------------------case 1----------------------------------
        ASSERT_EQ(run_graph(graph, {src_u8_ts, weight_s8_ts, bias_f32_ts},
                          {dst_f32_ts}, engine, strm),
                impl::status::success);

        // -------------------------case 2----------------------------------
        impl::pass::pass_base_ptr apass
                = get_pass("int8_convtranspose_post_ops_fusion");
        ASSERT_TRUE(apass != nullptr);
        apass->run(graph);
        ASSERT_EQ(graph.get_num_partitions(), 1);
        auto part = graph.get_partitions()[0];

        // compile
        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> lt_ins;
        if (with_bias)
            lt_ins = {&src_u8, &weight_s8, &bias_f32};
        else
            lt_ins = {&src_u8, &weight_s8};
        std::vector<const impl::logical_tensor_t *> lt_outs {&dst_eltwise_f32};

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

TEST(ExecuteSubgraphInt8, X8X8F32ConvTransposeSwish) {
    using dims = impl::dnnl_impl::dims;

    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    std::vector<std::string> weight_qtypes = {"per_tensor", "per_channel"};
    std::vector<std::string> src_qtypes = {"symmetric", "asymmetric"};

    if (engine.kind() == impl::engine_kind::gpu) return;
    static auto isa = dnnl_get_effective_cpu_isa();

    for_(const auto &src_qtype : src_qtypes)
    for (const auto &wei_qtype : weight_qtypes) {
        if (isa < dnnl_cpu_isa_avx512_core_vnni && src_qtype == "asymmetric")
            continue;

        // prepare data
        int64_t in_channel = 8, out_channel = 8;
        int64_t kernel_size = 3, g = 1;
        std::vector<int64_t> src_shape
                = std::vector<int64_t> {1, in_channel, 12, 12};
        std::vector<int64_t> weight_shape = std::vector<int64_t> {
                out_channel, in_channel, kernel_size, kernel_size};
        std::vector<int64_t> bias_shape {out_channel};
        std::vector<int64_t> dst_shape
                = std::vector<int64_t> {1, out_channel, 14, 14};

        test::vector<uint8_t> src_u8_data(product(src_shape));
        test::vector<int8_t> weight_s8_data(product(weight_shape));
        test::vector<float> case1_out_data(product(dst_shape));
        test::vector<float> case2_out_data(product(dst_shape));

        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> u8_distribution(1.0f, 25.0f);
        std::uniform_real_distribution<float> s8_distribution(-1.0f, 25.0f);
        std::uniform_real_distribution<float> f32_distribution(1.0f, 25.0f);
        std::generate(src_u8_data.begin(), src_u8_data.end(), [&]() {
            return static_cast<uint8_t>(u8_distribution(generator));
        });
        std::generate(weight_s8_data.begin(), weight_s8_data.end(), [&]() {
            return static_cast<int8_t>(s8_distribution(generator));
        });

        float scale_src = 1 / 255.f; // map to 0~255
        int64_t zp_src = src_qtype == "symmetric" ? 0 : -4;

        size_t scale_size = wei_qtype == "per_tensor" ? 1 : out_channel;
        std::vector<float> scale_wei(scale_size, 1 / 127.f);
        std::vector<int64_t> zp_wei(scale_size, 0);

        impl::op_t dqdata_node(0, impl::op_kind::Dequantize, "dqdata_node");
        SET_Q_DQ_DATA_ATTR(dqdata_node)

        impl::op_t dqweight_node(1, impl::op_kind::Dequantize, "dqweight_node");
        SET_Q_DQ_WEIGHT_ATTR(dqweight_node)

        impl::op_t convtranspose_node(
                2, impl::op_kind::ConvTranspose, "convtranspose_node");
        SET_CONV_ATTR(convtranspose_node, 2)

        impl::op_t sigmoid_node(3, impl::op_kind::Sigmoid, "sigmoid_node");
        impl::op_t multiply_node {4, impl::op_kind::Multiply, "multiply_node"};

        impl::logical_tensor_t src_u8
                = utils::logical_tensor_init(0, src_shape, impl::data_type::u8);
        impl::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                1, src_shape, impl::data_type::f32);
        impl::logical_tensor_t weight_s8 = utils::logical_tensor_init(
                2, weight_shape, impl::data_type::s8);
        impl::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
                3, weight_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_f32 = utils::logical_tensor_init(
                5, dst_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_sigmoid_f32 = utils::logical_tensor_init(
                6, dst_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_multiply_f32 = utils::logical_tensor_init(
                7, dst_shape, impl::data_type::f32);

        dqdata_node.add_input(src_u8);
        dqdata_node.add_output(src_f32_dq);

        dqweight_node.add_input(weight_s8);
        dqweight_node.add_output(weight_f32_dq);

        convtranspose_node.add_input(src_f32_dq);
        convtranspose_node.add_input(weight_f32_dq);
        convtranspose_node.add_output(dst_f32);

        sigmoid_node.add_input(dst_f32);
        sigmoid_node.add_output(dst_sigmoid_f32);
        multiply_node.add_input(dst_sigmoid_f32);
        multiply_node.add_input(dst_f32);
        multiply_node.add_output(dst_multiply_f32);

        impl::graph_t graph(engine.kind());
        graph.add_op(&dqdata_node);
        graph.add_op(&dqweight_node);
        graph.add_op(&convtranspose_node);
        graph.add_op(&sigmoid_node);
        graph.add_op(&multiply_node);
        graph.build_graph();

        impl::tensor_t src_u8_ts(src_u8, &engine, src_u8_data.data());
        impl::tensor_t weight_s8_ts(weight_s8, &engine, weight_s8_data.data());
        impl::tensor_t dst_f32_ts(
                dst_multiply_f32, &engine, case1_out_data.data());
        impl::tensor_t dst_f32_case2_ts(
                dst_multiply_f32, &engine, case2_out_data.data());

        // -------------------------case 1----------------------------------
        ASSERT_EQ(run_graph(graph, {src_u8_ts, weight_s8_ts}, {dst_f32_ts},
                          engine, strm),
                impl::status::success);

        // -------------------------case 2----------------------------------
        impl::pass::pass_base_ptr apass
                = get_pass("int8_convtranspose_post_ops_fusion");
        ASSERT_TRUE(apass != nullptr);
        apass->run(graph);
        ASSERT_EQ(graph.get_num_partitions(), 1);
        auto part = graph.get_partitions()[0];

        // compile
        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> lt_ins
                = {&src_u8, &weight_s8};
        std::vector<const impl::logical_tensor_t *> lt_outs {&dst_multiply_f32};

        p.compile(&cp, lt_ins, lt_outs, &engine);
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

TEST(ExecuteSubgraphInt8, ConvTranspose1d2d3dAdd) {
    using dims = impl::dnnl_impl::dims;

    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    std::vector<size_t> nds = {1, 2, 3};
    std::vector<int64_t> groups = {1, 4};
    std::vector<bool> with_biases = {true, false};
    // swap add's two inputs
    std::vector<bool> swaps = {true, false};
    std::vector<bool> with_broadcasts = {true, false};
    std::vector<std::string> weight_qtypes = {"per_tensor", "per_channel"};
    std::vector<std::string> src_qtypes = {"symmetric", "asymmetric"};
    std::vector<std::string> other_qtypes = {"symmetric", "asymmetric"};

    if (engine.kind() == impl::engine_kind::gpu) return;
    static auto isa = dnnl_get_effective_cpu_isa();

    for_(const auto &nd : nds)
    for_(const auto &g : groups)
    for_(const auto with_bias : with_biases)
    for_(const auto swap : swaps)
    for_(const auto with_broadcast : with_broadcasts)
    for_(const auto &src_qtype : src_qtypes)
    for_(const auto &other_qtype : other_qtypes)
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
        std::vector<int64_t> other_shape
                = with_broadcast ? std::vector<int64_t> {1, 1, 1} : dst_shape;

        test::vector<uint8_t> src_u8_data(product(src_shape));
        test::vector<int8_t> weight_s8_data(product(weight_shape));
        test::vector<int8_t> other_s8_data(product(other_shape));
        size_t bias_size = with_bias ? product(bias_shape) : 0;
        test::vector<float> bias_data(bias_size);
        test::vector<int8_t> case1_out_data(product(dst_shape));
        test::vector<int8_t> case2_out_data(product(dst_shape));

        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> u8_distribution(0.0f, 25.0f);
        std::uniform_real_distribution<float> s8_distribution(-1.0f, 25.0f);
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
        int64_t zp_src = src_qtype == "symmetric" ? 0 : -4;
        int64_t zp_other = other_qtype == "symmetric" ? 0 : -4;
        int64_t zp_out = 78;

        size_t scale_size = wei_qtype == "per_tensor" ? 1 : (out_channel / g);
        std::vector<float> scale_wei(scale_size, 1 / 127.f);
        std::vector<int64_t> zp_wei(scale_size, 0);

        impl::op_t dqdata_node(0, impl::op_kind::Dequantize, "dqdata_node");
        SET_Q_DQ_DATA_ATTR(dqdata_node)

        impl::op_t dqweight_node(1, impl::op_kind::Dequantize, "dqweight_node");
        SET_Q_DQ_WEIGHT_ATTR(dqweight_node)

        impl::op_t convtranspose_node(
                2, impl::op_kind::ConvTranspose, "convtranspose_node");
        SET_CONV_ATTR(convtranspose_node, nd)

        impl::op_t dqother_node(3, impl::op_kind::Dequantize, "dqother_node");
        dqother_node.set_attr<std::string>("qtype", "per_tensor");
        dqother_node.set_attr<std::vector<int64_t>>("zps", {zp_other});
        dqother_node.set_attr<std::vector<float>>("scales", {scale_other});
        dqother_node.set_attr<int64_t>("axis", 0);

        impl::op_t add_node(4, impl::op_kind::Add, "add_node");

        impl::op_t qout_node(5, impl::op_kind::Quantize, "qout_node");
        SET_Q_DQ_OUT_ATTR(qout_node)

        auto src_u8
                = utils::logical_tensor_init(0, src_shape, impl::data_type::u8);
        auto src_f32_dq = utils::logical_tensor_init(
                1, src_shape, impl::data_type::f32);
        auto weight_s8 = utils::logical_tensor_init(
                2, weight_shape, impl::data_type::s8);
        auto weight_f32_dq = utils::logical_tensor_init(
                3, weight_shape, impl::data_type::f32);
        auto dst_f32 = utils::logical_tensor_init(
                5, dst_shape, impl::data_type::f32);
        auto other_s8 = utils::logical_tensor_init(
                6, other_shape, impl::data_type::s8);
        auto other_f32_dq = utils::logical_tensor_init(
                7, other_shape, impl::data_type::f32);
        auto dst_add_f32 = utils::logical_tensor_init(
                8, dst_shape, impl::data_type::f32);
        auto dst_s8
                = utils::logical_tensor_init(9, dst_shape, impl::data_type::s8);
        impl::logical_tensor_t bias_f32;
        if (with_bias) {
            bias_f32 = utils::logical_tensor_init(
                    4, bias_shape, impl::data_type::f32);
        }

        dqdata_node.add_input(src_u8);
        dqdata_node.add_output(src_f32_dq);

        dqweight_node.add_input(weight_s8);
        dqweight_node.add_output(weight_f32_dq);

        convtranspose_node.add_input(src_f32_dq);
        convtranspose_node.add_input(weight_f32_dq);
        if (with_bias) convtranspose_node.add_input(bias_f32);
        convtranspose_node.add_output(dst_f32);

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

        qout_node.add_input(dst_add_f32);
        qout_node.add_output(dst_s8);

        impl::graph_t graph(engine.kind());
        graph.add_op(&dqdata_node);
        graph.add_op(&dqweight_node);
        graph.add_op(&convtranspose_node);
        graph.add_op(&dqother_node);
        graph.add_op(&add_node);
        graph.add_op(&qout_node);
        graph.build_graph();

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
        ASSERT_EQ(run_graph(graph,
                          {src_u8_ts, weight_s8_ts, bias_f32_ts, other_s8_ts},
                          {dst_s8_ts}, engine, strm),
                impl::status::success);

        // -------------------------case 2----------------------------------
        impl::pass::pass_base_ptr apass
                = get_pass("int8_convtranspose_post_ops_fusion");
        ASSERT_TRUE(apass != nullptr);
        apass->run(graph);
        ASSERT_EQ(graph.get_num_partitions(), 1);
        auto part = graph.get_partitions()[0];

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

TEST(ExecuteSubgraphInt8, ConvTranspose2dAddGetInplacePair) {
    using dims = impl::dnnl_impl::dims;

    impl::engine_t &engine = get_engine();

    std::vector<size_t> nds = {1, 2, 3};
    std::vector<int64_t> groups = {1, 4};
    std::vector<std::string> weight_qtypes = {"per_tensor", "per_channel"};
    std::vector<std::string> src_qtypes = {"symmetric", "asymmetric"};
    std::vector<std::string> other_qtypes = {"symmetric", "asymmetric"};

    if (engine.kind() == impl::engine_kind::gpu) return;
    static auto isa = dnnl_get_effective_cpu_isa();

    for_(const auto &nd : nds)
    for_(const auto &g : groups)
    for_(const auto &src_qtype : src_qtypes)
    for_(const auto &other_qtype : other_qtypes)
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
        std::vector<int64_t> other_shape = dst_shape;

        test::vector<uint8_t> src_u8_data(product(src_shape));
        test::vector<int8_t> weight_s8_data(product(weight_shape));
        test::vector<int8_t> other_s8_data(product(other_shape));
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
        std::generate(other_s8_data.begin(), other_s8_data.end(), [&]() {
            return static_cast<int8_t>(s8_distribution(generator));
        });

        float scale_src = 1 / 255.f; // map to 0~255
        float scale_other = 1 / 127.f;
        float scale_out = 1;
        int64_t zp_src = src_qtype == "symmetric" ? 0 : 128;
        int64_t zp_other = other_qtype == "symmetric" ? 0 : 128;
        int64_t zp_out = 78;

        size_t scale_size = wei_qtype == "per_tensor" ? 1 : (out_channel / g);
        std::vector<float> scale_wei(scale_size, 1 / 127.f);
        std::vector<int64_t> zp_wei(scale_size, 0);

        impl::op_t dqdata_node(0, impl::op_kind::Dequantize, "dqdata_node");
        SET_Q_DQ_DATA_ATTR(dqdata_node)

        impl::op_t dqweight_node(1, impl::op_kind::Dequantize, "dqweight_node");
        SET_Q_DQ_WEIGHT_ATTR(dqweight_node)

        impl::op_t convtranspose_node(
                2, impl::op_kind::ConvTranspose, "convtranspose_node");
        SET_CONV_ATTR(convtranspose_node, nd)

        impl::op_t dqother_node(3, impl::op_kind::Dequantize, "dqother_node");
        dqother_node.set_attr<std::string>("qtype", "per_tensor");
        dqother_node.set_attr<std::vector<int64_t>>("zps", {zp_other});
        dqother_node.set_attr<std::vector<float>>("scales", {scale_other});
        dqother_node.set_attr<int64_t>("axis", 0);

        impl::op_t add_node(4, impl::op_kind::Add, "add_node");

        impl::op_t qout_node(5, impl::op_kind::Quantize, "qout_node");
        SET_Q_DQ_OUT_ATTR(qout_node)

        impl::op_t dqdata_node2(6, impl::op_kind::Dequantize, "dqdata_node");
        SET_Q_DQ_DATA_ATTR(dqdata_node2)

        impl::op_t dqweight_node2(
                7, impl::op_kind::Dequantize, "dqweight_node");
        SET_Q_DQ_WEIGHT_ATTR(dqweight_node2)

        impl::op_t convtranspose_node2(
                8, impl::op_kind::ConvTranspose, "convtranspose_node");
        SET_CONV_ATTR(convtranspose_node2, nd)

        impl::op_t qout_node2(9, impl::op_kind::Quantize, "qout_node");
        SET_Q_DQ_OUT_ATTR(qout_node2)

        auto src_u8
                = utils::logical_tensor_init(0, src_shape, impl::data_type::u8);
        auto src_f32_dq = utils::logical_tensor_init(
                1, src_shape, impl::data_type::f32);
        auto weight_s8 = utils::logical_tensor_init(
                2, weight_shape, impl::data_type::s8);
        auto weight_f32_dq = utils::logical_tensor_init(
                3, weight_shape, impl::data_type::f32);
        auto dst_f32 = utils::logical_tensor_init(
                4, dst_shape, impl::data_type::f32);
        auto other_f32_dq = utils::logical_tensor_init(
                6, other_shape, impl::data_type::f32);
        auto dst_add_f32 = utils::logical_tensor_init(
                7, dst_shape, impl::data_type::f32);
        auto dst_s8
                = utils::logical_tensor_init(8, dst_shape, impl::data_type::s8);

        auto src_u8_2
                = utils::logical_tensor_init(9, src_shape, impl::data_type::u8);
        auto src_f32_dq_2 = utils::logical_tensor_init(
                10, src_shape, impl::data_type::f32);
        auto weight_s8_2 = utils::logical_tensor_init(
                11, weight_shape, impl::data_type::s8);
        auto weight_f32_dq_2 = utils::logical_tensor_init(
                12, weight_shape, impl::data_type::f32);
        auto dst_f32_2 = utils::logical_tensor_init(
                13, dst_shape, impl::data_type::f32);
        auto dst_s8_2 = utils::logical_tensor_init(
                14, dst_shape, impl::data_type::s8);

        dqdata_node.add_input(src_u8);
        dqdata_node.add_output(src_f32_dq);

        dqweight_node.add_input(weight_s8);
        dqweight_node.add_output(weight_f32_dq);

        convtranspose_node.add_input(src_f32_dq);
        convtranspose_node.add_input(weight_f32_dq);
        convtranspose_node.add_output(dst_f32);

        dqother_node.add_input(dst_s8_2);
        dqother_node.add_output(other_f32_dq);

        add_node.add_input(dst_f32);
        add_node.add_input(other_f32_dq);
        add_node.add_output(dst_add_f32);

        qout_node.add_input(dst_add_f32);
        qout_node.add_output(dst_s8);

        dqdata_node2.add_input(src_u8_2);
        dqdata_node2.add_output(src_f32_dq_2);

        dqweight_node2.add_input(weight_s8_2);
        dqweight_node2.add_output(weight_f32_dq_2);

        convtranspose_node2.add_input(src_f32_dq_2);
        convtranspose_node2.add_input(weight_f32_dq_2);
        convtranspose_node2.add_output(dst_f32_2);

        qout_node2.add_input(dst_f32_2);
        qout_node2.add_output(dst_s8_2);

        impl::graph_t graph(engine.kind());
        graph.add_op(&dqdata_node);
        graph.add_op(&dqweight_node);
        graph.add_op(&convtranspose_node);
        graph.add_op(&dqother_node);
        graph.add_op(&add_node);
        graph.add_op(&qout_node);
        graph.add_op(&dqdata_node2);
        graph.add_op(&dqweight_node2);
        graph.add_op(&convtranspose_node2);
        graph.add_op(&qout_node2);
        graph.build_graph();

        impl::pass::pass_base_ptr apass
                = get_pass("int8_convtranspose_post_ops_fusion");
        apass->run(graph);
        ASSERT_EQ(graph.get_num_partitions(), 2);
        auto part2 = graph.get_partitions()[0]; // int8_convtranspose
        auto part1 = graph.get_partitions()[1]; // int8_convtranspose_add

        // compile
        impl::partition_t p1, p2;
        p1.init(part1);
        p2.init(part2);

        impl::compiled_partition_t cp1(p1);
        impl::compiled_partition_t cp2(p2);

        std::vector<const impl::logical_tensor_t *> lt_ins1 {
                &src_u8_2, &weight_s8_2};

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

TEST(ExecuteSubgraphInt8, Conv2dRelu) {
    using dims = impl::dnnl_impl::dims;

    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    std::vector<int64_t> groups = {1, 4};
    std::vector<bool> with_biases = {true, false};
    std::vector<std::string> weight_qtypes = {"per_tensor", "per_channel"};

    if (engine.kind() == impl::engine_kind::gpu) return;
    static auto isa = dnnl_get_effective_cpu_isa();

    for_(const auto &g : groups)
    for_(const auto with_bias : with_biases)
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

        impl::graph_t g(engine.kind());
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
        impl::pass::pass_base_ptr apass = get_pass("int8_conv_post_ops_fusion");
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

TEST(ExecuteSubgraphInt8, Conv2dLeakyRelu) {
    using dims = impl::dnnl_impl::dims;

    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    std::vector<int64_t> groups = {1, 4};
    std::vector<bool> with_biases = {true, false};
    std::vector<std::string> weight_qtypes = {"per_tensor", "per_channel"};

    if (engine.kind() == impl::engine_kind::gpu) return;
    static auto isa = dnnl_get_effective_cpu_isa();

    for_(const auto &g : groups)
    for_(const auto with_bias : with_biases)
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

        impl::op_t relu_node(5, impl::op_kind::LeakyReLU, "leaky_relu_node");
        relu_node.set_attr<float>("alpha", 0.02f);

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

        impl::graph_t g(engine.kind());
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
        impl::pass::pass_base_ptr apass = get_pass("int8_conv_post_ops_fusion");
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

TEST(ExecuteSubgraphInt8, Conv2dSumRelu) {
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
    for_(const auto with_bias : with_biases)
    for_(const auto swap : swaps)
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

        impl::graph_t g(engine.kind());
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
        impl::pass::pass_base_ptr apass = get_pass("int8_conv_post_ops_fusion");

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

TEST(ExecuteSubgraphInt8, Conv2dSumReluNxc) {
    using dims = impl::dnnl_impl::dims;

    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    if (engine.kind() == impl::engine_kind::gpu) return;
    static auto isa = dnnl_get_effective_cpu_isa();

    std::vector<int64_t> groups = {1, 4};
    std::vector<bool> with_biases = {true, false};
    std::vector<std::string> weight_qtypes = {"per_tensor", "per_channel"};

    for_(const auto &g : groups)
    for_(const auto with_bias : with_biases)
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

        impl::graph_t g(engine.kind());
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
        impl::pass::pass_base_ptr apass = get_pass("int8_conv_post_ops_fusion");

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

TEST(ExecuteSubgraphInt8, Conv1d2d3dX8s8f32) {
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
    for_(const auto with_bias : with_biases)
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
        int64_t zp_src = src_qtype == "symmetric" ? 0 : 128;

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

        impl::graph_t g(engine.kind());
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
        impl::pass::pass_base_ptr apass = get_pass("int8_conv_post_ops_fusion");
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

TEST(ExecuteSubgraphInt8, Conv2dReluX8s8f32) {
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
    for_(const auto with_bias : with_biases)
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
        int64_t zp_src = 0;

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

        impl::graph_t g(engine.kind());
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
        impl::pass::pass_base_ptr apass = get_pass("int8_conv_post_ops_fusion");
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

TEST(ExecuteSubgraphInt8, Conv2dSumReluX8s8f32) {
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
    for_(const auto with_bias : with_biases)
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
        int64_t zp_src = 0;
        int64_t zp_other = 0;

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

        impl::graph_t g(engine.kind());
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
        impl::pass::pass_base_ptr apass = get_pass("int8_conv_post_ops_fusion");

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

TEST(ExecuteSubgraphInt8, Conv2dSumReluNxcX8s8f32) {
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
    for_(const auto with_bias : with_biases)
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
        int64_t zp_src = 0;
        int64_t zp_other = 0;

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

        impl::graph_t g(engine.kind());
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
        impl::pass::pass_base_ptr apass = get_pass("int8_conv_post_ops_fusion");

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

TEST(ExecuteSubgraphInt8, MatmulNdx2d) {
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

        impl::graph_t g(engine.kind());
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
        impl::pass::pass_base_ptr apass
                = get_pass("int8_matmul_post_ops_fusion");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1);
        auto part = g.get_partitions()[0];
        ASSERT_EQ(part->get_ops().size(), 4);

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

TEST(ExecuteSubgraphInt8, MatmulNdx1d) {
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

            impl::graph_t g(engine.kind());
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
            impl::pass::pass_base_ptr apass
                    = get_pass("int8_matmul_post_ops_fusion");
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

TEST(ExecuteSubgraphInt8, MatmulNdx2dWithTranspose) {
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

            impl::graph_t g(engine.kind());
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
                    = get_pass("int8_matmul_post_ops_fusion");
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

TEST(ExecuteSubgraphInt8, MatmulReluFusion) {
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

    impl::graph_t g(engine.kind());
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
    impl::pass::pass_base_ptr apass = get_pass("int8_matmul_post_ops_fusion");
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

TEST(Execute, InterpolateForwardNearest) {
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    test::vector<float> src {-2.0, -1.5, -1.0, -0.5};
    test::vector<float> dst {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    test::vector<float> ref_dst {
            -2.f, -1.5f, -1.5f, -1.f, -0.5f, -0.5f, -1.f, -0.5f, -0.5f};

    impl::op_t op(impl::op_kind::Interpolate);
    op.set_attr<std::string>("mode", "nearest");
    op.set_attr("sizes", std::vector<int64_t> {3, 3});
    op.set_attr<std::string>("coordinate_transformation_mode", "half_pixel");
    op.set_attr<std::string>("data_format", "NCX");
    auto &op_factory = get_dnnl_kernel_registry();
    auto kernel = op_factory.create_kernel(op);
    ASSERT_TRUE(kernel);

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, impl::data_type::f32, impl::layout_type::strided);

    op.add_input(src_lt);
    op.add_output(dst_lt);

    impl::graph_t g(engine.kind());
    g.add_op(&op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("interpolate_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];
    ASSERT_TRUE(part != nullptr);

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);
    std::vector<const impl::logical_tensor_t *> lt_ins {&src_lt};
    std::vector<const impl::logical_tensor_t *> lt_outs {&dst_lt};

    p.compile(&cp, lt_ins, lt_outs, &engine);

    impl::tensor_t src_ts(src_lt, &engine, src.data());
    impl::tensor_t dst_ts(*lt_outs[0], &engine, dst.data());
    cp.execute(&strm, {src_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, InterpolateAddForwardNearest) {
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    test::vector<float> src {-2.0, -1.5, -1.0, -0.5};
    test::vector<float> src1 {
            0.f, 0.5f, 1.f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f, 4.f};
    test::vector<float> dst_add {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    test::vector<float> ref_dst {
            -2.f, -1.f, -0.5f, 0.5f, 1.5f, 2.f, 2.f, 3.f, 3.5f};

    impl::op_t interpolate_node(0, impl::op_kind::Interpolate, "interpolate");
    interpolate_node.set_attr<std::string>("mode", "nearest");
    interpolate_node.set_attr("sizes", std::vector<int64_t> {3, 3});
    interpolate_node.set_attr<std::string>(
            "coordinate_transformation_mode", "half_pixel");
    interpolate_node.set_attr<std::string>("data_format", "NCX");

    impl::op_t add_node(1, impl::op_kind::Add, "add_node");

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, impl::data_type::f32, impl::layout_type::strided);

    impl::logical_tensor_t src1_lt = utils::logical_tensor_init(
            2, {1, 1, 3, 3}, impl::data_type::f32, impl::layout_type::strided);
    impl::logical_tensor_t dst_add_lt = utils::logical_tensor_init(
            3, {1, 1, 3, 3}, impl::data_type::f32, impl::layout_type::strided);

    interpolate_node.add_input(src_lt);
    interpolate_node.add_output(dst_lt);

    add_node.add_input(dst_lt);
    add_node.add_input(src1_lt);
    add_node.add_output(dst_add_lt);

    impl::graph_t g(engine.kind());
    g.add_op(&interpolate_node);
    g.add_op(&add_node);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("interpolate_post_ops_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];
    ASSERT_TRUE(part != nullptr);

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);
    std::vector<const impl::logical_tensor_t *> lt_ins {&src_lt, &src1_lt};
    std::vector<const impl::logical_tensor_t *> lt_outs {&dst_add_lt};

    p.compile(&cp, lt_ins, lt_outs, &engine);

    impl::tensor_t src_ts(src_lt, &engine, src.data());
    impl::tensor_t src1_ts(src1_lt, &engine, src1.data());
    impl::tensor_t dst_add_ts(*lt_outs[0], &engine, dst_add.data());
    cp.execute(&strm, {src_ts, src1_ts}, {dst_add_ts});
    strm.wait();

    for (size_t i = 0; i < dst_add.size(); ++i) {
        ASSERT_FLOAT_EQ(dst_add[i], ref_dst[i]);
    }
}

TEST(Execute, InterpolateSwish) {
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    test::vector<float> src {-2.0, -1.5, -1.0, -0.5};
    test::vector<float> dst_mul {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};

    impl::op_t interpolate_node(0, impl::op_kind::Interpolate, "interpolate");
    interpolate_node.set_attr<std::string>("mode", "nearest");
    interpolate_node.set_attr("sizes", std::vector<int64_t> {3, 3});
    interpolate_node.set_attr<std::string>(
            "coordinate_transformation_mode", "half_pixel");
    interpolate_node.set_attr<std::string>("data_format", "NCX");

    impl::op_t sigmoid_node(1, impl::op_kind::Sigmoid, "sigmoid_node");
    impl::op_t mul_node(2, impl::op_kind::Multiply, "multiply_node");

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 1, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, impl::data_type::f32, impl::layout_type::strided);

    impl::logical_tensor_t dst_sigmoid_lt = utils::logical_tensor_init(
            2, {1, 1, 3, 3}, impl::data_type::f32, impl::layout_type::strided);
    impl::logical_tensor_t dst_mul_lt = utils::logical_tensor_init(
            3, {1, 1, 3, 3}, impl::data_type::f32, impl::layout_type::strided);

    interpolate_node.add_input(src_lt);
    interpolate_node.add_output(dst_lt);
    sigmoid_node.add_input(dst_lt);
    sigmoid_node.add_output(dst_sigmoid_lt);
    mul_node.add_input(dst_sigmoid_lt);
    mul_node.add_input(dst_lt);
    mul_node.add_output(dst_mul_lt);

    impl::graph_t g(engine.kind());
    ASSERT_EQ(g.add_op(&interpolate_node), impl::status::success);
    ASSERT_EQ(g.add_op(&sigmoid_node), impl::status::success);
    ASSERT_EQ(g.add_op(&mul_node), impl::status::success);
    ASSERT_EQ(g.build_graph(), impl::status::success);
    ASSERT_EQ(g.num_ops(), 3);

    impl::pass::pass_base_ptr apass = get_pass("interpolate_post_ops_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];
    ASSERT_TRUE(part != nullptr);

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);
    std::vector<const impl::logical_tensor_t *> lt_ins {&src_lt};
    std::vector<const impl::logical_tensor_t *> lt_outs {&dst_mul_lt};

    p.compile(&cp, lt_ins, lt_outs, &engine);

    impl::tensor_t src_ts(src_lt, &engine, src.data());
    impl::tensor_t dst_mul_ts(*lt_outs[0], &engine, dst_mul.data());
    cp.execute(&strm, {src_ts}, {dst_mul_ts});
    strm.wait();
}

TEST(Execute, InterpolatePostOps) {
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    const std::vector<impl::op_kind_t> supported_post_ops = {impl::op_kind::Abs,
            impl::op_kind::Clamp, impl::op_kind::Elu, impl::op_kind::Exp,
            impl::op_kind::GELU, impl::op_kind::HardTanh,
            impl::op_kind::HardSwish, impl::op_kind::Log,
            impl::op_kind::Sigmoid, impl::op_kind::SoftPlus, impl::op_kind::Pow,
            impl::op_kind::ReLU, impl::op_kind::Round, impl::op_kind::Sqrt,
            impl::op_kind::Square, impl::op_kind::Tanh, impl::op_kind::Add,
            impl::op_kind::Multiply, impl::op_kind::Maximum,
            impl::op_kind::Minimum, impl::op_kind::Divide,
            impl::op_kind::Subtract};
    const std::vector<impl::op_kind_t> two_inputs_ops {impl::op_kind::Multiply,
            impl::op_kind::Add, impl::op_kind::Maximum, impl::op_kind::Minimum,
            impl::op_kind::Divide, impl::op_kind::Subtract, impl::op_kind::Pow};

    for (const auto &post_op_kind : supported_post_ops) {
        test::vector<float> src {-2.0, -1.5, -1.0, -0.5};
        test::vector<float> src1 {
                0.f, 0.5f, 1.f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f, 4.f};
        test::vector<float> dst_add {
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};

        impl::op_t interpolate_node(
                0, impl::op_kind::Interpolate, "interpolate");
        interpolate_node.set_attr<std::string>("mode", "nearest");
        interpolate_node.set_attr("sizes", std::vector<int64_t> {3, 3});
        interpolate_node.set_attr<std::string>(
                "coordinate_transformation_mode", "half_pixel");
        interpolate_node.set_attr<std::string>("data_format", "NCX");

        impl::op_t post_node(1, post_op_kind, "post_op_node");
        if (post_op_kind == impl::op_kind::Elu) {
            post_node.set_attr<float>("alpha", 1.0f);
        } else if (post_op_kind == impl::op_kind::Clamp) {
            post_node.set_attr<float>("min", 1.0f);
            post_node.set_attr<float>("max", 3.0f);
        } else if (post_op_kind == impl::op_kind::HardTanh) {
            post_node.set_attr<float>("min", 1.0f);
            post_node.set_attr<float>("max", 3.0f);
        }

        impl::logical_tensor_t src_lt = utils::logical_tensor_init(
                0, {1, 1, 2, 2}, impl::data_type::f32);
        impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
                1, impl::data_type::f32, impl::layout_type::strided);

        impl::logical_tensor_t src1_lt = utils::logical_tensor_init(2,
                {1, 1, 3, 3}, impl::data_type::f32, impl::layout_type::strided);
        impl::logical_tensor_t dst_post_lt = utils::logical_tensor_init(3,
                {1, 1, 3, 3}, impl::data_type::f32, impl::layout_type::strided);

        interpolate_node.add_input(src_lt);
        interpolate_node.add_output(dst_lt);

        post_node.add_input(dst_lt);
        if (std::find(
                    two_inputs_ops.begin(), two_inputs_ops.end(), post_op_kind)
                != two_inputs_ops.end()) {
            post_node.add_input(src1_lt);
        }
        post_node.add_output(dst_post_lt);

        impl::graph_t g(engine.kind());
        g.add_op(&interpolate_node);
        g.add_op(&post_node);
        g.build_graph();

        impl::pass::pass_base_ptr apass
                = get_pass("interpolate_post_ops_fusion");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1);
        auto part = g.get_partitions()[0];
        ASSERT_TRUE(part != nullptr);

        // compile
        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);
        std::vector<const impl::logical_tensor_t *> lt_ins {&src_lt};
        if (std::find(
                    two_inputs_ops.begin(), two_inputs_ops.end(), post_op_kind)
                != two_inputs_ops.end()) {
            lt_ins.emplace_back(&src1_lt);
        }
        std::vector<const impl::logical_tensor_t *> lt_outs {&dst_post_lt};

        p.compile(&cp, lt_ins, lt_outs, &engine);

        impl::tensor_t src_ts(src_lt, &engine, src.data());
        impl::tensor_t src1_ts(src1_lt, &engine, src1.data());
        impl::tensor_t dst_add_ts(*lt_outs[0], &engine, dst_add.data());
        if (std::find(
                    two_inputs_ops.begin(), two_inputs_ops.end(), post_op_kind)
                != two_inputs_ops.end()) {
            cp.execute(&strm, {src_ts, src1_ts}, {dst_add_ts});
        } else {
            cp.execute(&strm, {src_ts}, {dst_add_ts});
        }

        strm.wait();
    }
}

TEST(Execute, InterpolateForwardLinear) {
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    test::vector<float> src {-2.0, -1.5, -1.0, -0.5};
    test::vector<float> dst {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    test::vector<float> ref_dst {
            -2.f, -1.75f, -1.5f, -1.5f, -1.25f, -1.f, -1.f, -0.75f, -0.5f};

    impl::op_t op(impl::op_kind::Interpolate);
    op.set_attr<std::string>("mode", "linear");
    op.set_attr("sizes", std::vector<int64_t> {3, 3});
    op.set_attr<std::string>("coordinate_transformation_mode", "half_pixel");
    op.set_attr<std::string>("data_format", "NXC");

    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 2, 1}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, impl::data_type::f32, impl::layout_type::strided);

    op.add_input(src_lt);
    op.add_output(dst_lt);

    impl::graph_t g(engine.kind());
    g.add_op(&op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("interpolate_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];
    ASSERT_TRUE(part != nullptr);

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);
    std::vector<const impl::logical_tensor_t *> lt_ins {&src_lt};
    std::vector<const impl::logical_tensor_t *> lt_outs {&dst_lt};

    p.compile(&cp, lt_ins, lt_outs, &engine);

    impl::tensor_t src_ts(src_lt, &engine, src.data());
    impl::tensor_t dst_ts(*lt_outs[0], &engine, dst.data());
    cp.execute(&strm, {src_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, InterpolateBackwardNearest) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src {-2.0, -1.5, -1.0, -0.5};
    test::vector<float> diff_dst {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    test::vector<float> diff_src {0.0, 0.0, 0.0, 0.0};
    test::vector<float> ref_diff_src {0.f, 3.f, 9.f, 24.f};

    impl::op_t op(impl::op_kind::InterpolateBackprop);
    op.set_attr<std::string>("mode", "nearest");
    op.set_attr<std::string>("data_format", "NCX");
    auto &op_factory = get_dnnl_kernel_registry();
    auto kernel = op_factory.create_kernel(op);
    ASSERT_TRUE(kernel);

    impl::logical_tensor_t diff_dst_lt
            = utils::logical_tensor_init(0, {1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(1, {1, 1, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t diff_src_lt = utils::logical_tensor_init(
            2, {1, 1, 2, 2}, impl::data_type::f32, impl::layout_type::strided);

    op.add_input(src_lt);
    op.add_input(diff_dst_lt);
    op.add_output(diff_src_lt);

    impl::graph_t g(eng.kind());
    g.add_op(&op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("interpolate_bwd_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);
    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src_lt, &diff_dst_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&diff_src_lt};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t diff_dst_ts(diff_dst_lt, &eng, diff_dst.data());
    impl::tensor_t diff_src_ts(diff_src_lt, &eng, diff_src.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src_ts, diff_dst_ts}, {diff_src_ts});
    strm.wait();

    for (size_t i = 0; i < diff_src.size(); ++i) {
        ASSERT_FLOAT_EQ(diff_src[i], ref_diff_src[i]);
    }
}

TEST(Execute, InterpolateBackwardLinear) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src {-2.0, -1.5, -1.0, -0.5};
    test::vector<float> diff_dst {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    test::vector<float> diff_src {0.0, 0.0, 0.0, 0.0};
    test::vector<float> ref_diff_src {3.f, 6.f, 12.f, 15.f};

    impl::op_t op(impl::op_kind::InterpolateBackprop);
    op.set_attr<std::string>("mode", "linear");
    op.set_attr<std::string>("data_format", "NCX");
    auto &op_factory = get_dnnl_kernel_registry();
    auto kernel = op_factory.create_kernel(op);
    ASSERT_TRUE(kernel);

    impl::logical_tensor_t diff_dst_lt
            = utils::logical_tensor_init(0, {1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(1, {1, 1, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t diff_src_lt = utils::logical_tensor_init(
            2, {1, 1, 2, 2}, impl::data_type::f32, impl::layout_type::strided);

    op.add_input(src_lt);
    op.add_input(diff_dst_lt);
    op.add_output(diff_src_lt);

    impl::graph_t g(eng.kind());
    g.add_op(&op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("interpolate_bwd_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);
    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src_lt, &diff_dst_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&diff_src_lt};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t diff_dst_ts(diff_dst_lt, &eng, diff_dst.data());
    impl::tensor_t diff_src_ts(diff_src_lt, &eng, diff_src.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src_ts, diff_dst_ts}, {diff_src_ts});
    strm.wait();

    for (size_t i = 0; i < diff_src.size(); ++i) {
        ASSERT_FLOAT_EQ(diff_src[i], ref_diff_src[i]);
    }
}

TEST(ExecuteSubgraphInt8, MatmulBiasSumNdx2d) {
    // skip the test on AArch64 or some older machine without avx support
    SKIP_IF(dnnl_get_effective_cpu_isa() < dnnl_cpu_isa_avx,
            "skip on machine without AVX");
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

        impl::graph_t g(engine.kind());
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
                = get_pass("int8_matmul_post_ops_fusion");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1);
        auto part = g.get_partitions()[0];
        ASSERT_EQ(part->get_ops().size(), 6);

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

TEST(ExecuteSubgraphInt8, MatmulBiasSumNdx2dX8s8f32) {
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
        int64_t zp_src = 0;
        int64_t zp_other = 0;

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

        impl::graph_t g(engine.kind());
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
                = get_pass("int8_matmul_post_ops_fusion");
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

TEST(ExecuteSubgraphInt8, MatmulBiasNdx2dX8s8f32) {
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
        int64_t zp_src = 0;

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

        impl::graph_t g(engine.kind());
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
                = get_pass("int8_matmul_post_ops_fusion");
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

TEST(ExecuteSubgraphInt8, MatmulNdx2dX8s8f32) {
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
        int64_t zp_src = 0;

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

        impl::graph_t g(engine.kind());
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
        impl::pass::pass_base_ptr apass
                = get_pass("int8_matmul_post_ops_fusion");
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

TEST(ExecuteSubgraphInt8, MatmulBiasGeluNdx2dX8s8f32) {
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
        int64_t zp_src = 0;

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

        impl::graph_t g(engine.kind());
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
                = get_pass("int8_matmul_post_ops_fusion");
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

TEST(ExecuteSubgraphInt8, Conv2dSumReluGetInplacePair) {
    using dims = impl::dnnl_impl::dims;

    impl::engine_t &engine = get_engine();

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

        impl::graph_t g(engine.kind());
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

        impl::pass::pass_base_ptr apass = get_pass("int8_conv_post_ops_fusion");

        apass->run(g);

        ASSERT_EQ(g.get_num_partitions(), 2);
        auto part2 = g.get_partitions()[0]; // int8_conv_sum_relu
        auto part1 = g.get_partitions()[1]; // int8_conv

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

struct dnnl_graph_test_eltwise_binary_params {
    impl::op_kind_t eltwise_kind;
    impl::op_kind_t binary_kind;
    std::vector<impl::dim_t> binary_src_shape;
    test::vector<float> binary_src_data;
    bool swap;
};

class EltwiseBinary
    : public ::testing::TestWithParam<dnnl_graph_test_eltwise_binary_params> {
public:
    void TestEltwiseBinary() {
        const auto params = ::testing::TestWithParam<
                dnnl_graph_test_eltwise_binary_params>::GetParam();
        impl::engine_t &eng = get_engine();

        test::vector<float> src {-2.0, -1.5, 1.0, 0.5};
        test::vector<float> binary_src = params.binary_src_data;
        test::vector<float> dst {0.0, 0.0, 0.0, 0.0};

        impl::op_t eltwise_op(0, params.eltwise_kind, "elt");
        if (params.eltwise_kind == impl::op_kind::Elu) {
            eltwise_op.set_attr<float>("alpha", 1.f);
        } else if (params.eltwise_kind == impl::op_kind::HardTanh
                || params.eltwise_kind == impl::op_kind::Clamp) {
            eltwise_op.set_attr<float>("min", -1.f);
            eltwise_op.set_attr<float>("max", 2.f);
        }
        impl::op_t binary_op(1, params.binary_kind, "binary");

        impl::logical_tensor_t eltwise_src_lt = utils::logical_tensor_init(
                0, {1, 1, 2, 2}, impl::data_type::f32);
        impl::logical_tensor_t eltwise_dst_lt = utils::logical_tensor_init(
                1, {1, 1, 2, 2}, impl::data_type::f32, impl::layout_type::any);
        impl::logical_tensor_t binary_src_lt = utils::logical_tensor_init(
                2, params.binary_src_shape, impl::data_type::f32);
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

        impl::graph_t g(eng.kind());
        g.add_op(&eltwise_op);
        g.add_op(&binary_op);
        g.build_graph();

        impl::pass::pass_base_ptr apass = get_pass("eltwise_binary_fusion");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1);
        auto part = g.get_partitions()[0];

        // compile
        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> inputs {
                &eltwise_src_lt, &binary_src_lt};

        std::vector<const impl::logical_tensor_t *> outputs {&binary_dst_lt};

        ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

        impl::logical_tensor_t lt;
        cp.query_logical_tensor(binary_dst_lt.id, &lt);

        ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

        impl::tensor_t src_ts(eltwise_src_lt, &eng, src.data());
        impl::tensor_t binary_src_ts(binary_src_lt, &eng, binary_src.data());
        impl::tensor_t binary_dst_ts(binary_dst_lt, &eng, dst.data());

        impl::stream_t &strm = get_stream();

        ASSERT_EQ(cp.execute(&strm, {src_ts, binary_src_ts}, {binary_dst_ts}),
                impl::status::success);
        strm.wait();
    }
};

TEST_P(EltwiseBinary, TestEltwiseBinary) {
    TestEltwiseBinary();
}

INSTANTIATE_TEST_SUITE_P(Execute, EltwiseBinary,
        ::testing::Values(
                // with broadcast add and no swap inputs
                dnnl_graph_test_eltwise_binary_params {impl::op_kind::ReLU,
                        impl::op_kind::Add, {1}, {2.0}, false},
                // with broadcast add and swap inputs
                dnnl_graph_test_eltwise_binary_params {impl::op_kind::ReLU,
                        impl::op_kind::Add, {1}, {2.0}, true},
                // no broadcast add and no swap inputs
                dnnl_graph_test_eltwise_binary_params {impl::op_kind::ReLU,
                        impl::op_kind::Add, {1, 1, 2, 2}, {2.0, 2.0, 2.0, 2.0},
                        false},
                // no broadcast add and swap inputs
                dnnl_graph_test_eltwise_binary_params {impl::op_kind::ReLU,
                        impl::op_kind::Add, {1, 1, 2, 2}, {2.0, 2.0, 2.0, 2.0},
                        true},
                dnnl_graph_test_eltwise_binary_params {impl::op_kind::Clamp,
                        impl::op_kind::Multiply, {1}, {2.0}, false},
                dnnl_graph_test_eltwise_binary_params {impl::op_kind::Round,
                        impl::op_kind::Maximum, {1, 1, 2, 2},
                        {2.0, 2.0, 2.0, 2.0}, false},
                dnnl_graph_test_eltwise_binary_params {impl::op_kind::Sigmoid,
                        impl::op_kind::Divide, {1}, {2.0}, false},
                dnnl_graph_test_eltwise_binary_params {impl::op_kind::SoftPlus,
                        impl::op_kind::Minimum, {1, 1, 2, 2},
                        {2.0, 2.0, 2.0, 2.0}, false},
                dnnl_graph_test_eltwise_binary_params {impl::op_kind::Exp,
                        impl::op_kind::Minimum, {1, 1, 2, 2},
                        {2.0, 2.0, 2.0, 2.0}, false},
                dnnl_graph_test_eltwise_binary_params {impl::op_kind::Elu,
                        impl::op_kind::Subtract, {1, 1, 2, 2},
                        {2.0, 2.0, 2.0, 2.0}, false},
                dnnl_graph_test_eltwise_binary_params {impl::op_kind::Sqrt,
                        impl::op_kind::Minimum, {1}, {2.0}, false},
                dnnl_graph_test_eltwise_binary_params {impl::op_kind::HardSwish,
                        impl::op_kind::Minimum, {1}, {2.0}, false},
                dnnl_graph_test_eltwise_binary_params {impl::op_kind::Tanh,
                        impl::op_kind::Add, {1}, {2.0}, true}));

struct dnnl_graph_test_pool_binary_params {
    impl::op_kind_t pool_kind;
    impl::op_kind_t binary_kind;
};

class PoolBinary
    : public ::testing::TestWithParam<dnnl_graph_test_pool_binary_params> {
public:
    void TestPoolBinary() {
        const auto params = ::testing::TestWithParam<
                dnnl_graph_test_pool_binary_params>::GetParam();
        using dims = impl::dnnl_impl::dims;

        impl::engine_t &eng = get_engine();

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

            test::vector<float> src(product(src_shape), 4.0);
            test::vector<float> dst(product(dst_shape), 0.0);
            test::vector<float> post_src(product(post_src_shape), 2.0);

            impl::op_t pool_op(0, params.pool_kind, "pool");
            pool_op.set_attr<dims>("strides", dims(spatial_size, 2));
            pool_op.set_attr<dims>("kernel", dims(spatial_size, 2));
            pool_op.set_attr<dims>("pads_begin", dims(spatial_size, 0));
            pool_op.set_attr<dims>("pads_end", dims(spatial_size, 0));
            pool_op.set_attr<std::string>("data_format", data_format);
            if (params.pool_kind == impl::op_kind::AvgPool) {
                pool_op.set_attr<bool>("exclude_pad", false);
            } else {
                pool_op.set_attr<dims>("dilations", dims(spatial_size, 1));
            }

            impl::op_t binary_op(1, params.binary_kind, "binary");

            impl::logical_tensor_t src_lt
                    = utils::logical_tensor_init(0, src_shape, dt);
            impl::logical_tensor_t dst_lt
                    = utils::logical_tensor_init(1, dst_shape, dt);
            impl::logical_tensor_t post_src_lt
                    = utils::logical_tensor_init(2, post_src_shape, dt);
            impl::logical_tensor_t add_dst_lt
                    = utils::logical_tensor_init(3, dst_shape, dt);

            pool_op.add_input(src_lt);
            pool_op.add_output(dst_lt);
            binary_op.add_input(dst_lt);
            binary_op.add_input(post_src_lt);
            binary_op.add_output(add_dst_lt);

            impl::graph_t g(eng.kind());
            g.add_op(&pool_op);
            g.add_op(&binary_op);
            g.build_graph();

            impl::pass::pass_base_ptr apass = get_pass("pool_binary_fusion");
            apass->run(g);
            ASSERT_EQ(g.get_num_partitions(), 1);
            auto part = g.get_partitions()[0];

            // compile
            impl::partition_t p;
            p.init(part);

            impl::compiled_partition_t cp(p);

            std::vector<const impl::logical_tensor_t *> inputs {
                    &src_lt, &post_src_lt};
            std::vector<const impl::logical_tensor_t *> outputs {&add_dst_lt};

            ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng),
                    impl::status::success);

            impl::logical_tensor_t lt;
            cp.query_logical_tensor(add_dst_lt.id, &lt);
            ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

            impl::tensor_t src_ts(src_lt, &eng, src.data());
            impl::tensor_t post_src_ts(post_src_lt, &eng, post_src.data());
            impl::tensor_t add_dst_ts(add_dst_lt, &eng, dst.data());

            impl::stream_t &strm = get_stream();
            ASSERT_EQ(cp.execute(&strm, {src_ts, post_src_ts}, {add_dst_ts}),
                    impl::status::success);
            strm.wait();
        }
    }
};

TEST_P(PoolBinary, TestPoolBinary) {
    TestPoolBinary();
}

INSTANTIATE_TEST_SUITE_P(Execute, PoolBinary,
        ::testing::Values(
                dnnl_graph_test_pool_binary_params {
                        impl::op_kind::AvgPool, impl::op_kind::Add},
                dnnl_graph_test_pool_binary_params {
                        impl::op_kind::MaxPool, impl::op_kind::Add},
                dnnl_graph_test_pool_binary_params {
                        impl::op_kind::AvgPool, impl::op_kind::Divide},
                dnnl_graph_test_pool_binary_params {
                        impl::op_kind::AvgPool, impl::op_kind::Maximum},
                dnnl_graph_test_pool_binary_params {
                        impl::op_kind::MaxPool, impl::op_kind::Minimum},
                dnnl_graph_test_pool_binary_params {
                        impl::op_kind::AvgPool, impl::op_kind::Multiply},
                dnnl_graph_test_pool_binary_params {
                        impl::op_kind::MaxPool, impl::op_kind::Subtract}));

TEST(ExecuteSubgraphInt8, Maxpool) {
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

        test::vector<uint8_t> src_u8_data(product(src_shape));
        test::vector<int8_t> case1_out_data(product(dst_shape));
        test::vector<int8_t> case2_out_data(product(dst_shape));

        // random generate src, weight and bias data
        // random seed = 7
        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> u8_distribution(0.0f, 255.0f);
        std::generate(src_u8_data.begin(), src_u8_data.end(), [&]() {
            return static_cast<uint8_t>(u8_distribution(generator));
        });

        float scale_src = 1 / 127.f;
        float scale_out = 1 / 127.f;
        int64_t zp_src = 0;
        int64_t zp_out = 0;

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

        // prepare logical tensor
        impl::logical_tensor_t src_u8
                = utils::logical_tensor_init(1, src_shape, impl::data_type::u8);
        impl::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                2, src_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_f32 = utils::logical_tensor_init(
                3, dst_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_u8
                = utils::logical_tensor_init(4, dst_shape, impl::data_type::u8);

        dqdata_op.add_input(src_u8);
        dqdata_op.add_output(src_f32_dq);

        maxpool_op.add_input(src_f32_dq);
        maxpool_op.add_output(dst_f32);

        qout_op.add_input(dst_f32);
        qout_op.add_output(dst_u8);

        impl::graph_t g(engine.kind());
        g.add_op(&dqdata_op);
        g.add_op(&maxpool_op);
        g.add_op(&qout_op);
        g.build_graph();

        impl::tensor_t src_u8_ts(src_u8, &engine, src_u8_data.data());
        impl::tensor_t dst_u8_ts(dst_u8, &engine, case1_out_data.data());
        impl::tensor_t dst_u8_case2_ts(dst_u8, &engine, case2_out_data.data());

        // -------------------------case 1----------------------------------
        ASSERT_EQ(run_graph(g, {src_u8_ts}, {dst_u8_ts}, engine, strm),
                impl::status::success);

        // -------------------------case 2----------------------------------
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

TEST(ExecuteSubgraphInt8, Avgpool) {
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

        test::vector<uint8_t> src_u8_data(product(src_shape));
        test::vector<int8_t> case1_out_data(product(dst_shape));
        test::vector<int8_t> case2_out_data(product(dst_shape));

        // random generate src, weight and bias data
        // random seed = 7
        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> u8_distribution(0.0f, 255.0f);
        std::generate(src_u8_data.begin(), src_u8_data.end(), [&]() {
            return static_cast<uint8_t>(u8_distribution(generator));
        });

        float scale_src = 1 / 127.f;
        float scale_out = 1 / 127.f;
        int64_t zp_src = 0;
        int64_t zp_out = 0;

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

        // prepare logical tensor
        impl::logical_tensor_t src_u8
                = utils::logical_tensor_init(1, src_shape, impl::data_type::u8);
        impl::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                2, src_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_f32 = utils::logical_tensor_init(
                3, dst_shape, impl::data_type::f32);
        impl::logical_tensor_t dst_u8
                = utils::logical_tensor_init(4, dst_shape, impl::data_type::u8);

        dqdata_op.add_input(src_u8);
        dqdata_op.add_output(src_f32_dq);

        avgpool_op.add_input(src_f32_dq);
        avgpool_op.add_output(dst_f32);

        qout_op.add_input(dst_f32);
        qout_op.add_output(dst_u8);

        impl::graph_t g(engine.kind());
        g.add_op(&dqdata_op);
        g.add_op(&avgpool_op);
        g.add_op(&qout_op);
        g.build_graph();

        impl::tensor_t src_u8_ts(src_u8, &engine, src_u8_data.data());
        impl::tensor_t dst_u8_ts(dst_u8, &engine, case1_out_data.data());
        impl::tensor_t dst_u8_case2_ts(dst_u8, &engine, case2_out_data.data());

        // -------------------------case 1----------------------------------
        ASSERT_EQ(run_graph(g, {src_u8_ts}, {dst_u8_ts}, engine, strm),
                impl::status::success);

        // -------------------------case 2----------------------------------
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

TEST(ExecuteSubgraphInt8, PoolAdd) {
    using dims = impl::dnnl_impl::dims;
    using config_t = std::tuple<impl::op_kind_t, std::string, bool>;

    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    const std::vector<config_t> confs {
            config_t {impl::op_kind::AvgPool, "int8_avgpool_add_fusion", true},
            config_t {impl::op_kind::AvgPool, "int8_avgpool_add_fusion", false},
            config_t {impl::op_kind::MaxPool, "int8_maxpool_add_fusion", true},
            config_t {
                    impl::op_kind::MaxPool, "int8_maxpool_add_fusion", false}};
    const std::vector<std::string> qtypes {"symmetric", "asymmetric"};
    const std::vector<bool> swap_add_ins {true, false};

    for_(const auto swap_add_in : swap_add_ins)
    for_(const auto &qtype : qtypes)
    for (const auto &conf : confs) {
        if (engine.kind() == impl::engine_kind::gpu && qtype == "asymmetric")
            continue;

        impl::op_kind_t base_op = impl::op_kind::Wildcard;
        std::string fuse_name;
        bool per_channel_broadcast = false;
        std::tie(base_op, fuse_name, per_channel_broadcast) = conf;

        const std::string data_format {"NCX"};
        const int64_t channels = 2;
        std::vector<int64_t> src_shape {2, channels, 4, 4};
        std::vector<int64_t> dst_shape {2, channels, 2, 2};
        std::vector<int64_t> other_shape {1, 1, 1, 1};
        if (per_channel_broadcast) other_shape[1] = channels;

        test::vector<int8_t> src_s8_data(product(src_shape));
        test::vector<int8_t> other_s8_data(product(other_shape));
        test::vector<int8_t> case1_dst_s8_data(product(dst_shape));
        test::vector<int8_t> case2_dst_s8_data(product(dst_shape));

        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> s8_distribution(-127.0f, 128.0f);
        std::generate(src_s8_data.begin(), src_s8_data.end(), [&]() {
            return static_cast<uint8_t>(s8_distribution(generator));
        });
        std::generate(other_s8_data.begin(), other_s8_data.end(), [&]() {
            return static_cast<uint8_t>(s8_distribution(generator));
        });

        const float scale_src = 5 / 127.f;
        const float scale_out = 10 / 127.f;
        const float scale_other = 2 / 127.f;
        const int64_t zp_src = (qtype == "symmetric") ? 0 : -2;
        const int64_t zp_out = (qtype == "symmetric") ? 0 : -2;
        const int64_t zp_other = (qtype == "symmetric") ? 0 : 4;

        impl::op_t dqdata_op(0, impl::op_kind::Dequantize, "dqdata_op");
        dqdata_op.set_attr<std::string>("qtype", "per_tensor");
        dqdata_op.set_attr<std::vector<int64_t>>("zps", {zp_src});
        dqdata_op.set_attr<std::vector<float>>("scales", {scale_src});
        dqdata_op.set_attr<int64_t>("axis", 1);

        impl::op_t pool_op(1, base_op, "pool_op");
        size_t spatial_size = src_shape.size() - 2;
        pool_op.set_attr<dims>("strides", dims(spatial_size, 2));
        pool_op.set_attr<dims>("kernel", dims(spatial_size, 2));
        pool_op.set_attr<dims>("pads_begin", dims(spatial_size, 0));
        pool_op.set_attr<dims>("pads_end", dims(spatial_size, 0));
        pool_op.set_attr<std::string>("data_format", data_format);
        if (base_op == impl::op_kind::AvgPool)
            pool_op.set_attr<bool>("exclude_pad", false);
        else
            // MaxPool
            pool_op.set_attr<dims>("dilations", dims(spatial_size, 1));

        impl::op_t qout_op(2, impl::op_kind::Quantize, "qout_op");
        qout_op.set_attr<std::string>("qtype", "per_tensor");
        qout_op.set_attr<std::vector<int64_t>>("zps", {zp_out});
        qout_op.set_attr<std::vector<float>>("scales", {scale_out});
        qout_op.set_attr<int64_t>("axis", 1);

        impl::op_t dqother_op(3, impl::op_kind::Dequantize, "dqother_op");
        dqother_op.set_attr<std::string>("qtype", "per_tensor");
        dqother_op.set_attr<std::vector<int64_t>>("zps", {zp_other});
        dqother_op.set_attr<std::vector<float>>("scales", {scale_other});
        dqother_op.set_attr<int64_t>("axis", 1);

        impl::op_t add_op(4, impl::op_kind::Add, "add_op");

        auto src_s8
                = utils::logical_tensor_init(0, src_shape, impl::data_type::s8);
        auto src_f32_dq = utils::logical_tensor_init(
                1, src_shape, impl::data_type::f32);
        auto dst_f32 = utils::logical_tensor_init(
                2, dst_shape, impl::data_type::f32);
        auto dst_s8
                = utils::logical_tensor_init(3, dst_shape, impl::data_type::s8);
        auto other_s8 = utils::logical_tensor_init(
                4, other_shape, impl::data_type::s8);
        auto other_f32_dq = utils::logical_tensor_init(
                5, other_shape, impl::data_type::f32);
        auto dst_add_f32 = utils::logical_tensor_init(
                6, dst_shape, impl::data_type::f32);

        dqdata_op.add_input(src_s8);
        dqdata_op.add_output(src_f32_dq);

        pool_op.add_input(src_f32_dq);
        pool_op.add_output(dst_f32);

        dqother_op.add_input(other_s8);
        dqother_op.add_output(other_f32_dq);

        if (swap_add_in) {
            add_op.add_input(other_f32_dq);
            add_op.add_input(dst_f32);
        } else {
            add_op.add_input(dst_f32);
            add_op.add_input(other_f32_dq);
        }
        add_op.add_output(dst_add_f32);

        qout_op.add_input(dst_add_f32);
        qout_op.add_output(dst_s8);

        impl::graph_t g(engine.kind());
        g.add_op(&dqdata_op);
        g.add_op(&pool_op);
        g.add_op(&dqother_op);
        g.add_op(&add_op);
        g.add_op(&qout_op);
        g.build_graph();

        impl::tensor_t src_s8_ts(src_s8, &engine, src_s8_data.data());
        impl::tensor_t other_s8_ts(other_s8, &engine, other_s8_data.data());
        impl::tensor_t case1_dst_s8_ts(
                dst_s8, &engine, case1_dst_s8_data.data());
        impl::tensor_t case2_dst_s8_ts(
                dst_s8, &engine, case2_dst_s8_data.data());

        // -------------------------case 1----------------------------------
        ASSERT_EQ(run_graph(g, {src_s8_ts, other_s8_ts}, {case1_dst_s8_ts},
                          engine, strm),
                impl::status::success);

        // -------------------------case 2----------------------------------
        impl::pass::pass_base_ptr apass = get_pass(fuse_name);
        apass->run(g);

        ASSERT_EQ(g.get_num_partitions(), 1);
        auto part = g.get_partitions()[0];

        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);
        std::vector<const impl::logical_tensor_t *> lt_ins {&src_s8, &other_s8};
        std::vector<const impl::logical_tensor_t *> lt_outs {&dst_s8};
        p.compile(&cp, lt_ins, lt_outs, &engine);

        cp.execute(&strm, {src_s8_ts, other_s8_ts}, {case2_dst_s8_ts});
        strm.wait();

        static auto isa = dnnl_get_effective_cpu_isa();
        if (isa < dnnl_cpu_isa_avx512_core_vnni)
            ASSERT_TRUE(allclose(case1_dst_s8_data, case2_dst_s8_data,
                    /*rtol*/ 0.1f,
                    /*atol*/ 1.f));
        else
            ASSERT_TRUE(allclose(case1_dst_s8_data, case2_dst_s8_data,
                    /*rtol*/ 0.01f,
                    /*atol*/ 1.f));
    }
}

TEST(Compile, MatmulAddGetInplacePair) {
    impl::engine_t &eng = get_engine();

    impl::graph_t agraph(eng.kind());
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

    impl::pass::pass_base_ptr apass1 = get_pass("matmul_post_ops_chain_fusion");
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

TEST(ExecuteSubgraphInt8, QuantWeiConv2dSumRelu) {
    using dims = impl::dnnl_impl::dims;

    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    if (engine.kind() == impl::engine_kind::gpu) return;

    std::vector<int64_t> groups = {1, 4};
    std::vector<bool> with_biases = {true, false};
    std::vector<std::string> weight_qtypes = {"per_tensor", "per_channel"};

    for_(const auto &g : groups)
    for_(const auto with_bias : with_biases)
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

        impl::graph_t g(engine.kind());
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
        impl::pass::pass_base_ptr apass = get_pass("int8_conv_post_ops_fusion");

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

TEST(ExecuteSubgraphInt8, QuantWeiMatmulBiasSumNdx2d) {
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

        impl::graph_t g(engine.kind());
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
                = get_pass("int8_matmul_post_ops_fusion");
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

TEST(ExecuteSubgraphInt8, QuantWeiMatmulBiasNdx2dWithTranspose) {
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

        impl::graph_t g(engine.kind());
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
                = get_pass("int8_matmul_post_ops_fusion");
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

TEST(ExecuteSubgraphInt8, QuantWeiMatmulBiasReluNdx2d) {
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

        impl::graph_t g(engine.kind());
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
        impl::pass::pass_base_ptr apass
                = get_pass("int8_matmul_post_ops_fusion");
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

TEST(ExecuteSubgraphInt8, Matmul2dx3dWithTranspose) {
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

        impl::graph_t g(engine.kind());
        g.add_op(&dqdata_op);
        g.add_op(&dqweight_op);
        g.add_op(&matmul_op);
        g.add_op(&qout_op);
        g.build_graph();

        impl::pass::pass_base_ptr apass
                = get_pass("int8_matmul_post_ops_fusion");
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

TEST(ExecuteSubgraphInt8, MatmulBiasSumGetInplacePair) {
    // skip the test on AArch64 or some older machine without avx support
    SKIP_IF(dnnl_get_effective_cpu_isa() < dnnl_cpu_isa_avx,
            "skip on machine without AVX");
    impl::engine_t &engine = get_engine();

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

        impl::graph_t g(engine.kind());
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

        impl::pass::pass_base_ptr apass
                = get_pass("int8_matmul_post_ops_fusion");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 2);
        auto part2 = g.get_partitions()[0]; // int8_mamtul_bias_sum
        auto part1 = g.get_partitions()[1]; // int8_matmul_bias

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

TEST(Execute, MulAddPerTensorBroadcast) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src0 {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> src1 {2.0};
    test::vector<float> post_src {2.0};
    test::vector<float> ref_dst {6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0};
    test::vector<float> dst(src0.size(), 0.0);

    impl::op_t mul_op(0, impl::op_kind::Multiply, "mul");
    impl::op_t add_op(1, impl::op_kind::Add, "add");

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1}, impl::data_type::f32);
    impl::logical_tensor_t mul_dst_lt
            = utils::logical_tensor_init(2, {1, 1, 3, 3}, impl::data_type::f32);
    impl::logical_tensor_t post_src_lt
            = utils::logical_tensor_init(3, {1}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(4, {1, 1, 3, 3}, impl::data_type::f32);

    mul_op.add_input(src0_lt);
    mul_op.add_input(src1_lt);
    mul_op.add_output(mul_dst_lt);
    add_op.add_input(mul_dst_lt);
    add_op.add_input(post_src_lt);
    add_op.add_output(dst_lt);

    impl::graph_t g(eng.kind());
    g.add_op(&mul_op);
    g.add_op(&add_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("binary_post_ops_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {
            &src0_lt, &src1_lt, &post_src_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};
    p.compile(&cp, inputs, outputs, &eng);

    impl::tensor_t src0_ts(src0_lt, &eng, src0.data());
    impl::tensor_t src1_ts(src1_lt, &eng, src1.data());
    impl::tensor_t post_src_ts(post_src_lt, &eng, post_src.data());
    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src0_ts, src1_ts, post_src_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, MulAddPerHwBroadcast) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src0(18, 2.0);
    test::vector<float> src1(1, 2.0);
    test::vector<float> post_src(6, 2.0);
    test::vector<float> ref_dst(18, 6.0);
    test::vector<float> dst(src0.size(), 0.0);

    impl::op_t mul_op(0, impl::op_kind::Multiply, "mul");
    impl::op_t add_op(1, impl::op_kind::Add, "add");

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 3, 2, 3}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1}, impl::data_type::f32);
    impl::logical_tensor_t mul_dst_lt
            = utils::logical_tensor_init(2, {1, 3, 2, 3}, impl::data_type::f32);
    impl::logical_tensor_t post_src_lt
            = utils::logical_tensor_init(3, {2, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(4, {1, 3, 2, 3}, impl::data_type::f32);

    mul_op.add_input(src0_lt);
    mul_op.add_input(src1_lt);
    mul_op.add_output(mul_dst_lt);
    add_op.add_input(mul_dst_lt);
    add_op.add_input(post_src_lt);
    add_op.add_output(dst_lt);

    impl::graph_t g(eng.kind());
    g.add_op(&mul_op);
    g.add_op(&add_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("binary_post_ops_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {
            &src0_lt, &src1_lt, &post_src_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

    impl::tensor_t src0_ts(src0_lt, &eng, src0.data());
    impl::tensor_t src1_ts(src1_lt, &eng, src1.data());
    impl::tensor_t post_src_ts(post_src_lt, &eng, post_src.data());
    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src0_ts, src1_ts, post_src_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, MulAddPerChannelBroadcast) {
    impl::engine_t &eng = get_engine();

    test::vector<float> src0 {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    test::vector<float> src1 {2.0};
    test::vector<float> post_src {2.0, 2.0, 2.0};
    test::vector<float> ref_dst {6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0};
    test::vector<float> dst(src0.size(), 0.0);

    impl::op_t mul_op(0, impl::op_kind::Multiply, "mul");
    impl::op_t add_op(1, impl::op_kind::Add, "add");

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, {1, 3, 1, 3}, impl::data_type::f32);
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, {1}, impl::data_type::f32);
    impl::logical_tensor_t mul_dst_lt
            = utils::logical_tensor_init(2, {1, 3, 1, 3}, impl::data_type::f32);
    impl::logical_tensor_t post_src_lt
            = utils::logical_tensor_init(3, {3, 1, 1}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(4, {1, 3, 1, 3}, impl::data_type::f32);

    mul_op.add_input(src0_lt);
    mul_op.add_input(src1_lt);
    mul_op.add_output(mul_dst_lt);
    add_op.add_input(mul_dst_lt);
    add_op.add_input(post_src_lt);
    add_op.add_output(dst_lt);

    impl::graph_t g(eng.kind());
    g.add_op(&mul_op);
    g.add_op(&add_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("binary_post_ops_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {
            &src0_lt, &src1_lt, &post_src_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};
    p.compile(&cp, inputs, outputs, &eng);

    impl::tensor_t src0_ts(src0_lt, &eng, src0.data());
    impl::tensor_t src1_ts(src1_lt, &eng, src1.data());
    impl::tensor_t post_src_ts(post_src_lt, &eng, post_src.data());
    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src0_ts, src1_ts, post_src_ts}, {dst_ts});
    strm.wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, AddAdd) {
    impl::engine_t &eng = get_engine();

    std::vector<int64_t> src_shape = {8, 128, 768};
    test::vector<float> src0(product(src_shape));
    test::vector<float> src1(product(src_shape));
    test::vector<float> post_src(product(src_shape));
    test::vector<float> dst(src0.size(), 0.0);

    // random generate src, weight data
    // random seed = 7
    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> f32_distribution(-1.0f, 1.0f);
    std::generate(src0.begin(), src0.end(),
            [&]() { return f32_distribution(generator); });
    std::generate(src1.begin(), src1.end(),
            [&]() { return f32_distribution(generator); });
    std::generate(post_src.begin(), post_src.end(),
            [&]() { return f32_distribution(generator); });

    impl::op_t add0_op(0, impl::op_kind::Add, "add0");
    impl::op_t add1_op(1, impl::op_kind::Add, "add1");

    impl::logical_tensor_t src0_lt
            = utils::logical_tensor_init(0, src_shape, impl::data_type::f32);
    impl::logical_tensor_t src1_lt
            = utils::logical_tensor_init(1, src_shape, impl::data_type::f32);
    impl::logical_tensor_t add_dst_lt
            = utils::logical_tensor_init(2, src_shape, impl::data_type::f32);
    impl::logical_tensor_t post_src_lt
            = utils::logical_tensor_init(3, src_shape, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(4, src_shape, impl::data_type::f32);

    add0_op.add_input(src0_lt);
    add0_op.add_input(src1_lt);
    add0_op.add_output(add_dst_lt);
    add1_op.add_input(add_dst_lt);
    add1_op.add_input(post_src_lt);
    add1_op.add_output(dst_lt);

    impl::graph_t g(eng.kind());
    g.add_op(&add0_op);
    g.add_op(&add1_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("binary_post_ops_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {
            &src0_lt, &src1_lt, &post_src_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};
    p.compile(&cp, inputs, outputs, &eng);

    // inplace
    auto inplace_pair = cp.get_inplace_pairs();
    ASSERT_EQ(inplace_pair[0].input, post_src_lt.id);
    ASSERT_EQ(inplace_pair[0].output, dst_lt.id);

    impl::tensor_t src0_ts(src0_lt, &eng, src0.data());
    impl::tensor_t src1_ts(src1_lt, &eng, src1.data());
    impl::tensor_t post_src_ts(post_src_lt, &eng, post_src.data());
    impl::tensor_t dst_inplace_ts(dst_lt, &eng, post_src.data());
    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    cp.execute(&strm, {src0_ts, src1_ts, post_src_ts}, {dst_ts});
    cp.execute(&strm, {src0_ts, src1_ts, post_src_ts}, {dst_inplace_ts});
    strm.wait();

    for (size_t i = 0; i < src0.size(); ++i) {
        ASSERT_FLOAT_EQ(post_src[i], dst[i]);
    }
}

TEST(ExecuteSubgraphInt8, BmmU8u8f32) {
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

    impl::graph_t g(engine.kind());
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
    impl::pass::pass_base_ptr apass = get_pass("int8_matmul_post_ops_fusion");
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
    if (isa >= dnnl_cpu_isa_avx512_core_vnni) {
        ASSERT_TRUE(allclose(case1_out_data, case2_out_data, /*rtol*/ 0.01f,
                /*atol*/ 1.f));
    }
}

TEST(ExecuteSubgraphInt8, BmmDivU8u8f32) {
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
    binary_op.set_attr<std::string>("auto_broadcast", "numpy");

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

    impl::graph_t g(engine.kind());
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

TEST(ExecuteSubgraphInt8, BmmDivAddU8u8f32) {
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    if (engine.kind() == impl::engine_kind::gpu) return;
    std::string qtype = "per_tensor";
    // prepare fp32 data
    std::vector<int64_t> src_shape = {1, 4, 16, 8};
    std::vector<int64_t> weight_shape = {1, 4, 8, 16};
    std::vector<int64_t> dst_shape = {1, 4, 16, 16};
    std::vector<int64_t> post_add_shape = {1, 1, 1, 16};

    test::vector<uint8_t> src_data(product(src_shape));
    test::vector<uint8_t> weight_data(product(weight_shape));
    test::vector<uint8_t> add_src1_data(product(post_add_shape));

    // random generate src, weight data
    // random seed = 7
    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> u8_distribution(0.0f, 255.0f);
    std::generate(src_data.begin(), src_data.end(),
            [&]() { return static_cast<uint8_t>(u8_distribution(generator)); });
    std::generate(weight_data.begin(), weight_data.end(),
            [&]() { return static_cast<uint8_t>(u8_distribution(generator)); });
    std::generate(add_src1_data.begin(), add_src1_data.end(),
            [&]() { return u8_distribution(generator); });
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
    binary_op.set_attr<std::string>("auto_broadcast", "numpy");

    impl::op_t binary_op2(6, impl::op_kind::Add, "binary_add");
    binary_op2.set_attr<std::string>("auto_broadcast", "numpy");

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
    impl::logical_tensor_t add_src1 = utils::logical_tensor_init(
            10, post_add_shape, impl::data_type::f32);
    impl::logical_tensor_t add_f32
            = utils::logical_tensor_init(11, dst_shape, impl::data_type::f32);

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

    binary_op2.add_input(div_f32);
    binary_op2.add_input(add_src1);
    binary_op2.add_output(add_f32);

    impl::graph_t g(engine.kind());
    g.add_op(&dqdata_op);
    g.add_op(&dqweight_op);
    g.add_op(&matmul_op);
    g.add_op(&binary_op);
    g.add_op(&binary_op2);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("x8x8f32_matmul_div_add_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> lt_ins {
            &src_u8, &weight_u8, &div_src1, &add_src1};
    std::vector<const impl::logical_tensor_t *> lt_outs {&add_f32};

    p.compile(&cp, lt_ins, lt_outs, &engine);

    impl::tensor_t src_u8_ts(src_u8, &engine, src_data.data());
    impl::tensor_t weight_u8_ts(weight_u8, &engine, weight_data.data());
    impl::tensor_t div_src1_ts(div_src1, &engine, div_src1_data.data());
    impl::tensor_t add_src1_ts(add_src1, &engine, add_src1_data.data());
    test::vector<float> case2_out_data(product(dst_shape));
    impl::tensor_t dst_f32_case2_ts(add_f32, &engine, case2_out_data.data());
    cp.execute(&strm, {src_u8_ts, weight_u8_ts, div_src1_ts, add_src1_ts},
            {dst_f32_case2_ts});
    strm.wait();
}

TEST(ExecuteSubgraphInt8, BmmX8x8bf16) {
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

            impl::graph_t g(engine.kind());
            g.add_op(&dqdata_op);
            g.add_op(&dqweight_op);
            g.add_op(&matmul_op);
            g.add_op(&tcdata_op);
            g.add_op(&tcweight_op);
            g.build_graph();

            impl::pass::pass_base_ptr apass
                    = get_pass("x8x8bf16_matmul_fusion");
            apass->run(g);
            ASSERT_EQ(g.get_num_partitions(), 1);
            auto part = g.get_partitions()[0];

            // compile
            impl::partition_t p;
            p.init(part);

            impl::compiled_partition_t cp(p);

            std::vector<const impl::logical_tensor_t *> lt_ins {&src, &weight};
            std::vector<const impl::logical_tensor_t *> lt_outs {&dst_bf16};

            p.compile(&cp, lt_ins, lt_outs, &engine);

            test::vector<float> div_src1_data(1);
            test::vector<float> dst_data(product(dst_shape));
            impl::tensor_t src_ts(src, &engine, src_data.data());
            impl::tensor_t weight_ts(weight, &engine, weight_data.data());
            impl::tensor_t dst_ts(dst_bf16, &engine, dst_data.data());
            cp.execute(&strm, {src_ts, weight_ts}, {dst_ts});
            strm.wait();
        }
    }
}

TEST(ExecuteSubgraphInt8, BmmDivX8x8bf16) {
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
            binary_op.set_attr<std::string>("auto_broadcast", "numpy");

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
                    = utils::logical_tensor_init(7, {1}, impl::data_type::bf16);
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

            impl::graph_t g(engine.kind());
            ASSERT_EQ(g.add_op(&dqdata_op), impl::status::success);
            ASSERT_EQ(g.add_op(&dqweight_op), impl::status::success);
            ASSERT_EQ(g.add_op(&matmul_op), impl::status::success);
            ASSERT_EQ(g.add_op(&tcdata_op), impl::status::success);
            ASSERT_EQ(g.add_op(&tcweight_op), impl::status::success);
            ASSERT_EQ(g.add_op(&binary_op), impl::status::success);
            ASSERT_EQ(g.build_graph(), impl::status::success);

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

TEST(ExecuteSubgraphInt8, BmmDivBlockedX8x8bf16) {
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    if (engine.kind() == impl::engine_kind::gpu) return;
    std::vector<std::string> dtypes = {"uint8", "int8"};

    std::vector<int64_t> src_shape = {1, 4, 16, 8};
    std::vector<int64_t> src_stride = {512, 8, 32, 1};
    std::vector<int64_t> weight_shape = {1, 4, 8, 16};
    std::vector<int64_t> weight_stride = {512, 8, 1, 32};
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
            binary_op.set_attr<std::string>("auto_broadcast", "numpy");

            // prepare logical tensor
            impl::logical_tensor_t src = utils::logical_tensor_init(
                    0, src_shape, src_stride, src_lt_dtype);
            impl::logical_tensor_t src_f32_dq = utils::logical_tensor_init(
                    1, src_shape, src_stride, impl::data_type::f32);
            impl::logical_tensor_t src_bf16 = utils::logical_tensor_init(
                    2, src_shape, src_stride, impl::data_type::bf16);
            impl::logical_tensor_t weight = utils::logical_tensor_init(
                    3, weight_shape, weight_stride, weight_lt_dtype);
            impl::logical_tensor_t weight_f32_dq = utils::logical_tensor_init(
                    4, weight_shape, weight_stride, impl::data_type::f32);
            impl::logical_tensor_t weight_bf16 = utils::logical_tensor_init(
                    5, weight_shape, weight_stride, impl::data_type::bf16);
            impl::logical_tensor_t dst_bf16 = utils::logical_tensor_init(
                    6, dst_shape, impl::data_type::bf16);
            impl::logical_tensor_t div_src1
                    = utils::logical_tensor_init(7, {1}, impl::data_type::bf16);
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

            impl::graph_t g(engine.kind());
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

TEST(ExecuteSubgraphInt8, BmmDivAddX8x8bf16) {
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    if (engine.kind() == impl::engine_kind::gpu) return;

    static auto isa = dnnl_get_effective_cpu_isa();
    SKIP_IF(isa < dnnl_cpu_isa_avx512_core
                    && engine.kind() == impl::engine_kind::cpu,
            "Skip bf16 examples for systems that do not support avx512_core.");

    std::vector<std::string> dtypes = {"uint8", "int8"};

    std::vector<int64_t> src_shape = {8, 4, 16, 8};
    std::vector<int64_t> weight_shape = {8, 4, 8, 16};
    std::vector<int64_t> post_div_shape = {1};
    std::vector<int64_t> post_add_shape = {8, 1, 1, 16};
    std::vector<int64_t> dst_shape = {8, 4, 16, 16};

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
            binary_op.set_attr<std::string>("auto_broadcast", "numpy");

            impl::op_t binary_add_op(6, impl::op_kind::Add, "binary_add");
            binary_add_op.set_attr<std::string>("auto_broadcast", "numpy");

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
            impl::logical_tensor_t div_src1 = utils::logical_tensor_init(
                    7, post_div_shape, impl::data_type::bf16);
            impl::logical_tensor_t div_bf16 = utils::logical_tensor_init(
                    8, dst_shape, impl::data_type::bf16);
            impl::logical_tensor_t add_src1 = utils::logical_tensor_init(
                    9, post_add_shape, impl::data_type::bf16);
            impl::logical_tensor_t add_bf16 = utils::logical_tensor_init(
                    10, dst_shape, impl::data_type::bf16);

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

            binary_add_op.add_input(div_bf16);
            binary_add_op.add_input(add_src1);
            binary_add_op.add_output(add_bf16);

            impl::graph_t g;
            g.add_op(&dqdata_op);
            g.add_op(&dqweight_op);
            g.add_op(&matmul_op);
            g.add_op(&tcdata_op);
            g.add_op(&tcweight_op);
            g.add_op(&binary_op);
            g.add_op(&binary_add_op);
            g.build_graph();

            impl::pass::pass_base_ptr apass
                    = get_pass("x8x8bf16_matmul_div_add_fusion");
            apass->run(g);
            ASSERT_EQ(g.get_num_partitions(), 1);
            auto part = g.get_partitions()[0];

            // compile
            impl::partition_t p;
            p.init(part);

            impl::compiled_partition_t cp(p);

            std::vector<const impl::logical_tensor_t *> lt_ins {
                    &src, &weight, &div_src1, &add_src1};
            std::vector<const impl::logical_tensor_t *> lt_outs {&add_bf16};

            p.compile(&cp, lt_ins, lt_outs, &engine);

            test::vector<float> div_src1_data(product(post_div_shape));
            test::vector<float> add_src1_data(product(post_add_shape));
            test::vector<float> dst_data(product(dst_shape));
            impl::tensor_t src_ts(src, &engine, src_data.data());
            impl::tensor_t weight_ts(weight, &engine, weight_data.data());
            impl::tensor_t div_src1_ts(div_src1, &engine, div_src1_data.data());
            impl::tensor_t add_src1_ts(add_src1, &engine, add_src1_data.data());
            impl::tensor_t dst_ts(add_bf16, &engine, dst_data.data());
            cp.execute(&strm, {src_ts, weight_ts, div_src1_ts, add_src1_ts},
                    {dst_ts});
            strm.wait();
        }
    }
}

TEST(ExecuteSubgraphInt8, MatmulBiasU8s8bf16) {
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
    impl::logical_tensor_t bias_bf16
            = utils::logical_tensor_init(6, bias_shape, impl::data_type::bf16);
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
    matmul_op.add_input(bias_bf16);
    matmul_op.add_output(dst_bf16);

    impl::graph_t g(engine.kind());
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
            &src_u8, &weight_s8, &bias_bf16};
    std::vector<const impl::logical_tensor_t *> lt_outs {&dst_bf16};

    p.compile(&cp, lt_ins, lt_outs, &engine);

    test::vector<float> dst_data(product(dst_shape));
    impl::tensor_t src_u8_ts(src_u8, &engine, src_data.data());
    impl::tensor_t weight_s8_ts(weight_s8, &engine, weight_data.data());
    impl::tensor_t bias_bf16_ts(bias_bf16, &engine, bias_data.data());
    impl::tensor_t dst_ts(dst_bf16, &engine, dst_data.data());
    cp.execute(&strm, {src_u8_ts, weight_s8_ts, bias_bf16_ts}, {dst_ts});
    strm.wait();
}

TEST(ExecuteSubgraphInt8, MatmulBiasAddU8s8bf16) {
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
    impl::logical_tensor_t bias_bf16
            = utils::logical_tensor_init(6, bias_shape, impl::data_type::bf16);
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
    matmul_op.add_input(bias_bf16);
    matmul_op.add_output(matmul_bf16);

    add_op.add_input(other_bf16);
    add_op.add_input(matmul_bf16);
    add_op.add_output(dst_bf16);

    impl::graph_t g(engine.kind());
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
            &src_u8, &weight_s8, &bias_bf16, &other_u8};
    std::vector<const impl::logical_tensor_t *> lt_outs {&dst_bf16};

    p.compile(&cp, lt_ins, lt_outs, &engine);

    test::vector<float> dst_data(product(dst_shape));
    impl::tensor_t src_u8_ts(src_u8, &engine, src_data.data());
    impl::tensor_t weight_s8_ts(weight_s8, &engine, weight_data.data());
    impl::tensor_t bias_bf16_ts(bias_bf16, &engine, bias_data.data());
    impl::tensor_t other_u8_ts(other_u8, &engine, other_data.data());
    impl::tensor_t dst_ts(dst_bf16, &engine, dst_data.data());
    cp.execute(&strm, {src_u8_ts, weight_s8_ts, bias_bf16_ts, other_u8_ts},
            {dst_ts});
    strm.wait();
}

TEST(ExecuteSubgraphInt8, MatmulBiasAddBF16U8s8bf16) {
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
    test::vector<float> other_data(product(dst_shape));
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

    impl::op_t tcdata_op {2, impl::op_kind::TypeCast, "typecast_data"};
    impl::op_t tcweight_op {3, impl::op_kind::TypeCast, "typecast_weight"};

    impl::op_t matmul_op(4, impl::op_kind::MatMul, "matmul_op");
    matmul_op.set_attr<bool>("transpose_a", false);
    matmul_op.set_attr<bool>("transpose_b", true);

    impl::op_t add_op(5, impl::op_kind::Add, "add_op");

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
    impl::logical_tensor_t bias_bf16
            = utils::logical_tensor_init(6, bias_shape, impl::data_type::bf16);
    impl::logical_tensor_t other_bf16
            = utils::logical_tensor_init(7, dst_shape, impl::data_type::bf16);
    impl::logical_tensor_t matmul_bf16
            = utils::logical_tensor_init(8, dst_shape, impl::data_type::bf16);
    impl::logical_tensor_t dst_bf16
            = utils::logical_tensor_init(9, dst_shape, impl::data_type::bf16);

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
    matmul_op.add_input(bias_bf16);
    matmul_op.add_output(matmul_bf16);

    add_op.add_input(other_bf16);
    add_op.add_input(matmul_bf16);
    add_op.add_output(dst_bf16);

    impl::graph_t g(engine.kind());
    g.add_op(&dqdata_op);
    g.add_op(&dqweight_op);
    g.add_op(&matmul_op);
    g.add_op(&tcdata_op);
    g.add_op(&tcweight_op);
    g.add_op(&add_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass
            = get_pass("x8s8bf16_matmul_bias_add_bf16_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> lt_ins {
            &src_u8, &weight_s8, &bias_bf16, &other_bf16};
    std::vector<const impl::logical_tensor_t *> lt_outs {&dst_bf16};

    p.compile(&cp, lt_ins, lt_outs, &engine);

    test::vector<float> dst_data(product(dst_shape));
    impl::tensor_t src_u8_ts(src_u8, &engine, src_data.data());
    impl::tensor_t weight_s8_ts(weight_s8, &engine, weight_data.data());
    impl::tensor_t bias_bf16_ts(bias_bf16, &engine, bias_data.data());
    impl::tensor_t other_bf16_ts(other_bf16, &engine, other_data.data());
    impl::tensor_t dst_ts(dst_bf16, &engine, dst_data.data());
    cp.execute(&strm, {src_u8_ts, weight_s8_ts, bias_bf16_ts, other_bf16_ts},
            {dst_ts});
    strm.wait();
}

TEST(ExecuteSubgraphInt8, MatmulBiasU8s8u8MixBf16) {
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

    impl::op_t tcdst_op {5, impl::op_kind::TypeCast, "typecast_dst"};

    impl::op_t qout_op(6, impl::op_kind::Quantize, "qdout_op");
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
    impl::logical_tensor_t bias_bf16
            = utils::logical_tensor_init(6, bias_shape, impl::data_type::bf16);
    impl::logical_tensor_t matmul_bf16
            = utils::logical_tensor_init(7, dst_shape, impl::data_type::bf16);
    impl::logical_tensor_t matmul_f32
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
    matmul_op.add_input(bias_bf16);
    matmul_op.add_output(matmul_bf16);

    tcdst_op.add_input(matmul_bf16);
    tcdst_op.add_output(matmul_f32);

    qout_op.add_input(matmul_f32);
    qout_op.add_output(dst_u8);

    impl::graph_t g(engine.kind());
    g.add_op(&dqdata_op);
    g.add_op(&dqweight_op);
    g.add_op(&matmul_op);
    g.add_op(&tcdata_op);
    g.add_op(&tcweight_op);
    g.add_op(&tcdst_op);
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
            &src_u8, &weight_s8, &bias_bf16};
    std::vector<const impl::logical_tensor_t *> lt_outs {&dst_u8};

    p.compile(&cp, lt_ins, lt_outs, &engine);

    test::vector<uint8_t> dst_data(product(dst_shape));
    impl::tensor_t src_u8_ts(src_u8, &engine, src_data.data());
    impl::tensor_t weight_s8_ts(weight_s8, &engine, weight_data.data());
    impl::tensor_t bias_bf16_ts(bias_bf16, &engine, bias_data.data());
    impl::tensor_t dst_ts(dst_u8, &engine, dst_data.data());
    cp.execute(&strm, {src_u8_ts, weight_s8_ts, bias_bf16_ts}, {dst_ts});
    strm.wait();
}

TEST(ExecuteSubgraphInt8, MatmulBiasGeluU8s8u8MixBf16) {
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
    impl::logical_tensor_t bias_bf16
            = utils::logical_tensor_init(6, bias_shape, impl::data_type::bf16);
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
    matmul_op.add_input(bias_bf16);
    matmul_op.add_output(matmul_bf16);

    gelu_op.add_input(matmul_bf16);
    gelu_op.add_output(gelu_bf16);

    tcgelu_op.add_input(gelu_bf16);
    tcgelu_op.add_output(gelu_f32);

    qout_op.add_input(gelu_f32);
    qout_op.add_output(dst_u8);

    impl::graph_t g(engine.kind());
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
            &src_u8, &weight_s8, &bias_bf16};
    std::vector<const impl::logical_tensor_t *> lt_outs {&dst_u8};

    p.compile(&cp, lt_ins, lt_outs, &engine);

    test::vector<uint8_t> dst_data(product(dst_shape));
    impl::tensor_t src_u8_ts(src_u8, &engine, src_data.data());
    impl::tensor_t weight_s8_ts(weight_s8, &engine, weight_data.data());
    impl::tensor_t bias_bf16_ts(bias_bf16, &engine, bias_data.data());
    impl::tensor_t dst_ts(dst_u8, &engine, dst_data.data());
    cp.execute(&strm, {src_u8_ts, weight_s8_ts, bias_bf16_ts}, {dst_ts});
    strm.wait();
}

TEST(Execute, ConvResBlock) {
    impl::engine_t &eng = get_engine();
    impl::stream_t &strm = get_stream();

    impl::op_t conv0(0, impl::op_kind::Convolution, "conv0");
    impl::op_t relu0(1, impl::op_kind::ReLU, "relu0");
    impl::op_t conv1(2, impl::op_kind::Convolution, "conv1");
    impl::op_t relu1(3, impl::op_kind::ReLU, "relu1");
    impl::op_t conv2(4, impl::op_kind::Convolution, "conv2");
    impl::op_t relu2(5, impl::op_kind::ReLU, "relu2");
    impl::op_t add(6, impl::op_kind::Add, "add");

    std::vector<int64_t> v_1_1 = {1, 1}, v_0_0 = {0, 0};
    utils::set_conv_common_attr(
            conv0, v_1_1, v_0_0, v_0_0, v_1_1, "", "NCX", "OIX", 1);
    utils::set_conv_common_attr(
            conv1, v_1_1, v_0_0, v_0_0, v_1_1, "", "NCX", "OIX", 1);
    utils::set_conv_common_attr(
            conv2, v_1_1, v_0_0, v_0_0, v_1_1, "", "NCX", "OIX", 1);

    // prepare logical tensor
    std::vector<impl::dim_t> src_shape = {8, 8, 32, 32};
    std::vector<impl::dim_t> wei_shape = {8, 8, 1, 1};
    impl::data_type_t dtype = impl::data_type::f32;
    auto conv0_src = utils::logical_tensor_init(0, src_shape, dtype);
    auto conv0_wei = utils::logical_tensor_init(1, wei_shape, dtype);
    auto relu0_src = utils::logical_tensor_init(2, src_shape, dtype);
    auto conv1_src = utils::logical_tensor_init(3, src_shape, dtype);
    auto conv1_wei = utils::logical_tensor_init(4, wei_shape, dtype);
    auto relu1_src = utils::logical_tensor_init(5, src_shape, dtype);
    auto conv2_src = utils::logical_tensor_init(6, src_shape, dtype);
    auto conv2_wei = utils::logical_tensor_init(7, wei_shape, dtype);
    auto add_src0 = utils::logical_tensor_init(8, src_shape, dtype);
    auto relu2_src = utils::logical_tensor_init(9, src_shape, dtype);
    auto relu2_dst = utils::logical_tensor_init(10, src_shape, dtype);

    conv0.add_input(conv0_src);
    conv0.add_input(conv0_wei);
    conv0.add_output(relu0_src);
    relu0.add_input(relu0_src);
    relu0.add_output(conv1_src);
    conv1.add_input(conv1_src);
    conv1.add_input(conv1_wei);
    conv1.add_output(relu1_src);
    relu1.add_input(relu1_src);
    relu1.add_output(conv2_src);
    conv2.add_input(conv2_src);
    conv2.add_input(conv2_wei);
    conv2.add_output(add_src0);
    add.add_input(add_src0);
    add.add_input(conv0_src);
    add.add_output(relu2_src);
    relu2.add_input(relu2_src);
    relu2.add_output(relu2_dst);

    impl::graph_t g(eng.kind());
    g.add_op(&conv0);
    g.add_op(&relu0);
    g.add_op(&conv1);
    g.add_op(&relu1);
    g.add_op(&conv2);
    g.add_op(&add);
    g.add_op(&relu2);
    g.build_graph();

    ASSERT_EQ(g.get_ops().size(), 7);

    impl::pass::pass_base_ptr apass = get_pass("conv_simple_resblock_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {
            &conv0_src, &conv0_wei, &conv1_wei, &conv2_wei, &conv0_src};
    std::vector<const impl::logical_tensor_t *> outputs {&relu2_dst};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

    test::vector<float> conv0_src_data(8 * 8 * 32 * 32);
    test::vector<float> conv0_wei_data(8 * 8 * 1 * 1);
    test::vector<float> conv1_wei_data(8 * 8 * 1 * 1);
    test::vector<float> conv2_wei_data(8 * 8 * 1 * 1);
    test::vector<float> relu2_dst_data(8 * 8 * 32 * 32);
    test::vector<float> ref_dst_data(8 * 8 * 32 * 32);

    // Initialize
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0f, 0.1f);

    std::generate(conv0_src_data.begin(), conv0_src_data.end(),
            [&]() { return distribution(generator); });
    std::generate(conv0_wei_data.begin(), conv0_wei_data.end(),
            [&]() { return distribution(generator); });
    std::generate(conv1_wei_data.begin(), conv1_wei_data.end(),
            [&]() { return distribution(generator); });
    std::generate(conv2_wei_data.begin(), conv2_wei_data.end(),
            [&]() { return distribution(generator); });

    impl::tensor_t conv0_src_ts(conv0_src, &eng, conv0_src_data.data());
    impl::tensor_t conv0_wei_ts(conv0_wei, &eng, conv0_wei_data.data());
    impl::tensor_t conv1_wei_ts(conv1_wei, &eng, conv1_wei_data.data());
    impl::tensor_t conv2_wei_ts(conv2_wei, &eng, conv2_wei_data.data());
    impl::tensor_t relu2_dst_ts(relu2_dst, &eng, relu2_dst_data.data());
    impl::tensor_t ref_dst_ts(relu2_dst, &eng, ref_dst_data.data());

    ASSERT_EQ(cp.execute(&strm,
                      {conv0_src_ts, conv0_wei_ts, conv1_wei_ts, conv2_wei_ts},
                      {relu2_dst_ts}),
            impl::status::success);
    strm.wait();
}

TEST(Execute, ConvResBlockWithNhwcLayout) {
    impl::engine_t &eng = get_engine();
    impl::stream_t &strm = get_stream();

    impl::op_t conv0(0, impl::op_kind::Convolution, "conv0");
    impl::op_t relu0(1, impl::op_kind::ReLU, "relu0");
    impl::op_t conv1(2, impl::op_kind::Convolution, "conv1");
    impl::op_t relu1(3, impl::op_kind::ReLU, "relu1");
    impl::op_t conv2(4, impl::op_kind::Convolution, "conv2");
    impl::op_t relu2(5, impl::op_kind::ReLU, "relu2");
    impl::op_t add(6, impl::op_kind::Add, "add");

    std::vector<int64_t> v_1_1 = {1, 1}, v_0_0 = {0, 0};
    utils::set_conv_common_attr(
            conv0, v_1_1, v_0_0, v_0_0, v_1_1, "", "NCX", "OIX", 1);
    utils::set_conv_common_attr(
            conv1, v_1_1, v_0_0, v_0_0, v_1_1, "", "NCX", "OIX", 1);
    utils::set_conv_common_attr(
            conv2, v_1_1, v_0_0, v_0_0, v_1_1, "", "NCX", "OIX", 1);

    // prepare logical tensor
    std::vector<impl::dim_t> src_shape = {8, 8, 32, 32};
    // strides for nhwc
    std::vector<impl::dim_t> src_strides_nhwc {
            src_shape[2] * src_shape[3] * src_shape[1], 1,
            src_shape[3] * src_shape[1], src_shape[1]};
    std::vector<impl::dim_t> wei_shape = {8, 8, 1, 1};
    impl::data_type_t dtype = impl::data_type::f32;
    auto conv0_src
            = utils::logical_tensor_init(0, src_shape, src_strides_nhwc, dtype);
    auto conv0_wei = utils::logical_tensor_init(1, wei_shape, dtype);
    auto relu0_src = utils::logical_tensor_init(2, src_shape, dtype);
    auto conv1_src = utils::logical_tensor_init(3, src_shape, dtype);
    auto conv1_wei = utils::logical_tensor_init(4, wei_shape, dtype);
    auto relu1_src = utils::logical_tensor_init(5, src_shape, dtype);
    auto conv2_src = utils::logical_tensor_init(6, src_shape, dtype);
    auto conv2_wei = utils::logical_tensor_init(7, wei_shape, dtype);
    auto add_src0 = utils::logical_tensor_init(8, src_shape, dtype);
    auto relu2_src = utils::logical_tensor_init(9, src_shape, dtype);
    auto relu2_dst = utils::logical_tensor_init(
            10, src_shape, dtype, impl::layout_type::any);

    conv0.add_input(conv0_src);
    conv0.add_input(conv0_wei);
    conv0.add_output(relu0_src);
    relu0.add_input(relu0_src);
    relu0.add_output(conv1_src);
    conv1.add_input(conv1_src);
    conv1.add_input(conv1_wei);
    conv1.add_output(relu1_src);
    relu1.add_input(relu1_src);
    relu1.add_output(conv2_src);
    conv2.add_input(conv2_src);
    conv2.add_input(conv2_wei);
    conv2.add_output(add_src0);
    add.add_input(add_src0);
    add.add_input(conv0_src);
    add.add_output(relu2_src);
    relu2.add_input(relu2_src);
    relu2.add_output(relu2_dst);

    impl::graph_t g(eng.kind());
    g.add_op(&conv0);
    g.add_op(&relu0);
    g.add_op(&conv1);
    g.add_op(&relu1);
    g.add_op(&conv2);
    g.add_op(&add);
    g.add_op(&relu2);
    g.build_graph();

    ASSERT_EQ(g.get_ops().size(), 7);

    impl::pass::pass_base_ptr apass = get_pass("conv_simple_resblock_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {
            &conv0_src, &conv0_wei, &conv1_wei, &conv2_wei, &conv0_src};
    std::vector<const impl::logical_tensor_t *> outputs {&relu2_dst};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

    impl::logical_tensor_t queried_relu2_dst;
    cp.query_logical_tensor(relu2_dst.id, &queried_relu2_dst);
    ASSERT_EQ(queried_relu2_dst.layout_type, impl::layout_type::strided);

    test::vector<float> conv0_src_data(8 * 8 * 32 * 32);
    test::vector<float> conv0_wei_data(8 * 8 * 1 * 1);
    test::vector<float> conv1_wei_data(8 * 8 * 1 * 1);
    test::vector<float> conv2_wei_data(8 * 8 * 1 * 1);
    test::vector<float> relu2_dst_data(8 * 8 * 32 * 32);
    test::vector<float> ref_dst_data(8 * 8 * 32 * 32);

    // Initialize
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0f, 0.1f);

    std::generate(conv0_src_data.begin(), conv0_src_data.end(),
            [&]() { return distribution(generator); });
    std::generate(conv0_wei_data.begin(), conv0_wei_data.end(),
            [&]() { return distribution(generator); });
    std::generate(conv1_wei_data.begin(), conv1_wei_data.end(),
            [&]() { return distribution(generator); });
    std::generate(conv2_wei_data.begin(), conv2_wei_data.end(),
            [&]() { return distribution(generator); });

    impl::tensor_t conv0_src_ts(conv0_src, &eng, conv0_src_data.data());
    impl::tensor_t conv0_wei_ts(conv0_wei, &eng, conv0_wei_data.data());
    impl::tensor_t conv1_wei_ts(conv1_wei, &eng, conv1_wei_data.data());
    impl::tensor_t conv2_wei_ts(conv2_wei, &eng, conv2_wei_data.data());
    impl::tensor_t relu2_dst_ts(relu2_dst, &eng, relu2_dst_data.data());
    impl::tensor_t ref_dst_ts(relu2_dst, &eng, ref_dst_data.data());

    ASSERT_EQ(cp.execute(&strm,
                      {conv0_src_ts, conv0_wei_ts, conv1_wei_ts, conv2_wei_ts},
                      {relu2_dst_ts}),
            impl::status::success);
    strm.wait();
}

TEST(Execute, Int8Mha) {
    impl::engine_t &eng = get_engine();
    impl::stream_t &strm = get_stream();

    SKIP_IF(eng.kind() == impl::engine_kind::gpu,
            "Skip for GPU - not supported yet.");

    impl::graph_t g(eng.kind());
    utils::construct_int8_MHA(&g);
    g.build_graph();

    ASSERT_EQ(g.get_ops().size(), 21);

    impl::pass::pass_base_ptr apass = get_pass("int8_MHA_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(partition_inputs.size(), 5);
    ASSERT_EQ(partition_outputs.size(), 1);

    std::vector<const impl::logical_tensor_t *> inputs, outputs;
    for (auto &lt : partition_inputs) {
        inputs.emplace_back(&lt);
    }
    for (auto &lt : partition_outputs) {
        // set output to be strided
        lt = utils::logical_tensor_init(
                lt.id, lt.data_type, impl::layout_type::strided);
        outputs.emplace_back(&lt);
    }

    impl::compiled_partition_t cp(p);
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

    using ltw = impl::logical_tensor_wrapper_t;

    std::vector<test::vector<float>> inputs_data, outputs_data;
    std::vector<impl::tensor_t> inputs_ts, outputs_ts;

    for (auto &lt : inputs) {
        inputs_data.emplace_back(
                test::vector<float>(utils::product(ltw(lt).vdims())));
        inputs_ts.emplace_back(*lt, &eng, inputs_data.back().data());
    }

    for (auto &lt : outputs) {
        impl::logical_tensor_t compiled_output;
        cp.query_logical_tensor(lt->id, &compiled_output);
        outputs_data.emplace_back(test::vector<float>(
                utils::product(ltw(compiled_output).vdims())));
        outputs_ts.emplace_back(
                compiled_output, &eng, outputs_data.back().data());
    }

    ASSERT_EQ(cp.execute(&strm, inputs_ts, outputs_ts), impl::status::success);
    strm.wait();
}

TEST(Execute, F32Mha) {
    impl::engine_t &eng = get_engine();
    impl::stream_t &strm = get_stream();

    impl::graph_t g(eng.kind());
    utils::construct_f32_MHA(&g);
    g.build_graph();

    ASSERT_EQ(g.get_ops().size(), 13);

    impl::pass::pass_base_ptr apass = get_pass("f32_MHA_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(partition_inputs.size(), 5);
    ASSERT_EQ(partition_outputs.size(), 1);

    std::vector<const impl::logical_tensor_t *> inputs, outputs;
    for (auto &lt : partition_inputs) {
        inputs.emplace_back(&lt);
    }
    for (auto &lt : partition_outputs) {
        // set output to be strided
        lt = utils::logical_tensor_init(
                lt.id, lt.data_type, impl::layout_type::strided);
        outputs.emplace_back(&lt);
    }

    impl::compiled_partition_t cp(p);
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

    using ltw = impl::logical_tensor_wrapper_t;

    std::vector<test::vector<float>> inputs_data, outputs_data;
    std::vector<impl::tensor_t> inputs_ts, outputs_ts;

    for (auto &lt : inputs) {
        inputs_data.emplace_back(
                test::vector<float>(utils::product(ltw(lt).vdims())));
        inputs_ts.emplace_back(*lt, &eng, inputs_data.back().data());
    }

    for (auto &lt : outputs) {
        impl::logical_tensor_t compiled_output;
        cp.query_logical_tensor(lt->id, &compiled_output);
        outputs_data.emplace_back(test::vector<float>(
                utils::product(ltw(compiled_output).vdims())));
        outputs_ts.emplace_back(
                compiled_output, &eng, outputs_data.back().data());
    }

    ASSERT_EQ(cp.execute(&strm, inputs_ts, outputs_ts), impl::status::success);
    strm.wait();
}

TEST(Execute, Int8Bf16Mha) {
    impl::engine_t &eng = get_engine();
    impl::stream_t &strm = get_stream();

    if (eng.kind() == impl::engine_kind::gpu) return;

    static auto isa = dnnl_get_effective_cpu_isa();
    SKIP_IF(isa < dnnl_cpu_isa_avx512_core
                    && eng.kind() == impl::engine_kind::cpu,
            "Skip bf16 examples for systems that do not support avx512_core.");

    impl::graph_t g(eng.kind());
    utils::construct_int8_bf16_MHA(&g);
    g.build_graph();

    ASSERT_EQ(g.get_ops().size(), 29);

    impl::pass::pass_base_ptr apass = get_pass("int8_bf16_MHA_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(partition_inputs.size(), 5);
    ASSERT_EQ(partition_outputs.size(), 1);

    std::vector<const impl::logical_tensor_t *> inputs, outputs;
    for (auto &lt : partition_inputs) {
        inputs.emplace_back(&lt);
    }
    for (auto &lt : partition_outputs) {
        outputs.emplace_back(&lt);
    }

    impl::compiled_partition_t cp(p);
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

    using ltw = impl::logical_tensor_wrapper_t;

    std::vector<test::vector<uint16_t>> inputs_data, outputs_data;
    std::vector<impl::tensor_t> inputs_ts, outputs_ts;

    for (auto &lt : partition_inputs) {
        inputs_data.emplace_back(
                test::vector<uint16_t>(utils::product(ltw(lt).vdims())));
        inputs_ts.emplace_back(lt, &eng, inputs_data.back().data());
    }

    for (auto &lt : partition_outputs) {
        outputs_data.emplace_back(
                test::vector<uint16_t>(utils::product(ltw(lt).vdims())));
        outputs_ts.emplace_back(lt, &eng, outputs_data.back().data());
    }

    ASSERT_EQ(cp.execute(&strm, inputs_ts, outputs_ts), impl::status::success);
    strm.wait();
}

TEST(Execute, ChainedReLU) {
    impl::engine_t &eng = get_engine();
    impl::stream_t &strm = get_stream();

    impl::graph_t g(eng.kind());
    utils::construct_chained_relu(&g);
    g.build_graph();

    ASSERT_EQ(g.get_ops().size(), 3);

    impl::pass::pass_base_ptr apass = get_pass("chained_relu_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(partition_inputs.size(), 1);
    ASSERT_EQ(partition_outputs.size(), 1);

    std::vector<const impl::logical_tensor_t *> inputs, outputs;
    for (auto &lt : partition_inputs) {
        inputs.emplace_back(&lt);
    }
    for (auto &lt : partition_outputs) {
        outputs.emplace_back(&lt);
    }

    impl::compiled_partition_t cp(p);
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

    using ltw = impl::logical_tensor_wrapper_t;

    std::vector<test::vector<float>> inputs_data, outputs_data;
    std::vector<impl::tensor_t> inputs_ts, outputs_ts;

    for (auto &lt : partition_inputs) {
        inputs_data.emplace_back(
                test::vector<float>(utils::product(ltw(lt).vdims())));
        inputs_ts.emplace_back(lt, &eng, inputs_data.back().data());
    }

    for (auto &lt : partition_outputs) {
        outputs_data.emplace_back(
                test::vector<float>(utils::product(ltw(lt).vdims())));
        outputs_ts.emplace_back(lt, &eng, outputs_data.back().data());
    }

    ASSERT_EQ(cp.execute(&strm, inputs_ts, outputs_ts), impl::status::success);
    strm.wait();
}

TEST(Execute, Int8ConvBiasReluConvBiasReluBlock) {
    impl::engine_t &eng = get_engine();
    impl::stream_t &strm = get_stream();

    utils::id_generator id_gen;
    impl::graph_t g(eng.kind());
    utils::construct_int8_conv_bias_relu_conv_bias_relu_block(&g, id_gen);
    g.build_graph();

    ASSERT_EQ(g.get_ops().size(), 10);

    impl::pass::pass_base_ptr apass
            = get_pass("int8_conv_bias_relu_conv_bias_relu_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(partition_inputs.size(), 5);
    ASSERT_EQ(partition_outputs.size(), 1);

    std::vector<const impl::logical_tensor_t *> inputs, outputs;
    for (auto &lt : partition_inputs) {
        inputs.emplace_back(&lt);
    }
    for (auto &lt : partition_outputs) {
        // set output to be strided
        lt = utils::logical_tensor_init(
                lt.id, lt.data_type, impl::layout_type::strided);
        outputs.emplace_back(&lt);
    }

    impl::compiled_partition_t cp(p);
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

    using ltw = impl::logical_tensor_wrapper_t;

    std::vector<test::vector<float>> inputs_data, outputs_data;
    std::vector<impl::tensor_t> inputs_ts, outputs_ts;

    for (auto &lt : inputs) {
        inputs_data.emplace_back(
                test::vector<float>(utils::product(ltw(lt).vdims())));
        inputs_ts.emplace_back(*lt, &eng, inputs_data.back().data());
    }

    for (auto &lt : outputs) {
        impl::logical_tensor_t compiled_output;
        cp.query_logical_tensor(lt->id, &compiled_output);
        outputs_data.emplace_back(test::vector<float>(
                utils::product(ltw(compiled_output).vdims())));
        outputs_ts.emplace_back(
                compiled_output, &eng, outputs_data.back().data());
    }

    ASSERT_EQ(cp.execute(&strm, inputs_ts, outputs_ts), impl::status::success);
    strm.wait();
}

TEST(Execute, ReorderAddBf16) {
    impl::engine_t &eng = get_engine();
    impl::stream_t &strm = get_stream();

    static auto isa = dnnl_get_effective_cpu_isa();
    SKIP_IF(isa < dnnl_cpu_isa_avx512_core
                    && eng.kind() == impl::engine_kind::cpu,
            "Skip bf16 examples for systems that do not support avx512_core.");

    test::vector<uint16_t> src {1, 2, 3, 4, 5, 6};
    test::vector<uint16_t> post_src {1, 2, 3, 4, 5, 6};
    test::vector<uint16_t> dst(src.size(), 0);

    impl::graph_t agraph(eng.kind());
    impl::op_t reorder {0, impl::op_kind::Reorder, "reorder"};
    impl::op_t add {1, impl::op_kind::Add, "add"};
    add.set_attr<std::string>("auto_broadcast", "none");

    impl::logical_tensor_t src_lt = utils::logical_tensor_init(
            0, {2, 3}, {3, 1}, impl::data_type::bf16);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {2, 3}, {1, 2}, impl::data_type::bf16);
    reorder.add_input(src_lt);
    reorder.add_output(dst_lt);

    impl::logical_tensor_t add_src_lt = utils::logical_tensor_init(
            2, {2, 3}, {1, 2}, impl::data_type::bf16);
    impl::logical_tensor_t add_dst_lt = utils::logical_tensor_init(
            3, {2, 3}, {1, 2}, impl::data_type::bf16);
    add.add_input(dst_lt);
    add.add_input(add_src_lt);
    add.add_output(add_dst_lt);

    ASSERT_EQ(agraph.add_op(&reorder), impl::status::success);
    ASSERT_EQ(agraph.add_op(&add), impl::status::success);

    ASSERT_EQ(agraph.build_graph(), impl::status::success);

    impl::pass::pass_base_ptr apass = get_pass("reorder_sum_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1);
    auto part = agraph.get_partitions()[0];
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    // compile reorder_add partition
    std::vector<const impl::logical_tensor_t *> inputs {&src_lt, &add_src_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&add_dst_lt};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t post_src_ts(add_src_lt, &eng, post_src.data());
    impl::tensor_t dst_ts(add_dst_lt, &eng, dst.data());
    cp.execute(&strm, {src_ts, post_src_ts}, {dst_ts});
    strm.wait();
}

TEST(Compile, ReorderAddGetInplacePair) {
    impl::engine_t &eng = get_engine();

    impl::graph_t agraph(eng.kind());
    impl::op_t reorder {0, impl::op_kind::Reorder, "reorder"};
    impl::op_t add {1, impl::op_kind::Add, "add"};
    add.set_attr<std::string>("auto_broadcast", "none");

    impl::logical_tensor_t src_lt = utils::logical_tensor_init(
            0, {2, 3}, {3, 1}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {2, 3}, {1, 2}, impl::data_type::f32);
    reorder.add_input(src_lt);
    reorder.add_output(dst_lt);

    impl::logical_tensor_t add_src_lt = utils::logical_tensor_init(
            2, {2, 3}, {1, 2}, impl::data_type::f32);
    impl::logical_tensor_t add_dst_lt = utils::logical_tensor_init(
            3, {2, 3}, {1, 2}, impl::data_type::f32);
    add.add_input(dst_lt);
    add.add_input(add_src_lt);
    add.add_output(add_dst_lt);

    ASSERT_EQ(agraph.add_op(&reorder), impl::status::success);
    ASSERT_EQ(agraph.add_op(&add), impl::status::success);

    ASSERT_EQ(agraph.build_graph(), impl::status::success);

    impl::pass::pass_base_ptr apass = get_pass("reorder_sum_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1);
    auto part = agraph.get_partitions()[0];
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    // compile reorder_add partition
    std::vector<const impl::logical_tensor_t *> inputs {&src_lt, &add_src_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&add_dst_lt};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

    std::vector<impl::inplace_pair_t> inplace_pairs = cp.get_inplace_pairs();
    ASSERT_EQ(inplace_pairs.size(), 1);
    ASSERT_EQ(inplace_pairs[0].input, add_src_lt.id);
    ASSERT_EQ(inplace_pairs[0].output, add_dst_lt.id);
}

TEST(Execute, Int8ReorderAdd) {
    /*
        dequant
        |
      reorder dequant
        |    /
        add
        |
        quant
    */
    impl::engine_t &engine = get_engine();

    test::vector<uint8_t> int8_src {1, 2, 3, 4, 5, 6};
    test::vector<uint8_t> int8_src_other {1, 2, 3, 4, 5, 6};
    test::vector<uint8_t> int8_dst(int8_src.size(), 0);

    // int8 = f32 / scales + zero_points
    test::vector<uint8_t> ref_dst {2, 6, 5, 9, 8, 12};

    impl::graph_t agraph(engine.kind());
    std::vector<int64_t> zps = {0};
    std::vector<float> scales = {3.f};
    impl::op_t dequant {0, impl::op_kind::Dequantize, "dequant"};
    dequant.set_attr("scales", scales);
    dequant.set_attr("zps", zps);
    impl::op_t reorder {1, impl::op_kind::Reorder, "reorder"};
    impl::op_t dequant_other {2, impl::op_kind::Dequantize, "dequant_other"};
    dequant_other.set_attr("scales", scales);
    dequant_other.set_attr("zps", zps);
    impl::op_t add {3, impl::op_kind::Add, "add"};
    add.set_attr<std::string>("auto_broadcast", "none");
    impl::op_t quant {4, impl::op_kind::Quantize, "quant"};
    quant.set_attr("scales", scales);
    quant.set_attr("zps", zps);

    // 1,2,3
    // 4,5,6
    impl::logical_tensor_t int8_src_lt = utils::logical_tensor_init(
            0, {2, 3}, {3, 1}, impl::data_type::u8);
    // 1,3,5
    // 2,4,6
    impl::logical_tensor_t int8_src_other_lt = utils::logical_tensor_init(
            1, {2, 3}, {1, 2}, impl::data_type::u8);
    impl::logical_tensor_t src_lt = utils::logical_tensor_init(
            2, {2, 3}, {3, 1}, impl::data_type::f32);
    impl::logical_tensor_t src_other_lt = utils::logical_tensor_init(
            3, {2, 3}, {1, 2}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            4, {2, 3}, {1, 2}, impl::data_type::f32);
    impl::logical_tensor_t dst_add_lt = utils::logical_tensor_init(
            5, {2, 3}, {1, 2}, impl::data_type::f32);
    impl::logical_tensor_t int8_dst_add_lt = utils::logical_tensor_init(
            6, {2, 3}, {1, 2}, impl::data_type::u8);
    dequant.add_input(int8_src_lt);
    dequant.add_output(src_lt);
    reorder.add_input(src_lt);
    reorder.add_output(dst_lt);
    dequant_other.add_input(int8_src_other_lt);
    dequant_other.add_output(src_other_lt);
    add.add_input(dst_lt);
    add.add_input(src_other_lt);
    add.add_output(dst_add_lt);
    quant.add_input(dst_add_lt);
    quant.add_output(int8_dst_add_lt);

    ASSERT_EQ(agraph.add_op(&dequant), impl::status::success);
    ASSERT_EQ(agraph.add_op(&dequant_other), impl::status::success);
    ASSERT_EQ(agraph.add_op(&reorder), impl::status::success);
    ASSERT_EQ(agraph.add_op(&add), impl::status::success);
    ASSERT_EQ(agraph.add_op(&quant), impl::status::success);

    ASSERT_EQ(agraph.build_graph(), impl::status::success);

    impl::pass::pass_base_ptr apass = get_pass("int8_reorder_sum_fusion");
    apass->run(agraph);

    ASSERT_EQ(agraph.get_num_partitions(), 1);
    auto part = agraph.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);
    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {
            &int8_src_lt, &int8_src_other_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&int8_dst_add_lt};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &engine), impl::status::success);

    impl::stream_t &stream = get_stream();
    impl::tensor_t src_ts(int8_src_lt, &engine, int8_src.data());
    impl::tensor_t src_other_ts(
            int8_src_other_lt, &engine, int8_src_other.data());
    impl::tensor_t dst_ts(int8_dst_add_lt, &engine, int8_dst.data());

    cp.execute(&stream, {src_ts, src_other_ts}, {dst_ts});
    stream.wait();
    for (size_t i = 0; i < int8_src.size(); ++i) {
        ASSERT_EQ(int8_dst[i], ref_dst[i]);
    }
}

struct dnnl_graph_test_prelu_params {
    dnnl::graph::impl::dims wei_dims;
    test::vector<float> wei;
    test::vector<float> ref_dst;
    std::string data_format;
    bool per_channel_broadcast;
};

class Prelu : public ::testing::TestWithParam<dnnl_graph_test_prelu_params> {
public:
    void TestPrelu() {
        const auto params = ::testing::TestWithParam<
                dnnl_graph_test_prelu_params>::GetParam();
        impl::engine_t &eng = get_engine();

        impl::op_t op(impl::op_kind::PReLU, "prelu");
        op.set_attr<std::string>("data_format", params.data_format);
        op.set_attr<bool>(
                "per_channel_broadcast", params.per_channel_broadcast);

        test::vector<float> src {-2.0, -1.5, -1.0, -0.5, 0.0, 3.5, -1.0, 1.0};
        test::vector<float> wei = params.wei;
        test::vector<float> dst(src.size(), 0.0);

        dnnl::graph::impl::dims dims {1, 2, 2, 2};

        impl::logical_tensor_t src_lt
                = utils::logical_tensor_init(0, dims, impl::data_type::f32);
        impl::logical_tensor_t wei_lt = utils::logical_tensor_init(
                1, params.wei_dims, impl::data_type::f32);
        impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
                2, dims, impl::data_type::f32, impl::layout_type::any);

        op.add_input(src_lt);
        op.add_input(wei_lt);
        op.add_output(dst_lt);

        impl::graph_t g(eng.kind());
        g.add_op(&op);
        g.build_graph();

        impl::pass::pass_base_ptr apass = get_pass("prelu_pass");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1);
        auto part = g.get_partitions()[0];

        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> inputs {&src_lt, &wei_lt};
        std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};

        p.compile(&cp, inputs, outputs, &eng);

        impl::logical_tensor_t lt;
        cp.query_logical_tensor(dst_lt.id, &lt);

        ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

        impl::tensor_t src_ts(src_lt, &eng, src.data());
        impl::tensor_t wei_ts(wei_lt, &eng, wei.data());
        impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

        impl::stream_t &strm = get_stream();

        cp.execute(&strm, {src_ts, wei_ts}, {dst_ts});
        strm.wait();

        for (size_t i = 0; i < src.size(); ++i) {
            ASSERT_FLOAT_EQ(dst[i], params.ref_dst[i]);
        }
    }
};

TEST_P(Prelu, TestPrelu) {
    TestPrelu();
}

INSTANTIATE_TEST_SUITE_P(Execute, Prelu,
        ::testing::Values(
                // no broadcast
                dnnl_graph_test_prelu_params {{1, 2, 2, 2},
                        {2.0, 2.0, 2.0, 0.0, 0.0, 1.0, 1.0, 1.0},
                        {-4.0, -3.0, -2.0, 0.0, 0.0, 3.5, -1.0, 1.0}, "NXC",
                        false},
                // channel-shared broadcast
                dnnl_graph_test_prelu_params {{1, 1, 1, 1}, {2.0},
                        {-4.0, -3.0, -2.0, -1.0, 0.0, 3.5, -2.0, 1.0}, "NXC",
                        false},
                // shared-axes broadcast
                dnnl_graph_test_prelu_params {{1, 2, 2, 1},
                        {2.0, 1.0, 1.0, 2.0},
                        {-4.0, -3.0, -1.0, -0.5, 0.0, 3.5, -2.0, 1.0}, "NCX",
                        false},
                // channel-wise broadcast, NCX
                dnnl_graph_test_prelu_params {{1, 2, 1, 1}, {1.0, 0.0},
                        {-2.0, -1.5, -1.0, -0.5, 0.0, 3.5, 0.0, 1.0}, "NCX",
                        true},
                // channel-wise broadcast, NXC
                dnnl_graph_test_prelu_params {{1, 1, 1, 2}, {1.0, 0.0},
                        {-2.0, 0.0, -1.0, 0.0, 0.0, 3.5, -1.0, 1.0}, "NXC",
                        true},
                // 1D weights broadcast, NXC
                dnnl_graph_test_prelu_params {{2}, {1.0, 2.0},
                        {-2.0, -3.0, -1.0, -1.0, 0.0, 3.5, -1.0, 1.0}, "NXC",
                        true},
                // 1d weights, no channel-wise broadcast, NCX
                dnnl_graph_test_prelu_params {{2}, {1.0, 2.0},
                        {-2.0, -3.0, -1.0, -1.0, 0.0, 3.5, -1.0, 1.0}, "NCX",
                        false},
                // 1d weights, channel-wise broadcast, NCX
                dnnl_graph_test_prelu_params {{2}, {1.0, 2.0},
                        {-2.0, -1.5, -1.0, -0.5, 0.0, 3.5, -2.0, 1.0}, "NCX",
                        true}));

struct dnnl_graph_test_prelu_bwd_params {
    dnnl::graph::impl::dims data_dims;
    dnnl::graph::impl::dims wei_dims;
    dnnl::graph::impl::dims diff_wei_dims;
    test::vector<float> src;
    test::vector<float> wei;
    test::vector<float> diff_dst;
    test::vector<float> ref_diff_src;
    test::vector<float> ref_diff_wei;
    std::string data_format;
};

class PreluBackprop
    : public ::testing::TestWithParam<dnnl_graph_test_prelu_bwd_params> {
public:
    void TestPreluBackprop() {
        const auto params = ::testing::TestWithParam<
                dnnl_graph_test_prelu_bwd_params>::GetParam();
        impl::engine_t &eng = get_engine();
        impl::stream_t &strm = get_stream();

        test::vector<float> src = params.src;
        test::vector<float> wei = params.wei;
        test::vector<float> diff_dst = params.diff_dst;
        test::vector<float> diff_src(src.size(), 0.f);

        size_t diff_wei_size = 1;
        for (auto dim : params.diff_wei_dims) {
            diff_wei_size *= dim;
        }
        test::vector<float> diff_wei(diff_wei_size, 0.f);

        impl::op_t prelu_op(impl::op_kind::PReLUBackprop, "prelu_bwd");
        prelu_op.set_attr<std::string>("data_format", params.data_format);

        impl::logical_tensor_t src_lt = utils::logical_tensor_init(
                0, params.data_dims, impl::data_type::f32);
        impl::logical_tensor_t wei_lt = utils::logical_tensor_init(
                1, params.wei_dims, impl::data_type::f32);
        impl::logical_tensor_t diff_dst_lt = utils::logical_tensor_init(
                2, params.data_dims, impl::data_type::f32);
        impl::logical_tensor_t diff_src_lt = utils::logical_tensor_init(
                3, params.data_dims, impl::data_type::f32);
        impl::logical_tensor_t diff_wei_lt = utils::logical_tensor_init(
                4, params.diff_wei_dims, impl::data_type::f32);

        prelu_op.add_input(src_lt);
        prelu_op.add_input(wei_lt);
        prelu_op.add_input(diff_dst_lt);
        prelu_op.add_output(diff_src_lt);
        prelu_op.add_output(diff_wei_lt);

        impl::graph_t g(eng.kind());
        g.add_op(&prelu_op);
        g.build_graph();

        impl::pass::pass_base_ptr apass = get_pass("prelu_bwd_pass");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1);
        auto part = g.get_partitions()[0];

        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> inputs {
                &src_lt, &wei_lt, &diff_dst_lt};
        std::vector<const impl::logical_tensor_t *> outputs {
                &diff_src_lt, &diff_wei_lt};

        ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

        impl::tensor_t src_ts(src_lt, &eng, src.data());
        impl::tensor_t wei_ts(wei_lt, &eng, wei.data());
        impl::tensor_t diff_dst_ts(diff_dst_lt, &eng, diff_dst.data());
        impl::tensor_t diff_src_ts(diff_src_lt, &eng, diff_src.data());
        impl::tensor_t diff_wei_ts(diff_wei_lt, &eng, diff_wei.data());

        ASSERT_EQ(cp.execute(&strm, {src_ts, wei_ts, diff_dst_ts},
                          {diff_src_ts, diff_wei_ts}),
                impl::status::success);
        strm.wait();

        for (size_t i = 0; i < params.ref_diff_src.size(); ++i) {
            ASSERT_FLOAT_EQ(diff_src[i], params.ref_diff_src[i]);
        }
        for (size_t i = 0; i < params.ref_diff_wei.size(); ++i) {
            ASSERT_FLOAT_EQ(diff_wei[i], params.ref_diff_wei[i]);
        }
    }
};

TEST_P(PreluBackprop, TestPreluBackprop) {
    TestPreluBackprop();
}

INSTANTIATE_TEST_SUITE_P(Execute, PreluBackprop,
        ::testing::Values(
                // NCX, 1d slope, per tensor broadcast, pytorch case
                dnnl_graph_test_prelu_bwd_params {{1, 2, 2, 2}, {1},
                        {1, 1, 1, 1},
                        {-0.0, 0.0, 0.0, 1.0, -1.0, 1.0, 1.0, 2.0}, {-4.0},
                        {-0.0625, 0.125, 0.0, 0.0625, -0.125, 0.0, 0.0625,
                                0.125},
                        {0.25, -0.5, -0.0, 0.0625, 0.5, 0.0, 0.0625, 0.125},
                        {0.125}, "NCX"},
                // NCX, 1d slope, per channel broadcast, pytorch case
                dnnl_graph_test_prelu_bwd_params {{1, 2, 2, 2}, {2},
                        {1, 2, 1, 1},
                        {-0.0, 0.0, 0.0, 1.0, -1.0, 1.0, 1.0, 2.0}, {-4.0, 2.0},
                        {-0.0625, 0.125, 0.0, 0.0625, -0.125, 0.0, 0.0625,
                                0.125},
                        {0.25, -0.5, -0.0, 0.0625, -0.25, 0.0, 0.0625, 0.125},
                        {0.0, 0.125}, "NCX"},
                // NCX, tensorflow case
                dnnl_graph_test_prelu_bwd_params {{1, 2, 2, 2}, {2, 2, 2},
                        {1, 2, 2, 2},
                        {-0.0, 0.0, 0.0, 1.0, -1.0, 1.0, 1.0, 2.0},
                        {-4.0, 2.0, 1.0, 0.5, -0.25, 8.0, 4.0, 2.0},
                        {-0.0625, 0.125, 0.0, 0.0625, -0.125, 0.0, 0.0625,
                                0.125},
                        {0.25, 0.25, 0.0, 0.0625, 0.03125, 0.0, 0.0625, 0.125},
                        {0.0, 0.0, 0.0, 0.0, 0.125, 0.0, 0.0, 0.0}, "NCX"},
                // NXC, 1d slope, per tensor broadcast, pytorch case
                dnnl_graph_test_prelu_bwd_params {{1, 2, 2, 2}, {1},
                        {1, 1, 1, 1},
                        {-0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 2.0}, {-4.0},
                        {-0.0625, -0.125, 0.125, 0.0, 0.0, 0.0625, 0.0625,
                                0.125},
                        {0.25, 0.5, -0.5, 0.0, -0.0, 0.0625, 0.0625, 0.125},
                        {0.125}, "NXC"},
                // 2d input, per tensor broadcast
                dnnl_graph_test_prelu_bwd_params {{1, 2}, {1}, {1, 1},
                        {-0.0, 0.0}, {-4.0}, {-0.0625, 0.125}, {0.25, -0.5},
                        {0.0}, "NCX"},
                // 2d input, per channel broadcast
                dnnl_graph_test_prelu_bwd_params {{1, 2}, {2}, {1, 2},
                        {-0.0, 0.0}, {-4.0, 2.0}, {-0.0625, 0.125},
                        {0.25, 0.25}, {0.0, 0.0}, "NCX"}));

TEST(ExecuteSubgraphInt8, Relu) {
    // compare results between:
    // case 1: [quantize] - [dequantize] - [fp32_relu] - [quantize]
    // case 2: [quantize] - [int8_relu]
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    if (engine.kind() == impl::engine_kind::gpu) return;

    // prepare fp32 data
    std::vector<int64_t> shape {1, 3, 3};

    test::vector<uint8_t> src_u8_data(product(shape));
    test::vector<int8_t> case1_out_data(product(shape));
    test::vector<int8_t> case2_out_data(product(shape));

    // random generate src data
    // random seed = 7
    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    std::generate(src_u8_data.begin(), src_u8_data.end(),
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

    impl::op_t relu_op(2, impl::op_kind::ReLU, "relu_op");

    impl::op_t qout_op(3, impl::op_kind::Quantize, "qout_op");
    qout_op.set_attr<std::string>("qtype", "per_tensor");
    qout_op.set_attr<std::vector<int64_t>>("zps", {zp_out});
    qout_op.set_attr<std::vector<float>>("scales", {scale_out});
    qout_op.set_attr<int64_t>("axis", 0);

    // prepare logical tensor
    impl::logical_tensor_t src_u8
            = utils::logical_tensor_init(0, shape, impl::data_type::u8);
    impl::logical_tensor_t src_f32_dq
            = utils::logical_tensor_init(1, shape, impl::data_type::f32);
    impl::logical_tensor_t dst_f32
            = utils::logical_tensor_init(2, shape, impl::data_type::f32);
    impl::logical_tensor_t dst_u8
            = utils::logical_tensor_init(3, shape, impl::data_type::u8);

    dqdata_op.add_input(src_u8);
    dqdata_op.add_output(src_f32_dq);

    relu_op.add_input(src_f32_dq);
    relu_op.add_output(dst_f32);

    qout_op.add_input(dst_f32);
    qout_op.add_output(dst_u8);

    impl::graph_t g;
    g.add_op(&dqdata_op);
    g.add_op(&relu_op);
    g.add_op(&qout_op);
    g.build_graph();

    impl::tensor_t src_u8_ts(src_u8, &engine, src_u8_data.data());
    impl::tensor_t dst_u8_ts(dst_u8, &engine, case1_out_data.data());
    impl::tensor_t dst_u8_case2_ts(dst_u8, &engine, case2_out_data.data());

    // -------------------------case 1----------------------------------
    ASSERT_EQ(run_graph(g, {src_u8_ts}, {dst_u8_ts}, engine, strm),
            impl::status::success);

    // -------------------------case 2----------------------------------
    impl::pass::pass_base_ptr apass = get_pass("int8_relu_fusion");
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

TEST(ExecuteSubgraphInt8, ReluAdd) {
    // compare results between:
    // case 1: [quantize] - [dequantize] - [fp32_relu] - [add] - [quantize]
    // case 2: [quantize] - [int8_relu] - [int8_add]
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    // prepare fp32 data
    std::vector<int64_t> shape {1, 3, 3};
    std::vector<bool> swaps {false, true};

    test::vector<uint8_t> src_u8_data(product(shape));
    test::vector<uint8_t> src_add_u8_data(product(shape));
    test::vector<uint8_t> case1_out_data(product(shape));
    test::vector<uint8_t> case2_out_data(product(shape));

    for (auto swap : swaps) {
        // random generate src data
        // random seed = 7
        std::default_random_engine generator(7);
        std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
        std::generate(src_u8_data.begin(), src_u8_data.end(),
                [&]() { return distribution(generator); });
        std::generate(src_add_u8_data.begin(), src_add_u8_data.end(),
                [&]() { return distribution(generator); });
        float scale_src = 1 / 127.f;
        float scale_add = 1 / 127.f;
        float scale_out = 1 / 127.f;
        int64_t zp_src = 0;
        int64_t zp_add = 0;
        int64_t zp_out = 0;

        // -------------------------case 1----------------------------------

        impl::op_t dqdata_relu_op(
                1, impl::op_kind::Dequantize, "dqdata_relu_op");
        dqdata_relu_op.set_attr<std::string>("qtype", "per_tensor");
        dqdata_relu_op.set_attr<std::vector<int64_t>>("zps", {zp_src});
        dqdata_relu_op.set_attr<std::vector<float>>("scales", {scale_src});
        dqdata_relu_op.set_attr<int64_t>("axis", 0);

        impl::op_t dqdata_add_op(2, impl::op_kind::Dequantize, "dqdata_add_op");
        dqdata_add_op.set_attr<std::string>("qtype", "per_tensor");
        dqdata_add_op.set_attr<std::vector<int64_t>>("zps", {zp_add});
        dqdata_add_op.set_attr<std::vector<float>>("scales", {scale_add});
        dqdata_add_op.set_attr<int64_t>("axis", 0);

        impl::op_t relu_op(3, impl::op_kind::ReLU, "relu_op");

        impl::op_t add_op(4, impl::op_kind::Add, "add_op");

        impl::op_t qout_op(5, impl::op_kind::Quantize, "qout_op");
        qout_op.set_attr<std::string>("qtype", "per_tensor");
        qout_op.set_attr<std::vector<int64_t>>("zps", {zp_out});
        qout_op.set_attr<std::vector<float>>("scales", {scale_out});
        qout_op.set_attr<int64_t>("axis", 0);

        // prepare logical tensor
        impl::logical_tensor_t src_u8
                = utils::logical_tensor_init(0, shape, impl::data_type::u8);
        impl::logical_tensor_t src_add_u8
                = utils::logical_tensor_init(1, shape, impl::data_type::u8);
        impl::logical_tensor_t src_f32_dq
                = utils::logical_tensor_init(2, shape, impl::data_type::f32);
        impl::logical_tensor_t src_add_f32_dq
                = utils::logical_tensor_init(3, shape, impl::data_type::f32);
        impl::logical_tensor_t dst_f32
                = utils::logical_tensor_init(4, shape, impl::data_type::f32);
        impl::logical_tensor_t dst_add_f32
                = utils::logical_tensor_init(5, shape, impl::data_type::f32);
        impl::logical_tensor_t dst_u8
                = utils::logical_tensor_init(6, shape, impl::data_type::u8);

        dqdata_relu_op.add_input(src_u8);
        dqdata_relu_op.add_output(src_f32_dq);

        relu_op.add_input(src_f32_dq);
        relu_op.add_output(dst_f32);

        dqdata_add_op.add_input(src_add_u8);
        dqdata_add_op.add_output(src_add_f32_dq);

        if (swap) {
            add_op.add_input(dst_f32);
            add_op.add_input(src_add_f32_dq);
        } else {
            add_op.add_input(src_add_f32_dq);
            add_op.add_input(dst_f32);
        }
        add_op.add_output(dst_add_f32);

        qout_op.add_input(dst_add_f32);
        qout_op.add_output(dst_u8);

        impl::graph_t g(engine.kind());
        g.add_op(&dqdata_relu_op);
        g.add_op(&relu_op);
        g.add_op(&dqdata_add_op);
        g.add_op(&add_op);
        g.add_op(&qout_op);
        g.build_graph();

        impl::tensor_t src_u8_ts(src_u8, &engine, src_u8_data.data());
        impl::tensor_t src_add_u8_ts(
                src_add_u8, &engine, src_add_u8_data.data());
        impl::tensor_t dst_u8_case1_ts(dst_u8, &engine, case1_out_data.data());
        impl::tensor_t dst_u8_case2_ts(dst_u8, &engine, case2_out_data.data());

        // -------------------------case 1----------------------------------
        ASSERT_EQ(run_graph(g, {src_u8_ts, src_add_u8_ts}, {dst_u8_case1_ts},
                          engine, strm),
                impl::status::success);

        // -------------------------case 2----------------------------------
        impl::pass::pass_base_ptr apass = get_pass("int8_relu_add_fusion");
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1);
        auto part = g.get_partitions()[0];

        // compile
        impl::partition_t p;
        p.init(part);

        impl::compiled_partition_t cp(p);

        std::vector<const impl::logical_tensor_t *> lt_ins {
                &src_u8, &src_add_u8};
        std::vector<const impl::logical_tensor_t *> lt_outs {&dst_u8};

        ASSERT_EQ(p.compile(&cp, lt_ins, lt_outs, &engine),
                impl::status::success);

        cp.execute(&strm, {src_u8_ts, src_add_u8_ts}, {dst_u8_case2_ts});
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

TEST(Execute, DynamicQuantizeS32ZpsPerTensor) {
    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();

    SKIP_IF(eng.kind() == impl::engine_kind::gpu,
            "Skip dynamic quantize test for GPU device.");

    impl::op_t dync_quantize(impl::op_kind::DynamicQuantize);
    dync_quantize.set_attr<std::string>("qtype", "per_tensor");
    dync_quantize.set_attr<int64_t>("axis", 0);

    test::vector<float> src {-1.0, 0.0, 1.0, 2.0};
    test::vector<float> scales {0.1f};
    test::vector<int32_t> zps {10};
    test::vector<uint8_t> dst(src.size(), 0);
    // int8 = f32 / scales + zero_points
    test::vector<uint8_t> ref_dst {0, 10, 20, 30};

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t scales_lt
            = utils::logical_tensor_init(1, {1}, impl::data_type::f32);
    impl::logical_tensor_t zps_lt
            = utils::logical_tensor_init(2, {1}, impl::data_type::s32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(3, {1, 2, 2}, impl::data_type::u8);

    dync_quantize.add_input(src_lt);
    dync_quantize.add_input(scales_lt);
    dync_quantize.add_input(zps_lt);
    dync_quantize.add_output(dst_lt);

    impl::graph_t g(eng.kind());
    g.add_op(&dync_quantize);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("dync_quant_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {
            &src_lt, &scales_lt, &zps_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

    impl::logical_tensor_t lt;
    cp.query_logical_tensor(dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t scales_ts(scales_lt, &eng, scales.data());
    impl::tensor_t zps_ts(zps_lt, &eng, zps.data());
    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    ASSERT_EQ(cp.execute(&strm, {src_ts, scales_ts, zps_ts}, {dst_ts}),
            impl::status::success);
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, DynamicQuantizeS32ZpsPerChannel) {
    // oneDNN reorder primitive didn't support per channel asymmetric quantize
    // regression?
    SKIP_IF(true,
            "oneDNN reorder primitive didn't support per channel asymmetric "
            "quantize");

    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();

    SKIP_IF(eng.kind() == impl::engine_kind::gpu,
            "Skip dynamic quantize test for GPU device.");

    impl::op_t dync_quantize(impl::op_kind::DynamicQuantize);
    dync_quantize.set_attr<std::string>("qtype", "per_channel");
    dync_quantize.set_attr<int64_t>("axis", 1);

    test::vector<float> src {-1.0, 0.0, 1.0, 2.0};
    test::vector<float> scales {0.1f, 0.2f};
    test::vector<int32_t> zps {10, 20};
    test::vector<uint8_t> dst(src.size(), 0);
    // int8 = f32 / scales + zero_points
    test::vector<uint8_t> ref_dst {0, 10, 25, 30};

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t scales_lt
            = utils::logical_tensor_init(1, {2}, impl::data_type::f32);
    impl::logical_tensor_t zps_lt
            = utils::logical_tensor_init(2, {2}, impl::data_type::s32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(3, {1, 2, 2}, impl::data_type::u8);

    dync_quantize.add_input(src_lt);
    dync_quantize.add_input(scales_lt);
    dync_quantize.add_input(zps_lt);
    dync_quantize.add_output(dst_lt);

    impl::graph_t g(eng.kind());
    g.add_op(&dync_quantize);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("dync_quant_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {
            &src_lt, &scales_lt, &zps_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

    impl::logical_tensor_t lt;
    cp.query_logical_tensor(dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t scales_ts(scales_lt, &eng, scales.data());
    impl::tensor_t zps_ts(zps_lt, &eng, zps.data());
    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    ASSERT_EQ(cp.execute(&strm, {src_ts, scales_ts, zps_ts}, {dst_ts}),
            impl::status::success);
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, DynamicQuantizeS8ZpsPerTensor) {
    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();

    SKIP_IF(eng.kind() == impl::engine_kind::gpu,
            "Skip dynamic quantize test for GPU device.");

    impl::op_t dync_quantize(impl::op_kind::DynamicQuantize);
    dync_quantize.set_attr<std::string>("qtype", "per_tensor");
    dync_quantize.set_attr<int64_t>("axis", 0);

    test::vector<float> src {-1.0, 0.0, 1.0, 2.0};
    test::vector<float> scales {0.1f};
    test::vector<int8_t> zps {10};
    test::vector<uint8_t> dst(src.size(), 0);
    // int8 = f32 / scales + zero_points
    test::vector<uint8_t> ref_dst {0, 10, 20, 30};

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t scales_lt
            = utils::logical_tensor_init(1, {1}, impl::data_type::f32);
    impl::logical_tensor_t zps_lt
            = utils::logical_tensor_init(2, {1}, impl::data_type::s8);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(3, {1, 2, 2}, impl::data_type::u8);

    dync_quantize.add_input(src_lt);
    dync_quantize.add_input(scales_lt);
    dync_quantize.add_input(zps_lt);
    dync_quantize.add_output(dst_lt);

    impl::graph_t g(eng.kind());
    g.add_op(&dync_quantize);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("dync_quant_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {
            &src_lt, &scales_lt, &zps_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

    impl::logical_tensor_t lt;
    cp.query_logical_tensor(dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t scales_ts(scales_lt, &eng, scales.data());
    impl::tensor_t zps_ts(zps_lt, &eng, zps.data());
    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    ASSERT_EQ(cp.execute(&strm, {src_ts, scales_ts, zps_ts}, {dst_ts}),
            impl::status::success);
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, DynamicQuantizeNoZpsPerTensor) {
    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();

    SKIP_IF(eng.kind() == impl::engine_kind::gpu,
            "Skip dynamic quantize test for GPU device.");

    impl::op_t dync_quantize(impl::op_kind::DynamicQuantize);
    dync_quantize.set_attr<std::string>("qtype", "per_tensor");
    dync_quantize.set_attr<int64_t>("axis", 0);

    test::vector<float> src {-1.0, 0.0, 1.0, 2.0};
    test::vector<float> scales {0.1f};
    test::vector<int8_t> dst(src.size(), 0);
    // int8 = f32 / scales + zero_points
    test::vector<int8_t> ref_dst {-10, 0, 10, 20};

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 2}, impl::data_type::f32);
    impl::logical_tensor_t scales_lt
            = utils::logical_tensor_init(1, {1}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(3, {1, 2, 2}, impl::data_type::s8);

    dync_quantize.add_input(src_lt);
    dync_quantize.add_input(scales_lt);
    dync_quantize.add_output(dst_lt);

    impl::graph_t g(eng.kind());
    g.add_op(&dync_quantize);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("dync_quant_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src_lt, &scales_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

    impl::logical_tensor_t lt;
    cp.query_logical_tensor(dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t scales_ts(scales_lt, &eng, scales.data());
    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    ASSERT_EQ(cp.execute(&strm, {src_ts, scales_ts}, {dst_ts}),
            impl::status::success);
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, DynamicDequantizeS32ZpsPerTensor) {
    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();

    SKIP_IF(eng.kind() == impl::engine_kind::gpu,
            "Skip dynamic quantize test for GPU device.");

    impl::op_t dync_dequantize(impl::op_kind::DynamicDequantize);
    dync_dequantize.set_attr<std::string>("qtype", "per_tensor");
    dync_dequantize.set_attr<int64_t>("axis", 0);

    test::vector<uint8_t> src {0, 10, 20, 30};
    test::vector<float> scales {0.1f};
    test::vector<int32_t> zps {10};
    test::vector<float> dst(src.size(), 0);
    // f32 = scales * (int8 - zero_points)
    test::vector<float> ref_dst {-1.0, 0.0, 1.0, 2.0};

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 2}, impl::data_type::u8);
    impl::logical_tensor_t scales_lt
            = utils::logical_tensor_init(1, {1}, impl::data_type::f32);
    impl::logical_tensor_t zps_lt
            = utils::logical_tensor_init(2, {1}, impl::data_type::s32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(3, {1, 2, 2}, impl::data_type::f32);

    dync_dequantize.add_input(src_lt);
    dync_dequantize.add_input(scales_lt);
    dync_dequantize.add_input(zps_lt);
    dync_dequantize.add_output(dst_lt);

    impl::graph_t g(eng.kind());
    g.add_op(&dync_dequantize);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("dync_dequant_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {
            &src_lt, &scales_lt, &zps_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

    impl::logical_tensor_t lt;
    cp.query_logical_tensor(dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t scales_ts(scales_lt, &eng, scales.data());
    impl::tensor_t zps_ts(zps_lt, &eng, zps.data());
    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    ASSERT_EQ(cp.execute(&strm, {src_ts, scales_ts, zps_ts}, {dst_ts}),
            impl::status::success);
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Execute, DynamicDequantizeNoZpsPerTensor) {
    // default engine kind is cpu.
    impl::engine_t &eng = get_engine();

    SKIP_IF(eng.kind() == impl::engine_kind::gpu,
            "Skip dynamic quantize test for GPU device.");

    impl::op_t dync_dequantize(impl::op_kind::DynamicDequantize);
    dync_dequantize.set_attr<std::string>("qtype", "per_tensor");
    dync_dequantize.set_attr<int64_t>("axis", 0);

    test::vector<uint8_t> src {0, 10, 20, 30};
    test::vector<float> scales {0.1f};
    test::vector<float> dst(src.size(), 0);
    // f32 = scales * (int8 - zero_points)
    test::vector<float> ref_dst {0, 1.0, 2.0, 3.0};

    // prepare logical tensor
    impl::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {1, 2, 2}, impl::data_type::u8);
    impl::logical_tensor_t scales_lt
            = utils::logical_tensor_init(1, {1}, impl::data_type::f32);
    impl::logical_tensor_t dst_lt
            = utils::logical_tensor_init(3, {1, 2, 2}, impl::data_type::f32);

    dync_dequantize.add_input(src_lt);
    dync_dequantize.add_input(scales_lt);
    dync_dequantize.add_output(dst_lt);

    impl::graph_t g(eng.kind());
    g.add_op(&dync_dequantize);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("dync_dequant_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src_lt, &scales_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

    impl::logical_tensor_t lt;
    cp.query_logical_tensor(dst_lt.id, &lt);
    ASSERT_EQ(lt.layout_type, impl::layout_type::strided);

    impl::tensor_t src_ts(src_lt, &eng, src.data());
    impl::tensor_t scales_ts(scales_lt, &eng, scales.data());
    impl::tensor_t dst_ts(dst_lt, &eng, dst.data());

    impl::stream_t &strm = get_stream();
    ASSERT_EQ(cp.execute(&strm, {src_ts, scales_ts}, {dst_ts}),
            impl::status::success);
    strm.wait();
    for (size_t i = 0; i < dst.size(); ++i) {
        ASSERT_FLOAT_EQ(dst[i], ref_dst[i]);
    }
}

TEST(Compile, SoftmaxGetInplacePair) {
    impl::engine_t &eng = get_engine();

    impl::op_t softmax_op(impl::op_kind::SoftMax);
    softmax_op.set_attr<int64_t>("axis", 0);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {2, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst
            = utils::logical_tensor_init(1, {2, 3}, impl::data_type::f32);

    softmax_op.add_input(src);
    softmax_op.add_output(dst);

    impl::graph_t g(eng.kind());
    g.add_op(&softmax_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("softmax_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];
    impl::partition_t p;
    p.init(part);

    // compile
    std::vector<const impl::logical_tensor_t *> inputs {&src};
    std::vector<const impl::logical_tensor_t *> outputs {&dst};

    impl::compiled_partition_t cp(p);
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

    auto pairs = cp.get_inplace_pairs();
    ASSERT_EQ(pairs.size(), 1);
    ASSERT_EQ(pairs[0].input, 0);
    ASSERT_EQ(pairs[0].output, 1);
}

TEST(Compile, EltwiseGetInplacePair) {
    // TODO(qun): re-enable this test once library and bridge align the inplace
    // logic
    SKIP_IF(true, "library and bridge have different inplace logic");

    impl::engine_t &eng = get_engine();

    impl::op_t eltwise_op(impl::op_kind::Tanh);

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, {2, 3}, impl::data_type::f32);
    impl::logical_tensor_t dst
            = utils::logical_tensor_init(1, {2, 3}, impl::data_type::f32);

    eltwise_op.add_input(src);
    eltwise_op.add_output(dst);

    impl::graph_t g(eng.kind());
    g.add_op(&eltwise_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("tanh_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];
    impl::partition_t p;
    p.init(part);

    // compile
    std::vector<const impl::logical_tensor_t *> inputs {&src};
    std::vector<const impl::logical_tensor_t *> outputs {&dst};

    impl::compiled_partition_t cp(p);
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

    auto pairs = cp.get_inplace_pairs();
    ASSERT_EQ(pairs.size(), 1);
    ASSERT_EQ(pairs[0].input, 0);
    ASSERT_EQ(pairs[0].output, 1);
}

TEST(Execute, F32ConvolutionalBottleneckResBlock) {
    impl::engine_t &eng = get_engine();
    impl::stream_t &strm = get_stream();

    utils::id_generator id_gen;
    impl::graph_t g(eng.kind());
    utils::construct_convolutional_bottleneck_resblock(&g, id_gen);
    g.build_graph();

    ASSERT_EQ(g.get_ops().size(), 8);

    impl::pass::pass_base_ptr apass
            = get_pass("convolutional_bottleneck_resblock_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(partition_inputs.size(), 10);
    ASSERT_EQ(partition_outputs.size(), 1);

    std::vector<const impl::logical_tensor_t *> inputs, outputs;
    for (auto &lt : partition_inputs) {
        // skip alias inputs
        auto pos = std::find_if(inputs.begin(), inputs.end(),
                [&](const impl::logical_tensor_t *item) {
                    return item->id == lt.id;
                });
        if (pos != inputs.end()) continue;
        inputs.emplace_back(&lt);
    }
    for (auto &lt : partition_outputs) {
        // set output to be strided
        lt = utils::logical_tensor_init(
                lt.id, lt.data_type, impl::layout_type::strided);
        outputs.emplace_back(&lt);
    }

    impl::compiled_partition_t cp(p);
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

    using ltw = impl::logical_tensor_wrapper_t;

    std::vector<test::vector<float>> inputs_data, outputs_data;
    std::vector<impl::tensor_t> inputs_ts, outputs_ts;

    for (auto &lt : inputs) {
        inputs_data.emplace_back(
                test::vector<float>(utils::product(ltw(lt).vdims())));
        inputs_ts.emplace_back(*lt, &eng, inputs_data.back().data());
    }

    for (auto &lt : outputs) {
        impl::logical_tensor_t compiled_output;
        cp.query_logical_tensor(lt->id, &compiled_output);
        outputs_data.emplace_back(test::vector<float>(
                utils::product(ltw(compiled_output).vdims())));
        outputs_ts.emplace_back(
                compiled_output, &eng, outputs_data.back().data());
    }

    ASSERT_EQ(cp.execute(&strm, inputs_ts, outputs_ts), impl::status::success);
    std::cout << "-------------------\n";
    ASSERT_EQ(cp.execute(&strm, inputs_ts, outputs_ts), impl::status::success);

    strm.wait();
}

TEST(Execute, Int8IdenticalBottleneckResBlock) {
    impl::engine_t &eng = get_engine();
    impl::stream_t &strm = get_stream();

    utils::id_generator id_gen;
    impl::graph_t g(eng.kind());
    utils::construct_int8_identical_bottleneck_resblock(&g, id_gen);
    g.build_graph();

    ASSERT_EQ(g.get_ops().size(), 17);

    impl::pass::pass_base_ptr apass
            = get_pass("int8_identical_bottleneck_resblock_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(partition_inputs.size(), 8);
    ASSERT_EQ(partition_outputs.size(), 1);

    std::vector<const impl::logical_tensor_t *> inputs, outputs;
    for (auto &lt : partition_inputs) {
        inputs.emplace_back(&lt);
    }
    for (auto &lt : partition_outputs) {
        // set output to be strided
        lt = utils::logical_tensor_init(
                lt.id, lt.data_type, impl::layout_type::strided);
        outputs.emplace_back(&lt);
    }

    impl::compiled_partition_t cp(p);
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

    using ltw = impl::logical_tensor_wrapper_t;

    std::vector<test::vector<float>> inputs_data, outputs_data;
    std::vector<impl::tensor_t> inputs_ts, outputs_ts;

    for (auto &lt : inputs) {
        inputs_data.emplace_back(
                test::vector<float>(utils::product(ltw(lt).vdims())));
        inputs_ts.emplace_back(*lt, &eng, inputs_data.back().data());
    }

    for (auto &lt : outputs) {
        impl::logical_tensor_t compiled_output;
        cp.query_logical_tensor(lt->id, &compiled_output);
        outputs_data.emplace_back(test::vector<float>(
                utils::product(ltw(compiled_output).vdims())));
        outputs_ts.emplace_back(
                compiled_output, &eng, outputs_data.back().data());
    }

    ASSERT_EQ(cp.execute(&strm, inputs_ts, outputs_ts), impl::status::success);
    strm.wait();
}

TEST(Execute, Int8ConvBiasReluConvBiasReluConvBiasConvBiasAddReluBlock) {
    impl::engine_t &eng = get_engine();
    impl::stream_t &strm = get_stream();

    utils::id_generator id_gen;
    impl::graph_t g(eng.kind());
    utils::construct_int8_convolutional_bottleneck_resblock(&g, id_gen);
    g.build_graph();

    ASSERT_EQ(g.get_ops().size(), 21);

    impl::pass::pass_base_ptr apass
            = get_pass("int8_convolutional_bottleneck_resblock_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(partition_inputs.size(), 10);
    ASSERT_EQ(partition_outputs.size(), 1);

    std::vector<const impl::logical_tensor_t *> inputs, outputs;
    for (auto &lt : partition_inputs) {
        inputs.emplace_back(&lt);
    }
    for (auto &lt : partition_outputs) {
        // set output to be strided
        lt = utils::logical_tensor_init(
                lt.id, lt.data_type, impl::layout_type::strided);
        outputs.emplace_back(&lt);
    }

    impl::compiled_partition_t cp(p);
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

    using ltw = impl::logical_tensor_wrapper_t;

    std::vector<test::vector<float>> inputs_data, outputs_data;
    std::vector<impl::tensor_t> inputs_ts, outputs_ts;

    for (auto &lt : inputs) {
        inputs_data.emplace_back(
                test::vector<float>(utils::product(ltw(lt).vdims())));
        inputs_ts.emplace_back(*lt, &eng, inputs_data.back().data());
    }

    for (auto &lt : outputs) {
        impl::logical_tensor_t compiled_output;
        cp.query_logical_tensor(lt->id, &compiled_output);
        outputs_data.emplace_back(test::vector<float>(
                utils::product(ltw(compiled_output).vdims())));
        outputs_ts.emplace_back(
                compiled_output, &eng, outputs_data.back().data());
    }

    ASSERT_EQ(cp.execute(&strm, inputs_ts, outputs_ts), impl::status::success);
    strm.wait();
}

TEST(Execute, Int8Resnet50Stage2Block) {
    impl::engine_t &eng = get_engine();
    impl::stream_t &strm = get_stream();

    utils::id_generator id_gen;
    impl::graph_t g(eng.kind());
    utils::construct_int8_resnet50_stage2_block(&g, id_gen, 3);
    g.build_graph();

    ASSERT_EQ(g.get_ops().size(), 72);

    impl::pass::pass_base_ptr apass = get_pass("int8_resnet50_stage_2_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(partition_inputs.size(), 28);
    ASSERT_EQ(partition_outputs.size(), 1);

    std::vector<const impl::logical_tensor_t *> inputs, outputs;
    for (auto &lt : partition_inputs) {
        inputs.emplace_back(&lt);
    }
    for (auto &lt : partition_outputs) {
        // set output to be strided
        lt = utils::logical_tensor_init(
                lt.id, lt.data_type, impl::layout_type::strided);
        outputs.emplace_back(&lt);
    }

    impl::compiled_partition_t cp(p);
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

    using ltw = impl::logical_tensor_wrapper_t;

    std::vector<test::vector<float>> inputs_data, outputs_data;
    std::vector<impl::tensor_t> inputs_ts, outputs_ts;

    for (auto &lt : inputs) {
        inputs_data.emplace_back(
                test::vector<float>(utils::product(ltw(lt).vdims())));
        inputs_ts.emplace_back(*lt, &eng, inputs_data.back().data());
    }

    for (auto &lt : outputs) {
        impl::logical_tensor_t compiled_output;
        cp.query_logical_tensor(lt->id, &compiled_output);
        outputs_data.emplace_back(test::vector<float>(
                utils::product(ltw(compiled_output).vdims())));
        outputs_ts.emplace_back(
                compiled_output, &eng, outputs_data.back().data());
    }

    ASSERT_EQ(cp.execute(&strm, inputs_ts, outputs_ts), impl::status::success);
    strm.wait();
}

TEST(Compile, Int8ConvBlockGetInplacePair) {
    impl::engine_t &eng = get_engine();

    utils::id_generator id_gen;
    impl::graph_t g(eng.kind());
    utils::construct_int8_convolutional_bottleneck_resblock(&g, id_gen);
    g.build_graph();

    ASSERT_EQ(g.get_ops().size(), 21);

    impl::pass::pass_base_ptr apass1
            = get_pass("int8_identical_bottleneck_resblock_fusion");
    impl::pass::pass_base_ptr apass2 = get_pass("int8_conv_post_ops_fusion");
    apass1->run(g);
    apass2->run(g);
    ASSERT_EQ(g.get_num_partitions(), 2);
    auto part0 = g.get_partitions()[0];
    auto part1 = g.get_partitions()[1];

    // compile part1 (single conv)
    impl::partition_t p1;
    p1.init(part1);

    auto partition_inputs1 = p1.get_inputs();
    auto partition_outputs1 = p1.get_outputs();
    ASSERT_EQ(partition_inputs1.size(), 3);
    ASSERT_EQ(partition_outputs1.size(), 1);

    std::vector<const impl::logical_tensor_t *> inputs1, outputs1;
    for (auto &lt : partition_inputs1) {
        inputs1.emplace_back(&lt);
    }
    for (auto &lt : partition_outputs1) {
        // set output to be any
        lt = utils::logical_tensor_init(
                lt.id, lt.data_type, impl::layout_type::any);
        outputs1.emplace_back(&lt);
    }

    impl::compiled_partition_t cp1(p1);
    ASSERT_EQ(p1.compile(&cp1, inputs1, outputs1, &eng), impl::status::success);

    // compile part0 (three conv, one with sum)
    impl::partition_t p0;
    p0.init(part0);

    auto partition_inputs0 = p0.get_inputs();
    auto partition_outputs0 = p0.get_outputs();
    ASSERT_EQ(partition_inputs0.size(), 8);
    ASSERT_EQ(partition_outputs0.size(), 1);

    std::vector<const impl::logical_tensor_t *> inputs0, outputs0;
    for (auto &lt : partition_inputs0) {
        if (lt.id == outputs1[0]->id) {
            // use precursor's output
            cp1.query_logical_tensor(lt.id, &lt);
        }
        inputs0.emplace_back(&lt);
    }
    for (auto &lt : partition_outputs0) {
        // set output to be any
        lt = utils::logical_tensor_init(
                lt.id, lt.data_type, impl::layout_type::any);
        outputs0.emplace_back(&lt);
    }

    impl::compiled_partition_t cp0(p0);
    ASSERT_EQ(p0.compile(&cp0, inputs0, outputs0, &eng), impl::status::success);
    auto pairs = cp0.get_inplace_pairs();
    ASSERT_EQ(pairs.size(), 1);
    ASSERT_EQ(pairs[0].input, outputs1[0]->id);
    ASSERT_EQ(pairs[0].output, outputs0[0]->id);
}

TEST(Execute, F32Resnet50Stage2Block) {
    impl::engine_t &eng = get_engine();
    impl::stream_t &strm = get_stream();

    utils::id_generator id_gen;
    impl::graph_t g(eng.kind());
    utils::construct_f32_resnet50_stage2_block(
            &g, id_gen, 3, /* use biasadd */ true);
    g.build_graph();

    ASSERT_EQ(g.get_ops().size(), 42);

    impl::pass::pass_base_ptr apass = get_pass("f32_resnet50_stage_2_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(partition_inputs.size(), 28);
    ASSERT_EQ(partition_outputs.size(), 1);

    std::vector<const impl::logical_tensor_t *> inputs, outputs;
    for (auto &lt : partition_inputs) {
        inputs.emplace_back(&lt);
    }
    for (auto &lt : partition_outputs) {
        // set output to be strided
        lt = utils::logical_tensor_init(
                lt.id, lt.data_type, impl::layout_type::strided);
        outputs.emplace_back(&lt);
    }

    impl::compiled_partition_t cp(p);
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

    using ltw = impl::logical_tensor_wrapper_t;

    std::vector<test::vector<float>> inputs_data, outputs_data;
    std::vector<impl::tensor_t> inputs_ts, outputs_ts;

    for (auto &lt : inputs) {
        inputs_data.emplace_back(
                test::vector<float>(utils::product(ltw(lt).vdims())));
        inputs_ts.emplace_back(*lt, &eng, inputs_data.back().data());
    }

    for (auto &lt : outputs) {
        impl::logical_tensor_t compiled_output;
        cp.query_logical_tensor(lt->id, &compiled_output);
        outputs_data.emplace_back(test::vector<float>(
                utils::product(ltw(compiled_output).vdims())));
        outputs_ts.emplace_back(
                compiled_output, &eng, outputs_data.back().data());
    }

    std::cout << "----------------iter 1----------------\n";
    ASSERT_EQ(cp.execute(&strm, inputs_ts, outputs_ts), impl::status::success);
    std::cout << "----------------iter 2----------------\n";
    ASSERT_EQ(cp.execute(&strm, inputs_ts, outputs_ts), impl::status::success);
    strm.wait();
}

TEST(Execute, ItexInt8Resnet50Stage2Block) {
    impl::engine_t &eng = get_engine();
    impl::stream_t &strm = get_stream();

    utils::id_generator id_gen;
    impl::graph_t g(eng.kind());
    utils::construct_itex_int8_resnet50_stage2_block(&g, id_gen, 3);
    g.build_graph();

    ASSERT_EQ(g.get_ops().size(), 98);

    impl::pass::pass_base_ptr apass
            = get_pass("itex_int8_resnet50_stage_2_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(partition_inputs.size(), 28);
    ASSERT_EQ(partition_outputs.size(), 1);

    std::vector<const impl::logical_tensor_t *> inputs, outputs;
    for (auto &lt : partition_inputs) {
        inputs.emplace_back(&lt);
    }
    for (auto &lt : partition_outputs) {
        // set output to be strided
        lt = utils::logical_tensor_init(
                lt.id, lt.data_type, impl::layout_type::strided);
        outputs.emplace_back(&lt);
    }

    impl::compiled_partition_t cp(p);
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

    using ltw = impl::logical_tensor_wrapper_t;

    std::vector<test::vector<float>> inputs_data, outputs_data;
    std::vector<impl::tensor_t> inputs_ts, outputs_ts;

    for (auto &lt : inputs) {
        inputs_data.emplace_back(
                test::vector<float>(utils::product(ltw(lt).vdims())));
        inputs_ts.emplace_back(*lt, &eng, inputs_data.back().data());
    }

    for (auto &lt : outputs) {
        impl::logical_tensor_t compiled_output;
        cp.query_logical_tensor(lt->id, &compiled_output);
        outputs_data.emplace_back(test::vector<float>(
                utils::product(ltw(compiled_output).vdims())));
        outputs_ts.emplace_back(
                compiled_output, &eng, outputs_data.back().data());
    }

    std::cout << "----------------iter 1----------------\n";
    ASSERT_EQ(cp.execute(&strm, inputs_ts, outputs_ts), impl::status::success);
    std::cout << "----------------iter 2----------------\n";
    ASSERT_EQ(cp.execute(&strm, inputs_ts, outputs_ts), impl::status::success);
    strm.wait();
}

TEST(ExecuteSubgraphInt8, U8u8bf16DivBmm) {
    impl::engine_t &engine = get_engine();
    impl::stream_t &strm = get_stream();

    std::vector<int64_t> src_shape = {1, 12, 128, 64};
    std::vector<int64_t> weight_shape = {1, 12, 64, 128};
    std::vector<int64_t> dst_shape = {1, 12, 128, 128};

    test::vector<uint8_t> src_data(product(src_shape));
    test::vector<uint8_t> weight_data(product(weight_shape));

    // u8 2 s8 shift by using reorder with -128 zps is not supported on
    // GPU
    SKIP_IF(engine.kind() == impl::engine_kind::gpu, "Skip on GPU device.");

    // random generate src, weight data
    // random seed = 7
    std::default_random_engine generator(7);
    std::uniform_real_distribution<float> src_distribution(0.0f, 255.f);
    std::uniform_real_distribution<float> weight_distribution(0.0f, 255.f);
    std::generate(src_data.begin(), src_data.end(),
            [&]() { return src_distribution(generator); });
    std::generate(weight_data.begin(), weight_data.end(),
            [&]() { return weight_distribution(generator); });
    float scale_src = 1 / 255.f;
    float scale_wei = 1 / 255.f;
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
    impl::op_t tcweight_op {3, impl::op_kind::TypeCast, "typecast_weight"};

    impl::op_t matmul_op(4, impl::op_kind::MatMul, "matmul_op");
    matmul_op.set_attr<bool>("transpose_a", false);
    matmul_op.set_attr<bool>("transpose_b", false);

    impl::op_t binary_op(5, impl::op_kind::Divide, "binary_div");
    binary_op.set_attr<std::string>("auto_broadcast", "numpy");

    // prepare logical tensor
    impl::logical_tensor_t src
            = utils::logical_tensor_init(0, src_shape, impl::data_type::u8);
    impl::logical_tensor_t src_f32_dq
            = utils::logical_tensor_init(1, src_shape, impl::data_type::f32);
    impl::logical_tensor_t src_bf16
            = utils::logical_tensor_init(2, src_shape, impl::data_type::bf16);
    impl::logical_tensor_t weight
            = utils::logical_tensor_init(3, weight_shape, impl::data_type::u8);
    impl::logical_tensor_t weight_f32_dq
            = utils::logical_tensor_init(4, weight_shape, impl::data_type::f32);
    impl::logical_tensor_t weight_bf16 = utils::logical_tensor_init(
            5, weight_shape, impl::data_type::bf16);
    impl::logical_tensor_t dst_bf16
            = utils::logical_tensor_init(6, dst_shape, impl::data_type::bf16);
    impl::logical_tensor_t div_src1
            = utils::logical_tensor_init(7, {1}, impl::data_type::bf16);
    impl::logical_tensor_t div_bf16
            = utils::logical_tensor_init(8, src_shape, impl::data_type::bf16);

    dqdata_op.add_input(src);
    dqdata_op.add_output(src_f32_dq);

    dqweight_op.add_input(weight);
    dqweight_op.add_output(weight_f32_dq);

    tcdata_op.add_input(src_f32_dq);
    tcdata_op.add_output(src_bf16);

    tcweight_op.add_input(weight_f32_dq);
    tcweight_op.add_output(weight_bf16);

    binary_op.add_input(src_bf16);
    binary_op.add_input(div_src1);
    binary_op.add_output(div_bf16);

    matmul_op.add_input(div_bf16);
    matmul_op.add_input(weight_bf16);
    matmul_op.add_output(dst_bf16);

    impl::graph_t g(engine.kind());
    ASSERT_EQ(g.add_op(&dqdata_op), impl::status::success);
    ASSERT_EQ(g.add_op(&dqweight_op), impl::status::success);
    ASSERT_EQ(g.add_op(&matmul_op), impl::status::success);
    ASSERT_EQ(g.add_op(&tcdata_op), impl::status::success);
    ASSERT_EQ(g.add_op(&tcweight_op), impl::status::success);
    ASSERT_EQ(g.add_op(&binary_op), impl::status::success);
    ASSERT_EQ(g.build_graph(), impl::status::success);

    impl::pass::pass_base_ptr apass = get_pass("x8x8bf16_div_matmul_fusion");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);

    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> lt_ins {
            &src, &weight, &div_src1};
    std::vector<const impl::logical_tensor_t *> lt_outs {&dst_bf16};

    ASSERT_EQ(p.compile(&cp, lt_ins, lt_outs, &engine), impl::status::success);

    test::vector<float> div_src1_data(1);
    test::vector<float> dst_data(product(dst_shape));
    impl::tensor_t src_ts(src, &engine, src_data.data());
    impl::tensor_t weight_ts(weight, &engine, weight_data.data());
    impl::tensor_t div_src1_ts(div_src1, &engine, div_src1_data.data());
    impl::tensor_t dst_ts(dst_bf16, &engine, dst_data.data());
    ASSERT_EQ(cp.execute(&strm, {src_ts, weight_ts, div_src1_ts}, {dst_ts}),
            impl::status::success);
    strm.wait();
}
