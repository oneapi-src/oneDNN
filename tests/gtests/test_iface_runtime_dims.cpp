/*******************************************************************************
* Copyright 2019-2022 Intel Corporation
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

#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include "oneapi/dnnl/dnnl.hpp"

namespace dnnl {

// short names for brevity
using data_type = memory::data_type;
using tag = memory::format_tag;

class runtime_dim_test_t : public ::testing::Test {
protected:
    engine eng = get_test_engine();
    void SetUp() override {}

    template <typename F>
    void check_status(const F &f, dnnl_status_t status) {
        catch_expected_failures(f, status != dnnl_success, status, false);
    }
};
#define CHECK_STATUs(status, ...) check_status([&]() { __VA_ARGS__; }, status)
#define CHECK_STATUS(status, ...) CHECK_STATUs(status, __VA_ARGS__)

#define CHECK_OK(...) CHECK_STATUS(dnnl_success, __VA_ARGS__)
#define CHECK_INVALID(...) CHECK_STATUS(dnnl_invalid_arguments, __VA_ARGS__)
#define CHECK_UNIMPL(...) CHECK_STATUS(dnnl_unimplemented, __VA_ARGS__)

TEST_F(runtime_dim_test_t, TestMemory) {
    memory::desc md_tag {{DNNL_RUNTIME_DIM_VAL, DNNL_RUNTIME_DIM_VAL},
            data_type::f32, tag::ab};
    ASSERT_EQ(md_tag.get_size(), DNNL_RUNTIME_SIZE_VAL);
    CHECK_INVALID(test::make_memory(md_tag, eng));

    memory::desc md_strides {{DNNL_RUNTIME_DIM_VAL, DNNL_RUNTIME_DIM_VAL},
            data_type::f32, {100, 1}};
    ASSERT_EQ(md_strides.get_size(), DNNL_RUNTIME_SIZE_VAL);
    CHECK_INVALID(test::make_memory(md_strides, eng));
}

TEST_F(runtime_dim_test_t, TestBNorm) {
    memory::desc md {
            {DNNL_RUNTIME_DIM_VAL, 16, 3, 3}, data_type::f32, tag::abcd};
    normalization_flags flags {};
    CHECK_UNIMPL(batch_normalization_forward::primitive_desc(
            eng, prop_kind::forward, md, md, 0.1f, flags));

    batch_normalization_forward::primitive_desc fwd_hint;
    {
        auto valid_md = memory::desc({2, 16, 3, 3}, data_type::f32, tag::abcd);
        CHECK_OK(fwd_hint = batch_normalization_forward::primitive_desc(eng,
                         prop_kind::forward, valid_md, valid_md, 0.1f, flags));
    }
    CHECK_UNIMPL(batch_normalization_backward::primitive_desc(
            eng, prop_kind::backward_data, md, md, md, 0.1f, flags, fwd_hint));
}

TEST_F(runtime_dim_test_t, TestBinary) {
    memory::desc md {
            {DNNL_RUNTIME_DIM_VAL, 16, 3, 3}, data_type::f32, tag::abcd};
    CHECK_UNIMPL(
            binary::primitive_desc(eng, algorithm::binary_add, md, md, md));
}

TEST_F(runtime_dim_test_t, TestConcat) {
    memory::desc md {
            {DNNL_RUNTIME_DIM_VAL, 16, 3, 3}, data_type::f32, tag::abcd};
    CHECK_UNIMPL(concat::primitive_desc(eng, 1, {md, md}));
}

TEST_F(runtime_dim_test_t, TestConv) {
    memory::desc src_md {
            {DNNL_RUNTIME_DIM_VAL, 16, 7, 7}, data_type::f32, tag::abcd};
    memory::desc wei_md {{32, 16, 3, 3}, data_type::f32, tag::abcd};
    memory::desc dst_md {
            {DNNL_RUNTIME_DIM_VAL, 32, 7, 7}, data_type::f32, tag::abcd};
    CHECK_UNIMPL(convolution_forward::primitive_desc(eng, prop_kind::forward,
            algorithm::convolution_direct, src_md, wei_md, dst_md, {1, 1},
            {1, 1}, {1, 1}));

    convolution_forward::primitive_desc fwd_hint;
    {
        auto valid_src_md
                = memory::desc({2, 16, 7, 7}, data_type::f32, tag::abcd);
        auto valid_dst_md
                = memory::desc({2, 32, 7, 7}, data_type::f32, tag::abcd);
        CHECK_OK(fwd_hint
                = convolution_forward::primitive_desc(eng, prop_kind::forward,
                        algorithm::convolution_direct, valid_src_md, wei_md,
                        valid_dst_md, {1, 1}, {1, 1}, {1, 1}));
    }

    CHECK_UNIMPL(convolution_backward_data::primitive_desc(eng,
            algorithm::convolution_direct, src_md, wei_md, dst_md, {1, 1},
            {1, 1}, {1, 1}, fwd_hint));
    CHECK_UNIMPL(convolution_backward_weights::primitive_desc(eng,
            algorithm::convolution_direct, src_md, wei_md, dst_md, {1, 1},
            {1, 1}, {1, 1}, fwd_hint));
}

TEST_F(runtime_dim_test_t, TestDeconv) {
    memory::desc src_md {
            {DNNL_RUNTIME_DIM_VAL, 16, 7, 7}, data_type::f32, tag::abcd};
    memory::desc wei_md {{32, 16, 3, 3}, data_type::f32, tag::abcd};
    memory::desc dst_md {
            {DNNL_RUNTIME_DIM_VAL, 32, 7, 7}, data_type::f32, tag::abcd};
    CHECK_UNIMPL(deconvolution_forward::primitive_desc(eng, prop_kind::forward,
            algorithm::deconvolution_direct, src_md, wei_md, dst_md, {1, 1},
            {1, 1}, {1, 1}));

    deconvolution_forward::primitive_desc fwd_hint;
    {
        auto valid_src_md
                = memory::desc({2, 16, 7, 7}, data_type::f32, tag::abcd);
        auto valid_dst_md
                = memory::desc({2, 32, 7, 7}, data_type::f32, tag::abcd);
        CHECK_OK(fwd_hint
                = deconvolution_forward::primitive_desc(eng, prop_kind::forward,
                        algorithm::deconvolution_direct, valid_src_md, wei_md,
                        valid_dst_md, {1, 1}, {1, 1}, {1, 1}));
    }
    CHECK_UNIMPL(deconvolution_backward_data::primitive_desc(eng,
            algorithm::deconvolution_direct, src_md, wei_md, dst_md, {1, 1},
            {1, 1}, {1, 1}, fwd_hint));
    CHECK_UNIMPL(deconvolution_backward_weights::primitive_desc(eng,
            algorithm::deconvolution_direct, src_md, wei_md, dst_md, {1, 1},
            {1, 1}, {1, 1}, fwd_hint));
}

TEST_F(runtime_dim_test_t, TestEltwise) {
    memory::desc md {
            {DNNL_RUNTIME_DIM_VAL, 16, 3, 3}, data_type::f32, tag::abcd};
    CHECK_UNIMPL(eltwise_forward::primitive_desc(
            eng, prop_kind::forward, algorithm::eltwise_relu, md, md, 0.1f));

    eltwise_forward::primitive_desc fwd_hint;
    {
        auto valid_md = memory::desc({2, 16, 3, 3}, data_type::f32, tag::abcd);
        CHECK_OK(fwd_hint
                = eltwise_forward::primitive_desc(eng, prop_kind::forward,
                        algorithm::eltwise_relu, valid_md, valid_md, 0.1f));
    }

    CHECK_UNIMPL(eltwise_backward::primitive_desc(
            eng, algorithm::eltwise_relu, md, md, md, 0.1f, fwd_hint));
}

TEST_F(runtime_dim_test_t, TestInnerProduct) {
    memory::desc src_md {
            {DNNL_RUNTIME_DIM_VAL, 16, 7, 7}, data_type::f32, tag::abcd};
    memory::desc wei_md {{32, 16, 7, 7}, data_type::f32, tag::abcd};
    memory::desc dst_md {{DNNL_RUNTIME_DIM_VAL, 32}, data_type::f32, tag::ab};
    CHECK_UNIMPL(inner_product_forward::primitive_desc(
            eng, prop_kind::forward, src_md, wei_md, dst_md));

    inner_product_forward::primitive_desc fwd_hint;
    {
        auto valid_src_md
                = memory::desc({2, 16, 7, 7}, data_type::f32, tag::abcd);
        auto valid_dst_md = memory::desc({2, 32}, data_type::f32, tag::ab);
        CHECK_OK(fwd_hint
                = inner_product_forward::primitive_desc(eng, prop_kind::forward,
                        valid_src_md, wei_md, valid_dst_md));
    }

    CHECK_UNIMPL(inner_product_backward_data::primitive_desc(
            eng, src_md, wei_md, dst_md, fwd_hint));
    CHECK_UNIMPL(inner_product_backward_weights::primitive_desc(
            eng, src_md, wei_md, dst_md, fwd_hint));
}

TEST_F(runtime_dim_test_t, TestLNorm) {
    memory::desc md {{DNNL_RUNTIME_DIM_VAL, 16, 16}, data_type::f32, tag::abc};
    memory::desc stat_md {{DNNL_RUNTIME_DIM_VAL, 16}, data_type::f32, tag::ab};
    normalization_flags flags {};
    CHECK_UNIMPL(layer_normalization_forward::primitive_desc(
            eng, prop_kind::forward, md, md, stat_md, 0.1f, flags));

    layer_normalization_forward::primitive_desc fwd_hint;
    {
        auto valid_md = memory::desc({2, 16, 16}, data_type::f32, tag::abc);
        auto valid_stat_md = memory::desc({2, 16}, data_type::f32, tag::ab);
        CHECK_OK(fwd_hint = layer_normalization_forward::primitive_desc(eng,
                         prop_kind::forward, valid_md, valid_md, valid_stat_md,
                         0.1f, flags));
    }
    CHECK_UNIMPL(layer_normalization_backward::primitive_desc(eng,
            prop_kind::backward_data, md, md, md, stat_md, 0.1f, flags,
            fwd_hint));
}

TEST_F(runtime_dim_test_t, TestLRN) {
    memory::desc md {
            {DNNL_RUNTIME_DIM_VAL, 16, 7, 7}, data_type::f32, tag::abcd};

    CHECK_UNIMPL(lrn_forward::primitive_desc(eng, prop_kind::forward,
            algorithm::lrn_across_channels, md, md, 5, 1.f, 0.75f, 1.0f));

    lrn_forward::primitive_desc fwd_hint;
    {
        auto valid_md = memory::desc({2, 16, 7, 7}, data_type::f32, tag::abcd);
        CHECK_OK(fwd_hint = lrn_forward::primitive_desc(eng, prop_kind::forward,
                         algorithm::lrn_across_channels, valid_md, valid_md, 5,
                         1.f, 0.75f, 1.0f));
    }
    CHECK_UNIMPL(
            lrn_backward::primitive_desc(eng, algorithm::lrn_across_channels,
                    md, md, md, 5, 1.f, 0.75f, 1.0f, fwd_hint));
}

CPU_TEST_F(runtime_dim_test_t, TestMatmul) {
    memory::desc a_md {{DNNL_RUNTIME_DIM_VAL, 3}, data_type::f32, tag::ab};
    memory::desc b_md {{3, DNNL_RUNTIME_DIM_VAL}, data_type::f32, tag::ba};
    memory::desc c_md {{DNNL_RUNTIME_DIM_VAL, DNNL_RUNTIME_DIM_VAL},
            data_type::f32, tag::ab};
    CHECK_OK(matmul::primitive_desc(eng, a_md, b_md, c_md));
}

TEST_F(runtime_dim_test_t, TestPool) {
    memory::desc src_md {
            {DNNL_RUNTIME_DIM_VAL, 16, 8, 8}, data_type::f32, tag::abcd};
    memory::desc dst_md {
            {DNNL_RUNTIME_DIM_VAL, 16, 4, 4}, data_type::f32, tag::abcd};
    CHECK_UNIMPL(pooling_forward::primitive_desc(eng, prop_kind::forward,
            algorithm::pooling_max, src_md, dst_md, {2, 2}, {2, 2}, {0, 0},
            {0, 0}, {0, 0}));

    pooling_forward::primitive_desc fwd_hint;
    {
        auto valid_src_md
                = memory::desc({2, 16, 8, 8}, data_type::f32, tag::abcd);
        auto valid_dst_md
                = memory::desc({2, 16, 4, 4}, data_type::f32, tag::abcd);
        CHECK_OK(fwd_hint
                = pooling_forward::primitive_desc(eng, prop_kind::forward,
                        algorithm::pooling_max, valid_src_md, valid_dst_md,
                        {2, 2}, {2, 2}, {0, 0}, {0, 0}, {0, 0}));
    }

    CHECK_UNIMPL(pooling_backward::primitive_desc(eng, algorithm::pooling_max,
            src_md, dst_md, {2, 2}, {2, 2}, {0, 0}, {0, 0}, {0, 0}, fwd_hint));
}

TEST_F(runtime_dim_test_t, TestPReLU) {
    memory::desc data_md {
            {DNNL_RUNTIME_DIM_VAL, 16, 3, 3}, data_type::f32, tag::abcd};
    memory::desc weights_md {
            {DNNL_RUNTIME_DIM_VAL, 16, 3, 3}, data_type::f32, tag::abcd};

    CHECK_UNIMPL(prelu_forward::primitive_desc(
            eng, prop_kind::forward, data_md, weights_md, data_md));

    prelu_forward::primitive_desc fwd_hint;
    {
        auto valid_md = memory::desc({2, 16, 3, 3}, data_type::f32, tag::abcd);
        CHECK_OK(fwd_hint = prelu_forward::primitive_desc(eng,
                         prop_kind::forward, valid_md, valid_md, valid_md));
    }

    memory::desc diff_data_desc {
            {DNNL_RUNTIME_DIM_VAL, 16, 3, 3}, data_type::f32, tag::abcd};
    memory::desc diff_weights_desc {
            {DNNL_RUNTIME_DIM_VAL, 16, 3, 3}, data_type::f32, tag::abcd};

    CHECK_UNIMPL(prelu_backward::primitive_desc(eng, data_md, weights_md,
            diff_data_desc, diff_weights_desc, diff_data_desc, fwd_hint));
}

CPU_TEST_F(runtime_dim_test_t, TestReorder) {
    memory::desc src_md {
            {DNNL_RUNTIME_DIM_VAL, 16, 8, 8}, data_type::f32, tag::abcd};
    memory::desc dst_md {
            {DNNL_RUNTIME_DIM_VAL, 16, 8, 8}, data_type::f32, tag::acdb};
    CHECK_OK(reorder::primitive_desc(eng, src_md, eng, dst_md));
}

TEST_F(runtime_dim_test_t, TestRNN) {
    memory::dim l = 10, c = 8, g = 1, d = 1;
    memory::desc src_layer_md {{DNNL_RUNTIME_DIM_VAL, DNNL_RUNTIME_DIM_VAL, c},
            data_type::f32, tag::tnc};
    memory::desc src_iter_md {
            {l, d, DNNL_RUNTIME_DIM_VAL, c}, data_type::f32, tag::ldnc};
    memory::desc wei_layer_md {{l, d, c, g, c}, data_type::f32, tag::ldigo};
    memory::desc wei_iter_md {{l, d, c, g, c}, data_type::f32, tag::ldigo};
    memory::desc bia_md {{l, d, g, c}, data_type::f32, tag::ldgo};
    memory::desc dst_layer_md {{DNNL_RUNTIME_DIM_VAL, DNNL_RUNTIME_DIM_VAL, c},
            data_type::f32, tag::tnc};
    memory::desc dst_iter_md {
            {l, d, DNNL_RUNTIME_DIM_VAL, c}, data_type::f32, tag::ldnc};
    CHECK_UNIMPL(vanilla_rnn_forward::primitive_desc(eng, prop_kind::forward,
            algorithm::eltwise_relu, rnn_direction::unidirectional_left2right,
            src_layer_md, src_iter_md, wei_layer_md, wei_iter_md, bia_md,
            dst_layer_md, dst_iter_md));
}

TEST_F(runtime_dim_test_t, TestShuffle) {
    memory::desc md {
            {DNNL_RUNTIME_DIM_VAL, 16, 3, 3}, data_type::f32, tag::abcd};
    CHECK_UNIMPL(shuffle_forward::primitive_desc(
            eng, prop_kind::forward, md, md, 1, 4));

    shuffle_forward::primitive_desc fwd_hint;
    {
        auto valid_md = memory::desc({2, 16, 3, 3}, data_type::f32, tag::abcd);
        CHECK_OK(fwd_hint = shuffle_forward::primitive_desc(
                         eng, prop_kind::forward, valid_md, valid_md, 1, 4));
    }

    CHECK_UNIMPL(shuffle_backward::primitive_desc(eng, md, md, 1, 4, fwd_hint));
}

TEST_F(runtime_dim_test_t, TestSoftmax) {
    memory::desc md {{DNNL_RUNTIME_DIM_VAL, 16}, data_type::f32, tag::ab};
    CHECK_UNIMPL(softmax_forward::primitive_desc(
            eng, prop_kind::forward, algorithm::softmax_accurate, md, md, 1));

    softmax_forward::primitive_desc fwd_hint;
    {
        auto valid_md = memory::desc({2, 16}, data_type::f32, tag::ab);
        CHECK_OK(fwd_hint
                = softmax_forward::primitive_desc(eng, prop_kind::forward,
                        algorithm::softmax_accurate, valid_md, valid_md, 1));
    }

    CHECK_UNIMPL(softmax_backward::primitive_desc(
            eng, algorithm::softmax_accurate, md, md, md, 1, fwd_hint));
}

TEST_F(runtime_dim_test_t, TestSum) {
    memory::desc md {
            {DNNL_RUNTIME_DIM_VAL, 16, 3, 3}, data_type::f32, tag::abcd};
    CHECK_UNIMPL(sum::primitive_desc(eng, {1.f, 1.f}, {md, md}));
}

} // namespace dnnl
