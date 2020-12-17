/*******************************************************************************
* Copyright 2020 Intel Corporation
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
#include "gtest/gtest.h"

#include "backend/dnnl/tensor.hpp"
#include "interface/allocator.hpp"

#include "unit_test_common.hpp"

using namespace llga::impl::dnnl_impl;

TEST(dnnl_tensor, create) {
    llga::impl::engine_t graph_eng = get_engine();
    engine dnnl_eng = engine(graph_eng);

    const dims adims {8, 32, 64, 64};
    const data_type adata_type = data_type::f32;
    const format_tag aformat_tag = format_tag::abcd;

    // create tensor without buffer (alloc internal)
    tensor::desc td {adims, adata_type, aformat_tag};
    tensor t1 {td, dnnl_eng};
    ASSERT_FALSE(t1.is_empty());

    // create tensor with buffer
    test::vector<float> buffer(td.get_size());
    tensor t2 {td, buffer.data(), dnnl_eng};
    ASSERT_FALSE(t2.is_empty());
}

TEST(dnnl_tensor, reinit) {
    llga::impl::engine_t graph_eng = get_engine();
    engine dnnl_eng = engine(graph_eng);

    const dims adims = {8, 32, 64, 64};
    const data_type adata_type = data_type::f32;
    const format_tag aformat_tag = format_tag::abcd;

    tensor::desc td {adims, adata_type, aformat_tag};
    tensor t1 {td, dnnl_eng};

    tensor t2;
    t2.reinit_like(t1);
    ASSERT_FALSE(t2.is_empty());

    // empty tensor have no engine, so can't be
    // reinit through this method
    tensor t3;
    bool ret = t3.reinit_if_possible(td);
    ASSERT_FALSE(ret);

    // if expected desc is different from this desc,
    // reinit method will alloc new memory
    tensor::desc td4 {adims, adata_type, format_tag::abdc};
    test::vector<float> buffer(td4.get_size());
    tensor t4(td4, buffer.data(), dnnl_eng);
    t4.reinit_if_possible(td);
    ASSERT_NE(buffer.data(), t4.get_data_handle());
}

TEST(dnnl_tensor, reorder) {
    llga::impl::engine_t graph_eng = get_engine();
    engine dnnl_eng = engine(graph_eng);

    const dims adims = {8, 32, 64, 64};
    const data_type adata_type = data_type::f32;
    const format_tag aformat_tag = format_tag::abcd;

    tensor::desc td {adims, adata_type, aformat_tag};
    tensor t1 {td, dnnl_eng};

    tensor::desc td2 {adims, adata_type, format_tag::abdc};
    tensor t2 = t1.reorder_if_differ_in(td2);
    ASSERT_FALSE(t2.is_empty());
}

TEST(dnnl_tensor, make_grouped_weight) {
    llga::impl::engine_t graph_eng = get_engine();
    engine dnnl_eng = engine(graph_eng);

    const dims adims = {8, 32, 3, 3};
    const data_type adata_type = data_type::f32;
    const format_tag aformat_tag = format_tag::abcd;

    tensor::desc td {adims, adata_type, aformat_tag};
    tensor t1 {td, dnnl_eng};

    dim groups = 2;
    tensor t2 = t1.make_grouped_weights(groups);
    ASSERT_FALSE(t2.is_empty());
}

TEST(dnnl_tensor, resize) {
    llga::impl::engine_t graph_eng = get_engine();
    engine dnnl_eng = engine(graph_eng);

    const dims adims = {8, 32, 3, 3};
    const data_type adata_type = data_type::f32;
    const format_tag aformat_tag = format_tag::abcd;

    tensor::desc td {adims, adata_type, aformat_tag};
    tensor t1 {td, dnnl_eng};

    const dims new_dims = {16, 32, 3, 3};
    const data_type new_data_type = data_type::f16;

    t1.resize(new_dims, new_data_type);
    ASSERT_EQ(t1.get_dim(0), 16);
    ASSERT_EQ(t1.get_data_type(), data_type::f16);
}

TEST(dnnl_tensor, reshape) {
    llga::impl::engine_t graph_eng = get_engine();
    engine dnnl_eng = engine(graph_eng);

    const dims adims = {8, 32, 4, 4};
    const data_type adata_type = data_type::f32;
    const format_tag aformat_tag = format_tag::abcd;

    tensor::desc td {adims, adata_type, aformat_tag};
    tensor t1 {td, dnnl_eng};

    const dims new_shape = {16, 16, 2, 8};

    t1.reshape(new_shape);
    ASSERT_EQ(t1.get_dims(), new_shape);
}

TEST(dnnl_tensor, to_format) {
    llga::impl::engine_t graph_eng = get_engine();
    engine dnnl_eng = engine(graph_eng);

    const dims adims = {8, 32, 4, 4};
    const data_type adata_type = data_type::f32;
    const format_tag aformat_tag = format_tag::abcd;

    tensor::desc td {adims, adata_type, aformat_tag};
    tensor t1 {td, dnnl_eng};

    t1.to_format(format_tag::abdc);
    ASSERT_EQ(t1.get_desc().get_format_tag(), format_tag::abdc);
}

TEST(dnnl_tensor, to_public) {
    llga::impl::engine_t graph_eng = get_engine();
    engine dnnl_eng = engine(graph_eng);

    const dims adims = {8, 32, 64, 84};
    const data_type adata_type = data_type::f32;
    const format_tag aformat_tag = format_tag::aBcd8b;

    tensor::desc td {adims, adata_type, aformat_tag};
    tensor t1 {td, dnnl_eng};

    tensor t2 = t1.to_public();
    ASSERT_TRUE(t2.is_public_format());
}

// TODO(qun) seldom used methods (such as methods about
// quantization and submemory) are not tested now. they
// should be added if used later.
