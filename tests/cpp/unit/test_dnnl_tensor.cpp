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
#include "gtest/gtest.h"

#include "backend/dnnl/common.hpp"
#include "backend/dnnl/tensor.hpp"
#include "interface/allocator.hpp"

#include "unit_test_common.hpp"

namespace impl = dnnl::graph::impl;
namespace dnnl_impl = dnnl::graph::impl::dnnl_impl;

using tensor = dnnl_impl::tensor;
using dim = dnnl_impl::tensor::desc::dim;
using dims = dnnl_impl::tensor::desc::dims;
using data_type = dnnl_impl::tensor::desc::data_type;
using format_tag = dnnl_impl::tensor::desc::format_tag;

TEST(dnnl_tensor, create) {
    impl::engine_t graph_eng = get_engine();
    dnnl::engine dnnl_eng = dnnl_impl::make_dnnl_engine(graph_eng);
    impl::allocator_t *alc = graph_eng.get_allocator();

    const dims adims {8, 32, 64, 64};
    const data_type adata_type = data_type::f32;
    const format_tag aformat_tag = format_tag::abcd;

    // create tensor without buffer (alloc internal)
    tensor::desc td {adims, adata_type, aformat_tag};
    tensor t1 {td, dnnl_eng, alc};
    ASSERT_FALSE(t1.is_empty());

    // create tensor with buffer
    test::vector<float> buffer(td.get_size());
    tensor t2 {td, dnnl_eng, alc, buffer.data()};
    ASSERT_FALSE(t2.is_empty());
}

TEST(dnnl_tensor, reinit) {
    impl::engine_t graph_eng = get_engine();
    dnnl::engine dnnl_eng = dnnl_impl::make_dnnl_engine(graph_eng);
    impl::allocator_t *alc = graph_eng.get_allocator();
    impl::stream_t &graph_stream = get_stream();
    dnnl::stream s = dnnl_impl::make_dnnl_stream(dnnl_eng, graph_stream);

    const dims adims = {8, 32, 64, 64};
    const data_type adata_type = data_type::f32;
    const format_tag aformat_tag = format_tag::abcd;

    tensor::desc td {adims, adata_type, aformat_tag};
    tensor t1 {td, dnnl_eng, alc};

    // empty tensor have no engine, so can't be
    // reinit through this method
    tensor t3;
    bool ret = t3.reinit_if_possible(s, td);
    graph_stream.wait();
    ASSERT_FALSE(ret);

    // if expected desc is different from this desc,
    // reinit method will alloc new memory
    tensor::desc td4 {adims, adata_type, format_tag::abdc};
    test::vector<float> buffer(td4.get_size());
    tensor t4(td4, dnnl_eng, alc, buffer.data());
    t4.reinit_if_possible(s, td);
    graph_stream.wait();
    ASSERT_NE(buffer.data(), t4.get_data_handle());
}

TEST(dnnl_tensor, reorder) {
    impl::engine_t graph_eng = get_engine();
    dnnl::engine dnnl_eng = dnnl_impl::make_dnnl_engine(graph_eng);
    impl::allocator_t *alc = graph_eng.get_allocator();
    impl::stream_t &graph_stream = get_stream();
    dnnl::stream s = dnnl_impl::make_dnnl_stream(dnnl_eng, graph_stream);

    const dims adims = {8, 32, 64, 64};
    const data_type adata_type = data_type::f32;
    const format_tag aformat_tag = format_tag::abcd;

    tensor::desc td {adims, adata_type, aformat_tag};
    tensor t1 {td, dnnl_eng, alc};

    tensor::desc td2 {adims, adata_type, format_tag::abdc};
    tensor t2 = t1.reorder_if_differ_in(s, td2);
    graph_stream.wait();
    ASSERT_FALSE(t2.is_empty());
}

TEST(dnnl_tensor, make_grouped_weight) {
    impl::engine_t graph_eng = get_engine();
    dnnl::engine dnnl_eng = dnnl_impl::make_dnnl_engine(graph_eng);
    impl::allocator_t *alc = graph_eng.get_allocator();

    const dims adims = {8, 32, 3, 3};
    const data_type adata_type = data_type::f32;
    const format_tag aformat_tag = format_tag::abcd;

    tensor::desc td {adims, adata_type, aformat_tag};
    tensor t1 {td, dnnl_eng, alc};

    dim groups = 2;
    tensor t2 = t1.make_grouped_weights(groups);
    ASSERT_FALSE(t2.is_empty());
}

TEST(dnnl_tensor, reshape) {
    impl::engine_t graph_eng = get_engine();
    dnnl::engine dnnl_eng = dnnl_impl::make_dnnl_engine(graph_eng);
    impl::allocator_t *alc = graph_eng.get_allocator();
    impl::stream_t &graph_stream = get_stream();
    dnnl::stream s = dnnl_impl::make_dnnl_stream(dnnl_eng, graph_stream);

    const dims adims = {8, 32, 4, 4};
    const data_type adata_type = data_type::f32;
    const format_tag aformat_tag = format_tag::abcd;

    tensor::desc td {adims, adata_type, aformat_tag};
    tensor t1 {td, dnnl_eng, alc};

    const dims new_shape = {16, 16, 2, 8};

    t1.reshape(s, new_shape);
    graph_stream.wait();
    ASSERT_EQ(t1.get_dims(), new_shape);
}

TEST(dnnl_tensor, to_format) {
    impl::engine_t graph_eng = get_engine();
    dnnl::engine dnnl_eng = dnnl_impl::make_dnnl_engine(graph_eng);
    impl::allocator_t *alc = graph_eng.get_allocator();
    impl::stream_t &graph_stream = get_stream();
    dnnl::stream s = dnnl_impl::make_dnnl_stream(dnnl_eng, graph_stream);

    const dims adims = {8, 32, 4, 4};
    const data_type adata_type = data_type::f32;
    const format_tag aformat_tag = format_tag::abcd;

    tensor::desc td {adims, adata_type, aformat_tag};
    tensor t1 {td, dnnl_eng, alc};

    t1.to_format(s, format_tag::abdc);
    graph_stream.wait();
    ASSERT_EQ(t1.get_desc().get_format_tag(), format_tag::abdc);
}

TEST(dnnl_tensor, to_public) {
    impl::engine_t graph_eng = get_engine();
    dnnl::engine dnnl_eng = dnnl_impl::make_dnnl_engine(graph_eng);
    impl::allocator_t *alc = graph_eng.get_allocator();
    impl::stream_t &graph_stream = get_stream();
    dnnl::stream s = dnnl_impl::make_dnnl_stream(dnnl_eng, graph_stream);

    const dims adims = {8, 32, 64, 84};
    const data_type adata_type = data_type::f32;
    const format_tag aformat_tag = format_tag::aBcd8b;

    tensor::desc td {adims, adata_type, aformat_tag};
    tensor t1 {td, dnnl_eng, alc};

    tensor t2 = t1.to_public(s);
    graph_stream.wait();
    ASSERT_TRUE(t2.is_public_format());
}

TEST(dnnl_tensor_desc, to_grouped) {
    tensor::desc td {dims {64, 8, 7, 7}, data_type::f32, format_tag::cdba};

    dim groups = 4;
    tensor::desc grouped_td = td.to_grouped(groups);
    ASSERT_TRUE(grouped_td.is_grouped());

    ASSERT_EQ(grouped_td.data.ndims, 5);
    ASSERT_EQ(grouped_td.data.dims[0], 4);
    ASSERT_EQ(grouped_td.data.dims[1], 16);
    ASSERT_EQ(grouped_td.data.dims[2], 8);
    ASSERT_EQ(grouped_td.data.dims[3], 7);
    ASSERT_EQ(grouped_td.data.dims[4], 7);

    ASSERT_EQ(grouped_td.data.format_desc.blocking.strides[0], 16);
    ASSERT_EQ(grouped_td.data.format_desc.blocking.strides[1], 1);
    ASSERT_EQ(grouped_td.data.format_desc.blocking.strides[2], 64);
    ASSERT_EQ(grouped_td.data.format_desc.blocking.strides[3], 3584);
    ASSERT_EQ(grouped_td.data.format_desc.blocking.strides[4], 512);
}

TEST(dnnl_tensor_desc, transpose) {
    tensor::desc td {{64, 3, 7, 7}, data_type::f32, format_tag::abcd};
    tensor::desc new_desc = td.transpose(0, 1);

    tensor::desc truth {{3, 64, 7, 7}, data_type::f32, format_tag::bacd};

    ASSERT_EQ(new_desc.data.ndims, 4);
    ASSERT_EQ(new_desc.data.dims[0], truth.data.dims[0]);
    ASSERT_EQ(new_desc.data.dims[1], truth.data.dims[1]);
    ASSERT_EQ(new_desc.data.dims[2], truth.data.dims[2]);
    ASSERT_EQ(new_desc.data.dims[3], truth.data.dims[3]);

    ASSERT_EQ(new_desc.data.format_desc.blocking.strides[0],
            truth.data.format_desc.blocking.strides[0]);
    ASSERT_EQ(new_desc.data.format_desc.blocking.strides[1],
            truth.data.format_desc.blocking.strides[1]);
    ASSERT_EQ(new_desc.data.format_desc.blocking.strides[2],
            truth.data.format_desc.blocking.strides[2]);
    ASSERT_EQ(new_desc.data.format_desc.blocking.strides[3],
            truth.data.format_desc.blocking.strides[3]);
}

// TODO(qun) seldom used methods (such as methods about
// quantization and submemory) are not tested now. they
// should be added if used later.
