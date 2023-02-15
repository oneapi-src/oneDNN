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

#include <gtest/gtest.h>

#include "oneapi/dnnl/dnnl_graph.h"

TEST(CAPI, LogicalTensorInit) {
    dnnl_graph_logical_tensor_t lt;
    const size_t id = 123;

    ASSERT_EQ(dnnl_graph_logical_tensor_init(&lt, id, dnnl_f32,
                      DNNL_GRAPH_UNKNOWN_NDIMS, dnnl_graph_layout_type_strided,
                      dnnl_graph_tensor_property_undef),
            dnnl_success);
    ASSERT_EQ(lt.id, id);
    ASSERT_EQ(lt.data_type, dnnl_f32);
    ASSERT_EQ(lt.ndims, DNNL_GRAPH_UNKNOWN_NDIMS);
    ASSERT_EQ(lt.layout_type, dnnl_graph_layout_type_strided);
    ASSERT_EQ(lt.property, dnnl_graph_tensor_property_undef);

    ASSERT_EQ(dnnl_graph_logical_tensor_init(&lt, id, dnnl_f32, 0,
                      dnnl_graph_layout_type_strided,
                      dnnl_graph_tensor_property_undef),
            dnnl_success);
    ASSERT_EQ(lt.id, id);
    ASSERT_EQ(lt.data_type, dnnl_f32);
    ASSERT_EQ(lt.ndims, 0);
    ASSERT_EQ(lt.layout_type, dnnl_graph_layout_type_strided);
    ASSERT_EQ(lt.property, dnnl_graph_tensor_property_undef);

    ASSERT_EQ(dnnl_graph_logical_tensor_init(&lt, id, dnnl_f32, 4,
                      dnnl_graph_layout_type_strided,
                      dnnl_graph_tensor_property_undef),
            dnnl_success);
    ASSERT_EQ(lt.id, id);
    ASSERT_EQ(lt.data_type, dnnl_f32);
    ASSERT_EQ(lt.ndims, 4);
    ASSERT_EQ(lt.layout_type, dnnl_graph_layout_type_strided);
    ASSERT_EQ(lt.property, dnnl_graph_tensor_property_undef);

    ASSERT_EQ(dnnl_graph_logical_tensor_init(&lt, id, dnnl_f32, 4,
                      dnnl_graph_layout_type_any,
                      dnnl_graph_tensor_property_undef),
            dnnl_success);
    ASSERT_EQ(lt.id, id);
    ASSERT_EQ(lt.data_type, dnnl_f32);
    ASSERT_EQ(lt.ndims, 4);
    ASSERT_EQ(lt.layout_type, dnnl_graph_layout_type_any);
    ASSERT_EQ(lt.property, dnnl_graph_tensor_property_undef);

    ASSERT_EQ(dnnl_graph_logical_tensor_init(&lt, id, dnnl_boolean, 4,
                      dnnl_graph_layout_type_any,
                      dnnl_graph_tensor_property_undef),
            dnnl_success);
    ASSERT_EQ(lt.id, id);
    ASSERT_EQ(lt.data_type, dnnl_boolean);
    ASSERT_EQ(lt.ndims, 4);
    ASSERT_EQ(lt.layout_type, dnnl_graph_layout_type_any);
    ASSERT_EQ(lt.property, dnnl_graph_tensor_property_undef);
}

TEST(CAPI, LogicalTensorInitProperty) {
    dnnl_graph_logical_tensor_t lt;
    const size_t id = 123;

    ASSERT_EQ(dnnl_graph_logical_tensor_init(&lt, id, dnnl_f32,
                      DNNL_GRAPH_UNKNOWN_NDIMS, dnnl_graph_layout_type_strided,
                      dnnl_graph_tensor_property_constant),
            dnnl_success);
    ASSERT_EQ(lt.id, id);
    ASSERT_EQ(lt.data_type, dnnl_f32);
    ASSERT_EQ(lt.ndims, DNNL_GRAPH_UNKNOWN_NDIMS);
    ASSERT_EQ(lt.layout_type, dnnl_graph_layout_type_strided);
    ASSERT_EQ(lt.property, dnnl_graph_tensor_property_constant);

    ASSERT_EQ(dnnl_graph_logical_tensor_init(&lt, id, dnnl_f32, 0,
                      dnnl_graph_layout_type_strided,
                      dnnl_graph_tensor_property_constant),
            dnnl_success);
    ASSERT_EQ(lt.id, id);
    ASSERT_EQ(lt.data_type, dnnl_f32);
    ASSERT_EQ(lt.ndims, 0);
    ASSERT_EQ(lt.layout_type, dnnl_graph_layout_type_strided);
    ASSERT_EQ(lt.property, dnnl_graph_tensor_property_constant);

    ASSERT_EQ(dnnl_graph_logical_tensor_init(&lt, id, dnnl_f32, 4,
                      dnnl_graph_layout_type_strided,
                      dnnl_graph_tensor_property_constant),
            dnnl_success);
    ASSERT_EQ(lt.id, id);
    ASSERT_EQ(lt.data_type, dnnl_f32);
    ASSERT_EQ(lt.ndims, 4);
    ASSERT_EQ(lt.layout_type, dnnl_graph_layout_type_strided);
    ASSERT_EQ(lt.property, dnnl_graph_tensor_property_constant);

    ASSERT_EQ(dnnl_graph_logical_tensor_init(&lt, id, dnnl_f32, 4,
                      dnnl_graph_layout_type_any,
                      dnnl_graph_tensor_property_constant),
            dnnl_success);
    ASSERT_EQ(lt.id, id);
    ASSERT_EQ(lt.data_type, dnnl_f32);
    ASSERT_EQ(lt.ndims, 4);
    ASSERT_EQ(lt.layout_type, dnnl_graph_layout_type_any);
    ASSERT_EQ(lt.property, dnnl_graph_tensor_property_constant);
}

TEST(CAPI, LogicalTensorInitWithDims) {
    dnnl_graph_logical_tensor_t lt;
    const size_t id = 123;

    dnnl_dims_t dims = {1, 2, 3, 4};
    ASSERT_EQ(dnnl_graph_logical_tensor_init_with_dims(&lt, id, dnnl_f32, 1,
                      dims, dnnl_graph_layout_type_strided,
                      dnnl_graph_tensor_property_undef),
            dnnl_success);
    ASSERT_EQ(lt.id, id);
    ASSERT_EQ(lt.data_type, dnnl_f32);
    ASSERT_EQ(lt.ndims, 1);
    ASSERT_EQ(lt.layout_type, dnnl_graph_layout_type_strided);
    ASSERT_EQ(lt.dims[0], dims[0]);
    ASSERT_EQ(lt.layout.strides[0], 1);
    ASSERT_EQ(lt.property, dnnl_graph_tensor_property_undef);

    ASSERT_EQ(dnnl_graph_logical_tensor_init_with_dims(&lt, id, dnnl_f32, 2,
                      dims, dnnl_graph_layout_type_strided,
                      dnnl_graph_tensor_property_undef),
            dnnl_success);
    ASSERT_EQ(lt.id, id);
    ASSERT_EQ(lt.data_type, dnnl_f32);
    ASSERT_EQ(lt.ndims, 2);
    ASSERT_EQ(lt.layout_type, dnnl_graph_layout_type_strided);
    ASSERT_EQ(lt.dims[0], dims[0]);
    ASSERT_EQ(lt.dims[1], dims[1]);
    ASSERT_EQ(lt.layout.strides[0], 2);
    ASSERT_EQ(lt.layout.strides[1], 1);
    ASSERT_EQ(lt.property, dnnl_graph_tensor_property_undef);

    ASSERT_EQ(dnnl_graph_logical_tensor_init_with_dims(&lt, id, dnnl_f32, 3,
                      dims, dnnl_graph_layout_type_strided,
                      dnnl_graph_tensor_property_undef),
            dnnl_success);
    ASSERT_EQ(lt.id, id);
    ASSERT_EQ(lt.data_type, dnnl_f32);
    ASSERT_EQ(lt.ndims, 3);
    ASSERT_EQ(lt.layout_type, dnnl_graph_layout_type_strided);
    ASSERT_EQ(lt.dims[0], dims[0]);
    ASSERT_EQ(lt.dims[1], dims[1]);
    ASSERT_EQ(lt.dims[2], dims[2]);
    ASSERT_EQ(lt.layout.strides[0], 6);
    ASSERT_EQ(lt.layout.strides[1], 3);
    ASSERT_EQ(lt.layout.strides[2], 1);
    ASSERT_EQ(lt.property, dnnl_graph_tensor_property_undef);

    // test 0-sized dimension tensor
    dnnl_dims_t zero_sized_dims = {2, 3, 0};
    ASSERT_EQ(dnnl_graph_logical_tensor_init_with_dims(&lt, id, dnnl_f32, 3,
                      zero_sized_dims, dnnl_graph_layout_type_strided,
                      dnnl_graph_tensor_property_undef),
            dnnl_success);
    ASSERT_EQ(lt.id, id);
    ASSERT_EQ(lt.data_type, dnnl_f32);
    ASSERT_EQ(lt.ndims, 3);
    ASSERT_EQ(lt.layout_type, dnnl_graph_layout_type_strided);
    ASSERT_EQ(lt.dims[0], zero_sized_dims[0]);
    ASSERT_EQ(lt.dims[1], zero_sized_dims[1]);
    ASSERT_EQ(lt.dims[2], zero_sized_dims[2]);
    ASSERT_EQ(lt.layout.strides[0], 3);
    ASSERT_EQ(lt.layout.strides[1], 1);
    ASSERT_EQ(lt.layout.strides[2], 1);
    ASSERT_EQ(lt.property, dnnl_graph_tensor_property_undef);
}

TEST(CAPI, LogicalTensorInitWithStrides) {
    dnnl_graph_logical_tensor_t lt;
    const size_t id = 123;

    dnnl_dims_t dims = {1, 2, 3, 4};
    dnnl_dims_t strides = {30, 20, 10, 1};
    ASSERT_EQ(dnnl_graph_logical_tensor_init_with_strides(&lt, id, dnnl_f32, 1,
                      dims, strides, dnnl_graph_tensor_property_undef),
            dnnl_success);
    ASSERT_EQ(lt.id, id);
    ASSERT_EQ(lt.data_type, dnnl_f32);
    ASSERT_EQ(lt.ndims, 1);
    ASSERT_EQ(lt.layout_type, dnnl_graph_layout_type_strided);
    ASSERT_EQ(lt.dims[0], dims[0]);
    ASSERT_EQ(lt.layout.strides[0], strides[0]);
    ASSERT_EQ(lt.property, dnnl_graph_tensor_property_undef);

    ASSERT_EQ(dnnl_graph_logical_tensor_init_with_strides(&lt, id, dnnl_f32, 2,
                      dims, strides, dnnl_graph_tensor_property_undef),
            dnnl_success);
    ASSERT_EQ(lt.id, id);
    ASSERT_EQ(lt.data_type, dnnl_f32);
    ASSERT_EQ(lt.ndims, 2);
    ASSERT_EQ(lt.layout_type, dnnl_graph_layout_type_strided);
    ASSERT_EQ(lt.dims[0], dims[0]);
    ASSERT_EQ(lt.dims[1], dims[1]);
    ASSERT_EQ(lt.layout.strides[0], strides[0]);
    ASSERT_EQ(lt.layout.strides[1], strides[1]);
    ASSERT_EQ(lt.property, dnnl_graph_tensor_property_undef);

    ASSERT_EQ(dnnl_graph_logical_tensor_init_with_strides(&lt, id, dnnl_f32, 2,
                      dims, strides, dnnl_graph_tensor_property_undef),
            dnnl_success);
    ASSERT_EQ(lt.id, id);
    ASSERT_EQ(lt.data_type, dnnl_f32);
    ASSERT_EQ(lt.ndims, 2);
    ASSERT_EQ(lt.layout_type, dnnl_graph_layout_type_strided);
    ASSERT_EQ(lt.dims[0], dims[0]);
    ASSERT_EQ(lt.dims[1], dims[1]);
    ASSERT_EQ(lt.layout.strides[0], strides[0]);
    ASSERT_EQ(lt.layout.strides[1], strides[1]);
    ASSERT_EQ(lt.property, dnnl_graph_tensor_property_undef);

    ASSERT_EQ(dnnl_graph_logical_tensor_init_with_strides(&lt, id, dnnl_f32, 3,
                      dims, strides, dnnl_graph_tensor_property_undef),
            dnnl_success);
    ASSERT_EQ(lt.id, id);
    ASSERT_EQ(lt.data_type, dnnl_f32);
    ASSERT_EQ(lt.ndims, 3);
    ASSERT_EQ(lt.layout_type, dnnl_graph_layout_type_strided);
    ASSERT_EQ(lt.dims[0], dims[0]);
    ASSERT_EQ(lt.dims[1], dims[1]);
    ASSERT_EQ(lt.dims[2], dims[2]);
    ASSERT_EQ(lt.layout.strides[0], strides[0]);
    ASSERT_EQ(lt.layout.strides[1], strides[1]);
    ASSERT_EQ(lt.layout.strides[2], strides[2]);
    ASSERT_EQ(lt.property, dnnl_graph_tensor_property_undef);
}

TEST(CAPI, LogicalTensorInitFull) {
    dnnl_graph_logical_tensor_t lt;
    const size_t id = 123;

    dnnl_dims_t dims = {1, 2, 3, 4};
    dnnl_dims_t strides = {30, 20, 10, 1};
    ASSERT_EQ(dnnl_graph_logical_tensor_init_with_strides(&lt, id, dnnl_f32, 1,
                      dims, strides, dnnl_graph_tensor_property_constant),
            dnnl_success);
    ASSERT_EQ(lt.id, id);
    ASSERT_EQ(lt.data_type, dnnl_f32);
    ASSERT_EQ(lt.ndims, 1);
    ASSERT_EQ(lt.layout_type, dnnl_graph_layout_type_strided);
    ASSERT_EQ(lt.dims[0], dims[0]);
    ASSERT_EQ(lt.layout.strides[0], strides[0]);
    ASSERT_EQ(lt.property, dnnl_graph_tensor_property_constant);

    ASSERT_EQ(dnnl_graph_logical_tensor_init_with_strides(&lt, id, dnnl_f32, 2,
                      dims, strides, dnnl_graph_tensor_property_constant),
            dnnl_success);
    ASSERT_EQ(lt.id, id);
    ASSERT_EQ(lt.data_type, dnnl_f32);
    ASSERT_EQ(lt.ndims, 2);
    ASSERT_EQ(lt.layout_type, dnnl_graph_layout_type_strided);
    ASSERT_EQ(lt.dims[0], dims[0]);
    ASSERT_EQ(lt.dims[1], dims[1]);
    ASSERT_EQ(lt.layout.strides[0], strides[0]);
    ASSERT_EQ(lt.layout.strides[1], strides[1]);
    ASSERT_EQ(lt.property, dnnl_graph_tensor_property_constant);

    ASSERT_EQ(dnnl_graph_logical_tensor_init_with_strides(&lt, id, dnnl_f32, 2,
                      dims, strides, dnnl_graph_tensor_property_constant),
            dnnl_success);
    ASSERT_EQ(lt.id, id);
    ASSERT_EQ(lt.data_type, dnnl_f32);
    ASSERT_EQ(lt.ndims, 2);
    ASSERT_EQ(lt.layout_type, dnnl_graph_layout_type_strided);
    ASSERT_EQ(lt.dims[0], dims[0]);
    ASSERT_EQ(lt.dims[1], dims[1]);
    ASSERT_EQ(lt.layout.strides[0], strides[0]);
    ASSERT_EQ(lt.layout.strides[1], strides[1]);
    ASSERT_EQ(lt.property, dnnl_graph_tensor_property_constant);

    ASSERT_EQ(dnnl_graph_logical_tensor_init_with_strides(&lt, id, dnnl_f32, 3,
                      dims, strides, dnnl_graph_tensor_property_constant),
            dnnl_success);
    ASSERT_EQ(lt.id, id);
    ASSERT_EQ(lt.data_type, dnnl_f32);
    ASSERT_EQ(lt.ndims, 3);
    ASSERT_EQ(lt.layout_type, dnnl_graph_layout_type_strided);
    ASSERT_EQ(lt.dims[0], dims[0]);
    ASSERT_EQ(lt.dims[1], dims[1]);
    ASSERT_EQ(lt.dims[2], dims[2]);
    ASSERT_EQ(lt.layout.strides[0], strides[0]);
    ASSERT_EQ(lt.layout.strides[1], strides[1]);
    ASSERT_EQ(lt.layout.strides[2], strides[2]);
    ASSERT_EQ(lt.property, dnnl_graph_tensor_property_constant);
}
