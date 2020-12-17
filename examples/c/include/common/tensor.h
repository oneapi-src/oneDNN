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
#ifndef TENSOR_H
#define TENSOR_H

#include "common/utils.h"

#define MAX_USER_NUM (1000)
#define MAX_DIMS_NUM (12)

typedef struct {
    void *producer_;
    uint64_t from_offset_;
} producer_t;

typedef struct {
    void *users_;
    uint64_t to_offset_;
} user_t;

typedef enum {
    undef = 0,
    f16 = 1,
    bf16 = 2,
    f32 = 3,
    s32 = 4,
    s8 = 5,
    u8 = 6,
} example_data_type_t;

typedef enum {
    any = 1,

    // Plain formats
    a, ///< plain 1D tensor
    ab, ///< plain 2D tensor
    abc, ///< plain 3D tensor
    abcd, ///< plain 4D tensor
    abcde, ///< plain 5D tensor
    abcdef, ///< plain 6D tensor

    // Permuted plain formats
    abdc, ///< permuted 4D tensor
    abdec, ///< permuted 5D tensor
    acb, ///< permuted 3D tensor
    acbde, ///< permuted 5D tensor
    acbdef, ///< permuted 6D tensor
    acdb, ///< permuted 4D tensor
    acdeb, ///< permuted 5D tensor
    ba, ///< permuted 2D tensor
    bac, ///< permuted 3D tensor
    bacd, ///< permuted 4D tensor
    bacde, ///< permuted 5D tensor
    bca, ///< permuted 3D tensor
    bcda, ///< permuted 4D tensor
    bcdea, ///< permuted 5D tensor
    cba, ///< permuted 3D tensor
    cdba, ///< permuted 4D tensor
    dcab, ///< permuted 4D tensor
    cdeba, ///< permuted 5D tensor
    decab, ///< permuted 5D tensor
    defcab, ///< permuted 6D tensor

    /// where we start to count for non-plain layout
    first_non_plain_layout
} example_layout_id_t;

typedef struct {
    example_data_type_t dtype_;
    int64_t layout_id_;
    int32_t ndims_;

    int64_t dims_[MAX_DIMS_NUM];
    int64_t strides_[MAX_DIMS_NUM];

    void *data_;
    int64_t bytes_;

    producer_t producer_;
    user_t users_[MAX_USER_NUM];
    size_t users_num_;

    size_t id_;
} example_tensor_t;

example_result_t example_tensor_create(example_tensor_t **created_tensor,
        example_data_type_t type, int32_t ndims, const int64_t *dims,
        int64_t layout_id);

void example_tensor_destroy_all();

example_result_t example_tensor_add_user(
        example_tensor_t *tensor, void *user_op, uint64_t to_offset);

example_result_t example_tensor_set_producer(
        example_tensor_t *tensor, void *producer_op, uint64_t from_offset);

example_result_t example_tensor_erase_user(
        example_tensor_t *tensor, void *user_op);
example_result_t example_tensor_erase_producer(
        example_tensor_t *tensor, void *producer_op);

example_result_t example_tensor_find_by_id(
        example_tensor_t **found_tensor, size_t id);

#endif
