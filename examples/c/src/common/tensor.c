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
#include "stdlib.h"
#include "string.h"

#include "common/tensor.h"
#include "common/utils.h"

#define MAX_TENSORS_NUM (10000)

static example_tensor_t *s_tensors[MAX_TENSORS_NUM];
static size_t s_tensors_num = 0;

static size_t s_tensor_id = 10000;

example_result_t example_tensor_create(example_tensor_t **created_tensor,
        example_data_type_t type, int32_t ndims, const int64_t *dims,
        size_t layout_id) {
    *created_tensor = (example_tensor_t *)malloc(sizeof(example_tensor_t));
    if (*created_tensor == NULL) return example_result_error_common_fail;

    (*created_tensor)->dtype_ = type;
    (*created_tensor)->ndims_ = ndims;
    (*created_tensor)->layout_id_ = layout_id;
    (*created_tensor)->data_ = NULL;
    (*created_tensor)->bytes_ = 0;
    (*created_tensor)->producer_.producer_ = NULL;
    (*created_tensor)->producer_.from_offset_ = 0;
    (*created_tensor)->users_num_ = 0;
    memcpy((*created_tensor)->dims_, dims, (size_t)ndims * sizeof(int64_t));

    (*created_tensor)->id_ = s_tensor_id;
    s_tensor_id++;

    s_tensors[s_tensors_num] = *created_tensor;
    s_tensors_num++;

    return example_result_success;
}

void example_tensor_destroy(example_tensor_t *tensor) {
    if (tensor->data_ != NULL) free(tensor->data_);
    free(tensor);
}

void example_tensor_destroy_all() {
    for (int i = 0; i < s_tensors_num; i++) {
        example_tensor_destroy(s_tensors[i]);
    }
}

example_result_t example_tensor_set_producer(
        example_tensor_t *tensor, void *producer_op, uint64_t from_offset) {
    if (!tensor) return example_result_error_common_fail;

    producer_t producer = {producer_op, from_offset};
    tensor->producer_ = producer;
    return example_result_success;
}

example_result_t example_tensor_add_user(
        example_tensor_t *tensor, void *user_op, uint64_t to_offset) {
    if (!tensor) return example_result_error_common_fail;

    user_t user = {user_op, to_offset};
    tensor->users_[tensor->users_num_] = user;
    tensor->users_num_++;
    return example_result_success;
}

example_result_t example_tensor_erase_user(
        example_tensor_t *tensor, void *user_op) {
    if (!tensor) return example_result_error_common_fail;

    int idx = -1;
    for (int i = 0; i < tensor->users_num_; i++) {
        if (tensor->users_[i].users_ == user_op) {
            idx = i;
            break;
        }
    }

    // no such user is not an error
    if (idx < 0) return example_result_success;

    for (int i = idx; i < tensor->users_num_ - 1; i++) {
        tensor->users_[i] = tensor->users_[i + 1];
    }
    tensor->users_num_--;

    return example_result_success;
}

example_result_t example_tensor_erase_producer(
        example_tensor_t *tensor, void *producer_op) {
    if (!tensor) return example_result_error_common_fail;

    if (tensor->producer_.producer_ != producer_op)
        return example_result_success;

    tensor->producer_.producer_ = NULL;
    tensor->producer_.from_offset_ = 0;

    return example_result_success;
}

example_result_t example_tensor_find_by_id(
        example_tensor_t **found_tensor, size_t id) {
    if (!found_tensor) return example_result_error_common_fail;

    for (int i = 0; i < s_tensors_num; i++) {
        if (id == s_tensors[i]->id_) {
            *found_tensor = s_tensors[i];
            return example_result_success;
        }
    }

    return example_result_error_common_fail;
}
