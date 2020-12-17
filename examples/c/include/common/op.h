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
#ifndef OP_H
#define OP_H

#include "common/tensor.h"
#include "common/utils.h"

#define MAX_INPUTS_OUTPUTS_NUM (1000)
#define MAX_ATTRS_NUM (1000)

typedef void (*kernel_func)(void);

typedef enum {
    attr_kind_f = 0,
    attr_kind_fs,
    attr_kind_i,
    attr_kind_is,
    attr_kind_s
} example_attr_kind_t;

typedef enum {
    e_kconv2d = 0,
    e_krelu,
    e_kmax_pool2d,
    e_kbatch_normal,
    e_kfake = 1000,
} example_op_kind_t;

typedef struct {
    const char *name_;
    example_attr_kind_t kind_;
    void *data_;
    int64_t data_num_;
} example_attr_t;

typedef struct {
    const char *name_;
    example_op_kind_t kind_;

    uint64_t inputs_num_;
    uint64_t outputs_num_;
    example_tensor_t *inputs_[MAX_INPUTS_OUTPUTS_NUM];
    example_tensor_t *outputs_[MAX_INPUTS_OUTPUTS_NUM];

    int64_t attrs_num_;
    example_attr_t *attrs_[MAX_ATTRS_NUM];

    kernel_func kernel_;

    size_t id_;
} example_op_t;

example_result_t example_op_attr_create(example_attr_t **created_op_attr,
        const char *name, example_attr_kind_t kind, void *data, size_t num);

void example_op_attr_destroy(example_attr_t *attr);

example_result_t example_op_create_base(
        example_op_t **created_op, const char *name, example_op_kind_t kind);

void example_op_destroy(example_op_t *op);

example_result_t example_op_add_attr(example_op_t *op, example_attr_t *attr);

example_tensor_t *conv2d(example_op_t **conv_op, const char *name,
        example_tensor_t *input, int32_t out_channels, int64_t *kernel_size,
        int64_t *strides, int64_t *padding, int64_t *dilations,
        int64_t *groups);

example_tensor_t *relu(
        example_op_t **relu_op, const char *name, example_tensor_t *input);

#endif
