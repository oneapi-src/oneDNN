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
#include "stdlib.h"
#include "string.h"

#include "common/op.h"
#include "common/utils.h"

static size_t s_op_id = 10000;

example_result_t example_op_attr_create(example_attr_t **created_op_attr,
        const char *name, example_attr_kind_t kind, void *data, size_t num) {
    *created_op_attr = (example_attr_t *)malloc(sizeof(example_attr_t));
    if (*created_op_attr == NULL) return example_result_error_common_fail;

    (*created_op_attr)->name_ = name;
    (*created_op_attr)->kind_ = kind;
    (*created_op_attr)->data_num_ = (int64_t)(num);

    size_t bytes;
    if (kind == attr_kind_i) {
        bytes = sizeof(int64_t);
    } else if (kind == attr_kind_is)
        bytes = sizeof(int64_t) * (size_t)num;
    else if (kind == attr_kind_s) {
        bytes = sizeof(char) * (size_t)num + 1;
    } else {
        bytes = sizeof(float) * (size_t)num;
    }
    (*created_op_attr)->data_ = malloc(bytes);
    memcpy((*created_op_attr)->data_, data, bytes);
    return example_result_success;
}

void example_op_attr_destroy(example_attr_t *attr) {
    free(attr->data_);
    free(attr);
}

example_result_t example_op_create_base(
        example_op_t **created_op, const char *name, example_op_kind_t kind) {
    *created_op = (example_op_t *)malloc(sizeof(example_op_t));
    if (*created_op == NULL) return example_result_error_common_fail;
    (*created_op)->name_ = name;
    (*created_op)->kind_ = kind;

    (*created_op)->inputs_num_ = 0;
    (*created_op)->outputs_num_ = 0;
    (*created_op)->attrs_num_ = 0;

    (*created_op)->kernel_ = NULL;

    (*created_op)->id_ = s_op_id;
    s_op_id++;

    return example_result_success;
}

void example_op_destroy(example_op_t *op) {
    for (int i = 0; i < op->attrs_num_; i++) {
        example_op_attr_destroy(op->attrs_[i]);
    }
    free(op);
}

example_result_t example_op_add_attr(example_op_t *op, example_attr_t *attr) {
    op->attrs_[op->attrs_num_] = attr;
    op->attrs_num_++;
    return example_result_success;
}

static example_result_t example_op_create(example_op_t **created_op,
        const char *name, example_op_kind_t kind, example_tensor_t **inputs,
        size_t input_num, example_tensor_t **outputs, size_t output_num,
        example_attr_t **attrs, size_t attr_num) {

    // create op base
    CHECK_EXAMPLE(example_op_create_base(created_op, name, kind));

    // add inputs
    uint64_t to_offset = 0;
    for (int i = 0; i < input_num; i++) {
        (*created_op)->inputs_[(*created_op)->inputs_num_] = inputs[i];
        example_tensor_add_user(inputs[i], *created_op, to_offset);
        (*created_op)->inputs_num_++;
        to_offset++;
    }

    // add outputs
    uint64_t from_offset = 0;
    for (int i = 0; i < output_num; i++) {
        (*created_op)->outputs_[(*created_op)->outputs_num_] = outputs[i];
        example_tensor_set_producer(outputs[i], *created_op, from_offset);
        (*created_op)->outputs_num_++;
        from_offset++;
    }

    // add attrs
    for (int i = 0; i < attr_num; i++) {
        (*created_op)->attrs_[(*created_op)->attrs_num_] = attrs[i];
        (*created_op)->attrs_num_++;
    }

    return example_result_success;
}

example_tensor_t *conv2d(example_op_t **conv_op, const char *name,
        example_tensor_t *input, int32_t out_channels, int64_t *kernel_size,
        int64_t *strides, int64_t *padding, int64_t *dilations,
        int64_t *groups) {

    if (input == NULL) return NULL;

    // get input shape
    int32_t ndims = input->ndims_;
    if (ndims != 4) return NULL;

    int64_t ib = input->dims_[0], ic = input->dims_[1], ih = input->dims_[2],
            iw = input->dims_[3];
    int64_t kh = kernel_size[0], kw = kernel_size[1];
    int64_t sh = strides[0], sw = strides[1];
    int64_t ph = padding[0], pw = padding[1];
    int64_t dh = dilations[0], dw = dilations[1];
    int64_t oc = out_channels;

    // prepare weight and bias
    const int64_t weight_dims[] = {oc, ic, kh, kw};
    example_tensor_t *weight = NULL;
    example_tensor_create(&weight, f32, 4, weight_dims, any);

    // prepare output
    int64_t _kh = kh + (kh - 1) * (dh - 1);
    int64_t _kw = kw + (kw - 1) * (dw - 1);
    int64_t oh = (ih - _kh + 2 * ph) / sh + 1;
    int64_t ow = (iw - _kw + 2 * pw) / sw + 1;
    const int64_t out_dims[] = {ib, oc, oh, ow};
    char *data_format = "NCX";
    char *filter_format = "OIX";
    example_tensor_t *output = NULL;
    example_tensor_create(&output, f32, 4, out_dims, any);

    // prepare attrs
    example_attr_t *attr0 = NULL, *attr1 = NULL, *attr2 = NULL, *attr3 = NULL,
                   *attr4 = NULL, *attr5 = NULL, *attr6 = NULL;
    example_op_attr_create(&attr0, "strides", attr_kind_is, strides, 2);
    example_op_attr_create(&attr1, "pads_begin", attr_kind_is, padding, 2);
    example_op_attr_create(&attr2, "pads_end", attr_kind_is, padding, 2);
    example_op_attr_create(&attr3, "dilations", attr_kind_is, dilations, 2);
    example_op_attr_create(&attr4, "groups", attr_kind_i, groups, 0);
    example_op_attr_create(&attr5, "data_format", attr_kind_s, data_format,
            strlen(data_format));
    example_op_attr_create(&attr6, "filter_format", attr_kind_s, filter_format,
            strlen(filter_format));

    example_tensor_t *inputs[] = {input, weight};
    example_attr_t *attrs[] = {attr0, attr1, attr2, attr3, attr4, attr5, attr6};

    example_op_create(
            conv_op, name, e_kconv2d, inputs, 2, &output, 1, attrs, 7);

    return output;
}

example_tensor_t *relu(
        example_op_t **relu_op, const char *name, example_tensor_t *input) {
    if (input == NULL) return NULL;

    // get input shape
    int32_t ndims = input->ndims_;
    if (ndims != 4) return NULL;

    int64_t ib = input->dims_[0], ic = input->dims_[1], ih = input->dims_[2],
            iw = input->dims_[3];

    // create output
    const int64_t out_dims[] = {ib, ic, ih, iw};
    example_tensor_t *output = NULL;
    example_tensor_create(&output, f32, 4, out_dims, any);

    example_op_create(relu_op, name, e_krelu, &input, 1, &output, 1, NULL, 0);

    return output;
}
