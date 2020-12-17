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
#ifndef OP_DEF_H
#define OP_DEF_H

#include "common/graph.h"
#include "common/op.h"
#include "common/tensor.h"
#include "common/utils.h"

#define Conv2d(graph, name, input, out_channels, kernel_size, strides, \
        padding, dilations, groups) \
    ({ \
        int32_t oc = out_channels; \
        int64_t ks[] = {kernel_size, kernel_size}; \
        int64_t s[] = {strides, strides}; \
        int64_t p[] = {padding, padding}; \
        int64_t d[] = {dilations, dilations}; \
        int64_t g = groups; \
        example_op_t *op; \
        example_tensor_t *output \
                = conv2d(&op, name, input, oc, ks, s, p, d, &g); \
        CHECK_EXAMPLE(example_graph_add_op(graph, op)); \
        output; \
    })

#define Relu(graph, name, input) \
    ({ \
        example_op_t *op; \
        example_tensor_t *output = relu(&op, name, input); \
        CHECK_EXAMPLE(example_graph_add_op(graph, op)); \
        output; \
    })

#endif
