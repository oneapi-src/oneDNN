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
#ifndef GRAPH_H
#define GRAPH_H

#include "common/op.h"
#include "common/utils.h"

#define MAX_OPS_NUM (10000)

typedef struct {
    example_op_t *ops_[MAX_OPS_NUM];
    int64_t op_num_;
} example_graph_t;

example_result_t example_graph_create(example_graph_t **created_graph);

void example_graph_destroy(example_graph_t *graph);

example_result_t example_graph_add_op(example_graph_t *graph, example_op_t *op);

#endif
