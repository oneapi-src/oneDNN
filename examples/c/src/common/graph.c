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
#include <stdlib.h>
#include <string.h>

#include "common/graph.h"
#include "common/tensor.h"
#include "common/utils.h"

example_result_t example_graph_create(example_graph_t **created_graph) {
    *created_graph = (example_graph_t *)malloc(sizeof(example_graph_t));
    if (*created_graph == NULL) return example_result_error_common_fail;

    memset(*created_graph, 0, sizeof(example_graph_t));
    return example_result_success;
}

void example_graph_destroy(example_graph_t *graph) {
    // destroy every op
    for (int i = 0; i < graph->op_num_; i++) {
        example_op_destroy(graph->ops_[i]);
    }
    // free the graph buffer
    free(graph);
}
example_result_t example_graph_add_op(
        example_graph_t *graph, example_op_t *op) {
    graph->ops_[graph->op_num_] = op;
    graph->op_num_++;
    return example_result_success;
}
