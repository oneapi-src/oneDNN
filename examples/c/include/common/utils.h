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
#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>

#include "oneapi/dnnl/dnnl_graph.h"

typedef enum {
    example_result_success = 0,
    example_result_error_common_fail,
    // TODO(qun) support more error code
    example_result_error_unknown = 0x7fffffff
} example_result_t;

const char *dnnl_graph_result2str(dnnl_graph_status_t v);
const char *example_result2str(example_result_t v);
dnnl_graph_engine_kind_t parse_engine_kind(int argc, char **argv);

#define COMPLAIN_DNNL_GRAPH_ERROR_AND_EXIT(what, status) \
    do { \
        printf("[%s:%d] `%s` returns oneDNN Graph error: %s.\n", __FILE__, \
                __LINE__, what, dnnl_graph_result2str(status)); \
        printf("Example failed.\n"); \
        exit(1); \
    } while (0)

#define COMPLAIN_EXAMPLE_ERROR_AND_EXIT(what, status) \
    do { \
        printf("[%s:%d] returns example error: %s.\n", __FILE__, __LINE__, \
                example_result2str(status)); \
        printf("Example failed.\n"); \
        exit(2); \
    } while (0)

#define DNNL_GRAPH_CHECK(f) \
    do { \
        dnnl_graph_status_t s_ = f; \
        if (s_ != dnnl_graph_success) \
            COMPLAIN_DNNL_GRAPH_ERROR_AND_EXIT(#f, s_); \
    } while (0)

#define CHECK_EXAMPLE(f) \
    do { \
        example_result_t s_ = f; \
        if (s_ != example_result_success) \
            COMPLAIN_EXAMPLE_ERROR_AND_EXIT(#f, s_); \
    } while (0)

#endif
