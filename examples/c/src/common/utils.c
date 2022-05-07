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

#include <assert.h>
#include <string.h>

#include "oneapi/dnnl/dnnl_graph.h"

#include "common/utils.h"

const char *dnnl_graph_result2str(dnnl_graph_status_t v) {
    switch (v) {
        case dnnl_graph_success: return "success";
        case dnnl_graph_out_of_memory: return "out of memory";
        case dnnl_graph_invalid_arguments: return "invalid arguments";
        case dnnl_graph_unimplemented: return "unimplemented";
        case dnnl_graph_iterator_ends: return "iterator ends";
        case dnnl_graph_runtime_error: return "runtime error";
        case dnnl_graph_not_required: return "not required";
        case dnnl_graph_invalid_graph: return "invalid graph";
        case dnnl_graph_invalid_graph_op: return "invalid op";
        case dnnl_graph_invalid_shape: return "invalid shape";
        case dnnl_graph_invalid_data_type: return "invalid data type";
        default: return "unknown error";
    }
}

const char *example_result2str(example_result_t v) {
    if (v == example_result_success) return "success";
    if (v == example_result_error_common_fail) return "common fail";
    return "unknown status";
}

dnnl_graph_engine_kind_t parse_engine_kind(int argc, char **argv) {
    // Returns default engine kind, i.e. CPU, if none given
    if (argc == 1) {
        return dnnl_graph_cpu;
    } else if (argc == 2) {
        // Checking the engine type, i.e. CPU or GPU
        char *engine_kind_str = argv[1];
        if (!strcmp(engine_kind_str, "cpu")) {
            return dnnl_graph_cpu;
        } else if (!strcmp(engine_kind_str, "gpu")) {
            // Checking if a GPU exists on the machine
            // if (!dnnl_engine_get_count(dnnl_gpu))
            //     COMPLAIN_EXAMPLE_ERROR_AND_EXIT("%s",
            //             "could not find compatible GPU\nPlease run the example "
            //             "with CPU instead");
            return dnnl_graph_gpu;
        }
    }

    exit(1);
}
