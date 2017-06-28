/*******************************************************************************
* Copyright 2017 Intel Corporation
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

#ifndef _MKLDNN_COMMON_HPP
#define _MKLDNN_COMMON_HPP

#include "mkldnn.h"

#include "common.hpp"

inline const char *status2str(mkldnn_status_t status) {
#   define CASE(s) case s: return #s
    switch (status) {
    CASE(mkldnn_success);
    CASE(mkldnn_out_of_memory);
    CASE(mkldnn_try_again);
    CASE(mkldnn_invalid_arguments);
    CASE(mkldnn_not_ready);
    CASE(mkldnn_unimplemented);
    CASE(mkldnn_iterator_ends);
    CASE(mkldnn_runtime_error);
    CASE(mkldnn_not_required);
    }
    return "unknown error";
#   undef CASE
}

#define DNN_SAFE(f, s) do { \
    mkldnn_status_t status = f; \
    if (status != mkldnn_success) { \
        if (s == CRIT || s == WARN) { \
            print(0, "error [%s:%d]: '%s' -> %s(%d)\n", \
                    __PRETTY_FUNCTION__, __LINE__, \
                    #f, status2str(status), (int)status); \
            fflush(0); \
            if (s == CRIT) exit(2); \
        } \
        return FAIL; \
    } \
} while(0)

/* simplification */
extern mkldnn_engine_t engine;

inline int execute(mkldnn_primitive_t p) {
    mkldnn_stream_t stream;
    DNN_SAFE(mkldnn_stream_create(&stream, mkldnn_eager), CRIT);
    DNN_SAFE(mkldnn_stream_submit(stream, 1, &p, NULL), CRIT);
    DNN_SAFE(mkldnn_stream_wait(stream, 1, NULL), CRIT);
    DNN_SAFE(mkldnn_stream_destroy(stream), CRIT);
    return OK;
}

inline int init() {
    DNN_SAFE(mkldnn_engine_create(&engine, mkldnn_cpu, 0), CRIT);
    return OK;
}

inline int finalize() {
    DNN_SAFE(mkldnn_engine_destroy(engine), CRIT);
    return OK;
}

inline const char *query_impl_info(const_mkldnn_primitive_desc_t pd) {
    const char *str;
    mkldnn_primitive_desc_query(pd, mkldnn_query_impl_info_str, 0, &str);
    return str;
}

#endif
