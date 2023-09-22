/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "utils/stream_kind.hpp"
#include "common.hpp"

stream_kind_t default_stream_kind {stream_kind_t::def};
stream_kind_t stream_kind {default_stream_kind};

dnnl_stream_flags_t stream_kind2stream_flags(
        stream_kind_t stream_kind, bool use_profiling) {
    dnnl_stream_flags_t flags = dnnl_stream_default_flags;
    switch (stream_kind) {
        case stream_kind_t::def: break;
        case stream_kind_t::in_order: flags = dnnl_stream_in_order; break;
        case stream_kind_t::out_of_order:
            flags = dnnl_stream_out_of_order;
            break;
        default: SAFE_V(FAIL);
    }

#ifdef DNNL_EXPERIMENTAL_PROFILING
    dnnl_stream_flags_t profiling_flag = dnnl_stream_profiling;
#else
    dnnl_stream_flags_t profiling_flag = static_cast<dnnl_stream_flags_t>(0x4);
#endif
    if (use_profiling)
        flags = static_cast<dnnl_stream_flags_t>(flags | profiling_flag);
    return flags;
}

stream_kind_t str2stream_kind(const char *str) {
#define CASE(param) \
    if (!strcasecmp(#param, str)) return stream_kind_t::param

    CASE(def);
    CASE(in_order);
    CASE(out_of_order);

#undef CASE

    BENCHDNN_PRINT(
            0, "Error: stream kind value \'%s\' is not recognized.\n", str);
    SAFE_V(FAIL);
    return stream_kind_t::def;
}

std::ostream &operator<<(std::ostream &s, stream_kind_t stream_kind) {
    if (stream_kind == stream_kind_t::def) s << "def";
    if (stream_kind == stream_kind_t::in_order) s << "in_order";
    if (stream_kind == stream_kind_t::out_of_order) s << "out_of_order";

    return s;
}
