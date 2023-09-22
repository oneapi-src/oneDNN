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

#include "oneapi/dnnl/dnnl.h"

#include "common/c_types_map.hpp"
#include "common/stream.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::status;

// In order to be able to use the profiling API in benchdnn
// regardless of whether the DNNL_EXPERIMENTAL_PROFILING macro is defined
// or not we have to expose the API as internal one when the macro is not
// defined.
#ifdef DNNL_EXPERIMENTAL_PROFILING
#define INTERNAL_API_ATTRIBUTE(rtype) rtype
#else
#define INTERNAL_API_ATTRIBUTE(rtype) extern "C" rtype DNNL_API
#endif

INTERNAL_API_ATTRIBUTE(status_t) dnnl_reset_profiling(stream_t *stream) {
    const auto eng_kind = stream->engine()->kind();
    if (eng_kind != engine_kind::gpu) {
        VERROR(common, common, "CPU engine does not support profiling");
        return status::unimplemented;
    }
    return stream->reset_profiling();
}

INTERNAL_API_ATTRIBUTE(status_t)
dnnl_query_profiling_data(stream_t *stream, profiling_data_kind_t data_kind,
        int *num_entries, uint64_t *data) {
    const auto eng_kind = stream->engine()->kind();
    if (eng_kind != engine_kind::gpu) {
        VERROR(common, common, "CPU engine does not support profiling");
        return status::unimplemented;
    }
    return stream->get_profiling_data(data_kind, num_entries, data);
}

extern "C" status_t DNNL_API dnnl_impl_notify_profiling_complete(
        stream_t *stream) {
    const auto eng_kind = stream->engine()->kind();
    if (eng_kind != engine_kind::gpu) {
        VERROR(common, common, "CPU engine does not support profiling");
        return status::unimplemented;
    }
    return stream->notify_profiling_complete();
}
