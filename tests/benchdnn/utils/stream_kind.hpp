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

#ifndef UTILS_STREAM_KIND_HPP
#define UTILS_STREAM_KIND_HPP

#include <sstream>

#include "oneapi/dnnl/dnnl_types.h"

enum class stream_kind_t {
    // The library-defined stream kind.
    def = 0x0,
    in_order = 0x1,
    out_of_order = 0x2,
};

extern stream_kind_t stream_kind; // user stream kind
extern stream_kind_t default_stream_kind; // the default stream kind

dnnl_stream_flags_t stream_kind2stream_flags(
        stream_kind_t stream_kind, bool use_profiling);

stream_kind_t str2stream_kind(const char *str);

std::ostream &operator<<(std::ostream &s, stream_kind_t stream_kind);

#endif
