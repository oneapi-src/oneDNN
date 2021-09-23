/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#ifndef UTILS_DIMS_T_HPP
#define UTILS_DIMS_T_HPP

#include <cassert>
#include <iostream>
#include <vector>

#include "oneapi/dnnl/dnnl_types.h"

struct dims_t : public std::vector<int64_t> {
    //  using vector<int64_t>::vector;
    //  There is a bug in Intel compiler 19.0 on MacOS which prevents
    //  using-declaration from being used here. The workaround is to introduce
    //  constructors explicitly.
    dims_t() = default;
    dims_t(size_t size) : vector(size) {}
    dims_t(size_t size, int64_t value) : vector(size, value) {}
    dims_t(const dnnl_memory_desc_t &md) : dims_t(md.ndims) {
        for (int d = 0; d < md.ndims; ++d)
            this->at(d) = md.dims[d];
    }
};

// strides for SRC, WEI, and DST
using strides_t = std::vector<dims_t>;
enum {
    STRIDES_SRC = 0,
    STRIDES_WEI = 1,
    STRIDES_DST = 2,
    STRIDES_SIZE = 3,
};

dims_t off2dims_idx(const dims_t &dims, int64_t off);
std::ostream &operator<<(std::ostream &s, const dims_t &dims);

#endif
