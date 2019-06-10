/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#ifndef GEMM_THREADING_HPP
#define GEMM_THREADING_HPP

#include <cstdint>
#include "c_types_map.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

enum class partition_type {
    row_1d,
    col_1d,
    col_major_2d,
};

enum class copy_type {
    nonshared,
    shared_a,
    no_copy,
};

struct gemm_threading_t {
    int nthrs_m, nthrs_n, nthrs_k;
    partition_type partition;
    copy_type copy;

    int nthrs() const { return nthrs_m * nthrs_n * nthrs_k; }

    friend bool operator==(const gemm_threading_t &t1,
            const gemm_threading_t &t2) {
        return (t1.nthrs_m == t2.nthrs_m && t1.nthrs_n == t2.nthrs_n
                && t1.nthrs_k == t2.nthrs_k && t1.partition == t2.partition
                && t1.copy == t2.copy);
    }

    friend bool operator!=(const gemm_threading_t &t1,
            const gemm_threading_t &t2) {
        return !(t1 == t2);
    }
};


}
}
}

#endif
