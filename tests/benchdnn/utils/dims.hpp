/*******************************************************************************
* Copyright 2021-2023 Intel Corporation
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
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using dims_t = std::vector<int64_t>;
using vdims_t = std::vector<dims_t>;

struct prb_dims_t {
    dims_t dims;
    int ndims = 0;
    std::string name;

    int64_t nelems(int mask) const;
};

// Note: we could use a single type to contain both dims_t and vdims_t versions.
// Two different types allow to separate features and members availability which
// don't make much sense for dims_t.
struct prb_vdims_t {
    prb_vdims_t() = default;
    prb_vdims_t(const vdims_t &avdims, const std::string &name = {});

    vdims_t vdims;
    // Destination dimensions with all broadcasts incorporated. Drivers inherit
    // this member and may modify it due to driver specifics.
    dims_t dst_dims;
    int ndims = 0;
    std::string name;

    int n_inputs() const { return static_cast<int>(vdims.size()); }
    int get_broadcast_mask(int i_input = 1) const;
    int64_t nelems(int i_input, int mask) const;
};

// strides for SRC, WEI, and DST
enum {
    STRIDES_SRC = 0,
    STRIDES_WEI = 1,
    STRIDES_DST = 2,
    STRIDES_SIZE = 3,
};

int64_t dims_nelems(const dims_t &dims, int ndims, int mask);
dims_t off2dims_idx(const dims_t &dims, int64_t off);
std::string dims2str(const dims_t &dims);
std::string vdims2str(const vdims_t &vdims);
std::ostream &operator<<(std::ostream &s, const prb_dims_t &prb_dims);
std::ostream &operator<<(std::ostream &s, const prb_vdims_t &prb_vdims);

#endif
