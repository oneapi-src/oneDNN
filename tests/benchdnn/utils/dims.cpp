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

#include <algorithm>

#include "utils/dims.hpp"

int64_t dims_nelems(const dims_t &dims, int ndims, int mask) {
    int64_t nelems = 1;
    for (int d = 0; d < ndims; d++) {
        nelems *= (mask & (1 << d)) ? dims[d] : 1;
    }
    return nelems;
}

int64_t prb_dims_t::nelems(int mask) const {
    return dims_nelems(dims, ndims, mask);
}

prb_vdims_t::prb_vdims_t(const vdims_t &avdims, const std::string &name)
    : vdims(avdims), ndims(static_cast<int>(avdims[0].size())), name(name) {
    // First tensor in `vdims` provided defined `ndims`.
    // If second and consecutive inputs are provided with less dimensions, e.g.
    // (ndims0 > ndims1), then fill these tensors with ones to match `ndims`,
    // e.g., 8x3x5:8 -> 8x3x5:8x1x1
    for (int i = 1; i < n_inputs(); i++)
        if (ndims > static_cast<int>(vdims[i].size()))
            vdims[i].resize(ndims, 1);

    dst_dims = vdims[0];
    for (int i = 1; i < n_inputs(); i++) {
        for (int d = 0; d < ndims; ++d) {
            bool has_zero_dim = dst_dims[d] == 0 || vdims[i][d] == 0;
            dst_dims[d] = has_zero_dim ? 0 : std::max(dst_dims[d], vdims[i][d]);
        }
    }
}

int64_t prb_vdims_t::nelems(int i_input, int mask) const {
    assert(i_input <= n_inputs() && i_input >= 0);
    const auto &dims = i_input == n_inputs() ? dst_dims : vdims[i_input];
    return dims_nelems(dims, ndims, mask);
}

int prb_vdims_t::get_broadcast_mask(int i_input) const {
    int broadcast_mask = 0;
    for (int d = 0; d < ndims; ++d)
        broadcast_mask += dst_dims[d] == vdims[i_input][d] ? (1 << d) : 0;
    return broadcast_mask;
}

// returns dims with current @p off values using actual values from @p dims
dims_t off2dims_idx(const dims_t &dims, int64_t off) {
    dims_t dims_idx;
    dims_idx.reserve(dims.size());

    for (int i = (int)dims.size() - 1; i >= 0; --i) {
        dims_idx.insert(dims_idx.begin(), off % dims[i]);
        off /= dims[i];
    }
    assert(off == 0);
    return dims_idx;
}

std::string dims2str(const dims_t &dims) {
    std::string s;
    if (dims.empty()) return s;

    s += std::to_string(dims[0]);
    for (auto d = dims.begin() + 1; d != dims.end(); d++)
        s += "x" + std::to_string(*d);

    return s;
}

std::string vdims2str(const vdims_t &vdims) {
    std::string s;
    if (vdims.empty()) return s;

    s += dims2str(vdims[0]);
    for (auto it = vdims.begin() + 1; it != vdims.end(); it++) {
        const auto &dims = *it;
        s += ":" + dims2str(dims);
    }
    return s;
}

std::ostream &operator<<(std::ostream &s, const prb_dims_t &prb_dims) {
    s << dims2str(prb_dims.dims);
    if (!prb_dims.name.empty()) s << "_n" << prb_dims.name;
    return s;
}

std::ostream &operator<<(std::ostream &s, const prb_vdims_t &prb_vdims) {
    s << vdims2str(prb_vdims.vdims);
    if (!prb_vdims.name.empty()) s << "_n" << prb_vdims.name;
    return s;
}
