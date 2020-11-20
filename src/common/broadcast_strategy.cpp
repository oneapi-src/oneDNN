/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include <bitset>

#include "common/broadcast_strategy.hpp"

namespace dnnl {
namespace impl {

output_dims_t make_output_dims(const memory_desc_wrapper &dst_d) {
    output_dims_t od {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
    for (int i = 0; i < dst_d.ndims(); ++i)
        od[i] = dst_d.dims()[i];
    return od;
}

broadcasting_strategy_t get_rhs_arg_broadcasting_strategy(
        const memory_desc_t &rhs_arg_md, const memory_desc_wrapper &dst_d) {

    static const bcast_set_t all_bcast_strategies {
            broadcasting_strategy_t::scalar, broadcasting_strategy_t::per_oc,
            broadcasting_strategy_t::per_oc_spatial,
            broadcasting_strategy_t::shared_axes,
            broadcasting_strategy_t::no_broadcast};

    return get_rhs_arg_broadcasting_strategy(
            rhs_arg_md, dst_d, all_bcast_strategies);
}

broadcasting_strategy_t get_rhs_arg_broadcasting_strategy(
        const memory_desc_t &rhs_arg_md, const memory_desc_wrapper &dst_d,
        const bcast_set_t &supported_strategy_set) {

    const auto is_enabled = [&](const broadcasting_strategy_t &bcast) {
        return supported_strategy_set.find(bcast)
                != supported_strategy_set.cend();
    };

    const int ndims = rhs_arg_md.ndims;
    const auto output_dims = make_output_dims(dst_d);

    bool all_ones = true;
    std::bitset<DNNL_MAX_NDIMS> mask(0);
    for (int d = 0; d < ndims; d++) {
        const auto &rhs_arg_dim = rhs_arg_md.dims[d];
        if (rhs_arg_md.dims[d] != 1 && rhs_arg_md.dims[d] != output_dims[d])
            return broadcasting_strategy_t::unsupported;

        if (rhs_arg_dim != 1) all_ones = false;

        const auto both_one_dim
                = (output_dims[d] == 1 && rhs_arg_md.dims[d] == 1);
        if ((output_dims[d] != rhs_arg_md.dims[d] || output_dims[d] == 1)
                && !both_one_dim)
            mask.set(d);
    }

    broadcasting_strategy_t bcast = broadcasting_strategy_t::shared_axes;

    const auto &mb_rhs = rhs_arg_md.dims[0];
    const bool broadcast_per_mb = !mask.test(0);
    const bool broadcast_per_oc = !mask.test(1);

    if (all_ones)
        bcast = broadcasting_strategy_t::scalar;
    else if (mask.none())
        bcast = broadcasting_strategy_t::no_broadcast;
    else if (broadcast_per_oc && !(broadcast_per_mb && mb_rhs != 1)) {
        const bool use_per_oc_spatial_strategy
                = is_enabled(broadcasting_strategy_t::per_oc_spatial);

        if (use_per_oc_spatial_strategy && dst_d.is_blocking_desc()) {
            const auto &strides = dst_d.blocking_desc().strides;

            //per_oc_spatial basically used in nchw data format
            bcast = dst_d.is_plain() && strides[1] != 1
                            && strides[0] >= strides[1]
                            && IMPLICATION(ndims >= 3, strides[1] >= strides[2])
                    ? broadcasting_strategy_t::per_oc_spatial
                    : broadcasting_strategy_t::per_oc;
        } else {
            bcast = broadcasting_strategy_t::per_oc;
        }
    }

    if (is_enabled(bcast)) return bcast;

    return broadcasting_strategy_t::unsupported;
}

} // namespace impl
} // namespace dnnl
