/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
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
    for (size_t i = 0; i < static_cast<size_t>(dst_d.ndims()); ++i)
        od[i] = dst_d.dims()[i];
    return od;
}

broadcasting_strategy_t get_rhs_arg_broadcasting_strategy(
        const memory_desc_t &rhs_arg_md, const memory_desc_wrapper &dst_d) {

    static const bcast_set_t all_bcast_strategies {
            broadcasting_strategy_t::scalar, broadcasting_strategy_t::per_oc,
            broadcasting_strategy_t::per_oc_spatial,
            broadcasting_strategy_t::shared_axes,
            broadcasting_strategy_t::per_mb_spatial,
            broadcasting_strategy_t::per_mb_w, broadcasting_strategy_t::per_w,
            broadcasting_strategy_t::no_broadcast};

    return get_rhs_arg_broadcasting_strategy(
            rhs_arg_md, dst_d, all_bcast_strategies);
}

namespace {

// Checks if mask corresponds to broadcast per first and last dimensions
// Returns true if mask (5D) is equal to [0, 1, 1, 1, 0]
bool is_per_mb_w_bcast(const std::bitset<DNNL_MAX_NDIMS> mask,
        const memory_desc_wrapper &dst_d) {
    const auto ndims = dst_d.ndims();
    assert(ndims > 0);
    const size_t last_dim = static_cast<size_t>(ndims) - 1;

    bool per_mb_w_bcast = !mask.test(0) && !mask.test(last_dim);
    if (!per_mb_w_bcast) return false;

    for (size_t d = 1; d < last_dim; ++d)
        per_mb_w_bcast = per_mb_w_bcast && mask.test(d);
    return per_mb_w_bcast;
}

// Checks if mask corresponds to broadcast per last dimension
// Returns true if mask (5D) is equal to [1, 1, 1, 1, 0]
bool is_per_w_bcast(const std::bitset<DNNL_MAX_NDIMS> mask,
        const memory_desc_wrapper &dst_d) {
    const auto ndims = dst_d.ndims();
    assert(ndims > 0);
    const size_t last_dim = static_cast<size_t>(ndims) - 1;

    bool per_w_bcast = !mask.test(last_dim);
    if (!per_w_bcast) return false;

    for (size_t d = 0; d < last_dim; ++d)
        per_w_bcast = per_w_bcast && mask.test(d);
    return per_w_bcast;
}

// Checks if mask corresponds to broadcast per batch and spatial dimensions
// Returns true if mask (5D) is equal to [0, 1, 0, 0, 0] and
// also if any of mask bits equal 0 will be equal to 1,
// but only if corresponding output dimensions are also equal to 1.
bool is_channel_bcast(const std::bitset<DNNL_MAX_NDIMS> mask,
        const memory_desc_wrapper &dst_d) {
    for (size_t d = 0; d < static_cast<size_t>(dst_d.ndims()); ++d) {
        if (d == 1 && !mask.test(1)) return false;
        if (d != 1 && mask.test(d) && dst_d.dims()[d] != 1) return false;
    }
    return true;
}

// Check if mask corresponds to broadcast per oc
// Returns true if mask (5D) is equal to [1, 0, 1, 1, 1]
bool is_per_oc_bcast(const std::bitset<DNNL_MAX_NDIMS> mask,
        const memory_desc_t &rhs_arg_md) {
    const bool broadcast_per_oc = !mask.test(1);

    if (!broadcast_per_oc) return false;

    const auto ndims = rhs_arg_md.ndims;

    if (ndims > 0 && rhs_arg_md.dims[0] != 1) return false;

    for (int dim = 2; dim < ndims; dim++) {
        if (rhs_arg_md.dims[dim] != 1) return false;
    }
    return true;
}

bool bcast_strategy_enabled(const bcast_set_t &supported_strategy_set,
        const broadcasting_strategy_t &bcast) {
    return supported_strategy_set.find(bcast) != supported_strategy_set.cend();
}

broadcasting_strategy_t get_per_oc_bcast(
        const bcast_set_t &supported_strategy_set,
        const memory_desc_wrapper &dst_d) {

    const auto ndims = dst_d.ndims();
    const bool use_per_oc_spatial_strategy = bcast_strategy_enabled(
            supported_strategy_set, broadcasting_strategy_t::per_oc_spatial);

    if (use_per_oc_spatial_strategy && dst_d.is_blocking_desc()) {
        const auto &strides = dst_d.blocking_desc().strides;

        //per_oc_spatial used in nchw data format and matmul having ndims >= 3
        return (dst_d.is_plain() && strides[0] >= strides[1]
                       && IMPLICATION(ndims < 3, strides[1] != 1)
                       && IMPLICATION(ndims >= 3, strides[1] >= strides[2]))
                ? broadcasting_strategy_t::per_oc_spatial
                : broadcasting_strategy_t::per_oc;
    }

    return broadcasting_strategy_t::per_oc;
}
} // namespace

// Compares dimensions of rhs arg (src1) with dimensions of destination.
// Produces broadcast mask and returns broadcast strategy which
// corresponds to given mask.
// Mask bits are set to 1 if corresponding dimensions are different
// or if both dimensions are equal to 1.
// Otherwise mask bits are set to 0.
broadcasting_strategy_t get_rhs_arg_broadcasting_strategy(
        const memory_desc_t &rhs_arg_md, const memory_desc_wrapper &dst_d,
        const bcast_set_t &supported_strategy_set) {

    const auto is_enabled = [&](const broadcasting_strategy_t &bcast) {
        return bcast_strategy_enabled(supported_strategy_set, bcast);
    };

    const int ndims = rhs_arg_md.ndims;
    const auto output_dims = make_output_dims(dst_d);

    bool all_ones = true;
    bool all_equal = true;
    std::bitset<DNNL_MAX_NDIMS> mask(0);
    for (size_t d = 0; d < static_cast<size_t>(ndims); d++) {
        const auto &rhs_arg_dim = rhs_arg_md.dims[d];
        if (rhs_arg_md.dims[d] != 1 && rhs_arg_md.dims[d] != output_dims[d])
            return broadcasting_strategy_t::unsupported;

        if (rhs_arg_dim != 1) all_ones = false;

        const bool different_dims = output_dims[d] != rhs_arg_md.dims[d];
        if (different_dims) all_equal = false;

        if (different_dims || output_dims[d] == 1) mask.set(d);
    }

    broadcasting_strategy_t bcast = broadcasting_strategy_t::unsupported;

    if (all_ones && is_enabled(broadcasting_strategy_t::scalar))
        bcast = broadcasting_strategy_t::scalar;
    else if (all_equal && is_enabled(broadcasting_strategy_t::no_broadcast))
        bcast = broadcasting_strategy_t::no_broadcast;
    else if (is_per_mb_w_bcast(mask, dst_d)
            && is_enabled(broadcasting_strategy_t::per_mb_w))
        bcast = broadcasting_strategy_t::per_mb_w;
    else if (is_per_oc_bcast(mask, rhs_arg_md)
            && (is_enabled(broadcasting_strategy_t::per_oc)
                    || is_enabled(broadcasting_strategy_t::per_oc_spatial))) {
        bcast = get_per_oc_bcast(supported_strategy_set, dst_d);
    } else if (is_per_w_bcast(mask, dst_d)
            && is_enabled(broadcasting_strategy_t::per_w))
        bcast = broadcasting_strategy_t::per_w;
    else if (is_channel_bcast(mask, dst_d)
            && is_enabled(broadcasting_strategy_t::per_mb_spatial))
        bcast = broadcasting_strategy_t::per_mb_spatial;
    else if (is_enabled(broadcasting_strategy_t::shared_axes))
        bcast = broadcasting_strategy_t::shared_axes;

    return bcast;
}

} // namespace impl
} // namespace dnnl
