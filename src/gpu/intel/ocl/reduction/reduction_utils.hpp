/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
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

#ifndef GPU_REDUCTION_UTILS_H
#define GPU_REDUCTION_UTILS_H

#include <assert.h>
#include <sstream>
#include <vector>

#include "common/c_types_map.hpp"
#include "gpu/intel/block_structure.hpp"
#include "gpu/intel/compute/kernel_ctx.hpp"
#include "gpu/intel/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

// Same as reduction portions of alg_kind_t, plus:
// lp_norm_power_p:
//     dst = sum(|src|^p)
// pth_root_max:
//     dst = root(max(sum(src), eps), p)
// pth_root_sum:
//     dst = root(sum(src) + eps, p)
// final_max:
//     dst = max(sum(src), eps)
// final_sum:
//     dst = sum(src) + eps
enum class reduction_alg_kind_t {
    undef = alg_kind::undef,
    max = alg_kind::reduction_max,
    min = alg_kind::reduction_min,
    sum = alg_kind::reduction_sum,
    mul = alg_kind::reduction_mul,
    mean = alg_kind::reduction_mean,
    lp_norm_max = alg_kind::reduction_norm_lp_max,
    lp_norm_sum = alg_kind::reduction_norm_lp_sum,
    lp_norm_power_p_max = alg_kind::reduction_norm_lp_power_p_max,
    lp_norm_power_p_sum = alg_kind::reduction_norm_lp_power_p_sum,
    // Extra algs
    lp_norm_power_p,
    pth_root_max,
    pth_root_sum,
    final_max,
    final_sum,
};

// Convert from the basic alg_kind_t to the expanded reduction_alg_kind_t. Since
// we break reduction problems into phases, this assigns a unique alg to any
// phase. e.g. mean -> sum, sum, sum, ..., mean (with the full div)
inline reduction_alg_kind_t from_alg(alg_kind_t alg, bool first, bool final) {
    using namespace alg_kind;
    switch (alg) {
        case (reduction_max): return reduction_alg_kind_t::max;
        case (reduction_min): return reduction_alg_kind_t::min;
        case (reduction_sum): return reduction_alg_kind_t::sum;
        case (reduction_mul): return reduction_alg_kind_t::mul;
        case (reduction_mean):
            return final ? reduction_alg_kind_t::mean
                         : reduction_alg_kind_t::sum;
        case (reduction_norm_lp_max):
            return first && final     ? reduction_alg_kind_t::lp_norm_max
                    : first && !final ? reduction_alg_kind_t::lp_norm_power_p
                    : !first && final ? reduction_alg_kind_t::pth_root_max
                                      : reduction_alg_kind_t::sum;
        case (reduction_norm_lp_sum):
            return first && final     ? reduction_alg_kind_t::lp_norm_sum
                    : first && !final ? reduction_alg_kind_t::lp_norm_power_p
                    : !first && final ? reduction_alg_kind_t::pth_root_sum
                                      : reduction_alg_kind_t::sum;
        case (reduction_norm_lp_power_p_max):
            return first && final ? reduction_alg_kind_t::lp_norm_power_p_max
                    : first && !final ? reduction_alg_kind_t::lp_norm_power_p
                    : !first && final ? reduction_alg_kind_t::final_max
                                      : reduction_alg_kind_t::sum;
        case (reduction_norm_lp_power_p_sum):
            return first && final ? reduction_alg_kind_t::lp_norm_power_p_sum
                    : first && !final ? reduction_alg_kind_t::lp_norm_power_p
                    : !first && final ? reduction_alg_kind_t::final_sum
                                      : reduction_alg_kind_t::sum;
        default: gpu_assert(false) << "Unexpected alg";
    }
    return reduction_alg_kind_t::undef;
}

inline int to_int(reduction_alg_kind_t alg) {
    return static_cast<int>(alg);
}

inline void def_reduction_alg_kinds(compute::kernel_ctx_t &kernel_ctx) {
#define CASE(alg, str) \
    kernel_ctx.define_int("REDUCTION_" str, to_int(reduction_alg_kind_t::alg));
    CASE(max, "MAX");
    CASE(min, "MIN");
    CASE(sum, "SUM");
    CASE(mul, "MUL");
    CASE(mean, "MEAN");
    CASE(lp_norm_max, "LP_NORM_MAX");
    CASE(lp_norm_sum, "LP_NORM_SUM");
    CASE(lp_norm_power_p_max, "LP_NORM_POWER_P_MAX");
    CASE(lp_norm_power_p_sum, "LP_NORM_POWER_P_SUM");
    CASE(lp_norm_power_p, "LP_NORM_POWER_P");
    CASE(pth_root_max, "PTH_ROOT_MAX");
    CASE(pth_root_sum, "PTH_ROOT_SUM");
    CASE(final_max, "FINAL_MAX");
    CASE(final_sum, "FINAL_SUM");
#undef CASE
}

// Zero padding splits one block into two, filling with
// zeros. This is a kind of reorder that can be used
// to short-circuit calculations to avoid reading/writing zeros.
struct zero_padding_t {
    zero_padding_t(const dim_idx_t dim_idx, const dim_t data_size,
            const dim_t outer_stride, const dim_t outer_size,
            const dim_t inner_stride, const dim_t inner_size)
        : dim_idx(dim_idx)
        , data_size(data_size)
        , outer_stride(outer_stride)
        , outer_size(outer_size)
        , inner_stride(inner_stride)
        , inner_size(inner_size) {}
    zero_padding_t(
            const dim_t data_size, const block_t &outer, const block_t &inner)
        : data_size(data_size)
        , outer_stride(outer.stride)
        , outer_size(outer.block)
        , inner_stride(inner.stride)
        , inner_size(inner.block) {
        assert(outer.dim_idx == inner.dim_idx);
        assert(outer.block * inner.block >= data_size);
        dim_idx = outer.dim_idx;
    }

    // Prints the indexing this zero-padding enforces. e.g.:
    // (idx / 1) % 16 + [(idx / 256) % 2] * 16 < 30 (aren't zeros)
    std::string str() const {
        std::stringstream os;
        os << dim_idx << ": ";
        os << "(idx / " << inner_stride << ") % " << inner_size;
        os << " + " << inner_size << " * (";
        os << "(idx / " << outer_stride << ") % " << outer_size;
        os << ") < " << data_size;
        return os.str();
    }

    dim_idx_t dim_idx;
    dim_t data_size;
    dim_t outer_stride, outer_size;
    dim_t inner_stride, inner_size;
};

class reduction_subproblem_t {
public:
    reduction_subproblem_t(
            dim_t inner_size, dim_t reduction_size, dim_t outer_size)
        : inner_block(0, inner_size, 1)
        , reduction_block(1, reduction_size, inner_size)
        , outer_block(2, outer_size, inner_size * reduction_size) {}

    std::string str() const {
        std::stringstream os;
        os << "subproblem:" << std::endl;
        os << "outer: " << outer_block.str() << std::endl;
        os << "reduction: " << reduction_block.str() << std::endl;
        os << "inner: " << inner_block.str() << std::endl;
        os << " -- src zero pads: " << src_zpads.size() << std::endl;
        for (const auto &zp : src_zpads) {
            os << "    + " << zp.str() << std::endl;
        }
        os << " -- dst zero pads: " << dst_zpads.size() << std::endl;
        for (const auto &zp : dst_zpads) {
            os << "    + " << zp.str() << std::endl;
        }
        return os.str();
    }

    block_t inner_block;
    block_t reduction_block;
    block_t outer_block;

    std::vector<zero_padding_t> src_zpads;
    std::vector<zero_padding_t> dst_zpads;
};

status_t generate_reduction_phases(const memory_desc_t *src,
        const memory_desc_t *dst, std::vector<reduction_subproblem_t> &subprbs);

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
