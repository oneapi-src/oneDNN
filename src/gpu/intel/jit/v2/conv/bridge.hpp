/*******************************************************************************
* Copyright 2023-2025 Intel Corporation
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

#ifndef GPU_INTEL_JIT_V2_CONV_BRIDGE_HPP
#define GPU_INTEL_JIT_V2_CONV_BRIDGE_HPP

#include "common/convolution_pd.hpp"
#include "gpu/intel/jit/ir/core.hpp"
#include "gpu/intel/jit/ir/message.hpp"
#include "gpu/intel/jit/ir/tensor.hpp"
#include "gpu/intel/jit/v2/conv/plan.hpp"
#include "gpu/intel/jit/v2/conv/tensor_utils.hpp"
#include "gpu/intel/jit/v2/ir/bridge.hpp"
#include "gpu/intel/jit/v2/ir/tensor.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {
namespace v2 {
namespace conv {

inline pvar_tile_t to_shape(const convolution_pd_t *pd) {
    pvar_tile_t shape;
    shape[pvars::mb] = pd->MB();
    shape[pvars::ic] = ir_utils::safe_div(pd->IC(), pd->G());
    shape[pvars::oc] = ir_utils::safe_div(pd->OC(), pd->G());
    shape[pvars::g] = pd->G();
    shape[pvars::id] = pd->ID();
    shape[pvars::ih] = pd->IH();
    shape[pvars::iw] = pd->IW();
    shape[pvars::od] = pd->OD();
    shape[pvars::oh] = pd->OH();
    shape[pvars::ow] = pd->OW();
    shape[pvars::kd] = pd->KD();
    shape[pvars::kh] = pd->KH();
    shape[pvars::kw] = pd->KW();
    shape[pvars::sd] = pd->KSD();
    shape[pvars::sh] = pd->KSH();
    shape[pvars::sw] = pd->KSW();
    shape[pvars::dd] = pd->KDD();
    shape[pvars::dh] = pd->KDH();
    shape[pvars::dw] = pd->KDW();
    shape[pvars::pd] = pd->padFront();
    shape[pvars::ph] = pd->padT();
    shape[pvars::pw] = pd->padL();
    return shape;
}

inline problem_t to_problem(
        const convolution_pd_t *pd, const impl::engine_t *engine) {
    auto prop = pd->desc()->prop_kind;
    auto src = make_conv_layout_tag(
            tensor_kind_t::src, pd->ndims(), *pd->invariant_src_md());
    auto wei = make_conv_layout_tag(
            tensor_kind_t::wei, pd->ndims(), *pd->invariant_wei_md());
    auto dst = make_conv_layout_tag(
            tensor_kind_t::dst, pd->ndims(), *pd->invariant_dst_md());
    auto shape = to_shape(pd);

    problem_t prb;
    prb.set_hw(hw_t(engine));
    prb.set_prop(prop);
    prb.set_bias_type(type_t(pd->invariant_bia_md()->data_type));
    prb.set_src_tag(src);
    prb.set_wei_tag(wei);
    prb.set_dst_tag(dst);
    prb.set_shape(shape);
    prb.normalize();

    return prb;
}

inline jit::layout_t to_conv_layout(const layout_tag_t &_tag,
        const memory_desc_t &md, bool remove_a_dim = false) {
    auto tag = _tag.raw_tag();
    dim_idx_t non_spatial_ndims = tag.ndims() - 3;
    if (remove_a_dim) {
        tag.remove_dim('a');
        non_spatial_ndims--;
    }
    while (tag.ndims() > into<dim_idx_t>(md.ndims)) {
        tag.remove_dim(dim_idx::as_tag(non_spatial_ndims));
    }
    return jit::layout_t(md, tag.str(), /*do_normalize=*/false);
}

inline jit::grid_info_t to_grid_info(
        const grid_t &grid, const pvar_tile_t &tile) {
    std::vector<dim_t> dims;
    std::vector<expr_t> idxs;
    for (int i = 0; i < grid_t::N; i++) {
        dim_t size = grid.size(i, tile);
        dims.push_back(size);
        idxs.push_back(size == 1 ? 0 : grid.index_var(i));
    }
    return jit::grid_info_t(dims, idxs);
}

} // namespace conv
} // namespace v2
} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
