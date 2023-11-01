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

#ifndef GPU_JIT_V2_CONV_BRIDGE_HPP
#define GPU_JIT_V2_CONV_BRIDGE_HPP

#include "common/convolution_pd.hpp"
#include "gpu/jit/ir/core.hpp"
#include "gpu/jit/ir/message.hpp"
#include "gpu/jit/ir/tensor.hpp"
#include "gpu/jit/v2/conv/plan.hpp"
#include "gpu/jit/v2/ir/tensor.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {
namespace v2 {
namespace conv {

inline jit::send_op_t to_ir(send_op_t op, bool is_2d = false) {
    jit::send_op_t ret = jit::send_op_t::undef;
    switch (op) {
#define CASE(name) \
    case v2::send_op_t::name: ret = jit::send_op_t::name; break;
        CASE(atomic_fadd);
        CASE(load);
        CASE(prefetch);
        CASE(store);
#undef CASE
        default: ir_error_not_expected();
    }
    if (is_2d) {
        switch (ret) {
            case jit::send_op_t::load: ret = jit::send_op_t::load_2d; break;
            case jit::send_op_t::prefetch:
                ret = jit::send_op_t::prefetch_2d;
                break;
            case jit::send_op_t::store: ret = jit::send_op_t::store_2d; break;
            default: ir_error_not_expected();
        }
    }
    return ret;
}

inline jit::layout_t to_ir(const layout_t &layout) {
    ir_assert(layout.has_zero_base());
    ir_assert(layout.has_const_sizes());
    ir_assert(layout.has_const_strides());
    std::vector<gpu::block_t> blocks;
    for (auto &b : layout.blocks()) {
        int dim_idx = layout.desc().dim_index(b.dim);
        blocks.emplace_back(dim_idx, b.int_size(), b.int_stride());
    }

    return jit::layout_t(layout.type(), layout.desc().ndims(), 0, blocks);
}

inline prb_tile_t to_shape(const convolution_pd_t *pd) {
    prb_tile_t shape;
    shape[prb_dims::mb] = pd->MB();
    shape[prb_dims::ic] = ir_utils::safe_div(pd->IC(), pd->G());
    shape[prb_dims::oc] = ir_utils::safe_div(pd->OC(), pd->G());
    shape[prb_dims::g] = pd->G();
    shape[prb_dims::id] = pd->ID();
    shape[prb_dims::ih] = pd->IH();
    shape[prb_dims::iw] = pd->IW();
    shape[prb_dims::od] = pd->OD();
    shape[prb_dims::oh] = pd->OH();
    shape[prb_dims::ow] = pd->OW();
    shape[prb_dims::kd] = pd->KD();
    shape[prb_dims::kh] = pd->KH();
    shape[prb_dims::kw] = pd->KW();
    shape[prb_dims::sd] = pd->KSD();
    shape[prb_dims::sh] = pd->KSH();
    shape[prb_dims::sw] = pd->KSW();
    shape[prb_dims::dd] = pd->KDD();
    shape[prb_dims::dh] = pd->KDH();
    shape[prb_dims::dw] = pd->KDW();
    shape[prb_dims::pd] = pd->padFront();
    shape[prb_dims::ph] = pd->padT();
    shape[prb_dims::pw] = pd->padL();
    return shape;
}

inline problem_t to_problem(
        const convolution_pd_t *pd, const engine_t *engine) {
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
    prb.set_src_tag(src);
    prb.set_wei_tag(wei);
    prb.set_dst_tag(dst);
    prb.set_shape(shape);

    return prb;
}

} // namespace conv
} // namespace v2
} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
