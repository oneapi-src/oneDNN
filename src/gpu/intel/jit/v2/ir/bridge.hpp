/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#ifndef GPU_INTEL_JIT_V2_IR_BRIDGE_HPP
#define GPU_INTEL_JIT_V2_IR_BRIDGE_HPP

#include "gpu/intel/jit/ir/message.hpp"
#include "gpu/intel/jit/ir/tensor.hpp"
#include "gpu/intel/jit/v2/ir/send.hpp"
#include "gpu/intel/jit/v2/ir/tensor.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {
namespace v2 {

inline jit::send_address_t to_ir(send_address_t address) {
    jit::send_address_t ret = jit::send_address_t::a64;
    switch (address) {
#define CASE(name) \
    case v2::send_address_t::name: ret = jit::send_address_t::name; break;
        CASE(a64);
        CASE(slm);
#undef CASE
        default: ir_error_not_expected();
    }
    return ret;
}

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
    ir_assert(layout.has_const_sizes());
    ir_assert(layout.has_const_strides());
    std::vector<gpu::intel::block_t> blocks;
    for (auto &b : layout.blocks()) {
        int dim_idx = layout.desc().dim_index(b.dim);
        blocks.emplace_back(dim_idx, b.int_size(), b.int_stride());
    }

    return jit::layout_t(
            layout.type(), layout.desc().ndims(), layout.base(), blocks);
}

} // namespace v2
} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
