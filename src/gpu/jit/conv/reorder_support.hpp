/*******************************************************************************
* Copyright 2021-2022 Intel Corporation
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

#ifndef GPU_JIT_CONV_REORDER_SUPPORT_HPP
#define GPU_JIT_CONV_REORDER_SUPPORT_HPP

#include <array>
#include <memory>
#include <string>

#include "gpu/jit/conv/ir.hpp"
#include "gpu/jit/conv/tensor.hpp"
#include "gpu/jit/ngen/ngen.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

// Helper class to permute registers. Used to permute registers after applying
// dpas -> dpasw transformation.
class grf_permutation_t {
public:
    grf_permutation_t() { permutation_.fill(-1); }

    int map(int off) const {
        ir_assert(off >= 0 && off < max_regs);
        if (permutation_[off] == -1) return off;
        return permutation_[off];
    }

    bool is_empty() const { return is_empty_; }

    void set_permute(int old_off, int new_off) {
        ir_assert(old_off >= 0 && old_off < max_regs);
        if (old_off == new_off || new_off == -1) return;
        is_empty_ = false;
        ir_assert(permutation_[old_off] == -1) << "Already assigned.";
        permutation_[old_off] = new_off;
    }

    bool operator==(const grf_permutation_t &other) const {
        for (int i = 0; i < max_regs; i++) {
            if (permutation_[i] != other.permutation_[i]) return false;
        }
        return true;
    }

    bool operator!=(const grf_permutation_t &other) const {
        return !operator==(other);
    }

private:
    static const int max_regs = 256;

    std::array<int, max_regs> permutation_;
    bool is_empty_ = true;
};

// Implements reorder between GRF buffers in given layouts. Conversion between
// data types is supported.
class reorder_t : public func_impl_t {
public:
    IR_DECL_DERIVED_TYPE_ID(reorder_t, func_impl_t)

    static func_t make(const layout_t &src_layout, const layout_t &dst_layout) {
        return func_t(new reorder_t(src_layout, dst_layout));
    }

    bool is_equal(const object_impl_t &obj) const override {
        if (!obj.is<self_type>()) return false;
        auto &other = obj.as<self_type>();

        return (src_layout == other.src_layout)
                && (dst_layout == other.dst_layout);
    }

    size_t get_hash() const override {
        return ir_utils::get_hash(src_layout, dst_layout);
    }

    std::string str() const override {
        std::ostringstream oss;
        oss << "reorder[" << src_layout << ", " << dst_layout << "]";
        return oss.str();
    }

    IR_DEFINE_ARG_GET(dst_buf, 0)
    IR_DEFINE_ARG_GET(src_buf, 1)

    layout_t src_layout;
    layout_t dst_layout;

private:
    reorder_t(const layout_t &src_layout, const layout_t &dst_layout)
        : src_layout(src_layout), dst_layout(dst_layout) {}
};

inline stmt_t create_reorder_stmt(const layout_t &src, const layout_t &dst,
        const expr_t &src_buf, const expr_t &dst_buf) {
    ir_assert(src.ndims() == dst.ndims()) << "Layouts are incompatible.";
    ir_assert(src.elems() == dst.elems()) << "Layouts are incompatible.";
    auto func = reorder_t::make(src, dst);
    return func.call({dst_buf, src_buf});
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
