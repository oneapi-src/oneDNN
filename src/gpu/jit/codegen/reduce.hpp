/*******************************************************************************
* Copyright 2022-2023 Intel Corporation
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

#ifndef GPU_JIT_CODEGEN_REDUCE_HPP
#define GPU_JIT_CODEGEN_REDUCE_HPP

#include "gpu/jit/codegen/register_scope.hpp"
#include "gpu/jit/codegen/reorder.hpp"
#include "gpu/jit/ir/reduce.hpp"
#include "gpu/jit/ngen/ngen.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

class reduce_impl_t {
public:
    reduce_impl_t(ngen::HW hw, const reduce_t &reduce, int simd_size)
        : hw_(hw)
        , src_layout_(reduce.src_layout)
        , dst_layout_(reduce.dst_layout)
        , simd_size_(simd_size) {}

    template <typename GeneratorT>
    void emit(GeneratorT *host, ngen_register_scope_t &scope,
            const reg_buf_data_t &src_rd, const reg_buf_data_t &dst_rd) {
        auto &src_type = src_layout_.type();
        auto &dst_type = dst_layout_.type();

        bool is_inplace = (src_rd.base() == dst_rd.base()
                && src_rd.byte_offset() == dst_rd.byte_offset());
        if (is_inplace) {
            ir_assert(src_type == dst_type)
                    << "Inplace operation is supported for the same type only.";
        }

        std::vector<bool> seen(src_layout_.size() * src_type.size());

        tensor_t tile = find_1d_tile(src_layout_, dst_layout_);
        int tile_elems = (int)tile.elems();
        auto src_tile_layout = src_layout_.map(tile);
        auto src_tile_blocks = src_tile_layout.blocks();
        ir_assert(src_tile_blocks.size() <= 1);
        ngen_register_scope_t block_scope(scope.register_allocator());
        int src_stride
                = src_tile_blocks.empty() ? 1 : (int)src_tile_blocks[0].stride;
        int grf_size = ngen::GRF::bytes(hw_);
        src_layout_.for_each_tile(
                tile, [&](const std::vector<dim_t> &src_start) {
                    ngen_register_scope_t tile_scope(
                            scope.register_allocator());
                    auto dst_start = src_start;
                    for (int i = 0; i < dst_layout_.ndims(); i++) {
                        if (dst_layout_.dims()[i] == 1) dst_start[i] = 0;
                    }
                    int src_off = int(src_layout_(src_start) * src_type.size());
                    int dst_off = int(dst_layout_(dst_start) * dst_type.size());

                    if (is_inplace) {
                        bool same_src_dst = (dst_off == src_off);
                        if (!seen[dst_off] && !same_src_dst) {
                            ir_error_not_expected()
                                    << "Invalid inplace reduction.";
                        }
                        seen[dst_off] = true;
                        if (same_src_dst) return;
                    }

                    auto d = dst_rd.format(
                            dst_off, to_ngen(dst_type), tile_elems, 1);
                    auto s = src_rd.format(
                            src_off, to_ngen(src_type), tile_elems, src_stride);
                    bool s_half_grf_aligned
                            = utils::one_of(s.byte_offset(), 0, grf_size / 2);
                    bool s_is_bf = src_type.is_bf16();
                    bool s_is_hf = src_type.is_f16();
                    bool d_is_f = dst_type.is_f32();

                    if (src_stride != 1 || s_is_hf
                            || (s_is_bf && !s_half_grf_aligned)) {
                        auto tmp_type = src_type;
                        if ((s_is_hf && d_is_f)
                                || ((d.offset() != 0 || !s_half_grf_aligned)
                                        && (s_is_bf))) {
                            tmp_type = type_t::f32();
                        }
                        auto tmp = tile_scope.alloc_reg_data(
                                tmp_type.with_elems(tile_elems));
                        emit_reorder_1d_tile(hw_, host, tile_scope, tile_elems,
                                s, src_stride, tmp, 1);
                        s = tmp.format(0, to_ngen(tmp_type), tile_elems, 1);
                    }
                    align_src_dst_offset(host, tile_scope, tile_elems, d, s);
                    host->add(tile_elems, d.reg_data(), d.reg_data(),
                            s.reg_data());
                });
    }

private:
    tensor_t find_1d_tile(layout_t a, layout_t b) const {
        layout_t::align_layouts(a, b);

        ir_assert(!a.blocks().empty());
        // Allow trivial tile for scalar dst.
        if (b.blocks().empty()) { return tensor_t(dst_layout_.dims()); }

        auto &a0 = a.blocks()[0];
        auto &b0 = b.blocks()[0];

        bool ok = (a0.dim_idx == b0.dim_idx && a0.block == b0.block);
        if (!ok) {
            // Try to match strided layout.
            if (a0.block == 2) {
                auto a_blocks = a.blocks();
                a_blocks.erase(a_blocks.begin());
                a = layout_t(a.type(), a.ndims(), 0, a_blocks);
                return find_1d_tile(a, b);
            }
        }

        ir_assert(ok) << "Incompatible layouts for reduction.";
        ir_assert(dim_t(b0.stride) == 1)
                << "Reduction is not supported for non-unit dst stride.";

        int grf_size = ngen::GRF::bytes(hw_);
        int a_grf_elems = grf_size / a.type().size();
        int b_grf_elems = grf_size / b.type().size();

        int min_step = std::min(a_grf_elems, b_grf_elems);
        int max_step = 2 * min_step;

        min_step = std::min(
                std::min(hw_ <= ngen::HW::XeLP ? 8 : simd_size_, min_step),
                (int)a0.block);

        if (a0.block % min_step != 0) {
            // TODO: Extend implementation to support this case.
            ir_except_not_implemented("Reduction is not supported.");
        }

        std::vector<dim_t> tile_dims(src_layout_.ndims(), 1);
        tile_dims[a0.dim_idx]
                = ir_utils::max_divisor(int(a0.block), {min_step, max_step});

        return tensor_t(tile_dims);
    }

    ngen::HW hw_;
    layout_t src_layout_;
    layout_t dst_layout_;
    int simd_size_;
};

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
