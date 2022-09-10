/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#ifndef GPU_JIT_REORDER_REORDER_KERNEL_HPP
#define GPU_JIT_REORDER_REORDER_KERNEL_HPP

#include "gpu/jit/codegen/codegen.hpp"
#include "gpu/jit/codegen/kernel.hpp"
#include "gpu/jit/codegen/ngen_helpers.hpp"
#include "gpu/jit/codegen/register_scope.hpp"
#include "gpu/jit/ir/ir_builder.hpp"
#include "gpu/jit/ir/message.hpp"
#include "gpu/jit/ir/reorder.hpp"
#include "gpu/jit/ir/tensor.hpp"
#include "gpu/jit/reorder/ir_builder.hpp"
#include "gpu/jit/utils/ngen_type_bridge.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

template <ngen::HW hw = ngen::HW::Unknown>
class reorder_kernel_t : public ir_kernel_t<hw> {
public:
    IR_KERNEL_FORWARD(hw)

    reorder_kernel_t(const hw_config_t &hw_cfg,
            const kernel_info_t &kernel_info, const layout_t &src_layout,
            const layout_t &dst_layout, bool require_dpas, grf_mode_t grf_mode)
        : ir_kernel_t<hw>("reorder", hw_cfg, kernel_info, require_dpas,
                /*require_global_atomics=*/false, grf_mode) {

        if (reorder_kernel_t<>::is_ir_based_reorder(src_layout, dst_layout)) {
            reorder_ir_builder_t builder(
                    hw_cfg, kernel_info, src_layout, dst_layout);
            stmt_t body = builder.stmt();
            setup_interface(body);
            generate_prologue();
            expr_binding_t expr_binding(hw);
            bind_external_vars(body, builder.kernel_grid(), builder.local_id(),
                    expr_binding);

            // Generate assembly from IR.
            ir_to_ngen_t<hw> visitor(this, expr_binding);
            visitor.visit(body);

            generate_epilogue();
            return;
        }

        // Handle specific reorder versions.
        setup_interface();
        generate_prologue();

        std::vector<std::string> arg_names(kernel_info.nargs());
        for (int i = 0; i < kernel_info.nargs(); i++) {
            arg_names[i] = kernel_info.arg_name(i);
        }
        src_ptr_ = getArgument(arg_names[0]);
        dst_ptr_ = getArgument(arg_names[1]);
        src_surf_ = Surface(getArgumentSurface(arg_names[0]));
        dst_surf_ = Surface(getArgumentSurface(arg_names[1]));
        elems_ = getArgument(arg_names[2]);

        global_id_ = ra_.template alloc_sub<uint32_t>();

        mul(1, global_id_, r0.ud(1), getLocalSize(0).uw());
        add(1, global_id_, global_id_, getLocalID(0));

        int elems_per_thr;
        if (is_f32_to_bf16(src_layout, dst_layout)) {
            emit_f32_to_bf16();
        } else if (is_2d_reorder(src_layout, dst_layout, elems_per_thr)) {
            emit_2d_reorder(src_layout, dst_layout, elems_per_thr);
        } else {
            ir_error_not_expected();
        }

        generate_epilogue();
    }

    void emit_f32_to_bf16() {
        int elems_per_thr = f32_to_bf16_elems_per_thr();
        int simd_size = getSIMD();

        bool use_a64 = false;
        // XXX: Stateful messages don't work on XeHPC.
        use_a64 = (hw == ngen::HW::XeHPC);

        int grf_size = ngen::GRF::bytes(hw);
        int ud_size = sizeof(uint32_t);
        int uq_size = sizeof(uint64_t);
        int f_size = sizeof(float);
        int bf_size = sizeof(uint16_t);

        auto elem_vec = ra_.alloc_range(elems_per_thr * ud_size / grf_size);
        auto elem_vec_q_strided
                = ra_.alloc_range(elems_per_thr * uq_size / grf_size);
        auto src_ptr_vec = ra_.alloc_range(elems_per_thr * uq_size / grf_size);

        auto get_elem = [&](int i) {
            return get_subregister(hw, ngen::DataType::ud, elem_vec, i);
        };

        auto get_elem_q_strided = [&](int i) {
            return get_subregister(
                    hw, ngen::DataType::ud, elem_vec_q_strided, i * 2);
        };

        auto S = ra_.alloc_range(elems_per_thr * f_size / grf_size);
        // D is for bf16 but allocated as dword-strided to use with
        // scattered_byte(2) messages.
        auto D = ra_.alloc_range(elems_per_thr * f_size / grf_size);

        auto get_src_reg = [&](int i) {
            return get_subregister(hw, ngen::DataType::f, S, i);
        };

        auto get_dst_reg = [&](int i) {
            return get_subregister(hw, ngen::DataType::bf, D, i);
        };

        auto idx_vec = ra_.alloc().uw();
        mov(8, idx_vec, ngen::Immediate::uv(0, 1, 2, 3, 4, 5, 6, 7));
        for (int i = 0; i < elems_per_thr; i += 8)
            shl(8, get_elem(i), global_id_,
                    math::ilog2q(elems_per_thr / simd_size));
        for (int i = 0; i < elems_per_thr; i += 8) {
            add3(8, get_elem(i), get_elem(i), idx_vec, i);
            if (use_a64) {
                auto src_ptr_sub_vec = get_subregister(
                        hw, ngen::DataType::uq, src_ptr_vec, i)(1);
                emov(8, get_elem_q_strided(i)(2), get_elem(i)(1));
                eshl(8, src_ptr_sub_vec, get_elem_q_strided(i)(2),
                        math::ilog2q(f_size));
                eadd(8, src_ptr_sub_vec, src_ptr_sub_vec, src_ptr_);
            }
        }

        int elems_per_load = 16;
        for (int i = 0; i < elems_per_thr; i += elems_per_load) {
            cmp(16 | lt | f0[0], get_elem(i)(1), elems_);
            if (use_a64) {
                auto h_a64 = get_subregister(
                        hw, ngen::DataType::uq, src_ptr_vec, i);
                load(16 | f0[0], get_src_reg(i), ngen::scattered_dword(), A64,
                        h_a64);
            } else {
                auto h_bts = get_elem(i);
                load(16 | f0[0], get_src_reg(i), ngen::scattered_dword(),
                        src_surf_, h_bts);
            }
        }

        int mov_step = (grf_size == 32 ? 8 : 16);
        for (int i = 0; i < elems_per_thr; i += mov_step) {
            // dst is dword-strided.
            mov(mov_step, get_dst_reg(i * 2)(2), get_src_reg(i)(1));
        }

        auto dst_header = ra_.alloc_range(
                elems_per_load * (use_a64 ? uq_size : ud_size) / grf_size);
        for (int i = 0; i < elems_per_thr; i += elems_per_load) {
            for (int j = 0; j < elems_per_load; j += 8) {
                ngen::RegData h;
                if (use_a64) {
                    int off = j * uq_size;
                    h = dst_header[off / grf_size].uq(
                            (off % grf_size) / uq_size)(1);
                } else {
                    int off = j * ud_size;
                    h = dst_header[off / grf_size].ud(
                            (off % grf_size) / ud_size)(1);
                }
                emov(8, get_elem_q_strided(i + j)(2), get_elem(i + j)(1));
                eshl(8, h, get_elem_q_strided(i + j)(2), math::ilog2q(bf_size));
                if (use_a64) eadd(8, h, h, dst_ptr_);
            }

            cmp(16 | lt | f0[0], get_elem(i)(1), elems_);
            if (use_a64) {
                store(16 | f0[0], ngen::scattered_byte(2), A64, dst_header[0],
                        get_dst_reg(i * 2));
            } else {
                store(16 | f0[0], ngen::scattered_byte(2), dst_surf_,
                        dst_header[0], get_dst_reg(i * 2));
            }
        }
    }

    void emit_2d_reorder(
            const layout_t &_src, const layout_t &_dst, int elems_per_thr) {
        auto tile = _src.split_into_max_tile(elems_per_thr, /*is_dense=*/true);
        ir_assert(!tile.is_empty()) << "Can't split " << _src;

        int simd_size = getSIMD();
        auto src = _src.map(tile);
        auto dst = _dst.map(tile);

        int src_size = src.type().size();
        int dst_size = dst.type().size();
        int src_tile_bytes = src_size * elems_per_thr;
        int dst_tile_bytes = dst_size * elems_per_thr;

        int grf_size = ngen::GRF::bytes(hw);

        auto S = ra_.alloc_range(utils::div_up(src_tile_bytes, grf_size));
        auto D = ra_.alloc_range(utils::div_up(dst_tile_bytes, grf_size));

        auto src_header = ra_.alloc();
        auto dst_header = ra_.alloc();

        // Prepare headers for loads and stores.
        eshl(1, src_header.uq(0), global_id_,
                math::ilog2q(src_tile_bytes / simd_size));
        eshl(1, dst_header.uq(0), global_id_,
                math::ilog2q(dst_tile_bytes / simd_size));
        eadd(1, src_header.uq(0), src_header.uq(0), src_ptr_);
        eadd(1, dst_header.uq(0), dst_header.uq(0), dst_ptr_);

        int oword_bytes = 16;

        // Load source tile.
        int src_off = 0;
        while (src_tile_bytes > 0) {
            for (int i = 3; i >= 0; i--) {
                int size = (1 << i) * oword_bytes;
                if (src_tile_bytes >= size) {
                    load(16, S[src_off / grf_size],
                            ngen::block_oword(size / oword_bytes), A64,
                            src_header);
                    eadd(1, src_header.uq(0), src_header.uq(0), size);
                    src_tile_bytes -= size;
                    src_off += size;
                    break;
                }
            }
        }

        // Reorder source tile to destination tile.
        ngen_register_scope_t scope(ra_);
        reorder_2d_impl_t r(hw, src, dst);
        reg_buf_t S_buf(hw, S);
        reg_buf_t D_buf(hw, D);
        r.emit(this, scope, S_buf, D_buf);

        // Store destination tile.
        int dst_off = 0;
        while (dst_tile_bytes > 0) {
            for (int i = 3; i >= 0; i--) {
                int size = (1 << i) * oword_bytes;
                if (dst_tile_bytes >= size) {
                    store(16, ngen::block_oword(size / oword_bytes), A64,
                            dst_header, D[dst_off / grf_size]);
                    eadd(1, dst_header.uq(0), dst_header.uq(0), size);
                    dst_tile_bytes -= size;
                    dst_off += size;
                    break;
                }
            }
        }
    }

    static bool is_ir_based_reorder(const layout_t &src, const layout_t &dst) {
        int dummy;
        if (is_2d_reorder(src, dst, dummy)) return false;
        if (is_f32_to_bf16(src, dst)) return false;
        return true;
    }

    static compute::nd_range_t nd_range(
            int simd, const layout_t &src, const layout_t &dst) {

        if (is_f32_to_bf16(src, dst)) {
            ir_assert(src.elems() == dst.elems());
            int elems_per_thr = f32_to_bf16_elems_per_thr();
            return compute::nd_range_t(
                    {(int)utils::div_up(src.elems(), elems_per_thr) * simd, 1,
                            1});
        }

        int elems_per_thr;
        if (is_2d_reorder(src, dst, elems_per_thr)) {
            ir_assert(src.elems() == dst.elems());
            return compute::nd_range_t(
                    {(int)utils::div_up(src.elems(), elems_per_thr) * simd, 1,
                            1});
        }

        // Handle IR-based reorder.
        ir_assert(reorder_kernel_t<>::is_ir_based_reorder(src, dst));

        return reorder_ir_builder_t::nd_range(simd, src, dst);
    }

private:
    static bool is_f32_to_bf16(const layout_t &src, const layout_t &dst) {
        if (src.type() != type_t::f32()) return false;
        if (dst.type() != type_t::bf16()) return false;
        if (src.retype(type_t::u8()) != dst.retype(type_t::u8())) return false;
        return true;
    }

    static int f32_to_bf16_elems_per_thr() { return 32; }

    static bool is_2d_reorder(
            const layout_t &src, const layout_t &dst, int &elems_per_thr) {
        return false;
        if (!src.type().is_bitwise_compatible(dst.type())) return false;

        const int hword_bytes = 32;
        const int min_bytes_per_thr = hword_bytes;
        const int max_bytes_per_thr = 32 * hword_bytes;

        int type_size = src.type().size();
        int max_elems_per_thr = max_bytes_per_thr / type_size;

        auto tile = reorder_2d_impl_t::find_2d_tile(
                src, dst, max_elems_per_thr, /*match_outer=*/true);

        if (tile.is_empty()) return false;
        if (tile.ndims() < 2) return false;

        elems_per_thr = tile.elems();
        if (!math::is_pow2(elems_per_thr)) return false;

        int bytes_per_thr = elems_per_thr * type_size;
        if (bytes_per_thr % hword_bytes != 0) return false;
        if (bytes_per_thr < min_bytes_per_thr) return false;
        if (bytes_per_thr > max_bytes_per_thr) return false;

        return true;
    }

    ngen::Subregister src_ptr_;
    ngen::Subregister dst_ptr_;
    ngen::AddressBase src_surf_;
    ngen::AddressBase dst_surf_;
    ngen::Subregister elems_;
    ngen::Subregister global_id_;
};

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
