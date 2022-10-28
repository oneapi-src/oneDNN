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

    reorder_kernel_t(const exec_config_t &exec_cfg,
            const std::string &kernel_name, const kernel_info_t &kernel_info,
            const layout_t &src_layout, const layout_t &dst_layout,
            bool require_dpas, grf_mode_t grf_mode)
        : ir_kernel_t<hw>(
                kernel_name, exec_cfg, kernel_info, require_dpas, grf_mode) {

        if (reorder_kernel_t<>::is_ir_based_reorder(src_layout, dst_layout)) {
            reorder_ir_builder_t builder(
                    exec_cfg, kernel_info, src_layout, dst_layout);
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
        if (is_2d_reorder(src_layout, dst_layout, elems_per_thr)) {
            emit_2d_reorder(src_layout, dst_layout, elems_per_thr);
        } else {
            ir_error_not_expected();
        }

        generate_epilogue();
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
        return true;
    }

    static compute::nd_range_t nd_range(const exec_config_t &exec_cfg,
            const layout_t &src, const layout_t &dst) {
        const int simd = exec_cfg.simd();

        int elems_per_thr;
        if (is_2d_reorder(src, dst, elems_per_thr)) {
            ir_assert(src.elems() == dst.elems());
            return compute::nd_range_t(
                    {(int)utils::div_up(src.elems(), elems_per_thr) * simd, 1,
                            1});
        }

        // Handle IR-based reorder.
        ir_assert(reorder_kernel_t<>::is_ir_based_reorder(src, dst));

        return reorder_ir_builder_t::nd_range(exec_cfg, src, dst);
    }

private:
    static bool is_2d_reorder(
            const layout_t &src, const layout_t &dst, int &elems_per_thr) {
        if (!src.type().is_bitwise_compatible(dst.type())) return false;
        if (src.is_equal(dst)) return false;

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
