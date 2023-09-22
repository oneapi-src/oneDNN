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

#ifndef GPU_JIT_CODEGEN_REORDER_HPP
#define GPU_JIT_CODEGEN_REORDER_HPP

#include "common/utils.hpp"
#include "gpu/jit/codegen/operand.hpp"
#include "gpu/jit/codegen/register_scope.hpp"
#include "gpu/jit/ir/reorder.hpp"
#include "gpu/jit/ir/tensor.hpp"
#include "gpu/jit/ngen/ngen.hpp"
#include "gpu/jit/utils/iterator.hpp"
#include "gpu/jit/utils/range.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

// Rewrites a single-src instruction to avoid GRF boundary issues.
struct op_plan_t {
private:
    template <typename... ArgT>
    using op_t = std::function<void(ArgT...)>;

    using inst_mod_t = ngen::InstructionModifier;
    using reg_data_t = ngen::RegData;
    using single_src_op_t = op_t<inst_mod_t, reg_data_t, reg_data_t>;

public:
    op_plan_t(int grf_size) : grf_size_(grf_size) {}

    void operator()(const single_src_op_t &op, inst_mod_t mod, reg_data_t dst,
            reg_data_t src) const {
        // Rewrite a single-src instruction that spans more than 2 GRFs into
        // multiple ops
        auto dst_esize = max_esize(dst, /*is_dst=*/true);
        auto src_esize = max_esize(src, /*is_dst=*/false);
        auto original_esize = mod.getExecSize();
        auto original_width = src.getWidth();
        auto esize = std::min(std::min(dst_esize, src_esize), original_esize);

        mod.setExecSize(esize);
        if (esize < original_width)
            // Width must be at most esize
            set_contiguous_region(src, esize, src.getHS());

        for (int i = 0; i < original_esize; i += esize) {
            fixup(op, mod, dst, src);
            shift_offset(dst, esize * dst.getHS());
            shift_offset(src, esize * src.getHS());
        }
    }

private:
    int max_esize(const reg_data_t &reg, bool is_dst) const {
        auto size = reg.getBytes();
        auto width = reg.getWidth();
        auto hs = reg.getHS();
        auto vs = reg.getVS();
        auto remaining_bytes = 2 * grf_size_ - reg.getByteOffset();
        auto stride = hs;
        if (!is_dst && width == 1) stride = vs;
        if (is_dst && stride == 0) stride = 1;
        if (stride == 0) return 16; // Broadcast can have max step
        auto max_step = (remaining_bytes - 1) / (stride * size) + 1;
        return utils::rnd_down_pow2(max_step);
    }

    void fixup(const single_src_op_t &op, inst_mod_t mod, reg_data_t dst,
            reg_data_t src) const {
        // Rewrite src0 to cross GRF boundaries using vertical striding
        auto exec_size = mod.getExecSize();
        auto offset = src.getOffset();
        auto width = src.getWidth();
        auto hs = src.getHS();
        auto vs = src.getVS();
        auto size = src.getBytes();

        if (!width) width = exec_size;
        auto height = exec_size / width;
        auto grf_elems = grf_size_ / size;

        bool crosses_grf_boundary = false;
        auto begin = offset;
        for (int i = 0; i < height; ++i) {
            auto reg_off = begin % grf_elems;
            crosses_grf_boundary
                    |= (reg_off + (width - 1) * hs + 1 > grf_elems);
            begin += vs;
        }

        if (!crosses_grf_boundary) {
            // op is valid
            op(mod, dst, src);
        } else if (vs == width * hs) {
            // rewrite src as a valid access with shorter width and vs
            auto elems_to_grf_boundary = (grf_elems - offset - 1) / hs + 1;
            auto tentative_width = utils::rnd_down_pow2(elems_to_grf_boundary);
            while (tentative_width > 1) {
                if (elems_to_grf_boundary % tentative_width == 0) break;
                tentative_width /= 2;
            }

            set_contiguous_region(src, tentative_width, hs);
            op(mod, dst, src);
        } else {
            // break op into multiple row-wise ops
            mod.setExecSize(width);
            set_contiguous_region(src, width, hs);
            for (int i = 0; i < height; ++i) {
                fixup(op, mod, dst, src);
                shift_offset(dst, width * dst.getHS());
                shift_offset(src, vs);
            }
        }
    }

    void set_contiguous_region(reg_data_t &rr, int width, int hs) const {
        if (width > 1)
            rr.setRegion(width * hs, width, hs);
        else
            // Each element occupies its own row. width = 1 requires hs = 0
            rr.setRegion(hs, 1, 0);
    }

    void shift_offset(reg_data_t &rr, int offset) const {
        auto new_offset = rr.getOffset() + offset;
        auto type_size = rr.getBytes();
        auto grf_elems = grf_size_ / type_size;
        rr.setBase(rr.getBase() + new_offset / grf_elems);
        rr.setOffset(new_offset % grf_elems);
    };

    int grf_size_;
};

// Aligns src offset with dst offset when src is not broadcasted.
template <typename GeneratorT>
void align_src_dst_offset(GeneratorT *host, ngen_register_scope_t &scope,
        const ngen::InstructionModifier &mod, const reg_buf_data_t &dst,
        reg_buf_data_t &src);

template <typename GeneratorT>
bool try_emit_batched_reorder_1d_tile(ngen::HW hw, GeneratorT *host,
        ngen_register_scope_t &scope, int width, const reg_buf_data_t &src,
        int src_stride, const reg_buf_data_t &dst, int dst_stride) {
    ngen::DataType src_type = src.type();
    ngen::DataType dst_type = dst.type();
    int src_type_size = ngen::getBytes(src_type);
    int dst_type_size = ngen::getBytes(dst_type);
    auto large_type = (src_type_size > dst_type_size) ? src_type : dst_type;
    auto small_type = (src_type_size < dst_type_size) ? src_type : dst_type;
    ngen_register_scope_t lex_scope {scope.register_allocator()};

    if (!utils::one_of(large_type, ngen::DataType::f, ngen::DataType::d))
        return false;
    if (!utils::one_of(small_type, ngen::DataType::b, ngen::DataType::ub))
        return false;
    if (src_stride != 1) return false;
    if (dst_stride != 1) return false;

    int batch = 128;
    int max_step = 8;
    // Small width may indicate many small reorders, which may require alignment
    // workarounds. Defer to the non-batched implemenation.
    if (width < max_step) return false;

    const int grf_size = ngen::GRF::bytes(hw);
    op_plan_t plan = grf_size;
    int tmp_regs = utils::div_up(int(batch * sizeof(uint32_t)), grf_size);
    auto tmp_range = lex_scope.try_alloc_range(tmp_regs);
    if (tmp_range.isInvalid()) return false;
    reg_buf_data_t tmp(reg_buf_t(hw, tmp_range));
    using inst_mod_t = ngen::InstructionModifier;
    using reg_data_t = ngen::RegData;
    auto mov = [&](inst_mod_t mod, reg_data_t dst, reg_data_t src) {
        host->emov(mod, dst, src);
    };

    // Do not attempt to match offsets when not moving data between pipes
    const auto dst_off = (to_ir(dst_type).is_fp()) ? dst.byte_offset() : 0;
    for (int i = 0; i < width; i += batch) {
        int i_beg = i;
        int i_end = std::min(width, i + batch);

        for (int ii = i_beg; ii < i_end;) {
            int esize = std::min(max_step, i_end - ii);
            esize = utils::rnd_down_pow2(esize);

            auto s = src.subregister(ii, esize, src_type_size);
            auto t = tmp.subregister(dst_off + (ii - i_beg) * 4, small_type)(4);
            ngen::InstructionModifier mod = esize;
            if (src_type == ngen::DataType::f && hw == ngen::HW::Gen9)
                host->rnde(esize, s(1), s(1));
            if (dst_type == small_type) mod |= host->sat;
            plan(mov, mod, t, s(1));
            ii += esize;
        }
        for (int ii = i_beg; ii < i_end;) {
            int esize = std::min(max_step, i_end - ii);
            esize = utils::rnd_down_pow2(esize);

            auto d = dst.subregister(ii, esize, dst_type_size);
            auto t = tmp.subregister(dst_off + (ii - i_beg) * 4, small_type)(4);
            plan(mov, esize, d(1), t);
            ii += esize;
        }
    }
    return true;
}

// Performs 1D reorder, possibly with strides and type conversion.
template <typename GeneratorT>
void emit_reorder_1d_tile(ngen::HW hw, GeneratorT *host,
        ngen_register_scope_t &scope, int width, const reg_buf_data_t &_src,
        int src_stride, const reg_buf_data_t &_dst, int dst_stride) {

    if (try_emit_batched_reorder_1d_tile(
                hw, host, scope, width, _src, src_stride, _dst, dst_stride))
        return;

    auto src = _src;
    auto dst = _dst;
    ngen::DataType src_type = src.type();
    ngen::DataType dst_type = dst.type();
    // Replace (float -> float) by (int -> int) as word/dword moves have less
    // restrictions.
    if (src_type == dst_type
            && utils::one_of(src_type, ngen::DataType::bf, ngen::DataType::hf,
                    ngen::DataType::f, ngen::DataType::df)) {
        int factor = (src_type == ngen::DataType::df ? 2 : 1);
        if (factor == 1 || (src_stride == 1 && dst_stride == 1)) {
            src_type
                    = to_ngen(type_t::u(ngen::getBytes(src_type) / factor * 8));
            dst_type = src_type;
            width *= factor;
            src = src.reinterpret(src_type);
            dst = dst.reinterpret(dst_type);
        }
    }

    const int grf_size = ngen::GRF::bytes(hw);
    int src_type_size = ngen::getBytes(src_type);
    int dst_type_size = ngen::getBytes(dst_type);
    int src_stride_bytes = src_stride * src_type_size;
    int dst_stride_bytes = dst_stride * dst_type_size;
    bool dst_b = ngen_is_b(dst_type);
    bool dst_d = ngen_is_dw(dst_type);
    bool dst_q = ngen_is_qw(dst_type);
    bool dst_f = (dst_type == ngen::DataType::f);
    bool dst_hf = (dst_type == ngen::DataType::hf);
    bool dst_bf = (dst_type == ngen::DataType::bf);
    bool dst_df = (dst_type == ngen::DataType::df);
    bool dst_xf = dst_bf || dst_f || dst_hf || dst_df;
    bool src_b = ngen_is_b(src_type);
    bool src_d = ngen_is_dw(src_type);
    bool src_q = ngen_is_qw(src_type);
    bool src_f = (src_type == ngen::DataType::f);
    bool src_hf = (src_type == ngen::DataType::hf);
    bool src_bf = (src_type == ngen::DataType::bf);
    bool src_df = (src_type == ngen::DataType::df);
    bool src_xf = src_bf || src_f || src_hf || src_df;
    bool f_to_xf = (src_f && (dst_bf || dst_hf));
    op_plan_t plan = grf_size;
    ngen_register_scope_t lex_scope {scope.register_allocator()};

    auto get_step = [&]() {
        int step = (width < 16 ? 8 : 16);

        // f32 -> bf16 or f32 -> f16: SIMD16 does not support mixed mode move.
        if (hw < ngen::HW::XeHPC)
            if (f_to_xf) step = 8;

        //dpasw as well as potential bank conflict allocation permutes registers; -> so use register granularity
        if (hw < ngen::HW::XeHPC
                && (!src.check_bounds(0, 64, /*is_dense=*/true)
                        || !dst.check_bounds(0, 64, /*is_dense=*/true)))
            step = 8;

        if (src_df || dst_df) step = 8;

        // Max supported stride is 4.
        if (src_stride > 4 || dst_stride > 4) step = 1;

        // Qword does not appear to support swizzling.
        if (src_q && dst_q && src_stride != dst_stride) step = 1;

        return step;
    };

    using inst_mod_t = ngen::InstructionModifier;
    using reg_data_t = ngen::RegData;
    auto shl16 = [&](inst_mod_t mod, reg_data_t dst, reg_data_t src) {
        host->eshl(mod, dst, src, 16);
    };
    auto mov = [&](inst_mod_t mod, reg_data_t dst, reg_data_t src) {
        host->emov(mod, dst, src);
    };

    // bf16 -> f32:
    // - bf16 must be packed: use left shift instead.
    if (src_bf && dst_f) {
        int step = get_step();
        for (int i = 0; i < width; i += step) {
            step = std::min(step, width - i);
            step = utils::rnd_down_pow2(step);
            int esize = step;
            auto s = src.subregister(
                    i, esize, src_stride_bytes, ngen::DataType::uw);
            auto d = dst.subregister(
                    i, esize, dst_stride_bytes, ngen::DataType::ud);
            plan(shl16, esize, d(dst_stride), s(src_stride));
        }
        return;
    }

    // d -> bf/hf:
    // - Use d -> f -> bf/hf conversion with temporary
    if (src_d && (dst_bf || dst_hf)) {
        const int nregs = utils::div_up(width * (int)sizeof(float), grf_size);
        auto tmp = lex_scope.alloc_reg_buf_data(nregs).format(
                0, ngen::DataType::f);
        emit_reorder_1d_tile(hw, host, scope, width, src, src_stride, tmp, 1);
        emit_reorder_1d_tile(hw, host, scope, width, tmp, 1, dst, dst_stride);
        return;
    }

    // b -> hf
    // - Direct b -> float conversion not supported: use s16 temporary
    // - int -> hf must be DW-aligned & strided: use f temporary
    // - Use b -> w -> f -> hf
    if (src_b && dst_hf) {
        ir_assert(utils::one_of(dst_stride_bytes, 2, 4));
        ir_assert(utils::one_of(src_stride_bytes, 1, 4));
        int step = get_step();
        const int align_boundary = grf_size / 2;
        const int step_size = step * (int)sizeof(uint32_t);
        const int nregs = utils::div_up(step_size, grf_size);
        auto tmp1 = lex_scope.alloc_reg_buf_data(nregs);
        auto tmp2 = lex_scope.alloc_reg_buf_data(nregs);
        for (int i = 0; i < width; i += step) {
            step = std::min(step, width - i);
            step = utils::rnd_down_pow2(step);
            int esize = step;

            auto s = src.subregister(i, esize, src_stride_bytes);
            auto d = dst.subregister(i, esize, dst_stride_bytes);
            auto byte_offset = 2 * (d.getByteOffset() % align_boundary);
            auto t1 = tmp1.subregister(byte_offset, ngen::DataType::w);
            auto t2 = tmp2.subregister(byte_offset, ngen::DataType::f);
            auto t1_as_hf = t1.reinterpret(0, ngen::DataType::hf);
            auto d_as_w = d.reinterpret(0, ngen::DataType::w);

            plan(mov, esize, t1(2), s(src_stride));
            plan(mov, esize, t2(1), t1(2));
            plan(mov, esize, t1_as_hf(2), t2(1));
            plan(mov, esize, d_as_w(dst_stride), t1(2));
        }
        return;
    }

    // hf -> b
    if (src_hf && dst_b) {
        ir_assert(utils::one_of(src_stride_bytes, 2, 4));
        ir_assert(utils::one_of(dst_stride_bytes, 1, 4));
        int step = get_step();
        const int tmp_stride = 4;
        const int tmp_stride_bytes = tmp_stride * dst_type_size;
        const int step_size = step * tmp_stride_bytes;
        const int nregs = 1 + utils::div_up(step_size, grf_size);
        auto tmp1 = lex_scope.alloc_reg_buf_data(nregs);
        auto tmp2 = lex_scope.alloc_reg_buf_data(nregs);
        for (int i = 0; i < width; i += step) {
            step = std::min(step, width - i);
            step = utils::rnd_down_pow2(step);
            int esize = step;

            auto s = src.subregister(i, esize, src_stride_bytes);
            auto d = dst.subregister(i, esize, dst_stride_bytes);
            const int t1_offset = (esize == 1) ? 0 : s.getByteOffset();
            const int t2_offset = (d.getOffset() % 16) * tmp_stride_bytes;
            auto t1 = tmp1.subregister(t1_offset, dst_type);
            auto t2 = tmp2.subregister(t2_offset, dst_type);

            if (esize == 1) {
                if (hw == ngen::HW::Gen9) {
                    auto t1_f = tmp1.subregister(0, ngen::DataType::f);
                    plan(mov, 1, t1_f, s);
                    host->rnde(1, t1_f, t1_f);
                    auto t2_h = tmp2.subregister(t1_offset, ngen::DataType::hf);
                    plan(mov, 1, t2_h, t1_f);
                    s = t2_h;
                }
                plan(mov, 2 | host->sat, t1(tmp_stride), s);
                plan(mov, 1, d, t1);
                continue;
            }

            // Operands are already dword aligned as required by F-pipe
            if (dst_stride_bytes >= tmp_stride_bytes) {
                if (hw != ngen::HW::Gen9) {
                    plan(mov, esize | host->sat, d(dst_stride), s(src_stride));
                } else {
                    ir_assert(dst_stride_bytes % tmp_stride_bytes == 0);
                    auto d_f = dst.format(i * ngen::getBytes(ngen::DataType::f),
                            ngen::DataType::f, esize,
                            dst_stride_bytes / tmp_stride_bytes);
                    plan(mov, esize, d_f, s(src_stride));
                    host->rnde(esize, d_f, d_f);
                    plan(mov, esize | host->sat, d(dst_stride), d_f);
                }
                continue;
            }

            if (hw != ngen::HW::Gen9) {
                plan(mov, esize | host->sat, t1(tmp_stride), s(src_stride));
            } else {
                auto t1_f = tmp1.format(t1_offset, ngen::DataType::f, esize);
                plan(mov, esize, t1_f, s(src_stride));
                host->rnde(esize, t1_f, t1_f);
                plan(mov, esize | host->sat, t1(tmp_stride), t1_f);
            }
            if (t1_offset != t2_offset)
                plan(mov, esize, t2(tmp_stride), t1(tmp_stride));
            else
                std::swap(t1, t2);
            plan(mov, esize, d(dst_stride), t2(tmp_stride));
        }
        return;
    }

    // f -> df
    // - f/df mixed operands must be qword aligned
    // - f -> f striding: use s32
    if (src_f && dst_df) {
        int step = get_step();
        const auto tmp_type = src_type;
        const int tmp_stride = 2;
        const int tmp_stride_bytes = tmp_stride * src_type_size;
        const int reg_size = dst.byte_offset() + width * tmp_stride_bytes;
        const int nregs = utils::div_up(reg_size, grf_size);
        auto tmp = lex_scope.alloc_reg_buf_data(nregs);
        for (int i = 0; i < width; i += step) {
            step = std::min(step, width - i);
            step = utils::rnd_down_pow2(step);
            int esize = step;

            auto s = src.subregister(i, esize, src_stride_bytes);
            auto d = dst.subregister(i, esize, dst_stride_bytes);
            auto t = tmp.subregister(d.getByteOffset(), tmp_type);
            plan(mov, esize, t.d()(tmp_stride), s.d()(src_stride));
            plan(mov, esize, d(dst_stride), t(tmp_stride));
        }
        return;
    }

    // df -> f
    // - f/df mixed operands must be qword aligned
    // - f -> f packing: use s32
    if (dst_f && src_df) {
        int step = get_step();
        const auto tmp_type = dst_type;
        const int tmp_stride = 2;
        const int tmp_stride_bytes = tmp_stride * src_type_size;
        const int reg_size = dst.byte_offset() + width * tmp_stride_bytes;
        const int nregs = utils::div_up(reg_size, grf_size);
        auto tmp = lex_scope.alloc_reg_buf_data(nregs);
        for (int i = 0; i < width; i += step) {
            step = std::min(step, width - i);
            step = utils::rnd_down_pow2(step);
            int esize = step;

            auto s = src.subregister(i, esize, src_stride_bytes);
            auto d = dst.subregister(i, esize, dst_stride_bytes);
            auto t = tmp.subregister(s.getByteOffset(), tmp_type);
            plan(mov, esize, t(tmp_stride), s(src_stride));
            plan(mov, esize, d.d()(dst_stride), t.d()(tmp_stride));
        }
        return;
    }

    // f -> hf
    if (src_f && dst_hf) {
        int step = get_step();
        const auto tmp_type = dst_type;
        const int reg_size
                = src.byte_offset() + step * 2 * src_stride * dst_type_size;
        const int nregs = utils::div_up(reg_size, grf_size);
        auto tmp1 = lex_scope.alloc_reg_buf_data(nregs);
        auto tmp2 = lex_scope.alloc_reg_buf_data(nregs);
        for (int i = 0; i < width; i += step) {
            int tmp_stride = 2;
            step = std::min(step, width - i);
            step = utils::rnd_down_pow2(step);
            int esize = step;

            auto s = src.subregister(i, esize, src_stride_bytes);
            auto d = dst.subregister(i, esize, dst_stride_bytes);

            if (esize == 1
                    || (d.getByteOffset() == s.getByteOffset()
                            && 2 * src_stride == dst_stride)) {
                plan(mov, esize, d(dst_stride), s(src_stride));
                continue;
            }

            const auto align_boundary = grf_size / 2;
            auto tmp_offset = 2 * dst_type_size
                    * (d.getOffset() % (align_boundary / 2));
            auto t1 = tmp1.subregister(
                    s.getByteOffset() % (nregs * grf_size), tmp_type);
            plan(mov, esize, t1(2 * src_stride), s(src_stride));
            if (dst_stride == 1
                    && (s.getByteOffset() != tmp_offset || src_stride != 1)) {
                // Packed word dst needs specially aligned and strided src
                auto t2 = tmp2.subregister(tmp_offset, tmp_type);
                plan(mov, esize, t2.w()(tmp_stride), t1.w()(2 * src_stride));
                t1 = t2;
            } else
                tmp_stride = 2 * src_stride;
            plan(mov, esize, d.w()(dst_stride), t1.w()(tmp_stride));
        }
        return;
    }

    // hf -> f
    if (dst_f && src_hf) {
        int step = get_step();
        const auto tmp_type = src_type;
        const int reg_size = step * 2 * dst_stride * src_type_size;
        const int nregs = utils::div_up(reg_size, grf_size);
        auto tmp = lex_scope.alloc_reg_buf_data(nregs);
        for (int i = 0; i < width; i += step) {
            step = std::min(step, width - i);
            step = utils::rnd_down_pow2(step);
            int esize = step;

            auto s = src.subregister(i, esize, src_stride_bytes);
            auto d = dst.subregister(i, esize, dst_stride_bytes);

            int tmp_stride = 2 * dst_stride;
            if (esize > 1
                    && (s.getByteOffset() != d.getByteOffset()
                            || src_stride != 2 * dst_stride)) {
                auto t = tmp.subregister(d.getByteOffset(), tmp_type);
                plan(mov, esize, t.w()(tmp_stride), s.w()(src_stride));
                s = t;
            } else
                tmp_stride = src_stride;

            plan(mov, esize, d(dst_stride), s(tmp_stride));
        }
        return;
    }

    // f32/f16/s32 -> s8/u8 and s8/u8 -> f32/s32
    // - Use saturation
    // - s8/u8 must be DW-strided: use temporary
    bool d_or_f_to_b = (src_d || src_f) && dst_b;
    bool b_to_d_or_f = (dst_d || dst_f) && src_b;
    if (d_or_f_to_b || b_to_d_or_f) {
        if (dst_d || dst_f) ir_assert(dst_stride_bytes == 4);
        if (src_d || src_f) ir_assert(src_stride_bytes == 4);
        if (dst_b) ir_assert(utils::one_of(dst_stride_bytes, 1, 4, 8));
        int step = get_step();
        const int step_size = step * (int)sizeof(uint32_t);
        const int nregs = 1 + utils::div_up(step_size, grf_size);
        auto tmp1 = lex_scope.alloc_reg_buf_data(nregs);
        auto tmp2 = lex_scope.alloc_reg_buf_data(nregs);
        for (int i = 0; i < width; i += step) {
            step = std::min(step, width - i);
            step = utils::rnd_down_pow2(step);
            int esize = step;

            auto s = src.subregister(i, esize, src_stride_bytes);
            auto d = dst.subregister(i, esize, dst_stride_bytes);
            if (src_d || src_f) {
                // d -> b.
                if (src_f && hw == ngen::HW::Gen9)
                    host->rnde(esize, s(src_stride), s(src_stride));
                if (esize == 1) {
                    // relaxed F-pipe alignment requirements for f32 broadcast
                    auto t = tmp1.subregister(0, dst_type);
                    plan(mov, 2 | host->sat, t(4), s);
                    plan(mov, 1, d, t);
                } else if (dst_stride_bytes == 1) {
                    auto offset_bytes = src_f ? s.getByteOffset()
                                              : 4 * (d.getByteOffset() % 16);
                    auto t = tmp1.subregister(offset_bytes, dst_type)(4);
                    plan(mov, esize | host->sat, t, s(src_stride));
                    if (offset_bytes != 4 * (d.getByteOffset() % 16)) {
                        auto t2 = tmp2.subregister(
                                4 * (d.getByteOffset() % 16), dst_type)(4);
                        plan(mov, esize, t2, t);
                        t = t2;
                    }
                    plan(mov, esize, d(dst_stride), t);
                } else {
                    plan(mov, esize | host->sat, d(dst_stride), s(src_stride));
                }
            } else {
                if (esize == 1) {
                    // Direct x8 -> x32 scalar cast is not always
                    // supported. Use intermediate cast to s16.
                    auto t = tmp1.subregister(0, ngen::DataType::w)(1);
                    plan(mov, esize, t, s(src_stride));
                    plan(mov, esize, d(dst_stride), t);
                } else if (src_b) {
                    auto offset_bytes = dst_f ? d.getByteOffset() : 0;
                    auto t = tmp1.subregister(offset_bytes, src_type)(4);
                    plan(mov, esize, t, s(src_stride));
                    plan(mov, esize, d(dst_stride), t);
                } else {
                    plan(mov, esize, d(dst_stride), s(src_stride));
                }
            }
        }
        return;
    }

    // Handle mov(src.uw(x)(1), dst.uw(y)(2)).
    if (src_type_size == 2 && dst_type_size == 2 && src_stride == 2
            && dst_stride == 1 && width > 1) {
        int step = get_step();
        auto step_size = step * src_type_size * src_stride;
        auto tmp_regs = utils::div_up(step_size, grf_size);
        auto tmp = lex_scope.alloc_reg_buf_data(tmp_regs);
        auto tmp2_regs = utils::div_up(step * dst_type_size, grf_size);
        auto tmp2 = lex_scope.alloc_reg_buf_data(tmp2_regs);
        for (int i = 0; i < width; i += step) {
            step = std::min(step, width - i);
            step = utils::rnd_down_pow2(step);
            int esize = step;
            auto s = src.format(i * src_stride_bytes, ngen::DataType::invalid,
                    esize, src_stride);
            auto d = dst.format(i * dst_stride_bytes, ngen::DataType::invalid,
                    esize, dst_stride);
            auto d_old = d;
            bool d_half_grf_aligned
                    = utils::one_of(d.byte_offset(), 0, grf_size / 2);
            if (!d_half_grf_aligned) { d = tmp2.format(0, dst_type, esize); }
            if (s.offset() != 0) {
                auto t = tmp.format(0, src_type, esize, src_stride);
                plan(mov, esize, t, s);
                s = t;
            }
            plan(mov, esize, d, s);
            if (!d_half_grf_aligned) plan(mov, esize, d_old, d);
        }
        return;
    }

    // Perform FP to FP move.
    // Float pipe has some register regioning limitations. If mov is not
    // allowed then fix regioning by switching to integer pipe which has
    // less limitations.
    if (src_xf || dst_xf) {
        // forcing floats if on Gen9, for RNDE only works with floats there
        auto real_type_size = (src_xf && !src_f && hw == ngen::HW::Gen9)
                ? ngen::getBytes(ngen::DataType::f)
                : dst_type_size;

        bool local_src_f = src_f;
        bool local_src_hf = src_hf;
        bool local_src_bf = src_bf;
        auto local_src_type = src_type;

        int step = get_step();
        auto tmp_regs = utils::div_up(step * real_type_size, grf_size);
        auto tmp = lex_scope.alloc_reg_buf_data(tmp_regs);
        for (int i = 0; i < width; i += step) {
            step = std::min(step, width - i);
            step = utils::rnd_down_pow2(step);
            int esize = step;
            ir_assert(math::is_pow2(esize));
            auto s = src.format(i * src_stride_bytes, ngen::DataType::invalid,
                    esize, src_stride);
            auto d = dst.format(i * dst_stride_bytes, ngen::DataType::invalid,
                    esize, dst_stride);
            auto d_old = d;

            if (hw == ngen::HW::Gen9) {
                auto t = tmp.format(0, ngen::DataType::f, esize);
                if (src_f)
                    host->rnde(esize, t, s);
                else if (src_xf) {
                    plan(mov, esize, t, s);
                    host->rnde(esize, t, t);
                }
                if (src_xf) {
                    s = t;
                    local_src_f = true;
                    local_src_hf = local_src_bf = false;
                    local_src_type = ngen::DataType::f;
                }
            }
            bool do_d0_align = false;
            if (esize > 1 && dst_bf) {
                ir_assert(hw != ngen::HW::Gen9); // not gonna happen on Gen9
                bool d_0_aligned = (d.byte_offset() == 0);
                bool d_half_grf_aligned = (d.byte_offset() == grf_size / 2);
                if (!d_0_aligned && (!d_half_grf_aligned || dst_stride != 1)) {
                    do_d0_align = true;
                }
            }
            if (do_d0_align) { d = tmp.format(0, dst_type, esize); }

            bool do_align = false;
            if (esize > 1 && s.hs() != 0) {
                if ((local_src_f && dst_hf) || (dst_f && local_src_hf)) {
                    if (s.byte_offset() != d.byte_offset()) do_align = true;
                } else if (s.offset() != d.offset())
                    do_align = true;
            }
            if (do_align) {
                bool s_half_grf_aligned = (local_src_hf || local_src_bf)
                        && utils::one_of(s.byte_offset(), 0, grf_size / 2);
                bool d_half_grf_aligned = (dst_hf || dst_bf)
                        && utils::one_of(d.byte_offset(), 0, grf_size / 2);
                if (dst_f && d.offset() == 0 && s_half_grf_aligned)
                    do_align = false;
                if (local_src_f && s.offset() == 0 && d_half_grf_aligned)
                    do_align = false;
            }

            if (do_align) {
                auto i_type = to_ngen(
                        type_t::u(ngen::getBytes(local_src_type) * 8));
                s = s.reinterpret(i_type);
                align_src_dst_offset(host, scope, esize, d, s);
                s = s.reinterpret(local_src_type);
            }
            plan(mov, esize, d, s);

            if (do_d0_align) {
                auto i_type = to_ngen(type_t::u(ngen::getBytes(dst_type) * 8));
                auto d_int = d_old.reinterpret(i_type);
                auto s_int = d.reinterpret(i_type);
                plan(mov, esize, d_int, s_int);
            }
        }
        return;
    }

    if (src_b && dst_b) {
        const int tmp_stride = 4;
        const int tmp_stride_bytes = tmp_stride * dst_type_size;
        // Any byte conversion requires saturation:
        // - ub -> b loses 1 bit of precision
        // - b -> ub loses sign bit
        const bool needs_saturation = src_type != dst_type;

        int step = get_step();
        const int nregs = 1 + utils::div_up(step * tmp_stride_bytes, grf_size);
        auto tmp = lex_scope.alloc_reg_buf_data(nregs);
        for (int i = 0; i < width; i += step) {
            step = std::min(step, width - i);
            step = utils::rnd_down_pow2(step);
            int esize = step;
            auto s = src.format(i * src_stride_bytes, ngen::DataType::invalid,
                    esize, src_stride);
            auto d = dst.format(i * dst_stride_bytes, ngen::DataType::invalid,
                    esize, dst_stride);
            ngen::InstructionModifier mod = esize;
            if (needs_saturation) mod |= host->sat;

            bool aligned = true;
            const bool needs_raw_mov = dst_stride == 1 || esize == 1;
            if ((src_stride == 1 || esize == 1) && needs_raw_mov) {
                // Note: This case does not appear to be documented. Experiments
                // seem to indicate that packed byte-to-packed byte move must be
                // word-aligned on the destination.
                aligned = d.offset() % 2 == 0;
            } else if (dst_stride <= 2 && src_stride >= 2 * dst_stride) {
                const int rel_stride = src_stride / dst_stride;
                const int alignment_bdy = grf_size / rel_stride;
                const int dst_aligned_offset = d.offset() % alignment_bdy;
                const int src_aligned_offset = s.offset() / rel_stride;
                aligned = dst_aligned_offset == src_aligned_offset;
            }

            // Workaround for scalar byte conversion:
            // - Broadcast to two locations with stride and conversion
            // - Move one copy to the destination
            if (needs_saturation && esize == 1) mod.setExecSize(2);

            if (!aligned || (needs_saturation && needs_raw_mov)) {
                const int tmp_rel_stride = tmp_stride / dst_stride;
                const int tmp_alignment_bdy = grf_size / tmp_rel_stride;
                const int tmp_aligned_offset = d.offset() % tmp_alignment_bdy;
                const int tmp_offset = tmp_rel_stride * tmp_aligned_offset;
                const int allowed_bytes = 2 * grf_size - tmp_offset;

                if ((mod.getExecSize() - 1) * tmp_stride + 1 > allowed_bytes) {
                    // Workaround for cases where temporary is not grf aligned
                    // and esize == 16 on XeHPG and below
                    auto max_width = (allowed_bytes - 1) / tmp_stride + 1;
                    auto tmp_esize = utils::rnd_down_pow2(max_width);
                    mod.setExecSize(tmp_esize);
                    esize = tmp_esize;
                    step = tmp_esize;
                }

                auto t = tmp.format(
                        tmp_offset, dst_type, mod.getExecSize(), tmp_stride);
                plan(mov, mod, t, s);
                mod = esize;
                s = tmp.format(tmp_offset, dst_type, esize, tmp_stride);
            }
            plan(mov, mod, d, s);
        }
        return;
    }

    // Perform regular move.
    int step = get_step();
    for (int i = 0; i < width; i += step) {
        step = std::min(step, width - i);
        step = utils::rnd_down_pow2(step);
        int esize = step;
        ir_assert(math::is_pow2(esize));
        auto s = src.format(i * src_stride_bytes, ngen::DataType::invalid,
                esize, src_stride);
        auto d = dst.format(i * dst_stride_bytes, ngen::DataType::invalid,
                esize, dst_stride);
        plan(mov, esize, d, s);
    }
}

template <typename GeneratorT>
void align_src_dst_offset(GeneratorT *host, ngen_register_scope_t &scope,
        const ngen::InstructionModifier &mod, const reg_buf_data_t &dst,
        reg_buf_data_t &src) {
    int src_stride = src.hs();
    // src is broadcasted, no need to align, return.
    if (src_stride == 0) return;

    bool is_xf = ngen_is_xf(src.type()) || ngen_is_xf(dst.type());
    bool is_bf_to_f = (src.type() == ngen::DataType::bf)
            && (dst.type() == ngen::DataType::f);
    int src_type_size = ngen::getBytes(src.type());
    int src_off = src.offset();
    int dst_off = dst.offset();
    int src_byte_off = src.byte_offset();
    int dst_byte_off = dst.byte_offset();

    // If src is aligned with dst, return.
    if ((is_xf || is_bf_to_f) && src_off == dst_off) return;
    if (!is_xf && src_byte_off == dst_byte_off) return;

    int new_src_byte_off = (is_xf ? dst_off * src_type_size : dst_byte_off);

    int esize = mod.getExecSize();
    int grf_size = ngen::GRF::bytes(scope.hw());
    int src_size = std::max(src_type_size * esize * src_stride, src_type_size);

    auto new_src = scope.alloc_reg_buf_data(
            utils::div_up(src_size + new_src_byte_off, grf_size));
    new_src = new_src.format(new_src_byte_off, src.type(), esize, src_stride);
    emit_reorder_1d_tile(scope.hw(), host, scope, esize, src, src_stride,
            new_src, src_stride);
    src = new_src;
}

template <typename GeneratorT>
void align_src_dst_offset(GeneratorT *host, ngen_register_scope_t &scope,
        const ngen::InstructionModifier &mod, const reg_buf_data_t &dst,
        reg_buf_data_t &src0, reg_buf_data_t &src1) {
    align_src_dst_offset(host, scope, mod, dst, src0);
    align_src_dst_offset(host, scope, mod, dst, src1);
}

template <typename GeneratorT>
void align_src_dst_offset(GeneratorT *host, ngen_register_scope_t &scope,
        const ngen::InstructionModifier &mod, const ngen_operand_t &dst,
        ngen_operand_t &src) {
    if (!dst.is_reg_data()) return;
    if (!src.is_reg_data()) return;

    auto rd = src.reg_buf_data();
    align_src_dst_offset(host, scope, mod, dst.reg_buf_data(), rd);
    if (rd == src.reg_buf_data()) return;

    bool is_negated = src.is_negated();
    src = ngen_operand_t(rd, src.mod());
    if (is_negated) src = -src;
}

template <typename GeneratorT>
void align_src_dst_offset(GeneratorT *host, ngen_register_scope_t &scope,
        const ngen::InstructionModifier &mod, const ngen_operand_t &dst,
        ngen_operand_t &src0, ngen_operand_t &src1) {
    align_src_dst_offset(host, scope, mod, dst, src0);
    align_src_dst_offset(host, scope, mod, dst, src1);
}

// Implementation of GRF reorder between 2D dense layouts.
// Requirements for A -> B reorder:
// - A and B must have the same data type
// - Layouts must be 2D and dense
// Reorder may require several steps, in this case a temporary buffer T is
// allocated. For example: A -> T -> B or A -> B -> T -> B
class reorder_2d_impl_t {
public:
    reorder_2d_impl_t(ngen::HW hw, tensor_t tile, const layout_t &src_layout,
            const layout_t &dst_layout)
        : hw_(hw), src_(src_layout), dst_(dst_layout), tile_(std::move(tile)) {
        ir_assert(src_.type() == dst_.type());
    }

    const tensor_t &tile() const { return tile_; }

    template <typename GeneratorT>
    void emit(GeneratorT *host, ngen_register_scope_t &scope,
            const reg_buf_data_t &src_rd, const reg_buf_data_t &dst_rd) {
        int a_idx, b_idx;
        int tile_a, tile_b;
        tile_to_2d_dims(tile_, a_idx, b_idx, tile_a, tile_b);

        // Convert src/dst to 2D layouts.
        dim_assignment_t to_ab(src_.ndims(), 2);
        to_ab.assign(a_idx, 0);
        to_ab.assign(b_idx, 1);
        auto src_ab = to_ab.map(src_);
        auto dst_ab = to_ab.map(dst_);

        // Find minimal cost reorder path between layouts.
        auto path = find_min_cost_path(hw_, src_ab, dst_ab, tile_a, tile_b);

        // Allocate a temporary GRF buffer if needed.
        reg_buf_data_t tmp;
        if (path.size() > 1) {
            const int grf_size = ngen::GRF::bytes(hw_);
            tmp = scope.alloc_reg_buf_data(
                    utils::div_up(dst_ab.size(), grf_size));
        }

        // Iterate through found reorders.
        auto *prev_layout = &src_ab;
        auto prev_rd = src_rd;
        int path_len = int(path.size());
        auto &orig_type = src_ab.type();
        for (int i = 0; i < path_len; i++) {
            auto &step = path[i];
            auto &tile = step.tile;
            auto &type = step.type;
            auto *next_layout = &step.layout;

            // x -> y reorder.
            auto x = prev_layout->map(tile).reinterpret(type);
            auto y = next_layout->map(tile).reinterpret(type);

            bool use_dst = ((path_len - i) % 2 == 1);
            auto next_rd = (use_dst ? dst_rd : tmp);
            auto &x_blocks = x.blocks();
            auto &y_blocks = y.blocks();
            ir_assert(x_blocks.size() <= 1);
            ir_assert(y_blocks.size() <= 1);
            int x_stride = (x_blocks.empty() ? 1 : int(x_blocks[0].stride));
            int y_stride = (y_blocks.empty() ? 1 : int(y_blocks[0].stride));
            int width = int(tile.elems()) * orig_type.size() / type.size();
            next_layout->for_each_tile(
                    tile, [&](const std::vector<dim_t> &start) {
                        int prev_off = int(prev_layout->offset_in_bytes(start));
                        int next_off = int(next_layout->offset_in_bytes(start));
                        auto x_sub = prev_rd.format(prev_off, to_ngen(type), 1);
                        auto y_sub = next_rd.format(next_off, to_ngen(type), 1);
                        emit_reorder_1d_tile(hw_, host, scope, width, x_sub,
                                x_stride, y_sub, y_stride);
                    });
            prev_layout = next_layout;
            prev_rd = next_rd;
        }
    }

    static const int max_tile_blocks = 4;

private:
    // Represents 2D reorder corresponding to (a x b) tile.
    struct edge_t {
        edge_t() = default;
        edge_t(int idx, int a, int b) : idx(idx), a(a), b(b) {}

        tensor_t tile() const { return tensor_t({a, b}); }

        std::string str() const {
            std::ostringstream oss;
            oss << "edge(idx = " << idx << ", a = " << a << ", b = " << b
                << ")";
            return oss.str();
        }

        int idx; // Identifier of the edge.
        int a = 0, b = 0; // Specify tile (a x b).
    };

    // Represents GRF layout between edges-reorders.
    struct vertex_t {
        vertex_t(ngen::HW hw, int idx, const layout_t &layout)
            : hw(hw), idx(idx), layout(layout) {}

        std::string str() const {
            std::ostringstream oss;
            oss << "vertex(idx = " << idx << ", layout = " << layout << ")";
            return oss.str();
        }

        void set_edges(const std::vector<edge_t> &edges) {
            adj_edge_type_masks.resize(edges.size());
            int type_size = layout.type().size();
            for (int i = 0; i < int(edges.size()); i++) {
                auto &e = edges[i];
                auto tile = e.tile();
                int max_type_size;
                bool ok = layout_t::try_reinterpret_to_wider_type(
                        layout, layout, tile, false, &max_type_size);
                if (!ok) max_type_size = type_size;
                int from = math::ilog2q(type_size);
                int to = math::ilog2q(max_type_size);
                for (int j = from; j <= to; j++) {
                    type_t type = type_t::u(8 << j);
                    if (can_reorder(tile, type))
                        adj_edge_type_masks[i] |= (1 << j);
                }
            }
        }

        void add_neighbor(const vertex_t *v) { adj_vertices.push_back(v); }

        bool is_neighbor(const vertex_t &v) const {
            for (auto *n : adj_vertices)
                if (n == &v) return true;
            return false;
        }

        // Check the following limitations:
        // - Assume at most one block (maybe with non-dense stride)
        // - Horizontal stride must be <= 4 for GRF region
        // - GRF region can't span more than 2 registers
        bool can_reorder(const tensor_t &tile, const type_t &type) const {
            auto ab_layout = layout.map(tile).reinterpret(type);
            int nblocks = int(ab_layout.blocks().size());
            if (nblocks == 0) return true;
            if (nblocks > 1) return false;
            auto &last = ab_layout.blocks().back();
            int max_stride = int(last.stride * last.block);
            if (last.stride > 4) return false;
            if ((int)last.stride == 4 && type.size() <= 2) return false;
            if (!math::is_pow2(last.stride)) return false;
            int max_stride_bytes = max_stride * type.size();
            int grf_size = ngen::GRF::bytes(hw);
            if (max_stride_bytes > 2 * grf_size) return false;
            return true;
        }

        // Finds the minimal cost of reordering from this vertex to vertex v.
        int cost(const vertex_t &v, const std::vector<edge_t> &edges,
                edge_t &min_edge, type_t &min_type) const {
            int min_cost = std::numeric_limits<int>::max();
            for (int i = 0; i < int(edges.size()); i++) {
                type_t i_min_type;
                int new_cost = cost(edges[i], v, i_min_type);
                if (new_cost < min_cost) {
                    min_cost = new_cost;
                    min_edge = edges[i];
                    min_type = i_min_type;
                }
            }
            return min_cost;
        }

        // Finds the minimal cost of reordering from this vertex to vertex `v`
        // through edge `e`. If the reorder is possible, `type` contains the
        // reorder type with the minimal cost.
        int cost(const edge_t &e, const vertex_t &v, type_t &type) const {
            uint32_t mask = (adj_edge_type_masks[e.idx]
                    & v.adj_edge_type_masks[e.idx]);
            if (mask == 0) return std::numeric_limits<int>::max();
            int cur_size = layout.type().size();
            int cur_cost = layout.elems() / (e.a * e.b);
            int min_log_bytes = math::ilog2q(cur_size);
            int max_log_bytes = 3;
            int min_cost = std::numeric_limits<int>::max();
            for (int i = min_log_bytes; i <= max_log_bytes; i++) {
                if ((mask & (1 << i)) == 0) continue;
                if (i > min_log_bytes) {
                    ir_assert(!layout.blocks().empty());
                    ir_assert(!v.layout.blocks().empty());
                    int dim_idx0 = layout.blocks()[0].dim_idx;
                    int dim_idx1 = v.layout.blocks()[0].dim_idx;
                    if (dim_idx0 != dim_idx1) continue;
                }
                min_cost = cur_cost;
                type = type_t::u(8 << i);
                break;
            }
            return min_cost;
        }

        ngen::HW hw;
        int idx; // Identifier of the vertex.
        layout_t layout; // Layout of the vertex.
        // Specifies a bitmask for every edge: if adj_edge_type_masks[E_idx]
        // has b-th bit set then this vertex can be reordered through E edge
        // using the data type with size 2^b bytes.
        std::vector<uint32_t> adj_edge_type_masks;
        std::vector<const vertex_t *> adj_vertices; // Adjacent vertices.
    };

    // Represents a reorder step.
    struct reorder_step_t {
        reorder_step_t() = default;
        reorder_step_t(const layout_t &layout, const tensor_t &tile,
                const type_t &type)
            : layout(layout), tile(tile), type(type) {}

        layout_t layout; // Destination layout.
        tensor_t tile; // Tile corresponding to one instruction.
        type_t type; // Registers should be reinterpreted to `type` for reorder.
    };

    // Extracts dimension sizes and their indices from a multidimensional
    // tensor.
    static void tile_to_2d_dims(
            const tensor_t &tile, int &a_idx, int &b_idx, int &a, int &b) {
        a_idx = -1;
        b_idx = -1;
        for (int i = 0; i < tile.ndims(); i++) {
            if (tile.dims()[i] == 1) continue;
            if (a_idx == -1) {
                a_idx = i;
                continue;
            }
            if (b_idx == -1) {
                b_idx = i;
                continue;
            }
            ir_error_not_expected();
        }

        for (int i = 0; i < tile.ndims(); i++) {
            if (utils::one_of(i, a_idx, b_idx)) continue;
            if (a_idx == -1) {
                a_idx = i;
                continue;
            }
            if (b_idx == -1) {
                b_idx = i;
                continue;
            }
        }

        if (a_idx > b_idx) std::swap(a_idx, b_idx);

        a = tile.dims()[a_idx];
        b = tile.dims()[b_idx];
    }

    // Finds the optimal sequence of reorders between src and dst layouts.
    static std::vector<reorder_step_t> find_min_cost_path(ngen::HW hw,
            const layout_t &src, const layout_t &dst, int tile_a, int tile_b) {
        // Create all possible edges - 2D reorders.
        std::vector<edge_t> edges;
        for (int a = 1; a <= tile_a; a *= 2) {
            for (int b = 1; b <= tile_b; b *= 2) {
                if (src.dim(0) % a != 0) continue;
                if (src.dim(1) % b != 0) continue;
                int idx = int(edges.size());
                edges.emplace_back(idx, a, b);
            }
        }

        int nedges = int(edges.size());

        // Create all possible layouts for tile_a x tile_b tensor.
        std::vector<vertex_t> vertices;
        std::vector<std::vector<std::pair<int, uint32_t>>> edge_vertices(
                nedges);
        auto all_layouts = generate_all_layouts(src.type(), tile_a, tile_b);
        for (auto &l : all_layouts) {
            // Skip if too many blocks.
            if (int(l.blocks().size()) > max_tile_blocks) continue;
            int v_idx = int(vertices.size());
            vertices.emplace_back(hw, v_idx, l);
            auto &v = vertices.back();
            // Pass all known reorders, the vertex/layout will filter out
            // incompatible reorders.
            v.set_edges(edges);
            // Store all vertices adjacent to a specific edge.
            for (int i = 0; i < nedges; i++) {
                uint32_t mask = v.adj_edge_type_masks[i];
                if (mask != 0) edge_vertices[i].emplace_back(v_idx, mask);
            }
        }

        // Find neighbors between all vertices.
        int nvertices = int(vertices.size());
        for (int i = 0; i < nvertices; i++) {
            auto &v = vertices[i];
            for (int j = 0; j < nedges; j++) {
                uint32_t mask = v.adj_edge_type_masks[j];
                if (mask != 0) {
                    for (auto &idx_mask : edge_vertices[j]) {
                        int v_idx = idx_mask.first;
                        if (v_idx == i) continue;
                        uint32_t common_mask = (mask
                                & vertices[v_idx].adj_edge_type_masks[j]);
                        if (common_mask != 0) v.add_neighbor(&vertices[v_idx]);
                    }
                }
            }
        }

        // Identify source and destination vertices.
        int src_idx = -1;
        int dst_idx = -1;
        for (int i = 0; i < nvertices; i++) {
            auto &v = vertices[i];
            if (src_idx == -1
                    && v.layout.is_strictly_equal(
                            src, /*compare_offset=*/false))
                src_idx = i;
            if (dst_idx == -1
                    && v.layout.is_strictly_equal(
                            dst, /*compare_offset=*/false))
                dst_idx = i;
        }

        ir_assert(src_idx != -1);
        ir_assert(dst_idx != -1);

        // Layouts are the same, just copy.
        if (src_idx == dst_idx) {
            auto &v = vertices[src_idx];
            edge_t min_edge;
            type_t min_type;
            v.cost(v, edges, min_edge, min_type);
            reorder_step_t step(v.layout, min_edge.tile(), min_type);
            return {step};
        }

        // Dijkstra's algorithm, find the minimal cost path between src and
        // dst. Use the number of instructions to estimate the cost.
        int inf_cost = std::numeric_limits<int>::max();
        std::vector<int> cost(nvertices, inf_cost);
        std::vector<int> prev(nvertices);
        std::vector<reorder_step_t> reorder_steps(nvertices);
        std::vector<bool> seen(nvertices, false);
        cost[src_idx] = 0;
        for (int i = 0; i < nvertices; i++) {
            int min_idx = -1;
            int min_cost = inf_cost;
            for (int j = 0; j < nvertices; j++) {
                if (seen[j]) continue;
                if (cost[j] < min_cost) {
                    min_idx = j;
                    min_cost = cost[j];
                }
            }
            seen[min_idx] = true;
            auto &v_min = vertices[min_idx];
            for (auto *v : v_min.adj_vertices) {
                edge_t min_edge;
                type_t min_type;
                int new_cost = cost[min_idx]
                        + v_min.cost(*v, edges, min_edge, min_type);
                if (new_cost < cost[v->idx]) {
                    cost[v->idx] = new_cost;
                    prev[v->idx] = min_idx;
                    reorder_steps[v->idx] = reorder_step_t(
                            v->layout, min_edge.tile(), min_type);
                }
            }
        }

        // Sanity check, ensure the reorder sequence is not too long.
        int max_cost = 256;
        ir_assert(cost[dst_idx] <= max_cost);
        MAYBE_UNUSED(max_cost);

        // Restore the shortest reorder path.
        std::vector<reorder_step_t> ret;
        int idx = dst_idx;
        while (idx != src_idx) {
            ret.push_back(reorder_steps[idx]);
            idx = prev[idx];
        }
        std::reverse(ret.begin(), ret.end());
        return ret;
    }

    // Returns all possible layouts for (a x b) tensor.
    static std::vector<layout_t> generate_all_layouts(
            const type_t &type, int a, int b) {
        std::vector<layout_t> ret;
        std::vector<block_t> blocks;
        generate_all_layouts_impl(ret, blocks, type, a, b, 1);
        return ret;
    }

    static void generate_all_layouts_impl(std::vector<layout_t> &layouts,
            std::vector<block_t> &blocks, const type_t &type, int a, int b,
            int stride) {
        if (a == 1 && b == 1) {
            layouts.emplace_back(type, 2, 0, blocks);
            return;
        }
        bool iterate_a = true;
        bool iterate_b = true;

        // Avoid repeating indices to keep only unique layouts.
        if (!blocks.empty()) {
            auto &last = blocks.back();
            iterate_a &= (last.dim_idx != 0);
            iterate_b &= (last.dim_idx != 1);
        }

        if (iterate_a) {
            for (int a_blk = 2; a_blk <= a; a_blk++) {
                if (a % a_blk != 0) continue;
                blocks.emplace_back(0, a_blk, stride);
                generate_all_layouts_impl(
                        layouts, blocks, type, a / a_blk, b, stride * a_blk);
                blocks.pop_back();
            }
        }
        if (iterate_b) {
            for (int b_blk = 2; b_blk <= b; b_blk++) {
                if (b % b_blk != 0) continue;
                blocks.emplace_back(1, b_blk, stride);
                generate_all_layouts_impl(
                        layouts, blocks, type, a, b / b_blk, stride * b_blk);
                blocks.pop_back();
            }
        }
    }

    ngen::HW hw_;

    layout_t src_;
    layout_t dst_;

    tensor_t tile_;
};

class reorder_impl_t {
public:
    reorder_impl_t(ngen::HW hw, const reorder_t &reorder)
        : hw_(hw)
        , src_layout_(reorder.src_layout)
        , dst_layout_(reorder.dst_layout) {
        layout_t::try_reinterpret_to_wider_type(src_layout_, dst_layout_);

        // Pure bf moves are not supported.
        if (utils::everyone_is(
                    type_t::bf16(), src_layout_.type(), dst_layout_.type())) {
            src_layout_ = src_layout_.retype(type_t::u16());
            dst_layout_ = dst_layout_.retype(type_t::u16());
        }
    }

    template <typename GeneratorT>
    void emit(GeneratorT *host, ngen_register_scope_t &scope,
            const reg_buf_data_t &src, const reg_buf_data_t &dst) {
        if (try_emit_2d(host, scope, src, dst)) return;
        emit_1d(host, scope, src, dst);
    }

private:
    template <typename GeneratorT>
    void emit_1d(GeneratorT *host, ngen_register_scope_t &scope,
            const reg_buf_data_t &src_rd, const reg_buf_data_t &dst_rd) {
        int src_stride;
        int dst_stride;
        auto tile = find_max_tile_with_fixed_stride(
                src_layout_, dst_layout_, src_stride, dst_stride);

        int tile_elems = int(tile.elems());
        auto &src_type = src_layout_.type();
        auto &dst_type = dst_layout_.type();
        dst_layout_.for_each_tile(tile, [&](const std::vector<dim_t> &start) {
            int src_off = int(src_layout_(start) * src_type.size());
            int dst_off = int(dst_layout_(start) * dst_type.size());
            auto sub_src = src_rd.format(src_off, to_ngen(src_type), 1);
            auto sub_dst = dst_rd.format(dst_off, to_ngen(dst_type), 1);

            ngen_register_scope_t tile_scope(scope.register_allocator());
            emit_reorder_1d_tile(hw_, host, tile_scope, tile_elems, sub_src,
                    src_stride, sub_dst, dst_stride);
        });
    }

    static std::vector<tensor_t> find_2d_dense_tiles(
            const layout_t &a, const layout_t &b) {
        using tile_pair_t = std::array<tensor_t, 2>;
        static constexpr int max_tile_blocks
                = reorder_2d_impl_t::max_tile_blocks;
        auto dense_2d_blocks = []() {
            dim_t stride = 1;
            int non_one_dims = 0;
            int count = 0;
            std::unordered_set<int> seen;
            return [=](const block_t &b) mutable {
                if ((dim_t)b.stride != stride) return false;
                if (b.block != 1) {
                    count++;
                    stride *= b.block;
                    auto ret = seen.insert(b.dim_idx);
                    if (ret.second) non_one_dims++;
                }
                return non_one_dims <= 2 && count <= max_tile_blocks;
            };
        };

        auto take_smaller = [](const tensor_t &a, const tensor_t &b) {
            return a.elems() <= b.elems();
        };

        auto equal_tiles
                = [](const tile_pair_t &p) { return p[0].is_equal(p[1]); };

        auto to_single_tile = [](const tile_pair_t &p) { return p[0]; };

        auto all_dims_pow2 = [](const tensor_t &tile) {
            for (auto d : tile.dims())
                if (!math::is_pow2(d)) return false;
            return true;
        };

        auto a_tiles = inner_tiles(
                a.blocks() | filter(dense_2d_blocks()), a.ndims());
        auto b_tiles = inner_tiles(
                b.blocks() | filter(dense_2d_blocks()), b.ndims());
        auto tiles = merge(a_tiles, b_tiles, take_smaller) | filter(equal_tiles)
                | transform(to_single_tile) | filter(all_dims_pow2);
        std::vector<tensor_t> ret;
        for (const auto &tile : tiles)
            ret.insert(ret.begin(), tile);
        return ret;
    }

    template <typename GeneratorT>
    bool try_emit_2d(GeneratorT *host, ngen_register_scope_t &scope,
            const reg_buf_data_t &src_rd, const reg_buf_data_t &dst_rd) {
        const int grf_size = ngen::GRF::bytes(hw_);

        if (src_layout_.type() != dst_layout_.type()) return false;
        // long / f64 swizzle emits scalar instructions
        if (src_layout_.type().scalar().size() >= 8) return false;
        if (!src_layout_.is_dense()) return false;
        if (!dst_layout_.is_dense()) return false;

        const auto type = to_ngen(src_layout_.type());
        for (const auto &tile : find_2d_dense_tiles(src_layout_, dst_layout_)) {
            if (tile.ndims() < 2) continue;
            if (tile.elems() < 4) break;
            auto src_tile_layout = src_layout_.map(tile);
            auto dst_tile_layout = dst_layout_.map(tile);
            if (!dst_tile_layout.is_dense()) continue;

            // Set layout offset to 0 since the offset is handled by fixing up
            // the register input to try_emit_2d_impl
            src_tile_layout.set_offset(0);
            dst_tile_layout.set_offset(0);

            // Try to allocate/release a temporary buffer to avoid
            // out_of_registers exception.
            auto dummy = scope.try_alloc_range(
                    utils::div_up(dst_tile_layout.size(), grf_size));
            if (dummy.isInvalid()) continue;

            // Allocation succeeded, can proceed further.
            scope.safeRelease(dummy);

            reorder_2d_impl_t r(hw_, tile, src_tile_layout, dst_tile_layout);
            src_layout_.for_each_tile(
                    tile, [&](const std::vector<dim_t> &start) {
                        auto src_off
                                = src_layout_.offset_in_bytes<dim_t>(start);
                        auto dst_off
                                = dst_layout_.offset_in_bytes<dim_t>(start);
                        auto src_tile_rd = src_rd.format(int(src_off), type);
                        auto dst_tile_rd = dst_rd.format(int(dst_off), type);

                        ngen_register_scope_t tile_scope(
                                scope.register_allocator());
                        r.emit(host, tile_scope, src_tile_rd, dst_tile_rd);
                    });
            return true;
        }
        return false;
    }

    static tensor_t find_max_tile_with_fixed_stride(const layout_t &src,
            const layout_t &dst, int &src_stride, int &dst_stride) {
        // 1. Split layouts to have aligned blocks.
        auto a = src;
        auto b = dst;
        layout_t::align_layouts(a, b);

        // 2. Find the max innermost tile.
        auto a_blocks = a.blocks();
        auto b_blocks = b.blocks();

        std::vector<dim_t> tile_dims(a.ndims(), 1);
        src_stride = (a_blocks.empty() ? 1 : int(a_blocks[0].stride));
        dst_stride = (b_blocks.empty() ? 1 : int(b_blocks[0].stride));
        int src_cur_stride = src_stride;
        int dst_cur_stride = dst_stride;

        int min_blocks = int(std::min(a_blocks.size(), b_blocks.size()));
        for (int i = 0; i < min_blocks; i++) {
            auto &ab = a_blocks[i];
            auto &bb = b_blocks[i];
            if (ab.dim_idx != bb.dim_idx || ab.block != bb.block) break;

            // Strides are supported for the innermost block only.
            if (src_cur_stride != int(ab.stride)) break;
            if (dst_cur_stride != int(bb.stride)) break;

            src_cur_stride = int(ab.block * ab.stride);
            dst_cur_stride = int(bb.block * bb.stride);
            tile_dims[ab.dim_idx] *= ab.block;
        }
        return tensor_t(tile_dims);
    }

    ngen::HW hw_;
    layout_t src_layout_;
    layout_t dst_layout_;
};

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
