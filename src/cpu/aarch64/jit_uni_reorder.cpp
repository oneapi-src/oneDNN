/*******************************************************************************
* Copyright 2018-2023 Intel Corporation
* Copyright 2020-2023 FUJITSU LIMITED
* Copyright 2022-2023 Arm Ltd. and affiliates
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

#include <algorithm>
#include <assert.h>
#include <numeric>

#include "oneapi/dnnl/dnnl_debug.h"

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/nstl.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_primitive.hpp"
#include "cpu/reorder/cpu_reorder_pd.hpp"

#include "cpu/aarch64/jit_uni_reorder.hpp"

#include "cpu/aarch64/jit_generator.hpp"

// #define DNNL_DEV_MODE
#if defined(DNNL_DEV_MODE)
#define DEBUg(...) \
    do { \
        if (get_verbose(verbose_t::debuginfo) > 1) { __VA_ARGS__ } \
    } while (0)
#else
#define DEBUg(...)
#endif
#define DEBUG(...) DEBUg(__VA_ARGS__)

using namespace Xbyak_aarch64;
using namespace dnnl::impl::types;

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

namespace tr {

static bool prb_has_small_strides(const prb_t &prb) {
    constexpr ptrdiff_t max_stride = (1LL << 31) - 1;
    for (int d = 0; d < prb.ndims; ++d) {
        const ptrdiff_t cms = max_stride / prb.nodes[d].n;
        const bool small_strides = true
                && prb.nodes[d].is < cms / (int)data_type_size(prb.itype)
                && prb.nodes[d].os < cms / (int)data_type_size(prb.otype);
        if (!small_strides) return false;
    }
    return true;
}

/** Minimal reasonable/desirable kernel size.
 * The constant might be used to determine how a problem should be split
 * between kernel and threading driver. */
const size_t ker_prb_size_min = 64;

/* kernel */
struct jit_uni_reorder_kernel_f32_t : public kernel_t, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_reorder_kernel_f32)

    void operator()(const call_param_t *c) const override {
        jit_generator::operator()(c);
    }
    void operator()(const tail_call_param_t *c) const override {
        jit_generator::operator()(c);
    }

    status_t create_kernel() override { return jit_generator::create_kernel(); }

    enum class scale_arg_t { NONE, SRC, DST };

    enum {
        len_unroll_max = 256,
        ndims_jit_loop_max = 3,
    };

    struct simple_impl_desc_t {
        int ndims_full_unroll = 0;
        int len_last_dim_unroll = 0;
        int tail_len_unroll = 0;
        int len_unroll = 0;
    };

#define PARAM(x) \
    abi_param1, \
            prb_.is_tail_present ? offsetof(tail_call_param_t, base_params) \
                    + offsetof(call_param_t, x) \
                                 : offsetof(call_param_t, x)
#define TAIL_PARAM(x) abi_param1, offsetof(tail_call_param_t, x)

    static bool simple_impl_desc_init(
            const prb_t &prb, simple_impl_desc_t *desc) {
        const int ndims = prb.ndims;

        int ndims_full_unroll = 0;
        int len_last_dim_unroll = 1;
        int tail_len_unroll = 0;
        int len_unroll = 1;

        // It is responsible for finding as many values
        // as kernel can unroll. If tail is present then
        // kernel will unroll only last node (possible improvement).
        // If there is no tail kernel can unroll a few nodes without any loops etc.
        // ndims_full_unroll - how many nodes will be unrolled
        // len_last_dim_unroll - what piece of last unrolled node will be unrolled
        if (prb.is_tail_present) {
            ndims_full_unroll = 1;
            len_unroll = prb.nodes[0].n;
            tail_len_unroll = prb.nodes[0].is_zero_pad_needed
                    ? 0
                    : static_cast<int>(prb.nodes[0].tail_size);
        } else {
            for (int d = 0; d < ndims; ++d) {
                const auto &node = prb.nodes[d];
                if (len_unroll * node.n <= len_unroll_max) {
                    ndims_full_unroll++;
                    len_unroll *= node.n;
                } else {
                    len_last_dim_unroll = len_unroll_max / len_unroll;
                    while (node.n % len_last_dim_unroll)
                        --len_last_dim_unroll;
                    len_unroll *= len_last_dim_unroll;
                    break;
                }
            }
        }

        if (prb.ndims - ndims_full_unroll > ndims_jit_loop_max) return false;

        if (desc) {
            desc->ndims_full_unroll = ndims_full_unroll;
            desc->len_last_dim_unroll = len_last_dim_unroll;
            desc->tail_len_unroll = tail_len_unroll;
            desc->len_unroll = len_unroll;
        }

        return true;
    }

    static bool applicable(const prb_t &p) {
        using namespace data_type;

        bool ok = true && p.ndims > 0
                && utils::one_of(p.itype, f32, s32, data_type::s8, u8)
                && utils::one_of(p.otype, f32, bf16, s32, data_type::s8, u8)
                && utils::everyone_is(0, p.ioff, p.ooff) /* do we need this? */
                && utils::one_of(p.beta, 0.f, 1.f) /* anything else? */
                && simple_impl_desc_init(p, nullptr) && prb_has_small_strides(p)
                && ((p.otype != bf16) || (p.itype == f32 && mayiuse_bf16()));

        return ok;
    }

    XReg o_addr(int o_off, bool with_type_multiplier = true) {
        if (o_off) {
            add_imm(X_DEFAULT_ADDR, x_ptr_out_off,
                    o_off * (with_type_multiplier ? otype_sz_ : 1), X_TMP);
            return X_DEFAULT_ADDR;
        }

        return x_ptr_out_off;
    }

    XReg src_s_addr(int s_off) {
        if (s_off) {
            add_imm(X_DEFAULT_ADDR, x_ptr_src_scale_off, s_off * stype_sz_,
                    X_TMP);
            return X_DEFAULT_ADDR;
        } else {
            return x_ptr_src_scale_off;
        }
    }

    XReg dst_s_addr(int s_off) {
        if (s_off) {
            add_imm(X_DEFAULT_ADDR, x_ptr_dst_scale_off, s_off * stype_sz_,
                    X_TMP);
            return X_DEFAULT_ADDR;
        } else {
            return x_ptr_dst_scale_off;
        }
    }

    XReg c_addr(int c_off) {
        if (c_off) {
            add_imm(X_DEFAULT_ADDR, x_ptr_comp_off, c_off * sizeof(int32_t),
                    X_TMP);
            return X_DEFAULT_ADDR;
        }

        return x_ptr_comp_off;
    }

    XReg data_chunk_addr(int node_id) {
        add_imm(X_DEFAULT_ADDR, abi_param1,
                offsetof(tail_call_param_t, curr_data_chunks)
                        + sizeof(int64_t) * (node_id),
                X_TMP);
        return X_DEFAULT_ADDR;
    }

    void step(int off, int prev_i_off, int prev_o_off, int prev_s_off,
            int prev_c_off, int &i_off, int &o_off, int &s_off, int &c_off,
            int step_size = 1) {
        i_off = prev_i_off;
        o_off = prev_o_off;
        s_off = prev_s_off;
        c_off = prev_c_off;

        if (off == 0) return;

        int start_dim = 0, dims_prod = 1;
        for (; start_dim < prb_.ndims && dims_prod != step_size; ++start_dim)
            dims_prod *= prb_.n(start_dim);
        assert(start_dim < prb_.ndims);
        off /= step_size;

        for (int dim_id = start_dim; dim_id < prb_.ndims; ++dim_id) {
            i_off += prb_.is(dim_id);
            o_off += prb_.os(dim_id);
            s_off += prb_.ss(dim_id);
            c_off += prb_.cs(dim_id);

            if (off % prb_.n(dim_id)) break;

            i_off += -prb_.n(dim_id) * prb_.is(dim_id);
            o_off += -prb_.n(dim_id) * prb_.os(dim_id);
            s_off += -prb_.n(dim_id) * prb_.ss(dim_id);
            c_off += -prb_.n(dim_id) * prb_.cs(dim_id);

            off /= prb_.n(dim_id);

            if (off == 0) break; /* FIXME: is it really required? */
        }
    }

    void step(int off, int prev_i_off, int prev_o_off, int &i_off, int &o_off,
            int step_size = 1) {
        int dummy = 0;
        step(off, prev_i_off, prev_o_off, dummy, dummy, i_off, o_off, dummy,
                dummy, step_size);
    }

    void tr8x8_sve256(int i_off, int o_off) {
        using namespace data_type;

        const auto cvt2ps
                = [=](const int startIdx, const int regNum, data_type_t idt) {
                      switch (idt) {
                          case f32:
                              /* do nothing */
                              break;
                          case s32: cvt_z_s32_f32(startIdx, regNum); break;
                          case data_type::s8:
                              cvt_z_s8_s32(startIdx, regNum);
                              cvt_z_s32_f32(startIdx, regNum);
                              break;
                          case u8:
                              cvt_z_u8_s32(startIdx, regNum);
                              cvt_z_s32_f32(startIdx, regNum);
                              break;
                          default: assert(!"unreachable");
                      }
                  };

        const auto cvt2odt = [=](const int startIdx, const int regNum,
                                     data_type_t odt, data_type_t idt) {
            switch (odt) {
                case s32:
                    if (idt == f32)
                        cvt_z_f32_s32(startIdx, regNum);
                    else if (idt == data_type::s8)
                        cvt_z_s8_s32(startIdx, regNum);
                    else if (idt == u8)
                        cvt_z_u8_s32(startIdx, regNum);
                    break;
                case data_type::s8:
                    if (idt == f32) cvt_z_f32_s32(startIdx, regNum);
                    if (utils::one_of(idt, f32, s32))
                        cvt_z_s32_s8(startIdx, regNum);
                    if (idt == u8) cvt_z_u8_s8(startIdx, regNum);
                    break;
                case u8:
                    if (idt == f32) cvt_z_f32_s32(startIdx, regNum);
                    if (utils::one_of(idt, f32, s32))
                        cvt_z_s32_u8(startIdx, regNum);
                    if (idt == data_type::s8) cvt_z_s8_u8(startIdx, regNum);
                    break;
                default: assert(!"unreachable");
            }
        };

        const int unroll = 8;

        const bool interim_f32 = (prb_.itype != f32)
                || utils::one_of(f32, prb_.itype, prb_.otype);

        const bool need_saturation
                = (utils::one_of(prb_.otype, u8, data_type::s8, s32)
                        && interim_f32);
        const uint64_t sveLen = get_sve_length();

        PReg p_size(DUMMY_IDX);
        switch (unroll * itype_sz_) {
            case 32: p_size = p_lsb_256; break;
            case 16: p_size = p_lsb_128; break;
            case 8: p_size = p_lsb_64; break;
            default: assert(!"unreachable");
        }

        const int node_0_input_stride = prb_.is(0);
        add_imm(X_TMP_0, XReg(x_ptr_in_off), itype_sz_ * i_off, X_DEFAULT_ADDR);
        for (int i = 1; i < unroll / 2; i++)
            add_imm(x_tmp_vec[i], x_tmp_vec[i - 1],
                    itype_sz_ * node_0_input_stride, X_DEFAULT_ADDR);
        for (uint32_t i = 0; i < unroll / 2; i++)
            ld1w(ZRegS {i}, p_size / T_z, ptr(x_tmp_vec[i]));
        for (int i = 0; i < unroll / 2; i++)
            add_imm(x_tmp_vec[i], x_tmp_vec[(i + 3) % 4],
                    itype_sz_ * node_0_input_stride, X_DEFAULT_ADDR);
        for (uint32_t i = 0; i < unroll / 2; i++)
            ld1w(ZRegS {4 + i}, p_size / T_z, ptr(x_tmp_vec[i]));

        if (interim_f32) cvt2ps(0, unroll, prb_.itype);

#if 0
        /* Debug code to forcedly set test pattern. */
        index(z0.s, 0, 1);
        mov(z0.s, P_NOT_256/T_m, 0);
        mov(z_tmp_vec[0].s, 16);
        for(uint32_t i=1; i<8; i++) {
          add(ZRegS{i}, ZRegS{i-1}, z_tmp_vec[0].s);
          mov(ZRegS{i}, P_NOT_256/T_m, 0);
        }
#endif

        ptrue(p_tmp0.s, VL4);
        /* 1st turn */
        for (uint32_t i = 0; i < unroll / 2; i++) {
            trn1(z_tmp_vec[i].s, ZRegS {2 * i}, ZRegS {2 * i + 1});
            trn2(z_tmp_vec[unroll / 2 + i].s, ZRegS {2 * i}, ZRegS {2 * i + 1});
        }

        /* 2nd turn */
        trn1(z4.d, z_tmp_vec[0].d, z_tmp_vec[1].d);
        trn1(z5.d, z_tmp_vec[4].d, z_tmp_vec[5].d);
        trn2(z6.d, z_tmp_vec[0].d, z_tmp_vec[1].d);
        trn2(z7.d, z_tmp_vec[4].d, z_tmp_vec[5].d);
        trn1(z_tmp_vec[0].d, z_tmp_vec[2].d, z_tmp_vec[3].d);
        trn1(z_tmp_vec[1].d, z_tmp_vec[6].d, z_tmp_vec[7].d);
        trn2(z_tmp_vec[2].d, z_tmp_vec[2].d, z_tmp_vec[3].d);
        trn2(z_tmp_vec[3].d, z_tmp_vec[6].d, z_tmp_vec[7].d);

        /* 3rd turn */
        for (uint32_t i = 0; i < unroll / 2; i++) {
            mov(ZRegD {i}, ZRegD {unroll / 2 + i});
            mov(z_tmp_vec[unroll / 2 + i].d, z_tmp_vec[i].d);
        }

        /* 4th turn */
        for (uint32_t i = 0; i < unroll / 2; i++) {
            ZRegB z {unroll / 2 + i};
            ZRegB z_tmp = z_tmp_vec[unroll / 2 + i].b;
            /* Move bit 0-127 to 128-255. */
            ext(z, z, 16);
            /* Move bit 128-255 to 0-127. */
            ext(z_tmp, z_tmp, sveLen - 16);
        }

        /* 5th turn */
        for (uint32_t i = 0; i < unroll / 2; i++) {
            ZRegS z0 {i};
            ZRegS z1 {unroll / 2 + i};
            sel(z0, p_tmp0.s, z0, z_tmp_vec[unroll / 2 + i].s);
            sel(z1, p_tmp0, z1, z_tmp_vec[i].s);
        }

        if (need_saturation) {
            init_saturate_f32(ymm_zero_, ymm_saturation_ubound_, X_TMP_0,
                    interim_f32 ? f32 : prb_.itype, prb_.otype);
            for (int i = 0; i < unroll; i++)
                saturate_f32(ZRegS(i), ymm_zero_, ymm_saturation_ubound_,
                        prb_.otype, P_ALL_ONE);
        }

        if (prb_.otype != f32)
            cvt2odt(0, unroll, prb_.otype, interim_f32 ? f32 : prb_.itype);

        const int node_1_output_stride = prb_.os(1);

        switch (unroll * otype_sz_) {
            case 32: p_size = p_lsb_256; break;
            case 16: p_size = p_lsb_128; break;
            case 8: p_size = p_lsb_64; break;
            default: assert(!"unreachable");
        }

        add_imm(X_TMP_0, XReg(x_ptr_out_off), otype_sz_ * o_off,
                X_DEFAULT_ADDR);
        for (int i = 1; i < unroll / 2; i++)
            add_imm(x_tmp_vec[i], x_tmp_vec[i - 1],
                    otype_sz_ * node_1_output_stride, X_DEFAULT_ADDR);
        for (uint32_t i = 0; i < 4; i++)
            st1w(ZRegS {i}, p_size / T_z, ptr(x_tmp_vec[i]));
        for (int i = 0; i < unroll / 2; i++)
            add_imm(x_tmp_vec[i], x_tmp_vec[(i + 3) % 4],
                    otype_sz_ * node_1_output_stride, X_DEFAULT_ADDR);

        for (uint32_t i = 0; i < unroll / 2; i++)
            st1w(ZRegS {4 + i}, p_size / T_z, ptr(x_tmp_vec[i]));
    }

    bool can_do_tr8x8() {
        using namespace data_type;

        static constexpr int desirable_node_size = 8;
        static constexpr int desirable_stride = 1;

        // This processing is relied on swaping two innermost dimension.
        // Therefore, input stride in second node and output stride in first node
        // have to be equal to 1.

        return mayiuse(sve_256) && prb_.ndims >= 2
                && ((utils::one_of(prb_.itype, u8, data_type::s8, s32, f32)
                        && utils::one_of(
                                prb_.otype, u8, data_type::s8, s32, f32)))
                && utils::everyone_is(desirable_node_size, prb_.n(0), prb_.n(1))
                && utils::everyone_is(desirable_stride, prb_.os(0), prb_.is(1))
                && !prb_.is_tail_present
                && prb_.src_scale_type == scale_type_t::NONE
                && prb_.dst_scale_type == scale_type_t::NONE
                && prb_.beta == 0.f;
    }

    bool process_unroll_tr8x8(const int ndims, const int len) {
        if (!can_do_tr8x8()) return false;

        const int step_size = prb_.n(0) * prb_.n(1);
        int i_off = 0, o_off = 0;
        for (int off = 0; off < len; off += step_size) {
            step(off, i_off, o_off, i_off, o_off, step_size);
            tr8x8_sve256(i_off, o_off);
        }

        return true;
    }

    template <cpu_isa_t isa>
    bool process_direct_copy(const int ndims, const int len) {
        using namespace data_type;

        static constexpr int desirable_stride = 1;
        using TRegS =
                typename utils::conditional<isa == asimd, VReg4S, ZRegS>::type;
        const int simd_w = cpu_isa_traits<isa>::vlen / itype_sz_;

        // TODO: support tail_processing for direct copy

        const bool do_src_zp = prb_.req_src_zp;
        const bool do_dst_zp = prb_.req_dst_zp;
        const bool zp_applicable = IMPLICATION(
                (do_src_zp || do_dst_zp), utils::one_of(prb_.itype, s32, f32));
        const bool can_do = true && mayiuse(isa)
                && compensation_needed_ == false
                && utils::everyone_is(desirable_stride, prb_.os(0), prb_.is(0))
                && (false || (prb_.itype == prb_.otype ? zp_applicable : false)
                        || (prb_.itype == s32 && prb_.otype == f32)
                        || (prb_.itype == f32 && prb_.otype == s32))
                && len % simd_w == 0 && prb_.n(0) % len == 0
                && !prb_.is_tail_present
                && prb_.src_scale_type == scale_type_t::NONE
                && prb_.dst_scale_type == scale_type_t::NONE
                && prb_.beta == 0.f;
        if (!can_do) return false;

        static constexpr int vmm_zp_last_idx = 15;
        const auto vmm_src_zp
                = TRegS(do_dst_zp ? vmm_zp_last_idx - 1 : vmm_zp_last_idx);
        if (do_src_zp) {
            uni_ld1rw(vmm_src_zp, PARAM(src_zp));
            uni_scvtf(vmm_src_zp, vmm_src_zp);
        }
        const auto vmm_dst_zp = TRegS(vmm_zp_last_idx);
        if (do_dst_zp) {
            uni_ld1rw(vmm_dst_zp, PARAM(dst_zp));
            uni_scvtf(vmm_dst_zp, vmm_dst_zp);
        }

        const auto apply_zp_ps = [&](const TRegS vmm) {
            if (do_src_zp) fsub(vmm, vmm, vmm_src_zp);
            if (do_dst_zp) fadd(vmm, vmm, vmm_dst_zp);
        };

        for (int off = 0; off < len;) {
            // TODO: we need extra reg for proper saturation if otype == s32
            int unroll
                    = nstl::min(16 - (prb_.otype == s32), (len - off) / simd_w);
            unroll = (do_src_zp || do_dst_zp)
                    ? nstl::min(unroll, 16 - do_src_zp - do_dst_zp)
                    : unroll;

            int ur = 0;
            int tmp_ur = 0;
            while (ur < unroll) {
                int count = 0;
                const int vlen = cpu_isa_traits<isa>::vlen;

                do {
                    add_imm(x_tmp_vec[count++], x_ptr_in_off,
                            (off + ur * simd_w) * itype_sz_, X_DEFAULT_ADDR);
                    ur++;
                } while (ur < unroll && count < x_tmp_vec_size);

                for (int i = 0; i < count; i++) {
                    if (vlen == 64 || vlen == 32)
                        ld1w(ZRegS(tmp_ur + i), p_lsb_256 / T_z,
                                ptr(x_tmp_vec[i]));
                    else if (vlen == 16)
                        ldr(QReg(tmp_ur + i), ptr(x_tmp_vec[i]));
                    else
                        assert(!"unreachable");
                }
                tmp_ur += count;
            }

            if (prb_.itype != prb_.otype) {
                for (int ur = 0; ur < unroll; ++ur) {
                    TRegS r(ur);
                    if (prb_.itype == s32 && prb_.otype == f32) {
                        uni_scvtf(r, r);
                        apply_zp_ps(r);
                    } else if (prb_.itype == f32 && prb_.otype == s32) {
                        apply_zp_ps(r);
                        uni_frinti(r, r);
                        uni_fcvtzs(r, r);
                    } else
                        assert(!"unreachable");
                }
            } else if (do_src_zp || do_dst_zp) {
                for (int ur = 0; ur < unroll; ++ur) {
                    const auto vmm = TRegS(ur);
                    if (prb_.otype == f32) {
                        apply_zp_ps(vmm);
                    } else if (prb_.otype == s32) {
                        uni_scvtf(vmm, vmm);
                        apply_zp_ps(vmm);
                        uni_frinti(vmm, vmm);
                        uni_fcvtzs(vmm, vmm);
                    }
                }
            }

            ur = 0;
            tmp_ur = 0;
            while (ur < unroll) {
                int count = 0;
                const int vlen = cpu_isa_traits<isa>::vlen;

                do {
                    add_imm(x_tmp_vec[count++], x_ptr_out_off,
                            (off + ur * simd_w) * otype_sz_, X_DEFAULT_ADDR);
                    ur++;
                } while (ur < unroll && count < x_tmp_vec_size);

                for (int i = 0; i < count; i++) {
                    if (vlen == 64 || vlen == 32)
                        st1w(ZRegS(tmp_ur + i), p_lsb_256 / T_z,
                                ptr(x_tmp_vec[i]));
                    else if (vlen == 16)
                        str(QReg(tmp_ur + i), ptr(x_tmp_vec[i]));
                    else
                        assert(!"unreachable");
                }
                tmp_ur += count;
            }

            off += unroll * simd_w;
        }

        return true;
    }

    void process_unroll_generic_step(int reg_unroll, const int *i_off,
            const int *o_off, const int *s_off, const int *c_off,
            const int *zero_padding, const bool tail_processing) {
        using namespace data_type;

        auto cvt2ps
                = [=](const int startIdx, const int regNum, data_type_t idt) {
                      switch (idt) {
                          case f32:
                              /* do nothing */
                              break;
                          case s32: cvt_v_s32_f32(startIdx, regNum); break;
                          case data_type::s8:
                              cvt_v_s8_s32(startIdx, regNum);
                              cvt_v_s32_f32(startIdx, regNum);
                              break;
                          case u8:
                              cvt_v_u8_s32(startIdx, regNum);
                              cvt_v_s32_f32(startIdx, regNum);
                              break;
                          default: assert(!"unreachable");
                      }
                  };

        auto cvt2odt = [=](const int startIdx, const int regNum,
                               data_type_t odt, data_type_t idt) {
            switch (odt) {
                case s32:
                    if (idt == f32)
                        cvt_v_f32_s32(startIdx, regNum);
                    else if (idt == data_type::s8)
                        cvt_v_s8_s32(startIdx, regNum);
                    else if (idt == u8)
                        cvt_v_u8_s32(startIdx, regNum);
                    break;
                case data_type::s8:
                    if (idt == f32) cvt_v_f32_s32(startIdx, regNum);
                    if (idt == f32 || idt == s32)
                        cvt_v_s32_s8(startIdx, regNum);
                    if (idt == u8) { cvt_v_u8_s8(startIdx, regNum); }
                    break;
                case u8:
                    if (idt == f32) cvt_v_f32_s32(startIdx, regNum);
                    if (idt == f32 || idt == s32)
                        cvt_v_s32_u8(startIdx, regNum);
                    if (idt == data_type::s8) cvt_v_s8_u8(startIdx, regNum);
                    break;
                case bf16:
                    if (idt == f32) cvt_v_f32_bf16(startIdx, regNum);
                    break;
                default: assert(!"unreachable");
            }
        };

        auto load_bytes_addr = [=](const int ur, const int r) {
            add_imm(x_tmp_vec[r], x_ptr_in_off, i_off[ur + r] * itype_sz_,
                    X_DEFAULT_ADDR);
        };
        auto load_bytes = [=](const int ur, int size, int r) {
            switch (size) {
                case 4: ld1(VReg4S(ur)[r], ptr(x_tmp_vec[r])); break;
                case 2: ld1(VReg8H(ur)[r], ptr(x_tmp_vec[r])); break;
                case 1: ld1(VReg16B(ur)[r], ptr(x_tmp_vec[r])); break;
                default: assert(!"unreachable");
            }
        };

        auto store = [=](const XReg &addr, const VReg ymm, int size) {
            const uint32_t xmm = ymm.getIdx();
            switch (size) {
                case 16: str(QReg(xmm), ptr(addr)); break;
                case 8: str(DReg(xmm), ptr(addr)); break;
                case 4: str(SReg(xmm), ptr(addr)); break;
                case 2: str(HReg(xmm), ptr(addr)); break;
                case 1: str(BReg(xmm), ptr(addr)); break;
                default: assert(!"unreachable");
            }
        };

        /* check whether loading 4 values at once is possible */
        static constexpr int xmm_vlen = 4;
        bool can_load_xmm = reg_unroll % xmm_vlen == 0;
        for (int ur = 1; ur < reg_unroll; ++ur)
            if (i_off[ur] != i_off[ur - 1] + 1) {
                can_load_xmm = false;
                break;
            }
        const int load_step = can_load_xmm ? xmm_vlen : 1;

        /* check whether storing 4 values at once is possible */
        bool can_store_xmm = reg_unroll % xmm_vlen == 0;
        for (int ur = 1; ur < reg_unroll; ++ur)
            if (o_off[ur] != o_off[ur - 1] + 1) {
                can_store_xmm = false;
                break;
            }
        const int ur_step = can_store_xmm ? 4 : 1;
        const int load_tail_step
                = !can_load_xmm && can_store_xmm ? ur_step : load_step;

        const bool interim_f32 = interim_f32_needed();

        const bool need_saturation
                = (utils::one_of(prb_.otype, u8, data_type::s8, s32)
                        && interim_f32);

        std::vector<int> store_masks;
        if (tail_processing) {
            for (int ur = 0; ur < reg_unroll; ur += load_tail_step) {
                uni_clear(VReg(ur));
                store_masks.push_back(0);
                for (int r = 0; r < load_tail_step; ++r) {
                    if (zero_padding[ur + r] == 0) {
                        store_masks.back() += 1 << r;
                        load_bytes_addr(ur, r);
                    }
                }

                for (int r = 0; r < load_tail_step; ++r)
                    if (zero_padding[ur + r] == 0) load_bytes(ur, itype_sz_, r);
            }
        } else {
            if (!can_load_xmm && can_store_xmm) {
                assert(ur_step == xmm_vlen);
                /* load with stride */
                for (int ur = 0; ur < reg_unroll; ur += ur_step) {
                    for (int r = 0; r < ur_step; ++r) {
                        load_bytes_addr(ur, r);
                    }
                    for (int r = 0; r < ur_step; ++r)
                        load_bytes(ur, itype_sz_, r);
                }
            } else {
                int ur = 0;
                int tmp_ur = 0;
                while (ur < reg_unroll) {
                    int count = 0;

                    do {
                        add_imm(x_tmp_vec[count++], x_ptr_in_off,
                                i_off[ur] * itype_sz_, X_DEFAULT_ADDR);
                        ur += load_step;
                    } while (ur < reg_unroll && count < x_tmp_vec_size);

                    for (int i = 0; i < count; i++) {

                        switch (load_step * itype_sz_) {
                            case 16:
                                ldr(QReg(tmp_ur), ptr(x_tmp_vec[i]));
                                break;
                            case 8: ldr(DReg(tmp_ur), ptr(x_tmp_vec[i])); break;
                            case 4: ldr(SReg(tmp_ur), ptr(x_tmp_vec[i])); break;
                            case 2: ldr(HReg(tmp_ur), ptr(x_tmp_vec[i])); break;
                            case 1: ldr(BReg(tmp_ur), ptr(x_tmp_vec[i])); break;
                            default: assert(!"unreachable");
                        }
                        tmp_ur += load_step;
                    }
                }
            }
        }

        /* xmm[:] <-- (f32)xmm[:] */
        if (interim_f32) {
            const int cvt_step = nstl::max(load_step, ur_step);
            for (int ur = 0; ur < reg_unroll; ur += cvt_step)
                cvt2ps(ur, 1, prb_.itype);
        }

        if (can_load_xmm && !can_store_xmm) {
            // transposition on the fly
            const bool fast_return = prb_.src_scale_type != scale_type_t::MANY
                    && prb_.dst_scale_type != scale_type_t::MANY
                    && prb_.beta == 0.f;
            if (fast_return) {
                if (prb_.src_scale_type == scale_type_t::COMMON)
                    for (int ur = 0; ur < reg_unroll; ur += load_step)
                        fmul(VReg4S(ur), VReg4S(ur), xmm_src_scales_);
                if (prb_.dst_scale_type == scale_type_t::COMMON)
                    for (int ur = 0; ur < reg_unroll; ur += load_step)
                        fmul(VReg4S(ur), VReg4S(ur), xmm_dst_scales_);
                if (prb_.otype != f32) {
                    init_saturate_f32(xmm_zero_, xmm_saturation_ubound_,
                            X_TMP_0, interim_f32 ? f32 : prb_.itype,
                            prb_.otype);
                    for (int ur = 0; ur < reg_unroll; ur += load_step) {
                        if (need_saturation)
                            saturate_f32(VReg4S(ur), xmm_zero_,
                                    xmm_saturation_ubound_, prb_.otype,
                                    P_ALL_ONE);
                    }

                    for (int ur = 0; ur < reg_unroll; ur += load_step)
                        cvt2odt(ur, 1, prb_.otype,
                                interim_f32 ? f32 : prb_.itype);
                }
                for (int ur = 0; ur < reg_unroll; ur += load_step) {
                    for (int r = 0; r < load_step; ++r) {
                        add_imm(x_tmp_vec[r], x_ptr_out_off,
                                o_off[ur + r] * otype_sz_, X_DEFAULT_ADDR);
                    }

                    for (int r = 0; r < load_step; ++r) {
                        if (otype_sz_ == 4)
                            st1(VReg4S(ur)[r], ptr(x_tmp_vec[r]));
                        else if (otype_sz_ == 2)
                            st1(VReg8H(ur)[r], ptr(x_tmp_vec[r]));
                        else
                            st1(VReg16B(ur)[r], ptr(x_tmp_vec[r]));
                    }
                }
                return;
            }

            /* scatter elements of xmm into 4 xmms */
            if (itype_sz_ == 4 || interim_f32) {
                for (int ur = 0; ur < reg_unroll; ur += load_step)
                    for (int r = 1; r < load_step; ++r) {
                        VReg4S v(ur);
                        VReg4S v_r(ur + r);
                        dup(VReg16B(ur + r), VReg16B(ur)[0]);
                        ins(VReg4S(ur + r)[0], VReg4S(ur)[r]);
                    }
            } else {
                for (int ur = 0; ur < reg_unroll; ur += load_step)
                    for (int r = 1; r < load_step; ++r)
                        ext(VReg16B(ur + r), VReg16B(ur), VReg16B(ur),
                                itype_sz_ * r);
            }
        }

        /* src zero point application */
        if (prb_.req_src_zp) {
            for (int ur = 0; ur < reg_unroll; ur += ur_step) {
                const auto xmm = VReg4S(ur);
                if (interim_f32)
                    fsub(xmm, xmm, xmm_src_zp_);
                else
                    sub(xmm, xmm, xmm_src_zp_);
            }
        }

        /* scale and beta processing */
        if (can_store_xmm) {
            const auto apply_scales = [&](const VReg4S &vreg_scales,
                                              scale_arg_t scale_arg,
                                              scale_type_t scale_type) {
                if (scale_type == scale_type_t::COMMON) {
                    for (int ur = 0; ur < reg_unroll; ur += ur_step)
                        fmul(VReg4S(ur), VReg4S(ur), vreg_scales);
                } else if (scale_type == scale_type_t::MANY) {
                    enum class scale_load_type_t { bcast, load, gather };
                    const uint32_t idx = vreg_scales.getIdx();

                    uni_clear(VReg(idx));
                    for (int ur = 0; ur < reg_unroll; ur += ur_step) {
                        scale_load_type_t scale_load_type
                                = scale_load_type_t::bcast; // the best case

                        for (int r = ur + 1; r < ur + ur_step; ++r)
                            if (s_off[r] != s_off[r - 1] + 0)
                                scale_load_type = scale_load_type_t::load;

                        if (scale_load_type == scale_load_type_t::bcast
                                && !tail_processing) {
                            if (scale_arg == scale_arg_t::SRC)
                                ld1r(vreg_scales, ptr(src_s_addr(s_off[ur])));
                            else
                                ld1r(vreg_scales, ptr(dst_s_addr(s_off[ur])));
                            fmul(VReg4S(ur), VReg4S(ur), vreg_scales);
                            continue;
                        }

                        // bcast doesn't work, the next try -- load
                        for (int r = ur + 1; r < ur + ur_step; ++r)
                            if (s_off[r] != s_off[r - 1] + 1)
                                scale_load_type = scale_load_type_t::gather;

                        if (scale_load_type == scale_load_type_t::load
                                && !tail_processing) {
                            if (scale_arg == scale_arg_t::SRC)
                                ldr(QReg {idx}, ptr(src_s_addr(s_off[ur])));
                            else
                                ldr(QReg {idx}, ptr(dst_s_addr(s_off[ur])));

                            fmul(VReg4S(ur), VReg4S(ur), VReg4S {idx});
                            continue;
                        }

                        // load doesn't work as well
                        // so gather the scale factors one by one
                        for (int r = ur; r < ur + ur_step; ++r)
                            if (zero_padding[r] == 0 || !tail_processing) {
                                if (scale_arg == scale_arg_t::SRC)
                                    mov(x_tmp_vec[r - ur],
                                            src_s_addr(s_off[r]));
                                else
                                    mov(x_tmp_vec[r - ur],
                                            dst_s_addr(s_off[r]));
                            }
                        for (int r = ur; r < ur + ur_step; ++r)
                            if (zero_padding[r] == 0 || !tail_processing)
                                ld1(vreg_scales[r - ur],
                                        ptr(x_tmp_vec[r - ur]));
                        fmul(VReg4S(ur), VReg4S(ur), vreg_scales);
                    }
                }
            };
            /* xmm <-- src_scales * xmm[:] */
            apply_scales(
                    xmm_src_scales_, scale_arg_t::SRC, prb_.src_scale_type);

            /* xmm[:] <-- beta * dst + xmm[:] */
            assert(prb_.beta == 0.f || prb_.beta == 1.f);
            if (prb_.beta == 1.f) {
                int ur = 0;
                int tmp_ur = 0;

                while (ur < reg_unroll) {
                    int count = 0;

                    do {
                        add_imm(x_tmp_vec[count++], x_ptr_out_off,
                                o_off[ur] * otype_sz_, X_DEFAULT_ADDR);
                        ur += ur_step;
                    } while (ur < reg_unroll && count < x_tmp_vec_size);

                    assert(count <= z_tmp_vec_size);
                    /* Firstly, data is loaded. */
                    for (int i = 0; i < count; i++) {

                        if (prb_.otype == f32 || prb_.otype == s32) {
                            ldr(QReg(tmp_vec_idx[i]), ptr(x_tmp_vec[i])); // bug
                        } else if (prb_.otype == data_type::s8
                                || prb_.otype == u8) {
                            ldr(SReg(tmp_vec_idx[i]), ptr(x_tmp_vec[i])); // bug
                        } else
                            assert(!"unreachable");
                    }

                    /* Secondly, it is added. */
                    if (prb_.otype == f32) {
                        for (int i = 0; i < count; i++) {
                            VReg4S v(tmp_ur);
                            fadd(v, v, VReg4S(tmp_vec_idx[i]));
                            tmp_ur += ur_step;
                        }
                    } else {
                        for (int i = 0; i < count; i++) {
                            /* cvt2ps() generate successive instructions
                               which have save destination operand,
                               but out of order can be expected. */
                            cvt2ps(tmp_vec_idx[i], 1, prb_.otype);
                        }
                        for (int i = 0; i < count; i++) {
                            VReg4S v(tmp_ur);
                            fadd(v, v, VReg4S(tmp_vec_idx[i]));
                            tmp_ur += ur_step;
                        }
                    }
                }
            }

            /* dst <-- dst_scales * xmm[:] */
            apply_scales(
                    xmm_dst_scales_, scale_arg_t::DST, prb_.dst_scale_type);
        } else {
            const auto apply_scales = [&](const VReg4S &vreg_scales,
                                              scale_arg_t scale_arg,
                                              scale_type_t scale_type) {
                if (scale_type == scale_type_t::COMMON) {
                    for (int ur = 0; ur < reg_unroll; ur += ur_step)
                        fmul(VReg4S(ur), VReg4S(ur), vreg_scales);
                } else if (scale_type == scale_type_t::MANY) {
#define DUMMY_IDX_ (99)
                    std::vector<uint32_t> idx_list;
                    std::vector<int> offt_list;
                    std::vector<uint32_t> vec_reg;
                    std::vector<XReg> addr_reg;
                    const size_t max_cnt_per_loop
                            = std::min(tmp_vec_idx.size(), x_tmp_vec.size());
                    size_t cnt = 0; // valid unroll steps count

                    // 1. Listing up valid steps
                    for (int ur = 0; ur < reg_unroll; ur += ur_step) {
                        if (zero_padding[ur] == 0 || !tail_processing) {
                            idx_list.push_back(ur);
                            offt_list.push_back(s_off[ur]);
                            vec_reg.push_back(
                                    tmp_vec_idx[cnt % max_cnt_per_loop]);
                            if (s_off[ur])
                                addr_reg.push_back(
                                        x_tmp_vec[cnt % max_cnt_per_loop]);
                            else
                                addr_reg.push_back(scale_arg == scale_arg_t::SRC
                                                ? x_ptr_src_scale_off
                                                : x_ptr_dst_scale_off);
                            cnt++;
                        }
                    }
                    /* 2. Generate instructions considering instruction order.
		       If cnt > max_cnt_per_loop, the following instruction sets are
		       generated several times.
		       add x?, ..., add x? for calculating address
		       ldr s?, ..., ldr s? for loading data
		       fmul v?, ..., fmul v? for scaling */
                    for (size_t ur = 0; ur < cnt;) {
                        // Calculating address
                        for (size_t i = ur;
                                i < cnt && i - ur < max_cnt_per_loop; i++)
                            add_imm(addr_reg[i],
                                    scale_arg == scale_arg_t::SRC
                                            ? x_ptr_src_scale_off
                                            : x_ptr_dst_scale_off,
                                    offt_list[i] * stype_sz_, X_TMP);
                        // Loading data
                        for (size_t i = ur;
                                i < cnt && i - ur < max_cnt_per_loop; i++)
                            ldr(SReg(vec_reg[i]), ptr(addr_reg[i]));
                        // Scaling
                        for (size_t i = ur;
                                i < cnt && i - ur < max_cnt_per_loop; i++) {
                            VReg4S v(idx_list[i]);
                            fmul(v, v, VReg4S(vec_reg[i]));
                        }
                        ur += std::min(cnt, max_cnt_per_loop);
                    }
                }
#undef DUMMY_IDX_
            };

            /* xmm[0] <-- src_scales * xmm[0] */
            apply_scales(
                    xmm_src_scales_, scale_arg_t::SRC, prb_.src_scale_type);

            /* xmm[0] <-- beta * dst + xmm[0] */
            assert(prb_.beta == 0.f || prb_.beta == 1.f);
            if (prb_.beta == 1.f) {
                int ur = 0;
                int tmp_ur = 0;
                while (ur < reg_unroll) {
                    int count = 0;

                    do {
                        add_imm(x_tmp_vec[count++], x_ptr_out_off,
                                o_off[ur] * otype_sz_, X_DEFAULT_ADDR);
                        ur += ur_step;
                    } while (ur < reg_unroll && count < (x_tmp_vec_size / 2));

                    assert(static_cast<size_t>(count) <= z_tmp_vec.size());

                    if (prb_.otype == f32) {
                        /* addss: dest[31:0] <- src1[31:0] + src2[31:0]
                         dset[MAXVL-1:32] (Unmodified) */
                        for (int i = 0; i < count; i++) {
                            ld1(VReg4S(z_tmp_vec[i].getIdx())[0],
                                    ptr(x_tmp_vec[i]));
                        }
                        for (int i = 0; i < count; i++) {
                            SReg s {tmp_vec_idx[i]};
                            fadd(s, s, SReg(tmp_ur + ur_step * i));
                        }
                        for (int i = 0; i < count; i++) {
                            mov(VReg4S(tmp_ur)[0], VReg4S(tmp_vec_idx[i])[0]);
                            tmp_ur += ur_step;
                        }
                    } else {
                        for (int i = 0; i < count; i++) {
                            if (prb_.otype == s32) {
                                ldr(SReg(tmp_vec_idx[i]), ptr(x_tmp_vec[i]));
                            } else if (utils::one_of(
                                               prb_.otype, data_type::s8, u8)) {
                                ldr(BReg(tmp_vec_idx[i]), ptr(x_tmp_vec[i]));
                            } else {
                                assert(!"unsupported o_type");
                            }
                            cvt2ps(tmp_vec_idx[i], 1, prb_.otype);
                        }
                        for (int i = 0; i < count; i++) {
                            VReg4S v(tmp_ur);
                            fadd(v, v, VReg4S(tmp_vec_idx[i]));
                            tmp_ur += ur_step;
                        }
                    }
                }
            }

            /* dst <-- dst_scales * xmm[0] */
            apply_scales(
                    xmm_dst_scales_, scale_arg_t::DST, prb_.dst_scale_type);
        }

        /* dst zero point application */
        if (prb_.req_dst_zp) {
            for (int ur = 0; ur < reg_unroll; ur += ur_step) {
                const auto xmm = VReg4S(ur);
                if (interim_f32)
                    fadd(xmm, xmm, xmm_dst_zp_);
                else
                    add(xmm, xmm, xmm_dst_zp_);
            }
        }

        /* adjust scale application */
        if (prb_.scale_adjust != 1.f) {
            dup(xmm_tmp_, reg_scale_adjust_);
            for (int ur = 0; ur < reg_unroll; ur += ur_step) {
                fmul(VReg4S(ur), VReg4S(ur), xmm_tmp_);
            }
        }

        if (need_saturation) {
            init_saturate_f32(xmm_zero_, xmm_saturation_ubound_, X_TMP_0, f32,
                    prb_.otype, compensation_needed_);
            for (int ur = 0; ur < reg_unroll; ur += ur_step) {
                saturate_f32(VReg4S(ur), xmm_zero_, xmm_saturation_ubound_,
                        prb_.otype, P_ALL_ONE, compensation_needed_);
            }

            // reset back xmm_zero_ if needed.
            if (compensation_needed_ && (prb_.req_src_zp || prb_.req_dst_zp))
                uni_clear(VReg(xmm_zero_.getIdx()));
        }

        if (compensation_needed_) {
            const uint32_t xmm_begin = 9;
            const uint32_t xmm_end = 11;
            uint32_t xmm_id = xmm_begin;
            const auto get_temp_xmm = [&] {
                const Xbyak_aarch64::VReg temp {xmm_id++};

                if (xmm_id > xmm_end) { xmm_id = xmm_begin; }

                return temp;
            };
            if (can_store_xmm) {
                enum class comp_load_type_t { bcast, load, gather };

                for (int ur = 0; ur < reg_unroll; ur += ur_step) {

                    bool all_ip_padding_one = true;
                    bool all_ip_padding_zero = true;
                    for (int r = ur; r < ur + ur_step; r++) {
                        if (zero_padding[r] != 1)
                            all_ip_padding_one = false;
                        else
                            all_ip_padding_zero = false;
                    }
                    if (all_ip_padding_one) continue;

                    comp_load_type_t comp_load_type = comp_load_type_t::bcast;

                    for (int r = ur + 1; r < ur + ur_step; ++r)
                        if (c_off[r] != c_off[r - 1] + 0) {
                            comp_load_type = comp_load_type_t::load;
                            break;
                        }

                    if (comp_load_type == comp_load_type_t::bcast
                            && all_ip_padding_zero) {
                        frinti(xmm_compensation, VReg4S(ur));
                        fcvtzs(xmm_compensation, xmm_compensation);
                        addv(SReg(xmm_compensation.getIdx()), xmm_compensation);
                        addv(SReg(xmm_compensation.getIdx()), xmm_compensation);
                        const auto comp_addr = c_addr(c_off[ur]);
                        const auto xmm_tmp_ = get_temp_xmm().s4;
                        ldr(SReg(xmm_tmp_.getIdx()), ptr(comp_addr));
                        add(xmm_tmp_, xmm_tmp_, xmm_compensation);
                        str(SReg(xmm_tmp_.getIdx()), ptr(comp_addr));
                        continue;
                    }

                    if (comp_load_type == comp_load_type_t::load)
                        for (int r = ur + 1; r < ur + ur_step; ++r)
                            if (c_off[r] != c_off[r - 1] + 1) {
                                comp_load_type = comp_load_type_t::gather;
                                break;
                            }

                    if (comp_load_type == comp_load_type_t::load
                            && all_ip_padding_zero) {
                        const auto xmm_reorder_result = VReg4S(ur);
                        const auto comp_addr = c_addr(c_off[ur]);
                        frinti(xmm_compensation, xmm_reorder_result);
                        fcvtzs(xmm_compensation, xmm_compensation);
                        const auto xmm_tmp_ = get_temp_xmm().s4;
                        ldr(QReg(xmm_tmp_.getIdx()), ptr(comp_addr));
                        add(xmm_compensation, xmm_compensation, xmm_tmp_);
                        str(QReg(xmm_compensation.getIdx()), ptr(comp_addr));
                        continue;
                    }

                    frinti(xmm_compensation, VReg4S(ur));
                    fcvtzs(xmm_compensation, xmm_compensation);
                    for (int r = ur; r < ur + ur_step; ++r) {
                        if (zero_padding[r] == 0 || !tail_processing) {
                            mov(W_TMP_0, xmm_compensation[r % 4]);
                            const auto comp_addr = c_addr(c_off[r]);
                            ldr(W_TMP_1, ptr(comp_addr));
                            add(W_TMP_0, W_TMP_0, W_TMP_1);
                            str(W_TMP_0, ptr(comp_addr));
                        }
                    }
                }
            } else {
                for (int ur = 0; ur < reg_unroll; ur += ur_step) {
                    if (zero_padding[ur] == 0 || !tail_processing) {
                        const auto comp_addr = c_addr(c_off[ur]);
                        frinti(xmm_compensation, VReg4S(ur));
                        fcvtzs(xmm_compensation, xmm_compensation);
                        const auto xmm_tmp_ = get_temp_xmm().s4;
                        ld1(xmm_tmp_, ptr(comp_addr));
                        add(xmm_compensation, xmm_compensation, xmm_tmp_);
                        st1(VReg(xmm_compensation.getIdx()).s[0],
                                ptr(comp_addr));
                    }
                }
            }
        }

        for (int ur = 0; ur < reg_unroll; ur += ur_step) {
            if (prb_.req_src_zp || prb_.req_dst_zp) {
                const bool use_store_masks = !store_masks.empty();
                if (use_store_masks) {
                    const auto mask = (~store_masks[ur / ur_step]) & 0xF;
                    switch (mask) {
                        case 0x0:
                            /* Do nothing */
                            break;
                        case 0x1: ins(VReg4S(ur)[0], xmm_zero_[0]); break;
                        case 0x2: ins(VReg4S(ur)[1], xmm_zero_[1]); break;
                        case 0x3:
                            ins(VReg2D(ur)[0], VReg2D(xmm_zero_.getIdx())[0]);
                            break;
                        case 0x4: ins(VReg4S(ur)[2], xmm_zero_[2]); break;
                        case 0x5:
                            ins(VReg4S(ur)[0], xmm_zero_[0]);
                            ins(VReg4S(ur)[2], xmm_zero_[2]);
                            break;
                        case 0x6:
                            ins(VReg4S(ur)[1], xmm_zero_[1]);
                            ins(VReg4S(ur)[2], xmm_zero_[2]);
                            break;
                        case 0x7:
                            ins(VReg2D(ur)[0], VReg2D(xmm_zero_.getIdx())[0]);
                            ins(VReg4S(ur)[2], xmm_zero_[2]);
                            break;
                        case 0x8: ins(VReg4S(ur)[3], xmm_zero_[3]); break;
                        case 0x9:
                            ins(VReg4S(ur)[0], xmm_zero_[0]);
                            ins(VReg4S(ur)[3], xmm_zero_[3]);
                            break;
                        case 0xa:
                            ins(VReg4S(ur)[1], xmm_zero_[1]);
                            ins(VReg4S(ur)[3], xmm_zero_[3]);
                            break;
                        case 0xb:
                            ins(VReg2D(ur)[0], VReg2D(xmm_zero_.getIdx())[0]);
                            ins(VReg4S(ur)[3], xmm_zero_[3]);
                            break;
                        case 0xc:
                            ins(VReg2D(ur)[1], VReg2D(xmm_zero_.getIdx())[1]);
                            break;
                        case 0xd:
                            ins(VReg4S(ur)[0], xmm_zero_[0]);
                            ins(VReg2D(ur)[1], VReg2D(xmm_zero_.getIdx())[1]);
                            break;
                        case 0xe:
                            ins(VReg4S(ur)[1], xmm_zero_[1]);
                            ins(VReg2D(ur)[1], VReg2D(xmm_zero_.getIdx())[1]);
                            break;
                        case 0xf: movi(VReg16B(ur), 0); break;
                        default: assert(!"unreachable");
                    }
                }
            }
            if (prb_.otype != f32)
                cvt2odt(ur, 1, prb_.otype, interim_f32 ? f32 : prb_.itype);

            store(o_addr(o_off[ur]), VReg(ur), ur_step * otype_sz_);
        }
    }

    bool interim_f32_needed() {
        using namespace data_type;

        return utils::one_of(f32, prb_.itype, prb_.otype)
                || prb_.src_scale_type != scale_type_t::NONE
                || prb_.dst_scale_type != scale_type_t::NONE || prb_.beta != 0.f
                || ((prb_.req_src_zp || prb_.req_dst_zp)
                                ? !(prb_.itype == s32 && prb_.otype == s32)
                                : false)
                || (prb_.itype != f32 && compensation_needed_)
                || prb_.scale_adjust != 1.f;
    }

    void process_unroll_generic(
            const int ndims, int len, const bool tail_processing) {
        assert(IMPLICATION(prb_.nodes[0].tail_size > 0,
                len == static_cast<int>(prb_.nodes[0].n)
                        || len == static_cast<int>(prb_.nodes[0].tail_size)));

        const int blk = 8;

        int i_off[2 * blk] = {0};
        int o_off[2 * blk] = {0};
        int s_off[2 * blk] = {0};
        int c_off[2 * blk] = {0};

        int curr = 0; // will switch between 0 and 1

        const bool interim_f32 = interim_f32_needed();

        if (prb_.req_src_zp) {
            add_imm(X_DEFAULT_ADDR, PARAM(src_zp), X_TMP_0);
            ld1r(xmm_src_zp_, ptr(X_DEFAULT_ADDR));
            if (interim_f32) scvtf(xmm_src_zp_, xmm_src_zp_);
        }
        if (prb_.req_dst_zp) {
            add_imm(X_DEFAULT_ADDR, PARAM(dst_zp), X_TMP_0);
            ld1r(xmm_dst_zp_, ptr(X_DEFAULT_ADDR));
            if (interim_f32) scvtf(xmm_dst_zp_, xmm_dst_zp_);
        }

        for (int off = 0; off < len; off += blk) {
            const int reg_unroll = nstl::min(off + blk, len) - off;
            int zero_padding[blk] = {0};
            const auto curr_blk = curr * blk;

            /* compute offsets and tail*/
            for (int ur = off != 0 ? 0 : 1; ur < reg_unroll; ++ur) {
                const int ur_c = curr_blk + ur;
                const int ur_p = (ur_c - 1 + 2 * blk) % (2 * blk); // prev ur
                const bool is_tail
                        = off + ur >= static_cast<int>(prb_.nodes[0].tail_size);
                step(off + ur, i_off[ur_p], o_off[ur_p], s_off[ur_p],
                        c_off[ur_p], i_off[ur_c], o_off[ur_c], s_off[ur_c],
                        c_off[ur_c]);
                if (tail_processing && is_tail) zero_padding[ur] = 1;
            }

            process_unroll_generic_step(reg_unroll, i_off + curr_blk,
                    o_off + curr_blk, s_off + curr_blk, c_off + curr_blk,
                    zero_padding, tail_processing);

            curr = 1 - curr;
        }
    }

    void compute_ker(
            const int ndims, const int len_unroll, const bool tail_processing) {
        bool optimized = false;
        optimized = optimized || process_direct_copy<sve_256>(ndims, len_unroll)
                || process_direct_copy<asimd>(ndims, len_unroll)
                || process_unroll_tr8x8(ndims, len_unroll);
        if (!optimized)
            process_unroll_generic(ndims, len_unroll, tail_processing);
    }

    void loop_begin(Label &l, XReg reg_cnt, int len) {
        mov(reg_cnt, len);
        L(l);
    }

    void check_if_this_is_last_chunk(const XReg reg_curr_chunk, int node_id) {
        // Chunks are backwards numered i.e:
        // [0] -> [node_size]
        // [1] -> [node_size - 1]
        // ...
        // [node_size - 1] -> [1]

        // It is done like this, because it is easier to decrement counter
        // and check if it is equal to zero than increment and check
        // if it is equal to node_size.
        static constexpr int64_t last_chunk = 1;
        cmp(reg_curr_chunk, last_chunk);
    }

    void zero_dst_memory(const int bytes_to_zeroing) {
        static constexpr int num_of_bytes_in_xmm = 128 / 8;

        const int xmms_to_zeroing
                = std::div(bytes_to_zeroing, num_of_bytes_in_xmm).quot;
        const int tail_to_zeroing
                = std::div(bytes_to_zeroing, num_of_bytes_in_xmm).rem;

        movi(xmm_tmp_, 0);

        if (xmms_to_zeroing > 0) {
            Label loop;

            mov(X_TMP_4, xmms_to_zeroing);
            L(loop);
            str(QReg(xmm_tmp_.getIdx()), ptr(o_addr(0)));
            add_imm(reg_off_out_, reg_off_out_, num_of_bytes_in_xmm, X_TMP_0);
            add_imm(x_ptr_out_off, x_ptr_out_off, num_of_bytes_in_xmm, X_TMP_0);
            subs(X_TMP_4, X_TMP_4, 1);
            b(NE, loop);
        }

        if (tail_to_zeroing) mov_imm(W_TMP_4, 0);
        for (int i = 0; i < tail_to_zeroing; i++)
            strb(W_TMP_4, ptr(o_addr(i, false)));

        // Restore dst offset to initial value
        if (xmms_to_zeroing > 0) {
            sub_imm(reg_off_out_, reg_off_out_,
                    num_of_bytes_in_xmm * xmms_to_zeroing, X_TMP_0);
            sub_imm(x_ptr_out_off, x_ptr_out_off,
                    num_of_bytes_in_xmm * xmms_to_zeroing, X_TMP_0);
        }
    }

    void finalize_tail_loop(int i_step, int o_step, int s_step, int c_step,
            const int curr_node_id) {
        static constexpr int empty_chunk_info = -1;

        mov(X_TMP_0, empty_chunk_info);
        str(X_TMP_0, ptr(data_chunk_addr(curr_node_id)));

        const int padded_area = prb_.nodes[curr_node_id].n
                - prb_.nodes[curr_node_id].tail_size;

        if (prb_.nodes[curr_node_id].is_zero_pad_needed) {
            int num_of_zero_padded_values = padded_area;
            for (int i = curr_node_id - 1; i >= 0; i--) {
                num_of_zero_padded_values *= prb_.nodes[i].n;
            }

            const int bytes_to_zeroing = num_of_zero_padded_values * otype_sz_;
            zero_dst_memory(bytes_to_zeroing);
        }

        // This function is called by loop_end. At the end
        // of loop_end is section that is responsible for
        // restoring offset values. Restoring is based on
        // len value which is equal to prb.nodes[x].n.
        // If fill_zero_padded_area is called then it means
        // offsets were shifted prb.nodes[x].tail_size times.
        // Therefore, this function has to shift offsets by
        // zero pad area.
        add_imm(reg_off_in_, reg_off_in_, padded_area * i_step * itype_sz_,
                X_TMP_0);
        add_imm(reg_off_out_, reg_off_out_, padded_area * o_step * otype_sz_,
                X_TMP_0);
        add_imm(x_ptr_in_off, x_ptr_in_off, padded_area * i_step * itype_sz_,
                X_TMP_0);
        add_imm(x_ptr_out_off, x_ptr_out_off, padded_area * o_step * otype_sz_,
                X_TMP_0);
        if (prb_.src_scale_type == scale_type_t::MANY)
            add_imm(x_ptr_src_scale_off, x_ptr_src_scale_off,
                    padded_area * s_step * stype_sz_, X_TMP_0);
        if (prb_.dst_scale_type == scale_type_t::MANY)
            add_imm(x_ptr_dst_scale_off, x_ptr_dst_scale_off,
                    padded_area * s_step * stype_sz_, X_TMP_0);

        if (compensation_needed_) {
            add_imm(reg_off_comp_, reg_off_comp_,
                    padded_area * c_step * sizeof(int32_t), X_TMP_0);
            add_imm(x_ptr_comp_off, x_ptr_comp_off,
                    padded_area * c_step * sizeof(int32_t), X_TMP_0);
        }
    }

    void loop_end(Label &l, XReg reg_cnt, int len, int i_step, int o_step,
            int s_step, int c_step, const int curr_node_id) {
        add_imm(reg_off_in_, reg_off_in_, i_step * itype_sz_, X_TMP_0);
        add_imm(reg_off_out_, reg_off_out_, o_step * otype_sz_, X_TMP_0);
        add_imm(x_ptr_in_off, x_ptr_in_off, i_step * itype_sz_, X_TMP_0);
        add_imm(x_ptr_out_off, x_ptr_out_off, o_step * otype_sz_, X_TMP_0);

        if (prb_.src_scale_type == scale_type_t::MANY)
            add_imm(x_ptr_src_scale_off, x_ptr_src_scale_off,
                    s_step * stype_sz_, X_TMP_0);
        if (prb_.dst_scale_type == scale_type_t::MANY)
            add_imm(x_ptr_dst_scale_off, x_ptr_dst_scale_off,
                    s_step * stype_sz_, X_TMP_0);

        if (compensation_needed_) {
            add_imm(reg_off_comp_, reg_off_comp_, c_step * sizeof(int32_t),
                    X_TMP_0);
            add_imm(x_ptr_comp_off, x_ptr_comp_off, c_step * sizeof(int32_t),
                    X_TMP_0);
        }

        subs(reg_cnt, reg_cnt, 1);
        b(NE, l);

        if (prb_.tail(curr_node_id) != 0) {
            Label if_end;

            // On the stack should be an information if node
            // was processed with tail or not.
            ldr(X_TMP_0, post_ptr(X_SP, X_TMP_0.getBit() / 8));

            cmp(X_TMP_0, with_tail_info_);
            b(NE, if_end);
            finalize_tail_loop(i_step, o_step, s_step, c_step, curr_node_id);
            L(if_end);
        }

        // Restore offset to initial values. It means before
        // loop execution.
        sub_imm(reg_off_in_, reg_off_in_, len * i_step * itype_sz_, X_TMP_0);
        sub_imm(reg_off_out_, reg_off_out_, len * o_step * otype_sz_, X_TMP_0);
        sub_imm(x_ptr_in_off, x_ptr_in_off, len * i_step * itype_sz_, X_TMP_0);
        sub_imm(x_ptr_out_off, x_ptr_out_off, len * o_step * otype_sz_,
                X_TMP_0);

        if (prb_.src_scale_type == scale_type_t::MANY)
            sub_imm(x_ptr_src_scale_off, x_ptr_src_scale_off,
                    len * s_step * stype_sz_, X_TMP_0);
        if (prb_.dst_scale_type == scale_type_t::MANY)
            sub_imm(x_ptr_dst_scale_off, x_ptr_dst_scale_off,
                    len * s_step * stype_sz_, X_TMP_0);
        if (compensation_needed_) {
            sub_imm(reg_off_comp_, reg_off_comp_,
                    len * c_step * sizeof(int32_t), X_TMP_0);
            sub_imm(x_ptr_comp_off, x_ptr_comp_off,
                    len * c_step * sizeof(int32_t), X_TMP_0);
        }
    }

    void compute_blk_ker(const simple_impl_desc_t &desc) {
        static constexpr bool with_tail_processing = true;
        Label no_last_chunk, end_label;
        int omp_ndims = prb_.full_ndims - prb_.ndims;

        if (prb_.nodes[0].tail_size > 0) {
            if (!prb_.nodes[0].is_parent_empty()) {
                const int parent_node_id = prb_.nodes[0].parent_node_id;
                ldr(X_TMP_0, ptr(data_chunk_addr(parent_node_id)));
                check_if_this_is_last_chunk(X_TMP_0, parent_node_id);
                b(NE, no_last_chunk);
            }

            const int len_unroll = desc.tail_len_unroll > 0
                    ? desc.tail_len_unroll
                    : desc.len_unroll;
            compute_ker(omp_ndims, len_unroll, with_tail_processing);
            b(end_label);
        }

        L(no_last_chunk);
        compute_ker(omp_ndims, desc.len_unroll, !with_tail_processing);
        L(end_label);
    }

    void create_loops(const simple_impl_desc_t &desc,
            const std::array<const XReg, 3> &reg_cnt, int jit_loop) {
        assert(jit_loop <= ndims_jit_loop_max);

        if (jit_loop > 0) {
            const int nfu = desc.ndims_full_unroll;
            const int unroll_factor
                    = jit_loop == 1 ? desc.len_last_dim_unroll : 1;
            const int curr_node_id = nfu + (jit_loop - 1);
            const int parent_node_id = prb_.nodes[curr_node_id].parent_node_id;
            const int tail_size = prb_.tail(curr_node_id) / unroll_factor;
            const int node_size = prb_.n(curr_node_id) / unroll_factor;
            const XReg reg_loop_cnt = reg_cnt[jit_loop - 1];
            const bool curr_node_has_tail = prb_.tail(curr_node_id) != 0;
            Label loop, if_no_tail, if_end;

            if (curr_node_has_tail) {
                const size_t reg_bytes = X_TMP_0.getBit() / 8;
                if (prb_.nodes[curr_node_id].is_parent_empty()) {
                    mov(reg_loop_cnt, tail_size);
                    // Put info that node is being processed with tail.
                    mov(X_TMP_0, with_tail_info_);
                    str(X_TMP_0, pre_ptr(X_SP, -reg_bytes));
                } else {
                    ldr(X_TMP_0, ptr(data_chunk_addr(parent_node_id)));
                    check_if_this_is_last_chunk(X_TMP_0, parent_node_id);
                    b(NE, if_no_tail);
                    mov(reg_loop_cnt, tail_size);
                    // Put info that node is being processed with tail.
                    mov(X_TMP_0, with_tail_info_);
                    str(X_TMP_0, pre_ptr(X_SP, -reg_bytes));
                    b(if_end);

                    L(if_no_tail);
                    mov(reg_loop_cnt, node_size);
                    // Put info that node is being processed without tail.
                    mov(X_TMP_0, without_tail_info_);
                    str(X_TMP_0, pre_ptr(X_SP, -reg_bytes));
                    L(if_end);
                }
            }

            if (prb_.is_tail_in_one_of_child_nodes(curr_node_id)) {
                if (!curr_node_has_tail) {
                    mov(reg_loop_cnt, node_size);
                    str(reg_loop_cnt, ptr(data_chunk_addr(curr_node_id)));
                }
                L(loop);
                if (!prb_.nodes[curr_node_id].is_parent_empty()) {
                    Label if_no_tail_in_child_node;
                    ldr(X_TMP_0, ptr(data_chunk_addr(parent_node_id)));
                    check_if_this_is_last_chunk(X_TMP_0, parent_node_id);
                    b(NE, if_no_tail_in_child_node);
                    str(reg_loop_cnt, ptr(data_chunk_addr(curr_node_id)));
                    L(if_no_tail_in_child_node);
                } else {
                    str(reg_loop_cnt, ptr(data_chunk_addr(curr_node_id)));
                }
            } else if (curr_node_has_tail) {
                L(loop);
            } else {
                loop_begin(loop, reg_loop_cnt, node_size);
            }

            create_loops(desc, reg_cnt, jit_loop - 1);

            loop_end(loop, reg_loop_cnt, node_size,
                    prb_.is(curr_node_id) * unroll_factor,
                    prb_.os(curr_node_id) * unroll_factor,
                    prb_.ss(curr_node_id) * unroll_factor,
                    prb_.cs(curr_node_id) * unroll_factor, curr_node_id);
        } else {
            compute_blk_ker(desc);
        }
    }

    bool simple_impl() {
        simple_impl_desc_t d;
        if (!simple_impl_desc_init(prb_, &d)) return false;

        eor(reg_off_in_, reg_off_in_, reg_off_in_);
        eor(reg_off_out_, reg_off_out_, reg_off_out_);

        if (prb_.src_scale_type == scale_type_t::MANY)
            mov(x_ptr_src_scale_off, reg_ptr_src_scales_);
        if (prb_.dst_scale_type == scale_type_t::MANY)
            mov(x_ptr_dst_scale_off, reg_ptr_dst_scales_);

        if (compensation_needed_)
            eor(reg_off_comp_, reg_off_comp_, reg_off_comp_);

        std::array<const XReg, 3> reg_cnt({{x15, x14, x13}});

        const int n_jit_loops = prb_.ndims - d.ndims_full_unroll;
        create_loops(d, reg_cnt, n_jit_loops);

        return true;
    }

    void impl() {
        if (simple_impl()) return;
        assert(!"no implementation available");
    }

#define UNROLL_INST(inst, reg, ...) \
    for (size_t i = startIdx; i < startIdx + regNum; i++) { \
        reg tmp(i); \
        inst(__VA_ARGS__); \
    }
#define UNROLL_INST2(inst, ...) \
    for (size_t i = startIdx; i < startIdx + regNum; i++) \
        inst(__VA_ARGS__);

    void cvt_z_s32_f32(const size_t startIdx, const size_t regNum) {
        UNROLL_INST(scvtf, ZRegS, tmp, P_ALL_ONE / T_m, tmp);
    }

    void cvt_v_s32_f32(const size_t startIdx, const size_t regNum) {
        UNROLL_INST(scvtf, VReg4S, tmp, tmp);
    }

    void cvt_z_f32_s32(const size_t startIdx, const size_t regNum) {
        UNROLL_INST(frinti, ZRegS, tmp, P_ALL_ONE / T_m, tmp);
        UNROLL_INST(fcvtzs, ZRegS, tmp, P_ALL_ONE / T_m, tmp);
    }

    void cvt_v_f32_s32(const size_t startIdx, const size_t regNum) {
        UNROLL_INST(frinti, VReg4S, tmp, tmp);
        UNROLL_INST(fcvtzs, VReg4S, tmp, tmp);
    }

    void cvt_v_f32_bf16(const size_t startIdx, const size_t regNum) {
        UNROLL_INST2(bfcvtn, VReg4H(i), VReg4S(i));
    }

    void cvt_z_s8_s32(const size_t startIdx, const size_t regNum) {
        cvt_z_b_s(startIdx, regNum);
        UNROLL_INST(sxtb, ZRegS, tmp, P_ALL_ONE / T_m, tmp);
    }

    void cvt_v_s8_s32(const size_t startIdx, const size_t regNum) {
        UNROLL_INST(sxtl, VReg, tmp.h8, tmp.b8);
        UNROLL_INST(sxtl, VReg, tmp.s4, tmp.h4);
    }

    void cvt_z_s8_f32(const size_t startIdx, const size_t regNum) {
        cvt_z_b_s(startIdx, regNum);
        cvt_z_s32_f32(startIdx, regNum);
    }

    void cvt_v_s8_f32(const size_t startIdx, const size_t regNum) {
        cvt_v_b_s(startIdx, regNum);
        cvt_v_s32_f32(startIdx, regNum);
    }

    void cvt_z_b_s(const size_t startIdx, const size_t regNum) {
        assert(z_tmp7.getIdx() < startIdx
                || startIdx + regNum - 1 < z_tmp7.getIdx());

        dup(z_tmp7.b, 0);
        UNROLL_INST(zip1, ZRegB, tmp, tmp, z_tmp7.b);
        UNROLL_INST(zip1, ZRegH, tmp, tmp, z_tmp7.h);
    }

    void cvt_v_b_s(const size_t startIdx, const size_t regNum) {
        assert(v_tmp7.getIdx() < startIdx
                || startIdx + regNum - 1 < v_tmp7.getIdx());

        mov_imm(W_TMP_0, 0);
        dup(v_tmp7.b16, W_TMP_0);
        UNROLL_INST(zip1, VReg16B, tmp, tmp, v_tmp7.b16);
        UNROLL_INST(zip1, VReg8H, tmp, tmp, v_tmp7.h8);
    }

    void cvt_z_u8_s32(const size_t startIdx, const size_t regNum) {
        cvt_z_b_s(startIdx, regNum);
        UNROLL_INST(uxtb, ZRegS, tmp, P_ALL_ONE / T_m, tmp);
    }

    void cvt_v_u8_s32(const size_t startIdx, const size_t regNum) {
        UNROLL_INST(uxtl, VReg, tmp.h8, tmp.b8);
        UNROLL_INST(uxtl, VReg, tmp.s4, tmp.h4);
    }

    void cvt_z_s32_s8(const size_t startIdx, const size_t regNum) {
        assert(z_tmp7.getIdx() < startIdx
                || startIdx + regNum - 1 < z_tmp7.getIdx());

        dup(z_tmp7.s, 0);
        UNROLL_INST2(smin, ZRegS(i), 127);
        UNROLL_INST2(smax, ZRegS(i), -128);
        UNROLL_INST(uzp1, ZRegH, tmp, tmp, z_tmp7.h);
        UNROLL_INST(uzp1, ZRegB, tmp, tmp, z_tmp7.b);
    }

    void cvt_v_s32_s8(const size_t startIdx, const size_t regNum) {
        assert(v_tmp7.getIdx() < startIdx
                || startIdx + regNum - 1 < v_tmp7.getIdx());

        mov_imm(W_TMP_0, 127);
        dup(v_tmp7.s4, W_TMP_0);
        mov_imm(W_TMP_0, -128);
        UNROLL_INST2(smin, VReg4S(i), VReg4S(i), v_tmp7.s4);
        dup(v_tmp7.s4, W_TMP_0);
        UNROLL_INST2(smax, VReg4S(i), VReg4S(i), v_tmp7.s4);
        mov_imm(W_TMP_0, 0);
        dup(v_tmp7.s4, W_TMP_0);
        UNROLL_INST(uzp1, VReg8H, tmp, tmp, v_tmp7.h8);
        UNROLL_INST(uzp1, VReg16B, tmp, tmp, v_tmp7.b16);
    }

    void cvt_z_u8_s8(const size_t startIdx, const size_t regNum) {
        UNROLL_INST2(umin, ZRegB(i), 127);
    }

    void cvt_v_u8_s8(const size_t startIdx, const size_t regNum) {
        assert(v_tmp7.getIdx() < startIdx
                || startIdx + regNum - 1 < v_tmp7.getIdx());

        mov_imm(W_TMP_0, 127);
        dup(v_tmp7.b16, W_TMP_0);
        UNROLL_INST(umin, VReg16B, tmp, tmp, v_tmp7.b16);
    }

    void cvt_z_u32_u8(const size_t startIdx, const size_t regNum) {
        UNROLL_INST2(umin, ZRegS(i), 255);
        UNROLL_INST(uzp1, ZRegH, tmp, tmp, tmp);
        UNROLL_INST(uzp1, ZRegB, tmp, tmp, tmp);
    }

    void cvt_v_u32_u8(const size_t startIdx, const size_t regNum) {
        assert(v_tmp7.getIdx() < startIdx
                || startIdx + regNum - 1 < v_tmp7.getIdx());

        mov_imm(W_TMP_0, 255);
        dup(v_tmp7.s4, W_TMP_0);
        UNROLL_INST(umin, VReg4S, tmp, tmp, v_tmp7.s4);
        UNROLL_INST(uzp1, VReg8H, tmp, tmp, tmp);
        UNROLL_INST(uzp1, VReg16B, tmp, tmp, tmp);
    }

    void cvt_z_s32_u8(const size_t startIdx, const size_t regNum) {
        assert(z_tmp7.getIdx() < startIdx
                || startIdx + regNum - 1 < z_tmp7.getIdx());

        dupm(z_tmp7.s, 255);
        UNROLL_INST2(smax, ZRegS(i), 0);
        UNROLL_INST2(smin, ZRegS(i), P_ALL_ONE / T_m, z_tmp7.s);
        UNROLL_INST(uzp1, ZRegH, tmp, tmp, tmp);
        UNROLL_INST(uzp1, ZRegB, tmp, tmp, tmp);
        UNROLL_INST2(mov, ZRegB(i), P_NOT_128 / T_m, 0);
    }

    void cvt_v_s32_u8(const size_t startIdx, const size_t regNum) {
        assert(v_tmp7.getIdx() < startIdx
                || startIdx + regNum - 1 < v_tmp7.getIdx());

        mov_imm(W_TMP_0, 0);
        dup(v_tmp7.s4, W_TMP_0);
        mov_imm(W_TMP_0, 255);
        UNROLL_INST(smax, VReg4S, tmp, tmp, v_tmp7.s4);
        dup(v_tmp7.s4, W_TMP_0);
        UNROLL_INST(smin, VReg4S, tmp, tmp, v_tmp7.s4);
        UNROLL_INST(uzp1, VReg8H, tmp, tmp, tmp);
        UNROLL_INST(uzp1, VReg16B, tmp, tmp, tmp);
    }

    void cvt_z_s8_u8(const size_t startIdx, const size_t regNum) {
        UNROLL_INST2(smax, ZRegB(i), 0);
    }

    void cvt_v_s8_u8(const size_t startIdx, const size_t regNum) {
        assert(v_tmp7.getIdx() < startIdx
                || startIdx + regNum - 1 < v_tmp7.getIdx());

        mov_imm(W_TMP_0, 0);
        dup(v_tmp7.b16, W_TMP_0);
        UNROLL_INST(smax, VReg16B, tmp, tmp, v_tmp7.b16);
    }
#undef UNROLL_INST
#undef UNROLL_INST

    jit_uni_reorder_kernel_f32_t(const desc_t &desc)
        : kernel_t(desc), isa_(get_max_cpu_isa()) {
        assert(!utils::one_of(isa_, isa_undef, isa_all));
        itype_sz_ = data_type_size(prb_.itype);
        otype_sz_ = data_type_size(prb_.otype);
        stype_sz_ = sizeof(float);
    }

    void generate() override {
        using namespace Xbyak_aarch64::util;
        uint64_t sveLen = get_sve_length();
        Label end_of_kernel;

        preamble();

        if (prb_.src_scale_type == scale_type_t::COMMON) {
            add_imm(X_DEFAULT_ADDR, PARAM(src_scales), X_TMP_1);
            ldr(X_TMP_0, ptr(X_DEFAULT_ADDR));
            ld1r(xmm_src_scales_, ptr(X_TMP_0));
        } else if (prb_.src_scale_type == scale_type_t::MANY) {
            add_imm(X_DEFAULT_ADDR, PARAM(src_scales), X_TMP_0);
            ldr(reg_ptr_src_scales_, ptr(X_DEFAULT_ADDR));
        }

        if (prb_.dst_scale_type == scale_type_t::COMMON) {
            add_imm(X_DEFAULT_ADDR, PARAM(dst_scales), X_TMP_1);
            ldr(X_TMP_0, ptr(X_DEFAULT_ADDR));
            ld1r(xmm_dst_scales_, ptr(X_TMP_0));
        } else if (prb_.dst_scale_type == scale_type_t::MANY) {
            add_imm(X_DEFAULT_ADDR, PARAM(dst_scales), X_TMP_0);
            ldr(reg_ptr_dst_scales_, ptr(X_DEFAULT_ADDR));
        }

        if (compensation_needed_) {
            add_imm(X_DEFAULT_ADDR, PARAM(compensation_scratch), X_TMP_0);
            ldr(reg_ptr_comp_, ptr(X_DEFAULT_ADDR));
        }
        if (prb_.scale_adjust == 0.5f) { mov(reg_scale_adjust_, 0x3f000000); }
        add_imm(X_TMP_0, PARAM(in), X_TMP_2);
        add_imm(X_TMP_1, PARAM(out), X_TMP_2);
        ldr(reg_ptr_in_, ptr(X_TMP_0));
        ldr(reg_ptr_out_, ptr(X_TMP_1));

        if (sveLen) { /* SVE is available. */
            ptrue(p_lsb_256.b, VL32);
            ptrue(p_lsb_128.b, VL16);
            ptrue(p_lsb_64.b, VL8);
        }

        bool is_tail_in_drv_dims = false;
        for (int i = prb_.ndims; i < prb_.full_ndims; i++)
            if (prb_.nodes[i].tail_size > 0) {
                is_tail_in_drv_dims = true;
                break;
            }

        if (is_tail_in_drv_dims) {
            Label reorder_kernel;
            add_imm(X_DEFAULT_ADDR, TAIL_PARAM(skip_kernel_execution), X_TMP_0);
            ldr(X_TMP_0, ptr(X_DEFAULT_ADDR));
            cmp(X_TMP_0, static_cast<int64_t>(true));
            b(EQ, end_of_kernel);

            add_imm(X_DEFAULT_ADDR, TAIL_PARAM(zeroing_data), X_TMP_0);
            ldr(X_TMP_0, ptr(X_DEFAULT_ADDR));
            cmp(X_TMP_0, static_cast<int64_t>(false));
            b(EQ, reorder_kernel);
            // If zeroing data is set then all dst memory
            // will be zeroed and nothing more will be done.
            int bytes_to_zeroing = otype_sz_;
            for (int i = 0; i < prb_.ndims; i++) {
                bytes_to_zeroing *= prb_.nodes[i].n;
            }
            eor(reg_off_out_, reg_off_out_, reg_off_out_);
            mov(x_ptr_out_off, reg_ptr_out_);
            zero_dst_memory(bytes_to_zeroing);
            b(end_of_kernel);
            L(reorder_kernel);
        }

        if (can_do_tr8x8()) {
            dup(ymm_zero_, 0);
        } else {
            movi(xmm_zero_, 0);
        }

        impl();

        L(end_of_kernel);
        postamble();
    }

    ~jit_uni_reorder_kernel_f32_t() override = default;

#undef TAIL_PARAM
#undef PARAM

private:
    static constexpr int64_t with_tail_info_ = static_cast<int64_t>(true);
    static constexpr int64_t without_tail_info_ = static_cast<int64_t>(false);

    int itype_sz_;
    int otype_sz_;
    int stype_sz_;

    const cpu_isa_t isa_;

    const XReg reg_ptr_in_ = x6;
    const XReg reg_ptr_out_ = x2;
    const XReg reg_ptr_src_scales_ = x1;
    const XReg reg_ptr_dst_scales_ = x12;
    const XReg reg_ptr_comp_ = x3;
    const WReg reg_scale_adjust_ = w5;

    const XReg reg_off_in_ = x8;
    const XReg reg_off_out_ = x9;
    const XReg reg_off_comp_ = x11;

    /* X_TMP is required to set address to
     x_tmp_vec(X_TMP_0 - X_TMP_4). */
    XReg X_TMP = x20;

    VReg4S xmm_src_scales_ = v15.s;
    VReg4S xmm_dst_scales_ = v11.s;
    VReg4S xmm_zero_ = v14.s;
    ZRegS ymm_zero_ = z14.s;
    VReg4S xmm_tmp_ = v12.s;
    const VReg4S xmm_src_zp_ = v9.s;
    const VReg4S xmm_dst_zp_ = v10.s;
    const VReg4S xmm_compensation = v8.s;
    VReg4S xmm_saturation_ubound_ = v12.s;
    ZRegS ymm_saturation_ubound_ = z12.s;

    /* Note: x22 - x28 are already used as temporal registgers
       in jit_generator.hpp.
       x_ptr_(in|out|scale|comp)_off keeps (base + offset) address. */
    XReg x_ptr_in_off = reg_ptr_in_;
    XReg x_ptr_out_off = reg_ptr_out_;
    XReg x_ptr_comp_off = reg_ptr_comp_;
    XReg x_ptr_src_scale_off = x19;
    XReg x_ptr_dst_scale_off = x29;

    /* Caution: Chose predicate registers not used by x64's implementation. */
    PReg p_lsb_256 = p7;
    PReg p_lsb_128 = p6;
    PReg p_lsb_64 = p4;
    PReg p_tmp0 = p5;

    const std::vector<uint32_t> tmp_vec_idx = {20, 21, 22, 23, 24, 25, 26, 27};
    VReg v_tmp0 = v20;
    ZReg z_tmp0 = z20;
    ZReg z_tmp1 = z21;
    ZReg z_tmp2 = z22;
    ZReg z_tmp3 = z23;
    ZReg z_tmp4 = z24;
    ZReg z_tmp5 = z25;
    ZReg z_tmp6 = z26;
    ZReg z_tmp7 = z27;
    VReg v_tmp7 = v27;

    const std::vector<ZReg> z_tmp_vec
            = {z_tmp0, z_tmp1, z_tmp2, z_tmp3, z_tmp4, z_tmp5, z_tmp6, z_tmp7};
    constexpr static int z_tmp_vec_size = 8;
};

// Seperate class for no unroll/threading burden
struct jit_single_blk_kernel_t : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_single_blk_kernel)
    static bool applicable(const prb_t &p) {
        using namespace data_type;

        bool ok = p.ndims >= 2 && mayiuse(sve_256)
                && p.src_scale_type == scale_type_t::NONE
                && p.dst_scale_type == scale_type_t::NONE
                && utils::one_of(p.itype, f32) && utils::one_of(p.otype, f32)
                && utils::everyone_is(0, p.ioff, p.ooff) && p.beta == 0.f
                && prb_has_small_strides(p);
        if (!ok) return false;

        int64_t n0 = p.nodes[0].n;
        auto i0 = p.nodes[0].is;
        auto o0 = p.nodes[0].os;
        int64_t n1 = p.nodes[1].n;
        auto i1 = p.nodes[1].is;
        auto o1 = p.nodes[1].os;

        /*
         * for a transpose of plain to 8c case, nodes would be like:
         *     n    is   os
         *     m    1    8
         *     8    m    1
         * or
         *     8    m    1
         *     m    1    8
         */
        ok = (utils::one_of(n0, 8, 16) || utils::one_of(n1, 8, 16))
                && ((i0 == 1 && o1 == 1 && n0 == i1 && o0 == n1)
                        || (o0 == 1 && i1 == 1 && n0 == o1 && i0 == n1));
        if (!ok) return false;

        // Do not handle transpose of dimensions other than last 2
        for (int i = 2; i < p.ndims; ++i) {
            if (p.nodes[i].is != p.nodes[i].os) {
                ok = false;
                break;
            }
        }

        return ok;
    }

    jit_single_blk_kernel_t(const tr::prb_t &prb)
        : jit_generator()
        , prb_(prb)
        , itype_sz_(data_type_size(prb_.itype))
        , otype_sz_(data_type_size(prb_.otype))
        , block_sz(prb.nodes[0].n) {}

    void generate() override {
        auto input_stride
                = prb_.nodes[0].is != 1 ? prb_.nodes[0].is : prb_.nodes[1].is;
        auto output_stride
                = prb_.nodes[0].os != 1 ? prb_.nodes[0].os : prb_.nodes[1].os;

        Label tail_processing;

        const auto load_zp = [&](const ZRegS ymm_zp, const XReg reg_zp) {
            dup(ymm_zp, WReg(reg_zp.getIdx()));
            scvtf(ymm_zp, P_ALL_ONE / T_m, ymm_zp);
        };

        preamble();

        if (prb_.req_src_zp) load_zp(ymm_src_zp, reg_src_zp);

        if (prb_.req_dst_zp) load_zp(ymm_dst_zp, reg_dst_zp);

        cmp(reg_ptr_tail, true);
        b(EQ, tail_processing);

        if (block_sz == 8) {
            gen_ker8x8(0, 0, input_stride, output_stride, 8, 8);
            block_sz = 8;
        } else if (block_sz == 16) {
            gen_ker16x16_in_8x8(input_stride, output_stride);
            block_sz = 16;
        } else {
            assert(!"unimplemented");
        }

        postamble();

        L(tail_processing);

        if (block_sz == 8) {
            auto i_tail = input_stride % 8 != 0 ? input_stride % 8 : 8;
            auto o_tail = output_stride % 8 != 0 ? output_stride % 8 : 8;
            if (i_tail != o_tail) {
                auto t_mask = i_tail == 8 ? o_tail : i_tail;
                gen_setmask(t_mask);
                gen_ker8x8(0, 0, input_stride, output_stride, i_tail, o_tail);
            }
        } else if (block_sz == 16) {
            auto i_tail = input_stride % 16 != 0 ? input_stride % 16 : 16;
            auto o_tail = output_stride % 16 != 0 ? output_stride % 16 : 16;
            if (i_tail != o_tail) {
                auto t_mask = i_tail == 16 ? o_tail : i_tail;
                t_mask %= 8;
                if (t_mask != 0) gen_setmask(t_mask);
                gen_ker16x16_in_8x8(
                        input_stride, output_stride, i_tail, o_tail);
            }
        } else {
            assert(!"unimplemented");
        }

        postamble();
    }

    void gen_loadu(const ZRegS ymm, const XReg &addr, int size) {
        QReg xmm(ymm.getIdx());
        switch (size) {
            case 32: ld1w(ymm, p_lsb_256 / T_z, ptr(addr)); break;
            case 16: ldr(xmm, ptr(addr)); break;
            default: assert(!"unreachable");
        }
    }

    void gen_storeu(const XReg &addr, const ZRegS ymm, int size) {
        QReg xmm(ymm.getIdx());
        switch (size) {
            case 32: st1w(ymm, p_lsb_256, ptr(addr)); break;
            case 16: str(xmm, ptr(addr)); break;
            default: assert(!"unreachable");
        }
    }

    void gen_maskloadu(
            const ZRegS ymm, const XReg &addr, const PReg mask, int size) {
        switch (size) {
            case 32:
            case 16: ld1w(ymm, mask / T_z, ptr(addr)); break;
            default: assert(!"unreachable");
        }
    }

    void gen_maskstoreu(
            const XReg &addr, const ZRegS ymm, const PReg mask, int size) {
        switch (size) {
            case 32:
            case 16: st1w(ymm, mask, ptr(addr)); break;
            default: assert(!"unreachable");
        }
    }

    // Register allocation xmm0~11
    void gen_transpose_8x8() {
        const uint64_t sveLen = get_sve_length();
        constexpr int lane = 8;

#if 0
        /* Debug code
	   z0:   7,  6,  5,  4,  3,  2,  1,  0
	   z1:  15, 14, 13, 12, 11, 10,  9,  8
	   ...
	   z17: 63, 62, 61, 60, 59, 58, 57, 56
	*/
	ptrue(P_ALL_ONE.b);
	ptrue(P_TMP.s, VL8);
	not_(P_TMP.b, P_ALL_ONE/T_z, P_TMP.b);
        index(z0.s, 0, 1);
        mov(z0.s, P_TMP/T_m, 0);
        mov(z_tmp_vec[0].s, 8);
        mov(z_tmp_vec[0].s, P_TMP/T_m, 0);
        for(uint32_t i=1; i<lane; i++)
          add(ZRegS{i}, ZRegS{i-1}, z_tmp_vec[0].s);
#endif

        ptrue(P_TMP.s, VL4);

        /* 1st turn */
        for (uint32_t i = 0; i < lane / 2; i++) {
            trn1(z_tmp_vec[i].s, ZRegS {2 * i}, ZRegS {2 * i + 1});
            trn2(z_tmp_vec[lane / 2 + i].s, ZRegS {2 * i}, ZRegS {2 * i + 1});
        }

        /* 2nd turn */
        trn1(z4.d, z_tmp_vec[0].d, z_tmp_vec[1].d);
        trn1(z5.d, z_tmp_vec[4].d, z_tmp_vec[5].d);
        trn2(z6.d, z_tmp_vec[0].d, z_tmp_vec[1].d);
        trn2(z7.d, z_tmp_vec[4].d, z_tmp_vec[5].d);
        trn1(z_tmp_vec[0].d, z_tmp_vec[2].d, z_tmp_vec[3].d);
        trn1(z_tmp_vec[1].d, z_tmp_vec[6].d, z_tmp_vec[7].d);
        trn2(z_tmp_vec[2].d, z_tmp_vec[2].d, z_tmp_vec[3].d);
        trn2(z_tmp_vec[3].d, z_tmp_vec[6].d, z_tmp_vec[7].d);

        /* 3rd turn */
        for (uint32_t i = 0; i < lane / 2; i++) {
            mov(ZRegD {i}, ZRegD {lane / 2 + i});
            mov(z_tmp_vec[lane / 2 + i].d, z_tmp_vec[i].d);
        }

        /* 4th turn */
        for (uint32_t i = 0; i < lane / 2; i++) {
            ZRegB z {lane / 2 + i};
            ZRegB z_tmp = z_tmp_vec[lane / 2 + i].b;
            /* Move bit 0-127 to 128-255. */
            ext(z, z, 16);
            /* Move bit 128-255 to 0-127. */
            ext(z_tmp, z_tmp, sveLen - 16);
        }

        /* 5th turn */
        for (uint32_t i = 0; i < lane / 2; i++) {
            ZRegS z0 {i};
            ZRegS z1 {lane / 2 + i};
            sel(z0, P_TMP, z0, z_tmp_vec[lane / 2 + i].s);
            sel(z1, P_TMP, z1, z_tmp_vec[i].s);
        }
    }

    // keep order nchw -> nChw()C
    // or nChw()C -> nchw
    void gen_setmask(int mask) { set_preg(p_mask.s, mask, x_tmp_0, x_tmp_1); }

    // TODO: Mark parameter with type information
    // XXX: !
    // offset in byte offset
    // stride in element number
    //
    // Gen specific 8x8 transform respect to certain tail condition
    void gen_tr8x8(int i_off, int o_off, int input_stride, int output_stride,
            int in_tail, int out_tail) {
        constexpr int lane = 8;

        if (in_tail == 0 || out_tail == 0) return;

        for (int i = 0; i < out_tail; ++i) {
            if (in_tail != lane) {
                add_imm(x_addr, reg_ptr_in_,
                        i_off + i * input_stride * itype_sz_, x_tmp_0);
                gen_maskloadu(ZRegS(i), x_addr, p_mask, lane * itype_sz_);
            } else {
                add_imm(x_addr, reg_ptr_in_,
                        i_off + i * input_stride * itype_sz_, x_tmp_0);
                gen_loadu(ZRegS(i), x_addr, lane * itype_sz_);
            }
            if (prb_.req_src_zp) { fsub(ZRegS(i), ZRegS(i), ymm_src_zp); }
        }

        gen_transpose_8x8();

        for (int i = 0; i < in_tail; ++i) {
            if (prb_.req_dst_zp) { fadd(ZRegS(i), ZRegS(i), ymm_dst_zp); }
            if (out_tail == lane) {
                add_imm(x_addr, reg_ptr_out_,
                        o_off + i * output_stride * otype_sz_, x_tmp_0);
                gen_storeu(x_addr, ZRegS(i), lane * otype_sz_);
            } else {
                add_imm(x_addr, reg_ptr_out_,
                        o_off + i * output_stride * otype_sz_, x_tmp_0);
                gen_maskstoreu(x_addr, ZRegS(i), p_mask, lane * otype_sz_);
            }
        }
    }

    // tail: 0 ~ 8
    // support: either in_tail or out_tail is not 8, but not both
    void gen_ker8x8(int i_off, int o_off, int input_stride, int output_stride,
            int in_tail, int out_tail) {
        gen_tr8x8(i_off, o_off, input_stride, output_stride, in_tail, out_tail);
    }

    void gen_ker16x16_in_8x8(int input_stride, int output_stride) {
        const auto lane = 16;
        const auto sub_lane = lane / 2;
        gen_tr8x8(0, 0, input_stride, output_stride, sub_lane, sub_lane);
        gen_tr8x8(input_stride * sub_lane * itype_sz_, sub_lane * otype_sz_,
                input_stride, output_stride, sub_lane, sub_lane);
        gen_tr8x8(sub_lane * itype_sz_, output_stride * sub_lane * otype_sz_,
                input_stride, output_stride, sub_lane, sub_lane);
        gen_tr8x8((input_stride * sub_lane + sub_lane) * itype_sz_,
                (output_stride * sub_lane + sub_lane) * otype_sz_, input_stride,
                output_stride, sub_lane, sub_lane);
    }

    // tail can be 1 ~ 16, using avx2 for now
    void gen_ker16x16_in_8x8(
            int input_stride, int output_stride, int in_tail, int out_tail) {
        constexpr auto lane = 16;
        constexpr auto sub_lane = lane / 2;
        auto tail = in_tail != lane ? in_tail : out_tail;

        const auto l_tail = tail < sub_lane ? tail : sub_lane;
        const auto u_tail = tail < sub_lane ? 0 : tail - sub_lane;

        if (tail == in_tail) {
            gen_tr8x8(0, 0, input_stride, output_stride, l_tail, sub_lane);
            gen_tr8x8(input_stride * sub_lane * itype_sz_, sub_lane * otype_sz_,
                    input_stride, output_stride, l_tail, sub_lane);
            gen_tr8x8(sub_lane * itype_sz_,
                    output_stride * sub_lane * otype_sz_, input_stride,
                    output_stride, u_tail, sub_lane);
            gen_tr8x8(itype_sz_ * (input_stride * sub_lane + sub_lane),
                    otype_sz_ * (output_stride * sub_lane + sub_lane),
                    input_stride, output_stride, u_tail, sub_lane);
        } else {
            gen_tr8x8(0, 0, input_stride, output_stride, sub_lane, l_tail);
            gen_tr8x8(input_stride * sub_lane * itype_sz_, sub_lane * otype_sz_,
                    input_stride, output_stride, sub_lane, u_tail);
            gen_tr8x8(sub_lane * itype_sz_,
                    output_stride * sub_lane * itype_sz_, input_stride,
                    output_stride, sub_lane, l_tail);
            gen_tr8x8(itype_sz_ * (input_stride * sub_lane + sub_lane),
                    otype_sz_ * (output_stride * sub_lane + sub_lane),
                    input_stride, output_stride, sub_lane, u_tail);
        }
    }

private:
    // 6 ~ 12
    constexpr static int xmm_save_start_from = 6;
    constexpr static int xmm_width = 16;

    void preamble() { ptrue(p_lsb_256.b, VL32); }

    void postamble() { ret(); }

    const prb_t &prb_;

    int itype_sz_;
    int otype_sz_;
    int block_sz;

    XReg reg_ptr_in_ = abi_param1;
    XReg reg_ptr_out_ = abi_param2;
    XReg reg_ptr_tail = abi_param3;
    XReg reg_src_zp = abi_param4;
    XReg reg_dst_zp = abi_param5;

    /* Because the callee-saved registers are not restored blk_reorder,
     the temporary registers (x9-x15) must be assigned.
     Must be selected from the temporary registers (x9-x15). */
    XReg x_addr = x10;
    XReg x_tmp_0 = x11;
    XReg x_tmp_1 = x12;

    /* Avoid P_TMP(p7) in jit_generator.hpp. */
    PReg p_lsb_256 = p6;
    PReg p_mask = p5;

    ZRegS ymm_tmp = z0.s;
    ZRegS ymm_src_zp = z14.s;
    ZRegS ymm_dst_zp = z15.s;

    const std::vector<uint32_t> tmp_vec_idx = {20, 21, 22, 23, 24, 25, 26, 27};
    VReg v_tmp0 = v20;
    ZReg z_tmp0 = z20;
    ZReg z_tmp1 = z21;
    ZReg z_tmp2 = z22;
    ZReg z_tmp3 = z23;
    ZReg z_tmp4 = z24;
    ZReg z_tmp5 = z25;
    ZReg z_tmp6 = z26;
    ZReg z_tmp7 = z27;
    VReg v_tmp7 = v27;

    const std::vector<ZReg> z_tmp_vec
            = {z_tmp0, z_tmp1, z_tmp2, z_tmp3, z_tmp4, z_tmp5, z_tmp6, z_tmp7};
    constexpr static int z_tmp_vec_size = 8;
};

status_t kernel_t::desc_init(
        kernel_t::desc_t &desc, const prb_t &prb, int ndims_ker_max) {
    desc.prb = prb;
    desc.prb.ioff = desc.prb.ooff = 0;

    if (ndims_ker_max > prb.ndims) return status::invalid_arguments;

    auto ndims_ker_max_f = [&]() {
        size_t cur_size = 1;
        for (int d = 0; d < prb.ndims; cur_size *= prb.nodes[d++].n)
            if (cur_size >= ker_prb_size_min) return d;
        return prb.ndims;
    };

    if (ndims_ker_max <= 0) ndims_ker_max = ndims_ker_max_f();

    /* traverse through kernel implementations */
    /* TODO: find a better way to do that... */
    desc.id = 0;
    for (int ndims_ker = ndims_ker_max; ndims_ker > 0; --ndims_ker) {
        desc.prb.ndims = ndims_ker;
        if (jit_uni_reorder_kernel_f32_t::applicable(desc.prb))
            return status::success;
    }

    return status::unimplemented;
}

kernel_t *kernel_t::create(const kernel_t::desc_t &desc) {
    switch (desc.id) {
        case 0: return new jit_uni_reorder_kernel_f32_t(desc);
        default: assert(!"unknown kernel id"); return nullptr;
    }

    return nullptr;
}

} // namespace tr

static void prb_block_for_cache(tr::prb_t &prb) {
    /* If strides for 0th and 1st nodes are cache friendly
     * then one can altogether do away with blocking ! */
    static constexpr int num_elems_thr = 16;
    const bool stride_cache_friendly
            = ((prb.nodes[0].is % 64 == 0 && prb.nodes[0].n > num_elems_thr)
                      || (prb.ndims > 1 && prb.nodes[1].is % num_elems_thr == 0
                              && prb.nodes[1].n > num_elems_thr))
            && !prb.is_tail_present;

    // performance improvement for shapes with large inner-most dimension
    const size_t L1_cache_sz
            = size_t(3) * platform::get_per_core_cache_size(1) / 4;
    const size_t itype_sz_ = data_type_size(prb.itype);
    const size_t inner_block_sz = prb.nodes[0].n * itype_sz_;
    const bool requires_inner_blocking = inner_block_sz > L1_cache_sz
            // 'is_tail_present' is not supported for cache_blocking when
            // asymmetric_comp is executed.
            && IMPLICATION(prb.req_asymmetric_comp, !prb.is_tail_present);

    const bool cache_blocking_needed
            = stride_cache_friendly || requires_inner_blocking;
    if (!cache_blocking_needed) return;

    int unit_input_stride_idx = -1;
    for (auto idx = 0; idx < prb.ndims; ++idx) {
        if (prb.nodes[idx].is == 1) unit_input_stride_idx = idx;
    }

    /* Re-prioritize the sequential read over sequential write:
     *                             /-> [n0:is0:1][16n1:1:osk]...
     * [n0:is0:1]...[nk:1:osk] -->     or
     *                             \-> [16n1:1:osk][n0:is0:1]... */
    if (unit_input_stride_idx != -1) {
        const auto output_stride = prb.nodes[unit_input_stride_idx].os;
        const auto num_elems = prb.nodes[unit_input_stride_idx].n;

        const bool split_needed = (num_elems > num_elems_thr)
                && (num_elems % num_elems_thr == 0);
        const int move_location = (output_stride % 4 != 0) ? 0 : 1;
        if (split_needed)
            prb_node_split(prb, unit_input_stride_idx, num_elems_thr);

        /* Because of cache-unfriendly nature of unit-output stride node, let
         * us move unit-input stride node on or near front! */
        if (unit_input_stride_idx != move_location)
            prb_node_move(prb, unit_input_stride_idx, move_location);
    }

    /* Potentially, split the node with os=1 in two and pull in the node with
     * is=1 between them for better cache reuse:
     * [n0:is0:1][n1:1:os1] --> [16n0:is0:1][n1:1:os1][n0/16:is0*16:16] */
    if (prb.ndims >= 2 && prb.nodes[0].os == 1 && prb.nodes[1].is == 1) {
        const auto num_elems = prb.nodes[0].n;

        const bool split_needed = (num_elems > num_elems_thr)
                && (num_elems % num_elems_thr == 0);
        if (split_needed) {
            prb_node_split(prb, 0, num_elems_thr);
            prb_node_move(prb, 1, 2);

            // Update node information
            prb_node_dependency(prb);

            // heuristics - looping over the unrolled dims should maximize reuse
            // of the already cached data; observation is choosing the smallest
            // dim from the remaining (from 2 up to ndims) gives good results
            constexpr int new_position = 2;
            const auto dim_beg_it = std::begin(prb.nodes);
            const auto dim_two_it = dim_beg_it + new_position;
            const auto dim_last_it = dim_beg_it + prb.ndims;
            const auto min_n_node_it = std::min_element(dim_two_it, dim_last_it,
                    [](const tr::node_t &lhs, const tr::node_t &rhs) {
                        return lhs.n < rhs.n;
                    });
            const auto min_idx = std::distance(dim_beg_it, min_n_node_it);
            // check if min_idx node is parent of node with tail processing which
            // is currently unsupported (i.e. tail processing can only be handled
            // at the inner-most dimension)
            bool inner_block_has_tail = false;
            for (int idx = min_idx - 1; idx >= new_position; idx--) {
                if (prb.nodes[idx].parent_node_id == min_idx) {
                    inner_block_has_tail = true;
                    break;
                }
            }

            if (min_idx > new_position && (!inner_block_has_tail))
                prb_node_move(prb, min_idx, new_position);
        }
    }
}

/** finds the maximum number of dimension the kernel should process and
 * optionally splits one of the dimension to achieve better balance between
 * parallel driver and the kernel. */
static void prb_thread_kernel_balance(
        tr::prb_t &prb, int &ndims_ker_max, int nthr) {
    size_t size_total = 1;
    for (int d = 0; d < prb.ndims; ++d)
        size_total *= prb.nodes[d].n;

    /* The general expression for size_drv_thr can be written as
     * size_drv_min = C0 + FC * (nthr > 1 ? 1 : 0) + VC * (nthr - 1)
     * where FC and VC are fixed and variable costs respectively.
     * Though for now, the below heuristic seems to be good enough */
    const size_t size_drv_thr = (nthr > 1) ? 16 * nthr : 1;

    /* size_drv_min is the minimal size for the parallel
     * driver required for good parallelization */
    const size_t size_drv_min
            = nstl::min<size_t>(size_drv_thr, utils::div_up(size_total, 1024));

    /* kdims -- # of dimensions processed by a kernel
     * size_ker_cur -- product of the dimension processed by a kernel
     * size_drv_cur -- product of the dimension processed by a driver */

    int kdims = prb.ndims;
    size_t size_drv_cur = 1;
    for (; kdims > 1 && size_drv_cur < size_drv_min; --kdims)
        size_drv_cur *= prb.nodes[kdims - 1].n;

    size_t size_ker_cur = 1;
    for (int d = 0; d < kdims; ++d)
        size_ker_cur *= prb.nodes[d].n;

    /* Initially kdims is chosen so that size_drv_cur >= size_drv_min.
     *
     * It might happen that for chosen kdims the size_ker_cur is too small
     * (less than tr::ker_prb_size_min). In that case try to split the
     * innermost driver dimension into two, to increase size_ker_cur. */
    const bool want_borrow_ker_from_drv = kdims < prb.ndims
            && size_ker_cur < tr::ker_prb_size_min
            && size_drv_cur > size_drv_min;
    if (want_borrow_ker_from_drv) {
        /* size_want_borrow is the minimal size, so that:
         *  o) size_ker_cur * size_want_borrow >= tr::ker_prb_size_min
         *  o) current innermost driver dimension is divisible by
         *     size_want_borrow (so that we can evenly split that
         *     dimension into two)
         *
         *  In the worst case the minimal size_want_borrow is equal
         *  to the innermost driver dimension itself. In that case
         *  we will sacrifice it in favor of kernel (is it fine?). */
        size_t size_want_borrow
                = utils::div_up(tr::ker_prb_size_min, size_ker_cur);
        for (; prb.nodes[kdims].n % size_want_borrow; ++size_want_borrow)
            ;

        if (size_want_borrow != prb.nodes[kdims].n)
            prb_node_split(prb, kdims, size_want_borrow);
        kdims += 1;
    }

    /* On the other hand it might happen that for chosen kdims
     * the size_drv_cur is too small (less than size_drv_min). In that case
     * try to split the outermost kernel dimension into two, to increase
     * size_drv_cur. */
    const bool want_borrow_drv_from_ker = size_ker_cur > tr::ker_prb_size_min
            && size_drv_cur < size_drv_min;
    if (want_borrow_drv_from_ker) {
        size_t size_want_borrow = utils::div_up(size_drv_min, size_drv_cur);
        for (; prb.nodes[kdims - 1].n % size_want_borrow; ++size_want_borrow)
            ;

        if (size_want_borrow != prb.nodes[kdims - 1].n)
            prb_node_split(
                    prb, kdims - 1, prb.nodes[kdims - 1].n / size_want_borrow);
    }

    ndims_ker_max = kdims;

    if (want_borrow_ker_from_drv || want_borrow_drv_from_ker) {
        DEBUG({
            printf("split: ");
            prb_dump(prb);
            printf("ndims_ker_max = %d\n", ndims_ker_max);
        });
    }
}

status_t jit_uni_reorder_t::pd_t::init(
        engine_t *engine, engine_t *src_engine, engine_t *dst_engine) {
    CHECK(cpu_reorder_pd_t::init(engine, src_engine, dst_engine));

    CHECK(init_scratchpad());

    return status::success;
}

status_t jit_uni_reorder_t::pd_t::init_scratchpad() {
    auto scratchpad = scratchpad_registry().registrar();

    const bool compensation_needed
            = prb_.req_s8s8_comp || prb_.req_asymmetric_comp;
    if (compensation_needed) {
        const memory_desc_wrapper od(dst_md());
        const auto G = with_groups_ ? od.padded_dims()[0] : 1;
        const auto N = od.padded_dims()[with_groups_ ? 1 : 0];
        static constexpr int cache_line_size = 16;
        const auto wspace_per_thr_size
                = utils::rnd_up(G * N, cache_line_size) * sizeof(int32_t);

        const auto compensation_reduce_size = wspace_per_thr_size * nthr_;

        // Every thread gets its own scratchpad space for each N.
        scratchpad.template book<int32_t>(
                memory_tracking::names::key_reorder_space,
                compensation_reduce_size);
    }

    const memory_desc_wrapper input_d(src_md());
    int scales_mask = -1;
    bool is_set = false;
    CHECK(attr()->scales_.get(DNNL_ARG_DST, &scales_mask, &is_set));

    if (is_set && scales_mask > 0) {
        get_D_values(input_d, scales_mask, nullptr, &D_mask_, nullptr);
        if (D_mask_ > 1) {
            scratchpad.template book<float>(
                    memory_tracking::names::key_reorder_precomputed_dst_scales,
                    D_mask_);
        }
    }

    return status::success;
}

status_t jit_uni_reorder_t::pd_t::create(reorder_pd_t **reorder_pd,
        engine_t *engine, const primitive_attr_t *attr, engine_t *src_engine,
        const memory_desc_t *src_md, engine_t *dst_engine,
        const memory_desc_t *dst_md) {
    auto prb = tr::prb_t();

    status_t prb_init_status = prb_init(prb, *src_md, *dst_md, attr);
    if (prb_init_status != status::success) return prb_init_status;

    prb_block_for_cache(prb);
    DEBUG({
        printf("cache: ");
        prb_dump(prb);
    });

    int ndims_ker_max {};
    int nthr = dnnl_get_max_threads();
    prb_thread_kernel_balance(prb, ndims_ker_max, nthr);

    if (prb.is_tail_present) prb_node_dependency(prb);

    tr::kernel_t::desc_t ker_desc;
    status_t ker_init_status
            = tr::kernel_t::desc_init(ker_desc, prb, ndims_ker_max);
    if (ker_init_status != status::success) return ker_init_status;

    const int ndims_driver = prb.ndims - ker_desc.prb.ndims;
    if (ndims_driver > jit_uni_reorder_t::ndims_driver_max)
        return status::unimplemented;

    DEBUG({
        printf("ker  : ");
        prb_dump(ker_desc.prb);
    });

    auto _pd = make_unique_pd<pd_t>(
            attr, src_engine->kind(), src_md, dst_engine->kind(), dst_md);
    if (_pd == nullptr) return status::out_of_memory;

    _pd->nthr_ = nthr;
    _pd->prb_ = prb;
    _pd->with_groups_
            = prb.compensation_mask == tr::prb_t::comp_mask_with_groups;
    CHECK(_pd->init(engine, src_engine, dst_engine));
    _pd->ker_desc_ = ker_desc;
    CHECK(_pd->init_scratchpad_md());

    return safe_ptr_assign(*reorder_pd, _pd.release());
}

void jit_uni_reorder_t::omp_driver_0d(int off, const char *in, char *out,
        const float *src_scales, const float *dst_scales, int src_zp,
        int dst_zp, int32_t *compensation_scratch) const {
    const tr::prb_t &prb = pd()->prb_;

    tr::call_param_t base_params;
    base_params.in = in;
    base_params.out = out;
    base_params.src_scales = src_scales;
    base_params.dst_scales = dst_scales;
    base_params.src_zp = src_zp;
    base_params.dst_zp = dst_zp;
    base_params.compensation_scratch = compensation_scratch;

    if (prb.is_tail_present) {
        tr::tail_call_param_t tail_params;
        tail_params.base_params = base_params;

        static constexpr int omp_ndims = 0;
        fill_curr_data_chunks(prb, off, nullptr, omp_ndims, tail_params);

        (*kernel_)(&tail_params);
    } else {
        (*kernel_)(&base_params);
    }
}

void jit_uni_reorder_t::omp_driver_1d(int ithr, int nthr, int off,
        const char *in, char *out, const float *src_scales,
        const float *dst_scales, int src_zp, int dst_zp,
        int32_t *compensation_scratch) const {
    const tr::prb_t &prb = pd()->prb_;
    const tr::node_t *ns = prb.nodes + off;
    for_nd(ithr, nthr, (ptrdiff_t)ns[0].n, [&](ptrdiff_t d0) {
        tr::call_param_t base_params;
        base_params.in = in + d0 * ns[0].is * data_type_size(prb.itype);
        base_params.out = out + d0 * ns[0].os * data_type_size(prb.otype);
        base_params.src_scales = src_scales + d0 * ns[0].ss;
        base_params.dst_scales = dst_scales + d0 * ns[0].ss;
        base_params.src_zp = src_zp;
        base_params.dst_zp = dst_zp;
        base_params.compensation_scratch = compensation_scratch + d0 * ns[0].cs;

        if (prb.is_tail_present) {
            tr::tail_call_param_t tail_params;
            tail_params.base_params = base_params;

            static constexpr int omp_ndims = 1;
            const ptrdiff_t omp_data_chunks[omp_ndims] = {d0};
            fill_curr_data_chunks(
                    prb, off, omp_data_chunks, omp_ndims, tail_params);

            (*kernel_)(&tail_params);
        } else {
            (*kernel_)(&base_params);
        }
    });
}

void jit_uni_reorder_t::omp_driver_2d(int ithr, int nthr, int off,
        const char *in, char *out, const float *src_scales,
        const float *dst_scales, int src_zp, int dst_zp,
        int32_t *compensation_scratch) const {
    const tr::prb_t &prb = pd()->prb_;
    const tr::node_t *ns = prb.nodes + off;
    for_nd(ithr, nthr, (ptrdiff_t)ns[1].n, (ptrdiff_t)ns[0].n,
            [&](ptrdiff_t d1, ptrdiff_t d0) {
                tr::call_param_t base_params;
                base_params.in = in
                        + (d0 * ns[0].is + d1 * ns[1].is)
                                * data_type_size(prb.itype);
                base_params.out = out
                        + (d0 * ns[0].os + d1 * ns[1].os)
                                * data_type_size(prb.otype);
                base_params.src_scales
                        = src_scales + d0 * ns[0].ss + d1 * ns[1].ss;
                base_params.dst_scales
                        = dst_scales + d0 * ns[0].ss + d1 * ns[1].ss;
                base_params.src_zp = src_zp;
                base_params.dst_zp = dst_zp;
                base_params.compensation_scratch
                        = compensation_scratch + d0 * ns[0].cs + d1 * ns[1].cs;

                if (prb.is_tail_present) {
                    tr::tail_call_param_t tail_params;
                    tail_params.base_params = base_params;

                    static constexpr int omp_ndims = 2;
                    const ptrdiff_t omp_data_chunks[omp_ndims] = {d0, d1};
                    fill_curr_data_chunks(
                            prb, off, omp_data_chunks, omp_ndims, tail_params);

                    (*kernel_)(&tail_params);
                } else {
                    (*kernel_)(&base_params);
                }
            });
}

void jit_uni_reorder_t::omp_driver_3d(int ithr, int nthr, int off,
        const char *in, char *out, const float *src_scales,
        const float *dst_scales, int src_zp, int dst_zp,
        int32_t *compensation_scratch) const {
    const tr::prb_t &prb = pd()->prb_;
    const tr::node_t *ns = prb.nodes + off;
    for_nd(ithr, nthr, (ptrdiff_t)ns[2].n, (ptrdiff_t)ns[1].n,
            (ptrdiff_t)ns[0].n, [&](ptrdiff_t d2, ptrdiff_t d1, ptrdiff_t d0) {
                tr::call_param_t base_params;
                base_params.in = in
                        + (d0 * ns[0].is + d1 * ns[1].is + d2 * ns[2].is)
                                * data_type_size(prb.itype);
                base_params.out = out
                        + (d0 * ns[0].os + d1 * ns[1].os + d2 * ns[2].os)
                                * data_type_size(prb.otype);
                base_params.src_scales = src_scales + d0 * ns[0].ss
                        + d1 * ns[1].ss + d2 * ns[2].ss;
                base_params.dst_scales = dst_scales + d0 * ns[0].ss
                        + d1 * ns[1].ss + d2 * ns[2].ss;
                base_params.src_zp = src_zp;
                base_params.dst_zp = dst_zp;
                base_params.compensation_scratch = compensation_scratch
                        + d0 * ns[0].cs + d1 * ns[1].cs + d2 * ns[2].cs;

                if (prb.is_tail_present) {
                    tr::tail_call_param_t tail_params;
                    tail_params.base_params = base_params;

                    static constexpr int omp_ndims = 3;
                    const ptrdiff_t omp_data_chunks[omp_ndims] = {d0, d1, d2};
                    fill_curr_data_chunks(
                            prb, off, omp_data_chunks, omp_ndims, tail_params);

                    (*kernel_)(&tail_params);
                } else {
                    (*kernel_)(&base_params);
                }
            });
}

void jit_uni_reorder_t::omp_driver_4d(int ithr, int nthr, int off,
        const char *in, char *out, const float *src_scales,
        const float *dst_scales, int src_zp, int dst_zp,
        int32_t *compensation_scratch) const {
    const tr::prb_t &prb = pd()->prb_;
    const tr::node_t *ns = prb.nodes + off;
    for_nd(ithr, nthr, (ptrdiff_t)ns[3].n, (ptrdiff_t)ns[2].n,
            (ptrdiff_t)ns[1].n, (ptrdiff_t)ns[0].n,
            [&](ptrdiff_t d3, ptrdiff_t d2, ptrdiff_t d1, ptrdiff_t d0) {
                tr::call_param_t base_params;
                base_params.in = in
                        + (d0 * ns[0].is + d1 * ns[1].is + d2 * ns[2].is
                                  + d3 * ns[3].is)
                                * data_type_size(prb.itype);
                base_params.out = out
                        + (d0 * ns[0].os + d1 * ns[1].os + d2 * ns[2].os
                                  + d3 * ns[3].os)
                                * data_type_size(prb.otype);
                base_params.src_scales = src_scales + d0 * ns[0].ss
                        + d1 * ns[1].ss + d2 * ns[2].ss + d3 * ns[3].ss;
                base_params.dst_scales = dst_scales + d0 * ns[0].ss
                        + d1 * ns[1].ss + d2 * ns[2].ss + d3 * ns[3].ss;
                base_params.src_zp = src_zp;
                base_params.dst_zp = dst_zp;
                base_params.compensation_scratch = compensation_scratch
                        + d0 * ns[0].cs + d1 * ns[1].cs + d2 * ns[2].cs
                        + d3 * ns[3].cs;

                if (prb.is_tail_present) {
                    tr::tail_call_param_t tail_params;
                    tail_params.base_params = base_params;

                    static constexpr int omp_ndims = 4;
                    const ptrdiff_t omp_data_chunks[omp_ndims]
                            = {d0, d1, d2, d3};
                    fill_curr_data_chunks(
                            prb, off, omp_data_chunks, omp_ndims, tail_params);

                    (*kernel_)(&tail_params);
                } else {
                    (*kernel_)(&base_params);
                }
            });
}

void jit_uni_reorder_t::omp_driver(const char *in, char *out,
        const float *src_scales, const float *dst_scales, int src_zp,
        int dst_zp, const memory_tracking::grantor_t &scratchpad) const {
    in += pd()->prb_.ioff * data_type_size(pd()->prb_.itype);
    out += pd()->prb_.ooff * data_type_size(pd()->prb_.otype);

    DEBUG({
        printf("prb : ");
        tr::prb_dump(pd()->prb_);
    });
    DEBUG({
        printf("ker : ");
        tr::prb_dump(pd()->ker_desc_.prb);
    });

    int ndims = pd()->prb_.ndims;
    int ndims_ker = pd()->ker_desc_.prb.ndims;
    const bool req_s8s8_comp = pd()->prb_.req_s8s8_comp;
    const bool req_asymmetric_comp = pd()->prb_.req_asymmetric_comp;
    const bool req_compensation = req_s8s8_comp || req_asymmetric_comp;
    assert(ndims - ndims_ker <= ndims_driver_max);

    int32_t *compensation_reduce_scratch = scratchpad.template get<int32_t>(
            memory_tracking::names::key_reorder_space);

    const memory_desc_wrapper od(pd()->dst_md());
    const auto G = pd()->with_groups_ ? od.padded_dims()[0] : 1;
    const auto N = od.padded_dims()[pd()->with_groups_ ? 1 : 0];
    static constexpr int cache_line_size = 16;
    const auto wspace_per_thr_size = utils::rnd_up(G * N, cache_line_size);
    const auto wspace_per_thr_bytes = wspace_per_thr_size * sizeof(int32_t);

    if (ndims - ndims_ker == 0) {
        if (req_compensation)
            std::memset(compensation_reduce_scratch, 0, wspace_per_thr_bytes);

        omp_driver_0d(ndims_ker, in, out, src_scales, dst_scales, src_zp,
                dst_zp, compensation_reduce_scratch);
    } else {
        parallel(pd()->nthr_, [&](const int ithr, const int nthr) {
            int32_t *compensation_scratch = nullptr;
            if (req_compensation) {
                compensation_scratch = &compensation_reduce_scratch[ithr
                        * wspace_per_thr_size];
                std::memset(compensation_scratch, 0, wspace_per_thr_bytes);
            }

            switch (ndims - ndims_ker) {
                case 1:
                    omp_driver_1d(ithr, nthr, ndims_ker, in, out, src_scales,
                            dst_scales, src_zp, dst_zp, compensation_scratch);
                    break;
                case 2:
                    omp_driver_2d(ithr, nthr, ndims_ker, in, out, src_scales,
                            dst_scales, src_zp, dst_zp, compensation_scratch);
                    break;
                case 3:
                    omp_driver_3d(ithr, nthr, ndims_ker, in, out, src_scales,
                            dst_scales, src_zp, dst_zp, compensation_scratch);
                    break;
                case 4:
                    omp_driver_4d(ithr, nthr, ndims_ker, in, out, src_scales,
                            dst_scales, src_zp, dst_zp, compensation_scratch);
                    break;
                default: assert(!"unimplemented");
            }
        });
    }

    //reduction of intermediate compensation results to the final output
    if (req_compensation) {
        const int nthr = ndims - ndims_ker == 0 ? 1 : pd()->nthr_;
        reduce_compensation(
                out, compensation_reduce_scratch, nthr, wspace_per_thr_size);
    }
}

void jit_uni_reorder_t::reduce_compensation(char *out,
        const int32_t *compensation_reduce_scratch, const int nthr,
        const dim_t wspace_per_thr_size) const {

    const memory_desc_wrapper od(pd()->dst_md());
    const size_t offset = od.size() - od.additional_buffer_size();

    static constexpr auto comp_dt_size = sizeof(int32_t);
    static constexpr int32_t comp_s8s8_shift = 128;

    // Note: We do not need to explicitly zero-out compensation buffer, as the
    // per_thread buffers are already zeroed out in the padded area.
    const auto G = pd()->with_groups_ ? od.padded_dims()[0] : 1;
    const auto N = od.padded_dims()[pd()->with_groups_ ? 1 : 0];
    const auto GN = G * N;
    const bool req_s8s8_comp = pd()->prb_.req_s8s8_comp;
    const bool req_asymmetric_comp = pd()->prb_.req_asymmetric_comp;
    const size_t zp_offset
            = offset + (pd()->prb_.req_s8s8_comp ? GN * comp_dt_size : 0);

    parallel_nd(GN, [&](int idx) {
        int32_t acc = 0;
        for (int ithr = 0; ithr < nthr; ithr++) {
            acc -= compensation_reduce_scratch[ithr * wspace_per_thr_size
                    + idx];
        }
        if (req_s8s8_comp) {
            int32_t *out_comp = reinterpret_cast<int32_t *>(&out[offset]);
            out_comp[idx] = comp_s8s8_shift * acc;
        }
        if (req_asymmetric_comp) {
            int32_t *out_asym_comp
                    = reinterpret_cast<int32_t *>(&out[zp_offset]);
            out_asym_comp[idx] = acc;
        }
    });
}

void jit_uni_reorder_t::fill_curr_data_chunks(const tr::prb_t &prb,
        const int off, const ptrdiff_t *omp_data_chunks, const int omp_ndims,
        tr::tail_call_param_t &c) const {
    // Chunks are backwards numered i.e:
    // [0] -> [node_size]
    // [1] -> [node_size - 1]
    // ...
    // [node_size - 1] -> [1]

    // It is done like this, because it is easier to decrement counter
    // and check if it is equal to zero than increment and check
    // if it is equal to node_size in jit kernel.

    static constexpr int64_t empty_chunk_info = -1;
    static constexpr int64_t last_chunk = 1;

    for (int curr_node_id = prb.ndims - 1; curr_node_id >= 0; curr_node_id--) {
        const int parent_node_id = prb.nodes[curr_node_id].parent_node_id;
        const bool is_drv_processing_this_node
                = curr_node_id >= off && curr_node_id <= off + omp_ndims - 1;
        const bool is_tail_processing
                = prb.is_tail_in_one_of_child_nodes(curr_node_id)
                || prb.nodes[curr_node_id].tail_size > 0;

        if (is_drv_processing_this_node && is_tail_processing) {
            const int inner_idx = curr_node_id - off;
            assert(inner_idx < omp_ndims);
            const int64_t node_size = prb.nodes[curr_node_id].tail_size > 0
                    ? prb.nodes[curr_node_id].tail_size
                    : prb.nodes[curr_node_id].n;
            const int64_t data_chunk = node_size - omp_data_chunks[inner_idx];

            if (!prb.nodes[curr_node_id].is_parent_empty()) {
                const bool is_parent_chunk_last
                        = c.curr_data_chunks[parent_node_id] == last_chunk;
                c.curr_data_chunks[curr_node_id]
                        = is_parent_chunk_last ? data_chunk : empty_chunk_info;
                c.zeroing_data = static_cast<int64_t>(
                        is_parent_chunk_last && data_chunk <= 0);
            } else {
                c.curr_data_chunks[curr_node_id] = data_chunk;
                c.zeroing_data = static_cast<int64_t>(data_chunk <= 0);
            }
            c.skip_kernel_execution = static_cast<int64_t>(c.zeroing_data
                    && !prb.nodes[curr_node_id].is_zero_pad_needed);
            if (c.zeroing_data || c.skip_kernel_execution) break;
        } else
            c.curr_data_chunks[curr_node_id] = empty_chunk_info;
    }
}

status_t jit_uni_reorder_t::init(engine_t *engine) {
    CHECK(safe_ptr_assign(kernel_, tr::kernel_t::create(pd()->ker_desc_)));
    return kernel_->create_kernel();
}

status_t jit_uni_reorder_t::execute(const exec_ctx_t &ctx) const {
    const auto &scratchpad = ctx.get_scratchpad_grantor();

    auto in = CTX_IN_MEM(const char *, DNNL_ARG_FROM);
    auto out = CTX_OUT_MEM(char *, DNNL_ARG_TO);
    DEFINE_ARG_SCALES_BUFFER(src_scales, DNNL_ARG_SRC);
    DEFINE_ARG_SCALES_BUFFER(dst_scales_, DNNL_ARG_DST);

    const float *dst_scales = pd()->precompute_scales(
            scratchpad, pd()->attr(), pd()->D_mask_, dst_scales_);
    assert(dst_scales);

    DEFINE_ZERO_POINT_VALUE(src_zp, DNNL_ARG_FROM);
    DEFINE_ZERO_POINT_VALUE(dst_zp, DNNL_ARG_TO);

    omp_driver(in, out, src_scales, dst_scales, src_zp, dst_zp, scratchpad);

    return status::success;
}

status_t jit_blk_reorder_t::pd_t::create(reorder_pd_t **reorder_pd,
        engine_t *engine, const primitive_attr_t *attr, engine_t *src_engine,
        const memory_desc_t *src_md, engine_t *dst_engine,
        const memory_desc_t *dst_md) {
    auto prb = tr::prb_t();

    status_t prb_init_status = prb_init(prb, *src_md, *dst_md, attr);
    if (prb_init_status != status::success) return prb_init_status;
    // only uni_reorder supports tail processing now
    // TODO: Add tail processing support in blk_reorder
    if (prb.is_tail_present) return status::unimplemented;

    prb_tile_normalize(prb);
    DEBUG({
        printf("tile : ");
        prb_dump(prb);
    });

    if (!tr::jit_single_blk_kernel_t::applicable(prb)) {
        return status::unimplemented;
    }

    auto _pd = make_unique_pd<pd_t>(
            attr, src_engine->kind(), src_md, dst_engine->kind(), dst_md);
    if (_pd == nullptr) return status::out_of_memory;
    _pd->prb_ = prb;
    CHECK(_pd->init(engine, src_engine, dst_engine));
    CHECK(_pd->init_scratchpad_md());

    return safe_ptr_assign(*reorder_pd, _pd.release());
}

void jit_blk_reorder_t::pd_t::prb_tile_normalize(tr::prb_t &p) {
    if (!utils::one_of(p.nodes[0].n, 8ul, 16ul)
            && utils::one_of(p.nodes[1].n, 8ul, 16ul)) {
        nstl::swap(p.nodes[0], p.nodes[1]);
    }
}

jit_blk_reorder_t::jit_blk_reorder_t(const pd_t *apd) : primitive_t(apd) {}
jit_blk_reorder_t::~jit_blk_reorder_t() = default;

status_t jit_blk_reorder_t::init(engine_t *engine) {
    kernel_ = utils::make_unique<tr::jit_single_blk_kernel_t>(pd()->prb_);
    return kernel_->create_kernel();
}

status_t jit_blk_reorder_t::execute(const exec_ctx_t &ctx) const {
    const auto in = CTX_IN_MEM(const char *, DNNL_ARG_FROM);
    auto out = CTX_OUT_MEM(char *, DNNL_ARG_TO);
    DEFINE_ZERO_POINT_VALUE(src_zp, DNNL_ARG_FROM);
    DEFINE_ZERO_POINT_VALUE(dst_zp, DNNL_ARG_TO);

    // kernel handle 2-dimension tiles, a tail is possible
    auto &prb = this->pd()->prb_;
    ptrdiff_t BH = 1;
    for (int i = 2; i < prb.ndims; ++i) {
        BH *= prb.nodes[i].n;
    }

    auto block_sz = prb.n(0);
    auto n1 = prb.n(1);
    auto i1 = prb.is(1);
    auto o1 = prb.os(1);
    auto FL = (n1 + block_sz - 1) / block_sz;
    auto bh_stride = BH == 1 ? 0 : prb.is(2);

    auto itype_sz_ = data_type_size(pd()->prb_.itype);
    auto otype_sz_ = data_type_size(pd()->prb_.otype);

    parallel_nd(BH, FL, [&](dim_t bh, dim_t fl) {
        auto fl_b = fl * block_sz;
        auto bh_b = bh_stride * bh;
        auto *i = in + (bh_b + fl_b * i1) * itype_sz_;
        auto *o = out + (bh_b + fl_b * o1) * otype_sz_;
        (*kernel_)(i, o, n1 - fl_b < block_sz, src_zp, dst_zp);
    });

    return status::success;
}

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
