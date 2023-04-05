/*******************************************************************************
* Copyright 2017-2023 Intel Corporation
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

#include "common/c_types_map.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "cpu/x64/cpu_barrier.hpp"
#include "cpu/x64/jit_generator.hpp"

#include "cpu/x64/jit_transpose_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::utils;
using namespace Xbyak;

#define GET_OFF(x) offsetof(ctx_t, x)

struct jit_trans_iw_ic_int16_t : public jit_trans_src_t, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_trans_iw_ic_int16_t)
    jit_trans_iw_ic_int16_t(const jit_conv_conf_t *conf)
        : jit_trans_src_t(conf), jit_generator(jit_name()) {}

    void operator()(ctx_t *ctx) override { jit_generator::operator()(ctx); }

    status_t create_kernel() override { return jit_generator::create_kernel(); }

private:
    using reg64_t = const Xbyak::Reg64;
    using reg32_t = const Xbyak::Reg32;
    using opmask_t = const Xbyak::Opmask;

    enum {
        typesize = sizeof(int16_t),
        transpose_size = 16,
        small_spatial = 14
    };
    size_t src_stride = 0, tr_src_stride = 0;
    int tail = 0;
    bool enable_prefetch = false;

    opmask_t kFFFF = k1;
    opmask_t k5555 = k2;
    opmask_t kAAAA = k3;
    opmask_t kAA = k4;
    opmask_t k55 = k5;
    opmask_t kCC = k6;
    opmask_t k33 = k7;
    opmask_t kTail = k1;

    reg64_t reg_src = r8;
    reg64_t reg_tr_src = r9;
    reg64_t reg_src_prf = r10;
    reg64_t reg_tr_src_prf = r11;
    reg64_t reg_loop = r12;
    reg64_t reg_tr_src_tmp = r13;
    reg32_t regw_tmp = r14d;
    reg64_t imm_addr64 = rbx;

    Xbyak::Zmm vidx1 = zmm31;
    Xbyak::Zmm vidx2 = zmm30;
    Xbyak::Zmm vidx3 = zmm29;
    Xbyak::Zmm vidx4 = zmm28;
    Xbyak::Zmm vidx5 = zmm27;
    Xbyak::Zmm zmm_tmp = zmm26;

    void transpose(int nrows, int l_pad, int r_pad, bool nontemporal_stores);
    void generate() override;
};

void jit_trans_iw_ic_int16_t::transpose(
        int nrows, int l_pad, int r_pad, bool nontemporal_stores) {
    assert(nrows >= 0 && nrows <= transpose_size);
    static_assert(transpose_size == 16, "Unsupported transpose size");
    if (!nrows) return;

    auto src_zmm = [](int i) { return Zmm(i); };

    auto src_ymm = [](int i) {
        assert(i >= 0 && i < 16);
        return Ymm(i);
    };

    auto load_ymm = [this, src_ymm](int i) {
        vmovups(src_ymm(i), EVEX_compress_addr(reg_src, i * src_stride));
    };

    auto kmovw = [this](Opmask k, unsigned w) {
        mov(regw_tmp, w);
        jit_generator::kmovw(k, regw_tmp);
    };

    auto kmovd = [this](Opmask k, unsigned w) {
        mov(regw_tmp, w);
        jit_generator::kmovd(k, regw_tmp);
    };

    auto store = [&](Zmm r, int i) {
        auto padding
                = [this, i, kmovw](Reg64 base, int pad_rows, int pad_tail) {
                      // note: pad can be bigger than 16 because of dilation
                      const size_t row_offset = 2 * transpose_size * typesize;
                      auto zmm_zero = zmm_tmp;
                      vpxord(zmm_zero, zmm_zero, zmm_zero);
                      for (int i_row = 0; i_row < pad_rows; i_row++) {
                          auto addr = EVEX_compress_addr(
                                  base, i * tr_src_stride + i_row * row_offset);
                          vmovups(addr, zmm_zero);
                      }
                      if (pad_tail > 0) {
                          kmovw(kTail, (1 << pad_tail) - 1);
                          base.setOpmaskIdx(kTail.getIdx(), true);
                          auto addr = EVEX_compress_addr(base,
                                  i * tr_src_stride + pad_rows * row_offset);
                          vmovups(addr, zmm_zero);
                      }
                  };

        mov(reg_tr_src_tmp, reg_tr_src);
        if (l_pad > 0) {
            int store_pad = 2 * transpose_size;
            int pad_rows = l_pad / store_pad;
            int tail = l_pad % store_pad;
            padding(reg_tr_src_tmp, pad_rows, div_up(tail, 2));
            add(reg_tr_src_tmp, (pad_rows * store_pad + tail) * typesize);
        }
        if (r_pad > 0) {
            int addr_shift = nrows - r_pad % 2;
            int store_pad = div_up(r_pad, 2);
            int pad_rows = store_pad / transpose_size;
            add(reg_tr_src_tmp, addr_shift * typesize);
            padding(reg_tr_src_tmp, pad_rows, store_pad % transpose_size);
            sub(reg_tr_src_tmp, addr_shift * typesize);
        }

        int store_tail = rnd_up(nrows, 2);
        kmovw(kTail, (1 << store_tail / 2) - 1);
        auto k = kTail;
        auto base = reg_tr_src_tmp;
        base.setOpmaskIdx(k.getIdx(), true);

        auto addr = EVEX_compress_addr(base, i * tr_src_stride);
        vmovups(addr, r);
    };

    const bool is_layout_nxc = utils::one_of(conf_->src_tag, format_tag::ndhwc,
            format_tag::nhwc, format_tag::nwc);
    const int ic_block = conf_->ic_block;
    const bool is_tail_block = ic_block != 16;
    const int ic_tail = conf_->ic_tail;
    // Assertion below as we need vmovdqu16 for ic_tails.
    // If needed, can be extended by using load_bytes() helper.
    assert(IMPLICATION(ic_tail, mayiuse(avx512_core)));
    if (mayiuse(avx512_core)) {
        if (conf_->stride_w > 1 || nrows % 2 || is_layout_nxc)
            kmovd(kFFFF, (1 << ic_block) - 1);
        if (conf_->stride_w > 1 || is_layout_nxc) kmovd(k33, 0xffff0000);
        if (is_layout_nxc && conf_->ic_tail) {
            Label done;
            cmp(dword[param1 + GET_OFF(ch_work)], ic_block);
            je(done, T_NEAR);
            kmovd(kFFFF, (1 << conf_->ic_tail) - 1);
            kshiftld(k33, kFFFF, 16);
            L(done);
        }

        for (int i = 0; i < nrows / 2; i++) {
            auto zmm_src0 = src_zmm(2 * i);
            if (conf_->stride_w == 1 && !is_layout_nxc) {
                vmovdqu16(zmm_src0,
                        EVEX_compress_addr(reg_src, 2 * i * src_stride));
            } else {
                vmovdqu16(zmm_src0 | kFFFF | T_z,
                        EVEX_compress_addr(reg_src, 2 * i * src_stride));
                if (is_tail_block || ic_tail) {
                    auto zmm_tmp = src_zmm(2 * i + 1);
                    vmovdqu16(zmm_tmp | kFFFF | T_z,
                            EVEX_compress_addr(
                                    reg_src, (2 * i + 1) * src_stride));
                    vinsertf64x4(zmm_src0, zmm_src0, src_ymm(2 * i + 1), 1);
                } else {
                    vmovdqu16(zmm_src0 | k33,
                            EVEX_compress_addr(
                                    reg_src, (2 * i + 1) * src_stride - 32));
                }
            }
            vpermw(zmm_src0, vidx5, zmm_src0);
        }

        // for odd numbers we need to mix row with zeroes
        if (nrows % 2) {
            int i = nrows / 2;
            auto zmm_src0 = src_zmm(2 * i);
            vmovdqu16(zmm_src0 | kFFFF | T_z,
                    EVEX_compress_addr(reg_src, 2 * i * src_stride));
            vpermw(zmm_src0, vidx5, zmm_src0);
        }

        if (conf_->stride_w > 1 || is_layout_nxc) kmovw(k33, 0x33);

        for (int i = rnd_up(nrows, 2); i < 16; i += 2) {
            vpxord(src_zmm(i), src_zmm(i), src_zmm(i));
        }
    } else {
        kmovw(kFFFF, 0xffff);
        // all loads
        for (int i = 0; i < 16; i++) {
            vpxord(src_zmm(i), src_zmm(i), src_zmm(i));
        }

        for (int i = 0; i < nrows / 2; i++) {
            auto src0 = src_ymm(2 * i);
            auto src1 = src_ymm(2 * i + 1);
            auto zmm_src0 = src_zmm(2 * i);
            load_ymm(2 * i);

            vpunpcklwd(src1, src0,
                    EVEX_compress_addr(reg_src, (2 * i + 1) * src_stride));
            vpunpckhwd(src0, src0,
                    EVEX_compress_addr(reg_src, (2 * i + 1) * src_stride));
            vinserti64x4(zmm_src0, zmm_src0, src1, 1);
            vpermps(zmm_src0 | kFFFF, vidx4, zmm_src0);
        }

        // for odd numbers we need to mix row with zeroes
        if (nrows % 2) {
            int i = nrows - 1;
            auto src0 = src_ymm(i);
            auto src1 = src_ymm(i + 1); // zero

            auto zmm_src0 = src_zmm(i);
            vpxor(src1, src1, src1);

            load_ymm(i);
            vpunpckhwd(src0, src0, src1);
            vinserti64x4(zmm_tmp, zmm_tmp, src0, 0);
            vpxor(src0, src0, src0);
            load_ymm(i);
            vpunpcklwd(src1, src0, src1);
            vinserti64x4(zmm_tmp, zmm_tmp, src1, 1);
            vpxord(zmm_src0, zmm_src0, zmm_src0);
            vmovups(zmm_src0, zmm_tmp);
            vpermps(zmm_src0 | kFFFF, vidx4, zmm_src0);
        }
    }

    // swap 1
    for (int i = 0; i < 4; i++) {
        auto zmm0 = src_zmm(4 * i);
        auto zmm1 = src_zmm(4 * i + 2);
        auto tmp0 = src_zmm(4 * i + 1);
        auto tmp1 = src_zmm(4 * i + 3);

        vmovups(tmp0, zmm0);
        vmovups(tmp1, zmm1);

        vpermps(tmp0 | kAAAA, vidx3, zmm1);
        vpermps(tmp1 | k5555, vidx3, zmm0);
    }
    // swap 2
    int base_idx;
    base_idx = 0;
    for (int i = 0; i < 2; i++) {
        auto zmm0 = src_zmm(base_idx + 2 * i + 1);
        auto zmm1 = src_zmm(base_idx + 2 * i + 5);

        auto tmp0 = src_zmm(base_idx + 2 * i);
        auto tmp1 = src_zmm(base_idx + 2 * i + 4);

        vmovupd(tmp0, zmm0);
        vmovupd(tmp1, zmm1);

        vpermpd(tmp0 | kAA, vidx2, zmm1);
        vpermpd(tmp1 | k55, vidx2, zmm0);
    }
    base_idx = 8;
    for (int i = 0; i < 2; i++) {
        auto zmm0 = src_zmm(base_idx + 2 * i + 1);
        auto zmm1 = src_zmm(base_idx + 2 * i + 5);

        auto tmp0 = src_zmm(base_idx + 2 * i);
        auto tmp1 = src_zmm(base_idx + 2 * i + 4);

        vmovupd(tmp0, zmm0);
        vmovupd(tmp1, zmm1);

        vpermpd(tmp0 | kAA, vidx2, zmm1);
        vpermpd(tmp1 | k55, vidx2, zmm0);
    }

    // swap 3
    for (int i = 0; i < 4; i++) {
        auto zmm0 = src_zmm(2 * i);
        auto zmm1 = src_zmm(2 * i + 8);

        auto tmp0 = src_zmm(2 * i + 1);
        auto tmp1 = src_zmm(2 * i + 9);

        vmovupd(tmp0, zmm0);
        vmovupd(tmp1, zmm1);

        vpermpd(tmp0 | kCC, vidx1, zmm1);
        vpermpd(tmp1 | k33, vidx1, zmm0);
    }

    // all stores
    for (int i = 0; i < 8; i++)
        vextracti64x4(src_ymm(2 * i), src_zmm(2 * i + 1), 1);

    auto get_vec_idx = [](int ic_idx) {
        assert(ic_idx < 16 && ic_idx >= 0);
        switch (ic_idx) {
            case 0: return 1;
            case 1: return 0;
            case 2: return 3;
            case 3: return 2;
            case 4: return 9;
            case 5: return 8;
            case 6: return 11;
            case 7: return 10;
            case 8: return 5;
            case 9: return 4;
            case 10: return 7;
            case 11: return 6;
            case 12: return 13;
            case 13: return 12;
            case 14: return 15;
            default: return 14;
        }
    };

    for (int ic = 0; ic < ic_block; ic++)
        store(src_zmm(get_vec_idx(ic)), ic);
}

void jit_trans_iw_ic_int16_t::generate() {
    preamble();

    alignas(64) static constexpr const int64_t idx1[8]
            = {2, 3, 0, 1, 6, 7, 4, 5};
    alignas(64) static constexpr const int64_t idx2[8]
            = {1, 0, 3, 2, 5, 4, 7, 6};
    alignas(64) static constexpr const int32_t idx3[16]
            = {1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14};
    alignas(64) static constexpr const int32_t idx4[16]
            = {8, 10, 12, 14, 0, 2, 4, 6, 9, 11, 13, 15, 1, 3, 5, 7};
    alignas(64) static constexpr const uint16_t idx5[32]
            = {0, 16, 2, 18, 8, 24, 10, 26, 4, 20, 6, 22, 12, 28, 14, 30, 1, 17,
                    3, 19, 9, 25, 11, 27, 5, 21, 7, 23, 13, 29, 15, 31};

    const int ic_block = conf_->ic_block;
    const bool is_layout_nxc = utils::one_of(conf_->src_tag, format_tag::ndhwc,
            format_tag::nhwc, format_tag::nwc);
    const size_t src_mult
            = is_layout_nxc ? conf_->ngroups * conf_->ic : ic_block;
    const int iw = conf_->iw;
    const int tr_iw = conf_->tr_iw;
    const int str_w = conf_->stride_w;
    assert(tr_iw % str_w == 0);
    const int tr_iw_s = tr_iw / str_w;
    assert(transpose_size >= ic_block);

    auto kmovw = [this](Opmask k, unsigned w) {
        mov(regw_tmp, w);
        jit_generator::kmovw(k, regw_tmp);
    };

    kmovw(kFFFF, 0xffff);
    kmovw(k5555, 0x5555);
    kmovw(kAAAA, 0xaaaa);
    kmovw(kAA, 0xaa);
    kmovw(k55, 0x55);
    kmovw(kCC, 0xcc);
    kmovw(k33, 0x33);

    auto vmovdqa64 = [this](Zmm z, const int64_t *addr) {
        mov(imm_addr64, reinterpret_cast<size_t>(addr));
        jit_generator::vmovdqa64(z, ptr[imm_addr64]);
    };

    auto vmovdqa32 = [this](Zmm z, const int32_t *addr) {
        mov(imm_addr64, reinterpret_cast<size_t>(addr));
        jit_generator::vmovdqa32(z, ptr[imm_addr64]);
    };

    vmovdqa64(vidx1, idx1);
    vmovdqa64(vidx2, idx2);
    vmovdqa32(vidx3, idx3);
    vmovdqa32(vidx4, idx4);
    vmovdqa32(vidx5, (const int32_t *)idx5);

    // Data for every strided case is placed consecutively
    // For 1x1 convolutions with strides we transpose only needed elements
    const auto str_w_end = (conf_->kw == 1) ? 1 : str_w;
    for (int s = 0; s < str_w_end; s++) {
        const int left_pad = div_up(conf_->l_pad - s, str_w);
        const int iw1 = iw + conf_->l_pad;
        const int iw_s = (s < (iw1 % str_w) ? div_up(iw1, str_w) : iw1 / str_w)
                - left_pad;
        const int right_pad = tr_iw_s - iw_s - left_pad;

        const int transposes = utils::div_up(iw_s, transpose_size);
        int loop_iters = nstl::max(0, transposes - 1);
        tail = iw_s - loop_iters * transpose_size;

        src_stride = src_mult * typesize * str_w;
        tr_src_stride = tr_iw * typesize;

        bool nontemporal_stores = false;
        enable_prefetch = iw > small_spatial ? true : false;

        const size_t src_step = src_mult * transpose_size * str_w * typesize;
        const size_t tr_src_step = transpose_size * typesize;

        mov(reg_src, ptr[param1 + GET_OFF(src)]);
        mov(reg_tr_src, ptr[param1 + GET_OFF(tr_src)]);
        mov(reg_src_prf, ptr[param1 + GET_OFF(src_prf)]);
        mov(reg_tr_src_prf, ptr[param1 + GET_OFF(tr_src_prf)]);

        if (str_w > 1) {
            int tr_src_shift = s;
            int src_shift = (str_w - (conf_->l_pad % str_w) + s) % str_w;
            add(reg_src, src_shift * src_mult * typesize);
            add(reg_tr_src, tr_src_shift * tr_iw_s * typesize);
            add(reg_src_prf, src_shift * src_mult * typesize);
            add(reg_tr_src_prf, tr_src_shift * tr_iw_s * typesize);
        }

        if (left_pad > 0 && loop_iters > 0) {
            loop_iters--;
            transpose(transpose_size, left_pad, 0, nontemporal_stores);
            add(reg_src, src_step);
            add(reg_tr_src, tr_src_step + left_pad * typesize);
            add(reg_src_prf, src_step);
            add(reg_tr_src_prf, tr_src_step + left_pad * typesize);
        }

        if (loop_iters) {
            mov(reg_loop, loop_iters);
            Label loop;
            L(loop);
            {
                transpose(transpose_size, 0, 0, nontemporal_stores);
                add(reg_src, src_step);
                add(reg_tr_src, tr_src_step);
                add(reg_src_prf, src_step);
                add(reg_tr_src_prf, tr_src_step);
                sub(reg_loop, 1);
                jnz(loop);
            }
        }
        if (transposes > 1)
            transpose(tail, 0, right_pad, nontemporal_stores);
        else
            transpose(tail, left_pad, right_pad, nontemporal_stores);
    }
    postamble();
}

struct jit_trans_ow_oc_t : public jit_trans_dst_t, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_trans_ow_oc_t)
    jit_trans_ow_oc_t(const jit_conv_conf_t *conf)
        : jit_trans_dst_t(conf), jit_generator(jit_name()) {}

    void operator()(ctx_t *ctx) override { jit_generator::operator()(ctx); }

    status_t create_kernel() override { return jit_generator::create_kernel(); }

private:
    using reg64_t = const Xbyak::Reg64;
    using reg32_t = const Xbyak::Reg32;
    using opmask_t = const Xbyak::Opmask;
    using zmm = const Xbyak::Zmm;

    enum {
        typesize = sizeof(int16_t),
        transpose_size = 16,
        small_spatial = 14
    };
    size_t src_stride = 0, tr_src_stride = 0;
    int tail = 0;
    bool enable_prefetch = false;

    opmask_t kFF = k1;
    opmask_t mask_lo = k2;
    opmask_t k_oc_tail = k3;

    zmm vidx1 = zmm31;
    zmm vidx2 = zmm30;

    reg64_t reg_src = r8;
    reg64_t reg_tr_src = r9;
    reg64_t reg_src_prf = r10;
    reg64_t reg_tr_src_prf = r11;
    reg64_t reg_loop = r12;
    reg64_t reg_tr_src_tmp = r13;
    reg32_t regw_tmp = r14d;
    reg64_t imm_addr64 = rbx;

    void transpose(int nrows, int l_pad, int r_pad, bool nontemporal_stores,
            bool do_convert = true);
    void generate() override;
};

// do_convert (default is 'true') is a flag that determines when to do the
// transformation of the input data and when to simply zero out the output data
void jit_trans_ow_oc_t::transpose(int nrows, int l_pad, int r_pad,
        bool nontemporal_stores, bool do_convert) {
    assert(nrows >= 0 && nrows <= transpose_size);
    static_assert(transpose_size == 16, "Unsupported transpose size");
    if (!nrows) return;

    auto src_zmm = [](int i) { return Zmm(i); };

    auto src_ymm = [](int i) {
        assert(i >= 0 && i < 16);
        return Ymm(i);
    };

    auto load_ymm = [this, src_ymm](int i) {
        auto ymm_reg = src_ymm(i);
        auto addr = EVEX_compress_addr(reg_src, i * src_stride);
        if (conf_->oc_tail) {
            ymm_reg = ymm_reg | k_oc_tail | T_z;
            // Assertion below as we need vmovdqu16 for tails.
            // If needed, can be removed by using load_bytes() helper.
            assert(mayiuse(avx512_core));
            vmovdqu16(ymm_reg, addr);
        } else {
            vmovups(ymm_reg, addr);
        }
    };

    auto store = [this, nontemporal_stores](Zmm r, int i) {
        auto addr = EVEX_compress_addr(reg_tr_src, i * tr_src_stride);
        if (nontemporal_stores)
            vmovntps(addr, r);
        else
            vmovups(addr, r);
    };
    const bool is_layout_nxc = utils::one_of(conf_->dst_tag, format_tag::ndhwc,
            format_tag::nhwc, format_tag::nwc);

    if (mayiuse(avx512_core) && !is_layout_nxc) {
        // TODO: adopt for nhwc?
        for (int i = 0; i < nrows / 2; i++) {
            auto zmm_src0 = src_zmm(i);
            if (do_convert) {
                vmovdqu16(zmm_src0,
                        EVEX_compress_addr(reg_src, 2 * i * src_stride));
                vpermw(zmm_src0, vidx2, zmm_src0);
            } else {
                vpxord(zmm_src0, zmm_src0, zmm_src0);
            }
            store(zmm_src0, 2 * i);
        }
        if (r_pad > 0) {
            auto zmm_src0 = src_zmm(29);
            if (do_convert) {
                vmovdqu16(zmm_src0 | mask_lo | T_z,
                        EVEX_compress_addr(reg_src, (nrows - 1) * src_stride));
                vpermw(zmm_src0, vidx2, zmm_src0);
            } else {
                vpxord(zmm_src0, zmm_src0, zmm_src0);
            }
            store(zmm_src0, nrows - 1);
        }
    } else {
        for (int i = 0; i < nrows / 2; i++) {
            auto src0 = src_ymm(2 * i);
            auto src1 = src_ymm(2 * i + 1);
            auto zmm_src0 = src_zmm(2 * i);
            if (do_convert) {
                load_ymm(2 * i);
                if (is_layout_nxc && conf_->oc_tail) {
                    load_ymm(2 * i + 1);
                    auto ymm_tmp = Ymm(30);
                    vpunpcklwd(ymm_tmp, src0, src1);
                    vpunpckhwd(src0, src0, src1);
                    vinserti64x4(zmm_src0, zmm_src0, ymm_tmp, 1);
                } else {
                    vpunpcklwd(src1, src0,
                            EVEX_compress_addr(
                                    reg_src, (2 * i + 1) * src_stride));
                    vpunpckhwd(src0, src0,
                            EVEX_compress_addr(
                                    reg_src, (2 * i + 1) * src_stride));
                    vinserti64x4(zmm_src0, zmm_src0, src1, 1);
                }
                vpermpd(zmm_src0 | kFF, vidx1, zmm_src0);
            } else {
                vpxord(zmm_src0, zmm_src0, zmm_src0);
            }
            store(zmm_src0, 2 * i);
        }
        if (r_pad > 0) {
            auto src0 = src_ymm(nrows - 1);
            auto src1 = src_ymm(nrows);
            auto zmm_src0 = src_zmm(30);
            if (do_convert) {
                load_ymm(nrows - 1);

                vpxor(src1, src1, src1);
                vpunpckhwd(src1, src0, src1);
                vinserti64x4(zmm_src0, zmm_src0, src1, 0);
                vpxor(src1, src1, src1);
                vpunpcklwd(src0, src0, src1);
                vinserti64x4(zmm_src0, zmm_src0, src0, 1);
                vpermpd(zmm_src0 | kFF, vidx1, zmm_src0);
            } else {
                vpxord(zmm_src0, zmm_src0, zmm_src0);
            }
            store(zmm_src0, nrows - 1);
        }
    }
}

void jit_trans_ow_oc_t::generate() {
    preamble();

    alignas(64) static constexpr const int64_t idx1[8]
            = {4, 5, 0, 1, 6, 7, 2, 3};
    alignas(64) static constexpr const int16_t idx2[32]
            = {0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23, 8, 24, 9,
                    25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31};

    const int oc_block = conf_->oc_block;
    const bool is_layout_nxc = utils::one_of(conf_->dst_tag, format_tag::ndhwc,
            format_tag::nhwc, format_tag::nwc);
    const size_t src_mult
            = is_layout_nxc ? conf_->ngroups * conf_->oc : oc_block;
    const int ow = conf_->ow;
    const int transposes = utils::div_up(ow, transpose_size);
    int loop_iters = nstl::max(0, transposes - 1);
    tail = ow - loop_iters * transpose_size;

    src_stride = src_mult * typesize;
    tr_src_stride = oc_block * typesize;

    bool nontemporal_stores = conf_->use_nt_stores_ddst;
    enable_prefetch = ow > small_spatial;

    const size_t src_step = src_mult * transpose_size * typesize;
    const size_t tr_src_step = (size_t)oc_block * transpose_size * typesize;
    const int right_pad = ow % 2;

    const auto zero_tr_ow = nstl::max(0, conf_->tr_ow - ow - right_pad);

    mov(reg_src, ptr[param1 + GET_OFF(src)]);
    mov(reg_tr_src, ptr[param1 + GET_OFF(tr_src)]);
    mov(reg_src_prf, ptr[param1 + GET_OFF(src_prf)]);
    mov(reg_tr_src_prf, ptr[param1 + GET_OFF(tr_src_prf)]);

    auto kmovw = [this](Opmask k, unsigned w) {
        mov(regw_tmp, w);
        jit_generator::kmovw(k, regw_tmp);
    };
    auto kmovd = [this](Opmask k, unsigned w) {
        mov(regw_tmp, w);
        jit_generator::kmovd(k, regw_tmp);
    };

    kmovw(kFF, 0xFF);
    kmovd(mask_lo, 0x0000ffff);

    if (is_layout_nxc && conf_->oc_tail) {
        Label done;
        kxnorw(k_oc_tail, k_oc_tail, k_oc_tail);
        cmp(dword[param1 + GET_OFF(ch_work)], conf_->oc_block);
        je(done, T_NEAR);
        kmovw(k_oc_tail, (1 << conf_->oc_tail) - 1);
        L(done);
    }

    auto vmovdqa64 = [this](Zmm z, const int64_t *addr) {
        mov(imm_addr64, reinterpret_cast<size_t>(addr));
        jit_generator::vmovdqa64(z, ptr[imm_addr64]);
    };

    vmovdqa64(vidx1, idx1);
    vmovdqa64(vidx2, (const int64_t *)idx2);
    if (loop_iters) {
        mov(reg_loop, loop_iters);
        Label loop;
        L(loop);
        {
            transpose(transpose_size, 0, 0, nontemporal_stores);
            add(reg_src, src_step);
            add(reg_tr_src, tr_src_step);
            add(reg_src_prf, src_step);
            add(reg_tr_src_prf, tr_src_step);
            sub(reg_loop, 1);
            jnz(loop);
        }
    }
    transpose(tail, 0, right_pad, nontemporal_stores);
    if (zero_tr_ow) {
        const auto zero_transposes = utils::div_up(zero_tr_ow, transpose_size);
        const auto zero_loop_iters = nstl::max(0, zero_transposes - 1);
        const auto zero_tail = zero_tr_ow - zero_loop_iters * transpose_size;
        const auto zero_right_pad = zero_tr_ow % 2;

        // shift over tail
        auto tr_src_tail_step
                = (size_t)oc_block * (tail + right_pad) * typesize;
        add(reg_tr_src, tr_src_tail_step);
        add(reg_tr_src_prf, tr_src_tail_step);

        // zero the tr_ow - ow
        if (zero_loop_iters) {
            mov(reg_loop, zero_loop_iters);
            Label zero_loop;
            L(zero_loop);
            {
                transpose(transpose_size, 0, 0, nontemporal_stores, false);
                add(reg_tr_src, tr_src_step);
                add(reg_tr_src_prf, tr_src_step);
                sub(reg_loop, 1);
                jnz(zero_loop);
            }
        }
        transpose(zero_tail, 0, zero_right_pad, nontemporal_stores, false);
    }

    postamble();
}

/*
// -------------------------------------------------
// jit_transpose4x16_src
// -------------------------------------------------
*/

void jit_transpose4x16_src::transpose(int nrows) {
    assert(nrows >= 0 && nrows <= transpose_size);
    static_assert(transpose_size == 4, "Unsupported transpose size");
    if (!nrows) return;

    auto pf_src_t0 = [this](int i) {
        if (tparams->src_pf0_distance)
            prefetcht0(EVEX_compress_addr(
                    reg_src, (tparams->src_pf0_distance + i) * src_stride));
    };

    auto pf_tr_src_t0 = [this](int i) {
        if (tparams->tr_src_pf0_distance)
            prefetcht0(EVEX_compress_addr(reg_tr_src,
                    (tparams->tr_src_pf0_distance + i) * src_stride));
    };

    auto pf_src_t1 = [this](int i) {
        if (tparams->src_pf1)
            prefetcht1(EVEX_compress_addr(reg_src_prf, i * src_stride));
    };

    auto pf_tr_src_t1 = [this](int i) {
        if (tparams->tr_src_pf1)
            prefetchwt1(EVEX_compress_addr(reg_tr_src_prf, i * tr_src_stride));
    };

    auto src_zmm = [](int i) {
        assert(i >= 0 && i < 4);
        return Zmm(i);
    };

    auto tmp_zmm = [](int i) {
        assert(i >= 0 && i < 4);
        return Zmm(4 + i);
    };

    auto load = [this, src_zmm](int i) {
        vmovups(src_zmm(i), EVEX_compress_addr(reg_src, i * src_stride));
    };

    auto store = [this](Zmm r, int i) {
        vmovups(EVEX_compress_addr(reg_tr_src, i * tr_src_stride), r);
    };

    auto tmp0 = tmp_zmm(0);
    auto tmp1 = tmp_zmm(1);
    auto tmp2 = tmp_zmm(2);
    auto tmp3 = tmp_zmm(3);

    auto src0 = src_zmm(0);
    auto src1 = src_zmm(1);
    auto src2 = src_zmm(2);
    auto src3 = src_zmm(3);
    for (int i = 0; i < nrows; i++) {
        load(i);
    }

    for (size_t i = nrows; i < 4; i++) {
        vpxord(src_zmm(i), src_zmm(i), src_zmm(i));
    }

    vmovupd(tmp0, src0);
    vmovupd(tmp1, src1);
    pf_src_t0(0);
    vpermpd(tmp0 | kF0, vidx01, src2);
    vpermpd(tmp1 | kF0, vidx01, src3);

    valignd(src0, src0, src0, 8);
    valignd(src1, src1, src1, 8);
    pf_src_t0(1);
    vmovupd(tmp2, src0);
    vmovupd(tmp3, src1);
    pf_src_t0(2);
    vpermpd(tmp2 | kF0, vidx10, src2);
    vpermpd(tmp3 | kF0, vidx10, src3);
    pf_src_t0(3);

    vmovupd(src0, tmp0);
    pf_src_t1(0);
    vmovupd(src1, tmp2);
    pf_src_t1(1);
    vmovupd(src2, tmp1);
    pf_src_t1(2);
    vmovupd(src3, tmp3);
    pf_src_t1(3);
    vpermpd(src0 | kCC, vidx1, tmp1);
    vpermpd(src1 | kCC, vidx1, tmp3);
    pf_tr_src_t0(0);
    vpermpd(src2 | k33, vidx1, tmp0);
    vpermpd(src3 | k33, vidx1, tmp2);
    pf_tr_src_t0(1);

    vmovupd(tmp0, src0);
    vmovupd(tmp1, src2);
    pf_tr_src_t0(2);
    vmovupd(tmp2, src1);
    vmovupd(tmp3, src3);
    pf_tr_src_t0(3);
    vpermps(tmp0 | kFFFF, vidxP, src0);
    pf_tr_src_t1(0);
    vpermps(tmp1 | kFFFF, vidxP, src2);
    pf_tr_src_t1(1);
    vpermps(tmp2 | kFFFF, vidxP, src1);
    pf_tr_src_t1(3);
    vpermps(tmp3 | kFFFF, vidxP, src3);
    pf_tr_src_t1(4);

    store(tmp0, 0);
    store(tmp1, 1);
    store(tmp2, 2);
    store(tmp3, 3);
}

alignas(64) static constexpr const int64_t idx01[8] = {0, 0, 0, 0, 0, 1, 2, 3};
alignas(64) static constexpr const int64_t idx10[8] = {0, 0, 0, 0, 4, 5, 6, 7};
alignas(64) static constexpr const int64_t idx1[8] = {2, 3, 0, 1, 6, 7, 4, 5};
alignas(64) static constexpr const int32_t idxP[16]
        = {0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15};

void jit_transpose4x16_src::generate() {
    preamble();

    const int ic_block = params->ic_block;
    const int is = params->is;
    int tail = is % transpose_size;

    src_stride = ic_block * typesize;
    assert(src_stride == 64);
    tr_src_stride = ic_block * typesize;

    const int src_step = ic_block * transpose_size * typesize;
    const int tr_src_step = ic_block * transpose_size * typesize;

#define GET_TR_OFF(x) offsetof(jit_src_transpose_s, x)
    mov(reg_loop, ptr[param1 + GET_TR_OFF(size)]);
    mov(reg_src, ptr[param1 + GET_TR_OFF(src)]);
    mov(reg_tr_src, ptr[param1 + GET_TR_OFF(tr_src)]);
    mov(reg_src_prf, ptr[param1 + GET_TR_OFF(src_prf)]);
    mov(reg_tr_src_prf, ptr[param1 + GET_TR_OFF(tr_src_prf)]);
#undef GET_TR_OFF

    auto kmovw = [this](Opmask k, unsigned w) {
        mov(regw_tmp, w);
        jit_generator::kmovw(k, regw_tmp);
    };

    auto vmovdqa64 = [this](Zmm z, const int64_t *addr) {
        mov(imm_addr64, reinterpret_cast<size_t>(addr));
        jit_generator::vmovdqa64(z, ptr[imm_addr64]);
    };

    auto vmovdqa32 = [this](Zmm z, const int32_t *addr) {
        mov(imm_addr64, reinterpret_cast<size_t>(addr));
        jit_generator::vmovdqa32(z, ptr[imm_addr64]);
    };

    kmovw(kF0, 0xf0); // 11110000
    kmovw(kCC, 0xcc); // 11001100
    kmovw(k33, 0x33); // 00110011
    kmovw(kFFFF, 0xffff); // 1111111111111111

    vmovdqa64(vidx01, idx01);
    vmovdqa64(vidx10, idx10);
    vmovdqa64(vidx1, idx1);
    vmovdqa32(vidxP, idxP);

    Label loop_label;
    Label tail_label;

    cmp(reg_loop, transpose_size);
    jl(tail_label, T_NEAR);

    L(loop_label);
    {
        transpose(transpose_size);
        add(reg_src, src_step);
        add(reg_tr_src, tr_src_step);
        add(reg_src_prf, src_step);
        add(reg_tr_src_prf, tr_src_step);
        sub(reg_loop, transpose_size);
        cmp(reg_loop, transpose_size);
        jge(loop_label, T_NEAR);
    }
    L(tail_label);
    transpose(tail);

    postamble();
}

#undef GET_OFF

#define GET_OFF(field) offsetof(jit_conv_call_s, field)

void jit_diff_wei_trans_to_vnni_t::generate() {
    /* Reorder part of F32 weights tensor
       from [2I][kd][kh][kw][16i][16o] to VNNI format [kd][kh][kw][16i][16o][2i]
       and downconvert it to Bfloat16. */
    const int typesize_out = 2;
    const int typesize_acc = 4;
    const int simd_w = 16;

    using reg64_t = const Xbyak::Reg64;
    const reg64_t &reg_output = r15;
    const reg64_t &org_reg_output = r14;
    const reg64_t &reg_input = r13;
    const reg64_t &reg_input_1 = r12;
    const reg64_t &org_reg_input_1 = r11;
    const reg64_t &reg_input_2 = r10;
    const reg64_t &reg_prm_table = r9;
    const reg64_t &reg_last_ic_block = rax;
    const reg64_t &reg_kd = rsi;
    const reg64_t &reg_kh = abi_not_param1;
    const reg64_t &reg_tmp = rdx;

    const Xbyak::Zmm &zmm_idx = Xbyak::Zmm(31);
    auto get_zmm_src_0 = [&](int ic) { return Xbyak::Zmm(ic); };
    auto get_zmm_src_1 = [&](int ic) { return Xbyak::Zmm(4 + ic); };
    auto get_zmm_bf16 = [&](int ic) { return Xbyak::Zmm(8 + ic); };

    Xbyak::Label prm_table, zero_buffer;
    Xbyak::Label kd_loop_label, kh_loop_label;

    preamble();

    mov(reg_last_ic_block, ptr[abi_param1 + GET_OFF(last_ic_block)]);
    mov(org_reg_input_1, ptr[abi_param1 + GET_OFF(src)]);
    mov(org_reg_output, ptr[abi_param1 + GET_OFF(dst)]);

    mov(reg_prm_table, prm_table);
    vmovups(zmm_idx, ptr[reg_prm_table]);

    xor_(reg_kd, reg_kd);
    L(kd_loop_label);
    {
        mov(reg_output, org_reg_output);
        mov(reg_input_1, org_reg_input_1);
        xor_(reg_kh, reg_kh);
        L(kh_loop_label);
        {
            for (int kw = 0; kw < kw_; kw++) {
                Xbyak::Label last_ic_label, done_ic_label;

                dim_t out_offset
                        = (dim_t)typesize_out * kw * ic_block_ * oc_block_ * 2;
                dim_t inp_1_offset
                        = (dim_t)typesize_acc * kw * ic_block_ * oc_block_;
                dim_t inp_2_offset = (dim_t)typesize_acc
                        * (kd_ * kh_ * kw_ * ic_block_ * oc_block_
                                + kw * ic_block_ * oc_block_);

                cmp(reg_last_ic_block, 0);
                jne(last_ic_label, T_NEAR);

                mov(reg_input_2, reg_input_1);
                safe_add(reg_input_2, inp_2_offset, reg_tmp);
                jmp(done_ic_label, T_NEAR);

                L(last_ic_label);
                mov(reg_input_2, zero_buffer);

                L(done_ic_label);

                for (int ocb = 0; ocb < oc_block_; ocb += simd_w) {
                    int ic_count = 0;
                    for (int bc = 0; bc < 2; bc++) {
                        if (!bc) {
                            mov(reg_input, reg_input_1);
                            safe_add(reg_input, inp_1_offset, reg_tmp);
                        } else
                            mov(reg_input, reg_input_2);

                        for (int ic = 0; ic < ic_block_ / 2; ic++) {
                            auto zmm_src_0 = get_zmm_src_0(ic);
                            auto zmm_src_1 = get_zmm_src_1(ic);
                            auto zmm_out = get_zmm_bf16(ic);

                            vmovups(zmm_src_0,
                                    ptr[reg_input
                                            + typesize_acc
                                                    * ((2 * ic + 0) * oc_block_
                                                            + ocb)]);
                            vmovups(zmm_src_1,
                                    ptr[reg_input
                                            + typesize_acc
                                                    * ((2 * ic + 1) * oc_block_
                                                            + ocb)]);
                            if (out_dt_ == data_type::bf16) {
                                vcvtne2ps2bf16(zmm_out, zmm_src_1, zmm_src_0);
                            } else if (out_dt_ == data_type::f16) {
                                vcvtps2phx(Ymm(zmm_src_0.getIdx()), zmm_src_0);
                                vcvtps2phx(Ymm(zmm_src_1.getIdx()), zmm_src_1);
                                vinsertf32x8(zmm_out, zmm_src_0,
                                        Ymm(zmm_src_1.getIdx()), 1);
                            } else {
                                assert(!"unsupported data type");
                            }
                            vpermw(zmm_out, zmm_idx, zmm_out);

                            vmovups(ptr[reg_output + out_offset
                                            + typesize_out
                                                    * (ic_count * oc_block_ * 2
                                                            + ocb * 2)],
                                    zmm_out);
                            ic_count++;
                        }
                    }
                }
            }
            safe_add(reg_output,
                    (dim_t)typesize_out * kw_ * 2 * ic_block_ * oc_block_,
                    reg_tmp);
            safe_add(reg_input_1,
                    (dim_t)typesize_acc * kw_ * ic_block_ * oc_block_, reg_tmp);

            add(reg_kh, 1);
            cmp(reg_kh, kh_);
            jl(kh_loop_label, T_NEAR);
        }
        safe_add(org_reg_output,
                (dim_t)typesize_out * kh_ * kw_ * 2 * ic_block_ * oc_block_,
                reg_tmp);
        safe_add(org_reg_input_1,
                (dim_t)typesize_acc * kh_ * kw_ * ic_block_ * oc_block_,
                reg_tmp);

        add(reg_kd, 1);
        cmp(reg_kd, kd_);
        jl(kd_loop_label, T_NEAR);
    }

    postamble();

    align(64);
    L(prm_table);
    const uint16_t prm_array[32]
            = {0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23, 8, 24, 9,
                    25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31};
    for (size_t i = 0; i < 32; ++i)
        dw(prm_array[i]);

    align(64);
    L(zero_buffer);
    const uint16_t zero = 0;
    for (int i = 0; i < typesize_acc * oc_block_ * ic_block_; ++i)
        db(zero);
}

#undef GET_OFF

jit_trans_src_t *create_trans_src(const jit_conv_conf_t *conf) {
    if (conf->has_vnni && IMPLICATION(conf->is_1stconv, conf->transpose_src))
        return new jit_trans_iw_ic_int16_t(conf);
    assert(!"unsupported configuration");
    return nullptr;
}

jit_trans_dst_t *create_trans_dst(const jit_conv_conf_t *conf) {
    if (conf->has_vnni) return new jit_trans_ow_oc_t(conf);
    assert(!"unsupported configuration");
    return nullptr;
}
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
