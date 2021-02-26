/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "dnnl_types.h"

#include "common/bfloat16.hpp"
#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/x64/jit_brgemm_conv_utils.hpp"
#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::status;
using namespace dnnl::impl::format_tag;
using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;

using namespace prop_kind;
using namespace data_type;

namespace brgemm_convolution_utils {

inline status_t init_tag(format_tag_t &tag, memory_desc_t &md,
        const memory_desc_wrapper &mdw, const format_tag_t tag_value,
        bool any_eligible) {

    if (mdw.format_kind() == format_kind::any) {
        if (any_eligible) {
            CHECK(memory_desc_init_by_tag(md, tag_value));
            tag = tag_value;
        } else {
            tag = format_tag::undef;
        }
    } else {
        tag = mdw.matches_one_of_tag(tag_value);
    }

    if (tag != tag_value) return status::unimplemented;

    return status::success;
}

bool post_ops_ok(jit_brgemm_conv_conf_t &jcp, const primitive_attr_t &attr,
        const memory_desc_wrapper &dst_d) {
    using namespace injector;

    const auto &post_ops = attr.post_ops_;

    return injector::post_ops_ok(post_ops_ok_args_t(avx512_common,
            {sum, eltwise, binary}, post_ops, &dst_d,
            false /*sum_at_pos_0_only*/, false /*sum_requires_scale_one*/,
            {broadcasting_strategy_t::per_oc,
                    broadcasting_strategy_t::scalar}));
}

status_t pick_tags(jit_brgemm_conv_conf_t &jcp, memory_desc_t &src_md,
        memory_desc_t &weights_md, memory_desc_t &dst_md,
        memory_desc_t &bias_md) {
    format_tag_t src_tag, dst_tag, wei_tag;
    dst_tag = pick(jcp.ndims - 3, nwc, nhwc, ndhwc);

    const memory_desc_wrapper src_d(&src_md);
    const memory_desc_wrapper weights_d(&weights_md);
    const memory_desc_wrapper dst_d(&dst_md);
    const memory_desc_wrapper bias_d(&bias_md);
    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;

    const bool is_1d = jcp.ndims == 3;
    const bool is_2d = jcp.ndims == 4;
    const bool is_3d = jcp.ndims == 5;

    if (jcp.wei_plain) {
        jcp.LDB = jcp.oc;
        if (is_3d) {
            if (jcp.wei_dt == f32)
                wei_tag = with_groups ? gdhwio : dhwio;
            else if (jcp.wei_dt == s8)
                wei_tag = with_groups ? gdhwIo4i : dhwIo4i;
            else if (jcp.wei_dt == bf16) {
                wei_tag = with_groups ? gdhwIo2i : dhwIo2i;
            } else
                return status::unimplemented;
        } else if (is_1d) {
            if (jcp.wei_dt == f32)
                wei_tag = with_groups ? gwio : wio;
            else if (jcp.wei_dt == s8)
                wei_tag = with_groups ? gwIo4i : wIo4i;
            else if (jcp.wei_dt == bf16) {
                wei_tag = with_groups ? gwIo2i : wIo2i;
            } else
                return status::unimplemented;
        } else {
            assert(is_2d);
            UNUSED(is_2d);
            if (jcp.wei_dt == f32)
                wei_tag = with_groups ? ghwio : hwio;
            else if (jcp.wei_dt == s8)
                wei_tag = with_groups ? ghwIo4i : hwIo4i;
            else if (jcp.wei_dt == bf16) {
                wei_tag = with_groups ? ghwIo2i : hwIo2i;
            } else
                return status::unimplemented;
        }
    } else {
        jcp.LDB = jcp.oc_block;
        if (jcp.oc_block == 64) {
            if (is_3d) {
                if (jcp.wei_dt == f32)
                    wei_tag = with_groups ? gOdhwi64o : Odhwi64o;
                else if (jcp.wei_dt == s8)
                    wei_tag = with_groups ? gOdhwI64o4i : OdhwI64o4i;
                else if (jcp.wei_dt == bf16) {
                    wei_tag = with_groups ? gOdhwI64o2i : OdhwI64o2i;
                } else
                    return status::unimplemented;
            } else if (is_1d) {
                if (jcp.wei_dt == f32)
                    wei_tag = with_groups ? gOwi64o : Owi64o;
                else if (jcp.wei_dt == s8)
                    wei_tag = with_groups ? gOwI64o4i : OwI64o4i;
                else if (jcp.wei_dt == bf16) {
                    wei_tag = with_groups ? gOwI64o2i : OwI64o2i;
                } else
                    return status::unimplemented;
            } else {
                assert(is_2d);
                UNUSED(is_2d);
                if (jcp.wei_dt == f32)
                    wei_tag = with_groups ? gOhwi64o : Ohwi64o;
                else if (jcp.wei_dt == s8)
                    wei_tag = with_groups ? gOhwI64o4i : OhwI64o4i;
                else if (jcp.wei_dt == bf16) {
                    wei_tag = with_groups ? gOhwI64o2i : OhwI64o2i;
                } else
                    return status::unimplemented;
            }
        } else if (jcp.oc_block == 48) {
            if (is_3d) {
                if (jcp.wei_dt == f32)
                    wei_tag = with_groups ? gOdhwi48o : Odhwi48o;
                else if (jcp.wei_dt == s8)
                    wei_tag = with_groups ? gOdhwI48o4i : OdhwI48o4i;
                else if (jcp.wei_dt == bf16) {
                    wei_tag = with_groups ? gOdhwI48o2i : OdhwI48o2i;
                } else
                    return status::unimplemented;
            } else if (is_1d) {
                if (jcp.wei_dt == f32)
                    wei_tag = with_groups ? gOwi48o : Owi48o;
                else if (jcp.wei_dt == s8)
                    wei_tag = with_groups ? gOwI48o4i : OwI48o4i;
                else if (jcp.wei_dt == bf16) {
                    wei_tag = with_groups ? gOwI48o2i : OwI48o2i;
                } else
                    return status::unimplemented;
            } else {
                assert(is_2d);
                UNUSED(is_2d);
                if (jcp.wei_dt == f32)
                    wei_tag = with_groups ? gOhwi48o : Ohwi48o;
                else if (jcp.wei_dt == s8)
                    wei_tag = with_groups ? gOhwI48o4i : OhwI48o4i;
                else if (jcp.wei_dt == bf16) {
                    wei_tag = with_groups ? gOhwI48o2i : OhwI48o2i;
                } else
                    return status::unimplemented;
            }
        } else if (jcp.oc_block == 32) {
            if (is_3d) {
                if (jcp.wei_dt == f32)
                    wei_tag = with_groups ? gOdhwi32o : Odhwi32o;
                else if (jcp.wei_dt == s8)
                    wei_tag = with_groups ? gOdhwI32o4i : OdhwI32o4i;
                else if (jcp.wei_dt == bf16) {
                    wei_tag = with_groups ? gOdhwI32o2i : OdhwI32o2i;
                } else
                    return status::unimplemented;
            } else if (is_1d) {
                if (jcp.wei_dt == f32)
                    wei_tag = with_groups ? gOwi32o : Owi32o;
                else if (jcp.wei_dt == s8)
                    wei_tag = with_groups ? gOwI32o4i : OwI32o4i;
                else if (jcp.wei_dt == bf16) {
                    wei_tag = with_groups ? gOwI32o2i : OwI32o2i;
                } else
                    return status::unimplemented;
            } else {
                assert(is_2d);
                UNUSED(is_2d);
                if (jcp.wei_dt == f32)
                    wei_tag = with_groups ? gOhwi32o : Ohwi32o;
                else if (jcp.wei_dt == s8)
                    wei_tag = with_groups ? gOhwI32o4i : OhwI32o4i;
                else if (jcp.wei_dt == bf16) {
                    wei_tag = with_groups ? gOhwI32o2i : OhwI32o2i;
                } else
                    return status::unimplemented;
            }
        } else {
            if (is_3d) {
                if (jcp.wei_dt == f32)
                    wei_tag = with_groups ? gOdhwi16o : Odhwi16o;
                else if (jcp.wei_dt == s8)
                    wei_tag = with_groups ? gOdhwI16o4i : OdhwI16o4i;
                else if (jcp.wei_dt == bf16)
                    wei_tag = with_groups ? gOdhwI16o2i : OdhwI16o2i;
                else
                    return status::unimplemented;
            } else if (is_1d) {
                if (jcp.wei_dt == f32)
                    wei_tag = with_groups ? gOwi16o : Owi16o;
                else if (jcp.wei_dt == s8)
                    wei_tag = with_groups ? gOwI16o4i : OwI16o4i;
                else if (jcp.wei_dt == bf16)
                    wei_tag = with_groups ? gOwI16o2i : OwI16o2i;
                else
                    return status::unimplemented;
            } else {
                assert(is_2d);
                UNUSED(is_2d);

                if (jcp.wei_dt == f32)
                    wei_tag = with_groups ? gOhwi16o : Ohwi16o;
                else if (jcp.wei_dt == s8)
                    wei_tag = with_groups ? gOhwI16o4i : OhwI16o4i;
                else if (jcp.wei_dt == bf16)
                    wei_tag = with_groups ? gOhwI16o2i : OhwI16o2i;
                else
                    return status::unimplemented;
            }
        }
    }

    src_tag = dst_tag;

    const bool any_eligible = (jcp.prop_kind == prop_kind::forward_inference);
    CHECK(init_tag(jcp.src_tag, src_md, src_d, src_tag, any_eligible));
    CHECK(init_tag(jcp.dst_tag, dst_md, dst_d, dst_tag, any_eligible));
    CHECK(init_tag(jcp.wei_tag, weights_md, weights_d, wei_tag, true));

    return status::success;
}

struct brg_blocking_t : public jit_brgemm_conv_conf_t {
    struct array_in_loop_t {
        dim_t itersize;
        float repeatn;
        float overlap;
        void set(dim_t iter_s, float rpt, float ovlp = 1.f) {
            itersize = iter_s;
            repeatn = rpt;
            overlap = ovlp;
        }
    };

    struct loop_t {
        array_in_loop_t src;
        array_in_loop_t wei;
        array_in_loop_t dst;
    };

    brg_blocking_t() : jit_brgemm_conv_conf_t() { init(); }
    brg_blocking_t(const jit_brgemm_conv_conf_t &jcp)
        : jit_brgemm_conv_conf_t(jcp) {
        init();
    }
    void init() {
        ur = 0;
        eff = 0.f;
        nb_kd = 0;
        nb_kh = 0;
        nb_kw = 0;
        is_os_block = false;
        sp = 0;
        sp_block = 0;
        nb_sp = 0;
        eff = 0;
    }

    int ur;
    int nb_kd, nb_kh, nb_kw;
    float eff;
    bool is_os_block;
    static unsigned L1;
    static unsigned L2;
    static unsigned L3;
    static cpu_isa_t isa;
    // These are rough estimates of the latency (relative) of access to various
    // cache levels. This is enough for an estimation of data access cost.
    // TODO: Improve memory access estimates
    static constexpr float L1_k = 1.f;
    static constexpr float L2_k = 3.f;
    static constexpr float L3_k = 15.f;
    // TODO: At the moment, we are primarily evaluating the fit of the data into
    // the L1/L2. Need to take into account the difference between the L3 and
    // memory.
    static constexpr float mem_k = 15.f;
    static constexpr int bench_iterations = 1;
    static constexpr int max_regs = 32;
    static constexpr int bcast_simd = 16;

    int sp, sp_block, nb_sp;

    void get_from_jcp(const jit_brgemm_conv_conf_t &jcp) { *this = jcp; }
    void save_to_jcp(jit_brgemm_conv_conf_t &jcp) const { jcp = *this; }

    int estimate_brgemm_ur(int spb) const;
    int get_brgemm_ur(const primitive_attr_t *attr, const memory_desc_t &dst_md,
            bool is_1x1) const;

    float io_k(dim_t src, dim_t wei, dim_t dst, float n, float pk,
            bool is_broadcast, bool is_shared) const;

    float io_k(const loop_t loop, const array_in_loop_t arr, float pk,
            bool is_broadcast, bool is_shared) const;

    void select_ic_block();

    void update_blocks();
    bool fast_check_oc_block() const;
    float est_eff();
    void iterate_ker_block(brg_blocking_t &best_brgb, int kd_block,
            int kh_block, bool maybe_use_buffer, int max_ow_block_thr);
    void calc_blocks();

    bool fast_check_oc_block_1x1() const;
    float est_eff_1x1();
    void calc_blocks_1x1();

    // utils
    static int get_inp_size(
            int max_src_size, int dst_size, int k, int stride, int dilate) {
        auto adj_str = nstl::min(k, stride);
        const auto res = nstl::min(max_src_size,
                calculate_end_padding(0, dst_size, 0, adj_str,
                        calculate_extended_filter_size(k, dilate)));
        return res;
    }

    static float squeeze_val(float eff, float koeff) {
        if (koeff <= 0) return 1;
        if (koeff == 1) return eff;
        const auto k = 1.f / koeff;
        return (k > 1.f) ? (k - 1 + eff) / k : eff * koeff;
    }

    static int estimate_ur(int oc_block) {
        const auto est_ur = (oc_block == 64)
                ? 6
                : ((oc_block == 48) ? 9 : ((oc_block == 32) ? 14 : 28));
        return est_ur;
    }

    int inp_w(int out_w, int ker_w) const {
        return get_inp_size(iw, out_w, ker_w, stride_w, dilate_w);
    }

    int rnd_simd(int val) const { return rnd_up(val, simd_w); }

    int rnd_inp_simd(int out_w, int ker_w, int vic) const {
        const auto vsp = inp_w(out_w, ker_w);
        return ((stride_w == 1 && vic >= ic) ? rnd_up(vsp * vic, simd_w)
                                             : vsp * rnd_up(vic, simd_w));
    }

    static constexpr int MAXNLOOPS = 32;
    loop_t loop[MAXNLOOPS];
};

unsigned brg_blocking_t::L1;
unsigned brg_blocking_t::L2;
unsigned brg_blocking_t::L3;
cpu_isa_t brg_blocking_t::isa;

float brg_blocking_t::io_k(dim_t src, dim_t wei, dim_t dst, float n, float pk,
        bool is_broadcast, bool is_shared) const {
    if (n < 1) return 0;
    if (n == 1) return pk;
    const auto amount = src * src_dsz + wei * wei_dsz + dst * dst_dsz
            + (use_buffer ? dst * acc_dsz : 0);
    const auto amount_L1 = is_broadcast ? src * src_dsz : amount;
    const auto k = is_broadcast
            ? ((amount_L1 < L1) ? L1_k
                                : ((amount < L2) ? L2_k
                                                 : (is_shared ? L3_k : mem_k)))
            : ((amount < L2) ? L2_k : (is_shared ? L3_k : mem_k));
    const auto cost = pk + k * (n - 1);
    return cost / n;
}

float brg_blocking_t::io_k(const loop_t loop, const array_in_loop_t arr,
        float pk, bool is_broadcast, bool is_shared) const {
    return io_k(loop.src.itersize, loop.wei.itersize, loop.dst.itersize,
            arr.repeatn * arr.overlap, pk, is_broadcast, is_shared);
}

void brg_blocking_t::select_ic_block() {
    auto max_simd_blocks = nstl::min(5 * simd_w, div_up(ic, simd_w));
    const auto est_ur = nstl::min(sp_block, estimate_ur(oc_block));
    const auto inp_ur = is_os_block ? est_ur : inp_w(est_ur, kw_block);

    if (kw_block > 1) {
        // try to fit src into L1
        const auto inp_per_ic = (unsigned int)inp_ur * src_dsz;
        max_simd_blocks = saturate(
                1, max_simd_blocks, (int)(L1 / (inp_per_ic * simd_w)));
    }
    {
        // try to fit all batch for ur into L2
        const auto wei_per_ic = (unsigned int)kd_block * kh_block * kw_block
                * oc_block * wei_dsz;
        const auto inp_per_ic
                = (unsigned int)kd_block * kh_block * inp_ur * src_dsz;
        const auto out_size = (unsigned int)ur * oc_block * dst_dsz;

        max_simd_blocks = saturate(1, max_simd_blocks,
                (int)((L2 - out_size) / ((wei_per_ic + inp_per_ic) * simd_w)));
    }

    auto nb_simd = utils::div_up(ic, simd_w);
    const auto nb_icb_eff_threshold = 0.5f;
    auto simd_blocks = 1;
    for (int nb_icb = nstl::min(max_simd_blocks, nb_simd); nb_icb >= 1;
            nb_icb--) {
        auto nb_icb_eff = (float)nb_simd / rnd_up(nb_simd, nb_icb);
        if (nb_icb_eff >= nb_icb_eff_threshold) {
            simd_blocks = nb_icb;
            break;
        }
    }
    ic_block = simd_blocks * simd_w;
    nb_ic = utils::div_up(ic, ic_block);
}

int brg_blocking_t::estimate_brgemm_ur(int spb) const {
    // Simple simulation of brgemm_desc init
    brgemm_t brg;
    const auto LDA = (exec_type == exec_trans) ? stride_w * ic_block
                                               : stride_w * ic_without_padding;
    const auto LDB = oc_block;
    const auto LDC = use_buffer ? oc_block : oc_without_padding;
    const auto N = oc >= oc_block ? oc_block : oc;
    const auto K = ic >= ic_block ? ic_block : ic;

    const float alpha = 1.0;
    const float beta = 0.0;
    auto status = brgemm_desc_init(&brg, isa, brgemm_addr, src_dt, wei_dt,
            false, false, brgemm_row_major, alpha, beta, LDA, LDB, LDC, spb, N,
            K);
    return status == success ? brg.bd_block : 0;
}

int brg_blocking_t::get_brgemm_ur(const primitive_attr_t *attr,
        const memory_desc_t &dst_md, bool is_1x1) const {
    // Detailed simulation of brgemm convolution init
    brgemm_t brg;
    const auto LDA = (exec_type == exec_trans) ? stride_w * ic_block
                                               : stride_w * ic_without_padding;
    const auto LDB = oc_block;
    const auto LDC = (use_buffer) ? oc_block : oc_without_padding;
    const auto LDD = oc_without_padding;

    const float alpha = 1.0;
    const float beta = 1.0;
    const float beta_init = 0.0;

    const auto M = sp_block;
    const auto M_tail = is_os_block ? os % sp_block : ow % sp_block;
    const auto K = ic >= ic_block ? ic_block : 0;
    const auto K_tail = ic % ic_block;
    const auto N = oc >= oc_block ? oc_block : 0;
    const auto N_tail = oc % oc_block;

    status_t status = success;
    int res_ur = estimate_brgemm_ur(is_os_block ? os_block : ow_block);

    for (int i = 0; i < M; i++) {
        auto vM = i + 1;
        // init only needed brgemm descriptors
        if ((utils::one_of(exec_type, exec_trans, exec_vpad) || is_1x1)
                && vM != M && vM != M_tail)
            continue;
        for (int i_init = 0; i_init < 2; i_init++) {
            for (int i_N = 0; i_N < 2; i_N++) {
                for (int i_K = 0; i_K < 2; i_K++) {
                    auto vbeta = (i_init) ? beta_init : beta;
                    auto vN = (i_N) ? N_tail : N;
                    auto vK = (i_K) ? K_tail : K;
                    if (vN == 0 || vK == 0) continue;
                    brgemm_strides_t brg_strides;
                    brg_strides.stride_a
                            = ic_without_padding * (dilate_w + 1) * src_dsz;
                    //weights are padded by oc_block and last_ic_block
                    const auto last_ic_block
                            = (wei_dt == f32) ? 1 : ((wei_dt == bf16) ? 2 : 4);
                    brg_strides.stride_b = rnd_up(ic, last_ic_block)
                            * rnd_up(oc, oc_block) * wei_dsz;
                    const auto strides_ptr = (brg_type == brgemm_strd)
                            ? &brg_strides
                            : nullptr;
                    status = brgemm_desc_init(&brg, isa, brg_type, src_dt,
                            wei_dt, false, false, brgemm_row_major, alpha,
                            vbeta, LDA, LDB, LDC, vM, vN, vK, strides_ptr);
                    if (status != success) break;

                    brgemm_attr_t brgattr;
                    brgattr.max_bs = max_batch;
                    const auto max_vpad = (exec_type == exec_vpad)
                            ? nstl::max(l_pad, r_pad)
                            : 0;
                    brgattr.max_top_vpad = max_vpad;
                    brgattr.max_bottom_vpad = max_vpad;
                    status = brgemm_desc_set_attr(&brg, brgattr);
                    if (status != success) break;

                    brg.with_sum = with_sum;
                    status = brgemm_desc_set_postops(
                            &brg, attr, &dst_md, LDD, bia_dt);
                    if (status != success) break;
                }
                if (status != success) break;
            }
            if (status != success) break;
        }
        if (status != success) break;
    }

    return status == success ? res_ur : 0;
}

void brg_blocking_t::update_blocks() {
    nb_od = div_up(od, od_blk_size);
    nb_oh = div_up(oh, oh_blk_size);
    nb_ic = div_up(ic, ic_block);
    nb_oc = div_up(oc, oc_block);
    nb_kd = div_up(kd, kd_block);
    nb_kh = div_up(kh, kh_block);
    nb_kw = div_up(kw, kw_block);
    if (is_os_block) {
        nb_os = div_up(os, os_block);
        sp = os;
        sp_block = os_block;
        nb_sp = nb_os;
    } else {
        nb_ow = div_up(ow, ow_block);
        sp = ow;
        sp_block = ow_block;
        nb_sp = nb_ow;
    }
}

bool brg_blocking_t::fast_check_oc_block() const {
    // This function for reducing the number of blocking variants
    // TODO: eliminate heuristic in this function
    const auto rnd_oc = rnd_up(oc, 16);
    auto res = false;
    if (oc_block == 64) {
        res = (rnd_oc % oc_block == 0 && rnd_oc * wei_dsz < 192 * 4);
    } else if (oc_block == 48) {
        const bool big_spatial
                = id * ih * iw > 81 * stride_d * stride_h * stride_w;
        res = (rnd_oc % oc_block == 0 && rnd_oc * wei_dsz <= 384 * 4
                && big_spatial);
    } else
        res = true;

    return res;
}

float brg_blocking_t::est_eff() {
    const auto ocblock = oc_block / 16;
    const auto od_block = od_blk_size;
    const auto oh_block = oh_blk_size;

    const auto brgemm_microkernel_eff
            = ((float)ocblock * ur) / ((ur + ocblock) * max_regs);

    const auto ur_eff = (float)sp_block / rnd_up(sp_block, ur);
    const auto brgemm_eff = squeeze_val(
            ur * (2.f - nstl::min(1.9f, (float)ur / sp_block)) / 64, 0.5f);

    const auto sp_amount = nb_od * nb_oh * nb_sp;
    const auto work_amount = mb * ngroups * nb_oc * sp_amount;
    const auto sp_eff = ((float)sp / rnd_up(sp, sp_block));

    const auto thr_eff = (float)work_amount / utils::rnd_up(work_amount, nthr);

    const auto oc_block_eff = (float)oc / rnd_up(oc, oc_block);

    const auto job = div_up(work_amount, nthr);

    auto job_eff = 1.f;
    if (job < nthr) {
        std::vector<dim_t> thr_jobs(nthr);

        for (int ithr = 0; ithr < nthr; ithr++) {
            thr_jobs[ithr] = 0;
            if (ithr >= work_amount) continue;
            dim_t thr_job = 0;
            int start {0}, end {0};
            balance211(work_amount, nthr, ithr, start, end);
            int n {0}, g {0}, ocb {0}, odp {0}, ohp {0}, spb {0};
            if (loop_order == loop_ndhwgc)
                nd_iterator_init(start, n, mb, odp, od, ohp, oh, spb, nb_sp, g,
                        ngroups, ocb, nb_oc);
            else if (loop_order == loop_ngcdhw)
                nd_iterator_init(start, n, mb, g, ngroups, ocb, nb_oc, odp, od,
                        ohp, oh, spb, nb_sp);

            for (auto work = start; work < end; work++) {
                const int ocp = ocb * oc_block;
                const auto oc_sz = nstl::min(oc - ocp, oc_block);
                int sp_sz = 0;
                const int spp = spb * sp_block;
                sp_sz = nstl::min(sp - spp, sp_block);
                thr_job += sp_sz * oc_sz;

                if (loop_order == loop_ndhwgc)
                    nd_iterator_step(n, mb, odp, od, ohp, oh, spb, nb_sp, g,
                            ngroups, ocb, nb_oc);
                else if (loop_order == loop_ngcdhw)
                    nd_iterator_step(n, mb, g, ngroups, ocb, nb_oc, odp, od,
                            ohp, oh, spb, nb_sp);
            }
            thr_jobs[ithr] = thr_job;
        }

        dim_t max_job = 0;
        dim_t sum_job = 0;
        for (int ithr = 0; ithr < nthr; ithr++) {
            if (thr_jobs[ithr] > max_job) max_job = thr_jobs[ithr];
            sum_job += thr_jobs[ithr];
        }
        job_eff = max_job == 0 ? 1 : (float)sum_job / (max_job * nthr);

    } else {
        job_eff = thr_eff;
    }

    const auto ic_blocking_size = ic_block * nb_ic_blocking;
    const auto oc_blocking_size = oc_block * ic_blocking_size;

    int l = -1;

    // -- brgemm kernel: loop by simd_w  --
    l++;
    const auto inp_ur = inp_w(ur, kw_block);
    loop[l].src.set(inp_ur * simd_w, 1, bcast_simd);
    loop[l].dst.set(0, 1);
    loop[l].wei.set(oc_block, 1);

    // -- brgemm kernel: loop by kw in kw_block  --
    l++;
    auto src_is = rnd_inp_simd(ur, kw_block, ic_blocking_size);
    loop[l].src.set(src_is, 1, kw_block);
    loop[l].dst.set(0, 1);
    loop[l].wei.set(oc_blocking_size, 1);

    // -- brgemm kernel: loop by batch (grouped by kw_block) in ur  --
    l++;
    loop[l].src.set(src_is, 1);
    loop[l].dst.set(0, 1);
    auto wei_is = kw_block * oc_blocking_size;
    loop[l].wei.set(wei_is, 1);
    // -- brgemm kernel: loop by ur in sp_block --
    l++;
    const auto nb_ur = div_up(sp_block, ur);
    loop[l].src.set(kd_block * kh_block * src_is, 1);
    loop[l].dst.set(ur * oc_block, 1);
    wei_is = kd_block * kh_block * kw_block * oc_blocking_size;
    loop[l].wei.set(wei_is, nb_ur);

    // -- harness: loop by k_blocks in ks --
    l++;
    loop[l].src.set(kd_block * kh_block
                    * rnd_inp_simd(sp_block, kw_block, ic_blocking_size),
            1);
    loop[l].dst.set(sp_block * oc_block, nb_kd * nb_kh * nb_kw);
    loop[l].wei.set(wei_is, 1);

    // -- brgemm kernel: loop by ic_chunks --
    l++;
    const auto ic_chunks = div_up(nb_ic, nb_ic_blocking);
    loop[l].src.set(kd * kh * rnd_inp_simd(sp_block, kw, ic_blocking_size), 1);
    loop[l].dst.set(sp_block * oc_block, ic_chunks);
    wei_is = kd * kh * kw * oc_blocking_size;
    loop[l].wei.set(wei_is, 1);

    const auto dim_oc = (loop_order == loop_ndhwgc) ? 1 : sp_amount;
    const auto nb_oc_thr = nstl::min(nb_oc, div_up(job, dim_oc));
    const auto oc_thr = nstl::min(oc, nb_oc_thr * oc_block);
    const auto nsimd_oc_thr = div_up(oc_thr, simd_w);

    const auto dim_sp = (loop_order == loop_ndhwgc) ? ngroups * nb_oc : 1;
    const auto nb_sp_thr = nstl::min(nb_sp, div_up(job, dim_sp));
    const auto sp_thr = nstl::min(sp, nb_sp_thr * sp_block);

    const auto dim_oh = nb_sp * dim_sp;
    const auto nb_oh_thr = nstl::min(nb_oh, div_up(job, dim_oh));
    const auto oh_thr = nstl::min(oh, nb_oh_thr * oh_block);

    const auto dim_od = nb_oh * dim_oh;
    const auto nb_od_thr = nstl::min(nb_od, div_up(job, dim_od));
    const auto od_thr = nstl::min(od, nb_od_thr * od_block);

    src_is = kd * kh * rnd_inp_simd(sp_block, kw, ic);

    auto wei_op = kd * kh * kw * ocblock * ic;
    if (loop_order == loop_ndhwgc) {
        // -- harness: loop by oc_block --
        l++;
        loop[l].src.set(src_is, nb_oc_thr);
        loop[l].dst.set(sp_block * oc_block, 1);
        wei_is = kd * kh * kw * oc_block * ic;
        wei_op = kd * kh * kw * nsimd_oc_thr * ic;
        loop[l].wei.set(wei_is, 1);
    }

    // -- harness: loop by sp_blocks --
    l++;
    loop[l].src.set(src_is, 1);
    const auto rnd_oc_for_sp
            = simd_w * ((loop_order == loop_ndhwgc) ? nsimd_oc_thr : ocblock);
    loop[l].dst.set(sp_block * rnd_oc_for_sp, 1);
    loop[l].wei.set(wei_op * simd_w, nb_sp_thr);
    // oh_block almost all is 1. TODO: manage oh_block != 1
    // -- harness: loop by oh_blocks --
    l++;
    src_is = kd * kh * rnd_inp_simd(sp_thr, kw, ic);
    loop[l].src.set(oh_block * src_is, 1);
    loop[l].dst.set(sp_thr * rnd_oc_for_sp, 1);
    loop[l].wei.set(wei_op * simd_w, nb_oh_thr);
    // od_block almost all is 1. TODO: manage oh_block != 1
    // -- harness: loop by od_blocks --
    l++;
    loop[l].src.set(od_block * oh_thr * src_is, 1);
    loop[l].dst.set(oh_thr * sp_thr * rnd_oc_for_sp, 1);
    loop[l].wei.set(wei_op * simd_w, nb_od_thr);

    if (loop_order != loop_ndhwgc) {
        // -- harness: loop by oc_block --
        l++;
        loop[l].src.set(od_thr * oh_thr * src_is, nb_oc_thr);
        loop[l].dst.set(oc_block * od_thr * oh_thr * sp_thr, 1);
        loop[l].wei.set(kd * kh * kw * oc_block * ic, 1);
    }

    // -- harness: loop by mb --
    l++;
    const auto mb_thr = nstl::min(mb, div_up(job, sp_amount * ngroups * nb_oc));
    loop[l].src.set(od_thr * oh_thr * src_is, 1);
    loop[l].dst.set(od_thr * oh_thr * sp_thr * nsimd_oc_thr * simd_w, 1);
    loop[l].wei.set(kd * kh * kw * nsimd_oc_thr * simd_w * ic, mb_thr);

    const auto src_op = mb_thr * od_thr * oh_thr * sp_thr * kd * kh * kw * ic;
    const auto dst_op = mb_thr * od_thr * oh_thr * sp_thr * nsimd_oc_thr;
    wei_op = kd * kh * kw * nsimd_oc_thr * ic;

    // for "real" application set bench_iterations to 1
    const auto iterations = bench_iterations;
    l++;
    loop[l].src.set(src_op, iterations);
    loop[l].dst.set(dst_op * simd_w, iterations);
    loop[l].wei.set(wei_op * simd_w, iterations);

    auto src_mem_k = mem_k;
    auto dst_mem_k = mem_k;
    auto wei_mem_k = mem_k;
    float src_rp = 1;
    float dst_rp = 1;
    float wei_rp = 1;

    for (auto il = l; il >= 0; il--) {
        src_mem_k = io_k(loop[il], loop[il].src, src_mem_k, true,
                loop_order == loop_ndhwgc ? false : true);
        dst_mem_k = io_k(loop[il], loop[il].dst, dst_mem_k, false, false);
        wei_mem_k = io_k(loop[il], loop[il].wei, wei_mem_k, false,
                loop_order == loop_ndhwgc ? true : false);
        src_rp *= loop[il].src.repeatn;
        dst_rp *= loop[il].dst.repeatn;
        wei_rp *= loop[il].wei.repeatn;
    }
    const auto src_ops = (src_op * src_rp) / iterations;
    const auto dst_ops = (dst_op * dst_rp) / iterations;
    const auto wei_ops = (wei_op * wei_rp) / iterations;

    const auto src_cost = src_mem_k * src_ops;
    const auto dst_cost = dst_mem_k * dst_ops;
    const auto wei_cost = wei_mem_k * wei_ops;
    const auto call_kernel_cost
            = 1000.f * job * ic_chunks * nb_kd * nb_kh * nb_kw;

    const auto cache_eff = ((dim_t)mb * od * oh * sp * ic * oc)
            / (nthr * (src_cost + dst_cost + wei_cost + call_kernel_cost));
    const auto res_eff = oc_block_eff * brgemm_microkernel_eff * sp_eff
            * job_eff * ur_eff * cache_eff * brgemm_eff;
    return res_eff;
}

void brg_blocking_t::iterate_ker_block(brg_blocking_t &best_brgb, int kd_block_,
        int kh_block_, bool maybe_use_buffer, int max_ow_block_thr) {

    unsigned est_k_amount = ic * oc_block * wei_dsz;

    kd_block = kd_block_;
    kh_block = kh_block_;
    if (one_of(exec_type, exec_vpad, exec_trans)) {
        kw_block = kw;
        kd_block_pad = kd_block;
        kh_block_pad = kh_block;
        kw_block_pad = kw_block;
    } else {
        kw_block = (est_k_amount * kw < L2) ? kw : 1;
        kd_block_pad = kh_block >= kd ? kd : 1;
        kh_block_pad = kw_block >= kh ? kh : 1;
        kw_block_pad = kw;
    }

    if (exec_type == exec_vpad) {
        od_blk_size = 1;
        oh_blk_size = 1;
    } else if (exec_type == exec_trans) {
        // TODO: select od/oh block size for best balancing
        // and performance
        const auto w_block_size
                = 2 * src_dsz * ic * iwp + dst_dsz * oc_block * ow;
        const auto L2_available = L2 - wei_dsz * kd * kh * kw * ic * oc_block;
        if (idp * ihp * w_block_size > L2_available) {
            od_blk_size
                    = utils::saturate(1, od, int(L2 / (ihp * w_block_size)));
            if (od_blk_size == 1)
                oh_blk_size = utils::saturate(1, oh, int(L2 / (w_block_size)));
            else
                oh_blk_size = oh;
        } else {
            od_blk_size = 1;
            oh_blk_size = oh;
        }
    } else {
        od_blk_size = 1;
        oh_blk_size = 1;
    }

    // --- Select ow_block ----
    const auto max_ow_block_L2 = ow;
    auto start_ow_block = nstl::min(max_ow_block_thr, max_ow_block_L2);

    sp = ow;
    const auto start_sp_block = start_ow_block;
    auto prev_spb = 0;
    for (auto ns = 1; ns <= sp; ns++) {
        const auto spb = div_up(sp, ns);
        if (spb == prev_spb || spb > start_sp_block) continue;
        prev_spb = spb;
        ow_block = spb;
        sp_block = ow_block;

        select_ic_block();

        use_buffer = maybe_use_buffer
                && (ic_block * nb_ic_blocking < ic || kd_block != kd
                        || kh_block != kh || kw_block != kw
                        || kd_block_pad != kd || kh_block_pad != kh
                        || kw_block_pad != kw);
        if (exec_type == exec_base)
            use_buffer = use_buffer || (maybe_use_buffer && iwp != iw);

        ur = estimate_brgemm_ur(ow_block);
        update_blocks();

        eff = est_eff();

        if (eff > best_brgb.eff || best_brgb.eff == 0) best_brgb = *this;
    }
}

void brg_blocking_t::calc_blocks() {
    sp = ow;
    is_os_block = false;

    nb_ic_blocking = 1;
    // --- Select kernel blocking ---
    // if dst_dt != acc_dt and we need to store intermediate
    // results then we need the out buffer
    const auto maybe_use_buffer = (dst_dt != acc_dt || with_sum);

    std::vector<int> kd_blocks(1), kh_blocks(1);
    kd_blocks[0] = kd;
    kh_blocks[0] = kh;
    if (kd != 1) {
        kd_blocks.resize(2);
        kd_blocks[1] = 1;
    }
    if (kh != 1) {
        kh_blocks.resize(2);
        kh_blocks[1] = 1;
    }

    const auto thr_eff_threshold = 0.9f;
    const auto max_ow_block_thr = utils::saturate(1, ow,
            (int)div_up(mb * ngroups * nb_oc * os, thr_eff_threshold * nthr));

    ow_block = -1;
    brg_blocking_t best_brgb = *this;
    for (const auto &kd_block : kd_blocks) {
        for (const auto &kh_block : kh_blocks) {
            iterate_ker_block(best_brgb, kd_block, kh_block, maybe_use_buffer,
                    max_ow_block_thr);
        }
    }
    *this = best_brgb;
    ow_block = os_block = sp_block;
    update_blocks();
}

bool brg_blocking_t::fast_check_oc_block_1x1() const {
    // This function for reducing the number of blocking variants
    // TODO: eliminate heuristic in this function
    const auto rnd_oc = rnd_up(oc, 16);
    auto res = false;
    if (oc_block == 64) {
        const auto big_spatial
                = od * oh * ow >= 64 * stride_d * stride_h * stride_w;
        res = (rnd_oc % oc_block == 0 && big_spatial);
    } else if (oc_block == 48) {
        const auto oc_block_eff = (float)oc / rnd_up(oc, oc_block);
        res = (oc_block_eff >= 0.95);
    } else
        res = true;

    return res;
}

float brg_blocking_t::est_eff_1x1() {
    const auto ocblock = oc_block / 16;
    const auto od_block = od_blk_size;
    const auto oh_block = oh_blk_size;

    const auto brgemm_microkernel_eff
            = ((float)ocblock * ur) / ((ur + ocblock) * max_regs);
    const auto ur_eff = (float)sp_block / rnd_up(sp_block, ur);
    const auto brgemm_eff = squeeze_val(
            ur * (2.f - nstl::min(1.9f, (float)ur / sp_block)) / 64, 0.5f);

    const auto sp_amount = is_os_block ? div_up(nb_os, nb_os_blocking)
                                       : nb_od * nb_oh * nb_sp;
    const auto work_amount = mb * ngroups * nb_oc * sp_amount;

    const auto sp_eff = (float)sp / rnd_up(sp, sp_block);
    const auto thr_eff = (float)work_amount / utils::rnd_up(work_amount, nthr);
    const auto oc_block_eff = (float)oc / rnd_up(oc, oc_block);

    const auto job = div_up(work_amount, nthr);

    const auto dim_oc = (loop_order == loop_ndhwgc) ? 1 : sp_amount;
    const auto nb_oc_thr = nstl::min(nb_oc, div_up(job, dim_oc));
    const auto oc_thr = nstl::min(oc, nb_oc_thr * oc_block);
    const auto nsimd_oc_thr = div_up(oc_thr, simd_w);

    const auto dim_sp = (loop_order == loop_ndhwgc) ? ngroups * nb_oc : 1;
    const auto nb_sp_thr = nstl::min(nb_sp, div_up(job, dim_sp));
    const auto sp_thr = nstl::min(sp, nb_sp_thr * sp_block);

    const auto dim_oh = nb_sp * dim_sp;
    const auto nb_oh_thr = nstl::min(nb_oh, div_up(job, dim_oh));
    const auto oh_thr = is_os_block ? 1 : nstl::min(oh, nb_oh_thr * oh_block);

    const auto dim_od = nb_oh * dim_oh;
    const auto nb_od_thr = nstl::min(nb_od, div_up(job, dim_od));
    const auto od_thr = is_os_block ? 1 : nstl::min(od, nb_od_thr * od_block);

    auto job_eff = 1.f;
    if (job < nthr) {
        std::vector<dim_t> thr_jobs(nthr);
        for (int ithr = 0; ithr < nthr; ithr++) {
            thr_jobs[ithr] = 0;
            if (ithr >= work_amount) continue;
            dim_t thr_job = 0;
            int start {0}, end {0};
            balance211(work_amount, nthr, ithr, start, end);
            int n {0}, g {0}, ocb {0}, oss {0}, odp {0}, ohp {0}, spb {0};
            if (loop_order == loop_ndhwgc) {
                if (is_os_block)
                    nd_iterator_init(start, n, mb, oss, sp_amount, g, ngroups,
                            ocb, nb_oc);
                else
                    nd_iterator_init(start, n, mb, odp, od, ohp, oh, spb, nb_sp,
                            g, ngroups, ocb, nb_oc);
            } else if (loop_order == loop_ngcdhw) {
                if (is_os_block)
                    nd_iterator_init(start, n, mb, g, ngroups, ocb, nb_oc, oss,
                            sp_amount);
                else
                    nd_iterator_init(start, n, mb, g, ngroups, ocb, nb_oc, odp,
                            od, ohp, oh, spb, nb_sp);
            }

            for (auto work = start; work < end; work++) {
                const int ocp = ocb * oc_block;
                const auto oc_sz = nstl::min(oc - ocp, oc_block);
                int sp_sz = 0;
                if (is_os_block) {
                    const auto osb_start = oss * nb_os_blocking;
                    const auto osb_range
                            = nstl::min(nb_os - osb_start, nb_os_blocking);
                    for (int osb = 0; osb < osb_range; osb++) {
                        const int osp = (osb_start + osb) * sp_block;
                        sp_sz = nstl::min(os - osp, sp_block);
                    }
                } else {
                    const int spp = spb * sp_block;
                    sp_sz = nstl::min(sp - spp, sp_block);
                }
                thr_job += sp_sz * oc_sz;

                if (loop_order == loop_ndhwgc) {
                    if (is_os_block)
                        nd_iterator_step(
                                n, mb, oss, sp_amount, g, ngroups, ocb, nb_oc);
                    else
                        nd_iterator_step(n, mb, odp, od, ohp, oh, spb, nb_sp, g,
                                ngroups, ocb, nb_oc);
                } else if (loop_order == loop_ngcdhw) {
                    if (is_os_block)
                        nd_iterator_step(
                                n, mb, g, ngroups, ocb, nb_oc, oss, sp_amount);
                    else
                        nd_iterator_step(n, mb, g, ngroups, ocb, nb_oc, odp, od,
                                ohp, oh, spb, nb_sp);
                }
            }
            thr_jobs[ithr] = thr_job;
        }

        dim_t max_job = 0;
        dim_t sum_job = 0;
        for (int ithr = 0; ithr < nthr; ithr++) {
            if (thr_jobs[ithr] > max_job) max_job = thr_jobs[ithr];
            sum_job += thr_jobs[ithr];
        }

        job_eff = max_job == 0 ? 1 : (float)sum_job / (max_job * nthr);
    } else {
        job_eff = thr_eff;
    }

    const auto ic_blocking_size = ic_block * nb_ic_blocking;
    const auto oc_blocking_size = oc_block * ic_blocking_size;

    int l = -1;
    // -- brgemm kernel: loop by simd_w  --
    l++;
    loop[l].src.set(ur * simd_w, 1, bcast_simd);
    loop[l].dst.set(0, 1);
    loop[l].wei.set(oc_block, 1);

    // -- brgemm kernel: loop by ur in sp_block --
    l++;
    const auto nb_ur = div_up(sp_block, ur);
    loop[l].src.set(ur * rnd_simd(ic_blocking_size), 1);
    loop[l].dst.set(ur * oc_block, 1);
    loop[l].wei.set(oc_blocking_size, nb_ur);
    // -- brgemm kernel: loop by ic_chunks --
    l++;
    const auto ic_chunks = div_up(nb_ic, nb_ic_blocking);
    loop[l].src.set(sp_block * ic_blocking_size, 1);
    loop[l].dst.set(sp_block * oc_block, ic_chunks);
    auto wei_is = oc_blocking_size;
    auto wei_op = ocblock * ic;
    loop[l].wei.set(wei_is, 1);

    if (loop_order == loop_ndhwgc) {
        // -- harness: loop by oc_block --
        l++;
        loop[l].src.set(sp_block * rnd_simd(ic), nb_oc_thr);
        loop[l].dst.set(sp_block * oc_block, 1);
        wei_is = oc_block * ic;
        wei_op = nsimd_oc_thr * ic;
        loop[l].wei.set(wei_is, 1);
    }

    const auto rnd_oc_for_sp
            = simd_w * ((loop_order == loop_ndhwgc) ? nsimd_oc_thr : ocblock);
    if (is_os_block) {
        // -- harness: loop by os_blocks --
        l++;
        loop[l].src.set(sp_block * ic_blocking_size, 1);
        loop[l].dst.set(sp_block * rnd_oc_for_sp, 1);
        loop[l].wei.set(wei_op * simd_w, nb_sp_thr);
    } else {
        // -- harness: loop by sp_blocks --
        l++;
        loop[l].src.set(sp_block * ic_blocking_size, 1);
        loop[l].dst.set(sp_block * rnd_oc_for_sp, 1);
        loop[l].wei.set(wei_op * simd_w, nb_sp_thr);
        // -- harness: loop by oh_blocks --
        l++;
        loop[l].src.set(oh_block * sp_thr * rnd_simd(ic_blocking_size), 1);
        loop[l].dst.set(oh_block * sp_thr * rnd_oc_for_sp, 1);
        loop[l].wei.set(wei_op * simd_w, nb_oh_thr);
        // -- harness: loop by od_blocks --
        l++;
        loop[l].src.set(
                od_block * oh_thr * sp_thr * rnd_simd(ic_blocking_size), 1);
        loop[l].dst.set(od_block * oh_thr * sp_thr * rnd_oc_for_sp, 1);
        loop[l].wei.set(wei_op * simd_w, nb_od_thr);
    }

    if (loop_order != loop_ndhwgc) {
        // -- harness: loop by oc_block --
        l++;
        loop[l].src.set(od_thr * oh_thr * rnd_simd(sp_thr * ic_blocking_size),
                nb_oc_thr);
        loop[l].dst.set(oc_block * od_thr * oh_thr * sp_thr, 1);
        loop[l].wei.set(oc_block * ic, 1);
    }

    // -- harness: loop by mb --
    l++;
    const auto mb_thr = nstl::min(mb, div_up(job, sp_amount * ngroups * nb_oc));
    loop[l].src.set(od_thr * oh_thr * sp_thr * rnd_simd(ic_blocking_size), 1);
    loop[l].dst.set(nsimd_oc_thr * simd_w * od_thr * oh_thr * sp_thr, 1);
    loop[l].wei.set(nsimd_oc_thr * ic * simd_w, mb_thr);

    const auto src_op = mb_thr * od_thr * oh_thr * sp_thr * ic_blocking_size;
    const auto dst_op = mb_thr * nsimd_oc_thr * od_thr * oh_thr * sp_thr;
    wei_op = nsimd_oc_thr * ic;

    // for "real" application set bench_iterations to 1
    const auto iterations = bench_iterations;
    l++;
    loop[l].src.set(src_op, iterations);
    loop[l].dst.set(dst_op * simd_w, iterations);
    loop[l].wei.set(wei_op * simd_w, iterations);

    auto src_mem_k = mem_k;
    auto dst_mem_k = mem_k;
    auto wei_mem_k = mem_k;
    float src_rp = 1;
    float dst_rp = 1;
    float wei_rp = 1;

    for (auto il = l; il >= 0; il--) {
        src_mem_k = io_k(loop[il], loop[il].src, src_mem_k, true, false);
        dst_mem_k = io_k(loop[il], loop[il].dst, dst_mem_k, false, false);
        wei_mem_k = io_k(loop[il], loop[il].wei, wei_mem_k, false, true);
        src_rp *= loop[il].src.repeatn;
        dst_rp *= loop[il].dst.repeatn;
        wei_rp *= loop[il].wei.repeatn;
    }
    const auto src_ops = (src_op * src_rp) / iterations;
    const auto dst_ops = (dst_op * dst_rp) / iterations;
    const auto wei_ops = (wei_op * wei_rp) / iterations;

    const auto src_cost = src_mem_k * src_ops;
    const auto dst_cost = dst_mem_k * dst_ops;
    const auto wei_cost = wei_mem_k * wei_ops;
    const auto call_kernel_cost = 1000.f * job * ic_chunks;

    const auto up_sp_size = is_os_block ? 1 : od * oh;

    const auto cache_eff = ((dim_t)mb * up_sp_size * sp * ic * oc)
            / (nthr * (src_cost + dst_cost + wei_cost + call_kernel_cost));

    const auto res_eff = oc_block_eff * brgemm_microkernel_eff * sp_eff
            * job_eff * ur_eff * cache_eff * brgemm_eff;
    return res_eff;
}

void brg_blocking_t::calc_blocks_1x1() {
    if (stride_d == 1 && stride_h == 1) {
        sp = os;
        is_os_block = true;
    } else {
        sp = ow;
        is_os_block = false;
    }

    od_blk_size = 1;
    oh_blk_size = 1;
    kd_block = kh_block = kw_block = 1;
    kd_block_pad = kh_block_pad = kw_block_pad = 1;
    nb_ic_blocking = 1;

    const auto thr_eff_threshold = 0.9f;

    const auto max_sp_block_L2 = os;
    // TODO: nb_os_blocking always is 1 for now. Update this code
    nb_os_blocking = 1;
    int start_sp_block = 0;

    if (stride_d == 1 && stride_h == 1) {
        ow_block = 0;

        const auto max_os_block_thr = nstl::max(
                div_up(2048, oc_block), (int)div_up(mb * ngroups * os, nthr));
        const auto max_os_block_L2 = max_sp_block_L2;

        auto max_os_block_aliasing = 1000000 / nthr;
        if ((oc_without_padding * os * dst_dsz) % 4096 == 0) {
            max_os_block_aliasing /= 1;
            for (auto cur_oc = oc_without_padding;
                    max_os_block_aliasing * dst_dsz > 400 && cur_oc % 2 == 0
                    && cur_oc * os * dst_dsz >= 4096;
                    cur_oc /= 2) {
                max_os_block_aliasing /= 2;
            }
            max_os_block_aliasing += max_os_block_aliasing % 2 ? 0 : 1;
        }
        max_os_block_aliasing
                = nstl::min(div_up(1001, dst_dsz), max_os_block_aliasing);

        start_sp_block = utils::saturate(1, os,
                nstl::min(nstl::min(max_os_block_thr, max_os_block_L2),
                        max_os_block_aliasing));

    } else {
        os_block = 0;

        const auto max_ow_block_thr = utils::saturate(1, ow,
                (int)div_up(
                        mb * ngroups * nb_oc * os, thr_eff_threshold * nthr));
        const auto max_ow_block_L2 = max_sp_block_L2;

        start_sp_block = utils::saturate(
                1, ow, nstl::min(max_ow_block_thr, max_ow_block_L2));
    }
    brg_blocking_t best_brgb = *this;

    auto prev_spb = 0;
    for (auto ns = 1; ns <= sp; ns++) {
        const auto spb = div_up(sp, ns);
        if (spb == prev_spb || spb > start_sp_block) continue;
        prev_spb = spb;
        if (is_os_block)
            os_block = spb;
        else
            ow_block = spb;
        select_ic_block();
        ur = estimate_brgemm_ur(spb);
        update_blocks();

        use_buffer = (dst_dt != acc_dt || with_sum)
                && (ic_block * nb_ic_blocking < ic);

        eff = est_eff_1x1();
        if (eff > best_brgb.eff || best_brgb.eff == 0) { best_brgb = *this; }
    }
    *this = best_brgb;
    os_block = ow_block = sp_block;
    update_blocks();
}

status_t init_jcp(jit_brgemm_conv_conf_t &jcp, cpu_isa_t isa,
        const convolution_desc_t &cd, memory_desc_t &src_md,
        memory_desc_t &weights_md, memory_desc_t &dst_md,
        memory_desc_t &bias_md, const primitive_attr_t &attr, int nthreads) {
    using namespace prop_kind;

    brg_blocking_t::L1 = platform::get_per_core_cache_size(1);
    brg_blocking_t::L2 = platform::get_per_core_cache_size(2);
    brg_blocking_t::L3 = platform::get_per_core_cache_size(2);
    brg_blocking_t::isa = isa;

    if (!mayiuse(avx512_core)) return status::unimplemented;

    const memory_desc_wrapper src_d(&src_md);
    const memory_desc_wrapper weights_d(&weights_md);
    const memory_desc_wrapper dst_d(&dst_md);
    const memory_desc_wrapper bias_d(&bias_md);

    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;
    int ndims = src_d.ndims();

    jcp = zero<decltype(jcp)>();
    jcp.ndims = ndims;
    jcp.prop_kind = cd.prop_kind;
    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];
    jcp.oc_without_padding = dst_d.dims()[1];
    jcp.oc = jcp.oc_without_padding / jcp.ngroups;
    jcp.ic_without_padding = src_d.dims()[1];
    jcp.ic = jcp.ic_without_padding / jcp.ngroups;
    jcp.id = (ndims == 5) ? src_d.dims()[2] : 1;
    jcp.ih = (ndims == 3) ? 1 : src_d.dims()[ndims - 2];
    jcp.iw = src_d.dims()[ndims - 1];
    jcp.od = (ndims == 5) ? dst_d.dims()[2] : 1;
    jcp.oh = (ndims == 3) ? 1 : dst_d.dims()[ndims - 2];
    jcp.ow = dst_d.dims()[ndims - 1];
    jcp.kd = (ndims == 5) ? weights_d.dims()[with_groups + 2] : 1;
    jcp.kh = (ndims == 3) ? 1 : weights_d.dims()[with_groups + ndims - 2];
    jcp.kw = weights_d.dims()[with_groups + ndims - 1];
    jcp.f_pad = (ndims == 5) ? cd.padding[0][0] : 0;
    jcp.t_pad = (ndims == 3) ? 0 : cd.padding[0][ndims - 4];
    jcp.l_pad = cd.padding[0][ndims - 3];
    jcp.stride_d = (ndims == 5) ? cd.strides[0] : 1;
    jcp.stride_h = (ndims == 3) ? 1 : cd.strides[ndims - 4];
    jcp.stride_w = cd.strides[ndims - 3];

    jcp.dilate_d = (ndims == 5) ? cd.dilates[0] : 0;
    jcp.dilate_h = (ndims == 3) ? 0 : cd.dilates[ndims - 4];
    jcp.dilate_w = cd.dilates[ndims - 3];

    jcp.os = jcp.od * jcp.oh * jcp.ow;

    int ext_kd = calculate_extended_filter_size(jcp.kd, jcp.dilate_d);
    int ext_kh = calculate_extended_filter_size(jcp.kh, jcp.dilate_h);
    int ext_kw = calculate_extended_filter_size(jcp.kw, jcp.dilate_w);

    jcp.back_pad = calculate_end_padding(
            jcp.f_pad, jcp.od, jcp.id, jcp.stride_d, ext_kd);
    jcp.b_pad = calculate_end_padding(
            jcp.t_pad, jcp.oh, jcp.ih, jcp.stride_h, ext_kh);
    jcp.r_pad = calculate_end_padding(
            jcp.l_pad, jcp.ow, jcp.iw, jcp.stride_w, ext_kw);

    jcp.with_bias = cd.bias_desc.format_kind != format_kind::undef;

    jcp.src_dt = cd.src_desc.data_type;
    jcp.dst_dt = cd.dst_desc.data_type;
    jcp.wei_dt = cd.weights_desc.data_type;
    jcp.bia_dt = jcp.with_bias ? cd.bias_desc.data_type : data_type::undef;

    // TODO: optimize depthwise convolutions (for now direct approach is faster)
    const bool is_depthwise = with_groups && everyone_is(1, jcp.ic, jcp.oc);
    if (is_depthwise) return status::unimplemented;

    // TODO: support s8 by brgemm convolutions
    if (jcp.src_dt == s8) return status::unimplemented;

    if (!IMPLICATION(jcp.wei_dt == s8, mayiuse(avx512_core_vnni)))
        return status::unimplemented;
    if (!IMPLICATION(jcp.wei_dt == bf16, mayiuse(avx512_core_bf16)))
        return status::unimplemented;

    if (one_of(jcp.src_dt, u8, s8)) {
        jcp.acc_dt = s32;
    } else if (one_of(jcp.src_dt, f32, bf16)) {
        jcp.acc_dt = f32;
    } else
        return status::unimplemented;

    jcp.src_dsz = types::data_type_size(jcp.src_dt);
    jcp.wei_dsz = types::data_type_size(jcp.wei_dt);
    jcp.dst_dsz = types::data_type_size(jcp.dst_dt);
    jcp.acc_dsz = types::data_type_size(jcp.acc_dt);
    jcp.bia_dsz = jcp.with_bias ? types::data_type_size(jcp.bia_dt) : 0;

    if (!post_ops_ok(jcp, attr, dst_d)) return status::unimplemented;

    jcp.simd_w = cpu_isa_traits<avx512_common>::vlen / jcp.src_dsz;
    const auto &p = attr.post_ops_;
    jcp.with_sum = p.find(primitive_kind::sum) != -1;
    const int eltwise_ind = p.find(primitive_kind::eltwise);
    jcp.with_eltwise = eltwise_ind != -1;

    const int binary_ind = p.find(primitive_kind::binary);
    jcp.with_binary = binary_ind != -1;

    if (jcp.with_bias) {
        if (bias_d.format_kind() == format_kind::any)
            CHECK(memory_desc_init_by_tag(bias_md, x));
    }

    jcp.nthr = nthreads;

    // fast check data layout before spending time for blocking selection
    format_tag_t src_tag = pick(jcp.ndims - 3, nwc, nhwc, ndhwc);
    const bool any_eligible = (jcp.prop_kind == prop_kind::forward_inference);
    CHECK(init_tag(jcp.src_tag, src_md, src_d, src_tag, any_eligible));

    return status::success;
}

status_t init_conf(jit_brgemm_conv_conf_t &jcp, cpu_isa_t isa,
        const convolution_desc_t &cd, memory_desc_t &src_md,
        memory_desc_t &weights_md, memory_desc_t &dst_md,
        memory_desc_t &bias_md, const primitive_attr_t &attr, int nthreads) {

    using namespace prop_kind;

    CHECK(init_jcp(
            jcp, isa, cd, src_md, weights_md, dst_md, bias_md, attr, nthreads));

    const memory_desc_wrapper src_d(&src_md);
    const memory_desc_wrapper weights_d(&weights_md);
    const memory_desc_wrapper dst_d(&dst_md);
    const memory_desc_wrapper bias_d(&bias_md);

    jcp.idp = jcp.id + jcp.f_pad + jcp.back_pad;
    jcp.ihp = jcp.ih + jcp.t_pad + jcp.b_pad;
    jcp.iwp = jcp.iw + jcp.l_pad + jcp.r_pad;

    using namespace data_type;
    // ======================= blocking =================================

    auto bcast_amount = (size_t)jcp.id * jcp.ih * jcp.iw * jcp.src_dsz;
    auto wei_amount = (size_t)jcp.oc * jcp.kd * jcp.kh * jcp.kw * jcp.wei_dsz;

    jcp.loop_order = (bcast_amount < wei_amount) ? loop_ngcdhw : loop_ndhwgc;

    const int min_oc_block = 16;

    int selected_ur = 0;
    MAYBE_UNUSED(selected_ur);

    auto try_exec_type = [&]() {
        brg_blocking_t best_brgb = zero<decltype(best_brgb)>();
        best_brgb.oc_block = min_oc_block;
        brg_blocking_t cur_brgb = zero<decltype(best_brgb)>();
        cur_brgb.get_from_jcp(jcp);
        auto start_ocb = 4;
        if (jcp.wei_plain)
            start_ocb = nstl::min(jcp.ic > 128 ? (jcp.ic > 256 ? 8 : 16) : 32,
                    div_up(jcp.oc, 16));
        start_ocb = nstl::min(div_up(jcp.oc, 16), start_ocb);

        auto finish_ocb = 1;
        for (auto ocb = start_ocb; ocb >= finish_ocb; ocb--) {
            cur_brgb.oc_block = ocb * 16;
            cur_brgb.nb_oc = utils::div_up(jcp.oc, cur_brgb.oc_block);
            if (!cur_brgb.fast_check_oc_block()) continue;

            cur_brgb.calc_blocks();
            const bool is_1x1 = false;
            const auto ur = cur_brgb.get_brgemm_ur(&attr, dst_md, is_1x1);
            if (ur == 0) continue;
            cur_brgb.ur = ur;
            cur_brgb.eff = cur_brgb.est_eff();
            if (cur_brgb.eff > best_brgb.eff) best_brgb = cur_brgb;
        }
        if (best_brgb.oc_block == 0 || best_brgb.ic_block == 0
                || best_brgb.ow_block == 0)
            return false;
        best_brgb.save_to_jcp(jcp);
        selected_ur = best_brgb.ur;
        return true;
    };

    //-----------------------------------------------------------------------

    jcp.exec_type = exec_base;
    jcp.brg_type = brgemm_addr; // TODO: Choose right type of BRGEMM

    bool try_exec_vpad = false;
    bool try_exec_trans = false;

    if (div_up(jcp.l_pad, jcp.stride_w) < jcp.kw
            && div_up(jcp.r_pad, jcp.stride_w) < jcp.kw) {
        try_exec_vpad = true;
    }
    // TODO: remove this restriction
    if (jcp.dilate_d == 0 && jcp.dilate_h == 0 && jcp.dilate_w == 0
            && jcp.stride_w == 1) {
        // conv with transpose does not work well for 3d
        try_exec_trans = (jcp.l_pad > 0 || jcp.r_pad > 0)
                && (1.f - float(jcp.iw) / jcp.iwp)
                                / nstl::max(1, jcp.iwp - jcp.iw)
                        < 0.1f
                && jcp.id == 1;
    }

    // TODO: in future use (kd/kh/kw) and (kd/kh/kw)_pad blocks for more
    // precise calculation of jcp.max_batch
    jcp.max_batch = jcp.kd * jcp.kh * jcp.kw;

    //TODO: check wei plain
    jcp.wei_plain = false;
    jcp.wei_plain = jcp.exec_type == exec_vpad ? jcp.wei_plain : false;

    bool try_exec_type_res = false;

    if (try_exec_vpad) {
        jcp.exec_type = exec_vpad;
        try_exec_type_res = try_exec_type();
        const auto iw_block = (jcp.ow_block - 1) * jcp.stride_w + 1;
        // to avoid case when both top and bottom virtual padding are non-zero
        // TODO: remove this restriction
        if (iw_block > jcp.iw) try_exec_type_res = false;
    }
    if (try_exec_type_res == false && try_exec_trans) {
        jcp.exec_type = exec_trans;
        try_exec_type_res = try_exec_type();
        const auto iw_block = (jcp.ow_block - 1) * jcp.stride_w + 1;
        // transform kernel doesn't work well with big padding
        // TODO: remove this restriction
        if (iw_block < jcp.l_pad || iw_block < jcp.r_pad)
            try_exec_type_res = false;
    }
    if (try_exec_type_res == false) {
        jcp.exec_type = exec_base;
        try_exec_type_res = try_exec_type();
    }

    // ============ end blocking ===========================================
    if (jcp.exec_type == exec_vpad)
        jcp.max_vpad = nstl::max(jcp.l_pad, jcp.r_pad);
    else
        jcp.max_vpad = 0;

    if (jcp.ow_block == 0 || jcp.ic_block == 0 || jcp.oc_block == 0)
        return status::unimplemented;
    // Configure matrix sizes
    jcp.M = jcp.ow >= jcp.ow_block ? jcp.ow_block : 0;
    jcp.K = jcp.ic >= jcp.ic_block ? jcp.ic_block : 0;
    jcp.N = jcp.oc >= jcp.oc_block ? jcp.oc_block : 0;
    jcp.M_tail = jcp.ow % jcp.ow_block;
    jcp.K_tail = jcp.ic % jcp.ic_block;
    jcp.N_tail = jcp.oc % jcp.oc_block;

    jcp.gemm_batch_size = jcp.nb_ic_blocking
            * nstl::max(jcp.kd_block * jcp.kh_block * jcp.kw_block,
                    jcp.kd_block_pad * jcp.kh_block_pad * jcp.kw_block_pad);
    // to avoid cache concurrent write access from different threads
    size_t sc_size = sizeof(brgemm_batch_element_t);
    jcp.adjusted_batch_size
            = div_up(rnd_up(jcp.gemm_batch_size * sc_size, 4096), sc_size);

    jcp.LDA = (jcp.exec_type == exec_trans)
            ? jcp.stride_w * jcp.ic_block
            : jcp.stride_w * jcp.ic_without_padding;
    jcp.LDC = (jcp.use_buffer) ? jcp.oc_block : jcp.oc_without_padding;
    jcp.LDD = jcp.oc_without_padding;

    CHECK(pick_tags(jcp, src_md, weights_md, dst_md, bias_md));

    const auto &oscales = attr.output_scales_;
    jcp.is_oc_scale = oscales.mask_ == 1 << 1;

    // only common and per-oc-channel scales are supported
    const bool oscales_ok = one_of(oscales.mask_, 0, 1 << 1);
    if (!oscales_ok) return status::unimplemented;

    jcp.buffer_size = jcp.LDC * jcp.M;

    jcp.nb_od = div_up(jcp.od, jcp.od_blk_size);
    jcp.nb_oh = div_up(jcp.oh, jcp.oh_blk_size);

    if (jcp.exec_type == exec_trans) {
        // TODO: this is rough estimation of buffer for transpose input
        jcp.inp_buffer_size = rnd_up((dim_t)jcp.idp * jcp.ihp * jcp.iwp
                        * jcp.ngroups * jcp.nb_ic * jcp.ic_block,
                4096);
        jcp.inp_buffer_mask_size = rnd_up((dim_t)jcp.nb_od * jcp.nb_oh
                        * jcp.nb_ow * jcp.ngroups * jcp.nb_ic,
                4096);
    }

    return status::success;
}

status_t init_1x1_conf(jit_brgemm_conv_conf_t &jcp, cpu_isa_t isa,
        const convolution_desc_t &cd, memory_desc_t &src_md,
        memory_desc_t &weights_md, memory_desc_t &dst_md,
        memory_desc_t &bias_md, const primitive_attr_t &attr, int nthreads) {

    using namespace prop_kind;

    CHECK(init_jcp(
            jcp, isa, cd, src_md, weights_md, dst_md, bias_md, attr, nthreads));

    const memory_desc_wrapper src_d(&src_md);
    const memory_desc_wrapper weights_d(&weights_md);
    const memory_desc_wrapper dst_d(&dst_md);
    const memory_desc_wrapper bias_d(&bias_md);

    bool args_ok = true && jcp.f_pad <= 0 && jcp.back_pad <= 0 && jcp.t_pad <= 0
            && jcp.b_pad <= 0 && jcp.l_pad <= 0 && jcp.r_pad <= 0 && jcp.kd == 1
            && jcp.kh == 1 && jcp.kw == 1;
    if (!args_ok) return status::unimplemented;

    using namespace data_type;
    // ===================== blocking =================================

    auto bcast_amount = (size_t)jcp.id * jcp.ih * jcp.iw * jcp.src_dsz;
    auto wei_amount = (size_t)jcp.oc * jcp.wei_dsz;

    jcp.loop_order = (bcast_amount < wei_amount) ? loop_ngcdhw : loop_ndhwgc;

    const auto min_oc_block = 16;

    jcp.brg_type = brgemm_addr; // TODO: Choose right type of BRGEMM

    // max_batch is 1 and max_vpad is 0 for 1x1 convolutions
    jcp.max_batch = 1;
    jcp.max_vpad = 0;

    jcp.wei_plain = false;

    brg_blocking_t best_brgb = zero<decltype(best_brgb)>();
    best_brgb.oc_block = min_oc_block;
    brg_blocking_t cur_brgb = zero<decltype(cur_brgb)>();
    cur_brgb.get_from_jcp(jcp);
    auto start_ocb = 4;
    if (jcp.wei_plain)
        start_ocb = nstl::min(jcp.ic > 128 ? (jcp.ic > 256 ? 8 : 16) : 32,
                div_up(jcp.oc, 16));
    start_ocb = nstl::min(div_up(jcp.oc, 16), start_ocb);

    auto finish_ocb = 1;
    for (auto ocb = start_ocb; ocb >= finish_ocb; ocb--) {
        cur_brgb.oc_block = ocb * min_oc_block;
        cur_brgb.nb_oc = utils::div_up(jcp.oc, cur_brgb.oc_block);

        if (!cur_brgb.fast_check_oc_block_1x1()) continue;

        cur_brgb.calc_blocks_1x1();
        const bool is_1x1 = true;
        const auto ur = cur_brgb.get_brgemm_ur(&attr, dst_md, is_1x1);
        if (ur == 0) continue;
        cur_brgb.ur = ur;
        cur_brgb.eff = cur_brgb.est_eff_1x1();
        if (cur_brgb.eff > best_brgb.eff) best_brgb = cur_brgb;
    }
    best_brgb.save_to_jcp(jcp);

    // =============== end blocking =================================
    jcp.brg_stride_a = jcp.ic_block * jcp.src_dsz;
    jcp.brg_stride_b = jcp.ic_block * jcp.oc * jcp.wei_dsz;

    if (jcp.ic_block == 0 || jcp.oc_block == 0) return status::unimplemented;

    // Configure matrix sizes

    if (best_brgb.is_os_block) {
        if (jcp.os_block == 0) return status::unimplemented;
        jcp.M = jcp.os_block;
        jcp.M_tail = jcp.os % jcp.os_block;
    } else {
        if (jcp.ow_block == 0) return status::unimplemented;
        jcp.M = jcp.ow_block;
        jcp.M_tail = jcp.ow % jcp.ow_block;
    }

    jcp.K = jcp.ic >= jcp.ic_block ? jcp.ic_block : 0;
    jcp.N = jcp.oc >= jcp.oc_block ? jcp.oc_block : 0;
    jcp.N_tail = jcp.oc % jcp.oc_block;
    jcp.K_tail = jcp.ic % jcp.ic_block;

    jcp.gemm_batch_size = jcp.nb_ic_blocking;
    // to avoid cache concurrent access from different threads
    size_t sc_size = sizeof(brgemm_batch_element_t);
    jcp.adjusted_batch_size
            = div_up(rnd_up(jcp.gemm_batch_size * sc_size, 4096), sc_size);

    jcp.LDA = jcp.stride_w * jcp.ic_without_padding;
    jcp.LDC = (jcp.use_buffer) ? jcp.oc_block : jcp.oc_without_padding;
    jcp.LDD = jcp.oc_without_padding;

    CHECK(pick_tags(jcp, src_md, weights_md, dst_md, bias_md));

    const auto &oscales = attr.output_scales_;
    jcp.is_oc_scale = oscales.mask_ == 1 << 1;

    // only common and per-oc-channel scales are supported
    const bool oscales_ok = one_of(oscales.mask_, 0, 1 << 1);
    if (!oscales_ok) return status::unimplemented;

    // no inp buffer or brgemm_vpad for 1x1
    jcp.exec_type = exec_base;
    jcp.inp_buffer_size = 0;
    jcp.buffer_size = jcp.LDC * jcp.M;

    return status::success;
}

void init_scratchpad(memory_tracking::registrar_t &scratchpad,
        const jit_brgemm_conv_conf_t &jcp) {
    if (jcp.brg_type == brgemm_addr || jcp.brg_type == brgemm_offs
            || (jcp.brg_type == brgemm_strd && jcp.exec_type == exec_vpad))
        scratchpad.book(key_brgemm_primitive_batch,
                (size_t)jcp.nthr * jcp.adjusted_batch_size,
                sizeof(brgemm_batch_element_t), 64);
    if (jcp.exec_type == exec_trans) {
        size_t inp_buffer_size = (size_t)jcp.nthr * jcp.inp_buffer_size;
        scratchpad.book(
                key_conv_brgemm_inp_buffer, inp_buffer_size, jcp.src_dsz);
        size_t inp_buffer_mask_size
                = (size_t)jcp.nthr * jcp.inp_buffer_mask_size;
        scratchpad.book(key_conv_brgemm_inp_buffer_mask, inp_buffer_mask_size,
                sizeof(uint8_t));
    }
    if (jcp.use_buffer) {
        scratchpad.book(key_brgemm_primitive_buffer, jcp.nthr * jcp.buffer_size,
                jcp.acc_dsz);
    }
}

} // namespace brgemm_convolution_utils

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
