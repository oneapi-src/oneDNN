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

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "cpu/cpu_primitive.hpp"
#include "cpu/x64/injectors/jit_uni_binary_injector.hpp"

#include "cpu/x64/amx_tile_configure.hpp"
#include "cpu/x64/jit_brgemm_conv_bwd_str.hpp"
#include "cpu/x64/jit_brgemm_conv_bwd_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::status;
using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;

using namespace nstl;
using namespace data_type;

using namespace jit_avx512_core_brgemm_conv_bwd_trans_kernel;
using namespace jit_avx512_core_brgemm_conv_comp_pad_kernel;

#define ndims_pick(v5, v4, v3) \
    ((ndims == 5) ? (v5) : (ndims == 4) ? (v4) : (ndims == 3) ? (v3) : 0)

template <cpu_isa_t isa>
status_t brgemm_convolution_bwd_str_t<isa>::pd_t::init(engine_t *engine) {
    using namespace data_type;

    const auto diff_src_type = diff_src_md(0)->data_type;
    const auto wei_type = weights_md(0)->data_type;
    const auto diff_dst_type = diff_dst_md(0)->data_type;

    const bool ok = is_bwd_d()
            && set_default_alg_kind(alg_kind::convolution_direct)
            && platform::has_data_type_support(diff_src_type)
            && platform::has_data_type_support(wei_type)
            && platform::has_data_type_support(diff_dst_type)
            && one_of(diff_src_type, f32, bf16) && one_of(wei_type, f32, bf16)
            && one_of(diff_dst_type, f32, bf16) && diff_dst_type == wei_type
            && IMPLICATION(diff_dst_type == f32, diff_src_type == f32)
            && IMPLICATION(with_bias(),
                    (diff_dst_type == bf16
                            && one_of(bias_md_.data_type, f32, bf16))
                            || everyone_is(f32, diff_dst_type,
                                    bias_md_.data_type, f32))
            && attr()->has_default_values() && !has_zero_dim_memory();

    if (!ok) return status::unimplemented;

    const auto is_amx = brgemm_convolution_bwd_utils::is_amx(isa);

    CHECK(brgemm_convolution_bwd_utils::init_conf(jcp_, isa, desc_,
            diff_dst_md_, weights_md_, diff_src_md_, bias_md_, attr_,
            dnnl_get_max_threads()));

    const auto adj_M = nstl::max(jcp_.M, jcp_.M_tail);

    batchsizes.resize(jcp_.max_batch + 1);
    for (int i = 0; i <= jcp_.max_batch; i++)
        batchsizes[i] = -1;

    first_bs = 0;
    bs_c = 0;
    if (jcp_.use_uker) {
        // TODO
        return status::unimplemented;
    } else {
        batchsizes[jcp_.max_batch] = bs_c;
        first_bs = jcp_.max_batch;
        bs_c++;
    }

    brgs_sz_ = bs_c * adj_M * 2 * 2 * 2;
    brgs_.resize(brgs_sz_);
    bd_masks.resize(brgs_sz_);

    const float alpha = 1.0;
    const float beta = 1.0;

    const auto &p = attr()->post_ops_;
    const int sum_idx = p.find(primitive_kind::sum);
    const bool with_sum = (sum_idx != -1);

    // is_blocking is supported for exec_trans only
    assert(IMPLICATION(jcp_.exec_type != exec_trans, !jcp_.is_is_blocking));
    assert(IMPLICATION(jcp_.is_is_blocking,
            jcp_.is_block % jcp_.iw == 0 && jcp_.is_block / jcp_.iw <= jcp_.ih
                    && jcp_.is_block / jcp_.iw == jcp_.ih_block));

    auto maybe_M_mask = [&](int brg_idx, brgemm_attr_t &brgattr, int vM,
                                int vbrgM) {
        if (!jcp_.use_M_mask) return;
        auto sm_size = vbrgM;
        bd_masks[brg_idx] = std::make_shared<std::vector<char>>();
        bd_masks[brg_idx]->resize(sm_size);
        char *bd_mask = bd_masks[brg_idx]->data();
        if (jcp_.is_is_blocking) {
            int ibrgM = 0;
            int iM = 0;
            for (int hh = 0; hh < jcp_.ih_block; hh++) {
                auto M_mask = (iM >= vM) ? 0 : 1;
                for (int ww = 0; ww < jcp_.iw_block && ibrgM < sm_size;
                        ww++, ibrgM++, iM += M_mask) {
                    bd_mask[ibrgM] = M_mask;
                }
                for (int kk = 0; kk < jcp_.iskip && ibrgM < sm_size;
                        kk++, ibrgM++) {
                    bd_mask[ibrgM] = 0;
                }
            }
            for (; ibrgM < sm_size; ibrgM++) {
                bd_mask[ibrgM] = 0;
            }
        } else {
            for (int ibrgM = 0; ibrgM < sm_size; ibrgM++) {
                bd_mask[ibrgM] = 1;
            }
        }
        brgattr.bd_mask = bd_mask;
    };

    const auto M_end = nstl::max(jcp_.M, jcp_.M_tail);
    for (int i = 0; i < M_end; i++) {
        auto vM = i + 1;
        // init only needed brgemm descriptors
        if (one_of(jcp_.exec_type, exec_trans, exec_vpad) && vM != jcp_.M
                && vM != jcp_.M_tail)
            continue;
        for (int bs = 0; bs <= jcp_.max_batch; bs++) {
            if (batchsizes[bs] == -1) continue;
            for_(int i_init = 0; i_init < 2; i_init++)
            for_(int i_N = 0; i_N < 2; i_N++)
            for (int i_K = 0; i_K < 2; i_K++) {
                auto vbeta = (i_init) ? 0 : beta;
                auto vN = (i_N) ? jcp_.N_tail : jcp_.N;
                auto vK = (i_K) ? jcp_.K_tail : jcp_.K;
                auto vbrgM = jcp_.use_M_mask
                        ? (vM == jcp_.M ? jcp_.brgM : jcp_.brgM_tail)
                        : vM;
                auto brg_idx = get_brg_idx(bs, i, i_init, i_N, i_K);
                // if brgemm_t already created then skip this iteration
                if (brgs_[brg_idx] != nullptr) continue;
                brgs_[brg_idx] = std::make_shared<brgemm_t>();
                brgemm_t *brg = brgs_[brg_idx].get();
                if (vN == 0 || vK == 0) continue;
                brgemm_strides_t brg_strides;
                brg_strides.stride_a = jcp_.brg_stride_a;
                brg_strides.stride_b = jcp_.brg_stride_b;
                brg->req_cal_comp_pads = jcp_.req_brg_comp_pad
                        && nstl::max(jcp_.l_pad, jcp_.r_pad);
                const auto strides_ptr = (jcp_.brg_type == brgemm_strd)
                        ? &brg_strides
                        : nullptr;
                CHECK(brgemm_desc_init(brg, isa, jcp_.brg_type, diff_dst_type,
                        wei_type, false, false, brgemm_row_major, alpha, vbeta,
                        jcp_.LDA, jcp_.LDB, jcp_.LDC, vbrgM, vN, vK,
                        strides_ptr));

                brgemm_attr_t brgattr;
                brgattr.use_uker = jcp_.use_uker;
                brgattr.use_interleave_stores = jcp_.use_interleave_stores;
                brgattr.hint_prefetching = jcp_.hint_prefetching;
                brgattr.max_bs = bs;
                brgattr.hint_innermost_loop = jcp_.brgemm_bd_loop_innermost
                        ? brgemm_bd_loop_innermost
                        : brgemm_ld_loop_innermost;
                if (jcp_.amx_tile_load_xx) {
                    // assuming 2x2 decomposition in amx brgemm kernel
                    // and overlap of input by kw
                    const auto bd_blocking = 2 * jcp_.amx_h;
                    const auto ld_blocking = 2 * 16;
                    brgattr.hint_expected_A_size = bd_blocking * jcp_.K
                            * jcp_.kd_block * jcp_.kh_block;
                    brgattr.hint_expected_B_size = ld_blocking * jcp_.K
                            * jcp_.kd_block * jcp_.kh_block * jcp_.kw_block;
                    brgattr.hint_expected_C_size = bd_blocking * ld_blocking;
                } else {
                    brgattr.hint_expected_A_size = 0;
                    brgattr.hint_expected_B_size = 0;
                    brgattr.hint_expected_C_size = 0;
                }

                brgattr.wary_tail_read = false;
                maybe_M_mask(brg_idx, brgattr, vM, vbrgM);
                brgattr.bd_mask_level = jcp_.use_M_mask;

                if (is_amx) {
                    brgattr.max_top_vpad = 0;
                    brgattr.max_bottom_vpad = 0;
                } else {
                    brgattr.max_top_vpad = jcp_.max_vpad;
                    brgattr.max_bottom_vpad = jcp_.max_vpad;
                }
                brgattr.generate_skip_accumulation = true;
                CHECK(brgemm_desc_set_attr(brg, brgattr));

                auto LDD = jcp_.stride_w * jcp_.ic_without_padding;
                brg->with_sum = with_sum;
                CHECK(brgemm_desc_set_postops(
                        brg, attr(), &diff_src_md_, LDD, jcp_.bia_dt));
                jcp_.amx_buf_size_per_thread
                        = nstl::max(brg->get_wsp_buffer_size(),
                                jcp_.amx_buf_size_per_thread);
            }
        }
    }

    auto scratchpad = scratchpad_registry().registrar();
    brgemm_convolution_bwd_utils::init_scratchpad(scratchpad, jcp_);

    return status::success;
}

template <cpu_isa_t isa>
status_t brgemm_convolution_bwd_str_t<isa>::add_brg_kernel(
        int bs, int M, int i_N, int i_K, int i_init) {
    if (M <= 0) return status::success;
    const auto _pd = pd();
    const auto &jcp = _pd->jcp_;
    const auto &brgs = _pd->brgs_;

    auto N = (i_N) ? jcp.N_tail : jcp.N;
    auto K = (i_K) ? jcp.K_tail : jcp.K;
    if (N <= 0 || K <= 0) return status::success;
    auto brg_idx = _pd->get_brg_idx(bs, M - 1, i_init, i_N, i_K);
    auto brg = brgs[brg_idx];
    if (!brg_kernels_[brg_idx] && brg && brg->bcast_dim > 0 && brg->load_dim > 0
            && brg->reduce_dim > 0) {
        brgemm_kernel_t *brg_kernel = nullptr;
        CHECK(brgemm_kernel_create(&brg_kernel, *brg));
        CHECK(safe_ptr_assign(brg_kernels_[brg_idx], brg_kernel));
        if (is_amx) {
            CHECK(brgemm_init_tiles(*brg, &brg_kernel_palettes_[brg_idx].a[0]));
        }
    }
    return status::success;
}

template <cpu_isa_t isa>
status_t brgemm_convolution_bwd_str_t<isa>::init(engine_t *engine) {

    const auto _pd = pd();
    const auto &jcp = _pd->jcp_;

    oscales = _pd->attr()->output_scales_.scales_;
    bia_dsz = jcp.bia_dsz;
    acc_dsz = jcp.acc_dsz;
    src_dsz = jcp.src_dsz;
    wei_dsz = jcp.wei_dsz;
    dst_dsz = jcp.dst_dsz;

    auto ndims = _pd->ndims();
    if (ndims < 3 || ndims > 5) assert(!"Invalid ndims!");

    KD = ndims_pick(jcp.kd, 1, 1);
    KH = ndims_pick(jcp.kh, jcp.kh, 1);
    KW = jcp.kw;

    EXT_KD = ndims_pick(jcp.ext_kd, 1, 1);
    EXT_KH = ndims_pick(jcp.ext_kh, jcp.ext_kh, 1);
    EXT_KW = jcp.ext_kw;

    ODP = ndims_pick(jcp.odp, 1, 1);
    OHP = ndims_pick(jcp.ohp, jcp.ohp, 1);
    OWP = jcp.owp;

    KS = KD * KH * KW;
    KD_BLOCK = ndims_pick(jcp.kd_block, 1, 1);
    KH_BLOCK = ndims_pick(jcp.kh_block, jcp.kh_block, 1);
    KW_BLOCK = jcp.kw_block;
    KD_BLOCK_PAD = ndims_pick(jcp.kd_block_pad, 1, 1);
    KH_BLOCK_PAD = ndims_pick(jcp.kh_block_pad, jcp.kh_block_pad, 1);
    ID = ndims_pick(jcp.id, 1, 1);
    IH = ndims_pick(jcp.ih, jcp.ih, 1);
    IW = jcp.iw;
    OD = ndims_pick(jcp.od, 1, 1);
    OH = ndims_pick(jcp.oh, jcp.oh, 1);
    OW = jcp.ow;
    SD = ndims_pick(jcp.stride_d, 1, 1);
    SH = ndims_pick(jcp.stride_h, jcp.stride_h, 1);
    SW = jcp.stride_w;
    FP = ndims_pick(jcp.f_pad, 0, 0);
    TP = ndims_pick(jcp.t_pad, jcp.t_pad, 0);
    LP = jcp.l_pad;
    DD = ndims_pick(jcp.dilate_d, 0, 0) + 1;
    DH = ndims_pick(jcp.dilate_h, jcp.dilate_h, 0) + 1;
    DW = jcp.dilate_w + 1;

    oc_chunks = div_up(jcp.nb_oc, jcp.nb_oc_blocking);

    // const variables used for address calculations
    src_w_sz = static_cast<dim_t>(OW) * jcp.ngroups * jcp.oc_without_padding;
    src_h_sz = OH * src_w_sz;
    src_d_sz = OD * src_h_sz;
    dst_w_sz = static_cast<dim_t>(IW) * jcp.ic_without_padding;
    dst_h_sz = IH * dst_w_sz;
    dst_d_sz = ID * dst_h_sz;

    wei_oc_sz = static_cast<dim_t>(jcp.ocp) * jcp.ic_block;
    wei_kw_sz = KW * wei_oc_sz;
    wei_kh_sz = KH * wei_kw_sz;
    wei_kd_sz = KD * wei_kh_sz;
    wei_icb_sz = jcp.nb_ic * wei_kd_sz;

    need_postwork = jcp.with_bias || jcp.with_eltwise || jcp.with_binary
            || (jcp.dst_dt != jcp.acc_dt) || jcp.with_sum || jcp.use_M_mask
            || jcp.src_zero_point || jcp.dst_zero_point;

    // ---- Initialize arrays ---------------------
    brg_kernels_.resize(_pd->brgs_sz_);
    brg_kernel_palettes_.resize(_pd->brgs_sz_);

    for (int i = 0; i < _pd->brgs_sz_; i++)
        brg_kernels_[i] = nullptr;

    CHECK(safe_ptr_assign(copy_to_pbuffer_,
            new jit_avx512_core_brgemm_conv_bwd_trans_kernel_t(jcp)));
    CHECK(copy_to_pbuffer_->create_kernel());

    const auto ow_block = jcp.owp;
    const auto oh_block = jcp.ohp;
    const auto od_block = jcp.odp;

    pbuf_w_sz = (dim_t)jcp.oc_block * jcp.kh_sets * jcp.kw_sets * ow_block;
    pbuf_h_sz = pbuf_w_sz * oh_block;
    pbuf_d_sz = pbuf_h_sz * od_block;

    is_amx = brgemm_convolution_bwd_utils::is_amx(isa);

    // #TODO: this needed only if we have d/h padding more then kd/kh
    int M_begin = 0;
    int M_end = (jcp.M_tail == jcp.M) ? 1 : 2;
    int N_begin = 0;
    int N_end = (jcp.N_tail == jcp.N) ? 1 : 2;
    int K_begin = 0;
    int K_end = (jcp.K_tail == jcp.K) ? 1 : 2;
    int i_init_begin = (div_up(jcp.nb_oc, jcp.nb_oc_blocking) == 1
                               && KD_BLOCK == KD && KH_BLOCK == KH)
            ? 1
            : 0;
    int i_init_end = 2;

    for (int bs = 0; bs <= jcp.max_batch; bs++) {
        if (_pd->batchsizes[bs] == -1) continue;

        for_(int i_N = N_begin; i_N < N_end; i_N++)
        for_(int i_M = M_begin; i_M < M_end; i_M++)
        for_(int i_init = i_init_begin; i_init < i_init_end; i_init++)
        for (int i_K = K_begin; i_K < K_end; i_K++) {
            auto M = (i_M) ? jcp.M_tail : jcp.M;
            if (M <= 0) continue;
            add_brg_kernel(bs, M, i_N, i_K, i_init);
        }
    }

    return status::success;
}

template <cpu_isa_t isa>
status_t brgemm_convolution_bwd_str_t<isa>::execute(
        const exec_ctx_t &ctx) const {
    const auto _pd = pd();
    const auto &jcp = _pd->jcp_;

    const memory_tracking::grantor_t scratchpad = ctx.get_scratchpad_grantor();
    brgemm_batch_element_t *const __restrict brg_batch_global
            = (jcp.brg_type == brgemm_strd && jcp.exec_type != exec_vpad)
            ? nullptr
            : scratchpad.template get<brgemm_batch_element_t>(
                    key_brgemm_primitive_batch);
    char *const __restrict c_buffer_global = (jcp.use_buffer)
            ? scratchpad.template get<char>(key_brgemm_primitive_buffer)
            : nullptr;

    auto inp_p_buffer = (jcp.exec_type == exec_trans)
            ? scratchpad.template get<char>(key_conv_brgemm_inp_buffer)
            : nullptr;
    auto inp_p_buffer_mask = (jcp.exec_type == exec_trans)
            ? scratchpad.template get<uint8_t>(key_conv_brgemm_inp_buffer_mask)
            : nullptr;

    char *const wsp_tile_global = is_amx
            ? scratchpad.template get<char>(key_conv_amx_tile_buffer)
            : nullptr;

    brgemm_bwd_exec_ctx_t brgemm_ctx(ctx, _pd);

    const char *const __restrict diff_dst = brgemm_ctx.diff_dst;

    // --------------- Parallel section ------------------------------
    const dim_t work_amount = static_cast<dim_t>(jcp.mb) * jcp.ngroups
            * jcp.nb_ic * jcp.nb_id * jcp.nb_ih * jcp.nb_iw;

    parallel(jcp.nthr, [&](const int ithr, const int nthr) {
        if (ithr >= work_amount) return;

        brgemm_batch_element_t *const __restrict brg_batch = brg_batch_global
                + static_cast<size_t>(ithr) * jcp.adjusted_batch_size;
        char *const __restrict c_buffer = (jcp.use_buffer)
                ? c_buffer_global + ithr * acc_dsz * jcp.buffer_size
                : nullptr;
        char *inp_buffer = (jcp.exec_type == exec_trans)
                ? inp_p_buffer + src_dsz * ithr * jcp.inp_buffer_size
                : nullptr;
        if (is_amx) {
            // Workaround: for some machines SEGFAULT possible on tile load
            // if the page was not touched before it
            for (dim_t i = 0; i < jcp.inp_buffer_size;
                    i += brgemm_convolution_bwd_utils::P4K)
                inp_buffer[i] = 0;
        }

        uint8_t *__restrict inp_buffer_mask = (jcp.exec_type == exec_trans)
                ? inp_p_buffer_mask + ithr * jcp.inp_buffer_mask_size
                : nullptr;

        char *const wsp_tile = is_amx
                ? wsp_tile_global + ithr * 2 * brgemm_convolution_bwd_utils::P4K
                : nullptr;
        dim_t start {0}, end {0};
        balance211(work_amount, nthr, ithr, start, end);
        int n {0}, g {0}, icb {0}, idb {0}, ihb {0}, iwb {0};
        if (jcp.loop_order == loop_ndhwgc)
            nd_iterator_init(start, n, jcp.mb, idb, jcp.nb_id, ihb, jcp.nb_ih,
                    iwb, jcp.nb_iw, g, jcp.ngroups, icb, jcp.nb_ic);
        else if (jcp.loop_order == loop_ngcdhw)
            nd_iterator_init(start, n, jcp.mb, g, jcp.ngroups, icb, jcp.nb_ic,
                    idb, jcp.nb_id, ihb, jcp.nb_ih, iwb, jcp.nb_iw);
        else
            assert(!"Unknown loop order");

        brgemm_bwd_thread_ctx_t btc(
                brgemm_ctx, ithr, brg_batch, c_buffer, wsp_tile);
        std::memset(btc.cur_palette.a, 0, AMX_PALETTE_SIZE);

        int last_n = -1;
        int last_g = -1;
        int last_occ = -1;
        int last_idb = -1;
        int last_ihb = -1;
        int last_iwb = -1;
        for (auto work = start; work < end; work++) {
            btc.g = g;
            btc.n = n;
            btc.icb = icb;
            btc.idb = idb;
            btc.ihb = ihb;
            btc.iwb = iwb;

            auto id_begin = idb * jcp.id_block;
            auto id_end = nstl::min(ID, id_begin + jcp.id_block);
            auto ih_begin = ihb * jcp.ih_block;
            auto ih_end = nstl::min(IH, ih_begin + jcp.ih_block);

            for_(int id = id_begin; id < id_end; id++)
            for (int ih = ih_begin; ih < ih_end; ih++) {
                for (int occ = 0; occ < oc_chunks; occ++) {
                    btc.id = id;
                    btc.ih = ih;
                    btc.occ = occ;

                    if (jcp.exec_type == exec_trans) {
                        maybe_trans_inp(ithr, diff_dst, inp_buffer,
                                inp_buffer_mask, g, n, occ, idb, ihb, iwb,
                                last_g, last_n, last_occ, last_idb, last_ihb,
                                last_iwb);
                        for (int sw = 0; sw < SW; sw++) {
                            btc.sw = sw;
                            ker_trans(btc, inp_buffer);
                        }
                    } else {
                        assert(!"Unknown exec type");
                    }
                    last_n = n;
                    last_g = g;
                    last_occ = occ;
                    last_idb = idb;
                    last_ihb = ihb;
                    last_iwb = iwb;
                }
            }
            if (jcp.loop_order == loop_ndhwgc)
                nd_iterator_step(n, jcp.mb, idb, jcp.nb_id, ihb, jcp.nb_ih, iwb,
                        jcp.nb_iw, g, jcp.ngroups, icb, jcp.nb_ic);
            else if (jcp.loop_order == loop_ngcdhw)
                nd_iterator_step(n, jcp.mb, g, jcp.ngroups, icb, jcp.nb_ic, idb,
                        jcp.nb_id, ihb, jcp.nb_ih, iwb, jcp.nb_iw);
            else
                assert(!"Unknown loop order");
        }
        if (is_amx) { amx_tile_release(); }
    });

    return status::success;
}

template <cpu_isa_t isa>
void brgemm_convolution_bwd_str_t<isa>::call_brgemm_kernel(
        brgemm_bwd_thread_ctx_t &btc, int brg_idx, int batch_size, char *ptr_C,
        char *ptr_D, const char *bias_w, int g_ic, bool do_postops,
        const void *binary_post_ops_rhs, int32_t src_zp_vals,
        int32_t *src_zp_ptr, int32_t *dst_zp_ptr, int32_t *s8s8_comp,
        bool do_only_comp, bool is_first_call_postops) const {

    const auto _pd = pd();
    const auto &jcp = _pd->jcp_;

    const auto brg_ker = brg_kernels_[brg_idx].get();
    assert(brg_ker != nullptr);

    if (is_first_call_postops) return;

    if (is_amx) {
        if (std::memcmp(btc.cur_palette.a, brg_kernel_palettes_[brg_idx].a,
                    AMX_PALETTE_SIZE)
                != 0) {
            amx_tile_configure(brg_kernel_palettes_[brg_idx].a);
            std::memcpy(btc.cur_palette.a, brg_kernel_palettes_[brg_idx].a,
                    AMX_PALETTE_SIZE);
        }
    }

    const auto do_only_pass_comp = !do_postops && jcp.src_zero_point
            && (jcp.req_brg_comp_pad || jcp.max_vpad > 0);
    const auto do_skip_accm = batch_size == 0;
    const auto maybe_do_postops = one_of(
            true, do_postops, do_only_comp, do_only_pass_comp, do_skip_accm);
    if (maybe_do_postops) {
        const brgemm_post_ops_data_t post_ops_data {
                static_cast<const char *>(bias_w),
                &oscales[jcp.is_ic_scale * g_ic], binary_post_ops_rhs,
                static_cast<size_t>(g_ic), 0, btc.brgemm_ctx.dst, 0,
                static_cast<void *>(src_zp_ptr), nullptr,
                static_cast<void *>(dst_zp_ptr), do_skip_accm, src_zp_vals,
                do_only_comp, do_only_pass_comp};

        void *scratch = is_amx ? static_cast<void *>(btc.wsp_tile)
                               : static_cast<void *>(s8s8_comp);

        if (do_postops || do_skip_accm)
            brgemm_kernel_execute_postops(brg_ker, batch_size, btc.brg_batch,
                    ptr_C, ptr_D, post_ops_data, scratch);
        else
            brgemm_kernel_execute_postops(brg_ker, batch_size, btc.brg_batch,
                    ptr_C, ptr_C, post_ops_data, scratch);
    } else
        brgemm_kernel_execute(brg_ker, batch_size, btc.brg_batch, ptr_C,
                static_cast<void *>(btc.wsp_tile));
}

template <cpu_isa_t isa>
void brgemm_convolution_bwd_str_t<isa>::maybe_trans_inp(int ithr,
        const char *__restrict src, char *__restrict inp_buffer,
        uint8_t *__restrict inp_buffer_mask, int g, int n, int occ, int idb,
        int ihb, int iwb, int last_g, int last_n, int last_occ, int last_idb,
        int last_ihb, int last_iwb) const {

    const auto _pd = pd();
    const auto &jcp = _pd->jcp_;
    const auto ocb = occ * jcp.nb_oc_blocking;

    if (last_g == g && last_n == n && last_occ == occ && last_idb == idb
            && last_ihb == ihb && last_iwb == iwb)
        return;

    auto cp = jit_brgemm_conv_bwd_trans_kernel_call_s();

    const auto oc = ocb * jcp.oc_block;
    const auto g_oc = g * jcp.oc + oc;

    const auto sw = jcp.l_pad % jcp.stride_w;
    const auto kw = (jcp.kw - 1) % jcp.stride_w;
    const auto kw_x = (jcp.kw - 1) - nstl::modulo(kw - sw, jcp.stride_w);
    const auto ow = (iwb * jcp.iw_block + jcp.l_pad - kw_x * (jcp.dilate_w + 1))
            / jcp.stride_w;

    int od_start {0}, od_end {0}, oh_start {0}, oh_end {0};

    const auto sh = jcp.t_pad % jcp.stride_h;
    const auto kh = (jcp.kh - 1) % jcp.stride_h;
    const auto kh_x = (jcp.kh - 1) - nstl::modulo(kh - sh, jcp.stride_h);
    oh_start = (ihb * jcp.ih_block + jcp.t_pad - kh_x * (jcp.dilate_h + 1))
            / jcp.stride_h;
    oh_end = oh_start + jcp.oh_block;

    const auto sd = jcp.f_pad % jcp.stride_d;
    const auto kd = (jcp.kd - 1) % jcp.stride_d;
    const auto kd_x = (jcp.kd - 1) - nstl::modulo(kd - sd, jcp.stride_d);
    od_start = (idb * jcp.id_block + jcp.f_pad - kd_x * (jcp.dilate_d + 1))
            / jcp.stride_d;
    od_end = od_start + jcp.od_block;

    const auto rows_to_copy = min(jcp.oh, oh_end) - max(0, oh_start);
    cp.iwb = iwb;
    cp.oc = oc;
    const auto ow_buf = ow;
    dim_t inp_offset_start, out_offset_start;

    cp.t_pad = 0;
    cp.b_pad = 0;
    cp.h_count = max(0, rows_to_copy);

    const auto oh_buf = max(0, oh_start);

    inp_offset_start = static_cast<dim_t>(n) * src_d_sz
            + max(0, oh_start) * src_w_sz
            + max(0, ow) * jcp.ngroups * jcp.oc_without_padding + g_oc;
    out_offset_start = oh_buf * pbuf_w_sz
            + ow_buf * jcp.oc_block * jcp.kh_sets * jcp.kw_sets;

    for (int od = max(0, od_start); od < min(jcp.od, od_end); od++) {
        const auto inp_offset = inp_offset_start + od * src_h_sz;
        const auto od_buf = od;
        const auto out_offset = out_offset_start + od_buf * pbuf_h_sz;
        cp.src = src + src_dsz * inp_offset;
        cp.dst = inp_buffer + src_dsz * out_offset;

        (*copy_to_pbuffer_)(&cp);
    }
}

template <cpu_isa_t isa>
void brgemm_convolution_bwd_str_t<isa>::ker_trans(
        brgemm_bwd_thread_ctx_t &btc, char *inp_buffer) const {

    const auto _pd = pd();
    const auto &jcp = _pd->jcp_;
    auto ndims = _pd->ndims();

    const char *const __restrict weights = btc.brgemm_ctx.weights;
    const char *const __restrict bias = btc.brgemm_ctx.bias;
    char *const __restrict dst = btc.brgemm_ctx.dst;
    const std::vector<const void *> &post_ops_binary_rhs_arg_vec
            = btc.brgemm_ctx.post_ops_binary_rhs_arg_vec;
    const int ic = btc.icb * jcp.ic_block;
    const int g_ic = btc.g * jcp.ic + ic;
    const int ocb = btc.occ * jcp.nb_oc_blocking;
    const int oc = ocb * jcp.oc_block;
    const int iw = btc.iwb * jcp.iw_block + btc.sw;
    const int ih = btc.ih;
    const int id = btc.id;

    // od = (id + FP - kd * DD) / SD <-- general relation for all sets of (od, id, kd) that overlap
    // for a given index from diff_src, we need to find the appropriate stride sector
    int sd(0), sh(0), sw(0), od_test(0), oh_test(0), ow_test(0), kd_s_(0),
            kh_s_(0);

    while (true) {
        od_test = id + FP - sd * DD;
        if (od_test % SD == 0) break;
        sd++;
    }

    const int kd_f_ = min(KD, div_up(id + FP + 1, DD));
    kd_s_ = max(0, div_up(id + FP - OD * SD + 1, DD));

    while (kd_s_ % SD != sd) {
        kd_s_++;
    }

    while (true) {
        oh_test = ih + TP - sh * DH;
        if (oh_test % SH == 0) break;
        sh++;
    }

    const int kh_f_ = min(KH, div_up(ih + TP + 1, DH));
    kh_s_ = max(0, div_up(ih + TP - OH * SH + 1, DH));
    while (kh_s_ % SH != sh) {
        kh_s_++;
    }

    while (true) {
        ow_test = iw + LP - sw * DW;
        if (ow_test % SW == 0) break;
        sw++;
    }

    const int kw_f = KW;
    int kw_s = 0;

    while (kw_s % SW != sw) {
        kw_s++;
    }

    const auto kh_f = ndims_pick(kh_f_, kh_f_, 1);
    const auto kh_s = jcp.is_is_blocking ? 0 : ndims_pick(kh_s_, kh_s_, 0);

    const auto kd_f = ndims_pick(kd_f_, 1, 1);
    const auto kd_s = jcp.is_is_blocking ? 0 : ndims_pick(kd_s_, 0, 0);

    const bool is_oc_tail
            = (btc.occ == oc_chunks - 1 && ((jcp.oc - oc) % jcp.oc_block != 0));

    const bool is_ic_tail = (jcp.ic - ic < jcp.ic_block);
    const char *const __restrict bias_w
            = bias ? bias + (bias_d.blk_off(g_ic) * bia_dsz) : nullptr;
    const auto nb_oc_b = nstl::min(jcp.nb_oc_blocking, jcp.nb_oc - ocb)
            - (is_oc_tail ? 1 : 0);
    char *const __restrict dst_base = dst + dst_dsz * (btc.n * dst_d_sz + g_ic);
    char *ptr_C;
    char *ptr_D;
    int kd_b(0), kd_e(0), kh_b(0), kh_e(0), k_l(0);

    const auto wei_base
            = weights + wei_dsz * (btc.g * wei_icb_sz + btc.icb * wei_kd_sz);
    const int iw_b {iw};

    ptr_D = dst_base
            + dst_dsz
                    * (id * dst_h_sz + ih * dst_w_sz
                            + iw_b * jcp.ic_without_padding);
    ptr_C = (jcp.use_buffer) ? btc.c_buffer : static_cast<char *>(ptr_D);

    const auto ker_i = (jcp.M > 0 ? jcp.M : jcp.M_tail) - 1;

    bool is_first_call_postops = false,
         is_first_call_postops_state_changed = false;
    const auto call_brgemm = [&](int brg_idx, int oc_block_s, int n_oc_blocks,
                                     bool do_postops) {
        const auto kh_ee = kh_e;
        const auto kw_e = kw_f;
        const auto pbuf_base = inp_buffer;

        int k_sum = 0;
        for (int i_ocb = 0; i_ocb < n_oc_blocks; i_ocb++) {
            const auto oc_off = (oc_block_s + i_ocb) * jcp.oc_block;
            const auto wei_oc = oc + oc_off;
            const auto n_ocb_off = i_ocb * k_l;
            const auto pbuf_base_oc = pbuf_base;
            const auto wei_base_oc = wei_base + wei_dsz * wei_oc * jcp.ic_block;

            auto k = 0;
            for (int kd = kd_b; kd < kd_e; kd++) {
                auto od = (id - kd * DD + FP);
                if (od % SD != 0) continue;
                od /= SD;
                const auto pbuf_base_kd
                        = pbuf_base_oc + src_dsz * od * pbuf_h_sz;
                const auto wei_base_kd = wei_base_oc + wei_dsz * kd * wei_kh_sz;
                for (int kh = kh_b; kh < kh_ee; kh++) {
                    auto oh = (ih - kh * DH + TP);
                    if (oh % SH != 0) continue;
                    oh /= SH;
                    const auto pbuf_base_kh
                            = pbuf_base_kd + src_dsz * oh * pbuf_w_sz;
                    const auto wei_base_kh
                            = wei_base_kd + wei_dsz * kh * wei_kw_sz;
                    for (int kw = kw_s; kw < kw_e; kw += SW) {
                        const auto ow = (iw - kw * DW + LP) / SW;
                        // inp_buffer layout is Cdhw<oc_block>c
                        btc.brg_batch[n_ocb_off + k].ptr.A = pbuf_base_kh
                                + src_dsz * (ow + jcp.l_ovf) * jcp.oc_block;
                        btc.brg_batch[n_ocb_off + k].vvpad.top = 0;
                        btc.brg_batch[n_ocb_off + k].vvpad.bottom = 0;
                        // general wei layout is gIdhwO<block_i><block_o>
                        btc.brg_batch[n_ocb_off + k].ptr.B
                                = wei_base_kh + wei_dsz * kw * wei_oc_sz;
                        k++;
                    }
                }
            }
            k_sum += k;
        }
        call_brgemm_kernel(btc, brg_idx, k_sum, ptr_C, ptr_D, bias_w, g_ic,
                do_postops, post_ops_binary_rhs_arg_vec.data(), 0, nullptr,
                nullptr, nullptr, false, is_first_call_postops);
        if (!is_first_call_postops_state_changed) {
            const auto do_only_pass_comp = !do_postops && jcp.src_zero_point
                    && (jcp.req_brg_comp_pad || jcp.max_vpad > 0);
            const auto do_skip_accm = k_sum == 0;
            is_first_call_postops
                    = one_of(true, do_postops, do_only_pass_comp, do_skip_accm);
            is_first_call_postops_state_changed = true;
        }

        MAYBE_UNUSED(bias_w);
        MAYBE_UNUSED(ptr_C);
        MAYBE_UNUSED(post_ops_binary_rhs_arg_vec);
    };

    const auto kdhw_loop = [&]() {
        const auto do_init = btc.occ == 0 && kd_b == kd_s && kh_b == kh_s;
        const auto do_postwork = need_postwork && btc.occ == (oc_chunks - 1)
                && kd_e == kd_f && kh_e == kh_f;

        const int kd_l = div_up(kd_e - kd_b, SD);
        const int kh_l = div_up(kh_e - kh_b, SH);
        const int kw_l = div_up(kw_f - kw_s, SW);
        k_l = kd_l * kh_l * kw_l;

        int kernel_idx[2][2];
        kernel_idx[false][false]
                = _pd->get_brg_idx(k_l, ker_i, false, is_ic_tail, false);
        kernel_idx[true][false]
                = _pd->get_brg_idx(k_l, ker_i, true, is_ic_tail, false);
        kernel_idx[false][true]
                = _pd->get_brg_idx(k_l, ker_i, false, is_ic_tail, true);
        kernel_idx[true][true]
                = _pd->get_brg_idx(k_l, ker_i, true, is_ic_tail, true);

        if (nb_oc_b > 0) {
            const auto brg_idx = kernel_idx[do_init][false];
            call_brgemm(brg_idx, 0, nb_oc_b, do_postwork && !is_oc_tail);
        }

        if (is_oc_tail) {
            const auto use_init_ker = (do_init && nb_oc_b == 0);
            const auto brg_oc_tail_idx = kernel_idx[use_init_ker][true];
            call_brgemm(brg_oc_tail_idx, nb_oc_b, 1, do_postwork);
        }
    };

    if (kd_f > kd_s && kh_f > kh_s) {
        // kw values covering full ow_block
        for (kd_b = kd_s; kd_b < kd_f; kd_b += KD_BLOCK) {
            kd_e = nstl::min(kd_f, kd_b + KD_BLOCK);
            for (kh_b = kh_s; kh_b < kh_f; kh_b += KH_BLOCK) {
                kh_e = nstl::min(kh_f, kh_b + KH_BLOCK);
                kdhw_loop();
            }
        }
    } else {
        kd_b = kd_e = kd_s;
        kh_b = kh_e = kh_s;
        kdhw_loop();
    }
}

template struct brgemm_convolution_bwd_str_t<avx512_core>;
template struct brgemm_convolution_bwd_str_t<avx512_core_bf16>;
template struct brgemm_convolution_bwd_str_t<avx512_core_bf16_amx_bf16>;

} // namespace x64

} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
