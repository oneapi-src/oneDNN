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

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/x64/jit_brgemm_conv.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::status;
using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;

using namespace nstl;
using namespace data_type;

#define ndims_pick(v5, v4, v3) \
    ((ndims == 5) ? (v5) : (ndims == 4) ? (v4) : (ndims == 3) ? (v3) : 0)

#ifndef NDEBUG
namespace {
bool check_weight_layout(const jit_brgemm_conv_conf_t &jcp) {
    using namespace dnnl::impl::format_tag;
    return one_of(jcp.wei_tag, gOdhwi64o, Odhwi64o, gOdhwI64o4i, OdhwI64o4i,
            gOdhwI64o2i, OdhwI64o2i, gOwi64o, Owi64o, gOwI64o4i, OwI64o4i,
            gOwI64o2i, OwI64o2i, gOhwi64o, Ohwi64o, gOhwI64o4i, OhwI64o4i,
            gOhwI64o2i, OhwI64o2i, gOdhwi48o, Odhwi48o, gOdhwI48o4i, OdhwI48o4i,
            gOdhwI48o2i, OdhwI48o2i, gOwi48o, Owi48o, gOwI48o4i, OwI48o4i,
            gOwI48o2i, OwI48o2i, gOhwi48o, Ohwi48o, gOhwI48o4i, OhwI48o4i,
            gOhwI48o2i, OhwI48o2i, gOdhwi32o, Odhwi32o, gOdhwI32o4i, OdhwI32o4i,
            gOdhwI32o2i, OdhwI32o2i, gOwi32o, Owi32o, gOwI32o4i, OwI32o4i,
            gOwI32o2i, OwI32o2i, gOhwi32o, Ohwi32o, gOhwI32o4i, OhwI32o4i,
            gOhwI32o2i, OhwI32o2i, gOdhwi16o, Odhwi16o, gOdhwI16o4i, OdhwI16o4i,
            gOdhwI16o2i, OdhwI16o2i, gOwi16o, Owi16o, gOwI16o4i, OwI16o4i,
            gOwI16o2i, OwI16o2i, gOhwi16o, Ohwi16o, gOhwI16o4i, OhwI16o4i,
            gOhwI16o2i, OhwI16o2i);
}
} // namespace
#endif

template <cpu_isa_t isa, data_type_t src_type, data_type_t wei_type,
        data_type_t dst_type>
status_t
brgemm_convolution_fwd_t<isa, src_type, wei_type, dst_type>::pd_t::init(
        engine_t *engine) {
    auto check_attr = [=]() {
        if (utils::one_of(src_type, data_type::u8, data_type::s8)) {
            return attr()->has_default_values(
                    primitive_attr_t::skip_mask_t::oscale
                    | primitive_attr_t::skip_mask_t::post_ops);
        } else {
            return attr()->has_default_values(
                    primitive_attr_t::skip_mask_t::post_ops);
        }
    };

    bool ok = true && is_fwd()
            && set_default_alg_kind(alg_kind::convolution_direct)
            && expect_data_types(src_type, wei_type, data_type::undef, dst_type,
                    data_type::undef)
            && IMPLICATION(with_bias(),
                    ((utils::one_of(src_type, data_type::u8, data_type::s8)
                             && utils::one_of(bias_md_.data_type,
                                     data_type::f32, data_type::s32,
                                     data_type::s8, data_type::u8))
                            || (utils::one_of(src_type, data_type::bf16)
                                    && utils::one_of(bias_md_.data_type,
                                            data_type::f32, data_type::bf16))
                            || (utils::one_of(src_type, data_type::f32)
                                    && utils::one_of(bias_md_.data_type,
                                            data_type::f32))))
            && check_attr() && !has_zero_dim_memory();

    if (!ok) return status::unimplemented;

    CHECK(brgemm_convolution_utils::init_conf(jcp_, isa, *desc(), src_md_,
            weights_md_, dst_md_, bias_md_, *attr(), dnnl_get_max_threads()));

    brgs_sz_ = jcp_.ow_block * 2 * 2 * 2;
    brgs_.resize(brgs_sz_);
    for (int i = 0; i < brgs_sz_; i++)
        brgs_[i].bcast_dim = brgs_[i].load_dim = brgs_[i].reduce_dim = 0;

    const float alpha = 1.0;
    const float beta = 1.0;

    const auto &p = attr_->post_ops_;
    const int sum_idx = p.find(primitive_kind::sum);
    with_sum = (sum_idx != -1);

    for (int i = 0; i < jcp_.M; i++) {
        auto vM = i + 1;
        // init only needed brgemm descriptors
        if (utils::one_of(jcp_.exec_type, exec_trans, exec_vpad) && vM != jcp_.M
                && vM != jcp_.M_tail)
            continue;
        for_(int i_init = 0; i_init < 2; i_init++)
        for_(int i_N = 0; i_N < 2; i_N++)
        for (int i_K = 0; i_K < 2; i_K++) {
            auto vbeta = (i_init) ? 0 : beta;
            auto vN = (i_N) ? jcp_.N_tail : jcp_.N;
            auto vK = (i_K) ? jcp_.K_tail : jcp_.K;
            brgemm_t &brg = brgs_[get_brg_idx(i, i_init, i_N, i_K)];
            if (vN == 0 || vK == 0) continue;
            brgemm_strides_t brg_strides;
            brg_strides.stride_a = jcp_.brg_stride_a;
            brg_strides.stride_b = jcp_.brg_stride_b;
            const auto strides_ptr
                    = (jcp_.brg_type == brgemm_strd) ? &brg_strides : nullptr;
            CHECK(brgemm_desc_init(&brg, isa, jcp_.brg_type, src_type, wei_type,
                    false, false, brgemm_row_major, alpha, vbeta, jcp_.LDA,
                    jcp_.LDB, jcp_.LDC, vM, vN, vK, strides_ptr));

            brgemm_attr_t brgattr;
            brgattr.max_bs = jcp_.max_batch;
            brgattr.max_top_vpad = jcp_.max_vpad;
            brgattr.max_bottom_vpad = jcp_.max_vpad;
            brgattr.wary_tail_read = false;
            CHECK(brgemm_desc_set_attr(&brg, brgattr));

            auto dt_d = dst_type;
            auto LDD = jcp_.oc_without_padding;
            brg.with_sum = with_sum;
            CHECK(brgemm_desc_set_postops(
                    &brg, attr(), dt_d, LDD, jcp_.bia_dt));
        }
    }

    auto scratchpad = scratchpad_registry().registrar();
    brgemm_convolution_utils::init_scratchpad(scratchpad, jcp_);

    return status::success;
}

template <cpu_isa_t isa, data_type_t src_type, data_type_t wei_type,
        data_type_t dst_type>
brgemm_convolution_fwd_t<isa, src_type, wei_type,
        dst_type>::brgemm_convolution_fwd_t(const pd_t *apd)
    : primitive_t(apd), bias_d(pd()->weights_md(1)) {}

template <cpu_isa_t isa, data_type_t src_type, data_type_t wei_type,
        data_type_t dst_type>
void brgemm_convolution_fwd_t<isa, src_type, wei_type, dst_type>::get_kw_range(
        int ow, int &kw_s, int &kw_full_s, int &kw_full_f, int &kw_f) const {
    const auto &jcp = pd()->jcp_;
    // TODO: calculate these values instead direct loop by kw

    const bool is_ow_tail = (jcp.ow - ow < jcp.ow_block);
    const auto M = is_ow_tail ? jcp.M_tail : jcp.M;
    kw_s = kw_full_s = kw_full_f = kw_f = -1;
    for (int kw = 0; kw < jcp.kw; kw++) {
        int ow_s {0}, ow_f {0};
        get_ow_range(ow, kw, ow_s, ow_f);
        if (ow_s < ow_f) {
            if (kw_s == -1) kw_s = kw;
            kw_f = kw + 1;
            if (ow_f - ow_s == M) {
                if (kw_full_s == -1) kw_full_s = kw;
                kw_full_f = kw + 1;
            }
        }
    }
    if (kw_f == -1) {
        kw_s = 0;
        kw_f = 0;
    }
    if (kw_full_f == -1) kw_full_s = kw_full_f = kw_f;
}

template <cpu_isa_t isa, data_type_t src_type, data_type_t wei_type,
        data_type_t dst_type>
void brgemm_convolution_fwd_t<isa, src_type, wei_type, dst_type>::get_ow_range(
        int ow, int kw, int &ow_s, int &ow_f) const {
    const auto &jcp = pd()->jcp_;

    const bool is_ow_tail = (jcp.ow - ow < jcp.ow_block);
    const auto M = is_ow_tail ? jcp.M_tail : jcp.M;

    const auto IW = jcp.iw;
    const auto SW = jcp.stride_w;
    const auto LP = jcp.l_pad;
    const auto DW = jcp.dilate_w + 1;

    const auto iiw = ow * SW - LP;
    auto iw_lp = iiw + kw * DW;
    const auto iw_rp = iw_lp + (M - 1) * SW - IW + 1;
    ow_s = ow;

    int ker_idx = 0;
    if (iw_lp < 0) {
        iw_lp = nstl::abs(iw_lp);
        ker_idx += div_up(iw_lp, SW);
        ow_s += ker_idx;
    }
    if (iw_rp > 0) ker_idx += div_up(iw_rp, SW);
    ow_f = ow_s + (M - ker_idx);
    ow_s = nstl::min(ow_s, ow + M);
    ow_f = nstl::min(nstl::max(ow_f, ow_s), ow + M);
}

template <cpu_isa_t isa, data_type_t src_type, data_type_t wei_type,
        data_type_t dst_type>
status_t
brgemm_convolution_fwd_t<isa, src_type, wei_type, dst_type>::add_brg_kernel(
        int M, int i_N, int i_K, int i_init) {
    const auto &jcp = pd()->jcp_;
    const auto &brgs = pd()->brgs_;

    auto N = (i_N) ? jcp.N_tail : jcp.N;
    auto K = (i_K) ? jcp.K_tail : jcp.K;
    if (N <= 0 || K <= 0) return status::success;
    auto brg_idx = get_brg_idx(M - 1, i_init, i_N, i_K);
    auto brg = brgs[brg_idx];
    if (!brg_kernels_[brg_idx] && brg.bcast_dim > 0 && brg.load_dim > 0
            && brg.reduce_dim > 0) {
        brgemm_kernel_t *brg_kernel = nullptr;
        CHECK(brgemm_kernel_create(&brg_kernel, brg));
        CHECK(safe_ptr_assign(brg_kernels_[brg_idx], brg_kernel));
    }
    return status::success;
}

template <cpu_isa_t isa, data_type_t src_type, data_type_t wei_type,
        data_type_t dst_type>
status_t
brgemm_convolution_fwd_t<isa, src_type, wei_type, dst_type>::add_po_kernel(
        brgemm_t &bcfg, int ker_idx, bool is_init) {
    const auto &jcp = pd()->jcp_;

    bcfg.LDD = (is_init && jcp.use_buffer) ? jcp.LDC : jcp.LDD;
    bcfg.dt_c = (!is_init && jcp.use_buffer) ? jcp.acc_dt : jcp.dst_dt; // inp
    bcfg.dt_d = (is_init && jcp.use_buffer) ? jcp.acc_dt : jcp.dst_dt; // out
    bcfg.alpha = bcfg.beta = (is_init ? 0 : 1);
    CHECK(safe_ptr_assign(kernels_po_[ker_idx],
            new jit_brgemm_kernel_post_ops(jcp, bcfg, *pd()->attr())));
    kernels_po_[ker_idx]->create_kernel();
    return status::success;
}

template <cpu_isa_t isa, data_type_t src_type, data_type_t wei_type,
        data_type_t dst_type>
void brgemm_convolution_fwd_t<isa, src_type, wei_type,
        dst_type>::add_po_kernels(int i_N, int init_bcast_dim, int po_bcast_dim,
        bool need_postwork) {
    const auto &jcp = pd()->jcp_;
    const auto &brgs = pd()->brgs_;

    auto N = (i_N) ? jcp.N_tail : jcp.N;
    if (N <= 0) return;
    auto i_K = (jcp.K_tail > 0);

    if (init_bcast_dim > 0) {
        auto init_cfg = brgs[get_brg_idx(init_bcast_dim - 1, 0, i_N, i_K)];
        init_cfg.bcast_dim = init_bcast_dim;
        auto ker_init_idx = get_ker_po_idx(init_bcast_dim - 1, false, i_N);
        if (init_cfg.load_dim > 0 && kernels_po_[ker_init_idx] == nullptr)
            add_po_kernel(init_cfg, ker_init_idx, true);
    }

    if ((need_postwork || jcp.use_buffer) && po_bcast_dim > 0) {
        auto po_cfg = brgs[get_brg_idx(po_bcast_dim - 1, 0, i_N, i_K)];
        po_cfg.bcast_dim = po_bcast_dim;
        auto ker_po_idx = get_ker_po_idx(po_bcast_dim - 1, true, i_N);
        if (po_cfg.load_dim > 0 && kernels_po_[ker_po_idx] == nullptr)
            add_po_kernel(po_cfg, ker_po_idx, false);
    }
}

template <cpu_isa_t isa, data_type_t src_type, data_type_t wei_type,
        data_type_t dst_type>
status_t brgemm_convolution_fwd_t<isa, src_type, wei_type, dst_type>::init(
        engine_t *engine) {

    const auto &jcp = pd()->jcp_;
    assert(check_weight_layout(jcp));

    oscales = pd()->attr()->output_scales_.scales_;
    bia_dsz = jcp.bia_dsz;
    acc_dsz = jcp.acc_dsz;
    src_dsz = jcp.src_dsz;
    wei_dsz = jcp.wei_dsz;

    auto ndims = pd()->ndims();
    if (ndims < 3 || ndims > 5) assert(!"Invalid ndims!");

    KD = ndims_pick(jcp.kd, 1, 1);
    KH = ndims_pick(jcp.kh, jcp.kh, 1);
    KW = jcp.kw;
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

    ic_chunks = div_up(jcp.nb_ic, jcp.nb_ic_blocking);

    // const variables used for address calculations
    src_w_sz = (dim_t)IW * jcp.ic_without_padding;
    src_h_sz = IH * src_w_sz;
    src_d_sz = ID * src_h_sz;
    dst_w_sz = (dim_t)OW * jcp.oc_without_padding;
    dst_h_sz = OH * dst_w_sz;
    dst_d_sz = OD * dst_h_sz;

    last_ic_block = (wei_type == f32) ? 1 : ((wei_type == bf16) ? 2 : 4);
    wei_ic_sz = (dim_t)rnd_up(jcp.ic, last_ic_block) * jcp.oc_block;
    wei_kw_sz = KW * wei_ic_sz;
    wei_kh_sz = KH * wei_kw_sz;
    wei_kd_sz = KD * wei_kh_sz;
    wei_ocb_sz = jcp.nb_oc * wei_kd_sz;

    pbuf_w_sz = jcp.iwp * jcp.ic_block;
    pbuf_h_sz = jcp.ihp * pbuf_w_sz;
    pbuf_d_sz = jcp.idp * pbuf_h_sz;

    need_postwork = jcp.with_bias || jcp.with_eltwise
            || (one_of(src_type, u8, s8) && wei_type == s8) // oscales needed
            || (jcp.dst_dt != jcp.acc_dt) || jcp.with_sum;

    // ---- Initialize arrays ---------------------
    brg_kernels_.resize(pd()->brgs_sz_);
    for (int i = 0; i < pd()->brgs_sz_; i++)
        brg_kernels_[i] = nullptr;

    int num_kernels = jcp.ow_block;
    kernels_po_.resize(num_kernels * 2 * 2);
    for (int i = 0; i < num_kernels; i++) {
        for_(int i_init = 0; i_init < 2; i_init++)
        for (int i_N = 0; i_N < 2; i_N++)
            kernels_po_[get_ker_po_idx(i, i_init, i_N)] = nullptr;
    }
    // ----------------------------------------------

    CHECK(safe_ptr_assign(copy_to_pbuffer_,
            new jit_avx512_core_brgemm_conv_trans_kernel_t(jcp)));
    CHECK(copy_to_pbuffer_->create_kernel());

    // #TODO: this needed only if we have d/h padding more then kd/kh
    for_(int i_N = 0; i_N < 2; i_N++)
    for_(int i_M = 0; i_M < 2; i_M++)
    for_(int i_init = 0; i_init < 2; i_init++)
    for (int i_K = 0; i_K < 2; i_K++) {
        auto M = (i_M) ? jcp.M_tail : jcp.M;
        if (M <= 0) continue;
        add_brg_kernel(M, i_N, i_K, i_init);
    }

    for_(int i_N = 0; i_N < 2; i_N++)
    for (int i_M = 0; i_M < 2; i_M++) {
        // init init and po_kernels for cases then we never call brgemm kernels
        // e.g. for d/h padded areas
        // TODO: do this only if d/h padding > kd/kh
        auto M = (i_M) ? jcp.M_tail : jcp.M;
        add_po_kernels(i_N, M, M, need_postwork);
    }

    if (jcp.exec_type == exec_base) {
        // create brgemm kernels for ow_blocks with padded areas and
        // apply post-ops on final iteration by kw to padded areas in ow_block
        int kw_s {0}, kw_full_s {0}, kw_full_f {0}, kw_f {0}, ow_s {0},
                ow_f {0};
        for (int ow = 0; ow < OW; ow += jcp.ow_block) {
            get_kw_range(ow, kw_s, kw_full_s, kw_full_f, kw_f);
            for (int kw = kw_s; kw < kw_f; kw++) {
                get_ow_range(ow, kw, ow_s, ow_f);
                if (ow_f - ow_s <= 0) continue;

                auto M = ow_f - ow_s;
                if (M <= 0) continue;
                for_(int i_init = 0; i_init < 2; i_init++)
                for_(int i_N = 0; i_N < 2; i_N++)
                for (int i_K = 0; i_K < 2; i_K++) {
                    add_brg_kernel(M, i_N, i_K, i_init);
                }
            }

            bool is_ow_tail = (jcp.ow - ow < jcp.ow_block);
            for_(int i_N = 0; i_N < 2; i_N++)
            for (int i_side = 0; i_side < 2; i_side++) {
                auto M = is_ow_tail ? jcp.M_tail : jcp.M;
                if (M <= 0) continue;
                get_ow_range(ow, kw_s, ow_s, ow_f);
                const auto init_bcast_dim
                        = (i_side == 0) ? (ow_s - ow) : (ow + M - ow_f);
                get_ow_range(ow, kw_f - 1, ow_s, ow_f);
                const auto po_bcast_dim
                        = (i_side == 0) ? (ow_s - ow) : (ow + M - ow_f);
                add_po_kernels(
                        i_N, init_bcast_dim, po_bcast_dim, need_postwork);
            }

            if (kw_f == jcp.kw && kw_s == 0) break;
        }

        for (int ow = (jcp.nb_ow - 1) * jcp.ow_block; ow >= 0;
                ow -= jcp.ow_block) {
            get_kw_range(ow, kw_s, kw_full_s, kw_full_f, kw_f);
            for (int kw = kw_s; kw < kw_f; kw++) {
                get_ow_range(ow, kw, ow_s, ow_f);
                if (ow_f - ow_s <= 0) continue;

                auto M = ow_f - ow_s;
                if (M <= 0) continue;
                for_(int i_init = 0; i_init < 2; i_init++)
                for_(int i_N = 0; i_N < 2; i_N++)
                for (int i_K = 0; i_K < 2; i_K++) {
                    add_brg_kernel(M, i_N, i_K, i_init);
                }
            }

            bool is_ow_tail = (jcp.ow - ow < jcp.ow_block);

            for_(int i_N = 0; i_N < 2; i_N++)
            for (int i_side = 0; i_side < 2; i_side++) {
                auto M = is_ow_tail ? jcp.M_tail : jcp.M;
                if (M <= 0) continue;
                get_ow_range(ow, kw_s, ow_s, ow_f);
                const auto init_bcast_dim
                        = (i_side == 0) ? (ow_s - ow) : (ow + M - ow_f);
                get_ow_range(ow, kw_f - 1, ow_s, ow_f);
                const auto po_bcast_dim
                        = (i_side == 0) ? (ow_s - ow) : (ow + M - ow_f);
                add_po_kernels(
                        i_N, init_bcast_dim, po_bcast_dim, need_postwork);
            }

            if (kw_f == jcp.kw && kw_s == 0) break;
        }
    }

    // pre-calculated values
    if (jcp.exec_type == exec_vpad) {
        owb_kw_top_vpads.resize(jcp.nb_ow * jcp.kw);
        owb_kw_bottom_vpads.resize(jcp.nb_ow * jcp.kw);

        for (int owb = 0; owb < jcp.nb_ow; owb++) {
            const int ow = owb * jcp.ow_block;
            const bool is_ow_tail = (jcp.ow - ow < jcp.ow_block);
            const int ow_b {ow}, ow_e {ow + (is_ow_tail ? jcp.M_tail : jcp.M)};
            const auto ow_l = ow_e - ow_b;
            MAYBE_UNUSED(ow_l);
            assert(0 <= ow_l && ow_l <= jcp.ow_block);
            const auto iiw_b = ow_b * SW - LP;
            const auto iiw_e = (ow_e - 1) * SW - LP + 1;
            const auto iiw_l = iiw_e - iiw_b;
            for (int kw = 0; kw < KW; kw++) {
                const auto iw = iiw_b + kw * DW;
                const auto top_vpad = iw >= 0 ? 0 : div_up(abs(iw), SW);
                const auto bottom_vpad
                        = iw + iiw_l <= IW ? 0 : div_up(iw + iiw_l - IW, SW);
                assert(top_vpad == 0 || bottom_vpad == 0);
                owb_kw_top_vpads[owb * KW + kw] = top_vpad;
                owb_kw_bottom_vpads[owb * KW + kw] = bottom_vpad;
            }
        }
    }

    return status::success;
}
template <cpu_isa_t isa, data_type_t src_type, data_type_t wei_type,
        data_type_t dst_type>
status_t brgemm_convolution_fwd_t<isa, src_type, wei_type, dst_type>::execute(
        const exec_ctx_t &ctx) const {
    const auto &jcp = pd()->jcp_;

    const src_data_t *const __restrict src
            = CTX_IN_MEM(const src_data_t *, DNNL_ARG_SRC);
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
            ? scratchpad.template get<src_data_t>(key_conv_brgemm_inp_buffer)
            : nullptr;
    auto inp_p_buffer_mask = (jcp.exec_type == exec_trans)
            ? scratchpad.template get<uint8_t>(key_conv_brgemm_inp_buffer_mask)
            : nullptr;

    // --------------- Parallel section ------------------------------
    const int work_amount = jcp.mb * jcp.ngroups * jcp.nb_oc * jcp.nb_od
            * jcp.nb_oh * jcp.nb_ow;

    // DODO: consider loop by icc be innermost because for current
    // implementation if we use buffer then we accumulate in it only on row
    // or made ic_chunks = 1 if use_buffer
    // or (looks more general) increase buffer size to store several rows

#define BRGC_WO(...) \
    parallel(pd()->jcp_.nthr, [&](const int ithr, const int nthr) { \
        if (ithr >= work_amount) return; \
        brgemm_batch_element_t *const __restrict brg_batch \
                = brg_batch_global + ithr * 16 * jcp.gemm_batch_size; \
        char *const __restrict c_buffer = (jcp.use_buffer) \
                ? c_buffer_global + ithr * acc_dsz * jcp.LDC * jcp.M \
                : nullptr; \
        src_data_t *inp_buffer = (jcp.exec_type == exec_trans) \
                ? inp_p_buffer + ithr * jcp.inp_buffer_size \
                : nullptr; \
        uint8_t *__restrict inp_buffer_mask = (jcp.exec_type == exec_trans) \
                ? inp_p_buffer_mask + ithr * jcp.inp_buffer_mask_size \
                : nullptr; \
        int start {0}, end {0}; \
        balance211(work_amount, nthr, ithr, start, end); \
        int n {0}, g {0}, ocb {0}, odb {0}, ohb {0}, owb {0}; \
        nd_iterator_init(start, __VA_ARGS__); \
        int last_mb = -1; \
        int last_g = -1; \
        for (auto work = start; work < end; work++) { \
            if (jcp.exec_type == exec_trans && (last_mb != n || last_g != g)) \
                std::memset(inp_buffer_mask, false, jcp.inp_buffer_mask_size); \
            last_mb = n; \
            last_g = g; \
            auto od_begin = odb * jcp.od_blk_size; \
            auto od_end = nstl::min(OD, od_begin + jcp.od_blk_size); \
            auto oh_begin = ohb * jcp.oh_blk_size; \
            auto oh_end = nstl::min(OH, oh_begin + jcp.oh_blk_size); \
            for_(int od = od_begin; od < od_end; od++) \
            for (int oh = oh_begin; oh < oh_end; oh++) { \
                for (int icc = 0; icc < ic_chunks; icc++) { \
                    if (jcp.exec_type == exec_base) { \
                        ker_base(ctx, ithr, brg_batch, c_buffer, g, n, ocb, \
                                od, oh, owb, icc); \
                    } else if (jcp.exec_type == exec_trans) { \
                        maybe_conv_inp(ithr, src, inp_buffer, inp_buffer_mask, \
                                g, n, icc, odb, ohb, owb); \
                        ker_trans(ctx, ithr, brg_batch, c_buffer, inp_buffer, \
                                g, n, ocb, od, oh, owb, icc); \
                    } else if (jcp.exec_type == exec_vpad) { \
                        ker_vpad(ctx, ithr, brg_batch, c_buffer, g, n, ocb, \
                                od, oh, owb, icc); \
                    } else \
                        assert(0); \
                } \
            } \
            nd_iterator_step(__VA_ARGS__); \
        } \
    });

    if (jcp.loop_order == loop_ndhwgc)
        BRGC_WO(n, jcp.mb, odb, jcp.nb_od, ohb, jcp.nb_oh, owb, jcp.nb_ow, g,
                jcp.ngroups, ocb, jcp.nb_oc)
    else if (jcp.loop_order == loop_ngcdhw)
        BRGC_WO(n, jcp.mb, g, jcp.ngroups, ocb, jcp.nb_oc, odb, jcp.nb_od, ohb,
                jcp.nb_oh, owb, jcp.nb_ow)
    else
        assert(!"Unknown loop order");

#undef BRGC_WO

    if (pd()->wants_zero_pad_dst()) ctx.memory(DNNL_ARG_DST)->zero_pad(ctx);

    return status::success;
}

template <cpu_isa_t isa, data_type_t src_type, data_type_t wei_type,
        data_type_t dst_type>
void brgemm_convolution_fwd_t<isa, src_type, wei_type,
        dst_type>::perform_outwork(dst_data_t *dst_base, char *c_buffer,
        const char *bias_w, int od, int oh, int ow, int g_oc, bool is_oc_tail,
        int ker_ow_s, int ker_ow_f, int kd_l, int kh_l, bool do_init,
        bool do_postwork) const {
    if (!do_init && !do_postwork) return;

    const auto &jcp = pd()->jcp_;

    const bool is_ow_tail = (OW - ow < jcp.ow_block);

    const auto M = is_ow_tail ? jcp.M_tail : jcp.M;
    const auto kdh_l = kd_l * kh_l;
    const auto ow_s = (kdh_l <= 0) ? ow : ker_ow_s;
    const auto ow_f = (kdh_l <= 0) ? ow : ker_ow_f;
    assert(ow <= ow_s && ow_s <= ow_f && ow_f <= ow + M);

    brgemm_kernel_post_ops_t p;
    if (do_postwork) {
        p.ptr_bias = (void *)bias_w;
        p.ptr_scales = (void *)&oscales[jcp.is_oc_scale * g_oc];
    }

    auto call_outwork_ker = [&](bool is_postwork, int ow_pw_s, int ow_pw_l) {
        const auto outwork_ker = kernels_po_[get_ker_po_idx(ow_pw_l - 1,
                                                     is_postwork, is_oc_tail)]
                                         .get();
        assert(ow_pw_l == outwork_ker->brg.bcast_dim);
        if (is_postwork) {
            p.ptr_out = (void *)(dst_base + od * dst_h_sz + oh * dst_w_sz
                    + ow_pw_s * jcp.oc_without_padding);
            p.ptr_in = (void *)(jcp.use_buffer
                            ? (c_buffer + acc_dsz * (ow_pw_s - ow) * jcp.LDC)
                            : p.ptr_out);
        } else {
            char *const ptr_Cz = jcp.use_buffer
                    ? (c_buffer + acc_dsz * (ow_pw_s - ow) * jcp.LDC)
                    : (char *)(dst_base + od * dst_h_sz + oh * dst_w_sz
                            + ow_pw_s * jcp.oc_without_padding);
            p.ptr_out = (void *)ptr_Cz;
        }
        (*outwork_ker)(&p);
    };

    if (ow < ow_s) {
        // left side
        const auto ow_pw_l = ow_s - ow;
        if (do_init) call_outwork_ker(false, ow, ow_pw_l);
        if (do_postwork) call_outwork_ker(true, ow, ow_pw_l);
    }
    if (ow_f < ow + M) {
        // right side
        const auto ow_pw_l = ow + M - ow_f;
        if (do_init) call_outwork_ker(false, ow_f, ow_pw_l);
        if (do_postwork) call_outwork_ker(true, ow_f, ow_pw_l);
    }
}

template <cpu_isa_t isa, data_type_t src_type, data_type_t wei_type,
        data_type_t dst_type>
void brgemm_convolution_fwd_t<isa, src_type, wei_type,
        dst_type>::call_brgemm_kernel(brgemm_kernel_t *brg_ker, int batch_size,
        brgemm_batch_element_t *const __restrict brg_batch, char *ptr_C,
        dst_data_t *ptr_D, const char *bias_w, int g_oc,
        bool do_postops) const {
    const auto &jcp = pd()->jcp_;
    if (do_postops)
        brgemm_kernel_execute_postops(brg_ker, batch_size, brg_batch, ptr_C,
                ptr_D, (void *)bias_w, &oscales[jcp.is_oc_scale * g_oc]);
    else
        brgemm_kernel_execute(brg_ker, batch_size, brg_batch, ptr_C);
}

template <cpu_isa_t isa, data_type_t src_type, data_type_t wei_type,
        data_type_t dst_type>
void brgemm_convolution_fwd_t<isa, src_type, wei_type,
        dst_type>::maybe_conv_inp(int ithr, const src_data_t *__restrict src,
        src_data_t *__restrict inp_buffer, uint8_t *__restrict inp_buffer_mask,
        int g, int n, int icc, int odb, int ohb, int owb) const {

    const auto &jcp = pd()->jcp_;
    const auto icb = icc * jcp.nb_ic_blocking;

#define bmask(icb, odb, ohb, owb) \
    inp_buffer_mask[(((icb)*jcp.nb_od + (odb)) * jcp.nb_oh + (ohb)) \
                    * jcp.nb_ow \
            + (owb)]

    if (bmask(icb, odb, ohb, owb)) return;
    auto p = jit_conv_call_s();

    const auto prev_odb
            = (odb == 0 || bmask(icb, odb - 1, ohb, owb) == 0) ? false : true;

    const auto prev_ohb
            = (ohb == 0 || bmask(icb, odb, ohb - 1, owb) == 0) ? false : true;

    const auto prev_odb_ohb
            = (odb > 0 && ohb > 0 && bmask(icb, odb - 1, ohb - 1, owb) == 0)
            ? false
            : true;

    const auto ic = icb * jcp.ic_block;
    const auto g_ic = g * jcp.ic + ic;
    const auto ow = owb * jcp.ow_block;
    const auto iw = nstl::max(0, ow * SW - LP);

    auto get_start_end = [](int &start, int &end, int b, int bs, int i, int o,
                                 int s, int p, int k, bool prev) {
        const auto o_b = b * bs;
        const auto o_e = nstl::min(o, (b + 1) * bs);
        const auto prev_o_e = (b > 0) ? nstl::min(o, b * bs) : 0;
        const auto prev_end
                = prev ? utils::saturate(0, i, prev_o_e * s - p + k) : 0;
        const auto cur_start = utils::saturate(0, i, o_b * s - p);
        start = nstl::max(prev_end, cur_start);
        end = utils::saturate(0, i, o_e * s - p + k);
    };

    int id_start {0}, id_end {0}, ih_start {0}, ih_end {0};

    get_start_end(id_start, id_end, odb, jcp.od_blk_size, ID, OD, SD, FP, KD,
            prev_odb && prev_odb_ohb);
    get_start_end(ih_start, ih_end, ohb, jcp.oh_blk_size, IH, OH, SH, TP, KH,
            prev_ohb && prev_odb_ohb);

    const auto ih_zero_top = 0;
    const auto ih_zero_bottom = 0;

    // how many real data rows to copy (including padding)
    const auto rows_to_copy = ih_end - ih_start;
    p.kh_padding = max(0, rows_to_copy);
    p.t_overflow = ih_zero_top;
    p.b_overflow = ih_zero_bottom;
    p.owb = owb;
    const auto ih_buf = ih_start + TP;
    const auto iw_buf = ow * SW;

    const auto inp_offset_start = (dim_t)n * src_d_sz + ih_start * src_w_sz
            + iw * jcp.ic_without_padding + g_ic;
    // inp_buffer has physical padding
    const auto out_offset_start = (dim_t)icb * pbuf_d_sz + ih_buf * pbuf_w_sz
            + iw_buf * jcp.ic_block;

    for (int id = id_start; id < id_end; id++) {
        const auto inp_offset = inp_offset_start + id * src_h_sz;
        const auto id_buf = id + FP;
        const auto out_offset = out_offset_start + id_buf * pbuf_h_sz;
        p.src = src + inp_offset;
        p.dst = inp_buffer + out_offset;
        (*copy_to_pbuffer_)(&p);
    }

    bmask(icb, odb, ohb, owb) = 1;

#undef bmask
}

#define BRGEMM_CONV_KER_HEADER \
    const src_data_t *const __restrict src \
            = CTX_IN_MEM(const src_data_t *, DNNL_ARG_SRC); \
    const wei_data_t *const __restrict weights \
            = CTX_IN_MEM(const wei_data_t *, DNNL_ARG_WEIGHTS); \
    const char *const __restrict bias \
            = CTX_IN_MEM(const char *, DNNL_ARG_BIAS); \
    dst_data_t *const __restrict dst \
            = CTX_OUT_MEM(dst_data_t *, DNNL_ARG_DST); \
    const int oc = ocb * jcp.oc_block; \
    const int g_oc = g * jcp.oc + oc; \
    const int icb = icc * jcp.nb_ic_blocking; \
    const int ic = icb * jcp.ic_block; \
    const int g_ic = g * jcp.ic + ic; \
    const int ow = owb * jcp.ow_block; \
    const int iid = ndims_pick(od * SD - FP, 0, 0); \
    const int kd_s = ndims_pick(div_up(max(0, -iid), DD), 0, 0); \
    const int kd_f = ndims_pick( \
            KD - div_up(max(0, iid - ID + (KD - 1) * DD + 1), DD), 1, 1); \
    const auto kd_l = kd_f - kd_s; \
    const auto iih = ndims_pick(oh * SH - TP, oh * SH - TP, 0); \
    const auto kh_s_ = div_up(max(0, -iih), DH); \
    const auto kh_s = ndims_pick(kh_s_, kh_s_, 0); \
    const auto kh_f_ = KH - div_up(max(0, iih - IH + (KH - 1) * DH + 1), DH); \
    const auto kh_f = ndims_pick(kh_f_, kh_f_, 1); \
    const auto kh_l = kh_f - kh_s; \
    const bool is_oc_tail = (jcp.oc - oc < jcp.oc_block); \
    const bool is_ic_tail \
            = (icc == ic_chunks - 1 && ((jcp.ic - ic) % jcp.ic_block != 0)); \
    const bool is_ow_tail = (OW - ow < jcp.ow_block); \
    const char *const __restrict bias_w \
            = bias ? bias + (bias_d.blk_off(g_oc) * bia_dsz) : nullptr; \
    const auto nb_ic_b = nstl::min(jcp.nb_ic_blocking, jcp.nb_ic - icb) \
            - (is_ic_tail ? 1 : 0); \
    dst_data_t *const __restrict dst_base = dst + n * dst_d_sz + g_oc; \
    char *ptr_C; \
    dst_data_t *ptr_D; \
    int kd_b(0), kd_e(0), kh_b(0), kh_e(0), k_l(0), iiw_b(0);

template <cpu_isa_t isa, data_type_t src_type, data_type_t wei_type,
        data_type_t dst_type>
void brgemm_convolution_fwd_t<isa, src_type, wei_type, dst_type>::ker_base(
        const exec_ctx_t &ctx, int ithr,
        brgemm_batch_element_t *const __restrict brg_batch,
        char *const c_buffer, int g, int n, int ocb, int od, int oh, int owb,
        int icc) const {

    const auto &jcp = pd()->jcp_;
    auto ndims = pd()->ndims();

    BRGEMM_CONV_KER_HEADER;
    MAYBE_UNUSED(is_ow_tail);
    int kw_s {0}, kw_full_s {0}, kw_f {0}, kw_full_f {0}, kw_b(0), kw_e(0);

    get_kw_range(ow, kw_s, kw_full_s, kw_full_f, kw_f);

    const auto src_base = src + n * src_d_sz + g_ic;
    const auto wei_base = weights + g * wei_ocb_sz + ocb * wei_kd_sz;

    const auto call_brgemm = [&](brgemm_kernel_t *brg_ker, int ic_block_s,
                                     int n_ic_blocks, bool do_postops) {
        if (k_l <= 0) return;

        for (int i_icb = 0; i_icb < n_ic_blocks; i_icb++) {
            const auto ic_off = (ic_block_s + i_icb) * jcp.ic_block;
            const auto src_ic = ic_off;
            const auto wei_ic = ic + ic_off;
            const auto n_icb_off = i_icb * k_l;
            const auto src_base_ic = src_base + src_ic;
            const auto wei_base_ic = wei_base + wei_ic * jcp.oc_block;

            auto k = 0;
            for (int kd = kd_b; kd < kd_e; kd++) {
                const auto id = iid + kd * DD;
                const auto src_base_kd = src_base_ic + id * src_h_sz;
                const auto wei_base_kd = wei_base_ic + kd * wei_kh_sz;
                for (int kh = kh_b; kh < kh_e; kh++) {
                    const auto ih = iih + kh * DH;
                    const auto src_base_kh = src_base_kd + ih * src_w_sz;
                    const auto wei_base_kh = wei_base_kd + kh * wei_kw_sz;
                    for (int kw = kw_b; kw < kw_e; kw++) {
                        const auto iw = iiw_b + kw * DW;
                        brg_batch[n_icb_off + k].ptr.A
                                = src_base_kh + iw * jcp.ic_without_padding;
                        brg_batch[n_icb_off + k].vvpad.top = 0;
                        brg_batch[n_icb_off + k].vvpad.bottom = 0;
                        // general wei layout is gOdhwI<block_o><block_i>
                        brg_batch[n_icb_off + k].ptr.B
                                = wei_base_kh + kw * wei_ic_sz;
                        k++;
                    }
                }
            }
        }

        call_brgemm_kernel(brg_ker, k_l * n_ic_blocks, brg_batch, ptr_C, ptr_D,
                bias_w, g_oc, do_postops);
    };

    const auto kdhw_loop = [&]() {
        if (kw_e - kw_b <= 0) return;
        int ow_b {0}, ow_e {0};
        get_ow_range(ow, kw_b, ow_b, ow_e);

        const auto do_init
                = icc == 0 && kd_b == kd_s && kh_b == kh_s && kw_b == kw_s;
        const auto do_postwork = need_postwork && icc == (ic_chunks - 1)
                && kd_e == kd_f && kh_e == kh_f && kw_e == kw_f;
        if (ow_e - ow_b <= 0 && !do_init && !do_postwork) return;

        k_l = (kd_e - kd_b) * (kh_e - kh_b) * (kw_e - kw_b);
        iiw_b = ow_b * SW - LP;
        ptr_D = dst_base + od * dst_h_sz + oh * dst_w_sz
                + ow_b * jcp.oc_without_padding;
        ptr_C = (jcp.use_buffer) ? c_buffer + acc_dsz * (ow_b - ow) * jcp.LDC
                                 : (char *)ptr_D;

        const auto ow_l = ow_e - ow_b;
        assert(0 <= ow_l && ow_l <= jcp.ow_block);

        const auto ker_i = ow_l - 1;
        brgemm_kernel_t *__restrict cur_kernels[2][2];
        cur_kernels[false][false]
                = brg_kernels_[get_brg_idx(ker_i, false, is_oc_tail, false)]
                          .get();
        cur_kernels[true][false]
                = brg_kernels_[get_brg_idx(ker_i, true, is_oc_tail, false)]
                          .get();
        cur_kernels[false][true]
                = brg_kernels_[get_brg_idx(ker_i, false, is_oc_tail, true)]
                          .get();
        cur_kernels[true][true]
                = brg_kernels_[get_brg_idx(ker_i, true, is_oc_tail, true)]
                          .get();

        if (ow_l > 0 && k_l > 0) {
            if (nb_ic_b > 0) {
                const auto brg_ker = cur_kernels[do_init][false];
                call_brgemm(brg_ker, 0, nb_ic_b, do_postwork && !is_ic_tail);
            }

            if (is_ic_tail) {
                const auto use_init_ker = (do_init && nb_ic_b == 0);
                const auto brg_ic_tail_ker = cur_kernels[use_init_ker][true];
                call_brgemm(brg_ic_tail_ker, nb_ic_b, 1, do_postwork);
            }
        }
        perform_outwork(dst_base, c_buffer, bias_w, od, oh, ow, g_oc,
                is_oc_tail, ow_b, ow_e, kd_l, kh_l, do_init, do_postwork);
    };

    if (kd_f > kd_s && kh_f > kh_s && kw_f > kw_s) {
        // kw values with left padding
        if (kw_s < kw_full_s) {
            for (kd_b = kd_s; kd_b < kd_f; kd_b += KD_BLOCK_PAD) {
                kd_e = nstl::min(kd_f, kd_b + KD_BLOCK_PAD);
                for (kh_b = kh_s; kh_b < kh_f; kh_b += KH_BLOCK_PAD) {
                    kh_e = nstl::min(kh_f, kh_b + KH_BLOCK_PAD);
                    for (auto kw = kw_s; kw < kw_full_s; kw++) {
                        kw_b = kw;
                        kw_e = kw + 1;
                        kdhw_loop();
                    }
                }
            }
        }

        // kw values covering full ow_block
        if (kw_full_s < kw_full_f) {
            for (kd_b = kd_s; kd_b < kd_f; kd_b += KD_BLOCK) {
                kd_e = nstl::min(kd_f, kd_b + KD_BLOCK);
                for (kh_b = kh_s; kh_b < kh_f; kh_b += KH_BLOCK) {
                    kh_e = nstl::min(kh_f, kh_b + KH_BLOCK);
                    for (kw_b = kw_full_s; kw_b < kw_full_f; kw_b += KW_BLOCK) {
                        kw_e = nstl::min(kw_full_f, kw_b + KW_BLOCK);
                        kdhw_loop();
                    }
                }
            }
        }

        // kw values with right padding
        if (kw_full_f < kw_f) {
            for (kd_b = kd_s; kd_b < kd_f; kd_b += KD_BLOCK_PAD) {
                kd_e = nstl::min(kd_f, kd_b + KD_BLOCK_PAD);
                for (kh_b = kh_s; kh_b < kh_f; kh_b += KH_BLOCK_PAD) {
                    kh_e = nstl::min(kh_f, kh_b + KH_BLOCK_PAD);
                    for (int kw = kw_full_f; kw < kw_f; kw++) {
                        kw_b = kw;
                        kw_e = kw + 1;
                        kdhw_loop();
                    }
                }
            }
        }
    } else {
        const auto do_init = icc == 0;
        const auto do_postwork = need_postwork && icc == (ic_chunks - 1);
        perform_outwork(dst_base, c_buffer, bias_w, od, oh, ow, g_oc,
                is_oc_tail, ow, ow, kd_l, kh_l, do_init, do_postwork);
    }
}

template <cpu_isa_t isa, data_type_t src_type, data_type_t wei_type,
        data_type_t dst_type>
void brgemm_convolution_fwd_t<isa, src_type, wei_type, dst_type>::ker_trans(
        const exec_ctx_t &ctx, int ithr,
        brgemm_batch_element_t *const __restrict brg_batch,
        char *const c_buffer, src_data_t *inp_buffer, int g, int n, int ocb,
        int od, int oh, int owb, int icc) const {

    const auto &jcp = pd()->jcp_;
    auto ndims = pd()->ndims();

    BRGEMM_CONV_KER_HEADER;
    MAYBE_UNUSED(g_ic);
    MAYBE_UNUSED(src);

    const auto wei_base = weights + g * wei_ocb_sz + ocb * wei_kd_sz;
    const int ow_b {ow}, ow_e {ow + (is_ow_tail ? jcp.M_tail : jcp.M)};
    iiw_b = ow_b * SW - LP;
    ptr_D = dst_base + od * dst_h_sz + oh * dst_w_sz
            + ow_b * jcp.oc_without_padding;
    ptr_C = (jcp.use_buffer) ? c_buffer + acc_dsz * (ow_b - ow) * jcp.LDC
                             : (char *)ptr_D;

    const auto ow_l = ow_e - ow_b;
    assert(0 <= ow_l && ow_l <= jcp.ow_block);
    const auto ker_i = ow_l - 1;
    brgemm_kernel_t *__restrict cur_kernels[2][2];
    cur_kernels[false][false]
            = brg_kernels_[get_brg_idx(ker_i, false, is_oc_tail, false)].get();
    cur_kernels[true][false]
            = brg_kernels_[get_brg_idx(ker_i, true, is_oc_tail, false)].get();
    cur_kernels[false][true]
            = brg_kernels_[get_brg_idx(ker_i, false, is_oc_tail, true)].get();
    cur_kernels[true][true]
            = brg_kernels_[get_brg_idx(ker_i, true, is_oc_tail, true)].get();

    const auto call_brgemm = [&](brgemm_kernel_t *brg_ker, int ic_block_s,
                                     int n_ic_blocks, bool do_postops) {
        if (k_l <= 0) return;

        const auto pbuf_base = inp_buffer + (icb + ic_block_s) * pbuf_d_sz;
        for (int i_icb = 0; i_icb < n_ic_blocks; i_icb++) {
            const auto ic_off = (ic_block_s + i_icb) * jcp.ic_block;
            const auto wei_ic = ic + ic_off;
            const auto n_icb_off = i_icb * k_l;
            const auto pbuf_base_ic = pbuf_base + i_icb * pbuf_d_sz;
            const auto wei_base_ic = wei_base + wei_ic * jcp.oc_block;

            auto k = 0;
            for (int kd = kd_b; kd < kd_e; kd++) {
                const auto id = iid + kd * DD + FP;
                const auto pbuf_base_kd = pbuf_base_ic + id * pbuf_h_sz;
                const auto wei_base_kd = wei_base_ic + kd * wei_kh_sz;
                for (int kh = kh_b; kh < kh_e; kh++) {
                    const auto ih = iih + kh * DH + TP;
                    const auto pbuf_base_kh = pbuf_base_kd + ih * pbuf_w_sz;
                    const auto wei_base_kh = wei_base_kd + kh * wei_kw_sz;
                    for (int kw = 0; kw < KW; kw++) {
                        const auto iw = iiw_b + kw * DW + LP;
                        // inp_buffer layout is Cdhw<ic_block>c
                        brg_batch[n_icb_off + k].ptr.A
                                = pbuf_base_kh + iw * jcp.ic_block;
                        brg_batch[n_icb_off + k].vvpad.top = 0;
                        brg_batch[n_icb_off + k].vvpad.bottom = 0;
                        // general wei layout is gOdhwI<block_o><block_i>
                        brg_batch[n_icb_off + k].ptr.B
                                = wei_base_kh + kw * wei_ic_sz;
                        k++;
                    }
                }
            }
        }

        call_brgemm_kernel(brg_ker, k_l * n_ic_blocks, brg_batch, ptr_C, ptr_D,
                bias_w, g_oc, do_postops);
    };

    const auto kdhw_loop = [&]() {
        const auto do_init = icc == 0 && kd_b == kd_s && kh_b == kh_s;
        const auto do_postwork = need_postwork && icc == (ic_chunks - 1)
                && kd_e == kd_f && kh_e == kh_f;
        if (ow_e - ow_b <= 0 && !do_init && !do_postwork) return;

        k_l = (kd_e - kd_b) * (kh_e - kh_b) * KW;

        if (nb_ic_b > 0) {
            const auto brg_ker = cur_kernels[do_init][false];
            call_brgemm(brg_ker, 0, nb_ic_b, do_postwork && !is_ic_tail);
        }

        if (is_ic_tail) {
            const auto use_init_ker = (do_init && nb_ic_b == 0);
            const auto brg_ic_tail_ker = cur_kernels[use_init_ker][true];
            call_brgemm(brg_ic_tail_ker, nb_ic_b, 1, do_postwork);
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
        const auto do_init = icc == 0;
        const auto do_postwork = need_postwork && icc == (ic_chunks - 1);
        perform_outwork(dst_base, c_buffer, bias_w, od, oh, ow, g_oc,
                is_oc_tail, ow, ow, kd_l, kh_l, do_init, do_postwork);
    }
}

template <cpu_isa_t isa, data_type_t src_type, data_type_t wei_type,
        data_type_t dst_type>
void brgemm_convolution_fwd_t<isa, src_type, wei_type, dst_type>::ker_vpad(
        const exec_ctx_t &ctx, int ithr,
        brgemm_batch_element_t *const __restrict brg_batch,
        char *const c_buffer, int g, int n, int ocb, int od, int oh, int owb,
        int icc) const {

    const auto &jcp = pd()->jcp_;
    auto ndims = pd()->ndims();

    BRGEMM_CONV_KER_HEADER;

    const src_data_t *const __restrict src_base = src + n * src_d_sz + g_ic;

    const wei_data_t *const __restrict wei_base
            = weights + g * wei_ocb_sz + ocb * wei_kd_sz;

    const int ow_b {ow}, ow_e {ow + (is_ow_tail ? jcp.M_tail : jcp.M)};
    iiw_b = ow_b * SW - LP;
    ptr_D = dst_base + od * dst_h_sz + oh * dst_w_sz
            + ow_b * jcp.oc_without_padding;
    ptr_C = (jcp.use_buffer) ? c_buffer + acc_dsz * (ow_b - ow) * jcp.LDC
                             : (char *)ptr_D;

    const auto ow_l = ow_e - ow_b;
    assert(0 <= ow_l && ow_l <= jcp.ow_block);
    const auto ker_i = ow_l - 1;
    const dim_t *const __restrict kw_top_vpads
            = owb_kw_top_vpads.data() + owb * KW;
    const dim_t *const __restrict kw_bottom_vpads
            = owb_kw_bottom_vpads.data() + owb * KW;

    brgemm_kernel_t *__restrict cur_kernels[2][2];
    cur_kernels[false][false]
            = brg_kernels_[get_brg_idx(ker_i, false, is_oc_tail, false)].get();
    cur_kernels[true][false]
            = brg_kernels_[get_brg_idx(ker_i, true, is_oc_tail, false)].get();
    cur_kernels[false][true]
            = brg_kernels_[get_brg_idx(ker_i, false, is_oc_tail, true)].get();
    cur_kernels[true][true]
            = brg_kernels_[get_brg_idx(ker_i, true, is_oc_tail, true)].get();

    const auto call_brgemm = [&](brgemm_kernel_t *brg_ker, int ic_block_s,
                                     int n_ic_blocks, bool do_postops) {
        for (int i_icb = 0; i_icb < n_ic_blocks; i_icb++) {
            const auto ic_off = (ic_block_s + i_icb) * jcp.ic_block;
            const auto src_ic = ic_off;
            const auto wei_ic = ic + ic_off;
            const auto n_icb_off = i_icb * k_l;
            const src_data_t *const __restrict src_base_ic = src_base + src_ic;
            const wei_data_t *const __restrict wei_base_ic
                    = wei_base + wei_ic * jcp.oc_block;
            brgemm_batch_element_t *const __restrict icb_batch
                    = brg_batch + n_icb_off;

            auto k = 0;
            for (int kd = kd_b; kd < kd_e; kd++) {
                const auto id = iid + kd * DD;
                const src_data_t *const __restrict src_base_kd
                        = src_base_ic + id * src_h_sz;
                const wei_data_t *const __restrict wei_base_kd
                        = wei_base_ic + kd * wei_kh_sz;
                for (int kh = kh_b; kh < kh_e; kh++) {
                    const auto ih = iih + kh * DH;
                    const src_data_t *const __restrict src_base_kh
                            = src_base_kd + ih * src_w_sz;
                    const wei_data_t *const __restrict wei_base_kh
                            = wei_base_kd + kh * wei_kw_sz;
                    for (int kw = 0; kw < KW; kw++) {
                        const auto iw = iiw_b + kw * DW;
                        const auto ptr_A
                                = src_base_kh + iw * jcp.ic_without_padding;
                        if (jcp.max_vpad) {
                            icb_batch[k].vvpad.top = kw_top_vpads[kw];
                            icb_batch[k].vvpad.bottom = kw_bottom_vpads[kw];
                        }
                        // general wei layout is gOdhwI<block_o><block_i>
                        const auto ptr_B = wei_base_kh + kw * wei_ic_sz;

                        icb_batch[k].ptr.A = ptr_A;
                        icb_batch[k].ptr.B = ptr_B;

                        k++;
                    }
                }
            }
        }

        call_brgemm_kernel(brg_ker, k_l * n_ic_blocks, brg_batch, ptr_C, ptr_D,
                bias_w, g_oc, do_postops);
    };

    const auto kdhw_loop = [&]() {
        const auto do_init = icc == 0 && kd_b == kd_s && kh_b == kh_s;
        const auto do_postwork = need_postwork && icc == (ic_chunks - 1)
                && kd_e == kd_f && kh_e == kh_f;
        if (ow_e - ow_b <= 0 && !do_init && !do_postwork) return;

        k_l = (kd_e - kd_b) * (kh_e - kh_b) * KW;

        if (nb_ic_b > 0) {
            const auto brg_ker = cur_kernels[do_init][false];
            call_brgemm(brg_ker, 0, nb_ic_b, do_postwork && !is_ic_tail);
        }

        if (is_ic_tail) {
            const auto use_init_ker = (do_init && nb_ic_b == 0);
            const auto brg_ic_tail_ker = cur_kernels[use_init_ker][true];
            call_brgemm(brg_ic_tail_ker, nb_ic_b, 1, do_postwork);
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
        const auto do_init = icc == 0;
        const auto do_postwork = need_postwork && icc == (ic_chunks - 1);
        perform_outwork(dst_base, c_buffer, bias_w, od, oh, ow, g_oc,
                is_oc_tail, ow, ow, kd_l, kh_l, do_init, do_postwork);
    }
}

#undef BRGEMM_CONV_KER_HEADER

template struct brgemm_convolution_fwd_t<avx512_core, f32>;

template struct brgemm_convolution_fwd_t<avx512_core_vnni, u8, s8, f32>;
template struct brgemm_convolution_fwd_t<avx512_core_vnni, u8, s8, s32>;
template struct brgemm_convolution_fwd_t<avx512_core_vnni, u8, s8, u8>;
template struct brgemm_convolution_fwd_t<avx512_core_vnni, u8, s8, s8>;

template struct brgemm_convolution_fwd_t<avx512_core_vnni, s8, s8, f32>;
template struct brgemm_convolution_fwd_t<avx512_core_vnni, s8, s8, s32>;
template struct brgemm_convolution_fwd_t<avx512_core_vnni, s8, s8, u8>;
template struct brgemm_convolution_fwd_t<avx512_core_vnni, s8, s8, s8>;

template struct brgemm_convolution_fwd_t<avx512_core_bf16, bf16, bf16, bf16>;
template struct brgemm_convolution_fwd_t<avx512_core_bf16, bf16, bf16, f32>;

} // namespace x64

} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
