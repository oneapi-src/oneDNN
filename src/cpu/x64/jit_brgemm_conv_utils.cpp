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

status_t init_jcp(jit_brgemm_conv_conf_t &jcp, const convolution_desc_t &cd,
        memory_desc_t &src_md, memory_desc_t &weights_md, memory_desc_t &dst_md,
        memory_desc_t &bias_md, const primitive_attr_t &attr, int nthreads) {
    using namespace prop_kind;

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

    return status::success;
}

int select_ic_block(const jit_brgemm_conv_conf_t &jcp, int max_ic_blocks) {
    auto nb_simd = utils::div_up(jcp.ic, jcp.simd_w);
    const auto nb_icb_disb_threshold = 0.8f;
    auto ic_blocks = 1;
    for (int nb_icb = nstl::min(max_ic_blocks, nb_simd); nb_icb >= 1;
            nb_icb--) {
        auto nb_icb_disb = (float)nb_simd / rnd_up(nb_simd, nb_icb);
        if (nb_icb_disb >= nb_icb_disb_threshold) {
            ic_blocks = nb_icb;
            break;
        }
    }
    return ic_blocks * jcp.simd_w;
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

namespace {

struct brg_blocking_t {
    brg_blocking_t() {
        oc_block = 0;
        ow_block = 0;
        os_block = 0;
        ic_block = 0;
        nb_ic_blocking = 0;
        kd_block = 0;
        kh_block = 0;
        kw_block = 0;
        kd_block_pad = 0;
        kh_block_pad = 0;
        kw_block_pad = 0;
        od_blk_size = 0;
        oh_blk_size = 0;
        use_buffer = false;
        nb_oc = 0;
        nb_ic = 0;
        nb_ow = 0;
        nb_os = 0;
        nb_os_blocking = 0;
        is_os_block = false;
    }

    int ow_block, os_block, oc_block, ic_block;
    int nb_ic_blocking;
    int kd_block, kh_block, kw_block, kd_block_pad, kh_block_pad, kw_block_pad;
    int od_blk_size, oh_blk_size;
    bool use_buffer;
    int nb_oc, nb_ow, nb_os, nb_os_blocking, nb_ic;

    bool is_os_block;

    void get_from_jcp(const jit_brgemm_conv_conf_t &jcp) {
        oc_block = jcp.oc_block;
        ow_block = jcp.ow_block;
        os_block = jcp.os_block;
        ic_block = jcp.ic_block;
        nb_ic_blocking = jcp.nb_ic_blocking;
        kd_block = jcp.kd_block;
        kh_block = jcp.kh_block;
        kw_block = jcp.kw_block;
        kd_block_pad = jcp.kd_block_pad;
        kh_block_pad = jcp.kh_block_pad;
        kw_block_pad = jcp.kw_block_pad;
        od_blk_size = jcp.od_blk_size;
        oh_blk_size = jcp.oh_blk_size;
        use_buffer = jcp.use_buffer;
        nb_oc = jcp.nb_oc;
        nb_ic = jcp.nb_ic;
        nb_ow = jcp.nb_ow;
        nb_os = jcp.nb_os;
        nb_os_blocking = jcp.nb_os_blocking;

        is_os_block = false;
    }
    void save_to_jcp(jit_brgemm_conv_conf_t &jcp) {
        jcp.oc_block = oc_block;
        jcp.ow_block = ow_block;
        jcp.os_block = os_block;
        jcp.ic_block = ic_block;
        jcp.nb_ic_blocking = nb_ic_blocking;
        jcp.kd_block = kd_block;
        jcp.kh_block = kh_block;
        jcp.kw_block = kw_block;
        jcp.kd_block_pad = kd_block_pad;
        jcp.kh_block_pad = kh_block_pad;
        jcp.kw_block_pad = kw_block_pad;
        jcp.od_blk_size = od_blk_size;
        jcp.oh_blk_size = oh_blk_size;
        jcp.use_buffer = use_buffer;
        jcp.nb_oc = nb_oc;
        jcp.nb_ic = nb_ic;
        jcp.nb_ow = nb_ow;
        jcp.nb_os = nb_os;
        jcp.nb_os_blocking = nb_os_blocking;
    }
};

int get_brgemm_ur(const jit_brgemm_conv_conf_t &jcp, cpu_isa_t isa,
        const brg_blocking_t &brgb, const memory_desc_t &dst_md,
        const primitive_attr_t *attr, bool is_1x1) {
    // Simulation of brgemm_desc init
    brgemm_t brg;
    const auto ic_block = brgb.ic_block;
    const auto oc_block = brgb.oc_block;
    const auto sp_block = brgb.is_os_block ? brgb.os_block : brgb.ow_block;

    const auto LDA = (jcp.exec_type == exec_trans)
            ? jcp.stride_w * ic_block
            : jcp.stride_w * jcp.ic_without_padding;
    const auto LDB = oc_block;
    const auto LDC = (brgb.use_buffer) ? oc_block : jcp.oc_without_padding;
    const auto LDD = jcp.oc_without_padding;

    const float alpha = 1.0;
    const float beta = 1.0;
    const float beta_init = 0.0;

    const auto M = sp_block;
    const auto M_tail
            = brgb.is_os_block ? jcp.os % sp_block : jcp.ow % sp_block;
    const auto K = jcp.ic >= ic_block ? ic_block : 0;
    const auto K_tail = jcp.ic % ic_block;
    const auto N = jcp.oc >= oc_block ? oc_block : 0;
    const auto N_tail = jcp.oc % oc_block;

    status_t status = success;
    int res_ur = 0;

    for (int i = 0; i < M; i++) {
        auto vM = i + 1;
        // init only needed brgemm descriptors
        if ((utils::one_of(jcp.exec_type, exec_trans, exec_vpad) || is_1x1)
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
                    brg_strides.stride_a = jcp.ic_without_padding
                            * (jcp.dilate_w + 1) * jcp.src_dsz;
                    //weights are padded by oc_block and last_ic_block
                    const auto last_ic_block = (jcp.wei_dt == f32)
                            ? 1
                            : ((jcp.wei_dt == bf16) ? 2 : 4);
                    brg_strides.stride_b = rnd_up(jcp.ic, last_ic_block)
                            * rnd_up(jcp.oc, oc_block) * jcp.wei_dsz;
                    const auto strides_ptr = (jcp.brg_type == brgemm_strd)
                            ? &brg_strides
                            : nullptr;
                    status = brgemm_desc_init(&brg, isa, jcp.brg_type,
                            jcp.src_dt, jcp.wei_dt, false, false,
                            brgemm_row_major, alpha, vbeta, LDA, LDB, LDC, vM,
                            vN, vK, strides_ptr);
                    if (status != success) break;
                    if (res_ur == 0) res_ur = brg.bd_block;

                    brgemm_attr_t brgattr;
                    brgattr.max_bs = jcp.max_batch;
                    const auto max_vpad = (jcp.exec_type == exec_vpad)
                            ? nstl::max(jcp.l_pad, jcp.r_pad)
                            : 0;
                    brgattr.max_top_vpad = max_vpad;
                    brgattr.max_bottom_vpad = max_vpad;
                    status = brgemm_desc_set_attr(&brg, brgattr);
                    if (status != success) break;

                    brg.with_sum = jcp.with_sum;
                    status = brgemm_desc_set_postops(
                            &brg, attr, &dst_md, LDD, jcp.bia_dt);
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
} // namespace

status_t init_conf(jit_brgemm_conv_conf_t &jcp, cpu_isa_t isa,
        const convolution_desc_t &cd, memory_desc_t &src_md,
        memory_desc_t &weights_md, memory_desc_t &dst_md,
        memory_desc_t &bias_md, const primitive_attr_t &attr, int nthreads) {

    using namespace prop_kind;

    CHECK(init_jcp(
            jcp, cd, src_md, weights_md, dst_md, bias_md, attr, nthreads));

    const memory_desc_wrapper src_d(&src_md);
    const memory_desc_wrapper weights_d(&weights_md);
    const memory_desc_wrapper dst_d(&dst_md);
    const memory_desc_wrapper bias_d(&bias_md);

    jcp.idp = jcp.id + jcp.f_pad + jcp.back_pad;
    jcp.ihp = jcp.ih + jcp.t_pad + jcp.b_pad;
    jcp.iwp = jcp.iw + jcp.l_pad + jcp.r_pad;

    using namespace data_type;
    // ======================= blocking =================================

    const auto L2 = 3 * platform::get_per_core_cache_size(2) / 4;

    auto src_amount = (size_t)jcp.id * jcp.ih * jcp.iw * jcp.ngroups * jcp.ic
            * jcp.src_dsz;
    auto wei_amount = (size_t)jcp.ngroups * jcp.ic * jcp.oc * jcp.kd * jcp.kh
            * jcp.kw * jcp.wei_dsz;

    jcp.loop_order = (src_amount < wei_amount) ? loop_ngcdhw : loop_ndhwgc;

    auto calc_blocks = [L2, nthreads](const jit_brgemm_conv_conf_t &jcp,
                               brg_blocking_t &brgb) {
        auto max_ic_blocks = 64;
        // TODO: improve below solution of aliasing issue
        if (jcp.kd > 1 || jcp.kh > 1)
            for (auto bb = 32; bb <= 256; bb *= 2)
                if (jcp.ow % bb == 0) max_ic_blocks /= 2;

        const auto ic_block = select_ic_block(jcp, max_ic_blocks);
        brgb.ic_block = nstl::min(ic_block, jcp.ic);
        brgb.nb_ic_blocking
                = div_up(nstl::min(ic_block, jcp.ic), brgb.ic_block);

        brgb.nb_ic = utils::div_up(jcp.ic, brgb.ic_block);

        const auto thr_eff_threshold = 0.9f;
        const auto ow_disb_threshold = 0.8f;

        const auto max_ow_block_thr
                = (int)div_up(jcp.mb * jcp.ngroups * brgb.nb_oc * jcp.os,
                        thr_eff_threshold * nthreads);
        auto brg_wei_amount = brgb.ic_block * brgb.nb_ic_blocking
                * brgb.oc_block * brgb.kd_block * brgb.kh_block * brgb.kw_block
                * jcp.wei_dsz;
        const auto free_L2 = (brg_wei_amount < L2) ? (L2 - brg_wei_amount) : 1;
        const auto inp_size_per_pixel = nstl::min(jcp.kd, jcp.stride_d)
                * nstl::min(jcp.kh, jcp.stride_h)
                * nstl::min(jcp.kw, jcp.stride_w) * brgb.ic_block
                * brgb.nb_ic_blocking * brgb.kd_block * brgb.kh_block
                * jcp.src_dsz;
        const auto out_size_per_pixel = brgb.oc_block * jcp.dst_dsz
                + (brgb.use_buffer ? brgb.oc_block : 0) * jcp.acc_dsz;
        const auto max_ow_block_L2
                = (int)div_up(free_L2, inp_size_per_pixel + out_size_per_pixel);

        auto start_ow_block = nstl::max(1,
                nstl::min(
                        jcp.ow, nstl::min(max_ow_block_thr, max_ow_block_L2)));

        auto ow_block = start_ow_block;
        for (; ow_block > 1; ow_block--) {
            auto nb_ow = div_up(jcp.ow, ow_block);

            auto work = jcp.mb * jcp.ngroups * brgb.nb_oc * jcp.od * jcp.oh
                    * nb_ow;
            auto thr_eff = (float)work / utils::rnd_up(work, nthreads);
            if (thr_eff < thr_eff_threshold) continue;

            auto ow_disb = (float)jcp.ow / rnd_up(jcp.ow, ow_block);
            if (ow_disb < ow_disb_threshold) continue;

            // TODO: rough estimation of input width block
            auto brg_src_amount = (ow_block + jcp.kw) * inp_size_per_pixel;
            auto brg_wei_amount = brgb.ic_block * brgb.nb_ic_blocking
                    * brgb.oc_block * brgb.kd_block * brgb.kh_block
                    * brgb.kw_block * jcp.wei_dsz;
            auto brg_dst_amount = ow_block * out_size_per_pixel;
            if (brg_src_amount + brg_wei_amount + brg_dst_amount < L2) break;
        }
        brgb.ow_block = ow_block;
        brgb.nb_ow = div_up(jcp.ow, brgb.ow_block);

        auto k_amount = brgb.ic_block * brgb.nb_ic_blocking * brgb.oc_block
                * jcp.wei_dsz;

        // if jcp.dst_dt != jcp.acc_dt and we need to store intermediate
        // results then we need the out buffer
        const auto maybe_use_buffer
                = (jcp.dst_dt != jcp.acc_dt || jcp.with_sum);

        brgb.kd_block
                = (k_amount * jcp.kd * jcp.kh * jcp.kw < L2 / 2) ? jcp.kd : 1;
        brgb.kh_block = (k_amount * jcp.kh * jcp.kw < L2 / 2) ? jcp.kh : 1;

        if (jcp.exec_type == exec_vpad) {
            brgb.kw_block = jcp.kw;
            brgb.kd_block_pad = brgb.kd_block;
            brgb.kh_block_pad = brgb.kh_block;
            brgb.kw_block_pad = brgb.kw_block;

            const auto use_buffer = maybe_use_buffer
                    && (brgb.ic_block * brgb.nb_ic_blocking < jcp.ic
                            || brgb.kd_block != jcp.kd
                            || brgb.kh_block != jcp.kh
                            || brgb.kw_block != jcp.kw
                            || brgb.kd_block_pad != jcp.kd
                            || brgb.kh_block_pad != jcp.kh
                            || brgb.kw_block_pad != jcp.kw);

            brgb.od_blk_size = 1;
            brgb.oh_blk_size = 1;

            brgb.use_buffer = use_buffer;

        } else if (jcp.exec_type == exec_trans) {
            brgb.kw_block = jcp.kw;
            brgb.kd_block_pad = brgb.kd_block;
            brgb.kh_block_pad = brgb.kh_block;
            brgb.kw_block_pad = brgb.kw_block;

            const auto use_buffer = maybe_use_buffer
                    && (brgb.ic_block * brgb.nb_ic_blocking < jcp.ic
                            || brgb.kd_block != jcp.kd
                            || brgb.kh_block != jcp.kh
                            || brgb.kw_block != jcp.kw
                            || brgb.kd_block_pad != jcp.kd
                            || brgb.kh_block_pad != jcp.kh
                            || brgb.kw_block_pad != jcp.kw);

            // TODO: select od/oh block size for best balancing
            // and performance
            const auto w_block_size = 2 * jcp.src_dsz * brgb.ic_block * jcp.iwp
                    + jcp.dst_dsz * brgb.oc_block * jcp.ow;
            const auto L2_available = L2
                    - jcp.wei_dsz * jcp.kd * jcp.kh * jcp.kw * brgb.ic_block
                            * brgb.oc_block;
            if (jcp.idp * jcp.ihp * w_block_size > L2_available) {
                brgb.od_blk_size = utils::saturate(
                        1, jcp.od, int(L2 / (jcp.ihp * w_block_size)));
                if (jcp.od_blk_size == 1)
                    brgb.oh_blk_size = utils::saturate(
                            1, jcp.oh, int(L2 / (w_block_size)));
                else
                    brgb.oh_blk_size = jcp.oh;
            } else {
                brgb.od_blk_size = 1;
                brgb.oh_blk_size = jcp.oh;
            }

            brgb.use_buffer = use_buffer;

        } else {
            brgb.kw_block = (k_amount * jcp.kw < L2) ? jcp.kw : 1;
            brgb.kd_block_pad = brgb.kh_block >= jcp.kd ? jcp.kd : 1;
            brgb.kh_block_pad = brgb.kw_block >= jcp.kh ? jcp.kh : 1;
            brgb.kw_block_pad = jcp.kw;

            const auto use_buffer = maybe_use_buffer
                    && (brgb.ic_block * brgb.nb_ic_blocking < jcp.ic
                            || brgb.kd_block != jcp.kd
                            || brgb.kh_block != jcp.kh
                            || brgb.kw_block != jcp.kw
                            || brgb.kd_block_pad != jcp.kd
                            || brgb.kh_block_pad != jcp.kh
                            || brgb.kw_block_pad != jcp.kw);

            brgb.od_blk_size = 1;
            brgb.oh_blk_size = 1;

            brgb.use_buffer
                    = use_buffer || (maybe_use_buffer && jcp.iwp != jcp.iw);
        }
    };

    auto try_exec_type = [&]() {
        constexpr auto max_regs = 32; //TODO: implement more general code
        auto best_ocb_eff = 0.f;
        brg_blocking_t best_brgb;
        best_brgb.oc_block = 16;
        brg_blocking_t cur_brgb;
        cur_brgb.get_from_jcp(jcp);
        const auto start_ocb = jcp.wei_plain
                ? nstl::min(jcp.ic > 128 ? (jcp.ic > 256 ? 8 : 16) : 32,
                        div_up(jcp.oc, 16))
                : 4;
        for (auto ocb = start_ocb; ocb >= 1; ocb--) {
            cur_brgb.oc_block = ocb * 16;
            cur_brgb.nb_oc = utils::div_up(jcp.oc, cur_brgb.oc_block);

            calc_blocks(jcp, cur_brgb);
            const bool is_1x1 = false;
            const auto ur
                    = get_brgemm_ur(jcp, isa, cur_brgb, dst_md, &attr, is_1x1);
            if (ur == 0) continue;

            const auto oc_block_disb
                    = (float)jcp.oc / rnd_up(jcp.oc, cur_brgb.oc_block);
            const auto ur_disb
                    = (float)cur_brgb.ow_block / rnd_up(cur_brgb.ow_block, ur);
            const auto brgemm_microkernel_eff
                    = ((float)ocb * ur) / ((ur + ocb) * max_regs);
            const auto ocb_eff
                    = oc_block_disb * ur_disb * brgemm_microkernel_eff;
            if (ocb_eff > best_ocb_eff) {
                best_ocb_eff = ocb_eff;
                best_brgb = cur_brgb;
            }
        }
        if (best_brgb.oc_block == 0 || best_brgb.ic_block == 0
                || best_brgb.ow_block == 0)
            return false;
        best_brgb.save_to_jcp(jcp);
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
            jcp, cd, src_md, weights_md, dst_md, bias_md, attr, nthreads));

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

    const auto L2 = 3 * platform::get_per_core_cache_size(2) / 4;

    auto src_amount = (size_t)jcp.id * jcp.ih * jcp.iw * jcp.ngroups * jcp.ic
            * jcp.src_dsz;
    auto wei_amount = (size_t)jcp.ngroups * jcp.ic * jcp.oc * jcp.wei_dsz;

    jcp.loop_order = (src_amount < wei_amount) ? loop_ngcdhw : loop_ndhwgc;

    auto calc_blocks = [L2, nthreads](const jit_brgemm_conv_conf_t &jcp,
                               brg_blocking_t &brgb) {
        // TODO: check aliasing
        auto max_ic_blocks = 64;
        const auto ic_block = select_ic_block(jcp, max_ic_blocks);
        brgb.ic_block = nstl::min(ic_block, jcp.ic);
        brgb.nb_ic_blocking = 1;
        brgb.nb_ic = utils::div_up(jcp.ic, brgb.ic_block);

        brgb.use_buffer = (jcp.dst_dt != jcp.acc_dt || jcp.with_sum)
                && (brgb.ic_block * brgb.nb_ic_blocking < jcp.ic);

        if (jcp.stride_d == 1 && jcp.stride_h == 1) {
            brgb.is_os_block = true;
            brgb.ow_block = 0;
            // TODO: os_blocking always is 1 for now. Update this code
            auto nb_os_blocking = 1;

            const auto thr_eff_threshold = 0.9f;
            const auto os_disb1_threshold = 0.8f;
            const auto os_disb2_threshold = 0.8f;

            const auto max_os_block_thr = nstl::max(div_up(2048, brgb.oc_block),
                    (int)div_up(jcp.mb * jcp.ngroups * jcp.os, nthreads));
            auto brg_wei_amount = brgb.ic_block * brgb.nb_ic_blocking
                    * brgb.oc_block * jcp.wei_dsz;
            const auto free_L2
                    = (brg_wei_amount < L2) ? (L2 - brg_wei_amount) : 1;
            const auto inp_size_per_pixel
                    = brgb.ic_block * brgb.nb_ic_blocking * jcp.src_dsz;
            const auto out_size_per_pixel = brgb.oc_block * jcp.dst_dsz
                    + ((brgb.use_buffer) ? brgb.oc_block : 0) * jcp.acc_dsz;
            const auto max_os_block_L2 = (int)div_up(
                    free_L2, inp_size_per_pixel + out_size_per_pixel);

            auto max_os_block_aliasing = 1000000 / nthreads;
            if ((jcp.oc_without_padding * jcp.os * jcp.dst_dsz) % 4096 == 0) {
                max_os_block_aliasing /= 1;
                for (auto cur_oc = jcp.oc_without_padding;
                        max_os_block_aliasing * jcp.dst_dsz > 400
                        && cur_oc % 2 == 0
                        && cur_oc * jcp.os * jcp.dst_dsz >= 4096;
                        cur_oc /= 2) {
                    max_os_block_aliasing /= 2;
                }
                max_os_block_aliasing += max_os_block_aliasing % 2 ? 0 : 1;
            }
            max_os_block_aliasing = nstl::min(
                    div_up(1001, jcp.dst_dsz), max_os_block_aliasing);

            auto start_os_block = nstl::max(1,
                    nstl::min(jcp.os,
                            nstl::min(nstl::min(max_os_block_thr,
                                              max_os_block_L2),
                                    max_os_block_aliasing)));

            auto os_block = start_os_block;
            for (; os_block > 1; os_block--) {
                auto nb_os = div_up(jcp.os, os_block);
                auto os_chunks = div_up(nb_os, nb_os_blocking);

                auto work = jcp.mb * jcp.ngroups * brgb.nb_oc * os_chunks;
                auto thr_eff = (float)work / utils::rnd_up(work, nthreads);
                if (thr_eff < thr_eff_threshold) continue;

                auto disb1 = (float)jcp.os / rnd_up(jcp.os, os_block);
                if (disb1 < os_disb1_threshold) continue;

                auto disb2 = (float)jcp.os
                        / rnd_up(jcp.os, nb_os_blocking * os_block);
                if (disb2 < os_disb2_threshold) continue;

                auto brg_src_amount = os_block * inp_size_per_pixel;
                auto brg_dst_amount = os_block * out_size_per_pixel;
                if (brg_src_amount + brg_wei_amount + brg_dst_amount < L2)
                    break;
            }
            brgb.os_block = os_block;
            brgb.nb_os_blocking = nb_os_blocking;

            brgb.nb_os = div_up(jcp.os, brgb.os_block);
        } else {
            brgb.is_os_block = false;
            brgb.os_block = 0;
            const auto thr_eff_threshold = 0.9f;
            const auto ow_disb_threshold = 0.8f;

            const auto max_ow_block_thr
                    = (int)div_up(jcp.mb * jcp.ngroups * brgb.nb_oc * jcp.os,
                            thr_eff_threshold * nthreads);
            const auto brg_wei_amount = brgb.ic_block * brgb.nb_ic_blocking
                    * brgb.oc_block * jcp.wei_dsz;
            const auto free_L2
                    = (brg_wei_amount < L2) ? (L2 - brg_wei_amount) : 1;
            const auto inp_pixel_size
                    = brgb.ic_block * brgb.nb_ic_blocking * jcp.src_dsz;
            const auto out_pixel_size = brgb.oc_block * jcp.dst_dsz
                    + ((brgb.use_buffer) ? brgb.oc_block : 0) * jcp.acc_dsz;
            const auto max_ow_block_L2
                    = (int)div_up(free_L2, inp_pixel_size + out_pixel_size);

            auto start_ow_block = utils::saturate(
                    1, jcp.ow, nstl::min(max_ow_block_thr, max_ow_block_L2));

            auto ow_block = start_ow_block;
            for (; ow_block > 1; ow_block--) {
                auto nb_ow = div_up(jcp.ow, ow_block);

                auto work = jcp.mb * jcp.ngroups * brgb.nb_oc * jcp.od * jcp.oh
                        * nb_ow;
                auto thr_eff = (float)work / utils::rnd_up(work, nthreads);
                if (thr_eff < thr_eff_threshold) continue;

                auto ow_disb = (float)jcp.ow / rnd_up(jcp.ow, ow_block);
                if (ow_disb < ow_disb_threshold) continue;

                auto brg_src_amount = ow_block * inp_pixel_size;
                auto brg_dst_amount = ow_block * out_pixel_size;
                if (brg_src_amount + brg_wei_amount + brg_dst_amount < L2)
                    break;
            }
            brgb.ow_block = ow_block;

            brgb.nb_ow = div_up(jcp.ow, brgb.ow_block);
        }
    };

    jcp.brg_type = brgemm_addr; // TODO: Choose right type of BRGEMM

    // max_batch is 1 and max_vpad is 0 for 1x1 convolutions
    jcp.max_batch = 1;
    jcp.max_vpad = 0;

    jcp.wei_plain = false;

    constexpr auto max_regs = 32; //TODO: implement more general approach
    auto best_ocb_eff = 0.f;
    brg_blocking_t best_brgb;
    best_brgb.oc_block = 16;
    brg_blocking_t cur_brgb;
    cur_brgb.get_from_jcp(jcp);
    const auto start_ocb = jcp.wei_plain
            ? nstl::min(jcp.ic > 128 ? (jcp.ic > 256 ? 8 : 16) : 32,
                    div_up(jcp.oc, 16))
            : 4;

    for (auto ocb = start_ocb; ocb >= 1; ocb--) {
        cur_brgb.oc_block = ocb * 16;
        cur_brgb.nb_oc = utils::div_up(jcp.oc, cur_brgb.oc_block);
        calc_blocks(jcp, cur_brgb);
        const bool is_1x1 = true;
        const auto ur
                = get_brgemm_ur(jcp, isa, cur_brgb, dst_md, &attr, is_1x1);
        if (ur == 0) continue;

        const auto oc_block_disb
                = (float)jcp.oc / rnd_up(jcp.oc, cur_brgb.oc_block);
        const auto ur_disb = cur_brgb.is_os_block
                ? (float)cur_brgb.os_block / rnd_up(cur_brgb.os_block, ur)
                : (float)cur_brgb.ow_block / rnd_up(cur_brgb.ow_block, ur);
        const auto brgemm_microkernel_eff
                = ((float)ocb * ur) / ((ur + ocb) * max_regs);
        const auto ocb_eff = oc_block_disb * ur_disb * brgemm_microkernel_eff;
        if (ocb_eff > best_ocb_eff) {
            best_ocb_eff = ocb_eff;
            best_brgb = cur_brgb;
        }
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
