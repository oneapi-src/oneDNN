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

#include "dnnl_types.h"

#include "common/bfloat16.hpp"
#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"
#include "cpu/x64/brgemm/brgemm_utils.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/x64/jit_brgemm_conv_bwd_utils.hpp"
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

namespace brgemm_convolution_bwd_utils {

inline status_t init_tag(format_tag_t &tag, memory_desc_t &md,
        const memory_desc_wrapper &mdw, const format_tag_t tag_value) {
    if (mdw.format_kind() == format_kind::any) {
        CHECK(memory_desc_init_by_tag(md, tag_value));
        tag = tag_value;
    } else {
        tag = mdw.matches_one_of_tag(tag_value);
    }

    if (tag != tag_value) return status::unimplemented;

    return status::success;
}

bool is_amx(cpu_isa_t isa) {
    return is_superset(isa, avx512_core_amx);
}

bool post_ops_ok(jit_brgemm_conv_conf_t &jcp, primitive_attr_t &attr,
        const memory_desc_wrapper &dst_d, bool is_deconv) {
    using namespace injector;

    const auto &post_ops = attr.post_ops_;

    if (post_ops.len() > 0 && !is_deconv) return false;

    return injector::post_ops_ok(post_ops_ok_args_t(jcp.isa,
            {sum, eltwise, binary}, post_ops, &dst_d,
            false /*sum_at_pos_0_only*/, false /*sum_requires_scale_one*/,
            false /*sum_requires_zp_zero*/, true /*sum_requires_same_params*/,
            {broadcasting_strategy_t::per_oc, broadcasting_strategy_t::scalar,
                    broadcasting_strategy_t::no_broadcast}));
}

bool is_groups_ok(jit_brgemm_conv_conf_t &jcp) {
    // Enable grouped convs for the shapes not supported in direct convs
    // direct approach only supports int8/bf16 grouped conv
    // when channels per groups is at least multiple of 4
    // and bf16 grouped conv with layout nxc on jit_bf16 impl
    // TODO: remove this condition after the restriction on small oc is removed
    return jcp.ngroups > 1
            && IMPLICATION(one_of(jcp.src_dt, u8, s8, bf16, f16),
                    jcp.oc % 4 == 0 && jcp.ic % 4 == 0);
}

status_t pick_tags(jit_brgemm_conv_conf_t &jcp, memory_desc_t &diff_dst_md,
        memory_desc_t &weights_md, memory_desc_t &diff_src_md,
        memory_desc_t &bias_md) {
    format_tag_t src_tag, dst_tag, wei_tag;
    dst_tag = pick(jcp.ndims - 3, nwc, nhwc, ndhwc);

    const memory_desc_wrapper diff_dst_d(&diff_dst_md);
    const memory_desc_wrapper weights_d(&weights_md);
    const memory_desc_wrapper diff_src_d(&diff_src_md);
    const memory_desc_wrapper bias_d(&bias_md);
    const bool with_groups = weights_d.ndims() == diff_src_d.ndims() + 1;

    const bool is_1d = jcp.ndims == 3;
    const bool is_2d = jcp.ndims == 4;
    const bool is_3d = jcp.ndims == 5;

    if (jcp.wei_plain) {
        return status::unimplemented;
    } else {
        jcp.LDB = jcp.ic_block;
        const bool no_vnni_format = jcp.wei_dt == f32
                || (jcp.wei_dt == f16 && jcp.isa == avx512_core_fp16);
        if (jcp.ic_block == 64) {
            if (is_3d) {
                if (no_vnni_format)
                    wei_tag = with_groups ? gIdhwo64i : Idhwo64i;
                else if (jcp.wei_dt == s8) {
                    if (jcp.is_oc_padded)
                        wei_tag = with_groups ? gIdhwO16o64i4o : IdhwO16o64i4o;
                    else
                        wei_tag = with_groups ? gIdhwO64i4o : IdhwO64i4o;
                } else if (one_of(jcp.wei_dt, bf16, f16)) {
                    if (jcp.is_oc_padded)
                        wei_tag = with_groups ? gIdhwO16o64i2o : IdhwO16o64i2o;
                    else
                        wei_tag = with_groups ? gIdhwO64i2o : IdhwO64i2o;
                } else
                    return status::unimplemented;
            } else if (is_1d) {
                if (no_vnni_format)
                    wei_tag = with_groups ? gIwo64i : Iwo64i;
                else if (jcp.wei_dt == s8) {
                    if (jcp.is_oc_padded)
                        wei_tag = with_groups ? gIwO16o64i4o : IwO16o64i4o;
                    else
                        wei_tag = with_groups ? gIwO64i4o : IwO64i4o;
                } else if (one_of(jcp.wei_dt, bf16, f16)) {
                    if (jcp.is_oc_padded)
                        wei_tag = with_groups ? gIwO16o64i2o : IwO16o64i2o;
                    else
                        wei_tag = with_groups ? gIwO64i2o : IwO64i2o;
                } else
                    return status::unimplemented;
            } else {
                assert(is_2d);
                UNUSED(is_2d);
                if (no_vnni_format)
                    wei_tag = with_groups ? gIhwo64i : Ihwo64i;
                else if (jcp.wei_dt == s8) {
                    if (jcp.is_oc_padded)
                        wei_tag = with_groups ? gIhwO16o64i4o : IhwO16o64i4o;
                    else
                        wei_tag = with_groups ? gIhwO64i4o : IhwO64i4o;
                } else if (one_of(jcp.wei_dt, bf16, f16)) {
                    if (jcp.is_oc_padded)
                        wei_tag = with_groups ? gIhwO16o64i2o : IhwO16o64i2o;
                    else
                        wei_tag = with_groups ? gIhwO64i2o : IhwO64i2o;
                } else
                    return status::unimplemented;
            }
        } else if (jcp.ic_block == 48) {
            if (is_3d) {
                if (no_vnni_format)
                    wei_tag = with_groups ? gIdhwo48i : Idhwo48i;
                else if (jcp.wei_dt == s8) {
                    if (jcp.is_oc_padded)
                        wei_tag = with_groups ? gIdhwO16o48i4o : IdhwO16o48i4o;
                    else
                        wei_tag = with_groups ? gIdhwO48i4o : IdhwO48i4o;
                } else if (one_of(jcp.wei_dt, bf16, f16)) {
                    if (jcp.is_oc_padded)
                        wei_tag = with_groups ? gIdhwO16o48i2o : IdhwO16o48i2o;
                    else
                        wei_tag = with_groups ? gIdhwO48i2o : IdhwO48i2o;
                } else
                    return status::unimplemented;
            } else if (is_1d) {
                if (no_vnni_format)
                    wei_tag = with_groups ? gIwo48i : Iwo48i;
                else if (jcp.wei_dt == s8) {
                    if (jcp.is_oc_padded)
                        wei_tag = with_groups ? gIwO16o48i4o : IwO16o48i4o;
                    else
                        wei_tag = with_groups ? gIwO48i4o : IwO48i4o;
                } else if (one_of(jcp.wei_dt, bf16, f16)) {
                    if (jcp.is_oc_padded)
                        wei_tag = with_groups ? gIwO16o48i2o : IwO16o48i2o;
                    else
                        wei_tag = with_groups ? gIwO48i2o : IwO48i2o;
                } else
                    return status::unimplemented;
            } else {
                assert(is_2d);
                UNUSED(is_2d);
                if (no_vnni_format)
                    wei_tag = with_groups ? gIhwo48i : Ihwo48i;
                else if (jcp.wei_dt == s8) {
                    if (jcp.is_oc_padded)
                        wei_tag = with_groups ? gIhwO16o48i4o : IhwO16o48i4o;
                    else
                        wei_tag = with_groups ? gIhwO48i4o : IhwO48i4o;
                } else if (one_of(jcp.wei_dt, bf16, f16)) {
                    if (jcp.is_oc_padded)
                        wei_tag = with_groups ? gIhwO16o48i2o : IhwO16o48i2o;
                    else
                        wei_tag = with_groups ? gIhwO48i2o : IhwO48i2o;
                } else
                    return status::unimplemented;
            }
        } else if (jcp.ic_block == 32) {
            if (is_3d) {
                if (no_vnni_format)
                    wei_tag = with_groups ? gIdhwo32i : Idhwo32i;
                else if (jcp.wei_dt == s8) {
                    if (jcp.is_oc_padded)
                        wei_tag = with_groups ? gIdhwO16o32i4o : IdhwO16o32i4o;
                    else
                        wei_tag = with_groups ? gIdhwO32i4o : IdhwO32i4o;
                } else if (one_of(jcp.wei_dt, bf16, f16)) {
                    if (jcp.is_oc_padded)
                        wei_tag = with_groups ? gIdhwO16o32i2o : IdhwO16o32i2o;
                    else
                        wei_tag = with_groups ? gIdhwO32i2o : IdhwO32i2o;
                } else
                    return status::unimplemented;
            } else if (is_1d) {
                if (no_vnni_format)
                    wei_tag = with_groups ? gIwo32i : Iwo32i;
                else if (jcp.wei_dt == s8) {
                    if (jcp.is_oc_padded)
                        wei_tag = with_groups ? gIwO16o32i4o : IwO16o32i4o;
                    else
                        wei_tag = with_groups ? gIwO32i4o : IwO32i4o;
                } else if (one_of(jcp.wei_dt, bf16, f16)) {
                    if (jcp.is_oc_padded)
                        wei_tag = with_groups ? gIwO16o32i2o : IwO16o32i2o;
                    else
                        wei_tag = with_groups ? gIwO32i2o : IwO32i2o;
                } else
                    return status::unimplemented;
            } else {
                assert(is_2d);
                UNUSED(is_2d);
                if (no_vnni_format)
                    wei_tag = with_groups ? gIhwo32i : Ihwo32i;
                else if (jcp.wei_dt == s8) {
                    if (jcp.is_oc_padded)
                        wei_tag = with_groups ? gIhwO16o32i4o : IhwO16o32i4o;
                    else
                        wei_tag = with_groups ? gIhwO32i4o : IhwO32i4o;
                } else if (one_of(jcp.wei_dt, bf16, f16)) {
                    if (jcp.is_oc_padded)
                        wei_tag = with_groups ? gIhwO16o32i2o : IhwO16o32i2o;
                    else
                        wei_tag = with_groups ? gIhwO32i2o : IhwO32i2o;
                } else
                    return status::unimplemented;
            }
        } else if (jcp.ic_block == 24) {
            if (is_3d) {
                if (no_vnni_format)
                    wei_tag = with_groups ? gIdhwo24i : Idhwo24i;
                else if (jcp.wei_dt == s8)
                    wei_tag = with_groups ? gIdhwO24i4o : IdhwO24i4o;
                else if (one_of(jcp.wei_dt, bf16, f16))
                    wei_tag = with_groups ? gIdhwO24i2o : IdhwO24i2o;
                else
                    return status::unimplemented;
            } else if (is_1d) {
                if (no_vnni_format)
                    wei_tag = with_groups ? gIwo24i : Iwo24i;
                else if (jcp.wei_dt == s8)
                    wei_tag = with_groups ? gIwO24i4o : IwO24i4o;
                else if (one_of(jcp.wei_dt, bf16, f16))
                    wei_tag = with_groups ? gIwO24i2o : IwO24i2o;
                else
                    return status::unimplemented;
            } else {
                assert(is_2d);
                UNUSED(is_2d);

                if (no_vnni_format)
                    wei_tag = with_groups ? gIhwo24i : Ihwo24i;
                else if (jcp.wei_dt == s8)
                    wei_tag = with_groups ? gIhwO24i4o : IhwO24i4o;
                else if (one_of(jcp.wei_dt, bf16, f16))
                    wei_tag = with_groups ? gIhwO24i2o : IhwO24i2o;
                else
                    return status::unimplemented;
            }
        } else if (jcp.ic_block == 16) {
            if (is_3d) {
                if (no_vnni_format)
                    wei_tag = with_groups ? gIdhwo16i : Idhwo16i;
                else if (jcp.wei_dt == s8) {
                    if (jcp.is_oc_padded)
                        wei_tag = with_groups ? gIdhwO16o16i4o : IdhwO16o16i4o;
                    else
                        wei_tag = with_groups ? gIdhwO16i4o : IdhwO16i4o;
                } else if (one_of(jcp.wei_dt, bf16, f16)) {
                    if (jcp.is_oc_padded)
                        wei_tag = with_groups ? gIdhwO16o16i2o : IdhwO16o16i2o;
                    else
                        wei_tag = with_groups ? gIdhwO16i2o : IdhwO16i2o;
                } else
                    return status::unimplemented;
            } else if (is_1d) {
                if (no_vnni_format)
                    wei_tag = with_groups ? gIwo16i : Iwo16i;
                else if (jcp.wei_dt == s8) {
                    if (jcp.is_oc_padded)
                        wei_tag = with_groups ? gIwO16o16i4o : IwO16o16i4o;
                    else
                        wei_tag = with_groups ? gIwO16i4o : IwO16i4o;
                } else if (one_of(jcp.wei_dt, bf16, f16)) {
                    if (jcp.is_oc_padded)
                        wei_tag = with_groups ? gIwO16o16i2o : IwO16o16i2o;
                    else
                        wei_tag = with_groups ? gIwO16i2o : IwO16i2o;
                } else
                    return status::unimplemented;
            } else {
                assert(is_2d);
                UNUSED(is_2d);

                if (no_vnni_format)
                    wei_tag = with_groups ? gIhwo16i : Ihwo16i;
                else if (jcp.wei_dt == s8) {
                    if (jcp.is_oc_padded)
                        wei_tag = with_groups ? gIhwO16o16i4o : IhwO16o16i4o;
                    else
                        wei_tag = with_groups ? gIhwO16i4o : IhwO16i4o;
                } else if (one_of(jcp.wei_dt, bf16, f16)) {
                    if (jcp.is_oc_padded)
                        wei_tag = with_groups ? gIhwO16o16i2o : IhwO16o16i2o;
                    else
                        wei_tag = with_groups ? gIhwO16i2o : IhwO16i2o;
                } else
                    return status::unimplemented;
            }
        } else if (jcp.ic_block == 8) {
            if (is_3d) {
                if (no_vnni_format)
                    wei_tag = with_groups ? gIdhwo8i : Idhwo8i;
                else if (jcp.wei_dt == s8)
                    wei_tag = with_groups ? gIdhwO8i4o : IdhwO8i4o;
                else if (one_of(jcp.wei_dt, bf16, f16))
                    wei_tag = with_groups ? gIdhwO8i2o : IdhwO8i2o;
                else
                    return status::unimplemented;
            } else if (is_1d) {
                if (no_vnni_format)
                    wei_tag = with_groups ? gIwo8i : Iwo8i;
                else if (jcp.wei_dt == s8)
                    wei_tag = with_groups ? gIwO8i4o : IwO8i4o;
                else if (one_of(jcp.wei_dt, bf16, f16))
                    wei_tag = with_groups ? gIwO8i2o : IwO8i2o;
                else
                    return status::unimplemented;
            } else {
                assert(is_2d);
                UNUSED(is_2d);

                if (no_vnni_format)
                    wei_tag = with_groups ? gIhwo8i : Ihwo8i;
                else if (jcp.wei_dt == s8)
                    wei_tag = with_groups ? gIhwO8i4o : IhwO8i4o;
                else if (one_of(jcp.wei_dt, bf16, f16))
                    wei_tag = with_groups ? gIhwO8i2o : IhwO8i2o;
                else
                    return status::unimplemented;
            }
        } else
            return status::unimplemented;
    }

    src_tag = dst_tag;

    CHECK(init_tag(jcp.src_tag, diff_dst_md, diff_dst_d, src_tag));
    CHECK(init_tag(jcp.dst_tag, diff_src_md, diff_src_d, dst_tag));
    CHECK(init_tag(jcp.wei_tag, weights_md, weights_d, wei_tag));

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

    brg_blocking_t() {
        // TODO: This is a broken form of initialization for a base class.
        // Either set default values in a base class, or provide a proper
        // default ctor, or take a `jit_brgemm_conv_conf_t` object to initialize
        // a base class object.
        jit_brgemm_conv_conf_t *base
                = static_cast<jit_brgemm_conv_conf_t *>(this);
        *base = jit_brgemm_conv_conf_t();
        init();
    }
    brg_blocking_t(const jit_brgemm_conv_conf_t &jcp)
        : jit_brgemm_conv_conf_t(jcp) {
        init();
    }
    void init() {
        ur = 0;
        ur_block = 0;
        ur_block_tail = 0;
        eff = 0.f;
        nb_kd = 0;
        nb_kh = 0;
        nb_kw = 0;
        sp = 0;
        sp_block = 0;
        nb_sp = 0;
        eff = 0;
        max_regs = isa_num_vregs(isa);
        bcast_simd = acc_simd_w;
    }

    int ur, ur_block, ur_block_tail;
    int nb_kd, nb_kh, nb_kw;
    int max_regs;
    int bcast_simd;
    float eff;
    static unsigned L1;
    static unsigned L2;
    static unsigned L3;
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

    int sp, sp_block, nb_sp;
    static int last_oc_block_size;

    void get_from_jcp(const jit_brgemm_conv_conf_t &jcp) { *this = jcp; }
    void save_to_jcp(jit_brgemm_conv_conf_t &jcp) const { jcp = *this; }

    status_t estimate_brgemm_ur();
    status_t get_brgemm_ur(
            const primitive_attr_t *attr, const memory_desc_t &dst_md);

    float io_k(dim_t src, dim_t wei, dim_t dst, float n, float pk,
            bool is_broadcast, bool is_shared) const;

    float io_k(const loop_t loop, const array_in_loop_t arr, float pk,
            bool is_broadcast, bool is_shared) const;

    void select_oc_block();

    void update_blocks();
    bool fast_check_ic_block() const;
    float est_eff();
    void iterate_ker_block(brg_blocking_t &best_brgb, int kd_block,
            int kh_block, bool maybe_use_buffer, int max_iw_block_thr);
    status_t calc_blocks();

    bool fast_check_ic_block_1x1() const;
    float est_eff_1x1();

    // utils
    static int get_inp_size(
            int max_src_size, int dst_size, int k, int stride, int dilate) {
        auto adj_str = nstl::min(k, stride);
        const auto res = nstl::min(max_src_size,
                calculate_end_padding(0, dst_size, 0, adj_str,
                        calculate_extended_filter_size(k, dilate)));
        return res;
    }

    static int get_inp_block_size(
            int out_size, int stride, int ext_k, int padding) {
        const auto res = div_up(out_size + padding % stride, stride)
                + (ext_k - 1 - padding % stride) / stride;
        return res;
    }

    static float squeeze_val(float eff, float koeff) {
        if (koeff <= 0) return 1;
        if (koeff == 1) return eff;
        const auto k = 1.f / koeff;
        return (k > 1.f) ? (k - 1 + eff) / k : eff * koeff;
    }

    static int estimate_ur(int ic_block) {
        const auto est_ur = (ic_block == 64)
                ? 6
                : ((ic_block == 48) ? 9 : ((ic_block == 32) ? 14 : 28));
        return est_ur;
    }

    int inp_w(int out_w, int ker_w) const {
        return get_inp_size(ow, out_w, ker_w, stride_w, dilate_w);
    }

    int rnd_simd(int val) const { return rnd_up(val, simd_w); }

    int rnd_inp_simd(int out_w, int ker_w, int voc) const {
        const auto vsp = inp_w(out_w, ker_w);
        return ((stride_w == 1 && voc >= oc) ? rnd_up(vsp * voc, simd_w)
                                             : vsp * rnd_up(voc, simd_w));
    }

    static constexpr int MAXNLOOPS = 32;
    loop_t loop[MAXNLOOPS];
};

unsigned brg_blocking_t::L1;
unsigned brg_blocking_t::L2;
unsigned brg_blocking_t::L3;
int brg_blocking_t::last_oc_block_size;

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

void brg_blocking_t::select_oc_block() {
    const auto padded_oc = last_oc_block_size * (is_oc_padded ? acc_simd_w : 1);
    oc_block = (exec_type == exec_trans ? rnd_up(oc, padded_oc) : oc);
    nb_oc = utils::div_up(oc, oc_block);
}

status_t brg_blocking_t::estimate_brgemm_ur() {
    // Simple simulation of brgemm_desc init
    if (sp_block <= 0) return status::invalid_arguments;
    LDA = exec_type == exec_trans ? oc_block : oc_without_padding * ngroups;
    LDB = ic_block;
    LDC = use_buffer ? ic_block : stride_w * ic_without_padding;

    // Configure matrix sizes
    // for amx if oc_block != oc then we use exec_trans so K is oc_block
    const auto padded_oc = last_oc_block_size * (is_oc_padded ? acc_simd_w : 1);

    ocp = rnd_up(oc, padded_oc);

    const auto adj_sp = div_up(iw_block, stride_w);
    M = brgM = adj_sp >= sp_block ? sp_block : 0;
    M_tail = brgM_tail = adj_sp % sp_block;

    N = ic >= ic_block ? ic_block : 0;
    N_tail = ic % ic_block;
    K = oc >= oc_block ? oc_block : 0;
    K_tail = oc_block;

    const auto vK = K > 0 ? K : K_tail;
    const auto vM = M > 0 ? M : M_tail;
    const auto vN = N > 0 ? N : N_tail;

    const float alpha = 1.0;
    const float beta = 0.0;
    brgemm_t brg;
    brgemm_utils::init_brgemm_conf(&brg, isa, brgemm_addr, src_dt, wei_dt,
            brgemm_row_major, alpha, beta, LDA, LDB, LDC, vM, vN, vK, nullptr,
            is_bf32);
    CHECK(brgemm_utils::brgemm_blocking(&brg));
    ur = brg.bd_block * (is_amx(isa) ? brg.bd_block2 : 1);
    if (ur == 0) return status::invalid_arguments;
    ur_block = brg.bd_block;
    if (is_1x1 && is_amx(isa) && M > 0 && M_tail > 0) {
        brgemm_t brg_sp_tail;
        brgemm_utils::init_brgemm_conf(&brg_sp_tail, isa, brgemm_addr, src_dt,
                wei_dt, brgemm_row_major, alpha, beta, LDA, LDB, LDC, M_tail,
                vN, vK, nullptr, is_bf32);
        CHECK(brgemm_utils::brgemm_blocking(&brg_sp_tail));
        ur_block_tail = brg_sp_tail.bd_block;
    } else {
        ur_block_tail = 0;
    }
    return status::success;
}

status_t brg_blocking_t::get_brgemm_ur(
        const primitive_attr_t *attr, const memory_desc_t &dst_md) {
    // Detailed simulation of brgemm convolution init
    if (sp_block <= 0 || oc_block <= 0 || ic_block <= 0)
        return status::invalid_arguments;
    CHECK(estimate_brgemm_ur());

    LDD = stride_w * ic_without_padding;

    const float alpha = 1.0;
    const float beta = 1.0;
    const float beta_init = 0.0;

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
                    brgemm_t brg;
                    brgemm_strides_t brg_strides;
                    brg_strides.stride_a = ngroups * oc_without_padding
                            * (dilate_w + 1) * src_dsz;
                    //weights are padded by ic_block and last_oc_block
                    brg_strides.stride_b = rnd_up(oc, last_oc_block_size)
                            * rnd_up(ic, ic_block) * wei_dsz;
                    const auto strides_ptr = (brg_type == brgemm_strd)
                            ? &brg_strides
                            : nullptr;
                    brgemm_utils::init_brgemm_conf(&brg, isa, brg_type, src_dt,
                            wei_dt, brgemm_row_major, alpha, vbeta, LDA, LDB,
                            LDC, vM, vN, vK, strides_ptr, is_bf32);
                    CHECK(brgemm_utils::brgemm_blocking(&brg));

                    brgemm_attr_t brgattr;
                    brgattr.max_bs = max_batch;
                    const auto max_vpad = (exec_type == exec_vpad)
                            ? nstl::max(l_pad, r_pad)
                            : 0;
                    brgattr.max_top_vpad = max_vpad;
                    brgattr.max_bottom_vpad = max_vpad;
                    brgattr.fpmath_mode = attr->fpmath_mode_;
                    CHECK(brgemm_desc_set_attr(&brg, brgattr));

                    brg.with_sum = with_sum;
                    CHECK(brgemm_desc_set_postops(
                            &brg, attr, &dst_md, LDD, bia_dt));
                }
            }
        }
    }

    return status::success;
}

void brg_blocking_t::update_blocks() {
    if (sp_block <= 0
            || utils::one_of(0, id_block, ih_block, oc_block, ic_block,
                    kd_block, kh_block, kw_block, is_block, iw_block))
        return;

    const bool maskrcnn_cond = is_superset(isa, avx2_vnni) && ic == 256
            && oc == 256 && everyone_is(28, ih, iw) && everyone_is(14, oh, ow)
            && everyone_is(2, kh, stride_h, kw, stride_w);
    if (maskrcnn_cond) {
        ic_block = 64;
        iw_block = 28;
        ih_block = 14;
    }
    nb_id = div_up(id, id_block);
    nb_ih = div_up(ih, ih_block);
    nb_oc = div_up(oc, oc_block);
    nb_ic = div_up(ic, ic_block);
    nb_kd = div_up(kd, kd_block);
    nb_kh = div_up(kh, kh_block);
    nb_kw = div_up(kw, kw_block);
    nb_iw = div_up(iw, iw_block);

    sp = iw;
    sp_block = iw_block;
    nb_sp = nb_iw;

    ow_block = get_inp_block_size(iw_block, stride_w, ext_kw, l_pad);
    oh_block = get_inp_block_size(ih_block, stride_h, ext_kh, t_pad);
    od_block = get_inp_block_size(id_block, stride_d, ext_kd, f_pad);
}

bool brg_blocking_t::fast_check_ic_block() const {
    // This function is for reducing the number of blocking variants
    // TODO: eliminate heuristic in this function
    if (is_1x1) return fast_check_ic_block_1x1();
    const auto rnd_ic = rnd_up(ic, acc_simd_w);
    auto res = false;
    if (ic_block == 64) {
        res = (rnd_ic % ic_block == 0 && rnd_ic * wei_dsz < 192 * 4);
    } else if (ic_block == 48) {
        // TODO: edit this heuristic for bwd_d
        const bool big_spatial
                = od * oh * ow > 81 * stride_d * stride_h * stride_w;
        res = (rnd_ic % ic_block == 0 && rnd_ic * wei_dsz <= 384 * 4
                && big_spatial);
    } else
        res = true;

    return res;
}

float brg_blocking_t::est_eff() {
    if (is_1x1) return est_eff_1x1();
    const auto icblock = ic_block / acc_simd_w;

    const auto brgemm_microkernel_eff
            = (static_cast<float>(icblock) * ur) / ((ur + icblock) * max_regs);

    const auto ur_eff = static_cast<float>(sp_block) / rnd_up(sp_block, ur);
    const auto brgemm_eff = squeeze_val(ur
                    * (2.f - nstl::min(1.9f, static_cast<float>(ur) / sp_block))
                    / 64,
            0.5f);

    const auto sp_amount = nb_id * nb_ih * nb_sp;
    const auto work_amount = mb * ngroups * nb_ic * sp_amount;
    const auto sp_eff = (static_cast<float>(sp) / rnd_up(sp, sp_block));

    const auto thr_eff = static_cast<float>(work_amount)
            / utils::rnd_up(work_amount, nthr);

    const auto ic_block_eff = static_cast<float>(ic) / rnd_up(ic, ic_block);

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
            int n {0}, g {0}, icb {0}, idp {0}, ihp {0}, spb {0};
            if (loop_order == loop_ndhwgc)
                nd_iterator_init(start, n, mb, idp, id, ihp, ih, spb, nb_sp, g,
                        ngroups, icb, nb_ic);
            else if (loop_order == loop_ngcdhw)
                nd_iterator_init(start, n, mb, g, ngroups, icb, nb_ic, idp, id,
                        ihp, ih, spb, nb_sp);

            for (auto work = start; work < end; work++) {
                const int icp = icb * ic_block;
                const auto ic_sz = nstl::min(ic - icp, ic_block);
                int sp_sz = 0;
                const int spp = spb * sp_block;
                sp_sz = nstl::min(sp - spp, sp_block);
                thr_job += sp_sz * ic_sz;

                nd_iterator_step(n, mb, idp, id, ihp, ih, spb, nb_sp, g,
                        ngroups, icb, nb_ic);
                if (loop_order == loop_ndhwgc)
                    nd_iterator_step(n, mb, idp, id, ihp, ih, spb, nb_sp, g,
                            ngroups, icb, nb_ic);
                else if (loop_order == loop_ngcdhw)
                    nd_iterator_step(n, mb, g, ngroups, icb, nb_ic, idp, id,
                            ihp, ih, spb, nb_sp);
            }
            thr_jobs[ithr] = thr_job;
        }

        dim_t max_job = 0;
        dim_t sum_job = 0;
        for (int ithr = 0; ithr < nthr; ithr++) {
            if (thr_jobs[ithr] > max_job) max_job = thr_jobs[ithr];
            sum_job += thr_jobs[ithr];
        }
        job_eff = max_job == 0 ? 1
                               : static_cast<float>(sum_job) / (max_job * nthr);

    } else {
        job_eff = thr_eff;
    }

    const auto oc_blocking_size = oc_block * nb_oc_blocking;
    const auto ic_blocking_size = ic_block * oc_blocking_size;

    int l = -1;

    // -- brgemm kernel: loop by simd_w  --
    l++;
    const auto inp_ur = inp_w(ur, kw_block);
    loop[l].src.set(inp_ur * simd_w, 1, bcast_simd);
    loop[l].dst.set(0, 1);
    loop[l].wei.set(ic_block, 1);

    // -- brgemm kernel: loop by kw in kw_block  --
    l++;
    auto src_is = rnd_inp_simd(ur, kw_block, oc_blocking_size);
    loop[l].src.set(src_is, 1, kw_block);
    loop[l].dst.set(0, 1);
    loop[l].wei.set(ic_blocking_size, 1);

    // -- brgemm kernel: loop by batch (grouped by kw_block) in ur  --
    l++;
    loop[l].src.set(src_is, 1);
    loop[l].dst.set(0, 1);
    auto wei_is = kw_block * ic_blocking_size;
    loop[l].wei.set(wei_is, 1);
    // -- brgemm kernel: loop by ur in sp_block --
    l++;
    const auto nb_ur = div_up(sp_block, ur);
    loop[l].src.set(kd_block * kh_block * src_is, 1);
    loop[l].dst.set(ur * ic_block, 1);
    wei_is = kd_block * kh_block * kw_block * ic_blocking_size;
    loop[l].wei.set(wei_is, nb_ur);

    // -- harness: loop by k_blocks in ks --
    l++;
    loop[l].src.set(kd_block * kh_block
                    * rnd_inp_simd(sp_block, kw_block, oc_blocking_size),
            1);
    loop[l].dst.set(sp_block * ic_block, nb_kd * nb_kh * nb_kw);
    loop[l].wei.set(wei_is, 1);

    // -- brgemm kernel: loop by oc_chunks --
    l++;
    const auto oc_chunks = div_up(nb_oc, nb_oc_blocking);
    loop[l].src.set(kd * kh * rnd_inp_simd(sp_block, kw, oc_blocking_size), 1);
    loop[l].dst.set(sp_block * ic_block, oc_chunks);
    wei_is = kd * kh * kw * ic_blocking_size;
    loop[l].wei.set(wei_is, 1);

    const auto dim_ic = (loop_order == loop_ndhwgc) ? 1 : sp_amount;
    const auto nb_ic_thr = nstl::min(nb_ic, div_up(job, dim_ic));
    const auto ic_thr = nstl::min(ic, nb_ic_thr * ic_block);
    const auto nsimd_ic_thr = div_up(ic_thr, simd_w);

    const auto dim_sp = (loop_order == loop_ndhwgc) ? ngroups * nb_ic : 1;
    const auto nb_sp_thr = nstl::min(nb_sp, div_up(job, dim_sp));
    const auto sp_thr = nstl::min(sp, nb_sp_thr * sp_block);

    const auto dim_ih = nb_sp * dim_sp;
    const int nb_ih_thr = nstl::min(nb_ih, div_up(job, dim_ih));
    const int ih_thr = nstl::min(ih, nb_ih_thr * ih_block);

    const auto dim_id = nb_ih * dim_ih;
    const int nb_id_thr = nstl::min(nb_id, div_up(job, dim_id));
    const int id_thr = nstl::min(id, nb_id_thr * id_block);

    src_is = kd * kh * rnd_inp_simd(sp_block, kw, oc);

    auto wei_op = kd * kh * kw * icblock * oc;

    if (loop_order == loop_ndhwgc) {
        // -- harness: loop by ic_block --
        l++;
        loop[l].src.set(src_is, nb_ic_thr);
        loop[l].dst.set(sp_block * ic_block, 1);
        wei_is = kd * kh * kw * ic_block * oc;
        wei_op = kd * kh * kw * nsimd_ic_thr * oc;
        loop[l].wei.set(wei_is, 1);
    }

    // -- harness: loop by sp_blocks --
    l++;
    loop[l].src.set(src_is, 1);
    const auto rnd_ic_for_sp
            = simd_w * ((loop_order == loop_ndhwgc) ? nsimd_ic_thr : icblock);
    loop[l].dst.set(sp_block * rnd_ic_for_sp, 1);
    loop[l].wei.set(wei_op * simd_w, nb_sp_thr);
    // oh_block almost all is 1. TODO: manage oh_block != 1
    // -- harness: loop by oh_blocks --
    l++;
    src_is = kd * kh * rnd_inp_simd(sp_thr, kw, oc);
    loop[l].src.set(ih_block * src_is, 1);
    loop[l].dst.set(sp_thr * rnd_ic_for_sp, 1);
    loop[l].wei.set(wei_op * simd_w, nb_ih_thr);
    // od_block almost all is 1. TODO: manage oh_block != 1
    // -- harness: loop by od_blocks --
    l++;
    loop[l].src.set(id_block * ih_thr * src_is, 1);
    loop[l].dst.set(ih_thr * sp_thr * rnd_ic_for_sp, 1);
    loop[l].wei.set(wei_op * simd_w, nb_id_thr);

    if (loop_order != loop_ndhwgc) {
        // -- harness: loop by ic_block --
        l++;
        loop[l].src.set(id_thr * ih_thr * src_is, nb_ic_thr);
        loop[l].dst.set(ic_block * id_thr * ih_thr * sp_thr, 1);
        loop[l].wei.set(kd * kh * kw * ic_block * oc, 1);
    }

    // -- harness: loop by mb --
    l++;
    const auto mb_thr = nstl::min(mb, div_up(job, sp_amount * ngroups * nb_ic));
    loop[l].src.set(id_thr * ih_thr * src_is, 1);
    loop[l].dst.set(id_thr * ih_thr * sp_thr * nsimd_ic_thr * simd_w, 1);
    loop[l].wei.set(kd * kh * kw * nsimd_ic_thr * simd_w * oc, mb_thr);

    const auto src_op = static_cast<dim_t>(mb_thr) * id_thr * ih_thr * sp_thr
            * kd * kh * kw * oc;
    const auto dst_op = static_cast<dim_t>(mb_thr) * id_thr * ih_thr * sp_thr
            * nsimd_ic_thr;
    wei_op = kd * kh * kw * nsimd_ic_thr * oc;

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
    const auto call_kernel_cost = job * oc_chunks * nb_kd * nb_kh * nb_kw;

    const auto cache_eff = (static_cast<dim_t>(mb) * id * ih * sp * oc * ic)
            / (nthr * (src_cost + dst_cost + wei_cost + call_kernel_cost));
    const auto res_eff = ic_block_eff * brgemm_microkernel_eff * sp_eff
            * job_eff * ur_eff * cache_eff * brgemm_eff;
    return res_eff;
}

void brg_blocking_t::iterate_ker_block(brg_blocking_t &best_brgb, int kd_block_,
        int kh_block_, bool maybe_use_buffer, int max_iw_block_thr) {
    kd_block = kd_block_;
    kh_block = kh_block_;

    kw_block = kw;
    kd_block_pad = kd_block;
    kh_block_pad = kh_block;
    kw_block_pad = kw_block;

    const auto w_block_size = 2 * src_dsz * oc * owp + dst_dsz * iw * ic_block;
    const auto other_size = wei_dsz * kd * kh * kw * oc * ic_block
            + acc_dsz * 2 * amx_h * ic_block;
    const auto L2_available = nstl::min(static_cast<size_t>(div_up(L2, 2)),
            other_size > L2 ? 0 : L2 - other_size);
    if (odp * ohp * w_block_size > L2_available) {
        id_block = utils::saturate(
                1, id, int(L2_available / (ohp * w_block_size)));
        if (id_block == 1)
            ih_block = utils::saturate(
                    1, ih, int(L2_available / (w_block_size)));
        else
            ih_block = ih;
    } else {
        id_block = 1;
        ih_block = ih;
    }
    if (is_amx(isa)) {
        // try to fit into L1
        bool L1_fit_res = false;
        auto cur_id_block = id_block;
        auto cur_ih_block = ih_block;
        const auto src_w_block_size
                = src_dsz * oc * owp + dst_dsz * iw * ic_block;
        if (src_w_block_size < L1) {
            cur_id_block = utils::saturate(
                    1, id, int(L1 / (ohp * src_w_block_size)));
            if (cur_id_block == 1)
                cur_ih_block
                        = utils::saturate(1, ih, int(L1 / (src_w_block_size)));
        }
        for (; cur_id_block > 1; cur_id_block--) {
            const auto sp_size = cur_id_block * cur_ih_block * owp;
            if ((static_cast<float>(id) / rnd_up(id, cur_id_block)) > 0.9f
                    && static_cast<float>(sp_size) / rnd_up(sp, amx_h) > 0.8f) {
                L1_fit_res = true;
                break;
            }
        }
        if (cur_id_block == 1) {
            for (; cur_ih_block > 1; cur_ih_block--) {
                const auto sp_size = cur_ih_block * owp;
                if ((static_cast<float>(ih) / rnd_up(ih, cur_ih_block)) > 0.9f
                        && sp_size > 128) {
                    L1_fit_res = true;
                    break;
                }
            }
        }
        if (L1_fit_res) {
            id_block = cur_id_block;
            ih_block = cur_ih_block;
        }
    }

    // limit ih_block to have good threading
    const auto thr_ic_block
            = div_up(nthr, mb * div_up((ic > 32 ? ngroups : 1) * ic, ic_block));
    const auto thr_id_block = div_up(id, thr_ic_block);
    const auto thr_ih_block
            = div_up(ih, thr_ic_block * div_up(id, thr_id_block));
    id_block = nstl::min(id_block, thr_id_block);
    ih_block = nstl::min(ih_block, thr_ih_block);
    while ((id_block % stride_d != 0 || id % id_block != 0) && id_block < id)
        id_block++;
    while ((ih_block % stride_h != 0 || ih % ih_block != 0) && ih_block < ih)
        ih_block++;

    // --- Select iw_block ----
    const auto max_iw_block_L2 = iw;
    auto start_iw_block = nstl::min(max_iw_block_thr, max_iw_block_L2);

    sp = iw;
    const auto start_sp_block = start_iw_block;
    auto prev_spb = 0;
    for (auto ns = 1; ns <= sp; ns++) {
        const auto spb = div_up(sp, ns);
        if (spb == prev_spb || spb > start_sp_block) continue;
        if (spb % stride_w != 0) continue;
        if (iw % spb != 0) continue;

        prev_spb = spb;
        iw_block = spb;
        sp_block = iw_block;

        select_oc_block();

        use_buffer = maybe_use_buffer;

        const status_t st = estimate_brgemm_ur();
        if (st != status::success) continue;
        os_block = sp_block = iw_block;
        update_blocks();

        eff = est_eff();
        if (eff > best_brgb.eff || best_brgb.eff == 0) best_brgb = *this;
    }
}

status_t brg_blocking_t::calc_blocks() {
    sp = iw;

    nb_oc_blocking = 1;
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
    const auto max_iw_block_thr = utils::saturate(1, iw,
            static_cast<int>(div_up(
                    mb * ngroups * nb_ic * is, thr_eff_threshold * nthr)));

    iw_block = is_block = sp_block = -1;
    brg_blocking_t best_brgb = *this;
    for (const auto &kd_block : kd_blocks) {
        for (const auto &kh_block : kh_blocks) {
            iterate_ker_block(best_brgb, kd_block, kh_block, maybe_use_buffer,
                    max_iw_block_thr);
        }
    }
    *this = best_brgb;
    if (sp_block <= 0) return status::unimplemented;

    iw_block = is_block = sp_block;
    iw_tail = iw % iw_block;

    update_blocks();

    return status::success;
}

bool brg_blocking_t::fast_check_ic_block_1x1() const {
    // This function checks for reducing the number of blocking variants
    // TODO: eliminate heuristic in this function
    if (is_1x1 && is_amx(isa)) return true;
    const auto rnd_ic = rnd_up(ic, acc_simd_w);
    auto res = false;
    if (ic_block == 64) {
        const auto big_spatial
                = id * ih * iw >= 64 * stride_d * stride_h * stride_w;
        res = (rnd_ic % ic_block == 0 && big_spatial);
    } else if (ic_block == 48) {
        const auto ic_block_eff = static_cast<float>(ic) / rnd_up(ic, ic_block);
        res = (ic_block_eff >= 0.95f);
    } else
        res = true;

    return res;
}

float brg_blocking_t::est_eff_1x1() {
    const auto icblock = ic_block / acc_simd_w;

    auto calc_ave_blk = [&](int dim, int block, bool use_ave) -> float {
        const int nb = dim / block;
        constexpr int max_nb = 2; // only consider 2x2 tile blocking
        const int block2 = nstl::min(max_nb, nb);
        const int nb2 = nb / block2;
        const int nb2_tail = nb % block2;
        if (!use_ave) return block2;
        return (float(nb2) * block2 + nb2_tail) / div_up(nb, block2);
    };
    const bool use_ocb_ave = true;
    const auto icb_ave = calc_ave_blk(ic_block, acc_simd_w, use_ocb_ave);
    const bool use_spb_ave = false;
    const auto spb_ave = calc_ave_blk(sp_block, ur_block, use_spb_ave);
    const auto M_n_sp_blks = ur_block > 0 ? nstl::max(M, M_tail) / ur_block : 0;
    const auto M_tail_n_sp_blks
            = ur_block_tail > 0 ? M_tail / ur_block_tail : 0;

    // heuristic for maskrcnn workaround: use old blocking for some convolutions
    // TODO: remove this condition
    const bool maskrcnn_cond = (ic == 1024 && oc == 2048)
            || (ic == 1024 && oc == 512) || (ic == 256 && oc == 1024)
            || (ic == 512 && oc == 1024) || (ic == 512 && oc == 2048);
    const auto amx_fac = maskrcnn_cond
            ? (div_up(M + M_tail, 16) / (M_n_sp_blks + M_tail_n_sp_blks))
            : (static_cast<float>(div_up(M + M_tail, 16))
                    / (M_n_sp_blks + M_tail_n_sp_blks));

    const auto brgemm_microkernel_eff = is_amx(isa)
            ? amx_fac * (static_cast<float>(icb_ave) * spb_ave)
                    / (icb_ave + spb_ave)
            : (static_cast<float>(icblock) * ur) / ((ur + icblock) * max_regs);
    const auto ur_eff = static_cast<float>(sp_block) / rnd_up(sp_block, ur);
    const auto brgemm_eff = squeeze_val(ur
                    * (2.f - nstl::min(1.9f, static_cast<float>(ur) / sp_block))
                    / 64,
            0.5f);

    const auto sp_amount = nb_id * nb_ih * nb_sp;
    const auto work_amount = mb * ngroups * nb_ic * sp_amount;

    const auto sp_eff = static_cast<float>(sp) / rnd_up(sp, sp_block);
    const auto thr_eff = static_cast<float>(work_amount)
            / utils::rnd_up(work_amount, nthr);
    const auto ic_block_eff = static_cast<float>(ic) / rnd_up(ic, ic_block);

    const auto job = div_up(work_amount, nthr);

    const auto dim_ic = (loop_order == loop_ndhwgc) ? 1 : sp_amount;
    const auto nb_ic_thr = nstl::min(nb_ic, div_up(job, dim_ic));
    const auto ic_thr = nstl::min(ic, nb_ic_thr * ic_block);
    const auto nsimd_ic_thr = div_up(ic_thr, simd_w);

    const auto dim_sp = (loop_order == loop_ndhwgc) ? ngroups * nb_ic : 1;
    const auto nb_sp_thr = nstl::min(nb_sp, div_up(job, dim_sp));
    const auto sp_thr = nstl::min(sp, nb_sp_thr * sp_block);

    const auto dim_ih = nb_sp * dim_sp;
    const int nb_ih_thr = nstl::min(nb_ih, div_up(job, dim_ih));
    const int ih_thr = nstl::min(ih, nb_ih_thr * ih_block);

    const auto dim_id = nb_ih * dim_ih;
    const int nb_id_thr = nstl::min(nb_id, div_up(job, dim_id));
    const int id_thr = nstl::min(id, nb_id_thr * id_block);

    auto job_eff = 1.f;
    if (job < nthr) {
        std::vector<dim_t> thr_jobs(nthr);
        for (int ithr = 0; ithr < nthr; ithr++) {
            thr_jobs[ithr] = 0;
            if (ithr >= work_amount) continue;
            dim_t thr_job = 0;
            int start {0}, end {0};
            balance211(work_amount, nthr, ithr, start, end);
            int n {0}, g {0}, icb {0}, idp {0}, ihp {0}, spb {0};
            nd_iterator_init(start, n, mb, idp, id, ihp, ih, spb, nb_sp, g,
                    ngroups, icb, nb_ic);

            if (loop_order == loop_ndhwgc) {
                nd_iterator_init(start, n, mb, idp, id, ihp, ih, spb, nb_sp, g,
                        ngroups, icb, nb_ic);
            } else if (loop_order == loop_ngcdhw) {
                nd_iterator_init(start, n, mb, g, ngroups, icb, nb_ic, idp, id,
                        ihp, ih, spb, nb_sp);
            }

            for (auto work = start; work < end; work++) {
                const int icp = icb * ic_block;
                const auto ic_sz = nstl::min(ic - icp, ic_block);
                int sp_sz = 0;
                const int spp = spb * sp_block;
                sp_sz = nstl::min(sp - spp, sp_block);
                thr_job += sp_sz * ic_sz;
                nd_iterator_step(n, mb, idp, id, ihp, ih, spb, nb_sp, g,
                        ngroups, icb, nb_ic);
                if (loop_order == loop_ndhwgc) {
                    nd_iterator_step(n, mb, idp, id, ihp, ih, spb, nb_sp, g,
                            ngroups, icb, nb_ic);
                } else if (loop_order == loop_ngcdhw) {
                    nd_iterator_step(n, mb, g, ngroups, icb, nb_ic, idp, id,
                            ihp, ih, spb, nb_sp);
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

        job_eff = max_job == 0 ? 1
                               : static_cast<float>(sum_job) / (max_job * nthr);
    } else {
        job_eff = thr_eff;
    }

    const auto oc_blocking_size = oc_block * nb_oc_blocking;
    const auto ic_blocking_size = ic_block * oc_blocking_size;

    int l = -1;
    // -- brgemm kernel: loop by simd_w  --
    l++;
    loop[l].src.set(ur * simd_w, 1, bcast_simd);
    loop[l].dst.set(0, 1);
    loop[l].wei.set(ic_block, 1);

    // -- brgemm kernel: loop by ur in sp_block --
    l++;
    const auto nb_ur = div_up(sp_block, ur);
    const auto nb_sp_no_tail = sp / sp_block;
    const auto sp_block_tail = sp % sp_block;
    const auto nb_ur_average
            = (nb_sp_no_tail * nb_ur + div_up(sp_block_tail, ur)) / nb_sp;
    loop[l].src.set(ur * rnd_simd(oc_blocking_size), 1);
    loop[l].dst.set(ur * ic_block, 1);
    loop[l].wei.set(ic_blocking_size, is_amx(isa) ? nb_ur_average : nb_ur);
    // -- brgemm kernel: loop by ic_chunks --
    l++;
    const auto oc_chunks = div_up(nb_oc, nb_oc_blocking);
    loop[l].src.set(sp_block * oc_blocking_size, 1);
    loop[l].dst.set(sp_block * ic_block, oc_chunks);
    auto wei_is = ic_blocking_size;
    auto wei_op = icblock * oc;
    loop[l].wei.set(wei_is, 1);

    if (loop_order == loop_ndhwgc) {
        // -- harness: loop by oc_block --
        l++;
        loop[l].src.set(sp_block * rnd_simd(ic), nb_ic_thr);
        loop[l].dst.set(sp_block * ic_block, 1);
        wei_is = ic_block * ic;
        wei_op = nsimd_ic_thr * ic;
        loop[l].wei.set(wei_is, 1);
    }

    const auto rnd_ic_for_sp
            = simd_w * ((loop_order == loop_ndhwgc) ? nsimd_ic_thr : icblock);
    // -- harness: loop by sp_blocks --
    l++;
    loop[l].src.set(sp_block * oc_blocking_size, 1);
    loop[l].dst.set(sp_block * rnd_ic_for_sp, 1);
    loop[l].wei.set(wei_op * simd_w, nb_sp_thr);
    // -- harness: loop by oh_blocks --
    l++;
    loop[l].src.set(ih_block * sp_thr * rnd_simd(oc_blocking_size), 1);
    loop[l].dst.set(ih_block * sp_thr * rnd_ic_for_sp, 1);
    loop[l].wei.set(wei_op * simd_w, nb_ih_thr);
    // -- harness: loop by od_blocks --
    l++;
    loop[l].src.set(id_block * ih_thr * sp_thr * rnd_simd(oc_blocking_size), 1);
    loop[l].dst.set(id_block * ih_thr * sp_thr * rnd_ic_for_sp, 1);
    loop[l].wei.set(wei_op * simd_w, nb_id_thr);

    if (loop_order != loop_ndhwgc) {
        // -- harness: loop by oc_block --
        l++;
        loop[l].src.set(id_thr * ih_thr * rnd_simd(sp_thr * oc_blocking_size),
                nb_ic_thr);
        loop[l].dst.set(ic_block * id_thr * ih_thr * sp_thr, 1);
        loop[l].wei.set(ic_block * oc, 1);
    }

    // -- harness: loop by mb --
    l++;
    const auto mb_thr = nstl::min(mb, div_up(job, sp_amount * ngroups * nb_ic));
    loop[l].src.set(id_thr * ih_thr * sp_thr * rnd_simd(oc_blocking_size), 1);
    loop[l].dst.set(nsimd_ic_thr * simd_w * id_thr * ih_thr * sp_thr, 1);
    loop[l].wei.set(nsimd_ic_thr * oc * simd_w, mb_thr);

    const auto src_op = static_cast<dim_t>(mb_thr) * id_thr * ih_thr * sp_thr
            * oc_blocking_size;
    const auto dst_op = static_cast<dim_t>(mb_thr) * nsimd_ic_thr * id_thr
            * ih_thr * sp_thr;
    wei_op = nsimd_ic_thr * oc;

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
    const auto call_kernel_cost = job * oc_chunks;

    const auto up_sp_size = id * ih;

    const auto cache_eff = (static_cast<dim_t>(mb) * up_sp_size * sp * oc * ic)
            / (nthr * (src_cost + dst_cost + wei_cost + call_kernel_cost));

    const auto res_eff = ic_block_eff * brgemm_microkernel_eff * sp_eff
            * job_eff * ur_eff * cache_eff * brgemm_eff;
    return res_eff;
}

brgemm_broadcast_t get_zp_type(const primitive_attr_t &attr, int arg) {
    return attr.zero_points_.has_default_values(arg)
            ? brgemm_broadcast_t::none
            : brgemm_broadcast_t::per_tensor;
}

status_t init_jcp(jit_brgemm_conv_conf_t &jcp, cpu_isa_t isa,
        const convolution_desc_t &cd, memory_desc_t &diff_dst_md,
        memory_desc_t &weights_md, memory_desc_t &diff_src_md,
        memory_desc_t &bias_md, primitive_attr_t &attr, int nthreads,
        bool is_deconv) {
    using namespace prop_kind;

    brg_blocking_t::L1 = platform::get_per_core_cache_size(1);
    brg_blocking_t::L2 = platform::get_per_core_cache_size(2);
    brg_blocking_t::L3 = platform::get_per_core_cache_size(2);

    const memory_desc_wrapper diff_dst_d(&diff_dst_md);
    const memory_desc_wrapper weights_d(&weights_md);
    const memory_desc_wrapper diff_src_d(&diff_src_md);
    const memory_desc_wrapper bias_d(&bias_md);

    const bool with_groups = weights_d.ndims() == diff_src_d.ndims() + 1;
    int ndims = diff_src_d.ndims();

    jcp = zero<decltype(jcp)>();
    jcp.isa = isa;
    jcp.ndims = ndims;
    jcp.prop_kind = cd.prop_kind;
    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = diff_src_d.dims()[0];
    jcp.oc_without_padding = diff_dst_d.dims()[1] / jcp.ngroups;
    jcp.oc = jcp.oc_without_padding;
    jcp.ic_without_padding = diff_src_d.dims()[1];
    jcp.ic = jcp.ic_without_padding / jcp.ngroups;
    jcp.id = (ndims == 5) ? diff_src_d.dims()[2] : 1;
    jcp.ih = (ndims == 3) ? 1 : diff_src_d.dims()[ndims - 2];
    jcp.iw = diff_src_d.dims()[ndims - 1];
    jcp.od = (ndims == 5) ? diff_dst_d.dims()[2] : 1;
    jcp.oh = (ndims == 3) ? 1 : diff_dst_d.dims()[ndims - 2];
    jcp.ow = diff_dst_d.dims()[ndims - 1];
    jcp.kd = (ndims == 5) ? weights_d.dims()[with_groups + 2] : 1;
    jcp.kh = (ndims == 3) ? 1 : weights_d.dims()[with_groups + ndims - 2];
    jcp.kw = weights_d.dims()[with_groups + ndims - 1];
    jcp.f_pad = (ndims == 5) ? cd.padding[0][0] : 0;
    jcp.t_pad = (ndims == 3) ? 0 : cd.padding[0][ndims - 4];
    jcp.l_pad = cd.padding[0][ndims - 3];
    jcp.stride_d = (ndims == 5) ? cd.strides[0] : 1;
    jcp.stride_h = (ndims == 3) ? 1 : cd.strides[ndims - 4];
    jcp.stride_w = cd.strides[ndims - 3];

    if (everyone_is(1, jcp.stride_d, jcp.stride_h, jcp.stride_w))
        return status::unimplemented;

    if (jcp.id % jcp.stride_d != 0 || jcp.ih % jcp.stride_h != 0
            || jcp.iw % jcp.stride_w)
        return status::unimplemented;

    jcp.dilate_d = (ndims == 5) ? cd.dilates[0] : 0;
    jcp.dilate_h = (ndims == 3) ? 0 : cd.dilates[ndims - 4];
    jcp.dilate_w = cd.dilates[ndims - 3];

    if (!everyone_is(0, jcp.dilate_d, jcp.dilate_h, jcp.dilate_w))
        return status::unimplemented;

    jcp.is = jcp.id * jcp.ih * jcp.iw;

    jcp.ext_kd = calculate_extended_filter_size(jcp.kd, jcp.dilate_d);
    jcp.ext_kh = calculate_extended_filter_size(jcp.kh, jcp.dilate_h);
    jcp.ext_kw = calculate_extended_filter_size(jcp.kw, jcp.dilate_w);

    jcp.back_pad = calculate_end_padding(
            jcp.f_pad, jcp.od, jcp.id, jcp.stride_d, jcp.ext_kd);
    jcp.b_pad = calculate_end_padding(
            jcp.t_pad, jcp.oh, jcp.ih, jcp.stride_h, jcp.ext_kh);
    jcp.r_pad = calculate_end_padding(
            jcp.l_pad, jcp.ow, jcp.iw, jcp.stride_w, jcp.ext_kw);

    jcp.is_1x1 = jcp.f_pad <= 0 && jcp.back_pad <= 0 && jcp.t_pad <= 0
            && jcp.b_pad <= 0 && jcp.l_pad <= 0 && jcp.r_pad <= 0
            && utils::everyone_is(1, jcp.kd, jcp.kh, jcp.kw);

    jcp.with_bias = cd.bias_desc.format_kind != format_kind::undef;

    jcp.src_dt = diff_dst_md.data_type;
    jcp.dst_dt = diff_src_md.data_type;
    jcp.wei_dt = weights_md.data_type;
    jcp.bia_dt = jcp.with_bias ? bias_md.data_type : data_type::undef;

    jcp.acc_dt = types::is_integral_dt(jcp.src_dt) ? s32 : f32;

    jcp.src_dsz = types::data_type_size(jcp.src_dt);
    jcp.wei_dsz = types::data_type_size(jcp.wei_dt);
    jcp.dst_dsz = types::data_type_size(jcp.dst_dt);
    jcp.acc_dsz = types::data_type_size(jcp.acc_dt);
    jcp.bia_dsz = jcp.with_bias ? types::data_type_size(jcp.bia_dt) : 0;

    jcp.simd_w = isa_max_vlen(isa) / jcp.src_dsz;
    jcp.acc_simd_w = isa_max_vlen(isa) / jcp.acc_dsz;
    jcp.is_bf32 = everyone_is(f32, jcp.src_dt, jcp.wei_dt)
            && attr.fpmath_mode_ == fpmath_mode::bf16 && isa == avx512_core_amx;

    if (jcp.is_bf32) return status::unimplemented;

    brg_blocking_t::last_oc_block_size
            = (jcp.wei_dt == f16 && isa == avx512_core_fp16)
            ? 1
            : data_type_vnni_granularity(jcp.wei_dt);

    // TODO: optimize grouped convolutions with small oc
    const bool is_grouped_small_oc
            = jcp.prop_kind != prop_kind::backward_weights && with_groups
            && jcp.ngroups > 1 && jcp.oc <= jcp.acc_simd_w
            && IMPLICATION(is_amx(jcp.isa),
                    jcp.oc < 16
                            && jcp.ic < 16
                            // already optimized for amx 1x1 convs
                            && !jcp.is_1x1)
            // Enable the shapes not supported in direct convs
            && IMPLICATION(with_groups, is_groups_ok(jcp));
    if (is_grouped_small_oc) return status::unimplemented;

    // Dispatch the shapes to VNNI for better performance
    // TODO: optimize the perf of 3d shape with small oc and large spatial
    const auto max_small_shapes_sz = jcp.is_1x1
            ? static_cast<int32_t>(brg_blocking_t::L1) / 2
            : static_cast<int32_t>(brg_blocking_t::L1);
    const auto is_small_shape = is_amx(jcp.isa) && jcp.is <= 4 && jcp.oc <= 512
            && jcp.mb * jcp.ngroups * jcp.oc * jcp.ic <= max_small_shapes_sz;
    const auto is_3d_small_oc = is_amx(jcp.isa) && jcp.ndims == 5
            && jcp.oc * jcp.ic <= 32 && jcp.id >= 128 && jcp.ih >= 128
            && jcp.iw >= 128;
    if (is_small_shape || is_3d_small_oc) return status::unimplemented;

    const bool is_signed_input = jcp.src_dt == s8;
    jcp.s8s8_compensation_required = is_signed_input && !isa_has_s8s8(jcp.isa);
    jcp.has_int8_vnni = isa_has_int8_vnni(jcp.isa);

    if (!IMPLICATION(jcp.wei_dt == s8,
                is_superset(jcp.isa, avx512_core_vnni)
                        || one_of(jcp.isa, avx2_vnni, avx2_vnni_2)))
        return status::unimplemented;

    if (!IMPLICATION(jcp.wei_dt == bf16,
                is_superset(jcp.isa, avx512_core_bf16)
                        || is_superset(jcp.isa, avx2_vnni_2)))
        return status::unimplemented;

    if (!IMPLICATION(jcp.wei_dt == f16,
                is_superset(jcp.isa, avx512_core_fp16)
                        || is_superset(jcp.isa, avx2_vnni_2)))
        return status::unimplemented;

    const bool is_f32
            = utils::everyone_is(f32, jcp.src_dt, jcp.wei_dt, jcp.dst_dt);
    if (!IMPLICATION(is_f32, one_of(isa, avx512_core, avx2)))
        return status::unimplemented;

    if (!post_ops_ok(jcp, attr, diff_src_d, is_deconv))
        return status::unimplemented;

    jcp.amx_h = 16;
    jcp.amx_w = 64 / jcp.src_dsz;

    if (jcp.with_bias) {
        if (bias_d.format_kind() == format_kind::any)
            CHECK(memory_desc_init_by_tag(bias_md, x));
    }

    const auto &p = attr.post_ops_;
    jcp.with_sum = p.find(primitive_kind::sum) != -1;
    const int eltwise_ind = p.find(primitive_kind::eltwise);
    jcp.with_eltwise = eltwise_ind != -1;

    const int binary_ind = p.find(primitive_kind::binary);
    const int prelu_ind = p.find(primitive_kind::prelu);
    jcp.with_binary = !everyone_is(-1, binary_ind, prelu_ind);

    jcp.src_zero_point
            = get_zp_type(attr, is_deconv ? DNNL_ARG_SRC : DNNL_ARG_DIFF_DST)
            != brgemm_broadcast_t::none;
    jcp.dst_zero_point
            = get_zp_type(attr, is_deconv ? DNNL_ARG_DST : DNNL_ARG_DIFF_SRC)
            != brgemm_broadcast_t::none;

    // TODO: Extend zero points support to AMX
    const bool has_zero_points = jcp.src_zero_point || jcp.dst_zero_point;

    const bool params_ok = IMPLICATION(has_zero_points, !is_amx(jcp.isa))
            && IMPLICATION(has_zero_points, utils::one_of(jcp.src_dt, u8, s8))
            && IMPLICATION(jcp.src_zero_point,
                    attr.zero_points_.common(
                            is_deconv ? DNNL_ARG_SRC : DNNL_ARG_DIFF_DST))
            && IMPLICATION(jcp.dst_zero_point,
                    attr.zero_points_.common(
                            is_deconv ? DNNL_ARG_DST : DNNL_ARG_DIFF_SRC));
    if (!params_ok) return status::unimplemented;

    if (is_deconv) {
        const auto &src_scales = attr.scales_.get(DNNL_ARG_SRC);
        const auto &wei_scales = attr.scales_.get(DNNL_ARG_WEIGHTS);
        jcp.with_scales = !src_scales.has_default_values()
                || !wei_scales.has_default_values();
        jcp.is_ic_scale = wei_scales.mask_ != 0;
    }

    jcp.nthr = nthreads;
    jcp.kh_sets = 1;
    jcp.kw_sets = 1;
    jcp.copy_block_only = false;
    jcp.amx_tile_load_xx = false;
    jcp.use_M_mask = 0;
    jcp.is_is_blocking = false;
    jcp.oskip = 0;
    jcp.use_uker = false;
    jcp.use_interleave_stores = false;
    jcp.hint_prefetching = brgemm_kernel_prefetching_t::brgemm_prf_default;
    jcp.brgemm_bd_loop_innermost = false;

    // fast check data layout before spending time for blocking selection
    format_tag_t src_tag = pick(jcp.ndims - 3, nwc, nhwc, ndhwc);

    CHECK(init_tag(jcp.src_tag, diff_dst_md, diff_dst_d, src_tag));

    return status::success;
}

void set_k_range(int P, int D, int S, dim_t i, dim_t O, int K, int &k_s,
        int &k_f, bool is_w) {
    int s(0), o_test(0);
    while (true) {
        o_test = i + P - s * D;
        if (o_test % S == 0) break;
        s++;
    }

    k_f = is_w ? K : nstl::min(K, static_cast<int>(div_up(i + P + 1, D)));
    k_s = is_w ? 0
               : nstl::max(0, static_cast<int>(div_up(i + P - O * S + 1, D)));

    while (k_s % S != s)
        k_s++;
}

void get_iw_range(const jit_brgemm_conv_conf_t &jcp, int iw, int iw_raw, int kw,
        int &iw_s, int &M_without_overflow) {
    // This function is needed for exec_base only
    using namespace dnnl::impl::utils;
    using namespace nstl;

    const auto SW = jcp.stride_w;
    const auto LP = jcp.l_pad;
    const auto DW = jcp.dilate_w + 1;
    const auto IW = jcp.iw;
    const auto OW = jcp.ow;
    const auto IW_BLOCK = jcp.iw_block;
    const auto IW_TAIL = jcp.iw_block;
    const bool is_iw_tail = (IW - iw_raw < IW_BLOCK);
    const auto M = div_up(is_iw_tail ? IW_TAIL : IW_BLOCK, SW);

    auto ow = (iw - kw * DW + LP) / SW;
    auto ow_l_ovf = ow;
    const auto ow_r_ovf = ow_l_ovf + (M - 1) - OW + 1;
    iw_s = iw;

    int ker_idx = 0;
    if (ow_l_ovf < 0) {
        ow_l_ovf = nstl::abs(ow_l_ovf);
        ker_idx += ow_l_ovf;
        iw_s += ker_idx;
    }

    if (ow_r_ovf > 0) ker_idx += ow_r_ovf;

    int iw_f = iw_s + (M - ker_idx);

    iw_s = nstl::min(iw_s, iw + M);
    iw_f = nstl::min(nstl::max(iw_f, iw_s), iw + M);

    M_without_overflow = iw_f - iw_s;
    iw_s = iw;
    ow = (iw_s - kw * DW + LP) / SW;
    while (ow < 0 && iw_s + SW < IW) {
        iw_s += SW;
        ow = (iw_s - kw * DW + LP) / SW;
    }
}

void get_kw_range(const jit_brgemm_conv_conf_t &jcp, int iw, int iw_raw,
        int &kw_s, int &kw_full_s, int &kw_full_f, int &kw_f) {
    // This function is needed for exec_base only
    using namespace dnnl::impl::utils;

    const auto SW = jcp.stride_w;
    const auto KW = jcp.kw;
    const auto LP = jcp.l_pad;
    const auto DW = jcp.dilate_w + 1;
    const auto IW = jcp.iw;
    const auto IW_BLOCK = jcp.iw_block;
    const auto IW_TAIL = jcp.iw_tail;
    const bool is_iw_tail = (IW - iw_raw < IW_BLOCK);
    const auto M = div_up(is_iw_tail ? IW_TAIL : IW_BLOCK, SW);
    kw_s = kw_full_s = kw_full_f = kw_f = -1;
    for (int kw = 0; kw < KW; kw++) {
        int iw_s {0}, iw_f {0}, M_without_overflow {0};
        get_iw_range(jcp, iw, iw_raw, kw, iw_s, M_without_overflow);
        iw_f = iw_s + M_without_overflow;
        if (iw_s < iw_f) {
            if (kw_s == -1) kw_s = kw;
            kw_f = kw + 1;
            if (iw_f - iw_s == M) {
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

    int s(0), o_test(0);
    while (true) {
        o_test = iw + LP - s * DW;
        if (o_test % SW == 0) break;
        s++;
    }
    while (kw_s % SW != s)
        kw_s++;
    if (kw_full_s != -1) {
        while (kw_full_s % SW != s)
            kw_full_s++;
    }
}

dim_t precalculate_comp_pad_kernels(const jit_brgemm_conv_conf_t &jcp,
        std::vector<dim_t> *kd_bs, std::vector<dim_t> *kd_es,
        std::vector<dim_t> *kh_bs, std::vector<dim_t> *kh_es,
        std::vector<dim_t> *kw_bs, std::vector<dim_t> *kw_es) {
    using namespace dnnl::impl::utils;
    using namespace nstl;

#define ndims_pick(v5, v4, v3) \
    ((ndims == 5) ? (v5) : (ndims == 4) ? (v4) : (ndims == 3) ? (v3) : 0)
    const auto ndims = jcp.ndims;
    const auto KD = jcp.kd;
    const auto KH = jcp.kh;
    const auto KW = jcp.kw;
    const auto KW_BLOCK = jcp.kw_block;
    const auto ID = ndims_pick(jcp.id, 1, 1);
    const auto ID_BLOCK = jcp.id_block;
    const auto IH = ndims_pick(jcp.ih, jcp.ih, 1);
    const auto IH_BLOCK = jcp.ih_block;
    const auto IW_BLOCK = jcp.iw_block;
    const auto OD = ndims_pick(jcp.od, 1, 1);
    const auto OH = ndims_pick(jcp.oh, jcp.oh, 1);
    const auto SD = ndims_pick(jcp.stride_d, 1, 1);
    const auto SH = ndims_pick(jcp.stride_h, jcp.stride_h, 1);
    const auto SW = jcp.stride_w;
    const auto FP = ndims_pick(jcp.f_pad, 0, 0);
    const auto TP = ndims_pick(jcp.t_pad, jcp.t_pad, 0);
    const auto DD = ndims_pick(jcp.dilate_d, 0, 0) + 1;
    const auto DH = ndims_pick(jcp.dilate_h, jcp.dilate_h, 0) + 1;

    const bool fill_k_ranges
            = !any_null(kd_bs, kd_es, kh_bs, kh_es, kw_bs, kw_es);
    std::set<std::vector<int>> unique_kernels;
    dim_t k = 0;
    if (fill_k_ranges) {
        kd_bs->resize(jcp.ker_ranges_size);
        kd_es->resize(jcp.ker_ranges_size);
        kh_bs->resize(jcp.ker_ranges_size);
        kh_es->resize(jcp.ker_ranges_size);
        kw_bs->resize(jcp.ker_ranges_size);
        kw_es->resize(jcp.ker_ranges_size);
    }

    const auto update_kernels
            = [&](int kd_b, int kd_e, int kh_b, int kh_e, int kw_b, int kw_e) {
                  unique_kernels.insert({kd_b, kd_e, kh_b, kh_e, kw_b, kw_e});
                  if (k == static_cast<dim_t>(unique_kernels.size())) return;
                  if (fill_k_ranges) {
                      (*kd_bs)[k] = kd_b;
                      (*kd_es)[k] = kd_e;
                      (*kh_bs)[k] = kh_b;
                      (*kh_es)[k] = kh_e;
                      (*kw_bs)[k] = kw_b;
                      (*kw_es)[k] = kw_e;
                  }
                  k++;
                  assert(IMPLICATION(fill_k_ranges, k <= jcp.ker_ranges_size));
              };

    for_(int idb = 0; idb < jcp.nb_id; idb++)
    for_(int ihb = 0; ihb < jcp.nb_ih; ihb++)
    for (int iwb = 0; iwb < jcp.nb_iw; iwb++) {
        auto id_begin = idb * ID_BLOCK;
        auto id_end = nstl::min(ID, id_begin + ID_BLOCK);
        auto ih_begin = ihb * IH_BLOCK;
        auto ih_end = jcp.is_os_blocking ? ih_begin + 1
                                         : nstl::min(IH, ih_begin + IH_BLOCK);
        for_(int id = id_begin; id < id_end; id++)
        for_(int ih = ih_begin; ih < ih_end; ih++)
        for (int sw = 0; sw < SW; sw++) {
            const int iw = iwb * IW_BLOCK + sw;
            const int iw_raw = iwb * IW_BLOCK;

            int kw_s {0}, kw_full_s {0}, kw_f {0}, kw_full_f {0};
            int kd_s_(0), kh_s_(0), kd_f_(0), kh_f_(0);
            get_kw_range(jcp, iw, iw_raw, kw_s, kw_full_s, kw_full_f, kw_f);

            set_k_range(FP, DD, SD, id, OD, KD, kd_s_, kd_f_);
            set_k_range(TP, DH, SH, ih, OH, KH, kh_s_, kh_f_);
            const auto kh_f = ndims_pick(kh_f_, kh_f_, 1);
            const auto kh_s = ndims_pick(kh_s_, kh_s_, 0);

            const auto kd_f = ndims_pick(kd_f_, 1, 1);
            const auto kd_s = ndims_pick(kd_s_, 0, 0);

            if (kd_f > kd_s && kh_f > kh_s && kw_f > kw_s && kw_s < KW) {
                if (jcp.exec_type == exec_base) {
                    if (kw_s < kw_full_s) {
                        for (auto kw = kw_s; kw < kw_full_s; kw += SW) {
                            update_kernels(kd_s, kd_f, kh_s, kh_f, kw, kw + 1);
                        }
                    }
                    if (kw_full_s < kw_full_f) {
                        for (auto kw = kw_full_s; kw < kw_full_f;
                                kw += KW_BLOCK) {
                            const auto kw_e
                                    = nstl::min(kw_full_f, kw + KW_BLOCK);
                            update_kernels(kd_s, kd_f, kh_s, kh_f, kw, kw_e);
                        }
                    }
                    if (kw_full_f < kw_f) {
                        for (auto kw = kw_full_f; kw < kw_f; kw += SW) {
                            update_kernels(kd_s, kd_f, kh_s, kh_f, kw, kw + 1);
                        }
                    }
                } else
                    assert(!"Unsupported exec type.");
            }
        }
    }
    return k;
#undef ndims_pick
}

status_t init_conf(jit_brgemm_conv_conf_t &jcp, cpu_isa_t isa,
        const convolution_desc_t &cd, memory_desc_t &diff_dst_md,
        memory_desc_t &weights_md, memory_desc_t &diff_src_md,
        memory_desc_t &bias_md, primitive_attr_t &attr, int nthreads,
        bool is_deconv) {

    using namespace prop_kind;

    if (!mayiuse(isa)) return status::unimplemented;

    CHECK(init_jcp(jcp, isa, cd, diff_dst_md, weights_md, diff_src_md, bias_md,
            attr, nthreads, is_deconv));

    const memory_desc_wrapper diff_dst_d(&diff_dst_md);
    const memory_desc_wrapper weights_d(&weights_md);
    const memory_desc_wrapper diff_src_d(&diff_src_md);
    const memory_desc_wrapper bias_d(&bias_md);

    jcp.l_ovf = nstl::max(0, jcp.ext_kw - 1 - jcp.l_pad) / jcp.stride_w;
    jcp.r_ovf = nstl::max(0, jcp.ext_kw - 1 - jcp.r_pad) / jcp.stride_w;
    jcp.t_ovf = nstl::max(0, jcp.ext_kh - 1 - jcp.t_pad) / jcp.stride_h;
    jcp.b_ovf = nstl::max(0, jcp.ext_kh - 1 - jcp.b_pad) / jcp.stride_h;
    jcp.f_ovf = nstl::max(0, jcp.ext_kd - 1 - jcp.f_pad) / jcp.stride_d;
    jcp.back_ovf = nstl::max(0, jcp.kd - 1 - jcp.back_pad) / jcp.stride_d;

    jcp.odp = jcp.od + jcp.f_ovf + jcp.back_ovf;
    jcp.ohp = jcp.oh + jcp.t_ovf + jcp.b_ovf;
    jcp.owp = jcp.ow + jcp.l_ovf + jcp.r_ovf;

    using namespace data_type;
    // ======================= blocking =================================

    const int min_ic_block = jcp.acc_simd_w;
    int selected_ur = 0;

    //-----------------------------------------------------------------------

    jcp.exec_type = is_amx(isa) ? exec_trans : exec_base;
    jcp.brg_type = brgemm_addr; // TODO: Choose right type of BRGEMM

    // TODO: in future use (kd/kh/kw) and (kd/kh/kw)_pad blocks for more
    // precise calculation of jcp.max_batch
    jcp.max_batch = jcp.kd * jcp.kh * jcp.kw;

    jcp.wei_plain = false;

    auto bcast_amount
            = static_cast<size_t>(jcp.od) * jcp.oh * jcp.ow * jcp.src_dsz;
    auto wei_amount = static_cast<size_t>(jcp.ic) * jcp.kd * jcp.kh * jcp.kw
            * jcp.wei_dsz;

    // use loop_ndhwgc always for exec_trans
    jcp.loop_order = (bcast_amount < wei_amount && jcp.exec_type != exec_trans)
            ? loop_ngcdhw
            : loop_ndhwgc;

    jcp.copy_block_only = true;

    const auto oc_padded_block
            = jcp.acc_simd_w * brg_blocking_t::last_oc_block_size;
    jcp.is_oc_padded = one_of(jcp.wei_dt, bf16, f16, s8)
            && jcp.oc * jcp.kw_sets > oc_padded_block && is_amx(isa);

    if (is_amx(isa) && (/* heuristic */ jcp.kw_sets == 1 && jcp.iw < 256)) {
        jcp.use_M_mask = 0;

        jcp.hint_prefetching = brgemm_kernel_prefetching_t::brgemm_prf1;

        // assuming 2x2 decomposition in amx brgemm kernel
        // and overlap of input by kw
        const auto bd_blocking = 2 * jcp.amx_h;
        const auto ld_blocking = 2 * 16;
        const auto A_ds = jcp.src_dsz * bd_blocking * jcp.oc * jcp.kd * jcp.kh;
        const auto B_ds
                = jcp.wei_dsz * ld_blocking * jcp.oc * jcp.kd * jcp.kh * jcp.kw;
        const auto C_ds = jcp.acc_dsz * bd_blocking * ld_blocking;
        if (A_ds + B_ds + C_ds > brg_blocking_t::L1)
            jcp.amx_tile_load_xx = true;
    }

    auto try_exec_type = [&]() {
        brg_blocking_t best_brgb = zero<decltype(best_brgb)>();
        best_brgb.ic_block = min_ic_block;
        brg_blocking_t cur_brgb = zero<decltype(best_brgb)>();
        cur_brgb.get_from_jcp(jcp);
        const auto start_icb = nstl::min(div_up(jcp.ic, jcp.acc_simd_w), 4);

        auto finish_icb = 1;
        for (auto icb = start_icb; icb >= finish_icb; icb--) {
            cur_brgb.ic_block = icb * jcp.acc_simd_w;
            cur_brgb.nb_ic = utils::div_up(jcp.ic, cur_brgb.ic_block);
            if (!cur_brgb.fast_check_ic_block()) continue;

            const status_t blocking_ok = cur_brgb.calc_blocks();
            if (blocking_ok != status::success) continue;

            const status_t st = cur_brgb.get_brgemm_ur(&attr, diff_src_md);
            if (st != status::success) continue;
            cur_brgb.eff = cur_brgb.est_eff();
            if (cur_brgb.eff > best_brgb.eff) best_brgb = cur_brgb;
        }
        if (best_brgb.oc_block == 0 || best_brgb.ic_block == 0
                || best_brgb.iw_block == 0)
            return false;
        best_brgb.save_to_jcp(jcp);
        selected_ur = best_brgb.ur;
        return true;
    };

    if (!try_exec_type()) return status::unimplemented;

    // ============ end blocking ===========================================
    jcp.max_vpad = 0;

    if (jcp.iw_block == 0 || jcp.oc_block == 0 || jcp.ic_block == 0)
        return status::unimplemented;

    jcp.gemm_batch_size = jcp.nb_oc_blocking
            * nstl::max(jcp.kd_block * jcp.kh_block * jcp.kw_block,
                    jcp.kd_block_pad * jcp.kh_block_pad * jcp.kw_block_pad);
    // to avoid cache concurrent write access from different threads
    size_t sc_size = sizeof(brgemm_batch_element_t);
    jcp.adjusted_batch_size
            = div_up(rnd_up(jcp.gemm_batch_size * sc_size, P4K), sc_size);

    CHECK(pick_tags(jcp, diff_dst_md, weights_md, diff_src_md, bias_md));
    CHECK(attr.set_default_formats(&diff_src_md));

    jcp.buffer_size = jcp.LDC * (jcp.M > 0 ? jcp.M : jcp.M_tail);

    jcp.nb_id = div_up(jcp.id, jcp.id_block);
    jcp.nb_ih = div_up(jcp.ih, jcp.ih_block);

    jcp.inp_buffer_size = rnd_up(jcp.odp * jcp.ohp * jcp.owp * jcp.ngroups
                    * jcp.nb_oc * jcp.oc_block,
            P4K);
    jcp.inp_buffer_mask_size = rnd_up(static_cast<dim_t>(jcp.nb_id) * jcp.nb_ih
                    * jcp.nb_iw * jcp.ngroups * jcp.nb_oc,
            P4K);

    // Calculate the comp along with the computation inside brgemm kernel when
    // output size is small to get optimal perf
    // Otherwise we calculate the comp using brgemm_comp_pad kernel
    const auto output_sz = static_cast<dim_t>(jcp.mb) * jcp.ngroups * jcp.ic
            * jcp.id * jcp.ih * jcp.iw;
    const auto need_compensation
            = jcp.src_zero_point || jcp.s8s8_compensation_required;
    jcp.req_brg_comp_pad
            = need_compensation && output_sz <= 8192 && jcp.ic < 512;
    jcp.req_cal_comp_pad = need_compensation && !jcp.req_brg_comp_pad;

    if (jcp.req_cal_comp_pad) {
        jcp.ker_ranges_size = precalculate_comp_pad_kernels(jcp);
        jcp.comp_a_buffer_size = static_cast<dim_t>(jcp.ngroups) * jcp.nb_ic
                * jcp.ker_ranges_size * jcp.ic_block;
        jcp.s8s8_comp_buffer_size = jcp.comp_a_buffer_size;
    }

    return status::success;
}

void init_scratchpad(memory_tracking::registrar_t &scratchpad,
        const jit_brgemm_conv_conf_t &jcp) {
    if (jcp.brg_type == brgemm_addr || jcp.brg_type == brgemm_offs
            || (jcp.brg_type == brgemm_strd && jcp.exec_type == exec_vpad))
        scratchpad.book(key_brgemm_primitive_batch,
                static_cast<size_t>(jcp.nthr) * jcp.adjusted_batch_size,
                sizeof(brgemm_batch_element_t), 64, P4K);

    size_t inp_buffer_size
            = static_cast<size_t>(jcp.nthr) * jcp.inp_buffer_size;
    scratchpad.book(
            key_conv_brgemm_inp_buffer, inp_buffer_size, jcp.src_dsz, 0, P4K);
    size_t inp_buffer_mask_size
            = static_cast<size_t>(jcp.nthr) * jcp.inp_buffer_mask_size;
    scratchpad.book(key_conv_brgemm_inp_buffer_mask, inp_buffer_mask_size,
            sizeof(uint8_t), 0, P4K);

    if (jcp.use_buffer) {
        scratchpad.book(key_brgemm_primitive_buffer, jcp.nthr * jcp.buffer_size,
                jcp.acc_dsz, 0, P4K);
    }
    if (is_amx(jcp.isa)) {
        scratchpad.book(key_conv_amx_tile_buffer, jcp.nthr * 2 * P4K,
                sizeof(char), 0, P4K);
    }
    if (jcp.s8s8_compensation_required && jcp.req_cal_comp_pad) {
        scratchpad.book(key_brgemm_primitive_buffer_comp,
                jcp.s8s8_comp_buffer_size, sizeof(int32_t), 0, P4K);
    }
    if (jcp.src_zero_point && jcp.req_cal_comp_pad && !is_amx(jcp.isa)) {
        scratchpad.book(key_brgemm_primitive_zp_comp_a, jcp.comp_a_buffer_size,
                sizeof(int32_t), 0, P4K);
    }
}

} // namespace brgemm_convolution_bwd_utils

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
