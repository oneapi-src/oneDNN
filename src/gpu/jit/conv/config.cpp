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

#include "gpu/jit/conv/config.hpp"

#include <cctype>
#include <cstring>

#include "common/type_helpers.hpp"
#include "gpu/jit/conv/block_helper.hpp"
#include "gpu/jit/conv/config_lookup_table.hpp"
#include "gpu/jit/conv/grf_usage.hpp"
#include "gpu/jit/conv/normalization.hpp"
#include "gpu/jit/ir/block_2d_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

std::string conv_problem_t::desc_str(bool print_mb) const {
    std::ostringstream oss;
    if (print_mb) oss << "mb" << mb;
    if (g > 1) oss << "g" << g;
    oss << "ic" << ic;

    std::vector<int> xd = {id, od, kd, sd, dd, pd};
    std::vector<int> xh = {ih, oh, kh, sh, dh, ph};
    std::vector<int> xw = {iw, ow, kw, sw, dw, pw};
    std::vector<int> xdef = {1, 1, 1, 1, 0, 0};
    bool has_d = !ir_utils::is_equal(xd, xdef);
    bool has_h = !ir_utils::is_equal(xh, xdef);
    bool is_square = ir_utils::is_equal(xh, xw);
    bool is_cubic = is_square && ir_utils::is_equal(xd, xh);
    bool print_d = has_d;
    bool print_h = has_h && !is_cubic;
    bool print_w = !is_cubic && !is_square;

    if (print_d) oss << "id" << id;
    if (print_h) oss << "ih" << ih;
    if (print_w) oss << "iw" << iw;
    oss << "oc" << oc;
    if (print_d) oss << "od" << od;
    if (print_h) oss << "oh" << oh;
    if (print_w) oss << "ow" << ow;
    if (print_d) oss << "kd" << kd;
    if (print_h) oss << "kh" << kh;
    if (print_w) oss << "kw" << kw;
    if (print_d && sd != 1) oss << "sd" << sd;
    if (print_h && sh != 1) oss << "sh" << sh;
    if (print_w && sw != 1) oss << "sw" << sw;
    if (print_d && dd != 0) oss << "dd" << dd;
    if (print_h && dh != 0) oss << "dh" << dh;
    if (print_w && dw != 0) oss << "dw" << dw;
    if (print_d) oss << "pd" << pd;
    if (print_h) oss << "ph" << ph;
    if (print_w) oss << "pw" << pw;
    return oss.str();
}

status_t conv_config_t::init_common_blocking() {
    if (is_fwd && is_small_ic()) hw_cfg.set_max_tg_size(16);

    bh = std::make_shared<block_helper_t>();
    bh->set_hw_config(hw_cfg);
    bh->set_fma_kind(fma_kind);
    bh->set_abc_types(a_data_type, b_data_type, acc_data_type);

    bh->set_dim("mb", mb);
    bh->set_dim("g", g);
    bh->set_dim("oc", oc);
    //take into account blocked ic channels when selecting block sizes
    bh->set_dim("ic",
            is_bwd_w ? std::max(src_layout.dims()[2], wei_layout.dims()[2])
                     : ic);
    bh->set_dims({"kd", "kh", "kw"}, {kd, kh, kw});

    bh->set_b_dims({"g"});

    if (is_fwd) {
        if (fuse_spatial) {
            bh->set_dims({"osp"}, {osp});
            bh->set_m_dims({"mb", "osp"});
        } else {
            bh->set_dims({"od", "oh", "ow"}, {od, oh, ow});
            bh->set_m_dims({"mb", "od", "oh", "ow"});
        }
        bh->set_n_dims({"oc"});
        bh->set_k_dims({"ic", "kd", "kh", "kw"});
    } else if (is_bwd_d) {
        ir_assert(!fuse_spatial);
        bh->set_dims({"id", "ih", "iw"}, {id, ih, iw});
        bh->set_m_dims({"mb", "id", "ih", "iw"});
        bh->set_n_dims({"ic"});
        bh->set_k_dims({"oc", "kd", "kh", "kw"});
    } else if (is_bwd_w) {
        ir_assert(!fuse_spatial);
        bh->set_dims({"od", "oh", "ow"}, {od, oh, ow});
        bh->set_m_dims({"ic", "kd", "kh", "kw"});
        bh->set_n_dims({"oc"});
        bh->set_k_dims({"mb", "od", "oh", "ow"});
    } else {
        ir_error_not_expected();
    }

    // Set blocks for padding. This is to comply with zero-padding
    // requirements. For example if the output layout is nChw32c but there are
    // only 8 channels to compute and store, we still need to pad 8 to 32 and
    // spawn more thread groups to ensure 32c block is properly zero-padded.
    if (is_fwd) {
        bh->set_pad_block("mb", dst_layout.inner_block(0));
        bh->set_pad_block("g", dst_layout.inner_block(1));
        bh->set_pad_block("oc", dst_layout.inner_block(2));
    } else if (is_bwd_d) {
        bh->set_pad_block("mb", src_layout.inner_block(0));
        bh->set_pad_block("g", src_layout.inner_block(1));
        bh->set_pad_block("ic", src_layout.inner_block(2));
    } else if (is_bwd_w) {
        bh->set_pad_block("g", wei_layout.inner_block(0));
        bh->set_pad_block("oc", wei_layout.inner_block(1));
        bh->set_pad_block("ic", wei_layout.inner_block(2));
    }

    // Set base blocks to align kernel blocking with layout blocking.
    if (is_fwd) {
        bh->set_base_iter_block("mb", src_layout.inner_block(0));
        int src_g_blk = src_layout.inner_block(1);
        int wei_g_blk = wei_layout.inner_block(0);
        bh->set_base_iter_block("g", src_g_blk, wei_g_blk);
        int src_ic_blk = src_layout.inner_block(2);
        int wei_ic_blk = wei_layout.inner_block(2);
        bh->set_base_iter_block("ic", src_ic_blk, wei_ic_blk);
    } else if (is_bwd_d) {
        bh->set_base_iter_block("mb", dst_layout.inner_block(0));
        int dst_g_blk = dst_layout.inner_block(1);
        int wei_g_blk = wei_layout.inner_block(0);
        bh->set_base_iter_block("g", dst_g_blk, wei_g_blk);
        int dst_oc_blk = dst_layout.inner_block(2);
        int wei_oc_blk = wei_layout.inner_block(1);
        bh->set_base_iter_block("oc", dst_oc_blk, wei_oc_blk);
    } else if (is_bwd_w) {
        bh->set_base_iter_block("g", wei_layout.inner_block(0));
        int wei_oc_blk = wei_layout.inner_block(2);
        int dst_oc_blk = dst_layout.inner_block(2);
        bh->set_base_iter_block("oc", wei_oc_blk, dst_oc_blk);
        int src_ic_blk = src_layout.inner_block(2);
        int wei_ic_blk = wei_layout.inner_block(2);
        bh->set_base_iter_block("ic", src_ic_blk, wei_ic_blk);
        int src_mb_blk = src_layout.inner_block(0);
        int dst_mb_blk = dst_layout.inner_block(0);
        bh->set_base_iter_block("mb", src_mb_blk, dst_mb_blk);
    }

    return status::success;
}

status_t conv_config_t::init_fwd(convolution_pd_t *conv_pd) {
    using namespace ir_utils;

    init_use_ow_kw_grf_cache();

    const char *osp_name = fuse_spatial ? "osp" : "ow";

    if (use_ow_kw_grf_cache) {
        bh->set_max_iter_dim("mb", 1);
        bh->set_max_m_tg_dim(2);
        bh->set_max_n_tg_dim(2);
    }

    bh->set_thr_dim("kd", kd);
    bh->set_thr_dim("kh", kh);
    if (is_small_ic() && !is_dw_large_mb()) {
        bh->set_block_dims({"kw"});
    } else {
        bh->set_thr_dim("kw", kw);
        // mad is not tested with thread group k-slicing.
        if (is_dp_fma()) {
            bh->allow_k_tg_slicing();
            bh->set_max_k_tg_dim(8);
        }
    }

    bh->set_block_dims({"g", "oc", "ic", "mb", osp_name});
    bh->set_vector_dim(is_dw ? "g" : "oc");
    bh->allow_fuse({"ic", "kw"});
    bh->allow_split({"oc", "ic", "kw"});

    int mb_base_iter_blk = bh->dim("mb").base_iter_block();
    // mb blocking is always outer so we can safely use a smaller divisor to
    // have more flexible blocking for some cases.
    int mb_base_iter_divisor = is_dw_large_mb() ? 32 : 8;
    mb_base_iter_blk = math::gcd(mb_base_iter_divisor, mb_base_iter_blk);

    bh->set_base_iter_block("mb", mb_base_iter_blk);

    bool use_sp_blocking = false;
    if (is_compute_nhwc("src")) {
        use_sp_blocking = should_use_spatial_blocking(od, oh, ow);
    } else if (src_layout.inner_block(0) == 1) {
        use_sp_blocking = true;
    } else if (is_dw && !is_dw_large_mb()) {
        use_sp_blocking = true;
    }

    if (use_sp_blocking) {
        if (is_dw) bh->set_pref_tg_block(osp_name);
        bh->allow_split({osp_name, "mb"});
        bh->reorder({osp_name, "mb"});
        if (!is_int8_dst() && !fuse_spatial && mb < 16 && iw % 8 != 0
                && !is_dw) {
            bh->set_max_m_tg_dim(1);
        }
    } else {
        const int large_sp_threshold = is_ge_xe_hpc() ? 128 : 256;
        if (!is_dw && ow > large_sp_threshold)
            bh->set_pref_tg_block("oc");
        else if (is_dp_fma() && mb >= 16)
            bh->set_pref_tg_block(osp_name);
        bh->reorder({"mb", osp_name});
        auto spatial_dim = fuse_spatial ? osp : ow;
        if (!use_2d_send_nhwc && mb >= 128
                && (spatial_dim % 4 != 0 || spatial_dim < 64))
            bh->allow_split({"mb"});
    }

    if (mb < 8 && !bh->any_pref_tg_block())
        bh->set_pref_tg_block(ow > oc ? osp_name : "oc");

    bh->reorder({"ic", "kw"});

    if (use_2d_send_nhwc) {
        int src_type_size = (int)types::data_type_size(src_data_type);
        // Use 64-byte reduction step to avoid partial cache line loads.
        bh->set_base_iter_block("ic", 64 / src_type_size);
        bh->set_reduce_m_block_hint(false);
    }

    bh->init_blocks();
    maybe_override_from_lookup_table();
    bh->finalize();

    g_tg_blk = bh->tg_blk("g");
    g_thr_blk = bh->thr_blk("g");
    ic_blk = bh->iter_blk("ic");
    ic_thr_blk = bh->thr_blk("ic");
    kw_blk = bh->iter_blk("kw");
    mb_tg_blk = bh->tg_blk("mb");
    mb_thr_blk = bh->thr_blk("mb");
    oc_tg_blk = bh->tg_blk("oc");
    oc_thr_blk = bh->thr_blk("oc");
    osp_tg_blk = bh->tg_blk(osp_name);
    osp_thr_blk = bh->thr_blk(osp_name);

    mb_thr_dim = ir_utils::safe_divide(mb_tg_blk, mb_thr_blk);
    osp_thr_dim = ir_utils::safe_divide(osp_tg_blk, osp_thr_blk);
    oc_thr_dim = ir_utils::safe_divide(oc_tg_blk, oc_thr_blk);

    ic_thr_dim = bh->tg_dim("ic");

    tg_grid_dim[0] = oc_thr_dim;
    tg_grid_dim[1] = mb_thr_dim * osp_thr_dim;
    tg_grid_dim[2] = ic_thr_dim;

    b_blk = g_tg_blk;
    m_tg_blk = mb_tg_blk * osp_tg_blk;
    n_tg_blk = oc_tg_blk;
    k_blk = ic_blk * kw_blk;

    int g_grid_dim = bh->grid_dim("g");
    int mb_grid_dim = bh->grid_dim("mb");
    int oc_grid_dim = bh->grid_dim("oc");
    int osp_grid_dim = fuse_spatial ? bh->grid_dim(osp_name)
                                    : od * oh * bh->grid_dim("ow");

    kernel_grid_dim[0] = oc_grid_dim;

    kernel_grid_dim[1] = g_grid_dim * osp_grid_dim;
    kernel_grid_dim[2] = mb_grid_dim;

    if (use_2d_send_nhwc && sw != 1 && (kw != 1 || pw != 0)) {
        ir_assert(osp_thr_blk == 1)
                << "Can't use 2D block messages for non-trivial "
                   "strided dimensions.";
    }

    CHECK(init_zero_points_config(conv_pd));

    const int max_unroll = 9;
    if (kd * kh * kw > max_unroll) do_pipeline_unroll = false;
    if (is_small_ic()) {
        reuse_headers = true;
        do_pipeline_unroll = false;
    }

#ifdef GEN_CONV_DEBUG
    do_pipeline_unroll = getenv_bool("do_pipeline_unroll", do_pipeline_unroll);
    reuse_headers = getenv_bool("reuse_headers", reuse_headers);
    use_preload = getenv_bool("use_preload", use_preload);
#endif

    return status::success;
}

status_t conv_config_t::init_bwd_d(convolution_pd_t *conv_pd) {
    using namespace ir_utils;

    bh->set_thr_dim("kw", kw);
    bh->set_thr_dim("kd", kd);
    bh->set_thr_dim("kh", kh);
    bh->set_block_dims({"g", "oc", "ic", "mb", "iw"});
    bh->set_vector_dim(is_dw ? "g" : "ic");
    bh->allow_split({"ic"});

    bool use_w_blocking = false;
    if (is_compute_nhwc("dst")) {
        use_w_blocking = should_use_spatial_blocking(id, ih, iw);
    } else if (dst_layout.inner_block(0) == 1) {
        use_w_blocking = true;
    }

    if (use_w_blocking) {
        bh->allow_fuse({"iw", "mb"});
        bh->allow_split({"iw", "mb"});
        bh->reorder({"iw", "mb"});
    } else {
        bh->reorder({"mb", "iw"});
        bh->set_base_iter_block("mb", 8);
    }

    if (use_2d_send_nhwc) {
        int dst_type_size = (int)types::data_type_size(dst_data_type);
        bh->set_base_iter_block("oc", 64 / dst_type_size);
        if (!is_stride1()) bh->allow_split({"mb"});
        bh->set_reduce_m_block_hint(false);
    }

    bh->init_blocks();
    maybe_override_from_lookup_table();
    bh->finalize();

    // Try to enable special optimization for strided BWD_D convolution.
    init_bwd_d_optimize_strided(bh->thr_blk("iw"));

    if (bwd_d_optimize_strided_iw) {
        int iw_tg_dim0 = bh->tg_dim("iw");
        ir_assert(math::is_pow2(iw_tg_dim0));
        ir_assert(iw % sw == 0);
        for (int tg_dim = iw_tg_dim0; tg_dim >= 1; tg_dim /= 2) {
            if ((iw / sw) % tg_dim == 0) {
                bh->set_tg_dim("iw", tg_dim);
                int mb_iter_dim = bh->iter_dim("mb");
                int new_mb_tg_dim = bh->tg_dim("mb") * iw_tg_dim0 / tg_dim;
                // TODO: non-uniform thread group is unsupported
                while (new_mb_tg_dim > 1
                        && utils::rnd_up(mb, mb_iter_dim * new_mb_tg_dim) - mb
                                >= mb_iter_dim) {
                    new_mb_tg_dim /= 2;
                }
                if (mb_iter_dim * new_mb_tg_dim <= mb) {
                    bh->set_tg_dim("mb", new_mb_tg_dim);
                }
                break;
            }
        }
    }

    g_tg_blk = bh->tg_blk("g");
    g_thr_blk = bh->thr_blk("g");
    mb_tg_blk = bh->tg_blk("mb");
    mb_thr_blk = bh->thr_blk("mb");
    ic_tg_blk = bh->tg_blk("ic");
    ic_thr_blk = bh->thr_blk("ic");
    iw_tg_blk = bh->tg_blk("iw");
    iw_thr_blk = bh->thr_blk("iw");
    oc_blk = bh->iter_dim("oc");

    mb_thr_dim = ir_utils::safe_divide(mb_tg_blk, mb_thr_blk);
    iw_thr_dim = ir_utils::safe_divide(iw_tg_blk, iw_thr_blk);
    ic_thr_dim = ir_utils::safe_divide(ic_tg_blk, ic_thr_blk);

    tg_grid_dim[0] = ic_thr_dim;
    tg_grid_dim[1] = mb_thr_dim * iw_thr_dim;
    tg_grid_dim[2] = 1;

    b_blk = g_tg_blk;
    m_tg_blk = mb_tg_blk * iw_tg_blk;
    n_tg_blk = ic_tg_blk;
    k_blk = oc_blk;

    int g_grid_dim = bh->grid_dim("g");
    int ic_grid_dim = bh->grid_dim("ic");
    int mb_grid_dim = bh->grid_dim("mb");
    int iw_grid_dim = bh->grid_dim("iw");

    kernel_grid_dim[0] = ic_grid_dim;
    kernel_grid_dim[1] = g_grid_dim * id * ih * iw_grid_dim;
    kernel_grid_dim[2] = mb_grid_dim;

    if (use_2d_send_nhwc && mb < 16 && sw != 1) {
        ir_assert(bh->iter_blk("iw") == 1)
                << "Can't use 2D block messages for non-trivial "
                   "strided dimensions.";
    }

    CHECK(init_zero_points_config(conv_pd));

    // Do not perform full unrolling when there are too many inner
    // iterations.
    int kernel_limit = is_f32_conv() ? 4 : 9;
    if (kd * kh * kw > kernel_limit) do_pipeline_unroll = false;

    // Do not perform full unrolling with non-unit stride unless special
    // stride optimization is enabled. These cases have non-trivial
    // post-increment updates which result in unrolling all reduction loops
    // and exceeding the instruction cache.
    if (!is_stride1() && !bwd_d_optimize_strided_iw) do_pipeline_unroll = false;

    return status::success;
}

status_t conv_config_t::init_bwd_w(convolution_pd_t *conv_pd) {
    using namespace ir_utils;

    bh->allow_k_grid_slicing();

    bh->set_block_dims({"g", "oc", "ic", "mb", "oh", "ow"});
    bh->set_vector_dim(is_dw ? "g" : "oc");

    if (oc <= 32) bh->set_max_iter_dim("oc", 16);
    if (ic <= 32) bh->set_max_iter_dim("ic", 16);

    if (is_small_ic() && !is_dw) {
        bh->set_block_dims({"kw"});
        bh->set_max_tg_dim("kw", 1);
        bh->set_max_iter_dim("kw", 8);
    }

    // Avoid 2D spatial blocking when possible (when 1D blocking can be
    // enough). Extra oh/od loops may result in assembly bloat due to pipeline
    // unroll.
    if (mb >= 32 && ow >= 16) {
        bh->set_max_thr_dim("oh", 1);
        bh->set_max_thr_dim("od", 1);
    }

    bh->set_max_iter_dim("oh", 1);

    bh->allow_split({"oc", "ic", "mb", "ow"});
    bh->allow_fuse({"ic", "kw"});
    bh->allow_fuse({"mb", "oh", "ow"});
    bh->set_max_thr_dim("mb", 2);
    bh->set_base_iter_block(
            "mb", math::gcd(16, bh->dim("mb").base_iter_block()));

    bh->reorder({"mb", "ow", "oh"});

    if (use_2d_send_nhwc) bh->set_reduce_m_block_hint(false);

    bh->init_blocks();
    maybe_override_from_lookup_table();
    bh->finalize();

    g_tg_blk = bh->tg_blk("g");
    g_thr_blk = bh->thr_blk("g");
    ic_thr_blk = bh->thr_blk("ic");
    ic_tg_blk = bh->tg_blk("ic");
    mb_blk = bh->iter_blk("mb");
    mb_thr_blk = bh->thr_blk("mb");
    oc_thr_blk = bh->thr_blk("oc");
    oc_tg_blk = bh->tg_blk("oc");
    od_thr_blk = bh->thr_blk("od");
    oh_thr_blk = bh->thr_blk("oh");
    ow_blk = bh->iter_blk("ow");
    ow_thr_blk = bh->thr_blk("ow");
    kw_blk = bh->iter_blk("kw");
    kw_tg_blk = bh->tg_blk("kw");

    ic_thr_dim = ir_utils::safe_divide(ic_tg_blk, ic_thr_blk);
    oc_thr_dim = ir_utils::safe_divide(oc_tg_blk, oc_thr_blk);

    init_allow_slm_tg_slicing(ic_thr_blk, oc_thr_blk, ic_thr_dim, oc_thr_dim);

    tg_grid_dim[0] = oc_thr_dim;
    tg_grid_dim[1] = ic_thr_dim;
    tg_grid_dim[2] = 1;

    int g_grid_dim = bh->grid_dim("g");
    int ic_grid_dim = bh->grid_dim("ic");
    int kw_grid_dim = bh->grid_dim("kw");
    int mb_grid_dim = bh->grid_dim("mb");
    int oc_grid_dim = bh->grid_dim("oc");
    int od_grid_dim = bh->grid_dim("od");
    int oh_grid_dim = bh->grid_dim("oh");
    int ow_grid_dim = bh->grid_dim("ow");

    kernel_grid_dim[0] = oc_grid_dim;
    kernel_grid_dim[1] = ic_grid_dim * kd * kh * kw_grid_dim * od_grid_dim
            * oh_grid_dim * ow_grid_dim;
    kernel_grid_dim[2] = g_grid_dim * mb_grid_dim;

    if (use_2d_send_nhwc && sw != 1 && (kw != 1 || pw != 0)) {
        ir_assert(ow_blk == 1) << "Can't use 2D block messages for non-trivial "
                                  "strided dimensions.";
    }

    mb_unroll = mb_thr_blk / mb_blk;
    ow_unroll = (ow_blk > 1 && is_dp_fma()) ? ow_thr_blk / ow_blk : 1;
    if (ow_unroll > 8) ow_unroll = 1;

    b_blk = g_tg_blk;
    m_tg_blk = ic_tg_blk * kw_tg_blk;
    n_tg_blk = oc_tg_blk;
    k_blk = mb_blk * ow_blk;

    // Set BWD_W-specific settings.
    do_b_reduction = with_bias;
    do_pipeline_unroll = (is_ge_xe_hpc() && is_dp_fma() && mb_blk > 1);
    do_atomic_update = true;

    if (!with_sum) {
        tensor_config.require_zero_out("wei");
        if (with_bias) tensor_config.require_zero_out("bia");
    }

    return status::success;
}

void conv_config_t::maybe_override_from_lookup_table() {
    static conv_config_lookup_table_t table;
#ifdef GEN_CONV_DEBUG
    auto env_cfg = ir_utils::getenv_str("cfg", "");
    if (!env_cfg.empty()) {
        conv_config_params_t params(env_cfg);
        params.apply(*this);
        return;
    }
#endif
    auto params = table.find(*this, hw_cfg);
    if (params.is_empty()) return;
    params.apply(*this);
}

static std::string build_tag(const std::vector<int> &inner_blocks,
        const std::vector<int> &outer_blocks, const std::vector<char> &letters,
        const std::vector<int> &idxs) {
    size_t n = letters.size();
    ir_assert(inner_blocks.size() == n);
    ir_assert(outer_blocks.size() == n);
    ir_assert(idxs.size() == n);

    std::string tag;
    std::vector<bool> seen(n);

    // Iterate through outer blocks.
    for (int i = (int)n - 1; i >= 0; i--) {
        int idx = idxs[i];
        int blk = outer_blocks[idx];
        if (blk == 1) continue;
        seen[idx] = true;
        tag += std::to_string(blk) + letters[idx];
    }

    // Iterate through inner blocks.
    for (int i = (int)n - 1; i >= 0; i--) {
        int idx = idxs[i];
        int blk = inner_blocks[idx];
        if (blk == 1) continue;
        seen[idx] = true;
        tag += std::to_string(blk) + letters[idx];
    }

    if (tag.empty()) {
        // Assume this is an activations tag, use NHWC by default.
        tag = "axb";
    } else {
        tag = 'x' + tag;
        for (int i = (int)n - 1; i >= 0; i--) {
            char c = letters[i];
            if (c == ' ') continue;
            if (seen[i]) c = std::toupper(c);
            tag = c + tag;
        }
    }

    return tag;
}

int pick_block_impl(bool prefer_rnd_up, int dim, int b0, int b1, int b2) {
    int blocks[3] = {b0, b1, b2};
    int prev_blk = 1;
    for (int i = 0; i < 3; i++) {
        if (blocks[i] == 0) continue;
        if (prefer_rnd_up) {
            if (dim <= blocks[i] / 2) return prev_blk;
        } else {
            if (dim < blocks[i]) return prev_blk;
        }
        prev_blk = blocks[i];
    }
    return prev_blk;
}

int pick_block_rnd_up(int dim, int b0, int b1 = 0, int b2 = 0) {
    return pick_block_impl(true, dim, b0, b1, b2);
}

int pick_block(int dim, int b0, int b1 = 0, int b2 = 0) {
    return pick_block_impl(false, dim, b0, b1, b2);
}

struct nc_block_t {
    nc_block_t(int n_block, int c_block, bool nc_order = true)
        : n_block_(n_block), c_block_(c_block), nc_order_(nc_order) {}

    std::string tag() const {
        std::vector<int> idxs = {1, 0};
        if (!nc_order_) std::swap(idxs[0], idxs[1]);
        return build_tag({n_block_, c_block_}, {1, 1}, {'a', 'b'}, idxs);
    }

    // Ideally, this should only depend on data type, direction, mb, c, and g to
    // enable the same src/dst formats and avoid reorders between convolutions
    static nc_block_t get_default_blocking(type_t type, bool is_dw, int n,
            int c, int g, bool is_input, bool is_small_c,
            int min_block_size = 0, bool nc_order = true,
            bool force_default_c_blk = false) {
        bool is_small_c_input
                = (type.size() <= 2 && is_input && !is_dw && is_small_c);
        auto default_c_blk = type.size() == 1 ? 32 : 16;
        auto c_block = [&]() {
            if (force_default_c_blk) return default_c_blk;
            // Special case for small input channel shapes with dpas.
            if (is_small_c_input) {
                int packed_dword_elems = 4 / type.size();
                return std::max(packed_dword_elems, utils::rnd_up_pow2(c));
            }
            auto blk_dim = is_dw ? g : c;
            return pick_block_rnd_up(blk_dim, default_c_blk);
        }();

        // Non-depthwise convolutions currently require channel is a multiple of
        // c_block. If that implementation restriction is removed, this logic
        // could be removed.
        if (g > 1 && !is_dw && c % c_block != 0) c_block = 1;

        auto default_n_blk = type.size() < 4 ? 32 : 16;
        auto n_block = [&]() {
            if (c_block == 1)
                return 1;
            else if (is_small_c_input)
                return pick_block(n, 8, 16);
            else
                return pick_block(n, 16, default_n_blk);
        }();

        // Require minimum block size, used to enable better message behavior
        while (n_block * c_block * type.size() < min_block_size) {
            // Prefer increasing blocks in dimensions with available data, and
            // otherwise just increase c_block to meet requirements. Limit
            // blocking dimensions to avoid untested edge cases.
            if (c_block < c && c_block < default_c_blk)
                c_block *= 2;
            else if (n_block < n && n_block < default_n_blk)
                n_block *= 2;
            else {
                c_block = utils::div_up(min_block_size, type.size() * n_block);
                if (c_block > default_c_blk) c_block = default_c_blk;
                break;
            }
        }

        return nc_block_t(n_block, c_block, nc_order);
    }

private:
    int n_block_;
    int c_block_;
    bool nc_order_;
};

struct goi_block_t {
    goi_block_t(fma_kind_t fma_kind, bool is_dw, bool is_bwd_d, int g_block,
            int o_block, int i_block, int o_block_outer, int i_block_outer)
        : fma_kind_(fma_kind)
        , is_dw_(is_dw)
        , is_bwd_d_(is_bwd_d)
        , g_block_(g_block)
        , o_block_(o_block)
        , i_block_(i_block)
        , o_block_outer_(o_block_outer)
        , i_block_outer_(i_block_outer) {}

    std::string tag() const {
        std::vector<char> wei_letters(3, ' ');
        char wei_letter = 'a';
        for (int i = (is_dw_ ? 0 : 1); i < 3; i++) {
            wei_letters[i] = wei_letter++;
        }
        std::vector<int> wei_idxs = {0, 1, 2}; // g, ic, oc
        // dpas requires ic to go before oc in innermost blocks for weights.
        if (fma_kind_ != fma_kind_t::mad) std::swap(wei_idxs[1], wei_idxs[2]);
        if (is_bwd_d_) std::swap(wei_idxs[1], wei_idxs[2]);
        return build_tag({g_block_, o_block_, i_block_},
                {1, o_block_outer_, i_block_outer_}, wei_letters, wei_idxs);
    }

    static goi_block_t get_default_blocking(type_t type, int vec_size,
            fma_kind_t fma_kind, bool is_bwd_d, bool is_small_ic, int g, int o,
            int i) {
        int x = o;
        int y = i;
        int g_block = 1;
        int o_block = 1;
        int i_block = 1;
        int o_block_outer = 1;
        int i_block_outer = 1;
        int *x_block = &o_block;
        int *y_block = &i_block;
        int *x_block_outer = &o_block_outer;
        int *y_block_outer = &i_block_outer;
        // Backward by data requires flipped ic/oc in weights.
        if (is_bwd_d) {
            std::swap(x, y);
            std::swap(x_block, y_block);
            std::swap(x_block_outer, y_block_outer);
        }
        get_default_blocking(type, vec_size, fma_kind, is_bwd_d, is_small_ic, g,
                x, y, g_block, *x_block, *y_block, *x_block_outer,
                *y_block_outer);
        return goi_block_t(fma_kind, is_dw(g, o, i), is_bwd_d, g_block, o_block,
                i_block, o_block_outer, i_block_outer);
    }

    static void get_default_blocking(type_t type, int vec_size,
            fma_kind_t fma_kind, bool is_bwd_d, bool is_small_ic, int g, int x,
            int y, int &g_block, int &x_block, int &y_block, int &x_block_outer,
            int &y_block_outer) {
        if (is_dw(g, x, y)) {
            g_block = type.is_x8() ? 32 : 16;
        } else if (fma_kind == fma_kind_t::mad) {
            x_block = vec_size;
            y_block = pick_block(y, 8, 16);
        } else {
            int packed_dword_elems = 4 / type.size();
            x_block = vec_size;
            y_block = packed_dword_elems;
            if (is_bwd_d || !is_small_ic) y_block_outer = 8;
        }
    }

private:
    static bool is_dw(int g, int o, int i) {
        return (g > 1 && o == 1 && i == 1);
    }

    fma_kind_t fma_kind_;
    bool is_dw_;
    bool is_bwd_d_;
    int g_block_;
    int o_block_;
    int i_block_;
    int o_block_outer_;
    int i_block_outer_;
};

void conv_config_t::init_data_tags(bool allow_src_reorder,
        bool allow_wei_reorder, bool allow_dst_reorder,
        const memory_desc_t &src_md, const memory_desc_t &wei_md,
        const memory_desc_t &dst_md, std::string &src_tag, std::string &wei_tag,
        std::string &dst_tag, std::string &user_wei_tag) {

    auto src_compute_type = is_bwd_d ? c_data_type : a_data_type;
    auto dst_compute_type
            = is_fwd ? c_data_type : (is_bwd_d ? a_data_type : b_data_type);
    auto wei_compute_type = is_bwd_w ? c_data_type : b_data_type;

    int src_type_size = (int)types::data_type_size(src_compute_type);
    int vec_size = hw_cfg.vec_size();

    // Prefer larger messages for large mb bwd_w
    bool is_bwd_w_message_opt
            = is_bwd_w && src_type_size <= 2 && allow_src_reorder && mb >= 16;
    int min_block_size = is_bwd_w_message_opt ? 128 : 0;
    bool nc_order = is_bwd_w_message_opt ? false : true;

    nc_block_t src_blk = nc_block_t::get_default_blocking(src_compute_type,
            is_dw, mb, ic, g, is_fwd || is_bwd_w, is_small_ic(), min_block_size,
            nc_order);
    // TODO: Force use of default_c_blk for bwd_w with bias due to reduction
    // limitation to register granularity
    nc_block_t dst_blk = nc_block_t::get_default_blocking(dst_compute_type,
            is_dw, mb, oc, g, is_bwd_d || is_bwd_w, is_small_oc(), 0, true,
            is_bwd_w && with_bias);

    auto wei_blk = goi_block_t::get_default_blocking(wei_compute_type, vec_size,
            fma_kind, is_bwd_d, is_small_ic(), g, oc, ic);

    src_tag = src_blk.tag();
    wei_tag = wei_blk.tag();
    dst_tag = dst_blk.tag();

    // Use OhwIXoYi weights for small-channel forward convolution to ensure
    // c-after-w order of reduction blocks to match the source layout.
    if (is_small_ic() && !is_dw && is_fwd && is_dp_fma()) {
        const char *patterns[] = {"ABx", "AxB", "Abx", "Axb", nullptr};
        bool found = false;
        for (auto *p = patterns; *p; p += 2) {
            auto pos = wei_tag.find(*p);
            if (pos == std::string::npos) continue;
            wei_tag = wei_tag.replace(pos, std::strlen(*p), *(p + 1));
            found = true;
            break;
        }
        ir_assert(found) << wei_tag;
    }

    // Align weights layout between forward/backward by data in some cases via
    // internal reorder to eliminate user-side reorder.
    auto fwd_wei_blk = goi_block_t::get_default_blocking(wei_compute_type,
            vec_size, fma_kind, /*is_bwd_d=*/false, is_small_ic(), g, oc, ic);
    auto fwd_wei_tag = fwd_wei_blk.tag();
    if (fwd_wei_tag != wei_tag && allow_wei_reorder) {
        user_wei_tag = fwd_wei_tag;
    }

    // Override compute layouts when using nhwc with block 2D messages.
    if (use_2d_send_nhwc) {
        if (is_bwd_d && !is_small_ic()) {
            wei_tag = "xab";
        } else {
            wei_tag = "xba";
        }
        user_wei_tag = "xba";
    }

    // Override compute layouts for nhwc case.
    bool src_matches = matches_tag(src_md, src_tag);
    bool dst_matches = matches_tag(dst_md, dst_tag);
    bool src_axb = matches_tag(src_md, "axb");
    bool dst_axb = matches_tag(dst_md, "axb");
    if (src_axb && dst_axb && (!src_matches || !dst_matches)) {
        if (!allow_src_reorder) src_tag = "axb";
        if (!allow_dst_reorder) dst_tag = "axb";
    }

    // Override compute layouts for plain outputs.
    if (is_fwd && dst_axb) dst_tag = "axb";
    if (is_bwd_d && src_axb) src_tag = "axb";
}

status_t conv_config_t::init_data_layouts(convolution_pd_t *conv_pd) {
    // Compute layout tags and user layout tags. If a compute layout is
    // different from a user layout then an extra pre/post reorder will be
    // executed before/after convolution.
    std::string src_tag, user_src_tag;
    std::string wei_tag, user_wei_tag;
    std::string dst_tag, user_dst_tag;

    auto &src_md = *conv_pd->invariant_src_md();
    auto &wei_md = *conv_pd->invariant_wei_md();
    auto &dst_md = *conv_pd->invariant_dst_md();
    auto &bia_md = *conv_pd->invariant_bia_md();

    // If src/dst is nhwc then set the other one with any to nhwc too (except
    // 1st convolution).
    bool is_small_ic_non_dw = is_small_ic() && !is_dw;
    bool is_small_oc_non_dw = is_small_oc() && !is_dw;
    bool propagate_nhwc = (matches_tag(src_md, "axb") && !is_small_ic_non_dw)
            || matches_tag(dst_md, "axb");
    if (propagate_nhwc) {
        set_default_format(src_md, "axb");
        set_default_format(dst_md, "axb");
    }

    bool allow_src_reorder = false;
    // Allow internal weights reorder in some cases. The goal is to have
    // aligned weights layouts between fwd/bwd_d/bwd_w to reduce potential
    // weights reorders during training. In general it's more efficient than
    // the external reorder.
    bool allow_wei_reorder = is_ge_xe_hpc() && is_dp_fma();
    bool allow_dst_reorder = false;
    bool src_abx = matches_tag(src_md, "abx");
    bool src_axb = matches_tag(src_md, "axb");
    if ((src_abx || src_axb) && (is_fwd || is_bwd_w) && is_small_ic_non_dw) {
        allow_src_reorder = true;
    }

    init_data_tags(allow_src_reorder, allow_wei_reorder, allow_dst_reorder,
            src_md, wei_md, dst_md, src_tag, wei_tag, dst_tag, user_wei_tag);

    if (allow_src_reorder) {
        if (src_abx) user_src_tag = "abx";
        if (src_axb) user_src_tag = "axb";
    }

    // Prefer nhwc for small-channel inputs.
    if (user_src_tag.empty() && is_fwd && is_small_ic_non_dw) {
        if (!matches_tag(src_md, src_tag)) user_src_tag = "axb";
    }
    if (user_dst_tag.empty() && is_bwd_d && is_small_oc_non_dw) {
        if (!matches_tag(dst_md, dst_tag)) user_dst_tag = "axb";
    }

    // Allow internal reorder from oihw/ohwi to more optimal weights layout.
    if (allow_wei_reorder) {
        if (matches_tag(wei_md, "abx")) user_wei_tag = "abx";
        if (matches_tag(wei_md, "axb")) user_wei_tag = "axb";
    }

    if (user_src_tag.empty()) user_src_tag = src_tag;
    if (user_wei_tag.empty()) user_wei_tag = wei_tag;
    if (user_dst_tag.empty()) user_dst_tag = dst_tag;

    bool wei_prepend_groups = (with_groups && !is_dw);
    if (wei_prepend_groups) {
        wei_tag = prepend_groups_to_tag(wei_tag);
        user_wei_tag = prepend_groups_to_tag(user_wei_tag);
    }

#ifdef GEN_CONV_DEBUG
    src_tag = ir_utils::getenv_str("stag", src_tag);
    wei_tag = ir_utils::getenv_str("wtag", wei_tag);
    dst_tag = ir_utils::getenv_str("dtag", dst_tag);

    user_src_tag = ir_utils::getenv_str("user_stag", user_src_tag);
    user_wei_tag = ir_utils::getenv_str("user_wtag", user_wei_tag);
    user_dst_tag = ir_utils::getenv_str("user_dtag", user_dst_tag);
#endif

    // Select user layouts.
    auto src_user_layout = init_layout(src_md, user_src_tag);
    auto wei_user_layout = init_layout(wei_md, user_wei_tag);
    auto dst_user_layout = init_layout(dst_md, user_dst_tag);

    if (with_bias) bia_layout = init_layout(bia_md, "a");

    if (!src_user_layout.is_strictly_equal(make_layout(src_md, user_src_tag)))
        return status::unimplemented;
    if (!dst_user_layout.is_strictly_equal(make_layout(dst_md, user_dst_tag)))
        return status::unimplemented;
    if (!wei_user_layout.is_strictly_equal(make_layout(wei_md, user_wei_tag)))
        return status::unimplemented;

    tensor_config.add_tensor("src", src_arg_key(), is_src_input(),
            is_src_output(), src_user_layout);
    tensor_config.add_tensor("wei", wei_arg_key(), is_wei_input(),
            is_wei_output(), wei_user_layout);
    if (with_bias)
        tensor_config.add_tensor("bia", bia_arg_key(), is_bia_input(),
                is_bia_output(), bia_layout);
    tensor_config.add_tensor("dst", dst_arg_key(), is_dst_input(),
            is_dst_output(), dst_user_layout);

    if (src_tag != user_src_tag)
        tensor_config.set_compute_layout("src", make_layout(src_md, src_tag));

    if (wei_tag != user_wei_tag)
        tensor_config.set_compute_layout("wei", make_layout(wei_md, wei_tag));

    if (dst_tag != user_dst_tag)
        tensor_config.set_compute_layout("dst", make_layout(dst_md, dst_tag));

    if (is_bwd_w) {
        if (wei_data_type == data_type::bf16) {
            auto &bf16_layout = tensor_config.compute_layout("wei");
            tensor_config.set_compute_layout(
                    "wei", bf16_layout.retype(type_t::f32()));
        }
        if (bia_data_type == data_type::bf16) {
            auto &bf16_layout = tensor_config.compute_layout("bia");
            tensor_config.set_compute_layout(
                    "bia", bf16_layout.retype(type_t::f32()));
        }
    }

    src_layout = tensor_config.compute_layout("src");
    wei_layout = tensor_config.compute_layout("wei");
    dst_layout = tensor_config.compute_layout("dst");
    if (with_bias) bia_layout = tensor_config.compute_layout("bia");

    // Normalize layouts: add group dimension for all layouts and reduce/fuse
    // spatial dimensions when applicable.
    normalize_conv_layouts(src_layout, wei_layout, dst_layout, bia_layout,
            with_groups, g, is_dw, reduced_dim, fuse_spatial,
            /*add_groups=*/true);

    return status::success;
}

status_t conv_config_t::init_zero_points_config(convolution_pd_t *conv_pd) {
    const auto *attr = conv_pd->attr();
    zp_cfg.do_src_compensation
            = !attr->zero_points_.has_default_values(DNNL_ARG_SRC);
    zp_cfg.do_dst_compensation
            = !attr->zero_points_.has_default_values(DNNL_ARG_DST);
    zp_cfg.is_runtime_src_zero_points
            = !attr->zero_points_.defined(DNNL_ARG_SRC);
    zp_cfg.is_runtime_dst_zero_points
            = !attr->zero_points_.defined(DNNL_ARG_DST);
    zp_cfg.is_common_src_zero_point = attr->zero_points_.common(DNNL_ARG_SRC);
    zp_cfg.is_common_dst_zero_point = attr->zero_points_.common(DNNL_ARG_DST);
    zp_cfg.common_src_zero_point = attr->zero_points_.defined(DNNL_ARG_SRC)
            ? *attr->zero_points_.get(DNNL_ARG_SRC)
            : 0;
    zp_cfg.common_dst_zero_point = attr->zero_points_.defined(DNNL_ARG_DST)
            ? *attr->zero_points_.get(DNNL_ARG_DST)
            : 0;
    return status::success;
}

// Overwrites parameters that are implied by other parameters.
status_t conv_config_t::fixup_inference_consistency() {
    // Can't reuse headers with loop unroll and post-increment offset updates.
    if (reuse_headers) do_pipeline_unroll = false;

    // Unrolling with mad or dp4a results in too large kernels.
    if (utils::one_of(fma_kind, fma_kind_t::mad, fma_kind_t::dp4a)
            && (hw_cfg.hw() >= ngen::HW::XeHPG || mb != 1))
        do_pipeline_unroll = false;

    // Without unrolling there is no benefit in keeping per-message headers.
    if (!do_pipeline_unroll) reuse_headers = true;

    if (use_preload) {
        // Prefetches are only supported with loop unrolling.
        if (prefer_prefetch()) {
            enable_prefetch();
        } else {
            enable_slm_buffering();
        }
    }
    // Downgrade dpasw -> dpas for some cases.
    if (fma_kind == fma_kind_t::dpasw) {
        // dpasw is executed by fused EUs (across X thread group
        // dimension). Do not use dpasw if X is uneven.
        if (tg_grid_dim[0] % 2 != 0) fma_kind = fma_kind_t::dpas;
        // dpasw can't be generated in case of direct load from GMEM and reorder.
        if (is_bwd_w && (allow_a_grf_reorder || allow_b_grf_reorder)
                && (!use_a_slm || !use_b_slm))
            fma_kind = fma_kind_t::dpas;
    }

    ir_assert(slm_bufs <= max_slm_bufs)
            << "Unsupported number of SLM buffers: " << slm_bufs;

    if (check_slm_size && slm_size() > hw_cfg.max_slm_size()) {
        not_enough_slm = true;
        return status::runtime_error;
    }
    return status::success;
}

void conv_config_t::set_allow_grf_reorder() {
    bool is_mad = !is_dp_fma();
    if (is_mad && is_s32_accumulator()) {
        allow_a_grf_reorder = true;
        allow_b_grf_reorder = true;
        return;
    }

    if (is_mad && b_data_type == data_type::bf16) {
        allow_b_grf_reorder = true;
        return;
    }

    bool is_a_grf_blocked
            = (a_layout().innermost_block_layout().size() % grf_size() == 0);
    if ((is_fwd || is_bwd_d) && !can_use_a_2d_send && !is_a_grf_blocked) {
        const char *dim_name = (is_fwd ? "ic" : "oc");
        int dim = (is_fwd ? ic : oc);
        int blk = bh->iter_blk(dim_name);
        if (blk * a_data_type_size % grf_size() != 0
                || dim != padded_dim(dim_name)) {
            allow_a_grf_reorder = true;
        }
    }

    if (is_dp_fma() && !is_dw && a_is_small_c()) { allow_a_grf_reorder = true; }

    if (is_bwd_w && is_dp_fma()) {
        if (!can_use_a_2d_send) allow_a_grf_reorder = true;
        if (!can_use_b_2d_send) allow_b_grf_reorder = true;
    }
}

void conv_config_t::set_init_can_use_2d_send() {
    can_use_a_2d_send = can_use_2d_send(compute_layout("a"), true);
    can_use_b_2d_send = can_use_2d_send(compute_layout("b"), false);
}

int conv_config_t::grid_dim(const std::string &name) const {
    return bh->grid_dim(name);
}

int conv_config_t::padded_dim(const std::string &name) const {
    return bh->padded_size(name);
}

const std::unordered_map<std::string, int> &conv_config_t::padded_dims() const {
    return bh->padded_dim_sizes();
}

const std::unordered_map<std::string, int> &conv_config_t::dim_blocks() const {
    return bh->iter_dims();
}

bool conv_config_t::can_use_2d_send(const layout_t &l, bool is_a) const {
    bool is_b = !is_a;
    if (hw() < ngen::HW::XeHPC) return false;

    const char *sp_name = nullptr;
    if (bh) {
        for (auto *name : {"ow", "iw", "osp"}) {
            if (bh->has_dim(name)) {
                sp_name = name;
                break;
            }
        }
        ir_assert(sp_name);
    }

    // Can't use 2D block messages for non-trivial strided dimensions.
    if (is_a && (is_fwd || is_bwd_w) && sw != 1 && (kw != 1 || pw != 0)) {
        if (bh) {
            if (bh->thr_blk(sp_name) > 1) return false;
        } else if (mb < 16) {
            return false;
        }
    }
    if (is_a && is_bwd_d && sw != 1) {
        if (bh && bh->thr_blk(sp_name) > 1) {
            return false;
        } else if (mb < 16) {
            return false;
        }
    }

    // Can't use 2D block messages for compound blocks.
    if (is_a && bh) {
        bool has_mb_block = (bh->thr_blk("mb") > 1);
        bool has_sp_block = (bh->thr_blk(sp_name) > 1);
        if (has_mb_block && has_sp_block) return false;
    }

    // 2D messages does not support vnni format with 4 byte elements
    if (type_t(b_data_type).size() >= 4) return false;

    auto is_plain_ok = [&]() {
        if (is_a || is_bwd_w) return matches_tag_strict(l, "axb");
        if (is_b && l.is_empty()) return true;
        if (is_b && is_fwd) return matches_tag_strict(l, "xba");
        if (is_b && is_bwd_d) return matches_tag_strict(l, "xab");
        return false;
    };

    if (!is_plain_ok()) return false;

    // Check 2D block message limitations.
    // Layouts for A:
    //   FWD:   NHWC (src)
    //   BWD_D: NHWC (dst)
    //   BWD_W: NHWC (src)
    // Layouts for B:
    //   FWD:   HWIO (wei)
    //   BWD_D: HWOI (wei)
    //   BWD_W: NHWC (dst)
    int a_width = (is_fwd || is_bwd_w) ? ic : oc;
    int b_width = (is_fwd || is_bwd_w) ? oc : ic;
    int a_max_height = std::max((is_fwd || is_bwd_w) ? iw : ow, mb);
    int b_max_height = is_fwd ? ic : (is_bwd_d ? oc : std::max(ow, mb));
    int a_max_pitch
            = (is_fwd || is_bwd_w) ? (ic * id * ih * iw) : (oc * od * oh * ow);
    int b_max_pitch = (is_fwd || is_bwd_d) ? b_width : (oc * od * oh * ow);
    int data_type_size = (is_a ? a_data_type_size : b_data_type_size);
    int width = (is_a ? a_width : b_width);
    int max_height = (is_a ? a_max_height : b_max_height);
    int max_pitch = (is_a ? a_max_pitch : b_max_pitch);
    if (!block_2d_width_ok(width, data_type_size)) return false;
    if (!block_2d_height_ok(max_height)) return false;
    if (!block_2d_pitch_ok(hw_cfg, width, data_type_size)) return false;
    if (!block_2d_pitch_ok(hw_cfg, max_pitch, data_type_size)) return false;
    return true;
}

bool conv_config_t::should_use_spatial_blocking(int d, int h, int w) const {
    if (hw() <= ngen::HW::XeHPG) return true;
    if (bh->max_iter_dim("mb") == 1) return true;
    if (use_2d_send_nhwc && is_bwd_d && sw != 1) return false;
    int sp = (kd * kh * kw == 1 && is_fwd) ? (d * h * w) : w;
    int block = 16;
    double mb_ratio = (double)mb / utils::rnd_up(mb, block);
    double sp_ratio = (double)sp / utils::rnd_up(sp, block);
    return sp_ratio >= mb_ratio;
}

void conv_config_t::init_use_2d_send_nhwc(const convolution_pd_t *conv_pd) {
    layout_t a_layout;
    layout_t b_layout;
    auto &a_md = (is_fwd || is_bwd_w) ? *conv_pd->invariant_src_md()
                                      : *conv_pd->invariant_dst_md();
    auto &b_md = (is_fwd || is_bwd_d) ? *conv_pd->invariant_wei_md()
                                      : *conv_pd->invariant_dst_md();
    if (a_md.format_kind != format_kind::any) a_layout = layout_t(a_md);
    if (b_md.format_kind != format_kind::any) b_layout = layout_t(b_md);

    bool a_ok = can_use_2d_send(a_layout, true);
    bool b_ok = can_use_2d_send(b_layout, false);

    use_2d_send_nhwc = a_ok && b_ok;
}

void conv_config_t::init_fuse_spatial() {
    fuse_spatial = false;
#ifdef GEN_CONV_DEBUG
    int env_value = getenv_int("fuse_spatial", -1);
    if (env_value != -1) {
        fuse_spatial = (bool)env_value;
        return;
    }
#endif

    if (!is_fwd || is_small_ic()) return;

    // Enable spatial fusion only for large batches (when N is blocked).
    // Spatial fusion may be suboptimal for small batch due to:
    // - Using smaller messages (load blocks are not fully dense anymore)
    // - Extra division arithmetic to work with fused indices
    if (src_layout.inner_block(0) == 1) return;

    fuse_spatial = true;
}

void conv_config_t::init_hoist_masks_from_compute_loop() {
    hoist_masks_from_compute_loop = false;
#ifdef GEN_CONV_DEBUG
    int env_value = getenv_int("hoist_masks_from_compute_loop", -1);
    if (env_value != -1) {
        hoist_masks_from_compute_loop = (bool)env_value;
        return;
    }
#endif
    if (use_2d_send_nhwc) {
        hoist_masks_from_compute_loop = true;
        return;
    }
    if (!fuse_spatial) return;
    if (hw() < ngen::HW::XeHPC) return;

    // Both nhwc layouts and mask hoisting require extra GRF memory so avoid
    // enabling both.
    if (is_compute_nhwc("src")) return;

    hoist_masks_from_compute_loop = true;
}

void conv_config_t::init_bwd_d_optimize_strided(int iw_thr_blk) {
    bwd_d_optimize_strided = false;
    bwd_d_optimize_strided_iw = false;
    if (!is_bwd_d) return;
    if (is_stride1()) return;

    bwd_d_optimize_strided = true;

    if (iw_thr_blk > 1) return;
    if (iw % sw != 0) return;
    bwd_d_optimize_strided_iw = true;
}

void conv_config_t::init_use_ow_kw_grf_cache() {
    use_ow_kw_grf_cache = false;
    if (!is_fwd || !is_small_ic() || kw < 3 || is_dw_large_mb()) return;
    if (is_dp_fma()) return;
    if (fuse_spatial) return;

    int iw_blk_limit = 40;
    int max_ow_blk = 16;
    int max_iw_blk = (sw * (max_ow_blk - 1) + (kw - 1) * (1 + dw) + 1);
    if (max_iw_blk > iw_blk_limit) return;

    use_ow_kw_grf_cache = true;
}

void conv_config_t::init_allow_slm_tg_slicing(
        int m_blk, int n_blk, int m_tg_dim, int n_tg_dim) {
    allow_slm_tg_slicing = false;
    if (!is_bwd_w) return;
    if (!utils::everyone_is(a_data_type, b_data_type, data_type::bf16)) return;
    if (!is_dp_fma()) return;

    // Enable only for layouts with batch blocking.
    int src_mb_blk = src_layout.inner_block(0);
    int src_ic_blk = src_layout.inner_block(2);
    int dst_mb_blk = dst_layout.inner_block(0);
    int dst_oc_blk = dst_layout.inner_block(2);
    if (src_mb_blk < 16 || dst_mb_blk < 16) return;

    int k_blk = 16; // Assume bfloat16.
    int tg_size = m_tg_dim * n_tg_dim;

    // Backward by weights with dpas layouts requires GRF reorders for A/B
    // (e.g. 2c*16n16c -> 32c16n). When SLM is used, such reorders are
    // generated after load from GMEM and before store to SLM. For optimal
    // performance we need load/store layouts to have large dense blocks. This
    // means that in some cases we have to use only a sub-grid of thread group
    // (i.e. rely on TG slicing) to perform load-store operation, otherwise we
    // may end up with reorders like 8n16c -> 16c*8n which result in scattered
    // loads/stores).
    // At the same time using sub-grids results in higher GRF consumption so we
    // only enable TG slicing when the resulting sub-grid consists of at least
    // half of the total threads.
    int src_reorder_elems = k_blk * src_ic_blk;
    int src_tg_elems = m_blk * m_tg_dim * k_blk;
    if (src_tg_elems % tg_size != 0) return;
    int src_elems_per_thr = src_tg_elems / tg_size;
    int src_slices = utils::div_up(src_reorder_elems, src_elems_per_thr);
    if (src_slices > 2) return;

    int dst_reorder_elems = k_blk * dst_oc_blk;
    int dst_tg_elems = n_blk * n_tg_dim * k_blk;
    if (dst_tg_elems % tg_size != 0) return;
    int dst_elems_per_thr = dst_tg_elems / tg_size;
    int dst_slices = utils::div_up(dst_reorder_elems, dst_elems_per_thr);
    if (dst_slices > 2) return;

    allow_slm_tg_slicing = true;
}

std::string conv_config_t::str() const {
    using namespace ir_utils;

    std::ostringstream oss;
    // clang-format off
    oss << "  HW config:                  " << hw_cfg.str() << std::endl;
    oss << "  Problem:                    " << desc_str() << std::endl;
    const char *tags[] = {"src", "wei", "dst"};
    const char *names[] = {"Source", "Weights", "Destination"};
    for (int i = 0; i < 3; i++) {
        std::string desc = std::string(names[i]) + " layout:";
        desc.insert(desc.size(), 28 - desc.size(), ' ');
        auto &compute_layout = tensor_config.compute_layout(tags[i]);
        auto &user_layout = tensor_config.user_layout(tags[i]);
        oss << "  " << desc << compute_layout;
        if (user_layout != compute_layout) {
            oss << " (user: " << user_layout << ")";
        }
        oss << std::endl;
    }
    oss << bh->brief_str();
    oss << "  Kernel grid:                " << make_seq_print_helper(kernel_grid_dim, " x ") << std::endl;
    oss << "  Thread group:               " << make_seq_print_helper(tg_grid_dim, " x ") << std::endl;
    oss << "  Threads:                    " << get_thread_count() << " (utilization: "
        << get_thread_utilization() << "% thread, "
        << get_wave_utilization() << "% wave)" <<  std::endl;
    oss << "  FMA kind:                   " << fma_kind::to_string(fma_kind) << std::endl;
    oss << "  SLM buffering:              " << "A: " << to_string(use_a_slm) << ", B: " << to_string(use_b_slm)
                                            << ", buffers: " << slm_bufs << ", pad: " << to_string(pad_slm) << std::endl;
    oss << "  GRF buffers for GMEM load:  " << gmem_bufs << std::endl;
    oss << "  Prefetch:                   " << to_string(use_prefetch) << ", buffers: " << prefetch_bufs << std::endl;
    oss << "  Do pipeline unroll:         " << to_string(do_pipeline_unroll) << std::endl;
    oss << "  Assign SBIDs:               " << to_string(assign_sbids) << std::endl;
    oss << "  Reduce GRF usage:           " << to_string(reduce_grf_usage) << std::endl;
    oss << "  Reuse headers:              " << to_string(reuse_headers) << std::endl;
    oss << "  Allow GRF reorder:          " << "A: " << to_string(allow_a_grf_reorder) << ", B: " << to_string(allow_b_grf_reorder) << std::endl;
    oss << "  Sub-tiles:                  " << "A: " << a_sub_tiles << ", B: " << b_sub_tiles << std::endl;
    oss << "  Estimated GRF usage:        " << estimated_peak_grf_usage << std::endl;
    // clang-format on
    return oss.str();
}

int estimate_register_count(const conv_config_t &cfg) {
    return estimate_grf_usage(cfg).total();
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
