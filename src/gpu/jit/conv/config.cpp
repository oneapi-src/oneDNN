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

#include "common/type_helpers.hpp"
#include "gpu/jit/conv/block_2d_utils.hpp"
#include "gpu/jit/conv/block_helper.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

status_t conv_config_t::init_common_blocking() {
    if (is_fwd && is_small_ic()) hw_cfg.set_max_tg_size(16);

    bh = std::make_shared<block_helper_t>();
    bh->set_hw_config(hw_cfg);
    bh->set_fma_kind(fma_kind);
    bh->set_abc_types(a_data_type, b_data_type, acc_data_type);

    bh->set_dim("mb", mb);
    bh->set_dim("g", g);
    bh->set_dim("oc", oc);
    bh->set_dim("ic", ic);
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

    if (is_small_ic() && !is_dw && is_dp_fma()) {
        int wei_ic_blk = wei_layout.inner_block(2);
        if (kw != 1) bh->set_max_iter_dim("ic", wei_ic_blk);
    }

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
    } else {
        const int large_sp_threshold = is_ge_xe_hpc() ? 128 : 256;
        if (!is_dw && osp > large_sp_threshold) bh->set_pref_tg_block("oc");
        bh->reorder({"mb", osp_name});
        auto spatial_dim = fuse_spatial ? osp : ow;
        if (!use_2d_send_nhwc && mb >= 128
                && (spatial_dim % 4 != 0 || spatial_dim < 64))
            bh->allow_split({"mb"});
    }

    if (is_dp_fma()) { bh->set_base_iter_block("oc", 32); }

    if (mb < 8 && !bh->any_pref_tg_block())
        bh->set_pref_tg_block(ow > oc ? osp_name : "oc");

    bh->reorder({"ic", "kw"});

    if (use_2d_send_nhwc) {
        int src_type_size = (int)types::data_type_size(src_data_type);
        // Use 64-byte reduction step to avoid partial cache line loads.
        bh->set_base_iter_block("ic", 64 / src_type_size);
        bh->set_reduce_m_block_hint(false);
    }

    bh->compute();

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

    bh->compute();

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

    bh->compute();

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

    if (!with_sum_post_op(conv_pd)) {
        tensor_config.require_zero_out("wei");
        if (with_bias) tensor_config.require_zero_out("bia");
    }

    return status::success;
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

void conv_config_t::init_data_tags(bool allow_src_reorder,
        bool allow_wei_reorder, bool allow_dst_reorder, std::string &src_tag,
        std::string &wei_tag, std::string &dst_tag, std::string &user_wei_tag) {
    auto pick_block = [&](int dim, int b0, int b1 = 0, int b2 = 0) {
        int blocks[3] = {b0, b1, b2};
        int prev_blk = 1;
        for (int i = 0; i < 3; i++) {
            if (blocks[i] == 0) continue;
            if (dim < blocks[i]) return prev_blk;
            prev_blk = blocks[i];
        }
        return prev_blk;
    };

    bool is_src_byte
            = utils::one_of(src_data_type, data_type::s8, data_type::u8);
    int src_type_size = (int)types::data_type_size(src_data_type);
    bool is_dst_byte
            = utils::one_of(dst_data_type, data_type::s8, data_type::u8);
    bool is_mad = (fma_kind == fma_kind_t::mad);
    int vec_size = hw_cfg.vec_size();

    // Set order and letters for blocking dimensions.
    std::vector<int> src_idxs = {1, 0};
    std::vector<int> dst_idxs = {1, 0};
    std::vector<char> wei_letters(3, ' ');
    char wei_letter = 'a';
    for (int i = (is_dw ? 0 : 1); i < 3; i++) {
        wei_letters[i] = wei_letter++;
    }
    std::vector<int> wei_idxs = {0, 1, 2}; // g, ic, oc
    // dpas requires ic to go before oc in innermost blocks for weights.
    if (!is_mad) std::swap(wei_idxs[1], wei_idxs[2]);

    // Set blocks for source layout.
    int src_n_blk = 1;
    int src_c_blk = 1;
    if (is_small_ic() && !is_dw && !is_bwd_d) {
        if (is_dp_fma() && is_bwd_w && allow_src_reorder) {
            src_c_blk = 4;
            src_n_blk = pick_block(mb, 32 / src_type_size);
            std::swap(src_idxs[0], src_idxs[1]);
        } else if (is_dp_fma()) {
            src_c_blk = 4 / src_type_size;
            src_n_blk = pick_block(mb, 8, 16);
        }
    } else if (is_mad && is_f32_conv()) {
        src_c_blk = (is_src_byte ? 32 : 16);
        src_n_blk = pick_block(mb, 16);
    } else {
        src_c_blk = (is_src_byte ? 32 : 16);
        src_n_blk = pick_block(mb, 16, 32);
    }

    // Set blocks for destination layout.
    int dst_n_blk = 1;
    int dst_c_blk = 1;
    if (is_mad && is_f32_conv()) {
        dst_c_blk = (is_dst_byte ? 32 : 16);
        dst_n_blk = pick_block(mb, 16);
    } else {
        dst_c_blk = (is_dst_byte ? 32 : 16);
        dst_n_blk = pick_block(mb, 16, 32);
    }

    if (with_groups && g > 1 && !is_dw) {
        if (ic % src_c_blk != 0) src_c_blk = 1;
        if (oc % dst_c_blk != 0) dst_c_blk = 1;
    }

    if (is_mad) {
        int i_dim = (is_dw ? g : ic);
        int o_dim = (is_dw ? g : oc);
        if (src_c_blk / 2 > i_dim) src_c_blk = 1;
        if (dst_c_blk / 2 > o_dim) dst_c_blk = 1;
        if (is_fwd && vec_size < dst_c_blk && o_dim <= vec_size) dst_c_blk = 1;
        if (is_bwd_d && vec_size < src_c_blk && i_dim <= vec_size)
            src_c_blk = 1;
    }

    if (src_c_blk == 1) src_n_blk = 1;
    if (dst_c_blk == 1) dst_n_blk = 1;

    // Set blocks for weights layout.
    int wei_g_blk = 1;
    int wei_o_blk = 1;
    int wei_o_blk_outer = 1;
    int wei_i_blk = 1;
    int wei_i_blk_outer = 1;
    bool is_wei_byte
            = utils::one_of(wei_data_type, data_type::s8, data_type::u8);
    auto init_mad_wei_oi_blocks = [&](bool block_by_o, int &o, int &i) {
        if (block_by_o) {
            o = vec_size;
            i = pick_block(ic, 8, 16);
        } else {
            o = pick_block(oc, 8, 16);
            i = vec_size;
        }
    };
    if (is_dw) {
        wei_g_blk = (is_wei_byte ? 32 : 16);
    } else if (is_mad) {
        init_mad_wei_oi_blocks(true, wei_o_blk, wei_i_blk);
    } else {
        // Set dpas blocks.
        int packed_dword_elems = 4 / types::data_type_size(wei_data_type);
        wei_o_blk = vec_size;
        wei_i_blk = packed_dword_elems;
        if (!is_small_ic()) {
            wei_i_blk_outer = 8;

            // XXX: This is to keep compatibility with the current implementation
            // of zero points. Other than that the outer oc block doesn't affect
            // performance/correctness.
            if (is_src_byte) wei_o_blk_outer = 32 / vec_size;
        }
    }

    src_tag = build_tag({src_n_blk, src_c_blk}, {1, 1}, {'a', 'b'}, src_idxs);
    dst_tag = build_tag({dst_n_blk, dst_c_blk}, {1, 1}, {'a', 'b'}, dst_idxs);

    auto common_wei_tag = build_tag({wei_g_blk, wei_o_blk, wei_i_blk},
            {1, wei_o_blk_outer, wei_i_blk_outer}, wei_letters, wei_idxs);

    // Backward by data requires flipped ic/oc in weights.
    if (is_bwd_d) {
        if (is_mad) {
            init_mad_wei_oi_blocks(false, wei_o_blk, wei_i_blk);
        } else {
            std::swap(wei_o_blk, wei_i_blk);
        }
        std::swap(wei_o_blk_outer, wei_i_blk_outer);
        std::swap(wei_idxs[1], wei_idxs[2]);
        wei_tag = build_tag({wei_g_blk, wei_o_blk, wei_i_blk},
                {1, wei_o_blk_outer, wei_i_blk_outer}, wei_letters, wei_idxs);
        if ((wei_i_blk * wei_i_blk_outer) != (wei_o_blk * wei_o_blk_outer))
            allow_wei_reorder = false;

    } else {
        wei_tag = common_wei_tag;
    }

    if (common_wei_tag != wei_tag && allow_wei_reorder)
        user_wei_tag = common_wei_tag;
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
    // the external reorder but it has a number of restrictions.
    bool allow_wei_reorder = is_ge_xe_hpc() && is_dp_fma();
    bool allow_dst_reorder = false;
    bool src_abx = matches_tag(src_md, "abx");
    bool src_axb = matches_tag(src_md, "axb");
    if ((src_abx || src_axb) && (is_fwd || is_bwd_w) && is_small_ic_non_dw) {
        allow_src_reorder = true;
    }

    init_data_tags(allow_src_reorder, allow_dst_reorder, allow_wei_reorder,
            src_tag, wei_tag, dst_tag, user_wei_tag);

    if (allow_src_reorder) {
        if (src_abx) user_src_tag = "abx";
        if (src_axb) user_src_tag = "axb";
    }

    if (user_src_tag.empty()) user_src_tag = src_tag;
    if (user_wei_tag.empty()) user_wei_tag = wei_tag;
    if (user_dst_tag.empty()) user_dst_tag = dst_tag;

    bool wei_prepend_groups = (with_groups && !is_dw);

    if (is_pure_nhwc(src_md, user_src_tag)
            || is_pure_nhwc(dst_md, user_dst_tag)) {
        user_src_tag = "axb";
        user_dst_tag = "axb";
        if (!allow_src_reorder) src_tag = user_src_tag;
        if (!allow_dst_reorder) dst_tag = user_dst_tag;
        bool wei_hwio = false;
        bool user_wei_hwio = false;
        if (hw() >= ngen::HW::XeHPC) {
            if (use_2d_send_nhwc) {
                wei_hwio = true;
                user_wei_hwio = true;
            }
        }
        if (wei_hwio) {
            if (is_bwd_d && !is_small_ic()) {
                wei_tag = "xab";
            } else {
                wei_tag = "xba";
            }
        }
        if (user_wei_hwio) {
            auto tag
                    = wei_prepend_groups ? prepend_groups_to_tag("xba") : "xba";
            set_default_format(wei_md, tag);
            user_wei_tag = "xba";
        }
        // Allow internal reorder from OIHW/OHWI to more optimal weights layout.
        if (matches_tag(wei_md, "abx")) user_wei_tag = "abx";
        if (matches_tag(wei_md, "axb")) user_wei_tag = "axb";
    }

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

    if (slm_size() > hw_cfg.max_slm_size()) {
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

    if (is_fwd && is_dp_fma() && is_small_ic() && !is_dw) {
        allow_a_grf_reorder = true;
    }

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
    //if (!is_dp_fma()) return;
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
        if (is_a || is_bwd_w) return matches_tag(l, "axb");
        if (is_b && l.is_empty()) return true;
        if (is_b && is_fwd) return matches_tag(l, "xba");
        if (is_b && is_bwd_d) return matches_tag(l, "xab");
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
    oss << "  Source layout:              " << tensor_config.compute_layout("src") << std::endl;
    oss << "  Weights layout:             " << tensor_config.compute_layout("wei") << std::endl;
    oss << "  Destination layout:         " << tensor_config.compute_layout("dst") << std::endl;
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

class access_grf_usage_helper_t {
public:
    access_grf_usage_helper_t(const layout_t &mem_layout, int elems,
            int reg_bytes, bool is_slm, bool use_2d_send)
        : mem_type_size_(mem_layout.type().size())
        , reg_bytes_(reg_bytes)
        , is_slm_(is_slm)
        , use_2d_send_(use_2d_send) {
        init_message_size(mem_layout);
        init_payload_size(elems);
        init_header_size();
    }

    // This setting is related to dpasw loads. dpasw reuses registers between
    // fused threads so each of the fused threads need to load only half of the
    // data it will access.
    void enable_fused_eus_sharing() { enabled_fused_eus_sharing_ = true; }

    int payload_regs() const {
        int ret = payload_size_ / reg_bytes_;
        if (enabled_fused_eus_sharing_) ret = utils::div_up(ret, 2);
        return ret;
    }

    int header_regs_per_msg() const {
        return header_size_per_msg_ / reg_bytes_;
    }

    int header_regs() const {
        int ret = nmsgs_ * header_regs_per_msg();
        if (enabled_fused_eus_sharing_) ret = utils::div_up(ret, 2);
        return ret;
    }

private:
    void init_message_size(const layout_t &mem_layout) {
        auto l = mem_layout.innermost_block_layout();
        int block_bytes = (is_slm_ ? oword_bytes_ : hword_bytes_);
        int max_block_bytes = (is_slm_ ? 16 * oword_bytes_ : 8 * hword_bytes_);
        block_t b0;
        int b0_size = mem_type_size_;
        auto &mem_blocks = mem_layout.blocks();
        if (!mem_blocks.empty()) {
            b0 = mem_blocks[0];
            b0_size = b0.block * mem_type_size_;
        }
        if (use_2d_send_) {
            is_block_ = true;
            // It's hard to determine 2D block message decomposition at this
            // point but in general 2D block messages are larger so use 2x of a
            // regular block message (empirical estimate).
            msg_size_ = 2 * max_block_bytes;
            payload_bytes_per_elem_ = mem_type_size_;
        } else if (l.size() % block_bytes == 0) {
            is_block_ = true;
            msg_size_ = (l.size() % max_block_bytes == 0) ? max_block_bytes
                                                          : block_bytes;
            payload_bytes_per_elem_ = mem_type_size_;
        } else if (!b0.is_empty() && b0_size % block_bytes == 0) {
            is_block_ = true;
            msg_size_ = block_bytes;
            payload_bytes_per_elem_ = mem_type_size_;
        } else {
            ir_assert(!is_slm_) << "Unexpected scattered messages with SLM.";
            // Assume scattered byte SIMD16 load as the worst case. Check if
            // we can use byte x {1,2,4} messages.
            int slots = 16;
            int bytes_per_slot = 4;
            for (int x : {4, 2, 1}) {
                if (x < bytes_per_slot && mem_type_size_ != x) continue;
                if (b0_size % x == 0) {
                    msg_size_ = slots * x;
                    payload_bytes_per_elem_
                            = mem_type_size_ * (bytes_per_slot / x);
                    break;
                }
            }
            ir_assert(msg_size_ > 0);
        }
    }

    void init_payload_size(int elems) {
        int elems_per_msg = utils::div_up(msg_size_, mem_type_size_);
        int payload_per_msg = elems_per_msg * payload_bytes_per_elem_;
        int payload_per_msg_grf_aligned
                = utils::rnd_up(payload_per_msg, reg_bytes_);
        nmsgs_ = utils::div_up(elems * mem_type_size_, msg_size_);
        payload_size_ = nmsgs_ * payload_per_msg_grf_aligned;
    }

    void init_header_size() {
        if (is_block_) {
            // One register per header for block messages.
            header_size_per_msg_ = reg_bytes_;
        } else {
            // Assume SIMD16 with A64 address model.
            int slots = 16;
            int bytes_per_slot = sizeof(uint64_t);
            header_size_per_msg_
                    = utils::rnd_up(slots * bytes_per_slot, reg_bytes_);
        }
    }

    static const int oword_bytes_ = 16;
    static const int hword_bytes_ = 32;

    int mem_type_size_ = 0;
    int reg_bytes_ = 0;
    bool is_slm_ = false;
    bool use_2d_send_ = false;
    bool enabled_fused_eus_sharing_ = false;

    // Whether message is block or scattered.
    bool is_block_ = false;

    // Amount of memory that can be read by a single message from global memory.
    int msg_size_ = 0;

    // How many bytes are occupied by a single element in the message payload.
    int payload_bytes_per_elem_ = 0;

    // Size of GRF buffers for all messages to load data.
    int payload_size_ = 0;

    // Number of messages to load data.
    int nmsgs_ = 0;

    // Size of header buffer per message.
    int header_size_per_msg_ = 0;
};

// Helper class to provide GRF usage estimation.
class grf_usage_helper_t {
public:
    grf_usage_helper_t(const conv_config_t &cfg) : cfg_(cfg) {
        auto &tg_grid_dim = cfg_.tg_grid_dim;

        reg_bytes_ = cfg_.grf_size();
        tg_size_ = tg_grid_dim[0] * tg_grid_dim[1] * tg_grid_dim[2];
        m_tg_dim_ = tg_grid_dim[1];
        n_tg_dim_ = tg_grid_dim[0];

        int m_iter_blk = utils::div_up(cfg_.m_tg_blk, tg_grid_dim[1]);
        int n_iter_blk = utils::div_up(cfg_.n_tg_blk, tg_grid_dim[0]);

        if (!cfg_.use_ow_kw_grf_cache) {
            a_thr_elems_ = cfg_.b_blk * m_iter_blk * cfg_.k_blk;
        } else {
            ir_assert(!cfg_.use_a_slm);
            int a_m_blk = (cfg_.sw * (m_iter_blk - 1)
                    + (cfg_.kw - 1) * (1 + cfg_.dw) + 1);
            int a_k_blk = utils::div_up(cfg_.k_blk, cfg_.kw);
            a_thr_elems_ = cfg_.b_blk * a_m_blk * a_k_blk;
        }

        b_thr_elems_ = cfg_.b_blk * cfg_.k_blk * n_iter_blk;
        c_thr_elems_ = cfg_.b_blk * m_iter_blk * n_iter_blk;
        a_tg_elems_ = a_thr_elems_ * m_tg_dim_;
        b_tg_elems_ = b_thr_elems_ * n_tg_dim_;
        a_sub_tile_elems_ = utils::div_up(a_thr_elems_, cfg_.a_sub_tiles);
        b_sub_tile_elems_ = utils::div_up(b_thr_elems_, cfg_.b_sub_tiles);
    }

    int estimate() const {
        int regs = 0;

        int max_reuse_header_regs = 0;
        int a_slm_store_payload_regs = 0;
        int b_slm_store_payload_regs = 0;

        int c_buf_usage = estimate_c_buf_usage();
        int gmem_load_usage = estimate_gmem_load_usage(max_reuse_header_regs);
        int slm_store_usage = estimate_slm_store_usage(a_slm_store_payload_regs,
                b_slm_store_payload_regs, max_reuse_header_regs);
        int slm_load_usage = estimate_slm_load_usage(max_reuse_header_regs);
        int reorder_usage = estimate_reorder_usage(
                a_slm_store_payload_regs, b_slm_store_payload_regs);

        // clang-format off
        ir_trace() << "GRF estimate:" << std::endl;
        ir_trace() << "   c_buf_usage:           " << c_buf_usage << std::endl;
        ir_trace() << "   gmem_load_usage:       " << gmem_load_usage << std::endl;
        ir_trace() << "   slm_store_usage:       " << slm_store_usage << std::endl;
        ir_trace() << "   slm_load_usage:        " << slm_load_usage << std::endl;
        ir_trace() << "   reorder_usage:         " << reorder_usage << std::endl;
        ir_trace() << "   max_reuse_header_regs: " << max_reuse_header_regs << std::endl;
        // clang-format on

        regs += c_buf_usage;
        regs += gmem_load_usage;
        regs += slm_store_usage;
        regs += slm_load_usage;
        regs += reorder_usage;
        regs += max_reuse_header_regs;
        regs += cfg_.reserved_regs;

        return regs;
    }

private:
    int estimate_c_buf_usage() const {
        int c_bytes = c_thr_elems_ * cfg_.acc_data_type_size;
        return utils::div_up(c_bytes, reg_bytes_);
    }

    int estimate_gmem_load_usage(int &max_reuse_header_regs) const {
        int regs = 0;
        for (bool is_a : {true, false}) {
            bool use_slm = ab_use_slm(is_a);
            int per_thr_elems = utils::div_up(ab_tg_elems(is_a), tg_size_);
            int load_elems
                    = (use_slm ? per_thr_elems : ab_sub_tile_elems(is_a));
            auto layout = get_gmem_layout(is_a);
            bool use_2d_send
                    = (is_a ? cfg_.can_use_a_2d_send : cfg_.can_use_b_2d_send);
            access_grf_usage_helper_t load(layout, load_elems, reg_bytes_,
                    /*is_slm=*/false, use_2d_send);
            if (is_a && !use_slm && cfg_.fma_kind == fma_kind_t::dpasw)
                load.enable_fused_eus_sharing();
            int mult = (use_slm ? cfg_.gmem_bufs : 1);
            regs += mult * load.payload_regs();
            if (cfg_.reuse_headers) {
                max_reuse_header_regs = std::max(
                        max_reuse_header_regs, load.header_regs_per_msg());
            } else {
                int sub_tiles = (is_a ? cfg_.a_sub_tiles : cfg_.b_sub_tiles);
                int mult = (use_slm ? 1 : sub_tiles);
                bool use_2d_send = (is_a ? cfg_.can_use_a_2d_send
                                         : cfg_.can_use_b_2d_send);
                regs += mult * load.header_regs();
                if (cfg_.use_prefetch) {
                    access_grf_usage_helper_t prefetch(layout, per_thr_elems,
                            reg_bytes_, /*is_slm=*/false, use_2d_send);
                    regs += prefetch.header_regs();
                }
            }
        }
        return regs;
    }

    int estimate_slm_store_usage(int &a_payload_regs, int &b_payload_regs,
            int &max_reuse_header_regs) const {
        int regs = 0;
        for (bool is_a : {true, false}) {
            if (!ab_use_slm(is_a)) continue;

            int per_thr_elems = utils::div_up(ab_tg_elems(is_a), tg_size_);
            int bytes = per_thr_elems * ab_type_size(is_a);
            auto slm_layout = dummy_slm_layout(bytes);
            access_grf_usage_helper_t store(slm_layout, bytes, reg_bytes_,
                    /*is_slm=*/true, /*use_2d_send=*/false);
            int &payload_regs = (is_a ? a_payload_regs : b_payload_regs);
            payload_regs = store.payload_regs();
            if (cfg_.reuse_headers) {
                max_reuse_header_regs = std::max(
                        max_reuse_header_regs, store.header_regs_per_msg());
            } else {
                regs += store.header_regs();
            }
        }
        return regs;
    }

    int estimate_slm_load_usage(int &max_reuse_header_regs) const {
        int regs = 0;
        for (bool is_a : {true, false}) {
            if (!ab_use_slm(is_a)) continue;

            int bytes = ab_sub_tile_elems(is_a) * ab_type_size(is_a);
            auto slm_layout = dummy_slm_layout(bytes);
            access_grf_usage_helper_t load(slm_layout, bytes, reg_bytes_,
                    /*is_slm=*/true, /*use_2d_send=*/false);
            if (is_a && cfg_.fma_kind == fma_kind_t::dpasw)
                load.enable_fused_eus_sharing();
            regs += load.payload_regs();
            if (cfg_.reuse_headers) {
                max_reuse_header_regs = std::max(
                        max_reuse_header_regs, load.header_regs_per_msg());
            } else {
                regs += load.header_regs();
            }
        }

        return regs;
    }

    // Extra registers for GRF <-> GRF reorders.
    // Estimates upper bound for A/B reorders to temporary buffers.
    int estimate_reorder_usage(int a_payload_regs, int b_payload_regs) const {
        if (!cfg_.allow_a_grf_reorder && !cfg_.allow_b_grf_reorder) return 0;

        int regs = 0;
        if (cfg_.is_bwd_w) {
            // Hardcode the size of the temporary reorder buffer for BWD_W to
            // avoid suboptimal performance.
            int bwd_w_reorder_regs = 16;
            regs += bwd_w_reorder_regs;
        }

        for (bool is_a : {true, false}) {
            bool allow_grf_reorder = (is_a ? cfg_.allow_a_grf_reorder
                                           : cfg_.allow_b_grf_reorder);
            if (!allow_grf_reorder) continue;
            int reorder_regs = 0;
            if (ab_use_slm(is_a)) {
                int &payload_regs = (is_a ? a_payload_regs : b_payload_regs);
                reorder_regs = payload_regs;
            } else {
                int size = ab_sub_tile_elems(is_a) * ab_type_size(is_a);
                reorder_regs = utils::div_up(size, reg_bytes_);
            }
            regs += reorder_regs;
        }

        return regs;
    }

    layout_t get_gmem_layout(bool is_a) const {
        auto layout = (is_a ? cfg_.a_layout() : cfg_.b_layout());
        bool is_src_dst = is_a || cfg_.is_bwd_w;
        if (is_src_dst && cfg_.is_dw) {
            auto &blocks = layout.blocks();
            if (!blocks.empty()) {
                auto &b0 = blocks[0];
                std::vector<block_t> new_blocks(
                        blocks.begin() + 1, blocks.end());
                // Remove the innermost block of channels for depthwise
                // convolution.
                if (b0.dim_idx == 2 && b0.block == 1) {
                    layout = layout_t(layout.type(), layout.ndims(),
                            layout.offset(), new_blocks,
                            /*do_normalize=*/false);
                }
            }
        }
        return layout;
    }

    int ab_type_size(bool is_a) const {
        auto ret = is_a ? cfg_.a_data_type_size : cfg_.b_data_type_size;
        if (cfg_.is_s32_accumulator() && cfg_.fma_kind == fma_kind_t::mad) {
            // s8/u8 is converted to dword-strided word for mad.
            ir_assert(ret == 1);
            ret = 4;
        }
        return ret;
    }

    int ab_tg_elems(bool is_a) const {
        return is_a ? a_tg_elems_ : b_tg_elems_;
    }

    int ab_thr_elems(bool is_a) const {
        return is_a ? a_thr_elems_ : b_thr_elems_;
    }

    int ab_sub_tile_elems(bool is_a) const {
        return is_a ? a_sub_tile_elems_ : b_sub_tile_elems_;
    }

    int ab_use_slm(bool is_a) const {
        return is_a ? cfg_.use_a_slm : cfg_.use_b_slm;
    }

    layout_t dummy_slm_layout(int size) const {
        int inner_block = 16; // In bytes.
        int outer_block = utils::div_up(size, inner_block);
        std::vector<block_t> blocks;
        blocks.emplace_back(0, inner_block, 1);
        blocks.emplace_back(1, outer_block, inner_block);
        blocks.emplace_back(0, 1, size);
        blocks.emplace_back(1, 1, size);
        return layout_t(type_t::byte(), 2, 0, blocks, /*do_normalize=*/false);
    }

    const conv_config_t &cfg_;

    int reg_bytes_;
    int tg_size_;
    int m_tg_dim_;
    int n_tg_dim_;
    int a_tg_elems_;
    int b_tg_elems_;
    int a_thr_elems_;
    int b_thr_elems_;
    int c_thr_elems_;
    int a_sub_tile_elems_;
    int b_sub_tile_elems_;
};

int estimate_register_count(const conv_config_t &cfg) {
    grf_usage_helper_t helper(cfg);
    return helper.estimate();
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
