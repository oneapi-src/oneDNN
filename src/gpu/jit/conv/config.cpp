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

#include "gpu/jit/conv/config.hpp"

#include <cctype>
#include <cstring>

#include "common/type_helpers.hpp"
#include "gpu/jit/conv/block_helper.hpp"
#include "gpu/jit/conv/config_lookup_table.hpp"
#include "gpu/jit/conv/config_plan.hpp"
#include "gpu/jit/conv/grf_usage.hpp"
#include "gpu/jit/conv/message_patterns.hpp"
#include "gpu/jit/conv/normalization.hpp"
#include "gpu/jit/ir/block_2d_utils.hpp"
#include "gpu/jit/ir/gemm_schedule.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

// Helper functions.
layout_t make_layout(const memory_desc_t &md) {
    if (md.format_kind == format_kind::any) return layout_t();
    return layout_t(md, /*do_normalize=*/false);
}

layout_t make_layout(const memory_desc_t &md, const std::string &tag) {
    return layout_t(md, tag, /*do_normalize=*/false);
}

layout_t make_layout(const type_t &type, const std::vector<dim_t> &dims,
        const std::string &tag) {
    return layout_t(type, 0, tag, dims, /*do_normalize=*/false);
}

void set_default_format(memory_desc_t &md, const std::string &tag) {
    if (md.format_kind != format_kind::any) return;
    md = make_layout(md, tag).to_dnnl(md.dims);
}

bool matches_tag(const layout_t &layout, const std::string &tag,
        const std::vector<dim_t> &dims) {
    if (layout.is_empty()) return false;
    auto tag_layout = make_layout(layout.type(), dims, tag);
    if (layout != tag_layout) return false;
    return true;
}

bool matches_tag(const layout_t &layout, const std::string &tag) {
    return matches_tag(layout, tag, layout.dims());
}

bool matches_tag_strict(const layout_t &layout, const std::string &tag) {
    if (layout.is_empty()) return false;
    auto tag_layout = make_layout(layout.type(), layout.dims(), tag);
    if (!layout.is_strictly_equal(tag_layout)) return false;
    return true;
}

bool matches_tag(const memory_desc_t &md, const std::string &tag) {
    if (md.format_kind == format_kind::any) return false;
    std::vector<dim_t> dims(md.dims, md.dims + md.ndims);
    return matches_tag(make_layout(md), tag, dims);
}

bool matches_tag_strict(const memory_desc_t &md, const std::string &tag) {
    if (md.format_kind == format_kind::any) return false;
    return matches_tag_strict(make_layout(md), tag);
}

layout_t init_layout(memory_desc_t &user_md, const std::string &optimal_tag) {
    auto optimal = make_layout(user_md, optimal_tag);
    if (user_md.format_kind != format_kind::any) {
        auto user = make_layout(user_md);
        // If layouts are physically different return the layout passed by
        // the user and return unimplemented later.
        if (user != optimal) return user;
    } else {
        user_md = optimal.to_dnnl(user_md.dims);
    }
    return optimal;
}

std::string prepend_groups_to_tag(const std::string &tag) {
    auto ret = tag;
    for (auto &c : ret) {
        bool is_lower_dim = ('a' <= c && c < 'a' + DNNL_MAX_NDIMS);
        bool is_upper_dim = ('A' <= c && c < 'A' + DNNL_MAX_NDIMS);
        if (!is_lower_dim && !is_upper_dim) continue;
        c += 1;
    }
    return "a" + ret;
}

bool is_small_ic(const conv_problem_t &prb) {
    int size = (int)types::data_type_size(prb.src_data_type);
    if (size >= 4)
        return prb.ic <= 8;
    else
        return prb.ic * size <= 16;
}

bool is_small_oc(const conv_problem_t &prb) {
    int size = (int)types::data_type_size(prb.dst_data_type);
    if (size >= 4)
        return prb.oc <= 8;
    else
        return prb.oc * size <= 16;
}

bool is_mad_g_small_oc(const conv_config_t &cfg) {
    return (cfg.prb().g > 1 && cfg.fma_kind() == fma_kind_t::mad
            && is_small_oc(cfg.prb()));
}

bool is_dw_large_mb(const conv_problem_t &prb) {
    return prb.is_dw && prb.mb >= 16;
}

status_t conv_problem_t::init(
        const engine_t *engine, const convolution_pd_t *conv_pd) {
    using namespace compute;

    if (conv_pd->has_zero_dim_memory()) return status::unimplemented;

    this->conv_pd = conv_pd;
    attr = conv_pd->attr();
    is_fwd = conv_pd->is_fwd();
    is_bwd_d = conv_pd->is_bwd_d();
    is_bwd_w = conv_pd->is_bwd_w();
    with_bias = conv_pd->with_bias();
    with_groups = conv_pd->with_groups();
    with_sum = with_sum_post_op();

    src_data_type = conv_pd->invariant_src_md()->data_type;
    wei_data_type = conv_pd->invariant_wei_md()->data_type;
    bia_data_type = conv_pd->invariant_bia_md()->data_type;
    dst_data_type = conv_pd->invariant_dst_md()->data_type;
    fpmath_mode = attr->fpmath_mode_;

    ndims = conv_pd->ndims();

    mb = conv_pd->MB();
    g = conv_pd->G();
    ic = ir_utils::safe_divide(conv_pd->IC(), g);
    oc = ir_utils::safe_divide(conv_pd->OC(), g);

    // Input spatial.
    id = conv_pd->ID();
    ih = conv_pd->IH();
    iw = conv_pd->IW();

    // Output spatial.
    od = conv_pd->OD();
    oh = conv_pd->OH();
    ow = conv_pd->OW();

    // Kernel sizes.
    kd = conv_pd->KD();
    kh = conv_pd->KH();
    kw = conv_pd->KW();

    // Strides.
    sd = conv_pd->KSD();
    sh = conv_pd->KSH();
    sw = conv_pd->KSW();

    // Padding.
    pd = conv_pd->padFront();
    ph = conv_pd->padT();
    pw = conv_pd->padL();

    // Dilation.
    dd = conv_pd->KDD();
    dh = conv_pd->KDH();
    dw = conv_pd->KDW();

    try_reduce_to_1d();

    is_dw = with_groups && (g > 1) && (oc == 1) && (ic == 1);
    ksp = kd * kh * kw;
    isp = id * ih * iw;
    osp = od * oh * ow;

    auto *compute_engine = utils::downcast<const compute_engine_t *>(engine);
    auto *device_info = compute_engine->device_info();
    gpu_arch_t gpu_arch = device_info->gpu_arch();
    auto hw = convert_dnnl_arch_to_ngen(gpu_arch);

    CHECK(init_abc_data_types(hw));
    CHECK(init_acc_data_type());
    CHECK(init_zero_points_config());

    return status::success;
}

void conv_problem_t::try_reduce_to_1d() {
    bool is_1x1 = (kd * kh * kw == 1);
    bool is_eq_oi = (od == id && oh == ih && ow == iw);
    bool is_iw_1 = iw == 1 && kw == 1 && pw == 0 && ow == 1;
    bool is_ih_1 = ih == 1 && kh == 1 && ph == 0 && oh == 1;
    reduced_dim = 0;
    auto shift_oh_to_ow = [&]() {
        ow = oh;
        iw = ih;
        ih = 1;
        oh = 1;
        kw = kh;
        kh = 1;
        pw = ph;
        ph = 0;
        sw = sh;
        sh = 1;
        dw = dh;
        dh = 0;
        reduced_dim += 1;
    };
    auto shift_od_to_oh = [&]() {
        oh = od;
        ih = id;
        id = 1;
        od = 1;
        kh = kd;
        kd = 1;
        ph = pd;
        pd = 0;
        sh = sd;
        sd = 1;
        dh = dd;
        dd = 0;
        reduced_dim += 1;
    };

    if (is_iw_1) { shift_oh_to_ow(); }
    if (is_ih_1 || is_iw_1) { shift_od_to_oh(); }
    if (is_iw_1 && is_ih_1) { shift_oh_to_ow(); }

    if (is_1x1 && is_stride1() && is_eq_oi) {
        ir_assert(pd == 0 && ph == 0 && pw == 0);
        ow = od * oh * ow;
        iw = id * ih * iw;
        od = id = kd = 1;
        oh = ih = kh = 1;
        reduced_dim = 3;
    }
}

status_t conv_problem_t::init_zero_points_config() {
    zp_cfg = zero_points_config_t();
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
    zp_cfg.common_src_zero_point = 0;
    zp_cfg.common_dst_zero_point = 0;
    return status::success;
}

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

int get_default_max_tg_size(const hw_config_t &hw_cfg, int regs, int simd) {
    const compute::gpu_arch_t arch = convert_ngen_arch_to_dnnl(hw_cfg.hw());
    const int max_eus_per_wg = compute::device_info_t::max_eus_per_wg(arch);
    const int threads_per_eu
            = compute::device_info_t::threads_per_eu(arch, regs > 128);
    const int wg_per_thr = simd * compute::device_info_t::threads_per_eu(arch)
            / threads_per_eu;

    // Optimal thread group size may differ from hardware thread count due
    // to simd_size used in computation.
    return std::min(max_eus_per_wg * utils::rnd_down_pow2(threads_per_eu),
            static_cast<int>(hw_cfg.max_wg_size() / wg_per_thr));
}

std::vector<dim_t> get_prelu_weights_dims(
        uint32_t mask, const memory_desc_t &md) {
    std::vector<dim_t> dims(md.dims, md.dims + md.ndims);
    for (int i = 0; i < md.ndims; ++i)
        dims[i] = (mask & (1 << i)) ? dims[i] : 1;
    return dims;
}

std::string build_tag(const std::vector<int> &inner_blocks,
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
            bool channel_match = false, bool force_default_c_blk = false) {
        bool is_small_c_input
                = (type.size() <= 2 && is_input && g == 1 && is_small_c);
        auto default_c_blk = type.size() == 1 ? 32 : 16;
        auto c_block = [&]() {
            if (force_default_c_blk) return default_c_blk;
            // Special case for small input channel shapes with dpas.
            if (is_small_c_input) {
                if (c == 1 && n == 1 && channel_match) return 0;
                int packed_dword_elems = 4 / type.size();
                return std::max(packed_dword_elems, utils::rnd_up_pow2(c));
            }
            auto blk_dim = is_dw ? g : g * c;
            return pick_block_rnd_up(blk_dim, default_c_blk);
        }();

        // Non-depthwise convolutions currently require channel is a multiple of
        // c_block. If that implementation restriction is removed, this logic
        // could be removed.
        if (g > 1 && !is_dw && c % c_block != 0 && c_block % c != 0)
            c_block = 1;

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

// TODO: Remove this logic and switch to an IR generation-driven flow.
bool can_use_2d_send(const conv_config_t &cfg, const layout_t &l, bool is_a) {
    const auto &prb = cfg.prb();
    bool is_b = !is_a;
    if (!cfg.is_ge_xe_hpc()) return false;

    bool with_blocking
            = !cfg.iter_dims().is_empty() || !cfg.loop_dims().is_empty();

    // Can't use 2D block messages for non-trivial strided dimensions.
    if (is_a && (prb.is_fwd || prb.is_bwd_w) && prb.sw != 1
            && (prb.kw != 1 || prb.pw != 0)) {
        if (with_blocking) {
            if (cfg.iter_dim({"osp", "ow", "iw"}) > 1) return false;
        } else if (prb.mb < 16) {
            return false;
        }
    }
    if (is_a && prb.is_bwd_d && prb.sw != 1) {
        if (with_blocking && cfg.iter_dim({"osp", "ow", "iw"}) > 1) {
            return false;
        } else if (prb.mb < 16) {
            return false;
        }
    }

    // Can't use 2D block messages for compound blocks.
    if (is_a && with_blocking) {
        bool has_mb_block = (cfg.iter_dim("mb") > 1);
        bool has_sp_block = (cfg.iter_dim({"osp", "ow", "iw"}) > 1);
        if (has_mb_block && has_sp_block) return false;
    }

    // 2D messages does not support vnni format with 4 byte elements
    if (type_t(prb.b_data_type).size() >= 4) return false;

    auto is_plain_wei_ok = [&]() {
        if (l.is_empty()) return true;
        for (auto *t : {"xba", "xab", "axb"}) {
            if (matches_tag_strict(l, t)) return true;
        }
        return false;
    };

    auto is_plain_ok = [&]() {
        if (is_a || prb.is_bwd_w) return matches_tag_strict(l, "axb");
        bool is_wei = (is_b && prb.is_fwd) || (is_b && prb.is_bwd_d);
        if (is_wei) return is_plain_wei_ok();
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
    int a_width = (prb.is_fwd || prb.is_bwd_w) ? prb.ic : prb.oc;
    int b_width = (prb.is_fwd || prb.is_bwd_w) ? prb.oc : prb.ic;
    int a_max_height
            = std::max((prb.is_fwd || prb.is_bwd_w) ? prb.iw : prb.ow, prb.mb);
    int b_max_height = prb.is_fwd
            ? prb.ic
            : (prb.is_bwd_d ? prb.oc : std::max(prb.ow, prb.mb));
    int a_max_pitch = (prb.is_fwd || prb.is_bwd_w) ? (prb.ic * prb.isp)
                                                   : (prb.oc * prb.osp);
    int b_max_pitch
            = (prb.is_fwd || prb.is_bwd_d) ? b_width : (prb.oc * prb.osp);
    int data_type_size = (is_a ? prb.a_data_type_size : prb.b_data_type_size);
    int width = (is_a ? a_width : b_width);
    int max_height = (is_a ? a_max_height : b_max_height);
    int max_pitch = (is_a ? a_max_pitch : b_max_pitch);
    if (!block_2d_width_ok(width, data_type_size)) return false;
    if (!block_2d_height_ok(max_height)) return false;
    if (!block_2d_pitch_ok(cfg.hw_cfg(), width, data_type_size)) return false;
    if (!block_2d_pitch_ok(cfg.hw_cfg(), max_pitch, data_type_size))
        return false;
    return true;
}

void init_data_tags(const conv_config_t &cfg, bool allow_src_reorder,
        bool allow_wei_reorder, bool allow_dst_reorder,
        const memory_desc_t &src_md, const memory_desc_t &wei_md,
        const memory_desc_t &dst_md,

        std::string &src_tag, std::string &wei_tag, std::string &dst_tag,
        std::string &user_wei_tag) {
    const auto &prb = cfg.prb();
    auto src_compute_type = prb.is_bwd_d ? prb.c_data_type : prb.a_data_type;
    auto dst_compute_type = prb.is_fwd
            ? prb.c_data_type
            : (prb.is_bwd_d ? prb.a_data_type : prb.b_data_type);
    auto wei_compute_type = prb.is_bwd_w ? prb.c_data_type : prb.b_data_type;

    int src_type_size = (int)types::data_type_size(src_compute_type);

    // Prefer larger messages for large mb bwd_w
    bool is_bwd_w_message_opt = prb.is_bwd_w && src_type_size <= 2
            && allow_src_reorder && prb.mb >= 16;
    int min_block_size = is_bwd_w_message_opt ? 128 : 0;
    bool nc_order = is_bwd_w_message_opt ? false : true;
    bool channel_match = prb.ic == prb.oc;

    nc_block_t src_blk = nc_block_t::get_default_blocking(src_compute_type,
            prb.is_dw, prb.mb, prb.ic, prb.g, prb.is_fwd || prb.is_bwd_w,
            is_small_ic(prb), min_block_size, nc_order, channel_match);
    // TODO: Force use of default_c_blk for bwd_w with bias due to reduction
    // limitation to register granularity
    nc_block_t dst_blk = nc_block_t::get_default_blocking(dst_compute_type,
            prb.is_dw, prb.mb, prb.oc, prb.g, prb.is_bwd_d || prb.is_bwd_w,
            is_small_oc(prb), 0, true, channel_match,
            prb.is_bwd_w && prb.with_bias && prb.g == 1);

    auto wei_blk = goi_block_t::get_default_blocking(wei_compute_type,
            cfg.vec_size(), cfg.fma_kind(), prb.is_bwd_d, is_small_ic(prb),
            prb.g, prb.oc, prb.ic);

    src_tag = src_blk.tag();
    wei_tag = wei_blk.tag();
    dst_tag = dst_blk.tag();

    // Use OhwIXoYi weights for small-channel forward convolution to ensure
    // c-after-w order of reduction blocks to match the source layout.
    if (is_small_ic(prb) && !prb.is_dw && prb.is_fwd && cfg.is_dp_fma()) {
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
            cfg.vec_size(), cfg.fma_kind(), /*is_bwd_d=*/false,
            is_small_ic(prb), prb.g, prb.oc, prb.ic);
    auto fwd_wei_tag = fwd_wei_blk.tag();
    if (fwd_wei_tag != wei_tag && allow_wei_reorder) {
        user_wei_tag = fwd_wei_tag;
    }

    // Override compute layouts when using nhwc with block 2D messages.
    bool a_2d_ok = can_use_2d_send(cfg, make_layout(prb.a_md()), true);
    bool b_2d_ok = can_use_2d_send(cfg, make_layout(prb.b_md()), false);
    if (a_2d_ok && b_2d_ok) {
        if (prb.is_bwd_d && !is_small_ic(prb)) {
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
    if (prb.is_fwd && dst_axb) dst_tag = "axb";
    if (prb.is_bwd_d && src_axb) src_tag = "axb";
}

status_t init_tensor_layouts(conv_config_t &cfg, convolution_pd_t *pd) {
    const auto &prb = cfg.prb();
    // Compute layout tags and user layout tags. If a compute layout is
    // different from a user layout then an extra pre/post reorder will be
    // executed before/after convolution.
    std::string src_tag, user_src_tag;
    std::string wei_tag, user_wei_tag;
    std::string dst_tag, user_dst_tag;

    auto &src_md = *pd->invariant_src_md();
    auto &wei_md = *pd->invariant_wei_md();
    auto &dst_md = *pd->invariant_dst_md();
    auto &bia_md = *pd->invariant_bia_md();

    // If src/dst is nhwc then set the other one with any to nhwc too (except
    // 1st convolution).
    bool is_small_ic_g1 = is_small_ic(prb) && prb.g == 1;
    bool is_small_oc_g1 = is_small_oc(prb) && prb.g == 1;
    bool propagate_nhwc = (matches_tag(src_md, "axb") && !is_small_ic_g1)
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
    bool allow_wei_reorder = cfg.is_ge_xe_hpc() && cfg.is_dp_fma();
    bool allow_dst_reorder = false;
    bool src_abx = matches_tag(src_md, "abx");
    bool src_axb = matches_tag(src_md, "axb");
    if (src_abx) allow_src_reorder = true;
    if ((src_abx || src_axb) && (prb.is_fwd || prb.is_bwd_w)
            && is_small_ic_g1) {
        allow_src_reorder = true;
    }

    init_data_tags(cfg, allow_src_reorder, allow_wei_reorder, allow_dst_reorder,
            src_md, wei_md, dst_md, src_tag, wei_tag, dst_tag, user_wei_tag);

    if (allow_src_reorder) {
        if (src_abx) user_src_tag = "abx";
        if (src_axb) user_src_tag = "axb";
    }

    // Prefer nhwc for small-channel inputs.
    if (user_src_tag.empty() && prb.is_fwd && is_small_ic_g1) {
        if (!matches_tag(src_md, src_tag)) user_src_tag = "axb";
    }
    if (user_dst_tag.empty() && prb.is_bwd_d && is_small_oc_g1) {
        if (!matches_tag(dst_md, dst_tag)) user_dst_tag = "axb";
    }

    // Allow internal reorder for plain weights.
    for (auto *t : {"abx", "axb", "xba"}) {
        if (matches_tag(wei_md, t)) {
            user_wei_tag = t;
            break;
        }
    }

    if (user_src_tag.empty()) user_src_tag = src_tag;
    if (user_wei_tag.empty()) user_wei_tag = wei_tag;
    if (user_dst_tag.empty()) user_dst_tag = dst_tag;

    bool wei_prepend_groups = (prb.with_groups && !prb.is_dw);
    if (wei_prepend_groups) {
        wei_tag = prepend_groups_to_tag(wei_tag);
        user_wei_tag = prepend_groups_to_tag(user_wei_tag);
    }

    auto &src = cfg.src_layout();
    auto &wei = cfg.wei_layout();
    auto &dst = cfg.dst_layout();
    auto &bia = cfg.bia_layout();
    if (src.is_overridden()) {
        src_tag = src.compute_unnormalized_overridden_tag();
        user_src_tag = src.compute_unnormalized_overridden_tag();
    }
    if (wei.is_overridden()) {
        wei_tag = wei.compute_unnormalized_overridden_tag();
        user_wei_tag = wei.compute_unnormalized_overridden_tag();
    }
    if (dst.is_overridden()) {
        dst_tag = dst.compute_unnormalized_overridden_tag();
        user_dst_tag = dst.compute_unnormalized_overridden_tag();
    }

    // Select user layouts.
    auto user_src_layout = init_layout(src_md, user_src_tag);
    auto user_wei_layout = init_layout(wei_md, user_wei_tag);
    auto user_dst_layout = init_layout(dst_md, user_dst_tag);

    layout_t user_bia_layout;
    if (prb.with_bias) user_bia_layout = init_layout(bia_md, "a");

    if (!user_src_layout.is_strictly_equal(make_layout(src_md, user_src_tag)))
        return status::unimplemented;
    if (!user_dst_layout.is_strictly_equal(make_layout(dst_md, user_dst_tag)))
        return status::unimplemented;
    if (!user_wei_layout.is_strictly_equal(make_layout(wei_md, user_wei_tag)))
        return status::unimplemented;

    auto src_layout = (src_tag != user_src_tag) ? make_layout(src_md, src_tag)
                                                : user_src_layout;
    auto wei_layout = (wei_tag != user_wei_tag) ? make_layout(wei_md, wei_tag)
                                                : user_wei_layout;
    auto dst_layout = (dst_tag != user_dst_tag) ? make_layout(dst_md, dst_tag)
                                                : user_dst_layout;
    auto bia_layout = user_bia_layout;

    if (prb.is_bwd_w) {
        if (prb.wei_data_type == data_type::bf16)
            wei_layout = wei_layout.retype(type_t::f32());
        if (prb.bia_data_type == data_type::bf16)
            bia_layout = bia_layout.retype(type_t::f32());
    }

    src.set_compute_unnormalized(src_layout);
    src.set_user_unnormalized(user_src_layout);
    wei.set_compute_unnormalized(wei_layout);
    wei.set_user_unnormalized(user_wei_layout);
    dst.set_compute_unnormalized(dst_layout);
    dst.set_user_unnormalized(user_dst_layout);
    bia.set_compute_unnormalized(bia_layout);
    bia.set_user_unnormalized(user_bia_layout);

    // Normalize layouts: add group dimension for all layouts and reduce/fuse
    // spatial dimensions when applicable.
    normalize_conv_layouts(src_layout, wei_layout, dst_layout, bia_layout,
            prb.with_groups, prb.g, prb.ic, prb.oc, prb.is_dw, prb.reduced_dim,
            /*fuse_spatial=*/false,
            /*add_groups=*/true);
    normalize_conv_layouts(user_src_layout, user_wei_layout, user_dst_layout,
            user_bia_layout, prb.with_groups, prb.g, prb.ic, prb.oc, prb.is_dw,
            prb.reduced_dim,
            /*fuse_spatial=*/false,
            /*add_groups=*/true);

    src.set_compute(src_layout);
    src.set_user(user_src_layout);
    wei.set_compute(wei_layout);
    wei.set_user(user_wei_layout);
    dst.set_compute(dst_layout);
    dst.set_user(user_dst_layout);
    bia.set_compute(bia_layout);
    bia.set_user(user_bia_layout);

    return status::success;
}

bool hw_ok(const hw_config_t &hw_cfg) {
    // Disable pre-XeHP until performance parity is reached with OpenCL
    // kernels.
    if (hw_cfg.hw() < ngen::HW::XeHP) return false;
    return true;
}

bool data_types_ok(const conv_problem_t &prb, const hw_config_t &hw_cfg) {
    auto src = prb.src_data_type;
    auto wei = prb.wei_data_type;
    auto dst = prb.dst_data_type;
    auto bia = prb.bia_data_type;
    bool is_bf16 = utils::one_of(data_type::bf16, src, wei, dst, bia);
    if (!prb.is_f64_conv() && utils::one_of(data_type::f64, src, wei, dst, bia))
        return false;
    if (is_bf16 && hw_cfg.hw() <= ngen::HW::XeLP) return false;
    if (prb.is_f64_conv()
            && utils::one_of(hw_cfg.hw(), ngen::HW::XeLP, ngen::HW::XeHPG))
        return false;
    if (prb.is_fwd) return true;
    if (prb.is_bwd_d) return true;
    if (prb.is_bwd_w) {
        bool ok = true;
        data_type_t default_acc_type
                = src == data_type::f64 ? data_type::f64 : data_type::f32;
        ok &= utils::one_of(
                src, data_type::bf16, data_type::f32, data_type::f64);
        ok &= (dst == src);
        ok &= utils::one_of(wei, src, default_acc_type);

        if (prb.with_bias) { ok &= utils::one_of(bia, src, data_type::f32); }
        return ok;
    }
    return false;
}

bool zero_points_ok(const conv_problem_t &prb) {
    auto *pd = prb.conv_pd;
    auto *attr = pd->attr();

    using namespace data_type;
    const auto input_type = (prb.is_fwd) ? pd->invariant_src_md()->data_type
                                         : pd->invariant_dst_md()->data_type;
    int mask_src = 0, mask_dst = 0;
    if (attr->zero_points_.get(DNNL_ARG_SRC, &mask_src) != status::success)
        return false;
    if (attr->zero_points_.get(DNNL_ARG_DST, &mask_dst) != status::success)
        return false;

    return IMPLICATION(!utils::one_of(input_type, s8, u8),
                   attr->zero_points_.has_default_values())
            && attr->zero_points_.has_default_values(DNNL_ARG_WEIGHTS)
            && (mask_src == 0 || mask_src == 1 << 1)
            && (mask_dst == 0 || mask_dst == 1 << 1);
}

std::vector<int> get_scale_args(const conv_problem_t &prb) {
    conv_arg_helper_t h(prb);
    std::vector<int> ret = {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST};
    return ret;
}

bool post_ops_ok(const conv_problem_t &prb, const hw_config_t &hw_cfg) {
    auto *attr = prb.attr;

    // No post-ops are supported for f64
    if (prb.is_f64_conv() && !attr->has_default_values()) return false;

    if (prb.is_fwd || prb.is_bwd_d) {
        using sm = primitive_attr_t::skip_mask_t;
        auto attr_skip_mask = sm::post_ops | sm::sum_dt
                | sm::zero_points_runtime | sm::scales_runtime;
        if (!attr->has_default_values(attr_skip_mask)) return false;
    } else {
        if (!attr->has_default_values()) return false;
    }

    if (!attr->scales_.has_default_values())
        if (!prb.is_s32_accumulator()) return false;
    auto scale_args = get_scale_args(prb);
    if (!attr->scales_.has_default_values(scale_args)) return false;
    for (int arg : scale_args) {
        int mask = attr->scales_.get(arg).mask_;
        // XXX: per_oc for BWD_D is treated as per_ic assuming it's called from
        // deconvolution.
        if (arg == DNNL_ARG_WEIGHTS) {
            if (!utils::one_of(mask, 0, prb.with_groups ? 3 : 1)) return false;
        } else {
            if (mask != 0) return false;
        }
    }

    for (int i = 0; i < attr->post_ops_.len(); i++) {
        auto &po = attr->post_ops_.entry_[i];
        if (po.is_eltwise()) {
            if (!jit_eltwise_injector_f32_is_supported(po.eltwise.alg))
                return false;
            else if (po.eltwise.alg == alg_kind::eltwise_tanh
                    && hw_cfg.hw() == ngen::HW::XeHPG
                    && hw_cfg.eu_count() <= 128)
                // Workaround for hard to reproduce issue in end to end
                // workloads. It is unclear what the actual issue is as the
                // kernel always works correctly in benchdnn.
                return false;
        }
    }
    return true;
}

const memory_desc_t *output_md(const convolution_pd_t *pd) {
    if (pd->is_fwd()) return pd->dst_md();
    if (pd->is_bwd_d()) return pd->diff_src_md();
    if (pd->is_bwd_w()) return pd->diff_weights_md();
    ir_error_not_expected();
    return nullptr;
}

void maybe_override_from_lookup_table(conv_config_t &cfg) {
#ifdef GEN_CONV_DEBUG
    if (ir_utils::getenv_bool("lookup", true)) return;
#endif
    static conv_config_lookup_table_t table;
    auto *s_params = table.find(cfg);
    if (s_params) cfg.override_set(s_params, /*is_env=*/false);
}

void maybe_override_from_env(conv_config_t &cfg) {
    auto cfg_env = ir_utils::getenv_str("cfg", "");
    if (cfg_env.empty()) return;
    cfg.override_set(cfg_env.c_str(), /*is_env=*/true);
}

void maybe_override(conv_config_t &cfg) {
    maybe_override_from_lookup_table(cfg);
#ifdef GEN_CONV_DEBUG
    maybe_override_from_env(cfg);
#endif
}

status_t init_fma_kind(conv_config_t &cfg) {
    if (cfg.fma_kind_param().is_overridden()) return status::success;
    const auto &prb = cfg.prb();
    auto fma_kind = fma_kind::get_supported_kind(
            cfg.hw(), prb.a_data_type, prb.b_data_type, prb.acc_data_type);
    // Force mad for some cases.
    if (prb.is_dw || (prb.ic < 3 && prb.oc < 3 && prb.mb < 8))
        fma_kind = fma_kind_t::mad;
    if (fma_kind == fma_kind_t::unknown) return status::unimplemented;
    cfg.set_fma_kind(fma_kind);
    return status::success;
}

status_t init_simd(conv_config_t &cfg) {
    if (cfg.exec_cfg_param().is_overridden("simd")) return status::success;

    const auto &prb = cfg.prb();
    int simd = fma_kind::get_simd_size(cfg.hw(), cfg.fma_kind(),
            prb.a_data_type, prb.b_data_type, prb.acc_data_type);
    cfg.set_simd(simd);
    return status::success;
}

status_t init_vec_size(conv_config_t &cfg) {
    if (cfg.exec_cfg_param().is_overridden("vec")) return status::success;

    const auto &prb = cfg.prb();
    int vec_size = cfg.simd();
    if (cfg.fma_kind() == fma_kind_t::mad) {
        int grf_elems = cfg.grf_size() / prb.acc_data_type_size;
        int vec_dim = (prb.is_fwd || prb.is_bwd_w) ? prb.oc : prb.ic;
        if (utils::rnd_up(vec_dim, grf_elems) < vec_size) vec_size = grf_elems;
    }
    // SIMD32 produces invalid layouts in bwd_w.
    if (prb.is_bwd_w) vec_size = std::min(vec_size, 16);
    cfg.set_vec_size(vec_size);
    return status::success;
}

bool post_op_layouts_ok(const conv_problem_t &prb) {
    auto *pd = prb.conv_pd;
    auto *attr = pd->attr();

    for (int i = 0; i < attr->post_ops_.len(); i++) {
        auto &po = attr->post_ops_.entry_[i];
        if (po.is_binary() || po.is_prelu()) {
            int mask = po.is_prelu()
                    ? po.prelu.mask
                    : utils::get_dims_mask(pd->invariant_dst_md()->dims,
                            po.binary.src1_desc.dims, prb.ndims, true);
            // These cases don't have message-related limitations.
            if ((mask & (1 << 1)) == 0 || mask == (1 << 1)) continue;
            auto rhs_layout = po.is_prelu() ? layout_t(type_t::f32(), 0,
                                      get_prelu_weights_dims(po.prelu.mask,
                                              *pd->invariant_dst_md()))
                                            : layout_t(po.binary.src1_desc);
            // No blocks means it's a scalar, can be always loaded.
            if (rhs_layout.blocks().empty()) return true;

            auto rhs0 = rhs_layout.blocks()[0];
            // Innermost block must:
            // - be across output channels
            // - be dense
            if (rhs0.dim_idx != 1 || dim_t(rhs0.stride) != 1) return false;
        }
    }
    return true;
}

status_t init_pd_time_cfg(const conv_problem_t &prb, conv_config_t &cfg,
        const engine_t *engine, convolution_pd_t *pd, primitive_attr_t *attr) {
    hw_config_t hw_cfg(engine);

    if (!hw_ok(hw_cfg)) return status::unimplemented;
    if (!data_types_ok(prb, hw_cfg)) return status::unimplemented;
    if (!post_ops_ok(prb, hw_cfg)) return status::unimplemented;
    if (!zero_points_ok(prb)) return status::unimplemented;

    cfg.set_prb(prb);
    cfg.set_exec_cfg(exec_config_t(hw_cfg));

    maybe_override(cfg);

    CHECK(init_fma_kind(cfg));
    CHECK(init_simd(cfg));
    CHECK(init_vec_size(cfg));
    CHECK(init_tensor_layouts(cfg, pd));

    CHECK(attr->set_default_formats(&prb.c_md()));

    if (!post_op_layouts_ok(prb)) return status::unimplemented;

    return status::success;
}

void init_hint(conv_config_t &cfg) {
    const auto &prb = cfg.prb();
    if (prb.is_fwd && is_small_ic(prb)) {
        int max_tg = 16;
        auto hint = cfg.hint();
        if (hint.max_tg_size() > max_tg) {
            hint.set_max_tg_size(max_tg);
            cfg.set_hint(hint);
        }
    }
}

void init_pipeline(conv_config_t &cfg) {
    if (cfg.pipeline().is_overridden()) return;

    const auto &prb = cfg.prb();
    bool do_unroll = true;
    bool reuse_headers = false;
    if (prb.is_fwd) {
        const int max_unroll = 9;
        if (prb.ksp > max_unroll) do_unroll = false;
    } else if (prb.is_bwd_d) {
        // Do not perform full unrolling when there are too many inner
        // iterations.
        int kernel_limit = prb.is_f32_conv() ? 4 : 9;
        if (prb.ksp > kernel_limit) do_unroll = false;

        // Do not perform full unrolling with non-unit stride unless special
        // stride optimization is enabled. These cases have non-trivial
        // post-increment updates which result in unrolling all reduction loops
        // and exceeding the instruction cache.
        if (!prb.is_stride1() && !cfg.bwd_d_optimize_strided_iw())
            do_unroll = false;
    } else if (prb.is_bwd_w) {
        int mb_iter_blk = cfg.iter_dim("mb");
        do_unroll = (cfg.is_ge_xe_hpc() && cfg.is_dp_fma() && mb_iter_blk > 1);
    }
    // Unrolling with mad or dp4a results in too large kernels.
    if (utils::one_of(cfg.fma_kind(), fma_kind_t::mad, fma_kind_t::dp4a)
            && (cfg.hw() >= ngen::HW::XeHPG || prb.mb != 1))
        do_unroll = false;
    if (reuse_headers) do_unroll = false;
    cfg.pipeline().set(do_unroll, reuse_headers);
}

void init_send_2d_nhwc(conv_config_t &cfg) {
    const auto &prb = cfg.prb();

    bool a_ok = can_use_a_2d_send(cfg);
    bool b_ok = can_use_b_2d_send(cfg);

    int64_t est_threads = 1;
    est_threads *= prb.g;
    est_threads *= prb.ic;
    est_threads *= prb.ksp;
    est_threads *= prb.mb;
    est_threads *= prb.oc;
    est_threads *= prb.osp;

    // Estimated max reduction size per thread for BWD_W.
    const int bwd_w_max_k_per_thr = 1000;
    // Estimated M x N elements per thread.
    const int mn_per_thr = 16 * 16;
    // Crosspoint to enable 2D send and blocking.
    const int min_threads_to_enable_2d = 1024;

    int k_fwd = prb.ic;
    int k_bwd_d = prb.oc;
    int k_bwd_w = std::min(bwd_w_max_k_per_thr, prb.mb * prb.osp);
    int k = prb.pick_by_dir(k_fwd, k_bwd_d, k_bwd_w);
    est_threads /= mn_per_thr;
    est_threads /= k;

    if (est_threads < min_threads_to_enable_2d) {
        cfg.set_send_2d_nhwc(false);
        return;
    }

    cfg.set_send_2d_nhwc(a_ok && b_ok);
}

void init_fuse_spatial(conv_config_t &cfg) {
    if (cfg.fuse_spatial_param().is_overridden()) return;

    const auto &prb = cfg.prb();
    if (!prb.is_fwd || is_small_ic(prb)) return;

    // Spatial fusion may be suboptimal for small batch due to:
    // - Using smaller messages (load blocks are not fully dense anymore)
    // - Extra division arithmetic to work with fused indices
    if (cfg.src_layout().compute().inner_block(0) == 1) {
        if (!prb.is_fwd || cfg.is_ge_xe_hpc()) return;
        // Enable fusion for cases without m block with overwhelming spatial dim.
        if (prb.is_int8_dst() || (prb.osp < 4096)
                || !(prb.oh == prb.ow && prb.ow == prb.od)) {
            return;
        }
    }

    cfg.set_fuse_spatial(true);
}

void init_hoist_masks_from_compute_loop(conv_config_t &cfg) {
    if (cfg.send_2d_nhwc()) {
        cfg.set_hoist_masks_from_compute_loop(true);
        return;
    }
    if (!cfg.fuse_spatial()) return;
    if (cfg.hw() < ngen::HW::XeHPC) return;

    // Both nhwc layouts and mask hoisting require extra GRF memory so avoid
    // enabling both.
    if (matches_tag(cfg.a_layout().compute_unnormalized(), "axb")) return;

    cfg.set_hoist_masks_from_compute_loop(true);
}

void init_ow_kw_grf_cache(conv_config_t &cfg) {
    const auto &prb = cfg.prb();
    if (!prb.is_fwd || !is_small_ic(prb) || prb.kw < 3 || is_dw_large_mb(prb))
        return;
    if (cfg.is_dp_fma()) return;
    if (cfg.fuse_spatial()) return;

    const int iw_blk_limit = 40;
    const int max_ow_blk = 16;
    int max_iw_blk
            = (prb.sw * (max_ow_blk - 1) + (prb.kw - 1) * (1 + prb.dw) + 1);
    if (max_iw_blk > iw_blk_limit) return;

    cfg.set_ow_kw_grf_cache(true);
}

void init_common_blocking(conv_config_t &cfg, block_helper_t &bh) {
    const auto &prb = cfg.prb();

    auto &src_layout = cfg.src_layout().compute();
    auto &wei_layout = cfg.wei_layout().compute();
    auto &dst_layout = cfg.dst_layout().compute();

    bh.set_hw_config(cfg.hw_cfg());
    bh.set_fma_kind(cfg.fma_kind());
    bh.set_simd_size(cfg.simd());
    bh.set_vec_size(cfg.vec_size());
    bh.set_max_tg_size(cfg.hint().max_tg_size());
    bh.set_max_tg_overridden(cfg.hint().max_tg_overridden());
    bh.set_abc_types(prb.a_data_type, prb.b_data_type, prb.acc_data_type);

    bh.set_dim("mb", prb.mb);
    bh.set_dim("g", prb.g);
    bh.set_dim("oc", prb.oc);
    //take into account blocked ic channels when selecting block sizes
    bh.set_dim("ic",
            prb.is_bwd_w ? std::max(src_layout.dims()[2], wei_layout.dims()[2])
                         : prb.ic);
    bh.set_dims({"kd", "kh", "kw"}, {prb.kd, prb.kh, prb.kw});

    bh.set_b_dims({"g"});

    if (prb.is_fwd) {
        if (cfg.fuse_spatial()) {
            bh.set_dims({"osp"}, {prb.osp});
            bh.set_m_dims({"mb", "osp"});
        } else {
            bh.set_dims({"od", "oh", "ow"}, {prb.od, prb.oh, prb.ow});
            bh.set_m_dims({"mb", "od", "oh", "ow"});
        }
        bh.set_n_dims({"oc"});
        bh.set_k_dims({"ic", "kd", "kh", "kw"});
    } else if (prb.is_bwd_d) {
        ir_assert(!cfg.fuse_spatial());
        bh.set_dims({"id", "ih", "iw"}, {prb.id, prb.ih, prb.iw});
        bh.set_m_dims({"mb", "id", "ih", "iw"});
        bh.set_n_dims({"ic"});
        bh.set_k_dims({"oc", "kd", "kh", "kw"});
    } else if (prb.is_bwd_w) {
        ir_assert(!cfg.fuse_spatial());
        bh.set_dims({"od", "oh", "ow"}, {prb.od, prb.oh, prb.ow});
        bh.set_m_dims({"ic", "kd", "kh", "kw"});
        bh.set_n_dims({"oc"});
        bh.set_k_dims({"mb", "od", "oh", "ow"});
    } else {
        ir_error_not_expected();
    }

    for (auto &kv : bh.dims()) {
        bh.set_pad_block(kv.first, cfg.pad_block(kv.first));
    }

    // Set base blocks to align kernel blocking with layout blocking.
    if (prb.is_fwd) {
        bh.set_base_iter_block("mb", src_layout.inner_block(0));
        int src_g_blk = prb.is_dw ? src_layout.inner_block(1) : 1;
        int wei_g_blk = prb.is_dw ? wei_layout.inner_block(0) : 1;
        bh.set_base_iter_block("g", src_g_blk, wei_g_blk);
        int src_ic_blk = src_layout.inner_block(2);
        int wei_ic_blk = wei_layout.inner_block(2);
        bh.set_base_iter_block("ic", src_ic_blk, wei_ic_blk);
        if (cfg.is_g_mad()) {
            bh.set_base_iter_block(
                    "oc", dst_layout.inner_block(2), wei_layout.inner_block(1));
            bh.dim("oc").set_iter_dim(bh.dim("oc").base_iter_block());
        }
    } else if (prb.is_bwd_d) {
        bh.set_base_iter_block("mb", dst_layout.inner_block(0));
        int dst_oc_blk = dst_layout.inner_block(2);
        int wei_oc_blk = wei_layout.inner_block(1);
        bh.set_base_iter_block("oc", dst_oc_blk, wei_oc_blk);
        if (!prb.is_dw && !cfg.is_dp_fma()) {
            int dst_g_blk = dst_layout.inner_block(1);
            int wei_g_blk = wei_layout.inner_block(0);
            bh.set_base_iter_block("g", dst_g_blk, wei_g_blk);
            bh.dim("g").set_iter_dim(bh.dim("g").base_iter_block());
        }
    } else if (prb.is_bwd_w) {
        bh.set_base_iter_block("g", wei_layout.inner_block(0));
        int wei_oc_blk = wei_layout.inner_block(1);
        int dst_oc_blk = dst_layout.inner_block(2);
        if (!is_mad_g_small_oc(cfg)) {
            bh.set_base_iter_block("oc", wei_oc_blk, dst_oc_blk);
        }
        int src_ic_blk = src_layout.inner_block(2);
        int wei_ic_blk = wei_layout.inner_block(2);
        bh.set_base_iter_block("ic", src_ic_blk, wei_ic_blk);
        int src_mb_blk = src_layout.inner_block(0);
        int dst_mb_blk = dst_layout.inner_block(0);
        bh.set_base_iter_block("mb", src_mb_blk, dst_mb_blk);
    }
}

bool should_use_spatial_blocking(const conv_config_t &cfg,
        dim_value_t mb_max_iter_dim, int d, int h, int w) {
    const auto &prb = cfg.prb();
    if (!cfg.is_ge_xe_hpc()) return true;
    if (mb_max_iter_dim == 1) return true;
    if (cfg.send_2d_nhwc() && prb.is_bwd_d && prb.sw != 1) return false;
    int sp = (prb.ksp == 1 && prb.is_fwd) ? (d * h * w) : w;
    int block = 16;
    double mb_ratio = (double)prb.mb / utils::rnd_up(prb.mb, block);
    double sp_ratio = (double)sp / utils::rnd_up(sp, block);
    return sp_ratio >= mb_ratio;
}

send_pattern_t validate_blocking(const conv_config_t &cfg,
        const block_helper_t &bh, conv_stride_layout_t::input_tensor_t tensor,
        std::pair<const char *, const char *> translation = {"", ""}) {

    const compute::gpu_arch_t arch
            = convert_ngen_arch_to_dnnl(cfg.hw_cfg().hw());

    auto is_match = [&](const block_hint_t<conv_dim_t> &hint,
                            const block_helper_t &bh) {
        for (auto dim : conv_dim_t::dims()) {
            if (hint[dim]) {
                if (dim.str() == translation.first) {
                    if (bh.iter_dim(translation.second) % hint[dim])
                        return false;
                } else {
                    if (bh.iter_dim(dim.str()) % hint[dim]) return false;
                }
            }
        }
        return true;
    };

    for (const auto &load : get_uniform_blocked_patterns(arch)) {
        uniform_blocked_idiom_t<conv_dim_t> idiom(load);
        auto layout = conv_stride_layout_t(cfg.prb(), tensor);
        auto hints = idiom.get_hints(layout);

        if (hints.empty()) continue;

        bool found_match = false;
        for (const auto &hint : hints) {
            if (is_match(hint, bh)) {
                found_match = true;
                break;
            }
        }
        if (!found_match) {
            ir_suggestion()
                    << "blocking disables " << load.str() << " load of the "
                    << tensor << " tensor. Try a multiple of:\n";
            for (auto &hint : hints) {
                ir_suggestion() << "\t" << hint.str() << "\n";
            }
            return send_pattern_t();
        }
        return load;
    }
    return send_pattern_t();
}

void init_fwd(conv_config_t &cfg, block_helper_t &bh) {
    using namespace ir_utils;

    const auto &prb = cfg.prb();
    const char *osp_name = cfg.fuse_spatial() ? "osp" : "ow";

    //set iter block for cases with no m block and large spatial
    if (!cfg.is_ge_xe_hpc() && cfg.src_layout().compute().inner_block(0) == 1
            && prb.mb > 1 && (prb.oh == prb.ow && prb.ow == prb.od)
            && prb.osp >= 512 && !cfg.is_g_mad()) {
        bh.set_base_iter_block(osp_name, 16);
    }

    if (prb.oc == 1 && prb.ic == 1) { bh.set_expand_m_block_hint(); }

    if (cfg.ow_kw_grf_cache()) {
        bh.set_base_iter_block("mb", 1);
        bh.dim("mb").set_iter_dim(1);
        bh.set_max_iter_dim("mb", 1);
        bh.set_max_m_tg_dim(2);
        bh.set_max_n_tg_dim(2);
    }
    if (cfg.is_g_mad()) {
        bh.set_base_iter_block("mb", 1);
        bh.dim("mb").set_iter_dim(1);
        bh.set_max_iter_dim("mb", 1);
        bh.set_max_iter_dim(osp_name, 4);
    }
    bh.set_loop_dim("kd", prb.kd);
    bh.set_loop_dim("kh", prb.kh);
    if (is_small_ic(prb) && !is_dw_large_mb(prb)
            && (prb.g == 1 || prb.ic == prb.oc)) {
        bh.set_block_dims({"kw"});
    } else {
        bh.set_loop_dim("kw", prb.kw);
        // mad is not tested with thread group k-slicing.
        if (cfg.is_dp_fma()) {
            bh.allow_k_tg_slicing();
            bh.set_max_k_tg_dim(8);
        }
    }

    bh.set_block_dims({"g", "oc", "ic", "mb", osp_name});
    bh.set_vector_dim(prb.is_dw || cfg.is_g_mad() ? "g" : "oc");
    bh.allow_fuse({"ic", "kw"});
    bh.allow_split({"oc", "ic", "kw"});

    int mb_base_iter_blk = bh.dim("mb").base_iter_block();
    // mb blocking is always outer so we can safely use a smaller divisor to
    // have more flexible blocking for some cases.
    int mb_base_iter_divisor = is_dw_large_mb(prb) ? 32 : 8;
    mb_base_iter_blk = math::gcd(mb_base_iter_divisor, mb_base_iter_blk);

    bh.set_base_iter_block("mb", mb_base_iter_blk);

    bool use_sp_blocking = false;
    if (matches_tag(cfg.src_layout().compute_unnormalized(), "axb")) {
        use_sp_blocking = should_use_spatial_blocking(
                cfg, bh.max_iter_dim("mb"), prb.od, prb.oh, prb.ow);
    } else if (cfg.src_layout().compute().inner_block(0) == 1) {
        use_sp_blocking = true;
    } else if (prb.is_dw && !is_dw_large_mb(prb)) {
        use_sp_blocking = true;
    } else if (cfg.is_g_mad() || cfg.ow_kw_grf_cache()) {
        use_sp_blocking = true;
    }

    if (use_sp_blocking) {
        if (prb.is_dw) bh.set_pref_tg_block(osp_name);
        bh.allow_split({osp_name, "mb"});
        bh.reorder({osp_name, "mb"});
        if (!prb.is_int8_dst() && !cfg.fuse_spatial() && prb.mb < 16
                && prb.iw % 8 != 0 && !prb.is_dw) {
            int max_dim = (prb.ic < 3 && prb.oc < 3) ? 2 : 1;
            bh.set_max_m_tg_dim(max_dim);
        }
    } else {
        const int large_sp_threshold = cfg.is_ge_xe_hpc() ? 128 : 256;
        if (!prb.is_dw && prb.ow > large_sp_threshold)
            bh.set_pref_tg_block("oc");
        else if (cfg.is_dp_fma() && prb.mb >= 16)
            bh.set_pref_tg_block(osp_name);
        bh.reorder({"mb", osp_name});
        auto spatial_dim = cfg.fuse_spatial() ? prb.osp : prb.ow;
        if (!cfg.send_2d_nhwc() && prb.mb >= 128
                && (spatial_dim % 4 != 0 || spatial_dim < 64))
            bh.allow_split({"mb"});
        if (bh.expand_m_block_hint()) {
            // allow splitting for m tg dim
            bh.allow_split({osp_name, "mb"});
            // allow fusing for m iter dim
            bh.allow_fuse({osp_name, "mb"});
        }
    }

    if (prb.mb < 8 && !bh.any_pref_tg_block())
        bh.set_pref_tg_block(prb.ow > prb.oc ? osp_name : "oc");

    bh.reorder({"ic", "kw"});

    if (cfg.send_2d_nhwc()) {
        // Use 64-byte reduction step to avoid partial cache line loads.
        bh.set_base_iter_block("ic", 64 / prb.a_data_type_size);
        bh.set_reduce_m_block_hint(false);
    }

    bh.compute();
#ifdef GEN_CONV_DEBUG
    if (!can_use_2d_send(cfg, cfg.a_layout().compute_unnormalized(), true)
            && prb.g == 1)
        cfg.a_load_pattern = validate_blocking(cfg, bh,
                conv_stride_layout_t::input_tensor_t::src, {"ow", osp_name});
    if (!can_use_2d_send(cfg, cfg.b_layout().compute_unnormalized(), false)
            && prb.g == 1)
        cfg.b_load_pattern = validate_blocking(
                cfg, bh, conv_stride_layout_t::input_tensor_t::wei);
#endif
}

void init_bwd_d(conv_config_t &cfg, block_helper_t &bh) {
    using namespace ir_utils;

    const auto &prb = cfg.prb();
    bh.set_loop_dim("kw", prb.kw);
    bh.set_loop_dim("kd", prb.kd);
    bh.set_loop_dim("kh", prb.kh);
    bh.set_block_dims({"g", "oc", "ic", "mb", "iw"});
    bh.set_vector_dim(prb.is_dw ? "g" : "ic");
    bh.allow_split({"oc", "ic"});

    bool use_w_blocking = false;
    if (matches_tag(cfg.dst_layout().compute_unnormalized(), "axb")) {
        use_w_blocking = should_use_spatial_blocking(
                cfg, bh.max_iter_dim("mb"), prb.id, prb.ih, prb.iw);
    } else if (cfg.dst_layout().compute().inner_block(0) == 1) {
        use_w_blocking = true;
    }

    if (use_w_blocking) {
        bh.allow_fuse({"iw", "mb"});
        bh.allow_split({"iw", "mb"});
        bh.reorder({"iw", "mb"});
    } else {
        bh.reorder({"mb", "iw"});
        bh.set_base_iter_block("mb", 8);
    }

    if (cfg.send_2d_nhwc()) {
        bh.set_base_iter_block("oc", 64 / prb.a_data_type_size);
        if (!prb.is_stride1()) bh.allow_split({"mb"});
        bh.set_reduce_m_block_hint(false);
    }

    bh.compute();

#ifdef GEN_CONV_DEBUG
    if (!can_use_2d_send(cfg, cfg.a_layout().compute_unnormalized(), true))
        validate_blocking(cfg, bh, conv_stride_layout_t::input_tensor_t::dst);
    if (!can_use_2d_send(cfg, cfg.b_layout().compute_unnormalized(), false))
        validate_blocking(cfg, bh, conv_stride_layout_t::input_tensor_t::wei);
#endif
}

void init_bwd_w(conv_config_t &cfg, block_helper_t &bh) {
    const auto &prb = cfg.prb();
    bh.allow_k_grid_slicing();

    bh.set_block_dims({"g", "oc", "ic", "mb", "oh", "ow"});
    bool vectorize_g = is_mad_g_small_oc(cfg) || prb.is_dw;
    bh.set_vector_dim(vectorize_g ? "g" : "oc");

    int size = (int)types::data_type_size(prb.src_data_type);
    if (size >= 4) {
        if (prb.oc <= 32) bh.set_max_iter_dim("oc", 16);
        if (prb.ic <= 32) bh.set_max_iter_dim("ic", 16);
    } else {
        if (prb.oc < 32) bh.set_max_iter_dim("oc", 16);
        if (prb.ic < 32) bh.set_max_iter_dim("ic", 16);
    }

    if (is_small_ic(prb) && !vectorize_g) {
        bh.set_block_dims({"kw"});
        bh.set_max_tg_dim("kw", 1);
        bh.set_max_iter_dim("kw", 8);
    }

    // Avoid 2D spatial blocking when possible (when 1D blocking can be
    // enough). Extra oh/od loops may result in assembly bloat due to pipeline
    // unroll.
    if (prb.mb >= 32 && prb.ow >= 16) {
        bh.set_max_loop_dim("oh", 1);
        bh.set_max_loop_dim("od", 1);
    }

    bh.set_max_iter_dim("oh", 1);

    bh.allow_split({"oc", "ic", "mb", "ow"});
    bh.allow_fuse({"ic", "kw"});
    bh.allow_fuse({"mb", "oh", "ow"});
    bh.set_max_loop_dim("mb", 2);
    bh.set_base_iter_block("mb", math::gcd(16, bh.dim("mb").base_iter_block()));

    bh.reorder({"mb", "ow", "oh"});

    if (cfg.send_2d_nhwc()) bh.set_reduce_m_block_hint(false);

    bh.compute();
#ifdef GEN_CONV_DEBUG
    if (!can_use_2d_send(cfg, cfg.a_layout().compute_unnormalized(), true))
        validate_blocking(cfg, bh, conv_stride_layout_t::input_tensor_t::src);
    if (!can_use_2d_send(cfg, cfg.b_layout().compute_unnormalized(), false))
        validate_blocking(cfg, bh, conv_stride_layout_t::input_tensor_t::dst);
#endif
}

void init_blocking(conv_config_t &cfg) {
    const auto &prb = cfg.prb();
    block_helper_t bh;
    init_common_blocking(cfg, bh);
    if (prb.is_fwd) {
        init_fwd(cfg, bh);
    } else if (prb.is_bwd_d) {
        init_bwd_d(cfg, bh);
    } else if (prb.is_bwd_w) {
        init_bwd_w(cfg, bh);
    } else {
        ir_error_not_expected();
    }

    auto &dims = cfg.dims();
    auto &iter_dims = cfg.iter_dims();
    auto &thread_group_dims = cfg.thread_group_dims();
    auto &loop_dims = cfg.loop_dims();

    for (auto &kv : bh.dims()) {
        auto &name = kv.first;
        auto &d = kv.second;
        if (!dims.is_overridden()) dims.set(name, d.size());
        if (!iter_dims.is_overridden()) iter_dims.set(name, d.iter_dim());
        if (!thread_group_dims.is_overridden())
            thread_group_dims.set(name, d.tg_dim());
        if (!loop_dims.is_overridden()) loop_dims.set(name, d.loop_dim());
        if (cfg.shrink_tg_dims()) {
            int dim = cfg.dim(name);
            int iter = cfg.iter_dim(name);
            int tg = cfg.thread_group_dim(name);
            int loop = cfg.loop_dim(name);
            int pad_blk = cfg.pad_block(name);
            while (tg > 1) {
                int padded = utils::rnd_up(
                        dim, math::lcm(iter * tg * loop, pad_blk));
                if (dim * 2 > padded) break;
                tg = std::max(1, tg / 2);
            }
            cfg.thread_group_dims().set(name, tg);
        }
    }
}

const char **get_kernel_grid_conv_dims(const conv_problem_t &prb, int idx) {
    static const char *fwd_0[] = {"oc", nullptr};
    static const char *fwd_1[] = {"g", "osp", "od", "oh", "ow", nullptr};
    static const char *fwd_2[] = {"mb", nullptr};
    static const char *bwd_d_0[] = {"ic", nullptr};
    static const char *bwd_d_1[] = {"g", "id", "ih", "iw", nullptr};
    static const char *bwd_d_2[] = {"mb", nullptr};
    static const char *bwd_w_0[] = {"oc", nullptr};
    static const char *bwd_w_1[]
            = {"ic", "kd", "kh", "kw", "od", "oh", "ow", nullptr};
    static const char *bwd_w_2[] = {"g", "mb", nullptr};
    static const char **fwd[] = {fwd_0, fwd_1, fwd_2};
    static const char **bwd_d[] = {bwd_d_0, bwd_d_1, bwd_d_2};
    static const char **bwd_w[] = {bwd_w_0, bwd_w_1, bwd_w_2};
    ir_assert(idx >= 0 && idx < 3);
    if (prb.is_fwd) return fwd[idx];
    if (prb.is_bwd_d) return bwd_d[idx];
    if (prb.is_bwd_w) return bwd_w[idx];
    ir_error_not_expected();
    return nullptr;
}

const char **get_thread_group_grid_conv_dims(
        const conv_problem_t &prb, int idx) {
    static const char *fwd_0[] = {"oc", nullptr};
    static const char *fwd_1[] = {"mb", "osp", "ow", nullptr};
    static const char *fwd_2[] = {"ic", nullptr};
    static const char *bwd_d_0[] = {"ic", nullptr};
    static const char *bwd_d_1[] = {"mb", "iw", nullptr};
    static const char *bwd_d_2[] = {"oc", nullptr};
    static const char *bwd_w_0[] = {"oc", nullptr};
    static const char *bwd_w_1[] = {"ic", nullptr};
    static const char *bwd_w_2[] = {nullptr};
    static const char **fwd[] = {fwd_0, fwd_1, fwd_2};
    static const char **bwd_d[] = {bwd_d_0, bwd_d_1, bwd_d_2};
    static const char **bwd_w[] = {bwd_w_0, bwd_w_1, bwd_w_2};
    ir_assert(idx >= 0 && idx < 3);
    if (prb.is_fwd) return fwd[idx];
    if (prb.is_bwd_d) return bwd_d[idx];
    if (prb.is_bwd_w) return bwd_w[idx];
    ir_error_not_expected();
    return nullptr;
}

void init_padded_dims(conv_config_t &cfg) {
    for (auto &kv : cfg.dims().get()) {
        auto &name = kv.first;
        int dim = cfg.dim(name);
        int iter = cfg.iter_dim(name);
        int tg = cfg.thread_group_dim(name);
        int loop = cfg.loop_dim(name);
        int blk = iter * tg * loop;
        int pad_blk = cfg.pad_block(name);
        int padded = utils::rnd_up(dim, math::lcm(blk, pad_blk));
        cfg.padded_dims().set(name, padded);
    }
}

void init_kernel_grid(conv_config_t &cfg) {
    const auto &prb = cfg.prb();
    auto get = [&](const char *name) {
        int padded = cfg.padded_dim(name);
        int iter = cfg.iter_dim(name);
        int loop = cfg.loop_dim(name);
        int tg = cfg.thread_group_dim(name);
        int tg_block = iter * loop * tg;
        return ir_utils::safe_divide(padded, tg_block);
    };

    const int grid_ndims = 3;
    std::vector<int> dims = {1, 1, 1};
    for (int i = 0; i < grid_ndims; i++) {
        auto **dd = get_kernel_grid_conv_dims(prb, i);
        for (auto **d = dd; *d; d++)
            dims[i] *= get(*d);
    }
    cfg.set_kernel_grid(grid_info_t(dims, "grid_idx"));
}

void init_thread_group_grid(conv_config_t &cfg) {
    const auto &prb = cfg.prb();
    auto get = [&](const char *name) {
        return cfg.thread_group_dims().get(name);
    };

    const int grid_ndims = 3;
    std::vector<int> dims = {1, 1, 1};
    for (int i = 0; i < grid_ndims; i++) {
        auto **dd = get_thread_group_grid_conv_dims(prb, i);
        for (auto **d = dd; *d; d++)
            dims[i] *= get(*d);
    }
    cfg.set_thread_group_grid(grid_info_t(dims, "tg_idx"));
}

// Enable optimization for strided BWD_D convolution.
void init_bwd_d_optimize_strided(conv_config_t &cfg) {
    const auto &prb = cfg.prb();
    if (!prb.is_bwd_d) return;
    if (prb.is_stride1()) return;

    cfg.set_bwd_d_optimize_strided(true);

    if (cfg.iter_dim("iw") > 1) return;
    if (prb.iw % prb.sw != 0) return;
    cfg.set_bwd_d_optimize_strided_iw(true);

    // Update blocks.
    int iw_tg_dim0 = cfg.thread_group_dim("iw");
    ir_assert(math::is_pow2(iw_tg_dim0));
    ir_assert(prb.iw % prb.sw == 0);
    for (int tg_dim = iw_tg_dim0; tg_dim >= 1; tg_dim /= 2) {
        if ((prb.iw / prb.sw) % tg_dim != 0) continue;

        cfg.thread_group_dims().set("iw", tg_dim);
        int mb_iter_dim = cfg.iter_dim("mb");
        int new_mb_tg_dim = cfg.thread_group_dim("mb") * iw_tg_dim0 / tg_dim;
        // TODO: non-uniform thread group is unsupported
        while (new_mb_tg_dim > 1
                && utils::rnd_up(prb.mb, mb_iter_dim * new_mb_tg_dim) - prb.mb
                        >= mb_iter_dim) {
            new_mb_tg_dim /= 2;
        }
        if (mb_iter_dim * new_mb_tg_dim <= prb.mb) {
            cfg.thread_group_dims().set("mb", new_mb_tg_dim);
        }
        break;
    }
}

void init_unroll(conv_config_t &cfg) {
    if (cfg.unroll().is_overridden()) return;

    const auto &prb = cfg.prb();

    if (prb.is_bwd_w) {
        int mb_loop_dim = cfg.loop_dim("mb");
        int ow_loop_dim = cfg.loop_dim("ow");
        cfg.unroll().set("mb", mb_loop_dim);
        if (cfg.iter_dim("ow") > 1 && ow_loop_dim <= 8 && cfg.is_dp_fma()) {
            cfg.unroll().set("ow", ow_loop_dim);
        }
    }
}

bool can_split_across_thread_group(int tg_size, int elems, int type_size) {
    // Thread group grid is limited to powers of two. We can reliably split
    // only powers of two elements across such grids.
    if (!math::is_pow2(elems)) return false;

    // Check that the buffer can be uniformly distributed.
    if (elems % tg_size != 0) return false;

    // Check that SLM can be stored with oword messages.
    int bytes_per_thr = (elems / tg_size) * type_size;
    if (bytes_per_thr % 16 != 0) return false;

    return true;
}

void get_slm_enable(const conv_config_t &cfg, bool &enable_a, bool &enable_b) {
    if (cfg.with_plan()) {
        enable_a = cfg.plan().slm.has_a();
        enable_b = cfg.plan().slm.has_b();
        return;
    }

    enable_a = false;
    enable_b = false;
    if (cfg.hw() >= ngen::HW::XeHPC) return;

    const auto &prb = cfg.prb();
    auto &tg = cfg.thread_group_grid();
    int tg_size = tg.elems();
    bmnk_dim_helper_t h(cfg);
    int m_tg_blk = h.thread_group_dim('m') * h.iter_dim('m');
    int n_tg_blk = h.thread_group_dim('n') * h.iter_dim('n');
    int k_iter_blk = h.iter_dim('k');
    if (!cfg.ow_kw_grf_cache()) {
        // Check that SLM can be stored with oword messages.
        int tg_size = tg.elems();
        int bytes_per_tg = (m_tg_blk * k_iter_blk * prb.a_data_type_size);
        int align = prb.is_bwd_w ? 32 : 16;
        bool can_split_a = bytes_per_tg % align == 0
                && bytes_per_tg / tg_size >= k_iter_blk && k_iter_blk % 2 == 0;
        enable_a = (tg.dim(0) > 1) && can_split_a;
    }
    bool can_split_b = can_split_across_thread_group(
            tg_size, n_tg_blk * k_iter_blk, prb.b_data_type_size);
    enable_b = (tg.dim(1) > 1) && can_split_b;
}

void init_slm(conv_config_t &cfg) {
    if (cfg.slm().is_overridden()) return;

    const auto &prb = cfg.prb();

    int bufs = 0;
    int gmem_bufs = 0;
    bool enable_a = false;
    bool enable_b = false;
    get_slm_enable(cfg, enable_a, enable_b);
    auto &tg = cfg.thread_group_grid();
    if (enable_a || enable_b) {
        bool is_small_tg = (tg.dim(0) * tg.dim(1) <= 8);
        int pref_bufs
                = ((is_small_tg || prb.is_f32_conv()) && prb.mb > 1 ? 2 : 3);
        if (cfg.pipeline().do_unroll()) {
            bufs = pref_bufs;
            gmem_bufs = (cfg.is_dp_fma() ? 2 : 1);
        } else {
            // Double/triple SLM buffering is not supported when only one
            // matrix is SLM-buffered.
            // Limit the SLM buffer count to 1 in the presence of ZP, since
            // for now the masks are otherwise computed for next iterations.
            const bool use_pref_bufs = (enable_a == enable_b)
                    && (!prb.zp_cfg.do_src_compensation);
            bufs = (use_pref_bufs ? pref_bufs : 1);
            gmem_bufs = 1;
        }
    }
    if (cfg.with_plan())
        gmem_bufs = std::min(cfg.plan().max_gmem_bufs, gmem_bufs);
    cfg.slm().set(bufs, gmem_bufs, enable_a, enable_b);
}

void get_prefetch_enable(
        const conv_config_t &cfg, bool &enable_a, bool &enable_b) {
    if (cfg.with_plan()) {
        enable_a = cfg.plan().prefetch.has_a();
        enable_b = cfg.plan().prefetch.has_b();
        return;
    }

    enable_a = false;
    enable_b = false;
    if (cfg.hw() < ngen::HW::XeHPC) return;

    const auto &prb = cfg.prb();
    auto &tg = cfg.thread_group_grid();
    int tg_size = tg.elems();
    bmnk_dim_helper_t h(cfg);
    int m_tg_blk = h.thread_group_dim('m') * h.iter_dim('m');
    int n_tg_blk = h.thread_group_dim('n') * h.iter_dim('n');
    int k_iter_blk = h.iter_dim('k');
    bool can_split_a = (tg.dim(0) == 1)
            || can_split_across_thread_group(
                    tg_size, m_tg_blk * k_iter_blk, prb.a_data_type_size);
    bool can_split_b = (tg.dim(1) == 1)
            || can_split_across_thread_group(
                    tg_size, n_tg_blk * k_iter_blk, prb.b_data_type_size);

    enable_a = can_split_a;
    enable_b = can_split_b;

    if (!prb.is_bwd_d && is_small_ic(prb) && cfg.is_dp_fma()) enable_a = false;
}

void init_prefetch(conv_config_t &cfg) {
    if (cfg.prefetch().is_overridden()) return;

    bool enable_a = false;
    bool enable_b = false;
    get_prefetch_enable(cfg, enable_a, enable_b);

    if (!enable_a && !enable_b) return;

    cfg.prefetch().set_a(enable_a);
    cfg.prefetch().set_b(enable_b);

    int bufs = cfg.prb().is_f32_conv() ? 2 : 3;
    cfg.prefetch().set(bufs);
}

void init_allow_a_grf_reorder(conv_config_t &cfg) {
    if (cfg.allow_a_grf_reorder_param().is_overridden()) return;
    const auto &prb = cfg.prb();
    bool use_a_2d_send = can_use_a_2d_send(cfg);
    bool is_a_grf_blocked
            = (cfg.a_layout().compute().innermost_block_layout().size()
                            % cfg.grf_size()
                    == 0);
    bool is_mad = !cfg.is_dp_fma();
    cfg.set_allow_a_grf_reorder(!prb.matches_user_types());
    if (is_mad && prb.is_s32_accumulator()) {
        cfg.set_allow_a_grf_reorder(true);
        return;
    }
    if (is_mad && prb.b_data_type == data_type::bf16) { return; }
    if ((prb.is_fwd || prb.is_bwd_d) && !use_a_2d_send && !is_a_grf_blocked) {
        const char *dim_name = (prb.is_fwd ? "ic" : "oc");
        int dim = (prb.is_fwd ? prb.ic : prb.oc);
        int blk = cfg.iter_dim(dim_name);
        if (blk * prb.a_data_type_size % cfg.grf_size() != 0
                || dim != cfg.padded_dim(dim_name)) {
            cfg.set_allow_a_grf_reorder(true);
        }
    }
    if (cfg.send_2d_nhwc() && (!prb.is_fwd)) cfg.set_allow_a_grf_reorder(true);

    bool a_is_small_c = (prb.is_fwd || prb.is_bwd_w) ? is_small_ic(prb)
                                                     : is_small_oc(prb);
    if (cfg.is_dp_fma() && !prb.is_dw && a_is_small_c) {
        cfg.set_allow_a_grf_reorder(true);
    }
    if (prb.is_bwd_w && cfg.is_dp_fma()) { cfg.set_allow_a_grf_reorder(true); }
}

void init_allow_b_grf_reorder(conv_config_t &cfg) {
    if (cfg.allow_b_grf_reorder_param().is_overridden()) return;
    const auto &prb = cfg.prb();
    bool use_b_2d_send = can_use_b_2d_send(cfg);
    bool is_mad = !cfg.is_dp_fma();
    cfg.set_allow_b_grf_reorder(!prb.matches_user_types());
    if (is_mad && prb.is_s32_accumulator()) {
        cfg.set_allow_b_grf_reorder(true);
        return;
    }
    if (is_mad && prb.b_data_type == data_type::bf16) {
        cfg.set_allow_b_grf_reorder(true);
        return;
    }
    if (prb.is_bwd_w && cfg.is_dp_fma() && !use_b_2d_send) {
        cfg.set_allow_b_grf_reorder(true);
    }
}

void init_allow_slm_tg_slicing(conv_config_t &cfg) {
    const auto &prb = cfg.prb();
    if (!prb.is_bwd_w) return;
    if (!utils::everyone_is(prb.a_data_type, prb.b_data_type, data_type::bf16))
        return;
    if (!cfg.is_dp_fma()) return;

    // Enable only for layouts with batch blocking.
    int src_mb_blk = cfg.src_layout().compute().inner_block(0);
    int src_ic_blk = cfg.src_layout().compute().inner_block(2);
    int dst_mb_blk = cfg.dst_layout().compute().inner_block(0);
    int dst_oc_blk = cfg.dst_layout().compute().inner_block(2);
    if (src_mb_blk < 16 || dst_mb_blk < 16) return;

    bmnk_dim_helper_t h(cfg);
    int k_iter_blk = h.iter_dim('k');
    int m_iter_blk = h.iter_dim('m');
    int n_iter_blk = h.iter_dim('n');
    int m_tg_dim = h.thread_group_dim('m');
    int n_tg_dim = h.thread_group_dim('n');
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
    int src_reorder_elems = k_iter_blk * src_ic_blk;
    int src_tg_elems = m_iter_blk * m_tg_dim * k_iter_blk;
    if (src_tg_elems % tg_size != 0) return;
    int src_elems_per_thr = src_tg_elems / tg_size;
    int src_slices = utils::div_up(src_reorder_elems, src_elems_per_thr);
    if (src_slices > 2) return;

    int dst_reorder_elems = k_iter_blk * dst_oc_blk;
    int dst_tg_elems = n_iter_blk * n_tg_dim * k_iter_blk;
    if (dst_tg_elems % tg_size != 0) return;
    int dst_elems_per_thr = dst_tg_elems / tg_size;
    int dst_slices = utils::div_up(dst_reorder_elems, dst_elems_per_thr);
    if (dst_slices > 2) return;

    cfg.set_allow_slm_tg_slicing(true);
}

void init_assign_sbids(conv_config_t &cfg) {
    if (cfg.is_dp_fma()) cfg.set_assign_sbids(true);
}

void init_subtiles(conv_config_t &cfg) {
    if (cfg.subtiles().is_overridden()) return;
    if (!cfg.with_plan()) return;

    int a = 1;
    int b = 1;
    auto &p = cfg.plan();
    if (p.split_abc == abc_kind_t::a) a = p.split_factor;
    if (p.split_abc == abc_kind_t::b) b = p.split_factor;
    cfg.subtiles().set(a, b);
}

// Overwrites parameters that are implied by other parameters.
status_t fixup_config(conv_config_t &cfg) {
    const auto &prb = cfg.prb();

    // Downgrade dpasw -> dpas for some cases.
    if (cfg.fma_kind() == fma_kind_t::dpasw) {
        // dpasw is executed by fused EUs (across X thread group
        // dimension). Do not use dpasw if X is uneven.
        if (cfg.thread_group_grid().dim(0) % 2 != 0)
            cfg.set_fma_kind(fma_kind_t::dpas);
        // dpasw can't be generated in case of direct load from GMEM and reorder.
        if (prb.is_bwd_w
                && (cfg.allow_a_grf_reorder() || cfg.allow_b_grf_reorder())
                && (!cfg.slm().a() || !cfg.slm().b()))
            cfg.set_fma_kind(fma_kind_t::dpas);
    }

    return status::success;
}

template <typename GetFuncT>
bool in_grid_dims(
        GetFuncT get_func, const conv_problem_t &prb, const std::string &dim) {
    for (int i = 0; i < 3; i++) {
        auto **dd = get_func(prb, i);
        for (auto **d = dd; *d; d++)
            if (*d == dim) return true;
    }
    return false;
}

status_t check_plan(conv_config_t &cfg) {
    auto &plan = cfg.plan();
    ir_assert(cfg.slm().a() == plan.slm.has_a());
    ir_assert(cfg.slm().b() == plan.slm.has_b());
    ir_assert(cfg.pipeline().reuse_headers() == plan.reuse_headers);

#ifdef GEN_CONV_DEBUG
    auto dummy_mem(var_t::make(type_t::byte_ptr(), "mem"));
    auto dummy_reg(var_t::make(type_t::byte_ptr(), "reg"));
    if (!cfg.a_load_pattern.matches(
                plan.x2r.a_load.create_stmt(dummy_mem, dummy_reg))) {
        ir_warning() << "Generated load for tensor A does not match "
                     << cfg.a_load_pattern << " load idiom\n";
    }
    if (!cfg.b_load_pattern.matches(
                plan.x2r.b_load.create_stmt(dummy_mem, dummy_reg))) {
        ir_warning() << "Generated load for tensor B does not match "
                     << cfg.a_load_pattern << " load idiom\n";
    }
#endif
    return status::success;
}

status_t check_config(conv_config_t &cfg) {
    const auto &prb = cfg.prb();
    if (prb.is_fwd) {
        if (cfg.send_2d_nhwc() && prb.sw != 1 && (prb.kw != 1 || prb.pw != 0)) {
            int osp_iter_blk = cfg.iter_dim("osp") * cfg.iter_dim("ow");
            ir_assert(osp_iter_blk == 1)
                    << "Can't use 2D block messages for non-trivial "
                       "strided dimensions.";
        }
    } else if (prb.is_bwd_d) {
        if (cfg.send_2d_nhwc() && prb.mb < 16 && prb.sw != 1) {
            ir_assert(cfg.iter_dim("iw") == 1)
                    << "Can't use 2D block messages for non-trivial "
                       "strided dimensions.";
        }
    } else if (prb.is_bwd_w) {
        if (cfg.send_2d_nhwc() && prb.sw != 1 && (prb.kw != 1 || prb.pw != 0)) {
            ir_assert(cfg.iter_dim("ow") == 1)
                    << "Can't use 2D block messages for non-trivial "
                       "strided dimensions.";
        }
    }

    for (auto &kv : cfg.dims().get()) {
        auto &name = kv.first;
        int tg = cfg.thread_group_dim(name);
        int grid = cfg.grid_dim(name);
        if (tg != 1)
            ir_assert(in_grid_dims(get_thread_group_grid_conv_dims, prb, name))
                    << name;
        if (grid != 1)
            ir_assert(in_grid_dims(get_kernel_grid_conv_dims, prb, name))
                    << name;
    }
    if (cfg.with_plan()) CHECK(check_plan(cfg));
    return status::success;
}

status_t try_init_cfg(conv_config_t &cfg) {
    init_hint(cfg);
    init_send_2d_nhwc(cfg);
    init_fuse_spatial(cfg);
    init_hoist_masks_from_compute_loop(cfg);
    init_ow_kw_grf_cache(cfg);
    init_blocking(cfg);
    init_bwd_d_optimize_strided(cfg);
    init_pipeline(cfg);
    init_padded_dims(cfg);
    init_kernel_grid(cfg);
    init_thread_group_grid(cfg);
    CHECK(init_plan(cfg));
    init_unroll(cfg);
    init_slm(cfg);
    init_prefetch(cfg);
    init_allow_a_grf_reorder(cfg);
    init_allow_b_grf_reorder(cfg);
    init_allow_slm_tg_slicing(cfg);
    init_assign_sbids(cfg);
    init_subtiles(cfg);

    CHECK(fixup_config(cfg));
    CHECK(check_config(cfg));

    return status::success;
}

// Returns max SLM size per thread group assuming max utilization (max
// concurrent threads per EU).
int max_slm_size(const conv_config_t &cfg) {
    ngen::HW hw = cfg.hw();
    int regs = cfg.regs();
    return compute::device_info_t::max_slm_size_per_tg(
            convert_ngen_arch_to_dnnl(hw), regs > 128);
}

int slm_size(const conv_config_t &cfg) {
    const auto &prb = cfg.prb();
    auto &slm = cfg.slm();
    if (slm.bufs() == 0) return 0;

    bmnk_dim_helper_t h(cfg);
    int m_tg_blk = h.thread_group_dim('m') * h.iter_dim('m');
    int n_tg_blk = h.thread_group_dim('n') * h.iter_dim('n');
    int k_iter_blk = h.iter_dim('k');
    int a_slm_size = m_tg_blk * k_iter_blk * prb.a_data_type_size;
    int b_slm_size = n_tg_blk * k_iter_blk * prb.b_data_type_size;

    int ret = 0;
    if (slm.a()) ret += a_slm_size;
    if (slm.b()) ret += b_slm_size;
    ret *= slm.bufs();

    return ret;
}

status_t init_cfg(conv_config_t &cfg, const convolution_pd_t *pd) {
    cfg.set_pd(pd);

    // Try large GRF mode first.
    int try_regs = cfg.hw_cfg().large_grf_support() ? 256 : 128;
    //if (prb.g == 1 && prb.is_f32_conv()) try_regs = 128;

    int def_max_tg_size
            = get_default_max_tg_size(cfg.hw_cfg(), try_regs, cfg.simd());
    conv_hint_t hint(def_max_tg_size);

    // Use fixed iterations to avoid infinite loop.
    int max_iters = 10;
    bool ok = false;
    for (int i = 0; i < max_iters; i++) {
        conv_config_t try_cfg = cfg;
        try_cfg.set_regs(try_regs);
        try_cfg.set_hint(hint);

        CHECK(try_init_cfg(try_cfg));

        // Reduce thread group size if SLM size is too large.
        if (try_cfg.check_slm_size()) {
            if (slm_size(try_cfg) > max_slm_size(try_cfg)) {
                hint.set_max_tg_size(hint.max_tg_size() / 2);
                continue;
            }
        }

        // If the kernel fits 128 registers, switch to the normal mode which is
        // expected to have better performance for such cases.
        int bound = (!try_cfg.is_dp_fma() ? 128 : 116);
        int estimated_peak_regs = estimate_register_count(try_cfg);
        if (try_regs == 256 && estimated_peak_regs <= bound) {
            try_regs = 128;
            continue;
        }
        cfg = try_cfg;
        ok = true;
        break;
    }

    return ok ? status::success : status::runtime_error;
}

bool use_conv_plan(const conv_config_t &cfg) {
    return true;
}

conv_config_t::conv_config_t() = default;
conv_config_t::~conv_config_t() = default;

void conv_config_t::override_set(const std::string &s, bool is_env) {
    std::vector<param_t *> params;
    for (auto &gp : get_params_)
        params.push_back(gp(this));
    auto parts = ir_utils::split(s);
    for (auto &p : parts) {
        auto sub_parts = ir_utils::split(p, "=");
        ir_assert(sub_parts.size() == 2);
        auto &key = sub_parts[0];
        auto &value = sub_parts[1];
        bool found = false;
        for (auto *p : params) {
            if (p->accept_key(key)) {
                ir_info() << "Override " << p->name() << ": " << key << "="
                          << value << std::endl;
                p->override_set(key, value, is_env);
                found = true;
                break;
            }
        }
        if (!found) ir_warning() << "Unknown parameter: " << p << std::endl;
    }
}

int get_thread_count(const conv_config_t &cfg) {
    return cfg.kernel_grid().elems() * cfg.thread_group_grid().elems();
}

// Return thread utilization as a percentage. If this value is low,
// parallelism is a fundamental limitation to the current work scheduling.
float get_thread_utilization(const conv_config_t &cfg) {
    auto arch = convert_ngen_arch_to_dnnl(cfg.hw());
    int slice_eu_count = compute::device_info_t::max_eus_per_wg(arch);
    int slice_count = cfg.hw_cfg().eu_count() / slice_eu_count;

    int min_wg_per_slice_wave
            = std::max(slice_eu_count / cfg.thread_group_grid().elems(), 1);
    int min_wg_per_wave = slice_count * min_wg_per_slice_wave;
    int wg = cfg.kernel_grid().elems();
    return ((float)wg / utils::rnd_up(wg, min_wg_per_wave)) * 100;
}

// Return wave utilization as a percentage. If this value is low, memory
// latency may be an issue due to limited use of SMT to hide the latency.
float get_wave_utilization(const conv_config_t &cfg) {
    auto arch = convert_ngen_arch_to_dnnl(cfg.hw());
    int threads_per_eu = compute::device_info_t::threads_per_eu(
            arch, cfg.hw_cfg().large_grf_support());
    int slice_eu_count = compute::device_info_t::max_eus_per_wg(arch);
    int slice_count = cfg.hw_cfg().eu_count() / slice_eu_count;

    int max_wg_per_slice_wave
            = slice_eu_count * threads_per_eu / cfg.thread_group_grid().elems();
    int max_wg_per_wave = slice_count * max_wg_per_slice_wave;
    int wg = cfg.kernel_grid().elems();
    return ((float)wg / utils::rnd_up(wg, max_wg_per_wave)) * 100;
}

std::string conv_config_t::str() const {
    using namespace ir_utils;

    std::ostringstream oss;
    // clang-format off
    oss << "  HW config:                  " << exec_cfg().str(hint().max_tg_size()) << std::endl;
    oss << "  Problem:                    " << prb().desc_str() << std::endl;
    const char *names[] = {"Source", "Weights", "Destination"};
    const layout_param_t *layouts[] = {&src_layout(), &wei_layout(), &dst_layout()};
    for (int i = 0; i < 3; i++) {
        std::string desc = std::string(names[i]) + " layout:";
        desc.insert(desc.size(), 28 - desc.size(), ' ');
        auto &compute_layout = layouts[i]->compute_unnormalized();
        auto &user_layout = layouts[i]->user_unnormalized();
        oss << "  " << desc << compute_layout;
        if (user_layout != compute_layout) {
            oss << " (user: " << user_layout << ")";
        }
        oss << std::endl;
    }
    int estimated_peak_regs = estimate_register_count(*this);
    oss << blocking_brief_str();
    oss << "  Kernel grid:                " << kernel_grid() << std::endl;
    oss << "  Thread group:               " << thread_group_grid() << std::endl;
    oss << "  Threads:                    " << get_thread_count(*this) << " (utilization: "
        << get_thread_utilization(*this) << "% thread, "
        << get_wave_utilization(*this) << "% wave)" <<  std::endl;
    oss << "  FMA kind:                   " << fma_kind::to_string(fma_kind()) << std::endl;
    oss << "  SLM buffering:              " << "A: " << to_string(slm().a()) << ", B: " << to_string(slm().b())
                                            << ", buffers: " << slm().bufs() << ", pad: " << to_string(pad_slm()) << std::endl;
    oss << "  GRF buffers for GMEM load:  " << slm().gmem_bufs() << std::endl;
    oss << "  Prefetch:                   " << to_string(prefetch()) << ", buffers: " << prefetch().bufs() << std::endl;
    oss << "  Do pipeline unroll:         " << to_string(pipeline().do_unroll()) << std::endl;
    oss << "  Assign SBIDs:               " << to_string(assign_sbids()) << std::endl;
    oss << "  Reuse headers:              " << to_string(pipeline().reuse_headers()) << std::endl;
    oss << "  Allow GRF reorder:          " << "A: " << to_string(allow_a_grf_reorder()) << ", B: " << to_string(allow_b_grf_reorder()) << std::endl;
    oss << "  Subtiles:                   " << "A: " << subtiles().a() << ", B: " << subtiles().b() << std::endl;
    oss << "  Estimated GRF usage:        " << estimated_peak_regs << std::endl;
    oss << "  Use plan:                   " << to_string(with_plan()) << std::endl;
    // clang-format on
    return oss.str();
}

std::string pad_str(std::string s, int pad) {
    auto pos = (pad >= 0 ? 0 : s.length());
    int off = std::abs(pad) - (int)s.length();
    s.insert(pos, std::max(off, 0), ' ');
    return s;
}

std::string pad_int(int i, int pad) {
    return pad_str(std::to_string(i), pad);
}

std::string conv_config_t::blocking_brief_str() const {
    std::ostringstream oss;
    std::vector<std::string> names;
    for (auto &kv : dims().get()) {
        names.push_back(kv.first);
    }
    std::sort(names.begin(), names.end());
    for (auto &name : names) {
        int iter = iter_dim(name);
        int tg = thread_group_dim(name);
        int loop = loop_dim(name);
        int grid = grid_dim(name);
        if (iter == 1 && loop == 1 && tg == 1) continue;
        oss << "  Dimension " << name << pad_str(":", -18 + (int)name.length());
        oss << "(grid:" << pad_int(grid, 5) << ") x ";
        oss << "(tg:" << pad_int(tg, 5) << ") x ";
        oss << "(loop:" << pad_int(loop, 5) << ") x ";
        oss << "(iter:" << pad_int(iter, 5) << ")\n";
    }
    return oss.str();
}

void conv_config_t::set_plan(const std::shared_ptr<conv_plan_t> &plan) {
    if (!use_conv_plan(*this)) return;
    plan_ = plan;
}

bool conv_config_t::with_plan() const {
    return (bool)plan_;
}

const conv_plan_t &conv_config_t::plan() const {
    ir_assert(with_plan());
    return *plan_;
}

bool conv_config_t::can_skip_wei_zero_out() const {
    if (!prb().is_bwd_w) return true;
    bmnk_dim_helper_t h(*this);
    int k_iter_dim = h.iter_dim('k');
    int k_loop_dim = h.loop_dim('k');
    int k_tg_dim = h.thread_group_dim('k');
    int k_tg_block = k_iter_dim * k_loop_dim * k_tg_dim;
    int k_padded = padded_dim("mb") * padded_dim("od") * padded_dim("oh")
            * padded_dim("ow");
    return k_tg_block >= k_padded;
}

bool conv_config_t::can_skip_bia_zero_out() const {
    if (!prb().is_bwd_w || !prb().with_bias) return true;
    return can_skip_wei_zero_out() && !slm().b();
}

void init_extra_tensors(const conv_config_t &cfg, tensor_config_t &tensor_cfg) {
    const auto &prb = cfg.prb();
    auto &zp_cfg = prb.zp_cfg;
    auto *pd = prb.conv_pd;
    auto *attr = prb.attr;
    if (zp_cfg.do_src_compensation && zp_cfg.is_runtime_src_zero_points) {
        int zp_ic = (zp_cfg.is_common_src_zero_point) ? 1 : prb.ic;
        std::vector<dim_t> dims = {zp_ic};
        layout_t zp_layout(type_t::s32(), 0, dims);
        int arg_key = DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC;
        tensor_cfg.add_tensor("src_zero_points", arg_key,
                /*is_input=*/true, /*is_output=*/false, zp_layout);
    }
    if (zp_cfg.do_dst_compensation && zp_cfg.is_runtime_dst_zero_points) {
        std::vector<dim_t> dims = {prb.oc};
        layout_t zp_layout(type_t::s32(), 0, dims);
        int arg_key = DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST;
        tensor_cfg.add_tensor("dst_zero_points", arg_key,
                /*is_input=*/true, /*is_output=*/false, zp_layout);
    }
    auto scale_args = get_scale_args(prb);
    const char *scale_names[] = {"src_scales", "wei_scales", "dst_scales"};
    const int scale_names_len = sizeof(scale_names) / sizeof(scale_names[0]);
    ir_assert((int)scale_args.size() == scale_names_len);
    for (int i = 0; i < (int)scale_args.size(); i++) {
        int arg = scale_args[i];
        auto &s = attr->scales_.get(arg);
        if (s.has_default_values()) continue;
        int dim = s.mask_ == 0 ? 1 : (prb.is_fwd ? prb.oc : prb.ic);
        std::vector<dim_t> dims = {dim};
        layout_t layout(type_t::f32(), 0, dims);
        int arg_key = DNNL_ARG_ATTR_SCALES | arg;
        tensor_cfg.add_tensor(scale_names[i], arg_key, /*is_input=*/true,
                /*is_output=*/false, layout);
    }
    for (int i = 0; i < attr->post_ops_.len(); i++) {
        auto &po = attr->post_ops_.entry_[i];
        if (po.is_eltwise()
                || po.is_sum(/*require_scale_one=*/false,
                        /*require_zp_zero=*/false)) {
            // No extra tensors.
        } else if (po.is_binary()) {
            auto layout = make_layout(po.binary.src1_desc);
            int arg_key = DNNL_ARG_ATTR_MULTIPLE_POST_OP(i) | DNNL_ARG_SRC_1;
            tensor_cfg.add_tensor("binary_rhs_" + std::to_string(i), arg_key,
                    /*is_input=*/true,
                    /*is_output=*/false, layout);
        } else if (po.is_prelu()) {
            layout_t layout(type_t::f32(), 0,
                    get_prelu_weights_dims(
                            po.prelu.mask, *pd->invariant_dst_md()));
            int arg_key = DNNL_ARG_ATTR_MULTIPLE_POST_OP(i) | DNNL_ARG_WEIGHTS;
            tensor_cfg.add_tensor("prelu_rhs_" + std::to_string(i), arg_key,
                    /*is_input=*/true, /*is_output=*/false, layout);
        } else {
            ir_error_not_expected();
        }
    }
}

tensor_config_t get_tensor_config(const conv_config_t &cfg) {
    const auto &prb = cfg.prb();
    tensor_config_t tensor_cfg;
    conv_arg_helper_t h(prb);
    auto &src = cfg.src_layout();
    auto &wei = cfg.wei_layout();
    auto &bia = cfg.bia_layout();
    auto &dst = cfg.dst_layout();
    tensor_cfg.add_tensor("src", h.src_arg_key(), h.is_src_input(),
            h.is_src_output(), src.compute(), src.user());
    tensor_cfg.add_tensor("wei", h.wei_arg_key(), h.is_wei_input(),
            h.is_wei_output(), wei.compute(), wei.user());
    if (prb.with_bias)
        tensor_cfg.add_tensor("bia", h.bia_arg_key(), h.is_bia_input(),
                h.is_bia_output(), bia.compute(), bia.user());
    tensor_cfg.add_tensor("dst", h.dst_arg_key(), h.is_dst_input(),
            h.is_dst_output(), dst.compute(), dst.user());
    if (prb.is_bwd_w && !prb.with_sum) {
        tensor_cfg.require_zero_out("wei");
        if (prb.with_bias) tensor_cfg.require_zero_out("bia");
    }
    init_extra_tensors(cfg, tensor_cfg);
    return tensor_cfg;
}

int estimate_register_count(const conv_config_t &cfg) {
    if (cfg.with_plan()) return cfg.plan().grf_usage().total();
    return estimate_grf_usage(cfg).total();
}

bool can_use_a_2d_send(const conv_config_t &cfg) {
    return can_use_2d_send(cfg, cfg.a_layout().compute_unnormalized(), true);
}

bool can_use_b_2d_send(const conv_config_t &cfg) {
    return can_use_2d_send(cfg, cfg.b_layout().compute_unnormalized(), false);
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
