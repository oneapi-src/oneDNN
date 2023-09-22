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
#include <mutex>

#include "gpu/jit/conv/grf_usage.hpp"
#include "gpu/jit/conv/message_patterns.hpp"
#include "gpu/jit/conv/normalization.hpp"
#include "gpu/jit/conv/params.hpp"
#include "gpu/jit/conv/plan.hpp"
#include "gpu/jit/conv/tiler.hpp"
#include "gpu/jit/ir/block_2d_utils.hpp"
#include "gpu/jit/ir/gemm_schedule.hpp"
#include "gpu/jit/ir/tensor_config.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

// Helper functions.
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

memory_desc_t &get_src_md(convolution_pd_t *pd) {
    return *const_cast<memory_desc_t *>(pd->invariant_src_md());
}

memory_desc_t &get_wei_md(convolution_pd_t *pd) {
    return *const_cast<memory_desc_t *>(pd->invariant_wei_md());
}

memory_desc_t &get_dst_md(convolution_pd_t *pd) {
    return *const_cast<memory_desc_t *>(pd->invariant_dst_md());
}

memory_desc_t &get_bia_md(convolution_pd_t *pd) {
    return *const_cast<memory_desc_t *>(pd->invariant_bia_md());
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

int get_default_mad_block(const type_t &type) {
    switch (type.size()) {
        case 1: return 32;
        case 2:
        case 4: return 16;
        case 8: return 8;
        default: ir_error_not_expected() << type;
    }
    return 1;
}

bool is_small(const type_t &type, int elems) {
    int block = get_default_mad_block(type);
    return elems <= block / 2;
}

bool is_small_ic(const conv_problem_t &prb) {
    return is_small(prb.src_data_type, prb.ic);
}

bool is_small_oc(const conv_problem_t &prb) {
    return is_small(prb.src_data_type, prb.oc);
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
    auto *gpu_attr = utils::downcast<gpu_primitive_attr_t *>(
            conv_pd->attr()->gpu_attr_.get());
    bool large_grf_mode = gpu_attr && gpu_attr->threads_per_eu() == 4;

    hw_config_t hw_cfg(engine, large_grf_mode);

    init_transpose(hw_cfg);
    CHECK(init_abc_data_types(hw_cfg));
    CHECK(init_acc_data_type());

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
    bool is_square = !has_d && ir_utils::is_equal(xh, xw);
    bool is_cubic = ir_utils::is_equal(xd, xh) && ir_utils::is_equal(xd, xw);
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

int param_t::sort_key() const {
    static const char *ordered_params[] = {
            "exec-cfg",
            "fma",
            "l",
            "T",
            "i",
            "P",
            "p",
            "s",
            "src",
            "wei",
            "dst",
            "bia",
            nullptr,
    };
    for (const char **p = ordered_params; *p; p++) {
        if (short_name() == *p) return p - ordered_params;
    }
    return (int)(sizeof(ordered_params) / sizeof(ordered_params[0]));
}

const bwd_d_optimize_kind_t bwd_d_optimize_kind_param_t::default_value
        = bwd_d_optimize_kind_t::none;
const bool pad_slm_param_t::default_value = true;

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

int get_default_block(fma_kind_t fma, const type_t &type, int elems) {
    if (is_dp_fma(fma)) {
        if (is_small(type, elems)) {
            int packed_dword_elems = 4 / type.size();
            return std::max(utils::rnd_up_pow2(elems), packed_dword_elems);
        }
        return 32 / type.size();
    }
    if (is_small(type, elems)) return 1;
    return get_default_mad_block(type);
}

fma_kind_t get_default_fma(ngen::HW hw, const type_t &type) {
    switch (type.size()) {
        case 1:
            if (hw >= ngen::HW::XeHP) return fma_kind_t::dpas;
            return hw >= ngen::HW::XeLP ? fma_kind_t::dp4a : fma_kind_t::mad;
        case 2:
            return hw >= ngen::HW::XeHP ? fma_kind_t::dpas : fma_kind_t::mad;
        default: return fma_kind_t::mad;
    }
    return fma_kind_t::unknown;
}

struct nc_block_t {
    nc_block_t(int n_block, int c_block)
        : n_block_(n_block), c_block_(c_block) {}

    std::string tag() const {
        std::vector<int> idxs = {1, 0};
        return build_tag({n_block_, c_block_}, {1, 1}, {'a', 'b'}, idxs);
    }

    // Ideally, this should only depend on data type, direction, mb, c, and g to
    // enable the same src/dst formats and avoid reorders between convolutions
    static nc_block_t get_default_blocking(ngen::HW hw, fma_kind_t fma,
            type_t type, bool is_dw, int n, int c, int g,
            bool is_output = false) {
        // Select dst layout to align with fma kind of following conv
        // for non-depthwise cases.
        fma_kind_t tmp_fma
                = (is_output && !is_dw) ? get_default_fma(hw, type) : fma;
        int c_block = (is_dw ? get_default_block(tmp_fma, type, g)
                             : get_default_block(tmp_fma, type, c));
        if (g > 1 && !is_dw) {
            if (c % c_block != 0) c_block = 1;
            // Try to use the same layout between group/non-group convolution
            // to avoid reorder.
            auto default_gc_blk
                    = get_default_block(get_default_fma(hw, type), type, g * c);
            if (c_block != default_gc_blk) {
                if (default_gc_blk % c == 0 && g % (default_gc_blk / c) == 0) {
                    c_block = default_gc_blk;
                }
            }
        }
        auto default_n_blk = (type.size() <= 2) ? 32 : 16;
        int n_block = (c_block == 1) ? 1 : pick_block(n, 16, default_n_blk);
        return nc_block_t(n_block, c_block);
    }

private:
    int n_block_;
    int c_block_;
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
            fma_kind_t fma_kind, bool is_bwd_d, int g, int o, int i,
            bool ab_transpose) {
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
        get_default_blocking(type, vec_size, fma_kind, is_bwd_d, g, x, y,
                g_block, *x_block, *y_block, *x_block_outer, *y_block_outer,
                ab_transpose);
        return goi_block_t(fma_kind, is_dw(g, o, i), is_bwd_d, g_block, o_block,
                i_block, o_block_outer, i_block_outer);
    }

    static void get_default_blocking(type_t type, int vec_size,
            fma_kind_t fma_kind, bool is_bwd_d, int g, int x, int y,
            int &g_block, int &x_block, int &y_block, int &x_block_outer,
            int &y_block_outer, bool ab_transpose = false) {
        if (is_dw(g, x, y)) {
            g_block = vec_size;
        } else if (fma_kind == fma_kind_t::mad) {
            x_block = (ab_transpose && is_bwd_d) ? utils::rnd_up_pow2(x)
                                                 : vec_size;
            y_block = get_default_block(fma_kind, type, y);
        } else {
            int packed_dword_elems = 4 / type.size();
            x_block = ab_transpose ? utils::rnd_up_pow2(x) : vec_size;
            y_block = packed_dword_elems;
            // Fixing y outer block helps to avoid extra GRF reorders however
            // in small reduction cases it may result in excessive zero
            // padding. In such cases fused reduction can be used. E.g. in
            // non-1x1 small-ic fwd convolution kw and ic can be fused.
            if (y * type.size() >= 32) {
                if (ab_transpose) {
                    y_block *= 8;
                } else {
                    y_block_outer = 8;
                }
            }
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

// Matches the user-provided descriptor against the list of supported plain tags.
std::string get_plain_user_tag(
        const conv_problem_t &prb, const memory_desc_t &md, bool is_wei) {
    if (is_wei) {
        std::vector<const char *> plain_non_group_wei_tags
                = {"abx", "axb", "xba"};
        std::vector<const char *> plain_group_wei_tags
                = {"abcx", "abxc", "axcb"};
        auto &plain_wei_tags = (prb.with_groups ? plain_group_wei_tags
                                                : plain_non_group_wei_tags);
        ir_assert(
                plain_non_group_wei_tags.size() == plain_group_wei_tags.size());
        for (size_t i = 0; i < plain_wei_tags.size(); i++) {
            if (matches_tag(md, plain_wei_tags[i])) {
                return plain_non_group_wei_tags[i];
            }
        }
    } else {
        for (auto *t : {"axb", "abx"}) {
            if (matches_tag(md, t)) return t;
        }
    }
    return {};
}

std::string maybe_fixup_1st_conv_wei_tag(
        const conv_config_t &cfg, const std::string &tag) {
    auto &prb = cfg.prb();

    if (!cfg.is_dp_fma()) return tag;
    if (!is_small_ic(prb) || prb.is_dw) return tag;
    if (prb.ab_swap_transpose) return tag;
    if (!prb.is_fwd) return tag;

    // Use OhwIXoYi weights for small-channel forward convolution to ensure
    // c-after-w order of reduction blocks to match the source layout.
    const char *patterns[] = {"ABx", "AxB", "Abx", "Axb", nullptr};
    for (auto *p = patterns; *p; p += 2) {
        auto pos = tag.find(*p);
        if (pos == std::string::npos) continue;
        auto ret = tag;
        return ret.replace(pos, std::strlen(*p), *(p + 1));
    }
    ir_error_not_expected() << tag;
    return tag;
}

void maybe_set_plain_weights(const conv_config_t &cfg, bool src_dst_axb,
        const std::string &user_wei_req, std::string &wei_tag,
        std::string &user_wei_tag) {
    auto &prb = cfg.prb();
    // For XeHPC+ with nhwc activations always use hwio weights for
    // consistency for user-facing layouts.
    if (cfg.hw() >= ngen::HW::XeHPC && src_dst_axb) {
        bool channels_ok = (prb.ic % 16 == 0 && prb.oc % 16 == 0);
        if (prb.g == 1 && channels_ok) {
            // Use plain compute layout for weights, normally they are
            // supported via block 2D load.
            wei_tag = (prb.is_bwd_d ? "xab" : "xba");
            if (user_wei_req.empty()) user_wei_tag = "xba";
        }
    }
    if (user_wei_tag.empty()) user_wei_tag = user_wei_req;
}

void init_data_tags(const conv_config_t &cfg, const memory_desc_t &src_md,
        const memory_desc_t &wei_md, const memory_desc_t &dst_md,
        std::string &src_tag, std::string &wei_tag, std::string &dst_tag,
        std::string &user_src_tag, std::string &user_wei_tag,
        std::string &user_dst_tag) {
    const auto &prb = cfg.prb();
    auto src_compute_type = prb.is_bwd_d ? prb.c_data_type : prb.a_data_type;
    auto dst_compute_type = prb.is_fwd
            ? prb.c_data_type
            : (prb.is_bwd_d ? prb.a_data_type : prb.b_data_type);
    auto wei_compute_type = prb.is_bwd_w ? prb.c_data_type : prb.b_data_type;

    auto src_blk = nc_block_t::get_default_blocking(cfg.hw(), cfg.fma_kind(),
            src_compute_type, prb.is_dw, prb.mb, prb.ic, prb.g,
            /*is_output=*/prb.is_bwd_d);
    auto dst_blk = nc_block_t::get_default_blocking(cfg.hw(), cfg.fma_kind(),
            dst_compute_type, prb.is_dw, prb.mb, prb.oc, prb.g,
            /*is_output=*/prb.is_fwd);
    auto wei_blk = goi_block_t::get_default_blocking(wei_compute_type,
            cfg.vec_size(), cfg.fma_kind(), prb.is_bwd_d, prb.g, prb.oc, prb.ic,
            prb.ab_swap_transpose);

    src_tag = src_blk.tag();
    wei_tag = wei_blk.tag();
    dst_tag = dst_blk.tag();

    wei_tag = maybe_fixup_1st_conv_wei_tag(cfg, wei_tag);

    // Handle nhwc case.
    auto user_src_req = get_plain_user_tag(prb, src_md, /*is_wei=*/false);
    auto user_wei_req = get_plain_user_tag(prb, wei_md, /*is_wei=*/true);
    auto user_dst_req = get_plain_user_tag(prb, dst_md, /*is_wei=*/false);
    bool src_axb = (user_src_req == "axb");
    bool dst_axb = (user_dst_req == "axb");
    bool src_abx = (user_src_req == "abx");
    bool dst_abx = (user_dst_req == "abx");
    bool src_matches = matches_tag(src_md, src_tag);
    bool dst_matches = matches_tag(dst_md, dst_tag);
    bool src_output = prb.is_bwd_d;
    bool dst_output = prb.is_fwd;
    bool is_small_ic_g1 = is_small_ic(prb) && (prb.g == 1);
    bool is_small_oc_g1 = is_small_oc(prb) && (prb.g == 1);

    // Use nhwc for compute for non-small channels to avoid reorders.
    if (!src_matches && !is_small_ic_g1 && src_axb) src_tag = "axb";
    if (!dst_matches && !is_small_oc_g1 && dst_axb) dst_tag = "axb";

    // Use plain tags for user-facing activations for small-channel tensors.
    if (!matches_tag(src_md, src_tag) && is_small_ic_g1)
        user_src_tag = (user_src_req.empty() ? "axb" : user_src_req);
    if (!matches_tag(dst_md, dst_tag) && is_small_oc_g1)
        user_dst_tag = (user_dst_req.empty() ? "axb" : user_dst_req);

    maybe_set_plain_weights(
            cfg, src_axb && dst_axb, user_wei_req, wei_tag, user_wei_tag);

    // Use plain tag for output to avoid extra reorders.
    if (!user_src_tag.empty() && src_output) src_tag = user_src_tag;
    if (!user_dst_tag.empty() && dst_output) dst_tag = user_dst_tag;

    if (user_src_tag.empty()) user_src_tag = src_tag;
    if (user_wei_tag.empty()) user_wei_tag = wei_tag;
    if (user_dst_tag.empty()) user_dst_tag = dst_tag;
    if (src_abx && !src_matches) user_src_tag = "abx";
    if (dst_abx && !dst_matches) user_dst_tag = "abx";
}

status_t init_tensor_layouts(conv_config_t &cfg, convolution_pd_t *pd) {
    const auto &prb = cfg.prb();
    // Compute layout tags and user layout tags. If a compute layout is
    // different from a user layout then an extra pre/post reorder will be
    // executed before/after convolution.
    std::string src_tag, user_src_tag;
    std::string wei_tag, user_wei_tag;
    std::string dst_tag, user_dst_tag;
    std::string bia_tag = "a";
    std::string user_bia_tag = "a";

    auto &src_md = get_src_md(pd);
    auto &wei_md = get_wei_md(pd);
    auto &dst_md = get_dst_md(pd);
    auto &bia_md = get_bia_md(pd);

    // If src/dst is nhwc then set the other one with any to nhwc too (except
    // 1st convolution).
    bool is_small_ic_non_dw = is_small_ic(prb) && !prb.is_dw;
    bool propagate_nhwc = (matches_tag(src_md, "axb") && !is_small_ic_non_dw)
            || matches_tag(dst_md, "axb");
    if (propagate_nhwc) {
        set_default_format(src_md, "axb");
        set_default_format(dst_md, "axb");
    }

    init_data_tags(cfg, src_md, wei_md, dst_md, src_tag, wei_tag, dst_tag,
            user_src_tag, user_wei_tag, user_dst_tag);

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
        src_tag = src.compute_unnormalized_tag();
        user_src_tag = src.user_unnormalized_tag();
    }
    if (wei.is_overridden()) {
        wei_tag = wei.compute_unnormalized_tag();
        user_wei_tag = wei.user_unnormalized_tag();
    }
    if (dst.is_overridden()) {
        dst_tag = dst.compute_unnormalized_tag();
        user_dst_tag = dst.user_unnormalized_tag();
    }

    // Select user layouts.
    auto user_src_layout = init_layout(src_md, user_src_tag);
    auto user_wei_layout = init_layout(wei_md, user_wei_tag);
    auto user_dst_layout = init_layout(dst_md, user_dst_tag);

    layout_t user_bia_layout;
    if (prb.with_bias) user_bia_layout = init_layout(bia_md, user_bia_tag);

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
        if (utils::one_of(prb.wei_data_type, data_type::bf16, data_type::f16))
            wei_layout = wei_layout.retype(type_t::f32());
        if (utils::one_of(prb.bia_data_type, data_type::bf16, data_type::f16))
            bia_layout = bia_layout.retype(type_t::f32());
    }

    src.set_compute_unnormalized(src_layout, src_tag);
    src.set_user_unnormalized(user_src_layout, user_src_tag);
    wei.set_compute_unnormalized(wei_layout, wei_tag);
    wei.set_user_unnormalized(user_wei_layout, user_wei_tag);
    dst.set_compute_unnormalized(dst_layout, dst_tag);
    dst.set_user_unnormalized(user_dst_layout, user_dst_tag);
    bia.set_compute_unnormalized(bia_layout, bia_tag);
    bia.set_user_unnormalized(user_bia_layout, user_bia_tag);

    // Normalize layouts: add group dimension for all layouts and reduce/fuse
    // spatial dimensions when applicable.
    normalize_conv_layouts(src_layout, wei_layout, dst_layout, bia_layout,
            prb.with_groups, prb.g, prb.ic, prb.oc, prb.is_dw, prb.reduced_dim,
            /*add_groups=*/true);
    normalize_conv_layouts(user_src_layout, user_wei_layout, user_dst_layout,
            user_bia_layout, prb.with_groups, prb.g, prb.ic, prb.oc, prb.is_dw,
            prb.reduced_dim,
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
    if (hw_cfg.hw() < ngen::HW::Gen9) return false;
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
    bool is_xelpg
            = hw_cfg.hw() == ngen::HW::XeHPG && !hw_cfg.systolic_support();
    if (is_bf16 && (hw_cfg.hw() <= ngen::HW::XeLP || is_xelpg)) return false;
    if (prb.is_f64_conv()
            && (utils::one_of(hw_cfg.hw(), ngen::HW::XeLP, ngen::HW::XeHPG)
                    && !is_xelpg))
        return false;
    if (prb.is_fwd) return true;
    if (prb.is_bwd_d) return true;
    if (prb.is_bwd_w) {
        bool ok = true;
        data_type_t default_acc_type
                = src == data_type::f64 ? data_type::f64 : data_type::f32;
        ok &= utils::one_of(src, data_type::bf16, data_type::f16,
                data_type::f32, data_type::f64);
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

bool post_ops_ok(const conv_problem_t &prb, const hw_config_t &hw_cfg) {
    auto *pd = prb.conv_pd;
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

    using namespace data_type;
    const auto input_type = (prb.is_fwd) ? pd->invariant_src_md()->data_type
                                         : pd->invariant_dst_md()->data_type;
    if (!attr->post_ops_.check_sum_consistency(
                prb.c_data_type, utils::one_of(input_type, s8, u8), true))
        return false;

    if (!attr->scales_.has_default_values())
        if (!prb.is_s32_accumulator()) return false;
    auto scale_args = get_scale_args();
    std::vector<int> scales(scale_args.size());
    for (int i = 0; i < (int)scale_args.size(); i++)
        scales[i] = scale_args[i].second;
    if (!attr->scales_.has_default_values(scales)) return false;
    for (int arg : scales) {
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
                    && hw_cfg.systolic_support() && hw_cfg.eu_count() <= 128)
                // Workaround for hard to reproduce issue in end to end
                // workloads. It is unclear what the actual issue is as the
                // kernel always works correctly in benchdnn.
                return false;
        }
    }
    return true;
}

void maybe_override_from_env(conv_config_t &cfg) {
#ifdef DNNL_DEV_MODE
    auto cfg_env = gpu_utils::dev_getenv("cfg", std::string());
    if (cfg_env.empty()) return;
    cfg.override_set(cfg_env, /*is_env=*/true);
#else
    UNUSED(cfg);
#endif
}

status_t init_fma_kind(conv_config_t &cfg) {
    if (cfg.fma_kind_param().is_overridden()) return status::success;
    const auto &prb = cfg.prb();
    auto fma_kind = fma_kind::get_supported_kind(
            cfg.hw_cfg(), prb.a_data_type, prb.b_data_type, prb.acc_data_type);
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
        int vec_dim = prb.ab_swap_transpose
                ? ((prb.is_bwd_w) ? prb.ic : prb.mb)
                : ((prb.is_fwd || prb.is_bwd_w) ? prb.oc : prb.ic);
        if (utils::rnd_up(vec_dim, grf_elems) < vec_size) vec_size = grf_elems;
    }
    // SIMD32 produces invalid layouts in bwd_w.
    if (prb.is_bwd_w && !cfg.is_dpas_or_dpasw_fma()) {
        if (prb.is_f64_conv()) {
            vec_size = std::min(vec_size, 8);
        } else {
            vec_size = std::min(vec_size, 16);
        }
    }
    cfg.set_vec_size(vec_size);
    return status::success;
}

int default_regs(const conv_config_t &cfg) {
    if (!cfg.hw_cfg().large_grf_support()) return 128;
    if (cfg.is_dpas_or_dpasw_fma()) return 256;
    return 128;
}

status_t init_regs(conv_config_t &cfg) {
    if (cfg.exec_cfg_param().is_overridden("regs")) return status::success;

    cfg.set_regs(default_regs(cfg));
    return status::success;
}

bool post_op_layouts_ok(const conv_problem_t &prb) {
    auto *pd = prb.conv_pd;
    auto *attr = pd->attr();

    auto &output_md = prb.c_md();
    for (int i = 0; i < attr->post_ops_.len(); i++) {
        auto &po = attr->post_ops_.entry_[i];
        if (po.is_binary() || po.is_prelu()) {
            int mask = po.is_prelu()
                    ? po.prelu.mask
                    : utils::get_dims_mask(output_md.dims,
                            po.binary.src1_desc.dims, prb.ndims, true);
            // These cases don't have message-related limitations.
            if ((mask & (1 << 1)) == 0 || mask == (1 << 1)) continue;
            auto rhs_layout = po.is_prelu()
                    ? layout_t(type_t::f32(), 0,
                            get_prelu_weights_dims(po.prelu.mask, output_md))
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

bwd_d_optimize_kind_t bwd_d_optimize_kind_hint(const conv_problem_t &prb) {
    bool with_dilation = prb.dh || prb.dw || prb.dd;
    if (!prb.is_bwd_d || with_dilation) return bwd_d_optimize_kind_t::none;
    if (prb.is_stride1()) {
        // Count how many out-of-bound iw updates are applied.
        int oob_updates = 0;
        int iw_oob_idx = prb.ow - prb.pw;
        for (int iw = iw_oob_idx; iw < prb.iw; iw++) {
            for (int kw = 0; kw < prb.kw; kw++) {
                if (iw + prb.pw - kw * (1 + prb.dw) >= prb.ow) oob_updates++;
            }
        }
        double eff = 1 - oob_updates / (prb.iw * (double)prb.kw);
        if (eff < 0.85) return bwd_d_optimize_kind_t::skip_out_of_bound_w;
        return bwd_d_optimize_kind_t::none;
    }

    auto hint = bwd_d_optimize_kind_t::skip_strided_dhw;
    if (prb.iw % prb.sw != 0 || prb.mb < 16)
        hint = bwd_d_optimize_kind_t::skip_strided_dh;
    return hint;
}

// Enable optimization for strided BWD_D convolution.
void init_bwd_d_optimize(conv_config_t &cfg) {
    if (cfg.bwd_d_optimize_kind_param().is_overridden()) return;

    auto hint = bwd_d_optimize_kind_hint(cfg.prb());
    cfg.set_bwd_d_optimize_kind(hint);
}

status_t init_pd_time_cfg(const conv_problem_t &prb, conv_config_t &cfg,
        const engine_t *engine, convolution_pd_t *pd, primitive_attr_t *attr) {
    auto *gpu_attr
            = utils::downcast<gpu_primitive_attr_t *>(attr->gpu_attr_.get());
    bool large_grf_mode = gpu_attr && gpu_attr->threads_per_eu() == 4;
    hw_config_t hw_cfg(engine, large_grf_mode);

    if (!hw_ok(hw_cfg)) return status::unimplemented;
    if (!data_types_ok(prb, hw_cfg)) return status::unimplemented;
    if (!post_ops_ok(prb, hw_cfg)) return status::unimplemented;
    if (!zero_points_ok(prb)) return status::unimplemented;

    zero_points_config_t zp_cfg(pd);
    cfg.set_zp_cfg(zp_cfg);
    cfg.set_prb(prb);
    cfg.set_exec_cfg(exec_config_t(hw_cfg));

    maybe_override_from_env(cfg);

    CHECK(init_fma_kind(cfg));
    CHECK(init_simd(cfg));
    CHECK(init_vec_size(cfg));
    CHECK(init_regs(cfg));
    CHECK(init_tensor_layouts(cfg, pd));

    CHECK(attr->set_default_formats(&prb.c_md()));

    if (!post_op_layouts_ok(prb)) return status::unimplemented;

    init_bwd_d_optimize(cfg);

    return status::success;
}

bool pipeline_unroll_hint(const conv_problem_t &prb, fma_kind_t fma_kind,
        const exec_config_t &exec_cfg,
        bwd_d_optimize_kind_t bwd_d_optimize_kind) {
    bool do_unroll = true;
    if (prb.is_fwd) {
        const int max_unroll = exec_cfg.hw() <= ngen::HW::XeLP ? 4 : 9;
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
        if (!prb.is_stride1()
                && bwd_d_optimize_kind
                        != bwd_d_optimize_kind_t::skip_strided_dhw)
            do_unroll = false;
    }
    // Unrolling with mad or dp4a results in too large kernels.
    if (utils::one_of(fma_kind, fma_kind_t::mad, fma_kind_t::dp4a)
            && (exec_cfg.hw() >= ngen::HW::XeHPG || prb.mb != 1))
        do_unroll = false;
    return do_unroll;
}

void init_pipeline(conv_config_t &cfg) {
    if (cfg.pipeline().is_overridden()) return;

    bool do_unroll = pipeline_unroll_hint(cfg.prb(), cfg.fma_kind(),
            cfg.exec_cfg(), cfg.bwd_d_optimize_kind());
    if (cfg.plan().reuse_headers) do_unroll = false;
    cfg.pipeline().set(do_unroll, cfg.plan().reuse_headers);
}

send_pattern_t validate_blocking(
        const conv_config_t &cfg, conv_stride_layout_t::input_tensor_t tensor) {
    auto &prb = cfg.prb();
    const compute::gpu_arch_t arch
            = convert_ngen_arch_to_dnnl(cfg.hw_cfg().hw());

    auto is_match = [&](const block_hint_t<conv_dim_t> &hint) {
        for (auto dim : get_conv_dims(prb.prop_kind())) {
            if (hint[dim]) {
                if (cfg.iter_dim(dim) % hint[dim]) return false;
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
            if (is_match(hint)) {
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

void init_params(conv_config_t &cfg) {
    cfg.tiler().set_params(cfg);
}

const conv_tile_t *get_kernel_grid_conv_dims(
        const conv_problem_t &prb, int idx) {
    static const conv_tile_t fwd_0({conv_dims::oc});
    static const conv_tile_t fwd_1(
            {conv_dims::g, conv_dims::od, conv_dims::oh, conv_dims::ow});
    static const conv_tile_t fwd_2({conv_dims::mb});
    static const conv_tile_t bwd_d_0({conv_dims::ic});
    static const conv_tile_t bwd_d_1(
            {conv_dims::g, conv_dims::id, conv_dims::ih, conv_dims::iw});
    static const conv_tile_t bwd_d_2({conv_dims::mb});
    static const conv_tile_t bwd_w_0({conv_dims::oc});
    static const conv_tile_t bwd_w_1(
            {conv_dims::ic, conv_dims::kd, conv_dims::kh, conv_dims::kw,
                    conv_dims::od, conv_dims::oh, conv_dims::ow});
    static const conv_tile_t bwd_w_2({conv_dims::g, conv_dims::mb});
    static const conv_tile_t *fwd[] = {&fwd_0, &fwd_1, &fwd_2};
    static const conv_tile_t *bwd_d[] = {&bwd_d_0, &bwd_d_1, &bwd_d_2};
    static const conv_tile_t *bwd_w[] = {&bwd_w_0, &bwd_w_1, &bwd_w_2};
    ir_assert(idx >= 0 && idx < 3);
    if (prb.is_fwd) return fwd[idx];
    if (prb.is_bwd_d) return bwd_d[idx];
    if (prb.is_bwd_w) return bwd_w[idx];
    ir_error_not_expected();
    return nullptr;
}

const conv_tile_t *get_transpose_kernel_grid_conv_dims(
        const conv_problem_t &prb, int idx) {
    static const conv_tile_t fwd_0({conv_dims::mb});
    static const conv_tile_t fwd_1({conv_dims::oc});
    static const conv_tile_t fwd_2(
            {conv_dims::g, conv_dims::od, conv_dims::oh, conv_dims::ow});
    static const conv_tile_t bwd_d_0({conv_dims::mb});
    static const conv_tile_t bwd_d_1({conv_dims::ic});
    static const conv_tile_t bwd_d_2(
            {conv_dims::g, conv_dims::id, conv_dims::ih, conv_dims::iw});
    static const conv_tile_t bwd_w_0(
            {conv_dims::ic, conv_dims::kd, conv_dims::kh, conv_dims::kw,
                    conv_dims::od, conv_dims::oh, conv_dims::ow});
    static const conv_tile_t bwd_w_1({conv_dims::g, conv_dims::mb});
    static const conv_tile_t bwd_w_2({conv_dims::oc});
    static const conv_tile_t *fwd[] = {&fwd_0, &fwd_1, &fwd_2};
    static const conv_tile_t *bwd_d[] = {&bwd_d_0, &bwd_d_1, &bwd_d_2};
    static const conv_tile_t *bwd_w[] = {&bwd_w_0, &bwd_w_1, &bwd_w_2};
    ir_assert(idx >= 0 && idx < 3);
    if (prb.is_fwd) return fwd[idx];
    if (prb.is_bwd_d) return bwd_d[idx];
    if (prb.is_bwd_w) return bwd_w[idx];
    ir_error_not_expected();
    return nullptr;
}

const conv_tile_t *get_thread_group_grid_conv_dims(
        const conv_problem_t &prb, int idx) {
    static const conv_tile_t fwd_0({conv_dims::oc});
    static const conv_tile_t fwd_1({conv_dims::mb, conv_dims::ow});
    static const conv_tile_t fwd_2({conv_dims::ic});
    static const conv_tile_t bwd_d_0({conv_dims::ic});
    static const conv_tile_t bwd_d_1({conv_dims::mb, conv_dims::iw});
    static const conv_tile_t bwd_d_2({conv_dims::oc});
    static const conv_tile_t bwd_w_0({conv_dims::oc});
    static const conv_tile_t bwd_w_1({conv_dims::ic});
    static const conv_tile_t bwd_w_2;
    static const conv_tile_t *fwd[] = {&fwd_0, &fwd_1, &fwd_2};
    static const conv_tile_t *bwd_d[] = {&bwd_d_0, &bwd_d_1, &bwd_d_2};
    static const conv_tile_t *bwd_w[] = {&bwd_w_0, &bwd_w_1, &bwd_w_2};
    ir_assert(idx >= 0 && idx < 3);
    if (prb.is_fwd) return fwd[idx];
    if (prb.is_bwd_d) return bwd_d[idx];
    if (prb.is_bwd_w) return bwd_w[idx];
    ir_error_not_expected();
    return nullptr;
}

const conv_tile_t *get_transpose_thread_group_grid_conv_dims(
        const conv_problem_t &prb, int idx) {
    static const conv_tile_t fwd_0({conv_dims::mb, conv_dims::ow});
    static const conv_tile_t fwd_1({conv_dims::oc});
    static const conv_tile_t fwd_2({conv_dims::ic});
    static const conv_tile_t bwd_d_0({conv_dims::mb, conv_dims::iw});
    static const conv_tile_t bwd_d_1({conv_dims::ic});
    static const conv_tile_t bwd_d_2({conv_dims::oc});
    static const conv_tile_t bwd_w_0({conv_dims::ic});
    static const conv_tile_t bwd_w_1({conv_dims::oc});
    static const conv_tile_t bwd_w_2;
    static const conv_tile_t *fwd[] = {&fwd_0, &fwd_1, &fwd_2};
    static const conv_tile_t *bwd_d[] = {&bwd_d_0, &bwd_d_1, &bwd_d_2};
    static const conv_tile_t *bwd_w[] = {&bwd_w_0, &bwd_w_1, &bwd_w_2};
    ir_assert(idx >= 0 && idx < 3);
    if (prb.is_fwd) return fwd[idx];
    if (prb.is_bwd_d) return bwd_d[idx];
    if (prb.is_bwd_w) return bwd_w[idx];
    ir_error_not_expected();
    return nullptr;
}

void init_padded_dims(conv_config_t &cfg) {
    for (auto &d : get_conv_dims(cfg.prb().prop_kind())) {
        int dim = cfg.dim(d);
        int iter = cfg.iter_dim(d);
        int tg = cfg.thread_group_dim(d);
        int loop = cfg.loop_dim(d);
        int blk = iter * tg * loop;
        int pad_blk = cfg.pad_block(d);
        int padded = utils::rnd_up(dim, math::lcm(blk, pad_blk));
        cfg.padded_dims().set(d, padded);
    }
}

void init_kernel_grid(conv_config_t &cfg) {
    const auto &prb = cfg.prb();
    auto get = [&](const conv_dim_t &d) {
        int padded = cfg.padded_dim(d);
        int iter = cfg.iter_dim(d);
        int loop = cfg.loop_dim(d);
        int tg = cfg.thread_group_dim(d);
        int tg_block = iter * loop * tg;
        return ir_utils::safe_divide(padded, tg_block);
    };

    const int grid_ndims = 3;
    std::vector<int> dims = {1, 1, 1};
    for (int i = 0; i < grid_ndims; i++) {
        auto *tile = prb.ab_swap_transpose
                ? get_transpose_kernel_grid_conv_dims(prb, i)
                : get_kernel_grid_conv_dims(prb, i);
        for (auto d : *tile)
            dims[i] *= get(d);
    }
    cfg.set_kernel_grid(grid_info_t(dims, "grid_idx"));
}

void init_thread_group_grid(conv_config_t &cfg) {
    const auto &prb = cfg.prb();
    auto get = [&](const conv_dim_t &d) {
        return cfg.thread_group_dims().get(d);
    };

    const int grid_ndims = 3;
    std::vector<int> dims = {1, 1, 1};
    for (int i = 0; i < grid_ndims; i++) {
        auto *tile = prb.ab_swap_transpose
                ? get_transpose_thread_group_grid_conv_dims(prb, i)
                : get_thread_group_grid_conv_dims(prb, i);
        for (auto d : *tile)
            dims[i] *= get(d);
    }
    cfg.set_thread_group_grid(grid_info_t(dims, "tg_idx"));
}

int slm_bufs_hint(const conv_problem_t &prb, int m_tg, int n_tg,
        bool zp_do_src_compensation, bool enable_a, bool enable_b,
        bool do_unroll) {
    if (enable_a || enable_b) {
        bool is_small_tg = (m_tg * n_tg <= 8);
        int pref_bufs
                = ((is_small_tg || prb.is_f32_conv()) && prb.mb > 1 ? 2 : 3);
        if (do_unroll) return pref_bufs;
        const bool use_pref_bufs
                = (enable_a == enable_b) && !zp_do_src_compensation;
        return use_pref_bufs ? pref_bufs : 1;
    }
    return 0;
}

void init_slm(conv_config_t &cfg) {
    if (cfg.slm().is_overridden()) return;

    const auto &prb = cfg.prb();

    int bufs = 0;
    int gmem_bufs = 0;
    bool enable_a = cfg.plan().slm.has_a();
    bool enable_b = cfg.plan().slm.has_b();
    if (enable_a || enable_b) {
        auto &tg = cfg.thread_group_grid();
        bufs = slm_bufs_hint(prb, tg.dim(1), tg.dim(0),
                cfg.zp_cfg().do_src_compensation, enable_a, enable_b,
                cfg.pipeline().do_unroll());
        gmem_bufs = (cfg.is_dp_fma() && cfg.pipeline().do_unroll()) ? 2 : 1;
    }
    gmem_bufs = std::min(cfg.plan().max_gmem_bufs, gmem_bufs);
    cfg.slm().set(bufs, gmem_bufs, enable_a, enable_b);
}

void init_prefetch(conv_config_t &cfg) {
    if (cfg.prefetch().is_overridden()) return;

    bool enable_a = cfg.plan().prefetch.has_a();
    bool enable_b = cfg.plan().prefetch.has_b();

    if (!enable_a && !enable_b) return;

    int bufs = cfg.prb().is_f32_conv() ? 2 : 3;
    cfg.prefetch().set(bufs, enable_a, enable_b);
}

void init_subtiles(conv_config_t &cfg) {
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
        if (prb.is_bwd_w && (!cfg.slm().a() || !cfg.slm().b()))
            cfg.set_fma_kind(fma_kind_t::dpas);
    }

    return status::success;
}

template <typename GetFuncT>
bool in_grid_dims(
        GetFuncT get_func, const conv_problem_t &prb, const conv_dim_t &dim) {
    for (int i = 0; i < 3; i++) {
        auto *tile = get_func(prb, i);
        for (auto d : *tile)
            if (d == dim) return true;
    }
    return false;
}

status_t check_plan(conv_config_t &cfg) {
    auto &plan = cfg.plan();
    ir_assert(cfg.slm().a() == plan.slm.has_a());
    ir_assert(cfg.slm().b() == plan.slm.has_b());
    ir_assert(cfg.pipeline().reuse_headers() == plan.reuse_headers);

#ifdef DNNL_DEV_MODE
    auto &prb = cfg.prb();
    send_pattern_t a_load_pattern;
    send_pattern_t b_load_pattern;
    if (prb.is_fwd) {
        a_load_pattern = validate_blocking(
                cfg, conv_stride_layout_t::input_tensor_t::src);
        b_load_pattern = validate_blocking(
                cfg, conv_stride_layout_t::input_tensor_t::wei);
    } else if (prb.is_bwd_d) {
        a_load_pattern = validate_blocking(
                cfg, conv_stride_layout_t::input_tensor_t::dst);
        b_load_pattern = validate_blocking(
                cfg, conv_stride_layout_t::input_tensor_t::wei);
    } else if (prb.is_bwd_w) {
        a_load_pattern = validate_blocking(
                cfg, conv_stride_layout_t::input_tensor_t::src);
        b_load_pattern = validate_blocking(
                cfg, conv_stride_layout_t::input_tensor_t::dst);
    }
    auto dummy_mem(var_t::make(type_t::byte_ptr(), "mem"));
    auto dummy_reg(var_t::make(type_t::byte_ptr(), "reg"));
    if (!a_load_pattern.matches(
                plan.x2r.a_load.create_stmt(dummy_mem, dummy_reg))) {
        ir_warning() << "Generated load for tensor A does not match "
                     << a_load_pattern << " load idiom\n";
    }
    if (!b_load_pattern.matches(
                plan.x2r.b_load.create_stmt(dummy_mem, dummy_reg))) {
        ir_warning() << "Generated load for tensor B does not match "
                     << a_load_pattern << " load idiom\n";
    }
#endif
    return status::success;
}

status_t check_config(conv_config_t &cfg) {
    const auto &prb = cfg.prb();
    for (auto d : cfg.dims().get()) {
        int tg = cfg.thread_group_dim(d);
        int grid = cfg.grid_dim(d);
        if (tg != 1)
            ir_assert(in_grid_dims(get_thread_group_grid_conv_dims, prb, d))
                    << d.name();
        if (grid != 1)
            ir_assert(in_grid_dims(get_kernel_grid_conv_dims, prb, d))
                    << d.name();
    }
    CHECK(check_plan(cfg));
    return status::success;
}

status_t try_init_cfg(conv_config_t &cfg) {
    init_params(cfg);
    init_padded_dims(cfg);
    init_kernel_grid(cfg);
    init_thread_group_grid(cfg);
    CHECK(init_plan(cfg));
    init_pipeline(cfg);
    init_slm(cfg);
    init_prefetch(cfg);
    init_subtiles(cfg);

    CHECK(fixup_config(cfg));
    CHECK(check_config(cfg));

    return status::success;
}

status_t init_cfg(conv_config_t &cfg, const primitive_t *prim) {
    static std::mutex tune_mutex;
    std::unique_lock<std::mutex> lock(tune_mutex, std::defer_lock_t());
    if (cfg.tiler().is_tuning_mode()) lock.lock();
    while (cfg.tiler().can_move_next()) {
        auto try_cfg = cfg;
        auto status = try_init_cfg(try_cfg);
        if (status == status::success) {
            cfg = try_cfg;
            return status::success;
        }
    }
    return status::runtime_error;
}

int conv_config_t::reserved_regs() const {
    return constants::reserved_regs_default;
}

int conv_config_t::pad_block(const conv_dim_t &d) const {
    auto &src = src_layout().compute();
    auto &wei = wei_layout().compute();
    auto &dst = dst_layout().compute();

    const layout_t *layouts[] = {&src, &wei, &dst};
    // src, wei, dst
    int g_idxs[] = {1, 0, 1};
    int mb_idxs[] = {0, -1, 0};
    int oc_idxs[] = {-1, 1, 2};
    int ic_idxs[] = {2, 2, -1};
    int *idxs = nullptr;
    switch (d.kind()) {
        case conv_dim_kind_t::g: idxs = g_idxs; break;
        case conv_dim_kind_t::mb: idxs = mb_idxs; break;
        case conv_dim_kind_t::oc: idxs = oc_idxs; break;
        case conv_dim_kind_t::ic: idxs = ic_idxs; break;
        default: return 1;
    }

    int ret = 1;
    for (int i = 0; i < 3; i++) {
        if (idxs[i] == -1) continue;
        int blk = (int)layouts[i]->inner_block(
                idxs[i], /*skip_outer=*/true, /*inner_only=*/false);
        ret = math::lcm(ret, blk);
    }

    return ret;
}

int get_thread_count(const conv_config_t &cfg) {
    return cfg.kernel_grid().elems() * cfg.thread_group_grid().elems();
}

// Return thread utilization as a percentage. If this value is low,
// parallelism is a fundamental limitation to the current work scheduling.
float get_thread_utilization(const conv_config_t &cfg) {
    auto arch = convert_ngen_arch_to_dnnl(cfg.hw());
    int eus_per_slice = compute::device_info_t::max_eus_per_wg(arch);
    int slice_count = cfg.hw_cfg().eu_count() / eus_per_slice;

    int min_wg_per_slice_wave
            = std::max(eus_per_slice / cfg.thread_group_grid().elems(), 1);
    int min_wg_per_wave = slice_count * min_wg_per_slice_wave;
    int wg_count = cfg.kernel_grid().elems();
    return ((float)wg_count / utils::rnd_up(wg_count, min_wg_per_wave)) * 100;
}

// Return wave utilization as a percentage. If this value is low, memory
// latency may be an issue due to limited use of SMT to hide the latency.
float get_wave_utilization(const conv_config_t &cfg) {
    auto arch = convert_ngen_arch_to_dnnl(cfg.hw());
    int threads_per_eu
            = compute::device_info_t::threads_per_eu(arch, cfg.regs() > 128);
    int eus_per_slice = compute::device_info_t::max_eus_per_wg(arch);
    int slice_count = cfg.hw_cfg().eu_count() / eus_per_slice;

    int wgs_per_slice
            = eus_per_slice * threads_per_eu / cfg.thread_group_grid().elems();
    ir_assert(wgs_per_slice > 0);
    int wgs_per_tile = slice_count * wgs_per_slice;
    int wg_count = cfg.kernel_grid().elems();
    return ((float)wg_count / utils::rnd_up(wg_count, wgs_per_tile)) * 100;
}

std::string conv_config_t::str() const {
    using namespace ir_utils;

    std::ostringstream oss;
    // clang-format off
    oss << "  Exec config:                " << exec_cfg().str() << std::endl;
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
    oss << "  Reuse headers:              " << to_string(pipeline().reuse_headers()) << std::endl;
    oss << "  Subtiles:                   " << "A: " << subtiles().a() << ", B: " << subtiles().b() << std::endl;
    oss << "  Estimated GRF usage:        " << estimated_peak_regs << std::endl;
    oss << "  AB Swap Transpose:          " << to_string(prb().ab_swap_transpose) << std::endl;
    oss << "  Configuration line:         " << get_config_line() << std::endl;
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

conv_key_t conv_config_t::key() const {
    return conv_key_t(*this);
}

std::string conv_config_t::blocking_brief_str() const {
    std::ostringstream oss;
    for (auto &d : get_conv_dims(prb().prop_kind())) {
        int iter = iter_dim(d);
        int tg = thread_group_dim(d);
        int loop = loop_dim(d);
        int grid = grid_dim(d);
        if (iter == 1 && loop == 1 && tg == 1) continue;
        oss << "  Dimension " << d.name()
            << pad_str(":", -18 + (int)d.name().length());
        oss << "(grid:" << pad_int(grid, 5) << ") x ";
        oss << "(tg:" << pad_int(tg, 5) << ") x ";
        oss << "(loop:" << pad_int(loop, 5) << ") x ";
        oss << "(iter:" << pad_int(iter, 5) << ")\n";
    }
    return oss.str();
}

void conv_config_t::set_params_id(int id) {
    params_id_ = id;
}

conv_params_t conv_config_t::params() const {
    auto ret = conv_params_t(*this);
    ret.set_id(params_id_);
    return ret;
}

void conv_config_t::set_tiler(const std::shared_ptr<conv_tiler_t> &tiler) {
    tiler_ = tiler;
}

const conv_tiler_t &conv_config_t::tiler() const {
    return *tiler_;
}

conv_tiler_t &conv_config_t::tiler() {
    return *tiler_;
}

void conv_config_t::set_plan(const std::shared_ptr<conv_plan_t> &plan) {
    plan_ = plan;
}

const conv_plan_t &conv_config_t::plan() const {
    return *plan_;
}

bool conv_config_t::can_skip_wei_zero_out() const {
    if (!prb().is_bwd_w) return true;
    bmnk_dim_helper_t h(*this);
    int k_iter_dim = h.iter_dim(gemm_dims::k);
    int k_loop_dim = h.loop_dim(gemm_dims::k);
    int k_tg_dim = h.thread_group_dim(gemm_dims::k);
    int k_tg_block = k_iter_dim * k_loop_dim * k_tg_dim;
    int k_padded = padded_dim(conv_dims::mb) * padded_dim(conv_dims::od)
            * padded_dim(conv_dims::oh) * padded_dim(conv_dims::ow);
    return k_tg_block >= k_padded;
}

bool conv_config_t::can_skip_bia_zero_out() const {
    if (!prb().is_bwd_w || !prb().with_bias) return true;
    return can_skip_wei_zero_out() && !slm().b();
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
    init_extra_tensors(cfg.zp_cfg(), *prb.conv_pd->attr(),
            *prb.conv_pd->invariant_dst_md(), (prb.is_fwd) ? prb.ic : prb.oc,
            (prb.is_fwd) ? prb.oc : prb.ic, tensor_cfg);
    return tensor_cfg;
}

int estimate_register_count(const conv_config_t &cfg) {
    return cfg.plan().grf_usage().total();
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
