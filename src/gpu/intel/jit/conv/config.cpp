/*******************************************************************************
* Copyright 2022-2025 Intel Corporation
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

#include "gpu/intel/jit/conv/config.hpp"

#include <cctype>
#include <cstring>
#include <mutex>

#include "common/utils.hpp"
#include "gpu/intel/jit/conv/grf_usage.hpp"
#include "gpu/intel/jit/conv/message_patterns.hpp"
#include "gpu/intel/jit/conv/normalization.hpp"
#include "gpu/intel/jit/conv/plan.hpp"
#include "gpu/intel/jit/conv/problem.hpp"
#include "gpu/intel/jit/conv/tiler.hpp"
#include "gpu/intel/jit/eltwise_injector.hpp"
#include "gpu/intel/jit/ir/gemm_schedule.hpp"
#include "gpu/intel/jit/ir/tensor_config.hpp"

#define VDISPATCH_CHECK(pd, engine, cond, msg, ...) \
    VCONDCHECK(primitive, create, dispatch, convolution, (cond), \
            status::unimplemented, "%s," msg, pd->info(engine), ##__VA_ARGS__)

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
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
        default: gpu_error_not_expected() << type;
    }
    return 1;
}

bool is_small(const type_t &type, dim_t elems) {
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
        impl::engine_t *engine, const convolution_pd_t *conv_pd) {
    using namespace compute;

    VDISPATCH_CHECK(conv_pd, engine, !conv_pd->has_zero_dim_memory(),
            VERBOSE_EMPTY_TENSOR, "");

    this->conv_pd = conv_pd;
    attr = conv_pd->attr();
    is_fwd = conv_pd->is_fwd();
    is_bwd_d = conv_pd->is_bwd_d();
    is_bwd_w = conv_pd->is_bwd_w();
    with_bias = conv_pd->with_bias();
    with_groups = conv_pd->with_groups();
    with_sum = with_sum_post_op();
    memory_desc_wrapper mdw_src(conv_pd->invariant_src_md());
    memory_desc_wrapper mdw_wei(conv_pd->invariant_wei_md());
    memory_desc_wrapper mdw_dst(conv_pd->invariant_dst_md());

    strided = (mdw_src.is_plain() && !mdw_src.is_dense())
            || (mdw_wei.is_plain() && !mdw_wei.is_dense())
            || (mdw_dst.is_plain() && !mdw_dst.is_dense());

    src_data_type = conv_pd->invariant_src_md()->data_type;
    wei_data_type = conv_pd->invariant_wei_md()->data_type;
    bia_data_type = conv_pd->invariant_bia_md()->data_type;
    dst_data_type = conv_pd->invariant_dst_md()->data_type;
    fpmath_mode = attr->fpmath_.mode_;

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

    normalize_shape();

    is_dw = with_groups && (g > 1) && (oc == 1) && (ic == 1);
    ksp = kd * kh * kw;
    isp = id * ih * iw;
    osp = od * oh * ow;

    hw_t hw(engine);
    init_transpose(hw);
    CHECK(init_abc_data_types(hw));
    CHECK(init_acc_data_type());

    return status::success;
}

std::string conv_problem_t::desc_str(bool print_mb) const {
    std::ostringstream oss;
    if (print_mb) oss << "mb" << mb;
    if (g > 1) oss << "g" << g;
    oss << "ic" << ic;

    std::vector<dim_t> xd = {id, od, kd, sd, dd, pd};
    std::vector<dim_t> xh = {ih, oh, kh, sh, dh, ph};
    std::vector<dim_t> xw = {iw, ow, kw, sw, dw, pw};
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

int prim_config_t::sort_key(const param_t *param) const {
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
        if (param->short_name() == *p) return into<int>(p - ordered_params);
    }
    return (int)(sizeof(ordered_params) / sizeof(ordered_params[0]));
}

const bool allow_global_reduction_param_t::default_value = true;
const bwd_d_optimize_kind_t bwd_d_optimize_kind_param_t::default_value
        = bwd_d_optimize_kind_t::none;
const bool pad_slm_param_t::default_value = true;

std::string build_tag(const std::vector<int> &inner_blocks,
        const std::vector<int> &outer_blocks, const std::vector<char> &letters,
        const std::vector<int> &idxs) {
    dim_idx_t n = into<dim_idx_t>(letters.size());
    gpu_assert(inner_blocks.size() == n);
    gpu_assert(outer_blocks.size() == n);
    gpu_assert(idxs.size() == n);

    std::string tag;
    std::vector<bool> seen(n);

    // Iterate through outer blocks.
    for (int i = n - 1; i >= 0; i--) {
        int idx = idxs[i];
        int blk = outer_blocks[idx];
        if (blk == 1) continue;
        seen[idx] = true;
        tag += std::to_string(blk) + letters[idx];
    }

    // Iterate through inner blocks.
    for (int i = n - 1; i >= 0; i--) {
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
            if (seen[i]) c = static_cast<char>(std::toupper(c));
            tag = c + tag;
        }
    }

    return tag;
}

int pick_block_impl(bool prefer_rnd_up, dim_t dim, int b0, int b1, int b2) {
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

int pick_block_rnd_up(dim_t dim, int b0, int b1 = 0, int b2 = 0) {
    return pick_block_impl(true, dim, b0, b1, b2);
}

int pick_block(dim_t dim, int b0, int b1 = 0, int b2 = 0) {
    return pick_block_impl(false, dim, b0, b1, b2);
}

int get_default_block(fma_kind_t fma, const type_t &type, dim_t elems) {
    if (is_dp_fma(fma)) {
        if (is_small(type, elems)) {
            int packed_dword_elems = 4 / type.size();
            return std::max(
                    utils::rnd_up_pow2(into<int>(elems)), packed_dword_elems);
        }
        return 32 / type.size();
    }
    if (is_small(type, elems)) return 1;
    return get_default_mad_block(type);
}

fma_kind_t get_default_fma(const hw_t &hw, const type_t &type) {
    switch (type.size()) {
        case 1:
            if (hw >= ngen::HW::XeHP) return fma_kind_t::dpas;
            return hw >= ngen::HW::XeLP ? fma_kind_t::dp4a : fma_kind_t::mad;
        case 2:
            return hw >= ngen::HW::XeHP ? fma_kind_t::dpas : fma_kind_t::mad;
        default: return fma_kind_t::mad;
    }
    return fma_kind_t::undef;
}

struct nc_block_t {
    nc_block_t(int n_block, int c_block)
        : n_block_(n_block), c_block_(c_block) {}

    int n_block() const { return n_block_; }
    int c_block() const { return c_block_; }

    std::string tag() const {
        std::vector<int> idxs = {1, 0};
        return build_tag({n_block_, c_block_}, {1, 1}, {'a', 'b'}, idxs);
    }

    // Ideally, this should only depend on data type, direction, mb, c, and g to
    // enable the same src/dst formats and avoid reorders between convolutions
    static nc_block_t get_default_blocking(const hw_t &hw, fma_kind_t fma,
            type_t type, bool is_dw, dim_t n, dim_t c, dim_t g,
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
            fma_kind_t fma_kind, bool is_bwd_d, dim_t g, dim_t o, dim_t i,
            bool ab_transpose) {
        dim_t x = o;
        dim_t y = i;
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
                g_block, *x_block, *y_block, *y_block_outer, ab_transpose);
        return goi_block_t(fma_kind, is_dw(g, o, i), is_bwd_d, g_block, o_block,
                i_block, o_block_outer, i_block_outer);
    }

    static void get_default_blocking(type_t type, int vec_size,
            fma_kind_t fma_kind, bool is_bwd_d, dim_t g, dim_t x, dim_t y,
            int &g_block, int &x_block, int &y_block, int &y_block_outer,
            bool ab_transpose = false) {
        if (is_dw(g, x, y)) {
            g_block = vec_size;
        } else if (fma_kind == fma_kind_t::mad) {
            x_block = (ab_transpose && is_bwd_d)
                    ? into<int>(utils::rnd_up_pow2(x))
                    : vec_size;
            y_block = get_default_block(fma_kind, type, y);
        } else {
            int packed_dword_elems = 4 / type.size();
            x_block = ab_transpose ? into<int>(utils::rnd_up_pow2(x))
                                   : vec_size;
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
    static bool is_dw(dim_t g, dim_t o, dim_t i) {
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
    memory_desc_wrapper mdw(md);
    if (mdw.is_plain() && !mdw.is_dense()) return "user";
    if (is_wei) {
        std::vector<const char *> plain_non_group_wei_tags
                = {"abx", "axb", "xba"};
        std::vector<const char *> plain_group_wei_tags
                = {"abcx", "abxc", "axcb"};
        auto &plain_wei_tags = (prb.with_groups ? plain_group_wei_tags
                                                : plain_non_group_wei_tags);
        gpu_assert(
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
    gpu_error_not_expected() << tag;
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

bool is_plain_tag_optimal_for_output(
        const std::string &tag, const std::string &user_tag) {
    // NHWC is OK with output as C is used for blocking and C is dense.
    if (user_tag == "axb") return true;
    // NCHW is OK only when blocked by W (not N).
    if (user_tag == "abx") {
        bool is_n_blocked = (tag.find("A") != std::string::npos);
        return !is_n_blocked;
    }
    return false;
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
        user_src_tag = (user_src_req.empty() ? "axb" : std::move(user_src_req));
    if (!matches_tag(dst_md, dst_tag) && is_small_oc_g1)
        user_dst_tag = (user_dst_req.empty() ? "axb" : std::move(user_dst_req));

    // Avoid reorder for small shapes
    if (!user_src_tag.empty() && !user_dst_tag.empty() && prb.g == 1
            && prb.ic < 4 && prb.oc < 4 && prb.mb < 4 && prb.ksp == 1) {
        src_tag = user_src_tag;
        dst_tag = user_dst_tag;
    }
    maybe_set_plain_weights(
            cfg, src_axb && dst_axb, user_wei_req, wei_tag, user_wei_tag);

    if (user_src_tag.empty()) user_src_tag = src_tag;
    if (user_wei_tag.empty()) user_wei_tag = wei_tag;
    if (user_dst_tag.empty()) user_dst_tag = dst_tag;
    if (src_abx && !src_matches) user_src_tag = "abx";
    if (dst_abx && !dst_matches) user_dst_tag = "abx";

    // Use plain tag for output to avoid extra reorders when beneficial.
    if (src_output && is_plain_tag_optimal_for_output(src_tag, user_src_tag))
        src_tag = user_src_tag;
    if (dst_output && is_plain_tag_optimal_for_output(dst_tag, user_dst_tag))
        dst_tag = user_dst_tag;

    if (user_src_req == "user") src_tag = user_src_tag = "user";
    if (user_wei_req == "user") wei_tag = user_wei_tag = "user";
    if (user_dst_req == "user") dst_tag = user_dst_tag = "user";
}

void prepare_zp_precompute_conv(const conv_problem_t &prb, dim_t *idhw,
        dim_t *odhw, dim_t *pdhw, dim_t *ddhw) {
    const bool is_bwd_d = (prb.prop_kind() == prop_kind::backward_data);
    using memory_dims = std::vector<dim_t>;
    memory_dims I {prb.id, prb.ih, prb.iw};
    memory_dims O {prb.od, prb.oh, prb.ow};
    memory_dims K {prb.kd, prb.kh, prb.kw};
    memory_dims S {prb.sd, prb.sh, prb.sw};
    memory_dims D {prb.dd, prb.dh, prb.dw};
    memory_dims P {prb.pd, prb.ph, prb.pw};
    const int off = 5 - prb.ndims;
    const auto *w = prb.conv_pd->weights_md();

    // restore the original layout of the prb values
    const auto *s
            = (is_bwd_d) ? prb.conv_pd->diff_dst_md() : prb.conv_pd->src_md();
    const auto *d
            = (is_bwd_d) ? prb.conv_pd->diff_src_md() : prb.conv_pd->dst_md();
    auto has_dim = [&](int i) {
        return (s->dims[2 + i] > 1) || (d->dims[2 + i] > 1)
                || (w->dims[2 + i + prb.with_groups] > 1);
    };
    auto move_back = [&](int i, int off) {
        if (off == 0) return;
        I[i - off] = O[i - off] = K[i - off] = S[i - off] = 1;
        D[i - off] = P[i - off] = 0;
        std::swap(I[i - off], I[i]);
        std::swap(O[i - off], O[i]);
        std::swap(K[i - off], K[i]);
        std::swap(S[i - off], S[i]);
        std::swap(D[i - off], D[i]);
        std::swap(P[i - off], P[i]);
    };
    bool has_d = (off <= 0) && has_dim(0 - off);
    bool has_h = (off <= 1) && has_dim(1 - off);
    bool has_w = (off <= 2) && has_dim(2 - off);
    if (!has_d && !has_h && !has_w) has_w = true;
    move_back(1, has_d * (!has_h == has_w));
    move_back(2, !has_w * (!has_h + 1));

    for (int i = off; i < int(K.size()); i++) {
        const auto KD = (K[i] - 1) * (D[i] + 1) + 1;
        gpu_assert(w->dims[2 + i + prb.with_groups - off] == K[i]);
        O[i] = ir_utils::max_unique_pad_states(
                O[i], I[i], KD, P[i], S[i], true);
        I[i] = std::min(KD, I[i]);
    }
    for (int i = 0; i < 3; i++) {
        idhw[i] = (i < off) ? 0 : I[i];
        odhw[i] = (i < off) ? 0 : O[i];
        pdhw[i] = (i < off) ? 0 : P[i];
        ddhw[i] = (i < off) ? 0 : D[i];
    }
}

status_t init_tensor_layouts(
        conv_config_t &cfg, convolution_pd_t *pd, impl::engine_t *engine) {
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

    VDISPATCH_CHECK(pd, engine,
            user_src_layout.is_strictly_equal(
                    make_layout(src_md, user_src_tag)),
            VERBOSE_UNSUPPORTED_TAG);
    VDISPATCH_CHECK(pd, engine,
            user_dst_layout.is_strictly_equal(
                    make_layout(dst_md, user_dst_tag)),
            VERBOSE_UNSUPPORTED_TAG);
    VDISPATCH_CHECK(pd, engine,
            user_wei_layout.is_strictly_equal(
                    make_layout(wei_md, user_wei_tag)),
            VERBOSE_UNSUPPORTED_TAG);

    auto src_layout = (src_tag != user_src_tag) ? make_layout(src_md, src_tag)
                                                : user_src_layout;
    auto wei_layout = (wei_tag != user_wei_tag) ? make_layout(wei_md, wei_tag)
                                                : user_wei_layout;
    auto dst_layout = (dst_tag != user_dst_tag) ? make_layout(dst_md, dst_tag)
                                                : user_dst_layout;
    auto bia_layout = user_bia_layout;

    if (prb.is_bwd_w) {
        if (utils::one_of(prb.wei_data_type, data_type::bf16, data_type::f16,
                    data_type::f8_e5m2, data_type::f8_e4m3))
            wei_layout = wei_layout.retype(type_t::f32());
        if (utils::one_of(prb.bia_data_type, data_type::bf16, data_type::f16,
                    data_type::f8_e5m2, data_type::f8_e4m3))
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
            prb.with_groups, prb.g, prb.ic, prb.oc, prb.is_dw, prb.dhw_map,
            /*add_groups=*/true);
    normalize_conv_layouts(user_src_layout, user_wei_layout, user_dst_layout,
            user_bia_layout, prb.with_groups, prb.g, prb.ic, prb.oc, prb.is_dw,
            prb.dhw_map,
            /*add_groups=*/true);

    src.set_compute(src_layout);
    src.set_user(user_src_layout);
    wei.set_compute(wei_layout);
    wei.set_user(user_wei_layout);
    dst.set_compute(dst_layout);
    dst.set_user(user_dst_layout);
    bia.set_compute(bia_layout);
    bia.set_user(user_bia_layout);

    if (cfg.zp_cfg().needs_src_reorder_precalc) {
        auto get_channels = [](const layout_t &layout) {
            const dim_t min_esize = 16;
            return std::max(utils::rnd_up_pow2(layout.dim(1) * layout.dim(2)),
                    min_esize);
        };
        using namespace memory_extra_flags;
        prepare_zp_precompute_conv(prb, wei_md.extra.idhw, wei_md.extra.odhw,
                wei_md.extra.pdhw, wei_md.extra.ddhw);

        wei_md.extra.dst_size = sizeof(float);
        for (const auto &o : wei_md.extra.odhw)
            wei_md.extra.dst_size *= std::max(o, dim_t(1));
        if (prb.prop_kind() == prop_kind::backward_data) {
            wei_md.extra.flags |= compensation_gpu_conv_asymmetric_src_bwd;
            wei_md.extra.dst_size *= get_channels(src_layout);
        } else {
            wei_md.extra.dst_size *= get_channels(dst_layout);
        }
        wei_md.extra.flags |= compensation_gpu_conv_asymmetric_src;
        // since tmasks are used on precalc ZPs only if absolutely necessary
        // (due to significant computational costs in most cases) some block
        // reads can exceed the total buffer size, resulting in page faults;
        // padding at the end is the easiest way to avoid that, as 1-2 KB of
        // additional VRAM per precalc buffer is virtually free
        // TODO: vectorize send params (in jit:ir:v2 maybe?) and add tmasks!
        const dim_t max_read_blk_bytes = 2048;
        wei_md.extra.dst_size += max_read_blk_bytes * 2;
    }
    return status::success;
}

bool hw_ok(const hw_t &hw) {
    if (hw < ngen::HW::Gen9) return false;
    return true;
}

bool data_types_ok(
        const conv_problem_t &prb, const hw_t &hw, impl::engine_t *engine) {
    auto src = prb.src_data_type;
    auto wei = prb.wei_data_type;
    auto dst = prb.dst_data_type;
    auto bia = prb.bia_data_type;
    bool is_fp8 = utils::one_of(data_type::f8_e5m2, src, wei, dst, bia)
            || utils::one_of(data_type::f8_e4m3, src, wei, dst, bia);
    if (!prb.is_f64_accumulator()
            && utils::one_of(data_type::f64, src, wei, dst, bia))
        return false;
    auto *compute_engine
            = utils::downcast<const compute::compute_engine_t *>(engine);
    auto *device_info = compute_engine->device_info();
    if (prb.is_f64_accumulator() && !device_info->has_native(data_type::f64))
        return false;
    if (is_fp8
            && !(utils::one_of(hw, ngen::HW::XeHPC) && hw.systolic_support()))
        return false;
    if (prb.is_fwd) return true;
    if (prb.is_bwd_d) return true;
    if (prb.is_bwd_w) {
        bool ok = true;
        data_type_t default_acc_type
                = src == data_type::f64 ? data_type::f64 : data_type::f32;
        ok &= utils::one_of(src, data_type::f8_e5m2, data_type::f8_e4m3,
                data_type::bf16, data_type::f16, data_type::f32,
                data_type::f64);
        ok &= (dst == src);
        ok &= (utils::one_of(wei, src, default_acc_type)
                || (utils::one_of(src, data_type::f8_e4m3, data_type::f8_e5m2)
                        && utils::one_of(wei, data_type::f8_e4m3,
                                data_type::f8_e5m2, data_type::f32,
                                data_type::bf16, data_type::f16)));

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
    int mask_wei = 0, mask_src = 0, mask_dst = 0;
    if (attr->zero_points_.get(DNNL_ARG_WEIGHTS, &mask_wei) != status::success)
        return false;
    if (attr->zero_points_.get(DNNL_ARG_SRC, &mask_src) != status::success)
        return false;
    if (attr->zero_points_.get(DNNL_ARG_DST, &mask_dst) != status::success)
        return false;

    if (!attr->zero_points_.has_default_values(DNNL_ARG_WEIGHTS)) {
        if (attr->zero_points_.get_data_type(DNNL_ARG_WEIGHTS) != s8)
            return false;
        if (prb.with_groups) return false;
        if (mask_src != 0) return false; // zp_wei implies scalar zp_src
    }

    return IMPLICATION(!utils::one_of(input_type, s8, u8),
                   attr->zero_points_.has_default_values())
            && (mask_wei == 0) && (mask_src == 0 || mask_src == 1 << 1)
            && (mask_dst == 0 || mask_dst == 1 << 1);
}

bool post_ops_ok(const conv_problem_t &prb, const hw_t &hw) {
    auto *pd = prb.conv_pd;
    auto *attr = prb.attr;

    // No post-ops are supported for f64
    if (prb.is_f64_accumulator() && !attr->has_default_values()) return false;

    using sm = primitive_attr_t::skip_mask_t;
    auto attr_skip_mask = sm::fpmath_mode | sm::accumulation_mode;
    if (prb.is_fwd || prb.is_bwd_d) {
        attr_skip_mask |= sm::post_ops | sm::sum_dt | sm::zero_points_runtime
                | sm::zero_points_runtime_data_type | sm::scales_runtime
                | sm::rounding_mode | sm::scales_runtime_groups
                | sm::scales_runtime_data_type;
        if (!attr->has_default_values(attr_skip_mask)) return false;
    } else {
        if (!attr->has_default_values(attr_skip_mask)) return false;
    }

    using namespace data_type;
    const auto input_type = (prb.is_fwd) ? pd->invariant_src_md()->data_type
                                         : pd->invariant_dst_md()->data_type;
    if (!attr->post_ops_.check_sum_consistency(
                prb.c_data_type, utils::one_of(input_type, s8, u8), true))
        return false;

    if (!attr->scales_.has_default_values())
        if (!prb.is_s32_accumulator() && !prb.is_fp8_conv()) return false;
    auto scale_args = get_scale_args();
    std::vector<int> scales(scale_args.size());
    for (int i = 0; i < (int)scale_args.size(); i++)
        scales[i] = scale_args[i].second;
    if (!attr->scales_.has_default_values(scales)) return false;
    for (int arg : scales) {
        if (attr->scales_.has_default_values(arg)) continue;

        int mask = attr->scales_.get(arg).get_mask();
        // XXX: per_oc for BWD_D is treated as per_ic assuming it's called from
        // deconvolution.
        if (arg == DNNL_ARG_WEIGHTS) {
            if (!utils::one_of(mask, 0, prb.with_groups ? 3 : 1)) return false;
        } else if (arg == DNNL_ARG_DST) {
            if (!utils::one_of(mask, 0, 2)) return false;
        } else {
            if (mask != 0) return false;
        }
    }

    for (int i = 0; i < attr->post_ops_.len(); i++) {
        auto &po = attr->post_ops_.entry_[i];
        if (po.is_eltwise()) {
            if (!eltwise_injector_f32_is_supported(po.eltwise.alg))
                return false;
            else if (po.eltwise.alg == alg_kind::eltwise_tanh
                    && hw == ngen::HW::XeHPG
                    && utils::one_of(hw.product_family(),
                            ngen::ProductFamily::GenericXeHPG,
                            ngen::ProductFamily::DG2)
                    && hw.systolic_support() && hw.eu_count() <= 128)
                // Workaround for hard to reproduce issue in end to end
                // workloads. It is unclear what the actual issue is as the
                // kernel always works correctly in benchdnn.
                return false;
        }
    }
    return true;
}

bool should_use_mad(const conv_problem_t &prb) {
    bool small_ic_oc = prb.ic < 3 && prb.oc < 3 && prb.mb < 8;
    bool grouped_small_ic_oc = prb.ic < 4 && prb.oc < 4 && prb.g > 1;
    return prb.is_dw || small_ic_oc || grouped_small_ic_oc;
}

status_t init_fma_kind(
        conv_config_t &cfg, convolution_pd_t *pd, impl::engine_t *engine) {
    if (cfg.fma_kind_param().is_overridden()) return status::success;
    const auto &prb = cfg.prb();
    auto fma_kind = get_supported_fma_kind(
            cfg.hw(), prb.a_data_type, prb.b_data_type, prb.acc_data_type);
    // Force mad for some cases
    if (should_use_mad(prb)) fma_kind = fma_kind_t::mad;
    VDISPATCH_CHECK(pd, engine, fma_kind != fma_kind_t::undef,
            VERBOSE_UNSUPPORTED_DT_CFG);
    cfg.set_fma_kind(fma_kind);
    return status::success;
}

status_t init_simd(conv_config_t &cfg) {
    if (cfg.exec_cfg_param().is_overridden("simd")) return status::success;

    const auto &prb = cfg.prb();
    int simd = get_simd_size(cfg.hw(), cfg.fma_kind(), prb.a_data_type,
            prb.b_data_type, prb.acc_data_type);
    cfg.set_simd(simd);
    return status::success;
}

status_t init_vec_size(conv_config_t &cfg) {
    if (cfg.exec_cfg_param().is_overridden("vec")) return status::success;

    const auto &prb = cfg.prb();
    int vec_size = cfg.simd();
    if (cfg.fma_kind() == fma_kind_t::mad) {
        int grf_elems = cfg.grf_size() / prb.acc_data_type_size;
        dim_t vec_dim = prb.ab_swap_transpose
                ? ((prb.is_bwd_w) ? prb.ic : prb.mb)
                : ((prb.is_fwd || prb.is_bwd_w) ? prb.oc : prb.ic);
        if (utils::rnd_up(vec_dim, grf_elems) < vec_size || prb.is_bwd_d)
            vec_size = grf_elems;
    }
    // SIMD32 produces invalid layouts in bwd_w.
    if (prb.is_bwd_w && !cfg.is_dpas_or_dpasw_fma()) {
        if (prb.is_f64_accumulator()) {
            vec_size = std::min(vec_size, 8);
        } else {
            vec_size = std::min(vec_size, 16);
        }
    }
    cfg.set_vec_size(vec_size);
    return status::success;
}

int default_regs(const conv_config_t &cfg) {
    if (!cfg.hw().large_grf_support()) return 128;
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
    if (prb.is_stride1()) return bwd_d_optimize_kind_t::none;

    auto hint = bwd_d_optimize_kind_t::skip_strided_dhw;
    if (prb.iw % prb.sw != 0 || (prb.mb < 16 && prb.sw <= 8))
        hint = bwd_d_optimize_kind_t::skip_strided_dh;
    return hint;
}

void init_global_reduction(conv_config_t &cfg) {
    if (cfg.allow_global_reduction_param().is_overridden()) return;
    auto &prb = cfg.prb();
    auto *attr = prb.conv_pd->attr();
    bool value = true;
    if (attr->deterministic_) value = false;
    if (prb.is_f64_accumulator() && !cfg.hw().has_fp64_atomic_support())
        value = false;
    cfg.set_allow_global_reduction(value);
}

// Enable optimization for strided BWD_D convolution.
void init_bwd_d_optimize(conv_config_t &cfg) {
    if (cfg.bwd_d_optimize_kind_param().is_overridden()) return;

    auto hint = bwd_d_optimize_kind_hint(cfg.prb());
    cfg.set_bwd_d_optimize_kind(hint);
}

status_t init_pd_time_cfg(const conv_problem_t &prb, conv_config_t &cfg,
        impl::engine_t *engine, convolution_pd_t *pd, primitive_attr_t *attr) {
    hw_t hw(engine);

    VDISPATCH_CHECK(pd, engine, hw_ok(hw), VERBOSE_UNSUPPORTED_ISA);
    VDISPATCH_CHECK(
            pd, engine, data_types_ok(prb, hw, engine), VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_CHECK(
            pd, engine, post_ops_ok(prb, hw), VERBOSE_UNSUPPORTED_POSTOP);
    VDISPATCH_CHECK(
            pd, engine, zero_points_ok(prb), VERBOSE_UNSUPPORTED_ZP_CFG);

    zero_points_config_t zp_cfg(pd);
    cfg.set_zp_cfg(zp_cfg);
    cfg.set_prb(prb);
    cfg.set_exec_cfg(exec_config_t(hw));
    cfg.maybe_override_from_env();

    CHECK(init_fma_kind(cfg, pd, engine));
    CHECK(init_simd(cfg));
    CHECK(init_vec_size(cfg));
    CHECK(init_tensor_layouts(cfg, pd, engine));

    CHECK(attr->set_default_formats(&prb.c_md()));

    VDISPATCH_CHECK(
            pd, engine, post_op_layouts_ok(prb), VERBOSE_UNSUPPORTED_POSTOP);

    init_global_reduction(cfg);
    init_bwd_d_optimize(cfg);

    return status::success;
}

bool pipeline_unroll_hint(const conv_problem_t &prb, fma_kind_t fma_kind,
        const exec_config_t &exec_cfg,
        bwd_d_optimize_kind_t bwd_d_optimize_kind,
        bool allow_global_reduction) {
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
    } else if (prb.is_bwd_w) {
        // Disabled global reduction requires to have full reduction in one
        // thread which may result in multiple nested loops with large bounds
        // so disable unrolling to avoid code size blow-up.
        if (!allow_global_reduction) do_unroll = false;
    }
    // Unrolling with mad or dp4a results in too large kernels.
    if (utils::one_of(fma_kind, fma_kind_t::mad, fma_kind_t::dp4a)
            && (exec_cfg.hw() >= ngen::HW::XeHPG || prb.mb != 1))
        do_unroll = false;
    return do_unroll;
}

void init_pipeline(conv_config_t &cfg) {
    if (cfg.pipeline().is_overridden()) return;

    bool do_unroll
            = pipeline_unroll_hint(cfg.prb(), cfg.fma_kind(), cfg.exec_cfg(),
                    cfg.bwd_d_optimize_kind(), cfg.allow_global_reduction());
    if (cfg.plan().reuse_headers) do_unroll = false;
    cfg.pipeline().set(do_unroll, cfg.plan().reuse_headers);
}

send_pattern_t<pvar_t> validate_blocking(const conv_config_t &cfg,
        conv_stride_layout_t::input_tensor_t tensor, bool check_2d) {
    using send_pattern = send_pattern_t<pvar_t>;
    const compute::gpu_arch_t arch
            = convert_ngen_arch_to_dnnl(cfg.hw().to_ngen());

    auto is_match = [&](const send_hint_t<pvar_t> &hint) {
        for (auto &dim : cfg.index_dims()) {
            if (hint[dim]) {
                if (cfg.iter_dim(dim) % hint[dim]) return false;
            }
        }
        return true;
    };

    auto layout = conv_stride_layout_t(cfg.prb(), tensor);

    auto idiom = [&] {
        switch (arch) {
            case compute::gpu_arch_t::xe_hpc:
                return uniform_send_idiom_t<pvar_t>(
                        /*min_bytes=*/256, check_2d);
            default:
                return uniform_send_idiom_t<pvar_t>(
                        /*min_bytes=*/128, check_2d);
        }
    }();

    auto hints = [&]() {
        auto all_hints = idiom.get_hints(layout);
        if (!all_hints.empty()) {
            dim_t max = 0;
            std::vector<send_hint_t<pvar_t>> max_hints = {};
            for (auto &h : all_hints) {
                auto hint_size = h.size();
                if (max < hint_size) {
                    max = hint_size;
                    max_hints = {h};
                } else if (max == hint_size) {
                    max_hints.emplace_back(h);
                }
            }
            return max_hints;
        }
        return all_hints;
    }();
    if (hints.empty()) {
        gpu_suggestion() << "No hints generated!";
        return send_pattern();
    }

    for (const auto &h : hints) {
        if (is_match(h)) { return send_pattern(h); }
    }

    gpu_suggestion() << "blocking disables " << send_pattern(hints[0])
                     << " load of the " << tensor
                     << " tensor. Try a multiple of:";
    for (auto &hint : hints) {
        gpu_suggestion() << "\t" << hint.str();
    }

    return send_pattern();
}

void init_params(conv_config_t &cfg) {
    cfg.tiler().set_params(cfg);
}

std::array<pvar_tile_t, 3> get_kernel_grid_conv_dims(const conv_config_t &cfg) {
    std::array<pvar_tile_t, 3> grid_dims;
    for (int i = 0; i < 3; i++) {
        for (auto &d : cfg.walk_order().grid_dims(i)) {
            grid_dims[i][d] = 1;
        }
    }
    return grid_dims;
}

using pvar_tile_3 = std::array<pvar_tile_t, 3>;

pvar_tile_3 get_thread_group_grid_conv_dims(const conv_config_t &cfg) {
    static const pvar_tile_t fwd_0({pvars::oc}, 1);
    static const pvar_tile_t fwd_1({pvars::mb, pvars::ow}, 1);
    static const pvar_tile_t fwd_2({pvars::ic}, 1);

    static const pvar_tile_t bwd_d_0({pvars::ic}, 1);
    static const pvar_tile_t bwd_d_1({pvars::mb, pvars::iw}, 1);
    static const pvar_tile_t bwd_d_2({pvars::oc}, 1);

    static const pvar_tile_t bwd_w_0({pvars::oc}, 1);
    static const pvar_tile_t bwd_w_1({pvars::ic}, 1);
    static const pvar_tile_t bwd_w_2;

    // non-transposed
    static const pvar_tile_3 fwd = {fwd_0, fwd_1, fwd_2};
    static const pvar_tile_3 bwd_d = {bwd_d_0, bwd_d_1, bwd_d_2};
    static const pvar_tile_3 bwd_w = {bwd_w_0, bwd_w_1, bwd_w_2};
    // transposed
    static const pvar_tile_3 t_fwd = {fwd_1, fwd_0, fwd_2};
    static const pvar_tile_3 t_bwd_d = {bwd_d_1, bwd_d_0, bwd_d_2};
    static const pvar_tile_3 t_bwd_w = {bwd_w_1, bwd_w_0, bwd_w_2};

    auto &prb = cfg.prb();
    if (prb.is_fwd) return (prb.ab_swap_transpose) ? t_fwd : fwd;
    if (prb.is_bwd_d) return (prb.ab_swap_transpose) ? t_bwd_d : bwd_d;
    if (prb.is_bwd_w) return (prb.ab_swap_transpose) ? t_bwd_w : bwd_w;
    gpu_error_not_expected();
    return fwd;
}

void init_kernel_grid(conv_config_t &cfg) {
    cfg.init_kernel_grid(get_kernel_grid_conv_dims(cfg));
}

void init_thread_group_grid(conv_config_t &cfg) {
    cfg.init_thread_group_grid(get_thread_group_grid_conv_dims(cfg));
}

void get_layout_and_dims(tensor_kind_t ab_kind, const conv_config_t &cfg,
        layout_t &layout, std::vector<pvar_t> &dims) {
    auto &prb = cfg.prb();
    auto &src_dims
            = conv_layout_dims(tensor_kind_t::src, /*src_dst_with_group=*/true);
    auto &wei_dims
            = conv_layout_dims(tensor_kind_t::wei, /*src_dst_with_group=*/true);
    auto &dst_dims
            = conv_layout_dims(tensor_kind_t::dst, /*src_dst_with_group=*/true);
    switch (ab_kind) {
        case tensor_kind_t::a:
            layout = prb.pick_a<const layout_param_t &>(cfg.src_layout(),
                                cfg.wei_layout(), cfg.dst_layout())
                             .compute();
            dims = prb.pick_a<const std::vector<pvar_t> &>(
                    src_dims, wei_dims, dst_dims);
            break;
        case tensor_kind_t::b:
            layout = prb.pick_b<const layout_param_t &>(cfg.src_layout(),
                                cfg.wei_layout(), cfg.dst_layout())
                             .compute();
            dims = prb.pick_b<const std::vector<pvar_t> &>(
                    src_dims, wei_dims, dst_dims);
            break;
        default: gpu_error_not_expected();
    }
    gpu_assert(layout.ndims() == dims.size());
}

// Calculates the size of the range for spatial dimensions within a tile.
// For example, consider forward convolution with stride of 2 and tile ow8kw3.
// After mapping (iw = ow * SW + kw), "iw" range is [0, 16] of size 17.
dim_t map_spatial(
        const conv_config_t &cfg, const pvar_t &dim, const pvar_tile_t &tile) {
    auto &prb = cfg.prb();
    bool is_isp = utils::one_of(dim, pvars::id, pvars::ih, pvars::iw);
    bool is_osp = utils::one_of(dim, pvars::od, pvars::oh, pvars::ow);
    const pvar_t isp_dims[] = {pvars::id, pvars::ih, pvars::iw};
    const pvar_t ksp_dims[] = {pvars::kd, pvars::kh, pvars::kw};
    const pvar_t osp_dims[] = {pvars::od, pvars::oh, pvars::ow};
    dim_t isp[] = {prb.id, prb.ih, prb.iw};
    dim_t osp[] = {prb.od, prb.oh, prb.ow};
    dim_t padding[] = {prb.pd, prb.ph, prb.pw};
    dim_t stride[] = {prb.sd, prb.sh, prb.sw};
    dim_t dilation[] = {prb.dd, prb.dh, prb.dw};
    int idx = dim.spatial_index();
    gpu_assert(idx != -1);
    dim_t O = tile.get(osp_dims[idx], 1);
    dim_t I = tile.get(isp_dims[idx], 1);
    dim_t K = tile.get(ksp_dims[idx], 1);
    dim_t P = padding[idx];
    dim_t S = stride[idx];
    dim_t D = dilation[idx];
    if (is_isp) {
        // Source tensor, map ox, kx to ix.
        gpu_assert(prb.is_fwd || prb.is_bwd_w);
        dim_t i_min = -P;
        dim_t i_max = (O - 1) * S - P + (K - 1) * (1 + D);
        return std::min(isp[idx], i_max - i_min + 1);
    }
    // Destination tensor, map ix, kx to ox.
    gpu_assert(is_osp && prb.is_bwd_d);
    dim_t os_min = P - (K - 1) * (1 + D);
    dim_t os_max = (I - 1) + P;
    return std::min(osp[idx], utils::div_up(os_max - os_min + 1, S));
}

bool needs_spatial_mapping(const conv_config_t &cfg, const pvar_t &dim) {
    auto &prb = cfg.prb();
    if (utils::one_of(dim.name(), "od", "oh", "ow")) return prb.is_bwd_d;
    if (utils::one_of(dim.name(), "id", "ih", "iw"))
        return prb.is_fwd || prb.is_bwd_w;
    return false;
}

size_t get_memory_footprint(const tensor_kind_t &ab_kind,
        const conv_config_t &cfg, const pvar_tile_t &_tile) {
    layout_t layout;
    std::vector<pvar_t> dims;
    get_layout_and_dims(ab_kind, cfg, layout, dims);
    dim_t elems = 1;
    pvar_tile_t tile;
    for (dim_idx_t i = 0; i < layout.ndims(); i++) {
        auto &d = dims[i];
        dim_t d_size
                = (needs_spatial_mapping(cfg, d) ? map_spatial(cfg, d, _tile)
                                                 : _tile.get(d, 1));
        tile[d] = d_size;
        elems *= std::min(d_size, layout.dim(i));
    }
    gpu_assert(elems >= 1);
    return (size_t)layout.type().size() * elems;
}

// Returns the memory footprint in bytes for both input tensors accessed inside
// the tile that is combined from tg_tile and grid_tile.
size_t get_memory_footprint(const conv_config_t &cfg,
        const pvar_tile_t &tg_tile, const pvar_tile_t &grid_tile) {
    pvar_tile_t tile;
    for (auto &d : tg_tile) {
        if (tg_tile[d] == 1) continue;
        tile[d] = tg_tile[d];
    }
    for (auto &d : grid_tile) {
        if (grid_tile[d] == 1) continue;
        tile[d] = tile.get(d, 1) * grid_tile[d];
    }
    auto a_bytes = get_memory_footprint(tensor_kind_t::a, cfg, tile);
    auto b_bytes = get_memory_footprint(tensor_kind_t::b, cfg, tile);
    return a_bytes + b_bytes;
}

pvar_tile_t get_grid_tile(const conv_config_t &cfg) {
    pvar_tile_t grid_tile;
    for (auto &d : conv_index_dims(cfg.prb().prop_kind())) {
        dim_t size = cfg.grid_dim(d);
        if (size == 1) continue;
        grid_tile[d] = size;
    }
    return grid_tile;
}

// Adjusts walk order to iterate group dimension earlier to ensure better
// access locality for a higher cache hit rate.
walk_order_t maybe_fixup_group_with_small_channels(
        const conv_config_t &cfg, const walk_order_t &walk_order) {
    auto &prb = cfg.prb();
    auto grid_tile = get_grid_tile(cfg);
    if (prb.g == 1 || !grid_tile.has(pvars::g)) return walk_order;

    auto &layout = (prb.is_fwd || prb.is_bwd_w) ? cfg.src_layout().compute()
                                                : cfg.dst_layout().compute();
    const int g_dim_idx = 1;
    const int c_dim_idx = 2;
    if (layout.nblocks() <= 1) return walk_order;
    auto &b0 = layout.blocks()[0];
    auto &b1 = layout.blocks()[1];
    // Check that layout has groups followed by channels, i.e. *gc form.
    if (b0.dim_idx != c_dim_idx || b1.dim_idx != g_dim_idx) return walk_order;
    // If the full channel dimension exceeds the cache line size, cache reuse
    // should be already good enough.
    // Xe2 has 256 byte L3 cache block so try to span 4 cache lines.
    int factor = (cfg.hw() == ngen::HW::Xe2) ? 4 : 1;
    if (layout.type().size() * b0.block >= cfg.hw().cache_line_size() * factor)
        return walk_order;

    walk_order_t fixed;
    fixed.add(pvars::g, grid_tile.at(pvars::g), 0);
    for (auto &b : walk_order.blocks()) {
        if (b.dim == pvars::g) continue;
        fixed.add(b.dim, b.size, b.grid_id);
    }
    fixed.finalize(grid_tile);
    return fixed;
}

walk_order_t get_default_walk_order(
        const conv_config_t &cfg, const pvar_tile_t &grid_tile) {
    using vec_t = std::vector<pvar_t>;
    // Ordered from innermost to outermost.
    static const vec_t fwd_0({pvars::oc});
    static const vec_t fwd_1({pvars::ow, pvars::oh, pvars::od, pvars::g});
    static const vec_t fwd_2({pvars::mb});

    static const vec_t bwd_d_0({pvars::ic});
    static const vec_t bwd_d_1({pvars::iw, pvars::ih, pvars::id, pvars::g});
    static const vec_t bwd_d_2({pvars::mb});

    static const vec_t bwd_w_0({pvars::oc});
    static const vec_t bwd_w_1({pvars::ic, pvars::kw, pvars::kh, pvars::kd,
            pvars::ow, pvars::oh, pvars::od});
    static const vec_t bwd_w_2({pvars::g, pvars::mb});
    static const std::array<vec_t, 3> fwd = {fwd_0, fwd_1, fwd_2};
    static const std::array<vec_t, 3> bwd_d = {bwd_d_0, bwd_d_1, bwd_d_2};
    static const std::array<vec_t, 3> bwd_w = {bwd_w_0, bwd_w_1, bwd_w_2};
    auto grid_dims
            = (cfg.prb().is_fwd ? fwd : (cfg.prb().is_bwd_d ? bwd_d : bwd_w));
    walk_order_t walk_order;
    for (int i = 0; i < 3; i++) {
        for (auto &d : grid_dims[i]) {
            if (grid_tile.has(d)) walk_order.add(d, grid_tile[d], i);
        }
    }
    walk_order.finalize(grid_tile);
    walk_order = maybe_fixup_group_with_small_channels(cfg, walk_order);
    return walk_order;
}

// Helper class to iterate through M/N problem sizes in blocks to ensure
// squarish (M x N) size for more optimal cache reuse.
class mn_walker_t {
public:
    struct entry_t {
        pvar_t dim;
        dim_t size = 1;
        dim_t tile_size = 1;
        char mn_kind = ' ';

        bool has_next() const { return size < tile_size; }
    };

    mn_walker_t(const pvar_tile_t &tile, const conv_problem_t &prb)
        : prb_(prb) {
        for (auto &d : tile) {
            auto bmnk = to_gemm(d, prb);
            entry_t e;
            e.dim = d;
            e.tile_size = tile[d];
            if (!utils::one_of(bmnk, pvars::m, pvars::n)) continue;
            e.mn_kind = (bmnk == pvars::m ? 'm' : 'n');
            entries_.push_back(e);
        }
        // Put through spatial dimensions first and order spatial accordingly
        // (WHD, width is first).
        std::sort(entries_.begin(), entries_.end(),
                [&](const entry_t &a, const entry_t &b) {
                    int a_sp_idx = a.dim.spatial_index();
                    int b_sp_idx = b.dim.spatial_index();
                    if (a_sp_idx >= 0 && b_sp_idx >= 0)
                        return a_sp_idx > b_sp_idx;
                    return (a_sp_idx >= 0) && (b_sp_idx < 0);
                });
    }

    bool has_next() const {
        for (auto &e : entries_)
            if (e.has_next()) return true;
        return false;
    }

    entry_t next(const pvar_tile_t &inner) {
        int m_size = 1;
        int n_size = 1;
        for (auto &d : inner) {
            auto bmnk = to_gemm(d, prb_);
            if (bmnk == pvars::m) {
                m_size *= inner[d];
            } else if (bmnk == pvars::n) {
                n_size *= inner[d];
            }
        }
        auto mn_kind = (m_size < n_size ? 'm' : 'n');
        for (auto kind : {mn_kind, ' '}) {
            for (auto &e : entries_) {
                if (utils::one_of(kind, e.mn_kind, ' ') && e.has_next()) {
                    e.size *= 2;
                    return e;
                }
            }
        }
        gpu_error_not_expected();
        return entry_t();
    }

private:
    conv_problem_t prb_;
    std::vector<entry_t> entries_;
};

walk_order_t compute_walk_order(const conv_config_t &cfg) {
    auto &prb = cfg.prb();
    int tg_size = 1;
    pvar_tile_t inner;
    for (auto &d : conv_index_dims(cfg.prb().prop_kind())) {
        dim_t iter = cfg.iter_dim(d);
        dim_t tg = cfg.thread_group_dim(d);
        dim_t loop = cfg.loop_dim(d);
        dim_t size = iter * tg * loop;
        if (size == 1) continue;
        inner[d] = size;
        tg_size *= tg;
    }
    auto grid_tile = get_grid_tile(cfg);
    auto default_walk_order = get_default_walk_order(cfg, grid_tile);

    // Depthwise does not expose much reuse so keep the default order.
    if (prb.is_dw) return default_walk_order;

    // XXX: Workaround for XeHPG related issues, supposedly coming from
    // math.inv usage to emulate integer division when using blocked walk
    // order.
    if (cfg.hw() == ngen::HW::XeHPG
            && utils::one_of(cfg.hw().product_family(),
                    ngen::ProductFamily::GenericXeHPG,
                    ngen::ProductFamily::DG2))
        return default_walk_order;

    // If threadgroup memory footprint exceeds L3 then L3 blocking is not
    // applied.
    const size_t l3_size = cfg.hw().l3_cache_size();
    size_t inner_bytes = get_memory_footprint(cfg, inner, pvar_tile_t());
    if (inner_bytes > l3_size) return default_walk_order;

    // If input memory fits L3 then no L3 blocking is not applied.
    size_t ab_bytes = get_memory_footprint(cfg, inner, grid_tile);
    if (ab_bytes <= l3_size) return default_walk_order;

    // If the kernel does not require multiple waves then no L3 blocking is not
    // applied.
    int max_tgs_per_wave = conv_config_t::get_max_threadgroups_per_wave(
            cfg.exec_cfg(), tg_size);
    if (grid_tile.elems() <= max_tgs_per_wave) return default_walk_order;

    // Add M/N blocks until the full footprint fits L3 cache.
    pvar_tile_t grid_inner;
    pvar_tile_t rem_tile = grid_tile;
    ab_bytes = inner_bytes;
    mn_walker_t mn_walker(rem_tile, cfg.prb());
    while (mn_walker.has_next()) {
        auto entry = mn_walker.next(grid_inner);
        auto outer = grid_inner;
        outer[entry.dim] = std::min(rem_tile[entry.dim], entry.size);
        size_t ab_bytes = get_memory_footprint(cfg, inner, outer);
        if (ab_bytes <= l3_size) grid_inner = std::move(outer);
    }
    // Add the blocks in this order:
    // - Step 1. Add grid_inner blocks (fitting L3 cache)
    // - Step 2. Add the remaining M/N blocks
    // - Step 3. Add the remaining B/K blocks
    // Within a step follow the default walk order between dimensions.
    walk_order_t walk_order;
    for (int step = 0; step < 3; step++) {
        for (auto &b : default_walk_order.blocks()) {
            switch (step) {
                case 0:
                    if (grid_inner.has(b.dim)) {
                        walk_order.add(b.dim, grid_inner[b.dim], 0);
                    }
                    break;
                case 1:
                case 2:
                    dim_t rem = utils::div_up(
                            grid_tile[b.dim], grid_inner.get(b.dim, 1));
                    if (rem == 1) continue;
                    auto bmnk = to_gemm(b.dim, prb);
                    bool is_bk = utils::one_of(bmnk, pvars::b, pvars::k);
                    if ((step == 2) != is_bk) continue;
                    walk_order.add(b.dim, rem, 0);
                    break;
            }
        }
    }
    walk_order.finalize(grid_tile);
    walk_order = maybe_fixup_group_with_small_channels(cfg, walk_order);

    // Emulated integer division can handle a limited range only.
    const int max_size_per_grid_id = (1 << 20);
    for (int id = 0; id < 3; id++) {
        if (!walk_order.is_blocked(id)) continue;
        int size = 1;
        for (auto &b : walk_order.blocks()) {
            if (b.grid_id == id) size *= b.size;
        }
        if (size > max_size_per_grid_id) return default_walk_order;
    }

    return walk_order;
}

void init_walk_order(conv_config_t &cfg) {
    if (cfg.walk_order_param().is_overridden()) {
        auto walk_order = cfg.walk_order();
        walk_order.finalize(get_grid_tile(cfg));
        cfg.set_walk_order(walk_order);
        return;
    }
    auto walk_order = compute_walk_order(cfg);
    cfg.walk_order_param().set(walk_order);
}

int fixup_slm_bufs(const conv_problem_t &prb, int slm_bufs,
        bool zp_do_src_compensation, bool enable_a, bool enable_b,
        bool do_unroll) {
    if (do_unroll) return slm_bufs;
    // Multiple SLM buffering without unrolling has some limitations as compute
    // indices are not tracked: A/B buffers are directly loaded from SLM and
    // multiplied so some scenarios are not supported:
    // - Mixing SLM and direct memory load for A/B as memory load requires
    //   masking which relies on problem indices
    // - Source zero-points as compensation masks require problem indices
    if (enable_a != enable_b || zp_do_src_compensation)
        return std::min(slm_bufs, 1);
    return slm_bufs;
}

int slm_bufs_hint(const conv_problem_t &prb, dim_t m_tg, dim_t n_tg,
        bool zp_do_src_compensation, bool enable_a, bool enable_b,
        bool do_unroll) {
    if (!enable_a && !enable_b) return 0;
    bool is_small_tg = (m_tg * n_tg <= 8);
    int pref_bufs = ((is_small_tg || prb.is_f32_conv()) && prb.mb > 1 ? 2 : 3);
    return fixup_slm_bufs(prb, pref_bufs, zp_do_src_compensation, enable_a,
            enable_b, do_unroll);
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
        bufs = cfg.bufs_hint();
        if (bufs == blocking_params_t::bufs_hint_undef) {
            bufs = slm_bufs_hint(prb, tg.dim(1), tg.dim(0),
                    cfg.zp_cfg().do_src_compensation, enable_a, enable_b,
                    cfg.pipeline().do_unroll());
        } else if (cfg.zp_cfg().do_src_compensation) {
            bufs = std::min(bufs, 1);
        }
        bufs = fixup_slm_bufs(prb, bufs, cfg.zp_cfg().do_src_compensation,
                enable_a, enable_b, cfg.pipeline().do_unroll());
        gpu_assert(bufs > 0);
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

    int bufs = 0;
    bufs = cfg.bufs_hint();
    if (bufs == blocking_params_t::bufs_hint_undef) {
        bufs = cfg.prb().is_f32_conv() ? 2 : 3;
    }
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
void fixup_config(conv_config_t &cfg) {
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
}

void validate_config_and_plan(conv_config_t &cfg) {
    auto check_if_in_grid_dims
            = [](const std::array<pvar_tile_t, 3> &grid, const pvar_t &dim) {
                  for (auto &tile : grid)
                      for (auto &d : tile)
                          if (d == dim) return;
                  gpu_error_not_expected() << dim.name();
              };
    const auto &tg_dims = get_thread_group_grid_conv_dims(cfg);
    const auto &grid_dims = get_kernel_grid_conv_dims(cfg);
    for (auto &d : cfg.dims()) {
        if (cfg.thread_group_dim(d) != 1) check_if_in_grid_dims(tg_dims, d);
        if (cfg.grid_dim(d) != 1) check_if_in_grid_dims(grid_dims, d);
    }

    auto &plan = cfg.plan();
    gpu_assert(cfg.slm().a() == plan.slm.has_a());
    gpu_assert(cfg.slm().b() == plan.slm.has_b());
    gpu_assert(cfg.pipeline().reuse_headers() == plan.reuse_headers);

#ifdef DNNL_DEV_MODE
    using send_pattern = send_pattern_t<pvar_t>;
    send_pattern a_load_pattern;
    send_pattern b_load_pattern;
    bool a_2d = plan.uses_2d_load(abc_kind_t::a);
    bool b_2d = plan.uses_2d_load(abc_kind_t::b);
    auto &prb = cfg.prb();
    if (prb.is_fwd) {
        a_load_pattern = validate_blocking(
                cfg, conv_stride_layout_t::input_tensor_t::src, a_2d);
        b_load_pattern = validate_blocking(
                cfg, conv_stride_layout_t::input_tensor_t::wei, b_2d);
    } else if (prb.is_bwd_d) {
        a_load_pattern = validate_blocking(
                cfg, conv_stride_layout_t::input_tensor_t::dst, a_2d);
        b_load_pattern = validate_blocking(
                cfg, conv_stride_layout_t::input_tensor_t::wei, b_2d);
    } else if (prb.is_bwd_w) {
        a_load_pattern = validate_blocking(
                cfg, conv_stride_layout_t::input_tensor_t::src, a_2d);
        b_load_pattern = validate_blocking(
                cfg, conv_stride_layout_t::input_tensor_t::dst, b_2d);
    }
    auto dummy_mem(var_t::make(type_t::byte_ptr(), "mem"));
    auto dummy_reg(var_t::make(type_t::byte_ptr(), "reg"));
    if (!a_load_pattern.matches(
                plan.x2r.a_load.create_stmt(dummy_mem, dummy_reg))) {
        gpu_warning() << "Generated load for tensor A does not match "
                      << a_load_pattern << " load idiom";
    }
    if (!b_load_pattern.matches(
                plan.x2r.b_load.create_stmt(dummy_mem, dummy_reg))) {
        gpu_warning() << "Generated load for tensor B does not match "
                      << a_load_pattern << " load idiom";
    }
#endif
}

status_t try_init_cfg(conv_config_t &cfg) {
    init_params(cfg);
    init_walk_order(cfg);
    init_kernel_grid(cfg);
    init_thread_group_grid(cfg);

    CHECK(init_plan(cfg));

    init_pipeline(cfg);
    init_slm(cfg);
    init_prefetch(cfg);
    init_subtiles(cfg);

    fixup_config(cfg);
    validate_config_and_plan(cfg);

    return status::success;
}

status_t init_cfg(conv_config_t &cfg, const primitive_t *prim) {
    static std::mutex tune_mutex;
    std::unique_lock<std::mutex> lock(tune_mutex, std::defer_lock_t());
    if (cfg.tiler().is_tuning_mode()) lock.lock();
    while (cfg.tiler().is_valid()) {
        auto try_cfg = cfg;
        auto status = try_init_cfg(try_cfg);
        if (status == status::success) {
            if (cfg.tiler().is_tuning_mode()) cfg.tiler().move_next(cfg);
            cfg = std::move(try_cfg);
            return status::success;
        }
        cfg.tiler().move_next(cfg);
    }
    return status::runtime_error;
}

int conv_config_t::reserved_regs() const {
    return constants::reserved_regs_default;
}

int conv_config_t::pad_block(const pvar_t &d) const {
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
    if (d.name() == "g") {
        idxs = g_idxs;
    } else if (d.name() == "mb") {
        idxs = mb_idxs;
    } else if (d.name() == "oc") {
        idxs = oc_idxs;
    } else if (d.name() == "ic") {
        idxs = ic_idxs;
    } else {
        return 1;
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
    dim_t kg_elems = kernel_grid().elems(), tg_elems = thread_group_grid().elems();
    int estimated_peak_regs = estimate_register_count(*this);
    oss << blocking_brief_str();
    oss << "  Kernel grid:                " << kernel_grid() << std::endl;
    oss << "  Thread group:               " << thread_group_grid() << std::endl;
    oss << "  Threads:                    " << kg_elems * tg_elems << " (utilization: "
        << get_thread_utilization(exec_cfg(), kg_elems, tg_elems) << "% thread, "
        << get_wave_utilization(exec_cfg(), kg_elems, tg_elems) << "% wave)" << std::endl;
    oss << "  FMA kind:                   " << to_string(fma_kind()) << std::endl;
    oss << "  SLM buffering:              " << "A: " << to_string(slm().a()) << ", B: " << to_string(slm().b())
                                            << ", buffers: " << slm().bufs() << ", pad: " << to_string(pad_slm()) << std::endl;
    oss << "  GRF buffers for GMEM load:  " << slm().gmem_bufs() << std::endl;
    oss << "  Prefetch:                   " << to_string(prefetch()) << ", buffers: " << prefetch().bufs() << std::endl;
    oss << "  Do pipeline unroll:         " << to_string(pipeline().do_unroll()) << std::endl;
    oss << "  Reuse headers:              " << to_string(pipeline().reuse_headers()) << std::endl;
    oss << "  Subtiles:                   " << "A: " << subtiles().a() << ", B: " << subtiles().b() << std::endl;
    oss << "  Estimated GRF usage:        " << estimated_peak_regs << std::endl;
    oss << "  AB Swap Transpose:          " << to_string(prb().ab_swap_transpose) << std::endl;
    oss << "  Kernel grid walk order:     " << walk_order() << std::endl;
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

std::string pad_int(dim_t i, int pad) {
    return pad_str(std::to_string(i), pad);
}

conv_key_t conv_config_t::key() const {
    return conv_key_t(*this);
}

std::string conv_config_t::blocking_brief_str() const {
    std::ostringstream oss;
    for (auto &d : index_dims()) {
        dim_t iter = iter_dim(d);
        dim_t tg = thread_group_dim(d);
        dim_t loop = loop_dim(d);
        dim_t grid = grid_dim(d);
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
    dim_t k_iter_dim = h.iter_dim(pvars::k);
    dim_t k_loop_dim = h.loop_dim(pvars::k);
    dim_t k_tg_dim = h.thread_group_dim(pvars::k);
    dim_t k_tg_block = k_iter_dim * k_loop_dim * k_tg_dim;
    dim_t k_padded = padded_dim(pvars::mb) * padded_dim(pvars::od)
            * padded_dim(pvars::oh) * padded_dim(pvars::ow);
    return k_tg_block >= k_padded;
}

bool conv_config_t::can_skip_bia_zero_out() const {
    if (!prb().is_bwd_w || !prb().with_bias) return true;
    return can_skip_wei_zero_out() && !slm().b();
}

pvar_tile_t conv_config_t::shape(bool pad) const {
    auto &p = prb();
    pvar_tile_t ret;
#define SET(name) \
    ret[pvars::name] \
            = (pad ? utils::rnd_up(p.name, pad_block(pvars::name)) : p.name)
    SET(mb);
    SET(g);
    SET(oc);
    SET(ic);
    SET(kd);
    SET(kh);
    SET(kw);
    if (prb().is_fwd || prb().is_bwd_w) {
        SET(od);
        SET(oh);
        SET(ow);
    } else {
        SET(id);
        SET(ih);
        SET(iw);
    }
#undef SET
    return ret;
}

tensor_config_t get_tensor_config(
        const conv_config_t &cfg, const memory_desc_t *zp_src) {
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
    init_extra_tensors(cfg.zp_cfg(), *prb.conv_pd->attr(), zp_src,
            *prb.conv_pd->invariant_dst_md(), (prb.is_fwd) ? prb.ic : prb.oc,
            (prb.is_fwd) ? prb.oc : prb.ic, tensor_cfg);
    return tensor_cfg;
}

int estimate_register_count(const conv_config_t &cfg) {
    return cfg.plan().grf_usage().total();
}

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
