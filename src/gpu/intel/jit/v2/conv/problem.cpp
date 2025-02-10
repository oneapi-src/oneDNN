/*******************************************************************************
* Copyright 2023-2025 Intel Corporation
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

#include "gpu/intel/jit/v2/conv/problem.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {
namespace v2 {
namespace conv {

problem_t::problem_t(const std::string &line) {
    gpu_error_not_expected();
    auto s_desc = gpu_utils::split(line, " ").back();
    set_shape(s_desc);
}

pvar_map_t<dim_t> problem_t::vars() const {
    pvar_map_t<dim_t> map;
    map[pvar_t("outsz")] = out_type().size();
    return map;
}

const type_t &problem_t::out_type() const {
    switch (prop_) {
        case prop_kind::forward: return dst_tag_.type();
        case prop_kind::backward_data: return src_tag_.type();
        case prop_kind::backward_weights: return wei_tag_.type();
        default: gpu_error_not_expected();
    }
    return src_tag_.type();
}

void problem_t::set_shape(const std::string &s) {
    gpu_assert(prop_ != prop_kind::undef);
    pvar_tile_t s_tile(s);
    bool has_d = has_spatial(s_tile, 'd');
    bool has_h = has_spatial(s_tile, 'h');
    bool has_w = has_spatial(s_tile, 'w');
    if ((has_d && has_h && has_w) || (has_h && has_w) || has_w) {
        // Nothing to propagate.
    } else if (has_d && !has_h && !has_w) {
        s_tile[pvars::ih] = s_tile[pvars::iw] = s_tile[pvars::id];
        s_tile[pvars::oh] = s_tile[pvars::ow] = s_tile[pvars::od];
        s_tile[pvars::kh] = s_tile[pvars::kw] = s_tile[pvars::kd];
        s_tile[pvars::sh] = s_tile[pvars::sw] = s_tile[pvars::sd];
        s_tile[pvars::dh] = s_tile[pvars::dw] = s_tile[pvars::dd];
        s_tile[pvars::ph] = s_tile[pvars::pw] = s_tile[pvars::pd];
    } else if (has_h && !has_w) {
        s_tile[pvars::iw] = s_tile[pvars::ih];
        s_tile[pvars::ow] = s_tile[pvars::oh];
        s_tile[pvars::kw] = s_tile[pvars::kh];
        s_tile[pvars::sw] = s_tile[pvars::sh];
        s_tile[pvars::dw] = s_tile[pvars::dh];
        s_tile[pvars::pw] = s_tile[pvars::ph];
    } else {
        gpu_error_not_expected();
    }
    for (auto &d : default_shape()) {
        if (s_tile.has(d)) continue;
        s_tile.set(d, default_shape()[d]);
    }
    shape_ = std::move(s_tile);
}

double problem_t::ops() const {
    return ops(prop_, shape_);
}

void problem_t::normalize() {
#define GET(name) shape_[pvars::name]
    normalize_conv_shape(GET(od), GET(id), GET(kd), GET(sd), GET(dd), GET(pd),
            GET(oh), GET(ih), GET(kh), GET(sh), GET(dh), GET(ph), GET(ow),
            GET(iw), GET(kw), GET(sw), GET(dw), GET(pw),
            /*can_flatten_spatial=*/true, dhw_map_);
#undef GET
}

std::string problem_t::desc_str() const {
    dim_t g = shape_[pvars::g];
    dim_t mb = shape_[pvars::mb];
    dim_t oc = shape_[pvars::oc];
    dim_t ic = shape_[pvars::ic];
    dim_t id = shape_[pvars::id];
    dim_t ih = shape_[pvars::ih];
    dim_t iw = shape_[pvars::iw];
    dim_t od = shape_[pvars::od];
    dim_t oh = shape_[pvars::oh];
    dim_t ow = shape_[pvars::ow];
    dim_t kd = shape_[pvars::kd];
    dim_t kh = shape_[pvars::kh];
    dim_t kw = shape_[pvars::kw];
    dim_t sd = shape_[pvars::sd];
    dim_t sh = shape_[pvars::sh];
    dim_t sw = shape_[pvars::sw];
    dim_t pd = shape_[pvars::pd];
    dim_t ph = shape_[pvars::ph];
    dim_t pw = shape_[pvars::pw];
    dim_t dd = shape_[pvars::dd];
    dim_t dh = shape_[pvars::dh];
    dim_t dw = shape_[pvars::dw];
    std::ostringstream oss;
    oss << "mb" << mb;
    if (g > 1) oss << "g" << g;
    oss << "ic" << g * ic;

    std::vector<dim_t> xd = {id, od, kd, sd, dd, pd};
    std::vector<dim_t> xh = {ih, oh, kh, sh, dh, ph};
    std::vector<dim_t> xw = {iw, ow, kw, sw, dw, pw};
    std::vector<dim_t> xdef = {1, 1, 1, 1, 0, 0};
    bool has_d = (xd != xdef);
    bool has_h = (xh != xdef);
    bool is_square = !has_d && (xh == xw);
    bool is_cubic = (xd == xh) && (xd == xw);
    bool print_d = has_d;
    bool print_h = has_h && !is_cubic;
    bool print_w = !is_cubic && !is_square;

    if (print_d) oss << "id" << id;
    if (print_h) oss << "ih" << ih;
    if (print_w) oss << "iw" << iw;
    oss << "oc" << g * oc;
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

std::string problem_t::str() const {
    std::ostringstream oss;
    oss << "Conv problem" << std::endl;
    oss << "  HW:            " << to_string(hw_.to_ngen()) << std::endl;
    oss << "  Propagation:   " << jit::to_string(prop_) << std::endl;
    oss << "  Source:        " << src_tag_ << std::endl;
    oss << "  Weights:       " << wei_tag_ << std::endl;
    oss << "  Destination:   " << dst_tag_ << std::endl;
    oss << "  With post-ops: " << ir_utils::to_string(with_post_ops_)
        << std::endl;
    oss << "  Deterministic: " << ir_utils::to_string(deterministic_)
        << std::endl;
    oss << "  Descriptor:    " << desc_str();
    return oss.str();
}

std::string problem_t::csv_str() const {
    std::vector<std::string> parts;
    parts.push_back(hw_.brief_str());
    parts.push_back(jit::to_string(prop_));
    parts.push_back(src_tag_.str());
    parts.push_back(wei_tag_.str());
    parts.push_back(dst_tag_.str());
    parts.push_back(desc_str());
    std::ostringstream oss;
    bool is_first = true;
    for (auto &p : parts) {
        if (!is_first) oss << ",";
        oss << p;
        is_first = false;
    }
    return oss.str();
}

pvar_tile_t problem_t::default_shape() {
    static pvar_tile_t _default_shape = []() {
        static pvar_tile_t ret;
        ret[pvars::g] = 1;
        ret[pvars::mb] = 1;
        ret[pvars::id] = ret[pvars::ih] = ret[pvars::iw] = 1;
        ret[pvars::od] = ret[pvars::oh] = ret[pvars::ow] = 1;
        ret[pvars::kd] = ret[pvars::kh] = ret[pvars::kw] = 1;
        for (auto &d : conv_stride_dims())
            ret[d] = 1;
        for (auto &d : conv_dilation_dims())
            ret[d] = 0;
        for (auto &d : conv_padding_dims())
            ret[d] = 0;
        return ret;
    }();
    return _default_shape;
}

double problem_t::ops(prop_kind_t prop, const pvar_tile_t &shape) {
#define GET(name) shape[pvars::name]
    double ret = 2.0;
    ret *= (double)GET(g) * GET(mb) * GET(oc) * GET(ic);
    ret *= GET(kd) * GET(kh) * GET(kw);
    if (prop == prop_kind::backward_data) {
        ret *= GET(id) * GET(ih) * GET(iw);
    } else {
        ret *= GET(od) * GET(oh) * GET(ow);
    }
#undef GET
    return ret;
}

} // namespace conv
} // namespace v2
} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
