/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
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
    ir_error_not_expected();
    auto s_desc = gpu_utils::split(line, " ").back();
    set_shape(s_desc);
}

void problem_t::set_shape(const std::string &s) {
    ir_assert(prop_ != prop_kind::undef);
    prb_tile_t s_tile(s);
    bool has_d = has_spatial(s_tile, prb_dim_spatial_kind_t::d);
    bool has_h = has_spatial(s_tile, prb_dim_spatial_kind_t::h);
    bool has_w = has_spatial(s_tile, prb_dim_spatial_kind_t::w);
    if ((has_d && has_h && has_w) || (has_h && has_w) || has_w) {
        // Nothing to propagate.
    } else if (has_d && !has_h && !has_w) {
        s_tile[prb_dims::ih] = s_tile[prb_dims::iw] = s_tile[prb_dims::id];
        s_tile[prb_dims::oh] = s_tile[prb_dims::ow] = s_tile[prb_dims::od];
        s_tile[prb_dims::kh] = s_tile[prb_dims::kw] = s_tile[prb_dims::kd];
        s_tile[prb_dims::sh] = s_tile[prb_dims::sw] = s_tile[prb_dims::sd];
        s_tile[prb_dims::dh] = s_tile[prb_dims::dw] = s_tile[prb_dims::dd];
        s_tile[prb_dims::ph] = s_tile[prb_dims::pw] = s_tile[prb_dims::pd];
    } else if (has_h && !has_w) {
        s_tile[prb_dims::iw] = s_tile[prb_dims::ih];
        s_tile[prb_dims::ow] = s_tile[prb_dims::oh];
        s_tile[prb_dims::kw] = s_tile[prb_dims::kh];
        s_tile[prb_dims::sw] = s_tile[prb_dims::sh];
        s_tile[prb_dims::dw] = s_tile[prb_dims::dh];
        s_tile[prb_dims::pw] = s_tile[prb_dims::ph];
    } else {
        ir_error_not_expected();
    }
    for (auto &d : default_shape()) {
        if (s_tile.has(d)) continue;
        s_tile.set(d, default_shape()[d]);
    }
    shape_ = s_tile;
}

std::string problem_t::desc_str() const {
    int g = shape_[prb_dims::g];
    int mb = shape_[prb_dims::mb];
    int oc = shape_[prb_dims::oc];
    int ic = shape_[prb_dims::ic];
    int id = shape_[prb_dims::id];
    int ih = shape_[prb_dims::ih];
    int iw = shape_[prb_dims::iw];
    int od = shape_[prb_dims::od];
    int oh = shape_[prb_dims::oh];
    int ow = shape_[prb_dims::ow];
    int kd = shape_[prb_dims::kd];
    int kh = shape_[prb_dims::kh];
    int kw = shape_[prb_dims::kw];
    int sd = shape_[prb_dims::sd];
    int sh = shape_[prb_dims::sh];
    int sw = shape_[prb_dims::sw];
    int pd = shape_[prb_dims::pd];
    int ph = shape_[prb_dims::ph];
    int pw = shape_[prb_dims::pw];
    int dd = shape_[prb_dims::dd];
    int dh = shape_[prb_dims::dh];
    int dw = shape_[prb_dims::dw];
    std::ostringstream oss;
    oss << "mb" << mb;
    if (g > 1) oss << "g" << g;
    oss << "ic" << g * ic;

    std::vector<int> xd = {id, od, kd, sd, dd, pd};
    std::vector<int> xh = {ih, oh, kh, sh, dh, ph};
    std::vector<int> xw = {iw, ow, kw, sw, dw, pw};
    std::vector<int> xdef = {1, 1, 1, 1, 0, 0};
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

void problem_t::serialize(std::ostream &out) const {
    ir_utils::serialize(hw_, out);
    ir_utils::serialize(prop_, out);
    ir_utils::serialize(src_tag_, out);
    ir_utils::serialize(wei_tag_, out);
    ir_utils::serialize(dst_tag_, out);
    ir_utils::serialize(shape_, out);
}

void problem_t::deserialize(std::istream &in) {
    ir_utils::deserialize(hw_, in);
    ir_utils::deserialize(prop_, in);
    ir_utils::deserialize(src_tag_, in);
    ir_utils::deserialize(wei_tag_, in);
    ir_utils::deserialize(dst_tag_, in);
    ir_utils::deserialize(shape_, in);
}

std::string problem_t::str() const {
    std::ostringstream oss;
    oss << "Conv problem" << std::endl;
    oss << "  HW:          " << to_string(hw_.to_ngen()) << std::endl;
    oss << "  Propagation: " << ir_utils::to_string(prop_) << std::endl;
    oss << "  Source:      " << src_tag_ << std::endl;
    oss << "  Weights:     " << wei_tag_ << std::endl;
    oss << "  Destination: " << dst_tag_ << std::endl;
    oss << "  Descriptor:  " << desc_str();
    return oss.str();
}

std::string problem_t::csv_str() const {
    std::vector<std::string> parts;
    parts.push_back(hw_.brief_str());
    parts.push_back(ir_utils::to_string(prop_));
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

prb_tile_t problem_t::default_shape() {
    static prb_tile_t _default_shape = []() {
        static prb_tile_t ret;
        ret[prb_dims::g] = 1;
        ret[prb_dims::mb] = 1;
        ret[prb_dims::id] = ret[prb_dims::ih] = ret[prb_dims::iw] = 1;
        ret[prb_dims::od] = ret[prb_dims::oh] = ret[prb_dims::ow] = 1;
        ret[prb_dims::kd] = ret[prb_dims::kh] = ret[prb_dims::kw] = 1;
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

class arg_helper_t {
public:
    arg_helper_t(prop_kind_t prop) : prop_(prop) {}

    int src_arg_key() const {
        if (is_fwd()) return DNNL_ARG_SRC;
        if (is_bwd_d()) return DNNL_ARG_DIFF_SRC;
        if (is_bwd_w()) return DNNL_ARG_SRC;
        ir_error_not_expected();
        return DNNL_ARG_UNDEF;
    }

    bool is_src_input() const { return is_fwd() || is_bwd_w(); }
    bool is_src_output() const { return is_bwd_d(); }

    int wei_arg_key() const {
        if (is_fwd()) return DNNL_ARG_WEIGHTS;
        if (is_bwd_d()) return DNNL_ARG_WEIGHTS;
        if (is_bwd_w()) return DNNL_ARG_DIFF_WEIGHTS;
        ir_error_not_expected();
        return DNNL_ARG_UNDEF;
    }

    bool is_wei_input() const { return is_fwd() || is_bwd_d(); }
    bool is_wei_output() const { return is_bwd_w(); }

    int bia_arg_key() const {
        if (is_fwd()) return DNNL_ARG_BIAS;
        if (is_bwd_d()) return DNNL_ARG_BIAS;
        if (is_bwd_w()) return DNNL_ARG_DIFF_BIAS;
        ir_error_not_expected();
        return DNNL_ARG_UNDEF;
    }

    bool is_bia_input() const { return is_fwd() || is_bwd_d(); }
    bool is_bia_output() const { return is_bwd_w(); }

    int dst_arg_key() const {
        if (is_fwd()) return DNNL_ARG_DST;
        if (is_bwd_d()) return DNNL_ARG_DIFF_DST;
        if (is_bwd_w()) return DNNL_ARG_DIFF_DST;
        ir_error_not_expected();
        return DNNL_ARG_UNDEF;
    }

    bool is_dst_input() const { return is_bwd_d() || is_bwd_w(); }
    bool is_dst_output() const { return is_fwd(); }

private:
    bool is_fwd() const { return prop_ == prop_kind::forward; }
    bool is_bwd_d() const { return prop_ == prop_kind::backward_data; }
    bool is_bwd_w() const { return prop_ == prop_kind::backward_weights; }

    prop_kind_t prop_;
};

tensor_config_t get_tensor_config(prop_kind_t prop) {
    arg_helper_t h(prop);
    tensor_config_t tensor_cfg;
    tensor_cfg.add_tensor("src", h.src_arg_key(), h.is_src_input(),
            h.is_src_output(), jit::layout_t());
    tensor_cfg.add_tensor("wei", h.wei_arg_key(), h.is_wei_input(),
            h.is_wei_output(), jit::layout_t());
    tensor_cfg.add_tensor("dst", h.dst_arg_key(), h.is_dst_input(),
            h.is_dst_output(), jit::layout_t());
    return tensor_cfg;
}

} // namespace conv
} // namespace v2
} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
