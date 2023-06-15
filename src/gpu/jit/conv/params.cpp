/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "gpu/jit/conv/params.hpp"

#include <fstream>
#include <type_traits>

#include "gpu/jit/conv/config.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

std::string to_string(gemm_dim_kind_t kind) {
    std::ostringstream oss;
    switch (kind) {
#define CASE(name) \
    case gemm_dim_kind_t::name: return #name
        CASE(b);
        CASE(m);
        CASE(n);
        CASE(k);
#undef CASE
        default: ir_error_not_expected();
    }
    return oss.str();
}

std::string to_string(conv_dim_kind_t kind) {
    std::ostringstream oss;
    switch (kind) {
#define CASE(name) \
    case conv_dim_kind_t::name: return #name
        CASE(undef);
        CASE(g);
        CASE(ic);
        CASE(id);
        CASE(ih);
        CASE(iw);
        CASE(kd);
        CASE(kh);
        CASE(kw);
        CASE(mb);
        CASE(oc);
        CASE(od);
        CASE(oh);
        CASE(ow);
#undef CASE
        default: ir_error_not_expected();
    }
    return oss.str();
}

namespace gemm_dims {
gemm_dim_t b(gemm_dim_kind_t::b);
gemm_dim_t m(gemm_dim_kind_t::m);
gemm_dim_t n(gemm_dim_kind_t::n);
gemm_dim_t k(gemm_dim_kind_t::k);
} // namespace gemm_dims

namespace conv_dims {
conv_dim_t g(conv_dim_kind_t::g);
conv_dim_t ic(conv_dim_kind_t::ic);
conv_dim_t id(conv_dim_kind_t::id);
conv_dim_t ih(conv_dim_kind_t::ih);
conv_dim_t iw(conv_dim_kind_t::iw);
conv_dim_t kd(conv_dim_kind_t::kd);
conv_dim_t kh(conv_dim_kind_t::kh);
conv_dim_t kw(conv_dim_kind_t::kw);
conv_dim_t mb(conv_dim_kind_t::mb);
conv_dim_t oc(conv_dim_kind_t::oc);
conv_dim_t od(conv_dim_kind_t::od);
conv_dim_t oh(conv_dim_kind_t::oh);
conv_dim_t ow(conv_dim_kind_t::ow);
} // namespace conv_dims

const std::vector<conv_dim_t> &get_conv_dims(prop_kind_t prop) {
    auto get_dims = [&](prop_kind_t prop) {
        std::vector<conv_dim_t> ret;
        ret.push_back(conv_dims::mb);
        ret.push_back(conv_dims::g);
        ret.push_back(conv_dims::oc);
        ret.push_back(conv_dims::ic);
        ret.push_back(conv_dims::kd);
        ret.push_back(conv_dims::kh);
        ret.push_back(conv_dims::kw);
        if (prop != prop_kind::backward_data) {
            ret.push_back(conv_dims::od);
            ret.push_back(conv_dims::oh);
            ret.push_back(conv_dims::ow);
        } else {
            ret.push_back(conv_dims::id);
            ret.push_back(conv_dims::ih);
            ret.push_back(conv_dims::iw);
        }
        return ret;
    };
    static std::vector<conv_dim_t> fwd_dims = get_dims(prop_kind::forward);
    static std::vector<conv_dim_t> bwd_d_dims
            = get_dims(prop_kind::backward_data);
    static std::vector<conv_dim_t> bwd_w_dims
            = get_dims(prop_kind::backward_weights);
    switch (prop) {
        case prop_kind::forward: return fwd_dims;
        case prop_kind::backward_data: return bwd_d_dims;
        case prop_kind::backward_weights: return bwd_w_dims;
        default: ir_error_not_expected();
    }
    return fwd_dims;
}

template <typename KeyT>
void tile_generic_t<KeyT>::serialize(std::ostream &out) const {
    ir_utils::serialize(nkeys_, out);
    using key_int_type =
            typename std::underlying_type<typename KeyT::kind_type>::type;
    for (int i = 0; i < KeyT::max_id(); i++) {
        auto &e = entries_[i];
        if (e.index == -1) continue;
        ir_utils::serialize((key_int_type)i, out);
        ir_utils::serialize((key_int_type)e.index, out);
        ir_utils::serialize(e.value, out);
    }
}

template <typename KeyT>
void tile_generic_t<KeyT>::deserialize(std::istream &in) {
    nkeys_ = ir_utils::deserialize<int>(in);
    std::fill(entries_.begin(), entries_.end(), entry_t());
    using key_int_type =
            typename std::underlying_type<typename KeyT::kind_type>::type;
    for (int key_idx = 0; key_idx < nkeys_; key_idx++) {
        auto i = ir_utils::deserialize<key_int_type>(in);
        auto &e = entries_[i];
        e.index = ir_utils::deserialize<key_int_type>(in);
        e.value = ir_utils::deserialize<decltype(e.value)>(in);
    }
}

template class tile_generic_t<gemm_dim_t>;
template class tile_generic_t<conv_dim_t>;

gemm_dim_t to_gemm(const conv_dim_t &d, prop_kind_t prop, bool is_transpose) {
    bool is_fwd = (prop == prop_kind::forward);
    bool is_bwd_d = (prop == prop_kind::backward_data);
    bool is_bwd_w = (prop == prop_kind::backward_weights);
    auto transpose_gemm = [](const gemm_dim_t &d) {
        if (d == gemm_dims::m) return gemm_dims::n;
        if (d == gemm_dims::n) return gemm_dims::m;
        if (d == gemm_dims::k) return gemm_dims::k;
        ir_error_not_expected();
        return gemm_dim_t();
    };
    auto pick = [&](const gemm_dim_t &fwd, const gemm_dim_t &bwd_d,
                        const gemm_dim_t &bwd_w) {
        if (is_transpose) {
            if (is_fwd) return transpose_gemm(fwd);
            if (is_bwd_d) return transpose_gemm(bwd_d);
            if (is_bwd_w) return transpose_gemm(bwd_w);
        }
        if (is_fwd) return fwd;
        if (is_bwd_d) return bwd_d;
        if (is_bwd_w) return bwd_w;
        ir_error_not_expected();
        return gemm_dim_t();
    };
    switch (d.kind()) {
        case conv_dim_kind_t::g: return gemm_dims::b;
        case conv_dim_kind_t::mb:
            return pick(gemm_dims::m, gemm_dims::m, gemm_dims::k);
        case conv_dim_kind_t::oc:
            return pick(gemm_dims::n, gemm_dims::k, gemm_dims::n);
        case conv_dim_kind_t::ic:
            return pick(gemm_dims::k, gemm_dims::n, gemm_dims::m);
        case conv_dim_kind_t::kd:
        case conv_dim_kind_t::kh:
        case conv_dim_kind_t::kw:
            return pick(gemm_dims::k, gemm_dims::k, gemm_dims::m);
        case conv_dim_kind_t::od:
        case conv_dim_kind_t::oh:
        case conv_dim_kind_t::ow:
            return pick(gemm_dims::m, gemm_dim_t(), gemm_dims::k);
        case conv_dim_kind_t::id:
        case conv_dim_kind_t::ih:
        case conv_dim_kind_t::iw:
            return pick(gemm_dim_t(), gemm_dims::m, gemm_dim_t());
        default: ir_error_not_expected();
    }
    return gemm_dim_t();
}

gemm_tile_t to_gemm(const conv_tile_t &t, prop_kind_t prop, bool is_transpose) {
    gemm_tile_t ret;
    for (auto d : t) {
        auto gemm_d = to_gemm(d, prop, is_transpose);
        if (!ret.has(gemm_d)) ret[gemm_d] = 1;
        ret[gemm_d] *= t[d];
    }
    return ret;
}

void blocking_t::serialize(std::ostream &out) const {
    ir_utils::serialize(simd_, out);
    loop_.serialize(out);
    thread_group_.serialize(out);
    iter_.serialize(out);
}

void blocking_t::deserialize(std::istream &in) {
    simd_ = ir_utils::deserialize<int>(in);
    loop_.deserialize(in);
    thread_group_.deserialize(in);
    iter_.deserialize(in);
}

std::string blocking_t::str(bool csv) const {
    std::ostringstream oss;
    if (csv) {
        oss << simd_;
        oss << "," << loop_;
        oss << "," << thread_group_;
        oss << "," << iter_;
    } else {
        oss << "cfg=\"";
        oss << "simd=" << simd_;
        oss << " l=" << loop_;
        oss << " T=" << thread_group_;
        oss << " i=" << iter_;
        oss << "\"";
    }
    return oss.str();
}

conv_tile_t get_conv_shape(const conv_config_t &cfg, bool pad) {
    auto &prb = cfg.prb();
    conv_tile_t ret;
#define SET(name) \
    ret[conv_dims::name] \
            = (pad ? utils::rnd_up(prb.name, cfg.pad_block(conv_dims::name)) \
                   : prb.name)
    SET(mb);
    SET(g);
    SET(oc);
    SET(ic);
    SET(kd);
    SET(kh);
    SET(kw);
    if (prb.is_fwd || prb.is_bwd_w) {
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

conv_params_t::conv_params_t(const conv_config_t &cfg) {
    bufs_ = 0;
    if (cfg.slm()) bufs_ = cfg.slm().bufs();
    if (cfg.prefetch()) bufs_ = cfg.prefetch().bufs();
    for (auto &d : get_conv_dims(cfg.prb().prop_kind())) {
        int loop = cfg.loop_dims()(d);
        int tg = cfg.thread_group_dims()(d);
        int iter = cfg.iter_dims()(d);
        if (loop != 1) blocking_.set_loop(d, loop);
        if (tg != 1) blocking_.set_thread_group(d, tg);
        if (iter != 1) blocking_.set_iter(d, iter);
    }
    blocking_.set_simd(cfg.vec_size());
}

bool conv_params_t::is_empty() const {
    return blocking_.is_empty();
}

void conv_params_t::apply_to(conv_config_t &cfg) const {
    ir_assert(!is_empty());
    if (!cfg.loop_dims().is_overridden()) cfg.loop_dims().set(blocking_.loop());
    if (!cfg.thread_group_dims().is_overridden())
        cfg.thread_group_dims().set(blocking_.thread_group());
    if (!cfg.iter_dims().is_overridden()) cfg.iter_dims().set(blocking_.iter());
    cfg.set_params_id(id_);
}

void conv_params_t::serialize(std::ostream &out) const {
    blocking_.serialize(out);
    ir_utils::serialize(bufs_, out);
}

void conv_params_t::deserialize(std::istream &in) {
    blocking_.deserialize(in);
    bufs_ = ir_utils::deserialize<int>(in);
}

std::string conv_params_t::str(bool csv) const {
    std::ostringstream oss;
    oss << blocking_.str(csv);
    return oss.str();
}

std::vector<std::string> conv_params_t::csv_keys() {
    return {"simd", "loop", "tg", "iter"};
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
