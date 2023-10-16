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
        oss << "simd=" << simd_;
        oss << " l=" << loop_;
        oss << " T=" << thread_group_;
        oss << " i=" << iter_;
    }
    return oss.str();
}

prb_tile_t get_conv_shape(const conv_config_t &cfg, bool pad) {
    auto &prb = cfg.prb();
    prb_tile_t ret;
#define SET(name) \
    ret[prb_dims::name] \
            = (pad ? utils::rnd_up(prb.name, cfg.pad_block(prb_dims::name)) \
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
    for (auto &d : conv_index_dims(cfg.prb().prop_kind())) {
        int loop = cfg.loop_dims()(d);
        int tg = cfg.thread_group_dims()(d);
        int iter = cfg.iter_dims()(d);
        if (loop != 1) blocking_.set_loop(d, loop);
        if (tg != 1) blocking_.set_thread_group(d, tg);
        if (iter != 1) blocking_.set_iter(d, iter);
    }
    blocking_.set_simd(cfg.vec_size());
    if (!cfg.slm() && !cfg.prefetch()) bufs_hint_ = 0;
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
    cfg.set_bufs_hint(bufs_hint_);
}

void conv_params_t::serialize(std::ostream &out) const {
    blocking_.serialize(out);
    ir_utils::serialize(bufs_hint_, out);
}

void conv_params_t::deserialize(std::istream &in) {
    blocking_.deserialize(in);
    bufs_hint_ = ir_utils::deserialize<int>(in);
}

std::string conv_params_t::str(bool csv) const {
    std::ostringstream oss;
    if (csv) {
        oss << blocking_.str(csv);
        oss << "," << bufs_hint_;
    } else {
        oss << "cfg=\"";
        oss << blocking_.str(csv);
        if (bufs_hint_ == 0) oss << " s=x0 p=x0";
        oss << "\"";
    }
    return oss.str();
}

std::vector<std::string> conv_params_t::csv_keys() {
    return {"simd", "loop", "tg", "iter", "bufs_hint"};
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
