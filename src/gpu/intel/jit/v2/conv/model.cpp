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

#include "gpu/intel/jit/v2/conv/model.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {
namespace v2 {
namespace conv {

struct hw_config_t {
    hw_t hw;
    fma_kind_t fma = fma_kind_t::undef;
    type_t type;
    int regs = 0;

    hw_config_t() = default;
    hw_config_t(const hw_t &hw, fma_kind_t fma, const type_t &type)
        : hw(hw), fma(fma), type(type) {
        regs = (utils::one_of(fma, fma_kind_t::dpas, fma_kind_t::dpasw) ? 256
                                                                        : 128);
    }

    int max_tgs_per_gpu() const {
        int ss_per_gpu = hw.eu_count() / hw.eus_per_ss_or_dss();
        return ss_per_gpu * hw.threads_per_eu(regs);
    }

    int max_threads() const { return hw.eu_count() * hw.threads_per_eu(regs); }

    int f32_mad_ops_per_clock() const {
        switch (hw.to_ngen()) {
            case ngen::HW::XeHPC: return 32;
            default: ir_error_not_expected();
        }
        return 0;
    }

    int int8_dpas_ops_per_clock() const {
        switch (hw.to_ngen()) {
            case ngen::HW::XeHPC: return 1024;
            default: ir_error_not_expected();
        }
        return 0;
    }

    int ops_per_clock(fma_kind_t fma, const type_t &type) const;

    int ops_per_clock() const {
        bool is_mad = (fma == fma_kind_t::mad);
        bool is_dpas = utils::one_of(fma, fma_kind_t::dpas, fma_kind_t::dpasw);
        switch (type.size()) {
            case 1: {
                return is_dpas ? int8_dpas_ops_per_clock()
                               : f32_mad_ops_per_clock() * 4;
            }
            case 2: {
                return is_dpas ? int8_dpas_ops_per_clock() / 2
                               : f32_mad_ops_per_clock() * 2;
            }
            case 4: {
                return is_dpas ? int8_dpas_ops_per_clock() / 4
                               : f32_mad_ops_per_clock() * 1;
            }
            case 8: {
                ir_assert(is_mad);
                return f32_mad_ops_per_clock() / 2;
            }
            default: ir_error_not_expected();
        }
        return 0;
    }

    float freq() const {
        switch (hw.to_ngen()) {
            case ngen::HW::XeHPC: return 1.6e9;
            default: ir_error_not_expected();
        }
        return 0;
    }

    float max_gops_per_sec() const {
        float max_ops_per_sec = freq() * hw.eu_count() * ops_per_clock();
        return max_ops_per_sec / 1e9;
    }
};

struct sample_t {
    problem_t prb;
    kernel_desc_t kernel_desc;
    uint64_t time_ns = 0;

    hw_config_t hw_cfg;
    int b, m, n, k;
    int bt, mt, nt, kt;
    int bl, ml, nl, kl;
    int bi, mi, ni, ki;
    float pad_eff = 0;

    sample_t() = default;
    sample_t(const problem_t &prb, const kernel_desc_t &kernel_desc,
            uint64_t time_ns = 0)
        : prb(prb), kernel_desc(kernel_desc), time_ns(time_ns) {
        hw_cfg = hw_config_t(
                kernel_desc.hw, kernel_desc.fma, kernel_desc.src_tag.type());
        prb_tile_t padded_shape = prb.shape();
        pad_eff = 1;
        for (auto &d : padded_shape) {
            if (!is_conv_index(d)) continue;
            int tg = kernel_desc.thread_group_tile.get(d, 1);
            int iter = kernel_desc.iter_tile.get(d, 1);
            int dim = padded_shape[d];
            int padded_dim = utils::rnd_up(dim, tg * iter);
            padded_shape[d] = padded_dim;
            pad_eff *= ((float)dim / padded_dim);
        }
        to_bmnk(prb.prop(), padded_shape, b, m, n, k);
        to_bmnk(prb.prop(), kernel_desc.thread_group_tile, bt, mt, nt, kt);
        to_bmnk(prb.prop(), kernel_desc.iter_tile, bi, mi, ni, ki);
        bl = ml = nl = 1;
        kl = ir_utils::safe_div(k, kt * ki);
    }

    vec1d to_x() const {
        std::vector<float> ret;
        ret.push_back(thr_util());
        ret.push_back(tg_util());
        ret.push_back(inv_kl());
        return ret;
    }

    float to_y() const { return eff(); }

    float thr_util() const {
        return std::min(1.0f, threads() / (float)hw_cfg.max_threads());
    }

    float tg_util() const {
        float ntgs = 1.0f;
        ntgs *= ir_utils::safe_div(b, bl * bt * bi);
        ntgs *= ir_utils::safe_div(m, ml * mt * mi);
        ntgs *= ir_utils::safe_div(n, nl * nt * ni);
        ntgs *= ir_utils::safe_div(k, kl * kt * ki);
        return std::min(1.0f, ntgs / hw_cfg.max_tgs_per_gpu());
    }

    float inv_kl() const {
        int iters = bl * ml * nl * kl;
        return 1.0f / iters;
    }

    int64_t threads() const {
        int64_t ret = 1;
        ret *= ir_utils::safe_div(b, bl * bi);
        ret *= ir_utils::safe_div(m, ml * mi);
        ret *= ir_utils::safe_div(n, nl * ni);
        ret *= ir_utils::safe_div(k, kl * ki);
        return ret;
    }

    float ops() const { return 2.0f * b * m * n * k; }

    float eff() const {
        float sec = time_ns / 1e9;
        return ops() / 1e9 / sec / hw_cfg.max_gops_per_sec();
    }

    static void to_bmnk(prop_kind_t prop, const prb_tile_t &tile, int &b,
            int &m, int &n, int &k) {
        const auto t = to_gemm(tile, prop);
        b = t[prb_dims::b];
        m = t[prb_dims::m];
        n = t[prb_dims::n];
        k = t[prb_dims::k];
    }
};

float model_t::predict(const problem_t &prb) const {
    sample_t s(prb, kernel_desc_);
    return ml_model_.predict(s.to_x());
}

float model_t::eff(const problem_t &prb) const {
    sample_t s(prb, kernel_desc_);
    float raw_eff = ml_model_.predict(s.to_x());
    return raw_eff * s.pad_eff;
}

void model_t::score(const bench_data_t &bd) {
    vec2d X;
    X.reserve(bd.size());
    vec1d y_test;
    vec1d y_pred;
    for (int i = 0; i < bd.size(); i++) {
        sample_t s(bd.prbs[i], bd.kernel_desc, bd.times[i]);
        y_test.push_back(s.to_y());
        y_pred.push_back(predict(bd.prbs[i]));
    }
}

void model_t::serialize(std::ostream &out) const {
    ir_utils::serialize(kernel_desc_, out);
    ir_utils::serialize(ml_model_, out);
}

void model_t::deserialize(std::istream &in) {
    ir_utils::deserialize(kernel_desc_, in);
    ir_utils::deserialize(ml_model_, in);
}

void to_model_xy(const bench_data_t &bd, vec2d &X, vec1d &y) {
    X.clear();
    y.clear();
    X.reserve(bd.size());
    y.reserve(bd.size());
    for (int i = 0; i < bd.size(); i++) {
        sample_t s(bd.prbs[i], bd.kernel_desc, bd.times[i]);
        X.push_back(s.to_x());
        y.push_back(s.to_y());
    }
}

} // namespace conv
} // namespace v2
} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
