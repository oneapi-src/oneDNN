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

    int max_tgs_per_gpu(int tg_size) const {
        int tgs_per_ss
                = hw.eus_per_ss_or_dss() * hw.threads_per_eu(regs) / tg_size;
        return hw.eu_count() / hw.eus_per_ss_or_dss() * tgs_per_ss;
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
                prb.hw(), kernel_desc.fma, kernel_desc.src_tag.type());
        pvar_tile_t padded_shape = prb.shape();
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

    static std::vector<std::string> feature_names() {
        std::vector<std::string> ret;
        ret.push_back("kl");
        ret.push_back("waves");
        return ret;
    }

    vec1d to_x() const {
        std::vector<float> ret;
        ret.push_back(kl);
        ret.push_back(waves());
        return ret;
    }

    float to_y() const { return time_ns; }

    float ntgs() const {
        float ntgs = 1.0f;
        ntgs *= ir_utils::safe_div(b, bl * bt * bi);
        ntgs *= ir_utils::safe_div(m, ml * mt * mi);
        ntgs *= ir_utils::safe_div(n, nl * nt * ni);
        ntgs *= ir_utils::safe_div(k, kl * kt * ki);
        return ntgs;
    }

    float ops() const { return 2.0f * b * m * n * k; }

    float waves() const {
        int tgs_per_wave = hw_cfg.max_tgs_per_gpu(bt * mt * nt * kt);
        return ntgs() / tgs_per_wave;
    }

    float eff() const {
        float sec = time_ns / 1e9;
        return ops() / 1e9 / sec / hw_cfg.max_gops_per_sec();
    }

    static void to_bmnk(prop_kind_t prop, const pvar_tile_t &tile, int &b,
            int &m, int &n, int &k) {
        const auto t = to_gemm(tile, prop);
        b = t[pvars::b];
        m = t[pvars::m];
        n = t[pvars::n];
        k = t[pvars::k];
    }
};

float coef_kl(float x, float a, float b) {
    return 1 + 1.0f / (a * std::pow(x, b));
}

float coef_wp(float x, float a, float b) {
    return 1 - 1.0f / (a * std::pow(x, b));
}

// The performance model is based on two inputs:
// - kl:    the number of reduction iterations (integer)
// - waves: the number of thread waves to execute the kernel (may be fractional)
//
// waves input is split into wf/wp:
// - wf: number of full waves (integer)
// - wp: fractional number of waves, wp = 0 is translated to 1 to have smooth
//       function behavior.
//
// Model parameters:
// - T0 - "time per normalized wave-iteration",
//   For large kl/waves the total time is T0 * kl * wf
//
// Intermediate coefficients:
// - coef_kl = 1 + 1 / (a_kl * kl ^ b_kl)
//   This is for non-linear scaling of kl value
// - coef_wp = 1 - 1 / (a_wp * wf ^ b_wp)
//   This is for non-linear scaling of wp value
//
// The model evaluates the expected time as:
//   T = T0 * kl * coef_kl * (wf + wp * coef_wp)
//
// - For large kl/wf the coefficients approach 1
// - For small kl values (coef_kl > 1): the assumed iteration time in a small
//   loop is higher due to shorter pipeline and higher relative impact of kernel
//   prologue/epilogue
// - For small wf values (coef_wp < 1): this is to take into account the effect
//   of wave tails. For example when moving from one full wave to one full wave
//   and a few extra threadgroups a distinct increase in time is typically
//   observed. This effect is more pronounced with a smaller number of full
//   waves.
float model_t::predict(float kl, float waves, const vec1d &coef) {
    float waves_frac = waves - (int)waves;
    float wp = (waves_frac == 0 ? 1 : waves_frac);
    float wf = std::ceil(waves);
    float T0 = coef[0];
    float a_kl = coef[1];
    float b_kl = coef[2];
    float a_wp = coef[3];
    float b_wp = coef[4];
    float Tw = T0 * kl * coef_kl(kl, a_kl, b_kl);
    return Tw * (wf + wp * coef_wp(wf, a_wp, b_wp));
}

float model_t::predict(const vec1d &x) const {
    ir_assert(x.size() == 2);
    float kl = x[0];
    float waves = x[1];
    return model_t::predict(kl, waves, coef_);
}

float model_t::predict(const problem_t &prb, const kernel_desc_t &desc) const {
    sample_t s(prb, desc);
    return predict(s.to_x());
}

float model_t::eff(const problem_t &prb, const kernel_desc_t &desc) const {
    using namespace ir_utils;
    sample_t s(prb, desc);
    auto x = s.to_x();
    float raw_eff = s.ops() / predict(x);
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
        y_pred.push_back(predict(bd.prbs[i], bd.kernel_desc));
    }
}

void model_t::stringify(std::ostream &out) const {
    std::ostringstream oss;
    serialized_data_t s;
    s.append(coef_);
    for (uint8_t d : s.get_data()) {
        oss << std::uppercase << std::hex << std::setw(2) << std::setfill('0')
            << (int)d;
    }
    out << oss.str();
}

void model_t::parse(std::istream &in) {
    std::vector<uint8_t> data;
    auto s_data = stream_parse<std::string>(in);
    for (size_t i = 0; i < s_data.size(); i += 2) {
        data.push_back(std::stoi(s_data.substr(i, 2), nullptr, 16));
    }
    auto s = serialized_t::from_data(std::move(data));
    deserializer_t d(s);
    d.pop(coef_);
}

std::string to_str(const vec1d &x) {
    std::ostringstream oss;
    bool is_first = true;
    for (float f : x) {
        if (!is_first) oss << ",";
        oss << f;
        is_first = false;
    }
    return oss.str();
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

void dump_csv(const bench_data_t &bd, const model_t &model) {
    auto name = bd.kernel_desc.brief_str();
    std::ofstream out(name + ".csv");
    out << "desc,";
    for (auto &name : sample_t::feature_names()) {
        out << name << ",";
    }
    out << "time,model_time" << std::endl;
    for (int i = 0; i < bd.size(); i++) {
        sample_t s(bd.prbs[i], bd.kernel_desc, bd.times[i]);
        auto x = s.to_x();
        auto y = s.to_y();
        float model_time = model.predict(x);
        out << bd.prbs[i].desc_str() << "," << to_str(x) << "," << y << ","
            << model_time << std::endl;
    }
}

} // namespace conv
} // namespace v2
} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
