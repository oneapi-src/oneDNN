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

#include "gpu/intel/jit/v2/conv/model.hpp"

#include "gpu/intel/jit/v2/conv/tensor_utils.hpp"

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
    int regs = 0;

    hw_config_t() = default;
    hw_config_t(const hw_t &hw, fma_kind_t fma) : hw(hw), fma(fma) {
        regs = (utils::one_of(fma, fma_kind_t::dpas, fma_kind_t::dpasw) ? 256
                                                                        : 128);
    }

    int max_tgs_per_gpu(dim_t tg_size) const {
        int tgs_per_ss
                = hw.eus_per_ss_or_dss() * hw.threads_per_eu(regs) / tg_size;
        return hw.eu_count() / hw.eus_per_ss_or_dss() * tgs_per_ss;
    }
};

class sample_impl_t {
public:
    sample_impl_t(model_kind_t model_kind, const problem_t &prb,
            const kernel_desc_t &desc)
        : model_kind_(model_kind), prb_(prb), desc_(desc) {
        hw_cfg_ = hw_config_t(prb_.hw(), desc_.fma);
    }
    virtual ~sample_impl_t() = default;
    virtual vec1d to_x() const = 0;
    virtual float to_y() const = 0;

protected:
    model_kind_t model_kind_ = model_kind_t::undef;
    problem_t prb_;
    kernel_desc_t desc_;
    hw_config_t hw_cfg_;
};

std::vector<std::string> feature_names(model_kind_t kind) {
    switch (kind) {
        case model_kind_t::data_parallel:
            return std::vector<std::string>({"kl", "waves"});
        case model_kind_t::stream_k: return std::vector<std::string>({"iters"});
        default: gpu_error_not_expected();
    }
    return std::vector<std::string>();
}

void to_bmnk(prop_kind_t prop, const pvar_tile_t &tile, dim_t &b, dim_t &m,
        dim_t &n, dim_t &k) {
    const auto t = to_gemm(tile, prop);
    b = t[pvars::b];
    m = t[pvars::m];
    n = t[pvars::n];
    k = t[pvars::k];
}

struct bmnk_helper_t {
    dim_t b, m, n, k;
    dim_t bt, mt, nt, kt;
    dim_t bl, ml, nl, kl;
    dim_t bi, mi, ni, ki;
    dim_t tiles;
    dim_t iters;

    bmnk_helper_t(const problem_t &prb, const kernel_desc_t &desc) {
        auto padded_shape = prb.shape();
        dim_t tmp_iters = 1;
        for (auto &d : padded_shape) {
            if (!is_conv_index(d)) continue;
            dim_t tg = desc.thread_group_tile.get(d, 1);
            dim_t iter = desc.iter_tile.get(d, 1);
            dim_t dim = padded_shape[d];
            dim_t padded_dim = utils::rnd_up(dim, tg * iter);
            padded_shape[d] = padded_dim;
            if (!to_gemm(d, prb.prop()).is_undef()) {
                tmp_iters *= utils::div_up(dim, iter * tg);
            }
        }
        to_bmnk(prb.prop(), padded_shape, b, m, n, k);
        to_bmnk(prb.prop(), desc.thread_group_tile, bt, mt, nt, kt);
        to_bmnk(prb.prop(), desc.iter_tile, bi, mi, ni, ki);
        bl = ml = nl = 1;
        kl = ir_utils::safe_div(k, kt * ki);
        tiles = 1;
        tiles *= ir_utils::safe_div(b, bl * bt * bi);
        tiles *= ir_utils::safe_div(m, ml * mt * mi);
        tiles *= ir_utils::safe_div(n, nl * nt * ni);
        iters = tiles * kl;
        gpu_assert(tmp_iters == iters);
    }
};

dim_t layout_size(const layout_tag_t &tag, const problem_t &prb) {
    gpu_assert(!tag.is_any() && !tag.is_empty())
            << "Unexpected tag: " << tag.str();
    pvar_tile_t tile;
    for (auto &d : tag.desc().letter_map())
        tile[d] = prb.shape().at(d);
    dim_t elems = 1;
    for (auto &e : tag.raw_tag().entries()) {
        auto d = tag.desc().prb_dim(e.index());
        dim_t e_block = (e.block != 0 ? e.block : tile.at(d));
        elems *= e_block;
        tile[d] = utils::div_up(tile[d], e_block);
    }
    gpu_assert(tile.elems() == 1);
    return elems * tag.type().size();
}

float conv_time_nsec(const bench_time_t &time) {
    if (time.nkernels() == 0) return 0;
    if (time.nkernels() == 1) return time.total;
    gpu_assert(utils::one_of(time.nkernels(), 2, 3))
            << "Expecting zero-out -> conv [-> reorder] kernel sequence.";
    return time.kernel_times[1];
}

class data_parallel_sample_t : public sample_impl_t {
public:
    data_parallel_sample_t(const problem_t &prb, const kernel_desc_t &desc,
            const bench_time_t &time)
        : sample_impl_t(model_kind_t::data_parallel, prb, desc)
        , nsec_(conv_time_nsec(time)) {
        bmnk_helper_t h(prb, desc);
        int tgs_per_wave = hw_cfg_.max_tgs_per_gpu(h.bt * h.mt * h.nt * h.kt);
        kl_ = h.kl;
        waves_ = (float)h.tiles / tgs_per_wave;
    }

    vec1d to_x() const override {
        std::vector<float> ret;
        ret.push_back(kl_);
        ret.push_back(waves_);
        return ret;
    }

    float to_y() const override { return nsec_; }

private:
    uint64_t nsec_ = 0;
    dim_t kl_ = 0;
    float waves_ = 0;
};

class stream_k_sample_t : public sample_impl_t {
public:
    stream_k_sample_t(const problem_t &prb, const kernel_desc_t &desc,
            const bench_time_t &time)
        : sample_impl_t(model_kind_t::stream_k, prb, desc)
        , nsec_(conv_time_nsec(time)) {
        bmnk_helper_t h(prb, desc);
        iters_ = h.iters;
    }

    vec1d to_x() const override { return vec1d({(float)iters_}); }
    float to_y() const override { return nsec_; }

private:
    uint64_t nsec_ = 0;
    dim_t iters_;
};

class sample_t {
public:
    sample_t(model_kind_t kind, const problem_t &prb, const kernel_desc_t &desc,
            const bench_time_t &time = bench_time_t()) {
        switch (kind) {
            case model_kind_t::data_parallel:
                impl_ = std::make_shared<data_parallel_sample_t>(
                        prb, desc, time);
                break;
            case model_kind_t::stream_k:
                impl_ = std::make_shared<stream_k_sample_t>(prb, desc, time);
                break;
            default: gpu_error_not_expected();
        }
    }
    vec1d to_x() const { return impl_->to_x(); }
    float to_y() const { return impl_->to_y(); }

private:
    std::shared_ptr<sample_impl_t> impl_;
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
float predict_data_parallel(const vec1d &x, const vec1d &coef) {
    float kl = x[0];
    float waves = x[1];
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

float predict_stream_k(const vec1d &x, const vec1d &coef) {
    float iters = x[0];
    float a = coef[0];
    float b = coef[1];
    return a + b * iters;
}

float predict_data_copy(const problem_t &prb, const kernel_desc_t &desc) {
    auto tensor_kind = from_abc(desc.prop, tensor_kind_t::c);
    auto desc_tag = append_groups(
            tensor_kind, desc.layout_tag(tensor_kind_t::c), desc.is_dw);
    auto prb_tag = append_groups(
            tensor_kind, prb.layout_tag(tensor_kind_t::c), desc.is_dw);
    dim_t bytes = 0;
    if (desc.use_stream_k) bytes += layout_size(desc_tag, prb);
    if (prb_tag.is_any()) prb_tag = desc_tag.with_type(prb_tag.type());
    if (prb_tag != desc_tag) {
        bytes += layout_size(prb_tag, prb);
        bytes += layout_size(desc_tag, prb);
    }
    // XXX: Hardcoding for now, need a separate hardware-specific model.
    // Time is in nanoseconds.
    const int const_cost_time = 30000;
    const float time_per_byte = 1e-2f;
    return const_cost_time + time_per_byte * bytes;
}

void model_t::coef_ranges(model_kind_t kind, const vec2d &X, const vec1d &y,
        std::vector<std::string> &coef_names, vec1d &coef_init, vec1d &coef_min,
        vec1d &coef_max) {
    auto add = [&](const char *name, float init, float min, float max) {
        coef_names.emplace_back(name);
        coef_init.emplace_back(init);
        coef_min.emplace_back(min);
        coef_max.emplace_back(max);
    };
    switch (kind) {
        case model_kind_t::data_parallel:
            // Empirically-based parameter ranges.
            add("T0", 1000, 1, 100000);
            add("a_kl", 1, 0.0001f, 100);
            add("b_kl", 1, 0.0001f, 100);
            add("a_wp", 2, 1, 100);
            add("b_wp", 1, 0.0001f, 100);
            break;
        case model_kind_t::stream_k: {
            float t_min = *std::min_element(y.begin(), y.end());
            float t_max = *std::max_element(y.begin(), y.end());
            float t0 = *std::min_element(y.begin(), y.end());
            float t1 = 0;
            float x1 = 0;
            for (size_t i = 0; i < y.size(); i++) {
                if (y[i] < 0.5 * t_max) continue;
                t1 += (y[i] - t_min);
                x1 += X[i][0];
            }
            t1 /= x1;
            add("T0", t0, t0 / 10, t0 * 10);
            add("T1", t1, t1 / 10, t1 * 10);
            break;
        }
        default:
            gpu_error_not_expected() << "Unknown kind: " << to_string(kind);
    }
}

float model_t::predict(model_kind_t kind, const vec1d &x, const vec1d &coef) {
    switch (kind) {
        case model_kind_t::data_parallel: return predict_data_parallel(x, coef);
        case model_kind_t::stream_k: return predict_stream_k(x, coef);
        default:
            gpu_error_not_expected() << "Unknown kind: " << to_string(kind);
    }
    return 0;
}

float model_t::predict(const vec1d &x) const {
    return predict(kind_, x, coef_);
}

float model_t::predict(const problem_t &prb, const kernel_desc_t &desc) const {
    sample_t s(kind_, prb, desc);
    return predict(s.to_x());
}

void model_t::score(const bench_data_t &bd) {
    vec2d X;
    X.reserve(bd.size());
    vec1d y_test;
    vec1d y_pred;
    for (int i = 0; i < bd.size(); i++) {
        sample_t s(kind_, bd.prbs[i], bd.kernel_desc, bd.times[i]);
        y_test.push_back(s.to_y());
        y_pred.push_back(predict(bd.prbs[i], bd.kernel_desc));
    }
}

size_t model_t::coef_count(model_kind_t kind) {
    switch (kind) {
        case model_kind_t::data_parallel: return 5;
        case model_kind_t::stream_k: return 2;
        default:
            gpu_error_not_expected() << "Unknown kind: " << to_string(kind);
    }
    return 0;
}

std::string model_t::str() const {
    using namespace ir_utils;
    std::ostringstream oss;
    oss << to_string(kind_) << ": " << coef_;
    return oss.str();
}

bool with_data_copy(const problem_t &prb, const kernel_desc_t &desc) {
    if (desc.use_stream_k) return true;
    auto &prb_tag = prb.layout_tag(tensor_kind_t::c);
    auto &desc_tag = desc.layout_tag(tensor_kind_t::c);
    bool is_layout_compatible
            = (prb_tag.is_any() || prb_tag.raw_tag() == desc_tag.raw_tag());
    bool is_type_compatible = (prb_tag.type().size() == desc_tag.type().size());
    if (is_layout_compatible && is_type_compatible) return false;
    if (is_layout_compatible
            && desc.ext.has(extensions_t::out_size(prb_tag.type().size())))
        return false;
    return !is_layout_compatible || !is_type_compatible;
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

float model_set_t::time(const problem_t &prb, const kernel_desc_t &desc) const {
    float ret = 0;
    if (desc.use_stream_k) {
        ret += time(model_kind_t::stream_k, prb, desc);
    } else {
        ret += time(model_kind_t::data_parallel, prb, desc);
    }
    if (with_data_copy(prb, desc)) ret += predict_data_copy(prb, desc);
    return ret;
}

float model_set_t::time(model_kind_t kind, const problem_t &prb,
        const kernel_desc_t &desc) const {
    for (auto &m : models_) {
        if (m.kind() == kind) return m.predict(prb, desc);
    }
    gpu_error_not_expected() << "Unknown kind: " << to_string(kind);
    return 0;
}

void model_set_t::stringify(std::ostream &out) const {
    serialized_data_t s;
    for (auto &m : models_) {
        s.append(m.kind());
        for (auto &c : m.coef()) {
            s.append(c);
        }
    }
    out << data_to_hex(s.get_data());
}

void model_set_t::parse(std::istream &in) {
    auto s_data = stream_parse<std::string>(in);
    auto s = serialized_t::from_data(hex_to_data(s_data));
    deserializer_t d(s);
    while (!d.empty()) {
        auto kind = d.pop<model_kind_t>();
        size_t coef_count = model_t::coef_count(kind);
        vec1d coef(coef_count);
        for (size_t i = 0; i < coef_count; i++) {
            d.pop(coef[i]);
        }
        models_.emplace_back(kind, coef);
    }
}

std::string model_set_t::str() const {
    std::ostringstream oss;
    bool is_first = true;
    for (auto &m : models_) {
        if (!is_first) oss << std::endl;
        oss << m.str();
        is_first = false;
    }
    return oss.str();
}

void to_model_data(
        model_kind_t kind, const bench_data_t &bd, vec2d &X, vec1d &y) {
    X.clear();
    y.clear();
    X.reserve(bd.size());
    y.reserve(bd.size());
    for (int i = 0; i < bd.size(); i++) {
        sample_t s(kind, bd.prbs[i], bd.kernel_desc, bd.times[i]);
        X.push_back(s.to_x());
        y.push_back(s.to_y());
    }
}

void dump_csv(const bench_data_t &bd, const model_t &model) {
    auto name = bd.kernel_desc.brief_str();
    std::ofstream out(name + ".csv");
    out << "desc,";
    for (auto &name : feature_names(model.kind())) {
        out << name << ",";
    }
    out << "time,model_time" << std::endl;
    for (int i = 0; i < bd.size(); i++) {
        sample_t s(model.kind(), bd.prbs[i], bd.kernel_desc, bd.times[i]);
        auto x = s.to_x();
        auto y = s.to_y();
        float model_time = model.predict(x);
        out << bd.prbs[i].desc_str() << "," << to_str(x) << "," << y << ","
            << model_time << std::endl;
    }
}

void dump_model_params(const kernel_desc_t &kernel_desc, const model_t &model) {
    auto name = kernel_desc.brief_str();
    std::ofstream out(name + "_params.txt");
    bool is_first = true;
    for (auto &c : model.coef()) {
        if (!is_first) out << ", ";
        out << c;
        is_first = false;
    }
}

} // namespace conv
} // namespace v2
} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
