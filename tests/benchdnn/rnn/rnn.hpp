/*******************************************************************************
* Copyright 2018-2023 Intel Corporation
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

#ifndef RNN_HPP
#define RNN_HPP

#include <assert.h>
#include <limits.h>
#include <stdint.h>

#include <string>
#include <vector>

#include "common.hpp"
#include "dnn_types.hpp"
#include "dnnl_common.hpp"
#include "dnnl_debug.hpp"
#include "utils/perf_report.hpp"
#include "utils/settings.hpp"

#define AOC array_offset_calculator

namespace rnn {

enum alg_t {
    VANILLA_RNN,
    VANILLA_LSTM,
    VANILLA_GRU,
    LBR_GRU,
    VANILLA_AUGRU,
    LBR_AUGRU
};
alg_t str2alg(const char *str);
const char *alg2str(alg_t alg);
dnnl_alg_kind_t alg2kind(alg_t alg);

enum activation_t { UNDEF, RELU, LOGISTIC, TANH };
activation_t str2activation(const char *str);
const char *activation2str(activation_t alg);
dnnl_alg_kind_t activation2kind(activation_t alg);

dnnl_rnn_direction_t str2direction(const char *str);
const char *direction2str(dnnl_rnn_direction_t direction);

enum rnn_data_kind_t {
    SRC_LAYER,
    SRC_ITER,
    SRC_ITER_C,
    WEIGHTS_LAYER,
    WEIGHTS_ITER,
    BIAS,
    DST_ITER,
    DST_ITER_C,
    DST_LAYER,

    DIFF_SRC_LAYER,
    DIFF_SRC_ITER,
    DIFF_SRC_ITER_C,
    DIFF_WEIGHTS_LAYER,
    DIFF_WEIGHTS_ITER,
    DIFF_BIAS,
    DIFF_DST_ITER,
    DIFF_DST_ITER_C,
    DIFF_DST_LAYER,

    // FIXME: adding peephole related weights to the appropriate places will
    // cause false-positive accuracy check failures in unrelated test cases
    // (e.g.  backward vanilla RNN for bf16) due to the data fill seed being
    // dependent on the position of the tensor kind in the enum: adding
    // `WEIGHTS_PEEPHOLE` before `dst_*` and `*diff_*` results in initializing
    // the corresponding tensors differently.
    // We need a more robust way of testing RNN.
    WEIGHTS_PEEPHOLE,
    DIFF_WEIGHTS_PEEPHOLE,
    WEIGHTS_PROJECTION,
    DIFF_WEIGHTS_PROJECTION,
    // AUGRU requires an addtional argument for attention.
    AUGRU_ATTENTION,
    DIFF_AUGRU_ATTENTION,
    KIND_TOTAL,
};
const char *rnn_data_kind2str(rnn_data_kind_t kind);

using flags_t = unsigned;
// XXX: UNDEF is used in activation_t
const flags_t NONE = dnnl_rnn_flags_undef;
const flags_t DIFF_WEIGHTS_OVERWRITE = dnnl_rnn_flags_diff_weights_overwrite;
flags_t str2flags(const char *str);
std::string flags2str(flags_t flags);

// Gates indices
enum {
    LSTM_I = 0,
    LSTM_F = 1,
    LSTM_C = 2,
    LSTM_O = 3,
    GRU_U = 0,
    GRU_R = 1,
    GRU_O = 2,
    LBR_GRU_U_PRIME = 3,
};

// dlc is different at the cell level and the primitive level
// This enum enable to explicitely query the intended one
enum dlc_type_t { CELL, PRIMITIVE };

template <typename Telem>
struct array_offset_calculator {
    array_offset_calculator() = default;

    template <typename... Targs>
    array_offset_calculator(const dnn_mem_t &mem, Targs... dims)
        : base_ptr_(mem ? (Telem *)mem : nullptr), dims_({dims...}) {}

    template <typename... Targs>
    array_offset_calculator(Telem *base_ptr, Targs... dims)
        : base_ptr_(base_ptr), dims_({dims...}) {}

    // ctor for AOC<const T> based on const AOC<T> &
    template <typename Uelem>
    array_offset_calculator(const array_offset_calculator<Uelem> &rhs)
        : base_ptr_(rhs.base_ptr_), dims_(rhs.dims_) {}

    // to make the above ctor work AOC<const T> should be able to access
    // private fields of AOC<T>, hence let's friend them
    friend struct array_offset_calculator<const Telem>;

    template <typename... Targs>
    Telem &operator()(Targs... Fargs) const {
        assert(static_cast<bool>(base_ptr_));
        return *(base_ptr_ + offset(1, Fargs...));
    }

    int64_t nelems() const {
        int64_t res = 1;
        for (auto dim : dims_)
            res *= dim;
        return res;
    }

    void set_base_ptr(Telem *base_ptr) { base_ptr_ = base_ptr; }

private:
    template <typename... Targs>
    int64_t offset(int64_t d, int64_t pos) const {
        return pos;
    }

    template <typename... Targs>
    int64_t offset(int64_t d, int64_t off, int64_t pos) const {
        return off * dims_[d] + pos;
    }

    template <typename... Targs>
    int64_t offset(int64_t d, int64_t off, int64_t pos, Targs... rem) const {
        return offset(d + 1, off * dims_[d] + pos, rem...);
    }

    Telem *base_ptr_ = nullptr;
    std::vector<int64_t> dims_;
};

struct desc_t {
    int64_t sic;
    int64_t slc;
    int64_t dhc;
    int64_t dic;
    int64_t wc;
    int64_t mb;
    int64_t n_layer;
    int64_t n_iter;
    std::string name;
};
int str2desc(desc_t *desc, const char *str);
std::ostream &operator<<(std::ostream &s, const desc_t &d);

/** configuration structure, that controls initial data filling + error check
*
* dt defines precision
*
* for each lst data kind the values are filled as follows:
* if (rand() > f_sparsity) then:
*     v <-- f_base
* else:
*     v <-- f_min + rand() * f_step % (f_max - f_min)
*
* on final check the resulting values should be in [min .. max] range, the
* relative difference should not exceed eps
*/
struct dt_conf_t {
    struct entry_t {
        dnnl_data_type_t dt;
        int min, max; // representative
        float f_min, f_max; // fill range
        float f_mean, f_stddev; // parameters of normal distribution
        double eps; // acceptable error
    };

    dt_conf_t(const std::string &str) : str_(str) {}

    virtual const entry_t &operator[](rnn_data_kind_t kind) const = 0;

    const std::string &str() const { return str_; }
    bool is_int8() const {
        return operator[](SRC_LAYER).dt == dnnl_u8
                || operator[](SRC_LAYER).dt == dnnl_s8;
    }
    bool is_s8() const { return operator[](SRC_LAYER).dt == dnnl_s8; }

    static const dt_conf_t &create(const std::string &str, const attr_t &attr);

    std::string str_;
};

struct settings_t : public base_settings_t {
    settings_t() = default;

    // ctor to save certain fields from resetting
    settings_t(const char *perf_template) : settings_t() {
        this->perf_template = perf_template;
    }

    desc_t desc {};

    std::vector<dir_t> prop {FWD_I};
    std::vector<std::string> cfg {"f32"};
    std::vector<alg_t> alg {VANILLA_RNN};
    std::vector<dnnl_rnn_direction_t> direction {
            dnnl_unidirectional_left2right};
    std::vector<activation_t> activation {RELU};
    std::vector<bool> skip_nonlinear {false};
    std::vector<bool> trivial_strides {false};
    std::vector<bool> with_peephole {false};
    std::vector<bool> with_projection {false};
    std::vector<int64_t> n_layer {0}, n_iter {0};
    std::vector<policy_t> scale_policy {policy_t::COMMON};
    std::vector<policy_t> scale_proj_policy {policy_t::COMMON};
    std::vector<flags_t> flags {NONE};
    float alpha = 0.9f, beta = 0.0f;

    const char *perf_template_csv() const {
        static const std::string args
                = "%prop%,%cfg%,%alg%,%activation%,%direction%";
        return perf_template_csv_base(args);
    }

    void reset() { *this = settings_t(perf_template); }

    bool has_single_setup() const override {
        return prop.size() == 1 && cfg.size() == 1 && alg.size() == 1
                && direction.size() == 1 && activation.size() == 1
                && skip_nonlinear.size() == 1 && trivial_strides.size() == 1
                && with_peephole.size() == 1 && with_projection.size() == 1
                && n_layer.size() == 1 && n_iter.size() == 1
                && scale_policy.size() == 1 && scale_proj_policy.size() == 1
                && base_settings_t::has_single_setup();
    }
};

struct prb_t : public desc_t {
    // A ctor with common interface across all drivers.
    prb_t(const settings_t &s)
        : prb_t(s.desc,
                dt_conf_t::create(s.cfg[0],
                        settings_t::get_attr(s.scales[0], s.zero_points[0],
                                s.post_ops[0], s.scratchpad_mode[0],
                                s.fpmath_mode[0])),
                s.prop[0], s.alg[0], s.with_peephole[0], s.with_projection[0],
                s.direction[0], s.scale_policy[0], s.scale_proj_policy[0],
                s.flags[0], s.activation[0],
                settings_t::get_attr(s.scales[0], s.zero_points[0],
                        s.post_ops[0], s.scratchpad_mode[0], s.fpmath_mode[0]),
                s.ctx_init[0], s.ctx_exe[0], s.alpha, s.beta,
                s.skip_nonlinear[0], s.trivial_strides[0], s.n_layer[0],
                s.n_iter[0], s.mb[0]) {
        SAFE_V(s.has_single_setup() ? OK : FAIL);
        // Just for better styling, no real reason to keep it separate.
        this->attr = settings_t::get_attr(s.scales[0], s.zero_points[0],
                s.post_ops[0], s.scratchpad_mode[0], s.fpmath_mode[0]);
    }

    prb_t(const desc_t &desc, const dt_conf_t &cfg, dir_t prop, alg_t alg,
            bool with_peephole, bool with_projection,
            dnnl_rnn_direction_t direction, policy_t scale_policy,
            policy_t scale_proj_policy, unsigned int flags,
            activation_t activation, const attr_t &attr,
            const thr_ctx_t &ctx_init, const thr_ctx_t &ctx_exe, float alpha,
            float beta, bool skip_nonlinear, bool trivial_strides,
            int64_t n_layer, int64_t n_iter, int64_t mb = 0)
        : desc_t(desc)
        , cfg(cfg)
        , prop(prop2prop_kind(prop))
        , dir(prop)
        , alg(alg)
        , with_peephole(with_peephole)
        , with_projection(with_projection)
        , direction(direction)
        , wei_scales_policy(scale_policy)
        , wei_proj_scales_policy(scale_proj_policy)
        , flags(flags)
        , activation(activation)
        , attr(attr)
        , ctx_init(ctx_init)
        , ctx_exe(ctx_exe)
        , user_mb(mb)
        , alpha(alpha)
        , beta(beta)
        , skip_nonlinear(skip_nonlinear)
        , trivial_strides(trivial_strides)
        , ops(0.0)
        , linear_cscale(0.0f) {

        if (n_layer) this->n_layer = n_layer;
        if (n_iter) this->n_iter = n_iter;
        if (mb) this->mb = mb;
        count_ops();

        wei_scales = nullptr;
        wei_proj_scales = nullptr;
        linear_scales = nullptr;

        // We always allocate linear scales. Even if they are not
        // used, they get dereferenced when built in debug mode.
        linear_scales = (float *)zmalloc(sizeof(float) * n_gates(), 64);
        // Here we use the range of SRC_LAYER to set the scales
        set_tparams(cfg[SRC_LAYER].f_min, cfg[SRC_LAYER].f_max);

        switch (wei_scales_policy) {
            case policy_t::COMMON:
                wei_scales_mask = 0x0;
                wei_nscales = 1;
                break;
            case policy_t::PER_OC:
                wei_scales_mask = 0x18;
                wei_nscales = dhc * n_gates();
                break;
            default: SAFE_V(FAIL);
        }
        wei_scales = (float *)zmalloc(sizeof(float) * wei_nscales, 64);

        if (with_projection) {
            switch (wei_proj_scales_policy) {
                case policy_t::PER_OC:
                    wei_proj_scales_mask = 0x8;
                    wei_proj_nscales = dic;
                    break;
                case policy_t::COMMON:
                    wei_proj_scales_mask = 0x0;
                    wei_proj_nscales = 1;
                    break;
                default: SAFE_V(FAIL);
            }
            wei_proj_scales
                    = (float *)zmalloc(sizeof(float) * wei_proj_nscales, 64);
        }

        set_qparams(-1., 1.);
        repro = set_repro_line(); // must be last in ctor to collect right info
    }
    ~prb_t() {
        zfree(wei_scales);
        zfree(wei_proj_scales);
        zfree(linear_scales);
    }

    float get_wei_scale(int idx) const {
        return wei_scales[MIN2(idx, wei_nscales - 1)];
    }

    inline float get_wei_proj_scale(int idx) const {
        return wei_proj_scales[MIN2(idx, wei_proj_nscales - 1)];
    }

    void count_ops() {
        // Here, we count only the ops in GEMM portion as there is no
        // theoretical number of ops for the post-gemm operations
        int64_t num_cells = (int64_t)n_dir() * n_layer * n_iter;
        int64_t cell_ops = (int64_t)2 * (n_gates() * dhc) * mb * (sic + slc);
        if (with_projection) cell_ops += (int64_t)2 * dhc * mb * dic;
        int64_t prop_multiplier = prop == dnnl_backward ? 2 : 1;
        ops = prop_multiplier * num_cells * cell_ops;
    }

    int64_t n_dir() const {
        return (direction == dnnl_bidirectional_concat
                       || direction == dnnl_bidirectional_sum)
                ? 2
                : 1;
    }
    int64_t n_states() const { return alg == VANILLA_LSTM ? 2 : 1; }
    int64_t n_gates() const {
        return alg == VANILLA_LSTM
                ? 4
                : (alg == VANILLA_GRU || alg == LBR_GRU || alg == VANILLA_AUGRU
                                        || alg == LBR_AUGRU
                                ? 3
                                : 1);
    }
    int64_t n_bias() const {
        return alg == LBR_GRU || alg == LBR_AUGRU ? n_gates() + 1 : n_gates();
    }

    int64_t dlc(dlc_type_t type) const {
        if (type == PRIMITIVE)
            return (direction == dnnl_bidirectional_concat ? 2 : 1) * dic;
        if (type == CELL) return dic;
        assert(!"unsupported dlc type");
        return 0;
    }

    bool is_int8() const {
        return cfg[SRC_LAYER].dt == dnnl_u8 || cfg[SRC_LAYER].dt == dnnl_s8;
    }
    bool is_u8() const { return cfg[SRC_LAYER].dt == dnnl_u8; }
    bool is_s8() const { return cfg[SRC_LAYER].dt == dnnl_s8; }
    bool is_lstm_peephole() const { return with_peephole; }
    bool is_lstm_projection() const { return with_projection; }
    bool is_augru() const { return alg == VANILLA_AUGRU || alg == LBR_AUGRU; }

    // Used to construct memory desc when dimensions are runtime since such mds
    // can't be used directly from query and memory objects can't be constructed.
    benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> get_md(int arg) const {
        assert(!"No runtime dimensions support for this driver!");
        return make_benchdnn_dnnl_wrapper<dnnl_memory_desc_t>(nullptr);
    }

    const char *str() const { return repro.c_str(); }

    const dt_conf_t &cfg;
    dnnl_prop_kind_t prop;
    dir_t dir; // Same as `prop`, for compatibility. TODO: remove me;
    alg_t alg;
    bool with_peephole, with_projection;
    dnnl_rnn_direction_t direction;
    policy_t wei_scales_policy;
    policy_t wei_proj_scales_policy;
    unsigned int flags;
    activation_t activation;
    attr_t attr;
    thr_ctx_t ctx_init, ctx_exe;
    int64_t user_mb;
    float alpha;
    float beta;

    float data_scale, data_shift;

    float *wei_scales;
    int wei_nscales;
    int wei_scales_mask;

    float *wei_proj_scales;
    int wei_proj_nscales;
    int wei_proj_scales_mask;

    bool skip_nonlinear;
    bool trivial_strides;
    double ops;
    float *linear_scales;
    float linear_cscale;

private:
    std::string repro;

    std::string set_repro_line();

    /* Todo: fused the two functions in set_shifts_scales */
    void set_qparams(float fp_min, float fp_max);
    void set_tparams(float fp_min, float fp_max);
    prb_t(const prb_t &) = delete;
    prb_t &operator=(const prb_t &) = delete;
};

struct perf_report_t : public base_perf_report_t {
    perf_report_t(const prb_t *prb, const char *perf_template)
        : base_perf_report_t(perf_template), p_(prb) {}

    void dump_alg(std::ostream &s) const override { s << alg2str(p_->alg); }

    void dump_cfg(std::ostream &s) const override { s << p_->cfg.str(); }

    void dump_desc(std::ostream &s) const override {
        s << static_cast<const desc_t &>(*p_);
    }

    void dump_desc_csv(std::ostream &s) const override {
        s << p_->n_layer << "," << p_->n_iter << "," << p_->mb << "," << p_->sic
          << "," << p_->slc << "," << p_->dhc << "," << p_->dic;
    }

    void dump_rnn_activation(std::ostream &s) const override {
        s << activation2str(p_->activation);
    }

    void dump_rnn_direction(std::ostream &s) const override {
        s << direction2str(p_->direction);
    }

    double ops() const override { return p_->ops; }
    const int64_t *user_mb() const override { return &p_->user_mb; }
    const attr_t *attr() const override { return &p_->attr; }
    const thr_ctx_t *ctx_init() const override { return &p_->ctx_init; }
    const thr_ctx_t *ctx_exe() const override { return &p_->ctx_exe; }
    const std::string *name() const override { return &p_->name; }
    const dnnl_prop_kind_t *prop() const override { return &p_->prop; }

private:
    const prb_t *p_;
};

void prepare_ws_fwd(const prb_t &prb, std::vector<float> &ws_fwd_buffer,
        AOC<float> &ws_src_layer, AOC<float> &ws_src_iter,
        AOC<float> &ws_src_iter_c, AOC<float> &ws_gates, AOC<float> &ws_ht);

void rnn_linear_fwd(const prb_t &prb, const args_t &args,
        const AOC<float> &ws_src_layer, const AOC<float> &ws_src_iter,
        const AOC<float> &ws_src_iter_c, const AOC<float> &ws_gates,
        const AOC<float> &ws_ht);

dnnl_status_t init_pd(init_pd_args_t<prb_t> &init_pd_args);
void setup_cmp(compare::compare_t &cmp, const prb_t *prb, data_kind_t kind,
        const args_t &ref_args);
std::vector<int> supported_exec_args(dir_t dir);
int init_ref_memory_args(dnn_mem_map_t &ref_mem_map, dnn_mem_map_t &mem_map,
        dnnl_primitive_t prim, const prb_t *prb, res_t *res, dir_t dir,
        dnnl_primitive_t prim_ref = nullptr);

void skip_unimplemented_prb(const prb_t *prb, res_t *res);
void skip_invalid_prb(const prb_t *prb, res_t *res);
void compute_ref(const prb_t *prb, const args_t &args,
        dnnl_primitive_t prim_ref = nullptr);
void compute_ref_fwd(const prb_t &prb, const args_t &args);
void compute_ref_bwd(const prb_t &prb, const args_t &args);

int createit(std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const prb_t &prb, res_t *res);
int check_cacheit(
        std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const prb_t *prb, res_t *res);
int doit(const std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const prb_t &prb, res_t *res);
int bench(int argc, char **argv);

} // namespace rnn

#endif
