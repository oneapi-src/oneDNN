/*******************************************************************************
* Copyright 2018-2019 Intel Corporation
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

#include "dnnl.h"

#include "norm.hpp"
#include "rnn/rnn_aux.hpp"

namespace rnn {

alg_t str2alg(const char *str) {
#define CASE(_alg) \
    if (!strcasecmp(STRINGIFY(_alg), str)) return _alg
    CASE(VANILLA_RNN);
    CASE(VANILLA_LSTM);
    CASE(VANILLA_GRU);
    CASE(LBR_GRU);
#undef CASE
    assert(!"unknown algorithm");
    return VANILLA_RNN;
}

const char *alg2str(alg_t alg) {
    if (alg == VANILLA_RNN) return "VANILLA_RNN";
    if (alg == VANILLA_LSTM) return "VANILLA_LSTM";
    if (alg == VANILLA_GRU) return "VANILLA_GRU";
    if (alg == LBR_GRU) return "LBR_GRU";
    assert(!"unknown algorithm");
    return "unknown algorithm";
}

dnnl_alg_kind_t alg2kind(alg_t alg) {
    if (alg == VANILLA_RNN) return dnnl_vanilla_rnn;
    if (alg == VANILLA_LSTM) return dnnl_vanilla_lstm;
    if (alg == VANILLA_GRU) return dnnl_vanilla_gru;
    if (alg == LBR_GRU) return dnnl_lbr_gru;
    assert(!"unknown algorithm");
    return dnnl_alg_kind_undef;
}

activation_t str2activation(const char *str) {
#define CASE(_act) \
    if (!strcasecmp(STRINGIFY(_act), str)) return _act
    CASE(RELU);
    CASE(LOGISTIC);
    CASE(TANH);
    CASE(UNDEF);
#undef CASE
    assert(!"unknown activation");
    return TANH;
}

const char *activation2str(activation_t act) {
    const char *str = "unknown activation";
    switch (act) {
        case RELU: str = "RELU"; break;
        case LOGISTIC: str = "LOGISTIC"; break;
        case TANH: str = "TANH"; break;
        case UNDEF: str = "UNDEF"; break;
        default: assert(!"unknown activation");
    }
    return str;
}

dnnl_alg_kind_t activation2kind(activation_t act) {
    dnnl_alg_kind_t alg_kind = dnnl_alg_kind_undef;
    switch (act) {
        case RELU: alg_kind = dnnl_eltwise_relu; break;
        case LOGISTIC: alg_kind = dnnl_eltwise_logistic; break;
        case TANH: alg_kind = dnnl_eltwise_tanh; break;
        case UNDEF: alg_kind = dnnl_alg_kind_undef; break;
        default: assert(!"unknown activation");
    }
    return alg_kind;
}

dnnl_rnn_direction_t str2direction(const char *str) {
    if (!strcasecmp("left2right", str)) return dnnl_unidirectional_left2right;
    if (!strcasecmp("right2left", str)) return dnnl_unidirectional_right2left;
    if (!strcasecmp("concat", str)) return dnnl_bidirectional_concat;
    if (!strcasecmp("sum", str)) return dnnl_bidirectional_sum;
    assert(!"unknown direction");
    return dnnl_unidirectional_left2right;
}

const char *direction2str(dnnl_rnn_direction_t direction) {
    if (direction == dnnl_unidirectional_left2right) return "left2right";
    if (direction == dnnl_unidirectional_right2left) return "right2left";
    if (direction == dnnl_bidirectional_concat) return "concat";
    if (direction == dnnl_bidirectional_sum) return "sum";
    assert(!"unknown direction");
    return "unknown direction";
}

void check_case_validity(const dt_conf_t *cfg, policy_t policy) {
    if (is_cfg_u8(cfg)
            && (policy != policy_t::COMMON && policy != policy_t::PER_OC)) {
        fprintf(stderr,
                "%s driver: configuration `%s` requires scale policy "
                "to be policy_t::COMMON or policy_t::PER_OC, exiting...\n",
                driver_name, cfg2str(cfg));
        exit(2);
    }
    if (!(policy == policy_t::NONE || policy == policy_t::COMMON
                || policy == policy_t::PER_OC)) {
        fprintf(stderr,
                "rnn driver: scale_policy `%s` is not supported, exiting...\n",
                attr_t::scale_t::policy2str(policy));
        exit(2);
    }
}

int str2desc(desc_t *desc, const char *str) {
    desc_t d {0};

    /* canonical form:
     * lXtXmbXsicXslcXdicXdlcX
     *
     * where: X is number, S - string
     * note: symbol `_` is ignored
     *
     * implicit rules:
     *  - default values:
     *      l = 1, t = 1, mb = 2
     *  - if slc/dlc/dic is undefined => slc/dlc/dic = sic
     */

    d.n_layer = 1;
    d.n_iter = 1;
    d.mb = 2;

    const char *s = str;
    assert(s);

#define CASE_NN(p, c) \
    do { \
        if (!strncmp(p, s, strlen(p))) { \
            ok = 1; \
            s += strlen(p); \
            char *end_s; \
            d.c = strtol(s, &end_s, 10); \
            s += (end_s - s); \
            if (d.c < 0) return FAIL; \
        } \
    } while (0)
#define CASE_N(c) CASE_NN(#c, c)
    while (*s) {
        int ok = 0;
        CASE_NN("l", n_layer);
        CASE_NN("t", n_iter);
        CASE_N(mb);
        CASE_N(sic);
        CASE_N(slc);
        CASE_N(dic);
        CASE_N(dlc);
        if (*s == 'n') {
            d.name = s + 1;
            break;
        }
        if (*s == '_') ++s;
        if (!ok) return FAIL;
    }
#undef CASE_NN
#undef CASE_N

    if (d.sic == 0) return FAIL;
    if (d.slc == 0) d.slc = d.sic;
    if (d.dlc == 0) d.dlc = d.sic;
    if (d.dic == 0) d.dic = d.sic;

    *desc = d;

    return OK;
}

std::ostream &operator<<(std::ostream &s, const desc_t &d) {
    s << "l" << d.n_layer << "t" << d.n_iter << "mb" << d.mb << "sic" << d.sic
      << "slc" << d.slc << "dic" << d.dic << "dlc" << d.dlc;

    if (d.name) s << "n" << d.name;

    return s;
}

std::ostream &operator<<(std::ostream &s, const prb_t &p) {
    dump_global_params(s);

    s << "--prop=" << prop2str(p.prop) << " ";
    s << "--alg=" << alg2str(p.alg) << " ";
    s << "--skip-nonlinear=" << bool2str(p.skip_nonlinear) << " ";
    if (canonical || p.alg == VANILLA_LSTM)
        s << "--with-peephole=" << bool2str(p.with_peephole) << " ";
    if (canonical || p.alg == VANILLA_RNN)
        s << "--activation=" << activation2str(p.activation) << " ";

    s << "--direction=" << direction2str(p.direction) << " ";
    s << "--cfg=" << cfg2str(p.cfg) << " ";
    s << "--scaling=" << attr_t::scale_t::policy2str(p.scale_policy) << " ";

    s << static_cast<const desc_t &>(p);

    return s;
}

dnnl_status_t init_rnn_fwd_desc(dnnl_rnn_desc_t *rd, const prb_t &p,
        dnnl_prop_kind_t prop_kind, dnnl_memory_desc_t *src_layer_d,
        dnnl_memory_desc_t *src_iter_d, dnnl_memory_desc_t *src_iter_c_d,
        dnnl_memory_desc_t *weights_layer_d, dnnl_memory_desc_t *weights_iter_d,
        dnnl_memory_desc_t *weights_peephole_d, dnnl_memory_desc_t *bias_d,
        dnnl_memory_desc_t *dst_layer_d, dnnl_memory_desc_t *dst_iter_d,
        dnnl_memory_desc_t *dst_iter_c_d) {
    dnnl_alg_kind_t kind = alg2kind(p.alg);
    dnnl_alg_kind_t f = activation2kind(p.activation);

    dnnl_status_t init_status;
    switch (kind) {
        case dnnl_vanilla_rnn:
            init_status = dnnl_vanilla_rnn_forward_desc_init(rd, prop_kind, f,
                    p.direction, src_layer_d, src_iter_d, weights_layer_d,
                    weights_iter_d, bias_d, dst_layer_d, dst_iter_d, p.flags,
                    p.alpha, p.beta);
            break;
        case dnnl_vanilla_lstm:
            init_status = dnnl_lstm_forward_desc_init_v2(rd, prop_kind,
                    p.direction, src_layer_d, src_iter_d, src_iter_c_d,
                    weights_layer_d, weights_iter_d, weights_peephole_d, bias_d,
                    dst_layer_d, dst_iter_d, dst_iter_c_d, p.flags);
            break;
        case dnnl_vanilla_gru:
            init_status = dnnl_gru_forward_desc_init(rd, prop_kind, p.direction,
                    src_layer_d, src_iter_d, weights_layer_d, weights_iter_d,
                    bias_d, dst_layer_d, dst_iter_d, p.flags);
            break;
        case dnnl_lbr_gru:
            init_status = dnnl_lbr_gru_forward_desc_init(rd, prop_kind,
                    p.direction, src_layer_d, src_iter_d, weights_layer_d,
                    weights_iter_d, bias_d, dst_layer_d, dst_iter_d, p.flags);
            break;
        default: init_status = dnnl_unimplemented;
    }
    return init_status;
}

dnnl_status_t init_rnn_bwd_desc(dnnl_rnn_desc_t *rd, const prb_t &p,
        dnnl_prop_kind_t prop_kind, dnnl_memory_desc_t *src_layer_d,
        dnnl_memory_desc_t *src_iter_d, dnnl_memory_desc_t *src_iter_c_d,
        dnnl_memory_desc_t *weights_layer_d, dnnl_memory_desc_t *weights_iter_d,
        dnnl_memory_desc_t *weights_peephole_d, dnnl_memory_desc_t *bias_d,
        dnnl_memory_desc_t *dst_layer_d, dnnl_memory_desc_t *dst_iter_d,
        dnnl_memory_desc_t *dst_iter_c_d, dnnl_memory_desc_t *diff_src_layer_d,
        dnnl_memory_desc_t *diff_src_iter_d,
        dnnl_memory_desc_t *diff_src_iter_c_d,
        dnnl_memory_desc_t *diff_weights_layer_d,
        dnnl_memory_desc_t *diff_weights_iter_d,
        dnnl_memory_desc_t *diff_weights_peephole_d,
        dnnl_memory_desc_t *diff_bias_d, dnnl_memory_desc_t *diff_dst_layer_d,
        dnnl_memory_desc_t *diff_dst_iter_d,
        dnnl_memory_desc_t *diff_dst_iter_c_d) {
    dnnl_alg_kind_t kind = alg2kind(p.alg);
    dnnl_alg_kind_t f = activation2kind(p.activation);

    dnnl_status_t init_status;
    switch (kind) {
        case dnnl_vanilla_rnn:
            init_status = dnnl_vanilla_rnn_backward_desc_init(rd, prop_kind, f,
                    p.direction, src_layer_d, src_iter_d, weights_layer_d,
                    weights_iter_d, bias_d, dst_layer_d, dst_iter_d,
                    diff_src_layer_d, diff_src_iter_d, diff_weights_layer_d,
                    diff_weights_iter_d, diff_bias_d, diff_dst_layer_d,
                    diff_dst_iter_d, p.flags, p.alpha, p.beta);
            break;
        case dnnl_vanilla_lstm:
            init_status = dnnl_lstm_backward_desc_init_v2(rd, prop_kind,
                    p.direction, src_layer_d, src_iter_d, src_iter_c_d,
                    weights_layer_d, weights_iter_d, weights_peephole_d, bias_d,
                    dst_layer_d, dst_iter_d, dst_iter_c_d, diff_src_layer_d,
                    diff_src_iter_d, diff_src_iter_c_d, diff_weights_layer_d,
                    diff_weights_iter_d, diff_weights_peephole_d, diff_bias_d,
                    diff_dst_layer_d, diff_dst_iter_d, diff_dst_iter_c_d,
                    p.flags);
            break;
        case dnnl_vanilla_gru:
            init_status = dnnl_gru_backward_desc_init(rd, prop_kind,
                    p.direction, src_layer_d, src_iter_d, weights_layer_d,
                    weights_iter_d, bias_d, dst_layer_d, dst_iter_d,
                    diff_src_layer_d, diff_src_iter_d, diff_weights_layer_d,
                    diff_weights_iter_d, diff_bias_d, diff_dst_layer_d,
                    diff_dst_iter_d, p.flags);
            break;
        case dnnl_lbr_gru:
            init_status = dnnl_lbr_gru_backward_desc_init(rd, prop_kind,
                    p.direction, src_layer_d, src_iter_d, weights_layer_d,
                    weights_iter_d, bias_d, dst_layer_d, dst_iter_d,
                    diff_src_layer_d, diff_src_iter_d, diff_weights_layer_d,
                    diff_weights_iter_d, diff_bias_d, diff_dst_layer_d,
                    diff_dst_iter_d, p.flags);
            break;
        default: init_status = dnnl_unimplemented;
    }
    return init_status;
}

void init_buffer(float *buf, int64_t size, float value) {
    for (int64_t i = 0; i < size; i++)
        buf[i] = value;
}

float logistic(float x) {
    if (x < 0)
        return (expf(x) / (1 + expf(x)));
    else
        return 1.0f - (expf(-x) / (1 + expf(-x)));
}
float dlogistic(float x) {
    float tmp = logistic(x);
    return tmp * (1 - tmp);
}
float dtanhf(float x) {
    return (1 - tanhf(x)) * (1 + tanhf(x));
}
float x_m_square(float x) {
    return x - x * x;
}
float relu(float x, float alpha) {
    return x > 0 ? x : alpha * x;
}
float drelu(float x, float alpha) {
    return x > 0 ? 1.0f : alpha;
}
float one_m_square(float x) {
    return 1 - x * x;
}

int compare_dat(const prb_t &p, rnn_data_kind_t kind, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp, res_t *r, bool final_compare = false) {
    const auto nelems = mem_dt.nelems();

    const char *skind = rnn_data_kind2str(kind);

    diff_norm_t diff_norm;
    size_t errors = 0;
    r->total += nelems;

//#define BENCHDNN_RNN_PRINT_STATISTICS
#ifdef BENCHDNN_RNN_PRINT_STATISTICS
    double min_dt, max_dt, mean_dt = 0.0f, var_dt = 0.0f;
    double min_fp, max_fp, mean_fp = 0.0f, var_fp = 0.0f;
    min_dt = max_dt = mem_dt.get_elem(0);
    min_fp = max_fp = mem_fp.get_elem(0);
#endif

    int64_t fwd_acc_dim
            = 2 * p.n_gates() + 1; // factor 2 is because of the sum of 2 GEMMs
    if (p.alg == VANILLA_GRU) fwd_acc_dim *= p.sic;
    int64_t bwdd_acc_dim = p.n_gates() * p.dic;
    int64_t bwdw_acc_dim = p.mb;
    int64_t acc_dim = fwd_acc_dim;
    if (p.prop == dnnl_backward) acc_dim *= MAX2(bwdd_acc_dim, bwdw_acc_dim);
    // Here the factor 4 just gives some wiggle room for fp32 testing
    float rel_eps = 4
            * (1 + (p.prop == dnnl_backward)) // double wiggle room for bwd
            * ((p.direction == dnnl_bidirectional_sum)
                    + 1) // double threshold if bidir_sum
            * ceilf(log2f(acc_dim * p.n_iter)) * p.cfg[kind].eps;
#ifdef BENCHDNN_RNN_PRINT_STATISTICS
    printf("rel_eps(%a) eps(%a) %ld\n", rel_eps, p.cfg[kind].eps, acc_dim);
#endif

    /* Note: we do an eltwise comparison only when:
       - we use skip_nonlinear;
       - we do not use skip_nonlinear and we test only one cell execution;
       - for int8 computations the tensor is not dst_c_last_iteration;
       If the above conditions are not met, we check only norm-1,
       norm-2 and inf-norm.

       Rough rationale for the `dst_c_last_iteration` exception in int8 case:
       - The formula for one-step c-state is:
         c_t = f_t * c_{t−1} + i_t * c~_t.
         Here all computations happen in f32 (f_t, i_t, and c~_t are dequantized
         right before the computations + the corresponding bias added).
       - In int8 case we don't have much control over these components and
         cannot surmount potential cancellations, if any.
         In practice, I observed that the relative element-wise error of values
         in `dst_c_last_iteration` was bigger (up-to 8e-5) whenever the values
         themselves were smaller (which indirectly means the problem is exactly
         in the cancellation). Unfortunately, this even happened with only one
         layer and one time stamp.
       - So, for now the solution is to use l1- l2- and l_inf-norms to validate
         `dst_c_last_iteration`. When we switch testing on using precise
         integer arithmetic based on modulo operation in rnn_tparams (instead of
         current unreliable re-scaling), this testing weakness should go away.
       - Just an obvious side note: `dst_last_layer` and `dst_last_iteration`
         are immediate dequantization of the corresponding u8 tensors. Hence,
         as long as we get precise u8 intermediate results (and so far we do),
         the f32 result should be pretty accurate -- the dequantization is just
         two simple ops: f32 = scale * u8 + shift.
    */
    bool check_norm0
            = (p.skip_nonlinear || ((p.n_layer == 1) && (p.n_iter == 1)));
    if (p.is_int8() && kind == dst_c_last_iteration) check_norm0 = false;

    for (int64_t i = 0; i < nelems; ++i) {
        const float dt = mem_dt.get_elem(i);
        const float fp = mem_fp.get_elem(i);
#ifdef BENCHDNN_RNN_PRINT_STATISTICS
        min_dt = MIN2(dt, min_dt);
        min_fp = MIN2(dt, min_fp);
        max_dt = MAX2(dt, max_dt);
        max_fp = MAX2(dt, max_fp);
        mean_dt += dt;
        mean_fp += fp;
        if (i > 0) {
            double tmp_dt = (double(i + 1) * dt - mean_dt);
            var_dt += (tmp_dt * tmp_dt) / (i * (i + 1));
            double tmp_fp = (double(i + 1) * fp - mean_fp);
            var_fp += (tmp_fp * tmp_fp) / (i * (i + 1));
        }
#endif
        diff_norm.update(fp, dt);

        bool ok = true;
        if (check_norm0) {
            const float diff = fabsf(fp - dt);
            const float rel_diff = diff / (fabsf(fp) > FLT_MIN ? fabsf(fp) : 1);
            const float diff_threshold = p.cfg[kind].eps;

            // very strict error bound for int8 data type
            if (p.cfg[kind].dt == dnnl_u8)
                ok = diff == 0;
            else
                ok = (fabs(fp) > diff_threshold ? rel_diff : diff) <= rel_eps;

            if (!ok) {
                errors++;
                if (errors < 10 || verbose >= 10) {
                    int64_t n = 0, t = 0, c = 0, l = 0, d = 0, w = 0, ic = 0,
                            oc = 0, b = 0;
                    switch (kind) {
                        case input:
                        case dst_diff_input:
                            inv_ntc_off_f(p, i, n, t, c);
                            print(0,
                                    "%4ld, %s, [%s][" IFMT "," IFMT "," IFMT
                                    "] "
                                    "fp:%8g dt:%8g diff:%8g rdiff:%8g\n",
                                    (long)i, final_compare ? "" : "REORDER ",
                                    skind, n, t, c, fp, dt, diff, rel_diff);
                            break;
                        case states:
                        case dst_diff_states:
                            inv_ldnc_off_f(p, i, l, d, n, c);
                            print(0,
                                    "%4ld, %s, [%s][" IFMT "," IFMT "," IFMT
                                    "," IFMT
                                    "] "
                                    "fp:%8g dt:%8g diff:%8g rdiff:%8g\n",
                                    (long)i, final_compare ? "" : "REORDER ",
                                    skind, l, d, n, c, fp, dt, diff, rel_diff);
                            break;
                        case weights_input:
                        case dst_diff_weights_input:
                            inv_ldigo_off_f(p, i, l, d, w, ic, oc);
                            print(0,
                                    "%4ld, %s, [%s][" IFMT "," IFMT "," IFMT
                                    "," IFMT "," IFMT
                                    "] "
                                    "fp:%8g dt:%8g diff:%8g rdiff:%8g\n",
                                    (long)i, final_compare ? "" : "REORDER ",
                                    skind, l, d, w, ic, oc, fp, dt, diff,
                                    rel_diff);
                            break;
                        case weights_states:
                        case dst_diff_weights_states:
                            inv_ldigo_off_f(p, i, l, d, w, ic, oc);
                            print(0,
                                    "%4ld, %s, [%s][" IFMT "," IFMT "," IFMT
                                    "," IFMT "," IFMT
                                    "] "
                                    "fp:%8g dt:%8g diff:%8g rdiff:%8g\n",
                                    (long)i, final_compare ? "" : "REORDER ",
                                    skind, l, d, w, ic, oc, fp, dt, diff,
                                    rel_diff);
                            break;
                        case weights_peephole:
                        case dst_diff_weights_peephole:
                            inv_weights_peephole_ldgo_off_f(p, i, l, d, b, c);
                            print(0,
                                    "%4ld, %s, [%s][" IFMT "," IFMT "," IFMT
                                    "," IFMT
                                    "] "
                                    "fp:%8g dt:%8g diff:%8g rdiff:%8g\n",
                                    (long)i, final_compare ? "" : "REORDER ",
                                    skind, l, d, b, c, fp, dt, diff, rel_diff);
                            break;
                        case bias:
                        case dst_diff_bias:
                            inv_bias_ldgo_off_f(p, i, l, d, b, c);
                            print(0,
                                    "%4ld, %s, [%s][" IFMT "," IFMT "," IFMT
                                    "," IFMT
                                    "] "
                                    "fp:%8g dt:%8g diff:%8g rdiff:%8g\n",
                                    (long)i, final_compare ? "" : "REORDER ",
                                    skind, l, d, b, c, fp, dt, diff, rel_diff);
                            break;
                        case dst_last_layer:
                        case diff_last_layer:
                            inv_tnc_off_f(p, i, t, n, c);
                            print(0,
                                    "%4ld, %s, [%s][" IFMT "," IFMT "," IFMT
                                    "] "
                                    "fp:%8g dt:%8g diff:%8g rdiff:%8g\n",
                                    (long)i, final_compare ? "" : "REORDER ",
                                    skind, t, n, c, fp, dt, diff, rel_diff);
                            break;
                        case c_states:
                        case dst_last_iteration:
                        case dst_c_last_iteration:
                        case dst_diff_c_states:
                        case diff_last_iteration:
                        case diff_c_last_iteration:
                            inv_ldnc_off_f(p, i, l, d, n, c);
                            print(0,
                                    "%4ld, %s, [%s][" IFMT "," IFMT "," IFMT
                                    "," IFMT
                                    "] "
                                    "fp:%8g dt:%8g diff:%8g rdiff:%8g\n",
                                    (long)i, final_compare ? "" : "REORDER ",
                                    skind, l, d, n, c, fp, dt, diff, rel_diff);
                            break;
                        default: assert(!"unknown data kind"); return FAIL;
                    }
                }
            }
        }
#if 1
        /* for debug purposes only: dump the output */
        if (final_compare && verbose >= 50) {
            int64_t n = 0, t = 0, c = 0, l = 0, d = 0, w = 0, ic = 0, oc = 0,
                    b = 0;

            switch (kind) {
                case input:
                    inv_ntc_off_f(p, i, n, t, c);
                    print(0,
                            "[%4ld][%s][" IFMT "," IFMT "," IFMT
                            "] fp:%8g dt:%8g\n",
                            (long)i, skind, n, t, c, fp, dt);
                    break;
                case states:
                    inv_ldnc_off_f(p, i, l, d, n, c);
                    print(0,
                            "[%4ld][%s][" IFMT "," IFMT "," IFMT "," IFMT
                            "] fp:%8g dt:%8g\n",
                            (long)i, skind, l, d, n, c, fp, dt);
                    break;
                case weights_input:
                    inv_ldigo_off_f(p, i, l, d, w, ic, oc);
                    print(0,
                            "[%4ld][%s][" IFMT "," IFMT "," IFMT "," IFMT
                            "," IFMT "] fp:%8g dt:%8g\n",
                            (long)i, skind, l, d, w, ic, oc, fp, dt);
                    break;
                case weights_states:
                    inv_ldigo_off_f(p, i, l, d, w, ic, oc);
                    print(0,
                            "[%4ld][%s][" IFMT "," IFMT "," IFMT "," IFMT
                            "," IFMT "] fp:%8g dt:%8g\n",
                            (long)i, skind, l, d, w, ic, oc, fp, dt);
                    break;
                case weights_peephole:
                    inv_weights_peephole_ldgo_off_f(p, i, l, d, b, c);
                    print(0,
                            "[%4ld][%s][" IFMT "," IFMT "," IFMT "," IFMT
                            "] fp:%8g dt:%8g\n",
                            (long)i, skind, l, d, b, c, fp, dt);
                    break;
                case bias:
                    inv_bias_ldgo_off_f(p, i, l, d, b, c);
                    print(0,
                            "[%4ld][%s][" IFMT "," IFMT "," IFMT "," IFMT
                            "] fp:%8g dt:%8g\n",
                            (long)i, skind, l, d, b, c, fp, dt);
                    break;
                case dst_last_layer:
                    inv_tnc_off_f(p, i, t, n, c);
                    print(0,
                            "[%4ld][%s][" IFMT "," IFMT "," IFMT
                            "] fp:%8g dt:%8g\n",
                            (long)i, skind, n, t, c, fp, dt);
                    break;
                case c_states:
                case dst_last_iteration:
                case dst_c_last_iteration:
                    inv_ldnc_off_f(p, i, l, d, n, c);
                    print(0,
                            "[%4ld][%s][" IFMT "," IFMT "," IFMT "," IFMT
                            "] fp:%8g dt:%8g\n",
                            (long)i, skind, l, d, n, c, fp, dt);
                    break;
                default:
                    print(0, "[%4ld][unknown] fp:%8g dt:%8g\n", (long)i, fp,
                            dt);
                    break;
            }
        }
#endif
    }

    diff_norm.done();

    if (!check_norm0) {
        if ((diff_norm.rel_diff(norm_t::L1) > rel_eps)
                || (diff_norm.rel_diff(norm_t::L2) > rel_eps)
                || (diff_norm.rel_diff(norm_t::L8) > rel_eps))
            errors++;
    }

    if (final_compare || errors) {
        const int vl = errors ? 0 : 2;
        print(vl,
                "@@@ [%s] %sdiff: l0(``%g``) "
                "l1:(%g,%g,%g,``%g``) "
                "l2:(%g,%g,%g,``%g``) "
                "l8:(%g,%g,%g,``%g``)\n",
                skind, final_compare ? "final: " : "",
                diff_norm.rel_diff(norm_t::L0), diff_norm.a_[norm_t::L1],
                diff_norm.b_[norm_t::L1], diff_norm.diff_[norm_t::L1],
                diff_norm.rel_diff(norm_t::L1), diff_norm.a_[norm_t::L2],
                diff_norm.b_[norm_t::L2], diff_norm.diff_[norm_t::L2],
                diff_norm.rel_diff(norm_t::L2), diff_norm.a_[norm_t::L8],
                diff_norm.b_[norm_t::L8], diff_norm.diff_[norm_t::L8],
                diff_norm.rel_diff(norm_t::L8));
    }

    r->errors += errors;
    if (errors != 0) r->state = FAILED;

    if (final_compare && r->state == UNTESTED) r->state = PASSED; /* optimism */

#ifdef BENCHDNN_RNN_PRINT_STATISTICS
    printf("dt: min(%a) max(%a) mean(%a), var(%a)\n", min_dt, max_dt,
            mean_dt / nelems, var_dt / nelems);
    printf("fp: min(%a) max(%a) mean(%a), var(%a)\n", min_fp, max_fp,
            mean_fp / nelems, var_fp / nelems);
#endif

    return errors != 0 ? FAIL : OK;
}

int compare_input(const prb_t &p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp,
        res_t *r, bool final_compare = false) {
    return compare_dat(p, input, mem_dt, mem_fp, r, final_compare);
}
int compare_states(const prb_t &p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp,
        res_t *r, bool final_compare = false) {
    return compare_dat(p, states, mem_dt, mem_fp, r, final_compare);
}
int compare_weights_input(const prb_t &p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp,
        res_t *r, bool final_compare = false) {
    return compare_dat(p, weights_input, mem_dt, mem_fp, r, final_compare);
}
int compare_weights_states(const prb_t &p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp,
        res_t *r, bool final_compare = false) {
    return compare_dat(p, weights_states, mem_dt, mem_fp, r, final_compare);
}
int compare_bias(const prb_t &p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *r,
        bool final_compare = false) {
    return compare_dat(p, bias, mem_dt, mem_fp, r, final_compare);
}
int compare_dst_last_layer(const prb_t &p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp,
        res_t *r, bool final_compare = false) {
    return compare_dat(p, dst_last_layer, mem_dt, mem_fp, r, final_compare);
}
int compare_dst_last_iteration(const prb_t &p, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp, res_t *r, bool final_compare = false) {
    return compare_dat(p, dst_last_iteration, mem_dt, mem_fp, r, final_compare);
}

int compare_dst_c_last_iteration(const prb_t &p, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp, res_t *r, bool final_compare = false) {
    return compare_dat(
            p, dst_c_last_iteration, mem_dt, mem_fp, r, final_compare);
}

void prb_t::set_qparams(float fp_min, float fp_max) {
    if (!is_cfg_u8(cfg)) {
        data_shift = 0.;
        data_scale = 1.;
        wei_scale = 1.;
        return;
    }

    /* Set parameters for quantization of src and weights from fp32 data
     * in [-1, 1] to int8 data in a range specified in cfg */
    float fp_range = fp_max - fp_min;
    float int8_src_range = cfg[input].f_max - cfg[input].f_min,
          int8_wei_range = cfg[weights_input].f_max - cfg[weights_input].f_min;

    data_shift = cfg[input].f_mean;
    data_scale = int8_src_range / fp_range;

    if (scale_policy == policy_t::COMMON) {
        wei_scale = int8_wei_range / fp_range;
    } else if (scale_policy == policy_t::PER_OC) {
        float K = int8_wei_range / fp_range;
        const auto nelems = dic * n_gates();
        for (int64_t i = 0; i < nelems; i++) {
            wei_oc_scales[i] = K * (1. + (float)i / nelems);
        }
    }
}

void prb_t::set_tparams(float fp_min, float fp_max) {
    if (skip_nonlinear) {
        assert(linear_scales != nullptr);
        // Here, we assume that the inputs of the cells are in [fp_min,fp_max]. We
        // pick the scaling factors to ensure that the output of the
        // linear pre/post gemm is in [fp_min,fp_max]

        // Also, we rely on the fact that for forward, the weights
        // matrices are sparse, and contain coefficient equal to
        // 1/n_gates() to compensate for the gemm accumulation. So
        // here, we account only for the postgemm accumulation, and
        // the fact that we want to use diferent scales per gate.

        // For BWD_W, we cannot assume sparssness though since the
        // gates and diff_dst_* are dense.
        int64_t fwd_acc_dim = n_gates();
        int64_t bwdd_acc_dim = dic;
        int64_t bwdw_acc_dim = mb;
        int64_t acc_dim = 0;
        if (prop == dnnl_backward)
            acc_dim = n_gates()
                    * MAX2(fwd_acc_dim, MAX2(bwdd_acc_dim, bwdw_acc_dim));
        else
            acc_dim = fwd_acc_dim;
        // make scaling exact by choosing powers of two.
        int64_t n_cscale = (alg == VANILLA_LSTM);
        int64_t divisor = next_pow2((acc_dim + n_cscale) * (is_int8() ? 2 : 1));
        float factor = (1.0f / (float)(divisor));
        for (int64_t i = 0; i < n_gates(); i++)
            linear_scales[i] = (i + 1) * factor;
        if (n_cscale) linear_cscale = (n_gates() + 1) * factor;
    }
}

} // namespace rnn
