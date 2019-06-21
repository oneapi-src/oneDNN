/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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

#include <assert.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <math.h>

#include <sstream>

#include "mkldnn.h"

#include "common.hpp"
#include "dnn_types.hpp"
#include "mkldnn_common.hpp"
#include "mkldnn_debug.hpp"
#include "src/common/math_utils.hpp"

// returns dims with current @p off values using actual values from @p dims
dims_t off2dims_idx(const dims_t &dims, int64_t off) {
    dims_t dims_idx;
    dims_idx.reserve(dims.size());

    for (int i = (int)dims.size() - 1; i >= 0; --i) {
        dims_idx.insert(dims_idx.begin(), off % dims[i]);
        off /= dims[i];
    }
    assert(off == 0);
    return dims_idx;
}

std::ostream &operator<<(std::ostream &s, const dims_t &dims) {
    s << dims[0];
    for (size_t d = 1; d < dims.size(); ++d)
        s << "x" << dims[d];
    return s;
}

std::ostream &operator<<(std::ostream &s,
        const std::vector<mkldnn_data_type_t> &v_dt) {
    s << dt2str(v_dt[0]);
    for (size_t d = 1; d < v_dt.size(); ++d)
        s << ":" << dt2str(v_dt[d]);
    return s;
}

std::ostream &operator<<(std::ostream &s,
        const std::vector<mkldnn_format_tag_t> &v_tag) {
    s << fmt_tag2str(v_tag[0]);
    for (size_t d = 1; d < v_tag.size(); ++d)
        s << ":" << fmt_tag2str(v_tag[d]);
    return s;
}

dir_t str2dir(const char *str) {
#define CASE(x) if (!strcasecmp(STRINGIFY(x), str)) return x
    CASE(FWD_D);
    CASE(FWD_I);
    CASE(FWD_B);
    CASE(BWD_D);
    CASE(BWD_W);
    CASE(BWD_WB);
    CASE(BWD_DW);
#undef CASE
    assert(!"unknown dir");
    return DIR_UNDEF;
}

const char *dir2str(dir_t dir) {
#define CASE(x) if (dir == x) return STRINGIFY(x)
    CASE(FWD_D);
    CASE(FWD_I);
    CASE(FWD_B);
    CASE(BWD_D);
    CASE(BWD_DW);
    CASE(BWD_W);
    CASE(BWD_WB);
#undef CASE
    assert(!"unknown dir");
    return "DIR_UNDEF";
}

mkldnn_prop_kind_t prop2prop_kind(const dir_t dir) {
    if (dir == FWD_D)
        return mkldnn_forward;
    if (dir == BWD_DW)
        return mkldnn_backward;
    assert(!"unknown dir");
    return mkldnn_prop_kind_undef;
}

const char *prop2str(mkldnn_prop_kind_t prop) {
    if (prop == mkldnn_forward)
        return "FWD_D";
    if (prop == mkldnn_backward)
        return "BWD_DW";
    assert(!"unknown prop_kind");
    return "unknown prop_kind";
}

const char *data_kind2str(data_kind_t kind) {
    switch (kind) {
    case SRC: return "SRC";
    case WEI: return "WEI";
    case BIA: return "BIA";
    case DST: return "DST";
    case ACC: return "ACC";
    case DATA: return "DATA";
    case MEAN: return "MEAN";
    case VAR: return "VAR";
    case SS: return "SS";
    case GWEI: return "GWEI";
    }
    assert(!"incorrect data kind");
    return "incorrect data kind";
}

attr_t::scale_t::policy_t attr_t::scale_t::str2policy(const char *str) {
#define CASE(_plc) if (!strcasecmp(STRINGIFY(_plc), str)) return _plc
    CASE(NONE);
    CASE(COMMON);
    CASE(PER_OC);
    CASE(PER_DIM_0);
    CASE(PER_DIM_1);
    CASE(PER_DIM_01);
#undef CASE
    assert(!"unknown attr::scale::policy");
    return NONE;
}

const char *attr_t::scale_t::policy2str(attr_t::scale_t::policy_t policy) {
    if (policy == NONE) return "none";
    if (policy == COMMON) return "common";
    if (policy == PER_OC) return "per_oc";
    if (policy == PER_DIM_0) return "per_dim_0";
    if (policy == PER_DIM_1) return "per_dim_1";
    if (policy == PER_DIM_01) return "per_dim_01";
    assert(!"unknown attr::scale::policy");
    return "unknown attr::scale::policy";
}

int attr_t::scale_t::str2scale(const char *str, const char **end_s) {
    *this = attr_t::scale_t();

    if (str == NULL) return FAIL;

    const char *s_;
    const char * &s = end_s ? *end_s : s_;
    s = str;

    for (policy_t p = NONE; true; p = (policy_t)((int)p + 1)) {
        if (p == POLICY_TOTAL) return FAIL;

        const char *ps = policy2str(p);
        if (!strncasecmp(ps, s, strlen(ps))) {
            this->policy = p;
            s += strlen(ps);
            break;
        }
    }

    if (*s != ':') return OK;
    s++;

    char *end;
    this->scale = strtof(s, &end);
    if (this->scale < 0 || end == s) return FAIL;

    s = end;
    assert(*s == '\0' || *s == ';');

    return OK;
}

attr_t::post_ops_t::kind_t attr_t::post_ops_t::str2kind(const char *str) {
#define CASE(_knd) if (!strcasecmp(STRINGIFY(_knd), str)) return _knd
    CASE(SUM);
    CASE(RELU);
    CASE(TANH);
    CASE(ELU);
    CASE(SQUARE);
    CASE(ABS);
    CASE(SQRT);
    CASE(LINEAR);
    CASE(BRELU);
    CASE(SRELU);
    CASE(LOGISTIC);
    CASE(EXP);
    CASE(GELU);
#undef CASE
    assert(!"unknown attr::post_ops::kind");
    return KIND_TOTAL;
}

const char *attr_t::post_ops_t::kind2str(attr_t::post_ops_t::kind_t kind) {
#define CASE(_knd, str) if (kind == _knd) return str
    CASE(SUM, "sum");
    CASE(RELU, "relu");
    CASE(TANH, "tanh");
    CASE(ELU, "elu");
    CASE(SQUARE, "square");
    CASE(ABS, "abs");
    CASE(SQRT, "sqrt");
    CASE(LINEAR, "linear");
    CASE(BRELU, "brelu");
    CASE(SRELU, "srelu");
    CASE(LOGISTIC, "logistic");
    CASE(EXP, "exp");
    CASE(GELU, "gelu");
#undef CASE
    assert(!"unknown attr::post_ops::kind");
    return "unknown attr::post_ops::kind";
}

mkldnn_alg_kind_t attr_t::post_ops_t::kind2mkldnn_kind(
        attr_t::post_ops_t::kind_t kind) {
#define CASE(_knd, _mknd) if (kind == _knd) return _mknd
    CASE(RELU, mkldnn_eltwise_relu);
    CASE(TANH, mkldnn_eltwise_tanh);
    CASE(ELU, mkldnn_eltwise_elu);
    CASE(SQUARE, mkldnn_eltwise_square);
    CASE(ABS, mkldnn_eltwise_abs);
    CASE(SQRT, mkldnn_eltwise_sqrt);
    CASE(LINEAR, mkldnn_eltwise_linear);
    CASE(BRELU, mkldnn_eltwise_bounded_relu);
    CASE(SRELU, mkldnn_eltwise_soft_relu);
    CASE(LOGISTIC, mkldnn_eltwise_logistic);
    CASE(EXP, mkldnn_eltwise_exp);
    CASE(GELU, mkldnn_eltwise_gelu);
#undef CASE
    assert(!"unknown attr::post_ops::kind");
    return mkldnn_alg_kind_undef;
}

int attr_t::post_ops_t::from_str(const char *str, const char **end_s) {
    *this = post_ops_t();

    if (str == NULL || *str != '\'') return FAIL;

    const char *s_;
    const char * &s = end_s ? *end_s : s_;
    s = str;

    ++s;
    for (;;) {
        if (*s == '\'') { ++s; return OK; }
        if (len == capacity) return FAIL;

        for (kind_t k = SUM; true; k = (kind_t)((int)k + 1)) {
            if (k == KIND_TOTAL) return FAIL;

            const char *ks = kind2str(k);
            if (!strncasecmp(ks, s, strlen(ks))) {
                auto &e = entry[len];

                e.kind = k;
                s += strlen(ks);
                if (k == SUM) {
                    if (*s == ':') {
                        char *end;
                        e.sum.scale = strtof(++s, &end);
                        if (e.sum.scale <= 0 || end == s) return FAIL;
                        s = end;
                    } else {
                        e.sum.scale = 1.f;
                    }
                } else {
                    e.eltwise.alg = kind2mkldnn_kind(k);
                    e.eltwise.scale = 1.f;
                    e.eltwise.alpha = e.eltwise.beta = 0.f;

                    for (int i = 0; i < 3; ++i) {
                        // :alpha:beta:scale
                        float &val = i == 0 ? e.eltwise.alpha
                            : i == 1 ? e.eltwise.beta : e.eltwise.scale;
                        if (*s == ':') {
                            char *end;
                            val = strtof(++s, &end);
                            if (end == s) return FAIL;
                            s = end;
                        } else {
                            break;
                        }
                    }

                    if (e.eltwise.scale <= 0) return FAIL;
                }

                break;
            }
        }
        ++len;

        if (*s == ';') ++s;
    }

    return FAIL; /* unreachable */
}

bool attr_t::is_def() const {
    return true
        && oscale.is_def()
        && post_ops.is_def();
}

int str2attr(attr_t *attr, const char *str) {
    if (attr == NULL || str == NULL) return FAIL;
    *attr = attr_t();

    const char *s = str;

    while (*s != '\0') {
        int rc = FAIL;
        const char *param;

        param = "oscale=";
        if (!strncasecmp(param, s, strlen(param))) {
            s += strlen(param);
            rc = attr->oscale.str2scale(s, &s);
            if (rc != OK) return rc;
        }

        param = "post_ops=";
        if (!strncasecmp(param, s, strlen(param))) {
            s += strlen(param);
            rc = attr->post_ops.from_str(s, &s);
            if (rc != OK) return rc;
        }

        if (rc != OK) return FAIL;
        if (*s == ';') ++s;
    }

    return OK;
}

std::ostream &operator<<(std::ostream &s, const attr_t::scale_t &scale) {
    return s << attr_t::scale_t::policy2str(scale.policy) << ":" << scale.scale;
}

std::ostream &operator<<(std::ostream &s, const attr_t::post_ops_t &post_ops) {
    auto kind2str = &attr_t::post_ops_t::kind2str;

    s << "'";

    for (int idx = 0; idx < post_ops.len; ++idx) {
        if (idx > 0) s << ";";
        const auto &e = post_ops.entry[idx];

        using pk = attr_t::post_ops_t::kind_t;
        switch (e.kind) {
        case pk::SUM:
            s << kind2str(e.kind) << ":" << e.sum.scale;
            break;
        case pk::RELU:
        case pk::TANH:
        case pk::ELU:
        case pk::SQUARE:
        case pk::ABS:
        case pk::SQRT:
        case pk::LINEAR:
        case pk::BRELU:
        case pk::SRELU:
        case pk::LOGISTIC:
        case pk::EXP:
        case pk::GELU:
            s << kind2str(e.kind) << ":" << e.eltwise.alpha;
            if (e.eltwise.beta != 0.f || e.eltwise.scale != 1.f)
                s << ":" << e.eltwise.beta << ":" << e.eltwise.scale;
            break;
        default:
            assert(!"unknown kind");
            s << "unknown_kind";
        }
    }

    s << "'";

    return s;
}

std::ostream &operator<<(std::ostream &s, const attr_t &attr) {
    if (!attr.oscale.is_def())
        s << "oscale=" << attr.oscale << ";";
    if (!attr.post_ops.is_def())
        s << "post_ops=" << attr.post_ops << ";";
    return s;
}

std::ostream &dump_global_params(std::ostream &s) {
    if (engine_tgt_kind != mkldnn_cpu)
        s << "--engine=" << engine_kind2str(engine_tgt_kind) << " ";
    return s;
}

mkldnn_engine_kind_t str2engine_kind(const char *str) {
    const char *param = "cpu";
    if (!strncasecmp(param, str, strlen(param)))
        return mkldnn_cpu;

    param = "gpu";
    if (!strncasecmp(param, str, strlen(param)))
        return mkldnn_gpu;

    assert(!"not expected");
    return mkldnn_cpu;
}

const char *engine_kind2str(mkldnn_engine_kind_t engine) {
    switch (engine) {
    case mkldnn_any_engine: return "any";
    case mkldnn_cpu: return "cpu";
    case mkldnn_gpu: return "gpu";
    }
    assert(!"incorrect engine kind");
    return "incorrect engine kind";
}

mkldnn_primitive_attr_t create_mkldnn_attr(const attr_t &attr,
        int64_t scale_cnt, int scale_mask, const float *scales) {
    mkldnn_primitive_attr_t mkldnn_attr = NULL;
    DNN_SAFE_V(mkldnn_primitive_attr_create(&mkldnn_attr));

    if (!attr.oscale.is_def()) {
        using P = attr_t::scale_t::policy_t;
        int64_t count = attr.oscale.policy == P::COMMON ? 1 : scale_cnt;
        if (scale_mask == -1)
            scale_mask = attr.oscale.policy == P::PER_OC ? 1 << 1 : 0;

        float *gen_scs = NULL;
        if (scales == NULL) {
            gen_scs = (float *)zmalloc(count * sizeof(float), 64);
            SAFE_V(gen_scs != NULL ? OK : FAIL);
            for (int64_t i = 0; i < count; ++i)
                gen_scs[i] = attr.oscale.scale;
            scales = gen_scs;
        }

        DNN_SAFE_V(mkldnn_primitive_attr_set_output_scales(mkldnn_attr, count,
                    scale_mask, scales));

        if (gen_scs)
            zfree(gen_scs);
    }

    if (!attr.post_ops.is_def()) {
        mkldnn_post_ops_t ops;
        DNN_SAFE_V(mkldnn_post_ops_create(&ops));
        for (int idx = 0; idx < attr.post_ops.len; ++idx) {
            const auto &e = attr.post_ops.entry[idx];
            switch (attr.post_ops.entry[idx].kind) {
            case attr_t::post_ops_t::SUM:
                DNN_SAFE_V(mkldnn_post_ops_append_sum(ops, e.sum.scale));
                break;
            case attr_t::post_ops_t::RELU:
            case attr_t::post_ops_t::TANH:
            case attr_t::post_ops_t::ELU:
            case attr_t::post_ops_t::SQUARE:
            case attr_t::post_ops_t::ABS:
            case attr_t::post_ops_t::SQRT:
            case attr_t::post_ops_t::LINEAR:
            case attr_t::post_ops_t::BRELU:
            case attr_t::post_ops_t::SRELU:
            case attr_t::post_ops_t::LOGISTIC:
            case attr_t::post_ops_t::EXP:
            case attr_t::post_ops_t::GELU:
                DNN_SAFE_V(mkldnn_post_ops_append_eltwise(ops, e.eltwise.scale,
                            e.eltwise.alg, e.eltwise.alpha, e.eltwise.beta));
                break;
            default:
                assert(!"unknown attr::post_ops::kind");
            }
        }
        DNN_SAFE_V(mkldnn_primitive_attr_set_post_ops(mkldnn_attr, ops));

        const_mkldnn_post_ops_t c_ops;
        DNN_SAFE_V(mkldnn_primitive_attr_get_post_ops(mkldnn_attr, &c_ops));
        SAFE_V(mkldnn_post_ops_len(c_ops) == attr.post_ops.len ? OK : FAIL);

        DNN_SAFE_V(mkldnn_post_ops_destroy(ops));
    }

    return mkldnn_attr;
}

mkldnn_format_tag_t get_default_tag(int ndims) {
    switch (ndims) {
    case 1: return mkldnn_a;
    case 2: return mkldnn_ab;
    case 3: return mkldnn_abc;
    case 4: return mkldnn_abcd;
    case 5: return mkldnn_abcde;
    case 6: return mkldnn_abcdef;
    default: assert(!"unknown kind");
    }
    return mkldnn_format_tag_undef;
}

void maybe_scale(float &d, float *scales, int64_t oc, const attr_t &attr) {
    if (!attr.oscale.is_def()) {
        using policy_t = attr_t::scale_t::policy_t;
        const auto &s = attr.oscale;
        if (s.policy == policy_t::COMMON) {
            d *= s.scale;
        } else {
            d *= scales[oc];
        }
    }
}

float compute_eltwise_fwd(attr_t::post_ops_t::kind_t kind, float src,
        float scale, float alpha, float beta) {
    using namespace mkldnn::impl::math;
    using pk = attr_t::post_ops_t::kind_t;

    switch (kind) {
    case pk::RELU: return scale * relu_fwd(src, alpha);
    case pk::TANH: return scale * tanh_fwd(src);
    case pk::ELU: return scale * elu_fwd(src, alpha);
    case pk::SQUARE: return scale * square_fwd(src);
    case pk::ABS: return scale * abs_fwd(src);
    case pk::SQRT: return scale * sqrt_fwd(src);
    case pk::LINEAR: return scale * linear_fwd(src, alpha, beta);
    case pk::BRELU: return scale * bounded_relu_fwd(src, alpha);
    case pk::SRELU: return scale * soft_relu_fwd(src);
    case pk::LOGISTIC: return scale * logistic_fwd(src);
    case pk::EXP: return scale * exp_fwd(src);
    case pk::GELU: return scale * gelu_fwd(src);
    default: assert(!"unknown attr::post_ops::kind");
    };
    return NAN;
}

float compute_eltwise_bwd(attr_t::post_ops_t::kind_t kind, float d_dst,
        float src, float alpha, float beta) {
    using namespace mkldnn::impl::math;
    using pk = attr_t::post_ops_t::kind_t;

    switch (kind) {
    case pk::RELU: return relu_bwd(d_dst, src, alpha);
    case pk::TANH: return tanh_bwd(d_dst, src);
    case pk::ELU: return elu_bwd(d_dst, src, alpha);
    case pk::SQUARE: return square_bwd(d_dst, src);
    case pk::ABS: return abs_bwd(d_dst, src);
    case pk::SQRT: return sqrt_bwd(d_dst, src);
    case pk::LINEAR: return linear_bwd(d_dst, src, alpha, beta);
    case pk::BRELU: return bounded_relu_bwd(d_dst, src, alpha);
    case pk::SRELU: return soft_relu_bwd(d_dst, src);
    case pk::LOGISTIC: return logistic_bwd(d_dst, src);
    case pk::EXP: return exp_bwd(d_dst, src);
    case pk::GELU: return gelu_bwd(d_dst, src);
    default: assert(!"unknown attr::post_ops::kind");
    }
    return NAN;
}

void maybe_post_ops(float &d, float dst, const attr_t &attr) {
    using namespace mkldnn::impl::math;

    const auto &ops = attr.post_ops;
    for (int idx = 0; idx < ops.len; ++idx) {
        using pk = attr_t::post_ops_t::kind_t;
        const auto &e = ops.entry[idx];

        const auto &s = e.eltwise.scale;
        const auto &a = e.eltwise.alpha;
        const auto &b = e.eltwise.beta;

        if (e.kind == pk::SUM)
            d += e.sum.scale * dst;
        else
            d = compute_eltwise_fwd(e.kind, d, s, a, b);
    }
}
