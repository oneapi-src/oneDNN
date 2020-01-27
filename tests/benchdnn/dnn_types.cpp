/*******************************************************************************
* Copyright 2017-2019 Intel Corporation
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
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include <sstream>

#include "dnnl.h"

#include "common.hpp"
#include "dnn_types.hpp"
#include "dnnl_common.hpp"
#include "dnnl_debug.hpp"
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

std::ostream &operator<<(std::ostream &s, const std::vector<dims_t> &sdims) {
    s << sdims[0];
    for (size_t d = 1; d < sdims.size(); ++d)
        s << ":" << sdims[d];
    return s;
}

std::ostream &operator<<(
        std::ostream &s, const std::vector<dnnl_data_type_t> &v_dt) {
    s << dt2str(v_dt[0]);
    for (size_t d = 1; d < v_dt.size(); ++d)
        s << ":" << dt2str(v_dt[d]);
    return s;
}

std::ostream &operator<<(
        std::ostream &s, const std::vector<dnnl_format_tag_t> &v_tag) {
    s << fmt_tag2str(v_tag[0]);
    for (size_t d = 1; d < v_tag.size(); ++d)
        s << ":" << fmt_tag2str(v_tag[d]);
    return s;
}

dir_t str2dir(const char *str) {
#define CASE(x) \
    if (!strcasecmp(STRINGIFY(x), str)) return x
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
#define CASE(x) \
    if (dir == x) return STRINGIFY(x)
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

dnnl_prop_kind_t prop2prop_kind(const dir_t dir) {
    if (dir == FWD_D) return dnnl_forward;
    if (dir == BWD_DW) return dnnl_backward;
    assert(!"unknown dir");
    return dnnl_prop_kind_undef;
}

const char *prop2str(dnnl_prop_kind_t prop) {
    if (prop == dnnl_forward) return "FWD_D";
    if (prop == dnnl_backward) return "BWD_DW";
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

static const std::map<int, const char *> arg2str = {
        {DNNL_ARG_SRC, "src:"},
        {DNNL_ARG_SRC_1, "src1:"},
        {DNNL_ARG_WEIGHTS, "wei:"},
        {DNNL_ARG_DST, "dst:"},
};

attr_t::scale_t::policy_t attr_t::scale_t::str2policy(const char *str) {
#define CASE(_plc) \
    if (!strcasecmp(STRINGIFY(_plc), str)) return _plc
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
    const char *&s = end_s ? *end_s : s_;
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

    if (*s == '*') {
        ++s;
        if (this->policy != NONE) this->runtime = true;
    }

    assert(*s == '\0' || *s == ';');

    return OK;
}

int attr_t::zero_points_t::from_str(const char *str, const char **end_s) {
    *this = attr_t::zero_points_t();

    if (str == NULL) return FAIL;

    const char *s_;
    const char *&s = end_s ? *end_s : s_;
    s = str;

    while (isalpha(*s)) {
        for (const auto &arg : arg2str) {
            const size_t arg_name_len = strlen(arg.second);
            if (!strncasecmp(arg.second, s, arg_name_len)) {
                s += arg_name_len;
                char *end = NULL;
                int zero_point = (int)strtol(s, &end, 10);
                bool runtime = false;
                if (end == s) return FAIL;
                s = end;
                if (*s == '*') {
                    runtime = true;
                    ++s;
                }
                set(arg.first, {zero_point, runtime});
            }
        }

        while (*s == '_')
            ++s;
    }

    while (*s == '_')
        ++s;
    assert(*s == '\0' || *s == ';');

    return OK;
}

int attr_t::arg_scales_t::from_str(const char *str, const char **end_s) {
    *this = attr_t::arg_scales_t();

    if (str == NULL) return FAIL;

    const char *s_;
    const char *&s = end_s ? *end_s : s_;
    s = str;
    // skip '
    ++s;

    while (true) {
        if (*s == '\'') {
            ++s;
            return OK;
        }
        const char *s_init_pos = s;
        for (const auto &arg : arg2str) {
            const size_t arg_name_len = strlen(arg.second);
            if (!strncasecmp(arg.second, s, arg_name_len)) {
                s += arg_name_len;
                scale_t::policy_t policy;
                for (scale_t::policy_t p = scale_t::NONE; true;
                        p = (scale_t::policy_t)((int)p + 1)) {
                    if (p == scale_t::POLICY_TOTAL) return FAIL;

                    const char *ps = scale_t::policy2str(p);
                    if (!strncasecmp(ps, s, strlen(ps))) {
                        policy = p;
                        s += strlen(ps);
                        break;
                    }
                }
                // skip :
                s++;
                char *end;
                float scale = strtof(s, &end);
                if (scale < 0 || end == s) return FAIL;
                set(arg.first, policy, scale);
                s = end;
                break;
            }
        }
        if (s_init_pos == s) return FAIL;
        while (*s == '_' || isspace(*s))
            ++s;
    }
    assert(*s == '\0' || *s == ';');
    return OK;
}

using pk_t = attr_t::post_ops_t::kind_t;

typedef struct {
    pk_t kind;
    const char *kind_name;
    dnnl_alg_kind_t dnnl_kind;
} po_table_entry_t;

static po_table_entry_t kind_table[] = {{pk_t::SUM, "sum", dnnl_alg_kind_undef},
        {pk_t::RELU, "relu", dnnl_eltwise_relu},
        {pk_t::TANH, "tanh", dnnl_eltwise_tanh},
        {pk_t::ELU, "elu", dnnl_eltwise_elu},
        {pk_t::SQUARE, "square", dnnl_eltwise_square},
        {pk_t::ABS, "abs", dnnl_eltwise_abs},
        {pk_t::SQRT, "sqrt", dnnl_eltwise_sqrt},
        {pk_t::LINEAR, "linear", dnnl_eltwise_linear},
        {pk_t::BRELU, "brelu", dnnl_eltwise_bounded_relu},
        {pk_t::SRELU, "srelu", dnnl_eltwise_soft_relu},
        {pk_t::LOGISTIC, "logistic", dnnl_eltwise_logistic},
        {pk_t::EXP, "exp", dnnl_eltwise_exp},
        {pk_t::GELU, "gelu", dnnl_eltwise_gelu},
        {pk_t::SWISH, "swish", dnnl_eltwise_swish},
        {pk_t::LOG, "log", dnnl_eltwise_log},
        {pk_t::CLIP, "clip", dnnl_eltwise_clip},
        {pk_t::POW, "pow", dnnl_eltwise_pow},
        {pk_t::KIND_TOTAL, "unknown", dnnl_alg_kind_undef}};

pk_t attr_t::post_ops_t::str2kind(const char *str) {
    for (const auto &e : kind_table) {
        if (!strcasecmp(e.kind_name, str)) return e.kind;
    }
    assert(!"unknown attr::post_ops::kind");
    return kind_table[KIND_TOTAL].kind;
}

const char *attr_t::post_ops_t::kind2str(pk_t kind) {
    if (kind >= KIND_TOTAL) {
        assert(!"unknown attr::post_ops::kind");
        return kind_table[KIND_TOTAL].kind_name;
    }
    return kind_table[kind].kind_name;
}

dnnl_alg_kind_t attr_t::post_ops_t::kind2dnnl_kind(pk_t kind) {
    if (kind >= KIND_TOTAL) {
        assert(!"unknown attr::post_ops::kind");
        return kind_table[KIND_TOTAL].dnnl_kind;
    }
    return kind_table[kind].dnnl_kind;
}

int attr_t::post_ops_t::from_str(const char *str, const char **end_s) {
    *this = post_ops_t();

    if (str == NULL || *str != '\'') return FAIL;

    const char *s_;
    const char *&s = end_s ? *end_s : s_;
    s = str;

    ++s;
    for (;;) {
        if (*s == '\'') {
            ++s;
            return OK;
        }
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
                        if (end == s) return FAIL;
                        s = end;
                    } else {
                        e.sum.scale = 1.f;
                    }
                } else {
                    e.eltwise.alg = kind2dnnl_kind(k);
                    e.eltwise.scale = 1.f;
                    e.eltwise.alpha = e.eltwise.beta = 0.f;

                    for (int i = 0; i < 3; ++i) {
                        // :alpha:beta:scale
                        float &val = i == 0
                                ? e.eltwise.alpha
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

int attr_t::post_ops_t::find(pk_t kind, int start, int stop) const {
    if (stop == -1) stop = len;
    stop = MIN2(stop, len);
    for (int idx = start; idx < stop; ++idx)
        if (entry[idx].kind == kind) return idx;
    return -1;
}

bool attr_t::is_def() const {
    return oscale.is_def() && scales.is_def() && zero_points.is_def()
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

        param = "zero_points=";
        if (!strncasecmp(param, s, strlen(param))) {
            s += strlen(param);
            rc = attr->zero_points.from_str(s, &s);
            if (rc != OK) return rc;
        }

        param = "post_ops=";
        if (!strncasecmp(param, s, strlen(param))) {
            s += strlen(param);
            rc = attr->post_ops.from_str(s, &s);
            if (rc != OK) return rc;
        }

        param = "scales=";
        if (!strncasecmp(param, s, strlen(param))) {
            s += strlen(param);
            rc = attr->scales.from_str(s, &s);
            if (rc != OK) return rc;
        }

        if (rc != OK) return rc;
        if (*s == ';') ++s;
    }
    return OK;
}

std::ostream &operator<<(std::ostream &s, const attr_t::scale_t &scale) {
    s << attr_t::scale_t::policy2str(scale.policy) << ":" << scale.scale;
    if (scale.runtime) s << '*';
    return s;
}

std::ostream &operator<<(
        std::ostream &s, const attr_t::zero_points_t &zero_points) {
    bool first = true;
    for (const auto &point : zero_points.points) {
        if (!first) s << '_';
        first = false;

        s << arg2str.at(point.first) << point.second.value;
        if (point.second.runtime) s << '*';
    }

    return s;
}

std::ostream &operator<<(std::ostream &s, const attr_t::arg_scales_t &scales) {
    const char *delim = "";
    s << "'";
    for (const auto &v : scales.scales) {
        if (!v.second.is_def()) {
            s << delim << arg2str.at(v.first)
              << attr_t::scale_t::policy2str(v.second.policy) << ":"
              << v.second.scale;
            delim = "_";
        }
    }
    s << "'";
    return s;
}

std::ostream &operator<<(std::ostream &s, const attr_t::post_ops_t &post_ops) {
    auto kind2str = &attr_t::post_ops_t::kind2str;

    s << "'";

    for (int idx = 0; idx < post_ops.len; ++idx) {
        if (idx > 0) s << ";";
        const auto &e = post_ops.entry[idx];

        if (e.kind == pk_t::SUM) {
            s << kind2str(e.kind);
            if (e.sum.scale != 1.0f) s << ":" << e.sum.scale;
        } else if (e.kind < pk_t::KIND_TOTAL) {
            s << kind2str(e.kind);
            if (e.eltwise.scale != 1.f)
                s << ":" << e.eltwise.alpha << ":" << e.eltwise.beta << ":"
                  << e.eltwise.scale;
            else if (e.eltwise.beta != 0.f)
                s << ":" << e.eltwise.alpha << ":" << e.eltwise.beta;
            else if (e.eltwise.alpha != 0.f)
                s << ":" << e.eltwise.alpha;
        } else {
            assert(!"unknown kind");
            s << "unknown_kind";
        }
    }

    s << "'";

    return s;
}

std::ostream &operator<<(std::ostream &s, const attr_t &attr) {
    if (!attr.oscale.is_def()) s << "oscale=" << attr.oscale << ";";
    if (!attr.zero_points.is_def())
        s << "zero_points=" << attr.zero_points << ";";
    if (!attr.post_ops.is_def()) s << "post_ops=" << attr.post_ops << ";";
    if (!attr.scales.is_def()) s << "scales=" << attr.scales << ";";
    return s;
}

std::ostream &dump_global_params(std::ostream &s) {
    if (canonical || engine_tgt_kind != dnnl_cpu)
        s << "--engine=" << engine_kind2str(engine_tgt_kind) << " ";

    s << "--" << driver_name << " ";
    return s;
}

dnnl_engine_kind_t str2engine_kind(const char *str) {
    const char *param = "cpu";
    if (!strncasecmp(param, str, strlen(param))) return dnnl_cpu;

    param = "gpu";
    if (!strncasecmp(param, str, strlen(param))) return dnnl_gpu;

    assert(!"not expected");
    return dnnl_cpu;
}

const char *engine_kind2str(dnnl_engine_kind_t engine) {
    switch (engine) {
        case dnnl_any_engine: return "any";
        case dnnl_cpu: return "cpu";
        case dnnl_gpu: return "gpu";
    }
    assert(!"incorrect engine kind");
    return "incorrect engine kind";
}

void attr_bundle_t::init_zero_points() {
    for (const auto &arg_entry : attr.zero_points)
        zero_points[arg_entry.first] = {arg_entry.second.value};
}

int attr_bundle_t::generate(int scale_mask) {
    dnnl_primitive_attr_t dnnl_attr = create_dnnl_attr(
            attr, (int64_t)oscale.size(), scale_mask, oscale.data());
    if (dnnl_attr == NULL) return FAIL;

    scale_mask_ = scale_mask;
    dnnl_attr_.reset(dnnl_attr, &dnnl_primitive_attr_destroy);
    initialized_ = true;

    return OK;
}

dnnl_primitive_attr_t create_dnnl_attr(const attr_t &attr, int64_t scale_cnt,
        int scale_mask, const float *scales) {
    dnnl_primitive_attr_t dnnl_attr = NULL;
    DNN_SAFE_V(dnnl_primitive_attr_create(&dnnl_attr));

    using P = attr_t::scale_t::policy_t;

    if (!attr.oscale.is_def()) {
        int64_t count = attr.oscale.policy == P::COMMON ? 1 : scale_cnt;
        if (scale_mask == -1)
            scale_mask = attr.oscale.policy == P::PER_OC ? 1 << 1 : 0;

        const bool runtime = attr.oscale.runtime;
        SAFE_V(scales == NULL && runtime ? FAIL : OK);

        float *gen_scs = NULL;
        if (scales == NULL) {
            gen_scs = (float *)zmalloc(count * sizeof(float), 64);
            SAFE_V(gen_scs != NULL ? OK : FAIL);
            for (int64_t i = 0; i < count; ++i)
                gen_scs[i] = attr.oscale.scale;
            scales = gen_scs;
        }

        DNN_SAFE_V(dnnl_primitive_attr_set_output_scales(dnnl_attr, count,
                scale_mask, runtime ? &DNNL_RUNTIME_F32_VAL : scales));
        if (gen_scs) zfree(gen_scs);
    } else if (!attr.scales.is_def()) {
        // Only common policy is supported at this point
        for (const auto &s : attr.scales.scales) {
            int64_t count = -1;
            int mask = -1;
            count = s.second.policy == P::COMMON ? 1 : scale_cnt;
            if (scale_mask == -1)
                mask = s.second.policy == P::PER_OC ? 1 << 1 : 0;

            DNN_SAFE_V(dnnl_primitive_attr_set_scales(
                    dnnl_attr, s.first, count, mask, &s.second.scale));
        }
    }

    if (!attr.zero_points.is_def()) {
        for (const auto &zero_points : attr.zero_points) {
            DNN_SAFE_V(dnnl_primitive_attr_set_zero_points(dnnl_attr,
                    zero_points.first,
                    /* count */ 1, /* mask */ 0,
                    zero_points.second.runtime ? &DNNL_RUNTIME_S32_VAL
                                               : &zero_points.second.value));
        }
    }

    if (!attr.post_ops.is_def()) {
        dnnl_post_ops_t ops;
        DNN_SAFE_V(dnnl_post_ops_create(&ops));
        for (int idx = 0; idx < attr.post_ops.len; ++idx) {
            const auto &e = attr.post_ops.entry[idx];
            if (e.kind == pk_t::SUM) {
                DNN_SAFE_V(dnnl_post_ops_append_sum(ops, e.sum.scale));
            } else if (e.kind < pk_t::KIND_TOTAL) {
                DNN_SAFE_V(dnnl_post_ops_append_eltwise(ops, e.eltwise.scale,
                        e.eltwise.alg, e.eltwise.alpha, e.eltwise.beta));
            } else {
                assert(!"unknown attr::post_ops::kind");
            }
        }
        DNN_SAFE_V(dnnl_primitive_attr_set_post_ops(dnnl_attr, ops));

        const_dnnl_post_ops_t c_ops;
        DNN_SAFE_V(dnnl_primitive_attr_get_post_ops(dnnl_attr, &c_ops));
        SAFE_V(dnnl_post_ops_len(c_ops) == attr.post_ops.len ? OK : FAIL);

        DNN_SAFE_V(dnnl_post_ops_destroy(ops));
    }

    return dnnl_attr;
}

dnnl_format_tag_t get_default_tag(int ndims) {
    switch (ndims) {
        case 1: return dnnl_a;
        case 2: return dnnl_ab;
        case 3: return dnnl_abc;
        case 4: return dnnl_abcd;
        case 5: return dnnl_abcde;
        case 6: return dnnl_abcdef;
        default: assert(!"unsupported ndims");
    }
    return dnnl_format_tag_undef;
}

void maybe_scale(float &d, float *scales, int64_t oc, const attr_t &attr) {
    if (!attr.oscale.is_def()) {
        const auto &s = attr.oscale;
        if (s.policy == policy_t::COMMON) {
            d *= s.scale;
        } else {
            d *= scales[oc];
        }
    }
}

float compute_eltwise_fwd(
        pk_t kind, float src, float scale, float alpha, float beta) {
    using namespace dnnl::impl::math;

    switch (kind) {
        case pk_t::RELU: return scale * relu_fwd(src, alpha);
        case pk_t::TANH: return scale * tanh_fwd(src);
        case pk_t::ELU: return scale * elu_fwd(src, alpha);
        case pk_t::SQUARE: return scale * square_fwd(src);
        case pk_t::ABS: return scale * abs_fwd(src);
        case pk_t::SQRT: return scale * sqrt_fwd(src);
        case pk_t::LINEAR: return scale * linear_fwd(src, alpha, beta);
        case pk_t::BRELU: return scale * bounded_relu_fwd(src, alpha);
        case pk_t::SRELU: return scale * soft_relu_fwd(src);
        case pk_t::LOGISTIC: return scale * logistic_fwd(src);
        case pk_t::EXP: return scale * exp_fwd(src);
        case pk_t::GELU: return scale * gelu_fwd(src);
        case pk_t::SWISH: return scale * swish_fwd(src, alpha);
        case pk_t::LOG: return scale * log_fwd(src);
        case pk_t::CLIP: return scale * clip_fwd(src, alpha, beta);
        case pk_t::POW: return scale * pow_fwd(src, alpha, beta);
        default: assert(!"unknown attr::post_ops::kind");
    };
    return NAN;
}

float compute_eltwise_bwd(
        pk_t kind, float d_dst, float src, float alpha, float beta) {
    using namespace dnnl::impl::math;

    switch (kind) {
        case pk_t::RELU: return relu_bwd(d_dst, src, alpha);
        case pk_t::TANH: return tanh_bwd(d_dst, src);
        case pk_t::ELU: return elu_bwd(d_dst, src, alpha);
        case pk_t::SQUARE: return square_bwd(d_dst, src);
        case pk_t::ABS: return abs_bwd(d_dst, src);
        case pk_t::SQRT: return sqrt_bwd(d_dst, src);
        case pk_t::LINEAR: return linear_bwd(d_dst, src, alpha, beta);
        case pk_t::BRELU: return bounded_relu_bwd(d_dst, src, alpha);
        case pk_t::SRELU: return soft_relu_bwd(d_dst, src);
        case pk_t::LOGISTIC: return logistic_bwd(d_dst, src);
        case pk_t::EXP: return exp_bwd(d_dst, src);
        case pk_t::GELU: return gelu_bwd(d_dst, src);
        case pk_t::SWISH: return swish_bwd(d_dst, src, alpha);
        case pk_t::LOG: return log_bwd(d_dst, src);
        case pk_t::CLIP: return clip_bwd(d_dst, src, alpha, beta);
        case pk_t::POW: return pow_bwd(d_dst, src, alpha, beta);
        default: assert(!"unknown attr::post_ops::kind");
    }
    return NAN;
}

void maybe_post_ops(float &d, float dst, const attr_t &attr) {
    using namespace dnnl::impl::math;

    const auto &ops = attr.post_ops;
    for (int idx = 0; idx < ops.len; ++idx) {
        const auto &e = ops.entry[idx];

        const auto &s = e.eltwise.scale;
        const auto &a = e.eltwise.alpha;
        const auto &b = e.eltwise.beta;

        if (e.kind == pk_t::SUM)
            d += e.sum.scale * dst;
        else
            d = compute_eltwise_fwd(e.kind, d, s, a, b);
    }
}
