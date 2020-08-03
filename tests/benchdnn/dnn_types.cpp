/*******************************************************************************
* Copyright 2017-2020 Intel Corporation
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
#include "tests/test_thread.hpp"

#define BENCHDNN_DNNL_ARG_UNDEF 0

namespace tag {
const char *abx {"abx"};
const char *any {"any"};
const char *undef {"undef"};
} // namespace tag

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

std::ostream &operator<<(std::ostream &s, dir_t dir) {
#define CASE(x) \
    if (dir == x) return s << STRINGIFY(x)
    CASE(FWD_B);
    CASE(FWD_D);
    CASE(FWD_I);
    CASE(BWD_D);
    CASE(BWD_DW);
    CASE(BWD_W);
    CASE(BWD_WB);
#undef CASE
    SAFE_V(FAIL);
    return s;
}

std::ostream &operator<<(std::ostream &s, dnnl_data_type_t dt) {
    s << dt2str(dt);
    return s;
}

std::ostream &operator<<(std::ostream &s, dnnl_engine_kind_t ek) {
    s << engine_kind2str(ek);
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

static const std::map<int, const char *> supported_args {
        {DNNL_ARG_SRC, "src"},
        {DNNL_ARG_SRC_1, "src1"},
        {DNNL_ARG_WEIGHTS, "wei"},
        {DNNL_ARG_DST, "dst"},
};

static int str2arg(const std::string &str) {
    for (const auto &arg : supported_args)
        if (str.compare(arg.second) == 0) return arg.first;
    return BENCHDNN_DNNL_ARG_UNDEF;
}

policy_t attr_t::str2policy(const std::string &str) {
    std::string s(str);
    // s.compare is lexicographical, case matters
    std::transform(s.begin(), s.end(), s.begin(), ::toupper);
#define CASE(_plc) \
    if (s.compare(STRINGIFY(_plc)) == 0) return _plc
    CASE(COMMON);
    CASE(PER_OC);
    CASE(PER_DIM_0);
    CASE(PER_DIM_1);
    CASE(PER_DIM_01);
#undef CASE
    assert(!"unknown attr_t::policy_t policy");
    return POLICY_TOTAL;
}

const char *attr_t::policy2str(policy_t policy) {
    if (policy == COMMON) return "common";
    if (policy == PER_OC) return "per_oc";
    if (policy == PER_DIM_0) return "per_dim_0";
    if (policy == PER_DIM_1) return "per_dim_1";
    if (policy == PER_DIM_01) return "per_dim_01";
    assert(!"unknown attr_t::policy_t policy");
    return "unknown attr_t::policy_t policy";
}

int attr_t::get_default_mask(policy_t policy) {
    switch (policy) {
        case PER_DIM_0: return (1 << 0);
        case PER_OC:
        case PER_DIM_1: return (1 << 1);
        case PER_DIM_01: return (1 << 0) + (1 << 1);
        case COMMON: return 0;
        default: SAFE_V(FAIL); return 0;
    }
}

// This function return substring by @start_pos and @delim and shifts @start_pos
// to the place where @delim was found. Useful to parse peices of inputs
// separated by @delim.
std::string get_substr(
        const std::string &s, size_t &start_pos, char delim = ':') {
    auto end_pos = s.find_first_of(delim, start_pos);
    auto sub = s.substr(start_pos, end_pos - start_pos);
    start_pos = end_pos + (end_pos != std::string::npos);
    return sub;
}

// This function takes input string, extracts float value and runtime, if
// present, from the string. Updates @value and @runtime with extracted values.
int parse_value_and_runtime(float &value, bool &runtime, const std::string &s) {
    // process value
    size_t scale_pos = 0;
    value = std::stof(s, &scale_pos);
    runtime = false;
    if (scale_pos + 1 < s.size()) return FAIL;
    if (scale_pos == s.size()) return OK;
    if (s.back() != '*') return FAIL;
    runtime = true;
    return OK;
}

int attr_t::scale_t::from_str(const std::string &s) {
    *this = scale_t();
    if (s.empty()) return OK;

    size_t start_pos = 0;
    // process policy
    this->policy = str2policy(get_substr(s, start_pos));
    if (this->policy == POLICY_TOTAL) return FAIL;
    if (start_pos == std::string::npos) return OK;
    if (start_pos >= s.size()) return FAIL; // to catch dangling ':'

    int status = parse_value_and_runtime(
            this->scale, this->runtime, get_substr(s, start_pos));
    if (status != OK || this->scale < 0) return FAIL;
    return OK;
}

int attr_t::zero_points_t::from_str(const std::string &s) {
    *this = zero_points_t();
    if (s.empty()) return OK;

    size_t start_pos = 0;
    while (start_pos != std::string::npos) {
        auto arg = str2arg(get_substr(s, start_pos));
        if (arg == BENCHDNN_DNNL_ARG_UNDEF || start_pos == std::string::npos
                || start_pos >= s.size())
            return FAIL;

        auto policy = str2policy(get_substr(s, start_pos));
        if (policy == POLICY_TOTAL || start_pos == std::string::npos
                || start_pos >= s.size())
            return FAIL;

        float zp = 0;
        bool runtime = false;
        int status = parse_value_and_runtime(
                zp, runtime, get_substr(s, start_pos, '_'));
        if (status != OK) return status;

        set(arg, policy, (int)zp, runtime); // XXX: overflow/underflow?
    }
    return OK;
}

int attr_t::arg_scales_t::from_str(const std::string &s) {
    *this = arg_scales_t();
    if (s.empty()) return OK;

    size_t start_pos = 0;
    while (start_pos != std::string::npos) {
        auto arg = str2arg(get_substr(s, start_pos));
        if (arg == BENCHDNN_DNNL_ARG_UNDEF || start_pos == std::string::npos
                || start_pos >= s.size())
            return FAIL;

        auto policy_str = get_substr(s, start_pos);
        auto scale_str = policy_str + ":" + get_substr(s, start_pos, '_');
        scale_t arg_scale;
        auto status = arg_scale.from_str(scale_str.c_str());
        if (status != OK) return status;
        set(arg, arg_scale);
    }
    return OK;
}

using pk_t = attr_t::post_ops_t::kind_t;

typedef struct {
    pk_t kind;
    const char *kind_name;
    dnnl_alg_kind_t dnnl_kind;
} po_table_entry_t;

static po_table_entry_t kind_table[] = {
        // sum
        {pk_t::SUM, "sum", dnnl_alg_kind_undef},
        // depthwise convolution
        {pk_t::DW_K3S1P1, "dw_k3s1p1", dnnl_convolution_auto},
        {pk_t::DW_K3S2P1, "dw_k3s2p1", dnnl_convolution_auto},
        // eltwise
        {pk_t::ELTWISE_START, "eltwise_undef", dnnl_alg_kind_undef},
        {pk_t::ABS, "abs", dnnl_eltwise_abs},
        {pk_t::BRELU, "bounded_relu", dnnl_eltwise_bounded_relu},
        {pk_t::BRELU, "brelu", dnnl_eltwise_bounded_relu},
        {pk_t::CLIP, "clip", dnnl_eltwise_clip},
        {pk_t::ELU, "elu", dnnl_eltwise_elu},
        {pk_t::ELU_DST, "elu_dst", dnnl_eltwise_elu_use_dst_for_bwd},
        {pk_t::EXP, "exp", dnnl_eltwise_exp},
        {pk_t::EXP_DST, "exp_dst", dnnl_eltwise_exp_use_dst_for_bwd},
        {pk_t::GELU_ERF, "gelu_erf", dnnl_eltwise_gelu_erf},
        {pk_t::GELU_TANH, "gelu_tanh", dnnl_eltwise_gelu_tanh},
        {pk_t::LINEAR, "linear", dnnl_eltwise_linear},
        {pk_t::LOG, "log", dnnl_eltwise_log},
        {pk_t::LOGISTIC, "logistic", dnnl_eltwise_logistic},
        {pk_t::LOGISTIC_DST, "logistic_dst",
                dnnl_eltwise_logistic_use_dst_for_bwd},
        {pk_t::POW, "pow", dnnl_eltwise_pow},
        {pk_t::RELU, "relu", dnnl_eltwise_relu},
        {pk_t::RELU_DST, "relu_dst", dnnl_eltwise_relu_use_dst_for_bwd},
        {pk_t::ROUND, "round", dnnl_eltwise_round},
        {pk_t::SQRT, "sqrt", dnnl_eltwise_sqrt},
        {pk_t::SQRT_DST, "sqrt_dst", dnnl_eltwise_sqrt_use_dst_for_bwd},
        {pk_t::SQUARE, "square", dnnl_eltwise_square},
        {pk_t::SRELU, "soft_relu", dnnl_eltwise_soft_relu},
        {pk_t::SRELU, "srelu", dnnl_eltwise_soft_relu},
        {pk_t::SWISH, "swish", dnnl_eltwise_swish},
        {pk_t::TANH, "tanh", dnnl_eltwise_tanh},
        {pk_t::TANH_DST, "tanh_dst", dnnl_eltwise_tanh_use_dst_for_bwd},
        {pk_t::ELTWISE_END, "eltwise_undef", dnnl_alg_kind_undef},
        // binary
        {pk_t::BINARY_START, "binary_undef", dnnl_alg_kind_undef},
        {pk_t::ADD, "add", dnnl_binary_add},
        {pk_t::MAX, "max", dnnl_binary_max},
        {pk_t::MIN, "min", dnnl_binary_min},
        {pk_t::MUL, "mul", dnnl_binary_mul},
        {pk_t::BINARY_END, "binary_undef", dnnl_alg_kind_undef},
        // guard entry
        {pk_t::KIND_TOTAL, "kind_undef", dnnl_alg_kind_undef}};

pk_t attr_t::post_ops_t::str2kind(const std::string &str) {
    std::string s(str);
    // s.compare is lexicographical, case matters
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    for (const auto &e : kind_table) {
        if (s.compare(e.kind_name) == 0) return e.kind;
    }
    assert(!"unknown attr_t::post_ops_t::kind_t kind");
    const auto table_size = sizeof(kind_table) / sizeof(*kind_table);
    return kind_table[table_size - 1].kind;
}

const char *attr_t::post_ops_t::kind2str(pk_t kind) {
    for (const auto &e : kind_table) {
        if (e.kind == kind) return e.kind_name;
    }
    assert(!"unknown attr::post_ops::kind");
    const auto table_size = sizeof(kind_table) / sizeof(*kind_table);
    return kind_table[table_size - 1].kind_name;
}

dnnl_alg_kind_t attr_t::post_ops_t::kind2dnnl_kind(pk_t kind) {
    for (const auto &e : kind_table) {
        if (e.kind == kind) return e.dnnl_kind;
    }
    assert(!"unknown attr::post_ops::kind");
    const auto table_size = sizeof(kind_table) / sizeof(*kind_table);
    return kind_table[table_size - 1].dnnl_kind;
}

int attr_t::post_ops_t::from_str(const std::string &s) {
    *this = post_ops_t();
    // "'" is mandatory as long as ";" is used as alg delimiter
    if (s.front() != '\'' || s.back() != '\'') return FAIL;
    if (s.size() == 2) return OK; // empty input

    // strip quotes to simplify further logic
    auto s_no_quotes = s.substr(1, s.size() - 2);

    // operate over substrings separated by ';' represeting a single post op
    // with further parsing of specific kind
    size_t start_pos = 0;
    while (start_pos != s_no_quotes.size() && start_pos != std::string::npos) {
        auto subs = get_substr(s_no_quotes, start_pos, ';');
        size_t subs_pos = 0;

        auto kind = str2kind(get_substr(subs, subs_pos));
        if (kind == KIND_TOTAL) return FAIL;

        entry.emplace_back(kind);
        if (subs_pos == std::string::npos) continue;
        if (subs_pos >= subs.size()) return FAIL; // to catch dangling ':'

        auto &e = entry.back();
        if (e.is_sum_kind()) {
            e.sum.scale = std::stof(get_substr(subs, subs_pos));
            if (subs_pos == std::string::npos) continue;
            if (subs_pos >= subs.size()) return FAIL; // to catch dangling ':'

            e.sum.dt = str2dt(get_substr(subs, subs_pos).c_str());
        } else if (e.is_convolution_kind()) {
            e.convolution.dst_dt = str2dt(get_substr(subs, subs_pos).c_str());
            if (subs_pos == std::string::npos) continue;
            if (subs_pos >= subs.size()) return FAIL; // to catch dangling ':'

            auto policy_str = get_substr(subs, subs_pos);
            auto scale_str = policy_str + ":" + get_substr(subs, subs_pos);
            auto status = e.convolution.oscale.from_str(scale_str.c_str());
            if (status != OK) return status;
        } else if (e.is_eltwise_kind()) {
            e.eltwise.alpha = std::stof(get_substr(subs, subs_pos));
            if (subs_pos == std::string::npos) continue;
            if (subs_pos >= subs.size()) return FAIL; // to catch dangling ':'

            e.eltwise.beta = std::stof(get_substr(subs, subs_pos));
            if (subs_pos == std::string::npos) continue;
            if (subs_pos >= subs.size()) return FAIL; // to catch dangling ':'

            e.eltwise.scale = std::stof(get_substr(subs, subs_pos));
            if (e.eltwise.scale <= 0) return FAIL;
        }
    }
    return OK;
}

bool attr_t::is_def() const {
    return oscale.is_def() && scales.is_def() && zero_points.is_def()
            && post_ops.is_def()
            && scratchpad_mode == dnnl_scratchpad_mode_library;
}

int attr_t::post_ops_t::find(pk_t kind, int start, int stop) const {
    if (stop == -1) stop = len();
    stop = MIN2(stop, len());
    for (int idx = start; idx < stop; ++idx)
        if (entry[idx].kind == kind) return idx;
    return -1;
}

bool attr_t::post_ops_t::entry_t::is_sum_kind() const {
    return kind == SUM;
}
bool attr_t::post_ops_t::entry_t::is_convolution_kind() const {
    return kind == DW_K3S1P1 || kind == DW_K3S2P1;
}
bool attr_t::post_ops_t::entry_t::is_eltwise_kind() const {
    return kind > ELTWISE_START && kind < ELTWISE_END;
}

int attr_t::post_ops_t::eltwise_index() const {
    for (int i = 0; i < len(); ++i) {
        if (entry[i].is_eltwise_kind()) return i;
    }
    return -1;
}

int attr_t::post_ops_t::convolution_index() const {
    for (int i = 0; i < len(); ++i) {
        if (entry[i].is_convolution_kind()) return i;
    }
    return -1;
}

int str2attr(attr_t *attr, const char *str) {
    if (attr == NULL || str == NULL) return FAIL;

    *attr = attr_t();
    std::string s(str), entry_name;
    size_t start_pos = 0, end_pos = start_pos;

    entry_name = "oscale=";
    start_pos = s.find(entry_name);
    if (start_pos != std::string::npos) {
        start_pos += entry_name.size();
        // ';' is an attribute delimeter
        auto status = attr->oscale.from_str(get_substr(s, start_pos, ';'));
        if (status != OK) return status;
    }

    entry_name = "zero_points=";
    start_pos = s.find(entry_name);
    if (start_pos != std::string::npos) {
        start_pos += entry_name.size();
        auto status = attr->zero_points.from_str(get_substr(s, start_pos, ';'));
        if (status != OK) return status;
    }

    entry_name = "scales=";
    start_pos = s.find(entry_name);
    if (start_pos != std::string::npos) {
        start_pos += entry_name.size();
        auto status = attr->scales.from_str(get_substr(s, start_pos, ';'));
        if (status != OK) return status;
    }

    entry_name = "post_ops=";
    start_pos = s.find(entry_name);
    if (start_pos != std::string::npos) {
        start_pos += entry_name.size();
        // "\'\'" separates post-ops from everything else
        end_pos = s.find_first_of('\'', s.find_first_of('\'', start_pos) + 1);
        // +1 to include '\'' as an essential part of post-ops
        auto parse_str = s.substr(start_pos, end_pos - start_pos + 1);
        auto status = attr->post_ops.from_str(parse_str);
        if (status != OK) return status;
    }

    return OK;
}

void handle_legacy_attr(attr_t &attr, const attr_t &legacy_attr) {
    if (legacy_attr.is_def()) return;

    if (!attr.is_def()) {
        BENCHDNN_PRINT(0, "%s\n",
                "ERROR: both `--attr=` and one of `--attr-post-ops=`, "
                "`--attr-scales=`, `--attr-zero-points=` or `--attr-oscale=` "
                "options are specified, please use latter options.");
        SAFE_V(FAIL);
    }

    attr = legacy_attr;
}

std::ostream &operator<<(std::ostream &s, const policy_t &policy) {
    s << attr_t::policy2str(policy);
    return s;
}

std::ostream &operator<<(std::ostream &s, const attr_t::scale_t &scale) {
    s << scale.policy << ":" << scale.scale;
    if (scale.runtime) s << '*';
    return s;
}

std::ostream &operator<<(
        std::ostream &s, const attr_t::zero_points_t &zero_points) {
    bool first = true;
    for (const auto &point : zero_points.points) {
        if (!first) s << '_';
        first = false;

        s << supported_args.at(point.first) << ":" << point.second.policy << ":"
          << point.second.value;
        if (point.second.runtime) s << '*';
    }

    return s;
}

std::ostream &operator<<(std::ostream &s, const attr_t::arg_scales_t &scales) {
    bool first = true;
    for (const auto &v : scales.scales) {
        if (!v.second.is_def()) {
            if (!first) s << '_';
            first = false;

            s << supported_args.at(v.first) << ":" << v.second;
        }
    }
    return s;
}

std::ostream &operator<<(std::ostream &s, const attr_t::post_ops_t::kind_t &k) {
    s << attr_t::post_ops_t::kind2str(k);
    return s;
}

std::ostream &operator<<(std::ostream &s, const attr_t::post_ops_t &post_ops) {
    s << "'";

    for (int idx = 0; idx < post_ops.len(); ++idx) {
        if (idx > 0) s << ";";

        const auto &e = post_ops.entry[idx];
        s << e.kind;

        if (e.is_sum_kind()) {
            if (e.sum.scale != 1.0f || e.sum.dt != dnnl_data_type_undef)
                s << ":" << e.sum.scale;
            if (e.sum.dt != dnnl_data_type_undef) s << ":" << e.sum.dt;
        } else if (e.is_convolution_kind()) {
            if (e.convolution.dst_dt != dnnl_f32)
                s << ":" << e.convolution.dst_dt;
            const auto &co = e.convolution.oscale;
            if (!co.is_def()) s << ":" << co;
        } else if (e.is_eltwise_kind()) {
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

std::ostream &operator<<(std::ostream &s, dnnl_scratchpad_mode_t sm) {
    s << scratchpad_mode2str(sm);
    return s;
}

std::ostream &operator<<(std::ostream &s, const attr_t &attr) {
    if (!attr.is_def()) {
        if (!attr.oscale.is_def()) s << "--attr-oscale=" << attr.oscale << " ";
        if (!attr.scales.is_def()) s << "--attr-scales=" << attr.scales << " ";
        if (!attr.zero_points.is_def())
            s << "--attr-zero-points=" << attr.zero_points << " ";
        if (!attr.post_ops.is_def())
            s << "--attr-post-ops=\"" << attr.post_ops << "\" ";
        if (attr.scratchpad_mode != dnnl_scratchpad_mode_library)
            s << "--attr-scratchpad=" << attr.scratchpad_mode << " ";
    }
    return s;
}

std::ostream &dump_global_params(std::ostream &s) {
    s << "--" << driver_name << " ";
    if (canonical) s << "--canonical=" << bool2str(canonical) << " ";
    if (canonical || engine_tgt_kind != dnnl_cpu)
        s << "--engine=" << engine_tgt_kind << " ";
    if (canonical || fast_ref_gpu != true)
        s << "--fast-ref-gpu=" << bool2str(fast_ref_gpu) << " ";
    if (!skip_impl.empty()) s << "--skip-impl=" << skip_impl << " ";
    if (canonical || mem_check != true)
        s << "--mem-check=" << bool2str(mem_check) << " ";

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

dnnl_scratchpad_mode_t str2scratchpad_mode(const char *str) {
    const char *param = "library";
    if (!strncasecmp(param, str, strlen(param)))
        return dnnl_scratchpad_mode_library;

    param = "user";
    if (!strncasecmp(param, str, strlen(param)))
        return dnnl_scratchpad_mode_user;

    assert(!"not expected");
    return dnnl_scratchpad_mode_library;
}

void attr_args_t::prepare_output_scales(
        const attr_t &attr, const void *vals, int64_t count, int mask) {
    insert(DNNL_ARG_ATTR_OUTPUT_SCALES, vals, count, mask, attr.oscale.runtime);
}

dnnl_primitive_attr_t create_dnnl_attr(
        const attr_t &attr, const attr_args_t &attr_args) {
    dnnl_primitive_attr_t dnnl_attr = NULL;
    DNN_SAFE_V(dnnl_primitive_attr_create(&dnnl_attr));

    if (!attr.oscale.is_def()) {
        const auto &os_args = attr_args.get(DNNL_ARG_ATTR_OUTPUT_SCALES);
        const auto &policy = attr.oscale.policy;

        const auto count = os_args.get_count(policy);
        const auto mask = os_args.get_mask(policy);
        const auto scales = os_args.get_float_ptr();

        DNN_SAFE_V(dnnl_primitive_attr_set_output_scales(
                dnnl_attr, count, mask, scales));
    } else if (!attr.scales.is_def()) {
        for (const auto &arg : supported_args) {
            const auto arg_name = arg.first;
            const auto &as = attr.scales;
            if (as.is_def(arg_name)) continue;

            const auto &e = as.get(arg_name);
            // Only common policy is supported in the library at this point
            int64_t count = 1;
            int mask = attr_t::get_default_mask(e.policy);
            const float *scales = &e.scale;

            DNN_SAFE_V(dnnl_primitive_attr_set_scales(
                    dnnl_attr, arg_name, count, mask, scales));
        }
    }

    if (!attr.zero_points.is_def()) {
        for (const auto &arg : supported_args) {
            const auto arg_name = arg.first;
            const auto &zp = attr.zero_points;
            if (zp.is_def(arg_name)) continue;

            const auto &e = zp.get(arg_name);
            // Only common policy/single RT value are supported in the library
            // at this point
            int64_t count = 1;
            int mask = attr_t::get_default_mask(e.policy);
            const auto values = e.runtime ? &DNNL_RUNTIME_S32_VAL : &e.value;

            DNN_SAFE_V(dnnl_primitive_attr_set_zero_points(
                    dnnl_attr, arg_name, count, mask, values));
        }
    }

    if (!attr.post_ops.is_def()) {
        dnnl_post_ops_t ops;
        DNN_SAFE_V(dnnl_post_ops_create(&ops));

        const auto &po = attr.post_ops;
        for (int idx = 0; idx < po.len(); ++idx) {
            const auto &e = po.entry[idx];
            if (e.is_sum_kind()) {
                DNN_SAFE_V(dnnl_post_ops_append_sum_v2(
                        ops, e.sum.scale, e.sum.dt));
            } else if (e.is_eltwise_kind()) {
                DNN_SAFE_V(dnnl_post_ops_append_eltwise(ops, e.eltwise.scale,
                        e.eltwise.alg, e.eltwise.alpha, e.eltwise.beta));
            } else {
                assert(!"unknown attr::post_ops::kind");
            }
        }
        DNN_SAFE_V(dnnl_primitive_attr_set_post_ops(dnnl_attr, ops));

        const_dnnl_post_ops_t c_ops;
        DNN_SAFE_V(dnnl_primitive_attr_get_post_ops(dnnl_attr, &c_ops));
        SAFE_V(dnnl_post_ops_len(c_ops) == po.len() ? OK : FAIL);

        DNN_SAFE_V(dnnl_post_ops_destroy(ops));
    }

    DNN_SAFE_V(dnnl_primitive_attr_set_scratchpad_mode(
            dnnl_attr, attr.scratchpad_mode));

    return dnnl_attr;
}

dnnl_format_tag_t get_abx_tag(int ndims) {
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

dnnl_format_tag_t get_axb_tag(int ndims) {
    switch (ndims) {
        case 1: return dnnl_a;
        case 2: return dnnl_ab;
        case 3: return dnnl_acb;
        case 4: return dnnl_acdb;
        case 5: return dnnl_acdeb;
        default: assert(!"unsupported ndims");
    }
    return dnnl_format_tag_undef;
}

dnnl_format_tag_t get_xba_tag(int ndims) {
    switch (ndims) {
        case 1: return dnnl_a;
        case 2: return dnnl_ba;
        case 3: return dnnl_cba;
        case 4: return dnnl_cdba;
        case 5: return dnnl_cdeba;
        default: assert(!"unsupported ndims");
    }
    return dnnl_format_tag_undef;
}

dnnl_format_tag_t get_aBx4b_tag(int ndims) {
    switch (ndims) {
        case 3: return dnnl_aBc4b;
        case 4: return dnnl_aBcd4b;
        case 5: return dnnl_aBcde4b;
        default: assert(!"unsupported ndims");
    }
    return dnnl_format_tag_undef;
}

dnnl_format_tag_t get_aBx8b_tag(int ndims) {
    switch (ndims) {
        case 3: return dnnl_aBc8b;
        case 4: return dnnl_aBcd8b;
        case 5: return dnnl_aBcde8b;
        default: assert(!"unsupported ndims");
    }
    return dnnl_format_tag_undef;
}

dnnl_format_tag_t get_aBx16b_tag(int ndims) {
    switch (ndims) {
        case 3: return dnnl_aBc16b;
        case 4: return dnnl_aBcd16b;
        case 5: return dnnl_aBcde16b;
        default: assert(!"unsupported ndims");
    }
    return dnnl_format_tag_undef;
}

dnnl_format_tag_t get_ABx16a16b_tag(int ndims) {
    switch (ndims) {
        case 3: return dnnl_ABc16a16b;
        case 4: return dnnl_ABcd16a16b;
        case 5: return dnnl_ABcde16a16b;
        default: assert(!"unsupported ndims");
    }
    return dnnl_format_tag_undef;
}

dnnl_format_tag_t convert_tag(const std::string &tag_str, int ndims) {
    // List of supported meta-tags
    if (tag_str.compare("abx") == 0)
        return get_abx_tag(ndims);
    else if (tag_str.compare("axb") == 0)
        return get_axb_tag(ndims);
    else if (tag_str.compare("xba") == 0)
        return get_xba_tag(ndims);
    else if (tag_str.compare("aBx4b") == 0)
        return get_aBx4b_tag(ndims);
    else if (tag_str.compare("aBx8b") == 0)
        return get_aBx8b_tag(ndims);
    else if (tag_str.compare("aBx16b") == 0)
        return get_aBx16b_tag(ndims);
    else if (tag_str.compare("ABx16a16b") == 0)
        return get_ABx16a16b_tag(ndims);
    // fall-back to regular tag parse function
    return str2fmt_tag(tag_str.c_str());
}

void maybe_oscale(const attr_t &attr, float &d, float *scales, int64_t oc) {
    if (!attr.oscale.is_def()) {
        int64_t idx = attr.oscale.policy == policy_t::COMMON ? 0 : oc;
        d *= scales[idx];
    }
}

void maybe_zero_point(const attr_t &attr, float &d, const int32_t *zero_points,
        int64_t c, int arg, bool opposite_zero_point) {
    const auto &e = attr.zero_points.get(arg);

    if (!attr.zero_points.is_def(arg)) {
        const int idx = e.policy == policy_t::COMMON ? 0 : c;
        const int zp_sign = opposite_zero_point ? -1 : 1;
        d -= zp_sign * zero_points[idx];
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
        case pk_t::GELU_TANH: return scale * gelu_tanh_fwd(src);
        case pk_t::SWISH: return scale * swish_fwd(src, alpha);
        case pk_t::LOG: return scale * log_fwd(src);
        case pk_t::CLIP: return scale * clip_fwd(src, alpha, beta);
        case pk_t::POW: return scale * pow_fwd(src, alpha, beta);
        case pk_t::GELU_ERF: return scale * gelu_erf_fwd(src);
        case pk_t::ROUND: return scale * round_fwd(src);

        case pk_t::RELU_DST: return scale * relu_fwd(src, alpha);
        case pk_t::TANH_DST: return scale * tanh_fwd(src);
        case pk_t::ELU_DST: return scale * elu_fwd(src, alpha);
        case pk_t::SQRT_DST: return scale * sqrt_fwd(src);
        case pk_t::LOGISTIC_DST: return scale * logistic_fwd(src);
        case pk_t::EXP_DST: return scale * exp_fwd(src);

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
        case pk_t::GELU_TANH: return gelu_tanh_bwd(d_dst, src);
        case pk_t::SWISH: return swish_bwd(d_dst, src, alpha);
        case pk_t::LOG: return log_bwd(d_dst, src);
        case pk_t::CLIP: return clip_bwd(d_dst, src, alpha, beta);
        case pk_t::POW: return pow_bwd(d_dst, src, alpha, beta);
        case pk_t::GELU_ERF: return gelu_erf_bwd(d_dst, src);

        case pk_t::RELU_DST: return relu_bwd_use_dst(d_dst, src, alpha);
        case pk_t::TANH_DST: return tanh_bwd_use_dst(d_dst, src);
        case pk_t::ELU_DST: return elu_bwd_use_dst(d_dst, src, alpha);
        case pk_t::SQRT_DST: return sqrt_bwd_use_dst(d_dst, src);
        case pk_t::LOGISTIC_DST: return logistic_bwd_use_dst(d_dst, src);
        case pk_t::EXP_DST: return exp_bwd_use_dst(d_dst, src);

        default: assert(!"unknown attr::post_ops::kind");
    }
    return NAN;
}

float compute_binary(pk_t kind, float src0, float src1) {
    if (kind == pk_t::ADD) {
        return src0 + src1;
    } else if (kind == pk_t::MUL) {
        return src0 * src1;
    } else if (kind == pk_t::MAX) {
        return MAX2(src0, src1);
    } else if (kind == pk_t::MIN) {
        return MIN2(src0, src1);
    } else {
        assert(!"operation not supported!");
    }
    return 0;
}

void maybe_post_ops(const attr_t &attr, float &val, float sum_val) {
    using namespace dnnl::impl::math;

    const auto &po = attr.post_ops;
    for (int idx = 0; idx < po.len(); ++idx) {
        const auto &e = po.entry[idx];

        if (e.is_sum_kind()) {
            val += e.sum.scale * sum_val;
        } else if (e.is_convolution_kind()) {
            continue;
        } else if (e.is_eltwise_kind()) {
            const auto &s = e.eltwise.scale;
            const auto &a = e.eltwise.alpha;
            const auto &b = e.eltwise.beta;
            val = compute_eltwise_fwd(e.kind, val, s, a, b);
        }
    }
}

engine_t::engine_t(dnnl_engine_kind_t engine_kind) {
    DNN_SAFE_V(dnnl_engine_create(&engine_, engine_kind, 0));
}

engine_t::~engine_t() {
    DNN_SAFE_V(dnnl_engine_destroy(engine_));
}

stream_t::stream_t(dnnl_engine_t engine) {
    dnnl_engine_kind_t engine_kind;
    DNN_SAFE_V(dnnl_engine_get_kind(engine, &engine_kind));

    dnnl_stream_attr_t stream_attr;
    DNN_SAFE_V(dnnl_stream_attr_create(&stream_attr, engine_kind));
#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
    if (engine_kind == dnnl_cpu) {
        SAFE_V(dnnl_stream_attr_set_threadpool(
                stream_attr, dnnl::testing::get_threadpool()));
    }
#endif

    DNN_SAFE_V(dnnl_stream_create_v2(
            &stream_, engine, dnnl_stream_default_flags, stream_attr));
    dnnl_stream_attr_destroy(stream_attr);
}

stream_t::~stream_t() {
    DNN_SAFE_V(dnnl_stream_destroy(stream_));
}

#undef BENCHDNN_DNNL_ARG_UNDEF
