/*******************************************************************************
* Copyright 2017-2024 Intel Corporation
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
#include <cctype>
#include <cmath>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <string>

#include <algorithm>
#include <iostream>
#include <sstream>

#include "oneapi/dnnl/dnnl.h"

#include "src/common/math_utils.hpp"

#include "common.hpp"
#include "conv/conv_dw_fusion.hpp"
#include "dnn_types.hpp"
#include "dnnl_common.hpp"
#include "dnnl_debug.hpp"
#include "dnnl_memory.hpp"
#include "utils/cold_cache.hpp"
#include "utils/dims.hpp"
#include "utils/parser.hpp"
#include "utils/stream_kind.hpp"

namespace tag {
const char *x {"x"};
const char *abx {"abx"};
const char *axb {"axb"};
const char *any {"any"};
const char *undef {"undef"};
} // namespace tag

std::ostream &operator<<(std::ostream &s, dir_t dir) {
#define CASE(x) \
    if (dir == (x)) return s << STRINGIFY(x)
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

dnnl_prop_kind_t prop2prop_kind(const dir_t dir) {
    if (dir == FWD_D) return dnnl_forward_training;
    if (dir == FWD_I) return dnnl_forward_inference;
    if (dir == BWD_DW) return dnnl_backward;
    assert(!"unknown dir");
    return dnnl_prop_kind_undef;
}

const char *prop2str(dnnl_prop_kind_t prop) {
    if (prop == dnnl_forward_training) return "FWD_D";
    if (prop == dnnl_forward_inference) return "FWD_I";
    if (prop == dnnl_backward) return "BWD_DW";
    assert(!"unknown prop_kind");
    return "unknown prop_kind";
}

static const std::map<int, std::vector<const char *>> supported_args {
        {DNNL_ARG_SRC, {"src", "src0"}},
        {DNNL_ARG_SRC_1, {"src1"}},
        {DNNL_ARG_WEIGHTS, {"wei"}},
        {DNNL_ARG_DST, {"dst"}},
        {DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_DST, {"attr_post_op_dw_dst"}},
        {DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_WEIGHTS, {"attr_post_op_dw_wei"}},
};

static int str2arg(const std::string &str) {
    for (const auto &arg : supported_args)
        for (const auto &s : arg.second)
            if (str.compare(s) == 0) return arg.first;
    // multiple srcs
    std::string msrc = "msrc";
    if (str.compare(0, msrc.size(), msrc) == 0) {
        const auto &str_index = str.substr(msrc.size());
        if (str_index.empty() /* TODO: || non-digit string */) {
            BENCHDNN_PRINT(0, "%s\n",
                    "Error: \'msrc\' argument requires index to be specified.");
            return DNNL_ARG_UNDEF;
        }
        const auto index = stoul(str_index);
        return DNNL_ARG_MULTIPLE_SRC + index;
    }
    return DNNL_ARG_UNDEF;
}

static std::string arg2str(int arg) {
    if (supported_args.find(arg) != supported_args.end())
        return std::string(supported_args.at(arg)[0]);
    if (arg & DNNL_ARG_MULTIPLE_SRC) {
        std::string msrc("msrc");
        const int index = arg - DNNL_ARG_MULTIPLE_SRC;
        return msrc + std::to_string(index);
    }
    assert(!"unknown argument");
    return "unknown argument";
}

policy_t attr_t::str2policy(const std::string &str) {
    std::string s(str);
    // s.compare is lexicographical, case matters
    std::transform(s.begin(), s.end(), s.begin(), ::toupper);
#define CASE(_plc) \
    if (s.compare(STRINGIFY(_plc)) == 0) return _plc
    CASE(COMMON);
    CASE(PER_OC);
    CASE(PER_OCIC);
    CASE(PER_DIM_0);
    CASE(PER_DIM_1);
    CASE(PER_DIM_01);
    CASE(PER_DIM_2);
    CASE(PER_DIM_3);
    CASE(PER_TENSOR);
#undef CASE
    assert(!"unknown attr_t::policy_t policy");
    return POLICY_TOTAL;
}

const char *attr_t::policy2str(policy_t policy) {
    if (policy == COMMON) return "common";
    if (policy == PER_OC) return "per_oc";
    if (policy == PER_OCIC) return "per_ocic";
    if (policy == PER_DIM_0) return "per_dim_0";
    if (policy == PER_DIM_1) return "per_dim_1";
    if (policy == PER_DIM_01) return "per_dim_01";
    if (policy == PER_DIM_2) return "per_dim_2";
    if (policy == PER_DIM_3) return "per_dim_3";
    if (policy == PER_TENSOR) return "per_tensor";
    assert(!"unknown attr_t::policy_t policy");
    return "unknown attr_t::policy_t policy";
}

int attr_t::get_default_mask(policy_t policy) {
    switch (policy) {
        case PER_DIM_0: return (1 << 0);
        case PER_OC:
        case PER_DIM_1: return (1 << 1);
        case PER_OCIC:
        case PER_DIM_01: return (1 << 0) + (1 << 1);
        case PER_DIM_2: return (1 << 2);
        case PER_DIM_3: return (1 << 3);
        case PER_TENSOR: return (1 << DNNL_MAX_NDIMS) - 1;
        case COMMON: return 0;
        default: SAFE(FAIL, CRIT); return 0;
    }
}

int attr_t::policy2mask(int arg, policy_t policy,
        dnnl_primitive_kind_t prim_kind, int ndims, bool has_groups) {

    // Handle of weights mask for various primitives.
    if (prim_kind == dnnl_convolution || prim_kind == dnnl_deconvolution
            || prim_kind == dnnl_inner_product) {
        if (arg != DNNL_ARG_WEIGHTS || policy == policy_t::COMMON)
            return attr_t::get_default_mask(policy);

        switch (policy) {
            case PER_OC:
                if (has_groups)
                    return attr_t::get_default_mask(PER_DIM_01);
                else
                    return attr_t::get_default_mask(PER_DIM_0);
            default: SAFE(FAIL, CRIT); return -1;
        }
    } else if (prim_kind == dnnl_matmul) {
        if (arg != DNNL_ARG_WEIGHTS || policy == policy_t::COMMON)
            return attr_t::get_default_mask(policy);

        if (ndims <= 0) SAFE_V(FAIL);
        switch (policy) {
            case PER_OC: return (1 << (ndims - 1));
            case PER_OCIC: return (1 << (ndims - 1)) + (1 << (ndims - 2));
            default: SAFE_V(FAIL); return -1;
        }
    } else if (prim_kind == dnnl_layer_normalization) {
        if (arg != DNNL_ARG_SRC_1 || policy != policy_t::PER_OC)
            return attr_t::get_default_mask(policy);

        // PER_OC
        assert(policy == policy_t::PER_OC);
        if (ndims <= 0) SAFE_V(FAIL);
        return 1 << (ndims - 1);
    } else {
        // Default case
        return attr_t::get_default_mask(policy);
    }
}

// This function takes input string, extracts float value and asteriks, if
// present, from the string. Updates @value with extracted values.
int parse_value_and_runtime(float &value, const std::string &s) {
    // process value
    size_t scale_pos = 0;
    try {
        value = std::stof(s, &scale_pos);
    } catch (const std::invalid_argument &) {
        BENCHDNN_PRINT(0, "%s \'%s\'.\n",
                "Error: scale or zero point input value is expected to be a "
                "real number. Given input:",
                s.c_str());
        SAFE_V(FAIL);
    }
    if (scale_pos != s.size()) {
        BENCHDNN_PRINT(0, "%s \'%s\'. %s \'%g\'.\n",
                "Error: not every input symbol was processed. Given input:",
                s.c_str(), "Parsed value:", value);
        SAFE_V(FAIL);
    }
    return OK;
}

#define HANDLE_DANGLING_SYMBOL_AND_END_OF_STRING() \
    if (start_pos == std::string::npos) return OK; \
    if (start_pos >= s.size()) { \
        BENCHDNN_PRINT(0, "%s \'%s\'\n", \
                "Error: dangling symbol at the end of input", s.c_str()); \
        SAFE_V(FAIL); \
    }

int attr_t::arg_scales_t::entry_t::from_str(const std::string &s) {
    *this = arg_scales_t::entry_t();
    if (s.empty()) return OK;

    size_t start_pos = 0;
    // process policy
    const auto policy_str = parser::get_substr(s, start_pos, ':');
    this->policy = str2policy(policy_str);
    if (this->policy == POLICY_TOTAL) {
        BENCHDNN_PRINT(0, "%s \'%s\' %s\n", "Error: Scale entry policy",
                policy_str.c_str(), "is not recognized.");
        SAFE_V(FAIL);
    }
    HANDLE_DANGLING_SYMBOL_AND_END_OF_STRING();

    // process scale value for COMMON policy
    if (this->policy == COMMON) {
        SAFE(parse_value_and_runtime(
                     this->scale, parser::get_substr(s, start_pos, ':')),
                WARN);
        if (this->scale < 0) {
            BENCHDNN_PRINT(0, "%s \'%g\'\n",
                    "Error: the scale can't be negative:", this->scale);
            SAFE_V(FAIL);
        }
    }
    HANDLE_DANGLING_SYMBOL_AND_END_OF_STRING();

    // process data type
    const auto dt_str = parser::get_substr(s, start_pos, ':');
    this->dt = str2dt(dt_str.c_str());
    if (this->dt == dnnl_data_type_undef) {
        BENCHDNN_PRINT(0, "Error: data type \'%s\' was not recognized.\n",
                dt_str.c_str());
        SAFE_V(FAIL);
    }
    HANDLE_DANGLING_SYMBOL_AND_END_OF_STRING();

    // process groups
    const auto g_str = parser::get_substr(s, start_pos, ':');
    parser::parse_vector_str(this->groups, dims_t(),
            parser::parser_utils::stoll_safe, g_str, 'x');

    if (!groups.empty()) {
        switch (this->policy) {
            case PER_OCIC:
                if (this->groups.size() != 2) {
                    BENCHDNN_PRINT(0, "%s\n",
                            "Error: number of groups should be equal to number "
                            "of dimension bits set in the mask.");
                    SAFE_V(FAIL);
                }
                break;
            default:
                BENCHDNN_PRINT(0, "%s\n",
                        "Error: groups are supported only for policy PER_OCIC");
                SAFE_V(FAIL);
        }
    }
    HANDLE_DANGLING_SYMBOL_AND_END_OF_STRING();

    return OK;
}

int attr_t::zero_points_t::entry_t::from_str(const std::string &s) {
    *this = zero_points_t::entry_t();
    if (s.empty()) return OK;

    size_t start_pos = 0;

    // process policy
    const auto policy_str = parser::get_substr(s, start_pos, ':');
    this->policy = str2policy(policy_str);
    if (this->policy == POLICY_TOTAL) {
        BENCHDNN_PRINT(0, "Error: policy \'%s\' was not recognized.\n",
                policy_str.c_str());
        SAFE_V(FAIL);
    }
    HANDLE_DANGLING_SYMBOL_AND_END_OF_STRING();

    if (this->policy == COMMON) {
        float value = 0.0f;
        SAFE(parse_value_and_runtime(
                     value, parser::get_substr(s, start_pos, ':')),
                WARN);
        int zp_val = static_cast<int>(value);
        if (static_cast<float>(zp_val) != value) {
            BENCHDNN_PRINT(0, "%s \'%d\'\n",
                    "Error: the zero point is not exact:", zp_val);
            SAFE_V(FAIL);
        }
        this->value = zp_val;
    }
    HANDLE_DANGLING_SYMBOL_AND_END_OF_STRING();

    // process data type
    const auto dt_str = parser::get_substr(s, start_pos, ':');
    this->dt = str2dt(dt_str.c_str());
    if (this->dt == dnnl_data_type_undef) {
        BENCHDNN_PRINT(0, "Error: data type \'%s\' was not recognized.\n",
                dt_str.c_str());
        SAFE_V(FAIL);
    }
    HANDLE_DANGLING_SYMBOL_AND_END_OF_STRING();

    // process groups
    const auto g_str = parser::get_substr(s, start_pos, ':');
    parser::parse_vector_str(this->groups, dims_t(),
            parser::parser_utils::stoll_safe, g_str, 'x');
    if (!groups.empty()) {
        switch (this->policy) {
            case PER_OCIC:
                if (this->groups.size() != 2) {
                    BENCHDNN_PRINT(0, "%s\n",
                            "Error: number of groups should be equal to number "
                            "of dimension bits set in the mask.");
                    SAFE_V(FAIL);
                }
                break;
            default:
                BENCHDNN_PRINT(0, "%s\n",
                        "Error: groups are supported only for policy PER_OCIC");
                SAFE_V(FAIL);
        }
    }
    HANDLE_DANGLING_SYMBOL_AND_END_OF_STRING();

    return OK;
}

#undef HANDLE_DANGLING_SYMBOL_AND_END_OF_STRING

int attr_t::zero_points_t::from_str(const std::string &s) {
    *this = zero_points_t();
    if (s.empty()) return OK;

    size_t start_pos = 0;
    while (start_pos != std::string::npos) {
        auto subs = parser::get_substr(s, start_pos, '+');
        size_t subs_pos = 0;

        auto arg = str2arg(parser::get_substr(subs, subs_pos, ':'));
        if (arg == DNNL_ARG_UNDEF || subs_pos == std::string::npos
                || subs_pos >= subs.size()) {
            BENCHDNN_PRINT(0,
                    "Error: argument name \'%s\' was not recognized.\n",
                    subs.c_str());
            SAFE_V(FAIL);
        }

        zero_points_t::entry_t zero_point;
        SAFE(zero_point.from_str(parser::get_substr(subs, subs_pos, '\0')),
                WARN);
        set(arg, zero_point);
    }
    return OK;
}

int attr_t::arg_scales_t::from_str(const std::string &s) {
    *this = arg_scales_t();
    if (s.empty()) return OK;

    size_t start_pos = 0;
    while (start_pos != std::string::npos) {
        auto subs = parser::get_substr(s, start_pos, '+');
        // Special handling for really big float values
        if (subs.back() == 'e') {
            auto subs_add = parser::get_substr(s, start_pos, '+');
            subs += subs_add;
        }
        size_t subs_pos = 0;

        auto arg = str2arg(parser::get_substr(subs, subs_pos, ':'));
        if (arg == DNNL_ARG_UNDEF || subs_pos == std::string::npos
                || subs_pos >= s.size()) {
            BENCHDNN_PRINT(0,
                    "Error: argument name \'%s\' was not recognized.\n",
                    subs.c_str());
            SAFE_V(FAIL);
        }

        arg_scales_t::entry_t arg_scale;
        SAFE(arg_scale.from_str(parser::get_substr(subs, subs_pos, '\0')),
                WARN);
        set(arg, arg_scale);
    }
    return OK;
}

using pk_t = attr_t::post_ops_t::kind_t;

struct po_table_entry_t {
    pk_t kind;
    std::vector<std::string> kind_names;
    dnnl_alg_kind_t dnnl_kind;
};

static po_table_entry_t kind_table[] = {
        // sum
        {pk_t::SUM, {"sum"}, dnnl_alg_kind_undef},
        // depthwise convolution
        {pk_t::DW, {"dw"}, dnnl_convolution_direct},
        // eltwise
        {pk_t::ELTWISE_START, {"eltwise_undef"}, dnnl_alg_kind_undef},
        {pk_t::ABS, {"abs", "eltwise_abs"}, dnnl_eltwise_abs},
        {pk_t::CLIP, {"clip", "eltwise_clip"}, dnnl_eltwise_clip},
        {pk_t::CLIP_V2, {"clip_v2", "eltwise_clip_v2"}, dnnl_eltwise_clip_v2},
        {pk_t::CLIP_V2_DST, {"clip_v2_dst", "eltwise_clip_v2_use_dst_for_bwd"},
                dnnl_eltwise_clip_v2_use_dst_for_bwd},
        {pk_t::ELU, {"elu", "eltwise_elu"}, dnnl_eltwise_elu},
        {pk_t::ELU_DST, {"elu_dst", "eltwise_elu_use_dst_for_bwd"},
                dnnl_eltwise_elu_use_dst_for_bwd},
        {pk_t::EXP, {"exp", "eltwise_exp"}, dnnl_eltwise_exp},
        {pk_t::EXP_DST, {"exp_dst", "eltwise_exp_use_dst_for_bwd"},
                dnnl_eltwise_exp_use_dst_for_bwd},
        {pk_t::GELU_ERF, {"gelu_erf", "eltwise_gelu_erf"},
                dnnl_eltwise_gelu_erf},
        {pk_t::GELU_TANH, {"gelu_tanh", "eltwise_gelu_tanh"},
                dnnl_eltwise_gelu_tanh},
        {pk_t::HARDSIGMOID, {"hardsigmoid", "eltwise_hardsigmoid"},
                dnnl_eltwise_hardsigmoid},
        {pk_t::HARDSWISH, {"hardswish", "eltwise_hardswish"},
                dnnl_eltwise_hardswish},
        {pk_t::LINEAR, {"linear", "eltwise_linear"}, dnnl_eltwise_linear},
        {pk_t::LOG, {"log", "eltwise_log"}, dnnl_eltwise_log},
        {pk_t::LOGISTIC, {"logistic", "eltwise_logistic"},
                dnnl_eltwise_logistic},
        {pk_t::LOGISTIC_DST,
                {"logistic_dst", "eltwise_logistic_use_dst_for_bwd"},
                dnnl_eltwise_logistic_use_dst_for_bwd},
        {pk_t::MISH, {"mish", "eltwise_mish"}, dnnl_eltwise_mish},
        {pk_t::POW, {"pow", "eltwise_pow"}, dnnl_eltwise_pow},
        {pk_t::RELU, {"relu", "eltwise_relu"}, dnnl_eltwise_relu},
        {pk_t::RELU_DST, {"relu_dst", "eltwise_relu_use_dst_for_bwd"},
                dnnl_eltwise_relu_use_dst_for_bwd},
        {pk_t::ROUND, {"round", "eltwise_round"}, dnnl_eltwise_round},
        {pk_t::SQRT, {"sqrt", "eltwise_sqrt"}, dnnl_eltwise_sqrt},
        {pk_t::SQRT_DST, {"sqrt_dst", "eltwise_sqrt_use_dst_for_bwd"},
                dnnl_eltwise_sqrt_use_dst_for_bwd},
        {pk_t::SQUARE, {"square", "eltwise_square"}, dnnl_eltwise_square},
        {pk_t::SRELU, {"soft_relu", "eltwise_soft_relu", "srelu"},
                dnnl_eltwise_soft_relu},
        {pk_t::SWISH, {"swish", "eltwise_swish"}, dnnl_eltwise_swish},
        {pk_t::TANH, {"tanh", "eltwise_tanh"}, dnnl_eltwise_tanh},
        {pk_t::TANH_DST, {"tanh_dst", "eltwise_tanh_use_dst_for_bwd"},
                dnnl_eltwise_tanh_use_dst_for_bwd},
        {pk_t::ELTWISE_END, {"eltwise_undef"}, dnnl_alg_kind_undef},
        // binary
        {pk_t::BINARY_START, {"binary_undef"}, dnnl_alg_kind_undef},
        {pk_t::ADD, {"add", "binary_add"}, dnnl_binary_add},
        {pk_t::DIV, {"div", "binary_div"}, dnnl_binary_div},
        {pk_t::EQ, {"eq", "binary_eq"}, dnnl_binary_eq},
        {pk_t::GE, {"ge", "binary_ge"}, dnnl_binary_ge},
        {pk_t::GT, {"gt", "binary_gt"}, dnnl_binary_gt},
        {pk_t::LE, {"le", "binary_le"}, dnnl_binary_le},
        {pk_t::LT, {"lt", "binary_lt"}, dnnl_binary_lt},
        {pk_t::MAX, {"max", "binary_max"}, dnnl_binary_max},
        {pk_t::MIN, {"min", "binary_min"}, dnnl_binary_min},
        {pk_t::MUL, {"mul", "binary_mul"}, dnnl_binary_mul},
        {pk_t::NE, {"ne", "binary_ne"}, dnnl_binary_ne},
        {pk_t::SUB, {"sub", "binary_sub"}, dnnl_binary_sub},
        {pk_t::BINARY_END, {"binary_undef"}, dnnl_alg_kind_undef},
        // prelu
        {pk_t::PRELU, {"prelu"}, dnnl_alg_kind_undef},
        // guard entry
        {pk_t::KIND_TOTAL, {"kind_undef"}, dnnl_alg_kind_undef}};

pk_t attr_t::post_ops_t::str2kind(const std::string &str) {
    std::string s(str);
    // string::operator== is lexicographical, case matters
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    for (const auto &e : kind_table) {
        for (const auto &name : e.kind_names) {
            if (s == name) return e.kind;
        }
    }
    BENCHDNN_PRINT(0, "%s\'%s\' %s\n", "Error: ", str.c_str(),
            "kind of post operation entry was not recognized.");

    const auto table_size = sizeof(kind_table) / sizeof(*kind_table);
    return kind_table[table_size - 1].kind;
}

const char *attr_t::post_ops_t::kind2str(pk_t kind) {
    for (const auto &e : kind_table) {
        if (e.kind == kind) return e.kind_names[0].c_str();
    }
    assert(!"unknown attr::post_ops::kind");
    const auto table_size = sizeof(kind_table) / sizeof(*kind_table);
    return kind_table[table_size - 1].kind_names[0].c_str();
}

dnnl_alg_kind_t attr_t::post_ops_t::kind2dnnl_kind(pk_t kind) {
    for (const auto &e : kind_table) {
        if (e.kind == kind) return e.dnnl_kind;
    }
    assert(!"unknown attr::post_ops::kind");
    const auto table_size = sizeof(kind_table) / sizeof(*kind_table);
    return kind_table[table_size - 1].dnnl_kind;
}

std::vector<std::pair<int, int>> attr_t::post_ops_t::get_po_masks(
        int ndims, dnnl_primitive_kind_t prim_kind) const {
    std::vector<std::pair<int, int>> v_masks;
    for (int idx = 0; idx < len(); ++idx) {
        const auto &e = this->entry[idx];
        int mask = -1;
        int arg = DNNL_ARG_UNDEF;
        if (e.is_binary_kind()) {
            using mask_input_t = entry_t::binary_t::mask_input_t;
            auto mask_input = e.binary.mask_input;
            mask = mask_input == mask_input_t::mask
                    ? e.binary.mask
                    : policy2mask(
                            DNNL_ARG_SRC_1, e.binary.policy, prim_kind, ndims);
            arg = DNNL_ARG_SRC_1;
        } else if (e.is_prelu_kind()) {
            mask = attr_t::get_default_mask(e.prelu.policy);
            arg = DNNL_ARG_WEIGHTS;
        } else
            continue;

        assert(mask >= 0);
        v_masks.emplace_back(std::make_pair(
                DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx) | arg, mask));
    }
    return v_masks;
}

bool attr_t::is_def(bool skip_fpmath) const {
    return scales.is_def() && zero_points.is_def() && post_ops.is_def()
            && scratchpad_mode == get_default_scratchpad_mode()
            && IMPLICATION(!skip_fpmath, fpmath_mode.is_def())
            && acc_mode == dnnl_accumulation_mode_strict
            && deterministic.is_def();
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
    return kind == DW;
}
bool attr_t::post_ops_t::entry_t::is_eltwise_kind() const {
    return kind > ELTWISE_START && kind < ELTWISE_END;
}
bool attr_t::post_ops_t::entry_t::is_binary_kind() const {
    return kind > pk_t::BINARY_START && kind < pk_t::BINARY_END;
}
bool attr_t::post_ops_t::entry_t::is_prelu_kind() const {
    return kind == PRELU;
}

int attr_t::post_ops_t::convolution_index() const {
    for (int i = 0; i < len(); ++i) {
        if (entry[i].is_convolution_kind()) return i;
    }
    return -1;
}

int attr_t::post_ops_t::eltwise_index() const {
    for (int i = 0; i < len(); ++i) {
        if (entry[i].is_eltwise_kind()) return i;
    }
    return -1;
}

int attr_t::post_ops_t::binary_index() const {
    for (int i = 0; i < len(); ++i) {
        if (entry[i].is_binary_kind()) return i;
    }
    return -1;
}

int attr_t::post_ops_t::prelu_index() const {
    for (int i = 0; i < len(); ++i) {
        if (entry[i].is_prelu_kind()) return i;
    }
    return -1;
}

std::ostream &operator<<(std::ostream &s, const policy_t &policy) {
    s << attr_t::policy2str(policy);
    return s;
}

std::ostream &operator<<(
        std::ostream &s, const attr_t::arg_scales_t::entry_t &scale) {
    using ::operator<<;

    s << scale.policy;
    if (scale.policy == policy_t::COMMON) s << ":" << scale.scale;
    if (scale.dt != dnnl_f32) s << ':' << scale.dt;
    if (!scale.groups.empty()) s << ":" << dims2str(scale.groups);
    return s;
}

std::ostream &operator<<(
        std::ostream &s, const attr_t::zero_points_t &zero_points) {
    using ::operator<<;

    const char *delim = "";
    for (const auto &point : zero_points.points) {
        s << delim;
        s << arg2str(point.first) << ":" << point.second.policy;
        if (point.second.policy == policy_t::COMMON)
            s << ":" << point.second.value;
        if (point.second.dt != dnnl_s32) s << ':' << point.second.dt;
        if (!point.second.groups.empty())
            s << ":" << dims2str(point.second.groups);
        delim = "+";
    }

    return s;
}

std::ostream &operator<<(std::ostream &s, const attr_t::arg_scales_t &scales) {
    const char *delim = "";
    for (const auto &v : scales.scales) {
        if (!v.second.is_def()) {
            s << delim;
            s << arg2str(v.first) << ":" << v.second;
            delim = "+";
        }
    }
    return s;
}

std::ostream &operator<<(std::ostream &s, const attr_t::post_ops_t::kind_t &k) {
    s << attr_t::post_ops_t::kind2str(k);
    return s;
}

std::ostream &operator<<(std::ostream &s, const attr_t::post_ops_t &post_ops) {
    for (int idx = 0; idx < post_ops.len(); ++idx) {
        if (idx > 0) s << "+";

        const auto &e = post_ops.entry[idx];
        s << e.kind;

        if (e.is_sum_kind()) {
            if (e.sum.scale != 1.0f || e.sum.zero_point != 0
                    || e.sum.dt != dnnl_data_type_undef)
                s << ":" << e.sum.scale;
            if (e.sum.zero_point != 0 || e.sum.dt != dnnl_data_type_undef)
                s << ":" << e.sum.zero_point;
            if (e.sum.dt != dnnl_data_type_undef) s << ":" << e.sum.dt;
        } else if (e.is_convolution_kind()) {
            s << ":k" << e.convolution.kernel << "s" << e.convolution.stride
              << "p" << e.convolution.padding;
            if (e.convolution.dst_dt != dnnl_f32)
                s << ":" << e.convolution.dst_dt;
        } else if (e.is_eltwise_kind()) {
            if (e.eltwise.beta != 0.f)
                s << ":" << e.eltwise.alpha << ":" << e.eltwise.beta;
            else if (e.eltwise.alpha != 0.f)
                s << ":" << e.eltwise.alpha;
        } else if (e.is_binary_kind()) {
            s << ":" << e.binary.src1_dt;
            using mask_input_t
                    = attr_t::post_ops_t::entry_t::binary_t::mask_input_t;
            if (e.binary.mask_input != mask_input_t::none
                    || e.binary.tag != tag::any) {
                if (e.binary.mask_input == mask_input_t::mask) {
                    s << ":" << e.binary.mask;
                } else {
                    assert(e.binary.mask_input == mask_input_t::policy);
                    s << ":" << e.binary.policy;
                }
            }
            if (e.binary.tag != tag::any) s << ":" << e.binary.tag;
        } else if (e.is_prelu_kind()) {
            if (e.prelu.policy != policy_t::COMMON) {
                s << ":" << e.prelu.policy;
            }
        } else {
            assert(!"unknown kind");
            s << "unknown_kind";
        }
    }

    return s;
}

std::ostream &operator<<(std::ostream &s, dnnl_scratchpad_mode_t sm) {
    s << scratchpad_mode2str(sm);
    return s;
}

std::ostream &operator<<(std::ostream &s, const attr_t::fpmath_mode_t &fm) {
    s << fpmath_mode2str(fm.mode);
    if (fm.apply_to_int) s << ":" << bool2str(fm.apply_to_int);
    return s;
}

std::ostream &operator<<(std::ostream &s, dnnl_accumulation_mode_t am) {
    s << accumulation_mode2str(am);
    return s;
}

std::ostream &operator<<(std::ostream &s, const attr_t::deterministic_t &d) {
    s << bool2str(d.enabled);
    return s;
}

std::ostream &operator<<(std::ostream &s, const attr_t &attr) {
    if (!attr.is_def()) {
        if (!attr.scales.is_def()) s << "--attr-scales=" << attr.scales << " ";
        if (!attr.zero_points.is_def())
            s << "--attr-zero-points=" << attr.zero_points << " ";
        if (!attr.post_ops.is_def())
            s << "--attr-post-ops=" << attr.post_ops << " ";
        if (attr.scratchpad_mode != attr_t::get_default_scratchpad_mode())
            s << "--attr-scratchpad=" << attr.scratchpad_mode << " ";
        if (!attr.fpmath_mode.is_def())
            s << "--attr-fpmath=" << attr.fpmath_mode << " ";
        if (attr.acc_mode != dnnl_accumulation_mode_strict)
            s << "--attr-acc-mode=" << attr.acc_mode << " ";
        if (!attr.deterministic.is_def())
            s << "--attr-deterministic=" << attr.deterministic << " ";
    }
    return s;
}

#ifdef DNNL_EXPERIMENTAL_SPARSE
std::ostream &operator<<(std::ostream &s, dnnl_sparse_encoding_t se) {
    s << sparse_encoding2str(se);
    return s;
}

std::ostream &operator<<(
        std::ostream &s, const sparse_options_t &sparse_options) {
    if (!sparse_options.is_def()) {
        s << "--encoding=";
        const std::vector<int> args
                = {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST};

        for (int i = 0; i < (int)args.size(); i++) {
            const int arg = args[i];
            if (!sparse_options.is_encoding_def(arg)) {
                s << sparse_options.get_encoding(arg);
                if (!sparse_options.is_sparsity_def(arg))
                    s << "+" << sparse_options.get_sparsity(arg);
            }
            if (i != (int)args.size() - 1)
                s << ":";
            else
                s << " ";
        }
    }
    return s;
}
#endif

std::ostream &operator<<(std::ostream &s, memory_kind_ext_t memory_kind) {
    switch (memory_kind) {
        case memory_kind_ext_t::usm: s << "usm"; break;
        case memory_kind_ext_t::buffer: s << "buffer"; break;
        case memory_kind_ext_t::usm_device: s << "usm_device"; break;
        case memory_kind_ext_t::usm_shared: s << "usm_shared"; break;
        default: assert(!"unexpected"); break;
    }
    return s;
}

std::ostream &dump_global_params(std::ostream &s) {
    // Need to dump mode and modifiers in front of the driver name to make all
    // updated default values take effect before parsing a state of a problem.
    if (canonical || bench_mode != default_bench_mode)
        s << "--mode=" << bench_mode << " ";
    // Don't dump modifiers if F mode is used to keep the repro simple.
    if (canonical
            || (bench_mode != bench_mode_t::perf_fast
                    && bench_mode_modifier != default_bench_mode_modifier))
        s << "--mode-modifier=" << bench_mode_modifier << " ";
    // Don't dump max_ms_per_prb if F mode is used to keep the repro simple.
    if (canonical
            || (bench_mode != bench_mode_t::perf_fast
                    && max_ms_per_prb != default_max_ms_per_prb))
        s << "--max-ms-per-prb=" << max_ms_per_prb << " ";
    if (canonical || fix_times_per_prb != default_fix_times_per_prb)
        s << "--fix-times-per-prb=" << fix_times_per_prb << " ";

    s << "--" << driver_name << " ";
    if (canonical) s << "--canonical=" << bool2str(canonical) << " ";
    if (canonical || engine_tgt_kind != dnnl_cpu) {
        s << "--engine=" << engine_tgt_kind;
        if (engine_index != 0) s << ":" << engine_index;
        s << " ";
    }
    if (canonical || fast_ref != default_fast_ref)
        s << "--fast-ref=" << bool2str(fast_ref) << " ";
    if (!skip_impl.empty()) s << "--skip-impl=" << skip_impl << " ";
    if (canonical || mem_check != true)
        s << "--mem-check=" << bool2str(mem_check) << " ";
    if (canonical || allow_enum_tags_only != true)
        s << "--allow-enum-tags-only=" << bool2str(allow_enum_tags_only) << " ";
    if (canonical || hints.get() != isa_hints_t::none)
        s << "--cpu-isa-hints=" << isa_hints_t::hints2str(hints) << " ";
    if (canonical || attr_same_pd_check != false)
        s << "--attr-same-pd-check=" << bool2str(attr_same_pd_check) << " ";
    if (canonical || check_ref_impl != false)
        s << "--check-ref-impl=" << bool2str(check_ref_impl) << " ";
#if defined(DNNL_WITH_SYCL) || DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    if (canonical || memory_kind != default_memory_kind)
        s << "--memory-kind=" << memory_kind << " ";
    if (canonical || stream_kind != default_stream_kind)
        s << "--stream-kind=" << stream_kind << " ";
#endif
    if (canonical || cold_cache_mode != default_cold_cache_mode)
        s << "--cold-cache=" << cold_cache_mode << " ";

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
    return attr_t::get_default_scratchpad_mode();
}

dnnl_fpmath_mode_t str2fpmath_mode(const char *str) {
    if (std::strcmp(str, "") == 0) {
        dnnl_fpmath_mode_t ret;
        dnnl_get_default_fpmath_mode(&ret);
        return ret;
    }

#define CASE(fpm) \
    param = #fpm; \
    if (!strncasecmp(param, str, strlen(param))) return dnnl_fpmath_mode_##fpm;

    const char *param;

    CASE(strict);
    CASE(bf16);
    CASE(f16);
    CASE(tf32);
    CASE(any);

    assert(!"not expected");
    return dnnl_fpmath_mode_strict;

#undef CASE
}

dnnl_accumulation_mode_t str2accumulation_mode(const char *str) {

#define CASE(am) \
    param = #am; \
    if (!strncasecmp(param, str, strlen(param))) \
        return dnnl_accumulation_mode_##am;

    const char *param;

    CASE(strict);
    CASE(relaxed);
    CASE(any);
    CASE(f32);
    CASE(s32);
    CASE(f16);

    assert(!"not expected");
    return dnnl_accumulation_mode_strict;

#undef CASE
}

struct post_ops_rhs_tensor_entry_t {
    dnnl_data_type_t dt;
    int mask;
    std::string tag;
    int arg_attr_mask;
};

namespace {

post_ops_rhs_tensor_entry_t get_po_rhs_tensor_entry(
        const attr_t::post_ops_t::entry_t &entry, int ndims,
        dnnl_primitive_kind_t prim_kind) {
    if (entry.is_prelu_kind()) {
        const auto &prelu = entry.prelu;
        const int mask = attr_t::get_default_mask(prelu.policy);
        return {dnnl_f32, mask, tag::axb, DNNL_ARG_WEIGHTS};
    } else if (entry.is_binary_kind()) {
        const auto &binary = entry.binary;
        using mask_input_t
                = attr_t::post_ops_t::entry_t::binary_t::mask_input_t;
        int mask = -1;
        switch (binary.mask_input) {
            // `none` is treated as `policy_t::COMMON`.
            case mask_input_t::none: mask = 0; break;
            case mask_input_t::mask: mask = binary.mask; break;
            case mask_input_t::policy:
                mask = attr_t::policy2mask(
                        DNNL_ARG_SRC_1, binary.policy, prim_kind, ndims);
                break;
            default: assert(!"unknown mask_input value"); break;
        }
        return {binary.src1_dt, mask, binary.tag, DNNL_ARG_SRC_1};
    }

    return post_ops_rhs_tensor_entry_t {};
}

} // namespace

int attr_args_t::prepare_post_ops_mds(const attr_t &attr, int ndims,
        const dnnl_dims_t prb_dims, dnnl_primitive_kind_t prim_kind) {
    const auto &po = attr.post_ops;
    dnnl_dims_t dims;
    for (int d = 0; d < ndims; ++d)
        dims[d] = prb_dims[d];
    // iterate over all post ops and prepare md for each binary
    for (int idx = 0; idx < po.len(); ++idx) {
        const auto &e = po.entry[idx];
        if (e.is_binary_kind() || e.is_prelu_kind()) {

            const auto po_rhs_tensor_entry
                    = get_po_rhs_tensor_entry(e, ndims, prim_kind);
            const int mask = po_rhs_tensor_entry.mask;

            // deduce binary, prelu dims based on input policy
            dnnl_dims_t rhs_tensor_dims = {};
            for (auto d = 0; d < ndims; ++d)
                rhs_tensor_dims[d] = (!(mask & (1 << d))) ? 1 : dims[d];

            auto rhs_tensor_desc = dnn_mem_t::init_md(ndims, rhs_tensor_dims,
                    po_rhs_tensor_entry.dt, po_rhs_tensor_entry.tag);
            mds.emplace((DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx)
                                | po_rhs_tensor_entry.arg_attr_mask),
                    std::move(rhs_tensor_desc));
        } else if (e.is_convolution_kind()) {
            // Update dims for post operations appended after conv_dw
            conv_dw_fusion::get_fused_conv_dst_dims(ndims, e, dims, dims);
        }
    }

    return OK;
}

void attr_args_t::prepare_dw_post_op(
        const attr_t &attr, dnnl_data_type_t wei_dt, dnnl_data_type_t bia_dt) {
    const int dw_idx = attr.post_ops.convolution_index();
    if (dw_idx == -1) return;

    dw_entry.wei_dt = wei_dt;
    dw_entry.bia_dt = bia_dt;
}

dnnl_primitive_attr_t create_dnnl_attr(
        const attr_t &attr, const attr_args_t &attr_args) {
    dnnl_primitive_attr_t dnnl_attr = nullptr;
    DNN_SAFE_V(dnnl_primitive_attr_create(&dnnl_attr));

    if (!attr.scales.is_def()) {
        const auto &as = attr.scales;
        for (const auto &arg : as.scales) {
            const int arg_name = arg.first;
            if (as.is_def(arg_name)) continue;

            const auto &e = arg.second;
            // Weights mask differs from primitive to primitive, that's why it
            // is stashed in `attr_args` at primitive creation time.
            const bool is_wei_arg = arg_name == DNNL_ARG_WEIGHTS
                    || arg_name
                            == (DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_WEIGHTS);
            int mask = (is_wei_arg && e.policy != policy_t::COMMON)
                    ? attr_args.get_mask(arg_name)
                    : attr_t::policy2mask(arg_name, e.policy);

            DNN_SAFE_V(dnnl_primitive_attr_set_scales(dnnl_attr, arg_name, mask,
                    static_cast<int>(e.groups.size()), e.groups.data(), e.dt));
        }
    }

    if (!attr.zero_points.is_def()) {
        const auto &zp = attr.zero_points;
        for (const auto &arg : zp.points) {
            const auto arg_name = arg.first;
            if (zp.is_def(arg_name)) continue;

            const auto &e = arg.second;
            // Weights mask differs from primitive to primitive, that's why it
            // is stashed in `attr_args` at primitive creation time.
            const bool is_wei_arg = arg_name == DNNL_ARG_WEIGHTS
                    || arg_name
                            == (DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_WEIGHTS);
            int mask = (is_wei_arg && e.policy != policy_t::COMMON)
                    ? attr_args.get_mask(DNNL_ARG_ATTR_ZERO_POINTS | arg_name)
                    : attr_t::policy2mask(arg_name, e.policy);

            int ndims = static_cast<int>(e.groups.size());
            const auto &groups = e.groups.data();
            const auto dt = e.dt;

            DNN_SAFE_V(dnnl_primitive_attr_set_zero_points(
                    dnnl_attr, arg_name, mask, ndims, groups, dt));
        }
    }

    if (!attr.post_ops.is_def()) {
        dnnl_post_ops_t ops;
        DNN_SAFE_V(dnnl_post_ops_create(&ops));

        const auto &po = attr.post_ops;
        for (int idx = 0; idx < po.len(); ++idx) {
            const auto &e = po.entry[idx];
            if (e.is_sum_kind()) {
                DNN_SAFE_V(dnnl_post_ops_append_sum(
                        ops, e.sum.scale, e.sum.zero_point, e.sum.dt));
            } else if (e.is_convolution_kind()) {
                const auto wei_dt = attr_args.get_dw_arg(DNNL_ARG_WEIGHTS);
                const auto bia_dt = attr_args.get_dw_arg(DNNL_ARG_BIAS);

                DNN_SAFE_V(dnnl_post_ops_append_dw(ops, wei_dt, bia_dt,
                        e.convolution.dst_dt, e.convolution.kernel,
                        e.convolution.stride, e.convolution.padding));
            } else if (e.is_eltwise_kind()) {
                DNN_SAFE_V(dnnl_post_ops_append_eltwise(
                        ops, e.eltwise.alg, e.eltwise.alpha, e.eltwise.beta));
            } else if (e.is_binary_kind()) {
                const auto &src1_md = attr_args.get_md(
                        (DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx) | DNNL_ARG_SRC_1));
                assert(query_md_ndims(src1_md) != 0);
                DNN_SAFE_V(dnnl_post_ops_append_binary(
                        ops, e.binary.alg, src1_md));
            } else if (e.is_prelu_kind()) {
                const auto &policy = e.prelu.policy;
                const auto mask = attr_t::get_default_mask(policy);
                DNN_SAFE_V(dnnl_post_ops_append_prelu(ops, mask));
            } else {
                assert(!"unknown attr::post_ops::kind");
            }
        }
        DNN_SAFE_V(dnnl_primitive_attr_set_post_ops(dnnl_attr, ops));
        auto c_ops = query_post_ops(dnnl_attr);
        SAFE_V(dnnl_post_ops_len(c_ops) == po.len() ? OK : FAIL);

        DNN_SAFE_V(dnnl_post_ops_destroy(ops));
    }

    DNN_SAFE_V(dnnl_primitive_attr_set_scratchpad_mode(
            dnnl_attr, attr.scratchpad_mode));

    DNN_SAFE_V(dnnl_primitive_attr_set_fpmath_mode_v2(
            dnnl_attr, attr.fpmath_mode.mode, attr.fpmath_mode.apply_to_int));

    DNN_SAFE_V(dnnl_primitive_attr_set_accumulation_mode(
            dnnl_attr, attr.acc_mode));

    DNN_SAFE_V(dnnl_primitive_attr_set_deterministic(
            dnnl_attr, attr.deterministic.enabled));

    return dnnl_attr;
}

// Exception free version of std::stoi, sets idx to 0 and returns 0 in case of
// error.
static int stoi_safe(const std::string &s, size_t *idx) {
    if (s.empty() || !std::isdigit(s[0])) {
        *idx = 0;
        return 0;
    }
    return std::stoi(s, idx);
}

static bool is_abc_tag(const std::string &tag) {
    if (tag == tag::undef || tag == tag::any) return true;

    bool mask[DNNL_MAX_NDIMS] = {};
    for (auto &c : tag) {
        if (!std::isalpha(c)) continue;
        int idx = std::tolower(c) - 'a';
        if (idx < 0 || idx >= DNNL_MAX_NDIMS) return false;
        mask[idx] = true;
    }
    // Check there are no gaps, e.g. [1 1 1 1 0 0 ...].
    for (int i = 0; i < DNNL_MAX_NDIMS; i++) {
        if (mask[i]) continue;
        for (int j = i + 1; j < DNNL_MAX_NDIMS; j++)
            if (mask[j]) return false;
        break;
    }
    return true;
}

int check_abc_tag(const std::string &tag_, bool check_enum_tags_only) {
    if (tag_.empty()) return FAIL;
    if (!is_abc_tag(tag_)) return FAIL;
    if (check_enum_tags_only) {
        if (str2fmt_tag(tag_.c_str()) == dnnl_format_tag_last) return FAIL;
        return OK;
    }

    enum class dim_state_t { undef = 0, upper, lower, lower_with_block };
    dim_state_t dim_states[DNNL_MAX_NDIMS] = {};
    bool in_inner_block = false;
    auto tag = tag_;
    while (!tag.empty()) {
        // Parse block size if presented.
        size_t idx;
        int block = stoi_safe(tag, &idx);
        if (block == 0 && idx != 0) return FAIL;
        if (idx == 0) block = 0;
        if (block > 0) in_inner_block = true;

        // Move to the first position after the block.
        tag = tag.substr(idx);
        if (tag.empty()) return FAIL;

        char c = tag[0];
        bool is_lower = ('a' <= c && c <= 'a' + DNNL_MAX_NDIMS - 1);
        bool is_upper = ('A' <= c && c <= 'A' + DNNL_MAX_NDIMS - 1);
        if (!is_lower && !is_upper) return FAIL;

        // Uppercase cannot be with block.
        if (is_upper && block != 0) return FAIL;
        // Block sizes are required within inner block.
        if (block == 0 && in_inner_block) return FAIL;

        // Check rules related to lowercase/uppercase/block order.
        int dim_idx = std::tolower(c) - 'a';
        dim_state_t prev_state = dim_states[dim_idx];
        dim_state_t cur_state = is_upper ? dim_state_t::upper
                : block != 0             ? dim_state_t::lower_with_block
                                         : dim_state_t::lower;

        switch (cur_state) {
            case dim_state_t::upper:
            case dim_state_t::lower:
                // Letter without block must be the first.
                if (prev_state != dim_state_t::undef) return FAIL;
                break;
            case dim_state_t::lower_with_block:
                // Letter with block must be after uppercase or after a letter
                // with block.
                if (prev_state != dim_state_t::upper
                        && prev_state != dim_state_t::lower_with_block)
                    return FAIL;
                break;
            default: assert(!"not expected");
        }

        // Update state, move to the next position.
        dim_states[dim_idx] = cur_state;
        tag = tag.substr(1);
    }

    for (int i = 0; i < DNNL_MAX_NDIMS; i++) {
        // Uppercase letter must be followed by lowercase.
        if (dim_states[i] == dim_state_t::upper) return FAIL;

        // Ensure there are no gaps (e.g. acd).
        if (dim_states[i] == dim_state_t::undef) {
            for (int j = i + 1; j < DNNL_MAX_NDIMS; j++)
                if (dim_states[j] != dim_state_t::undef) return FAIL;
            break;
        }
    }

    return OK;
}

static std::string trim_letter(const std::string &tag_, char c) {
    auto tag = tag_;
    for (size_t pos = tag.find(c); pos != std::string::npos;
            pos = tag.find(c)) {
        tag.replace(pos, 1, "");
        if (pos == 0) return tag;

        pos--;
        while (std::isdigit(tag[pos])) {
            tag.replace(pos, 1, "");
            if (pos == 0) break;
            pos--;
        }
    }
    return tag;
}

// Tries to map a tag to an abc-tag according to a logical tag. For example:
// nchw -> abcd.
static std::string try_map_tag(
        const std::string &logical_tag, const std::string &tag, int *nmatched) {
    // Check if all the required letters are presented.
    for (auto &c : logical_tag) {
        if (std::toupper(c) == c
                && tag.find(std::tolower(c)) == std::string::npos)
            return {};
    }

    // Check that all letters are known and assign indices to letters.
    int logical_indices[DNNL_MAX_NDIMS] = {};
    for (auto &c : tag) {
        if (!std::isalpha(c)) continue;

        auto lower_pos = logical_tag.find(std::tolower(c));
        auto upper_pos = logical_tag.find(std::toupper(c));
        auto pos = (lower_pos == std::string::npos ? upper_pos : lower_pos);
        if (pos == std::string::npos) return {};

        logical_indices[pos] = 1;
    }

    for (int i = 0, idx = 0; i < (int)logical_tag.size(); i++) {
        if (logical_indices[i] == 0) continue;
        logical_indices[i] = idx++;
    }

    (*nmatched)++;
    std::string mapped_tag = tag;
    for (int i = 0; i < (int)tag.size(); i++) {
        char c = tag[i];
        if (!std::isalpha(tag[i])) continue;
        size_t pos = logical_tag.find(std::tolower(c));
        if (pos == std::string::npos) pos = logical_tag.find(std::toupper(c));
        if (pos >= logical_tag.size()) SAFE_V(FAIL);

        mapped_tag[i]
                = (char)(tag[i] - std::tolower(c) + 'a' + logical_indices[pos]);
    }
    return mapped_tag;
}

// Maps a tag to an abc-tag.
static std::string map_tag_letters(const std::string &tag) {
    int nmatched = 0;

    // Mapping rules:
    // - Uppercase letters are mandatory
    // - Lowercase letters are optional
    auto tag_goidhw = try_map_tag("GOIdhw", tag, &nmatched);
    auto tag_oidhw = try_map_tag("OIdhw", tag, &nmatched);
    auto tag_ncdhw = try_map_tag("NCdhw", tag, &nmatched);
    auto tag_tnc = try_map_tag("TNc", tag, &nmatched);
    auto tag_ldnc = try_map_tag("LDNC", tag, &nmatched);
    auto tag_ldigo = try_map_tag("LDigO", tag, &nmatched);

    if (nmatched == 0) return tag;
    if (nmatched > 1) assert(!"Not expected: ambiguous tag.");

    if (!tag_goidhw.empty()) return tag_goidhw;
    if (!tag_oidhw.empty()) return tag_oidhw;
    if (!tag_ncdhw.empty()) return tag_ncdhw;
    if (!tag_tnc.empty()) return tag_tnc;
    if (!tag_ldnc.empty()) return tag_ldnc;
    if (!tag_ldigo.empty()) return tag_ldigo;

    return tag;
}

std::string trim_tag(const std::string &tag, int ndims) {
    int mask = 0;
    for (int d = 0; d < ndims; d++) {
        mask += (1 << d);
    }
    return trim_tag_by_mask(tag, mask);
}

std::string trim_tag_by_mask(const std::string &tag, int mask) {
    std::string trimmed_tag = tag;
    int ndims_saved = 0;
    for (char c = 'a', d = 0; c < 'a' + (char)(DNNL_MAX_NDIMS); c++, d++) {
        if (!(mask & (1 << d))) {
            trimmed_tag = trim_letter(trimmed_tag, c);
            trimmed_tag = trim_letter(trimmed_tag, std::toupper(c));
        } else {
            ndims_saved++;
        }
    }

    // Mask may operate over non-consecutive dimensions. The piece below will
    // make trimmed_tag consist of consecutive dimensions starting from "a" or
    // "A". E.g., mask = 2 + 8 = 10, trimmed_tag will contain "b" and "d"
    // letters, and will be converted into one with "a" and "b".
    int mask_copy = mask;
    for (int i = 0; i < ndims_saved; i++) {
        int dist_to_a = 0;
        while (mask_copy % 2 == 0) {
            mask_copy /= 2;
            dist_to_a++;
        }
        mask_copy /= 2;
        if (dist_to_a == 0) continue;

        for (size_t j = 0; j < trimmed_tag.size(); j++) {
            char str_j = trimmed_tag[j];
            if (std::isalpha(str_j) && std::tolower(str_j) > 'a' + i) {
                std::string rep_str(1, str_j - dist_to_a);
                trimmed_tag.replace(j, 1, rep_str);
            }
        }
    }

    return trimmed_tag;
}

std::string normalize_tag(const std::string &tag_, int ndims) {
    std::string tag = tag_;
    if (tag == tag::undef || tag == tag::any || ndims == 0) return tag;
    if (tag == tag::x) {
        if (ndims >= 0) assert(ndims == 1);
        return "a";
    }

    // Handle meta-tags (abx, axb, etc).
    auto pos = tag.find("x");
    if (pos != std::string::npos) {
        // Non-grouped tags will start `x` from `c`, but grouped will most of
        // times start `x` from `d`.
        char start_x = 'c';
        for (char c = 'a' + DNNL_MAX_NDIMS - 1; c >= 'b'; c--) {
            if (tag.find(c) != std::string::npos) {
                start_x = c + 1;
                break;
            }
        }
        // Adjust ndims if they are not specified.
        int meta_ndims = (ndims == -1 ? (start_x - 'a' + 1) : ndims);
        std::string tail;
        for (int i = 0; i < meta_ndims - (start_x - 'a'); i++)
            tail += (start_x + i);
        return trim_tag(tag.replace(pos, 1, tail), meta_ndims);
    }

    return map_tag_letters(tag);
}

int check_tag(const std::string &tag_, bool check_enum_tags_only) {
    auto tag = normalize_tag(tag_);
    if (tag == tag::undef || tag == tag::any) return OK;
    return check_abc_tag(tag, check_enum_tags_only);
}

void maybe_scale(const attr_t &attr, float &d, const float *scales, int64_t c,
        int arg, bool opposite_scale) {
    if (attr.scales.is_def(arg)) return;

    const auto &e = attr.scales.get(arg);
    if (!e.is_def()) {
        int64_t idx = e.policy == policy_t::COMMON ? 0 : c;
        float s = scales[idx];
        if (opposite_scale) s = 1.f / s;
        d *= s;
    }
}

void maybe_zero_point(const attr_t &attr, float &d, const int32_t *zero_points,
        int64_t c, int arg, bool opposite_zero_point) {
    if (attr.zero_points.is_def()) return;

    const auto &e = attr.zero_points.get(arg);
    if (!e.is_def()) {
        const int idx = e.policy == policy_t::COMMON ? 0 : c;
        const int zp_sign = opposite_zero_point ? -1 : 1;
        d -= zp_sign * zero_points[idx];
    }
}

float compute_eltwise_fwd(pk_t kind, float src, float alpha, float beta) {
    // don't compute on nan, propagate it
    if (std::isnan(src)) return NAN;

    using namespace dnnl::impl::math;

    switch (kind) {
        case pk_t::RELU: return relu_fwd(src, alpha);
        case pk_t::TANH: return tanh_fwd(src);
        case pk_t::ELU: return elu_fwd(src, alpha);
        case pk_t::SQUARE: return square_fwd(src);
        case pk_t::ABS: return abs_fwd(src);
        case pk_t::SQRT: return sqrt_fwd(src);
        case pk_t::LINEAR: return linear_fwd(src, alpha, beta);
        case pk_t::SRELU: return soft_relu_fwd(src, alpha);
        case pk_t::MISH: return mish_fwd(src);
        case pk_t::LOGISTIC: return logistic_fwd(src);
        case pk_t::EXP: return exp_fwd(src);
        case pk_t::GELU_TANH: return gelu_tanh_fwd(src);
        case pk_t::SWISH: return swish_fwd(src, alpha);
        case pk_t::LOG: return log_fwd(src);
        case pk_t::CLIP: return clip_fwd(src, alpha, beta);
        case pk_t::CLIP_V2: return clip_v2_fwd(src, alpha, beta);
        case pk_t::POW: return pow_fwd(src, alpha, beta);
        case pk_t::GELU_ERF: return gelu_erf_fwd(src);
        case pk_t::ROUND: return round_fwd(src);
        case pk_t::HARDSWISH: return hardswish_fwd(src, alpha, beta);
        case pk_t::HARDSIGMOID: return hardsigmoid_fwd(src, alpha, beta);
        case pk_t::RELU_DST: return relu_fwd(src, alpha);
        case pk_t::TANH_DST: return tanh_fwd(src);
        case pk_t::ELU_DST: return elu_fwd(src, alpha);
        case pk_t::SQRT_DST: return sqrt_fwd(src);
        case pk_t::LOGISTIC_DST: return logistic_fwd(src);
        case pk_t::EXP_DST: return exp_fwd(src);
        case pk_t::CLIP_V2_DST: return clip_v2_fwd(src, alpha, beta);

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
        case pk_t::SRELU: return soft_relu_bwd(d_dst, src, alpha);
        case pk_t::MISH: return mish_bwd(d_dst, src);
        case pk_t::LOGISTIC: return logistic_bwd(d_dst, src);
        case pk_t::EXP: return exp_bwd(d_dst, src);
        case pk_t::GELU_TANH: return gelu_tanh_bwd(d_dst, src);
        case pk_t::SWISH: return swish_bwd(d_dst, src, alpha);
        case pk_t::LOG: return log_bwd(d_dst, src);
        case pk_t::CLIP: return clip_bwd(d_dst, src, alpha, beta);
        case pk_t::CLIP_V2: return clip_v2_bwd(d_dst, src, alpha, beta);
        case pk_t::POW: return pow_bwd(d_dst, src, alpha, beta);
        case pk_t::GELU_ERF: return gelu_erf_bwd(d_dst, src);
        case pk_t::HARDSWISH: return hardswish_bwd(d_dst, src, alpha, beta);
        case pk_t::HARDSIGMOID: return hardsigmoid_bwd(d_dst, src, alpha, beta);

        case pk_t::RELU_DST: return relu_bwd_use_dst(d_dst, src, alpha);
        case pk_t::TANH_DST: return tanh_bwd_use_dst(d_dst, src);
        case pk_t::ELU_DST: return elu_bwd_use_dst(d_dst, src, alpha);
        case pk_t::SQRT_DST: return sqrt_bwd_use_dst(d_dst, src);
        case pk_t::LOGISTIC_DST: return logistic_bwd_use_dst(d_dst, src);
        case pk_t::EXP_DST: return exp_bwd_use_dst(d_dst, src);
        case pk_t::CLIP_V2_DST:
            return clip_v2_bwd_use_dst(d_dst, src, alpha, beta);

        default: assert(!"unknown attr::post_ops::kind");
    }
    return NAN;
}

float compute_binary(pk_t kind, float src0, float src1) {
    // don't compute on nan, propagate it
    if (std::isnan(src0) || std::isnan(src1)) return NAN;

    if (kind == pk_t::ADD) {
        return src0 + src1;
    } else if (kind == pk_t::MUL) {
        return src0 * src1;
    } else if (kind == pk_t::MAX) {
        return MAX2(src0, src1);
    } else if (kind == pk_t::MIN) {
        return MIN2(src0, src1);
    } else if (kind == pk_t::DIV) {
        return src0 / src1;
    } else if (kind == pk_t::SUB) {
        return src0 - src1;
    } else if (kind == pk_t::GE) {
        return src0 >= src1;
    } else if (kind == pk_t::GT) {
        return src0 > src1;
    } else if (kind == pk_t::LE) {
        return src0 <= src1;
    } else if (kind == pk_t::LT) {
        return src0 < src1;
    } else if (kind == pk_t::EQ) {
        return src0 == src1;
    } else if (kind == pk_t::NE) {
        return src0 != src1;
    } else {
        assert(!"operation not supported!");
    }
    return NAN;
}

void maybe_post_ops(const attr_t &attr, float &val, float sum_val,
        const std::vector<float> &v_po_vals) {
    using namespace dnnl::impl::math;

    auto it_po = v_po_vals.begin();
    const auto &po = attr.post_ops;
    for (int idx = 0; idx < po.len(); ++idx) {
        const auto &e = po.entry[idx];

        if (e.is_sum_kind()) {
            val += e.sum.scale * (sum_val - e.sum.zero_point);
        } else if (e.is_convolution_kind()) {
            continue;
        } else if (e.is_eltwise_kind()) {
            const auto &a = e.eltwise.alpha;
            const auto &b = e.eltwise.beta;
            val = compute_eltwise_fwd(e.kind, val, a, b);
        } else if (e.is_binary_kind()) {
            val = compute_binary(e.kind, val, *it_po);
            it_po++;
        } else if (e.is_prelu_kind()) {
            val = val > 0 ? val : val * (*it_po);
            it_po++;
        }
    }
}

void update_cpu_ref_attrs(attr_t &attr, dnnl_data_type_t new_dt) {
    auto &po = attr.post_ops;
    for (int idx = 0; idx < po.len(); ++idx) {
        auto &e = po.entry[idx];
        if (!e.is_binary_kind()) continue;

        e.binary.src1_dt = new_dt;
        e.binary.tag = tag::abx; // Hardcoded in local fill functions.
        // Since tag is updated, it might get printed with policy, which means
        // that mask_input should be specified.
        using mask_input_t
                = attr_t::post_ops_t::entry_t::binary_t::mask_input_t;
        if (e.binary.mask_input == mask_input_t::none)
            e.binary.mask_input = mask_input_t::policy;
    }
}

#ifdef DNNL_EXPERIMENTAL_SPARSE
int sparse_options_t::from_str(const std::string &s) {
    *this = sparse_options_t();
    if (s.empty()) return OK;

    const auto get_arg = [](int i) {
        switch (i) {
            case 0: return DNNL_ARG_SRC;
            case 1: return DNNL_ARG_WEIGHTS;
            case 2: return DNNL_ARG_DST;
            default: return -1;
        }
    };

    int options_count = 0;
    size_t start_pos = 0;
    while (start_pos != std::string::npos) {
        auto subs = parser::get_substr(s, start_pos, ':');

        if (subs.empty()) {
            add(get_arg(options_count), sparse_options_t::def_encoding,
                    sparse_options_t::def_sparsity);
            options_count++;
            continue;
        }

        if (subs.find("+") == std::string::npos) {
            add(get_arg(options_count), str2sparse_encoding(subs.c_str()),
                    sparse_options_t::def_sparsity);
        } else {
            size_t subs_pos = 0;
            auto encoding_str = parser::get_substr(subs, subs_pos, '+');
            auto sparsity_str = parser::get_substr(subs, subs_pos, '+');
            if (encoding_str.empty() || sparsity_str.empty()) { return FAIL; }
            add(get_arg(options_count),
                    str2sparse_encoding(encoding_str.c_str()),
                    atof(sparsity_str.c_str()));
        }
        options_count++;
    }
    static const int expected_num_options = 3;
    return options_count == expected_num_options ? OK : FAIL;
}
#endif
