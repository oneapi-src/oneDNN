/*******************************************************************************
* Copyright 2019-2024 Intel Corporation
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

#include <algorithm>
#include <cctype>

#include "utils/cold_cache.hpp"
#include "utils/parser.hpp"
#include "utils/stream_kind.hpp"

#include "dnnl_common.hpp"

namespace parser {

bool last_parsed_is_problem = false;
const size_t eol = std::string::npos;
std::stringstream help_ss;

static const std::string benchdnn_url
        = "https://github.com/oneapi-src/oneDNN/blob/master/tests/benchdnn";
static const std::string doc_url = benchdnn_url + "/doc/";

namespace parser_utils {
std::string get_pattern(const std::string &option_name, bool with_args) {
    std::string s = std::string("--") + option_name;
    if (with_args) s += "=";
    return s;
}

void add_option_to_help(const std::string &option,
        const std::string &help_message, bool with_args) {
    static std::vector<std::string> help_added;
    for (const auto &e : help_added)
        if (e == option) return;

    std::string option_str = get_pattern(option, with_args);
    help_ss << option_str << help_message << "\n";
    help_added.push_back(option);
}

// Covers all integer parsing routines: `atoi`, `atol, `atoll`, `stoi`, `stol`.
int64_t stoll_safe(const std::string &s) {
    int64_t value = 0;
    try {
        value = std::stoll(s);
        if (std::to_string(value).size() != s.size()) {
            BENCHDNN_PRINT(0, "%s \'%s\'\n",
                    "Error: Parsed value is expected to be an integer number, "
                    "not",
                    s.c_str());
            SAFE_V(FAIL);
        }
    } catch (const std::invalid_argument &) {
        BENCHDNN_PRINT(0, "%s \'%s\'\n",
                "Error: Parsed value is expected to be an integer number, not",
                s.c_str());
        SAFE_V(FAIL);
    }
    return value;
}

// Covers all float parsing routines: `atof`.
float stof_safe(const std::string &s) {
    float value = 0;
    try {
        value = std::stof(s);
        // Can't compare input with output same way. The only way is to check
        // if input has `e`, `f` and digits. Seems an overkill, let function
        // decide on parsed output.
    } catch (const std::invalid_argument &) {
        BENCHDNN_PRINT(0, "%s \'%s\'\n",
                "Error: Parsed value is expected to be an floating-point "
                "number, not",
                s.c_str());
        SAFE_V(FAIL);
    }
    return value;
}

attr_t::post_ops_t parse_attr_post_ops_func(const std::string &s) {
    attr_t::post_ops_t v;
    if (s.empty()) return v;

    size_t start_pos = 0;
    while (start_pos != std::string::npos) {
        auto subs = get_substr(s, start_pos, '+');
        size_t subs_pos = 0;

        auto kind
                = attr_t::post_ops_t::str2kind(get_substr(subs, subs_pos, ':'));
        if (kind == attr_t::post_ops_t::kind_t::KIND_TOTAL) SAFE_V(FAIL);

#define CATCH_DANGLING_SYMBOL \
    if (subs_pos >= subs.size()) { \
        BENCHDNN_PRINT(0, "%s \'%s\'\n", \
                "Error: dangling symbol at the end of input", subs.c_str()); \
        SAFE_V(FAIL); \
    }

        v.entry.emplace_back(kind);
        if (subs_pos == std::string::npos) {
            if (kind != attr_t::post_ops_t::kind_t::DW) continue;

            BENCHDNN_PRINT(0, "%s\n",
                    "Error: depthwise post-op entry didn't recognize 'k', 's', "
                    "and 'p' values.");
            SAFE_V(FAIL);
        }
        CATCH_DANGLING_SYMBOL;

        auto &e = v.entry.back();
        if (e.is_sum_kind()) {
            e.sum.scale
                    = parser_utils::stof_safe(get_substr(subs, subs_pos, ':'));
            if (subs_pos == std::string::npos) continue;
            CATCH_DANGLING_SYMBOL;

            auto zp_str = get_substr(subs, subs_pos, ':');
            e.sum.zero_point = parser_utils::stoll_safe(zp_str);
            if (subs_pos == std::string::npos) continue;
            CATCH_DANGLING_SYMBOL;

            const auto dt_str = get_substr(subs, subs_pos, ':');
            e.sum.dt = str2dt(dt_str.c_str());
            // sum dt, if specified, should be defined
            if (e.sum.dt == dnnl_data_type_undef) {
                BENCHDNN_PRINT(0, "%s \'%s\' %s\n",
                        "Error: sum post-op data type", dt_str.c_str(),
                        "is not recognized.");
                SAFE_V(FAIL);
            }
        } else if (e.is_convolution_kind()) {
            // `DW` has input of `dw:kXsYpZ`.
            const auto str_dw_params = get_substr(subs, subs_pos, ':');
            size_t pos = 0, idx = 0;

            pos += idx;
            if (str_dw_params[pos] != 'k') {
                BENCHDNN_PRINT(0, "%s \'%s\' %s\n",
                        "Error: depthwise post-op entry", &str_dw_params[pos],
                        "is not 'k'.");
                SAFE_V(FAIL);
            }
            // TODO: some safe handling would help here
            e.convolution.kernel = std::stoi(&str_dw_params[++pos], &idx);
            if (e.convolution.kernel <= 0) {
                BENCHDNN_PRINT(0, "%s\n",
                        "Error: depthwise post-op kernel must be greater "
                        "than 0.");
                SAFE_V(FAIL);
            }

            pos += idx;
            if (str_dw_params[pos] != 's') {
                BENCHDNN_PRINT(0, "%s \'%s\' %s\n",
                        "Error: depthwise post-op entry", &str_dw_params[pos],
                        "is not 's'.");
                SAFE_V(FAIL);
            }
            e.convolution.stride = std::stoi(&str_dw_params[++pos], &idx);
            if (e.convolution.stride <= 0) {
                BENCHDNN_PRINT(0, "%s\n",
                        "Error: depthwise post-op stride must be greater "
                        "than 0.");
                SAFE_V(FAIL);
            }

            pos += idx;
            if (str_dw_params[pos] != 'p') {
                BENCHDNN_PRINT(0, "%s \'%s\' %s\n",
                        "Error: depthwise post-op entry", &str_dw_params[pos],
                        "is not 'p'.");
                SAFE_V(FAIL);
            }
            e.convolution.padding = std::stoi(&str_dw_params[++pos]);

            if (subs_pos == std::string::npos) continue;

            const auto dt_str = get_substr(subs, subs_pos, ':');
            e.convolution.dst_dt = str2dt(dt_str.c_str());
            if (e.convolution.dst_dt == dnnl_data_type_undef) {
                BENCHDNN_PRINT(0, "%s \'%s\' %s\n",
                        "Error: depthwise post-op data type", dt_str.c_str(),
                        "is not recognized.");
                SAFE_V(FAIL);
            }
            if (subs_pos < std::string::npos) {
                const auto leftover = get_substr(subs, subs_pos, '\0');
                BENCHDNN_PRINT(0,
                        "Error: no more inputs are expected. Provided: "
                        "\'%s\'.\n",
                        leftover.c_str());
                SAFE_V(FAIL);
            }
        } else if (e.is_eltwise_kind()) {
            e.eltwise.alpha
                    = parser_utils::stof_safe(get_substr(subs, subs_pos, ':'));
            if (subs_pos == std::string::npos) continue;
            CATCH_DANGLING_SYMBOL;

            e.eltwise.beta
                    = parser_utils::stof_safe(get_substr(subs, subs_pos, ':'));
            if (subs_pos == std::string::npos) continue;
        } else if (e.is_binary_kind()) {
            const auto dt_str = get_substr(subs, subs_pos, ':');
            e.binary.src1_dt = str2dt(dt_str.c_str());
            if (e.binary.src1_dt == dnnl_data_type_undef) {
                BENCHDNN_PRINT(0, "%s \'%s\' %s\n",
                        "Error: binary post-op data type", dt_str.c_str(),
                        "is not recognized.");
                SAFE_V(FAIL);
            }
            if (subs_pos == std::string::npos) continue;
            CATCH_DANGLING_SYMBOL;

            const auto mask_input_str = get_substr(subs, subs_pos, ':');
            // Check if `mask_input_str` consists of only digits.
            const bool only_digits = std::all_of(
                    mask_input_str.cbegin(), mask_input_str.cend(), [](int c) {
                        assert(c < UINT8_MAX);
                        return std::isdigit(c);
                    });

            using mask_input_t
                    = attr_t::post_ops_t::entry_t::binary_t::mask_input_t;
            if (only_digits) {
                // If digits only, then read it as integer value.
                e.binary.mask = parser_utils::stoll_safe(mask_input_str);
                e.binary.mask_input = mask_input_t::mask;
            } else {
                // Otherwise, re-direct to policy parsing.
                e.binary.policy = attr_t::str2policy(mask_input_str);
                if (e.binary.policy == attr_t::policy_t::POLICY_TOTAL) {
                    BENCHDNN_PRINT(0, "%s \'%s\' %s\n",
                            "Error: binary post-op policy",
                            mask_input_str.c_str(),
                            "is not recognized. Input also is not consisted of "
                            "only integers to process it as mask directly.");
                    SAFE_V(FAIL);
                }
                e.binary.mask_input = mask_input_t::policy;
            }
            if (subs_pos == std::string::npos) continue;
            CATCH_DANGLING_SYMBOL;

            const auto tag_str = get_substr(subs, subs_pos, ':');
            e.binary.tag = tag_str;
            if (check_tag(e.binary.tag) != OK) {
                BENCHDNN_PRINT(0, "%s \'%s\' %s\n", "Error: binary post-op tag",
                        tag_str.c_str(), "is not recognized.");
                SAFE_V(FAIL);
            }
        } else if (e.is_prelu_kind()) {
            const auto policy_str = get_substr(subs, subs_pos, ':');
            e.prelu.policy = attr_t::str2policy(policy_str);
            if (e.prelu.policy == attr_t::policy_t::POLICY_TOTAL) {
                BENCHDNN_PRINT(0, "%s \'%s\' %s\n",
                        "Error: prelu post-op policy", policy_str.c_str(),
                        "is not recognized.");
                SAFE_V(FAIL);
            }
        }
        if (subs_pos == std::string::npos) continue;
        CATCH_DANGLING_SYMBOL;
    }

    return v;
}

attr_t::deterministic_t parse_attr_deterministic_func(const std::string &s) {
    attr_t::deterministic_t v;
    if (s.empty()) return v;

    v.enabled = str2bool(s.c_str());
    return v;
}

attr_t::fpmath_mode_t parse_attr_fpmath_mode_func(const std::string &s) {
    attr_t::fpmath_mode_t v;
    if (s.empty()) return v;

    size_t start_pos = 0;
    auto subs = get_substr(s, start_pos, ':');
    v.mode = str2fpmath_mode(subs.c_str());
    if (start_pos == std::string::npos) return v;
    if (start_pos >= s.size()) {
        BENCHDNN_PRINT(0, "%s \'%s\'\n",
                "Error: dangling symbol at the end of input", s.c_str());
        SAFE_V(FAIL);
    }

    if (start_pos != std::string::npos) {
        subs = get_substr(s, start_pos, '\0');
        v.apply_to_int = str2bool(subs.c_str());

        if (start_pos != std::string::npos) {
            BENCHDNN_PRINT(0, "%s \'%s\'\n",
                    "Error: dangling symbol at the end of input", s.c_str());
            SAFE_V(FAIL);
        }
    }

    return v;
}

attr_t::rounding_mode_t parse_attr_rounding_mode_func(const std::string &s) {
    attr_t::rounding_mode_t rm;

    if (s.empty()) return rm;

    size_t start_pos = 0;
    while (start_pos != std::string::npos) {
        auto subs = parser::get_substr(s, start_pos, '+');
        size_t subs_pos = 0;

        auto arg = str2arg(parser::get_substr(subs, subs_pos, ':'));
        if (arg == DNNL_ARG_UNDEF) {
            BENCHDNN_PRINT(0, "%s\n", "Error: undefined argument index");
            SAFE_V(FAIL);
        }
        if (subs_pos != std::string::npos)
            rm.set(arg,
                    str2rounding_mode(parser::get_substr(subs, subs_pos, ':')));
        if (subs_pos != std::string::npos)
            rm.set_seed(stoll_safe(get_substr(subs, subs_pos, ':')));
    }
    return rm;
}

attr_t::dropout_t parse_attr_dropout_func(const std::string &s) {
    const char *err = "Error: dangling symbol at the end of input";
    attr_t::dropout_t v;
    if (s.empty()) return v;

    size_t start_pos = 0;
    auto subs = get_substr(s, start_pos, ':');
    v.p = stof_safe(subs);
    if ((v.p < 0.f) || (v.p > 1.f)) {
        BENCHDNN_PRINT(0, "Error: bad dropout probability value: %f\n", v.p);
        SAFE_V(FAIL);
    }
    if (start_pos == std::string::npos) return v;
    if (start_pos >= s.size()) {
        BENCHDNN_PRINT(0, "%s \'%s\'\n", err, s.c_str());
        SAFE_V(FAIL);
    }

    if (start_pos != std::string::npos) {
        subs = get_substr(s, start_pos, ':');
        v.seed = stoll_safe(subs);

        if (start_pos == std::string::npos) return v;
        if (start_pos >= s.size()) {
            BENCHDNN_PRINT(0, "%s \'%s\'\n", err, s.c_str());
            SAFE_V(FAIL);
        }

        if (start_pos != std::string::npos) {
            v.tag = get_substr(s, start_pos, '\0');

            if (check_tag(v.tag) != OK) {
                BENCHDNN_PRINT(0, "%s \'%s\' %s\n", "Error: dropout mask tag",
                        v.tag.c_str(), "is not recognized.");
                SAFE_V(FAIL);
            }

            if (start_pos != std::string::npos) {
                BENCHDNN_PRINT(0, "%s \'%s\'\n", err, s.c_str());
                SAFE_V(FAIL);
            }
        }
    }

    return v;
}

bool parse_impl_filter(impl_filter_t &impl_filter,
        const impl_filter_t &def_impl_filter, bool use_impl, const char *str,
        const std::string &option_name, const std::string &help) {
    const auto chars2chars = [](const char *str) { return str; };
    const auto str2impl_filter = [&](const char *str) {
        std::vector<std::string> v, def;
        parse_vector_str(v, def, chars2chars, str);

        // Remove all quotes from input string since they affect the search.
        for_(auto &e : v)
        for (auto c : {'"', '\''}) {
            size_t start_pos = 0;
            while (start_pos != eol) {
                start_pos = e.find_first_of(c, start_pos);
                if (start_pos != eol) e.erase(start_pos, 1);
            }
        }

        return impl_filter_t(v, use_impl);
    };
    return parse_single_value_option(impl_filter, def_impl_filter,
            str2impl_filter, str, option_name, help);
}

} // namespace parser_utils

// vector types
bool parse_dir(std::vector<dir_t> &dir, const std::vector<dir_t> &def_dir,
        const char *str, const std::string &option_name /* = "dir"*/) {
    static const std::string help
            = "DIR    (Default: `FWD_B` when bias applicable, `FWD_D` "
              "otherwise)\n    Specifies propagation kind `DIR` for operation. "
              "Has bias support incorporated with `_B` suffix.\n    `DIR` "
              "values can be `FWD_B`, `FWD_D`, `FWD_I`, `BWD_D`, `BWD_WB`, "
              "`BWD_W` and `BWD_DW`.\n    More details at "
            + doc_url + "knobs_dir.md\n";
    return parse_vector_option(dir, def_dir, str2dir, str, option_name, help);
}

bool parse_dt(std::vector<dnnl_data_type_t> &dt,
        const std::vector<dnnl_data_type_t> &def_dt, const char *str,
        const std::string &option_name /* = "dt"*/) {
    static const std::string help
            = "DT    (Default: `f32`)\n    Specifies data type `DT` for source "
              "and/or destination.\n    `DT` values can be `f32`, `bf16`, "
              "`f16`, `s32`, `s8`, `u8`.\n";
    return parse_vector_option(dt, def_dt, str2dt, str, option_name, help);
}

bool parse_multi_dt(std::vector<std::vector<dnnl_data_type_t>> &dt,
        const std::vector<std::vector<dnnl_data_type_t>> &def_dt,
        const char *str, const std::string &option_name /* = "sdt"*/) {
    static const std::string help
            = "DT0:DT1[:DTi]    (Default: `f32` for all)\n    When the driver "
              "supports the notion of multiple sources, the option specifies a "
              "data type `DTi` for a source `i`.\n    When the driver supports "
              "the notion of source, weights (optional), and destination, the "
              "option specifies data types for source, weights (optional) and "
              "destination correspondently.\n    The option may support "
              "broadcast semantics (check the driver online documentation), "
              "when a single value will be used for all inputs.\n    `DT` "
              "values can be `f32`, `bf16`, `f16`, `s32`, `s8`, `u8`.\n";
    return parse_multivector_option(dt, def_dt, str2dt, str, option_name, help);
}

bool parse_tag(std::vector<std::string> &tag,
        const std::vector<std::string> &def_tag, const char *str,
        const std::string &option_name /* = "tag"*/) {
    static const std::string help
            = "TAG    (Default: `any` for compute-bound, `abx` for rest)\n    "
              "Specifies memory format tag `TAG` for source, weights, or "
              "destination.\n    Valid `TAG` values can be found at "
            + doc_url + "knobs_tag.md\n";

    auto ret_string = [](const char *str) { return std::string(str); };
    bool ok = parse_vector_option(
            tag, def_tag, ret_string, str, option_name, help);
    if (!ok) return false;

    for (size_t i = 0; i < tag.size(); i++) {
        if (check_tag(tag[i], allow_enum_tags_only) != OK) {
            if (allow_enum_tags_only && check_tag(tag[i]) == OK) {
                fprintf(stderr,
                        "ERROR: tag `%s` is valid but not found in "
                        "`dnnl::memory::format_tag`. To force the testing with "
                        "this tag, please specify `--allow-enum-tags-only=0` "
                        "prior to any tag option.\n",
                        tag[i].c_str());
            } else {
                fprintf(stderr,
                        "ERROR: unknown or invalid tag: `%s`, exiting...\n",
                        tag[i].c_str());
            }
            exit(2);
        }
    }
    return true;
}

bool parse_encoding(std::vector<sparse_options_t> &sparse_options,
        const char *str, const std::string &option_name /* = "encoding"*/) {
    static const std::string help
            = "ENCODING[+SPARSITY]:ENCODING[+SPARSITY]:ENCODING[+SPARSITY]\n   "
              "Specifies sparse encodings and sparsity.\n    More details at "
              "https://github.com/oneapi-src/oneDNN/blob/master/tests/benchdnn/"
              "doc/knobs_encoding.md\n";

    std::vector<sparse_options_t> def {sparse_options_t()};
    auto parse_sparse_options_func = [](const std::string &s) {
        sparse_options_t v;
        SAFE_V(v.from_str(s));
        return v;
    };

    return parse_vector_option(sparse_options, def, parse_sparse_options_func,
            str, option_name, help);
}

bool parse_multi_tag(std::vector<std::vector<std::string>> &tag,
        const std::vector<std::vector<std::string>> &def_tag, const char *str,
        const std::string &option_name /* = "stag"*/) {
    static const std::string help
            = "TAG0:TAG1[:TAGi]    (Default: `any` for compute-bound, `abx` "
              "for rest)\n    Specifies memory format tag `TAGi` for source "
              "i.\n    Valid `TAGi` values can be found at "
            + doc_url + "knobs_tag.md\n";
    auto ret_string = [](const char *str) { return std::string(str); };
    return parse_multivector_option(
            tag, def_tag, ret_string, str, option_name, help);
}

bool parse_mb(std::vector<int64_t> &mb, const std::vector<int64_t> &def_mb,
        const char *str, const std::string &option_name /* = "mb"*/) {
    static const std::string help
            = "UINT    (Default: `0`)\n    Overrides mini-batch value "
              "specified in a problem descriptor with `UINT` value.\n    When "
              "set to `0`, takes no effect.\n";
    return parse_vector_option(
            mb, def_mb, parser_utils::stoll_safe, str, option_name, help);
}

bool parse_attr_post_ops(std::vector<attr_t::post_ops_t> &po, const char *str,
        const std::string &option_name = "attr-post-ops") {
    static const std::string help
            = "POST-OPS\n    Specifies post-ops attribute. `POST-OPS` syntax "
              "is one of those:\n    * SUM[:SCALE[:ZERO_POINT[:DATA_TYPE]]]\n  "
              "  * ELTWISE[:ALPHA[:BETA[:SCALE]]]\n    * DW:KkSsPp[:DST_DT]\n  "
              "  * BINARY:DT[:MASK_INPUT[:TAG]]\n    More details at "
              "https://github.com/oneapi-src/oneDNN/blob/master/tests/benchdnn/"
              "doc/knobs_attr.md\n";
    std::vector<attr_t::post_ops_t> def {attr_t::post_ops_t()};
    return parse_vector_option(po, def, parser_utils::parse_attr_post_ops_func,
            str, option_name, help);
}

bool parse_attr_scales(std::vector<attr_t::arg_scales_t> &scales,
        const char *str, const std::string &option_name = "attr-scales") {
    static const std::string help
            = "ARG:POLICY[:SCALE][+...]\n    Specifies input scales "
              "attribute.\n    More details at "
              "https://github.com/oneapi-src/oneDNN/blob/master/tests/benchdnn/"
              "doc/knobs_attr.md\n";
    return parse_subattr(scales, str, option_name, help);
}

bool parse_attr_zero_points(std::vector<attr_t::zero_points_t> &zp,
        const char *str, const std::string &option_name = "attr-zero-points") {
    static const std::string help
            = "ARG:POLICY[:ZEROPOINT][+...]\n    Specifies zero-points "
              "attribute.\n    More details at "
              "https://github.com/oneapi-src/oneDNN/blob/master/tests/benchdnn/"
              "doc/knobs_attr.md\n";
    return parse_subattr(zp, str, option_name, help);
}

bool parse_attr_rounding_mode(std::vector<attr_t::rounding_mode_t> &rm,
        const char *str,
        const std::string &option_name = "attr-rounding-mode") {
    static const std::string help
            = "ARG:MODE[:SEED][+...]    (Default: `environment`)\n    "
              "Specifies a "
              "rounding mode MODE to be applied upon conversion of argument "
              "ARG.\n    More details at "
            + doc_url + "knobs_attr.md\n";
    return parse_vector_option(rm, {},
            parser_utils::parse_attr_rounding_mode_func, str, option_name,
            help);
}

bool parse_attr_scratchpad_mode(
        std::vector<dnnl_scratchpad_mode_t> &scratchpad_mode,
        const std::vector<dnnl_scratchpad_mode_t> &def_scratchpad_mode,
        const char *str, const std::string &option_name = "attr-scratchpad") {
    static const std::string help
            = "MODE    (Default: `library`)\n    Specifies scratchpad "
              "attribute. `MODE` values can be `library` or `user`.\n    More "
              "details at "
            + doc_url + "knobs_attr.md\n";
    return parse_vector_option(scratchpad_mode, def_scratchpad_mode,
            str2scratchpad_mode, str, option_name, help);
}

bool parse_attr_fpmath_mode(std::vector<attr_t::fpmath_mode_t> &fpmath_mode,
        const std::vector<attr_t::fpmath_mode_t> &def_fpmath_mode,
        const char *str, const std::string &option_name = "attr-fpmath") {
    static const std::string help
            = "MODE[:APPLY_TO_INT]    (Default: `strict[:false]`)\n    "
              "Specifies "
              "fpmath_mode attribute. `MODE` values can be `strict` or "
              "`bf16`. `APPLY_TO_INT` values can be `true` or `false`.\n";
    return parse_vector_option(fpmath_mode, def_fpmath_mode,
            parser_utils::parse_attr_fpmath_mode_func, str, option_name, help);
}

bool parse_attr_dropout(std::vector<attr_t::dropout_t> &dropout,
        const std::vector<attr_t::dropout_t> &def_dropout, const char *str,
        const std::string &option_name = "attr-dropout") {
    static const std::string help
            = "PROBABILITY[:SEED[:TAG]]\n    Specifies dropout attribute.\n    "
              "More details at "
            + doc_url + "knobs_attr.md\n";
    return parse_vector_option(dropout, def_dropout,
            parser_utils::parse_attr_dropout_func, str, option_name, help);
}

bool parse_attr_acc_mode(std::vector<dnnl_accumulation_mode_t> &acc_mode,
        const std::vector<dnnl_accumulation_mode_t> &def_acc_mode,
        const char *str, const std::string &option_name = "attr-acc-mode") {
    static const std::string help
            = "MODE    (Default: `strict`)\n    Specifies accumulation mode "
              "attribute. `MODE` values can be `strict`, `relaxed`, `any`,"
              "`f32`, `f16` or `s32`.\n";
    return parse_vector_option(acc_mode, def_acc_mode, str2accumulation_mode,
            str, option_name, help);
}

bool parse_attr_deterministic(
        std::vector<attr_t::deterministic_t> &deterministic,
        const std::vector<attr_t::deterministic_t> &def_deterministic,
        const char *str,
        const std::string &option_name = "attr-deterministic") {
    static const std::string help
            = "MODE    (Default: `false`)\n    Specifies deterministic mode "
              "attribute. `MODE` values can be `true`, or `false`.\n";
    return parse_vector_option(deterministic, def_deterministic,
            parser_utils::parse_attr_deterministic_func, str, option_name,
            help);
}

bool parse_attributes(
        base_settings_t &s, const base_settings_t &def, const char *str) {
    const bool parsed_attrs = parse_attr_scales(s.scales, str)
            || parse_attr_zero_points(s.zero_points, str)
            || parse_attr_post_ops(s.post_ops, str)
            || parse_attr_dropout(s.dropout, def.dropout, str)
            || parse_attr_scratchpad_mode(
                    s.scratchpad_mode, def.scratchpad_mode, str)
            || parse_attr_fpmath_mode(s.fpmath_mode, def.fpmath_mode, str)
            || parse_attr_acc_mode(s.acc_mode, def.acc_mode, str)
            || parse_attr_deterministic(s.deterministic, def.deterministic, str)
            || parse_attr_rounding_mode(s.rounding_mode, str);
    return parsed_attrs;
}

bool parse_axis(std::vector<int> &axis, const std::vector<int> &def_axis,
        const char *str, const std::string &option_name /* = "axis"*/) {
    static const std::string help
            = "UINT    (Default: `1`)\n    Specifies axis dimension `UINT` for "
              "an operation.\n";
    return parse_vector_option(
            axis, def_axis, parser_utils::stoll_safe, str, option_name, help);
}

bool parse_test_pattern_match(const char *&match, const char *str,
        const std::string &option_name /* = "match"*/) {
    static const std::string help
            = "REGEX    (Default: not specified)\n    `REGEX` is a string "
              "literal representing a regular expression that filters problem "
              "descriptors.\n    Matched descriptors are executed, rest are "
              "skipped.\n";
    const char *def_match = "";
    const auto chars2chars = [](const char *str) { return str; };
    return parse_single_value_option(
            match, def_match, chars2chars, str, option_name, help);
}

bool parse_impl(impl_filter_t &impl_filter,
        const impl_filter_t &def_impl_filter, const char *str,
        const std::string &option_name) {
    static const std::string help
            = "STRINGS    (Default: not specified)\n    Instructs the driver "
              "to fetch the next implementation from the list if fetched "
              "implementation name doesn't match any from the `STRINGS` list "
              "(a.k.a. include-list).\n    `STRINGS` is a comma-separated list "
              "of string literal entries with no spaces.\n    When empty, the "
              "option has no effect. The option is opposite to "
              "`--skip-impl`.\n";

    return parser_utils::parse_impl_filter(impl_filter, def_impl_filter,
            /* use_impl = */ true, str, option_name, help);
}

bool parse_skip_impl(impl_filter_t &impl_filter,
        const impl_filter_t &def_impl_filter, const char *str,
        const std::string &option_name) {
    static const std::string help
            = "STRINGS    (Default: not specified)\n    Instructs the driver "
              "to fetch the next implementation from the list if fetched "
              "implementation name matches any from the `STRINGS` list (a.k.a. "
              "exclude-list).\n    `STRINGS` is a comma-separated list of "
              "string literal entries with no spaces.\n    When empty, the "
              "option has no effect. The option is opposite to `--impl`.\n";

    return parser_utils::parse_impl_filter(impl_filter, def_impl_filter,
            /* use_impl = */ false, str, option_name, help);
}

bool parse_inplace(std::vector<bool> &inplace,
        const std::vector<bool> &def_inplace, const char *str,
        const std::string &option_name /* = "inplace"*/) {
    static const std::string help
            = "BOOL    (Default: `false`)\n    Instructs the driver to use "
              "same memory data handle for source and destination when set to "
              "`true`.\n";
    return parse_vector_option(
            inplace, def_inplace, str2bool, str, option_name, help);
}

bool parse_skip_nonlinear(std::vector<bool> &skip,
        const std::vector<bool> &def_skip, const char *str,
        const std::string &option_name /* = "skip-nonlinear"*/) {
    static const std::string help
            = "BOOL    (Default: `false`)\n    Instructs the driver to treat "
              "transcendental activations as linear when set to `true`.\n";
    return parse_vector_option(
            skip, def_skip, str2bool, str, option_name, help);
}

bool parse_strides(std::vector<vdims_t> &strides,
        const std::vector<vdims_t> &def_strides, const char *str,
        const std::string &option_name /* = "strides"*/) {
    static const std::string help
            = "DIMS_SRC[:DIMS_WEI]:DIMS_DST    (Default: not specified)\n    "
              "Specifies strides `DIMS_ARG` for correspondent supported "
              "`ARG`.\n    If correspondent `DIMS_ARG` is empty, it does not "
              "take an effect.\n    More details at "
            + doc_url + "driver_" + driver_name + ".md\n";
    auto str2strides = [&](const char *str) -> vdims_t {
        vdims_t strides(STRIDES_SIZE);
        parse_multivector_str(
                strides, vdims_t(), parser_utils::stoll_safe, str, ':', 'x');
        return strides;
    };
    return parse_vector_option(
            strides, def_strides, str2strides, str, option_name, help);
}

bool parse_trivial_strides(std::vector<bool> &ts,
        const std::vector<bool> &def_ts, const char *str,
        const std::string &option_name /* = "trivial-strides"*/) {
    static const std::string help
            = "BOOL    (Default: `false`)\n    Instructs the driver to use "
              "dense (trivial) strides when set to `true`.\n";
    return parse_vector_option(ts, def_ts, str2bool, str, option_name, help);
}

bool parse_scale_policy(std::vector<policy_t> &policy,
        const std::vector<policy_t> &def_policy, const char *str,
        const std::string &option_name /*= "scaling"*/) {
    static const std::string help
            = "POLICY    (Default: `common`)\n    Specifies a mask for scales "
              "to be applied.\n    More details at "
            + doc_url + "knobs_attr.md\n";
    return parse_vector_option(
            policy, def_policy, attr_t::str2policy, str, option_name, help);
}

// plain types
bool parse_perf_template(const char *&pt, const char *pt_def,
        const char *pt_csv, const char *str,
        const std::string &option_name /* = "perf-template"*/) {
    static const std::string help
            = "TEMPLATE    (Default: `def`)\n    Specifies performance output "
              "template for perf mode. `TEMPLATE` values can be `def`, `csv` "
              "or customized set.\n    More details at "
            + doc_url + "knobs_perf_report.md\n";
    const auto str2pt = [&pt_def, &pt_csv](const char *str_) {
        const std::string csv_pattern = "csv";
        const std::string def_pattern = "def";
        if (csv_pattern.find(str_, 0, csv_pattern.size()) != eol)
            return pt_csv;
        else if (def_pattern.find(str_, 0, def_pattern.size()) != eol)
            return pt_def;
        else
            return str_;
    };
    return parse_single_value_option(
            pt, pt_def, str2pt, str, option_name, help);
}

bool parse_batch(const bench_f bench, const char *str,
        const std::string &option_name /* = "batch"*/) {
    static const std::string help
            = "FILE\n    Instructs the driver to take options and problem "
              "descriptors from a `FILE`.\n";
    int status = OK;
    const auto str2batch = [bench](const char *str_) {
        SAFE(batch(str_, bench), CRIT);
        return OK;
    };
    return parse_single_value_option(
            status, FAIL, str2batch, str, option_name, help);
}

bool parse_help(const char *str, const std::string &option_name /* = "help"*/) {
    std::string pattern = parser_utils::get_pattern(option_name, false);
    if (pattern.find(str, 0, pattern.size()) == eol) return false;

    BENCHDNN_PRINT(0, "%s\n", help_ss.str().c_str());
    exit(0);
}

bool parse_main_help(
        const char *str, const std::string &option_name /* = "help"*/) {
    std::string pattern = parser_utils::get_pattern(option_name, false);
    if (pattern.find(str, 0, pattern.size()) == eol) return false;

    static const std::string main_help
            = "Usage:\n    benchdnn --<driver> [global_options] "
              "[driver_options] problem_description\n\nList of supported "
              "<drivers> (lower case accepted only):\n    * binary\n    * "
              "bnorm\n    * concat\n    * conv\n    * deconv\n    * eltwise\n  "
              "  * ip\n    * lnorm\n    * lrn\n    * matmul\n    * pool\n    * "
              "prelu\n    * reduction\n    * reorder\n    * resampling\n    * "
              "rnn\n    * shuffle\n    * softmax\n    * sum\n    * "
              "zeropad\n\nFor global and specific driver options, use:\n    "
              "benchdnn --<driver> --help\n\nMore details at "
            + benchdnn_url + "\n";

    BENCHDNN_PRINT(0, "%s\n", main_help.c_str());
    exit(0);
}

// prb_dims_t type
void parse_prb_vdims(
        prb_vdims_t &prb_vdims, const std::string &str, size_t min_inputs) {
    assert(!str.empty());

    size_t start_pos = 0;
    // `n` is an indicator for a name supplied with dims_t object.
    std::string vdims_str = get_substr(str, start_pos, 'n');
    // Potential trailing underscore before `n` shouldn't be parsed as dims.
    if (vdims_str.back() == '_') vdims_str.pop_back();

    // Sanity check that dims start with a digit.
    if (!std::isdigit(vdims_str[0])) {
        BENCHDNN_PRINT(0, "%s\n%s \'%s\'\n",
                "ERROR: dims are expected to start with an integer value.",
                "Given input:", str.c_str());
        SAFE_V(FAIL);
    }

    std::string name;
    if (start_pos != eol) name = str.substr(start_pos);

    vdims_t vdims;
    parse_multivector_str(vdims, {dims_t()}, parser_utils::stoll_safe,
            vdims_str, ':', 'x', /* allow_empty = */ false);
    // Expect at least two inputs provided
    SAFE_V(vdims.size() >= min_inputs ? OK : FAIL);

    prb_vdims = prb_vdims_t(vdims, name);
}

void parse_prb_dims(prb_dims_t &prb_dims, const std::string &str) {
    size_t start_pos = 0;
    // `n` is an indicator for a name supplied with dims_t object.
    std::string dims_str = get_substr(str, start_pos, 'n');
    // Potential trailing underscore before `n` shouldn't be parsed as dims.
    if (dims_str.back() == '_') dims_str.pop_back();

    parse_vector_str(
            prb_dims.dims, dims_t(), parser_utils::stoll_safe, dims_str, 'x');

    prb_dims.ndims = static_cast<int>(prb_dims.dims.size());

    if (start_pos != eol) prb_dims.name = str.substr(start_pos);
}

// Global options
static bool parse_allow_enum_tags_only(const char *str,
        const std::string &option_name = "allow-enum-tags-only") {
    static const std::string help
            = "BOOL    (Default: `true`)\n    Instructs the driver to validate "
              "format tags against the documented tags from "
              "`dnnl_format_tag_t` enumeration only.\n    When set to `true`, "
              "the only allowed format tags are the ones from "
              "`dnnl_format_tag_t` enumeration.\n";
    return parse_single_value_option(
            allow_enum_tags_only, true, str2bool, str, option_name, help);
}

static bool parse_attr_same_pd_check(const char *str,
        const std::string &option_name = "attr-same-pd-check") {
    static const std::string help
            = "BOOL    (Default: `false`)\n    Instructs the driver to compare "
              "two primitive descriptors - one with requested attributes and "
              "one without them.\n    When set to `true`, check would return "
              "an error if attributes caused fallback to a different "
              "implementation.\n";
    return parse_single_value_option(
            attr_same_pd_check, false, str2bool, str, option_name, help);
}

static bool parse_canonical(
        const char *str, const std::string &option_name = "canonical") {
    static const std::string help
            = "BOOL    (Default: `false`)\n    Instructs the driver to print a "
              "canonical form of a reproducer line.\n    When set to `true`, "
              "the driver prints all options and their values, including "
              "default ones.\n";
    return parse_single_value_option(
            canonical, false, str2bool, str, option_name, help);
}

static bool parse_check_ref_impl(
        const char *str, const std::string &option_name = "check-ref-impl") {
    static const std::string help
            = "BOOL    (Default: `false`)\n    Instructs the driver to compare "
              "an implementation name against the \'ref\' string pattern.\n    "
              "When set to `true`, the check would return an error if the "
              "implementation name contains such pattern.\n";
    return parse_single_value_option(
            check_ref_impl, false, str2bool, str, option_name, help);
}

static bool parse_cold_cache(
        const char *str, const std::string &option_name = "cold-cache") {
    static const std::string help
            = "MODE    (Default: `none`)\n    Instructs the driver to enable a "
              "cold cache for performance mode.\n    When set to `none` (the "
              "default), cold cache is disabled.\n    When set to `wei`, cold "
              "cache is enabled for weights argument only. Targets forward "
              "propagation kind.\n    When set to `all`, cold cache is enabled "
              "for each execution argument.\n    When set to `custom`, cold "
              "cache is enabled for custom arguments which should be specified "
              "directly in the code. Refer to doc for more details.\n";

    const auto str2cold_cache_mode = [](const std::string &_str) {
        cold_cache_mode_t cc_mode = default_cold_cache_mode;
        if (_str == "none") {
            cc_mode = cold_cache_mode_t::none;
        } else if (_str == "wei") {
            cc_mode = cold_cache_mode_t::wei;
        } else if (_str == "all") {
            cc_mode = cold_cache_mode_t::all;
        } else if (_str == "custom") {
            cc_mode = cold_cache_mode_t::custom;
        } else {
            BENCHDNN_PRINT(0, "%s \'%s\'\n%s", "Error: unknown cold cache mode",
                    _str.c_str(), help.c_str());
            SAFE_V(FAIL);
        }
        return cc_mode;
    };

    return parse_single_value_option(cold_cache_mode, cold_cache_mode_t::none,
            str2cold_cache_mode, str, option_name, help);
}

static bool parse_cpu_isa_hints(
        const char *str, const std::string &option_name = "cpu-isa-hints") {
    static const std::string help
            = "HINTS    (Default: `none`)\n    Specifies the ISA specific "
              "hints for CPU engine.\n    `HINTS` values can be `none`, "
              "`no_hints` or `prefer_ymm`.\n";
    const bool parsed
            = parse_single_value_option(hints, isa_hints_t {isa_hints_t::none},
                    isa_hints_t::str2hints, str, option_name, help);
    if (parsed) init_isa_settings();
    return parsed;
}

static bool parse_engine(
        const char *str, const std::string &option_name = "engine") {
    static const std::string help
            = "KIND[:INDEX]    (Default: `cpu`)\n    Instructs the driver to "
              "use an engine with requested `KIND`.\n    `KIND` values can be "
              "`cpu` or `gpu`.\n    `INDEX` is an integer value specifying "
              "which engine to use if several were identified.\n";
    if (!parse_single_value_option(engine_tgt_kind, dnnl_cpu, str2engine_kind,
                str, option_name, help))
        return false;
    // Parse engine index if present
    std::string s(str);
    auto start_pos = s.find_first_of(':');
    if (start_pos != eol) {
        // Let the library catch issues with incorrect engine indices.
        engine_index = std::stoi(s.substr(start_pos + 1));
    }

    return true;
}

static bool parse_fast_ref(
        const char *str, const std::string &option_name = "fast-ref") {
    static const std::string help
            = "BOOL    (Default: `true`)\n    Instructs the driver to use "
              "faster reference path when doing correctness testing for "
              "`--engine=gpu`.\n    When set to `true`, the library best fit "
              "CPU implementation is used to compute the reference path.\n";
    bool parsed = parse_single_value_option(
            fast_ref, default_fast_ref, str2bool, str, option_name, help);
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_NONE
    if (parsed && fast_ref) {
        fast_ref = false;
        fprintf(stderr,
                "%s driver: WARNING: option `fast_ref` is not supported "
                "for GPU only configurations.\n",
                driver_name.c_str());
    }
#endif
    return parsed;
}

static bool parse_global_impl(
        const char *str, const std::string &option_name = "global-impl") {
    static const std::string help
            = "STRINGS    (Default: not specified)\n    Same as `--impl` but "
              "overrides any values from `--impl` or `--skip-impl` options met "
              "on the way\n.";

    return parser_utils::parse_impl_filter(global_impl_filter, impl_filter_t(),
            /* use_impl = */ true, str, option_name, help);
}

static bool parse_global_skip_impl(
        const char *str, const std::string &option_name = "global-skip-impl") {
    static const std::string help
            = "STRINGS    (Default: not specified)\n    Same as `--skip-impl` "
              "but overrides any values from `--impl` or `--skip-impl` options "
              "met on the way\n.";

    return parser_utils::parse_impl_filter(global_impl_filter, impl_filter_t(),
            /* use_impl = */ false, str, option_name, help);
}

// TODO: remove
static bool parse_fast_ref_gpu(
        const char *str, const std::string &option_name = "fast-ref-gpu") {
    bool parsed = parse_single_value_option(fast_ref, default_fast_ref,
            str2bool, str, option_name, std::string());

    static bool msg_printed = false;
    if (parsed && !msg_printed) {
        BENCHDNN_PRINT(0, "%s\n",
                "Warning: \'--fast-ref-gpu\' is deprecated. Use \'--fast-ref\' "
                "instead.");
        msg_printed = true;
    }
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_NONE
    if (parsed && fast_ref) {
        fast_ref = false;
        fprintf(stderr,
                "%s driver: WARNING: option `fast_ref` is not supported "
                "for GPU only configurations.\n",
                driver_name.c_str());
    }
#endif
    return parsed;
}

bool parse_ctx(std::vector<thr_ctx_t> &ctx,
        const std::vector<thr_ctx_t> &def_ctx, const char *str,
        const std::string &option_name) {
    const std::string name_in_help
            = (option_name == "ctx-init") ? "initialization." : "execution.";
    const std::string help
            = std::string(
                      "MAX_CONCURENCY[:CORE_TYPE[:THREADS_PER_CORE]] "
                      "(Default:`auto:auto:auto`)\n    Specifies the threading "
                      "context used during primitive ")
            + name_in_help
            + std::string(
                    "\n    MAX_CONCURRENCY is the maximum number of threads.\n "
                    "   CORE_TYPE enables to select big (value 0) or small "
                    "cores (value 1) for hybrid CPUs (TBB runtime only).\n    "
                    "THREADS_PER_CORE allows to enable/disable hyper-threading "
                    "(TBB runtime only).\n");

    auto str2ctx = [&option_name](const char *str) {
        thr_ctx_t result = default_thr_ctx;
        try {
            size_t start_pos = 0;
            /* concurrency piece */
            std::string val_str = get_substr(str, start_pos, ':');
            if (val_str != "auto") result.max_concurrency = std::stoll(val_str);
            /* core_type piece */
            val_str = start_pos != eol ? get_substr(str, start_pos, ':') : "";
            if (val_str != "auto" && !val_str.empty())
                result.core_type = std::stoll(val_str);
            /* nthr_per_core piece */
            val_str = start_pos != eol ? get_substr(str, start_pos, ':') : "";
            if (val_str != "auto" && !val_str.empty())
                result.nthr_per_core = std::stoll(val_str);
        } catch (const std::invalid_argument &) {
            BENCHDNN_PRINT(0, "%s %s\n", option_name.c_str(),
                    "fields should be 'auto' or integer values");
            exit(1);
        }

        return result;
    };

    return parse_vector_option(ctx, def_ctx, str2ctx, str, option_name, help);
}

bool parse_ctx_init(std::vector<thr_ctx_t> &ctx,
        const std::vector<thr_ctx_t> &def_ctx, const char *str) {
    return parse_ctx(ctx, def_ctx, str, "ctx-init");
}
bool parse_ctx_exe(std::vector<thr_ctx_t> &ctx,
        const std::vector<thr_ctx_t> &def_ctx, const char *str) {
    return parse_ctx(ctx, def_ctx, str, "ctx-exe");
}

static bool parse_fix_times_per_prb(
        const char *str, const std::string &option_name = "fix-times-per-prb") {
    static const std::string help
            = "UINT    (Default: `0`)\n    Specifies the limit in `UINT` "
              "rounds for performance benchmarking per problem.\n    If `UINT` "
              "is greater than `0`, the number of rounds criterion takes place "
              "over the time criterion.\n";
    bool parsed = parse_single_value_option(fix_times_per_prb,
            default_fix_times_per_prb, parser_utils::stoll_safe, str,
            option_name, help);
    if (parsed) fix_times_per_prb = MAX2(0, fix_times_per_prb);
    return parsed;
}

static bool parse_max_ms_per_prb(
        const char *str, const std::string &option_name = "max-ms-per-prb") {
    static const std::string help
            = "MS    (Default: `3000`)\n    Specifies the limit in `MS` "
              "milliseconds for performance benchmarking per problem.\n    "
              "`MS` is a positive integer in a range [10, 60000].\n";
    bool parsed
            = parse_single_value_option(max_ms_per_prb, default_max_ms_per_prb,
                    parser_utils::stof_safe, str, option_name, help);
    if (parsed) {
        if (bench_mode == bench_mode_t::perf_fast) {
            BENCHDNN_PRINT(0, "%s\n",
                    "Error: mode=F can't be adjusted. Please use full command "
                    "mode=F aliases with custom max-ms-per-prb input.");
            exit(2);
        }

        max_ms_per_prb = MAX2(10, MIN2(max_ms_per_prb, 60e3));
    }
    return parsed;
}

static bool parse_num_streams(
        const char *str, const std::string &option_name = "num-streams") {
    static const std::string help
            = "N    (Default: `1`)\n    Specifies the number `N` of streams "
              "used for performance benchmarking.\n    `N` is a positive "
              "integer.\n";
    bool parsed = parse_single_value_option(num_streams, default_num_streams,
            parser_utils::stoll_safe, str, option_name, help);
    if (parsed) {
        if (num_streams <= 0) {
            BENCHDNN_PRINT(
                    0, "%s\n", "Error: number of streams must be positive.");
            SAFE_V(FAIL);
        }
    }
    return parsed;
}

static bool parse_repeats_per_prb(
        const char *str, const std::string &option_name = "repeats-per-prb") {
    static const std::string help
            = "N    (Default: `1`)\n    Specifies the number of times to "
              "repeat testing of the problem.\n";
    bool parsed = parse_single_value_option(repeats_per_prb,
            default_repeats_per_prb, parser_utils::stoll_safe, str, option_name,
            help);
    if (parsed) repeats_per_prb = MAX2(1, repeats_per_prb);
    return parsed;
}

static bool parse_mem_check(
        const char *str, const std::string &option_name = "mem-check") {
    static const std::string help
            = "BOOL    (Default: `true`)\n    Instructs the driver to perform "
              "a device RAM capability check if a problem fits a device, when "
              "set to `true`.\n";
    return parse_single_value_option(
            mem_check, true, str2bool, str, option_name, help);
}

static bool parse_memory_kind(
        const char *str, const std::string &option_name = "memory-kind") {
    static const std::string help
            = "KIND    (Default: `usm`)\n    Specifies a memory `KIND` to test "
              "with DPC++ and OpenCL engines.\n    `KIND` values are `usm`, "
              "`buffer`, `usm_device` (malloc_device) or `usm_shared` "
              "(malloc_shared).\n";
    bool parsed = parse_single_value_option(memory_kind, default_memory_kind,
            str2memory_kind, str, option_name, help);

#if !defined(DNNL_WITH_SYCL) && DNNL_GPU_RUNTIME != DNNL_RUNTIME_OCL
    if (parsed) {
        fprintf(stderr,
                "ERROR: option `--%s` is supported with DPC++ and OpenCL "
                "builds only, exiting...\n",
                option_name.c_str());
        exit(2);
    }
#endif
    return parsed;
}

static bool parse_mode(
        const char *str, const std::string &option_name = "mode") {
    static const std::string help
            = "MODE    (Default: `C`)\n"
              "    Specifies a `MODE` for benchmarking.\n"
              "    `MODE` values are:\n"
              "    * `L` for listing mode.\n"
              "    * `I` for initialization mode.\n"
              "    * `R` for execution mode (no correctness validation).\n"
              "    * `C` for correctness testing.\n"
              "    * `P` for performance testing.\n"
              "    * `F` for fast performance testing (GPU only).\n"
              "    * `B` for bitwise (numerical determinism) testing.\n"
              "    * `CP` for both correctness and performance testing.\n"
              "    More details at "
            + doc_url + "benchdnn_general_info.md\n";

    const auto str2bench_mode = [](const std::string &_str) {
        bench_mode_t mode = default_bench_mode;
        if (_str.size() > 2) {
            BENCHDNN_PRINT(
                    0, "%s\n%s", "Error: mode value is invalid.", help.c_str());
            SAFE_V(FAIL);
        } else if (_str.size() == 2) {
            for (size_t i = 0; i < _str.size(); i++) {
                switch (_str[i]) {
                    case 'c':
                    case 'C':
                    case 'p':
                    case 'P': break;
                    default:
                        BENCHDNN_PRINT(0, "%s\n%s",
                                "Error: mode value is invalid.", help.c_str());
                        SAFE_V(FAIL);
                }
            }
            mode = bench_mode_t::corr_perf;
        } else if (_str.size() == 1) {
            switch (_str[0]) {
                case 'l':
                case 'L': mode = bench_mode_t::list; break;
                case 'i':
                case 'I': mode = bench_mode_t::init; break;
                case 'r':
                case 'R':
                    mode = bench_mode_t::exec;
                    bench_mode_modifier |= mode_modifier_t::no_ref_memory;
                    break;
                case 'c':
                case 'C': mode = bench_mode_t::corr; break;
                case 'p':
                case 'P': mode = bench_mode_t::perf; break;
                case 'f':
                case 'F':
                    mode = bench_mode_t::perf_fast;
                    max_ms_per_prb = 10;
                    bench_mode_modifier |= mode_modifier_t::par_create
                            | mode_modifier_t::no_ref_memory;
                    break;
                case 'b':
                case 'B': mode = bench_mode_t::bitwise; break;
                default:
                    BENCHDNN_PRINT(0, "%s\n%s", "Error: mode value is invalid.",
                            help.c_str());
                    SAFE_V(FAIL);
            }
        }

        return mode;
    };

    return parse_single_value_option(bench_mode, default_bench_mode,
            str2bench_mode, str, option_name, help);
}

static bool parse_mode_modifier(
        const char *str, const std::string &option_name = "mode-modifier") {
    static const std::string help
            = "MODIFIER    (Default: empty)\n"
              "    Specifies a `MODIFIER` for selected benchmarking mode.\n"
              "    `MODIFIER` values are:\n"
              "    * `P` to enable parallel test objects creation.\n"
              "          The flow will create as many primitives in parallel \n"
              "          as number of threads identified on the system \n"
              "          first, then execute them one by one.\n"
              "    * `M` to disable usage of reference memory.\n"
              "          It removes any overheads for mapping, unmapping and \n"
              "          reorders used in filling functions (disabled).\n"
              "    More details at "
            + doc_url + "benchdnn_general_info.md\n";

    const auto str2mode_modifier = [](const std::string &_str) {
        mode_modifier_t modifier = bench_mode_modifier;
        for (auto s : _str) {
            switch (s) {
                case 'p':
                case 'P': modifier |= mode_modifier_t::par_create; break;
                case 'm':
                case 'M': modifier |= mode_modifier_t::no_ref_memory; break;
                default:
                    BENCHDNN_PRINT(0, "%s\n%s",
                            "Error: modifier value is invalid.", help.c_str());
                    exit(2);
            }
        }

        return modifier;
    };

    return parse_single_value_option(bench_mode_modifier,
            default_bench_mode_modifier, str2mode_modifier, str, option_name,
            help);
}

static bool parse_start(
        const char *str, const std::string &option_name = "start") {
    static const std::string help
            = "UINT    (Default: `0`)\n    Specifies the test case index "
              "`UINT` to start execution. All test cases up to `UINT` will be "
              "skipped.\n";
    return parse_single_value_option(
            test_start, 0, parser_utils::stoll_safe, str, option_name, help);
}

static bool parse_stream_kind(
        const char *str, const std::string &option_name = "stream-kind") {
    static const std::string help
            = "KIND    (Default: `def`)\n    Specifies a stream `KIND` to test "
              "with DPC++ and OpenCL engines through stream flags.\n    "
              "`KIND` values are `def` (the default flags), `in_order`, or "
              "`out_of_order`.\n";
    bool parsed = parse_single_value_option(stream_kind, default_stream_kind,
            str2stream_kind, str, option_name, help);

#if !defined(DNNL_WITH_SYCL) && DNNL_GPU_RUNTIME != DNNL_RUNTIME_OCL
    if (parsed) {
        BENCHDNN_PRINT(0,
                "Error: option `--%s` is supported with DPC++ and OpenCL "
                "builds only, exiting...\n",
                option_name.c_str());
        SAFE_V(FAIL);
    }
#endif
    return parsed;
}

static bool parse_verbose(
        const char *str, const std::string &option_name = "verbose") {
    static const std::string help
            = "UINT, -vUINT    (Default: `0`)\n    Instructs the driver to "
              "print additional information depending on `UINT`.\n    More "
              "details at "
            + doc_url + "knobs_verbose.md\n";
    bool parsed = parse_single_value_option(
            verbose, 0, parser_utils::stoll_safe, str, option_name, help);
    if (parsed) return parsed;

    const std::string pattern("-v"); // check short option first
    if (pattern.find(str, 0, pattern.size()) != eol) {
        verbose = parser_utils::stoll_safe(str + pattern.size());
        return true;
    }
    return false;
}

bool parse_bench_settings(const char *str) {
    last_parsed_is_problem = false; // if start parsing, expect an option

    static bool start_msg = false, end_msg = false;
    if (!start_msg) {
        help_ss << "===================\n";
        help_ss << "= Global options: =\n";
        help_ss << "===================\n";
        help_ss << "(More technical details available at "
                   "https://github.com/oneapi-src/oneDNN/blob/master/tests/"
                   "benchdnn/doc/knobs_common.md)\n\n";
        start_msg = true;
    }

    bool parsed = parse_allow_enum_tags_only(str)
            || parse_attr_same_pd_check(str) || parse_canonical(str)
            || parse_check_ref_impl(str) || parse_cold_cache(str)
            || parse_cpu_isa_hints(str) || parse_engine(str)
            || parse_fast_ref(str) || parse_fast_ref_gpu(str)
            || parse_fix_times_per_prb(str) || parse_global_impl(str)
            || parse_global_skip_impl(str) || parse_max_ms_per_prb(str)
            || parse_num_streams(str) || parse_repeats_per_prb(str)
            || parse_mem_check(str) || parse_memory_kind(str) || parse_mode(str)
            || parse_mode_modifier(str) || parse_start(str)
            || parse_stream_kind(str) || parse_verbose(str);

    // Last condition makes this help message to be triggered once driver_name
    // is already known.
    if (!parsed && !end_msg && !driver_name.empty()) {
        help_ss << "===================\n";
        help_ss << "= Driver options: =\n";
        help_ss << "===================\n";
        help_ss << "(More technical details available at "
                   "https://github.com/oneapi-src/oneDNN/blob/master/tests/"
                   "benchdnn/doc/driver_"
                << driver_name << ".md)\n\n";
        end_msg = true;
    }
    return parsed;
}

// Service functions
void catch_unknown_options(const char *str) {
    last_parsed_is_problem = true; // if reached, means problem parsing

    std::string pattern = "--";
    if (pattern.find(str, 0, pattern.size()) != eol) {
        BENCHDNN_PRINT(0, "%s %s \'%s\'\n",
                "driver: ERROR: unknown option:", driver_name.c_str(), str);
        exit(2);
    }

    // Must stay after `--` check.
    pattern = "-";
    if (pattern.find(str, 0, pattern.size()) != eol) {
        BENCHDNN_PRINT(0, "%s\n%s \'%s\'\n",
                "ERROR: options should be passed with `--` prefix.",
                "Given input:", str);
        exit(2);
    }
}

int parse_last_argument() {
    if (!last_parsed_is_problem)
        fprintf(stderr,
                "%s driver: WARNING: No problem found for a given option!\n",
                driver_name.c_str());
    return OK;
}

std::string get_substr(const std::string &s, size_t &start_pos, char delim) {
    auto end_pos = s.find_first_of(delim, start_pos);
    auto sub = s.substr(start_pos, end_pos - start_pos);
    start_pos = end_pos + (end_pos != eol);
    return sub;
}

} // namespace parser
