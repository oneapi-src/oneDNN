/*******************************************************************************
* Copyright 2022-2023 Intel Corporation
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
#include "gpu/ocl/bnorm/bnorm_lookup_table.hpp"

#include <string>
#include <vector>
#include <unordered_map>

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {
namespace bn_lookup_table {

using namespace compute;

// Gets bnorm parameters from BN_PARAMS env value
// Only used during tuning procedure, BN_TUNING env var must be set
void maybe_override_bn_conf_params_env(params_t &conf) {
    auto s_params = getenv_str("BN_PARAMS", "");
    assert(!s_params.empty());
    assert(conf.bn_tuning);
    conf.override_set(s_params, /*is_env*/ true);
}

// Gets bnorm parameters from a lookup table
// BN_TUNING env var must be unset or zero;
void maybe_override_bn_conf_params_table(params_t &conf, engine_t *engine) {
    assert(!conf.bn_tuning);
    auto *compute_engine = utils::downcast<compute::compute_engine_t *>(engine);
    auto gpu_arch = compute_engine->device_info()->gpu_arch();
    static bnorm_lookup_table_t table;
    auto *s_params = table.find(conf, gpu_arch);
    if (s_params) {
        conf.override_set(s_params, /*is_env*/ false);
        conf.found_in_table = true;
    }
}

void maybe_override_bn_conf_params(params_t &conf, engine_t *engine) {
    // Environment var BN_TUNING turns ON/OFF tuning mode
    conf.bn_tuning = getenv_int("BN_TUNING", 0);
    if (conf.bn_tuning) {
        maybe_override_bn_conf_params_env(conf);
    } else {
        // TODO: extend to 1pass
        if (!conf.use_stats_one_pass) {
            maybe_override_bn_conf_params_table(conf, engine);
        }
    }
}

gpu_arch_t to_hw(const std::string &s) {

#define CASE(name) \
    if (s == #name) return gpu_arch_t::name;
    CASE(xe_hp)
    CASE(xe_hpg)
    CASE(xe_hpc)
#undef CASE
    assert(!"Not expected");
    return gpu_arch_t::unknown;
}

int_filter_t::int_filter_t(const std::string &s) {
    cmp_op_ = op_kind_t::_eq;
    if (s.empty()) {
        value_ = 0;
        return;
    }
    auto end = s.size();
    auto last = s[end - 1];
    if (last == '+') {
        cmp_op_ = op_kind_t::_ge;
        end--;
    }
    value_ = std::stoi(s.substr(0, end));
}

bool int_filter_t::matches(int value) const {
    switch (cmp_op_) {
        case op_kind_t::_eq: return value == value_;
        case op_kind_t::_le: return value <= value_;
        case op_kind_t::_ge: return value >= value_;
        case op_kind_t::_lt: return value < value_;
        case op_kind_t::_gt: return value > value_;
        default: assert(!"Not expected");
    }
    return false;
}

type_filter_t::type_filter_t(const std::string &s) {
    for (auto &p : all_patterns()) {
        if (s == p) {
            pattern_ = p;
            break;
        }
    }
}

bool type_filter_t::matches(const data_type_t &dt) const {
    if (pattern_.empty())
        return (dt == data_type::f32);
    else if (pattern_ == "s8")
        return (dt == data_type::s8);
    else if (pattern_ == "f32")
        return (dt == data_type::f32);
    else if (pattern_ == "f16")
        return (dt == data_type::f16);
    else if (pattern_ == "bf16")
        return (dt == data_type::bf16);
    else
        assert(!"Not expected");
    return true;
}

std::vector<std::string> &type_filter_t::all_patterns() {
    static std::vector<std::string> ret = {
            "s8",
            "bf16",
            "f16",
            "f32",
    };
    return ret;
}

static std::vector<std::pair<char, normalization_flags_t>> all_patterns = {
        {'G', normalization_flags::use_global_stats},
        {'C', normalization_flags::use_scale},
        {'H', normalization_flags::use_shift},
        {'R', normalization_flags::fuse_norm_relu},
        {'A', normalization_flags::fuse_norm_add_relu},
};

normalization_flags_t flags_from_string(const std::string &s) {
    int ret = 0;
    for (const auto &pattern : all_patterns) {
        if (s.find(pattern.first) != std::string::npos) {
            ret |= pattern.second;
        }
    }
    return normalization_flags_t(ret);
}

std::string string_from_flags(normalization_flags_t flags) {
    std::string ret;
    for (const auto &pattern : all_patterns) {
        if (flags & pattern.second) { ret += pattern.first; }
    }
    return ret;
}

bnorm_problem_filter_t::bnorm_problem_filter_t(const std::string &s) {
    auto parts = split(s, " ");
    for (auto &part : parts) {
        auto sub_parts = split(part, "=");
        assert(sub_parts.size() == 2);
        auto &name = sub_parts[0];
        auto &value = sub_parts[1];
        if (name == "hw") {
            hw_ = to_hw(value);
        } else if (name == "dt") {
            type_filter_ = type_filter_t(value);
        } else if (name == "dir") {
            dir_ = value;
        } else if (name == "tag") {
            tag_ = value;
        } else if (name == "flags") {
            flags_filter_ = flags_from_string(value);
        } else if (name == "desc") {
            desc_ = value;
        } else {
            assert(!"Not expected");
        }
    }
}

bool bnorm_problem_filter_t::matches(
        const params_t &conf, const gpu_arch_t &gpu_arch) const {
    if (gpu_arch != hw_) return false;
    if (!matches_dir(conf)) return false;
    if (!matches_tag(conf)) return false;
    if (!type_filter_.matches({conf.data_type})) return false;
    if (flags_filter_ != conf.flags) return false;
    if (!matches_desc(conf)) return false;
    return true;
}

bool bnorm_problem_filter_t::matches_dir(const params_t &conf) const {
    // --dir={FWD_D [default], FWD_I, BWD_D, BWD_DW}
    if (dir_.empty()) return conf.is_forward;
    if (dir_ == "FWD_D" || dir_ == "FWD_I" || dir_ == "fwd_d"
            || dir_ == "fwd_i") {
        return conf.is_forward;
    } else if (dir_ == "BWD_D" || dir_ == "BWD_DW" || dir_ == "bwd_d"
            || dir_ == "bwd_dw") {
        return conf.is_backward;
    } else {
        assert(!"Not expected");
    }
    return false;
}

bool bnorm_problem_filter_t::matches_tag(const params_t &conf) const {
    // --tag={nchw [default], ...}
    bool default_tag = !(conf.is_nhwc || conf.is_blocked_16c
            || conf.is_blocked_16n16c || conf.is_blocked_32n16c);
    if (tag_.empty()) return default_tag;
    if (tag_ == "ab" || tag_ == "acb" || tag_ == "acdb" || tag_ == "acdeb"
            || tag_ == "axb") {
        return conf.is_nhwc;
    } else if (tag_ == "aBcd16b" || tag_ == "ABcd16a16b"
            || tag_ == "ABcde16a16b") {
        return conf.is_blocked_16n16c;
    } else if (tag_ == "ABcd32a16b" || tag_ == "ABcde32a16b") {
        return conf.is_blocked_32n16c;
    } else {
        assert(!"Not expected");
    }
    return false;
}

bool bnorm_problem_filter_t::matches_desc(const params_t &conf) const {
    return get_desc_str(conf) == desc_;
}

// Lookup table is a result of tuning procedure which can be implemented by
// some script that runs some given testcase with many different values of
// tunable parameters and then parses the best results.
// Env varibles BN_TUNING and BN_PARAMS must be set.
// BN_PARAMS syntax is {key=val,...}, for example
// BN_PARAMS="far=0 mv=4 sspb=14 uspb=4 uspu=4"
bnorm_lookup_table_t::bnorm_lookup_table_t() {
    // clang-format off
// resnet-50.tr.fp32.pt.mb16_pvc
add("hw=xe_hpc dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb16ic1024ih14iw14", "far=0 mv=4 sspb=14 uspb=4 uspu=4");
add("hw=xe_hpc dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb16ic128ih28iw28", "far=0 mv=-1 sspb=28 uspb=4 uspu=4");
add("hw=xe_hpc dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb16ic128ih56iw56", "far=1 mv=-1 sspb=-1 uspb=4 uspu=4");
add("hw=xe_hpc dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb16ic2048ih7iw7", "far=0 mv=4 sspb=7 uspb=4 uspu=-1");
add("hw=xe_hpc dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb16ic256ih14iw14", "far=0 mv=4 sspb=14 uspb=4 uspu=2");
add("hw=xe_hpc dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb16ic256ih28iw28", "far=0 mv=-1 sspb=28 uspb=4 uspu=2");
add("hw=xe_hpc dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb16ic256ih56iw56", "far=1 mv=-1 sspb=-1 uspb=4 uspu=4");
add("hw=xe_hpc dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb16ic512ih14iw14", "far=0 mv=-1 sspb=14 uspb=4 uspu=2");
add("hw=xe_hpc dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb16ic512ih28iw28", "far=0 mv=-1 sspb=49 uspb=4 uspu=4");
add("hw=xe_hpc dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb16ic512ih7iw7", "far=0 mv=4 sspb=7 uspb=4 uspu=4");
add("hw=xe_hpc dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb16ic64ih112iw112", "far=1 mv=-1 sspb=-1 uspb=4 uspu=4");
add("hw=xe_hpc dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb16ic64ih56iw56", "far=1 mv=-1 sspb=14 uspb=-1 uspu=2");
add("hw=xe_hpc dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb16ic1024ih14iw14", "far=0 mv=4 sspb=14 uspb=16 uspu=2");
add("hw=xe_hpc dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb16ic128ih28iw28", "far=0 mv=-1 sspb=16 uspb=4 uspu=4");
add("hw=xe_hpc dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb16ic128ih56iw56", "far=1 mv=-1 sspb=-1 uspb=4 uspu=4");
add("hw=xe_hpc dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb16ic2048ih7iw7", "far=0 mv=-1 sspb=7 uspb=16 uspu=4");
add("hw=xe_hpc dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb16ic256ih14iw14", "far=0 mv=-1 sspb=14 uspb=4 uspu=4");
add("hw=xe_hpc dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb16ic256ih28iw28", "far=0 mv=-1 sspb=28 uspb=4 uspu=4");
add("hw=xe_hpc dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb16ic256ih56iw56", "far=1 mv=-1 sspb=28 uspb=4 uspu=4");
add("hw=xe_hpc dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb16ic512ih14iw14", "far=0 mv=-1 sspb=14 uspb=16 uspu=4");
add("hw=xe_hpc dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb16ic512ih28iw28", "far=0 mv=-1 sspb=-1 uspb=4 uspu=4");
add("hw=xe_hpc dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb16ic512ih7iw7", "far=0 mv=-1 sspb=7 uspb=4 uspu=4");
add("hw=xe_hpc dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb16ic64ih112iw112", "far=1 mv=4 sspb=49 uspb=4 uspu=4");
add("hw=xe_hpc dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb16ic64ih56iw56", "far=1 mv=4 sspb=14 uspb=4 uspu=4");
// resnet-50.tr.bf16.pt.mb256_pvc
add("hw=xe_hpc dir=BWD_DW dt=bf16 tag=acdb flags=CH desc=mb256ic1024ih14iw14", "far=1 mv=-1 sspb=98 uspb=8 uspu=2");
add("hw=xe_hpc dir=BWD_DW dt=bf16 tag=acdb flags=CH desc=mb256ic128ih28iw28", "far=1 mv=-1 sspb=49 uspb=-1 uspu=-1");
add("hw=xe_hpc dir=BWD_DW dt=bf16 tag=acdb flags=CH desc=mb256ic128ih56iw56", "far=1 mv=4 sspb=-1 uspb=4 uspu=2");
add("hw=xe_hpc dir=BWD_DW dt=bf16 tag=acdb flags=CH desc=mb256ic2048ih7iw7", "far=0 mv=-1 sspb=-1 uspb=-1 uspu=-1");
add("hw=xe_hpc dir=BWD_DW dt=bf16 tag=acdb flags=CH desc=mb256ic256ih14iw14", "far=1 mv=-1 sspb=-1 uspb=-1 uspu=-1");
add("hw=xe_hpc dir=BWD_DW dt=bf16 tag=acdb flags=CH desc=mb256ic256ih28iw28", "far=1 mv=4 sspb=-1 uspb=16 uspu=4");
add("hw=xe_hpc dir=BWD_DW dt=bf16 tag=acdb flags=CH desc=mb256ic256ih56iw56", "far=1 mv=-1 sspb=392 uspb=4 uspu=4");
add("hw=xe_hpc dir=BWD_DW dt=bf16 tag=acdb flags=CH desc=mb256ic512ih14iw14", "far=1 mv=-1 sspb=-1 uspb=-1 uspu=-1");
add("hw=xe_hpc dir=BWD_DW dt=bf16 tag=acdb flags=CH desc=mb256ic512ih28iw28", "far=1 mv=-1 sspb=196 uspb=8 uspu=-1");
add("hw=xe_hpc dir=BWD_DW dt=bf16 tag=acdb flags=CH desc=mb256ic512ih7iw7", "far=0 mv=-1 sspb=49 uspb=16 uspu=2");
add("hw=xe_hpc dir=BWD_DW dt=bf16 tag=acdb flags=CH desc=mb256ic64ih112iw112", "far=1 mv=-1 sspb=-1 uspb=8 uspu=4");
add("hw=xe_hpc dir=BWD_DW dt=bf16 tag=acdb flags=CH desc=mb256ic64ih56iw56", "far=1 mv=-1 sspb=-1 uspb=16 uspu=4");
add("hw=xe_hpc dir=FWD_D dt=bf16 tag=acdb flags=CH desc=mb256ic1024ih14iw14", "far=0 mv=-1 sspb=98 uspb=4 uspu=4");
add("hw=xe_hpc dir=FWD_D dt=bf16 tag=acdb flags=CH desc=mb256ic128ih28iw28", "far=1 mv=-1 sspb=-1 uspb=8 uspu=4");
add("hw=xe_hpc dir=FWD_D dt=bf16 tag=acdb flags=CH desc=mb256ic128ih56iw56", "far=1 mv=-1 sspb=-1 uspb=4 uspu=4");
add("hw=xe_hpc dir=FWD_D dt=bf16 tag=acdb flags=CH desc=mb256ic2048ih7iw7", "far=0 mv=4 sspb=-1 uspb=8 uspu=4");
add("hw=xe_hpc dir=FWD_D dt=bf16 tag=acdb flags=CH desc=mb256ic256ih14iw14", "far=0 mv=-1 sspb=98 uspb=4 uspu=4");
add("hw=xe_hpc dir=FWD_D dt=bf16 tag=acdb flags=CH desc=mb256ic256ih28iw28", "far=1 mv=-1 sspb=-1 uspb=4 uspu=4");
add("hw=xe_hpc dir=FWD_D dt=bf16 tag=acdb flags=CH desc=mb256ic256ih56iw56", "far=1 mv=-1 sspb=392 uspb=4 uspu=4");
add("hw=xe_hpc dir=FWD_D dt=bf16 tag=acdb flags=CH desc=mb256ic512ih14iw14", "far=0 mv=-1 sspb=-1 uspb=4 uspu=4");
add("hw=xe_hpc dir=FWD_D dt=bf16 tag=acdb flags=CH desc=mb256ic512ih28iw28", "far=1 mv=-1 sspb=196 uspb=8 uspu=4");
add("hw=xe_hpc dir=FWD_D dt=bf16 tag=acdb flags=CH desc=mb256ic512ih7iw7", "far=0 mv=-1 sspb=49 uspb=4 uspu=4");
add("hw=xe_hpc dir=FWD_D dt=bf16 tag=acdb flags=CH desc=mb256ic64ih112iw112", "far=1 mv=4 sspb=-1 uspb=4 uspu=4");
add("hw=xe_hpc dir=FWD_D dt=bf16 tag=acdb flags=CH desc=mb256ic64ih56iw56", "far=1 mv=2 sspb=196 uspb=-1 uspu=4");
    // clang-format on
}

std::string get_desc_str(const params_t &conf) {
    std::ostringstream oss;
    oss << "mb" << conf.mb;
    oss << "ic" << conf.ic;
    oss << "ih" << conf.ih;
    oss << "iw" << conf.iw;
    return oss.str();
}

const char *bnorm_lookup_table_t::find(
        const params_t &conf, const gpu_arch_t &gpu_arch) const {
    auto key = get_desc_str(conf);
    auto it = map_.find(key);
    if (it == map_.end()) return nullptr;
    for (auto &e : it->second) {
        if (e.filter.matches(conf, gpu_arch)) { return e.s_params; }
    }
    return nullptr;
}

void bnorm_lookup_table_t::add(const char *s_prb, const char *s_params) {
    bnorm_problem_filter_t filter(s_prb);
    map_[filter.key()].push_back(entry_t {filter, s_params});
}

int params_t::sort_key(const param_t *param) const {
    static const char *ordered_params[] = {
            "far",
            "icb",
            "mv",
            "sspb",
            "uspb",
            "uspu",
            nullptr,
    };
    for (const char **p = ordered_params; *p; p++) {
        if (param->short_name() == *p) return p - ordered_params;
    }
    return (int)(sizeof(ordered_params) / sizeof(ordered_params[0]));
}

std::string params_t::str() const {
    std::ostringstream oss;
    oss << "Fused atomic reduction: " << use_fused_atomics_reduction_.str()
        << std::endl;
    oss << "IC block: " << ic_block_.str() << std::endl;
    oss << "Max vector size: " << max_vect_size_.str() << std::endl;
    oss << "Stat spatial blocking: " << stat_sp_block_.str() << std::endl;
    oss << "Update spatial blocking: " << update_sp_block_.str() << std::endl;
    oss << "Update spatial unrolling: " << update_sp_unroll_.str() << std::endl;
    return oss.str();
}

} // namespace bn_lookup_table
} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
