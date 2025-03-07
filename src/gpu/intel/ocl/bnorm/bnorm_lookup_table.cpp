/*******************************************************************************
* Copyright 2022-2025 Intel Corporation
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
#include "gpu/intel/ocl/bnorm/bnorm_lookup_table.hpp"
#include "gpu/intel/compute/compute_engine.hpp"

#include <string>
#include <vector>
#include <unordered_map>

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
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
void maybe_override_bn_conf_params_table(
        params_t &conf, impl::engine_t *engine) {
    assert(!conf.bn_tuning);
    auto *compute_engine = utils::downcast<compute::compute_engine_t *>(engine);
    auto gpu_arch = compute_engine->device_info()->gpu_arch();
    static bnorm_lookup_table_t table(conf.use_stats_one_pass);
    auto *s_params = table.find(conf, gpu_arch);
    if (s_params) {
        conf.override_set(s_params, /*is_env*/ false);
        conf.found_in_table = true;
    }
}

void maybe_override_bn_conf_params(params_t &conf, impl::engine_t *engine) {
    // Environment var BN_TUNING turns ON/OFF tuning mode
    conf.bn_tuning = getenv_int("BN_TUNING", 0);

    if (conf.bn_tuning) {
        maybe_override_bn_conf_params_env(conf);
    } else {
        // for performance debugging/analysis purposes
        if (getenv_int("BN_ENABLE_LOOKUP_TABLE", 1))
            maybe_override_bn_conf_params_table(conf, engine);
    }
}

gpu_arch_t to_hw(const std::string &s) {

#define CASE(name) \
    if (s == #name) return gpu_arch_t::name;
    CASE(xe_hp)
    CASE(xe_hpg)
    CASE(xe_hpc)
#undef CASE
    gpu_error_not_expected();
    return gpu_arch_t::unknown;
}

bn_impl_t to_impl(const std::string &s) {
#define CASE(name) \
    if (s == #name) return bn_impl_t::name;
    CASE(ref)
    CASE(simple)
    CASE(reusable)
    CASE(gen9)
    CASE(nhwc_opt)
    CASE(nhwc_reusable)
#undef CASE
    gpu_error_not_expected();
    return bn_impl_t::unknown;
}

int_filter_t::int_filter_t(const std::string &s) : cmp_op_(op_kind_t::_eq) {
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
        default: gpu_error_not_expected();
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
        gpu_error_not_expected();
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

static bool is_nhwc_impl(const params_t &conf) {
    return conf.impl == bn_impl_t::nhwc_reusable
            || conf.impl == bn_impl_t::nhwc_opt;
}

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
        } else if (name == "impl") {
            impl_ = to_impl(value);
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
        } else if (name == "nhwc_desc") {
            nhwc_desc_ = value;
        } else {
            gpu_error_not_expected();
        }
    }
}

bool bnorm_problem_filter_t::matches(
        const params_t &conf, const gpu_arch_t &gpu_arch) const {
    if (gpu_arch != hw_) return false;
    if (!matches_impl(conf)) return false;
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
        gpu_error_not_expected();
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
        gpu_error_not_expected();
    }
    return false;
}

bool bnorm_problem_filter_t::matches_impl(const params_t &conf) const {
    return conf.impl == impl_;
}

bool bnorm_problem_filter_t::matches_desc(const params_t &conf) const {
    return is_nhwc_impl(conf) ? get_nhwc_desc_str(conf) == nhwc_desc_
                              : get_desc_str(conf) == desc_;
}

// Lookup table is a result of tuning procedure which can be implemented by
// some script that runs some given testcase with many different values of
// tunable parameters and then parses the best results.
// Env varibles BN_TUNING and BN_PARAMS must be set.
// BN_PARAMS syntax is {key=val,...}, for example
// BN_PARAMS="far=0 mv=4 sspb=14 uspb=4 uspu=4"
bnorm_lookup_table_t::bnorm_lookup_table_t(bool use_stat_one_pass) {
    if (use_stat_one_pass) {
        // clang-format off
        // nhwc_opt version
        // resnet-50.tr.fp32.pt.mb16_pvc
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb16ic1024ih14iw14 nhwc_desc=ic1024sp3136", "far=0 mv=8 icb=32 sspb=14 uspb=8 uspu=4"); //, 0.028000
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb16ic128ih28iw28 nhwc_desc=ic128sp12544", "far=0 mv=8 icb=32 sspb=49 uspb=8 uspu=4"); //, 0.020320
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb16ic128ih56iw56 nhwc_desc=ic128sp50176", "far=1 mv=8 icb=64 sspb=28 uspb=4 uspu=4"); //, 0.048000
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb16ic2048ih7iw7 nhwc_desc=ic2048sp784", "far=0 mv=8 icb=32 sspb=14 uspb=8 uspu=2"); //, 0.019680
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb16ic256ih14iw14 nhwc_desc=ic256sp3136", "far=0 mv=8 icb=64 sspb=14 uspb=4 uspu=4"); //, 0.016960
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb16ic256ih28iw28 nhwc_desc=ic256sp12544", "far=0 mv=8 icb=64 sspb=49 uspb=4 uspu=4"); //, 0.029600
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb16ic256ih56iw56 nhwc_desc=ic256sp50176", "far=0 mv=8 icb=32 sspb=98 uspb=8 uspu=4"); //, 0.081280
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb16ic512ih14iw14 nhwc_desc=ic512sp3136", "far=0 mv=8 icb=32 sspb=7 uspb=8 uspu=4"); //, 0.020160
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb16ic512ih28iw28 nhwc_desc=ic512sp12544", "far=0 mv=8 icb=32 sspb=49 uspb=8 uspu=4"); //, 0.044960
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb16ic512ih7iw7 nhwc_desc=ic512sp784", "far=0 mv=8 icb=32 sspb=28 uspb=4 uspu=4"); //, 0.013440
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb16ic64ih112iw112 nhwc_desc=ic64sp200704", "far=1 mv=8 icb=64 sspb=49 uspb=16 uspu=4"); //, 0.082880
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb16ic64ih56iw56 nhwc_desc=ic64sp50176", "far=1 mv=8 icb=32 sspb=16 uspb=8 uspu=4"); //, 0.031360
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb16ic1024ih14iw14 nhwc_desc=ic1024sp3136", "far=0 mv=8 icb=32 sspb=28 uspb=8 uspu=4"); //, 0.019520
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb16ic128ih28iw28 nhwc_desc=ic128sp12544", "far=0 mv=8 icb=32 sspb=49 uspb=4 uspu=4"); //, 0.017440
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb16ic128ih56iw56 nhwc_desc=ic128sp50176", "far=0 mv=8 icb=32 sspb=98 uspb=16 uspu=4"); //, 0.032960
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb16ic2048ih7iw7 nhwc_desc=ic2048sp784", "far=0 mv=8 icb=64 sspb=14 uspb=16 uspu=4"); //, 0.016480
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb16ic256ih14iw14 nhwc_desc=ic256sp3136", "far=0 mv=8 icb=32 sspb=14 uspb=4 uspu=4"); //, 0.014880
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb16ic256ih28iw28 nhwc_desc=ic256sp12544", "far=0 mv=8 icb=32 sspb=28 uspb=8 uspu=4"); //, 0.021440
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb16ic256ih56iw56 nhwc_desc=ic256sp50176", "far=0 mv=8 icb=32 sspb=98 uspb=8 uspu=4"); //, 0.047680
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb16ic512ih14iw14 nhwc_desc=ic512sp3136", "far=0 mv=8 icb=32 sspb=14 uspb=-1 uspu=2"); //, 0.016000
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb16ic512ih28iw28 nhwc_desc=ic512sp12544", "far=0 mv=8 icb=32 sspb=49 uspb=8 uspu=4"); //, 0.029760
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb16ic512ih7iw7 nhwc_desc=ic512sp784", "far=0 mv=8 icb=32 sspb=28 uspb=4 uspu=4"); //, 0.011680
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb16ic64ih112iw112 nhwc_desc=ic64sp200704", "far=0 mv=8 icb=32 sspb=196 uspb=8 uspu=4"); //, 0.057760
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb16ic64ih56iw56 nhwc_desc=ic64sp50176", "far=1 mv=8 icb=32 sspb=28 uspb=16 uspu=4"); //, 0.024000
        // resnet-50.tr.bf16.pt.mb256_pvc
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=bf16 tag=acdb flags=CH desc=mb256ic1024ih14iw14 nhwc_desc=ic1024sp50176", "far=0 mv=8 icb=64 sspb=196 uspb=16 uspu=2"); //, 0.313760
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=bf16 tag=acdb flags=CH desc=mb256ic128ih28iw28 nhwc_desc=ic128sp200704", "far=1 mv=8 icb=32 sspb=196 uspb=-1 uspu=4"); //, 0.099040
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=bf16 tag=acdb flags=CH desc=mb256ic128ih56iw56 nhwc_desc=ic128sp802816", "far=1 mv=8 icb=128 sspb=196 uspb=4 uspu=4"); //, 0.949280
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=bf16 tag=acdb flags=CH desc=mb256ic2048ih7iw7 nhwc_desc=ic2048sp12544", "far=0 mv=8 icb=128 sspb=49 uspb=-1 uspu=1"); //, 0.092960
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=bf16 tag=acdb flags=CH desc=mb256ic256ih14iw14 nhwc_desc=ic256sp50176", "far=0 mv=8 icb=32 sspb=98 uspb=16 uspu=4"); //, 0.060320
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=bf16 tag=acdb flags=CH desc=mb256ic256ih28iw28 nhwc_desc=ic256sp200704", "far=0 mv=8 icb=128 sspb=392 uspb=16 uspu=2"); //, 0.317440
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=bf16 tag=acdb flags=CH desc=mb256ic256ih56iw56 nhwc_desc=ic256sp802816", "far=1 mv=8 icb=128 sspb=392 uspb=4 uspu=4"); //, 1.978720
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=bf16 tag=acdb flags=CH desc=mb256ic512ih14iw14 nhwc_desc=ic512sp50176", "far=0 mv=8 icb=32 sspb=196 uspb=-1 uspu=4"); //, 0.094720
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=bf16 tag=acdb flags=CH desc=mb256ic512ih28iw28 nhwc_desc=ic512sp200704", "far=0 mv=8 icb=64 sspb=392 uspb=4 uspu=4"); //, 0.923520
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=bf16 tag=acdb flags=CH desc=mb256ic512ih7iw7 nhwc_desc=ic512sp12544", "far=0 mv=8 icb=64 sspb=28 uspb=-1 uspu=4"); //, 0.034400
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=bf16 tag=acdb flags=CH desc=mb256ic64ih112iw112 nhwc_desc=ic64sp3211264", "far=1 mv=8 icb=256 sspb=196 uspb=4 uspu=2"); //, 2.047040
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=bf16 tag=acdb flags=CH desc=mb256ic64ih56iw56 nhwc_desc=ic64sp802816", "far=1 mv=8 icb=128 sspb=196 uspb=16 uspu=4"); //, 0.323680
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=bf16 tag=acdb flags=CH desc=mb256ic1024ih14iw14 nhwc_desc=ic1024sp50176", "far=0 mv=8 icb=32 sspb=392 uspb=-1 uspu=4"); //, 0.096640
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=bf16 tag=acdb flags=CH desc=mb256ic128ih28iw28 nhwc_desc=ic128sp200704", "far=0 mv=8 icb=32 sspb=196 uspb=-1 uspu=4"); //, 0.059040
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=bf16 tag=acdb flags=CH desc=mb256ic128ih56iw56 nhwc_desc=ic128sp802816", "far=0 mv=8 icb=64 sspb=392 uspb=8 uspu=4"); //, 0.492960
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=bf16 tag=acdb flags=CH desc=mb256ic2048ih7iw7 nhwc_desc=ic2048sp12544", "far=0 mv=8 icb=32 sspb=196 uspb=-1 uspu=4"); //, 0.053120
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=bf16 tag=acdb flags=CH desc=mb256ic256ih14iw14 nhwc_desc=ic256sp50176", "far=0 mv=8 icb=32 sspb=98 uspb=16 uspu=4"); //, 0.036480
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=bf16 tag=acdb flags=CH desc=mb256ic256ih28iw28 nhwc_desc=ic256sp200704", "far=0 mv=8 icb=32 sspb=392 uspb=-1 uspu=4"); //, 0.096320
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=bf16 tag=acdb flags=CH desc=mb256ic256ih56iw56 nhwc_desc=ic256sp802816", "far=1 mv=8 icb=128 sspb=392 uspb=4 uspu=2"); //, 1.179840
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=bf16 tag=acdb flags=CH desc=mb256ic512ih14iw14 nhwc_desc=ic512sp50176", "far=0 mv=8 icb=32 sspb=196 uspb=-1 uspu=4"); //, 0.054240
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=bf16 tag=acdb flags=CH desc=mb256ic512ih28iw28 nhwc_desc=ic512sp200704", "far=0 mv=8 icb=64 sspb=392 uspb=8 uspu=4"); //, 0.458880
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=bf16 tag=acdb flags=CH desc=mb256ic512ih7iw7 nhwc_desc=ic512sp12544", "far=0 mv=8 icb=64 sspb=28 uspb=8 uspu=4"); //, 0.023840
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=bf16 tag=acdb flags=CH desc=mb256ic64ih112iw112 nhwc_desc=ic64sp3211264", "far=1 mv=8 icb=128 sspb=196 uspb=4 uspu=4"); //, 1.209120
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=bf16 tag=acdb flags=CH desc=mb256ic64ih56iw56 nhwc_desc=ic64sp802816", "far=1 mv=8 icb=32 sspb=392 uspb=-1 uspu=4"); //, 0.106880
        // resnet-50.tr.fp32.pt.mb256_pvc_aurora
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb256ic1024ih14iw14 nhwc_desc=ic1024sp50176", "far=0 mv=8 icb=32 sspb=392 uspb=4 uspu=4"); //, 0.920960
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb256ic128ih28iw28 nhwc_desc=ic128sp200704", "far=0 mv=8 icb=64 sspb=392 uspb=4 uspu=4"); //, 0.294080
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb256ic128ih56iw56 nhwc_desc=ic128sp802816", "far=1 mv=8 icb=32 sspb=64 uspb=4 uspu=4"); //, 1.992000
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb256ic2048ih7iw7 nhwc_desc=ic2048sp12544", "far=0 mv=8 icb=32 sspb=196 uspb=8 uspu=4"); //, 0.293600
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb256ic256ih14iw14 nhwc_desc=ic256sp50176", "far=0 mv=8 icb=32 sspb=98 uspb=8 uspu=4"); //, 0.082240
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb256ic256ih28iw28 nhwc_desc=ic256sp200704", "far=0 mv=8 icb=32 sspb=392 uspb=4 uspu=2"); //, 0.919040
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb256ic256ih56iw56 nhwc_desc=ic256sp802816", "far=1 mv=8 icb=32 sspb=28 uspb=4 uspu=1"); //, 3.945280
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb256ic512ih14iw14 nhwc_desc=ic512sp50176", "far=0 mv=8 icb=32 sspb=196 uspb=16 uspu=4"); //, 0.289280
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb256ic512ih28iw28 nhwc_desc=ic512sp200704", "far=0 mv=8 icb=64 sspb=392 uspb=4 uspu=4"); //, 1.987680
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb256ic512ih7iw7 nhwc_desc=ic512sp12544", "far=0 mv=8 icb=32 sspb=49 uspb=8 uspu=4"); //, 0.045280
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb256ic64ih112iw112 nhwc_desc=ic64sp3211264", "far=1 mv=8 icb=32 sspb=64 uspb=4 uspu=1"); //, 3.997280
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb256ic64ih56iw56 nhwc_desc=ic64sp802816", "far=0 mv=8 icb=32 sspb=392 uspb=4 uspu=4"); //, 0.954560
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb256ic1024ih14iw14 nhwc_desc=ic1024sp50176", "far=0 mv=8 icb=64 sspb=196 uspb=4 uspu=2"); //, 0.458400
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb256ic128ih28iw28 nhwc_desc=ic128sp200704", "far=0 mv=8 icb=32 sspb=196 uspb=16 uspu=4"); //, 0.088800
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb256ic128ih56iw56 nhwc_desc=ic128sp802816", "far=0 mv=8 icb=64 sspb=392 uspb=4 uspu=2"); //, 1.143680
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb256ic2048ih7iw7 nhwc_desc=ic2048sp12544", "far=0 mv=8 icb=32 sspb=196 uspb=16 uspu=4"); //, 0.081600
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb256ic256ih14iw14 nhwc_desc=ic256sp50176", "far=0 mv=8 icb=32 sspb=98 uspb=8 uspu=4"); //, 0.047680
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb256ic256ih28iw28 nhwc_desc=ic256sp200704", "far=0 mv=8 icb=32 sspb=392 uspb=8 uspu=4"); //, 0.455840
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb256ic256ih56iw56 nhwc_desc=ic256sp802816", "far=1 mv=8 icb=32 sspb=98 uspb=4 uspu=1"); //, 2.330240
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb256ic512ih14iw14 nhwc_desc=ic512sp50176", "far=0 mv=8 icb=32 sspb=196 uspb=16 uspu=4"); //, 0.082400
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb256ic512ih28iw28 nhwc_desc=ic512sp200704", "far=0 mv=8 icb=64 sspb=392 uspb=4 uspu=2"); //, 1.126560
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb256ic512ih7iw7 nhwc_desc=ic512sp12544", "far=0 mv=8 icb=32 sspb=49 uspb=8 uspu=4"); //, 0.029600
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb256ic64ih112iw112 nhwc_desc=ic64sp3211264", "far=1 mv=8 icb=32 sspb=98 uspb=4 uspu=1"); //, 2.359040
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb256ic64ih56iw56 nhwc_desc=ic64sp802816", "far=0 mv=8 icb=32 sspb=392 uspb=4 uspu=4"); //, 0.477120
        // nhwc_reusable version
        // resnet-50.tr.fp32.pt.mb16_pvc
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb16ic1024ih14iw14 nhwc_desc=ic1024sp3136", "far=1 mv=8 icb=64 sspb=14 uspb=-1 uspu=-1"); //, 0.036640
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb16ic128ih28iw28 nhwc_desc=ic128sp12544", "far=1 mv=8 icb=64 sspb=14 uspb=8 uspu=-1"); //, 0.028000
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb16ic128ih56iw56 nhwc_desc=ic128sp50176", "far=1 mv=8 icb=64 sspb=28 uspb=-1 uspu=-1"); //, 0.058560
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb16ic2048ih7iw7 nhwc_desc=ic2048sp784", "far=0 mv=8 icb=64 sspb=7 uspb=-1 uspu=-1"); //, 0.025760
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb16ic256ih14iw14 nhwc_desc=ic256sp3136", "far=0 mv=8 icb=32 sspb=16 uspb=8 uspu=-1"); //, 0.024160
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb16ic256ih28iw28 nhwc_desc=ic256sp12544", "far=1 mv=8 icb=64 sspb=14 uspb=-1 uspu=-1"); //, 0.037440
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb16ic256ih56iw56 nhwc_desc=ic256sp50176", "far=0 mv=8 icb=64 sspb=49 uspb=-1 uspu=-1"); //, 0.093600
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb16ic512ih14iw14 nhwc_desc=ic512sp3136", "far=0 mv=8 icb=64 sspb=16 uspb=8 uspu=-1"); //, 0.028160
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb16ic512ih28iw28 nhwc_desc=ic512sp12544", "far=1 mv=8 icb=64 sspb=28 uspb=-1 uspu=-1"); //, 0.057120
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb16ic512ih7iw7 nhwc_desc=ic512sp784", "far=0 mv=8 icb=32 sspb=7 uspb=-1 uspu=-1"); //, 0.020640
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb16ic64ih112iw112 nhwc_desc=ic64sp200704", "far=1 mv=8 icb=64 sspb=49 uspb=-1 uspu=-1"); //, 0.093440
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb16ic64ih56iw56 nhwc_desc=ic64sp50176", "far=1 mv=8 icb=64 sspb=14 uspb=-1 uspu=-1"); //, 0.036640
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb16ic1024ih14iw14 nhwc_desc=ic1024sp3136", "far=1 mv=8 icb=64 sspb=14 uspb=-1 uspu=-1"); //, 0.030400
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb16ic128ih28iw28 nhwc_desc=ic128sp12544", "far=1 mv=8 icb=32 sspb=14 uspb=-1 uspu=-1"); //, 0.026400
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb16ic128ih56iw56 nhwc_desc=ic128sp50176", "far=1 mv=8 icb=64 sspb=28 uspb=-1 uspu=-1"); //, 0.047200
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb16ic2048ih7iw7 nhwc_desc=ic2048sp784", "far=0 mv=8 icb=64 sspb=7 uspb=8 uspu=-1"); //, 0.023520
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb16ic256ih14iw14 nhwc_desc=ic256sp3136", "far=0 mv=8 icb=32 sspb=16 uspb=8 uspu=-1"); //, 0.024000
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb16ic256ih28iw28 nhwc_desc=ic256sp12544", "far=1 mv=8 icb=64 sspb=14 uspb=16 uspu=-1"); //, 0.031840
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb16ic256ih56iw56 nhwc_desc=ic256sp50176", "far=0 mv=8 icb=64 sspb=49 uspb=-1 uspu=-1"); //, 0.072160
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb16ic512ih14iw14 nhwc_desc=ic512sp3136", "far=1 mv=8 icb=64 sspb=7 uspb=4 uspu=-1"); //, 0.026080
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb16ic512ih28iw28 nhwc_desc=ic512sp12544", "far=1 mv=8 icb=64 sspb=28 uspb=-1 uspu=-1"); //, 0.044640
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb16ic512ih7iw7 nhwc_desc=ic512sp784", "far=0 mv=8 icb=32 sspb=7 uspb=4 uspu=-1"); //, 0.019680
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb16ic64ih112iw112 nhwc_desc=ic64sp200704", "far=1 mv=8 icb=64 sspb=49 uspb=16 uspu=-1"); //, 0.077280
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb16ic64ih56iw56 nhwc_desc=ic64sp50176", "far=1 mv=8 icb=128 sspb=14 uspb=16 uspu=-1"); //, 0.032480
        // resnet-50.tr.bf16.pt.mb256_pvc
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=bf16 tag=acdb flags=CH desc=mb256ic1024ih14iw14 nhwc_desc=ic1024sp50176", "far=0 mv=8 icb=128 sspb=98 uspb=16 uspu=-1"); //, 0.396800
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=bf16 tag=acdb flags=CH desc=mb256ic128ih28iw28 nhwc_desc=ic128sp200704", "far=1 mv=8 icb=128 sspb=49 uspb=-1 uspu=-1"); //, 0.124960
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=bf16 tag=acdb flags=CH desc=mb256ic128ih56iw56 nhwc_desc=ic128sp802816", "far=1 mv=8 icb=128 sspb=196 uspb=8 uspu=-1"); //, 1.011040
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=bf16 tag=acdb flags=CH desc=mb256ic2048ih7iw7 nhwc_desc=ic2048sp12544", "far=0 mv=8 icb=128 sspb=49 uspb=-1 uspu=-1"); //, 0.120320
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=bf16 tag=acdb flags=CH desc=mb256ic256ih14iw14 nhwc_desc=ic256sp50176", "far=1 mv=8 icb=128 sspb=28 uspb=-1 uspu=-1"); //, 0.080320
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=bf16 tag=acdb flags=CH desc=mb256ic256ih28iw28 nhwc_desc=ic256sp200704", "far=1 mv=8 icb=64 sspb=196 uspb=16 uspu=-1"); //, 0.403840
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=bf16 tag=acdb flags=CH desc=mb256ic256ih56iw56 nhwc_desc=ic256sp802816", "far=0 mv=8 icb=128 sspb=392 uspb=4 uspu=-1"); //, 2.110560
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=bf16 tag=acdb flags=CH desc=mb256ic512ih14iw14 nhwc_desc=ic512sp50176", "far=0 mv=8 icb=128 sspb=49 uspb=-1 uspu=-1"); //, 0.123040
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=bf16 tag=acdb flags=CH desc=mb256ic512ih28iw28 nhwc_desc=ic512sp200704", "far=0 mv=8 icb=64 sspb=392 uspb=8 uspu=-1"); //, 1.018720
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=bf16 tag=acdb flags=CH desc=mb256ic512ih7iw7 nhwc_desc=ic512sp12544", "far=1 mv=8 icb=64 sspb=28 uspb=-1 uspu=-1"); //, 0.050880
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=bf16 tag=acdb flags=CH desc=mb256ic64ih112iw112 nhwc_desc=ic64sp3211264", "far=1 mv=8 icb=128 sspb=196 uspb=16 uspu=-1"); //, 2.164160
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=bf16 tag=acdb flags=CH desc=mb256ic64ih56iw56 nhwc_desc=ic64sp802816", "far=1 mv=8 icb=64 sspb=196 uspb=16 uspu=-1"); //, 0.413600
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=bf16 tag=acdb flags=CH desc=mb256ic1024ih14iw14 nhwc_desc=ic1024sp50176", "far=0 mv=8 icb=64 sspb=196 uspb=-1 uspu=-1"); //, 0.204320
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=bf16 tag=acdb flags=CH desc=mb256ic128ih28iw28 nhwc_desc=ic128sp200704", "far=1 mv=8 icb=64 sspb=98 uspb=-1 uspu=-1"); //, 0.114400
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=bf16 tag=acdb flags=CH desc=mb256ic128ih56iw56 nhwc_desc=ic128sp802816", "far=0 mv=8 icb=128 sspb=196 uspb=16 uspu=-1"); //, 0.624640
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=bf16 tag=acdb flags=CH desc=mb256ic2048ih7iw7 nhwc_desc=ic2048sp12544", "far=0 mv=8 icb=64 sspb=98 uspb=-1 uspu=-1"); //, 0.107520
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=bf16 tag=acdb flags=CH desc=mb256ic256ih14iw14 nhwc_desc=ic256sp50176", "far=0 mv=8 icb=64 sspb=49 uspb=-1 uspu=-1"); //, 0.064800
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=bf16 tag=acdb flags=CH desc=mb256ic256ih28iw28 nhwc_desc=ic256sp200704", "far=0 mv=8 icb=64 sspb=196 uspb=-1 uspu=-1"); //, 0.208000
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=bf16 tag=acdb flags=CH desc=mb256ic256ih56iw56 nhwc_desc=ic256sp802816", "far=0 mv=8 icb=128 sspb=392 uspb=4 uspu=-1"); //, 1.306880
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=bf16 tag=acdb flags=CH desc=mb256ic512ih14iw14 nhwc_desc=ic512sp50176", "far=0 mv=8 icb=64 sspb=98 uspb=-1 uspu=-1"); //, 0.105120
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=bf16 tag=acdb flags=CH desc=mb256ic512ih28iw28 nhwc_desc=ic512sp200704", "far=0 mv=8 icb=128 sspb=196 uspb=16 uspu=-1"); //, 0.616320
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=bf16 tag=acdb flags=CH desc=mb256ic512ih7iw7 nhwc_desc=ic512sp12544", "far=1 mv=8 icb=64 sspb=28 uspb=-1 uspu=-1"); //, 0.043040
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=bf16 tag=acdb flags=CH desc=mb256ic64ih112iw112 nhwc_desc=ic64sp3211264", "far=0 mv=8 icb=128 sspb=392 uspb=8 uspu=-1"); //, 1.424320
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=bf16 tag=acdb flags=CH desc=mb256ic64ih56iw56 nhwc_desc=ic64sp802816", "far=1 mv=8 icb=64 sspb=196 uspb=-1 uspu=-1"); //, 0.217120
        // resnet-50.tr.fp32.pt.mb256_pvc_aurora
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb256ic1024ih14iw14 nhwc_desc=ic1024sp50176", "far=0 mv=8 icb=64 sspb=196 uspb=4 uspu=-1"); //, 0.985440
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb256ic128ih28iw28 nhwc_desc=ic128sp200704", "far=0 mv=8 icb=64 sspb=98 uspb=16 uspu=-1"); //, 0.349760
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb256ic128ih56iw56 nhwc_desc=ic128sp802816", "far=1 mv=8 icb=32 sspb=49 uspb=4 uspu=-1"); //, 2.046240
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb256ic2048ih7iw7 nhwc_desc=ic2048sp12544", "far=0 mv=8 icb=64 sspb=98 uspb=8 uspu=-1"); //, 0.337440
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb256ic256ih14iw14 nhwc_desc=ic256sp50176", "far=1 mv=8 icb=64 sspb=49 uspb=-1 uspu=-1"); //, 0.092640
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb256ic256ih28iw28 nhwc_desc=ic256sp200704", "far=1 mv=8 icb=64 sspb=196 uspb=8 uspu=-1"); //, 0.974560
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb256ic256ih56iw56 nhwc_desc=ic256sp802816", "far=1 mv=8 icb=32 sspb=64 uspb=4 uspu=-1"); //, 4.052640
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb256ic512ih14iw14 nhwc_desc=ic512sp50176", "far=0 mv=8 icb=64 sspb=98 uspb=4 uspu=-1"); //, 0.343520
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb256ic512ih28iw28 nhwc_desc=ic512sp200704", "far=1 mv=8 icb=32 sspb=14 uspb=4 uspu=-1"); //, 2.043840
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb256ic512ih7iw7 nhwc_desc=ic512sp12544", "far=1 mv=8 icb=64 sspb=28 uspb=-1 uspu=-1"); //, 0.056800
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb256ic64ih112iw112 nhwc_desc=ic64sp3211264", "far=1 mv=8 icb=32 sspb=196 uspb=4 uspu=-1"); //, 4.126560
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb256ic64ih56iw56 nhwc_desc=ic64sp802816", "far=0 mv=8 icb=128 sspb=196 uspb=4 uspu=-1"); //, 0.987040
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb256ic1024ih14iw14 nhwc_desc=ic1024sp50176", "far=1 mv=8 icb=64 sspb=196 uspb=16 uspu=-1"); //, 0.501440
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb256ic128ih28iw28 nhwc_desc=ic128sp200704", "far=1 mv=8 icb=64 sspb=98 uspb=16 uspu=-1"); //, 0.132000
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb256ic128ih56iw56 nhwc_desc=ic128sp802816", "far=1 mv=8 icb=64 sspb=392 uspb=4 uspu=-1"); //, 1.201440
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb256ic2048ih7iw7 nhwc_desc=ic2048sp12544", "far=0 mv=8 icb=64 sspb=98 uspb=-1 uspu=-1"); //, 0.121760
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb256ic256ih14iw14 nhwc_desc=ic256sp50176", "far=0 mv=8 icb=64 sspb=49 uspb=-1 uspu=-1"); //, 0.072000
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb256ic256ih28iw28 nhwc_desc=ic256sp200704", "far=1 mv=8 icb=64 sspb=196 uspb=16 uspu=-1"); //, 0.506400
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb256ic256ih56iw56 nhwc_desc=ic256sp802816", "far=1 mv=8 icb=64 sspb=98 uspb=4 uspu=-1"); //, 2.495840
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb256ic512ih14iw14 nhwc_desc=ic512sp50176", "far=0 mv=8 icb=64 sspb=98 uspb=-1 uspu=-1"); //, 0.117280
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb256ic512ih28iw28 nhwc_desc=ic512sp200704", "far=0 mv=8 icb=64 sspb=392 uspb=4 uspu=-1"); //, 1.191840
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb256ic512ih7iw7 nhwc_desc=ic512sp12544", "far=1 mv=8 icb=64 sspb=28 uspb=-1 uspu=-1"); //, 0.044640
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb256ic64ih112iw112 nhwc_desc=ic64sp3211264", "far=1 mv=8 icb=128 sspb=392 uspb=4 uspu=-1"); //, 2.535680
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb256ic64ih56iw56 nhwc_desc=ic64sp802816", "far=1 mv=8 icb=64 sspb=196 uspb=8 uspu=-1"); //, 0.520000
    } else { // for regular algorithm
        // resnet-50.tr.fp32.pt.mb16_pvc
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb16ic1024ih14iw14 nhwc_desc=ic1024sp3136", "far=0 mv=8 icb=32 sspb=14 uspb=8 uspu=4"); //, 0.028000
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb16ic128ih28iw28 nhwc_desc=ic128sp12544", "far=0 mv=8 icb=32 sspb=49 uspb=8 uspu=4"); //, 0.020160
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb16ic128ih56iw56 nhwc_desc=ic128sp50176", "far=1 mv=8 icb=64 sspb=28 uspb=4 uspu=4"); //, 0.048160
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb16ic2048ih7iw7 nhwc_desc=ic2048sp784", "far=0 mv=8 icb=32 sspb=14 uspb=8 uspu=4"); //, 0.019200
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb16ic256ih14iw14 nhwc_desc=ic256sp3136", "far=0 mv=8 icb=32 sspb=7 uspb=-1 uspu=1"); //, 0.017120
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb16ic256ih28iw28 nhwc_desc=ic256sp12544", "far=0 mv=8 icb=64 sspb=49 uspb=4 uspu=4"); //, 0.029440
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb16ic256ih56iw56 nhwc_desc=ic256sp50176", "far=0 mv=8 icb=32 sspb=98 uspb=8 uspu=4"); //, 0.080480
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb16ic512ih14iw14 nhwc_desc=ic512sp3136", "far=0 mv=8 icb=32 sspb=7 uspb=8 uspu=4"); //, 0.020160
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb16ic512ih28iw28 nhwc_desc=ic512sp12544", "far=0 mv=8 icb=32 sspb=49 uspb=8 uspu=4"); //, 0.045120
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb16ic512ih7iw7 nhwc_desc=ic512sp784", "far=0 mv=8 icb=64 sspb=7 uspb=4 uspu=4"); //, 0.013920
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb16ic64ih112iw112 nhwc_desc=ic64sp200704", "far=1 mv=8 icb=64 sspb=49 uspb=16 uspu=4"); //, 0.082720
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb16ic64ih56iw56 nhwc_desc=ic64sp50176", "far=1 mv=8 icb=32 sspb=16 uspb=8 uspu=4"); //, 0.031200
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb16ic1024ih14iw14 nhwc_desc=ic1024sp3136", "far=0 mv=8 icb=32 sspb=14 uspb=16 uspu=4"); //, 0.028480
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb16ic128ih28iw28 nhwc_desc=ic128sp12544", "far=0 mv=8 icb=32 sspb=32 uspb=4 uspu=2"); //, 0.024320
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb16ic128ih56iw56 nhwc_desc=ic128sp50176", "far=1 mv=8 icb=32 sspb=49 uspb=8 uspu=4"); //, 0.046080
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb16ic2048ih7iw7 nhwc_desc=ic2048sp784", "far=0 mv=8 icb=64 sspb=14 uspb=-1 uspu=4"); //, 0.025440
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb16ic256ih14iw14 nhwc_desc=ic256sp3136", "far=0 mv=8 icb=64 sspb=16 uspb=-1 uspu=4"); //, 0.021600
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb16ic256ih28iw28 nhwc_desc=ic256sp12544", "far=0 mv=8 icb=32 sspb=28 uspb=4 uspu=4"); //, 0.029760
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb16ic256ih56iw56 nhwc_desc=ic256sp50176", "far=0 mv=8 icb=32 sspb=98 uspb=8 uspu=4"); //, 0.066880
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb16ic512ih14iw14 nhwc_desc=ic512sp3136", "far=0 mv=8 icb=64 sspb=28 uspb=4 uspu=4"); //, 0.024320
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb16ic512ih28iw28 nhwc_desc=ic512sp12544", "far=0 mv=8 icb=32 sspb=49 uspb=8 uspu=4"); //, 0.041120
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb16ic512ih7iw7 nhwc_desc=ic512sp784", "far=0 mv=8 icb=64 sspb=7 uspb=16 uspu=2"); //, 0.019520
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb16ic64ih112iw112 nhwc_desc=ic64sp200704", "far=1 mv=8 icb=64 sspb=49 uspb=16 uspu=4"); //, 0.072800
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb16ic64ih56iw56 nhwc_desc=ic64sp50176", "far=0 mv=8 icb=32 sspb=98 uspb=4 uspu=4"); //, 0.032160
        // resnet-50.tr.bf16.pt.mb256_pvc
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=bf16 tag=acdb flags=CH desc=mb256ic1024ih14iw14 nhwc_desc=ic1024sp50176", "far=0 mv=8 icb=64 sspb=196 uspb=16 uspu=2"); //, 0.311040
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=bf16 tag=acdb flags=CH desc=mb256ic128ih28iw28 nhwc_desc=ic128sp200704", "far=1 mv=8 icb=32 sspb=196 uspb=-1 uspu=4"); //, 0.097920
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=bf16 tag=acdb flags=CH desc=mb256ic128ih56iw56 nhwc_desc=ic128sp802816", "far=1 mv=8 icb=64 sspb=392 uspb=4 uspu=2"); //, 0.950880
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=bf16 tag=acdb flags=CH desc=mb256ic2048ih7iw7 nhwc_desc=ic2048sp12544", "far=0 mv=8 icb=128 sspb=49 uspb=-1 uspu=1"); //, 0.092800
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=bf16 tag=acdb flags=CH desc=mb256ic256ih14iw14 nhwc_desc=ic256sp50176", "far=0 mv=8 icb=32 sspb=98 uspb=-1 uspu=2"); //, 0.059680
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=bf16 tag=acdb flags=CH desc=mb256ic256ih28iw28 nhwc_desc=ic256sp200704", "far=0 mv=8 icb=128 sspb=392 uspb=8 uspu=2"); //, 0.317440
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=bf16 tag=acdb flags=CH desc=mb256ic256ih56iw56 nhwc_desc=ic256sp802816", "far=1 mv=8 icb=128 sspb=392 uspb=4 uspu=4"); //, 1.981120
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=bf16 tag=acdb flags=CH desc=mb256ic512ih14iw14 nhwc_desc=ic512sp50176", "far=0 mv=8 icb=32 sspb=196 uspb=-1 uspu=4"); //, 0.095520
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=bf16 tag=acdb flags=CH desc=mb256ic512ih28iw28 nhwc_desc=ic512sp200704", "far=0 mv=8 icb=64 sspb=392 uspb=4 uspu=4"); //, 0.929760
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=bf16 tag=acdb flags=CH desc=mb256ic512ih7iw7 nhwc_desc=ic512sp12544", "far=0 mv=8 icb=64 sspb=28 uspb=-1 uspu=4"); //, 0.034560
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=bf16 tag=acdb flags=CH desc=mb256ic64ih112iw112 nhwc_desc=ic64sp3211264", "far=1 mv=8 icb=256 sspb=196 uspb=4 uspu=4"); //, 2.051520
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=bf16 tag=acdb flags=CH desc=mb256ic64ih56iw56 nhwc_desc=ic64sp802816", "far=1 mv=8 icb=64 sspb=196 uspb=8 uspu=4"); //, 0.324000
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=bf16 tag=acdb flags=CH desc=mb256ic1024ih14iw14 nhwc_desc=ic1024sp50176", "far=0 mv=8 icb=32 sspb=392 uspb=-1 uspu=4"); //, 0.115200
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=bf16 tag=acdb flags=CH desc=mb256ic128ih28iw28 nhwc_desc=ic128sp200704", "far=1 mv=8 icb=32 sspb=196 uspb=-1 uspu=4"); //, 0.069280
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=bf16 tag=acdb flags=CH desc=mb256ic128ih56iw56 nhwc_desc=ic128sp802816", "far=1 mv=8 icb=256 sspb=196 uspb=8 uspu=4"); //, 0.537920
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=bf16 tag=acdb flags=CH desc=mb256ic2048ih7iw7 nhwc_desc=ic2048sp12544", "far=0 mv=8 icb=32 sspb=196 uspb=-1 uspu=4"); //, 0.064480
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=bf16 tag=acdb flags=CH desc=mb256ic256ih14iw14 nhwc_desc=ic256sp50176", "far=0 mv=8 icb=32 sspb=98 uspb=16 uspu=4"); //, 0.044160
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=bf16 tag=acdb flags=CH desc=mb256ic256ih28iw28 nhwc_desc=ic256sp200704", "far=0 mv=8 icb=32 sspb=392 uspb=-1 uspu=4"); //, 0.115520
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=bf16 tag=acdb flags=CH desc=mb256ic256ih56iw56 nhwc_desc=ic256sp802816", "far=1 mv=8 icb=128 sspb=392 uspb=4 uspu=2"); //, 1.543680
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=bf16 tag=acdb flags=CH desc=mb256ic512ih14iw14 nhwc_desc=ic512sp50176", "far=0 mv=8 icb=32 sspb=196 uspb=-1 uspu=4"); //, 0.065120
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=bf16 tag=acdb flags=CH desc=mb256ic512ih28iw28 nhwc_desc=ic512sp200704", "far=1 mv=8 icb=128 sspb=196 uspb=8 uspu=4"); //, 0.522720
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=bf16 tag=acdb flags=CH desc=mb256ic512ih7iw7 nhwc_desc=ic512sp12544", "far=0 mv=8 icb=32 sspb=49 uspb=16 uspu=4"); //, 0.030080
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=bf16 tag=acdb flags=CH desc=mb256ic64ih112iw112 nhwc_desc=ic64sp3211264", "far=1 mv=8 icb=128 sspb=392 uspb=4 uspu=1"); //, 1.655040
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=bf16 tag=acdb flags=CH desc=mb256ic64ih56iw56 nhwc_desc=ic64sp802816", "far=1 mv=8 icb=32 sspb=392 uspb=-1 uspu=4"); //, 0.120640
        // resnet-50.tr.fp32.pt.mb256_pvc_aurora
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb256ic1024ih14iw14 nhwc_desc=ic1024sp50176", "far=0 mv=8 icb=32 sspb=392 uspb=4 uspu=2"); //, 0.925440
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb256ic128ih28iw28 nhwc_desc=ic128sp200704", "far=0 mv=8 icb=64 sspb=392 uspb=4 uspu=4"); //, 0.290880
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb256ic128ih56iw56 nhwc_desc=ic128sp802816", "far=1 mv=8 icb=32 sspb=64 uspb=4 uspu=2"); //, 1.998720
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb256ic2048ih7iw7 nhwc_desc=ic2048sp12544", "far=0 mv=8 icb=32 sspb=196 uspb=8 uspu=4"); //, 0.292800
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb256ic256ih14iw14 nhwc_desc=ic256sp50176", "far=0 mv=8 icb=32 sspb=98 uspb=8 uspu=4"); //, 0.080960
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb256ic256ih28iw28 nhwc_desc=ic256sp200704", "far=0 mv=8 icb=32 sspb=392 uspb=4 uspu=1"); //, 0.925760
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb256ic256ih56iw56 nhwc_desc=ic256sp802816", "far=1 mv=8 icb=32 sspb=49 uspb=4 uspu=1"); //, 3.960640
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb256ic512ih14iw14 nhwc_desc=ic512sp50176", "far=0 mv=8 icb=32 sspb=196 uspb=8 uspu=4"); //, 0.288640
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb256ic512ih28iw28 nhwc_desc=ic512sp200704", "far=1 mv=8 icb=32 sspb=16 uspb=4 uspu=1"); //, 1.968960
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb256ic512ih7iw7 nhwc_desc=ic512sp12544", "far=0 mv=8 icb=32 sspb=49 uspb=8 uspu=4"); //, 0.045600
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb256ic64ih112iw112 nhwc_desc=ic64sp3211264", "far=1 mv=8 icb=32 sspb=196 uspb=4 uspu=2"); //, 4.016320
        add("hw=xe_hpc impl=nhwc_opt dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb256ic64ih56iw56 nhwc_desc=ic64sp802816", "far=1 mv=8 icb=128 sspb=196 uspb=4 uspu=4"); //, 0.941760
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb256ic1024ih14iw14 nhwc_desc=ic1024sp50176", "far=0 mv=8 icb=64 sspb=196 uspb=4 uspu=2"); //, 0.511520
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb256ic128ih28iw28 nhwc_desc=ic128sp200704", "far=1 mv=8 icb=64 sspb=98 uspb=8 uspu=2"); //, 0.125280
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb256ic128ih56iw56 nhwc_desc=ic128sp802816", "far=1 mv=8 icb=64 sspb=392 uspb=4 uspu=1"); //, 1.561600
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb256ic2048ih7iw7 nhwc_desc=ic2048sp12544", "far=0 mv=8 icb=64 sspb=98 uspb=8 uspu=4"); //, 0.112000
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb256ic256ih14iw14 nhwc_desc=ic256sp50176", "far=0 mv=8 icb=32 sspb=98 uspb=8 uspu=4"); //, 0.066880
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb256ic256ih28iw28 nhwc_desc=ic256sp200704", "far=1 mv=8 icb=32 sspb=392 uspb=4 uspu=4"); //, 0.514720
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb256ic256ih56iw56 nhwc_desc=ic256sp802816", "far=1 mv=8 icb=32 sspb=49 uspb=4 uspu=1"); //, 3.154240
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb256ic512ih14iw14 nhwc_desc=ic512sp50176", "far=0 mv=8 icb=32 sspb=196 uspb=8 uspu=4"); //, 0.113760
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb256ic512ih28iw28 nhwc_desc=ic512sp200704", "far=1 mv=8 icb=32 sspb=32 uspb=4 uspu=1"); //, 1.556960
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb256ic512ih7iw7 nhwc_desc=ic512sp12544", "far=0 mv=8 icb=32 sspb=49 uspb=8 uspu=4"); //, 0.040480
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb256ic64ih112iw112 nhwc_desc=ic64sp3211264", "far=1 mv=8 icb=32 sspb=98 uspb=4 uspu=1"); //, 3.230880
        add("hw=xe_hpc impl=nhwc_opt dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb256ic64ih56iw56 nhwc_desc=ic64sp802816", "far=1 mv=8 icb=128 sspb=196 uspb=4 uspu=2"); //, 0.527360
        // nhwc_reusable version
        // resnet-50.tr.fp32.pt.mb16_pvc
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb16ic1024ih14iw14 nhwc_desc=ic1024sp3136", "far=1 mv=8 icb=64 sspb=14 uspb=-1 uspu=-1"); //, 0.036480
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb16ic128ih28iw28 nhwc_desc=ic128sp12544", "far=1 mv=8 icb=64 sspb=7 uspb=8 uspu=-1"); //, 0.027680
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb16ic128ih56iw56 nhwc_desc=ic128sp50176", "far=1 mv=8 icb=64 sspb=28 uspb=-1 uspu=-1"); //, 0.058720
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb16ic2048ih7iw7 nhwc_desc=ic2048sp784", "far=0 mv=8 icb=64 sspb=7 uspb=-1 uspu=-1"); //, 0.025920
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb16ic256ih14iw14 nhwc_desc=ic256sp3136", "far=0 mv=8 icb=32 sspb=16 uspb=8 uspu=-1"); //, 0.024160
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb16ic256ih28iw28 nhwc_desc=ic256sp12544", "far=1 mv=8 icb=64 sspb=14 uspb=-1 uspu=-1"); //, 0.037280
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb16ic256ih56iw56 nhwc_desc=ic256sp50176", "far=1 mv=8 icb=64 sspb=49 uspb=-1 uspu=-1"); //, 0.093280
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb16ic512ih14iw14 nhwc_desc=ic512sp3136", "far=0 mv=8 icb=64 sspb=16 uspb=8 uspu=-1"); //, 0.027840
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb16ic512ih28iw28 nhwc_desc=ic512sp12544", "far=1 mv=8 icb=64 sspb=28 uspb=-1 uspu=-1"); //, 0.056960
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb16ic512ih7iw7 nhwc_desc=ic512sp784", "far=0 mv=8 icb=32 sspb=7 uspb=4 uspu=-1"); //, 0.020640
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb16ic64ih112iw112 nhwc_desc=ic64sp200704", "far=1 mv=8 icb=128 sspb=49 uspb=-1 uspu=-1"); //, 0.093120
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb16ic64ih56iw56 nhwc_desc=ic64sp50176", "far=1 mv=8 icb=128 sspb=14 uspb=-1 uspu=-1"); //, 0.036320
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb16ic1024ih14iw14 nhwc_desc=ic1024sp3136", "far=0 mv=8 icb=64 sspb=16 uspb=-1 uspu=-1"); //, 0.038720
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb16ic128ih28iw28 nhwc_desc=ic128sp12544", "far=0 mv=8 icb=64 sspb=7 uspb=8 uspu=-1"); //, 0.035360
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb16ic128ih56iw56 nhwc_desc=ic128sp50176", "far=1 mv=8 icb=128 sspb=14 uspb=8 uspu=-1"); //, 0.055680
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb16ic2048ih7iw7 nhwc_desc=ic2048sp784", "far=0 mv=8 icb=64 sspb=7 uspb=16 uspu=-1"); //, 0.035840
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb16ic256ih14iw14 nhwc_desc=ic256sp3136", "far=0 mv=8 icb=128 sspb=7 uspb=4 uspu=-1"); //, 0.034400
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb16ic256ih28iw28 nhwc_desc=ic256sp12544", "far=0 mv=8 icb=64 sspb=16 uspb=16 uspu=-1"); //, 0.042080
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb16ic256ih56iw56 nhwc_desc=ic256sp50176", "far=1 mv=8 icb=128 sspb=28 uspb=4 uspu=-1"); //, 0.086720
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb16ic512ih14iw14 nhwc_desc=ic512sp3136", "far=0 mv=8 icb=64 sspb=16 uspb=8 uspu=-1"); //, 0.035840
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb16ic512ih28iw28 nhwc_desc=ic512sp12544", "far=0 mv=8 icb=128 sspb=16 uspb=8 uspu=-1"); //, 0.053120
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb16ic512ih7iw7 nhwc_desc=ic512sp784", "far=0 mv=8 icb=64 sspb=14 uspb=4 uspu=-1"); //, 0.034720
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb16ic64ih112iw112 nhwc_desc=ic64sp200704", "far=1 mv=8 icb=64 sspb=49 uspb=8 uspu=-1"); //, 0.093440
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb16ic64ih56iw56 nhwc_desc=ic64sp50176", "far=1 mv=8 icb=64 sspb=14 uspb=-1 uspu=-1"); //, 0.043200
        // resnet-50.tr.bf16.pt.mb256_pvc
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=bf16 tag=acdb flags=CH desc=mb256ic1024ih14iw14 nhwc_desc=ic1024sp50176", "far=1 mv=8 icb=128 sspb=98 uspb=8 uspu=-1"); //, 0.396960
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=bf16 tag=acdb flags=CH desc=mb256ic128ih28iw28 nhwc_desc=ic128sp200704", "far=1 mv=8 icb=128 sspb=49 uspb=-1 uspu=-1"); //, 0.124640
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=bf16 tag=acdb flags=CH desc=mb256ic128ih56iw56 nhwc_desc=ic128sp802816", "far=1 mv=8 icb=128 sspb=196 uspb=8 uspu=-1"); //, 1.023840
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=bf16 tag=acdb flags=CH desc=mb256ic2048ih7iw7 nhwc_desc=ic2048sp12544", "far=0 mv=8 icb=128 sspb=49 uspb=-1 uspu=-1"); //, 0.120480
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=bf16 tag=acdb flags=CH desc=mb256ic256ih14iw14 nhwc_desc=ic256sp50176", "far=1 mv=8 icb=128 sspb=28 uspb=-1 uspu=-1"); //, 0.080160
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=bf16 tag=acdb flags=CH desc=mb256ic256ih28iw28 nhwc_desc=ic256sp200704", "far=0 mv=8 icb=128 sspb=196 uspb=8 uspu=-1"); //, 0.408160
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=bf16 tag=acdb flags=CH desc=mb256ic256ih56iw56 nhwc_desc=ic256sp802816", "far=0 mv=8 icb=128 sspb=392 uspb=4 uspu=-1"); //, 2.111840
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=bf16 tag=acdb flags=CH desc=mb256ic512ih14iw14 nhwc_desc=ic512sp50176", "far=0 mv=8 icb=128 sspb=49 uspb=-1 uspu=-1"); //, 0.123360
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=bf16 tag=acdb flags=CH desc=mb256ic512ih28iw28 nhwc_desc=ic512sp200704", "far=1 mv=8 icb=128 sspb=196 uspb=4 uspu=-1"); //, 1.015360
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=bf16 tag=acdb flags=CH desc=mb256ic512ih7iw7 nhwc_desc=ic512sp12544", "far=0 mv=8 icb=128 sspb=16 uspb=16 uspu=-1"); //, 0.051360
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=bf16 tag=acdb flags=CH desc=mb256ic64ih112iw112 nhwc_desc=ic64sp3211264", "far=1 mv=8 icb=128 sspb=196 uspb=16 uspu=-1"); //, 2.179360
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=bf16 tag=acdb flags=CH desc=mb256ic64ih56iw56 nhwc_desc=ic64sp802816", "far=1 mv=8 icb=64 sspb=196 uspb=16 uspu=-1"); //, 0.406240
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=bf16 tag=acdb flags=CH desc=mb256ic1024ih14iw14 nhwc_desc=ic1024sp50176", "far=0 mv=8 icb=128 sspb=98 uspb=-1 uspu=-1"); //, 0.167840
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=bf16 tag=acdb flags=CH desc=mb256ic128ih28iw28 nhwc_desc=ic128sp200704", "far=1 mv=8 icb=128 sspb=49 uspb=-1 uspu=-1"); //, 0.098240
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=bf16 tag=acdb flags=CH desc=mb256ic128ih56iw56 nhwc_desc=ic128sp802816", "far=1 mv=8 icb=128 sspb=196 uspb=4 uspu=-1"); //, 0.662560
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=bf16 tag=acdb flags=CH desc=mb256ic2048ih7iw7 nhwc_desc=ic2048sp12544", "far=0 mv=8 icb=128 sspb=49 uspb=-1 uspu=-1"); //, 0.088480
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=bf16 tag=acdb flags=CH desc=mb256ic256ih14iw14 nhwc_desc=ic256sp50176", "far=1 mv=8 icb=128 sspb=28 uspb=-1 uspu=-1"); //, 0.065600
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=bf16 tag=acdb flags=CH desc=mb256ic256ih28iw28 nhwc_desc=ic256sp200704", "far=1 mv=8 icb=128 sspb=98 uspb=-1 uspu=-1"); //, 0.173280
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=bf16 tag=acdb flags=CH desc=mb256ic256ih56iw56 nhwc_desc=ic256sp802816", "far=1 mv=8 icb=128 sspb=392 uspb=4 uspu=-1"); //, 1.704160
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=bf16 tag=acdb flags=CH desc=mb256ic512ih14iw14 nhwc_desc=ic512sp50176", "far=0 mv=8 icb=128 sspb=49 uspb=-1 uspu=-1"); //, 0.092480
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=bf16 tag=acdb flags=CH desc=mb256ic512ih28iw28 nhwc_desc=ic512sp200704", "far=0 mv=8 icb=128 sspb=196 uspb=8 uspu=-1"); //, 0.639360
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=bf16 tag=acdb flags=CH desc=mb256ic512ih7iw7 nhwc_desc=ic512sp12544", "far=1 mv=8 icb=128 sspb=14 uspb=-1 uspu=-1"); //, 0.044000
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=bf16 tag=acdb flags=CH desc=mb256ic64ih112iw112 nhwc_desc=ic64sp3211264", "far=0 mv=8 icb=128 sspb=392 uspb=8 uspu=-1"); //, 1.865120
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=bf16 tag=acdb flags=CH desc=mb256ic64ih56iw56 nhwc_desc=ic64sp802816", "far=1 mv=8 icb=64 sspb=196 uspb=-1 uspu=-1"); //, 0.274400
        // resnet-50.tr.fp32.pt.mb256_pvc_aurora
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb256ic1024ih14iw14 nhwc_desc=ic1024sp50176", "far=0 mv=8 icb=32 sspb=392 uspb=8 uspu=-1"); //, 0.984640
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb256ic128ih28iw28 nhwc_desc=ic128sp200704", "far=0 mv=8 icb=64 sspb=98 uspb=8 uspu=-1"); //, 0.353440
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb256ic128ih56iw56 nhwc_desc=ic128sp802816", "far=1 mv=8 icb=32 sspb=98 uspb=4 uspu=-1"); //, 2.055200
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb256ic2048ih7iw7 nhwc_desc=ic2048sp12544", "far=0 mv=8 icb=64 sspb=98 uspb=16 uspu=-1"); //, 0.341280
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb256ic256ih14iw14 nhwc_desc=ic256sp50176", "far=0 mv=8 icb=64 sspb=49 uspb=-1 uspu=-1"); //, 0.093760
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb256ic256ih28iw28 nhwc_desc=ic256sp200704", "far=0 mv=8 icb=32 sspb=392 uspb=4 uspu=-1"); //, 0.973920
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb256ic256ih56iw56 nhwc_desc=ic256sp802816", "far=1 mv=8 icb=32 sspb=32 uspb=4 uspu=-1"); //, 4.031360
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb256ic512ih14iw14 nhwc_desc=ic512sp50176", "far=0 mv=8 icb=64 sspb=98 uspb=8 uspu=-1"); //, 0.336800
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb256ic512ih28iw28 nhwc_desc=ic512sp200704", "far=1 mv=8 icb=32 sspb=32 uspb=4 uspu=-1"); //, 2.054240
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb256ic512ih7iw7 nhwc_desc=ic512sp12544", "far=0 mv=8 icb=64 sspb=32 uspb=-1 uspu=-1"); //, 0.057760
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb256ic64ih112iw112 nhwc_desc=ic64sp3211264", "far=1 mv=8 icb=32 sspb=64 uspb=8 uspu=-1"); //, 4.064320
        add("hw=xe_hpc impl=nhwc_reusable dir=BWD_DW dt=f32 tag=acdb flags=CH desc=mb256ic64ih56iw56 nhwc_desc=ic64sp802816", "far=1 mv=8 icb=128 sspb=196 uspb=8 uspu=-1"); //, 0.978240
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb256ic1024ih14iw14 nhwc_desc=ic1024sp50176", "far=0 mv=8 icb=128 sspb=98 uspb=4 uspu=-1"); //, 0.588960
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb256ic128ih28iw28 nhwc_desc=ic128sp200704", "far=1 mv=8 icb=128 sspb=49 uspb=8 uspu=-1"); //, 0.144640
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb256ic128ih56iw56 nhwc_desc=ic128sp802816", "far=1 mv=8 icb=64 sspb=392 uspb=4 uspu=-1"); //, 1.653440
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb256ic2048ih7iw7 nhwc_desc=ic2048sp12544", "far=0 mv=8 icb=128 sspb=49 uspb=-1 uspu=-1"); //, 0.136000
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb256ic256ih14iw14 nhwc_desc=ic256sp50176", "far=1 mv=8 icb=128 sspb=28 uspb=16 uspu=-1"); //, 0.086720
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb256ic256ih28iw28 nhwc_desc=ic256sp200704", "far=0 mv=8 icb=128 sspb=98 uspb=4 uspu=-1"); //, 0.586080
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb256ic256ih56iw56 nhwc_desc=ic256sp802816", "far=1 mv=8 icb=64 sspb=64 uspb=4 uspu=-1"); //, 3.301600
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb256ic512ih14iw14 nhwc_desc=ic512sp50176", "far=0 mv=8 icb=128 sspb=49 uspb=8 uspu=-1"); //, 0.138720
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb256ic512ih28iw28 nhwc_desc=ic512sp200704", "far=1 mv=8 icb=64 sspb=28 uspb=4 uspu=-1"); //, 1.634400
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb256ic512ih7iw7 nhwc_desc=ic512sp12544", "far=0 mv=8 icb=128 sspb=16 uspb=8 uspu=-1"); //, 0.053280
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb256ic64ih112iw112 nhwc_desc=ic64sp3211264", "far=1 mv=8 icb=128 sspb=196 uspb=4 uspu=-1"); //, 3.386400
        add("hw=xe_hpc impl=nhwc_reusable dir=FWD_D dt=f32 tag=acdb flags=CH desc=mb256ic64ih56iw56 nhwc_desc=ic64sp802816", "far=1 mv=8 icb=64 sspb=196 uspb=8 uspu=-1"); //, 0.643360
        // clang-format on
    }
}

std::string get_desc_str(const params_t &conf) {
    std::ostringstream oss;
    oss << "mb" << conf.mb;
    oss << "ic" << conf.ic;
    oss << "ih" << conf.ih;
    oss << "iw" << conf.iw;
    return oss.str();
}
std::string get_nhwc_desc_str(const params_t &conf) {
    std::ostringstream oss;
    oss << "ic" << conf.ic;
    oss << "sp" << conf.mb * conf.ih * conf.iw * conf.id;
    return oss.str();
}

const char *bnorm_lookup_table_t::find(
        const params_t &conf, const gpu_arch_t &gpu_arch) const {
    const auto &key
            = is_nhwc_impl(conf) ? get_nhwc_desc_str(conf) : get_desc_str(conf);
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
        if (param->short_name() == *p)
            return static_cast<int>(p - ordered_params);
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
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
