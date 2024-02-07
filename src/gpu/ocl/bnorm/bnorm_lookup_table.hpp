/*******************************************************************************
* Copyright 2022-2024 Intel Corporation
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

#ifndef GPU_OCL_BNORM_LOOKUP_TABLE_HPP
#define GPU_OCL_BNORM_LOOKUP_TABLE_HPP

#include <string>
#include <vector>
#include <unordered_map>

#include "gpu/compute/compute.hpp"
#include "gpu/config.hpp"
#include "gpu/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {
namespace bn_lookup_table {

class use_fused_atomics_reduction_param_t : public bool_param_t {
public:
    std::string name() const override { return "use-fused-atomics-reduction"; }
    std::string short_name() const override { return "far"; }
    std::string desc() const override {
        return "Whether to fuse reduction using atomic operations.";
    }
    bool is_overridable() const override { return true; }
};

class ic_block_param_t : public int_param_t {
public:
    std::string name() const override { return "ic-block"; }
    std::string short_name() const override { return "icb"; }
    std::string desc() const override { return "Size of ic blocking."; }
    bool is_overridable() const override { return true; }
};

class max_vect_size_param_t : public int_param_t {
public:
    std::string name() const override { return "vect-size"; }
    std::string short_name() const override { return "mv"; }
    std::string desc() const override {
        return "Maximum size of vectorization.";
    }
    bool is_overridable() const override { return true; }
};

class stat_sp_block_param_t : public int_param_t {
public:
    std::string name() const override { return "stat-sp-block"; }
    std::string short_name() const override { return "sspb"; }
    std::string desc() const override {
        return "Size of spatial dim blocking.";
    }
    bool is_overridable() const override { return true; }
};

class update_sp_block_param_t : public int_param_t {
public:
    std::string name() const override { return "update-sp-block"; }
    std::string short_name() const override { return "uspb"; }
    std::string desc() const override {
        return "Number of spatial elements handled by each work item.";
    }
    bool is_overridable() const override { return true; }
};

class update_sp_unroll_param_t : public int_param_t {
public:
    std::string name() const override { return "update-sp-unroll"; }
    std::string short_name() const override { return "uspu"; }
    std::string desc() const override {
        return "Size of unrolling while handling multiple spatial elements.";
    }
    bool is_overridable() const override { return true; }
};

struct params_t : public bnorm_conf_t, public container_config_t {
#define DECL_PARAM(name) \
    const name##_param_t &name##_param() const { \
        gpu_assert(!name##_.is_undef()); \
        (void)name##_init_; \
        return name##_; \
    } \
    name##_param_t &name##_param() { return name##_; } \
    const name##_param_t::value_t &name() const { \
        gpu_assert(!name##_.is_undef()); \
        return name##_.get(); \
    } \
    void set_##name(const name##_param_t::value_t &value) { \
        name##_.set(value); \
    }

    DECL_PARAM(use_fused_atomics_reduction);
    DECL_PARAM(ic_block);
    DECL_PARAM(max_vect_size);
    DECL_PARAM(stat_sp_block);
    DECL_PARAM(update_sp_block);
    DECL_PARAM(update_sp_unroll);

#undef DECL_PARAM

    std::string str() const override;
    int sort_key(const param_t *param) const override;

private:
#define INIT_PARAM(name) \
    name##_param_t name##_; \
    param_init_t name##_init_ \
            = register_param([](const container_config_t *c) { \
                  return &((const params_t *)c)->name##_; \
              });

    INIT_PARAM(use_fused_atomics_reduction);
    INIT_PARAM(ic_block);
    INIT_PARAM(max_vect_size);
    INIT_PARAM(stat_sp_block);
    INIT_PARAM(update_sp_block);
    INIT_PARAM(update_sp_unroll);

#undef INIT_PARAM

public:
    // Same as for container_config_t, but ignores "-1" values
    void override_set(const std::string &s, bool is_env) override {
        auto params = get_all_params();
        auto parts = gpu_utils::split(s);
        for (auto &p : parts) {
            if (p.empty()) continue;
            auto sub_parts = gpu_utils::split(p, "=");
            gpu_assert(sub_parts.size() == 2);
            auto &key = sub_parts[0];
            auto &value = sub_parts[1];
            if (value == "-1") continue;
            bool found = false;
            for (auto *p : params) {
                if (p->accepts_key(key)) {
                    p->override_set(key, value, is_env);
                    found = true;
                    break;
                }
            }
            // TODO: Get access to ir_info() and ir_warning() to use in
            // case of overriden/unknown parameters.
            gpu_assert(found) << "Unknown parameter";
        }
    }

    normalization_flags_t flags = normalization_flags::none;
    bool bn_tuning = false;
    bool found_in_table = false;
    bool is_blocked_16c = false;
    bool is_blocked_16n16c = false;
    bool is_blocked_32n16c = false;
    bool is_nhwc = false;
};

void maybe_override_bn_conf_params_env(params_t &conf);
void maybe_override_bn_conf_params_table(params_t &conf, engine_t *engine);
void maybe_override_bn_conf_params(params_t &conf, engine_t *engine);

inline std::string getenv_str(const char *s, const std::string &def) {
    char buf[1024];
    int ret = getenv(s, buf, sizeof(buf));
    if (ret > 0) return buf;
    return def;
}

inline std::vector<std::string> split(const std::string &s,
        const std::string &delimiter = std::string(1, ' ')) {
    size_t beg = 0;
    size_t end = 0;
    std::vector<std::string> ret;
    while (end != std::string::npos) {
        beg = (end == 0) ? 0 : end + delimiter.size();
        end = s.find(delimiter, beg);
        size_t len
                = (end == std::string::npos) ? std::string::npos : (end - beg);
        ret.push_back(s.substr(beg, len));
    }
    return ret;
}

std::string get_desc_str(const params_t &conf);

enum class op_kind_t {
    undef,
    _lt,
    _le,
    _gt,
    _ge,
    _ne,
    _eq,
};

class int_filter_t {
public:
    int_filter_t() = default;
    int_filter_t(const std::string &s);
    bool matches(int value) const;

private:
    int value_;
    op_kind_t cmp_op_;
};

class type_filter_t {
public:
    type_filter_t() = default;
    type_filter_t(const std::string &s);
    bool matches(const data_type_t &dt) const;

private:
    static std::vector<std::string> &all_patterns();
    std::string pattern_;
};

class bnorm_problem_filter_t {
public:
    using key_t = std::string;
    bnorm_problem_filter_t(const std::string &s);
    key_t key() const { return desc_; }
    bool matches(
            const params_t &conf, const compute::gpu_arch_t &gpu_arch) const;

private:
    bool matches_dir(const params_t &conf) const;
    bool matches_desc(const params_t &conf) const;
    bool matches_tag(const params_t &conf) const;
    bool matches_flags(const params_t &conf) const;

    std::string dir_;
    type_filter_t type_filter_;
    std::string desc_;
    std::string tag_;
    normalization_flags_t flags_filter_;
    compute::gpu_arch_t hw_;
};

class bnorm_lookup_table_t {
public:
    bnorm_lookup_table_t() = default;
    bnorm_lookup_table_t(bool use_stats_one_pass);
    const char *find(
            const params_t &conf, const compute::gpu_arch_t &gpu_arch) const;
    void get_params(const params_t &conf, const std::string &params) const;

private:
    struct entry_t {
        bnorm_problem_filter_t filter;
        const char *s_params;
    };
    void add(const char *s_prb, const char *s_params);
    using key_t = bnorm_problem_filter_t::key_t;
    std::unordered_map<key_t, std::vector<entry_t>> map_;
};

} // namespace bn_lookup_table
} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
