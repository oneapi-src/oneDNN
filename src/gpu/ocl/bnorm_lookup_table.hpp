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

#ifndef GPU_OCL_BNORM_LOOKUP_TABLE_HPP
#define GPU_OCL_BNORM_LOOKUP_TABLE_HPP

#include <string>
#include <vector>
#include <unordered_map>

#include "gpu/compute/compute.hpp"
#include "gpu/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {
namespace bn_lookup_table {

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

std::string get_desc_str(const bnorm_conf_t &conf);

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

class flags_filter_t {
public:
    flags_filter_t() = default;
    flags_filter_t(const std::string &s);
    bool matches(const std::string &values) const;

private:
    static std::vector<char> &all_patterns();
    std::vector<char> patterns_;
};

class bnorm_problem_filter_t {
public:
    using key_t = std::string;
    bnorm_problem_filter_t(const std::string &s);
    key_t key() const { return desc_; }
    bool matches(const bnorm_conf_t &conf,
            const compute::gpu_arch_t &gpu_arch) const;

private:
    bool matches_dir(const bnorm_conf_t &conf) const;
    bool matches_desc(const bnorm_conf_t &conf) const;
    bool matches_tag(const bnorm_conf_t &conf) const;
    bool matches_flags(const bnorm_conf_t &conf) const;

    std::string dir_;
    type_filter_t type_filter_;
    std::string desc_;
    std::string tag_;
    flags_filter_t flags_filter_;
    compute::gpu_arch_t hw_;
};

class bnorm_params_t {
public:
    bnorm_params_t() = default;
    bnorm_params_t(const std::string &s);
    void override_params(bnorm_conf_t &conf) const;

private:
    std::string s_use_fused_atomics_reduction_;
    std::string s_max_vect_size_;
    std::string s_ic_block_;
    std::string s_stat_sp_block_;
    std::string s_update_sp_block_;
    std::string s_update_sp_unroll_;
};

class bnorm_lookup_table_t {
public:
    bnorm_lookup_table_t();
    const char *find(const bnorm_conf_t &conf,
            const compute::gpu_arch_t &gpu_arch) const;
    void get_params(const bnorm_conf_t &conf, const std::string &params) const;

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
