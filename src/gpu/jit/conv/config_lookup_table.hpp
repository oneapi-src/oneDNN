/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#ifndef GPU_JIT_CONV_CONFIG_LOOKUP_TABLE_HPP
#define GPU_JIT_CONV_CONFIG_LOOKUP_TABLE_HPP

#include <string>
#include <vector>
#include <unordered_map>

#include "common/c_types_map.hpp"
#include "gpu/jit/ir/core.hpp"
#include "gpu/jit/ir/hw_config.hpp"
#include "gpu/jit/ngen/ngen_core.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

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

    bool matches(const std::vector<data_type_t> &values) const;

private:
    bool try_parse(
            const std::string &s, size_t &pos, const std::string &pattern);

    static std::vector<std::string> &all_patterns();

    std::vector<std::string> patterns_;
};

class fpmath_filter_t {
public:
    fpmath_filter_t() = default;

    bool matches(fpmath_mode_t mode) const { return mode == filter_; }

private:
    fpmath_mode_t filter_ = fpmath_mode::strict;
};

class conv_problem_t;

class conv_problem_filter_t {
public:
    using key_t = std::string;

    conv_problem_filter_t(const std::string &s);

    key_t key() const { return desc_; }

    bool matches(const conv_problem_t &prb, const hw_config_t &hw_cfg) const;

private:
    bool matches_dir(const conv_problem_t &prb) const;

    bool matches_desc(const conv_problem_t &prb) const;

    bool matches_post_ops(const conv_problem_t &prb) const;

    std::string dir_;
    type_filter_t type_filter_;
    fpmath_filter_t fpmath_filter_;
    int_filter_t mb_filter_;
    std::string desc_;
    std::string post_ops_;
    ngen::HW hw_;
};

class conv_config_t;

class conv_config_lookup_table_t {
public:
    conv_config_lookup_table_t();

    const char *find(const conv_config_t &cfg) const;

private:
    struct entry_t {
        conv_problem_filter_t filter;
        const char *s_params;
    };

    void add(const char *s_prb, const char *s_params);

    using key_t = conv_problem_filter_t::key_t;
    std::unordered_map<key_t, std::vector<entry_t>> map_;
};

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
