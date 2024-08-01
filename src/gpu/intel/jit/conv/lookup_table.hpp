/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
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

#include <iostream>
#include <unordered_map>

#include "gpu/intel/jit/conv/key.hpp"
#include "gpu/intel/jit/ir/blocking.hpp"
#include "gpu/intel/jit/ir/problem.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

class conv_lookup_table_t {
public:
    struct entry_t {
        conv_key_t key;
        blocking_params_t params;

        void stringify(std::ostream &out) const;
        void parse(std::istream &in);
    };

    conv_lookup_table_t() = default;
    conv_lookup_table_t(const char **entries);

    void set(const conv_key_t &key, const blocking_params_t &params);
    void merge(const conv_lookup_table_t &other);
    blocking_params_t find(const conv_key_t &key) const;
    bool is_empty() const { return data_.empty(); }
    void stringify(std::ostream &out) const;
    void parse(std::istream &in);

private:
    std::unordered_map<std::string, std::vector<entry_t>> data_;
};

const conv_lookup_table_t &const_conv_lookup_table();
conv_lookup_table_t &conv_lookup_table();

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
