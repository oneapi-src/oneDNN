/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "gpu/jit/conv/key.hpp"
#include "gpu/jit/ir/blocking.hpp"
#include "gpu/jit/ir/problem.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

class conv_lookup_table_t {
public:
    void set(const conv_key_t &key, const blocking_params_t &params) {
        auto it = data_.find(key);
        if (it == data_.end()) {
            data_.emplace(key, params);
            return;
        }
        for (; it != data_.end(); it++) {
            if (it->first == key) {
                it->second = params;
                return;
            }
        }
        data_.emplace(key, params);
    }
    void merge(const conv_lookup_table_t &other);
    blocking_params_t find(const conv_key_t &key) const;
    bool is_empty() const { return data_.empty(); }
    void serialize(std::ostream &out) const;
    void deserialize(std::istream &in);

private:
    std::unordered_multimap<conv_key_t, blocking_params_t, conv_key_hash_t,
            conv_key_lookup_table_equal_t>
            data_;
};

const conv_lookup_table_t &const_conv_lookup_table();
conv_lookup_table_t &conv_lookup_table();

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
