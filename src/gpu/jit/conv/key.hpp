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

#ifndef GPU_JIT_CONV_KEY_HPP
#define GPU_JIT_CONV_KEY_HPP

#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gpu/jit/ir/core.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

class conv_config_t;
class conv_key_impl_t;

// Represents a key with hash/equality functionality for convolution problems.
// Mainly used for the lookup table with key -> <optimal convolution parameters>
// mapping.
// When used for lookup tables a conv_key_t object represents a filter which is
// used with matches() API.
// Examples of differences between key and filter:
// 1) Type:
//    - Convolution problem: s8s8s32
//    - Filter:              x8x8*
// 2) Batch size
//    - Convolution problem: mb32(blocked)
//    - Filter:              mb32+(blocked)
class conv_key_t {
public:
    conv_key_t() = default;
    conv_key_t(const conv_config_t &cfg, bool make_filter = false);
    // Makes a filter from the given key.
    conv_key_t to_filter() const;
    // Computes the distance between this key and other key (must be
    // non-filter), a filter with a smaller distance is a better match for the
    // key.
    int distance(const conv_key_t &other) const;
    bool operator==(const conv_key_t &other) const;
    bool matches(const conv_key_t &other) const;
    bool is_desc_equal(const conv_key_t &other) const;
    size_t get_hash() const;
    void serialize(std::ostream &out) const;
    void deserialize(std::istream &in);
    std::string str(bool csv = false) const;
    static std::vector<std::string> csv_keys();

    IR_DEFINE_DUMP()

private:
    std::shared_ptr<conv_key_impl_t> impl_;
};

struct conv_key_hash_t {
    size_t operator()(const conv_key_t &key) const { return key.get_hash(); }
};

struct conv_key_lookup_table_equal_t {
    bool operator()(const conv_key_t &a, const conv_key_t &b) const {
        return a.is_desc_equal(b);
    }
};

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
