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

#include "gpu/jit/conv/lookup_table.hpp"

#include <mutex>

#include "common/utils.hpp"
#include "gpu/jit/conv/params.hpp"
#include "gpu/jit/utils/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

const std::vector<uint64_t> &get_conv_lookup_table_data();

void conv_lookup_table_t::merge(const conv_lookup_table_t &other) {
    for (auto &kv : other.data_) {
        set(kv.first, kv.second);
    }
}

conv_params_t conv_lookup_table_t::find(const conv_key_t &key) const {
    auto it = data_.find(key);
    auto best = data_.end();
    int best_dist = std::numeric_limits<int>::max();
    for (; it != data_.end(); it++) {
        if (!it->first.matches(key)) continue;
        int dist = it->first.distance(key);
        if (dist < best_dist) {
            best_dist = dist;
            best = it;
        }
    }
    return (best == data_.end()) ? conv_params_t() : best->second;
}

void conv_lookup_table_t::serialize(std::ostream &out) const {
    ir_utils::serialize(data_.size(), out);
    for (auto &kv : data_) {
        kv.first.serialize(out);
        kv.second.serialize(out);
    }
}

void conv_lookup_table_t::deserialize(std::istream &in) {
    auto n = ir_utils::deserialize<size_t>(in);
    for (size_t i = 0; i < n; i++) {
        conv_key_t key;
        conv_params_t params;
        key.deserialize(in);
        params.deserialize(in);
        data_.emplace(key, params);
    }
}

struct conv_lookup_table_instance_t {
    conv_lookup_table_instance_t() {
        table = ir_utils::deserialize_from_data<conv_lookup_table_t>(
                get_conv_lookup_table_data());
#ifdef DNNL_DEV_MODE
        table_path = getenv_string_user(env_table_path_name);
#endif
        if (!table_path.empty()) {
            std::ifstream in(table_path, std::ios::binary);
            if (!in.good()) return;
            conv_lookup_table_t file_table;
            file_table.deserialize(in);
            table.merge(file_table);
        }
    }

    ~conv_lookup_table_instance_t() {
        if (table_path.empty()) return;
        std::ofstream out(table_path, std::ios::binary);
        table.serialize(out);
    }

    static const char *env_table_path_name;
    std::string table_path;
    conv_lookup_table_t table;
};

const char *conv_lookup_table_instance_t::env_table_path_name
        = "GPU_CONV_LOOKUP_TABLE_PATH";

conv_lookup_table_t &conv_lookup_table_impl(bool read_only = true) {
    static conv_lookup_table_instance_t instance;
    if (!read_only && instance.table_path.empty()) {
        static std::once_flag flag;
        std::call_once(flag, [&] {
            printf("Warning: %s is not set. All tuning data will be lost.\n",
                    conv_lookup_table_instance_t::env_table_path_name);
        });
    }
    return instance.table;
}

const conv_lookup_table_t &const_conv_lookup_table() {
    return conv_lookup_table_impl();
}
conv_lookup_table_t &conv_lookup_table() {
    return conv_lookup_table_impl(/*read_only=*/false);
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
