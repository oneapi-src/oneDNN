/*******************************************************************************
* Copyright 2023-2025 Intel Corporation
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

#include "gpu/intel/jit/conv/lookup_table.hpp"

#include <mutex>

#include "common/utils.hpp"
#include "gpu/intel/jit/utils/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

const char **get_conv_lookup_table_entries();

void conv_lookup_table_t::entry_t::stringify(std::ostream &out) const {
    key.stringify(out);
    out << " ";
    params.stringify(out);
}

void conv_lookup_table_t::entry_t::parse(std::istream &in) {
    key.parse(in);
    params.parse(in);
}

conv_lookup_table_t::conv_lookup_table_t(const char **entries) {
    while (*entries) {
        conv_lookup_table_t::entry_t e;
        std::istringstream iss(*entries);
        e.parse(iss);
#ifdef DNNL_DEV_MODE
        {
            std::ostringstream oss;
            e.stringify(oss);
            gpu_assert(oss.str() == *entries)
                    << "parsed from:\n  " << *entries << "\nstringified to\n  "
                    << oss.str();
        }
#endif
        set(e.key, e.params);
        entries++;
    }
}

void conv_lookup_table_t::set(
        const conv_key_t &key, const blocking_params_t &params) {
    auto &desc_entries = data_[key.desc()];
    for (auto &e : desc_entries) {
        if (e.key == key) {
            e.params = params;
            return;
        }
    }
    desc_entries.push_back(entry_t {key, params});
}

void conv_lookup_table_t::merge(const conv_lookup_table_t &other) {
    for (auto &kv : other.data_) {
        for (auto &e : kv.second) {
            set(e.key, e.params);
        }
    }
}

blocking_params_t conv_lookup_table_t::find(const conv_key_t &key) const {
    auto entries_it = data_.find(key.desc());
    if (entries_it == data_.end()) return blocking_params_t();
    auto &desc_entries = entries_it->second;
    auto it = desc_entries.begin();
    auto best = desc_entries.end();
    dim_t best_dist = std::numeric_limits<dim_t>::max();
    for (; it != desc_entries.end(); it++) {
        if (!it->key.matches(key)) continue;
        dim_t dist = it->key.distance(key);
        if (dist < best_dist) {
            best_dist = dist;
            best = it;
        }
    }
    return (best == desc_entries.end()) ? blocking_params_t() : best->params;
}

void conv_lookup_table_t::stringify(std::ostream &out) const {
    bool is_first = true;
    for (auto &kv : data_) {
        for (auto &e : kv.second) {
            if (!is_first) out << "\n";
            e.stringify(out);
            is_first = false;
        }
    }
}

void conv_lookup_table_t::parse(std::istream &in) {
    data_.clear();
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        entry_t e;
        jit::parse(line, e);
        data_[e.key.desc()].push_back(e);
    }
}

struct conv_lookup_table_instance_t {
    conv_lookup_table_instance_t() {
        table = conv_lookup_table_t(get_conv_lookup_table_entries());
#ifdef DNNL_DEV_MODE
        table_path = getenv_string_user(env_table_path_name);
#endif
        if (!table_path.empty()) {
            std::ifstream in(table_path);
            if (!in.good()) return;
            conv_lookup_table_t file_table;
            file_table.parse(in);
            table.merge(file_table);
        }
    }

    ~conv_lookup_table_instance_t() {
        if (table_path.empty()) return;
        std::ofstream out(table_path, std::ios::binary);
        table.stringify(out);
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
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
