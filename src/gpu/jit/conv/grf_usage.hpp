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

#ifndef GPU_JIT_CONV_GRF_USAGE_HPP
#define GPU_JIT_CONV_GRF_USAGE_HPP

#include <unordered_map>

#include "gpu/jit/conv/config.hpp"
#include "gpu/jit/ir/ir.hpp"
#include "gpu/jit/utils/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

enum class grf_usage_label_t {
    unknown,
    gmem_load,
    out_buf,
    reorder,
    reserved,
    reused_headers,
    slm_load,
    slm_store,
    tmp_vars,
    zero_points,
    _last,
};

inline std::vector<grf_usage_label_t> all_grf_usage_labels() {
    std::vector<grf_usage_label_t> ret;
    for (int i = 0; i < (int)grf_usage_label_t::_last; i++) {
        ret.push_back((grf_usage_label_t)i);
    }
    return ret;
}

std::string to_string(grf_usage_label_t label);
std::ostream &operator<<(std::ostream &out, grf_usage_label_t label);

class grf_buf_usage_t {
public:
    grf_buf_usage_t(int grf_size) : grf_size_(grf_size) {}

    const object_set_t<expr_t> &bufs() const { return bufs_; }

    std::vector<expr_t> sorted_bufs() const {
        std::vector<expr_t> ret(bufs_.begin(), bufs_.end());
        std::sort(ret.begin(), ret.end(), [](const expr_t &a, const expr_t &b) {
            return a.as<var_t>().name < b.as<var_t>().name;
        });
        return ret;
    }

    bool has(const expr_t &buf) const { return bufs_.find(buf) != bufs_.end(); }

    grf_usage_label_t get_label(const expr_t &buf) const {
        auto it = buf_labels_.find(buf);
        ir_assert(it != buf_labels_.end()) << "Buffer not found: " << buf;
        return it->second;
    }

    int get_size(const expr_t &buf) const {
        auto it = buf_sizes_.find(buf);
        ir_assert(it != buf_sizes_.end()) << "Buffer not found: " << buf;
        return it->second;
    }

    void set_label(const expr_t &buf, grf_usage_label_t label) {
        buf_labels_[buf] = label;
    }

    void add(const expr_t &buf, int size, grf_usage_label_t label) {
        bufs_.insert(buf);
        buf_labels_.emplace(buf, label);
        buf_sizes_.emplace(buf, size);
    }

    void remove(const expr_t &buf) {
        bufs_.erase(buf);
        buf_labels_.erase(buf);
        buf_sizes_.erase(buf);
    }

    int total_regs(grf_usage_label_t label) const {
        int ret = 0;
        for (auto &kv : buf_labels_) {
            if (kv.second != label) continue;
            ret += utils::div_up(buf_sizes_.at(kv.first), grf_size_);
        }
        return ret;
    }

    std::string str() const;

private:
    int grf_size_;
    object_set_t<expr_t> bufs_;
    object_map_t<expr_t, int> buf_sizes_;
    object_map_t<expr_t, grf_usage_label_t> buf_labels_;
};

class grf_usage_t {
public:
    grf_usage_t(int grf_size = 0) : grf_size_(grf_size), buf_usage_(grf_size) {
        for (auto label : all_grf_usage_labels()) {
            regs_.emplace(label, 0);
        }
    }

    bool is_empty() const {
        for (auto &kv : regs_)
            if (kv.second != 0) return false;
        return true;
    }

    void add(grf_usage_label_t label, int regs) { regs_[label] += regs; }

    void add(const expr_t &buf, int size, grf_usage_label_t label) {
        add(label, utils::div_up(size, grf_size_));
        buf_usage_.add(buf, size, label);
    }

    void add(const grf_buf_usage_t &buf_usage) {
        for (auto &buf : buf_usage.bufs()) {
            add(buf, buf_usage.get_size(buf), buf_usage.get_label(buf));
        }
    }

    int get(grf_usage_label_t label) const { return regs_.at(label); }

    int total() const {
        int ret = 0;
        for (auto &kv : regs_)
            ret += kv.second;
        return ret;
    }

    const grf_buf_usage_t &buf_usage() const { return buf_usage_; }

    std::string str() const;

    IR_DEFINE_DUMP()

private:
    int grf_size_;

    using label_hash_t = ir_utils::enum_hash_t<grf_usage_label_t>;
    std::unordered_map<grf_usage_label_t, int, label_hash_t> regs_;
    grf_buf_usage_t buf_usage_;
};

grf_usage_t get_grf_usage(const stmt_t &body, int grf_size);

void verify_grf_usage(
        const conv_config_t &cfg, const stmt_t &body, int external_usage);

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
