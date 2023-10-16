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

#ifndef GPU_JIT_CONV_PARAMS_HPP
#define GPU_JIT_CONV_PARAMS_HPP

#include <array>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>

#include "common/c_types_map.hpp"
#include "gpu/jit/conv/key.hpp"
#include "gpu/jit/ir/core.hpp"
#include "gpu/jit/ir/problem.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

class blocking_t {
public:
    int simd() const { return simd_; }
    const prb_tile_t &loop() const { return loop_; }
    const prb_tile_t &thread_group() const { return thread_group_; }
    const prb_tile_t &iter() const { return iter_; }

    int loop_dim(const prb_dim_t &d) const { return loop_[d]; }
    int thread_group_dim(const prb_dim_t &d) const { return thread_group_[d]; }
    int iter_dim(const prb_dim_t &d) const { return iter_[d]; }

    void set_simd(int simd) { simd_ = simd; }
    void set_loop(const prb_dim_t &d, int value) { loop_[d] = value; }
    void set_thread_group(const prb_dim_t &d, int value) {
        thread_group_[d] = value;
    }
    void set_iter(const prb_dim_t &d, int value) { iter_[d] = value; }

    bool is_empty() const {
        return loop_.is_empty() && thread_group_.is_empty() && iter_.is_empty();
    }
    bool is_spatial() const {
        for (auto d : {prb_dims::iw, prb_dims::ow}) {
            if (iter_.has(d) && iter_[d] != 1) return true;
        }
        return false;
    }

    void unset(const prb_dim_t &d) {
        if (loop_.has(d)) loop_[d] = 1;
        if (thread_group_.has(d)) thread_group_[d] = 1;
        if (iter_.has(d)) iter_[d] = 1;
    }

    bool operator==(const blocking_t &other) const {
        return (loop_ == other.loop_) && (thread_group_ == other.thread_group_)
                && (iter_ == other.iter_);
    }

    void serialize(std::ostream &out) const;
    void deserialize(std::istream &in);

    size_t get_hash() const {
        return ir_utils::get_hash(loop_, thread_group_, iter_);
    }

    std::string str(bool csv = false) const;

    IR_DEFINE_DUMP()

private:
    int simd_ = 0;
    prb_tile_t loop_;
    prb_tile_t thread_group_;
    prb_tile_t iter_;
};

struct blocking_hash_t {
    size_t operator()(const blocking_t &b) const { return b.get_hash(); }
};

class conv_config_t;

prb_tile_t get_conv_shape(const conv_config_t &cfg, bool pad);

class conv_params_t {
public:
    static const int bufs_hint_undef = -1;

    conv_params_t() = default;
    conv_params_t(const blocking_t &blocking, int bufs_hint = bufs_hint_undef)
        : blocking_(blocking), bufs_hint_(bufs_hint) {}
    conv_params_t(const conv_config_t &cfg);
    int id() const { return id_; }
    const blocking_t &blocking() const { return blocking_; }
    void set_id(int id) { id_ = id; }
    bool is_empty() const;
    void apply_to(conv_config_t &cfg) const;
    void serialize(std::ostream &out) const;
    void deserialize(std::istream &in);
    std::string str(bool csv = false) const;
    IR_DEFINE_DUMP()

    static std::vector<std::string> csv_keys();

private:
    int id_ = -1;
    blocking_t blocking_;
    int bufs_hint_ = bufs_hint_undef;
};

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
