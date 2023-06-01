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
#include <sstream>
#include <string>
#include <unordered_map>

#include "common/c_types_map.hpp"
#include "gpu/jit/conv/key.hpp"
#include "gpu/jit/ir/core.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

enum class gemm_dim_kind_t {
    undef = 0,
    b,
    m,
    n,
    k,
    _max,
};

std::string to_string(gemm_dim_kind_t kind);

enum class conv_dim_kind_t : int8_t {
    undef = 0,
    g,
    ic,
    id,
    ih,
    iw,
    kd,
    kh,
    kw,
    mb,
    oc,
    od,
    oh,
    ow,
    _max,
};

std::string to_string(conv_dim_kind_t kind);

template <typename KindT>
class tile_key_t {
public:
    using kind_type = KindT;

    tile_key_t() = default;
    tile_key_t(KindT kind) : kind_(kind) {}
    KindT kind() const { return kind_; }
    int id() const { return static_cast<int>(kind_); }
    bool is_undef() const { return kind_ == KindT::undef; }
    bool operator==(const tile_key_t &other) const {
        return kind_ == other.kind_;
    }
    bool operator!=(const tile_key_t &other) const {
        return kind_ != other.kind_;
    }
    std::string name() const { return str(); }
    std::string str() const { return to_string(kind_); }

    IR_DEFINE_DUMP()

    static constexpr int max_id() { return static_cast<int>(KindT::_max); }
    static tile_key_t from_id(int id) {
        return tile_key_t(static_cast<KindT>(id));
    }
    static tile_key_t from_name(const std::string &name) {
        for (int id = 0; id < max_id(); id++) {
            auto key = from_id(id);
            if (key.name() == name) return key;
        }
        ir_error_not_expected() << name;
        return tile_key_t();
    }
    static tile_key_t undef() { return tile_key_t(KindT::undef); }
    static tile_key_t max() { return tile_key_t(KindT::_max); }

private:
    KindT kind_ = KindT::undef;
};

using gemm_dim_t = tile_key_t<gemm_dim_kind_t>;
using conv_dim_t = tile_key_t<conv_dim_kind_t>;

namespace gemm_dims {
extern gemm_dim_t b;
extern gemm_dim_t m;
extern gemm_dim_t n;
extern gemm_dim_t k;
} // namespace gemm_dims

namespace conv_dims {
extern conv_dim_t g;
extern conv_dim_t ic;
extern conv_dim_t id;
extern conv_dim_t ih;
extern conv_dim_t iw;
extern conv_dim_t kd;
extern conv_dim_t kh;
extern conv_dim_t kw;
extern conv_dim_t mb;
extern conv_dim_t oc;
extern conv_dim_t od;
extern conv_dim_t oh;
extern conv_dim_t ow;
} // namespace conv_dims

const std::vector<conv_dim_t> &get_conv_dims(prop_kind_t prop);

template <typename KeyT>
class tile_generic_t {
public:
    class iterator_t {
    public:
        iterator_t(const tile_generic_t *parent, KeyT key = KeyT::undef())
            : parent_(parent) {
            if (key.is_undef()) {
                for (int i = 0; i < KeyT::max_id(); i++) {
                    auto i_key = KeyT::from_id(i);
                    if (parent_->has(i_key)) {
                        key_ = i_key;
                        break;
                    }
                }
            }
        }

        iterator_t &operator++() {
            for (int i = key_.id() + 1; i < KeyT::max_id(); i++) {
                auto i_key = KeyT::from_id(i);
                if (parent_->has(i_key)) {
                    key_ = i_key;
                    return *this;
                }
            }
            key_ = KeyT::max();
            return *this;
        }

        bool operator!=(const iterator_t &other) const {
            return (parent_ != other.parent_) || (key_ != other.key_);
        }

        KeyT operator*() const { return key_; }

    private:
        const tile_generic_t *parent_ = nullptr;
        KeyT key_ = KeyT::max();
    };

    tile_generic_t() {}

    tile_generic_t(const std::initializer_list<KeyT> &keys) {
        for (auto &k : keys)
            operator[](k) = 1;
    }

    bool has(const KeyT &key) const { return entries_[key.id()].index != -1; }

    iterator_t begin() const { return iterator_t(this); }
    iterator_t end() const { return iterator_t(this, KeyT::max()); }

    bool is_empty() const { return nkeys_ == 0; }

    std::vector<KeyT> keys() const {
        std::vector<KeyT> ret;
        for (auto key : *this)
            ret.push_back(key);
        return ret;
    }

    size_t get_hash() const {
        size_t h = 0;
        for (int i = 0; i < KeyT::max_id(); i++) {
            auto key = KeyT::from_id(i);
            if (has(key)) {
                auto &e = entries_[key.id()];
                h = hash_combine(h, ir_utils::get_hash(e.index, e.value));
            }
        }
        return h;
    }

    bool operator==(const tile_generic_t &other) const {
        for (int i = 0; i < KeyT::max_id(); i++) {
            auto key = KeyT::from_id(i);
            if (has(key) != other.has(key)) return false;
            if (has(key) && operator[](key) != other.operator[](key))
                return false;
        }
        return true;
    }

    const int &operator[](const KeyT &key) const {
        ir_assert(has(key));
        return entries_[key.id()].value;
    }

    int &operator[](const KeyT &key) {
        auto &e = entries_[key.id()];
        if (!has(key)) e.index = nkeys_++;
        return e.value;
    }

    int at(const KeyT &key, int default_value = 0) const {
        if (!has(key)) return default_value;
        return operator[](key);
    }

    void erase(const KeyT &key) {
        ir_assert(has(key));
        entries_[key.id()].index = -1;
        nkeys_--;
    }

    void serialize(std::ostream &out) const;
    void deserialize(std::istream &in);

    std::unordered_map<std::string, int> to_map() const {
        std::unordered_map<std::string, int> ret;
        for (auto d : (*this)) {
            ret[d.name()] = at(d);
        }
        return ret;
    }

    std::string str() const {
        std::ostringstream oss;
        for (int i = 0; i < KeyT::max_id(); i++) {
            auto key = KeyT::from_id(i);
            if (!has(key)) continue;
            oss << key << entries_[i].value;
        }
        return oss.str();
    }

    IR_DEFINE_DUMP()

private:
    struct entry_t {
        int index = -1;
        int value;
    };

    std::array<entry_t, KeyT::max_id()> entries_;
    int nkeys_ = 0;
};

using gemm_tile_t = tile_generic_t<gemm_dim_t>;
using conv_tile_t = tile_generic_t<conv_dim_t>;

gemm_dim_t to_gemm(const conv_dim_t &d, prop_kind_t prop, bool is_transpose);
gemm_tile_t to_gemm(const conv_tile_t &t, prop_kind_t prop, bool is_transpose);

class blocking_t {
public:
    int simd() const { return simd_; }
    const conv_tile_t &loop() const { return loop_; }
    const conv_tile_t &thread_group() const { return thread_group_; }
    const conv_tile_t &iter() const { return iter_; }

    int loop_dim(const conv_dim_t &d) const { return loop_[d]; }
    int thread_group_dim(const conv_dim_t &d) const { return thread_group_[d]; }
    int iter_dim(const conv_dim_t &d) const { return iter_[d]; }

    void set_simd(int simd) { simd_ = simd; }
    void set_loop(const conv_dim_t &d, int value) { loop_[d] = value; }
    void set_thread_group(const conv_dim_t &d, int value) {
        thread_group_[d] = value;
    }
    void set_iter(const conv_dim_t &d, int value) { iter_[d] = value; }

    bool is_empty() const {
        return loop_.is_empty() && thread_group_.is_empty() && iter_.is_empty();
    }
    bool is_spatial() const {
        for (auto d : {conv_dims::iw, conv_dims::ow}) {
            if (iter_.has(d) && iter_[d] != 1) return true;
        }
        return false;
    }

    void unset(const conv_dim_t &d) {
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
    conv_tile_t loop_;
    conv_tile_t thread_group_;
    conv_tile_t iter_;
};

struct blocking_hash_t {
    size_t operator()(const blocking_t &b) const { return b.get_hash(); }
};

class conv_config_t;

conv_tile_t get_conv_shape(const conv_config_t &cfg, bool pad);

class conv_params_t {
public:
    conv_params_t() = default;
    conv_params_t(const blocking_t &blocking) : blocking_(blocking) {}
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
    int bufs_ = -1;
};

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
