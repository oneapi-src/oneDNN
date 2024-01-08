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

#ifndef GPU_JIT_IR_PROBLEM_HPP
#define GPU_JIT_IR_PROBLEM_HPP

#include <string>
#include <vector>

#include "gpu/jit/ir/core.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

enum class tensor_kind_t {
    undef,
    src,
    wei,
    dst,
    a,
    b,
    c,
};

std::string to_string(tensor_kind_t tensor);

enum class prb_dim_kind_t : int8_t {
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
    // Non-layout dimensions.
    sd,
    sh,
    sw,
    dd,
    dh,
    dw,
    pd,
    ph,
    pw,
    b,
    m,
    n,
    k,
    _max,
};

std::string to_string(prb_dim_kind_t kind);

inline std::ostream &operator<<(std::ostream &out, prb_dim_kind_t kind) {
    out << to_string(kind);
    return out;
}

enum class prb_dim_spatial_kind_t : uint32_t {
    undef,
    d,
    h,
    w,
};

prb_dim_spatial_kind_t to_spatial(prb_dim_kind_t kind);

template <typename KindT>
class map_key_t {
public:
    using kind_type = KindT;

    map_key_t() = default;
    map_key_t(KindT kind) : kind_(kind) {}
    KindT kind() const { return kind_; }
    int id() const { return static_cast<int>(kind_); }
    bool is_undef() const { return kind_ == KindT::undef; }
    bool is_max() const { return kind_ == KindT::_max; }
    bool operator==(const map_key_t &other) const {
        return kind_ == other.kind_;
    }
    bool operator!=(const map_key_t &other) const {
        return kind_ != other.kind_;
    }
    size_t get_hash() const { return ir_utils::get_hash(kind_); }
    void serialize(std::ostream &out) const { ir_utils::serialize(kind_, out); }
    void deserialize(std::istream &in) { ir_utils::deserialize(kind_, in); }
    std::string name() const { return str(); }
    std::string str() const { return to_string(kind_); }

    IR_DEFINE_DUMP()

    static constexpr int max_id() { return static_cast<int>(KindT::_max); }
    static map_key_t from_id(int id) {
        return map_key_t(static_cast<KindT>(id));
    }
    static map_key_t from_name(const std::string &name) {
        for (int id = 0; id < max_id(); id++) {
            auto key = from_id(id);
            if (key.name() == name) return key;
        }
        ir_error_not_expected() << name;
        return map_key_t();
    }
    static map_key_t undef() { return map_key_t(KindT::undef); }
    static map_key_t max() { return map_key_t(KindT::_max); }

    static std::vector<map_key_t> all() {
        static std::vector<map_key_t> _all_keys = [&]() {
            std::vector<map_key_t> ret;
            for (int i = 1; i < max_id(); i++)
                ret.push_back(from_id(i));
            return ret;
        }();
        return _all_keys;
    }

private:
    KindT kind_ = KindT::undef;
};

using prb_dim_t = map_key_t<prb_dim_kind_t>;

namespace prb_dims {
extern prb_dim_t undef;
extern prb_dim_t g;
extern prb_dim_t ic;
extern prb_dim_t id;
extern prb_dim_t ih;
extern prb_dim_t iw;
extern prb_dim_t kd;
extern prb_dim_t kh;
extern prb_dim_t kw;
extern prb_dim_t mb;
extern prb_dim_t oc;
extern prb_dim_t od;
extern prb_dim_t oh;
extern prb_dim_t ow;
extern prb_dim_t sd;
extern prb_dim_t sh;
extern prb_dim_t sw;
extern prb_dim_t dd;
extern prb_dim_t dh;
extern prb_dim_t dw;
extern prb_dim_t pd;
extern prb_dim_t ph;
extern prb_dim_t pw;
extern prb_dim_t b;
extern prb_dim_t m;
extern prb_dim_t n;
extern prb_dim_t k;
} // namespace prb_dims

template <typename KeyT, typename ValueT>
class dim_map_t {
public:
    class iterator_t {
    public:
        iterator_t(const dim_map_t *parent, KeyT key = KeyT::undef())
            : parent_(parent), key_(key) {
            move_next();
        }

        iterator_t &operator++() {
            move_next();
            return *this;
        }

        bool operator!=(const iterator_t &other) const {
            return (parent_ != other.parent_) || (key_ != other.key_);
        }

        const KeyT &operator*() const { return key_; }

    private:
        void move_next() {
            if (key_.is_max()) return;
            key_ = KeyT::from_id(key_.id() + 1);
            while (!key_.is_max() && !parent_->has(key_))
                key_ = KeyT::from_id(key_.id() + 1);
        }

        const dim_map_t *parent_ = nullptr;
        KeyT key_ = KeyT::max();
    };

    dim_map_t() {
        is_set_.fill(false);
        values_.fill(ValueT());
    }

    dim_map_t(const ValueT &value) {
        is_set_.fill(true);
        values_.fill(value);
    }

    dim_map_t(const std::initializer_list<KeyT> &keys) {
        is_set_.fill(false);
        values_.fill(ValueT());
        for (auto &k : keys)
            operator[](k) = 1;
    }

    dim_map_t(const std::string &s) {
        is_set_.fill(false);
        values_.fill(ValueT());
        for (auto &kv : ir_utils::to_string_int_map(s)) {
            operator[](KeyT::from_name(kv.first)) = ValueT(kv.second);
        }
    }

    virtual ~dim_map_t() = default;

    bool has(const KeyT &key) const { return is_set_[key.id()]; }

    iterator_t begin() const { return iterator_t(this); }
    iterator_t end() const { return iterator_t(this, KeyT::max()); }

    int size() const { return size_; }
    bool is_empty() const { return size_ == 0; }

    void set(const KeyT &key, const ValueT &value) {
        int idx = key.id();
        if (!is_set_[idx]) size_++;
        is_set_[idx] = true;
        values_[idx] = value;
    }

    void unset(const KeyT &key) {
        int idx = key.id();
        if (is_set_[idx]) size_--;
        is_set_[idx] = false;
        values_[idx] = ValueT();
    }

    std::vector<KeyT> keys() const {
        std::vector<KeyT> ret;
        for (auto key : *this)
            ret.push_back(key);
        return ret;
    }

    const ValueT &operator[](const KeyT &key) const {
        ir_assert(has(key));
        return values_[key.id()];
    }

    ValueT &operator[](const KeyT &key) {
        if (!has(key)) set(key, ValueT());
        return values_[key.id()];
    }

    const ValueT &at(const KeyT &key) const { return operator[](key); }

    ValueT get(const KeyT &key, const ValueT &default_value = ValueT()) const {
        if (!has(key)) return default_value;
        return at(key);
    }

    void erase(const KeyT &key) { unset(key); }

    void fill_missing(const ValueT &value) {
        for (int i = 1; i < KeyT::max_id(); i++) {
            if (is_set_[i]) continue;
            set(KeyT::from_id(i), value);
        }
    }

    std::unordered_map<std::string, int> to_map() const {
        std::unordered_map<std::string, int> ret;
        for (auto d : (*this)) {
            ret[d.name()] = at(d);
        }
        return ret;
    }

    bool operator==(const dim_map_t &other) const {
        if (is_set_ != other.is_set_) return false;
        if (values_ != other.values_) return false;
        if (size_ != other.size_) return false;
        return true;
    }

    bool operator!=(const dim_map_t &other) const { return !operator==(other); }

    size_t get_hash() const {
        return ir_utils::get_hash(is_set_, values_, size_);
    }

    void serialize(std::ostream &out) const {
        using key_int_type =
                typename std::underlying_type<typename KeyT::kind_type>::type;
        ir_utils::serialize(size_, out);
        for (int i = 0; i < KeyT::max_id(); i++) {
            if (!is_set_[i]) continue;
            ir_utils::serialize((key_int_type)i, out);
            // To keep binary compatibility with the old version.
            ir_utils::serialize((key_int_type)i, out);
            ir_utils::serialize(values_[i], out);
        }
    }

    void deserialize(std::istream &in) {
        using key_int_type =
                typename std::underlying_type<typename KeyT::kind_type>::type;
        ir_utils::deserialize(size_, in);
        is_set_.fill(false);
        values_.fill(ValueT());
        for (int j = 0; j < size_; j++) {
            auto i = ir_utils::deserialize<key_int_type>(in);
            (void)ir_utils::deserialize<key_int_type>(in);
            is_set_[i] = true;
            ir_utils::deserialize(values_[i], in);
        }
    }

    std::string str_impl(bool multiline) const {
        if (size_ == 0) return "x";
        std::ostringstream oss;
        bool is_first = true;
        for (auto &d : *this) {
            auto &value = operator[](d);
            if (multiline) {
                if (!is_first) oss << std::endl;
                oss << std::setw(4) << d << ": "
                    << ir_utils::str_helper_t<ValueT>::call(value);
                is_first = false;
            } else {
                oss << d << ir_utils::str_helper_t<ValueT>::call(value);
            }
        }
        return oss.str();
    }

    virtual std::string str() const { return str_impl(/*multiline=*/true); }

    IR_DEFINE_DUMP()

private:
    std::array<bool, KeyT::max_id()> is_set_;
    std::array<ValueT, KeyT::max_id()> values_;
    int size_ = 0;
};

template <typename KeyT>
class tile_t : public dim_map_t<KeyT, int> {
public:
    using dim_map_t<KeyT, int>::at;
    using dim_map_t<KeyT, int>::dim_map_t;
    using dim_map_t<KeyT, int>::has;
    using dim_map_t<KeyT, int>::operator[];
    using dim_map_t<KeyT, int>::str_impl;

    int elems() const {
        int ret = 1;
        for (auto &d : *this)
            ret *= at(d);
        return ret;
    }

    bool try_factor(const prb_dim_t &dim, int factor) {
        if (factor == 1) return true;
        if (!has(dim)) return false;
        int &value = operator[](dim);
        if (value % factor != 0) return false;
        value /= factor;
        return true;
    }

    std::string str() const override { return str_impl(/*multiline=*/false); }
};

template <typename ValueT>
class prb_coord_t : public dim_map_t<prb_dim_t, ValueT> {
public:
    using dim_map_t<prb_dim_t, ValueT>::dim_map_t;
};

template <typename T1, typename T2>
struct coord_add_type_t {
    using type = expr_t;
};

template <>
struct coord_add_type_t<int, int> {
    using type = int;
};

template <typename T1, typename T2,
        typename T = typename coord_add_type_t<T1, T2>::type>
inline prb_coord_t<T> operator+(
        const prb_coord_t<T1> &a, const prb_coord_t<T2> &b) {
    prb_coord_t<T> ret;
    for (auto &d : a) {
        ret[d] = a.get(d, T1(0)) + b.get(d, T2(0));
    }
    for (auto &d : b) {
        if (ret.has(d)) continue;
        ret[d] = a.get(d, T1(0)) + b.get(d, T2(0));
    }
    return ret;
}

using prb_tile_t = tile_t<prb_dim_t>;

inline prb_tile_t str_to_prb_tile(const std::string &s) {
    if (s.empty() || s == "x") return prb_tile_t();
    return prb_tile_t(s);
}

template <typename T>
bool has_spatial(const dim_map_t<prb_dim_t, T> &map,
        prb_dim_spatial_kind_t spatial_kind) {
    for (auto &d : prb_dim_t::all()) {
        if (to_spatial(d.kind()) == spatial_kind && map.has(d)) return true;
    }
    return false;
}

const expr_t &index_var(const prb_dim_t &prb_dim);
const expr_t &size_var(const prb_dim_t &prb_dim);
prb_dim_t index_to_prb_dim(const expr_t &var);
prb_dim_t size_to_prb_dim(const expr_t &var);

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
