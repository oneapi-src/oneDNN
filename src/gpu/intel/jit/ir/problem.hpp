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

#ifndef GPU_INTEL_JIT_IR_PROBLEM_HPP
#define GPU_INTEL_JIT_IR_PROBLEM_HPP

#include <string>
#include <vector>

#include "gpu/intel/jit/ir/core.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

enum class tensor_kind_t {
    undef,
    src,
    wei,
    dst,
    bias,
    a,
    b,
    c,
};

std::string to_string(tensor_kind_t tensor);

class pvar_t {
public:
    pvar_t() = default;
    pvar_t(const std::string &name) : name_(name) { gpu_assert(!name.empty()); }
    const std::string &name() const { return name_; }
    bool is_undef() const { return name_.empty(); }
    bool operator==(const pvar_t &other) const { return name_ == other.name_; }
    bool operator!=(const pvar_t &other) const { return name_ != other.name_; }
    bool operator<(const pvar_t &other) const { return name_ < other.name_; }
    size_t get_hash() const { return ir_utils::get_hash(name_); }
    std::string str() const { return name_; }

    IR_DEFINE_DUMP()

    const expr_t &index_var() const;
    const expr_t &var() const;
    static pvar_t from_index_var(const expr_t &ndex_var);
    static pvar_t from_var(const expr_t &var);

    char to_spatial() const;
    int spatial_index() const;

private:
    std::string name_;
};

namespace pvars {
extern pvar_t g;
extern pvar_t ic;
extern pvar_t id;
extern pvar_t ih;
extern pvar_t iw;
extern pvar_t kd;
extern pvar_t kh;
extern pvar_t kw;
extern pvar_t mb;
extern pvar_t oc;
extern pvar_t od;
extern pvar_t oh;
extern pvar_t ow;
extern pvar_t sd;
extern pvar_t sh;
extern pvar_t sw;
extern pvar_t dd;
extern pvar_t dh;
extern pvar_t dw;
extern pvar_t pd;
extern pvar_t ph;
extern pvar_t pw;
extern pvar_t b;
extern pvar_t m;
extern pvar_t n;
extern pvar_t k;
} // namespace pvars

template <typename ValueT>
class pvar_map_t {
public:
    class iterator_t {
    public:
        iterator_t(typename std::map<pvar_t, ValueT>::const_iterator it)
            : it_(it) {}

        iterator_t &operator++() {
            it_++;
            return *this;
        }
        bool operator!=(const iterator_t &other) const {
            return it_ != other.it_;
        }

        const pvar_t &operator*() const { return it_->first; }

    private:
        typename std::map<pvar_t, ValueT>::const_iterator it_;
    };

    pvar_map_t() = default;
    pvar_map_t(const std::initializer_list<pvar_t> &keys, const ValueT &value) {
        for (auto &k : keys)
            operator[](k) = value;
    }

    explicit pvar_map_t(const std::string &s) {
        for (auto &kv : ir_utils::to_string_int_pairs(s)) {
            operator[](pvar_t(kv.first)) = ValueT(kv.second);
        }
    }

    virtual ~pvar_map_t() = default;

    bool has(const pvar_t &key) const { return map_.count(key) != 0; }
    iterator_t begin() const { return iterator_t(map_.begin()); }
    iterator_t end() const { return iterator_t(map_.end()); }
    int size() const { return (int)map_.size(); }
    bool is_empty() const { return map_.empty(); }

    void set(const pvar_t &key, const ValueT &value) { map_[key] = value; }

    void unset(const pvar_t &key) {
        if (!has(key)) return;
        map_.erase(key);
    }

    std::vector<pvar_t> keys() const {
        std::vector<pvar_t> ret;
        for (auto &key : *this)
            ret.push_back(key);
        return ret;
    }

    const ValueT &operator[](const pvar_t &key) const {
        gpu_assert(has(key)) << "Key not found: " << key;
        return map_.at(key);
    }

    ValueT &operator[](const pvar_t &key) {
        if (!has(key)) set(key, ValueT());
        return map_[key];
    }

    const ValueT &at(const pvar_t &key) const { return operator[](key); }

    ValueT get(
            const pvar_t &key, const ValueT &default_value = ValueT()) const {
        if (!has(key)) return default_value;
        return at(key);
    }

    void erase(const pvar_t &key) { map_.erase(key); }

    std::unordered_map<std::string, dim_t> to_string_map() const {
        std::unordered_map<std::string, dim_t> ret;
        for (auto &kv : map_)
            ret[kv.first.name()] = kv.second;
        return ret;
    }

    pvar_map_t operator|(const pvar_map_t &other) const {
        pvar_map_t ret = *this;
        for (auto &kv : other.map_) {
            auto it = map_.find(kv.first);
            if (it != map_.end()) {
                gpu_assert(it->second == kv.second);
                continue;
            }
            ret[kv.first] = kv.second;
        }
        return ret;
    }

    bool operator==(const pvar_map_t &other) const {
        if (size() != other.size()) return false;
        auto it1 = map_.begin();
        auto it2 = other.map_.begin();
        for (int i = 0; i < size(); i++) {
            if (*it1 != *it2) return false;
            it1++;
            it2++;
        }
        return true;
    }

    bool operator!=(const pvar_map_t &other) const {
        return !operator==(other);
    }

    size_t get_hash() const { return ir_utils::get_hash(map_); }

    void stringify(std::ostream &out) const {
        if (is_empty()) {
            out << "x";
            return;
        }
        for (auto &d : *this) {
            auto &value = operator[](d);
            out << d << value;
        }
    }

    void parse(std::istream &in) {
        auto s = stream_parse<std::string>(in);
        if (s == "x") return;
        for (auto &kv : ir_utils::to_string_int_pairs(s)) {
            operator[](pvar_t(kv.first)) = ValueT(kv.second);
        }
    }

    std::string str_impl(bool multiline) const {
        if (is_empty()) return "x";
        std::ostringstream oss;
        bool is_first = true;
        for (auto &kv : map_) {
            auto &p = kv.first;
            auto &value = kv.second;
            if (multiline) {
                if (!is_first) oss << std::endl;
                oss << std::setw(4) << p << ": "
                    << ir_utils::str_helper_t<ValueT>::call(value);
                is_first = false;
            } else {
                oss << p << ir_utils::str_helper_t<ValueT>::call(value);
            }
        }
        return oss.str();
    }

    virtual std::string str() const {
        return str_impl(/*multiline=*/!std::is_integral<ValueT>::value);
    }

    IR_DEFINE_DUMP()

private:
    std::map<pvar_t, ValueT> map_;
};

class pvar_tile_t : public pvar_map_t<dim_t> {
public:
    using pvar_map_t<dim_t>::at;
    using pvar_map_t<dim_t>::pvar_map_t;
    using pvar_map_t<dim_t>::has;
    using pvar_map_t<dim_t>::operator[];
    using pvar_map_t<dim_t>::str_impl;
    using pvar_map_t<dim_t>::get_hash;

    dim_t elems() const {
        dim_t ret = 1;
        for (auto &d : *this)
            ret *= at(d);
        return ret;
    }

    bool try_factor(const pvar_t &key, dim_t factor) {
        if (factor == 1) return true;
        if (!has(key)) return false;
        dim_t &value = operator[](key);
        if (value % factor != 0) return false;
        value /= factor;
        return true;
    }
#if __cplusplus >= 202002L
    bool operator==(const pvar_tile_t &other) const = default;
#endif

    std::string str() const override { return str_impl(/*multiline=*/false); }
};

template <typename ValueT>
class pvar_coord_t : public pvar_map_t<ValueT> {
public:
    using pvar_map_t<ValueT>::pvar_map_t;
};

template <typename T1, typename T2>
struct coord_add_type_t {
    using type = expr_t;
};

template <>
struct coord_add_type_t<int, int> {
    using type = int;
};

template <>
struct coord_add_type_t<dim_t, dim_t> {
    using type = dim_t;
};

template <>
struct coord_add_type_t<dim_t, int> {
    using type = dim_t;
};

template <>
struct coord_add_type_t<int, dim_t> {
    using type = dim_t;
};

template <typename T1, typename T2,
        typename T = typename coord_add_type_t<T1, T2>::type>
inline pvar_coord_t<T> operator+(
        const pvar_coord_t<T1> &a, const pvar_coord_t<T2> &b) {
    pvar_coord_t<T> ret;
    for (auto &d : a) {
        ret[d] = a.get(d, T1(0)) + b.get(d, T2(0));
    }
    for (auto &d : b) {
        if (ret.has(d)) continue;
        ret[d] = a.get(d, T1(0)) + b.get(d, T2(0));
    }
    return ret;
}

template <typename T>
bool has_spatial(const pvar_map_t<T> &map, char spatial) {
    for (auto &d : map) {
        if (d.to_spatial() == spatial) return true;
    }
    return false;
}

bool is_input_spatial(const pvar_t &pvar);
bool is_output_spatial(const pvar_t &pvar);
bool is_kernel_spatial(const pvar_t &pvar);
bool is_dilation(const pvar_t &pvar);
bool is_padding(const pvar_t &pvar);

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

namespace std {
template <>
struct hash<dnnl::impl::gpu::intel::jit::pvar_t> {
    size_t operator()(const dnnl::impl::gpu::intel::jit::pvar_t &pvar) const {
        return std::hash<std::string>()(pvar.name());
    }
};
} // namespace std

#endif
