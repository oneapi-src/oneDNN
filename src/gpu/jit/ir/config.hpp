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

#ifndef GPU_JIT_IR_CONFIG_HPP
#define GPU_JIT_IR_CONFIG_HPP

#include <iostream>
#include <sstream>

#include "gpu/jit/ir/hw_config.hpp"
#include "gpu/jit/ir/post_ops.hpp"
#include "gpu/jit/ir/tensor.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

class param_t {
public:
    virtual std::string name() const = 0;
    virtual std::string short_name() const { return name(); }
    virtual std::string desc() const = 0;
    virtual bool is_undef() const { return false; }
    virtual bool is_overridable() const = 0;
    virtual std::vector<std::string> accepted_keys() const {
        return std::vector<std::string>({short_name()});
    }
    bool accepts_key(const std::string &key) const {
        auto keys = accepted_keys();
        auto it = std::find(keys.begin(), keys.end(), key);
        return it != keys.end();
    }

    virtual bool is_default() const {
        ir_error_not_expected() << name();
        return false;
    }
    virtual bool is_default(const std::string &key) const {
        if (key == short_name()) return is_default();
        ir_error_not_expected();
        return false;
    }
    virtual void set_from_str(const std::string &s) { ir_error_not_expected(); }
    virtual void set_from_str(
            const std::string &key, const std::string &value) {
        if (key == short_name()) {
            set_from_str(value);
            return;
        }
        ir_error_not_expected();
    }
    void override_set(
            const std::string &key, const std::string &value, bool is_env) {
        key_states_[key] = is_env ? key_state_t::env_overridden
                                  : key_state_t::overridden;
        set_from_str(key, value);
    }
    bool is_overridden() const { return is_overridden(short_name()); }
    bool is_env_overridden() const { return is_env_overridden(short_name()); }
    bool is_overridden(const std::string &key) const {
        return is_overridden_impl(key, /*only_env=*/false);
    }
    bool is_env_overridden(const std::string &key) const {
        return is_overridden_impl(key, /*only_env=*/true);
    }
    int sort_key() const;
    virtual std::string str() const {
        ir_error_not_expected();
        return std::string();
    }
    virtual std::string str(const std::string &key) const {
        if (key == short_name()) return str();
        ir_error_not_expected();
        return std::string();
    }

private:
    enum class key_state_t {
        overridden,
        env_overridden,
    };

    bool is_overridden_impl(const std::string &key, bool only_env) const {
        auto it = key_states_.find(key);
        if (it == key_states_.end()) return false;
        if (only_env) return it->second == key_state_t::env_overridden;
        return utils::one_of(it->second, key_state_t::overridden,
                key_state_t::env_overridden);
    }

    std::unordered_map<std::string, key_state_t> key_states_;
};

template <typename T>
class value_param_t : public param_t {
public:
    using value_t = T;
    using param_t::is_overridden;

    value_param_t() = default;
    value_param_t(const T &value) : value_(value) {}

    const T &get() const { return value_; }

    operator const T &() const { return get(); }

    void set(const T &value) { value_ = value; }

protected:
    T value_;
};

class bool_param_t : public value_param_t<bool> {
public:
    using value_param_t::value_param_t;

    void set_from_str(const std::string &s) override {
        value_ = ir_utils::to_bool(s);
    }

    std::string str() const override {
        std::ostringstream oss;
        oss << short_name() << "=" << (int)value_;
        return oss.str();
    }
};

class int_param_t : public value_param_t<int> {
public:
    using value_param_t::value_param_t;

    void set_from_str(const std::string &s) override { value_ = std::stoi(s); }

    std::string str() const override {
        std::ostringstream oss;
        oss << short_name() << "=" << value_;
        return oss.str();
    }
};

class layout_param_t : public param_t {
public:
    const layout_t &user() const { return user_; }
    const layout_t &compute() const { return compute_; }
    const layout_t &user_unnormalized() const { return user_unnormalized_; }
    const layout_t &compute_unnormalized() const {
        return compute_unnormalized_;
    }

    const std::string &user_unnormalized_tag() const {
        return user_unnormalized_tag_;
    }
    const std::string &compute_unnormalized_tag() const {
        return compute_unnormalized_tag_;
    }

    void set_from_str(const std::string &s) override {
        auto parts = ir_utils::split(s, ".");
        switch ((int)parts.size()) {
            case 1:
                compute_unnormalized_tag_ = parts[0];
                user_unnormalized_tag_ = parts[0];
                break;
            case 2:
                compute_unnormalized_tag_ = parts[0];
                user_unnormalized_tag_ = parts[1];
                break;
            default: ir_error_not_expected();
        }
    }

    void set_user(const layout_t &l) { user_ = l; }
    void set_compute(const layout_t &l) { compute_ = l; }
    void set_user_unnormalized(const layout_t &l, const std::string &tag) {
        user_unnormalized_ = l;
        user_unnormalized_tag_ = tag;
    }
    void set_compute_unnormalized(const layout_t &l, const std::string &tag) {
        compute_unnormalized_ = l;
        compute_unnormalized_tag_ = tag;
    }

    std::string str() const override {
        std::ostringstream oss;
        oss << short_name() << "=";
        oss << compute_unnormalized_tag_;
        if (user_unnormalized_tag_ != compute_unnormalized_tag_)
            oss << "." << user_unnormalized_tag_;
        return oss.str();
    }

private:
    layout_t user_;
    layout_t compute_;
    layout_t user_unnormalized_;
    layout_t compute_unnormalized_;
    std::string user_unnormalized_tag_;
    std::string compute_unnormalized_tag_;
};

class src_layout_param_t : public layout_param_t {
public:
    std::string name() const override { return "src"; }
    std::string desc() const override { return "Source layout."; }
    bool is_overridable() const override { return true; }
    bool is_default() const override { return false; }
};

class dst_layout_param_t : public layout_param_t {
    std::string name() const override { return "dst"; }
    std::string desc() const override { return "Destination layout."; }
    bool is_overridable() const override { return true; }
    bool is_default() const override { return false; }
};

class exec_cfg_param_t : public value_param_t<exec_config_t> {
public:
    using value_param_t::is_overridden;
    using value_param_t::value_param_t;

    std::string name() const override { return "exec-cfg"; }
    std::string desc() const override {
        return "Execution config (hardware config, number of registers, SIMD, "
               "etc).";
    }
    bool is_overridable() const override { return true; }
    bool is_default(const std::string &key) const override {
        if (key == "regs") return false;
        if (key == "simd") return false;
        if (key == "vec") return value_.vec_size() == value_.simd();
        ir_error_not_expected() << key;
        return false;
    }

    std::vector<std::string> accepted_keys() const override {
        std::vector<std::string> ret;
        ret.push_back("regs");
        ret.push_back("simd");
        ret.push_back("vec");
        return ret;
    }

    void set_from_str(
            const std::string &key, const std::string &value) override {
        if (key == "regs") {
            value_.set_regs(std::stoi(value));
        } else if (key == "simd") {
            value_.set_simd(std::stoi(value));
        } else if (key == "vec") {
            value_.set_vec_size(std::stoi(value));
        } else {
            ir_error_not_expected() << key;
        }
    }

    std::string str(const std::string &key) const override {
        std::ostringstream oss;
        if (key == "regs") {
            oss << "regs=" << value_.regs();
        } else if (key == "simd") {
            oss << "simd=" << value_.simd();
        } else if (key == "vec") {
            if (!is_default("vec")) oss << "vec=" << value_.vec_size();
        }
        return oss.str();
    }
};

class prim_config_t {
public:
    virtual ~prim_config_t() = default;
    virtual std::string str() const = 0;

    void override_set(const std::string &s, bool is_env) {
        auto params = get_all_params();
        auto parts = ir_utils::split(s);
        for (auto &p : parts) {
            if (p.empty()) continue;
            auto sub_parts = ir_utils::split(p, "=");
            ir_assert(sub_parts.size() == 2);
            auto &key = sub_parts[0];
            auto &value = sub_parts[1];
            bool found = false;
            for (auto *p : params) {
                if (p->accepts_key(key)) {
                    ir_info() << "Override " << p->name() << ": " << key << "="
                              << value << std::endl;
                    p->override_set(key, value, is_env);
                    found = true;
                    break;
                }
            }
            if (!found) ir_warning() << "Unknown parameter: " << p << std::endl;
        }
    }

    void set_zp_cfg(const zero_points_config_t &zp_cfg) { zp_cfg_ = zp_cfg; }
    const zero_points_config_t &zp_cfg() const { return zp_cfg_; }

#define DECL_PARAM(name) \
    const name##_param_t &name##_param() const { \
        (void)name##_init_; \
        ir_assert(!name##_.is_undef()); \
        return name##_; \
    } \
    name##_param_t &name##_param() { return name##_; } \
    const name##_param_t::value_t &name() const { \
        ir_assert(!name##_.is_undef()); \
        return name##_.get(); \
    } \
    void set_##name(const name##_param_t::value_t &value) { \
        name##_.set(value); \
    }
#define DECL_PARAM2(name) \
    const name##_param_t &name() const { \
        (void)name##_init_; \
        ir_assert(!name##_.is_undef()); \
        return name##_; \
    } \
    name##_param_t &name() { return name##_; }
    DECL_PARAM(exec_cfg)
    DECL_PARAM2(src_layout)
    DECL_PARAM2(dst_layout)
#undef DECL_PARAM
#undef DECL_PARAM2

protected:
    std::vector<std::function<const param_t *(const prim_config_t *)>>
            get_params_;
    zero_points_config_t zp_cfg_;

    struct param_init_t {};
    param_init_t register_param(
            std::function<const param_t *(const prim_config_t *)> f) {
        get_params_.emplace_back(std::move(f));
        return param_init_t();
    }

    std::vector<param_t *> get_all_params(bool do_sort = false) {
        auto *this_const = const_cast<const prim_config_t *>(this);
        std::vector<param_t *> ret;
        for (auto *p : this_const->get_all_params(do_sort)) {
            ret.push_back(const_cast<param_t *>(p));
        }
        return ret;
    }

    std::vector<const param_t *> get_all_params(bool do_sort = false) const {
        std::vector<const param_t *> ret;
        for (auto &gp : get_params_)
            ret.push_back(gp(this));
        if (do_sort) {
            std::sort(ret.begin(), ret.end(),
                    [](const param_t *a, const param_t *b) {
                        return a->sort_key() < b->sort_key();
                    });
        }
        return ret;
    }

    std::string get_config_line() const {
        std::ostringstream oss;
        auto params = get_all_params(/*do_sort=*/true);
        bool is_first = true;
        for (auto *p : params) {
            if (!p->is_overridable()) continue;
            auto keys = p->accepted_keys();
            for (auto &k : keys) {
                if (p->is_default(k)) continue;
                if (!is_first) oss << " ";
                oss << p->str(k);
                is_first = false;
            }
        }
        return oss.str();
    }

#define INIT_PARAM(name) \
    name##_param_t name##_; \
    param_init_t name##_init_ = register_param( \
            [](const prim_config_t *c) { return &c->name##_; });
    INIT_PARAM(exec_cfg)
    INIT_PARAM(src_layout)
    INIT_PARAM(dst_layout)
#undef INIT_PARAM
};

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
