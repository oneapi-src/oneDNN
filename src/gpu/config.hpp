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

#ifndef GPU_CONFIG_HPP
#define GPU_CONFIG_HPP

#include <functional>
#include <iostream>
#include <sstream>
#include <vector>
#include <unordered_map>

#include "gpu/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {

class param_t {
public:
    virtual ~param_t() = default;
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
        gpu_error_not_expected() << name();
        return false;
    }
    virtual bool is_default(const std::string &key) const {
        if (key == short_name()) return is_default();
        gpu_error_not_expected();
        return false;
    }
    virtual void set_from_str(const std::string &s) {
        gpu_error_not_expected();
    }
    virtual void set_from_str(
            const std::string &key, const std::string &value) {
        if (key == short_name()) {
            set_from_str(value);
            return;
        }
        gpu_error_not_expected();
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
    virtual std::string str() const {
        gpu_error_not_expected();
        return std::string();
    }
    virtual std::string str(const std::string &key) const {
        if (key == short_name()) return str();
        gpu_error_not_expected();
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
        value_ = gpu_utils::to_bool(s);
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

class container_config_t {
public:
    virtual ~container_config_t() = default;
    virtual std::string str() const = 0;

    virtual void override_set(const std::string &s, bool is_env) {
        auto params = get_all_params();
        auto parts = gpu_utils::split(s);
        for (auto &p : parts) {
            if (p.empty()) continue;
            auto sub_parts = gpu_utils::split(p, "=");
            gpu_assert(sub_parts.size() == 2);
            auto &key = sub_parts[0];
            auto &value = sub_parts[1];
            bool found = false;
            for (auto *p : params) {
                if (p->accepts_key(key)) {
                    p->override_set(key, value, is_env);
                    found = true;
                    break;
                }
            }
            // TODO: Get access to ir_info() and ir_warning() to use in
            // case of overriden/unknown parameters.
            gpu_assert(found) << "Unknown parameter";
        }
    }

    virtual int sort_key(const param_t *param) const = 0;

protected:
    std::vector<std::function<const param_t *(const container_config_t *)>>
            get_params_;

    struct param_init_t {};
    param_init_t register_param(
            std::function<const param_t *(const container_config_t *)> f) {
        get_params_.emplace_back(std::move(f));
        return param_init_t();
    }

    std::vector<param_t *> get_all_params(bool do_sort = false) {
        auto *this_const = const_cast<const container_config_t *>(this);
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
                    [this](const param_t *a, const param_t *b) {
                        return sort_key(a) < sort_key(b);
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
};

} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
