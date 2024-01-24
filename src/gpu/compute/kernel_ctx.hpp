/*******************************************************************************
* Copyright 2019-2024 Intel Corporation
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

#ifndef GPU_COMPUTE_KERNEL_CTX_HPP
#define GPU_COMPUTE_KERNEL_CTX_HPP

#include <cassert>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>

#include "common/bit_cast.hpp"
#include "common/primitive_attr.hpp"
#include "common/verbose.hpp"
#include "gpu/gpu_primitive_attr.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace compute {

class kernel_ctx_t {
public:
    kernel_ctx_t(const primitive_attr_t *attr = nullptr) {
        set_default_options(attr);
        set_default_macros(attr);
    }

    std::string options() const {
        std::ostringstream oss;
        for (auto &opt : option_set_)
            oss << " " << opt;

        for (auto &int_var : int_var_map_) {
            oss << " -D" << int_var.first << "=" << int_var.second;
            if (int_var.second > INT_MAX || int_var.second < INT_MIN)
                oss << "L";
        }

        for (auto &float_var : float_var_map_) {
            oss << " -D" << float_var.first << "=as_float(0x" << std::hex
                << utils::bit_cast<uint32_t>(float_var.second) << ")";
        }
        return oss.str();
    }

    void define_int(const char *variable, int64_t value) {
        int_var_map_.insert({variable, value});
    }

    void define_int(const std::string &variable, int64_t value) {
        define_int(variable.c_str(), value);
    }

    // TODO: should be removed, any float values should be passed in
    // kernel parameters
    void define_float(const char *variable, float value) {
        float_var_map_.insert({variable, value});
    }

    void add_option(const char *option) { option_set_.insert(option); }
    void add_option(const std::string &option) { add_option(option.c_str()); }

    bool has_macro(const char *name) const {
        std::string opt_start = std::string("-D") + name + "=";
        for (auto &opt : option_set_)
            if (opt.find(opt_start) != std::string::npos) return true;

        return int_var_map_.count(name) != 0 || float_var_map_.count(name) != 0;
    }
    bool has_macro(const std::string &name) const {
        return has_macro(name.c_str());
    }

    void set_data_type(data_type_t dt) {
        switch (dt) {
            case data_type::bf16: define_int("DT_BF16", 1); break;
            case data_type::f16: define_int("DT_F16", 1); break;
            case data_type::f32: define_int("DT_F32", 1); break;
            case data_type::f64: define_int("DT_F64", 1); break;
            case data_type::s8: define_int("DT_S8", 1); break;
            case data_type::u8: define_int("DT_U8", 1); break;
            case data_type::f8_e4m3: define_int("DT_HF8", 1); break;
            case data_type::f8_e5m2: define_int("DT_BF8", 1); break;
            case data_type::s32: define_int("DT_S32", 1); break;
            default: assert(!"unknown data type"); break;
        }
    }

    template <typename T>
    T get_scalar(const std::string &s) const {
        UNUSED(s);
        static_assert(!std::is_same<T, T>::value, "not expected");
        return {};
    }

    std::string data_type() const {
        if (int_var_map_.count("DT_F16") != 0) return "f16";

        if (int_var_map_.count("DT_F32") != 0) return "f32";

        if (int_var_map_.count("DT_F64") != 0) return "f64";

        if (int_var_map_.count("DT_S8") != 0) return "s8";

        return "";
    }

private:
    void set_default_options(const primitive_attr_t *attr) {
        // By default fp32 division and sqrt are not IEEE-compliant
        add_option("-cl-fp32-correctly-rounded-divide-sqrt");

        if (attr && attr->gpu_attr_) {
            auto *gpu_attr = utils::downcast<gpu_primitive_attr_t *>(
                    attr->gpu_attr_.get());
            if (gpu_attr->threads_per_eu() == 4) {
                add_option("-cl-intel-256-GRF-per-thread");
            }
        }
    }
    void set_default_macros(const primitive_attr_t *attr) {
        if (attr) { define_int("DETERMINISTIC", attr->deterministic_); }
    }

    std::map<std::string, int64_t> int_var_map_;
    std::map<std::string, float> float_var_map_;
    std::set<std::string> option_set_;
};

template <>
inline int64_t kernel_ctx_t::get_scalar(const std::string &name) const {
    assert(int_var_map_.count(name) != 0 && "not expected");
    return int_var_map_.at(name);
}

} // namespace compute
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_COMPUTE_KERNEL_CTX_HPP
