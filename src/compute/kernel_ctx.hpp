/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#ifndef COMPUTE_KERNEL_CTX_HPP
#define COMPUTE_KERNEL_CTX_HPP

#include <cassert>
#ifdef DEBUG_PRINT
#include <iostream>
#endif
#include <map>
#include <set>
#include <sstream>
#include <type_traits>

namespace dnnl {
namespace impl {
namespace compute {

class kernel_ctx_t {
public:
    kernel_ctx_t() { set_default_options(); }

    std::string options() const {
        std::ostringstream oss;
        for (auto &opt : option_set_)
            oss << " " << opt;

        for (auto &int_var : int_var_map_)
            oss << " -D" << int_var.first << "=" << int_var.second;

        for (auto &float_var : float_var_map_) {
            union {
                float f;
                uint32_t u;
            } f2u = {float_var.second};

            oss << " -D" << float_var.first << "=as_float(0x" << std::hex
                << f2u.u << ")";
        }
        return oss.str();
    }

    void define_int(const char *variable, int64_t value) {
        int_var_map_.insert({variable, value});
    }

    // TODO: should be removed, any float values should be passed in
    // kernel parameters
    void define_float(const char *variable, float value) {
        float_var_map_.insert({variable, value});
    }

    void add_option(const char *option) { option_set_.insert(option); }

    void set_data_type(data_type_t dt) {
        switch (dt) {
            case data_type::bf16: define_int("DT_BF16", 1); break;
            case data_type::f16: define_int("DT_F16", 1); break;
            case data_type::f32: define_int("DT_F32", 1); break;
            case data_type::s8: define_int("DT_S8", 1); break;
            case data_type::u8: define_int("DT_U8", 1); break;
            case data_type::s32: define_int("DT_S32", 1); break;
            default: assert(!"unknown data type"); break;
        }
    }

    void print_options() const {
#ifdef DEBUG_PRINT
        std::cout << "OPT:\n" << options() << std::endl;
#endif
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

        if (int_var_map_.count("DT_S8") != 0) return "s8";

        return "";
    }

private:
    void set_default_options() {
        // By default fp32 division and sqrt are not IEEE-compliant
        add_option("-cl-fp32-correctly-rounded-divide-sqrt");
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
} // namespace impl
} // namespace dnnl

#endif // COMPUTE_KERNEL_CTX_HPP
