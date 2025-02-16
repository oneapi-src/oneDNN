/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
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

#ifndef GPU_INTEL_COMPUTE_KERNEL_CTX_HPP
#define GPU_INTEL_COMPUTE_KERNEL_CTX_HPP

#include <cassert>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>

#include "common/bit_cast.hpp"
#include "gpu/intel/gpu_primitive_attr.hpp"
#include "gpu/intel/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
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

        if (use_int32_offset_) {
            oss << " -DUSE_INT32_OFFSET";
        } else {
            // TODO: Determine if specialization for buffers between 2GB and 4GB
            // is worthwhile
            oss << " -cl-intel-greater-than-4GB-buffer-required";
        }

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

    void register_buffer_size(size_t size) {
        if (size > INT_MAX) use_int32_offset(false);
    }

    void register_buffer_size(const memory_desc_wrapper &mdw) {
        register_buffer_size(mdw.size());
    }

    // Enable various optimizations when all buffers are < 2GB in size. In this
    // case, int32_t types can be used for data offsets and avoid int64_t
    // operations when native 64-bit operations are unsupported.
    void use_int32_offset(bool value) { use_int32_offset_ = value; }

    void define_int(const char *variable, int64_t value) {
        set_macro(variable, value, int_var_map_);
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
            case data_type::f4_e2m1: define_int("DT_F4_E2M1", 1); break;
            case data_type::f4_e3m0: define_int("DT_F4_E3M0", 1); break;
            case data_type::s32: define_int("DT_S32", 1); break;
            default: assert(!"unknown data type"); break;
        }
    }

    std::string data_type() const {
        if (int_var_map_.count("DT_F16") != 0) return "f16";

        if (int_var_map_.count("DT_F32") != 0) return "f32";

        if (int_var_map_.count("DT_F64") != 0) return "f64";

        if (int_var_map_.count("DT_S8") != 0) return "s8";

        return "";
    }

    void add_custom_header(
            const std::string &header_name, std::string &&source) {
        custom_headers_[header_name] = std::move(source);
    }

    const char *get_custom_header(const std::string &header_name) const {
        auto iter = custom_headers_.find(header_name);
        if (iter != custom_headers_.end()) return iter->second.c_str();
        return nullptr;
    }

    bool has_custom_headers() const { return !custom_headers_.empty(); }

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

        // Set override flag for checking compiler assumptions
        if (gpu_utils::dev_getenv("enable_check_assumptions", 0)) {
            add_option("-DENABLE_CHECK_ASSUMPTIONS");
        }

        if (gpu_utils::dev_getenv("ocl_debug", 0)) {
            add_option("-DOCL_DEBUG");
        }
    }
    void set_default_macros(const primitive_attr_t *attr) {
        if (attr) { define_int("DETERMINISTIC", attr->deterministic_); }
    }

    template <typename T>
    void set_macro(const char *variable, const T &value,
            std::map<std::string, T> &var_map) {
        gpu_assert(var_map.count(variable) == 0 || var_map[variable] == value)
                << "Error: macro " << variable
                << " is already set to a different value.\n  Old value: "
                << var_map[variable] << "\n  New value: " << value;
        var_map.insert({variable, value});
    }

    std::map<std::string, int64_t> int_var_map_;
    std::map<std::string, float> float_var_map_;
    std::set<std::string> option_set_;
    std::unordered_map<std::string, std::string> custom_headers_;
    bool use_int32_offset_ = true;
};

} // namespace compute
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_INTEL_COMPUTE_KERNEL_CTX_HPP
