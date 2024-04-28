/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#ifndef GPU_INTEL_COMPUTE_DATA_TYPE_CONVERTER_HPP
#define GPU_INTEL_COMPUTE_DATA_TYPE_CONVERTER_HPP

#include "common/c_types_map.hpp"
#include "gpu/intel/compute/kernel_ctx.hpp"
#include <unordered_map>

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace compute {

inline std::string get_ocl_type(data_type_t type) {
    switch (type) {
        case data_type::f16: return "half";
        case data_type::bf16: return "bf16";
        case data_type::f32: return "float";
        case data_type::s32: return "int";
        case data_type::s8: return "char";
        case data_type::u8: return "uchar";
        case data_type::f64: return "double";
        case data_type::f8_e5m2: return "f8_e5m2";
        case data_type::f8_e4m3: return "f8_e4m3";
        default: gpu_assert(false) << "Unexpected data type";
    }
    return "unexpected";
}

class data_type_converter_t {
public:
    void def_kernel_macros(kernel_ctx_t &kernel_ctx) const {
        bool uses_f8 = false;
        for (const auto &it : types) {
            std::string type_name = it.first;
            data_type_t type = it.second;

            uses_f8 |= utils::one_of(
                    type, data_type::f8_e4m3, data_type::f8_e5m2);

            std::string ocl_type = get_ocl_type(type);
            kernel_ctx.add_option(
                    utils::format("-D%s_DT=%s", type_name, ocl_type));
        }
        if (uses_f8) kernel_ctx.add_option("-DMATH_UTILS_DECLARE_BF8");
    }

    status_t register_type(const std::string &name, data_type_t type) {
        assert(types.find(name) == types.end());
        types[name] = type;
        return status::success;
    }

protected:
    std::unordered_map<std::string, data_type_t> types;
};

} // namespace compute
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
