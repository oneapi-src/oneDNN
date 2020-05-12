/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#ifndef SYCL_LEVEL_ZERO_UTILS_HPP
#define SYCL_LEVEL_ZERO_UTILS_HPP

#if defined(DNNL_WITH_LEVEL_ZERO)

#ifndef __linux__
#error "Level Zero is supported with Linux only"
#endif

#include <dlfcn.h>
#include <level_zero/ze_api.h>

#include "common/verbose.hpp"
#include <CL/sycl.hpp>

#define ZE_CHECK(f) \
    do { \
        ze_result_t res_ = (f); \
        if (res_ != ZE_RESULT_SUCCESS) { \
            if (get_verbose()) \
                printf("dnnl_verbose,gpu,ze_error,%d\n", (int)(res_)); \
            return status::runtime_error; \
        } \
    } while (false)

namespace dnnl {
namespace impl {
namespace sycl {

template <typename F>
F find_ze_symbol(const char *symbol) {
    const char *ze_loader_name = "libze_loader.so";
    void *handle = dlopen(ze_loader_name, RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        if (get_verbose())
            printf("dnnl_verbose,gpu,error,cannot find Level Zero loader "
                   "library\n");
        assert(!"not expected");
        return nullptr;
    }
    void *f = dlsym(handle, symbol);
    if (!f) {
        if (get_verbose())
            printf("dnnl_verbose,gpu,error,cannot find symbol: %s\n", symbol);
        assert(!"not expected");
    }
    return (F)f;
}

inline status_t func_zeModuleCreate(ze_device_handle_t hDevice,
        const ze_module_desc_t *desc, ze_module_handle_t *phModule,
        ze_module_build_log_handle_t *phBuildLog) {
    using func_type
            = ze_result_t (*)(ze_device_handle_t, const ze_module_desc_t *,
                    ze_module_handle_t *, ze_module_build_log_handle_t *);
    static auto f = find_ze_symbol<func_type>("zeModuleCreate");

    if (!f) return status::runtime_error;
    ZE_CHECK(f(hDevice, desc, phModule, phBuildLog));
    return status::success;
}

inline status_t func_zeModuleDestroy(ze_module_handle_t hModule) {
    using func_type = ze_result_t (*)(ze_module_handle_t);
    static auto f = find_ze_symbol<func_type>("zeModuleDestroy");

    if (!f) return status::runtime_error;
    ZE_CHECK(f(hModule));
    return status::success;
}

inline status_t func_zeDeviceGetProperties(
        ze_device_handle_t hDevice, ze_device_properties_t *pDeviceProperties) {
    using func_type
            = ze_result_t (*)(ze_device_handle_t, ze_device_properties_t *);
    static auto f = find_ze_symbol<func_type>("zeDeviceGetProperties");

    if (!f) return status::runtime_error;
    ZE_CHECK(f(hDevice, pDeviceProperties));
    return status::success;
}

using device_uuid_t = std::tuple<uint64_t, uint64_t>;

// FIXME: Currently SYCL doesn't provide any API to get device UUID so
// we query it directly from Level0 with the zeDeviceGetProperties function.
// The `get_device_uuid` function packs 128 bits of the device UUID, which are
// represented as an uint8_t array of size 16, to 2 uint64_t values.
inline device_uuid_t get_device_uuid(const cl::sycl::device &dev) {
    static_assert(ZE_MAX_DEVICE_UUID_SIZE == 16,
            "ZE_MAX_DEVICE_UUID_SIZE is expected to be 16");

    ze_device_properties_t ze_device_properties;
    auto ze_device = (ze_device_handle_t)dev.get();
    auto status = func_zeDeviceGetProperties(ze_device, &ze_device_properties);
    assert(status == status::success);

    const auto &ze_device_id = ze_device_properties.uuid.id;

    constexpr int half_ze_uuid_size = ZE_MAX_DEVICE_UUID_SIZE / 2;
    const int uuid_size = 2;
    uint64_t uuid[uuid_size] = {};

    for (int i = 0; i < uuid_size; i++) {
        for (int j = half_ze_uuid_size - 1; j >= 0; j--) {
            for (int k = CHAR_BIT - 1; k >= 0; k--) {
                uint64_t bit_pos = j * CHAR_BIT + k;

                int ze_device_id_idx
                        = i * sizeof(uint64_t) + (half_ze_uuid_size - 1) - j;
                uint64_t bit_val = (ze_device_id[ze_device_id_idx] >> k) & 1ULL;
                uuid[i] |= (bit_val << bit_pos);
            }
        }
    }
    return device_uuid_t(uuid[0], uuid[1]);
}

} // namespace sycl
} // namespace impl
} // namespace dnnl

#endif

#endif // SYCL_LEVEL_ZERO_UTILS_HPP
