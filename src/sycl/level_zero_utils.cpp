/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

#include "sycl/level_zero_utils.hpp"
#include "oneapi/dnnl/dnnl_config.h"

#include <stdio.h>

#if defined(__linux__)
#include <dlfcn.h>
#elif defined(_WIN32)
#include "windows.h"
#else
#error "Level Zero is supported on Linux and Windows only"
#endif

#include <level_zero/ze_api.h>

#if !defined(__SYCL_COMPILER_VERSION)
#error "Unsupported compiler"
#endif

#if (__SYCL_COMPILER_VERSION < 20200818)
#error "Level Zero is not supported with this compiler version"
#endif

#include "common/c_types_map.hpp"
#include "common/verbose.hpp"

#include "sycl/sycl_utils.hpp"
#include <sycl/ext/oneapi/backend/level_zero.hpp>

#include "sycl/sycl_engine_base.hpp"

namespace dnnl {
namespace impl {
namespace sycl {

namespace {

#define ZE_CHECK_COMMON(f, retval) \
    do { \
        ze_result_t res_ = (f); \
        if (res_ != ZE_RESULT_SUCCESS) { \
            VERROR(common, level_zero, "errcode %d", (int)(res_)); \
            return retval; \
        } \
    } while (false)

#define ZE_CHECK(f) ZE_CHECK_COMMON(f, status::runtime_error)
#define ZE_CHECK_VP(f) ZE_CHECK_COMMON(f, nullptr)

void *find_ze_symbol(const char *symbol) {
#if defined(__linux__)
    void *handle = dlopen("libze_loader.so.1", RTLD_NOW | RTLD_LOCAL);
#elif defined(_WIN32)
    // Use LOAD_LIBRARY_SEARCH_SYSTEM32 flag to avoid DLL hijacking issue.
    HMODULE handle = LoadLibraryExA(
            "ze_loader.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
#endif
    if (!handle) {
        VERROR(common, level_zero, "cannot find loader library");
        assert(!"not expected");
        return nullptr;
    }

    using zeInit_decl_t = ze_result_t (*)(ze_init_flags_t flags);
    const ze_init_flags_t default_ze_flags = 0;
#if defined(__linux__)
    static const ze_result_t ze_result = reinterpret_cast<zeInit_decl_t>(
            dlsym(handle, "zeInit"))(default_ze_flags);
    void *f = reinterpret_cast<void *>(dlsym(handle, symbol));
#elif defined(_WIN32)
    static const ze_result_t ze_result = reinterpret_cast<zeInit_decl_t>(
            GetProcAddress(handle, "zeInit"))(default_ze_flags);
    void *f = reinterpret_cast<void *>(GetProcAddress(handle, symbol));
#endif
    ZE_CHECK_VP(ze_result);

    if (!f) {
        VERROR(common, level_zero, "cannot find symbol: %s", symbol);
        assert(!"not expected");
    }
    return f;
}

template <typename F>
F find_ze_symbol(const char *symbol) {
    return (F)find_ze_symbol(symbol);
}

status_t func_zeModuleCreate(ze_context_handle_t hContext,
        ze_device_handle_t hDevice, const ze_module_desc_t *desc,
        ze_module_handle_t *phModule,
        ze_module_build_log_handle_t *phBuildLog) {
    static auto f = find_ze_symbol<decltype(&zeModuleCreate)>("zeModuleCreate");

    if (!f) return status::runtime_error;
    ZE_CHECK(f(hContext, hDevice, desc, phModule, phBuildLog));
    return status::success;
}

status_t func_zeDeviceGetProperties(
        ze_device_handle_t hDevice, ze_device_properties_t *pDeviceProperties) {
    static auto f = find_ze_symbol<decltype(&zeDeviceGetProperties)>(
            "zeDeviceGetProperties");

    if (!f) return status::runtime_error;
    ZE_CHECK(f(hDevice, pDeviceProperties));
    return status::success;
}

} // namespace

// This function is called from compatibility layer that ensures compatibility
// with SYCL 2017 API. Once the compatibility layer is removed this function
// can be moved to the anonymous namespace above and a function with SYCL
// data types in its interface can be created to call it.
status_t func_zeKernelCreate(ze_module_handle_t hModule,
        const ze_kernel_desc_t *desc, ze_kernel_handle_t *phKernel) {
    static auto f = find_ze_symbol<decltype(&zeKernelCreate)>("zeKernelCreate");

    if (!f) return status::runtime_error;
    ZE_CHECK(f(hModule, desc, phKernel));
    return status::success;
}

// FIXME: Currently SYCL doesn't provide any API to get device UUID so
// we query it directly from Level0 with the zeDeviceGetProperties function.
// The `get_device_uuid` function packs 128 bits of the device UUID, which are
// represented as an uint8_t array of size 16, to 2 uint64_t values.
device_uuid_t get_device_uuid(const ::sycl::device &dev) {
    static_assert(ZE_MAX_DEVICE_UUID_SIZE == 16,
            "ZE_MAX_DEVICE_UUID_SIZE is expected to be 16");

    auto ze_device_properties = ze_device_properties_t();
    ze_device_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;

    auto ze_device = compat::get_native<ze_device_handle_t>(dev);
    auto status = func_zeDeviceGetProperties(ze_device, &ze_device_properties);
    MAYBE_UNUSED(status);
    assert(status == status::success);

    const auto &ze_device_id = ze_device_properties.uuid.id;

    uint64_t uuid[ZE_MAX_DEVICE_UUID_SIZE / sizeof(uint64_t)] = {};
    for (size_t i = 0; i < ZE_MAX_DEVICE_UUID_SIZE; ++i) {
        size_t shift = i % sizeof(uint64_t) * CHAR_BIT;
        uuid[i / sizeof(uint64_t)] |= (((uint64_t)ze_device_id[i]) << shift);
    }
    return device_uuid_t(uuid[0], uuid[1]);
}

status_t sycl_create_kernel_with_level_zero(
        std::unique_ptr<::sycl::kernel> &sycl_kernel,
        const std::string &kernel_name, const sycl_engine_base_t *sycl_engine,
        const gpu::compute::binary_t &binary) {
    auto desc = ze_module_desc_t();
    desc.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
    desc.format = ZE_MODULE_FORMAT_NATIVE;
    desc.inputSize = binary.size();
    desc.pInputModule = binary.data();
    desc.pBuildFlags = "";
    desc.pConstants = nullptr;

    ze_module_handle_t ze_module;

    auto ze_device
            = compat::get_native<ze_device_handle_t>(sycl_engine->device());
    auto ze_ctx
            = compat::get_native<ze_context_handle_t>(sycl_engine->context());

    CHECK(func_zeModuleCreate(ze_ctx, ze_device, &desc, &ze_module, nullptr));
    ::sycl::kernel_bundle<::sycl::bundle_state::executable> kernel_bundle
            = ::sycl::make_kernel_bundle<::sycl::backend::ext_oneapi_level_zero,
                    ::sycl::bundle_state::executable>(
                    {ze_module}, sycl_engine->context());

    ze_kernel_handle_t ze_kernel;
    ze_kernel_desc_t ze_kernel_desc {
            ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0, kernel_name.c_str()};
    CHECK(func_zeKernelCreate(ze_module, &ze_kernel_desc, &ze_kernel));
    auto k = ::sycl::make_kernel<::sycl::backend::ext_oneapi_level_zero>(
            {kernel_bundle, ze_kernel}, sycl_engine->context());
    sycl_kernel = utils::make_unique<::sycl::kernel>(k);

    return status::success;
}

bool compare_ze_devices(const ::sycl::device &lhs, const ::sycl::device &rhs) {
    auto lhs_ze_handle = compat::get_native<ze_device_handle_t>(lhs);
    auto rhs_ze_handle = compat::get_native<ze_device_handle_t>(rhs);

    return lhs_ze_handle == rhs_ze_handle;
}

} // namespace sycl
} // namespace impl
} // namespace dnnl
