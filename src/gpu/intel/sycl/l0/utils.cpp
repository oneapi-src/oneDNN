/*******************************************************************************
* Copyright 2020-2025 Intel Corporation
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

#include "gpu/intel/sycl/l0/utils.hpp"
#include "oneapi/dnnl/dnnl_config.h"

#include "gpu/intel/jit/binary_format.hpp"
#include "gpu/intel/jit/utils/ngen_type_bridge.hpp"
#include "ngen_level_zero.hpp"

#if defined(__linux__)
#include <dlfcn.h>
#elif defined(_WIN32)
#include "windows.h"
#else
#error "Level Zero is supported on Linux and Windows only"
#endif

#include "level_zero/ze_api.h"
#include "level_zero/ze_intel_gpu.h"

#if !defined(__SYCL_COMPILER_VERSION)
#error "Unsupported compiler"
#endif

#if (__SYCL_COMPILER_VERSION < 20200818)
#error "Level Zero is not supported with this compiler version"
#endif

#include "common/c_types_map.hpp"
#include "common/verbose.hpp"

#include "gpu/intel/sycl/utils.hpp"
#include <sycl/ext/oneapi/backend/level_zero.hpp>

#include "gpu/intel/sycl/engine.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace sycl {

namespace {

std::string to_string(ze_result_t r) {
#define ZE_STATUS_CASE(status) \
    case status: return #status
    switch (r) {
        ZE_STATUS_CASE(ZE_RESULT_SUCCESS);
        ZE_STATUS_CASE(ZE_RESULT_NOT_READY);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_DEVICE_LOST);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_MODULE_BUILD_FAILURE);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_MODULE_LINK_FAILURE);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_DEVICE_REQUIRES_RESET);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_DEVICE_IN_LOW_POWER_STATE);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_NOT_AVAILABLE);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_UNINITIALIZED);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_UNSUPPORTED_VERSION);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_UNSUPPORTED_FEATURE);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INVALID_ARGUMENT);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INVALID_NULL_HANDLE);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INVALID_NULL_POINTER);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INVALID_SIZE);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_UNSUPPORTED_SIZE);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INVALID_ENUMERATION);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INVALID_NATIVE_BINARY);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INVALID_GLOBAL_NAME);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INVALID_KERNEL_NAME);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INVALID_FUNCTION_NAME);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_OVERLAPPING_REGIONS);
        ZE_STATUS_CASE(ZE_RESULT_ERROR_UNKNOWN);
        ZE_STATUS_CASE(ZE_RESULT_FORCE_UINT32);
        default: return std::to_string((int)r);
    }
#undef ZE_STATUS_CASE
};

#define ZE_CHECK_COMMON(f, retval) \
    do { \
        ze_result_t res_ = (f); \
        if (res_ != ZE_RESULT_SUCCESS) { \
            std::string err_str_ = to_string(res_); \
            VERROR(common, level_zero, "errcode %s", err_str_.c_str()); \
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

status_t func_zeDeviceGetModuleProperties(ze_device_handle_t hDevice,
        ze_device_module_properties_t *pDeviceProperties) {
    static auto f = find_ze_symbol<decltype(&zeDeviceGetModuleProperties)>(
            "zeDeviceGetModuleProperties");

    if (!f) {
        VERROR(common, level_zero,
                "failed to find systolic query extension (maybe update the "
                "driver?)");
        return status::runtime_error;
    }
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

status_t func_zeModuleGetNativeBinary(ze_module_handle_t hModule, size_t *pSize,
        uint8_t *pModuleNativeBinary) {
    static auto f = find_ze_symbol<decltype(&zeModuleGetNativeBinary)>(
            "zeModuleGetNativeBinary");

    if (!f) return status::runtime_error;
    ZE_CHECK(f(hModule, pSize, pModuleNativeBinary));
    return status::success;
}

// FIXME: Currently SYCL doesn't provide any API to get device UUID so
// we query it directly from Level0 with the zeDeviceGetProperties function.
// The `get_device_uuid` function packs 128 bits of the device UUID, which are
// represented as an uint8_t array of size 16, to 2 uint64_t values.
xpu::device_uuid_t get_device_uuid(const ::sycl::device &dev) {
    static_assert(ZE_MAX_DEVICE_UUID_SIZE == 16,
            "ZE_MAX_DEVICE_UUID_SIZE is expected to be 16");

    auto ze_device_properties = ze_device_properties_t();
    ze_device_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;

    auto ze_device = xpu::sycl::compat::get_native<ze_device_handle_t>(dev);
    auto status = func_zeDeviceGetProperties(ze_device, &ze_device_properties);
    MAYBE_UNUSED(status);
    assert(status == status::success);

    const auto &ze_device_id = ze_device_properties.uuid.id;

    uint64_t uuid[ZE_MAX_DEVICE_UUID_SIZE / sizeof(uint64_t)] = {};
    for (size_t i = 0; i < ZE_MAX_DEVICE_UUID_SIZE; ++i) {
        size_t shift = i % sizeof(uint64_t) * CHAR_BIT;
        uuid[i / sizeof(uint64_t)] |= (((uint64_t)ze_device_id[i]) << shift);
    }
    return xpu::device_uuid_t(uuid[0], uuid[1]);
}

status_t sycl_create_kernels_with_level_zero(
        std::vector<std::unique_ptr<::sycl::kernel>> &sycl_kernels,
        const std::vector<const char *> &kernel_names,
        const gpu::intel::sycl::engine_t *sycl_engine,
        const xpu::binary_t &binary) {
    auto desc = ze_module_desc_t();
    desc.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
    desc.format = ZE_MODULE_FORMAT_NATIVE;
    desc.inputSize = binary.size();
    desc.pInputModule = binary.data();
    desc.pBuildFlags = "";
    desc.pConstants = nullptr;

    ze_module_handle_t ze_module;

    auto ze_device = xpu::sycl::compat::get_native<ze_device_handle_t>(
            sycl_engine->device());
    auto ze_ctx = xpu::sycl::compat::get_native<ze_context_handle_t>(
            sycl_engine->context());

    CHECK(func_zeModuleCreate(ze_ctx, ze_device, &desc, &ze_module, nullptr));
    ::sycl::kernel_bundle<::sycl::bundle_state::executable> kernel_bundle
            = ::sycl::make_kernel_bundle<::sycl::backend::ext_oneapi_level_zero,
                    ::sycl::bundle_state::executable>(
                    {ze_module}, sycl_engine->context());

    sycl_kernels.resize(kernel_names.size());
    for (size_t i = 0; i < kernel_names.size(); i++) {
        if (kernel_names[i] == nullptr) continue;
        ze_kernel_handle_t ze_kernel;
        ze_kernel_desc_t ze_kernel_desc {
                ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0, kernel_names[i]};
        CHECK(func_zeKernelCreate(ze_module, &ze_kernel_desc, &ze_kernel));
        auto k = ::sycl::make_kernel<::sycl::backend::ext_oneapi_level_zero>(
                {kernel_bundle, ze_kernel}, sycl_engine->context());
        sycl_kernels[i] = utils::make_unique<::sycl::kernel>(k);
    }

    return status::success;
}

bool compare_ze_devices(const ::sycl::device &lhs, const ::sycl::device &rhs) {
    auto lhs_ze_handle = xpu::sycl::compat::get_native<ze_device_handle_t>(lhs);
    auto rhs_ze_handle = xpu::sycl::compat::get_native<ze_device_handle_t>(rhs);

    return lhs_ze_handle == rhs_ze_handle;
}

status_t get_device_ip(ze_device_handle_t device, uint32_t &ip_version) {
    auto devicePropsIP = ze_device_ip_version_ext_t();
    devicePropsIP.stype = ZE_STRUCTURE_TYPE_DEVICE_IP_VERSION_EXT;

    auto deviceProps = ze_device_properties_t();
    deviceProps.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    deviceProps.pNext = &devicePropsIP;

    CHECK(func_zeDeviceGetProperties(device, &deviceProps));
    ip_version = devicePropsIP.ipVersion;
    return status::success;
}

status_t get_l0_device_enabled_systolic_intel(
        ze_device_handle_t device, bool &mayiuse_systolic) {
    // Note: supported by Intel Driver 24.05 and onwards
    auto deviceModPropsExt = ze_intel_device_module_dp_exp_properties_t();
    deviceModPropsExt.stype
            = ZE_STRUCTURE_INTEL_DEVICE_MODULE_DP_EXP_PROPERTIES;

    auto deviceModProps = ze_device_module_properties_t();
    deviceModProps.stype = ZE_STRUCTURE_TYPE_DEVICE_MODULE_PROPERTIES;
    deviceModProps.pNext = &deviceModPropsExt;

    CHECK(func_zeDeviceGetModuleProperties(device, &deviceModProps));
    mayiuse_systolic
            = deviceModPropsExt.flags & ZE_INTEL_DEVICE_MODULE_EXP_FLAG_DPAS;
    return status::success;
}

status_t get_l0_device_enabled_native_float_atomics(
        ze_device_handle_t device, uint64_t native_extensions) {
    using namespace gpu::intel::compute;

    auto fltAtom = ze_float_atomic_ext_properties_t();
    fltAtom.stype = ZE_STRUCTURE_TYPE_FLOAT_ATOMIC_EXT_PROPERTIES;

    auto deviceProps = ze_device_properties_t();
    deviceProps.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    deviceProps.pNext = &fltAtom;

    CHECK(func_zeDeviceGetProperties(device, &deviceProps));

    ze_device_fp_atomic_ext_flags_t atomic_load_store
            = ZE_DEVICE_FP_ATOMIC_EXT_FLAG_GLOBAL_LOAD_STORE
            | ZE_DEVICE_FP_ATOMIC_EXT_FLAG_LOCAL_LOAD_STORE;
    ze_device_fp_atomic_ext_flags_t atomic_add
            = ZE_DEVICE_FP_ATOMIC_EXT_FLAG_GLOBAL_ADD
            | ZE_DEVICE_FP_ATOMIC_EXT_FLAG_LOCAL_ADD;
    ze_device_fp_atomic_ext_flags_t atomic_min_max
            = ZE_DEVICE_FP_ATOMIC_EXT_FLAG_GLOBAL_MIN_MAX
            | ZE_DEVICE_FP_ATOMIC_EXT_FLAG_LOCAL_MIN_MAX;

    if ((fltAtom.fp16Flags & atomic_load_store) == atomic_load_store)
        native_extensions |= (uint64_t)native_ext_t::fp16_atomic_load_store;
    if ((fltAtom.fp16Flags & atomic_add) == atomic_add)
        native_extensions |= (uint64_t)native_ext_t::fp16_atomic_add;
    if ((fltAtom.fp16Flags & atomic_add) == atomic_min_max)
        native_extensions |= (uint64_t)native_ext_t::fp16_atomic_min_max;

    if ((fltAtom.fp32Flags & atomic_load_store) == atomic_load_store)
        native_extensions |= (uint64_t)native_ext_t::fp32_atomic_load_store;
    if ((fltAtom.fp32Flags & atomic_add) == atomic_add)
        native_extensions |= (uint64_t)native_ext_t::fp32_atomic_add;
    if ((fltAtom.fp32Flags & atomic_add) == atomic_min_max)
        native_extensions |= (uint64_t)native_ext_t::fp32_atomic_min_max;

    if ((fltAtom.fp64Flags & atomic_load_store) == atomic_load_store)
        native_extensions |= (uint64_t)native_ext_t::fp64_atomic_load_store;
    if ((fltAtom.fp64Flags & atomic_add) == atomic_add)
        native_extensions |= (uint64_t)native_ext_t::fp64_atomic_add;
    if ((fltAtom.fp64Flags & atomic_add) == atomic_min_max)
        native_extensions |= (uint64_t)native_ext_t::fp64_atomic_min_max;

    return status::success;
}

status_t get_l0_device_eu_count(ze_device_handle_t device, int &eu_count) {
    auto eucnt = ze_eu_count_ext_t();
    auto deviceProps = ze_device_properties_t();
    deviceProps.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    deviceProps.pNext = &eucnt;

    CHECK(func_zeDeviceGetProperties(device, &deviceProps));
    eu_count = eucnt.numTotalEUs;
    return status::success;
}

status_t init_gpu_hw_info(impl::engine_t *engine, ze_device_handle_t device,
        ze_context_handle_t context, uint32_t &ip_version,
        compute::gpu_arch_t &gpu_arch, int &gpu_product_family,
        int &stepping_id, uint64_t &native_extensions, bool &mayiuse_systolic,
        bool &mayiuse_ngen_kernels) {
    using namespace ngen;
    HW hw = HW::Unknown;
    Product product = {ProductFamily::Unknown, 0};
    LevelZeroCodeGenerator<HW::Unknown>::detectHWInfo(
            context, device, hw, product);

    gpu_arch = jit::convert_ngen_arch_to_dnnl(hw);
    gpu_product_family = static_cast<int>(product.family);
    stepping_id = product.stepping;

    mayiuse_systolic = false;
    CHECK(get_l0_device_enabled_systolic_intel(device, mayiuse_systolic));

    CHECK(get_l0_device_enabled_native_float_atomics(
            device, native_extensions));

    auto status
            = jit::gpu_supports_binary_format(&mayiuse_ngen_kernels, engine);
    if (status != status::success) mayiuse_ngen_kernels = false;

    ip_version = 0;
    return get_device_ip(device, ip_version);
}

} // namespace sycl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
