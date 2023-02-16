/*******************************************************************************
* Copyright 2019-2023 Intel Corporation
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

#ifndef NGEN_OPENCL_HPP
#define NGEN_OPENCL_HPP

#include "ngen_config.hpp"

#include <CL/cl.h>

#include <atomic>
#include <sstream>

#include "ngen_elf.hpp"
#include "ngen_interface.hpp"

#include "npack/neo_packager.hpp"

namespace ngen {


// Exceptions.
class unsupported_opencl_runtime : public std::runtime_error {
public:
    unsupported_opencl_runtime() : std::runtime_error("Unsupported OpenCL runtime.") {}
};
class opencl_error : public std::runtime_error {
public:
    opencl_error(cl_int status_ = 0) : std::runtime_error("An OpenCL error occurred: " + std::to_string(status_)), status(status_) {}
protected:
    cl_int status;
};

// OpenCL program generator class.
template <HW hw>
class OpenCLCodeGenerator : public ELFCodeGenerator<hw>
{
public:
    explicit OpenCLCodeGenerator(Product product_)  : ELFCodeGenerator<hw>(product_) {}
    explicit OpenCLCodeGenerator(int stepping_ = 0) : ELFCodeGenerator<hw>(stepping_) {}

    inline std::vector<uint8_t> getBinary(cl_context context, cl_device_id device, const std::string &options = "-cl-std=CL2.0");
    inline cl_kernel getKernel(cl_context context, cl_device_id device, const std::string &options = "-cl-std=CL2.0");
    static inline HW detectHW(cl_context context, cl_device_id device);
    static inline void detectHWInfo(cl_context context, cl_device_id device, HW &outHW, Product &outProduct);

    /* Deprecated. Use the Product-based API instead. */
    static inline void detectHWInfo(cl_context context, cl_device_id device, HW &outHW, int &outStepping);

private:
    inline std::vector<uint8_t> getPatchTokenBinary(cl_context context, cl_device_id device, const std::vector<uint8_t> *code = nullptr, const std::string &options = "-cl-std=CL2.0");
};

#define NGEN_FORWARD_OPENCL(hw) NGEN_FORWARD_ELF(hw)

namespace detail {

static inline void handleCL(cl_int result)
{
    if (result != CL_SUCCESS)
        throw opencl_error{result};
}

static inline std::vector<uint8_t> getOpenCLCProgramBinary(cl_context context, cl_device_id device, const char *src, const char *options)
{
    cl_int status;

    auto program = clCreateProgramWithSource(context, 1, &src, nullptr, &status);

    detail::handleCL(status);
    if (program == nullptr)
        throw opencl_error();

    detail::handleCL(clBuildProgram(program, 1, &device, options, nullptr, nullptr));
    cl_uint nDevices = 0;
    detail::handleCL(clGetProgramInfo(program, CL_PROGRAM_NUM_DEVICES, sizeof(cl_uint), &nDevices, nullptr));
    std::vector<cl_device_id> devices(nDevices);
    detail::handleCL(clGetProgramInfo(program, CL_PROGRAM_DEVICES, sizeof(cl_device_id) * nDevices, devices.data(), nullptr));
    size_t deviceIdx = std::distance(devices.begin(), std::find(devices.begin(), devices.end(), device));

    if (deviceIdx >= nDevices)
        throw opencl_error();

    std::vector<size_t> binarySize(nDevices);
    std::vector<uint8_t *> binaryPointers(nDevices);
    std::vector<std::vector<uint8_t>> binaries(nDevices);

    detail::handleCL(clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t) * nDevices, binarySize.data(), nullptr));
    for (size_t i = 0; i < nDevices; i++) {
        binaries[i].resize(binarySize[i]);
        binaryPointers[i] = binaries[i].data();
    }

    detail::handleCL(clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(uint8_t *) * nDevices, binaryPointers.data(), nullptr));
    detail::handleCL(clReleaseProgram(program));

    return binaries[deviceIdx];
}

inline bool tryZebinFirst(cl_device_id device, bool setDefault = false, bool newDefault = false)
{
    static std::atomic<bool> hint(false);
    if (setDefault) hint = newDefault;

    return hint;
}

}; /* namespace detail */

template <HW hw>
std::vector<uint8_t> OpenCLCodeGenerator<hw>::getPatchTokenBinary(cl_context context, cl_device_id device, const std::vector<uint8_t> *code, const std::string &options)
{
    using super = ELFCodeGenerator<hw>;
    std::ostringstream dummyCL;
    auto modOptions = options;

    if ((hw >= HW::XeHP) && (super::interface_.needGRF > 128))
        modOptions.append(" -cl-intel-256-GRF-per-thread");

    super::interface_.generateDummyCL(dummyCL);
    auto dummyCLString = dummyCL.str();

    auto binary = detail::getOpenCLCProgramBinary(context, device, dummyCLString.c_str(), modOptions.c_str());

    npack::replaceKernel(binary, code ? *code : this->getCode());

    return binary;
}

template <HW hw>
std::vector<uint8_t> OpenCLCodeGenerator<hw>::getBinary(cl_context context, cl_device_id device, const std::string &options)
{
    using super = ELFCodeGenerator<hw>;
    bool zebinFirst = detail::tryZebinFirst(device);

    auto code = this->getCode();

    for (bool defaultFormat : {true, false}) {
        bool legacy = defaultFormat ^ zebinFirst;

        if (legacy) {
            try {
                return getPatchTokenBinary(context, device, &code, options);
            } catch (...) {
                (void) detail::tryZebinFirst(device, true, true);
                continue;
            }
        } else
            return super::getBinary(code);
    }

    return std::vector<uint8_t>();      // Unreachable.
}

template <HW hw>
cl_kernel OpenCLCodeGenerator<hw>::getKernel(cl_context context, cl_device_id device, const std::string &options)
{
    using super = ELFCodeGenerator<hw>;
    cl_int status = CL_SUCCESS;
    cl_program program = nullptr;
    bool good = false;
    bool zebinFirst = detail::tryZebinFirst(device);
    std::vector<uint8_t> binary;

    auto code = this->getCode();

    for (bool defaultFormat : {true, false}) {
        bool legacy = defaultFormat ^ zebinFirst;

        if (legacy) {
            try {
                binary = getPatchTokenBinary(context, device, &code);
            } catch (...) {
                continue;
            }
        } else
            binary = super::getBinary(code);

        const auto *binaryPtr = binary.data();
        size_t binarySize = binary.size();
        status = CL_SUCCESS;
        program = clCreateProgramWithBinary(context, 1, &device, &binarySize, &binaryPtr, nullptr, &status);

        if ((program == nullptr) || (status != CL_SUCCESS))
            continue;

        status = clBuildProgram(program, 1, &device, options.c_str(), nullptr, nullptr);

        good = (status == CL_SUCCESS);
        if (good) {
            (void) detail::tryZebinFirst(device, true, !legacy);
            break;
        } else
            detail::handleCL(clReleaseProgram(program));
    }

    if (!good)
        throw opencl_error(status);

    auto kernel = clCreateKernel(program, super::interface_.getExternalName().c_str(), &status);
    detail::handleCL(status);
    if (kernel == nullptr)
        throw opencl_error();

    detail::handleCL(clReleaseProgram(program));

    return kernel;
}

template <HW hw>
HW OpenCLCodeGenerator<hw>::detectHW(cl_context context, cl_device_id device)
{
    HW outHW;
    Product outProduct;

    detectHWInfo(context, device, outHW, outProduct);

    return outHW;
}

template <HW hw>
void OpenCLCodeGenerator<hw>::detectHWInfo(cl_context context, cl_device_id device, HW &outHW, int &outStepping)
{
    Product outProduct;
    detectHWInfo(context, device, outHW, outProduct);
    outStepping = outProduct.stepping;
}

template <HW hw>
void OpenCLCodeGenerator<hw>::detectHWInfo(cl_context context, cl_device_id device, HW &outHW, Product &outProduct)
{
    const char *dummyCL = "kernel void _ngen_hw_detect(){}";
    const char *dummyOptions = "";

    auto binary = detail::getOpenCLCProgramBinary(context, device, dummyCL, dummyOptions);

    ELFCodeGenerator<hw>::getBinaryHWInfo(binary, outHW, outProduct);
}

} /* namespace ngen */

#endif
