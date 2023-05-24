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

#include "gpu/jit/binary_format.hpp"

#include "common/utils.hpp"
#include "gpu/compute/compute_engine.hpp"
#include "gpu/compute/compute_stream.hpp"
#include "gpu/jit/jit_generator.hpp"

#define MAGIC0 0xBEEFCAFEu
#define MAGIC1 0x3141592653589793ull
#define MAGIC2 0xBEAD
#define MAGIC3 0xFACE
#define MAGIC4 0x0123456789ABCDEFull
#define MAGIC5 0xFEDCBA9876543210ull
#define MAGICPTR 0xABADFEEDu

#define MAGICSIZEX 4
#define MAGICSIZEY 2
#define MAGICSIZEZ 1

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

using namespace ngen;

template <HW hw>
class binary_format_kernel_t : public jit_generator<hw> {
    NGEN_FORWARD_OPENCL(hw);

public:
    binary_format_kernel_t() {

        auto low_half = [](uint64_t q) -> uint32_t { return q & 0xFFFFFFFF; };
        auto high_half = [](uint64_t q) -> uint32_t { return q >> 32; };

        newArgument("src0", DataType::ud); // r5.4:ud
        newArgument("src1", DataType::uq); // r5.3:uq
        newArgument("src2", DataType::uw); // r6.0:uw
        newArgument("src3", DataType::uw); // r6.2:uw
        newArgument("src4", DataType::uq); // r6.1:uq
        newArgument("src5", DataType::uq); // r6.2:uq
        newArgument("src_ptr", ExternalArgumentType::GlobalPtr);
        newArgument("ok", ExternalArgumentType::GlobalPtr);

        setDefaultAutoSWSB();
        requireSIMD((GRF::bytes(hw) == 64) ? 16 : 8);
        requireLocalID(3); // r1-r3
        requireLocalSize(); // r7.0-2:ud
        finalizeInterface();

        Label doWrite;

        auto src0 = getArgument("src0");
        auto src1 = getArgument("src1");
        auto src2 = getArgument("src2");
        auto src3 = getArgument("src3");
        auto src4 = getArgument("src4");
        auto src5 = getArgument("src5");
        auto src_ptr = getArgument("src_ptr");
        auto ok_surface = Surface(getArgumentSurfaceIfExists("ok"));

        auto data = r30;
        auto data2 = r31;
        auto ok = data.ud(0);
        auto header = r64;

        prologue();
        setDefaultNoMask();

        // Default: test failure.
        mov(1, ok, uint16_t(0));

        // Validate scalar arguments
        cmp(1 | eq | f0[0], null.ud(), src0, uint32_t(MAGIC0));
        jmpi(1 | ~f0[0], doWrite);
        cmp(1 | eq | f0[0], null.ud(), src1.ud(0), low_half(MAGIC1));
        jmpi(1 | ~f0[0], doWrite);
        cmp(1 | eq | f0[0], null.ud(), src1.ud(1), high_half(MAGIC1));
        jmpi(1 | ~f0[0], doWrite);
        cmp(1 | eq | f0[0], null.uw(), src2, uint16_t(MAGIC2));
        jmpi(1 | ~f0[0], doWrite);
        cmp(1 | eq | f0[0], null.uw(), src3, uint16_t(MAGIC3));
        jmpi(1 | ~f0[0], doWrite);
        cmp(1 | eq | f0[0], null.ud(), src4.ud(0), low_half(MAGIC4));
        jmpi(1 | ~f0[0], doWrite);
        cmp(1 | eq | f0[0], null.ud(), src4.ud(1), high_half(MAGIC4));
        jmpi(1 | ~f0[0], doWrite);
        cmp(1 | eq | f0[0], null.ud(), src5.ud(0), low_half(MAGIC5));
        jmpi(1 | ~f0[0], doWrite);
        cmp(1 | eq | f0[0], null.ud(), src5.ud(1), high_half(MAGIC5));
        jmpi(1 | ~f0[0], doWrite);

        // Validate A64 pointer argument.
        mov<uint32_t>(2, header[0](1), src_ptr.ud(0)(1));
        load(1 | SWSB(sb0, 1), data2, scattered_dword(), A64, header);
        cmp(1 | eq | f0[0] | sb0.dst, null.ud(), data2.ud(0),
                uint32_t(MAGICPTR));
        jmpi(1 | ~f0[0], doWrite);

        // Validate OCL local size arguments
        cmp(1 | eq | f0[0], null.ud(), getLocalSize(0), uint32_t(MAGICSIZEX));
        jmpi(1 | ~f0[0], doWrite);
        cmp(1 | eq | f0[0], null.ud(), getLocalSize(1), uint32_t(MAGICSIZEY));
        jmpi(1 | ~f0[0], doWrite);
        cmp(1 | eq | f0[0], null.ud(), getLocalSize(2), uint32_t(MAGICSIZEZ));
        jmpi(1 | ~f0[0], doWrite);

        // Test passed.
        mov(1, ok, uint16_t(1));

        mark(doWrite);

        // Write out results.
        if (hw >= HW::XeHPC) {
            // stateless
            mov<uint32_t>(2, header, getArgument("ok").d(0)(1));
            store.ugm(1 | SWSB(sb2, 1), D32(1), A64, header, data);
        } else {
            // bti surface
            mov<uint32_t>(1, header, uint16_t(0));
            store(1 | SWSB(sb2, 1), scattered_dword(), ok_surface, header,
                    data);
        }

        if (hw >= HW::XeHP) memfence(sb2, header);
        mov<uint32_t>(8, r127, r0);
        threadend(SWSB(sb2, 1), r127);
    }

    static compute::kernel_t make_kernel(compute::compute_engine_t *engine) {
        compute::kernel_t kernel;

        if (hw != HW::Unknown) {
            binary_format_kernel_t<hw> binary_format_kernel;

            auto status
                    = engine->create_kernel(&kernel, &binary_format_kernel, {});
            if (status != status::success) return nullptr;
        } else {
            switch (engine->device_info()->gpu_arch()) {
                case compute::gpu_arch_t::gen9:
                    kernel = binary_format_kernel_t<HW::Gen9>::make_kernel(
                            engine);
                    break;
                case compute::gpu_arch_t::gen11:
                    kernel = binary_format_kernel_t<HW::Gen11>::make_kernel(
                            engine);
                    break;
                case compute::gpu_arch_t::xe_lp:
                    kernel = binary_format_kernel_t<HW::XeLP>::make_kernel(
                            engine);
                    break;
                case compute::gpu_arch_t::xe_hp:
                    kernel = binary_format_kernel_t<HW::XeHP>::make_kernel(
                            engine);
                    break;
                case compute::gpu_arch_t::xe_hpg:
                    kernel = binary_format_kernel_t<HW::XeHPG>::make_kernel(
                            engine);
                    break;
                case compute::gpu_arch_t::xe_hpc:
                    kernel = binary_format_kernel_t<HW::XeHPC>::make_kernel(
                            engine);
                    break;
                case compute::gpu_arch_t::unknown: kernel = nullptr; break;
            }
        }
        return kernel;
    }
};

status_t gpu_supports_binary_format(bool *ok, engine_t *engine) {
    *ok = false;

    auto gpu_engine = utils::downcast<compute::compute_engine_t *>(engine);
    if (!gpu_engine) return status::invalid_arguments;

    stream_t *stream_generic;
    auto status = gpu_engine->get_service_stream(stream_generic);
    if (status != status::success) return status::runtime_error;

    auto stream = utils::downcast<compute::compute_stream_t *>(stream_generic);
    if (!stream) return status::invalid_arguments;

    auto kernel = binary_format_kernel_t<HW::Unknown>::make_kernel(gpu_engine);
    if (!kernel) return status::success;

    // Binary kernel check.
    uint32_t magic0 = MAGIC0;
    uint64_t magic1 = MAGIC1;
    uint16_t magic2 = MAGIC2;
    uint16_t magic3 = MAGIC3;
    uint64_t magic4 = MAGIC4;
    uint64_t magic5 = MAGIC5;
    uint32_t magic_ptr = MAGICPTR;

    size_t gws[3] = {MAGICSIZEX, MAGICSIZEY, MAGICSIZEZ};
    size_t lws[3] = {MAGICSIZEX, MAGICSIZEY, MAGICSIZEZ};

    memory_storage_t *storage = nullptr;
    std::unique_ptr<memory_storage_t> magic_buf, result_buf;

    status = engine->create_memory_storage(&storage, sizeof(int32_t));
    if (status != status::success) return status::runtime_error;
    magic_buf.reset(storage);

    status = engine->create_memory_storage(&storage, sizeof(int32_t));
    if (status != status::success) return status::runtime_error;
    result_buf.reset(storage);

    void *magic_host = nullptr;
    magic_buf->map_data(&magic_host, nullptr, sizeof(int32_t));
    if (!magic_host) return status::runtime_error;

    *reinterpret_cast<uint32_t *>(magic_host) = magic_ptr;

    magic_buf->unmap_data(magic_host, nullptr);

    void *result_host = nullptr;
    result_buf->map_data(&result_host, nullptr, sizeof(int32_t));
    if (!result_host) return status::runtime_error;

    *reinterpret_cast<uint32_t *>(result_host) = 0;

    result_buf->unmap_data(result_host, nullptr);

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, magic0);
    arg_list.set(1, magic1);
    arg_list.set(2, magic2);
    arg_list.set(3, magic3);
    arg_list.set(4, magic4);
    arg_list.set(5, magic5);
    arg_list.set(6, *magic_buf.get());
    arg_list.set(7, *result_buf.get());

    auto nd_range = compute::nd_range_t(gws, lws);

    auto compute_stream = utils::downcast<compute::compute_stream_t *>(stream);
    status = kernel.parallel_for(*stream, nd_range, arg_list,
            compute_stream->ctx().get_deps(), compute_stream->ctx().get_deps());

    if (status != status::success) return status::runtime_error;

    status = stream->wait();
    if (status != status::success) return status::runtime_error;

    result_host = nullptr;
    result_buf->map_data(&result_host, nullptr, sizeof(int32_t));
    if (!result_host) return status::runtime_error;

    auto result = *reinterpret_cast<uint32_t *>(result_host);

    result_buf->unmap_data(result_host, nullptr);

    *ok = (result != 0);

    return status::success;
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
