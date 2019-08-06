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

#include "mkldnn.hpp"

#if MKLDNN_CPU_RUNTIME == MKLDNN_RUNTIME_SYCL

#include "sycl/sycl_stream_submit_cpu_primitive.hpp"

#include "common/mkldnn_traits.hpp"
#include "common/nstl.hpp"
#include "common/primitive.hpp"
#include "common/stream.hpp"
#include "common/utils.hpp"
#include "sycl/sycl_memory_storage.hpp"

#include <CL/sycl.hpp>
#include <assert.h>
#include <exception>
#include <tuple>
#include <vector>

// A global scope tag type to use for enqueueing a single task
template <int memory_api_kind, typename... types>
class mkldnn_submit_primitive_tag_t;

namespace mkldnn {
namespace impl {
namespace sycl {

namespace {

template <memory_api_kind_t memory_api_kind>
struct init_thunk_params_t {};

template <>
struct init_thunk_params_t<memory_api_kind_t::buffer> {
    template <size_t N>
    static void call(thunk_params_t *p) {
        p->size = N;
    }

    template <size_t N, typename accessor_t, typename... accessor_types>
    static void call(
            thunk_params_t *p, accessor_t acc, accessor_types... accessors) {
        p->native_pointers[N - sizeof...(accessor_types) - 1]
                = reinterpret_cast<uintptr_t>(&acc[0]);
        call<N>(p, accessors...);
    }
};

#ifdef MKLDNN_SYCL_INTEL
template <>
struct init_thunk_params_t<memory_api_kind_t::usm> {
    template <size_t N>
    static void call(thunk_params_t *p) {
        p->size = N;
    }
    template <size_t N, typename... ptr_types>
    static void call(thunk_params_t *p, void *ptr, ptr_types... ptrs) {
        p->native_pointers[N - sizeof...(ptr_types) - 1]
                = reinterpret_cast<uintptr_t>(ptr);
        call<N>(p, ptrs...);
    }
};
#endif

template <memory_api_kind_t memory_api_kind>
struct make_kernel_tag_t {};

template <>
struct make_kernel_tag_t<memory_api_kind_t::buffer> {
    template <typename... accessor_types>
    using type = mkldnn_submit_primitive_tag_t<(int)memory_api_kind_t::buffer,
            typename accessor_types::value_type...>;
};

#ifdef MKLDNN_SYCL_INTEL
template <>
struct make_kernel_tag_t<memory_api_kind_t::usm> {
    template <typename... param_types>
    using type = mkldnn_submit_primitive_tag_t<(int)memory_api_kind_t::usm,
            param_types...>;
};
#endif

template <memory_api_kind_t memory_api_kind, typename... param_types>
status_t submit_cpu_primitive_with_params_impl(submit_ctx_t *submit_ctx,
        cl::sycl::handler &cgh, param_types... params) {
    // Trick the compiler by capturing scalar values in the kernel
    // instead of pointers what is not allowed.
    uintptr_t submit_ctx_ptr = reinterpret_cast<uintptr_t>(submit_ctx);
    using tag_type = typename make_kernel_tag_t<memory_api_kind>::template type<
            param_types...>;

    cgh.single_task<tag_type>([=]() {
        thunk_params_t thunk_params;
        thunk_params.submit_ctx_ptr = submit_ctx_ptr;

        constexpr size_t nparams = sizeof...(param_types);

        // Extract pointers from params
        init_thunk_params_t<memory_api_kind>::template call<nparams>(
                &thunk_params, params...);

        // Call C-linkage thunk which executes CPU primitive natively
        mkldnn_impl_sycl_cpu_thunk(&thunk_params);
    });
    return status::success;
}

template <memory_api_kind_t memory_api_kind>
struct submit_t {};

template <>
struct submit_t<memory_api_kind_t::buffer> {
    template <typename params_tuple_t, size_t... Is>
    static void call(submit_ctx_t *submit_ctx, cl::sycl::handler &cgh,
            params_tuple_t &params_tp, nstl::index_sequence<Is...>) {
        submit_cpu_primitive_with_params_impl<memory_api_kind_t::buffer>(
                submit_ctx, cgh,
                std::get<Is>(params_tp)
                        .template get_access<
                                cl::sycl::access::mode::read_write>(cgh)...);
    }
};

#ifdef MKLDNN_SYCL_INTEL
template <>
struct submit_t<memory_api_kind_t::usm> {
    template <typename params_tuple_t, size_t... Is>
    static void call(submit_ctx_t *submit_ctx, cl::sycl::handler &cgh,
            params_tuple_t &params_tp, nstl::index_sequence<Is...>) {
        submit_cpu_primitive_with_params_impl<memory_api_kind_t::usm>(
                submit_ctx, cgh, std::get<Is>(params_tp)...);
    }
};
#endif

memory_api_kind_t get_memory_api_kind() {
    // Shouldn't matter as an empty call happens only if no storages are
    // converted
    return memory_api_kind_t::buffer;
}

template <typename storage_t, typename... storage_types>
memory_api_kind_t get_memory_api_kind(
        const storage_t *storage, const storage_types *... storages) {
    return utils::downcast<const sycl_memory_storage_base_t *>(storage->impl())
            ->memory_api_kind();
}

template <typename... storage_types>
void fast_dispatch_by_size(submit_ctx_t *submit_ctx, cl::sycl::handler &cgh,
        const storage_types *... storages) {
    // XXX: all storages must be of the same memory API kind
    auto mem_api_kind = get_memory_api_kind(storages...);
    constexpr size_t nparams = sizeof...(storage_types);

    switch (mem_api_kind) {
    case memory_api_kind_t::buffer: {
        auto params_tp = std::make_tuple(
                utils::downcast<const sycl_buffer_memory_storage_t *>(
                        storages->impl())
                        ->buffer()...);
        submit_t<memory_api_kind_t::buffer>::call(submit_ctx, cgh, params_tp,
                nstl::make_index_sequence<nparams>{});
        break;
    }
#ifdef MKLDNN_SYCL_INTEL
    case memory_api_kind_t::usm: {
        auto params_tp = std::make_tuple(
                utils::downcast<const sycl_usm_memory_storage_t *>(
                        storages->impl())
                        ->usm_ptr()...);
        submit_t<memory_api_kind_t::usm>::call(submit_ctx, cgh, params_tp,
                nstl::make_index_sequence<nparams>{});
        break;
    }
#endif
    default: assert(!"not expected");
    }
}

} // namespace

// CPU primitive submission is implemented this way:
// 1. Obtain all accessible SYCL memory storages from iterating
//    over the execution context.
// 2. Use variadic templates to pass SYCL accessors for these
//    storages to the SYCL kernel inside single_task().
// 3. Stream, primitive and execution context pointers are
//    passed to the kernel via the submit context structure.
// 4. Pass a submit context via uintptr_t to work around
//    SYCL kernel restrictions. The context structure is
//    unpacked and deallocated on kernel side.
// 5. The SYCL kernel "registers" mapping
//    memory storage -> raw pointer via execution context.
// 6. Call the thunk function that executes the primitve
//    natively.
void submit_cpu_primitive(stream_t *stream, const primitive_t *prim,
        const exec_ctx_t &exec_ctx, cl::sycl::handler &cgh) {
    const_cast<primitive_t *>(prim)->retain();

    std::vector<const memory_storage_t *> sycl_mem_storages;
    for (auto &a : exec_ctx.args()) {
        if (a.second.mem->engine()->backend_kind() == backend_kind::sycl) {
            auto *mem_storage = a.second.mem->memory_storage();
            if (!mem_storage->is_null()) {
                sycl_mem_storages.push_back(mem_storage);
            }
        }
    }

    // Keep unique only
    std::sort(sycl_mem_storages.begin(), sycl_mem_storages.end(),
            [](const memory_storage_t *lhs, const memory_storage_t *rhs) {
                return lhs->impl() < rhs->impl();
            });
    auto last = std::unique(sycl_mem_storages.begin(), sycl_mem_storages.end(),
            [](const memory_storage_t *lhs, const memory_storage_t *rhs) {
                return lhs->impl() == rhs->impl();
            });
    sycl_mem_storages.erase(last, sycl_mem_storages.end());

    // XXX: validate that all the storages use the same memory API
    if (!sycl_mem_storages.empty()) {
        auto *mem0 = sycl_mem_storages[0];
        auto mem_api_kind0
                = utils::downcast<const sycl_memory_storage_base_t *>(mem0->impl())
                          ->memory_api_kind();
        for (auto *mem : sycl_mem_storages) {
            auto mem_api_kind = utils::downcast<const sycl_memory_storage_base_t *>(
                    mem->impl())
                                        ->memory_api_kind();
            if (mem_api_kind != mem_api_kind0) {
                throw std::runtime_error(
                        "Memory objects must use the same memory API");
            }
        }
    }

    auto *submit_ctx = new submit_ctx_t();
    submit_ctx->stream = stream;
    submit_ctx->prim = prim;
    submit_ctx->exec_ctx = exec_ctx;
    submit_ctx->sycl_mem_storages = sycl_mem_storages;

    switch (sycl_mem_storages.size()) {
    case 0: fast_dispatch_by_size(submit_ctx, cgh); break;
    case 1: fast_dispatch_by_size(submit_ctx, cgh, sycl_mem_storages[0]); break;
    case 2:
        fast_dispatch_by_size(
                submit_ctx, cgh, sycl_mem_storages[0], sycl_mem_storages[1]);
        break;
    case 3:
        fast_dispatch_by_size(submit_ctx, cgh, sycl_mem_storages[0],
                sycl_mem_storages[1], sycl_mem_storages[2]);
        break;
    case 4:
        fast_dispatch_by_size(submit_ctx, cgh, sycl_mem_storages[0],
                sycl_mem_storages[1], sycl_mem_storages[2],
                sycl_mem_storages[3]);
        break;
    case 5:
        fast_dispatch_by_size(submit_ctx, cgh, sycl_mem_storages[0],
                sycl_mem_storages[1], sycl_mem_storages[2],
                sycl_mem_storages[3], sycl_mem_storages[4]);
        break;
    case 6:
        fast_dispatch_by_size(submit_ctx, cgh, sycl_mem_storages[0],
                sycl_mem_storages[1], sycl_mem_storages[2],
                sycl_mem_storages[3], sycl_mem_storages[4],
                sycl_mem_storages[5]);
        break;
    case 7:
        fast_dispatch_by_size(submit_ctx, cgh, sycl_mem_storages[0],
                sycl_mem_storages[1], sycl_mem_storages[2],
                sycl_mem_storages[3], sycl_mem_storages[4],
                sycl_mem_storages[5], sycl_mem_storages[6]);
        break;
    case 8:
        fast_dispatch_by_size(submit_ctx, cgh, sycl_mem_storages[0],
                sycl_mem_storages[1], sycl_mem_storages[2],
                sycl_mem_storages[3], sycl_mem_storages[4],
                sycl_mem_storages[5], sycl_mem_storages[6],
                sycl_mem_storages[7]);
        break;
    case 9:
        fast_dispatch_by_size(submit_ctx, cgh, sycl_mem_storages[0],
                sycl_mem_storages[1], sycl_mem_storages[2],
                sycl_mem_storages[3], sycl_mem_storages[4],
                sycl_mem_storages[5], sycl_mem_storages[6],
                sycl_mem_storages[7], sycl_mem_storages[8]);
        break;
    case 10:
        fast_dispatch_by_size(submit_ctx, cgh, sycl_mem_storages[0],
                sycl_mem_storages[1], sycl_mem_storages[2],
                sycl_mem_storages[3], sycl_mem_storages[4],
                sycl_mem_storages[5], sycl_mem_storages[6],
                sycl_mem_storages[7], sycl_mem_storages[8],
                sycl_mem_storages[9]);
        break;
    case 11:
        fast_dispatch_by_size(submit_ctx, cgh, sycl_mem_storages[0],
                sycl_mem_storages[1], sycl_mem_storages[2],
                sycl_mem_storages[3], sycl_mem_storages[4],
                sycl_mem_storages[5], sycl_mem_storages[6],
                sycl_mem_storages[7], sycl_mem_storages[8],
                sycl_mem_storages[9], sycl_mem_storages[10]);
        break;
    case 12:
        fast_dispatch_by_size(submit_ctx, cgh, sycl_mem_storages[0],
                sycl_mem_storages[1], sycl_mem_storages[2],
                sycl_mem_storages[3], sycl_mem_storages[4],
                sycl_mem_storages[5], sycl_mem_storages[6],
                sycl_mem_storages[7], sycl_mem_storages[8],
                sycl_mem_storages[9], sycl_mem_storages[10],
                sycl_mem_storages[11]);
        break;
    case 13:
        fast_dispatch_by_size(submit_ctx, cgh, sycl_mem_storages[0],
                sycl_mem_storages[1], sycl_mem_storages[2],
                sycl_mem_storages[3], sycl_mem_storages[4],
                sycl_mem_storages[5], sycl_mem_storages[6],
                sycl_mem_storages[7], sycl_mem_storages[8],
                sycl_mem_storages[9], sycl_mem_storages[10],
                sycl_mem_storages[11], sycl_mem_storages[12]);
        break;
    case 14:
        fast_dispatch_by_size(submit_ctx, cgh, sycl_mem_storages[0],
                sycl_mem_storages[1], sycl_mem_storages[2],
                sycl_mem_storages[3], sycl_mem_storages[4],
                sycl_mem_storages[5], sycl_mem_storages[6],
                sycl_mem_storages[7], sycl_mem_storages[8],
                sycl_mem_storages[9], sycl_mem_storages[10],
                sycl_mem_storages[11], sycl_mem_storages[12],
                sycl_mem_storages[13]);
        break;
    case 15:
        fast_dispatch_by_size(submit_ctx, cgh, sycl_mem_storages[0],
                sycl_mem_storages[1], sycl_mem_storages[2],
                sycl_mem_storages[3], sycl_mem_storages[4],
                sycl_mem_storages[5], sycl_mem_storages[6],
                sycl_mem_storages[7], sycl_mem_storages[8],
                sycl_mem_storages[9], sycl_mem_storages[10],
                sycl_mem_storages[11], sycl_mem_storages[12],
                sycl_mem_storages[13], sycl_mem_storages[14]);
        break;
    case 16:
        fast_dispatch_by_size(submit_ctx, cgh, sycl_mem_storages[0],
                sycl_mem_storages[1], sycl_mem_storages[2],
                sycl_mem_storages[3], sycl_mem_storages[4],
                sycl_mem_storages[5], sycl_mem_storages[6],
                sycl_mem_storages[7], sycl_mem_storages[8],
                sycl_mem_storages[9], sycl_mem_storages[10],
                sycl_mem_storages[11], sycl_mem_storages[12],
                sycl_mem_storages[13], sycl_mem_storages[14],
                sycl_mem_storages[15]);
        break;
    case 17:
        fast_dispatch_by_size(submit_ctx, cgh, sycl_mem_storages[0],
                sycl_mem_storages[1], sycl_mem_storages[2],
                sycl_mem_storages[3], sycl_mem_storages[4],
                sycl_mem_storages[5], sycl_mem_storages[6],
                sycl_mem_storages[7], sycl_mem_storages[8],
                sycl_mem_storages[9], sycl_mem_storages[10],
                sycl_mem_storages[11], sycl_mem_storages[12],
                sycl_mem_storages[13], sycl_mem_storages[14],
                sycl_mem_storages[15], sycl_mem_storages[16]);
        break;
    case 18:
        fast_dispatch_by_size(submit_ctx, cgh, sycl_mem_storages[0],
                sycl_mem_storages[1], sycl_mem_storages[2],
                sycl_mem_storages[3], sycl_mem_storages[4],
                sycl_mem_storages[5], sycl_mem_storages[6],
                sycl_mem_storages[7], sycl_mem_storages[8],
                sycl_mem_storages[9], sycl_mem_storages[10],
                sycl_mem_storages[11], sycl_mem_storages[12],
                sycl_mem_storages[13], sycl_mem_storages[14],
                sycl_mem_storages[15], sycl_mem_storages[16],
                sycl_mem_storages[17]);
        break;
    case 19:
        fast_dispatch_by_size(submit_ctx, cgh, sycl_mem_storages[0],
                sycl_mem_storages[1], sycl_mem_storages[2],
                sycl_mem_storages[3], sycl_mem_storages[4],
                sycl_mem_storages[5], sycl_mem_storages[6],
                sycl_mem_storages[7], sycl_mem_storages[8],
                sycl_mem_storages[9], sycl_mem_storages[10],
                sycl_mem_storages[11], sycl_mem_storages[12],
                sycl_mem_storages[13], sycl_mem_storages[14],
                sycl_mem_storages[15], sycl_mem_storages[16],
                sycl_mem_storages[17], sycl_mem_storages[18]);
        break;
    case 20:
        fast_dispatch_by_size(submit_ctx, cgh, sycl_mem_storages[0],
                sycl_mem_storages[1], sycl_mem_storages[2],
                sycl_mem_storages[3], sycl_mem_storages[4],
                sycl_mem_storages[5], sycl_mem_storages[6],
                sycl_mem_storages[7], sycl_mem_storages[8],
                sycl_mem_storages[9], sycl_mem_storages[10],
                sycl_mem_storages[11], sycl_mem_storages[12],
                sycl_mem_storages[13], sycl_mem_storages[14],
                sycl_mem_storages[15], sycl_mem_storages[16],
                sycl_mem_storages[17], sycl_mem_storages[18],
                sycl_mem_storages[19]);
        break;
    default:
        delete submit_ctx;
        assert(!"Please add another case");
        throw std::runtime_error("Internal error");
    }
}

} // namespace sycl
} // namespace impl
} // namespace mkldnn

#endif
