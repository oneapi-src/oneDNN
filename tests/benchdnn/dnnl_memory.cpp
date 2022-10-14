/*******************************************************************************
* Copyright 2019-2022 Intel Corporation
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

#include <algorithm>
#include <atomic>
#include <cctype>
#include <memory>
#include <numeric>
#include <string>

#include "oneapi/dnnl/dnnl.h"

#ifdef DNNL_WITH_SYCL
#include "oneapi/dnnl/dnnl_sycl.hpp"
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#include "oneapi/dnnl/dnnl_ocl.hpp"
#include "src/gpu/ocl/ocl_usm_utils.hpp"
#endif

#include "tests/test_thread.hpp"

#include "dnn_types.hpp"
#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"
#include "utils/dnnl_query.hpp"
#include "utils/parallel.hpp"

extern "C" dnnl_status_t dnnl_memory_desc_create_with_string_tag(
        dnnl_memory_desc_t *, int, const dnnl_dims_t, dnnl_data_type_t,
        const char *);

extern "C" dnnl_status_t dnnl_memory_desc_set_data_type(
        dnnl_memory_desc_t memory_desc, dnnl_data_type_t data_type);

dnn_mem_t::dnn_mem_t(const_dnnl_memory_desc_t md, dnnl_engine_t engine,
        const handle_info_t &handle_info) {
    if (query_md_ndims(md) > 0) {
        auto status = dnnl_memory_desc_clone(&md_, md);
        (void)status;
        assert(status == dnnl_success);
        active_ = (initialize(engine, handle_info) == OK);
    }
}

dnn_mem_t::dnn_mem_t(const_dnnl_memory_desc_t md, dnnl_data_type_t dt,
        const std::string &tag, dnnl_engine_t engine) {
    auto md_wrapper = dnn_mem_t::init_md(
            query_md_ndims(md), query_md_dims(md), dt, tag);
    md_ = md_wrapper.release();
    if (ndims() > 0) active_ = (initialize(engine) == OK);
}

dnn_mem_t::dnn_mem_t(int ndims, const dnnl_dims_t dims, dnnl_data_type_t dt,
        const std::string &tag, dnnl_engine_t engine) {
    auto md_wrapper = dnn_mem_t::init_md(ndims, dims, dt, tag);
    md_ = md_wrapper.release();
    if (ndims > 0) active_ = (initialize(engine) == OK);
}

dnn_mem_t::dnn_mem_t(int ndims, const dnnl_dims_t dims, dnnl_data_type_t dt,
        const dnnl_dims_t strides, dnnl_engine_t engine) {
    auto status = dnnl_memory_desc_create_with_strides(
            &md_, ndims, dims, dt, strides);
    (void)status;
    assert(status == dnnl_success);
    if (ndims > 0) active_ = (initialize(engine) == OK);
}

dnn_mem_t::dnn_mem_t(const dnn_mem_t &rhs, dnnl_data_type_t dt,
        const std::string &tag, dnnl_engine_t engine)
    : dnn_mem_t(rhs.md_, dt, tag, engine) {
    if (active_) reorder(rhs);
}

int execute_reorder(const dnn_mem_t &src, dnn_mem_t &dst,
        const_dnnl_primitive_attr_t attr) {
    std::shared_ptr<const dnn_mem_t> r_src(&src, [](const dnn_mem_t *) {});
    std::shared_ptr<dnn_mem_t> r_dst(&dst, [](dnn_mem_t *) {});

    dnnl_primitive_desc_t r_pd_ {};
    dnnl_primitive_t prim_ {};

    // Optimization to reduce testing time for GPU.
    //
    // For CPU <-> GPU reorders, the library creates GPU-side kernels.
    // Benchdnn heavily relies on reorders and this greatly increases execution
    // time because of big overhead on building OpenCL kernels.
    //
    // First, try to create CPU reorder for the requested GPU reorder. If
    // succeeded, then create CPU memory object wrapping mapped pointers of
    // source and destination and execute CPU reorder. If CPU reorder can't be
    // create, then just execute a regular GPU reorder.
    //
    // This optimization is skipped when testing reorder, sum and concat
    // primitives because they are used specifically to test GPU reorders.
#if ((DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL) \
        || (DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL)) \
        && DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
    bool is_reorder_related_driver = (driver_name == "reorder"
            || driver_name == "sum" || driver_name == "concat");
    const auto &cpu_engine = get_cpu_engine();
    if (!is_reorder_related_driver
            && (src.engine_kind() == dnnl_gpu
                    || dst.engine_kind() == dnnl_gpu)) {

        dnnl_status_t status = dnnl_reorder_primitive_desc_create(
                &r_pd_, src.md_, cpu_engine, dst.md_, cpu_engine, attr);
        if (status == dnnl_success) {
            // Create CPU memory objects wrapping mapped pointers of source and
            // destination
            r_src = std::make_shared<dnn_mem_t>(dnn_mem_t::create_from_host_ptr(
                    src.md_, cpu_engine, (void *)src));
            r_dst = std::make_shared<dnn_mem_t>(dnn_mem_t::create_from_host_ptr(
                    dst.md_, cpu_engine, (void *)dst));
        }
    }
#endif

    if (!r_pd_) {
        DNN_SAFE(dnnl_reorder_primitive_desc_create(&r_pd_, src.md_,
                         src.engine(), dst.md_, dst.engine(), attr),
                WARN);
    }
    auto r_pd = make_benchdnn_dnnl_wrapper(r_pd_);
    const auto &scratchpad_md = query_md(r_pd, DNNL_ARG_SCRATCHPAD);
    dnn_mem_t scratchpad(scratchpad_md, src.engine());

    DNN_SAFE(dnnl_primitive_create(&prim_, r_pd), CRIT);
    auto prim = make_benchdnn_dnnl_wrapper(prim_);

    args_t args;
    args.set(DNNL_ARG_FROM, *r_src);
    args.set(DNNL_ARG_TO, *r_dst);
    args.set(DNNL_ARG_SCRATCHPAD, scratchpad);

    return execute_and_wait(prim, args);
}
int dnn_mem_t::reorder(const dnn_mem_t &rhs, const_dnnl_primitive_attr_t attr) {
    if (this == &rhs) return OK;
    return execute_reorder(rhs, *this, attr);
}

size_t dnn_mem_t::size() const {
    return dnnl_memory_desc_get_size(md_);
}

size_t dnn_mem_t::sizeof_dt() const {
    return dnnl_data_type_size(dt());
}

float dnn_mem_t::get_elem(int64_t idx) const {
    void *data = (void *)*this;
    float elem = 0.0;
    switch (dt()) {
        case dnnl_s8: elem = static_cast<int8_t *>(data)[idx]; break;
        case dnnl_u8: elem = static_cast<uint8_t *>(data)[idx]; break;
        case dnnl_s32: elem = static_cast<int32_t *>(data)[idx]; break;
        case dnnl_f32: elem = static_cast<float *>(data)[idx]; break;
        case dnnl_f64: elem = static_cast<double *>(data)[idx]; break;
        case dnnl_f16: elem = static_cast<float16_t *>(data)[idx]; break;
        case dnnl_bf16: elem = static_cast<bfloat16_t *>(data)[idx]; break;
        default: assert(!"bad data type");
    }
    return elem;
}

void dnn_mem_t::set_elem(int64_t idx, float value) const {
    void *data = (void *)*this;
    switch (dt()) {
        case dnnl_s8: ((int8_t *)data)[idx] = value; break;
        case dnnl_u8: ((uint8_t *)data)[idx] = value; break;
        case dnnl_s32: ((int32_t *)data)[idx] = value; break;
        case dnnl_f32: ((float *)data)[idx] = value; break;
        case dnnl_f64: ((double *)data)[idx] = value; break;
        case dnnl_f16: ((float16_t *)data)[idx] = value; break;
        case dnnl_bf16: ((bfloat16_t *)data)[idx] = value; break;
        default: assert(!"bad data type");
    }
}

// Creates a memory object from the underlying buffer of an existing memory
// object `mem`. The size of `mem` must not be less than the size of `md`.
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL || defined(DNNL_WITH_SYCL)
static int init_memory(
        dnnl_memory_t *ret, const dnnl_memory_desc_t &md, dnnl_memory_t mem) {
    void *handle;
    DNN_SAFE(dnnl_memory_get_data_handle(mem, &handle), CRIT);

    dnnl_engine_t engine;
    DNN_SAFE(dnnl_memory_get_engine(mem, &engine), CRIT);

    bool is_sycl = is_sycl_engine(engine);
    bool is_opencl = is_opencl_engine(engine);

    *ret = nullptr;

    if (is_opencl) {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
        dnnl_ocl_interop_memory_kind_t mem_kind;
        DNN_SAFE(dnnl_ocl_interop_memory_get_memory_kind(mem, &mem_kind), CRIT);
        DNN_SAFE(dnnl_ocl_interop_memory_create(
                         ret, md, engine, mem_kind, handle),
                CRIT);
#endif
    } else if (is_sycl) {
#ifdef DNNL_WITH_SYCL
        dnnl_sycl_interop_memory_kind_t mem_kind;
        DNN_SAFE(
                dnnl_sycl_interop_memory_get_memory_kind(mem, &mem_kind), CRIT);
        DNN_SAFE(dnnl_sycl_interop_memory_create(
                         ret, md, engine, mem_kind, handle),
                CRIT);
#endif
    }

    // Memory must be initialized at this point in some of the branches above.
    if (!*ret) assert(!"not expected");

    return OK;
}
#endif

void dnn_mem_t::map() const {
    assert(!is_mapped_ && "memory is already mapped");
    is_mapped_ = true;

    if (!m_) return;
    auto mem = m_padded_ ? m_padded_ : m_;
    DNN_SAFE_V(dnnl_memory_map_data(mem, &mapped_ptr_));
}

void dnn_mem_t::unmap() const {
    assert(is_mapped_ && "memory is not mapped");
    is_mapped_ = false;

    if (!m_) return;
    auto mem = m_padded_ ? m_padded_ : m_;
    DNN_SAFE_V(dnnl_memory_unmap_data(mem, mapped_ptr_));
    mapped_ptr_ = nullptr;
}

void dnn_mem_t::memset(int value, size_t size) const {
    bool is_opencl = is_opencl_engine(engine_);
    bool is_sycl = is_sycl_engine(engine_);
    auto mem = m_padded_ ? m_padded_ : m_;
    void *mem_handle;
    DNN_SAFE_V(dnnl_memory_get_data_handle(mem, &mem_handle));
    if (is_opencl) {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
        stream_t stream(engine_);
        switch (memory_kind) {
            case memory_kind_ext_t::buffer: {
                auto buf = static_cast<cl_mem>(mem_handle);
                cl_command_queue queue;
                DNN_SAFE_V(dnnl_ocl_interop_stream_get_command_queue(
                        stream, &queue));
                cl_int err = clEnqueueFillBuffer(queue, buf, &value,
                        sizeof(uint8_t), 0, size, 0, nullptr, nullptr);
                if (err != CL_SUCCESS) SAFE_V(FAIL);
                DNN_SAFE_V(dnnl_stream_wait(stream));
                return;
            }
            case memory_kind_ext_t::usm:
            case memory_kind_ext_t::usm_device:
            case memory_kind_ext_t::usm_shared: {
                DNN_SAFE_V(dnnl::impl::gpu::ocl::usm::memset(
                        stream, mem_handle, value, size));
                DNN_SAFE_V(dnnl_stream_wait(stream));
                return;
            }
        }
#endif
    } else if (is_sycl) {
#ifdef DNNL_WITH_SYCL
        stream_t stream(engine_);
        void *queue_ptr;
        DNN_SAFE_V(dnnl_sycl_interop_stream_get_queue(stream, &queue_ptr));
        auto &queue = *static_cast<::sycl::queue *>(queue_ptr);
        switch (memory_kind) {
            case memory_kind_ext_t::buffer: {
                auto &buf = *static_cast<::sycl::buffer<uint8_t, 1> *>(
                        mem_handle);
                queue.submit([&](::sycl::handler &cgh) {
#ifdef DNNL_SYCL_INTEROP_USE_SYCL121
                    constexpr auto target_device
                            = ::sycl::target::global_buffer;
#else
                    constexpr auto target_device = ::sycl::target::device;
#endif
                    ::sycl::accessor<uint8_t, 1, ::sycl::access::mode::write,
                            target_device>
                            acc(buf, cgh);
                    cgh.fill(acc, static_cast<uint8_t>(value));
                });
                DNN_SAFE_V(dnnl_stream_wait(stream));
                return;
            }
            case memory_kind_ext_t::usm:
            case memory_kind_ext_t::usm_device:
            case memory_kind_ext_t::usm_shared: {
                queue.submit([&](::sycl::handler &cgh) {
                    cgh.memset(mem_handle, value, size);
                });
                DNN_SAFE_V(dnnl_stream_wait(stream));
                return;
            }
        }
#endif
    }
    if (is_cpu(engine_)) {
        ::memset(mem_handle, value, size);
        return;
    }
    SAFE_V(FAIL);
}

dnn_mem_t dnn_mem_t::create_from_host_ptr(
        const dnnl_memory_desc_t &md, dnnl_engine_t engine, void *host_ptr) {
    return dnn_mem_t(md, engine, {true, host_ptr});
}

size_t dnn_mem_t::pad_memory_size(
        size_t sz, dnnl_engine_kind_t engine_kind, bool *was_padded) {
    if (was_padded) *was_padded = false;
    if (sz == 0 || !is_bench_mode(CORR) || engine_kind == dnnl_cpu) return sz;

    const int pad_size = 4096;
    if (was_padded) *was_padded = true;
    return sz + pad_size;
}

dnnl_memory_desc_t dnn_mem_t::pad_memory_desc(const_dnnl_memory_desc_t md,
        dnnl_engine_kind_t engine_kind, bool *was_padded) {
    if (was_padded) *was_padded = false;
    size_t old_sz = dnnl_memory_desc_get_size(md);
    if (old_sz == 0 || !is_bench_mode(CORR) || engine_kind == dnnl_cpu)
        return nullptr;

    size_t sz = pad_memory_size(old_sz, engine_kind, was_padded);
    if (sz == old_sz) return nullptr;

    dnnl_memory_desc_t ret;
    dnnl_dims_t dims = {(dnnl_dim_t)sz};
    DNN_SAFE_V(
            dnnl_memory_desc_create_with_tag(&ret, 1, dims, dnnl_u8, dnnl_x));
    return ret;
}

benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> dnn_mem_t::init_md(int ndims,
        const dnnl_dims_t dims, dnnl_data_type_t data_type,
        const std::string &tag_, const dims_t &strides_) {
    dnnl_memory_desc_t md {};
    const bool use_strides = !strides_.empty();
    // Ignore tag_ in case strides_ are explicitly provided
    if (use_strides) {
        std::vector<dnnl_dim_t> strides(strides_);
        DNN_SAFE_V(dnnl_memory_desc_create_with_strides(
                &md, ndims, dims, data_type, strides.data()));
        return md;
    }

    auto tag = normalize_tag(tag_, ndims);
    if (tag == tag::undef || tag == tag::any || ndims == 0) {
        dnnl_format_tag_t enum_tag = (tag == tag::undef || ndims == 0)
                ? dnnl_format_tag_undef
                : dnnl_format_tag_any;
        DNN_SAFE_V(dnnl_memory_desc_create_with_tag(
                &md, ndims, dims, data_type, enum_tag));
        return md;
    }

    DNN_SAFE_V(dnnl_memory_desc_create_with_string_tag(
            &md, ndims, dims, data_type, tag.data()));

    return md;
}

int dnn_mem_t::initialize_memory_create_sycl(const handle_info_t &handle_info) {
#ifdef DNNL_WITH_SYCL
    if (handle_info.is_host_ptr) {
        // Ignore memory_kind with host pointers and force USM.
        DNN_SAFE(dnnl_sycl_interop_memory_create(&m_, md_, engine_,
                         dnnl_sycl_interop_usm, handle_info.ptr),
                CRIT);
        return OK;
    }

    auto md_padded = pad_memory_desc(md_, engine_kind_, &is_canary_protected_);
    if (!md_padded) md_padded = md_;

    switch (memory_kind) {
        case memory_kind_ext_t::usm:
        case memory_kind_ext_t::buffer: {
            dnnl_sycl_interop_memory_kind_t mem_kind
                    = (memory_kind == memory_kind_ext_t::usm
                                    ? dnnl_sycl_interop_usm
                                    : dnnl_sycl_interop_buffer);
            DNN_SAFE(dnnl_sycl_interop_memory_create(&m_padded_, md_padded,
                             engine_, mem_kind, handle_info.ptr),
                    CRIT);
            SAFE(init_memory(&m_, md_, m_padded_), CRIT);
            break;
        }
        case memory_kind_ext_t::usm_device:
        case memory_kind_ext_t::usm_shared: {
            SAFE(handle_info.is_allocate() ? OK : FAIL, CRIT);
            is_data_owner_ = true;
            size_t sz = dnnl_memory_desc_get_size(md_padded);
            auto eng = dnnl::engine(engine_, true);
            auto dev = dnnl::sycl_interop::get_device(eng);
            auto ctx = dnnl::sycl_interop::get_context(eng);
            if (memory_kind == memory_kind_ext_t::usm_device) {
                data_ = ::sycl::malloc_device(sz, dev, ctx);
            } else {
                data_ = ::sycl::malloc_shared(sz, dev, ctx);
            }
            DNN_SAFE((sz > 0 && !data_) ? dnnl_out_of_memory : dnnl_success,
                    CRIT);
            DNN_SAFE(dnnl_sycl_interop_memory_create(&m_padded_, md_padded,
                             engine_, dnnl_sycl_interop_usm, data_),
                    CRIT);
            SAFE(init_memory(&m_, md_, m_padded_), CRIT);
            break;
        }
        default: assert(!"not expected");
    }
    if (md_padded != md_) DNN_SAFE(dnnl_memory_desc_destroy(md_padded), CRIT);

#else
    (void)handle_info;
#endif
    return OK;
}

int dnn_mem_t::initialize_memory_create_opencl(
        const handle_info_t &handle_info) {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    if (handle_info.is_host_ptr) {
        // Ignore memory_kind with host pointers and force USM.
        DNN_SAFE(dnnl_ocl_interop_memory_create(&m_, md_, engine_,
                         dnnl_ocl_interop_usm, handle_info.ptr),
                CRIT);
        return OK;
    }

    SAFE(handle_info.is_allocate() ? OK : FAIL, CRIT);

    auto md_padded = pad_memory_desc(md_, engine_kind_, &is_canary_protected_);
    if (!md_padded) md_padded = md_;

    switch (memory_kind) {
        case memory_kind_ext_t::usm:
        case memory_kind_ext_t::buffer: {
            dnnl_ocl_interop_memory_kind_t mem_kind
                    = (memory_kind == memory_kind_ext_t::usm
                                    ? dnnl_ocl_interop_usm
                                    : dnnl_ocl_interop_buffer);
            DNN_SAFE(dnnl_ocl_interop_memory_create(&m_padded_, md_padded,
                             engine_, mem_kind, handle_info.ptr),
                    CRIT);
            SAFE(init_memory(&m_, md_, m_padded_), CRIT);
            break;
        }
        case memory_kind_ext_t::usm_device:
        case memory_kind_ext_t::usm_shared: {
            is_data_owner_ = true;
            size_t sz = dnnl_memory_desc_get_size(md_padded);
            if (memory_kind == memory_kind_ext_t::usm_device) {
                data_ = dnnl::impl::gpu::ocl::usm::malloc_device(engine_, sz);
            } else {
                data_ = dnnl::impl::gpu::ocl::usm::malloc_shared(engine_, sz);
            }
            DNN_SAFE((sz > 0 && !data_) ? dnnl_out_of_memory : dnnl_success,
                    CRIT);
            DNN_SAFE(dnnl_ocl_interop_memory_create(&m_padded_, md_padded,
                             engine_, dnnl_ocl_interop_usm, data_),
                    CRIT);
            SAFE(init_memory(&m_, md_, m_padded_), CRIT);
            break;
        }
        default: assert(!"not expected");
    }
    if (md_padded != md_) DNN_SAFE(dnnl_memory_desc_destroy(md_padded), CRIT);
#else
    (void)handle_info;
#endif
    return OK;
}

int dnn_mem_t::initialize_memory_create(const handle_info_t &handle_info) {
    bool is_sycl = is_sycl_engine(engine_);
    bool is_opencl = is_opencl_engine(engine_);

    if (handle_info.is_host_ptr) {
        // Host pointer can be used with CPU memory only.
        // XXX: assumption is that SYCL can work with native host pointers.
        SAFE(is_cpu(engine_) ? OK : FAIL, CRIT);
    }

    if (is_cpu(engine_) && handle_info.is_allocate() && !is_sycl) {
        // Allocate memory for native runtime directly.
        is_data_owner_ = true;
        const size_t alignment = 2 * 1024 * 1024;
        size_t sz = dnnl_memory_desc_get_size(md_);
        data_ = zmalloc(sz, alignment);
        DNN_SAFE(!data_ ? dnnl_out_of_memory : dnnl_success, CRIT);
        DNN_SAFE(dnnl_memory_create(&m_, md_, engine_, data_), CRIT);
    } else if (is_sycl) {
        SAFE(initialize_memory_create_sycl(handle_info), CRIT);
    } else if (is_opencl) {
        SAFE(initialize_memory_create_opencl(handle_info), CRIT);
    } else {
        is_data_owner_ = false;
        data_ = nullptr;
        DNN_SAFE(dnnl_memory_create(&m_, md_, engine_, handle_info.ptr), CRIT);
    }
    return OK;
}

int dnn_mem_t::initialize(
        dnnl_engine_t engine, const handle_info_t &handle_info) {
    is_mapped_ = false;
    engine_ = engine;
    engine_kind_ = query_engine_kind(engine_);

    SAFE(initialize_memory_create(handle_info), CRIT);

    size_t sz = dnnl_memory_desc_get_size(md_);
    if (is_canary_protected_) sz = pad_memory_size(sz, engine_kind_);

    // Do not fill a memory if its size is zero. Moreover, memset expects
    // defined pointer, nullptr is not allowed.
    if (sz != 0 && handle_info.is_allocate()) {
        // Fill memory with a magic number (NAN for fp data types) to catch
        // possible uninitialized access.
        map();
        ::memset(mapped_ptr_, dnnl_mem_default_value, sz);
        unmap();
    }

    // Keep memory mapped and unmap only before execution
    map();

    return OK;
}

static int cleanup_sycl(const dnnl_engine_t &engine, void *data) {
#ifdef DNNL_WITH_SYCL
    switch (memory_kind) {
        case memory_kind_ext_t::usm_device:
        case memory_kind_ext_t::usm_shared: {
            auto eng = dnnl::engine(engine, true);
            auto ctx = dnnl::sycl_interop::get_context(eng);
            ::sycl::free(data, ctx);
            break;
        }
        default: break;
    }
#endif
    return OK;
}

static int cleanup_opencl(const dnnl_engine_t &engine, void *data) {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    switch (memory_kind) {
        case memory_kind_ext_t::usm_device:
        case memory_kind_ext_t::usm_shared:
            dnnl::impl::gpu::ocl::usm::free(engine, data);
            break;
        default: break;
    }
#endif
    return OK;
}

int dnn_mem_t::cleanup() {
    if (!active_) return OK;
    unmap();
    DNN_SAFE(dnnl_memory_desc_destroy(md_), CRIT);
    DNN_SAFE(dnnl_memory_destroy(m_), CRIT);
    if (is_data_owner_) {
        if (is_sycl_engine(engine_)) {
            SAFE(cleanup_sycl(engine_, data_), CRIT);
        } else if (is_opencl_engine(engine_)) {
            SAFE(cleanup_opencl(engine_, data_), CRIT);
        } else {
            zfree(data_);
        }
    }
    DNN_SAFE(dnnl_memory_destroy(m_padded_), CRIT);
    return OK;
}

void dnn_mem_t::set_dt(dnnl_data_type_t dt) const {
    // NOLINTNEXTLINE(readability-make-member-function-const)
    dnnl_memory_desc_set_data_type(md_, dt);
}

// Queries from memory descriptor.
int dnn_mem_t::ndims() const {
    return query_md_ndims(md_);
}

// Can't merge two below because compiler doesn't like conversion from
// pointer to reference type.
const dnnl_dims_t &dnn_mem_t::dims() const {
    return query_md_dims(md_);
}

const dnnl_dims_t &dnn_mem_t::padded_dims() const {
    return query_md_padded_dims(md_);
}

dnnl_data_type_t dnn_mem_t::dt() const {
    return query_md_data_type(md_);
}

const dnnl_dims_t &dnn_mem_t::padded_offsets() const {
    return query_md_padded_offsets(md_);
}

dnnl_dim_t dnn_mem_t::offset0() const {
    return query_md_submemory_offset(md_);
}

dnnl_format_kind_t dnn_mem_t::format_kind() const {
    return query_md_format_kind(md_);
}

const dnnl_dims_t &dnn_mem_t::strides() const {
    return query_md_strides(md_);
}

int dnn_mem_t::inner_nblks() const {
    return query_md_inner_nblks(md_);
}

const dnnl_dims_t &dnn_mem_t::inner_blks() const {
    return query_md_inner_blks(md_);
}

const dnnl_dims_t &dnn_mem_t::inner_idxs() const {
    return query_md_inner_idxs(md_);
}

// Returns physical offset by logical one. logical offset is represented by a
// scalar l_offset. If is_pos_padded is true, l_offset represents logical
// offset in already padded area.
static dnnl_dim_t md_off_l(dnnl_dims_t _pos, const dnn_mem_t &mem,
        dnnl_dim_t l_offset, bool is_pos_padded = false) {
    dnnl_dims_t pos;
    const auto &_dims = is_pos_padded ? mem.padded_dims() : mem.dims();
    for (int rd = 0; rd < mem.ndims(); ++rd) {
        const int d = mem.ndims() - 1 - rd;
        const dnnl_dim_t cur_dim = _dims[d];
        pos[d] = l_offset % cur_dim;
        if (_pos) _pos[d] = pos[d];
        l_offset /= cur_dim;
    }
    return md_off_v(mem, pos, is_pos_padded);
}

template <typename T>
static int check_zero_padding_impl(
        const dnn_mem_t &mem, int arg, res_t *res, int *error_count) {
    const int ndims = mem.ndims();
    const auto &dims = mem.dims();
    const auto &pdims = mem.padded_dims();

    if (ndims == 0) return OK;
    if (mem.format_kind() != dnnl_blocked) return OK;

    auto product = [](const dnnl_dim_t *beg, const dnnl_dim_t *end) {
        return std::accumulate(
                beg, end, (dnnl_dim_t)1, std::multiplies<dnnl_dim_t>());
    };

    int errors = 0;
    std::atomic<int> ok(true);

    const T *mem_ptr = (const T *)mem;

    for (int dim_m_idx = 0; dim_m_idx < ndims; ++dim_m_idx) {
        if (dims[dim_m_idx] == pdims[dim_m_idx]) continue;

        auto dim_l = product(pdims, pdims + dim_m_idx);
        auto dim_r = product(pdims + dim_m_idx + 1, pdims + ndims);

        benchdnn_parallel_nd(dim_l, dim_r, [&](dnnl_dim_t l, dnnl_dim_t r) {
            for (dnnl_dim_t m = dims[dim_m_idx]; m < pdims[dim_m_idx]; ++m) {
                auto l_idx = (l * pdims[dim_m_idx] + m) * dim_r + r;
                auto idx = md_off_l(nullptr, mem, l_idx, true);
                if (!(mem_ptr[idx] == 0)) ok = false;
            }
        });

        // Run the check one more time to report incorrect elements. This check
        // is sequential.
        if (!ok) {
            for_(dnnl_dim_t l = 0; l < dim_l; ++l)
            for_(dnnl_dim_t m = dims[dim_m_idx]; m < pdims[dim_m_idx]; ++m)
            for (dnnl_dim_t r = 0; r < dim_r; ++r) {
                auto l_idx = (l * pdims[dim_m_idx] + m) * dim_r + r;
                dnnl_dims_t pos = {};
                auto idx = md_off_l(pos, mem, l_idx, true);

                bool idx_ok = (mem_ptr[idx] == 0);
                if (!idx_ok) errors++;

                const bool dump = (!idx_ok && (errors < 10 || verbose >= 10))
                        || (verbose >= 99);
                if (dump) {
                    BENCHDNN_PRINT(0,
                            "[%4ld][arg:%d]"
                            "[" IFMT "," IFMT "," IFMT "," IFMT "," IFMT
                            "," IFMT "] fp:  0.f dt:% 9.6g \n",
                            (long)idx, arg, pos[0], pos[1], pos[2], pos[3],
                            pos[4], pos[5], mem.get_elem(idx));
                }
            }
        }
    }

    if (!ok) {
        BENCHDNN_PRINT(0, "@@@ [arg:%d] check_zero_padding failed\n", arg);
        if (res) res->state = FAILED;
    }

    if (error_count != nullptr) *error_count = errors;

    return ok ? OK : FAIL;
}

int check_zero_padding(
        const dnn_mem_t &mem, int arg, res_t *res, int *error_count) {
#define CASE(dt, type) \
    case dt: return check_zero_padding_impl<type>(mem, arg, res, error_count);

    switch (mem.dt()) {
        case dnnl_data_type_undef:
            return OK;

            CASE(dnnl_bf16, bfloat16_t);
            CASE(dnnl_f16, float16_t);
            CASE(dnnl_f32, float);
            CASE(dnnl_f64, double);
            CASE(dnnl_s32, int32_t);
            CASE(dnnl_s8, int8_t);
            CASE(dnnl_u8, uint8_t);

        default: assert(!"bad data_type");
    };
#undef CASE

    return FAIL;
}

int check_buffer_overwrite(const dnn_mem_t &mem, int arg, res_t *res) {
    if (!mem.is_canary_protected()) return OK;

    size_t sz = mem.size();
    size_t sz_padded = dnn_mem_t::pad_memory_size(sz, mem.engine_kind());

    auto *mem_ptr = (const uint8_t *)mem;
    for (size_t i = sz; i < sz_padded; i++) {
        if (mem_ptr[i] == dnnl_mem_default_value) continue;

        BENCHDNN_PRINT(0,
                "@@@ [arg:%d] check_buffer_overwrite failed. Expected: %d at "
                "byte: %lld but found: %d\n",
                arg, dnnl_mem_default_value, (long long)i, mem_ptr[i]);
        if (res) res->state = FAILED;
        return FAIL;
    }
    return OK;
}

// Returns physical offset by logical one. Logical offset is represented by an
// array pos. If is_pos_padded is true pos represents the position in already
// padded area.
dnnl_dim_t md_off_v(
        const dnn_mem_t &mem, const dnnl_dims_t pos, bool is_pos_padded) {
    assert(mem.format_kind() == dnnl_blocked);

    dnnl_dims_t pos_copy = {0};
    for (int d = 0; d < mem.ndims(); ++d)
        pos_copy[d] = pos[d] + (is_pos_padded ? 0 : mem.padded_offsets()[d]);

    dnnl_dim_t phys_offset = mem.offset0();

    const int nblks = mem.inner_nblks();
    if (nblks > 0) {
        const auto &inner_idxs = mem.inner_idxs();
        const auto &inner_blks = mem.inner_blks();
        dnnl_dim_t blk_stride = 1;
        for (int iblk = nblks - 1; iblk >= 0; --iblk) {
            const int d = inner_idxs[iblk];

            dnnl_dim_t p = pos_copy[d] % inner_blks[iblk];
            pos_copy[d] /= inner_blks[iblk];

            phys_offset += p * blk_stride;
            blk_stride *= inner_blks[iblk];
        }
    }

    for (int d = 0; d < mem.ndims(); ++d) {
        const dnnl_dim_t p = pos_copy[d];
        phys_offset += p * mem.strides()[d];
    }

    return phys_offset;
}

dnnl_memory_desc_t clone_md(const_dnnl_memory_desc_t md) {
    dnnl_memory_desc_t cloned_md;
    auto status = dnnl_memory_desc_clone(&cloned_md, md);
    if (status != dnnl_success) return nullptr;
    return cloned_md;
}
