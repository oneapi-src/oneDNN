/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
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

#ifndef GRAPH_EXAMPLE_UTILS_HPP
#define GRAPH_EXAMPLE_UTILS_HPP

#include <mutex>
#include <unordered_set>

#include "oneapi/dnnl/dnnl_graph.hpp"
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#include "oneapi/dnnl/dnnl_graph_ocl.hpp"
#include "oneapi/dnnl/dnnl_ocl.hpp"
#elif DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
#include "oneapi/dnnl/dnnl_graph_sycl.hpp"
#include "oneapi/dnnl/dnnl_sycl.hpp"
#endif

#include "example_utils.hpp"

#ifndef UNUSED
#define UNUSED(x) ((void)(x))
#endif

/// Set any layout according to the connection relationship of partitions
///
/// @param partitions a list of partitions
/// @param id_to_set_any_layout a set of ids of logical tensors with any layout
///     type
void set_any_layout(const std::vector<dnnl::graph::partition> &partitions,
        std::unordered_set<size_t> &id_to_set_any_layout) {
    // mapping from output tensor id to the all supported flags of
    // supported partitions, we may only need outputs' supported flags
    std::unordered_map<size_t, std::vector<bool>> output_to_flag_map;
    for (const auto &p : partitions) {
        for (const auto &out : p.get_output_ports()) {
            size_t id = out.get_id();
            if (p.is_supported()
                    && output_to_flag_map.find(id)
                            == output_to_flag_map.end()) {
                output_to_flag_map[id] = {};
            }
        }

        for (const auto &in : p.get_input_ports()) {
            size_t id = in.get_id();
            auto iter = output_to_flag_map.find(id);
            if (iter != output_to_flag_map.end()) {
                // collect all of supported flags of this tensor's uses
                // Considering we have such a graph:
                //
                //   partition_A  partition_B
                //        \           |
                //      tensor1    tensor2
                //           \     /     |
                //         partition_C  unsupported partition
                //              |
                //           tensor3
                //              |
                //          framework op
                //
                // so the mapping of partition_A's output will be { true }
                // the mapping of partition_B's output will be { true, false }
                // The mapping of partition_C's output will be { false }
                // Only when all supported flags are true, users can set any
                // layout.
                iter->second.push_back(p.is_supported());
            }
        }
    }

    for (const auto &p : partitions) {
        // no need to set `any` layout if this partition is not supported
        if (!p.is_supported()) continue;
        for (const auto &in : p.get_input_ports()) {
            size_t id = in.get_id();
            auto iter = output_to_flag_map.find(id);
            // if this input tensor is not an output of another supported
            // partition, just skip
            if (iter == output_to_flag_map.end()) continue;
            std::vector<bool> flag_vec = iter->second;
            // check if all of uses of this tensor are supported partitions,
            // if not, no need to set ANY layout.
            bool need_set_any = std::all_of(flag_vec.begin(), flag_vec.end(),
                    [](const bool a) { return a; });
            if (!need_set_any) continue;

            /// record the id of logical tensor that will be set to ANY layout
            id_to_set_any_layout.insert(id);
        }
    }
}

struct cpu_deletor {
    cpu_deletor() = default;
    void operator()(void *ptr) {
        if (ptr) free(ptr);
    }
};

#ifdef DNNL_WITH_SYCL
struct sycl_deletor {
    sycl_deletor() = delete;
    ::sycl::context ctx_;
    sycl_deletor(const ::sycl::context &ctx) : ctx_(ctx) {}
    void operator()(void *ptr) {
        if (ptr) ::sycl::free(ptr, ctx_);
    }
};

void *sycl_malloc_wrapper(
        size_t size, size_t alignment, const void *dev, const void *ctx) {
    return malloc_shared(size, *static_cast<const ::sycl::device *>(dev),
            *static_cast<const ::sycl::context *>(ctx));
}

void sycl_free_wrapper(
        void *ptr, const void *device, const void *context, void *event) {
    // Device is not used in this example, but it may be useful for some users
    // application.
    UNUSED(device);
    // immediate synchronization here is for test purpose. For performance,
    // users may need to store the ptr and event and handle them separately
    if (event) {
        auto sycl_deps_ptr = static_cast<::sycl::event *>(event);
        sycl_deps_ptr->wait();
    }
    free(ptr, *static_cast<const ::sycl::context *>(context));
}
#endif

void allocate_graph_mem(std::vector<dnnl::graph::tensor> &tensors,
        const std::vector<dnnl::graph::logical_tensor> &lts,
        std::vector<std::shared_ptr<void>> &data_buffer,
        const dnnl::engine &eng) {
    tensors.reserve(lts.size());
    for (const auto &lt : lts) {
        const auto mem_size = lt.get_mem_size();

        // memory allocation
        data_buffer.push_back({});
        data_buffer.back().reset(malloc(mem_size), cpu_deletor {});

        dnnl::graph::tensor new_ts {lt, eng, data_buffer.back().get()};
        tensors.push_back(new_ts);
    }
}

void allocate_graph_mem(std::vector<dnnl::graph::tensor> &tensors,
        const std::vector<dnnl::graph::logical_tensor> &lts,
        std::vector<std::shared_ptr<void>> &data_buffer,
        std::unordered_map<size_t, dnnl::graph::tensor> &global_outputs_ts_map,
        const dnnl::engine &eng, bool is_input) {
    tensors.reserve(lts.size());
    for (const auto &lt : lts) {
        const auto lt_id = lt.get_id();
        const auto mem_size = lt.get_mem_size();

        // check if the input is an output of another partition
        if (is_input) {
            auto pos = global_outputs_ts_map.find(lt_id);
            if (pos != global_outputs_ts_map.end()) {
                tensors.push_back(pos->second);
                continue;
            }
        }

        // memory allocation
        data_buffer.push_back({});
        data_buffer.back().reset(malloc(mem_size), cpu_deletor {});

        dnnl::graph::tensor new_ts {lt, eng, data_buffer.back().get()};
        tensors.push_back(new_ts);

        // record the connection relationship between partitions
        if (!is_input) global_outputs_ts_map[lt_id] = tensors.back();
    }
}

#ifdef DNNL_WITH_SYCL
void allocate_sycl_graph_mem(std::vector<dnnl::graph::tensor> &tensors,
        const std::vector<dnnl::graph::logical_tensor> &lts,
        std::vector<std::shared_ptr<void>> &data_buffer, sycl::queue &q,
        const dnnl::engine &eng) {
    tensors.reserve(lts.size());
    for (const auto &lt : lts) {
        const auto mem_size = lt.get_mem_size();

        // memory allocation
        data_buffer.push_back({});
        data_buffer.back().reset(::sycl::malloc_shared(mem_size, q.get_device(),
                                         q.get_context()),
                sycl_deletor {q.get_context()});

        dnnl::graph::tensor new_ts {lt, eng, data_buffer.back().get()};
        tensors.push_back(new_ts);
    }
}

void allocate_sycl_graph_mem(std::vector<dnnl::graph::tensor> &tensors,
        const std::vector<dnnl::graph::logical_tensor> &lts,
        std::vector<std::shared_ptr<void>> &data_buffer,
        std::unordered_map<size_t, dnnl::graph::tensor> &global_outputs_ts_map,
        sycl::queue &q, const dnnl::engine &eng, bool is_input) {
    tensors.reserve(lts.size());
    for (const auto &lt : lts) {
        const auto lt_id = lt.get_id();
        const auto mem_size = lt.get_mem_size();

        // check if the input is an output of another partition
        if (is_input) {
            auto pos = global_outputs_ts_map.find(lt_id);
            if (pos != global_outputs_ts_map.end()) {
                tensors.push_back(pos->second);
                continue;
            }
        }

        // memory allocation
        data_buffer.push_back({});
        data_buffer.back().reset(::sycl::malloc_shared(mem_size, q.get_device(),
                                         q.get_context()),
                sycl_deletor {q.get_context()});

        dnnl::graph::tensor new_ts {lt, eng, data_buffer.back().get()};
        tensors.push_back(new_ts);

        // record the connection relationship between partitions
        if (!is_input) global_outputs_ts_map[lt_id] = tensors.back();
    }
}
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#define OCL_CHECK(x) \
    do { \
        cl_int s = (x); \
        if (s != CL_SUCCESS) { \
            std::cout << "[" << __FILE__ << ":" << __LINE__ << "] '" << #x \
                      << "' failed (status code: " << s << ")." << std::endl; \
            exit(1); \
        } \
    } while (0)

static void *ocl_malloc_shared(
        size_t size, size_t alignment, cl_device_id dev, cl_context ctx) {
    using F = void *(*)(cl_context, cl_device_id, cl_ulong *, size_t, cl_uint,
            cl_int *);
    if (size == 0) return nullptr;

    cl_platform_id platform;
    OCL_CHECK(clGetDeviceInfo(
            dev, CL_DEVICE_PLATFORM, sizeof(platform), &platform, nullptr));
    const char *f_name = "clSharedMemAllocINTEL";
    auto f = reinterpret_cast<F>(
            clGetExtensionFunctionAddressForPlatform(platform, f_name));
    cl_int err;
    void *p = f(ctx, dev, nullptr, size, static_cast<cl_uint>(alignment), &err);
    OCL_CHECK(err);
    return p;
}

static void *ocl_malloc_device(
        size_t size, size_t alignment, cl_device_id dev, cl_context ctx) {
    using F = void *(*)(cl_context, cl_device_id, cl_ulong *, size_t, cl_uint,
            cl_int *);
    if (size == 0) return nullptr;

    cl_platform_id platform;
    OCL_CHECK(clGetDeviceInfo(
            dev, CL_DEVICE_PLATFORM, sizeof(platform), &platform, nullptr));
    const char *f_name = "clDeviceMemAllocINTEL";
    auto f = reinterpret_cast<F>(
            clGetExtensionFunctionAddressForPlatform(platform, f_name));
    cl_int err;
    void *p = f(ctx, dev, nullptr, size, static_cast<cl_uint>(alignment), &err);
    OCL_CHECK(err);
    return p;
}

static void ocl_free(
        void *ptr, cl_device_id dev, const cl_context ctx, cl_event event) {
    if (nullptr == ptr) return;
    using F = cl_int (*)(cl_context, void *);
    if (event) { OCL_CHECK(clWaitForEvents(1, &event)); }
    cl_platform_id platform;
    OCL_CHECK(clGetDeviceInfo(
            dev, CL_DEVICE_PLATFORM, sizeof(platform), &platform, nullptr));
    const char *f_name = "clMemBlockingFreeINTEL";
    auto f = reinterpret_cast<F>(
            clGetExtensionFunctionAddressForPlatform(platform, f_name));
    OCL_CHECK(f(ctx, ptr));
}

void allocate_ocl_graph_mem(std::vector<dnnl::graph::tensor> &tensors,
        const std::vector<dnnl::graph::logical_tensor> &lts,
        std::vector<std::shared_ptr<void>> &data_buffer,
        std::unordered_map<size_t, dnnl::graph::tensor> &global_outputs_ts_map,
        const dnnl::engine &eng, bool is_input) {
    tensors.reserve(lts.size());
    cl_context ctx = dnnl::ocl_interop::get_context(eng);
    cl_device_id dev = dnnl::ocl_interop::get_device(eng);

    for (const auto &lt : lts) {
        const auto lt_id = lt.get_id();
        const auto mem_size = lt.get_mem_size();

        // check if the input is an output of another partition
        if (is_input) {
            auto pos = global_outputs_ts_map.find(lt_id);
            if (pos != global_outputs_ts_map.end()) {
                tensors.push_back(pos->second);
                continue;
            }
        }

        // memory allocation
        data_buffer.push_back({});
        void *p = ocl_malloc_shared(
                mem_size, 0, dnnl::ocl_interop::get_device(eng), ctx);
        data_buffer.back().reset(
                p, [ctx, dev](void *p) { ocl_free(p, dev, ctx, {}); });
        dnnl::graph::tensor new_ts {lt, eng, data_buffer.back().get()};
        tensors.push_back(new_ts);

        // record the connection relationship between partitions
        if (!is_input) global_outputs_ts_map[lt_id] = tensors.back();
    }
}

void ocl_memcpy(dnnl::engine &eng, void *dst, const void *src, size_t size) {
    using F = cl_int (*)(cl_command_queue, cl_bool, void *, const void *,
            size_t, cl_uint, const cl_event *, cl_event *);
    if (!src || !dst) return;
    cl_platform_id platform;
    cl_context ctx = dnnl::ocl_interop::get_context(eng);
    cl_device_id dev = dnnl::ocl_interop::get_device(eng);
    cl_int err = 0;

// clCreateCommandQueue is deprecated in OpenCL.
#ifdef CL_VERSION_2_0
    cl_command_queue queue
            = clCreateCommandQueueWithProperties(ctx, dev, nullptr, &err);
#else
    cl_command_queue queue = clCreateCommandQueue(ctx, dev, {}, &err);
#endif
    if (err != CL_SUCCESS)
        throw std::runtime_error("cannot create a cl_command_queue");

    err = clGetDeviceInfo(
            dev, CL_DEVICE_PLATFORM, sizeof(platform), &platform, nullptr);
    if (err != CL_SUCCESS) throw std::runtime_error("clGetDeviceInfo failed");

    const char *f_name = "clEnqueueMemcpyINTEL";
    auto f = reinterpret_cast<F>(
            clGetExtensionFunctionAddressForPlatform(platform, f_name));
    err = f(queue, CL_FALSE, dst, src, size, 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
        throw std::runtime_error("clEnqueueMemcpyINTEL failed");

    return;
}
#endif

inline dnnl::memory::desc make_md(const dnnl::graph::logical_tensor &lt,
        dnnl::memory::data_type dt = dnnl::memory::data_type::undef) {
    using layout_type = dnnl::graph::logical_tensor::layout_type;
    using dims = dnnl::memory::dims;

    // if not specified, use the tensor data type.
    if (dt == dnnl::memory::data_type::undef)
        dt = static_cast<dnnl::memory::data_type>(lt.get_data_type());

    if (lt.get_layout_type() != layout_type::strided) {
        throw std::runtime_error("make_md: bad layout type");
    } else {
        const auto sz = lt.get_dims();
        const auto st = lt.get_strides();
        const auto nd = sz.size();
        if (nd > 0) {
            return dnnl::memory::desc(sz, dt, st);
        } else {
            // nd == 0
            return dnnl::memory::desc(dims {1}, dt, dims {1});
        }
    }
}

inline void write_dt(void *handle, dnnl::graph::tensor &ts) {
    dnnl::engine eng = ts.get_engine();
    size_t size = ts.get_logical_tensor().get_mem_size();

    if (!handle) throw std::runtime_error("handle is nullptr.");

#ifdef DNNL_WITH_SYCL
    bool is_cpu_sycl = (DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
            && eng.get_kind() == dnnl::engine::kind::cpu);
    bool is_gpu_sycl = (DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
            && eng.get_kind() == dnnl::engine::kind::gpu);
    if (is_cpu_sycl || is_gpu_sycl) {
        // only usm is supported in graph API.
        uint8_t *dst_ptr = (uint8_t *)ts.get_data_handle();
        if (!dst_ptr)
            throw std::runtime_error("get_data_handle returned nullptr.");
        if (is_cpu_sycl) {
            for (size_t i = 0; i < size; ++i)
                dst_ptr[i] = ((uint8_t *)handle)[i];
        } else {
            auto sycl_queue = dnnl::sycl_interop::get_queue(dnnl::stream(eng));
            sycl_queue.memcpy(dst_ptr, handle, size).wait();
        }
        return;
    }
#endif
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    if (eng.get_kind() == dnnl::engine::kind::gpu) {
        // only usm is supported in graph API.
        uint8_t *dst_ptr = (uint8_t *)ts.get_data_handle();
        if (!dst_ptr)
            throw std::runtime_error("get_data_handle returned nullptr.");
        ocl_memcpy(eng, dst_ptr, handle, size);
        return;
    }
#endif

    if (eng.get_kind() == dnnl::engine::kind::cpu) {
        uint8_t *dst = static_cast<uint8_t *>(ts.get_data_handle());
        if (!dst) throw std::runtime_error("get_data_handle returned nullptr.");
        for (size_t i = 0; i < size; ++i)
            dst[i] = ((uint8_t *)handle)[i];
        return;
    }

    assert(!"not expected");
}

// Read from handle, write to tensor. Assume handle contains f32 data.
inline void write_to_dnnl_tensor(void *handle, dnnl::graph::tensor &ts) {
    if (!handle) throw std::runtime_error("handle is nullptr.");

    dnnl::engine eng = ts.get_engine();
    const dnnl::graph::logical_tensor lt = ts.get_logical_tensor();
    const dnnl::graph::logical_tensor::data_type dt = lt.get_data_type();

    if (dt != dnnl::graph::logical_tensor::data_type::f32) {
        // if non-f32 data type, use reorder to convert.
        const auto f32_md = make_md(lt, dnnl::memory::data_type::f32);
        auto f32_mem = dnnl::memory(f32_md, eng);
        write_to_dnnl_memory(handle, f32_mem);

        const auto dt_md = make_md(lt);
        if (dt_md.get_size() != lt.get_mem_size()) {
            throw std::runtime_error("incorrect memory size.");
        }

        dnnl::memory dt_mem;
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
        if (eng.get_kind() == dnnl::engine::kind::gpu) {
            dt_mem = dnnl::ocl_interop::make_memory(dt_md, eng,
                    dnnl::ocl_interop::memory_kind::usm, ts.get_data_handle());
        } else
#endif
            dt_mem = dnnl::memory(dt_md, eng, ts.get_data_handle());

        dnnl::stream strm(eng);
        dnnl::reorder(f32_mem, dt_mem).execute(strm, f32_mem, dt_mem);
        strm.wait();
    } else {
        // directly write to ts.
        write_dt(handle, ts);
    }
}

// This memory pool is for sdpa example. The clear and set_capacity functions
// aren't thread safe. The multi-threaded scenario is mainly used in Graph
// Compiler backend.
// There is an copy in benchdnn graph. It's just for simplification.
// TODO: add some comments to clarify the design and interfaces.
class simple_memory_pool_t {
public:
#if defined(DNNL_WITH_SYCL) || DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#ifdef DNNL_WITH_SYCL
    void *allocate(
            size_t size, size_t alignment, const void *dev, const void *ctx)
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
            void *allocate(size_t size, size_t alignment, cl_device_id dev,
                    cl_context ctx)
#endif
    {
        std::lock_guard<std::mutex> pool_guard(pool_lock);
        // fake malloc for 0 size
        if (size == 0) return nullptr;

        void *ptr {nullptr};
        bool need_alloc_new_mm = true;
        // find alloc mm with same size
        const auto cnt = map_size_ptr_.count(size);
        if (cnt > 0) {
            const auto Iter = map_size_ptr_.equal_range(size);
            for (auto it = Iter.first; it != Iter.second; ++it) {
                // check if same size mm is free
                if (is_free_ptr_[it->second.get()]) {
                    ptr = it->second.get();
                    is_free_ptr_[ptr] = false;
                    need_alloc_new_mm = false;
                }
            }
        }
        if (need_alloc_new_mm) {
#ifdef DNNL_WITH_SYCL
            auto sh_ptr = std::shared_ptr<void> {
                    sycl_malloc_wrapper(size, alignment, dev, ctx),
                    sycl_deletor {*static_cast<const sycl::context *>(ctx)}};
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
            auto sh_ptr = std::shared_ptr<void> {
                    ocl_malloc_device(size, alignment, dev, ctx),
                    [ctx, dev](void *p) { ocl_free(p, dev, ctx, {}); }};
#endif
            ptr = sh_ptr.get();
            // record the map of mm size and its ptr for reuse
            map_size_ptr_.emplace(std::make_pair(size, sh_ptr));
            is_free_ptr_[ptr] = false;
        }
        return ptr;
    }
#endif

    void *allocate_host(size_t size, size_t alignment) {
        std::lock_guard<std::mutex> pool_guard(pool_lock);
        if (size == 0) return nullptr;

        void *ptr {nullptr};
        bool need_alloc_new_mm = true;
        // find alloc mm with same size
        const auto cnt = map_size_ptr_.count(size);
        if (cnt > 0) {
            const auto Iter = map_size_ptr_.equal_range(size);
            for (auto it = Iter.first; it != Iter.second; ++it) {
                // check if same size mm is free
                if (is_free_ptr_[it->second.get()]) {
                    ptr = it->second.get();
                    is_free_ptr_[ptr] = false;
                    need_alloc_new_mm = false;
                }
            }
        }
        if (need_alloc_new_mm) {
            auto sh_ptr = std::shared_ptr<void> {malloc(size), cpu_deletor {}};
            ptr = sh_ptr.get();
            // record the map of mm size and its ptr for reuse
            map_size_ptr_.emplace(std::make_pair(size, sh_ptr));
            is_free_ptr_[ptr] = false;
        }
        return ptr;
    }

#ifdef DNNL_WITH_SYCL
    void deallocate(
            void *ptr, const void *device, const void *context, void *event) {
        std::lock_guard<std::mutex> pool_guard(pool_lock);
        // This example currently supports `in_order`. So the kernel are
        // executed in the order in which they are submitted. Don't need to wait
        // event.
        is_free_ptr_[ptr] = true;
        return;
    }
#endif
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    void deallocate(
            void *ptr, cl_device_id dev, const cl_context ctx, cl_event event) {
        std::lock_guard<std::mutex> pool_guard(pool_lock);
        // This example currently supports `In-order`. So the kernel are
        // executed in the order in which they are submitted. Don't need to wait
        // event.
        is_free_ptr_[ptr] = true;
        return;
    }
#endif
    void deallocate_host(void *ptr) {
        std::lock_guard<std::mutex> pool_guard(pool_lock);
        is_free_ptr_[ptr] = true;
        return;
    }
    void clear() {
        dnnl::graph::set_compiled_partition_cache_capacity(0);
        map_size_ptr_.clear();
        is_free_ptr_.clear();
    }

private:
    std::mutex pool_lock;
    std::unordered_multimap<size_t, std::shared_ptr<void>> map_size_ptr_;
    std::unordered_map<void *, bool> is_free_ptr_;
};

simple_memory_pool_t &get_mem_pool() {
    static simple_memory_pool_t mem_pool;
    return mem_pool;
}

dnnl::graph::allocator create_allocator(dnnl::engine::kind ekind) {
    if (ekind == dnnl::engine::kind::cpu) {
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
        auto alloc_func = [](size_t size, size_t alignment, const void *dev,
                                  const void *ctx) -> void * {
            return get_mem_pool().allocate(size, alignment, dev, ctx);
        };
        auto dealloc_func = [](void *ptr, const void *device,
                                    const void *context, void *event) {
            return get_mem_pool().deallocate(ptr, device, context, event);
        };

        return dnnl::graph::sycl_interop::make_allocator(
                alloc_func, dealloc_func);
#else
        auto alloc_func = [](size_t size, size_t alignment) -> void * {
            return get_mem_pool().allocate_host(size, alignment);
        };
        auto dealloc_func
                = [](void *ptr) { return get_mem_pool().deallocate_host(ptr); };
        return dnnl::graph::allocator(alloc_func, dealloc_func);
#endif
    } else {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
        auto alloc_func = [](size_t size, size_t alignment, const void *dev,
                                  const void *ctx) -> void * {
            return get_mem_pool().allocate(size, alignment, dev, ctx);
        };
        auto dealloc_func = [](void *ptr, const void *device,
                                    const void *context, void *event) {
            return get_mem_pool().deallocate(ptr, device, context, event);
        };
        return dnnl::graph::sycl_interop::make_allocator(
                alloc_func, dealloc_func);
#elif DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
        auto alloc_func = [](size_t size, size_t alignment, cl_device_id dev,
                                  cl_context ctx) -> void * {
            return get_mem_pool().allocate(size, alignment, dev, ctx);
        };
        auto dealloc_func = [](void *ptr, cl_device_id dev,
                                    const cl_context ctx, cl_event event) {
            return get_mem_pool().deallocate(ptr, dev, ctx, event);
        };
        return dnnl::graph::ocl_interop::make_allocator(
                alloc_func, dealloc_func);
#endif
    }
    return dnnl::graph::allocator {};
}

#endif
