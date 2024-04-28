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

#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#include "common/engine.hpp"
#include "common/utils.hpp"

#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
#include "cpu/cpu_engine.hpp"
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#include "gpu/intel/ocl/ocl_engine.hpp"
#endif

#ifdef DNNL_WITH_SYCL
#include "sycl/sycl_engine.hpp"
#endif

#include "graph/interface/backend.hpp"
#include "graph/interface/constant_tensor_cache.hpp"
#include "graph/utils/utils.hpp"

#ifdef _WIN32
#include <windows.h>
#endif

namespace std {
template <>
struct hash<dnnl::impl::engine_kind_t> {
    using argument_type = dnnl::impl::engine_kind_t;
    using result_type = std::size_t;
    result_type operator()(const argument_type &eng_kind) const {
        return static_cast<result_type>(eng_kind);
    }
};
} // namespace std

namespace dnnl {
namespace impl {
namespace graph {

using c_key_t = constant_tensor_cache_t::key_t;
using c_value_t = constant_tensor_cache_t::value_t;

static size_t get_timestamp() {
    return std::chrono::steady_clock::now().time_since_epoch().count();
}

constant_tensor_cache_t::constant_tensor_cache_t(
        size_t capacity_in_bytes, const std::string &name)
    : name_(name), capacity_in_bytes_(capacity_in_bytes), counter_(1) {
    constant_map_ = impl::utils::make_unique<
            std::unordered_map<c_key_t, timed_entry_t>>();
}

constant_tensor_cache_t::~constant_tensor_cache_t() {
    if (constant_map().empty()) return;

#if defined(_WIN32) && defined(DNNL_WITH_SYCL)
    // The library unloading issue affects only DPCPP runtimes on Windows when
    // DNNL_GRAPH_ENABLE_COMPILED_PARTITION_CACHE is ON. The ntdll.dll library
    // is located in system32 therefore setting additional environment is not
    // required.
    HMODULE handle = LoadLibraryExA(
            "ntdll.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!handle) {
        constant_map_.release();
        return;
    }

    // RtlDllShutdownInProgress returns TRUE if the whole process terminates and
    // FALSE if DLL is being unloaded dynamically or if it’s called from an
    // executable.
    auto f = reinterpret_cast<BOOLEAN (*)(void)>(
            GetProcAddress(handle, "RtlDllShutdownInProgress"));
    if (!f) {
        auto ret = FreeLibrary(handle);
        assert(ret);
        MAYBE_UNUSED(ret);
        constant_map_.release();
        return;
    }

    bool is_process_termination_in_progress = f();

    auto ret = FreeLibrary(handle);
    assert(ret);
    MAYBE_UNUSED(ret);

    if (is_process_termination_in_progress) {
        // The whole process is being terminated hence destroying content of the
        // primitive cache cannot be done safely. Theoretically, we can check
        // all entries and remove those that are not affected e.g. native CPU.
        // We can do this after switching to use dnnl engine.
        for (auto it = constant_map().begin(); it != constant_map().end();) {
#ifdef DNNL_WITH_SYCL
            ++it;
#else
            it = constant_map().erase(it);
#endif
        }
        constant_map_.release();
    } else {
        // Three scenarios possible:
        // 1. oneDNN Graph is being dynamically unloaded
        // 2. Another dynamic library that contains statically linked oneDNN
        //    Graph is dynamically unloaded
        // 3. oneDNN Graph is statically linked in an executable which is done
        //    and now the process terminates In all these scenarios content of
        //    the primitive cache can be safely destroyed.
        constant_map_.reset();
    }
#else
    // Always destroy the content of the constant tensor cache for
    // non-Windows OSes, and non-sycl and non-ocl runtimes because there is
    // no a problem with library unloading order in such cases.
    constant_map_.reset();
#endif
}

status_t constant_tensor_cache_t::set_capacity(size_t capacity) {
    lock_write();
    capacity_in_bytes_ = capacity;
    evict(get_size()); // completely flushed cache
    unlock_write();
    return status::success;
}

size_t constant_tensor_cache_t::get_capacity() {
    return capacity_in_bytes_.load();
}

c_key_t constant_tensor_cache_t::combine_key(
        c_key_t backend_id, c_key_t backend_specific_key) {
    size_t key = (backend_specific_key << BACKEND_ID_LENGTH)
            | (backend_id & (size_t)((1 << BACKEND_ID_LENGTH) - 1));
    return key;
}

c_value_t constant_tensor_cache_t::get_or_add(c_key_t backend_id,
        c_key_t backend_specific_key, size_t size, const c_value_t &value) {
    if (!size) { return c_value_t(); }

    c_key_t key = combine_key(backend_id, backend_specific_key);

    // 1. Section with shared access (read lock)
    lock_read();
    // Check if the cache is enabled.
    if (capacity_in_bytes_ == 0) {
        unlock_read();
        return c_value_t();
    }
    // Check if the requested entry is present in the cache (likely cache_hit)
    auto e = get(key);
    if (e.valid()) {
        unlock_read();
        return e;
    }

    unlock_read();

    // 2. Section with exclusive access (write lock).
    // In a multithreaded scenario, in the context of one thread the cache
    // may have changed by another thread between releasing the read lock and
    // acquiring the write lock (a.k.a. ABA problem), therefore additional
    // checks have to be performed for correctness.
    // Double check the capacity due to possible race condition
    lock_write();
    if (capacity_in_bytes_ == 0) {
        unlock_write();
        return c_value_t();
    }

    // Double check if the requested entry is present in the cache (unlikely
    // cache_hit).
    e = get(key);
    if (!e.valid()) {
        // If the entry is missing in the cache then add it (cache_miss)
        add(key, size, value);
    }
    unlock_write();
    return e;
}

void constant_tensor_cache_t::remove_if_exist(
        c_key_t backend_id, c_key_t backend_specific_key) {
    c_key_t key = combine_key(backend_id, backend_specific_key);

    lock_write();
    if (constant_map().count(key) == 0) {
        unlock_write();
    } else {
        // notify backend that this buffer will be evicted from the cache. If
        // backend hold a reference to this buffer, release it and don't use it
        // any more. Otherwise, the constant cache capacity may exceed the upper
        // bound and cause OOM in user application.
        auto &item = constant_map().at(key);
        item.value_.get()->notify_evict();
        constant_map().erase(key);
        unlock_write();
    }
}

// Get the total size of all cached buffers
size_t constant_tensor_cache_t::get_size() const {
    size_t total_size = 0;
    for (const auto &pair : constant_map()) {
        total_size += pair.second.value_.get()->size();
    }
    return total_size;
}

void constant_tensor_cache_t::add(
        const c_key_t &key, size_t size, const c_value_t &constant) {
    size_t current_size = get_size();

    // No enough capacity to cache the new tensor, ignore the new tensor
    // directly
    if (current_size + size > capacity_in_bytes_) { return; }

    // Cache tensors
    size_t timestamp = get_timestamp();

    auto res = constant_map().emplace(std::piecewise_construct,
            std::forward_as_tuple(key),
            std::forward_as_tuple(constant, timestamp));
    UNUSED(res);
    assert(res.second);
}

c_value_t constant_tensor_cache_t::get(const c_key_t &key) {
    auto it = constant_map().find(key);
    if (it == constant_map().end()) return c_value_t();

    size_t timestamp = get_timestamp();
    it->second.timestamp_.store(timestamp);
    // Return the entry
    return it->second.value_;
}

// Evict n size of cached buffers
void constant_tensor_cache_t::evict(size_t n) {
    using v_t = std::unordered_map<c_key_t, timed_entry_t>::value_type;
    if (n == get_size()) {
        constant_map().clear();
        return;
    }

    size_t evicted_size = 0;
    while (evicted_size < n) {
        // We evict the least recently used items
        auto it = std::min_element(constant_map().begin(), constant_map().end(),
                [&](const v_t &left, const v_t &right) {
                    // By default, load() and operator T use sequentially
                    // consistent memory ordering, which enforces writing the
                    // timestamps into registers in the same exact order they
                    // are read from the CPU cache line. Since eviction is
                    // performed under a write lock, this order is not
                    // important, therefore we can safely use the weakest memory
                    // ordering (relaxed).
                    return left.second.timestamp_.load(
                                   std::memory_order_relaxed)
                            < right.second.timestamp_.load(
                                    std::memory_order_relaxed);
                });
        evicted_size += it->second.value_.get()->size();
        auto res = constant_map().erase(it->first);
        UNUSED(res);
        assert(res);
    }
}

// copy from src/common/engine.cpp
static std::unique_ptr<impl::engine_factory_t> get_engine_factory(
        impl::engine_kind_t kind, impl::runtime_kind_t runtime_kind) {

#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
    if (kind == impl::engine_kind::cpu && is_native_runtime(runtime_kind)) {
        return std::unique_ptr<impl::engine_factory_t>(
                new impl::cpu::cpu_engine_factory_t());
    }
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    if (kind == engine_kind::gpu && runtime_kind == runtime_kind::ocl) {
        return std::unique_ptr<engine_factory_t>(
                new gpu::intel::ocl::ocl_engine_factory_t(kind));
    }
#endif

#ifdef DNNL_WITH_SYCL
    if (runtime_kind == impl::runtime_kind::sycl)
        return impl::sycl::get_engine_factory(kind);
#endif
    return nullptr;
}

using cache_ptr = std::shared_ptr<constant_tensor_cache_t>;

struct global_cache_manager_t {
public:
    global_cache_manager_t(const global_cache_manager_t &) = delete;
    global_cache_manager_t(global_cache_manager_t &&) = delete;
    global_cache_manager_t &operator=(const global_cache_manager_t &) = delete;
    global_cache_manager_t &operator=(global_cache_manager_t &&) = delete;

    static global_cache_manager_t &get_instance() {
        static global_cache_manager_t instance;
        return instance;
    }

    std::unordered_map<impl::engine_kind_t, std::vector<cache_ptr>> &
    get_caches() {
        return caches;
    }
    std::unordered_map<impl::engine_kind_t, size_t> &get_default_capacities() {
        return default_capacities;
    }
    std::unordered_map<impl::engine_kind_t, size_t> &get_user_capacities() {
        return user_capacities;
    }

private:
    global_cache_manager_t() {
        // The value of ONEDNN_GRAPH_CONSTANT_TENSOR_CACHE_CAPACITY is in this
        // form: engine_kind:size;engine_kind:size
        std::string str = impl::getenv_string_user(
                "GRAPH_CONSTANT_TENSOR_CACHE_CAPACITY");
        auto configs = graph::utils::split(str, ';');
        for (const auto &config : configs) {
            auto fields = graph::utils::split(config, ':');

            // The first field: engine kind
            impl::engine_kind_t env_eng_kind = impl::engine_kind::any_engine;
            if (!fields.empty() && !fields[0].empty()) {
                std::string eng_kind = fields[0];
                assertm(eng_kind == "cpu" || eng_kind == "gpu",
                        "engine kind must be cpu or gpu");
                env_eng_kind = eng_kind == "cpu" ? impl::engine_kind::cpu
                                                 : impl::engine_kind::gpu;
            }

            if (env_eng_kind == impl::engine_kind::any_engine) continue;

            // The second field: capacity
            if (fields.size() > 1 && !fields[1].empty()) {
                try {
                    size_t capacity = std::stoll(fields[1]);
                    // the capacity got from env var is in MBytes unit, we need to
                    // convert it to Bytes
                    static constexpr size_t converter = 1024 * 1024;
                    size_t capacity_byte
                            = capacity < std::numeric_limits<size_t>::max()
                                            / converter
                            ? capacity * converter
                            : std::numeric_limits<size_t>::max();
                    user_capacities[env_eng_kind] = capacity_byte;
                } catch (const std::invalid_argument &e) {
                    VERROR(graph, constant_tensor_cache,
                            "'%s': capacity setting is invalid ", e.what());
                } catch (const std::out_of_range &e) {
                    VERROR(graph, constant_tensor_cache,
                            "'%s': capacity setting exceeds numerical "
                            "representation limit ",
                            e.what());
                }
            }
        }

        // create cache for all engine kinds and all devices in the
        // system, to avoid potential data race when modifying caches vector at
        // runtime in multiple threads.
        std::vector<impl::engine_kind_t> eng_kinds {
                impl::engine_kind::cpu, impl::engine_kind::gpu};
        for (auto &kind : eng_kinds) {
            auto ef = get_engine_factory(kind, impl::get_default_runtime(kind));
            if (!ef) continue;
            auto device_count = ef->count();
            default_capacities[kind] = 0;
            size_t capacity_in_bytes = user_capacities.count(kind)
                    ? user_capacities[kind]
                    : default_capacities[kind];

            std::vector<cache_ptr> cache_list(device_count);
            for (size_t id = 0; id < device_count; id++) {
                cache_ptr &cache = cache_list[id];
                // NOLINTNEXTLINE(modernize-make-shared)
                cache.reset(new constant_tensor_cache_t(capacity_in_bytes),
                        [](constant_tensor_cache_t *ptr) {
                            return ptr->release();
                        });
            }
            caches.insert({kind, std::move(cache_list)});
        }
    }

    // The constant tensor cache should be dedicated for each engine kind and
    // each device, so the caches here would like to be in the following form:
    // <eng_kind::cpu, [cache0, cache1, ...]>
    // <eng_kind::gpu, [cache0, cache1, ...]>
    // And the cache instance will be created just before use.
    std::unordered_map<impl::engine_kind_t, std::vector<cache_ptr>> caches;
    std::unordered_map<impl::engine_kind_t, size_t> default_capacities;
    std::unordered_map<impl::engine_kind_t, size_t> user_capacities;
};

constant_tensor_cache_t *get_constant_tensor_cache(
        impl::engine_kind_t eng_kind, size_t index) {
    // get the index-th cache instance from the existing cache list.
    std::vector<cache_ptr> &cache_list
            = global_cache_manager_t::get_instance().get_caches().at(eng_kind);
    if (index >= cache_list.size()) {
        assertm(false, "given device index exceeds the detected device number");
        return nullptr;
    }
    cache_ptr &cache = cache_list[index];
    return cache.get();
}

} // namespace graph
} // namespace impl
} // namespace dnnl

// Constant cache control API
dnnl::impl::graph::status_t dnnl_graph_set_constant_tensor_cache(int flag) {
    if (flag < 0) return dnnl::impl::graph::status::invalid_arguments;
    size_t capacity = flag == 0 ? 0 : std::numeric_limits<size_t>::max();
    dnnl::impl::graph::status_t ret;
    ret = dnnl_graph_set_constant_tensor_cache_capacity(
            dnnl::impl::engine_kind::cpu, capacity);
    if (ret != dnnl::impl::graph::status::success) return ret;
    ret = dnnl_graph_set_constant_tensor_cache_capacity(
            dnnl::impl::engine_kind::gpu, capacity);
    if (ret != dnnl::impl::graph::status::success) return ret;
    return dnnl::impl::graph::status::success;
}

dnnl::impl::graph::status_t dnnl_graph_get_constant_tensor_cache(int *flag) {
    if (flag == nullptr) return dnnl::impl::graph::status::invalid_arguments;
    size_t cpu_flag, gpu_flag;
    dnnl::impl::graph::status_t ret;
    ret = dnnl_graph_get_constant_tensor_cache_capacity(
            dnnl::impl::engine_kind::cpu, &cpu_flag);
    if (ret != dnnl::impl::graph::status::success) return ret;
    ret = dnnl_graph_get_constant_tensor_cache_capacity(
            dnnl::impl::engine_kind::gpu, &gpu_flag);
    if (ret != dnnl::impl::graph::status::success) return ret;
    *flag = static_cast<int>((cpu_flag != 0) || (gpu_flag != 0));
    return dnnl::impl::graph::status::success;
}

dnnl::impl::graph::status_t dnnl_graph_set_constant_tensor_cache_capacity(
        dnnl_engine_kind_t eng_kind, size_t size) {
    // change the user capacity.
    static constexpr size_t converter = 1024 * 1024;
    size_t size_byte = size < std::numeric_limits<size_t>::max() / converter
            ? size * converter
            : std::numeric_limits<size_t>::max();
    dnnl::impl::graph::global_cache_manager_t::get_instance()
            .get_user_capacities()[eng_kind]
            = size_byte; // convert MB to Bytes

    // then reset the capacity to new value for all caches of this engine kind
    if (dnnl::impl::graph::global_cache_manager_t::get_instance()
                    .get_caches()
                    .count(eng_kind)) {
        for (auto &cache :
                dnnl::impl::graph::global_cache_manager_t::get_instance()
                        .get_caches()
                        .at(eng_kind)) {
            if (cache) {
                cache->set_capacity(
                        dnnl::impl::graph::global_cache_manager_t::
                                get_instance()
                                        .get_user_capacities()[eng_kind]);
            }
        }
    }

    return dnnl::impl::graph::status::success;
}

dnnl::impl::graph::status_t dnnl_graph_get_constant_tensor_cache_capacity(
        dnnl_engine_kind_t eng_kind, size_t *size) {
    static constexpr size_t converter = 1024 * 1024;
    if (dnnl::impl::graph::global_cache_manager_t::get_instance()
                    .get_user_capacities()
                    .count(eng_kind)) {
        // convert Bytes to MB
        *size = dnnl::impl::graph::global_cache_manager_t::get_instance()
                        .get_user_capacities()
                        .at(eng_kind)
                / converter;
    } else {
        // convert Bytes to MB
        *size = dnnl::impl::graph::global_cache_manager_t::get_instance()
                        .get_default_capacities()[eng_kind]
                / converter;
    }

    return dnnl::impl::graph::status::success;
}
