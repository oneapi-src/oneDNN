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

#ifndef LLGA_BACKEND_DNNL_ABSTRACT_TYPES_HPP
#define LLGA_BACKEND_DNNL_ABSTRACT_TYPES_HPP

#include <cstdlib>
#include <cstring>
#include <dnnl.h>
#include <dnnl.hpp>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <unordered_map>

#include "interface/allocator.hpp"
#include "interface/engine.hpp"

#if DNNL_GRAPH_WITH_SYCL
#include "dnnl_sycl.hpp"
#endif

namespace llga {
namespace impl {
namespace dnnl_impl {

using error = dnnl::error;
using memory = dnnl::memory;
using format_tag = memory::format_tag;
using tag = memory::format_tag;
using data_type = typename memory::data_type;
using dims = typename memory::dims;
using dim = memory::dim;
using query = dnnl::query;
using kind = dnnl::primitive::kind;
using prop_kind = dnnl::prop_kind;
using algorithm = dnnl::algorithm;
using normalization_flag = dnnl::normalization_flags;
using query = dnnl::query;
using scale_t = std::vector<float>;
using exec_args = std::unordered_map<int, memory>;

// for computation cache
using key_t = std::string;

#ifndef NDEBUG
#define BACKEND_DNNL_ENFORCE(condition, message) \
    do { \
        error::wrap_c_api((condition) ? dnnl_success : dnnl_invalid_arguments, \
                (message)); \
    } while (false)
#else
#define BACKEND_DNNL_ENFORCE(condition, message)
#endif

const size_t DNNL_CPU_MEMALIGNMENT = 4096;

#if DNNL_GRAPH_WITH_SYCL
const size_t DNNL_SYCL_MEMALIGNMENT = 16;
#endif

enum rnn_kind { RNN_RELU = 0, RNN_TANH = 1, LSTM = 2, GRU = 3 };

struct engine : public dnnl::engine {
    friend class tensor;

    engine() : dnnl::engine() {}

    // create dnnl::engine from llga engine
    explicit engine(const llga::impl::engine_t &eng)
#if DNNL_GRAPH_WITH_SYCL
        : dnnl::engine(dnnl::sycl_interop::make_engine(
                eng.sycl_device(), eng.sycl_context())) {
#else
        : dnnl::engine(static_cast<kind>(eng.kind()),
                static_cast<size_t>(eng.device_id())) {
#endif
        impl::allocator_t *allocator = eng.get_allocator();
        if (eng.kind() == llga::impl::engine_kind::cpu) {
            this->malloc = [this, allocator](size_t size) {
                // Now, we always have a default allocator for CPU.
                // Here, we always request for the persistent memory buffer,
                // so we also need free them manually.
                return allocator->allocate(size,
                        {llga::impl::allocator_lifetime::persistent,
                                DNNL_CPU_MEMALIGNMENT});
            };
            this->free = [this, allocator](
                                 void *p) { return allocator->deallocate(p); };
        } else {
#if DNNL_GRAPH_WITH_SYCL
            // Now, we always have a default allocator for SYCL device.
            // Here, we always request for the persistent memory buffer,
            // so we also need free them manually.
            this->malloc = [this, allocator](size_t size) {
                return allocator->allocate(size,
                        dnnl::sycl_interop::get_device(*this),
                        dnnl::sycl_interop::get_context(*this),
                        {llga::impl::allocator_lifetime::persistent,
                                DNNL_SYCL_MEMALIGNMENT});
            };
            this->free = [this, allocator](void *p) {
                return allocator->deallocate(
                        p, dnnl::sycl_interop::get_context(*this));
            };
#endif
        }
    }

    bool match(const llga::impl::engine_t &eng) {
        bool ok = true && (get_kind() == static_cast<kind>(eng.kind()));
#if DNNL_GRAPH_WITH_SYCL
        ok = ok && (eng.sycl_device() == dnnl::sycl_interop::get_device(*this))
                && (eng.sycl_context()
                        == dnnl::sycl_interop::get_context(*this));
#endif
        return ok;
    }

private:
    std::function<void *(size_t)> malloc;
    std::function<void(void *)> free;
};

class engine_manager {
public:
    static engine_manager *get() {
        static engine_manager inst;
        return &inst;
    }

    const engine *get_engine(const llga::impl::engine_t &eng) {
        std::lock_guard<std::mutex> guard(engines_.lock);

        auto it = std::find_if(engines_.data.begin(), engines_.data.end(),
                [&eng](const std::shared_ptr<engine> &v) {
                    return v->match(eng);
                });
        if (it == engines_.data.end()) {
            auto e = std::make_shared<engine>(eng);
            engines_.data.push_back(e);
            return e.get();
        } else {
            return it->get();
        }
    }

    // disable copy and assign
    engine_manager(const engine_manager &) = delete;
    engine_manager &operator=(const engine_manager &) = delete;

private:
    engine_manager() {}
    ~engine_manager() {}

    struct {
        std::vector<std::shared_ptr<engine>> data {};
        std::mutex lock;
    } engines_;
};

/// A default stream
struct stream : public dnnl::stream {
    explicit stream(const engine &aengine) : dnnl::stream(aengine) {}
};

} // namespace dnnl_impl
} // namespace impl
} // namespace llga

#endif
