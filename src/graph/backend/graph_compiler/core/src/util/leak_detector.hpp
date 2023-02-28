/*******************************************************************************
 * Copyright 2020-2023 Intel Corporation
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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_UTIL_LEAK_DETECTOR_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_UTIL_LEAK_DETECTOR_HPP

#include <atomic>
#include <functional>
#include <stdio.h>
#include <string>
#include <typeinfo>
#include <util/def.hpp>
#if SC_MEMORY_LEAK_CHECK == 2
#include <mutex>
#include <sstream>
#include <unordered_set>
#endif

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace utils {
struct instance_counter {
    const char *base_name;
    std::atomic<uint64_t> newcounter {0};
    std::atomic<uint64_t> delcounter {0};
#if SC_MEMORY_LEAK_CHECK == 2
    std::unordered_set<void *> pool;
    std::mutex lock;
#endif
    std::function<void(void *, FILE *)> obj_dumper;
    void on_new(void *ptr) {
        ++newcounter;
#if SC_MEMORY_LEAK_CHECK == 2
        std::lock_guard<std::mutex> guard(lock);
        pool.insert(ptr);
#endif
    }
    void on_delete(void *ptr) {
        ++delcounter;
#if SC_MEMORY_LEAK_CHECK == 2
        std::lock_guard<std::mutex> guard(lock);
        assert(pool.count(ptr));
        pool.erase(ptr);
#endif
    }
    instance_counter(const char *base_name,
            std::function<void(void *, FILE *)> obj_dumper = nullptr)
        : base_name(base_name), obj_dumper(obj_dumper) {}
    ~instance_counter() {
        auto fn = std::string(base_name) + ".counter.txt";
        FILE *f = fopen(fn.c_str(), "w");
        const char *result
                = (newcounter.load() - delcounter.load() == 0) ? "OK" : "LEAK";
        fprintf(f, "%s %s %lu - %lu = %lu\n", base_name, result,
                newcounter.load(), delcounter.load(),
                newcounter.load() - delcounter.load());

#if SC_MEMORY_LEAK_CHECK == 2
        if (pool.size()) {
            printf("Possible leak detected: Alive %s %lu\n", base_name,
                    pool.size());

            if (obj_dumper) {
                for (auto p : pool)
                    obj_dumper(p, f);
            }
        }
#endif
        fclose(f);
    }
};

namespace impl {

template <typename C>
struct has_to_string {
private:
    template <typename T>
    static constexpr auto check(T *) ->
            typename std::is_same<decltype(&T::to_string),
                    decltype(&T::to_string)>::type;
    template <typename>
    static constexpr std::false_type check(...);

    typedef decltype(check<C>(0)) type;

public:
    static constexpr bool value = type::value;
};

template <typename T, bool b_has_to_string>
struct obj_dumper_impl {
    constexpr static std::nullptr_t funct = nullptr;
};

#if SC_MEMORY_LEAK_CHECK == 2
template <typename T>
struct obj_dumper_impl<T, true> {
    static void funct(void *p, FILE *fp) {
        T *obj = (T *)p;
        std::stringstream ss;
        // obj->to_string(ss);
        ss << obj << '\n';
        fputs(ss.str().c_str(), fp);
    }
};
#endif

} // namespace impl

template <typename T>
struct SC_INTERNAL_API leak_detect_base {
    leak_detect_base() { cnter().on_new(static_cast<T *>(this)); }
    ~leak_detect_base() { cnter().on_delete(static_cast<T *>(this)); }
    static instance_counter &cnter() {
        static instance_counter v {typeid(T).name(),
                impl::obj_dumper_impl<T, impl::has_to_string<T>::value>::funct};
        return v;
    }
};

} // namespace utils
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
