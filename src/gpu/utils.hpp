/*******************************************************************************
* Copyright 2023 Intel Corporation
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
#ifndef GPU_UTILS_HPP
#define GPU_UTILS_HPP

#include <iostream>
#include <sstream>

#include "common/utils.hpp"
#include "gpu/compute/compute.hpp"

// Uncomment this when aborting on ir_assert is desired:
// #define GPU_ABORT_ON_ERROR

namespace dnnl {
namespace impl {
namespace gpu {
namespace gpu_utils {

class error_stream_t {
public:
    error_stream_t(const char *file, int line, const char *assert_msg)
        : data_(new data_t(file, line, assert_msg)) {}

    // This is to be able use a steam object in short-circuit evaluation with
    // booleans, see below.
    operator bool() const { return true; }

    template <typename T>
    error_stream_t &operator<<(const T &t) {
        data_->out << t;
        return *this;
    }

    ~error_stream_t() noexcept(false) {
        if (data_ == nullptr) return;

        std::cout << data_->out.str() << std::endl;
#ifdef GPU_ABORT_ON_ERROR
        std::abort();
#else
        auto err = std::runtime_error(data_->out.str());
        delete data_;
        data_ = nullptr;

        // This is techincally unsafe. Since error_stream_t is only used in
        // debug builds and since it is only used by ir_assert() which signals
        // an ill-defined program state, nested throws is not a concern.
        throw err; // NOLINT
#endif
    }

private:
    struct data_t {
        data_t(const char *file, int line, const char *assert_msg)
            : file(file), line(line) {
            out << "Assertion " << assert_msg << " failed at " << file << ":"
                << line << std::endl;
        }

        const char *file;
        int line;
        std::ostringstream out;
    };

    data_t *data_;
};

#if !defined(NDEBUG) || defined(DNNL_DEV_MODE)
#define gpu_assert(cond) \
    !(cond) \
            && dnnl::impl::gpu::gpu_utils::error_stream_t( \
                    __FILE__, __LINE__, #cond)
#else
#define gpu_assert(cond) \
    (false) && !(cond) \
            && dnnl::impl::gpu::gpu_utils::error_stream_t( \
                    __FILE__, __LINE__, #cond)
#endif

template <typename out_type, typename in_type,
        typename std::enable_if<!std::is_fundamental<out_type>::value
                || !std::is_fundamental<in_type>::value>::type>
inline bool validate_into(in_type in) {
    return true;
}
template <typename out_type, typename in_type,
        typename std::enable_if<std::is_fundamental<out_type>::value
                && std::is_fundamental<in_type>::value>::type>
inline bool validate_into(in_type in) {
    const double in_compare = static_cast<double>(in);
    const double out_max
            = static_cast<double>(std::numeric_limits<out_type>::max());
    const double out_lowest
            = static_cast<double>(std::numeric_limits<out_type>::lowest());
    return in_compare <= out_max && in_compare >= out_lowest;
}
template <typename out_type>
inline bool validate_into(bool b) {
    return std::is_integral<out_type>::value;
}

template <typename out_type, typename in_type>
inline out_type into(in_type in) {
    gpu_assert(validate_into<out_type>(in))
            << "Value " << in << " cannot be converted into type "
            << typeid(out_type).name();
    return static_cast<out_type>(in);
}

inline int dev_getenv(const char *name, int default_value) {
#ifdef DNNL_DEV_MODE
    return getenv_int(name, default_value);
#else
    return default_value;
#endif
}

inline bool dev_getenv(const char *s, bool def) {
    return dev_getenv(s, def ? 1 : 0) == 1;
}

inline std::string dev_getenv(const char *s, const std::string &def) {
#ifdef DNNL_DEV_MODE
    char buf[1024];
    int ret = getenv(s, buf, sizeof(buf));
    if (ret > 0) return buf;
    return def;
#else
    return def;
#endif
}

// Input is a comma separate list containing gpu_arch and optionally eu_count.
inline compute::gpu_arch_t dev_getenv(const char *s, compute::gpu_arch_t arch,
        int *eu_count = nullptr, int *max_wg_size = nullptr) {
#ifdef DNNL_DEV_MODE
    char buf[1024];
    int ret = getenv(s, buf, sizeof(buf));
    if (ret > 0) {
        char *arch_str = buf, *eu_str = nullptr;
        for (int i = 0; i < ret; i++) {
            if (buf[i] == ',') {
                buf[i] = 0;
                if (i < ret - 1) { eu_str = &buf[i + 1]; }
                break;
            }
        }
        arch = compute::str2gpu_arch(arch_str);
        if (eu_count && eu_str) { *eu_count = atoi(eu_str); }
        if (max_wg_size) {
            // Assume maximum wg size is basically the number of threads
            // available in a subslice with simd_size 16
            const int max_eus_per_wg
                    = compute::device_info_t::max_eus_per_wg(arch);
            const int simd_size = 16;
            const int thr_per_eu = utils::rnd_down_pow2(
                    compute::device_info_t::threads_per_eu(arch));
            *max_wg_size = simd_size * max_eus_per_wg * thr_per_eu;
        }
    }
    return arch;
#else
    return arch;
#endif
}

} // namespace gpu_utils
} // namespace gpu
} // namespace impl
} // namespace dnnl
#endif
