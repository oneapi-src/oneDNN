/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#ifndef GPU_COMPUTE_DEVICE_INFO_HPP
#define GPU_COMPUTE_DEVICE_INFO_HPP

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "common/c_types_map.hpp"
#include "common/utils.hpp"
#include "common/z_magic.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace compute {

enum class gpu_arch_t {
    unknown,
    gen9,
    gen12lp,
};

enum class device_ext_t : int64_t {
    intel_subgroups = 1 << 0,
    intel_subgroups_short = 1 << 1,
    khr_fp16 = 1 << 2,
    khr_int64_base_atomics = 1 << 3,
    intel_subgroup_local_block_io = 1 << 5,
    last
};

inline gpu_arch_t str2gpu_arch(const char *str) {
#define CASE(_case) \
    if (!strcmp(STRINGIFY(_case), str)) return gpu_arch_t::_case

    CASE(gen9);
    CASE(gen12lp);
    return gpu_arch_t::unknown;
#undef CASE
}

inline const char *gpu_arch2str(gpu_arch_t arch) {
#define CASE(_case) \
    case gpu_arch_t::_case: return STRINGIFY(_case)

    switch (arch) {
        CASE(gen9);
        CASE(gen12lp);
        CASE(unknown);
    }
    return "unknown";
#undef CASE
}

static inline const char *ext2cl_str(compute::device_ext_t ext) {
#define CASE(x) \
    case compute::device_ext_t::x: return STRINGIFY(CONCAT2(cl_, x));
    switch (ext) {
        CASE(intel_subgroups);
        CASE(intel_subgroups_short);
        CASE(intel_subgroup_local_block_io);
        CASE(khr_fp16);
        CASE(khr_int64_base_atomics);
        default: return nullptr;
    }
#undef CASE
}

struct runtime_version_t {
    int major;
    int minor;
    int build;

    runtime_version_t(int major = 0, int minor = 0, int build = 0)
        : major {major}, minor {minor}, build {build} {}

    bool operator==(const runtime_version_t &other) const {
        return (major == other.major) && (minor == other.minor)
                && (build == other.build);
    }

    bool operator!=(const runtime_version_t &other) const {
        return !(*this == other);
    }

    bool operator<(const runtime_version_t &other) const {
        if (major < other.major) return true;
        if (major > other.major) return false;
        if (minor < other.minor) return true;
        if (minor > other.minor) return false;
        return (build < other.build);
    }

    bool operator>(const runtime_version_t &other) const {
        return (other < *this);
    }

    bool operator<=(const runtime_version_t &other) const {
        return !(*this > other);
    }

    bool operator>=(const runtime_version_t &other) const {
        return !(*this < other);
    }

    status_t set_from_string(const char *s) {
        int i_major = 0, i = 0;

        for (; s[i] != '.'; i++)
            if (!s[i]) return status::invalid_arguments;

        auto i_minor = ++i;

        for (; s[i] != '.'; i++)
            if (!s[i]) return status::invalid_arguments;

        auto i_build = ++i;

        major = atoi(&s[i_major]);
        minor = atoi(&s[i_minor]);
        build = atoi(&s[i_build]);

        return status::success;
    }

    std::string str() const {
        return utils::format("%d.%d.%d", major, minor, build);
    }
};

struct device_info_t {
public:
    virtual ~device_info_t() = default;

    virtual status_t init() = 0;
    virtual bool has(device_ext_t ext) const = 0;

    virtual gpu_arch_t gpu_arch() const = 0;
    virtual int eu_count() const = 0;
    virtual int hw_threads() const = 0;
    virtual size_t llc_cache_size() const = 0;

    const runtime_version_t &runtime_version() const {
        return runtime_version_;
    }
    const std::string &name() const { return name_; }

protected:
    void set_runtime_version(const runtime_version_t &runtime_version) {
        runtime_version_ = runtime_version;
    }

    void set_name(const std::string &name) { name_ = name; }

private:
    runtime_version_t runtime_version_;
    std::string name_;
};

} // namespace compute
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
