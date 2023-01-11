/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
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

#ifndef GPU_OCL_GEMM_XE_SYSTOLIC_GEMM_COPY_KERNEL_HPP
#define GPU_OCL_GEMM_XE_SYSTOLIC_GEMM_COPY_KERNEL_HPP

#include "common/c_types_map.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

struct xe_systolic_gemm_copy_kernel_t {
    static status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx,
            compute::gpu_arch_t arch, data_type_t dt, int unroll_n, bool copyb,
            bool trans, bool sum = false, bool clear_sum = false) {

        auto dt_size = types::data_type_size(dt);

        if (dt_size == 1) kernel_ctx.add_option("-Dcl_intel_subgroups_char");
        kernel_ctx.define_int("ELEMENT_SIZE", int(dt_size));
        kernel_ctx.define_int("UNROLL_N", unroll_n);
        kernel_ctx.define_int("COPY_A", int(!copyb));
        kernel_ctx.define_int("COPY_B", int(copyb));
        kernel_ctx.define_int("COPY_TRANS", int(trans));
        kernel_ctx.define_int("COPY_SUM", int(sum));
        kernel_ctx.define_int("COPY_CLEAR_SUM", int(clear_sum));
        kernel_ctx.define_int("COPY_SIGNED", int(dt == data_type::s8));
        kernel_ctx.add_option("-cl-strict-aliasing");
        kernel_ctx.add_option(
                "-cl-intel-256-GRF-per-thread"); // avoid GRF mode switch
        if (arch == compute::gpu_arch_t::xe_hp)
            kernel_ctx.add_option("-DCOPY_XE_HP");
        if (arch == compute::gpu_arch_t::xe_hpg)
            kernel_ctx.add_option("-DCOPY_XE_HP");
        if (arch == compute::gpu_arch_t::xe_hpc)
            kernel_ctx.add_option("-DCOPY_XE_HPC");

        return status::success;
    }

    static const char *name(compute::gpu_arch_t arch) {
        switch (arch) {
            case compute::gpu_arch_t::xe_hp: return "xe_hp_systolic_gemm_copy";
            case compute::gpu_arch_t::xe_hpg: return "xe_hp_systolic_gemm_copy";
            case compute::gpu_arch_t::xe_hpc:
                return "xe_hpc_systolic_gemm_copy";
            default: assert(!"Unsupported architecture"); return "";
        }
    }

    static constexpr int unroll_k(
            compute::gpu_arch_t arch, size_t element_size) {
        return 32 / int(element_size);
    }

    static constexpr int unroll_r(compute::gpu_arch_t arch, size_t element_size,
            int unroll_n, bool copyb, bool trans) {
        return !copyb ? ((arch == compute::gpu_arch_t::xe_hpc) ? 64 : 32)
                      : unroll_k(arch, element_size);
    }

    static constexpr int unroll_c(compute::gpu_arch_t arch, size_t element_size,
            int unroll_n, bool copyb, bool trans) {
        return !copyb ? unroll_k(arch, element_size) : unroll_n;
    }

    static constexpr int subgroup_size(compute::gpu_arch_t arch,
            size_t element_size, bool copyb, bool trans) {
        return (arch == compute::gpu_arch_t::xe_hpc) ? 16 : 8;
    }

    static constexpr int subgroup_size_clear_sum(
            compute::gpu_arch_t arch, size_t element_size, bool copyb) {
        return (arch == compute::gpu_arch_t::xe_hpc) ? 16 : 8;
    }
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
