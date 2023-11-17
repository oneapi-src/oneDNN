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
    status_t init(compute::gpu_arch_t arch, data_type_t dt, int unroll_n,
            bool copyb, bool trans, bool sum = false, bool clear_sum = false) {
        *this = {};
        arch_ = arch;
        dt_ = dt;
        unroll_n_ = unroll_n;
        copyb_ = copyb;
        trans_ = trans;
        sum_ = sum;
        clear_sum_ = clear_sum;
        return status::success;
    };

    status_t create_generator(const compute::compute_engine_t &engine,
            compute::kernel_bundle_t &bundle) const {
        compute::kernel_ctx_t ctx;
        CHECK(init_kernel_ctx(ctx));
        return engine.create_kernel_bundle(bundle, {name()}, ctx);
    };

    status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const {
        return init_kernel_ctx(kernel_ctx, arch_, dt_, unroll_n_, copyb_,
                trans_, sum_, clear_sum_);
    }

    static status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx,
            compute::gpu_arch_t arch, data_type_t dt, int unroll_n, bool copyb,
            bool trans, bool sum = false, bool clear_sum = false) {

        using arch_t = compute::gpu_arch_t;

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
        if (utils::one_of(arch, arch_t::xe_hp, arch_t::xe_hpg))
            kernel_ctx.add_option("-DCOPY_XE_HP");
        if (utils::one_of(arch, arch_t::xe_hpc, arch_t::xe2))
            kernel_ctx.add_option("-DCOPY_XE_HPC");

        return status::success;
    }

    const char *name() const { return name(arch_); }
    static const char *name(compute::gpu_arch_t arch) {
        using arch_t = compute::gpu_arch_t;
        switch (arch) {
            case arch_t::xe_hp:
            case arch_t::xe_hpg: return "xe_hp_systolic_gemm_copy";
            case arch_t::xe_hpc:
            case arch_t::xe2: return "xe_hpc_systolic_gemm_copy";
            default: assert(!"Unsupported architecture"); return "";
        }
    }

    static constexpr int unroll_k(size_t element_size) {
        return 32 / int(element_size);
    }

    static constexpr int unroll_r(
            compute::gpu_arch_t arch, size_t element_size, bool copyb) {
        return !copyb ? ((arch >= compute::gpu_arch_t::xe_hpc) ? 64 : 32)
                      : unroll_k(element_size);
    }

    static constexpr int unroll_c(
            size_t element_size, int unroll_n, bool copyb) {
        return !copyb ? unroll_k(element_size) : unroll_n;
    }

    static constexpr int subgroup_size(compute::gpu_arch_t arch) {
        return (arch >= compute::gpu_arch_t::xe_hpc) ? 16 : 8;
    }

    static constexpr int subgroup_size_clear_sum(compute::gpu_arch_t arch) {
        return (arch >= compute::gpu_arch_t::xe_hpc) ? 16 : 8;
    }

#if __cplusplus >= 202002L
    bool operator==(const xe_systolic_gemm_copy_kernel_t &) const = default;
#endif

    serialized_t serialize() const {
        serialized_t s;
        s.append(*this);
        return s;
    }

    static xe_systolic_gemm_copy_kernel_t deserialize(const serialized_t &s) {
        xe_systolic_gemm_copy_kernel_t t {};
        deserializer_t d(s);
        d.pop(t);
        return t;
    }

private:
    compute::gpu_arch_t arch_;
    data_type_t dt_;
    int unroll_n_;
    bool copyb_;
    bool trans_;
    bool sum_;
    bool clear_sum_;
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
