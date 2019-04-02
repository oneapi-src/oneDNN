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

#ifndef JIT_GEN9_GEMM_KERNEL_HPP
#define JIT_GEN9_GEMM_KERNEL_HPP

#include "common/c_types_map.hpp"
#include "ocl/jit_primitive_conf.hpp"

namespace mkldnn {
namespace impl {
namespace ocl {

template <impl::data_type_t type>
struct jit_gen9_gemm_kernel_params {};

template <>
struct jit_gen9_gemm_kernel_params<data_type::f32> {
    static constexpr auto unroll_m = 16;
    static constexpr auto unroll_n = 32;
};

template <>
struct jit_gen9_gemm_kernel_params<data_type::f16> {
    static constexpr auto unroll_m = 16;
    static constexpr auto unroll_n = 32;
};

struct jit_gen9_gemm_kernel {
    template <impl::data_type_t type>
    static status_t init_cl_options(ocl_jit_t &jit) {
        using namespace data_type;

        if (type != f32 && type != f16)
            return status::unimplemented;

        jit.define_int("DT_F32", type == f32);
        jit.define_int("DT_F16", type == f16);

        jit.add_option("-cl-mad-enable");
        jit.add_option("-cl-strict-aliasing");
        jit.add_option("-cl-std=CL2.0");

        return status::success;
    }
};

template <impl::data_type_t type>
struct jit_gen9_gemm_beta_kernel : public jit_gen9_gemm_kernel {
    jit_gen9_gemm_beta_kernel() {}
    ~jit_gen9_gemm_beta_kernel() {}

    static status_t init_const_def(ocl_jit_t &jit) {
        auto status = init_cl_options<type>(jit);
        if (status)
            return status;

#ifdef DEBUG_PRINT
        printf("OPT:\n%s\n", jit.get_options());
#endif
        return status::success;
    }
};

template <impl::data_type_t type>
struct jit_gen9_gemm_copy_kernel : public jit_gen9_gemm_kernel {
    jit_gen9_gemm_copy_kernel() {}
    ~jit_gen9_gemm_copy_kernel() {}

    static status_t init_const_def(ocl_jit_t &jit, bool outer, bool trans) {
        auto status = init_cl_options<type>(jit);
        if (status)
            return status;

        jit.define_int("COPY_UNROLL",
                !outer ? jit_gen9_gemm_kernel_params<type>::unroll_m
                       : jit_gen9_gemm_kernel_params<type>::unroll_n);

        jit.add_option(trans ? "-DUSE_TRANS" : "-DUSE_NOTRANS");

#ifdef DEBUG_PRINT
        printf("OPT:\n%s\n", jit.get_options());
#endif
        return status::success;
    }
};

template <impl::data_type_t type>
struct jit_gen9_gemm_compute_kernel : public jit_gen9_gemm_kernel {
    jit_gen9_gemm_compute_kernel() {}
    ~jit_gen9_gemm_compute_kernel() {}

    static status_t init_const_def(ocl_jit_t &jit, bool beta0) {
        auto status = init_cl_options<type>(jit);
        if (status)
            return status;

        if (beta0)
            jit.add_option("-DBETA_ZERO");

        jit.define_int(
                "UNROLL_M", jit_gen9_gemm_kernel_params<type>::unroll_m);
        jit.define_int(
                "UNROLL_N", jit_gen9_gemm_kernel_params<type>::unroll_n);

#ifdef DEBUG_PRINT
        printf("OPT:\n%s\n", jit.get_options());
#endif
        return status::success;
    }
};

} // namespace ocl
} // namespace impl
} // namespace mkldnn

#endif
