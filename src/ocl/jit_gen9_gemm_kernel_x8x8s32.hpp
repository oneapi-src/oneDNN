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

#ifndef JIT_GEN9_GEMM_KERNEL_X8X8S32_HPP
#define JIT_GEN9_GEMM_KERNEL_X8X8S32_HPP

#include "common/c_types_map.hpp"
#include "compute/compute.hpp"
#include "ocl/jit_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace ocl {

struct jit_gen9_int8_gemm_kernel {
    template <impl::data_type_t a_type, impl::data_type_t b_type,
            impl::data_type_t c_type>
    static status_t init_cl_options(compute::kernel_ctx_t &kernel_ctx) {
        using namespace data_type;

        if (c_type != s32) return status::unimplemented;

        kernel_ctx.define_int("DT_S32", c_type == s32);

        kernel_ctx.add_option("-cl-mad-enable");
        kernel_ctx.add_option("-cl-strict-aliasing");
#ifdef CL_VERSION_2_0
        kernel_ctx.add_option("-cl-std=CL2.0");
#else
        kernel_ctx.add_option("-Dget_enqueued_local_size=get_local_size");
#endif
        return status::success;
    }

    struct copy_params {
        static constexpr auto unroll_m = 32;
        static constexpr auto unroll_n = 16;
        static constexpr auto unroll_k = 4;
    };
};

template <impl::data_type_t a_type, impl::data_type_t b_type,
        impl::data_type_t c_type>
struct jit_gen9_gemm_x8x8s32_kernel : public jit_gen9_int8_gemm_kernel {
    static status_t init_const_def(compute::kernel_ctx_t &kernel_ctx,
            bool trans_a, bool trans_b, bool fixed_c, bool column_c, bool row_c,
            bool with_eltwise, alg_kind_t alg, int m, int n, int k) {

        MAYBE_UNUSED(m);
        MAYBE_UNUSED(n);
        MAYBE_UNUSED(k);

        auto status = init_cl_options<a_type, b_type, c_type>(kernel_ctx);
        if (status) return status;

        using ao_type = typename prec_traits<a_type>::type;
        using bo_type = typename prec_traits<b_type>::type;

        if ((std::is_same<ao_type, int8_t>::value)
                && (std::is_same<bo_type, int8_t>::value)) {
            kernel_ctx.add_option("-DS8S8");
        } else if ((std::is_same<ao_type, uint8_t>::value)
                && (std::is_same<bo_type, int8_t>::value)) {
            kernel_ctx.add_option("-DU8S8");
        } else if ((std::is_same<ao_type, int8_t>::value)
                && (std::is_same<bo_type, uint8_t>::value)) {
            kernel_ctx.add_option("-DS8U8");
        } else {
            kernel_ctx.add_option("-DU8U8");
        }

        if (!trans_a && !trans_b)
            kernel_ctx.add_option("-DNN");
        else if (trans_a && !trans_b)
            kernel_ctx.add_option("-DTN");
        else if (!trans_a && trans_b)
            kernel_ctx.add_option("-DNT");
        else
            kernel_ctx.add_option("-DTT");

        if (fixed_c)
            kernel_ctx.add_option("-DFF");
        else if (column_c)
            kernel_ctx.add_option("-DCC");
        else if (row_c)
            kernel_ctx.add_option("-DRR");
        else
            return status::unimplemented;

        int block_m, block_n, block_k;
        get_blocks(block_m, block_n, block_k, true);

        int kk = ((block_k + 3) & ~3);
        kernel_ctx.define_int("A_LOCAL_SIZE", (kk + sizeof(int)) * block_m);
        kernel_ctx.define_int("B_LOCAL_SIZE", (kk + sizeof(int)) * block_n);

        kernel_ctx.define_int("UNROLL_M", copy_params::unroll_m);
        kernel_ctx.define_int("UNROLL_N", copy_params::unroll_n);
        kernel_ctx.define_int("UNROLL_K", copy_params::unroll_k);

        kernel_ctx.define_int("WITH_ELTWISE", with_eltwise);
        if (with_eltwise) def_postops(kernel_ctx, alg);

        kernel_ctx.print_options();
        return status::success;
    }

    static void get_unrolls(int &unroll_m, int &unroll_n) {
        unroll_m = copy_params::unroll_m;
        unroll_n = copy_params::unroll_n;
    }

    static void get_blocks(
            int &block_m, int &block_n, int &block_k, bool nocopy) {
        block_m = 6 * 32;
        block_n = 4 * 16;
        block_k = ((32768 / ((block_m >= block_n) ? block_m : block_n)
                           - sizeof(int))
                & ~3);
    }
};

template <impl::data_type_t a_type, impl::data_type_t b_type,
        impl::data_type_t c_type>
struct jit_gen9_gemm_scale_x8x8s32_kernel : public jit_gen9_int8_gemm_kernel {
    static status_t init_const_def(compute::kernel_ctx_t &kernel_ctx,
            bool with_eltwise, alg_kind_t alg) {

        auto status = init_cl_options<a_type, b_type, c_type>(kernel_ctx);
        if (status) return status;

        kernel_ctx.define_int("UNROLL_M", copy_params::unroll_m);
        kernel_ctx.define_int("UNROLL_N", copy_params::unroll_n);
        kernel_ctx.define_int("UNROLL_K", copy_params::unroll_k);

        kernel_ctx.define_int("WITH_ELTWISE", with_eltwise);
        if (with_eltwise) def_postops(kernel_ctx, alg);

        kernel_ctx.print_options();
        return status::success;
    }

    static void get_unrolls(int &unroll_m, int &unroll_n) {
        unroll_m = copy_params::unroll_m;
        unroll_n = copy_params::unroll_n;
    }
};

} // namespace ocl
} // namespace impl
} // namespace dnnl

#endif
