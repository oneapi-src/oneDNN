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

#ifndef JIT_REF_ELTWISE_COMMON_KERNEL_HPP
#define JIT_REF_ELTWISE_COMMON_KERNEL_HPP

#include "common/c_types_map.hpp"
#include "common/memory.hpp"
#include "compute/compute.hpp"
#include "ocl/jit_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace ocl {

struct jit_ref_eltwise_common_kernel {

    jit_ref_eltwise_common_kernel(const jit_eltwise_conf_t &ajel) : jel(ajel) {}

    ~jit_ref_eltwise_common_kernel() {}

    static status_t init_conf(jit_eltwise_conf_t &jel,
            const memory_desc_wrapper &data_d,
            const memory_desc_wrapper &diff_data_d, jit_offsets &jit_off,
            alg_kind_t alg, bool is_forward) {

        const int ndims = data_d.ndims();
        jel.ndims = ndims;

        jel.data_type = data_d.data_type();
        jel.alg = alg;
        jel.is_forward = is_forward;

        set_offsets(data_d, jit_off.src_off);
        set_offsets(diff_data_d, jit_off.dst_off);

        const auto &dims = data_d.dims();
        jel.gws_d[0] = utils::array_product(&dims[0], ndims);
        jel.gws_d[1] = 1;
        jel.gws_d[2] = 1;

        return status::success;
    }

    static status_t init_const_def(compute::kernel_ctx_t &kernel_ctx,
            const jit_eltwise_conf_t &jel, const jit_offsets &jit_off) {

        kernel_ctx.set_data_type(jel.data_type);
        kernel_ctx.define_int("RELU", alg_kind::eltwise_relu);
        kernel_ctx.define_int("LINEAR", alg_kind::eltwise_linear);
        kernel_ctx.define_int("BOUNDED_RELU", alg_kind::eltwise_bounded_relu);
        kernel_ctx.define_int("SOFT_RELU", alg_kind::eltwise_soft_relu);
        kernel_ctx.define_int("LOGISTIC", alg_kind::eltwise_logistic);
        kernel_ctx.define_int("TANH", alg_kind::eltwise_tanh);
        kernel_ctx.define_int("ELU", alg_kind::eltwise_elu);
        kernel_ctx.define_int("SQUARE", alg_kind::eltwise_square);
        kernel_ctx.define_int("SQRT", alg_kind::eltwise_sqrt);
        kernel_ctx.define_int("ABS", alg_kind::eltwise_abs);
        kernel_ctx.define_int("EXP", alg_kind::eltwise_exp);
        kernel_ctx.define_int("GELU", alg_kind::eltwise_gelu);
        kernel_ctx.define_int("SWISH", alg_kind::eltwise_swish);
        kernel_ctx.define_int("LOG", alg_kind::eltwise_log);
        kernel_ctx.define_int("CLIP", alg_kind::eltwise_clip);
        kernel_ctx.define_int("ALG_KIND", jel.alg);
        kernel_ctx.define_int("NDIMS", jel.ndims);

        def_offsets(jit_off.src_off, kernel_ctx, "DATA", jel.ndims);
        def_offsets(jit_off.dst_off, kernel_ctx, "DIFF_DATA",
                jel.is_forward ? 0 : jel.ndims);

        return status::success;
    }

    jit_eltwise_conf_t jel;
};

} // namespace ocl
} // namespace impl
} // namespace dnnl

#endif // JIT_REF_ELTWISE_COMMON_KERNEL_HPP
