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

#ifndef JIT_REF_BINARY_COMMON_KERNEL_HPP
#define JIT_REF_BINARY_COMMON_KERNEL_HPP

#include "common/c_types_map.hpp"
#include "common/memory.hpp"
#include "compute/compute.hpp"
#include "ocl/jit_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace ocl {

struct jit_ref_binary_common_kernel {

    jit_ref_binary_common_kernel(const jit_binary_conf_t &ajib) : jib(ajib) {}

    ~jit_ref_binary_common_kernel() {}

    static status_t init_conf(jit_binary_conf_t &jib,
            const memory_desc_wrapper &src0_d,
            const memory_desc_wrapper &src1_d, const memory_desc_wrapper &dst_d,
            alg_kind_t alg, const dims_t &broadcast_dims, bool is_tensor_op) {

        const int ndims = src0_d.ndims();
        jib.src0_md_info = jit_memory_desc_info_t::create(src0_d);
        jib.src1_md_info = jit_memory_desc_info_t::create(src1_d);
        jib.dst_md_info = jit_memory_desc_info_t::create(dst_d);
        jib.data_type = src0_d.data_type();
        jib.ndims = ndims;
        for (int i = 0; i < ndims; ++i) {
            jib.dim0[i] = src0_d.dims()[i];
            jib.bcast_dims[i] = broadcast_dims[i];
        }
        jib.is_add = (alg == alg_kind::binary_add);
        jib.is_mul = (alg == alg_kind::binary_mul);
        jib.is_tensor_op = is_tensor_op;

        jib.gws_d[0] = utils::array_product(src0_d.dims(), ndims);
        jib.gws_d[1] = 1;
        jib.gws_d[2] = 1;

        return status::success;
    }

    static status_t init_const_def(
            compute::kernel_ctx_t &kernel_ctx, const jit_binary_conf_t &jib) {

        kernel_ctx.set_data_type(jib.data_type);
        kernel_ctx.define_int("NDIMS", jib.ndims);
        kernel_ctx.define_int("IS_MUL", jib.is_mul);
        kernel_ctx.define_int("IS_ADD", jib.is_add);
        kernel_ctx.define_int("IS_TENSOR_OP", jib.is_tensor_op);
        kernel_ctx.define_int("DIM0", jib.dim0[0]);
        kernel_ctx.define_int("DIM1", jib.dim0[1]);
        kernel_ctx.define_int("DIM2", jib.dim0[2]);
        kernel_ctx.define_int("DIM3", jib.dim0[3]);
        kernel_ctx.define_int("DIM4", jib.dim0[4]);
        kernel_ctx.define_int("DIM5", jib.dim0[5]);
        kernel_ctx.define_int("BCAST_DIM0", jib.bcast_dims[0]);
        kernel_ctx.define_int("BCAST_DIM1", jib.bcast_dims[1]);
        kernel_ctx.define_int("BCAST_DIM2", jib.bcast_dims[2]);
        kernel_ctx.define_int("BCAST_DIM3", jib.bcast_dims[3]);
        kernel_ctx.define_int("BCAST_DIM4", jib.bcast_dims[4]);
        kernel_ctx.define_int("BCAST_DIM5", jib.bcast_dims[5]);

        def_memory_desc_info(kernel_ctx, jib.src0_md_info, "SRC0");
        def_memory_desc_info(kernel_ctx, jib.src1_md_info, "SRC1");
        def_memory_desc_info(kernel_ctx, jib.dst_md_info, "DST");

        return status::success;
    }

    jit_binary_conf_t jib;
};

} // namespace ocl
} // namespace impl
} // namespace dnnl

#endif // JIT_REF_BINARY_COMMON_KERNEL_HPP
