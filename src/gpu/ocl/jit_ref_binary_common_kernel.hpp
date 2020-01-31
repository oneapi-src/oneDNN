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

#ifndef GPU_OCL_JIT_REF_BINARY_COMMON_KERNEL_HPP
#define GPU_OCL_JIT_REF_BINARY_COMMON_KERNEL_HPP

#include "common/binary_pd.hpp"
#include "common/c_types_map.hpp"
#include "common/memory.hpp"
#include "compute/compute.hpp"
#include "gpu/ocl/jit_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

struct jit_ref_binary_common_kernel {

    jit_ref_binary_common_kernel(const jit_binary_conf_t &ajib) : jib(ajib) {}

    ~jit_ref_binary_common_kernel() {}

    static status_t init_conf(jit_binary_conf_t &jib, const binary_pd_t *pd) {

        const memory_desc_wrapper src0_d(pd->src_md(0));
        const memory_desc_wrapper src1_d(pd->src_md(1));
        const memory_desc_wrapper dst_d(pd->dst_md());

        alg_kind_t alg = pd->desc()->alg_kind;
        const dims_t &broadcast_dims = pd->broadcast_dims();
        bool is_tensor_op = pd->is_tensor_op();

        const int ndims = src0_d.ndims();
        jib.src0_md_info = jit_memory_desc_info_t::create(src0_d);
        jib.src1_md_info = jit_memory_desc_info_t::create(src1_d);
        jib.dst_md_info = jit_memory_desc_info_t::create(dst_d);
        jib.data_type = src0_d.data_type();
        jib.ndims = ndims;
        for (int i = 0; i < MAX_NDIMS; ++i) {
            jib.bcast_dims[i] = i < ndims ? broadcast_dims[i] : 1;
        }
        jib.is_add = (alg == alg_kind::binary_add);
        jib.is_mul = (alg == alg_kind::binary_mul);
        jib.is_max = (alg == alg_kind::binary_max);
        jib.is_min = (alg == alg_kind::binary_min);
        jib.is_tensor_op = is_tensor_op;
        jib.is_dense = dst_d.is_dense();
        jib.is_same_md = (src0_d == dst_d) && (src1_d == dst_d);

        auto *compute_engine
                = utils::downcast<compute::compute_engine_t *>(pd->engine());
        jib.dispatch = compute_engine->create_dispatch(dst_d.md_);
        if (jib.is_tensor_op && jib.is_dense && jib.is_same_md) {
            jib.dispatch.define_dim("IDX", 0, dst_d.nelems());
        } else {
            for (int i = 0; i < MAX_NDIMS; ++i) {
                jib.dispatch.define_dim(utils::format("D%d", i),
                        nstl::min(i, ndims - 1),
                        i < ndims ? dst_d.dims()[i] : 1);
            }
        }

        jib.dispatch.generate();

        return status::success;
    }

    static status_t init_const_def(
            compute::kernel_ctx_t &kernel_ctx, const jit_binary_conf_t &jib) {

        kernel_ctx.set_data_type(jib.data_type);
        kernel_ctx.define_int("NDIMS", jib.ndims);
        kernel_ctx.define_int("IS_MUL", jib.is_mul);
        kernel_ctx.define_int("IS_ADD", jib.is_add);
        kernel_ctx.define_int("IS_MAX", jib.is_max);
        kernel_ctx.define_int("IS_MIN", jib.is_min);
        kernel_ctx.define_int("IS_TENSOR_OP", jib.is_tensor_op);
        kernel_ctx.define_int("IS_DENSE", jib.is_dense);
        kernel_ctx.define_int("IS_SAME_MD", jib.is_same_md);
        kernel_ctx.define_int("BCAST_DIM0", jib.bcast_dims[0]);
        kernel_ctx.define_int("BCAST_DIM1", jib.bcast_dims[1]);
        kernel_ctx.define_int("BCAST_DIM2", jib.bcast_dims[2]);
        kernel_ctx.define_int("BCAST_DIM3", jib.bcast_dims[3]);
        kernel_ctx.define_int("BCAST_DIM4", jib.bcast_dims[4]);
        kernel_ctx.define_int("BCAST_DIM5", jib.bcast_dims[5]);

        def_memory_desc_info(kernel_ctx, jib.src0_md_info, "SRC0");
        def_memory_desc_info(kernel_ctx, jib.src1_md_info, "SRC1");
        def_memory_desc_info(kernel_ctx, jib.dst_md_info, "DST");

        def_dispatch(kernel_ctx, jib.dispatch);

        return status::success;
    }

    jit_binary_conf_t jib;
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_OCL_JIT_REF_BINARY_COMMON_KERNEL_HPP
