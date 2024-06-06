/*******************************************************************************
* Copyright 2019-2024 Intel Corporation
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

#include "gpu/intel/ocl/ref_binary.hpp"
#include "gpu/intel/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

status_t ref_binary_t::pd_t::init_conf(impl::engine_t *engine) {
    const memory_desc_wrapper src0_d(src_md(0));
    const memory_desc_wrapper src1_d(src_md(1));
    const memory_desc_wrapper dst_d(dst_md());

    const int ndims = src0_d.ndims();
    conf.src0_md_info = memory_desc_info_t::create(src0_d);
    conf.src1_md_info = memory_desc_info_t::create(src1_d);
    conf.dst_md_info = memory_desc_info_t::create(dst_d);
    conf.src0_data_type = src0_d.data_type();
    conf.src1_data_type = src1_d.data_type();
    conf.dst_data_type = dst_d.data_type();
    conf.ndims = ndims;
    bool is_src0_bcasted = false;
    for (int i = 0; i < MAX_NDIMS; ++i) {
        conf.src0_bcast_dims[i] = i < ndims
                ? src0_d.dims()[i] == 1 && src0_d.dims()[i] != src1_d.dims()[i]
                : 0;
        is_src0_bcasted = is_src0_bcasted || conf.src0_bcast_dims[i];
        conf.src1_bcast_dims[i] = i < ndims
                ? src1_d.dims()[i] == 1 && src0_d.dims()[i] != src1_d.dims()[i]
                : 0;
    }
    conf.alg = desc()->alg_kind;
    conf.is_tensor_op = is_tensor_op();
    conf.is_dense = dst_d.is_dense();
    conf.same_src_dt = (src0_d.data_type() == src1_d.data_type());
    conf.is_same_md = (src0_d == dst_d) && (src1_d == dst_d);
    conf.attr_info = attr_info_t::create(attr());
    conf.with_binary_post_op
            = attr()->post_ops_.find(primitive_kind::binary) != -1;
    int ic_block_sz = 1;
    conf.use_unroll_16b = false;
    conf.src0_unroll_16b = false;

    auto &blk0 = src0_d.blocking_desc();
    auto &blk1 = src1_d.blocking_desc();
    auto &blkd = dst_d.blocking_desc();
    bool is_16b_blk0 = (blk0.inner_nblks >= 1)
            && (blk0.inner_idxs[blk0.inner_nblks - 1] == 1)
            && (blk0.inner_blks[blk0.inner_nblks - 1] == 16);
    bool is_16b_blk1 = (blk1.inner_nblks >= 1)
            && (blk1.inner_idxs[blk1.inner_nblks - 1] == 1)
            && (blk1.inner_blks[blk1.inner_nblks - 1] == 16);
    bool is_16b_blkd = (blkd.inner_nblks >= 1)
            && (blkd.inner_idxs[blkd.inner_nblks - 1] == 1)
            && (blkd.inner_blks[blkd.inner_nblks - 1] == 16);

    if (is_16b_blkd && !conf.is_tensor_op && !is_src0_bcasted) {
        // If: in case when both are blocked
        // Else: only src0 is blocked
        if (is_16b_blk0 && is_16b_blk1) {
            ic_block_sz = 16;
            conf.use_unroll_16b = true;
        } else if (is_16b_blk0 && blk1.inner_nblks == 0) {
            ic_block_sz = 16;
            conf.src0_unroll_16b = true;
        }
    }

    auto *compute_engine = utils::downcast<compute::compute_engine_t *>(engine);
    conf.dispatch = compute_engine->create_dispatch(dst_d.md_);
    if (conf.is_tensor_op && conf.is_dense && conf.is_same_md
            && !conf.with_binary_post_op) {
        conf.dispatch.define_dim("IDX", 0, dst_d.nelems());
    } else {
        for (int i = 0; i < MAX_NDIMS; ++i) {
            if (i == 1 && (conf.use_unroll_16b || conf.src0_unroll_16b)) {
                // changing value for broadcasting offsets
                // division by IC for enabling blocking within kernel
                conf.dispatch.define_dim(utils::format("D%d", i),
                        nstl::min(i, ndims - 1),
                        i < ndims ? dst_d.padded_dims()[i] : 1, ic_block_sz);
            } else {
                conf.dispatch.define_dim(utils::format("D%d", i),
                        nstl::min(i, ndims - 1),
                        i < ndims ? dst_d.padded_dims()[i] : 1);
            }
        }
    }
    conf.dispatch.generate();
    return status::success;
}

status_t ref_binary_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    def_binary_alg_kinds(kernel_ctx);
    kernel_ctx.define_int("BINARY_ALG", conf.alg);

    kernel_ctx.set_data_type(conf.src0_data_type);
    kernel_ctx.set_data_type(conf.src1_data_type);
    kernel_ctx.set_data_type(conf.dst_data_type);
    kernel_ctx.define_int("NDIMS", conf.ndims);
    kernel_ctx.define_int("IS_PLAIN_LAYOUT", true);
    kernel_ctx.define_int("IS_TENSOR_OP", conf.is_tensor_op);
    kernel_ctx.define_int("IS_DENSE", conf.is_dense);
    kernel_ctx.define_int("IS_SAME_MD", conf.is_same_md);
    kernel_ctx.define_int("WITH_BINARY_POST_OP", conf.with_binary_post_op);
    kernel_ctx.define_int("SAME_SRC_DT", conf.same_src_dt);

    kernel_ctx.define_int("SRC0_BCAST_DIM0", conf.src0_bcast_dims[0]);
    kernel_ctx.define_int("SRC0_BCAST_DIM1", conf.src0_bcast_dims[1]);
    kernel_ctx.define_int("SRC0_BCAST_DIM2", conf.src0_bcast_dims[2]);
    kernel_ctx.define_int("SRC0_BCAST_DIM3", conf.src0_bcast_dims[3]);
    kernel_ctx.define_int("SRC0_BCAST_DIM4", conf.src0_bcast_dims[4]);
    kernel_ctx.define_int("SRC0_BCAST_DIM5", conf.src0_bcast_dims[5]);

    kernel_ctx.define_int("SRC1_BCAST_DIM0", conf.src1_bcast_dims[0]);
    kernel_ctx.define_int("SRC1_BCAST_DIM1", conf.src1_bcast_dims[1]);
    kernel_ctx.define_int("SRC1_BCAST_DIM2", conf.src1_bcast_dims[2]);
    kernel_ctx.define_int("SRC1_BCAST_DIM3", conf.src1_bcast_dims[3]);
    kernel_ctx.define_int("SRC1_BCAST_DIM4", conf.src1_bcast_dims[4]);
    kernel_ctx.define_int("SRC1_BCAST_DIM5", conf.src1_bcast_dims[5]);

    kernel_ctx.define_int("USE_UNROLL_16B", conf.use_unroll_16b);
    kernel_ctx.define_int("SRC0_UNROLL_16B", conf.src0_unroll_16b);
    kernel_ctx.define_int("SUB_GROUP_SIZE", 1);

    def_memory_desc_info(kernel_ctx, conf.src0_md_info, "SRC0");
    def_memory_desc_info(kernel_ctx, conf.src1_md_info, "SRC1");
    def_memory_desc_info(kernel_ctx, conf.dst_md_info, "DST");

    CHECK(def_attr_info(
            kernel_ctx, conf.attr_info, attr()->post_ops_, *dst_md()));

    def_dispatch(kernel_ctx, conf.dispatch);

    return status::success;
}

status_t ref_binary_t::execute_ref(const exec_ctx_t &ctx) const {

    status_t status = status::success;

    auto &src0 = CTX_IN_STORAGE(DNNL_ARG_SRC_0);
    auto &src1 = CTX_IN_STORAGE(DNNL_ARG_SRC_1);
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_DST);
    CHECK(status);

    const auto &conf = pd()->conf;

    auto &src0_scale = CTX_IN_STORAGE(DNNL_ARG_SRC_0 | DNNL_ARG_ATTR_SCALES);

    auto &src1_scale = CTX_IN_STORAGE(DNNL_ARG_SRC_1 | DNNL_ARG_ATTR_SCALES);

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src0);
    arg_list.set(1, src1);
    arg_list.set(2, dst);

    unsigned arg_idx = append_post_ops_to_arg_list(
            ctx, arg_list, 3, pd()->attr()->post_ops_);

    arg_list.set(arg_idx++, src0_scale);
    arg_list.set(arg_idx, src1_scale);

    auto nd_range = conf.dispatch.nd_range();

    status = parallel_for(ctx, nd_range, kernel_, arg_list);
    return status;
}

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
