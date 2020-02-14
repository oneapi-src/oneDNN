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

#include "gpu/ocl/ref_binary.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

status_t ref_binary_t::pd_t::init_conf() {
    const memory_desc_wrapper src0_d(src_md(0));
    const memory_desc_wrapper src1_d(src_md(1));
    const memory_desc_wrapper dst_d(dst_md());

    alg_kind_t alg = desc()->alg_kind;

    const auto &po = attr()->post_ops_;
    bool with_sum = po.contain(primitive_kind::sum, 0)
            && po.entry_[0].sum.scale != 0.f;
    float sum_scale = with_sum ? po.entry_[0].sum.scale : 1;
    int e_idx = po.find(primitive_kind::eltwise);
    bool with_eltwise = e_idx != -1 ? true : false;
    float eltwise_scale = with_eltwise ? po.entry_[e_idx].eltwise.scale : 1;

    const int ndims = src0_d.ndims();
    conf.src0_md_info = memory_desc_info_t::create(src0_d);
    conf.src1_md_info = memory_desc_info_t::create(src1_d);
    conf.dst_md_info = memory_desc_info_t::create(dst_d);
    conf.src0_data_type = src0_d.data_type();
    conf.src1_data_type = src1_d.data_type();
    conf.ndims = ndims;
    for (int i = 0; i < MAX_NDIMS; ++i) {
        conf.bcast_dims[i] = i < ndims ? broadcast_dims()[i] : 1;
    }
    conf.is_add = (alg == alg_kind::binary_add);
    conf.is_mul = (alg == alg_kind::binary_mul);
    conf.is_max = (alg == alg_kind::binary_max);
    conf.is_min = (alg == alg_kind::binary_min);
    conf.is_tensor_op = is_tensor_op();
    conf.is_dense = dst_d.is_dense();
    conf.same_src_dt = (src0_d.data_type() == src1_d.data_type());
    conf.is_same_md = (src0_d == dst_d) && (src1_d == dst_d);
    conf.with_src0_scale = with_scales(DNNL_ARG_SRC_0);
    conf.with_src1_scale = with_scales(DNNL_ARG_SRC_1);
    conf.with_eltwise = with_eltwise;
    conf.with_sum = with_sum;
    conf.sum_scale = sum_scale;
    conf.eltwise_scale = eltwise_scale;
    if (with_eltwise) { conf.eltwise = po.entry_[e_idx].eltwise; }
    int ic_block_sz = 1;
    conf.use_unroll_16b = false;
    conf.src0_unroll_16b = false;

    auto &blk0 = src0_d.blocking_desc();
    auto &blk1 = src1_d.blocking_desc();
    bool is_16b_blk0 = (blk0.inner_nblks >= 1)
            && (blk0.inner_idxs[blk0.inner_nblks - 1] == 1)
            && (blk0.inner_blks[blk0.inner_nblks - 1] == 16);
    bool is_16b_blk1 = (blk1.inner_nblks >= 1)
            && (blk1.inner_idxs[blk1.inner_nblks - 1] == 1)
            && (blk1.inner_blks[blk1.inner_nblks - 1] == 16);

    if (!conf.is_tensor_op) {
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

    auto *compute_engine
            = utils::downcast<compute::compute_engine_t *>(engine());
    conf.dispatch = compute_engine->create_dispatch(dst_d.md_);
    if (conf.is_tensor_op && conf.is_dense && conf.is_same_md) {
        conf.dispatch.define_dim("IDX", 0, dst_d.nelems());
    } else {
        // Setting the MB as the innermost dim for optimized performance
        // Hence starting i = 1, ignoring MB
        conf.dispatch.define_dim_with_nesting_level ("D0", ndims, dst_d.dims()[0], 1);
        for (int i = 1; i < MAX_NDIMS; ++i) {
            if ( i == 1 && (conf.use_unroll_16b || conf.src0_unroll_16b) ) {
                // changing value for broadcasting offsets
                // division by IC for enabling blocking within kernel
                conf.dispatch.define_dim(utils::format("D%d", i), nstl::min(i, ndims - 1),
                    i < ndims ? dst_d.padded_dims()[i] : 1, ic_block_sz);
            }
            else {
                conf.dispatch.define_dim(utils::format("D%d", i),
                    nstl::min(i, ndims - 1), i < ndims ? dst_d.dims()[i] : 1);
            }
        }
    }
    conf.dispatch.generate();
    return status::success;
}

status_t ref_binary_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    kernel_ctx.set_data_type(conf.src0_data_type);
    kernel_ctx.define_int("NDIMS", conf.ndims);
    kernel_ctx.define_int("IS_MUL", conf.is_mul);
    kernel_ctx.define_int("IS_ADD", conf.is_add);
    kernel_ctx.define_int("IS_MAX", conf.is_max);
    kernel_ctx.define_int("IS_MIN", conf.is_min);
    kernel_ctx.define_int("IS_TENSOR_OP", conf.is_tensor_op);
    kernel_ctx.define_int("IS_DENSE", conf.is_dense);
    kernel_ctx.define_int("IS_SAME_MD", conf.is_same_md);
    kernel_ctx.define_int("SAME_SRC_DT", conf.same_src_dt);
    kernel_ctx.define_int("BCAST_DIM0", conf.bcast_dims[0]);
    kernel_ctx.define_int("BCAST_DIM1", conf.bcast_dims[1]);
    kernel_ctx.define_int("BCAST_DIM2", conf.bcast_dims[2]);
    kernel_ctx.define_int("BCAST_DIM3", conf.bcast_dims[3]);
    kernel_ctx.define_int("BCAST_DIM4", conf.bcast_dims[4]);
    kernel_ctx.define_int("BCAST_DIM5", conf.bcast_dims[5]);
    kernel_ctx.define_int("WITH_ELTWISE", conf.with_eltwise);
    kernel_ctx.define_int("WITH_SUM", conf.with_sum);
    kernel_ctx.define_int("SUM_SCALE", conf.sum_scale == 1);
    kernel_ctx.define_int("SRC0_S", conf.src0_data_type == data_type::s8);
    kernel_ctx.define_int("SRC1_S", conf.src1_data_type == data_type::s8);
    kernel_ctx.define_int("SRC0_SCALE", conf.with_src0_scale);
    kernel_ctx.define_int("SRC1_SCALE", conf.with_src1_scale);
    kernel_ctx.define_int("USE_UNROLL_16B", conf.use_unroll_16b);
    kernel_ctx.define_int("SRC0_UNROLL_16B", conf.src0_unroll_16b);

    def_memory_desc_info(kernel_ctx, conf.src0_md_info, "SRC0");
    def_memory_desc_info(kernel_ctx, conf.src1_md_info, "SRC1");
    def_memory_desc_info(kernel_ctx, conf.dst_md_info, "DST");

    if (conf.with_eltwise) { def_postops(kernel_ctx, conf.eltwise.alg); }

    def_dispatch(kernel_ctx, conf.dispatch);

    return status::success;
}

status_t ref_binary_t::execute_ref(const exec_ctx_t &ctx) const {
    compute::compute_stream_t *compute_stream
            = utils::downcast<compute::compute_stream_t *>(ctx.stream());

    auto &src0 = CTX_IN_STORAGE(DNNL_ARG_SRC_0);
    auto &src1 = CTX_IN_STORAGE(DNNL_ARG_SRC_1);
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_DST);

    auto eltwise_alpha = pd()->eltwise_alpha();
    auto eltwise_beta = pd()->eltwise_beta();
    auto sum_scale = pd()->sum_scale();
    auto eltwise_scale = pd()->eltwise_scale();

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src0);
    arg_list.set(1, src1);
    arg_list.set(2, dst);
    arg_list.set(3, eltwise_alpha);
    arg_list.set(4, eltwise_beta);
    arg_list.set(5, sum_scale);
    arg_list.set(6, eltwise_scale);
    if (utils::one_of(
                pd()->src_md()->data_type, data_type::u8, data_type::s8)) {
        if (pd()->with_scales(DNNL_ARG_SRC_0)) {
            auto src0_scale = pd()->get_scale(DNNL_ARG_SRC_0);
            arg_list.set(7, src0_scale);
            if (pd()->with_scales(DNNL_ARG_SRC_1)) {
                auto src1_scale = pd()->get_scale(DNNL_ARG_SRC_1);
                arg_list.set(8, src1_scale);
            }
        } else if (pd()->with_scales(DNNL_ARG_SRC_1)) {
            auto src1_scale = pd()->get_scale(DNNL_ARG_SRC_1);
            arg_list.set(7, src1_scale);
        }
    }

    const auto &conf = pd()->conf;

    auto nd_range = conf.dispatch.nd_range();
    status_t status = compute_stream->parallel_for(nd_range, kernel_, arg_list);
    return status;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
