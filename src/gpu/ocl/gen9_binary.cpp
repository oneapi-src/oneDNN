/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
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

#include "gpu/ocl/gen9_binary.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

// For cases with src0 and dst blocked, and s1 in plain layout
// src1 channels are divisible by 16 and has broadcast on all other dims
// except n & c
bool check_layout_constraints(const memory_desc_t *md) {
    if (md->dims[1] % 16 != 0) { return false; }
    for (int i = 2; i < md->ndims; i++) {
        if (md->dims[i] != 1) { return false; }
    }
    return true;
}

// This is part-one of checking all the conditions to allow plain and
// single blocked layouts in src tensors & single blocked dst tensors
// src0_* and src1_* is nomenclature to distinguish between input tensors
bool check_mixed_layout(const memory_desc_wrapper &src0_d,
        const memory_desc_wrapper &dst_d, const memory_desc_t *src1_md) {
    using namespace dnnl::impl::format_tag;
    format_tag_t src0_tag
            = src0_d.matches_one_of_tag(aBc16b, aBcd16b, aBcde16b);
    const memory_desc_wrapper src1_d(src1_md);
    bool is_mixed = dst_d.matches_tag(src0_tag)
            && src1_d.matches_one_of_tag(abc, abcd, abcde) && src0_tag;
    return (is_mixed && src1_md->dims[1] % 16 == 0) ? true : false;
}

// This is part-two of checking mixed layouts, as it works with
// non-broadcast cases, mostly. Except for shapes like 1x16x16x16:1x16x16x16
// i.e. where mb = 1 (that is counted as a broadcast case in binary prim)
bool check_broadcast(
        const memory_desc_t *src0_md, const memory_desc_t *src1_md) {
    for (int i = 0; i < src0_md->ndims; i++) {
        if (src0_md->dims[i] != src1_md->dims[i]) { return false; }
    }
    return true;
}

status_t gen9_binary_t::pd_t::init_conf(engine_t *engine) {
    using namespace dnnl::impl::format_tag;
    const memory_desc_wrapper src0_d(src_md(0));
    const memory_desc_wrapper src1_d(src_md(1));
    const memory_desc_wrapper dst_d(dst_md());

    alg_kind_t alg = desc()->alg_kind;

    const int ndims = src0_d.ndims();
    conf.src0_md_info = memory_desc_info_t::create(src0_d);
    conf.src1_md_info = memory_desc_info_t::create(src1_d);
    conf.dst_md_info = memory_desc_info_t::create(dst_d);
    conf.attr_info = attr_info_t::create(attr());
    conf.src0_data_type = src0_d.data_type();
    conf.src1_data_type = src1_d.data_type();
    conf.dst_data_type = dst_d.data_type();
    conf.ndims = ndims;
    conf.is_add = (alg == alg_kind::binary_add);
    conf.is_mul = (alg == alg_kind::binary_mul);
    conf.is_max = (alg == alg_kind::binary_max);
    conf.is_min = (alg == alg_kind::binary_min);
    conf.is_div = (alg == alg_kind::binary_div);
    conf.is_sub = (alg == alg_kind::binary_sub);
    conf.is_ge = (alg == alg_kind::binary_ge);
    conf.is_gt = (alg == alg_kind::binary_gt);
    conf.is_le = (alg == alg_kind::binary_le);
    conf.is_lt = (alg == alg_kind::binary_lt);
    conf.is_eq = (alg == alg_kind::binary_eq);
    conf.is_ne = (alg == alg_kind::binary_ne);
    conf.is_tensor_op = is_tensor_op();
    conf.is_dense = dst_d.is_dense();
    conf.same_src_dt = (src0_d.data_type() == src1_d.data_type());
    conf.is_same_md = (src0_d == dst_d) && (src1_d == dst_d);
    conf.plain_to_ABcd4a4b = false;
    conf.isXa16b = false;
    conf.mb_block = 0;
    conf.is_src1_broadcast = check_layout_constraints(src_md(1));
    conf.is_src0_blocked = false;
    conf.has_tail = 0;

    for (int i = 0; i < MAX_NDIMS; ++i) {
        // Kernel doesn't support src0 broadcast
        if (i < ndims && src0_d.dims()[i] == 1
                && src0_d.dims()[i] != src1_d.dims()[i]) {
            return status::unimplemented;
        }
        conf.src1_bcast_dims[i] = i < ndims ? broadcast_dims()[i] : 1;
    }

    if (conf.src1_bcast_dims[1] && !conf.src1_bcast_dims[ndims - 1]) {
        conf.nvect = 1;
    } else {
        conf.nvect = 8;
        while (dst_d.dims()[ndims - 1] % conf.nvect != 0) {
            conf.nvect /= 2;
        }
    }

    auto *compute_engine = utils::downcast<compute::compute_engine_t *>(engine);
    conf.dispatch = compute_engine->create_dispatch(dst_d.md_);

    format_tag_t dst_tag = dst_d.matches_one_of_tag(nc, ncw, nchw, ncdhw);
    conf.is_plain_layout = dst_tag;

    conf.is_src0_blocked = check_mixed_layout(src0_d, dst_d, src_md(1));
    bool is_src1_blocked = check_mixed_layout(src1_d, dst_d, src_md(0));
    bool is_mixed_layout = check_broadcast(src_md(0), src_md(1))
            && (conf.is_src0_blocked || is_src1_blocked);

    format_tag_t src0_16b
            = src0_d.matches_one_of_tag(aBc16b, aBcd16b, aBcde16b);
    bool is_16b = src1_d.matches_tag(src0_16b) && dst_d.matches_tag(src0_16b);

    conf.isXa16b = src0_d.matches_one_of_tag(
                           ABcd32a16b, ABcde32a16b, ABcd16a16b, ABcde16a16b)
            && dst_d.matches_one_of_tag(
                    ABcd32a16b, ABcde32a16b, ABcd16a16b, ABcde16a16b)
            && src1_d.matches_one_of_tag(
                    ABcd32a16b, ABcde32a16b, ABcd16a16b, ABcde16a16b);

    format_tag_t src_plain = src0_d.matches_one_of_tag(abcd, acdb);
    const auto &padded_dims = dst_d.padded_dims();

    bool plain_and_X4a4b = (src1_d.matches_tag(src_plain)
            && dst_d.matches_one_of_tag(ABcd4a4b) && src0_d.is_dense()
            && dst_d.is_dense(true) && padded_dims[3] % 16 == 0
            && dst_d.data_type() != dnnl_f32);

    if (plain_and_X4a4b) {
        dim_t blocks[MAX_NDIMS] = {1, 1, 1, 1, 1, 1};
        auto &blk = dst_d.blocking_desc();
        int b_block = blk.inner_blks[blk.inner_nblks - 1];
        int sub_group_size = (b_block == 2 ? 8 : 16);
        blocks[0] = 4;
        blocks[1] = b_block;
        int vect_dim = 3;
        conf.nvect = 8;
        for (int i = 0; i < MAX_NDIMS; ++i) {
            auto dim_str = utils::format("D%d", i);
            if (i < dst_d.ndims()) {
                conf.dispatch.define_dim(dim_str, i, padded_dims[i], blocks[i]);
            } else {
                conf.dispatch.define_dim(dim_str, 1);
            }
        }

        auto dim_str = utils::format("D%d", vect_dim);
        CHECK(conf.dispatch.vectorize_dim(dim_str, sub_group_size));
        conf.plain_to_ABcd4a4b = true;
    } else if (conf.isXa16b) {
        if (is_broadcast()) return status::unimplemented;
        conf.nvect = 8;
        int channel_blk = 16;
        const int vect_dim_size = 16;
        const int padded_channels = padded_dims[1];
        conf.mb_block = dst_d.md_->format_desc.blocking.inner_blks[0];
        while (padded_channels % (vect_dim_size * channel_blk) != 0) {
            channel_blk /= 2;
        }
        dim_t blocks[MAX_NDIMS] = {8, channel_blk, 1, 1, 1, 1};
        for (int i = 0; i < MAX_NDIMS; ++i) {
            auto dim_str = utils::format("D%d", i);
            if (i < dst_d.ndims()) {
                conf.dispatch.define_dim(dim_str, i, padded_dims[i], blocks[i]);
                if (i == 1) {
                    CHECK(conf.dispatch.vectorize_dim(dim_str, vect_dim_size));
                }
            } else {
                conf.dispatch.define_dim(dim_str, 1);
            }
        }
    } else if ((conf.is_src0_blocked || is_16b) && conf.is_src1_broadcast) {
        int idx = 0;
        if (!is_16b) {
            idx = 1;
            // Setting the MB as the innermost dim for optimized performance
            // Hence starting i = 1, ignoring MB
            conf.dispatch.define_dim_with_nesting_level(
                    "D0", ndims, dst_d.dims()[0], 1);
        }
        for (int i = idx; i < MAX_NDIMS; ++i) {
            int dim = i < ndims ? dst_d.dims()[i] : 1;
            if (i == 1) {
                conf.dispatch.define_dim(utils::format("D%d", i),
                        nstl::min(i, ndims - 1), dim, 1);
                CHECK(conf.dispatch.vectorize_dim("D1", 16));
            } else if (i == ndims - 1) {
                conf.dispatch.define_dim(utils::format("D%d", i),
                        nstl::min(i, ndims - 1), dim, conf.nvect);
            } else {
                conf.dispatch.define_dim(utils::format("D%d", i),
                        nstl::min(i, ndims - 1), dim, 1);
            }
        }
    } else if (is_mixed_layout) {
        conf.nvect = 1;
        int block_size = 16;
        bool size_check = true;
        if (dst_d.matches_tag(aBc16b) && dst_d.dims()[2] % 16 != 0) {
            size_check = false;
        } else if (dst_d.matches_tag(aBcd16b)
                && (dst_d.dims()[3] % 16 != 0 || dst_d.dims()[2] % 16 != 0)) {
            size_check = false;
        } else if (dst_d.matches_tag(aBcde16b)
                && (dst_d.dims()[4] % 16 != 0 || dst_d.dims()[3] % 16 != 0
                        || dst_d.dims()[2] % 16 != 0)) {
            size_check = false;
        }
        if (!size_check) return status::unimplemented;
        for (int i = 0; i < MAX_NDIMS; ++i) {
            int dim = i < ndims ? dst_d.dims()[i] : 1;
            if (i == 1) {
                conf.dispatch.define_dim(utils::format("D%d", i),
                        nstl::min(i, ndims - 1), dim, 1);
                CHECK(conf.dispatch.vectorize_dim("D1", block_size));
            } else if (i == ndims - 1) {
                conf.dispatch.define_dim(utils::format("D%d", i),
                        nstl::min(i, ndims - 1), dim, block_size);
            } else {
                conf.dispatch.define_dim(utils::format("D%d", i),
                        nstl::min(i, ndims - 1), dim, 1);
            }
        }
    } else if (conf.is_plain_layout) {

        if (!src0_d.matches_tag(dst_tag) || !src1_d.matches_tag(dst_tag)) {
            return status::unimplemented;
        }

        const int subgroup_size = 16;
        int rem = (dst_d.dims()[ndims - 1]) % subgroup_size;
        if (rem) { conf.has_tail = 1; }
        conf.nvect = subgroup_size;
        bool all_dims_broadcast = true;

        for (int i = 0; i < ndims; i++) {
            if (src1_d.dims()[i] != 1) all_dims_broadcast = false;
        }

        if (rem && !all_dims_broadcast) { return status::unimplemented; }

        int rounded_last_dim = rem
                ? utils::rnd_up(dst_d.dims()[ndims - 1], subgroup_size)
                : dst_d.dims()[ndims - 1];

        while ((rounded_last_dim / 16) % conf.nvect != 0) {
            --conf.nvect;
        }

        dim_t mixed_dim = rounded_last_dim;
        for (int i = 0; i < (ndims - 1); ++i) {
            mixed_dim *= dst_d.dims()[i];
        }

        conf.dispatch.define_dim("MIXED_DIM", 0, mixed_dim, conf.nvect);
        CHECK(conf.dispatch.vectorize_dim("MIXED_DIM", 16));
    } else {
        return status::unimplemented;
    }

    conf.dispatch.generate();
    return status::success;
}

status_t gen9_binary_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    kernel_ctx.set_data_type(conf.src0_data_type);
    kernel_ctx.define_int("SUB_GROUP_SIZE", 16);
    kernel_ctx.define_int("NDIMS", conf.ndims);
    kernel_ctx.define_int("IS_PLAIN_LAYOUT", conf.is_plain_layout);
    kernel_ctx.define_int("PLAIN_TO_ABCD4AXB", conf.plain_to_ABcd4a4b);
    kernel_ctx.define_int("IS_XA16B", conf.isXa16b);
    kernel_ctx.define_int("IS_MUL", conf.is_mul);
    kernel_ctx.define_int("IS_ADD", conf.is_add);
    kernel_ctx.define_int("IS_MAX", conf.is_max);
    kernel_ctx.define_int("IS_MIN", conf.is_min);
    kernel_ctx.define_int("IS_DIV", conf.is_div);
    kernel_ctx.define_int("IS_SUB", conf.is_sub);
    kernel_ctx.define_int("IS_GE", conf.is_ge);
    kernel_ctx.define_int("IS_GT", conf.is_gt);
    kernel_ctx.define_int("IS_LE", conf.is_le);
    kernel_ctx.define_int("IS_LT", conf.is_lt);
    kernel_ctx.define_int("IS_EQ", conf.is_eq);
    kernel_ctx.define_int("IS_NE", conf.is_ne);
    kernel_ctx.define_int("MB_BLOCK", conf.mb_block);
    kernel_ctx.define_int("HAS_TAIL", conf.has_tail);
    kernel_ctx.define_int("SAME_SRC_DT", conf.same_src_dt);
    kernel_ctx.define_int("IS_SRC1_BROADCAST", conf.is_src1_broadcast);
    kernel_ctx.define_int("IS_SRC0_BLOCKED", conf.is_src0_blocked);
    kernel_ctx.define_int("BCAST_DIM0", conf.src1_bcast_dims[0]);
    kernel_ctx.define_int("BCAST_DIM1", conf.src1_bcast_dims[1]);
    kernel_ctx.define_int("BCAST_DIM2", conf.src1_bcast_dims[2]);
    kernel_ctx.define_int("BCAST_DIM3", conf.src1_bcast_dims[3]);
    kernel_ctx.define_int("BCAST_DIM4", conf.src1_bcast_dims[4]);
    kernel_ctx.define_int("BCAST_DIM5", conf.src1_bcast_dims[5]);
    kernel_ctx.define_int(
            "BCAST_AT_INNERMOST_DIM", conf.src1_bcast_dims[conf.ndims - 1]);
    kernel_ctx.define_int("NVECT", conf.nvect);
    kernel_ctx.add_option("-Dcl_intel_subgroups_char");
    kernel_ctx.add_option("-Dcl_intel_subgroups_uchar");

    def_memory_desc_info(kernel_ctx, conf.src0_md_info, "SRC0");
    def_memory_desc_info(kernel_ctx, conf.src1_md_info, "SRC1");
    def_memory_desc_info(kernel_ctx, conf.dst_md_info, "DST");

    CHECK(def_attr_info(kernel_ctx, conf.attr_info, attr()->post_ops_,
            conf.dst_md_info.dims));

    def_dispatch(kernel_ctx, conf.dispatch);

    return status::success;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
