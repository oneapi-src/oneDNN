/*******************************************************************************
* Copyright 2020-2024 Intel Corporation
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

#ifndef GPU_INTEL_OCL_GEN9_SOFTMAX_HPP
#define GPU_INTEL_OCL_GEN9_SOFTMAX_HPP

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "gpu/gpu_softmax_pd.hpp"
#include "gpu/intel/compute/utils.hpp"
#include "gpu/intel/gpu_primitive.hpp"
#include "gpu/intel/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

struct gen9_softmax_fwd_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_softmax_fwd_pd_t {
        using gpu_softmax_fwd_pd_t::gpu_softmax_fwd_pd_t;

        DECLARE_COMMON_PD_T("ocl:gen9", gen9_softmax_fwd_t);

        status_t init(impl::engine_t *engine) {
            using namespace dnnl::impl::format_tag;
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);

            auto arch = compute_engine->device_info()->gpu_arch();
            const memory_desc_wrapper src_d(src_md());
            const memory_desc_wrapper dst_d(dst_md());
            const auto src_dt = src_d.data_type();
            const auto dst_dt = dst_d.data_type();

            using namespace data_type;
            using skip_mask_t = primitive_attr_t::skip_mask_t;
            is_nhwc = (src_d.matches_one_of_tag(nwc, nhwc, ndhwc)
                    != format_tag::undef);
            is_blocked = (src_d.matches_one_of_tag(nCw16c, nChw16c, nCdhw16c)
                    != format_tag::undef);

            VDISPATCH_SOFTMAX(is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_SOFTMAX(
                    IMPLICATION(is_blocked, axis_size() % buffer_size == 0),
                    VERBOSE_BAD_AXIS);
            VDISPATCH_SOFTMAX(!memory_desc_ndims_ok(src_md(), dst_md()),
                    VERBOSE_INCONSISTENT_NDIMS, "src", "dst");
            VDISPATCH_SOFTMAX(axis() == src_d.ndims() - 1, VERBOSE_BAD_AXIS);
            VDISPATCH_SOFTMAX((src_d.is_plain() || is_blocked || is_nhwc),
                    VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_SOFTMAX(
                    utils::one_of(src_dt, f64, f32, f16, bf16, u8, s8),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_SOFTMAX(
                    utils::one_of(dst_dt, f64, f32, f16, bf16, u8, s8),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_SOFTMAX(IMPLICATION(utils::one_of(f16, src_dt, dst_dt),
                                      compute_engine->mayiuse(
                                              compute::device_ext_t::khr_fp16)),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_SOFTMAX(IMPLICATION(utils::one_of(data_type::f64,
                                                  dst_md()->data_type,
                                                  src_md()->data_type),
                                      compute_engine->mayiuse(
                                              compute::device_ext_t::khr_fp64)),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_SOFTMAX(
                    attr()->has_default_values(skip_mask_t::scales_runtime),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_SOFTMAX(attr_scales_ok(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_SOFTMAX_SC(
                    set_default_formats(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_SOFTMAX(compute_engine->mayiuse_sub_group(subgroup_size),
                    VERBOSE_UNSUPPORTED_DEVICE_FEATURE, "subgroup_size");

            VDISPATCH_SOFTMAX(
                    !(is_blocked && src_md()->dims[1] % subgroup_size != 0),
                    VERBOSE_UNSUPPORTED_MEM_STRIDE);
            // max lws size on Xe-HP* series is 1024 x 1024 x 1024
            // max lws size on Xe-LP is 512 x 512 x 512

            int max_lws = 256; // for Gen9, Gen11
            if (arch >= compute::gpu_arch_t::xe_hp) {
                max_lws = 1024;
            } else if (arch == compute::gpu_arch_t::xe_lp) {
                max_lws = 512;
            }

            if (is_nhwc) {
                int axis_padded = utils::rnd_up(axis_size(), subgroup_size);
                group_size = subgroup_size
                        * utils::div_up(axis_padded, buffer_size);
                if (group_size > (size_t)max_lws) {
                    int old_group_size = (int)group_size;
                    group_size = max_lws;

                    int lws_ratio = old_group_size / (int)group_size;
                    int rem_threads = old_group_size % max_lws;
                    int rem_reads = rem_threads * thread_buffer;

                    // initial calculation of repeating subgroups with the
                    // assumption that group size is divisible by 128
                    // and thread buffer is 8
                    subgroups_repeated = rem_threads / subgroup_size;
                    thread_reads = thread_buffer * lws_ratio;
                    buffer_size *= lws_ratio;

                    repeated_subgrp_buffer = thread_buffer = thread_reads;

                    // re-calculate repeated subgroups number with
                    // new buffer size (256, 512 ...) if conditions met
                    if (lws_ratio >= 2 && subgroups_repeated > 0) {
                        subgroups_repeated = rem_reads / buffer_size;
                        // The new buffer size may be too large for last subgroup
                        // as it will be handling 128-byte buffer size
                        if (rem_reads % buffer_size != 0) {
                            int tail_reads = rem_reads / subgroup_size;
                            repeated_subgrp_buffer = tail_reads % thread_buffer;
                            subgroups_repeated++;
                        }
                    }
                }
            } else {
                bool avoid_large_spatial
                        = (src_md()->dims[0] * src_md()->dims[1] > 128)
                        && (axis() > 1);
                if (!is_blocked
                        && (axis_size() % buffer_size != 0
                                || avoid_large_spatial)) {
                    group_size = subgroup_size;
                } else {
                    group_size = subgroup_size
                            * utils::div_up(axis_size(), buffer_size);
                }
                VDISPATCH_SOFTMAX(group_size <= (size_t)max_lws,
                        "unsupported group_size, %d <= %d", (int)group_size,
                        max_lws);
            }

            lws[0] = group_size;
            gws[0] = utils::array_product(&src_md()->dims[0], ndims() - 1)
                    * group_size;

            //subgroup block read requires the tensor to be 4-byte aligned, and
            //subgroup block write requires the tensor to be 16-byte aligned
            if ((axis_size() * types::data_type_size(src_dt))
                            % byte_alignment_read
                    == 0)
                is_read_aligned = true;
            if ((axis_size() * types::data_type_size(dst_dt))
                            % byte_alignment_write
                    == 0)
                is_write_aligned = true;
            return status::success;
        }

        bool is_nhwc = false;
        bool is_blocked = false;
        bool is_write_aligned = false;
        bool is_read_aligned = false;
        compute::range_t gws = compute::range_t::empty(1);
        compute::range_t lws = compute::range_t::empty(1);
        size_t group_size = 0;
        const int subgroup_size = 16;
        const int byte_alignment_read = 4;
        const int byte_alignment_write = 16;
        int thread_reads = 0;
        int thread_buffer = 8;
        int subgroups_repeated = 0;
        int repeated_subgrp_buffer = 8;
        // 8x16 load and store commands (Vector_Size x Sub_Group_Size)
        int buffer_size = 128;
    };

    status_t init(impl::engine_t *engine) override {
        if (pd()->has_zero_dim_memory()) return status::success;

        compute::kernel_ctx_t kernel_ctx;

        kernel_ctx.define_int("SOFTMAX_AXIS_IDX", pd()->axis());
        kernel_ctx.define_int("SOFTMAX_AXIS_SIZE", pd()->axis_size());
        kernel_ctx.define_int("SOFTMAX_BUF", pd()->buffer_size);
        kernel_ctx.define_int("GROUP_SIZE", pd()->group_size);
        kernel_ctx.define_int("SUB_GROUP_SIZE", pd()->subgroup_size);
        kernel_ctx.define_int("THREAD_BUF_SIZE", pd()->thread_buffer);
        kernel_ctx.define_int("SUBGROUPS_REPEATED", pd()->subgroups_repeated);
        kernel_ctx.define_int(
                "REPEAT_SUBGRP_BUF_SIZE", pd()->repeated_subgrp_buffer);
        kernel_ctx.define_int(
                "CHANNELS_PADDED", pd()->src_md()->padded_dims[1]);
        kernel_ctx.define_int("CHANNELS",
                pd()->is_blocked ? pd()->subgroup_size
                                 : pd()->src_md(0)->padded_dims[1]);
        kernel_ctx.define_int("IS_NHWC", pd()->is_nhwc);
        kernel_ctx.define_int("IS_BLOCKED", pd()->is_blocked);
        kernel_ctx.define_int("IS_READ_ALIGNED", pd()->is_read_aligned);
        kernel_ctx.define_int("IS_WRITE_ALIGNED", pd()->is_write_aligned);
        kernel_ctx.define_int("IS_FWD", 1);
        kernel_ctx.add_option("-cl-std=CL2.0");
        kernel_ctx.define_int("LOGSOFTMAX", pd()->is_logsoftmax());
        kernel_ctx.define_int("WITH_SRC_SCALES",
                !pd()->attr()->scales_.get(DNNL_ARG_SRC).has_default_values());
        kernel_ctx.define_int("WITH_DST_SCALES",
                !pd()->attr()->scales_.get(DNNL_ARG_DST).has_default_values());

        const memory_desc_wrapper dst_mdw(pd()->dst_md());
        const memory_desc_wrapper src_mdw(pd()->src_md());
        const auto dst_md_info = memory_desc_info_t::create(dst_mdw);
        const auto src_md_info = memory_desc_info_t::create(src_mdw);
        def_memory_desc_info(kernel_ctx, dst_md_info, "DST");
        def_memory_desc_info(kernel_ctx, src_md_info, "SRC");
        kernel_ctx.set_data_type(dst_mdw.data_type());
        set_offsets(kernel_ctx, pd()->dst_md(), "DATA");

        CHECK(create_kernel(engine, &kernel_, "gen9_softmax_fwd", kernel_ctx));
        if (!kernel_) return status::runtime_error;

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_generic(ctx);
    }

protected:
    status_t execute_generic(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    compute::kernel_t kernel_;
};

struct gen9_softmax_bwd_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_softmax_bwd_pd_t {
        using gpu_softmax_bwd_pd_t::gpu_softmax_bwd_pd_t;

        DECLARE_COMMON_PD_T("ocl:gen9", gen9_softmax_bwd_t);

        status_t init(impl::engine_t *engine) {
            using namespace dnnl::impl::format_tag;

            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);

            const memory_desc_wrapper diff_src_d(diff_src_md());
            const memory_desc_wrapper diff_dst_d(diff_dst_md());
            const memory_desc_wrapper dst_d(dst_md());

            using namespace data_type;
            VDISPATCH_SOFTMAX(!is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_SOFTMAX(axis_size() % buffer_size == 0, VERBOSE_BAD_AXIS);
            VDISPATCH_SOFTMAX(!memory_desc_ndims_ok(
                                      dst_md(), diff_src_md(), diff_dst_md()),
                    VERBOSE_INCONSISTENT_NDIMS, "dst, dst_src", "diff_dst");
            VDISPATCH_SOFTMAX(
                    axis() == diff_src_d.ndims() - 1, VERBOSE_BAD_AXIS);
            VDISPATCH_SOFTMAX(
                    utils::one_of(diff_src_d.data_type(), f64, f32, bf16, f16),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_SOFTMAX(
                    utils::one_of(diff_dst_d.data_type(), f64, f32, bf16, f16),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_SOFTMAX(compute_engine->mayiuse_sub_group(subgroup_size),
                    VERBOSE_UNSUPPORTED_DEVICE_FEATURE, "subgroup_size");
            VDISPATCH_SOFTMAX(IMPLICATION(utils::one_of(data_type::f64,
                                                  diff_dst_md()->data_type,
                                                  diff_src_md()->data_type),
                                      compute_engine->mayiuse(
                                              compute::device_ext_t::khr_fp64)),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_SOFTMAX(IMPLICATION(utils::one_of(data_type::f16,
                                                  diff_dst_md()->data_type,
                                                  diff_src_md()->data_type),
                                      compute_engine->mayiuse(
                                              compute::device_ext_t::khr_fp16)),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_SOFTMAX(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_SOFTMAX_SC(
                    set_default_formats(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_SOFTMAX(diff_dst_d.data_type() == dst_d.data_type(),
                    VERBOSE_INCONSISTENT_DT, "diff_dst", "dst");

            is_nhwc = (diff_src_d.matches_one_of_tag(nwc, nhwc, ndhwc)
                    != format_tag::undef);
            is_blk = (diff_src_d.matches_one_of_tag(nCw16c, nChw16c, nCdhw16c)
                    != format_tag::undef);
            if (is_nhwc || is_blk) {
                group_size = subgroup_size * (axis_size() / buffer_size);
            } else {
                group_size = subgroup_size;
            }
            lws[0] = group_size;
            gws[0] = utils::array_product(
                             &diff_src_md(0)->padded_dims[0], ndims() - 1)
                    * group_size;
            batches = diff_src_md(0)->padded_dims[0]
                    * diff_src_md(0)->padded_dims[2];
            return status::success;
        }

        compute::range_t gws = compute::range_t::empty(1);
        compute::range_t lws = compute::range_t::empty(1);
        size_t group_size = 0;
        size_t batches = 0;
        bool is_nhwc = false;
        bool is_blk = false;
        const int subgroup_size = 16;
        // 8x16 load and store commands (Vector_Size x Sub_Group_Size)
        const int buffer_size = 128;
    };

    status_t init(impl::engine_t *engine) override {
        if (pd()->has_zero_dim_memory()) return status::success;

        compute::kernel_ctx_t kernel_ctx;

        kernel_ctx.define_int("SOFTMAX_AXIS_IDX", pd()->axis());
        kernel_ctx.define_int("SOFTMAX_AXIS_SIZE", pd()->axis_size());
        kernel_ctx.define_int("SOFTMAX_BUF", pd()->buffer_size);
        kernel_ctx.define_int("SUB_GROUP_SIZE", pd()->subgroup_size);
        kernel_ctx.define_int("GROUP_SIZE", pd()->group_size);
        kernel_ctx.define_int("IS_BWD", 1);
        kernel_ctx.define_int("IS_16C", pd()->is_blk);
        kernel_ctx.define_int("BATCH", pd()->batches);
        kernel_ctx.define_int("IC_WO_PADDING", pd()->diff_src_md(0)->dims[1]);
        kernel_ctx.define_int(
                "IC_PADDED", pd()->diff_src_md(0)->padded_dims[1]);
        kernel_ctx.define_int("IC",
                pd()->is_blk ? pd()->subgroup_size
                             : pd()->diff_src_md(0)->padded_dims[1]);
        kernel_ctx.define_int("IS_NHWC", pd()->is_nhwc);
        kernel_ctx.add_option("-cl-std=CL2.0");
        kernel_ctx.define_int("LOGSOFTMAX", pd()->is_logsoftmax());

        const memory_desc_wrapper diff_src_mdw(pd()->diff_src_md());
        const memory_desc_wrapper diff_dst_mdw(pd()->diff_dst_md());
        const auto diff_src_md_info = memory_desc_info_t::create(diff_src_mdw);
        const auto diff_dst_md_info = memory_desc_info_t::create(diff_dst_mdw);
        def_memory_desc_info(kernel_ctx, diff_src_md_info, "SRC");
        def_memory_desc_info(kernel_ctx, diff_dst_md_info, "DST");
        kernel_ctx.set_data_type(pd()->diff_src_md()->data_type);
        set_offsets(kernel_ctx, *pd()->diff_src_md(), "DATA");

        CHECK(create_kernel(engine, &kernel_, "gen9_softmax_bwd", kernel_ctx));
        if (!kernel_) return status::runtime_error;

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_generic(ctx);
    }

protected:
    status_t execute_generic(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    compute::kernel_t kernel_;
};
} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
