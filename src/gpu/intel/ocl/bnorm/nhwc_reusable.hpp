/*******************************************************************************
* Copyright 2023-2025 Intel Corporation
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

#ifndef GPU_INTEL_OCL_BNORM_NHWC_REUSABLE_HPP
#define GPU_INTEL_OCL_BNORM_NHWC_REUSABLE_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "gpu/gpu_batch_normalization_pd.hpp"
#include "gpu/gpu_resource.hpp"
#include "gpu/intel/compute/dispatch_reusable.hpp"
#include "gpu/intel/compute/kernel.hpp"
#include "gpu/intel/gpu_primitive.hpp"
#include "gpu/intel/ocl/stream.hpp"
#include "gpu/intel/ocl/utils.hpp"
#include "gpu/intel/primitive_conf.hpp"
#include "gpu/intel/serialization.hpp"

#include "common/experimental.hpp"
#include "gpu/intel/ocl/bnorm/bnorm_utils.hpp"
#include "gpu/intel/ocl/bnorm/nhwc_batch_normalization.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

struct nhwc_reusable_bnorm_compile_params_t {
    status_t create_generator(const compute::compute_engine_t &engine,
            compute::kernel_bundle_t &bundle) const {
        auto status = engine.create_kernel_bundle(
                bundle, get_kernel_names(), get_kernel_ctx());
        return status;
    }

    const std::vector<const char *> &get_kernel_names() const {
        static const std::vector<const char *> kernel_names = {
                "nhwc_reusable_norm_fwd", "nhwc_reusable_calc_mean",
                "nhwc_reusable_calc_var", "nhwc_reusable_reduce_fwd_reg",
                "nhwc_reusable_calc_mean_var", "nhwc_reusable_reduce_fwd_1pass",
                "nhwc_reusable_reduce_aux", "nhwc_reusable_norm_bwd",
                "nhwc_reusable_calc_stat", "nhwc_reusable_reduce_stat",
                "nhwc_reusable_norm_fwd_buff", "nhwc_reusable_norm_bwd_buff",
                "nhwc_reusable_calc_mean_buff", "nhwc_reusable_calc_var_buff",
                "nhwc_reusable_calc_mean_var_buff",
                "nhwc_reusable_calc_stat_buff"};
        return kernel_names;
    }

#if __cplusplus >= 202002L
    bool operator==(
            const nhwc_reusable_bnorm_compile_params_t &) const = default;
#endif

    serialized_t serialize() const {
        assert_trivially_serializable(nhwc_reusable_bnorm_compile_params_t);
        return serialized_t(*this);
    }

    static nhwc_reusable_bnorm_compile_params_t deserialize(
            const serialized_t &s) {
        return deserializer_t(s).pop<nhwc_reusable_bnorm_compile_params_t>();
    }

    compute::kernel_ctx_t get_kernel_ctx() const;

    data_type_t data_type;
    int vect_size;
    int sub_group_size;
    int max_ic_block;
    bool use_scale;
    bool use_shift;
    bool is_training;
    bool fuse_norm_relu;
    bool fuse_norm_add_relu;
    bool with_relu;
    bool with_leaky_relu;
    bool calculate_stats;
    bool use_stats_one_pass;
    uint8_t padding[3] = {0};
};

struct nhwc_reusable_bnorm_runtime_params_t {
    dim_t ic_size, sp_size;
    dim_t update_sp_block, stat_sp_block, ic_block, update_sp_unroll;
    dim_t reduce_stat_nblocks;
    dim_t reduce_ic_sub_groups;
    dim_t sg_size;
    float relu_negative_slope;
    float eps;
    bool use_fused_atomics_reduction;
    bool use_buffers_calc;
    bool use_buffers_norm;
    compute::range_t calc_adj_lws;
};

struct nhwc_reusable_batch_normalization_fwd_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_batch_normalization_fwd_pd_t {
        using gpu_batch_normalization_fwd_pd_t::
                gpu_batch_normalization_fwd_pd_t;

        DECLARE_COMMON_PD_T(
                impl_name(), nhwc_reusable_batch_normalization_fwd_t);
        const char *impl_name() const {
            return bn_conf.use_stats_one_pass ? "ocl:nhwc_reusable:onepass"
                                              : "ocl:nhwc_reusable";
        }

        status_t init(impl::engine_t *engine) {
            using namespace data_type;
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);

            const auto attr_skip_mask = primitive_attr_t::skip_mask_t::post_ops;

            VDISPATCH_BNORM(is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_BNORM(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
            VDISPATCH_BNORM(
                    utils::one_of(src_md()->data_type, f32, bf16, f16, s8),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_BNORM(IMPLICATION(f16 == src_md()->data_type,
                                    compute_engine->mayiuse(
                                            compute::device_ext_t::khr_fp16)),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_BNORM(src_md()->data_type == dst_md()->data_type,
                    VERBOSE_INCONSISTENT_DT, "src", "dst");
            VDISPATCH_BNORM(IMPLICATION(src_md()->data_type == s8,
                                    !is_training() && stats_is_src()),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_BNORM(check_scale_shift_data_type(),
                    VERBOSE_UNSUPPORTED_FEATURE,
                    "unsupported scale, shift or datatype configuration");
            VDISPATCH_BNORM(attr()->has_default_values(attr_skip_mask),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_BNORM(
                    IMPLICATION(!attr()->has_default_values(),
                            attr()->post_ops_.len() == 1
                                    && with_relu_post_op(is_training())),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_BNORM(
                    set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_BNORM(memory_desc_wrapper(src_md())
                            == memory_desc_wrapper(dst_md()),
                    VERBOSE_INCONSISTENT_MDS, "src", "dst");
            VDISPATCH_BNORM(compute_engine->mayiuse(
                                    compute::device_ext_t::intel_subgroups),
                    VERBOSE_UNSUPPORTED_DEVICE_FEATURE, "subgroups");

            if (is_training() && (fuse_norm_relu() || fuse_norm_add_relu())) {
                VDISPATCH_BNORM_SC(init_default_ws(8), VERBOSE_WS_INIT);
            }

            VDISPATCH_BNORM_SC(init_conf(engine), "init_conf()");
            init_scratchpad();

            return status::success;
        }

        status_t init_conf(impl::engine_t *engine);
        void init_scratchpad();

        nhwc_reusable_bnorm_compile_params_t cmpl_conf;
        nhwc_reusable_bnorm_runtime_params_t rt_conf;
        nhwc_bnorm_params_t bn_conf;

        compute::dispatch_t dispatch_calc_stat;
        compute::dispatch_t dispatch_reduce_stat;
        compute::dispatch_t dispatch;
        compute::dispatch_t dispatch_reduce_aux;
    };

    status_t init(impl::engine_t *engine) override {
        if (pd()->has_zero_dim_memory()) return status::success;
        auto kernel_names = pd()->cmpl_conf.get_kernel_names();
        CHECK(create_kernels(engine, kernels_, kernel_names, pd()->cmpl_conf));
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::vector<compute::kernel_t> kernels_;
};

struct nhwc_reusable_batch_normalization_bwd_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_batch_normalization_bwd_pd_t {
        using gpu_batch_normalization_bwd_pd_t::
                gpu_batch_normalization_bwd_pd_t;

        DECLARE_COMMON_PD_T(
                "ocl:nhwc_reusable", nhwc_reusable_batch_normalization_bwd_t);

        status_t init(impl::engine_t *engine) {
            using namespace data_type;
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);

            VDISPATCH_BNORM(!is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_BNORM(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
            VDISPATCH_BNORM(utils::one_of(src_md()->data_type, f32, bf16, f16),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_BNORM(IMPLICATION(f16 == src_md()->data_type,
                                    compute_engine->mayiuse(
                                            compute::device_ext_t::khr_fp16)),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_BNORM(src_md()->data_type == diff_src_md()->data_type,
                    VERBOSE_INCONSISTENT_DT, "src", "diff_src");
            VDISPATCH_BNORM(
                    diff_src_md()->data_type == diff_dst_md()->data_type,
                    VERBOSE_INCONSISTENT_DT, "diff_src", "diff_dst");
            VDISPATCH_BNORM(check_scale_shift_data_type(),
                    VERBOSE_UNSUPPORTED_FEATURE,
                    "unsupported scale, shift or datatype configuration");
            VDISPATCH_BNORM(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_BNORM(
                    set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_BNORM(memory_desc_wrapper(diff_src_md())
                            == memory_desc_wrapper(diff_dst_md()),
                    VERBOSE_INCONSISTENT_MDS, "diff_src", "diff_dst");
            VDISPATCH_BNORM(compute_engine->mayiuse(
                                    compute::device_ext_t::intel_subgroups),
                    VERBOSE_UNSUPPORTED_DEVICE_FEATURE, "subgroups");

            if (fuse_norm_relu() || fuse_norm_add_relu()) {
                VDISPATCH_BNORM_SC(init_default_ws(8), VERBOSE_WS_INIT);
                VDISPATCH_BNORM(compare_ws(hint_fwd_pd_), VERBOSE_WS_MISMATCH);
            }

            VDISPATCH_BNORM_SC(init_conf(engine), "init_conf()");
            init_scratchpad();

            return status::success;
        }

        status_t init_conf(impl::engine_t *engine);
        void init_scratchpad();

        nhwc_reusable_bnorm_compile_params_t cmpl_conf;
        nhwc_reusable_bnorm_runtime_params_t rt_conf;
        nhwc_bnorm_params_t bn_conf;
        offsets_t off;
        compute::dispatch_t dispatch_calc_stat;
        compute::dispatch_t dispatch_reduce_stat;
        compute::dispatch_t dispatch;
        compute::dispatch_t dispatch_reduce_aux;
    };

    status_t init(impl::engine_t *engine) override {
        if (pd()->has_zero_dim_memory()) return status::success;
        auto kernel_names = pd()->cmpl_conf.get_kernel_names();
        CHECK(create_kernels(engine, kernels_, kernel_names, pd()->cmpl_conf));
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward(ctx);
    }

private:
    status_t execute_backward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::vector<compute::kernel_t> kernels_;
};

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
