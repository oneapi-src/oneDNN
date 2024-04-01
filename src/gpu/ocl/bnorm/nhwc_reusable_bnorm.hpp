/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
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

#ifndef GPU_OCL_NHWC_REUSABLE_BNORM_HPP
#define GPU_OCL_NHWC_REUSABLE_BNORM_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "gpu/compute/dispatch_reusable.hpp"
#include "gpu/compute/kernel.hpp"
#include "gpu/gpu_batch_normalization_pd.hpp"
#include "gpu/gpu_primitive.hpp"
#include "gpu/gpu_resource.hpp"
#include "gpu/ocl/ocl_stream.hpp"
#include "gpu/ocl/ocl_utils.hpp"
#include "gpu/primitive_conf.hpp"
#include "gpu/serialization.hpp"

#include "common/experimental.hpp"
#include "gpu/ocl/bnorm/bnorm_utils.hpp"
#include "gpu/ocl/bnorm/nhwc_batch_normalization.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
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
                "nhwc_reusable_update_fwd", "nhwc_reusable_calc_mean",
                "nhwc_reusable_calc_var", "nhwc_reusable_reduce_fwd_reg",
                "nhwc_reusable_calc_mean_var", "nhwc_reusable_reduce_fwd_1pass",
                "nhwc_reusable_reduce_aux"};
        return kernel_names;
    }

#if __cplusplus >= 202002L
    bool operator==(const reusable_bnorm_params_t &) const = default;
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
    compute::range_t calc_adj_lws;
};

struct nhwc_reusable_batch_normalization_fwd_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_batch_normalization_fwd_pd_t {
        pd_t(const batch_normalization_desc_t *adesc,
                const primitive_attr_t *attr,
                const batch_normalization_fwd_pd_t *hint_fwd_pd)
            : gpu_batch_normalization_fwd_pd_t(adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T(
                impl_name(), nhwc_reusable_batch_normalization_fwd_t);
        const char *impl_name() const {
            return bn_conf.use_stats_one_pass ? "ocl:nhwc_reusable:onepass"
                                              : "ocl:nhwc_reusable";
        }

        status_t init(engine_t *engine) {
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

        status_t init_conf(engine_t *engine);
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

    status_t init(engine_t *engine) override {
        if (pd()->has_zero_dim_memory()) return status::success;

        // Since selection of reduction kind occures at run-rime,
        // all kind of reduction kernels (trhu sratchpad or atomic-based)
        // must be build at once

        std::vector<const char *> kernel_names = {
                "nhwc_reusable_update_fwd", nullptr, nullptr, nullptr, nullptr};
        if (pd()->cmpl_conf.calculate_stats) {
            if (pd()->cmpl_conf.use_stats_one_pass) {
                kernel_names[1] = "nhwc_reusable_calc_mean_var";
                kernel_names[2] = "nhwc_reusable_reduce_fwd_1pass";
                kernel_names[3] = "nhwc_reusable_reduce_aux";
            } else { // regular algorithm
                kernel_names[1] = "nhwc_reusable_calc_mean";
                kernel_names[2] = "nhwc_reusable_calc_var";
                kernel_names[3] = "nhwc_reusable_reduce_fwd_reg";
                kernel_names[4] = "nhwc_reusable_reduce_aux";
            }
        }

        std::vector<compute::kernel_t> kernels;
        CHECK(create_kernels(engine, kernels, kernel_names, pd()->cmpl_conf));

        update_kernel_ = kernels[0];
        if (pd()->cmpl_conf.calculate_stats) {
            if (pd()->cmpl_conf.use_stats_one_pass) {
                calculate_mean_var_kernel_ = kernels[1];
                reduce_mean_var_kernel_ = kernels[2];
                reduce_aux_kernel_ = kernels[3];
            } else { // regular algorithm
                calculate_mean_kernel_ = kernels[1];
                calculate_var_kernel_ = kernels[2];
                reduce_fwd_reg_kernel_ = kernels[3];
                reduce_aux_kernel_ = kernels[4];
            }
        }

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    compute::kernel_t update_kernel_;
    compute::kernel_t calculate_mean_kernel_;
    compute::kernel_t calculate_var_kernel_;
    compute::kernel_t reduce_fwd_reg_kernel_;
    compute::kernel_t calculate_mean_var_kernel_;
    compute::kernel_t reduce_mean_var_kernel_;
    compute::kernel_t reduce_aux_kernel_;
};

// TODO: BWD part

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
