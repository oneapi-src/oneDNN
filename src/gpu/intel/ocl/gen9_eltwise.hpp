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

#ifndef GPU_INTEL_OCL_GEN9_ELTWISE_HPP
#define GPU_INTEL_OCL_GEN9_ELTWISE_HPP

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "gpu/gpu_eltwise_pd.hpp"
#include "gpu/intel/gpu_primitive.hpp"
#include "gpu/intel/serialization.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

struct gen9_eltwise_jit_params_t
    : public trivially_serializable_t<gen9_eltwise_jit_params_t> {
    status_t create_generator(const compute::compute_engine_t &engine,
            compute::kernel_bundle_t &bundle) const {
        return engine.create_kernel_bundle(
                bundle, get_kernel_names(), get_kernel_ctx());
    }

    const std::vector<const char *> &get_kernel_names() const {
        static const std::vector<const char *> names {
                "gen9_eltwise_fwd", "gen9_eltwise_bwd"};
        return names;
    }

    status_t init(impl::engine_t *engine, const memory_desc_wrapper data_d,
            alg_kind_t alg_kind);
    compute::kernel_ctx_t get_kernel_ctx() const;

    compute::gpu_arch_t arch;
    int vector_size;
    data_type_t data_type;
    alg_kind_t alg_kind;
    int work_group_size;
    int sub_group_size;
    bool with_overflow;
    uint8_t pad0[3] = {};
};

struct gen9_eltwise_fwd_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_eltwise_fwd_pd_t {
        using gpu_eltwise_fwd_pd_t::gpu_eltwise_fwd_pd_t;

        DECLARE_COMMON_PD_T("ocl:gen9:any", gen9_eltwise_fwd_t);

        status_t init(impl::engine_t *engine) {
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);

            using namespace alg_kind;
            VDISPATCH_ELTWISE(is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_ELTWISE(src_md()->data_type == dst_md()->data_type,
                    VERBOSE_INCONSISTENT_DT, "src", "dst");
            VDISPATCH_ELTWISE(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_ELTWISE(
                    set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_ELTWISE(memory_desc_wrapper(src_md())
                            == memory_desc_wrapper(dst_md()),
                    VERBOSE_INCONSISTENT_MDS, "src", "dst");
            VDISPATCH_ELTWISE(IMPLICATION(src_md()->data_type == data_type::f16,
                                      compute_engine->mayiuse(
                                              compute::device_ext_t::khr_fp16)),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_ELTWISE(IMPLICATION(src_md()->data_type == data_type::f64,
                                      compute_engine->mayiuse(
                                              compute::device_ext_t::khr_fp64)),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_ELTWISE(compute_engine->mayiuse_sub_group(16),
                    VERBOSE_UNSUPPORTED_DEVICE_FEATURE, "subgroups");

            VDISPATCH_ELTWISE_SC(init_conf(engine),
                    VERBOSE_PRIMITIVE_CREATION_FAIL, "eltwise");
            return status::success;
        }

        status_t init_conf(impl::engine_t *engine);

        gen9_eltwise_jit_params_t conf;
    };

    status_t init(impl::engine_t *engine) override {
        return create_kernel(engine, kernel_, "gen9_eltwise_fwd", pd()->conf);
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward_dense(ctx);
    }

private:
    status_t execute_forward_dense(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    compute::kernel_t kernel_;
};

struct gen9_eltwise_bwd_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_eltwise_bwd_pd_t {
        using gpu_eltwise_bwd_pd_t::gpu_eltwise_bwd_pd_t;

        DECLARE_COMMON_PD_T("ocl:gen9:any", gen9_eltwise_bwd_t);

        status_t init(impl::engine_t *engine) {
            using namespace prop_kind;
            using namespace utils;
            assert(engine->kind() == engine_kind::gpu);
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);

            using namespace alg_kind;
            VDISPATCH_ELTWISE(!is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_ELTWISE(
                    utils::one_of(data_md()->data_type, data_type::f32,
                            data_type::bf16, data_type::f16, data_type::f64),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_ELTWISE(
                    utils::everyone_is(data_md()->data_type,
                            diff_src_md()->data_type, diff_dst_md()->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_ELTWISE(
                    set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_ELTWISE(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_ELTWISE(
                    IMPLICATION(data_md()->data_type == data_type::f16,
                            compute_engine->mayiuse(
                                    compute::device_ext_t::khr_fp16)),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_ELTWISE(
                    IMPLICATION(data_md()->data_type == data_type::f64,
                            compute_engine->mayiuse(
                                    compute::device_ext_t::khr_fp64)),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_ELTWISE(memory_desc_wrapper(diff_dst_md())
                            == memory_desc_wrapper(diff_src_md()),
                    VERBOSE_INCONSISTENT_MDS, "diff_src", "diff_dst");

            VDISPATCH_ELTWISE_SC(init_conf(engine),
                    VERBOSE_PRIMITIVE_CREATION_FAIL, "eltwise");
            return status::success;
        }

        status_t init_conf(impl::engine_t *engine);

        gen9_eltwise_jit_params_t conf;
    };

    status_t init(impl::engine_t *engine) override {
        return create_kernel(engine, kernel_, "gen9_eltwise_bwd", pd()->conf);
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward_dense(ctx);
    }

private:
    status_t execute_backward_dense(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    compute::kernel_t kernel_;
};

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
