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

#ifndef GPU_OCL_GEN9_ELTWISE_HPP
#define GPU_OCL_GEN9_ELTWISE_HPP

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/gpu_eltwise_pd.hpp"
#include "gpu/gpu_primitive.hpp"
#include "gpu/gpu_resource.hpp"
#include "gpu/ocl/ocl_stream.hpp"
#include "gpu/ocl/ocl_utils.hpp"
#include "gpu/primitive_conf.hpp"
#include "gpu/serialization.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

struct gen9_eltwise_jit_params_t {
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

#if __cplusplus >= 202002L
    bool operator==(const gen9_eltwise_jit_params_t &) const = default;
#endif
    serialized_t serialize() const {
        serialized_t s {};
        // Explicitly maintain zero padding to keep the implementation simple and
        // robust
        s.append(*this);
        return s;
    }

    static gen9_eltwise_jit_params_t deserialize(const serialized_t &s) {
        gen9_eltwise_jit_params_t t {};
        deserializer_t d(s);
        d.pop(t);
        return t;
    }

    status_t init(engine_t *engine, const memory_desc_wrapper data_d,
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

        status_t init(engine_t *engine) {
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);

            using namespace alg_kind;
            bool ok = is_fwd() && src_md()->data_type == dst_md()->data_type
                    && attr()->has_default_values()
                    && set_default_formats_common()
                    && memory_desc_wrapper(src_md())
                            == memory_desc_wrapper(dst_md())
                    && IMPLICATION(src_md()->data_type == data_type::f16,
                            compute_engine->mayiuse(
                                    compute::device_ext_t::khr_fp16))
                    && IMPLICATION(src_md()->data_type == data_type::f64,
                            compute_engine->mayiuse(
                                    compute::device_ext_t::khr_fp64))
                    && compute_engine->mayiuse_sub_group(16);
            if (!ok) return status::unimplemented;

            return init_conf(engine);
        }

        status_t init_conf(engine_t *engine);

        gen9_eltwise_jit_params_t conf;
    };

    status_t init(engine_t *engine) override {
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
        pd_t(const eltwise_desc_t *adesc, const primitive_attr_t *attr,
                const eltwise_fwd_pd_t *hint_fwd_pd)
            : gpu_eltwise_bwd_pd_t(adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T("ocl:gen9:any", gen9_eltwise_bwd_t);

        status_t init(engine_t *engine) {
            using namespace prop_kind;
            using namespace utils;
            assert(engine->kind() == engine_kind::gpu);
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);

            using namespace alg_kind;
            bool ok = !is_fwd()
                    && utils::one_of(data_md()->data_type, data_type::f32,
                            data_type::bf16, data_type::f16, data_type::f64)
                    && utils::everyone_is(data_md()->data_type,
                            diff_src_md()->data_type, diff_dst_md()->data_type)
                    && set_default_formats_common()
                    && attr()->has_default_values()
                    && IMPLICATION(data_md()->data_type == data_type::f16,
                            compute_engine->mayiuse(
                                    compute::device_ext_t::khr_fp16))
                    && IMPLICATION(data_md()->data_type == data_type::f64,
                            compute_engine->mayiuse(
                                    compute::device_ext_t::khr_fp64))
                    && memory_desc_wrapper(diff_dst_md())
                            == memory_desc_wrapper(diff_src_md());
            if (!ok) return status::unimplemented;

            return init_conf(engine);
        }

        status_t init_conf(engine_t *engine);

        gen9_eltwise_jit_params_t conf;
    };

    status_t init(engine_t *engine) override {
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
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
