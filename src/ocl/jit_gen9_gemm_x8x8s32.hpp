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

#ifndef JIT_GEN9_GEMM_X8X8S32_HPP
#define JIT_GEN9_GEMM_X8X8S32_HPP

#include <assert.h>
#include <memory>

#include "common/c_types_map.hpp"
#include "common/gemm_utils.hpp"
#include "compute/compute.hpp"
#include "ocl/gemm/ocl_gemm.hpp"
#include "ocl/jit_gen9_gemm_kernel_x8x8s32.hpp"
#include "ocl/ocl_gemm_pd.hpp"
#include "ocl/ocl_stream.hpp"
#include "ocl/ocl_utils.hpp"

namespace dnnl {
namespace impl {
namespace ocl {

template <impl::data_type_t a_type, impl::data_type_t b_type,
        impl::data_type_t c_type>
struct jit_gen9_gemm_x8x8s32_t : public ocl_gemm_t {
    using c_t = typename prec_traits<c_type>::type;
    using ao_t = typename prec_traits<a_type>::type;
    using bo_t = typename prec_traits<b_type>::type;

    enum class type { no_copy };

    struct pd_t : public ocl_gemm_pd_t {
        using hint_class = void;

        pd_t(engine_t *engine, const gemm_desc_t *adesc,
                const primitive_attr_t *attr, const hint_class *)
            : ocl_gemm_pd_t(engine, adesc, attr) {}

        DECLARE_COMMON_PD_T("ocl:gemm:any", jit_gen9_gemm_x8x8s32_t);

        status_t init() {
            using namespace prop_kind;
            using namespace data_type;
            using namespace primitive_kind;

            assert(this->engine()->kind() == engine_kind::gpu);
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine());

            const auto attr_skip_mask = primitive_attr_t::skip_mask_t::post_ops | primitive_attr_t::skip_mask_t::zero_points_runtime;

            bool ok = true && desc()->a_type == a_type
                    && desc()->b_type == b_type && desc()->c_type == c_type
                    && desc()->acc_type == c_type
                    && compute_engine->mayiuse(
                            compute::device_ext_t::intel_subgroups)
                    && IMPLICATION(c_type == s32,
                            true
                                    && compute_engine->mayiuse(
                                            compute::device_ext_t::
                                                    intel_subgroups_short))
                    && attr()->has_default_values(attr_skip_mask)
                    && attr()->post_ops_.len_ <= 1
                    && IMPLICATION(attr()->post_ops_.len_ == 1,
                            attr()->post_ops_.find(eltwise) != -1);
            if (!ok) return status::unimplemented;

            return status::success;
        }

        bool with_eltwise() const {
            return attr()->post_ops_.find(primitive_kind::eltwise) != -1;
        }

        float eltwise_alpha() const {
            const int eltwise_idx
                    = attr()->post_ops_.find(primitive_kind::eltwise);
            return with_eltwise()
                    ? attr()->post_ops_.entry_[eltwise_idx].eltwise.alpha
                    : 1.0f;
        }

        float eltwise_beta() const {
            const int eltwise_idx
                    = attr()->post_ops_.find(primitive_kind::eltwise);
            return with_eltwise()
                    ? attr()->post_ops_.entry_[eltwise_idx].eltwise.beta
                    : 0.0f;
        }

        alg_kind_t eltwise_alg_kind() const {
            const int eltwise_idx
                    = attr()->post_ops_.find(primitive_kind::eltwise);
            return with_eltwise()
                    ? attr()->post_ops_.entry_[eltwise_idx].eltwise.alg
                    : dnnl_alg_kind_undef;
        }

        size_t dyn_offset_a = 0;
        size_t dyn_offset_b = 0;
        size_t dyn_offset_c = 0;
        size_t dyn_offset_co = 0;
    };

    status_t init() override {
        auto *compute_engine
                = utils::downcast<compute::compute_engine_t *>(engine());
        auto *dev_info = utils::downcast<const ocl_gpu_device_info_t *>(
                compute_engine->device_info());

        eu_count_ = dev_info->eu_count();
        hw_threads_ = dev_info->hw_threads();

        gemm_type_ = get_gemm_type();

        switch (gemm_type_) {
            case type::no_copy: return init_nocopy();
        }

        return status::invalid_arguments;
    }

    status_t init_nocopy() {
        const char *kernel_name = nullptr;

        //compute kernel
        switch (c_type) {
            case data_type::s32:
                kernel_name = "gen9_gemm_compute_x8x8s32";
                break;
            default: return status::unimplemented;
        }

        auto *compute_engine
                = utils::downcast<compute::compute_engine_t *>(engine());
        compute::kernel_ctx_t kernel_ctx;

        memory_storage_t *temp_buf_ptr;
        this->engine()->create_memory_storage(
                &temp_buf_ptr, pd()->desc()->m * pd()->desc()->n * sizeof(int));
        temp_buf_.reset(temp_buf_ptr);

        int cmask = 0;
        pd()->attr()->zero_points_.get(DNNL_ARG_DST, nullptr, &cmask, nullptr);
        bool fixed_c = (0 == cmask);
        bool column_c = (1 << 0 == cmask);
        bool row_c = (1 << 1 == cmask);

        auto status = jit_gen9_gemm_x8x8s32_kernel<a_type, b_type,
                c_type>::init_const_def(kernel_ctx, pd()->desc()->transa,
                pd()->desc()->transb, fixed_c, column_c, row_c,
                pd()->with_eltwise(), pd()->eltwise_alg_kind());
        if (status != status::success) return status;

        compute_engine->create_kernel(
                &compute_x8x8s32_kernel_, kernel_name, kernel_ctx);
        if (!compute_x8x8s32_kernel_) return status::runtime_error;

        //scale kernel
        kernel_name = "gen9_gemm_scale_x8x8s32";

        status = jit_gen9_gemm_scale_x8x8s32_kernel<a_type, b_type,
                c_type>::init_const_def(kernel_ctx, pd()->with_eltwise(),
                pd()->eltwise_alg_kind());
        if (status != status::success) return status;

        compute_engine->create_kernel(
                &scale_x8x8s32_kernel_, kernel_name, kernel_ctx);
        if (!scale_x8x8s32_kernel_) return status::runtime_error;

        return status::success;
    }

    jit_gen9_gemm_x8x8s32_t(const pd_t *apd) : ocl_gemm_t(apd) {}

    virtual status_t execute(const exec_ctx_t &ctx) const override;
    virtual status_t execute(const gemm_exec_ctx_t &ctx) const override;

private:
    status_t launch_x8x8s32(compute::compute_stream_t *s,
            const memory_storage_t &a, const memory_storage_t &b,
            const memory_storage_t &c, int64_t offset_a, int64_t offset_b,
            int64_t offset_c, int64_t lda, int64_t ldb, int64_t ldc, int64_t m,
            int64_t n, int64_t k, int64_t beta, ao_t ao, bo_t bo,
            const memory_storage_t &co, int64_t offset_co, bool apply_co,
            bool apply_eltwise, c_t eltwise_alpha, c_t eltwise_beta) const;

    status_t launch_scale_x8x8s32(compute::compute_stream_t *s,
            const memory_storage_t &c_temp, const memory_storage_t &c,
            char offsetc, int64_t offset_c, int64_t m, int64_t n, int64_t ldc,
            float alpha, float beta, const memory_storage_t &co,
            int64_t offset_co, bool alpha_is_zero, bool apply_eltwise,
            c_t eltwise_alpha, c_t eltwise_beta) const;

    virtual status_t execute_standard(const gemm_exec_ctx_t &ctx) const;

    compute::kernel_t compute_x8x8s32_kernel_;
    compute::kernel_t scale_x8x8s32_kernel_;

    std::unique_ptr<memory_storage_t> temp_buf_;

    type gemm_type_ = type::no_copy;
    int hw_threads_ = 0;
    int eu_count_ = 0;

    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }

    type get_gemm_type() const { return type::no_copy; }
};

} // namespace ocl
} // namespace impl
} // namespace dnnl
#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
