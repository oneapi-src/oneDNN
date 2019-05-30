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

#ifndef JIT_GEN9_GEMM_HPP
#define JIT_GEN9_GEMM_HPP

#include <assert.h>
#include <memory>

#include "common/c_types_map.hpp"
#include "common/gemm_utils.hpp"
#include "ocl/cl_engine.hpp"
#include "ocl/jit_gen9_gemm_kernel.hpp"
#include "ocl/ocl_gemm_pd.hpp"
#include "ocl/ocl_stream.hpp"
#include "ocl/ocl_utils.hpp"

extern const char *gen9_gemm_compute_kernel;
extern const char *gen9_gemm_copy_kernel;
extern const char *gen9_gemm_beta_kernel;
extern const char *gen9_gemm_nocopy_kernel;

namespace mkldnn {
namespace impl {
namespace ocl {

template <impl::data_type_t a_type, impl::data_type_t b_type = a_type,
        impl::data_type_t c_type = a_type>
struct jit_gen9_gemm_t : public primitive_t {
    using c_t = typename prec_traits<c_type>::type;

    struct pd_t : public ocl_gemm_pd_t {
        using hint_class = void;

        pd_t(engine_t *engine, const gemm_desc_t *adesc,
                const primitive_attr_t *attr, const hint_class *)
            : ocl_gemm_pd_t(engine, adesc, attr) {}

        DECLARE_COMMON_PD_T("ocl:gemm:any", jit_gen9_gemm_t);

        status_t init() {
            using namespace prop_kind;
            using namespace data_type;

            assert(this->engine()->kind() == engine_kind::gpu);
            auto *cl_engine = utils::downcast<cl_engine_t *>(engine());

            bool ok = true
                    && desc()->a_type == a_type
                    && desc()->b_type == b_type
                    && desc()->c_type == c_type
                    && cl_engine->mayiuse(cl_device_ext_t::intel_subgroups)
                    && IMPLICATION(c_type == f16,
                               true
                                       && cl_engine->mayiuse(
                                                  cl_device_ext_t::khr_fp16)
                                       && cl_engine->mayiuse(cl_device_ext_t::
                                                  intel_subgroups_short));
            if (!ok)
                return status::unimplemented;

            return status::success;
        }

        bool with_eltwise() const {
            return attr()->post_ops_.find(primitive_kind::eltwise) != -1;
        }

        float eltwise_alpha() const {
            const int eltwise_idx =
                attr()->post_ops_.find(primitive_kind::eltwise);
            return with_eltwise()
                ? attr()->post_ops_.entry_[eltwise_idx].eltwise.alpha
                : 1.0f;
        }

        float eltwise_beta() const {
            const int eltwise_idx =
                attr()->post_ops_.find(primitive_kind::eltwise);
            return with_eltwise()
                ? attr()->post_ops_.entry_[eltwise_idx].eltwise.beta
                : 0.0f;
        }

        alg_kind_t eltwise_alg_kind() const {
            const int eltwise_idx =
                attr()->post_ops_.find(primitive_kind::eltwise);
            return with_eltwise()
                ? attr()->post_ops_.entry_[eltwise_idx].eltwise.alg
                : mkldnn_alg_kind_undef;
        }

        size_t dyn_offset_a = 0;
        size_t dyn_offset_b = 0;
        size_t dyn_offset_c = 0;
    };

    status_t init() override {
        nocopy_ = use_nocopy();
        return nocopy_ ? init_nocopy() : init_copy_based();
    }

    status_t init_copy_based() {
        memory_storage_t *temp_buf_ptr;
        this->engine()->create_memory_storage(&temp_buf_ptr, 128 << 20);
        temp_buf_.reset(temp_buf_ptr);

        for (bool beta0 : { false, true }) {
            if (beta0 && pd()->desc()->beta != 0)
                continue;

            auto jit = ocl_jit_t(gen9_gemm_compute_kernel);

            auto status = jit_gen9_gemm_compute_kernel<c_type>::init_const_def(
                jit, beta0);
            if (status != status::success)
                return status;

            status = jit.build(engine());
            if (status != status::success)
                return status;

            compute_kernel_[beta0] = jit.get_kernel("gen9_gemm_compute_kernel");
            if (!compute_kernel_[beta0])
                return status::runtime_error;
        }

        for (bool outer : { false, true }) {
            auto trans = !outer ? !pd()->desc()->transa : pd()->desc()->transb;
            auto jit = ocl_jit_t(gen9_gemm_copy_kernel);

            auto status = jit_gen9_gemm_copy_kernel<c_type>::init_const_def(
                jit, outer, trans);
            if (status != status::success)
                return status;

            status = jit.build(engine());
            if (status != status::success)
                return status;

            copy_kernel_[outer][trans]
                = jit.get_kernel("gen9_gemm_copy_kernel");
            if (!copy_kernel_[outer][trans])
                return status::runtime_error;
        }

        auto jit = ocl_jit_t(gen9_gemm_beta_kernel);

        auto status = jit_gen9_gemm_beta_kernel<c_type>::init_const_def(jit);
        if (status != status::success)
            return status;

        status = jit.build(engine());
        if (status != status::success)
            return status;

        beta_kernel_ = jit.get_kernel("gen9_gemm_beta_kernel");
        if (!beta_kernel_)
            return status::runtime_error;

        return status::success;
    }

    status_t init_nocopy() {
        if (c_type != data_type::f32)
            return status::unimplemented;

        auto jit = ocl_jit_t(gen9_gemm_nocopy_kernel);

        auto status = jit_gen9_gemm_nocopy_kernel<c_type>::init_const_def(jit,
            pd()->desc()->transa, pd()->desc()->transb, pd()->with_eltwise(),
            pd()->eltwise_alg_kind());
        if (status != status::success)
            return status;

        status = jit.build(engine());
        if (status != status::success)
            return status;

        nocopy_kernel_ = jit.get_kernel("gen9_gemm_nocopy_kernel");
        if (!nocopy_kernel_)
            return status::runtime_error;

        return status::success;
    }

    jit_gen9_gemm_t(const pd_t *apd) : primitive_t(apd) {}

    virtual status_t execute(const exec_ctx_t &ctx) const override;

private:
    status_t launch_beta(stream_t *s, int64_t m, int64_t n, c_t alpha,
            const memory_storage_t &a, int64_t offseta, int64_t lda) const;

    status_t launch_copy(stream_t *s, int64_t m, int64_t n,
            const memory_storage_t &a, int64_t offseta, int64_t lda, c_t alpha,
            const memory_storage_t &b, int64_t offsetb, bool outer,
            bool trans) const;

    status_t launch_compute(stream_t *s, int64_t m, int64_t n, int64_t k,
            const memory_storage_t &base, int32_t offset_a, int32_t offset_b,
            const memory_storage_t &c, int64_t offset_c, int64_t ldc,
            bool beta0) const;

    status_t launch_nocopy(stream_t *s, const memory_storage_t &a,
        const memory_storage_t &b, const memory_storage_t &c, int64_t offset_a,
        int64_t offset_b, int64_t offset_c, int32_t lda, int32_t ldb,
        int32_t ldc, int32_t m, int32_t n, int32_t k, float alpha, float beta,
        int last_k_block, float eltwise_alpha, float eltwise_beta) const;

    ocl_kernel_t compute_kernel_[2];
    ocl_kernel_t copy_kernel_[2][2];
    ocl_kernel_t beta_kernel_;
    ocl_kernel_t nocopy_kernel_;

    bool nocopy_ = false;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }

    bool use_nocopy() const {
        return (pd()->with_eltwise() || ((c_type == data_type::f32) &&
            ((pd()->desc()->transa == mkldnn_notrans)
                || (pd()->desc()->transb == mkldnn_trans))));
    }

    std::unique_ptr<memory_storage_t> temp_buf_;
};

}
}
}
#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
