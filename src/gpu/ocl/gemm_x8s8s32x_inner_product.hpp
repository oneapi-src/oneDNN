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

#ifndef GPU_OCL_GEMM_X8S8S32X_INNER_PRODUCT_HPP
#define GPU_OCL_GEMM_X8S8S32X_INNER_PRODUCT_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/gemm_types.hpp"
#include "common/primitive.hpp"
#include "common/primitive_iterator.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/gemm/gpu_gemm.hpp"
#include "gpu/gemm/gpu_gemm_utils.hpp"
#include "gpu/gpu_inner_product_pd.hpp"
#include "gpu/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

namespace {
// XXX: Can be unified and used across all primitives
inline status_t create_gemm_x8s8s32x_pd(
        std::unique_ptr<primitive_desc_t> &gemm_pd, engine_t *engine,
        transpose_t transa, transpose_t transb, int m, int n, int k, int lda,
        int ldb, int ldc, data_type_t a_dt, data_type_t b_dt, data_type_t c_dt,
        const primitive_attr_t &attr) {
    gemm_desc_t gemm_desc;
    gemm_desc.primitive_kind = primitive_kind::gemm;
    gemm_desc.transa = transa;
    gemm_desc.transb = transb;
    gemm_desc.batch = 1;
    gemm_desc.m = m;
    gemm_desc.n = n;
    gemm_desc.k = k;
    gemm_desc.lda = lda;
    gemm_desc.ldb = ldb;
    gemm_desc.ldc = ldc;
    gemm_desc.stride_a = lda;
    gemm_desc.stride_b = ldb;
    gemm_desc.stride_c = ldc;
    gemm_desc.a_type = a_dt;
    gemm_desc.b_type = b_dt;
    gemm_desc.c_type = c_dt;
    gemm_desc.acc_type = c_dt;

    primitive_attr_t gemm_attr = attr;
    gemm_attr.set_scratchpad_mode(scratchpad_mode::user);
    dnnl_primitive_desc_iterator it(
            engine, (op_desc_t *)&gemm_desc, &gemm_attr, nullptr);
    ++it;
    gemm_pd.reset(it.fetch_once());
    if (!gemm_pd) return status::unimplemented;
    return status::success;
}
} // namespace

struct gemm_x8s8s32x_inner_product_fwd_t : public primitive_t {
    struct pd_t : public gpu_inner_product_fwd_pd_t {
        pd_t(const inner_product_desc_t *adesc, const primitive_attr_t *attr,
                const inner_product_fwd_pd_t *hint_fwd_pd)
            : gpu_inner_product_fwd_pd_t(adesc, attr, hint_fwd_pd) {}

        pd_t(const pd_t &rhs) : gpu_inner_product_fwd_pd_t(rhs) {
            gemm_pd_.reset(rhs.gemm_pd_->clone());
            ip_scratchpad_md_ = rhs.ip_scratchpad_md_;
            scales_md_ = rhs.scales_md_;
        }

        ~pd_t() = default;

        pd_t &operator=(const pd_t &rhs) {
            DNNL_SHORT_CIRCUIT_SELF_ASSIGN(rhs);
            gemm_pd_.reset(rhs.gemm_pd_->clone());
            ip_scratchpad_md_ = rhs.ip_scratchpad_md_;
            scales_md_ = rhs.scales_md_;
            return *this;
        }

        DECLARE_COMMON_PD_T(
                "ocl:gemm_x8s8s32x", gemm_x8s8s32x_inner_product_fwd_t);

        status_t init(engine_t *engine) {
            using namespace status;
            using namespace utils;
            using namespace data_type;
            using namespace primitive_kind;

            assert(engine->kind() == engine_kind::gpu);

            primitive_attr_t::skip_mask_t attr_skip_mask
                    = primitive_attr_t::skip_mask_t::oscale
                    | primitive_attr_t::skip_mask_t::post_ops;
            bool ok = is_fwd() && set_default_params() == success
                    && one_of(src_md()->data_type, s8, u8)
                    && weights_md()->data_type == s8
                    && IMPLICATION(with_bias(),
                            one_of(weights_md(1)->data_type, s8, u8, f32, s32))
                    && one_of(dst_md()->data_type, u8, s8, f32, s32)
                    && dense_consitency_check(src_md(), weights_md(), dst_md())
                    && dense_gemm_consitency_check(
                            src_md(), weights_md(), dst_md())
                    && attr()->has_default_values(attr_skip_mask)
                    && IMPLICATION(!attr()->output_scales_.has_default_values(),
                            attr()->scratchpad_mode_ == scratchpad_mode::library
                                    && one_of(attr()->output_scales_.mask_, 0,
                                            1 << 1))
                    && IMPLICATION(with_eltwise() && with_sum(),
                            post_op_idx(sum) == 0 && post_op_idx(eltwise) == 1);
            if (!ok) return unimplemented;

            // XXX: Empty attributes increase chances of creating a gemm
            // primitive. Ideally gemm should be created multiple times with
            // different attr combinations, but this mechanism might be tricky.
            // Current implementation computes attr - related things in the post
            // process kernel.
            primitive_attr_t gemm_attr;

            const auto &wmd = *this->weights_md();
            bool wei_tr = wmd.format_desc.blocking.strides[0] != 1;

            const int mb = this->MB();
            const int oc = this->OC();
            const int ic_total = this->IC_total_padded();

            bool gemm_ok = status::success
                    == create_gemm_x8s8s32x_pd(gemm_pd_, engine,
                            wei_tr ? transpose::trans : transpose::notrans,
                            transpose::notrans, oc, mb, ic_total,
                            wei_tr ? ic_total : oc, ic_total, oc,
                            weights_md()->data_type, src_md()->data_type, s32,
                            gemm_attr);
            if (!gemm_ok) return status::unimplemented;

            status_t scratchpad_status = init_ip_scratchpad_md();
            if (scratchpad_status != success) return scratchpad_status;

            status_t scales_status = init_scales_md();
            if (scales_status != success) return scales_status;

            return success;
        }

        bool with_scales() const {
            return !attr()->output_scales_.has_default_values();
        }
        bool with_post_process() const {
            return use_scratchpad() || dst_md()->data_type != data_type::s32
                    || with_bias() || with_scales() || with_eltwise()
                    || with_sum();
        }
        bool use_scratchpad() const { return use_temp_dst(); }

        bool use_temp_dst() const {
            using namespace data_type;
            return !utils::one_of(dst_md()->data_type, s32, f32) || with_sum();
        }
        const memory_desc_t *ip_scratchpad_md() const {
            return &ip_scratchpad_md_;
        }
        const memory_desc_t *scales_md() const { return &scales_md_; }
        // post ops
        int post_op_idx(primitive_kind_t kind) const {
            return attr()->post_ops_.find(kind);
        }
        bool with_eltwise() const {
            return post_op_idx(primitive_kind::eltwise) != -1;
        }
        bool with_sum() const { return post_op_idx(primitive_kind::sum) != -1; }
        alg_kind_t eltwise_algorithm() const {
            const int eltwise_idx = post_op_idx(primitive_kind::eltwise);
            return eltwise_idx != -1
                    ? attr()->post_ops_.entry_[eltwise_idx].eltwise.alg
                    : alg_kind::undef;
        }
        float eltwise_alpha() const {
            const int eltwise_idx = post_op_idx(primitive_kind::eltwise);
            return eltwise_idx != -1
                    ? attr()->post_ops_.entry_[eltwise_idx].eltwise.alpha
                    : 1.0f;
        }
        float eltwise_beta() const {
            const int eltwise_idx = post_op_idx(primitive_kind::eltwise);
            return eltwise_idx != -1
                    ? attr()->post_ops_.entry_[eltwise_idx].eltwise.beta
                    : 0.0f;
        }
        float eltwise_scale() const {
            const int eltwise_idx = post_op_idx(primitive_kind::eltwise);
            return eltwise_idx != -1
                    ? attr()->post_ops_.entry_[eltwise_idx].eltwise.scale
                    : 1.0f;
        }
        float sum_scale() const {
            const int sum_idx = post_op_idx(primitive_kind::sum);
            return sum_idx != -1 ? attr()->post_ops_.entry_[sum_idx].sum.scale
                                 : 0.0f;
        }

        status_t init_ip_scratchpad_md() {
            if (use_scratchpad()) {
                ip_scratchpad_md_.data_type = data_type::s32;
                ip_scratchpad_md_.ndims = 1;
                ip_scratchpad_md_.dims[0] = 0;

                if (use_temp_dst()) {
                    const size_t temp_dst_size = MB() * OC();
                    ip_scratchpad_md_.dims[0] += temp_dst_size;
                }
                return memory_desc_init_by_tag(
                        ip_scratchpad_md_, format_tag::x);
            }

            return status::success;
        }

        status_t init_scales_md() {
            if (with_scales()) {
                scales_md_.data_type = data_type::f32;
                scales_md_.ndims = 1;
                scales_md_.dims[0] = attr()->output_scales_.count_;
                return memory_desc_init_by_tag(scales_md_, format_tag::x);
            }

            return status::success;
        }

        std::unique_ptr<primitive_desc_t> gemm_pd_;

        memory_desc_t scales_md_;
        memory_desc_t ip_scratchpad_md_;

    private:
        void init_scratchpad() {
            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.book(memory_tracking::names::key_nested,
                    gemm_pd_->scratchpad_registry().size());
        }
    };

    status_t init(engine_t *engine) override {
        status_t gemm_status = pd()->gemm_pd_->create_primitive(gemm_, engine);
        if (gemm_status != status::success) return gemm_status;

        const size_t mb = pd()->MB();
        const size_t oc = pd()->OC();

        // Prepare scratchpad memory storage
        if (pd()->use_scratchpad()) {
            memory_desc_wrapper scratchpad_mdw(pd()->ip_scratchpad_md());
            const size_t scratchpad_size = scratchpad_mdw.size();
            memory_storage_t *scratchpad_ptr = nullptr;
            engine->create_memory_storage(&scratchpad_ptr, scratchpad_size);
            scratchpad_.reset(scratchpad_ptr);
            if (!scratchpad_) return status::runtime_error;
        }

        // Prepare output scales defined by attributes. Doesn't introduce
        // primitive state, because it is a constant memory -- will not be
        // changed during execution.
        if (pd()->with_scales()) {
            memory_desc_wrapper scales_mdw(pd()->scales_md());
            scales_mem_.reset(new memory_t(
                    engine, pd()->scales_md(), memory_flags_t::alloc, nullptr));
            void *scales_ptr = nullptr;
            status_t status
                    = scales_mem_->memory_storage()->map_data(&scales_ptr);
            if (status != status::success) return status;
            utils::array_copy((float *)scales_ptr,
                    pd()->attr()->output_scales_.scales_,
                    pd()->attr()->output_scales_.count_);
            status = scales_mem_->memory_storage()->unmap_data(scales_ptr);
            if (status != status::success) return status;
        }

        // Prepare post process kernel
        if (pd()->with_post_process()) {
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);
            compute::kernel_ctx_t kernel_ctx;

            kernel_ctx.define_int("MB", mb);
            kernel_ctx.define_int("OC", oc);

            kernel_ctx.set_data_type(data_type::f32);
            def_data_type(kernel_ctx, data_type::s32, "SRC");
            def_data_type(kernel_ctx, data_type::f32, "ACC");
            def_data_type(kernel_ctx,
                    pd()->with_bias() ? pd()->weights_md(1)->data_type
                                      : data_type::f32,
                    "BIAS");
            def_data_type(kernel_ctx, pd()->dst_md()->data_type, "DST");

            kernel_ctx.define_int("USE_TEMP_DST", pd()->use_temp_dst());

            kernel_ctx.define_int("WITH_BIAS", pd()->with_bias());

            // output scales
            kernel_ctx.define_int("WITH_SCALES", pd()->with_scales());

            const auto scales_mask = pd()->attr()->output_scales_.mask_;
            kernel_ctx.define_int(
                    "SCALES_COMMON", pd()->with_scales() && (scales_mask == 0));
            kernel_ctx.define_int("SCALES_PER_OC",
                    pd()->with_scales() && (scales_mask == (1 << 1)));
            // post ops
            kernel_ctx.define_int("WITH_ELTWISE", pd()->with_eltwise());
            if (pd()->with_eltwise()) {
                def_postops(kernel_ctx, pd()->eltwise_algorithm());
            }
            kernel_ctx.define_int("WITH_SUM", pd()->with_sum());

            compute_engine->create_kernel(&post_process_kernel_,
                    "gemm_x8s8s32x_inner_product_post_process", kernel_ctx);
            if (!post_process_kernel_) return status::runtime_error;
        }

        return status::success;
    }

    gemm_x8s8s32x_inner_product_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    std::shared_ptr<primitive_t> gemm_;
    compute::kernel_t post_process_kernel_;
    std::unique_ptr<memory_t> scales_mem_;
    std::unique_ptr<memory_storage_t> scratchpad_;
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
