/*******************************************************************************
* Copyright 2020-2024 Intel Corporation
* Copyright 2020 Codeplay Software Limited
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

#ifndef GPU_NVIDIA_CUDNN_REORDER_LT_HPP
#define GPU_NVIDIA_CUDNN_REORDER_LT_HPP

#include "common/memory_desc_wrapper.hpp"
#include "common/reorder.hpp"
#include "common/reorder_pd.hpp"
#include "gpu/gpu_primitive.hpp"
#include "gpu/gpu_reorder_pd.hpp"
#include "gpu/nvidia/cudnn_reorder_lt_impl.hpp"
#include "gpu/nvidia/engine.hpp"
#include "gpu/nvidia/sycl_cuda_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

struct cudnn_reorder_lt_t : public gpu::primitive_t {
    using gpu::primitive_t::primitive_t;

    struct pd_t : public gpu_reorder_pd_t {
        using gpu_reorder_pd_t::gpu_reorder_pd_t;

        DECLARE_COMMON_PD_T("cuda:cublaslt:any", cudnn_reorder_lt_t);

        // Function to verify data and memory format
        bool valid_data_n_mem_format(impl::engine_t *engine) {
            auto src_dt_ = src_md()->data_type;
            auto dst_dt_ = dst_md()->data_type;
            bool ok = utils::one_of(
                    src_dt_, data_type::f32, data_type::s8, data_type::s32);
            ok = ok
                    && utils::one_of(dst_dt_, data_type::f32, data_type::s8,
                            data_type::s32);
            // to ampere blocked
            ok = ok
                    && IMPLICATION(utils::one_of(src_dt_, data_type::s8,
                                           data_type::s32, data_type::f32),
                            dst_dt_ == data_type::s8);
            // from ampere blocked
            ok = ok
                    && IMPLICATION(src_dt_ == data_type::s8,
                            utils::one_of(dst_dt_, data_type::f32,
                                    data_type::s32, data_type::s8));

            src_float_ = src_dt_ == data_type::f32;
            dst_float_ = dst_dt_ == data_type::f32;

            if (!ok) return ok;

            memory_desc_wrapper src_wrap(*src_md());
            memory_desc_wrapper dst_wrap(*dst_md());

            ok = ok && src_wrap.ndims() <= 3 && dst_wrap.ndims() <= 3;

            // Only support transforming from plain to blocked format and vice-versa.
            ok = ok && IMPLICATION(src_wrap.is_plain(), !dst_wrap.is_plain());
            ok = ok && IMPLICATION(dst_wrap.is_plain(), !src_wrap.is_plain());
            ok = ok && IMPLICATION(!src_wrap.is_plain(), dst_wrap.is_plain());
            ok = ok && IMPLICATION(!dst_wrap.is_plain(), src_wrap.is_plain());
            ok = ok && IMPLICATION(src_float_, src_wrap.is_plain());
            ok = ok && IMPLICATION(dst_float_, dst_wrap.is_plain());

            if (!ok) return ok;

            auto check_tag = [&](const memory_desc_wrapper &wrap,
                                     bool &transpose, format_kind_t &kind) {
                kind = format_kind_t::dnnl_blocked;
                if (wrap.is_cublaslt_blocked_desc()) {
                    transpose = false;
                    kind = format_kind::cublaslt_blocked;
                    return format_tag::undef;
                }
                if (wrap.is_plain()) {
                    auto tag = wrap.matches_one_of_tag(
                            format_tag::ab, format_tag::abc);
                    if (tag != format_tag::undef) {
                        transpose = false;
                        return tag;
                    }
                    tag = wrap.matches_one_of_tag(
                            format_tag::ba, format_tag::acb);
                    if (tag != format_tag::undef) {
                        transpose = true;
                        return tag;
                    }
                }
                return dnnl_format_tag_undef;
            };
            ok = ok && src_wrap.ndims() == dst_wrap.ndims();
            format_kind_t kind;

            src_tag_ = check_tag(src_wrap, src_trans_, kind);
            ok = IMPLICATION(
                    kind == dnnl_blocked, src_tag_ != dnnl_format_tag_undef);

            dst_tag_ = check_tag(dst_wrap, dst_trans_, kind);
            ok = IMPLICATION(
                    kind == dnnl_blocked, dst_tag_ != dnnl_format_tag_undef);
            return ok;
        }

        bool scales_ok() const {
            const auto &scales = attr()->scales_;
            const auto &supported_args = {DNNL_ARG_FROM, DNNL_ARG_TO};
            if (!scales.has_default_values(supported_args)) return false;
            // cuDNN does not support scaling per dimension.
            for (auto arg : supported_args)
                if (scales.get(arg).mask_ != 0) return false;
            return true;
        }

        bool post_ops_ok() const {
            // only sum post-op is supported
            const auto &p = attr()->post_ops_;
            return p.len() == 0 || (p.len() == 1 && p.entry_[0].is_sum(false));
        }

        status_t init(impl::engine_t *engine, impl::engine_t *src_engine,
                impl::engine_t *dst_engine) {
            const auto attr_skip_mask
                    = primitive_attr_t::skip_mask_t::scales_runtime
                    | primitive_attr_t::skip_mask_t::post_ops;
            bool ok = engine == dst_engine && src_engine == dst_engine
                    && valid_data_n_mem_format(engine)
                    && attr()->has_default_values(attr_skip_mask) && scales_ok()
                    && post_ops_ok();
            if (!ok) return status::unimplemented;

            primitive_attr_t r_attr;
            int mask = 0;
            bool is_set = false;
            auto src = DNNL_ARG_DST;
            auto dst = DNNL_ARG_SRC;
            if (src_float_) {
                src_scratch_md_ = *src_md();
                dst_scratch_md_ = create_temp_md(src_scratch_md_);
                this->src_md_ = dst_scratch_md_;
            } else if (dst_float_) {
                src_scratch_md_ = create_temp_md(dst_scratch_md_);
                dst_scratch_md_ = *dst_md();
            }
            attr()->scales_.get(src, &mask, &is_set);
            if (is_set) { r_attr.scales_.set(src, mask); }

            attr()->scales_.get(dst, &mask, &is_set);
            if (is_set) { r_attr.scales_.set(dst, mask); }

            status_t generic_ok = reorder_primitive_desc_create(
                    generic_reorder_desc_, engine, &src_scratch_md_,
                    &dst_scratch_md_, &r_attr);
            ok = ok && (generic_ok == status::success);

            if (!ok) return status::unimplemented;
            init_scratchpad();

            return dnnl_success;
        }

        void init_scratchpad() {
            memory_desc_wrapper src_wrap(src_md());
            memory_desc_wrapper dst_wrap(dst_md());

            int dims[DNNL_MAX_NDIMS];

            if (!src_float_) {
                convert_dims(dst_md()->padded_dims, dims, dst_md()->ndims);
            } else {
                convert_dims(src_md()->padded_dims, dims, src_md()->ndims);
            }

            const bool is_batched = src_wrap.ndims() > 2 && dst_wrap.dims()[0];

            uint64_t row = dims[is_batched + 0];
            uint64_t col = dims[is_batched + 1];

            uint64_t blocked_ld
                    = ceildiv(row, static_cast<uint64_t>(32)) * 32 * 32;
            auto stride_b_blocked_
                    = ceildiv(col, static_cast<uint64_t>(32)) * blocked_ld;

            uint64_t src_scratch_size = 0;
            uint64_t dst_scratch_size = 0;

            if (src_float_) {
                dst_scratch_size
                        = stride_b_blocked_ * src_wrap.data_type_size() * 32;
                src_scratch_size
                        = src_wrap.nelems() * src_wrap.data_type_size();
            } else {
                src_scratch_size
                        = stride_b_blocked_ * src_wrap.data_type_size() * 32;
                dst_scratch_size
                        = dst_wrap.nelems() * dst_wrap.data_type_size();
            }

            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.book(memory_tracking::names::key_nested_multiple,
                    generic_reorder_desc_->scratchpad_registry());
            if (src_scratch_size) {
                scratchpad.book(
                        memory_tracking::names::key_reorder_cublaslt_src_float,
                        src_scratch_size, 1, 256);
            }
            if (dst_scratch_size) {
                scratchpad.book(
                        memory_tracking::names::key_reorder_cublaslt_dst_float,
                        dst_scratch_size, 1, 256);
            }
        }

        // Needed for internal reorder to convert src/dst from f32 to s8
        memory_desc_t create_temp_md(const memory_desc_t &md) {
            memory_desc_t temp;
            temp = md;
            temp.data_type = dnnl_s8;

            return temp;
        }

        bool src_trans_ = false;
        bool dst_trans_ = false;
        bool src_float_ = false;
        bool dst_float_ = false;

        data_type_t src_dt_;
        data_type_t dst_dt_;
        memory_desc_t src_scratch_md_;
        memory_desc_t dst_scratch_md_;
        format_tag_t src_tag_;
        format_tag_t dst_tag_;
        std::shared_ptr<impl::primitive_desc_t> generic_reorder_desc_;

    private:
        DECLARE_GPU_REORDER_CREATE();
    };

    status_t init(impl::engine_t *engine) override {
        cublaslt_reorder_.reset(new cublaslt_reorder_t);
        CHECK(cublaslt_reorder_->init((reorder_pd_t *)pd()));
        if ((pd()->src_float_ || pd()->dst_float_)) {
            CHECK(create_nested_primitive(
                    generic_reorder_, pd()->generic_reorder_desc_, engine));
        }

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override;
    status_t execute_internal_reorder(const exec_ctx_t &ctx,
            const memory_arg_t &src, const memory_arg_t &dst,
            const memory_arg_t *src_scales,
            const memory_arg_t *dst_scales) const;

private:
    std::shared_ptr<impl::primitive_t> generic_reorder_;
    std::shared_ptr<cublaslt_reorder_t> cublaslt_reorder_;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
