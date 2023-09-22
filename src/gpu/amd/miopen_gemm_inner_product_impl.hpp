/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
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

#ifndef GPU_AMD_MIOPEN_GEMM_INNER_PRODUCT_IMPL_HPP
#define GPU_AMD_MIOPEN_GEMM_INNER_PRODUCT_IMPL_HPP

#include <miopen/miopen.h>
#include <rocblas/rocblas.h>

#include "common/type_helpers.hpp"
#include "gpu/amd/miopen_inner_product_impl.hpp"
#include "gpu/amd/sycl_hip_engine.hpp"
#include "gpu/amd/sycl_hip_scoped_context.hpp"
#include "gpu/amd/sycl_hip_stream.hpp"
#include "gpu/amd/sycl_hip_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

struct miopen_gemm_inner_product_base_t {
protected:
    int m_, n_, k_, lda_, ldb_, ldc_;
    rocblas_operation trans_a_, trans_b_, trans_c_;
    rocblas_datatype a_type_, b_type_, c_type_,
            compute_type_ = rocblas_datatype_f32_r;
    rocblas_gemm_algo algo_ = rocblas_gemm_algo_standard;

    int32_t solution_index = 0;
    uint32_t flags = 0;
    float filter_alpha_ = 1, filter_beta_ = 0;

    miopenTensorDescriptor_t current_filter_desc_, transform_filter_desc_;

    status_t get_rocblas_data_type(const miopenDataType_t &miopen_dt,
            rocblas_datatype &blas_dt) const {
        switch (miopen_dt) {
            case miopenFloat:
                blas_dt = rocblas_datatype_f32_r;
                return status::success;
            case miopenHalf:
                blas_dt = rocblas_datatype_f16_r;
                return status::success;
            case miopenInt8:
                blas_dt = rocblas_datatype_i8_r;
                return status::success;
            case miopenInt8x4:
                blas_dt = rocblas_datatype_i8_r;
                return status::success;
            case miopenInt32:
                blas_dt = rocblas_datatype_i32_r;
                return status::success;
            case miopenBFloat16:
                blas_dt = rocblas_datatype_bf16_r;
                return status::success;
            default: return status::unimplemented;
        }
        return status::unimplemented;
    }

    void propagate_strides(int *strides, const int *dims,
            std::initializer_list<int> perm) const {
        int prev_p = -1;
        for (auto p : perm) {
            strides[p] = prev_p == -1 ? 1 : strides[prev_p] * dims[prev_p];
            prev_p = p;
        }
    }

    virtual status_t init_filter_transformation(
            miopenDataType_t filter_data_types, int filter_ndims,
            int *filter_dims, int *current_filter_strides,
            int *transform_filter_strides) {
        // Set a descriptor for the current filter.
        CHECK(create_and_set_tensor_descriptor(&current_filter_desc_,
                filter_data_types, filter_ndims, filter_dims,
                current_filter_strides));

        // Set a descriptor for the transform filter.
        CHECK(create_and_set_tensor_descriptor(&transform_filter_desc_,
                filter_data_types, filter_ndims, filter_dims,
                transform_filter_strides));
        return status::success;
    }

    virtual void set_filter_nchw(
            int filter_ndims, int *transform_filter_strides, int *filter_dims) {
        switch (filter_ndims) {
            case 4:
                return propagate_strides(
                        transform_filter_strides, filter_dims, {3, 2, 1, 0});
            case 5:
                return propagate_strides(
                        transform_filter_strides, filter_dims, {4, 3, 2, 1, 0});
            case 6:
                return propagate_strides(transform_filter_strides, filter_dims,
                        {5, 4, 3, 2, 1, 0});
        }
    }

    virtual void set_filter_nhwc(
            int filter_ndims, int *transform_filter_strides, int *filter_dims) {
        switch (filter_ndims) {
            case 4:
                return propagate_strides(
                        transform_filter_strides, filter_dims, {1, 3, 2, 0});
            case 5:
                return propagate_strides(
                        transform_filter_strides, filter_dims, {1, 4, 3, 2, 0});
            case 6:
                return propagate_strides(transform_filter_strides, filter_dims,
                        {1, 5, 4, 3, 2, 0});
        }
    }

    void set_filter_format(int filter_ndims, int *filter_dims,
            int *transform_filter_strides, miopenTensorLayout_t format) {
        if (format == miopenTensorNCHW) {
            set_filter_nchw(
                    filter_ndims, transform_filter_strides, filter_dims);
        } else {
            set_filter_nhwc(
                    filter_ndims, transform_filter_strides, filter_dims);
        }
    }
    void transform_filter(miopenHandle_t handle, void *current_filter,
            void *transform_filter) const {
        MIOPEN_EXECUTE_FUNC(miopenTransformTensor, handle, &filter_alpha_,
                current_filter_desc_, current_filter, &filter_beta_,
                transform_filter_desc_, transform_filter);
    }
    void undo_transform_filter(miopenHandle_t handle, void *transform_filter,
            void *current_filter) const {
        MIOPEN_EXECUTE_FUNC(miopenTransformTensor, handle, &filter_alpha_,
                transform_filter_desc_, transform_filter, &filter_beta_,
                current_filter_desc_, current_filter);
    }

    virtual ~miopen_gemm_inner_product_base_t() {
        if (current_filter_desc_) {
            MIOPEN_EXECUTE_FUNC_V(
                    miopenDestroyTensorDescriptor, current_filter_desc_);
        }
        if (transform_filter_desc_) {
            MIOPEN_EXECUTE_FUNC_V(
                    miopenDestroyTensorDescriptor, transform_filter_desc_);
        }
    }
};

struct miopen_gemm_inner_product_fwd_impl_t
    : public miopen_inner_product_fwd_base_t,
      public miopen_gemm_inner_product_base_t {
    miopenActivationDescriptor_t act_desc_;
    bool use_acc_dst_;
    miopenTensorDescriptor_t y_acc_desc_;
    bool need_reorder_;

    int alpha_s32 = 1, beta_s32 = 0;
    float alpha_f32 = 1.0f, beta_f32 = 0.0f;

    const void *get_gemm_alpha() const {
        switch (compute_type_) {
            case rocblas_datatype::rocblas_datatype_i32_r:
                return reinterpret_cast<const void *>(&alpha_s32);
            case rocblas_datatype::rocblas_datatype_f32_r:
                return reinterpret_cast<const void *>(&alpha_f32);
            default: assert(!"unknown compute type"); return nullptr;
        }
    }

    const void *get_gemm_beta() const {
        switch (compute_type_) {
            case rocblas_datatype::rocblas_datatype_i32_r:
                return reinterpret_cast<const void *>(&beta_s32);
            case rocblas_datatype::rocblas_datatype_f32_r:
                return reinterpret_cast<const void *>(&beta_f32);
            default: assert(!"unknown compute type"); return nullptr;
        }
    }

    bool ip_using_scratchpad() const override { return (use_acc_dst_ > 0); }
    virtual bool need_to_transform_filter() const override {
        return need_reorder_;
    }

    virtual status_t init(engine_t *, inner_product_pd_t *pd, bool with_relu,
            bool with_eltwise, bool with_sum, bool need_reorder) override {
        need_reorder_ = need_reorder;

        int ic = pd->IC_total_padded();
        int oc = pd->OC();
        int mb = pd->MB();

        bool wie_tr = (pd->weights_md()->format_desc.blocking.strides[0] != 1);
        bool src_tr = (pd->src_md()->format_desc.blocking.strides[0] == 1)
                && (ic > 1);

        CHECK(convert_data_type(pd->src_md(), &data_types_[io::src]));
        CHECK(convert_data_type(pd->weights_md(0), &data_types_[io::wei]));
        if (need_reorder) {
            miopenTensorLayout_t source_format;
            CHECK(get_format(pd->src_md(), source_format));
            ndims_ = pd->ndims() < 4 ? 4 : pd->ndims();
            get_4d_tensor_descriptor(
                    pd->weights_md(0), dims_[io::wei], strides_[io::wei]);
            set_filter_format(ndims_, dims_[io::wei], strides_[io::NUM_IO],
                    source_format);
            CHECK(init_filter_transformation(data_types_[io::wei], ndims_,
                    dims_[io::wei], strides_[io::wei], strides_[io::NUM_IO]));

            pd->scratchpad_registry().registrar().book(
                    memory_tracking::names::key_none,
                    memory_desc_wrapper(pd->weights_md(0)).size(), size_t(1));
            wie_tr = strides_[io::NUM_IO][0] != 1;
        }

        trans_a_
                = wie_tr ? rocblas_operation_transpose : rocblas_operation_none;
        trans_b_
                = src_tr ? rocblas_operation_transpose : rocblas_operation_none;

        n_ = mb;
        k_ = ic;
        m_ = oc;

        lda_ = wie_tr ? k_ : m_;
        ldb_ = src_tr ? n_ : k_;
        ldc_ = m_;

        with_bias_ = pd->with_bias();
        with_eltwise_ = with_eltwise || with_relu;
        with_relu_ = with_eltwise;

        output_scales_ = 1.0f;
        alpha_s32 = output_scales_;
        alpha_f32 = output_scales_;

        with_sum_ = with_sum;
        sum_scale_ = sum_scale(pd);
        beta_s32 = sum_scale_;
        beta_f32 = sum_scale_;

        ndims_ = 4;
        bool input_is_blocked
                = pd->src_md()->format_desc.blocking.inner_blks[0] == 4
                && pd->weights_md(0)->format_desc.blocking.inner_blks[0] == 4;
        if (input_is_blocked) {
            data_types_[io::src] = miopenInt8;
            data_types_[io::wei] = miopenInt8;
            data_types_[io::dst] = miopenInt8;
        } else {
            CHECK(convert_data_type(pd->dst_md(), &data_types_[io::dst]));
        }
        CHECK(get_rocblas_data_type(data_types_[io::wei], a_type_));
        CHECK(get_rocblas_data_type(data_types_[io::src], b_type_));

        get_4d_tensor_descriptor(
                pd->dst_md(), dims_[io::dst], strides_[io::dst]);

        CHECK(create_and_set_tensor_descriptor(&tensor_descs_[io::dst],
                data_types_[io::dst], ndims_, dims_[io::dst],
                strides_[io::dst]));

        if (with_bias_) {
            CHECK(convert_data_type(pd->weights_md(1), &data_types_[io::bia]));
            set_bias_dims(miopenTensorNCHW, ndims_, pd->OC());
            CHECK(create_and_set_tensor_descriptor(&tensor_descs_[io::bia],
                    data_types_[io::bia], ndims_, dims_[io::bia],
                    strides_[io::bia]));
        }

        y_acc_desc_ = tensor_descs_[io::dst];

        if (with_eltwise_) { CHECK(create_and_set_op_descriptor(pd)); }

        CHECK(convert_data_type(pd->dst_md(), &data_types_[io::dst]));
        CHECK(get_rocblas_data_type(data_types_[io::dst], c_type_));
        compute_type_ = (c_type_ == rocblas_datatype_bf16_r
                                || c_type_ == rocblas_datatype_f16_r)
                ? rocblas_datatype_f32_r
                : c_type_;

        return status::success;
    }

    void execute(miopenHandle_t miopen_handle, rocblas_handle blas_handle,
            const std::vector<void *> &args) const override {
        assert(args.size() == 8);
        auto x = args[0], w = args[1], b = args[2], y = args[3];

        auto y_dst = y;
        auto w_arg = w;
        const void *alpha = get_gemm_alpha();
        const void *beta = get_gemm_beta();
        float alpha2 = 0;

        if (need_reorder_) {
            void *transformed_w = args[5];
            transform_filter(miopen_handle, w, transformed_w);
            w_arg = transformed_w;
        }

        ROCBLAS_EXECUTE_FUNC(rocblas_gemm_ex, blas_handle, trans_a_, trans_b_,
                m_, n_, k_, alpha, w_arg, a_type_, lda_, x, b_type_, ldb_, beta,
                y_dst, c_type_, ldc_, y_dst, c_type_, ldc_, compute_type_,
                algo_, solution_index, flags);

        if (with_bias_) {
            MIOPEN_EXECUTE_FUNC(miopenOpTensor, miopen_handle,
                    miopenTensorOpAdd, &alpha_, y_acc_desc_, y_dst,
                    &output_scales_, tensor_descs_[io::bia], b, &alpha2,
                    y_acc_desc_, y_dst);
        }

        if (with_eltwise_) {
            MIOPEN_EXECUTE_FUNC(miopenActivationForward, miopen_handle,
                    act_desc_, &alpha_, tensor_descs_[io::dst], y, &beta_,
                    tensor_descs_[io::dst], y);
        }
    }

    status_t create_and_set_op_descriptor(const inner_product_pd_t *pd) {

        CHECK(MIOPEN_EXECUTE_FUNC_S(
                miopenCreateActivationDescriptor, &act_desc_));

        miopenActivationMode_t act_mode;
        switch (eltwise_algorithm_kind(pd)) {
            case alg_kind::eltwise_tanh: act_mode = miopenActivationTANH; break;
            case alg_kind::eltwise_elu: act_mode = miopenActivationELU; break;
            case alg_kind::eltwise_relu:
                act_mode = miopenActivationLEAKYRELU;
                break;
            case alg_kind::eltwise_logistic:
                act_mode = miopenActivationLOGISTIC;
                break;
            default: return status::unimplemented;
        }

        float activeAlpha;
        float activeBeta;
        float activeGamma;

        double ceiling = eltwise_alpha(pd);

        if (act_mode == miopenActivationMode_t::miopenActivationTANH)
            activeAlpha = activeBeta = 1;
        else if (act_mode == miopenActivationMode_t::miopenActivationELU)
            activeAlpha = ceiling;
        else if (act_mode == miopenActivationMode_t::miopenActivationLEAKYRELU)
            activeAlpha = ceiling;

        CHECK(MIOPEN_EXECUTE_FUNC_S(miopenSetActivationDescriptor, act_desc_,
                act_mode, activeAlpha, activeBeta, activeGamma));
        return status::success;
    }
};

struct miopen_gemm_inner_product_bwd_data_impl_t
    : public miopen_inner_product_impl_base_t,
      public miopen_gemm_inner_product_base_t {
    bool need_reorder_;

    virtual bool need_to_transform_filter() const override {
        return need_reorder_;
    }

    virtual status_t init(engine_t *, inner_product_pd_t *pd,
            bool /*with_relu*/, bool /*with_eltwise*/, bool /*with_sum */,
            bool need_reorder) override {
        need_reorder_ = need_reorder;

        int ic = pd->IC_total_padded();
        int oc = pd->OC();
        int mb = pd->MB();

        bool wie_tr = (pd->weights_md(0)->format_desc.blocking.strides[0] == 1);
        bool diff_src_tr
                = (pd->diff_src_md()->format_desc.blocking.strides[0] == 1)
                && (ic > 1);

        CHECK(convert_data_type(pd->diff_src_md(), &data_types_[io::src]));
        CHECK(convert_data_type(pd->weights_md(0), &data_types_[io::wei]));
        CHECK(convert_data_type(pd->diff_dst_md(), &data_types_[io::dst]));
        if (need_reorder) {
            miopenTensorLayout_t diff_source_format_;
            CHECK(get_format(pd->diff_src_md(), diff_source_format_));
            ndims_ = pd->ndims() < 4 ? 4 : pd->ndims();
            get_4d_tensor_descriptor(
                    pd->weights_md(0), dims_[io::wei], strides_[io::wei]);
            set_filter_format(ndims_, dims_[io::wei], strides_[NUM_IO],
                    diff_source_format_);
            CHECK(init_filter_transformation(data_types_[io::wei], ndims_,
                    dims_[io::wei], strides_[io::wei], strides_[NUM_IO]));

            pd->scratchpad_registry().registrar().book(
                    memory_tracking::names::key_none,
                    memory_desc_wrapper(pd->weights_md(0)).size(), size_t(1));
            wie_tr = strides_[NUM_IO][0] == 1;
        }

        trans_a_
                = wie_tr ? rocblas_operation_transpose : rocblas_operation_none;
        trans_b_ = rocblas_operation_none;
        trans_c_ = diff_src_tr ? rocblas_operation_transpose
                               : rocblas_operation_none;

        n_ = mb;
        k_ = oc;
        m_ = ic;

        lda_ = wie_tr ? k_ : m_;
        ldb_ = k_;
        ldc_ = m_;

        CHECK(get_rocblas_data_type(data_types_[io::wei], a_type_));
        CHECK(get_rocblas_data_type(data_types_[io::dst], b_type_));
        CHECK(get_rocblas_data_type(data_types_[io::src], c_type_));
        return status::success;
    }
    void execute(miopenHandle_t miopen_handle, rocblas_handle blas_handle,
            const std::vector<void *> &args) const override {
        assert(args.size() == 5);
        auto dx = args[0], w = args[1], dy = args[2];
        auto w_arg = w;

        if (need_reorder_) {
            void *transformed_w = args[4];
            transform_filter(miopen_handle, w, transformed_w);
            w_arg = transformed_w;
        }

        auto flip_op = [](rocblas_operation op) {
            return (op == rocblas_operation::rocblas_operation_transpose)
                    ? rocblas_operation::rocblas_operation_none
                    : rocblas_operation::rocblas_operation_transpose;
        };

        if (trans_c_ == rocblas_operation_transpose) {
            ROCBLAS_EXECUTE_FUNC(rocblas_gemm_ex, blas_handle,
                    flip_op(trans_b_), flip_op(trans_a_), n_, m_, k_, &alpha_,
                    dy, b_type_, ldb_, w_arg, a_type_, lda_, &beta_, dx,
                    c_type_, ldc_, dx, c_type_, ldc_, compute_type_, algo_,
                    solution_index, flags);
        } else {
            ROCBLAS_EXECUTE_FUNC(rocblas_gemm_ex, blas_handle, trans_a_,
                    trans_b_, m_, n_, k_, &alpha_, w_arg, a_type_, lda_, dy,
                    b_type_, ldb_, &beta_, dx, c_type_, ldc_, dx, c_type_, ldc_,
                    compute_type_, algo_, solution_index, flags);
        }
    }
};

struct miopen_gemm_inner_product_bwd_weights_impl_t
    : public miopen_inner_product_impl_base_t,
      public miopen_gemm_inner_product_base_t {
    miopenReduceTensorDescriptor_t reduceTensorDesc_ = nullptr;
    bool wie_tr_;
    bool need_reorder_;

    virtual bool need_to_transform_filter() const override {
        return need_reorder_;
    }

    virtual ~miopen_gemm_inner_product_bwd_weights_impl_t() {
        if (reduceTensorDesc_) {
            MIOPEN_EXECUTE_FUNC_V(
                    miopenDestroyReduceTensorDescriptor, reduceTensorDesc_);
        }
    }
    status_t create_and_set_reduce_descriptor() {
        MIOPEN_EXECUTE_FUNC_S(
                miopenCreateReduceTensorDescriptor, &reduceTensorDesc_);
        MIOPEN_EXECUTE_FUNC_S(miopenSetReduceTensorDescriptor,
                reduceTensorDesc_, MIOPEN_REDUCE_TENSOR_ADD, miopenFloat,
                MIOPEN_PROPAGATE_NAN, MIOPEN_REDUCE_TENSOR_NO_INDICES,
                MIOPEN_32BIT_INDICES);
        return status::success;
    }
    virtual status_t init(engine_t *engine, inner_product_pd_t *pd,
            bool /*with_relu*/, bool /*with_eltwise*/, bool /*with_sum */,
            bool need_reorder) override {
        need_reorder_ = need_reorder;
        with_bias_ = pd->with_bias();

        int ic = pd->IC_total_padded();
        int oc = pd->OC();
        int mb = pd->MB();

        wie_tr_ = (pd->diff_weights_md(0)->format_desc.blocking.strides[0]
                == 1);
        bool src_tr_ = (pd->src_md()->format_desc.blocking.strides[0] == 1)
                && (ic > 1);
        bool dst_tr_
                = (pd->diff_dst_md(0)->format_desc.blocking.strides[0] == 1)
                && (oc > 1);

        CHECK(convert_data_type(pd->src_md(), &data_types_[io::src]));
        CHECK(convert_data_type(pd->diff_weights_md(0), &data_types_[io::wei]));
        CHECK(convert_data_type(pd->diff_dst_md(), &data_types_[io::dst]));
        if (need_reorder_) {
            miopenTensorLayout_t source_format;
            CHECK(get_format(pd->src_md(), source_format));
            ndims_ = pd->ndims() < 4 ? 4 : pd->ndims();
            get_4d_tensor_descriptor(
                    pd->diff_weights_md(0), dims_[io::wei], strides_[io::wei]);
            set_filter_format(
                    ndims_, dims_[io::wei], strides_[NUM_IO], source_format);
            CHECK(init_filter_transformation(data_types_[io::wei], ndims_,
                    dims_[io::wei], strides_[NUM_IO], strides_[io::wei]));
            pd->scratchpad_registry().registrar().book(
                    memory_tracking::names::key_none,
                    memory_desc_wrapper(pd->diff_weights_md(0)).size(),
                    size_t(1));
            wie_tr_ = (strides_[NUM_IO][0] == 1);
        }

        trans_a_ = src_tr_ ? rocblas_operation_transpose
                           : rocblas_operation_none;
        trans_b_ = dst_tr_ ? rocblas_operation_none
                           : rocblas_operation_transpose;
        trans_c_ = wie_tr_ ? rocblas_operation_transpose
                           : rocblas_operation_none;

        n_ = wie_tr_ ? ic : oc;
        k_ = mb;
        m_ = wie_tr_ ? oc : ic;

        lda_ = m_;
        ldb_ = n_;
        ldc_ = m_;

        CHECK(get_rocblas_data_type(data_types_[io::src], a_type_));
        CHECK(get_rocblas_data_type(data_types_[io::dst], b_type_));
        CHECK(get_rocblas_data_type(data_types_[io::wei], c_type_));

        if (with_bias_) {
            ndims_ = 4;
            get_4d_tensor_descriptor(
                    pd->diff_dst_md(), dims_[io::dst], strides_[io::dst]);
            CHECK(convert_data_type(pd->diff_dst_md(), &data_types_[io::dst]));
            set_bias_dims(miopenTensorNCHW, ndims_, pd->OC());
            CHECK(convert_data_type(
                    pd->diff_weights_md(1), &data_types_[io::bia]));
            CHECK(create_and_set_tensor_descriptor(&tensor_descs_[io::dst],
                    data_types_[io::dst], ndims_, dims_[io::dst],
                    strides_[io::dst]));
            CHECK(create_and_set_tensor_descriptor(&tensor_descs_[io::bia],
                    data_types_[io::bia], ndims_, dims_[io::bia],
                    strides_[io::bia]));
            CHECK(create_and_set_reduce_descriptor());

            auto &sycl_engine = *utils::downcast<sycl_hip_engine_t *>(engine);
            stream_t *service_stream;
            CHECK(sycl_engine.get_service_stream(service_stream));

            auto hip_stream
                    = utils::downcast<sycl_hip_stream_t *>(service_stream);
            auto handle = hip_stream->get_miopen_handle();

            // get the required workspace size
            MIOPEN_EXECUTE_FUNC_S(miopenGetReductionWorkspaceSize, handle,
                    reduceTensorDesc_, tensor_descs_[io::dst],
                    tensor_descs_[io::bia], &workspace_size_);
        }

        if (workspace_size_ > 0) {
            pd->scratchpad_registry().registrar().book(
                    memory_tracking::names::key_iprod_int_dat_in_acc_dt,
                    workspace_size_, size_t(1));
        }

        return status::success;
    }
    void execute(miopenHandle_t miopen_handle, rocblas_handle blas_handle,
            const std::vector<void *> &args) const override {
        assert(args.size() == 6);
        auto x = args[0], dy = args[1], dw = args[2], db = args[3],
             workspace = args[4];
        auto dw_arg = need_reorder_ ? args[5] : dw;

        auto flip_op = [](rocblas_operation op) {
            return (op == rocblas_operation::rocblas_operation_transpose)
                    ? rocblas_operation::rocblas_operation_none
                    : rocblas_operation::rocblas_operation_transpose;
        };

        ROCBLAS_EXECUTE_FUNC(rocblas_gemm_ex, blas_handle,
                wie_tr_ ? flip_op(trans_b_) : trans_a_,
                wie_tr_ ? flip_op(trans_a_) : trans_b_, m_, n_, k_, &alpha_,
                (wie_tr_ ? dy : x), a_type_, lda_, (wie_tr_ ? x : dy), b_type_,
                ldb_, &beta_, dw_arg, c_type_, ldc_, dw_arg, c_type_, ldc_,
                compute_type_, algo_, solution_index, flags);

        if (need_reorder_) { transform_filter(miopen_handle, dw_arg, dw); }
        if (with_bias_) {
            MIOPEN_EXECUTE_FUNC(miopenReduceTensor, miopen_handle,
                    reduceTensorDesc_, nullptr, 0, workspace, workspace_size_,
                    &alpha_, tensor_descs_[io::dst], dy, &beta_,
                    tensor_descs_[io::bia], db);
        }
    }
};

} // namespace amd
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
