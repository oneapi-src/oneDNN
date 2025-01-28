/*******************************************************************************
* Copyright 2024-2025 Intel Corporation
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

/// @file
/// ukernel C API

#ifndef ONEAPI_DNNL_DNNL_UKERNEL_H
#define ONEAPI_DNNL_DNNL_UKERNEL_H

#include "oneapi/dnnl/dnnl.h"
#include "oneapi/dnnl/dnnl_ukernel_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/// @addtogroup dnnl_api
/// @{

/// @addtogroup dnnl_api_ukernel
/// @{

#ifdef DNNL_EXPERIMENTAL_UKERNEL

/// Creates a ukernel attributes memory storage.
///
/// @param attr_params Output ukernel attributes memory storage.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_ukernel_attr_params_create(
        dnnl_ukernel_attr_params_t *attr_params);

/// Sets post-operations arguments to a storage.
///
/// @param attr_params Memory pointers storage object.
/// @param post_ops_args A pointer to pointers of post_ops storages. Expected to
///     be packed together.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_ukernel_attr_params_set_post_ops_args(
        dnnl_ukernel_attr_params_t attr_params, const void **post_ops_args);

/// Sets tensor A scales argument to a storage.
///
/// @param attr_params Memory pointers storage object.
/// @param a_scales Pointer to the scales storage.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_ukernel_attr_params_set_A_scales(
        dnnl_ukernel_attr_params_t attr_params, const void *a_scales);

/// Sets tensor B scales argument to a storage.
///
/// If `dnnl_brgemm_set_B_scales` used mask of 2, then at least N values of
/// selected data type are expected.
///
/// @param attr_params Memory pointers storage object.
/// @param b_scales Pointer to the scales storage.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_ukernel_attr_params_set_B_scales(
        dnnl_ukernel_attr_params_t attr_params, const void *b_scales);

/// Sets tensor D scales argument to a storage.
///
/// @param attr_params Memory pointers storage object.
/// @param d_scales Pointer to the scales storage.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_ukernel_attr_params_set_D_scales(
        dnnl_ukernel_attr_params_t attr_params, const void *d_scales);

/// Destroys a ukernel attributes memory storage.
///
/// @param attr_params Memory pointers storage object to destroy.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_ukernel_attr_params_destroy(
        dnnl_ukernel_attr_params_t attr_params);

/// @addtogroup dnnl_api_ukernel_brgemm
/// @{

/// Creates a BRGeMM ukernel object. Operates by the following formula:
/// `C = [A x B]`.
///
/// @param brgemm Output BRGeMM ukernel object.
/// @param M Dimension M of tensor A.
/// @param N Dimension N of tensor B.
/// @param K Dimension K of tensors A and B.
/// @param batch_size Number of batches to process.
/// @param lda Leading dimension of tensor A.
/// @param ldb Leading dimension of tensor B.
/// @param ldc Leading dimension of tensor C.
/// @param a_dt Data type of tensor A.
/// @param b_dt Data type of tensor B.
/// @param c_dt Data type of tensor C. Must be dnnl_f32.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_brgemm_create(dnnl_brgemm_t *brgemm, dnnl_dim_t M,
        dnnl_dim_t N, dnnl_dim_t K, dnnl_dim_t batch_size, dnnl_dim_t lda,
        dnnl_dim_t ldb, dnnl_dim_t ldc, dnnl_data_type_t a_dt,
        dnnl_data_type_t b_dt, dnnl_data_type_t c_dt);

/// Sets adding an intermediate result to the output tensor C instead of
/// writing: `C += [A x B]`.
///
/// @param brgemm BRGeMM ukernel object.
/// @param add_C Value to indicate addition. Can be `0` to skip addition, and
///     `1` to apply addition.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_brgemm_set_add_C(dnnl_brgemm_t brgemm, int add_C);

/// Sets post-operations to a BRGeMM ukernel object: `D = post-operations(C)`.
///
/// Post-operations applies if one of the following holds:
/// * Non-empty attributes are specified.
/// * Output data type `d_dt` is different from accumulation data type `c_dt`.
///
/// If any of conditions happens, the final call of the accumulation chain
/// must be `dnnl_brgemm_execute_postops`, and `dnnl_brgemm_execute`, otherwise.
///
/// @param brgemm BRGeMM ukernel object.
/// @param ldd Leading dimension of tensor D.
/// @param d_dt Data type of tensor D.
/// @param post_ops Primitive post operations attribute to extend the kernel
///     operations.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_brgemm_set_post_ops(dnnl_brgemm_t brgemm,
        dnnl_dim_t ldd, dnnl_data_type_t d_dt, const_dnnl_post_ops_t post_ops);

/// Sets tensor A scales mask to a BRGeMM ukernel object.
///
/// For quantization flavor tensor A scales apply to accumulation buffer once C
/// is ready.
///
/// @param brgemm BRGeMM ukernel object.
/// @param a_scale_mask Tensor A scale mask. Can be `0` only.
dnnl_status_t DNNL_API dnnl_brgemm_set_A_scales(
        dnnl_brgemm_t brgemm, int a_scale_mask);

/// Sets tensor B scales mask to a BRGeMM ukernel object.
///
/// For quantization flavor tensor B scales apply to accumulation buffer once C
/// is ready.
///
/// @param brgemm BRGeMM ukernel object.
/// @param b_scale_mask Tensor B scale mask. Can be `0` and `2` only.
dnnl_status_t DNNL_API dnnl_brgemm_set_B_scales(
        dnnl_brgemm_t brgemm, int b_scale_mask);

/// Sets tensor D scales mask to a BRGeMM ukernel object.
///
/// For quantization flavor tensor D scales apply after all post-ops are
/// applied.
///
/// @param brgemm BRGeMM ukernel object.
/// @param d_scale_mask Tensor D scale mask. Can be `0` only.
dnnl_status_t DNNL_API dnnl_brgemm_set_D_scales(
        dnnl_brgemm_t brgemm, int d_scale_mask);

/// Finalizes initialization of a BRGeMM ukernel object.
///
/// This step is mandatory to query information from the object.
///
/// @param brgemm Output BRGeMM ukernel object.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_brgemm_finalize(dnnl_brgemm_t brgemm);

/// Returns the packing type expected by a tensor B of a BRGeMM ukernel object.
///
/// @param pack_type Output packing type. Can be `dnnl_brgemm_no_trans` if
///     packing is not expected, and `dnnl_pack_type_pack32`, otherwise.
/// @param a_dt Data type of tensor A.
/// @param b_dt Data type of tensor B.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_brgemm_get_B_pack_type(dnnl_pack_type_t *pack_type,
        dnnl_data_type_t dt_a, dnnl_data_type_t dt_b);

/// Returns the size of a scratchpad memory needed for the BRGeMM ukernel
/// object.
///
/// @param brgemm BRGeMM ukernel object.
/// @param size Output size of a buffer required for the BRGeMM ukernel object.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_brgemm_get_scratchpad_size(
        const_dnnl_brgemm_t brgemm, size_t *size);

/// Returns the flag indicating when the call to `dnnl_brgemm_execute_postops`
/// is valid.
///
/// @param brgemm BRGeMM ukernel object.
/// @param valid The flag indicating if `dnnl_brgemm_execute_postops` is valid
///     for a given ukernel object. `1` is for valid and `0`, otherwise.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_brgemm_is_execute_postops_valid(
        const_dnnl_brgemm_t brgemm, int *valid);

/// Initializes the hardware-specific context. If no initialization required,
/// returns the success status.
///
/// @param brgemm BRGeMM ukernel object.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_brgemm_set_hw_context(const_dnnl_brgemm_t brgemm);

/// Releases the hardware-specific context. Must be used after all the execution
/// calls to BRGeMM ukernel objects.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_brgemm_release_hw_context();

/// Generates an executable part of BRGeMM ukernel object.
/// @param brgemm BRGeMM ukernel object.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_brgemm_generate(dnnl_brgemm_t brgemm);

/// Executes a BRGeMM ukernel object.
///
/// @param brgemm BRGeMM ukernel object.
/// @param A_ptr Base pointer to a tensor A.
/// @param B_ptr Base pointer to a tensor B.
/// @param A_B_offsets Pointer to the set of tensor A and tensor B offsets for
///     each batch; the set must be contiguous in memory. Single batch should
///     supply offsets for both tensors A and B simultaneously. The number of
///     batches must coincide with the `batch_size` value passed at the creation
///     stage.
/// @param C_ptr Pointer to a tensor C (accumulation buffer).
/// @param scratchpad_ptr Pointer to a scratchpad buffer.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_brgemm_execute(const_dnnl_brgemm_t brgemm,
        const void *A_ptr, const void *B_ptr, const dnnl_dim_t *A_B_offsets,
        void *C_ptr, void *scratchpad_ptr);

/// Executes a BRGeMM ukernel object with post operations.
///
/// @param brgemm BRGeMM ukernel object.
/// @param A Base pointer to a tensor A.
/// @param B Base pointer to a tensor B.
/// @param A_B_offsets Pointer to a set of tensor A and tensor B offsets for
///     each batch. A set must be contiguous in memory. A single batch should
///     supply offsets for both tensors A and B simultaneously. The number of
///     batches must coincide with the `batch_size` value passed at the creation
///     stage.
/// @param C_ptr Pointer to a tensor C (accumulation buffer).
/// @param D_ptr Pointer to a tensor D (output buffer).
/// @param scratchpad_ptr Pointer to a scratchpad buffer.
/// @param attr_params Ukernel attributes memory storage.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_brgemm_execute_postops(const_dnnl_brgemm_t brgemm,
        const void *A, const void *B, const dnnl_dim_t *A_B_offsets,
        const void *C_ptr, void *D_ptr, void *scratchpad_ptr,
        const_dnnl_ukernel_attr_params_t attr_params);

/// Destroys a BRGeMM ukernel object.
///
/// @param brgemm BRGeMM ukernel object to destroy.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_brgemm_destroy(dnnl_brgemm_t brgemm);

/// Creates a transform object.
///
/// @param transform Output transform object.
/// @param K Dimension K.
/// @param N Dimension N.
/// @param in_pack_type Input packing type. Must be one of
///     `dnnl_pack_type_no_trans`, or `dnnl_pack_type_trans`.
/// @param in_ld Input leading dimension.
/// @param out_ld Output leading dimension. When packing data, it specifies a
///     block by N dimension.
/// @param in_dt Input data type.
/// @param out_dt Output data type.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_transform_create(dnnl_transform_t *transform,
        dnnl_dim_t K, dnnl_dim_t N, dnnl_pack_type_t in_pack_type,
        dnnl_dim_t in_ld, dnnl_dim_t out_ld, dnnl_data_type_t in_dt,
        dnnl_data_type_t out_dt);

/// Generates an executable part of transform object.
/// @param transform Transform object.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_transform_generate(dnnl_transform_t transform);

/// Executes a transform object.
///
/// @param transform Transform object.
/// @param in_ptr Pointer to an input buffer.
/// @param out_ptr Pointer to an output buffer.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_transform_execute(
        const_dnnl_transform_t transform, const void *in_ptr, void *out_ptr);

/// Destroys a transform object.
///
/// @param transform Transform object.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_transform_destroy(dnnl_transform_t transform);

/// @} dnnl_api_ukernel_brgemm

#endif

/// @} dnnl_api_ukernel

/// @} dnnl_api

#ifdef __cplusplus
}
#endif

#endif /* ONEAPI_DNNL_DNNL_UKERNEL_H */
