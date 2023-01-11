/*
 * Copyright (C) 2020 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 * @file zel_tracing_api.h
 */
#ifndef _ZEL_TRACING_API_H
#define _ZEL_TRACING_API_H
#if defined(__cplusplus)
#pragma once
#endif

// 'core' API headers
#include "../ze_api.h"

#if defined(__cplusplus)
extern "C" {
#endif

// Intel 'oneAPI' Level-Zero Loader Layer Extension APIs for API Tracing
#if !defined(__GNUC__)
#pragma region zel_tracing
#endif

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of tracer object
typedef struct _zel_tracer_handle_t *zel_tracer_handle_t;

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
#ifndef ZEL_API_TRACING_NAME
/// @brief API Tracing Extension Name
#define ZEL_API_TRACING_NAME  "ZEL_api_tracing"
#endif // ZEL_API_TRACING_NAME

///////////////////////////////////////////////////////////////////////////////
/// @brief API Tracing Extension Version(s)
typedef enum _zel_api_tracing_version_t
{
    ZEL_API_TRACING_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ),  ///< version 1.0
    ZEL_API_TRACING_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ),  ///< latest known version
    ZEL_API_TRACING_VERSION_FORCE_UINT32 = 0x7fffffff

} zel_api_tracing_version_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Alias the existing callbacks definition for 'core' callbacks
typedef ze_callbacks_t zel_core_callbacks_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Defines structure types
typedef enum _zel_structure_type_t
{
    ZEL_STRUCTURE_TYPE_TRACER_DESC = 0x1  ,///< ::zel_tracer_desc_t
    // This enumeration value is deprecated.
    // Pluse use ZEL_STRUCTURE_TYPE_TRACER_DESC.
    ZEL_STRUCTURE_TYPE_TRACER_EXP_DESC = 0x1  ,///< ::zel_tracer_desc_t
    ZEL_STRUCTURE_TYPE_FORCE_UINT32 = 0x7fffffff

} zel_structure_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Tracer descriptor
typedef struct _zel_tracer_desc_t
{
    zel_structure_type_t stype;                     ///< [in] type of this structure
    const void* pNext;                              ///< [in][optional] pointer to extension-specific structure
    void* pUserData;                                ///< [in] pointer passed to every tracer's callbacks

} zel_tracer_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Creates a tracer
/// 
/// @details
///     - The tracer is created in the disabled state.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function must be thread-safe.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == desc`
///         + `nullptr == desc->pUserData`
///         + `nullptr == phTracer`
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
ZE_APIEXPORT ze_result_t ZE_APICALL
zelTracerCreate(
    const zel_tracer_desc_t* desc,              ///< [in] pointer to tracer descriptor
    zel_tracer_handle_t* phTracer               ///< [out] pointer to handle of tracer object created
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Destroys a tracer.
/// 
/// @details
///     - The application must **not** call this function from simultaneous
///       threads with the same tracer handle.
///     - The implementation of this function must be thread-safe.
///     - The implementation of this function will stall and wait on any
///       outstanding threads executing callbacks before freeing any Host
///       allocations associated with this tracer.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hTracer`
///     - ::ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE
ZE_APIEXPORT ze_result_t ZE_APICALL
zelTracerDestroy(
    zel_tracer_handle_t hTracer                 ///< [in][release] handle of tracer object to destroy
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Sets the collection of callbacks to be executed **before** driver
///        execution.
/// 
/// @details
///     - The application only needs to set the function pointers it is
///       interested in receiving; all others should be 'nullptr'
///     - The application must ensure that no other threads are executing
///       functions for which the tracing functions are changing.
///     - The application must **not** call this function from simultaneous
///       threads with the same tracer handle.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hTracer`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pCoreCbs`
ZE_APIEXPORT ze_result_t ZE_APICALL
zelTracerSetPrologues(
    zel_tracer_handle_t hTracer,                ///< [in] handle of the tracer
    zel_core_callbacks_t* pCoreCbs                  ///< [in] pointer to table of 'core' callback function pointers
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Sets the collection of callbacks to be executed **after** driver
///        execution.
/// 
/// @details
///     - The application only needs to set the function pointers it is
///       interested in receiving; all others should be 'nullptr'
///     - The application must ensure that no other threads are executing
///       functions for which the tracing functions are changing.
///     - The application must **not** call this function from simultaneous
///       threads with the same tracer handle.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hTracer`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pCoreCbs`
ZE_APIEXPORT ze_result_t ZE_APICALL
zelTracerSetEpilogues(
    zel_tracer_handle_t hTracer,                ///< [in] handle of the tracer
    zel_core_callbacks_t* pCoreCbs                  ///< [in] pointer to table of 'core' callback function pointers
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Enables (or disables) the tracer
/// 
/// @details
///     - The application must **not** call this function from simultaneous
///       threads with the same tracer handle.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hTracer`
ZE_APIEXPORT ze_result_t ZE_APICALL
zelTracerSetEnabled(
    zel_tracer_handle_t hTracer,                ///< [in] handle of the tracer
    ze_bool_t enable                                ///< [in] enable the tracer if true; disable if false
    );

#if !defined(__GNUC__)
#pragma endregion
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // _ZEL_TRACING_API_H