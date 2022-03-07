/*
 * Copyright (C) 2020 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 * @file zel_tracing_ddi.h
 *
 * This file has been manually generated.
 * There is no "spec" for this loader layer "tracer" API.
 */

#ifndef _ZEL_TRACING_DDI_H
#define _ZEL_TRACING_DDI_H
#if defined(__cplusplus)
#pragma once
#endif
#include "layers/zel_tracing_api.h"

#if defined(__cplusplus)
extern "C" {
#endif

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zelTracerCreate 
typedef ze_result_t (ZE_APICALL *zel_pfnTracerCreate_t)(
    const zel_tracer_desc_t*,
    zel_tracer_handle_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zetTracerDestroy 
typedef ze_result_t (ZE_APICALL *zel_pfnTracerDestroy_t)(
    zel_tracer_handle_t
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zetTracerSetPrologues 
typedef ze_result_t (ZE_APICALL *zel_pfnTracerSetPrologues_t)(
    zel_tracer_handle_t,
    zel_core_callbacks_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zetTracerSetEpilogues 
typedef ze_result_t (ZE_APICALL *zel_pfnTracerSetEpilogues_t)(
    zel_tracer_handle_t,
    zel_core_callbacks_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zetTracerSetEnabled 
typedef ze_result_t (ZE_APICALL *zel_pfnTracerSetEnabled_t)(
    zel_tracer_handle_t,
    ze_bool_t
    );


///////////////////////////////////////////////////////////////////////////////
/// @brief Table of Tracer functions pointers
typedef struct _zel_tracer_dditable_t
{
    zel_pfnTracerCreate_t                                    pfnCreate;
    zel_pfnTracerDestroy_t                                   pfnDestroy;
    zel_pfnTracerSetPrologues_t                              pfnSetPrologues;
    zel_pfnTracerSetEpilogues_t                              pfnSetEpilogues;
    zel_pfnTracerSetEnabled_t                                pfnSetEnabled;
} zel_tracer_dditable_t;


///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Tracer table
///        with current process' addresses
///
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_VERSION
ZE_DLLEXPORT ze_result_t ZE_APICALL
zelGetTracerApiProcAddrTable(
    ze_api_version_t version,                       ///< [in] API version requested
    zel_tracer_dditable_t* pDdiTable            ///< [in,out] pointer to table of DDI function pointers
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zelGetTracerApiProcAddrTable
typedef ze_result_t (ZE_APICALL *zel_pfnGetTracerApiProcAddrTable_t)(
    ze_api_version_t,
    zel_tracer_dditable_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Container for tracing DDI tables
typedef struct _zel_tracing_dditable_t
{
   zel_tracer_dditable_t         Tracer;
} zel_tracing_dditable_t;

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // _ZEL_TRACING_DDI_H
