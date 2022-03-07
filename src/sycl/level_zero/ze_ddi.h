/*
 *
 * Copyright (C) 2019-2021 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 * @file ze_ddi.h
 * @version v1.3-r1.3.7
 *
 */
#ifndef _ZE_DDI_H
#define _ZE_DDI_H
#if defined(__cplusplus)
#pragma once
#endif
#include "ze_api.h"

#if defined(__cplusplus)
extern "C" {
#endif

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeInit 
typedef ze_result_t (ZE_APICALL *ze_pfnInit_t)(
    ze_init_flags_t
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of Global functions pointers
typedef struct _ze_global_dditable_t
{
    ze_pfnInit_t                                                pfnInit;
} ze_global_dditable_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Global table
///        with current process' addresses
///
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_VERSION
ZE_DLLEXPORT ze_result_t ZE_APICALL
zeGetGlobalProcAddrTable(
    ze_api_version_t version,                       ///< [in] API version requested
    ze_global_dditable_t* pDdiTable                 ///< [in,out] pointer to table of DDI function pointers
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeGetGlobalProcAddrTable
typedef ze_result_t (ZE_APICALL *ze_pfnGetGlobalProcAddrTable_t)(
    ze_api_version_t,
    ze_global_dditable_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeDriverGet 
typedef ze_result_t (ZE_APICALL *ze_pfnDriverGet_t)(
    uint32_t*,
    ze_driver_handle_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeDriverGetApiVersion 
typedef ze_result_t (ZE_APICALL *ze_pfnDriverGetApiVersion_t)(
    ze_driver_handle_t,
    ze_api_version_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeDriverGetProperties 
typedef ze_result_t (ZE_APICALL *ze_pfnDriverGetProperties_t)(
    ze_driver_handle_t,
    ze_driver_properties_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeDriverGetIpcProperties 
typedef ze_result_t (ZE_APICALL *ze_pfnDriverGetIpcProperties_t)(
    ze_driver_handle_t,
    ze_driver_ipc_properties_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeDriverGetExtensionProperties 
typedef ze_result_t (ZE_APICALL *ze_pfnDriverGetExtensionProperties_t)(
    ze_driver_handle_t,
    uint32_t*,
    ze_driver_extension_properties_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeDriverGetExtensionFunctionAddress 
typedef ze_result_t (ZE_APICALL *ze_pfnDriverGetExtensionFunctionAddress_t)(
    ze_driver_handle_t,
    const char*,
    void**
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of Driver functions pointers
typedef struct _ze_driver_dditable_t
{
    ze_pfnDriverGet_t                                           pfnGet;
    ze_pfnDriverGetApiVersion_t                                 pfnGetApiVersion;
    ze_pfnDriverGetProperties_t                                 pfnGetProperties;
    ze_pfnDriverGetIpcProperties_t                              pfnGetIpcProperties;
    ze_pfnDriverGetExtensionProperties_t                        pfnGetExtensionProperties;
    ze_pfnDriverGetExtensionFunctionAddress_t                   pfnGetExtensionFunctionAddress;
} ze_driver_dditable_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Driver table
///        with current process' addresses
///
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_VERSION
ZE_DLLEXPORT ze_result_t ZE_APICALL
zeGetDriverProcAddrTable(
    ze_api_version_t version,                       ///< [in] API version requested
    ze_driver_dditable_t* pDdiTable                 ///< [in,out] pointer to table of DDI function pointers
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeGetDriverProcAddrTable
typedef ze_result_t (ZE_APICALL *ze_pfnGetDriverProcAddrTable_t)(
    ze_api_version_t,
    ze_driver_dditable_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeDeviceGet 
typedef ze_result_t (ZE_APICALL *ze_pfnDeviceGet_t)(
    ze_driver_handle_t,
    uint32_t*,
    ze_device_handle_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeDeviceGetSubDevices 
typedef ze_result_t (ZE_APICALL *ze_pfnDeviceGetSubDevices_t)(
    ze_device_handle_t,
    uint32_t*,
    ze_device_handle_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeDeviceGetProperties 
typedef ze_result_t (ZE_APICALL *ze_pfnDeviceGetProperties_t)(
    ze_device_handle_t,
    ze_device_properties_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeDeviceGetComputeProperties 
typedef ze_result_t (ZE_APICALL *ze_pfnDeviceGetComputeProperties_t)(
    ze_device_handle_t,
    ze_device_compute_properties_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeDeviceGetModuleProperties 
typedef ze_result_t (ZE_APICALL *ze_pfnDeviceGetModuleProperties_t)(
    ze_device_handle_t,
    ze_device_module_properties_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeDeviceGetCommandQueueGroupProperties 
typedef ze_result_t (ZE_APICALL *ze_pfnDeviceGetCommandQueueGroupProperties_t)(
    ze_device_handle_t,
    uint32_t*,
    ze_command_queue_group_properties_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeDeviceGetMemoryProperties 
typedef ze_result_t (ZE_APICALL *ze_pfnDeviceGetMemoryProperties_t)(
    ze_device_handle_t,
    uint32_t*,
    ze_device_memory_properties_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeDeviceGetMemoryAccessProperties 
typedef ze_result_t (ZE_APICALL *ze_pfnDeviceGetMemoryAccessProperties_t)(
    ze_device_handle_t,
    ze_device_memory_access_properties_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeDeviceGetCacheProperties 
typedef ze_result_t (ZE_APICALL *ze_pfnDeviceGetCacheProperties_t)(
    ze_device_handle_t,
    uint32_t*,
    ze_device_cache_properties_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeDeviceGetImageProperties 
typedef ze_result_t (ZE_APICALL *ze_pfnDeviceGetImageProperties_t)(
    ze_device_handle_t,
    ze_device_image_properties_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeDeviceGetExternalMemoryProperties 
typedef ze_result_t (ZE_APICALL *ze_pfnDeviceGetExternalMemoryProperties_t)(
    ze_device_handle_t,
    ze_device_external_memory_properties_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeDeviceGetP2PProperties 
typedef ze_result_t (ZE_APICALL *ze_pfnDeviceGetP2PProperties_t)(
    ze_device_handle_t,
    ze_device_handle_t,
    ze_device_p2p_properties_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeDeviceCanAccessPeer 
typedef ze_result_t (ZE_APICALL *ze_pfnDeviceCanAccessPeer_t)(
    ze_device_handle_t,
    ze_device_handle_t,
    ze_bool_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeDeviceGetStatus 
typedef ze_result_t (ZE_APICALL *ze_pfnDeviceGetStatus_t)(
    ze_device_handle_t
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeDeviceGetGlobalTimestamps 
typedef ze_result_t (ZE_APICALL *ze_pfnDeviceGetGlobalTimestamps_t)(
    ze_device_handle_t,
    uint64_t*,
    uint64_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeDeviceReserveCacheExt 
typedef ze_result_t (ZE_APICALL *ze_pfnDeviceReserveCacheExt_t)(
    ze_device_handle_t,
    size_t,
    size_t
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeDeviceSetCacheAdviceExt 
typedef ze_result_t (ZE_APICALL *ze_pfnDeviceSetCacheAdviceExt_t)(
    ze_device_handle_t,
    void*,
    size_t,
    ze_cache_ext_region_t
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeDevicePciGetPropertiesExt 
typedef ze_result_t (ZE_APICALL *ze_pfnDevicePciGetPropertiesExt_t)(
    ze_device_handle_t,
    ze_pci_ext_properties_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of Device functions pointers
typedef struct _ze_device_dditable_t
{
    ze_pfnDeviceGet_t                                           pfnGet;
    ze_pfnDeviceGetSubDevices_t                                 pfnGetSubDevices;
    ze_pfnDeviceGetProperties_t                                 pfnGetProperties;
    ze_pfnDeviceGetComputeProperties_t                          pfnGetComputeProperties;
    ze_pfnDeviceGetModuleProperties_t                           pfnGetModuleProperties;
    ze_pfnDeviceGetCommandQueueGroupProperties_t                pfnGetCommandQueueGroupProperties;
    ze_pfnDeviceGetMemoryProperties_t                           pfnGetMemoryProperties;
    ze_pfnDeviceGetMemoryAccessProperties_t                     pfnGetMemoryAccessProperties;
    ze_pfnDeviceGetCacheProperties_t                            pfnGetCacheProperties;
    ze_pfnDeviceGetImageProperties_t                            pfnGetImageProperties;
    ze_pfnDeviceGetExternalMemoryProperties_t                   pfnGetExternalMemoryProperties;
    ze_pfnDeviceGetP2PProperties_t                              pfnGetP2PProperties;
    ze_pfnDeviceCanAccessPeer_t                                 pfnCanAccessPeer;
    ze_pfnDeviceGetStatus_t                                     pfnGetStatus;
    ze_pfnDeviceGetGlobalTimestamps_t                           pfnGetGlobalTimestamps;
    ze_pfnDeviceReserveCacheExt_t                               pfnReserveCacheExt;
    ze_pfnDeviceSetCacheAdviceExt_t                             pfnSetCacheAdviceExt;
    ze_pfnDevicePciGetPropertiesExt_t                           pfnPciGetPropertiesExt;
} ze_device_dditable_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Device table
///        with current process' addresses
///
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_VERSION
ZE_DLLEXPORT ze_result_t ZE_APICALL
zeGetDeviceProcAddrTable(
    ze_api_version_t version,                       ///< [in] API version requested
    ze_device_dditable_t* pDdiTable                 ///< [in,out] pointer to table of DDI function pointers
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeGetDeviceProcAddrTable
typedef ze_result_t (ZE_APICALL *ze_pfnGetDeviceProcAddrTable_t)(
    ze_api_version_t,
    ze_device_dditable_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeContextCreate 
typedef ze_result_t (ZE_APICALL *ze_pfnContextCreate_t)(
    ze_driver_handle_t,
    const ze_context_desc_t*,
    ze_context_handle_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeContextDestroy 
typedef ze_result_t (ZE_APICALL *ze_pfnContextDestroy_t)(
    ze_context_handle_t
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeContextGetStatus 
typedef ze_result_t (ZE_APICALL *ze_pfnContextGetStatus_t)(
    ze_context_handle_t
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeContextSystemBarrier 
typedef ze_result_t (ZE_APICALL *ze_pfnContextSystemBarrier_t)(
    ze_context_handle_t,
    ze_device_handle_t
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeContextMakeMemoryResident 
typedef ze_result_t (ZE_APICALL *ze_pfnContextMakeMemoryResident_t)(
    ze_context_handle_t,
    ze_device_handle_t,
    void*,
    size_t
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeContextEvictMemory 
typedef ze_result_t (ZE_APICALL *ze_pfnContextEvictMemory_t)(
    ze_context_handle_t,
    ze_device_handle_t,
    void*,
    size_t
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeContextMakeImageResident 
typedef ze_result_t (ZE_APICALL *ze_pfnContextMakeImageResident_t)(
    ze_context_handle_t,
    ze_device_handle_t,
    ze_image_handle_t
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeContextEvictImage 
typedef ze_result_t (ZE_APICALL *ze_pfnContextEvictImage_t)(
    ze_context_handle_t,
    ze_device_handle_t,
    ze_image_handle_t
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeContextCreateEx 
typedef ze_result_t (ZE_APICALL *ze_pfnContextCreateEx_t)(
    ze_driver_handle_t,
    const ze_context_desc_t*,
    uint32_t,
    ze_device_handle_t*,
    ze_context_handle_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of Context functions pointers
typedef struct _ze_context_dditable_t
{
    ze_pfnContextCreate_t                                       pfnCreate;
    ze_pfnContextDestroy_t                                      pfnDestroy;
    ze_pfnContextGetStatus_t                                    pfnGetStatus;
    ze_pfnContextSystemBarrier_t                                pfnSystemBarrier;
    ze_pfnContextMakeMemoryResident_t                           pfnMakeMemoryResident;
    ze_pfnContextEvictMemory_t                                  pfnEvictMemory;
    ze_pfnContextMakeImageResident_t                            pfnMakeImageResident;
    ze_pfnContextEvictImage_t                                   pfnEvictImage;
    ze_pfnContextCreateEx_t                                     pfnCreateEx;
} ze_context_dditable_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Context table
///        with current process' addresses
///
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_VERSION
ZE_DLLEXPORT ze_result_t ZE_APICALL
zeGetContextProcAddrTable(
    ze_api_version_t version,                       ///< [in] API version requested
    ze_context_dditable_t* pDdiTable                ///< [in,out] pointer to table of DDI function pointers
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeGetContextProcAddrTable
typedef ze_result_t (ZE_APICALL *ze_pfnGetContextProcAddrTable_t)(
    ze_api_version_t,
    ze_context_dditable_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeCommandQueueCreate 
typedef ze_result_t (ZE_APICALL *ze_pfnCommandQueueCreate_t)(
    ze_context_handle_t,
    ze_device_handle_t,
    const ze_command_queue_desc_t*,
    ze_command_queue_handle_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeCommandQueueDestroy 
typedef ze_result_t (ZE_APICALL *ze_pfnCommandQueueDestroy_t)(
    ze_command_queue_handle_t
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeCommandQueueExecuteCommandLists 
typedef ze_result_t (ZE_APICALL *ze_pfnCommandQueueExecuteCommandLists_t)(
    ze_command_queue_handle_t,
    uint32_t,
    ze_command_list_handle_t*,
    ze_fence_handle_t
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeCommandQueueSynchronize 
typedef ze_result_t (ZE_APICALL *ze_pfnCommandQueueSynchronize_t)(
    ze_command_queue_handle_t,
    uint64_t
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of CommandQueue functions pointers
typedef struct _ze_command_queue_dditable_t
{
    ze_pfnCommandQueueCreate_t                                  pfnCreate;
    ze_pfnCommandQueueDestroy_t                                 pfnDestroy;
    ze_pfnCommandQueueExecuteCommandLists_t                     pfnExecuteCommandLists;
    ze_pfnCommandQueueSynchronize_t                             pfnSynchronize;
} ze_command_queue_dditable_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's CommandQueue table
///        with current process' addresses
///
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_VERSION
ZE_DLLEXPORT ze_result_t ZE_APICALL
zeGetCommandQueueProcAddrTable(
    ze_api_version_t version,                       ///< [in] API version requested
    ze_command_queue_dditable_t* pDdiTable          ///< [in,out] pointer to table of DDI function pointers
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeGetCommandQueueProcAddrTable
typedef ze_result_t (ZE_APICALL *ze_pfnGetCommandQueueProcAddrTable_t)(
    ze_api_version_t,
    ze_command_queue_dditable_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeCommandListCreate 
typedef ze_result_t (ZE_APICALL *ze_pfnCommandListCreate_t)(
    ze_context_handle_t,
    ze_device_handle_t,
    const ze_command_list_desc_t*,
    ze_command_list_handle_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeCommandListCreateImmediate 
typedef ze_result_t (ZE_APICALL *ze_pfnCommandListCreateImmediate_t)(
    ze_context_handle_t,
    ze_device_handle_t,
    const ze_command_queue_desc_t*,
    ze_command_list_handle_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeCommandListDestroy 
typedef ze_result_t (ZE_APICALL *ze_pfnCommandListDestroy_t)(
    ze_command_list_handle_t
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeCommandListClose 
typedef ze_result_t (ZE_APICALL *ze_pfnCommandListClose_t)(
    ze_command_list_handle_t
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeCommandListReset 
typedef ze_result_t (ZE_APICALL *ze_pfnCommandListReset_t)(
    ze_command_list_handle_t
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeCommandListAppendWriteGlobalTimestamp 
typedef ze_result_t (ZE_APICALL *ze_pfnCommandListAppendWriteGlobalTimestamp_t)(
    ze_command_list_handle_t,
    uint64_t*,
    ze_event_handle_t,
    uint32_t,
    ze_event_handle_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeCommandListAppendBarrier 
typedef ze_result_t (ZE_APICALL *ze_pfnCommandListAppendBarrier_t)(
    ze_command_list_handle_t,
    ze_event_handle_t,
    uint32_t,
    ze_event_handle_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeCommandListAppendMemoryRangesBarrier 
typedef ze_result_t (ZE_APICALL *ze_pfnCommandListAppendMemoryRangesBarrier_t)(
    ze_command_list_handle_t,
    uint32_t,
    const size_t*,
    const void**,
    ze_event_handle_t,
    uint32_t,
    ze_event_handle_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeCommandListAppendMemoryCopy 
typedef ze_result_t (ZE_APICALL *ze_pfnCommandListAppendMemoryCopy_t)(
    ze_command_list_handle_t,
    void*,
    const void*,
    size_t,
    ze_event_handle_t,
    uint32_t,
    ze_event_handle_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeCommandListAppendMemoryFill 
typedef ze_result_t (ZE_APICALL *ze_pfnCommandListAppendMemoryFill_t)(
    ze_command_list_handle_t,
    void*,
    const void*,
    size_t,
    size_t,
    ze_event_handle_t,
    uint32_t,
    ze_event_handle_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeCommandListAppendMemoryCopyRegion 
typedef ze_result_t (ZE_APICALL *ze_pfnCommandListAppendMemoryCopyRegion_t)(
    ze_command_list_handle_t,
    void*,
    const ze_copy_region_t*,
    uint32_t,
    uint32_t,
    const void*,
    const ze_copy_region_t*,
    uint32_t,
    uint32_t,
    ze_event_handle_t,
    uint32_t,
    ze_event_handle_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeCommandListAppendMemoryCopyFromContext 
typedef ze_result_t (ZE_APICALL *ze_pfnCommandListAppendMemoryCopyFromContext_t)(
    ze_command_list_handle_t,
    void*,
    ze_context_handle_t,
    const void*,
    size_t,
    ze_event_handle_t,
    uint32_t,
    ze_event_handle_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeCommandListAppendImageCopy 
typedef ze_result_t (ZE_APICALL *ze_pfnCommandListAppendImageCopy_t)(
    ze_command_list_handle_t,
    ze_image_handle_t,
    ze_image_handle_t,
    ze_event_handle_t,
    uint32_t,
    ze_event_handle_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeCommandListAppendImageCopyRegion 
typedef ze_result_t (ZE_APICALL *ze_pfnCommandListAppendImageCopyRegion_t)(
    ze_command_list_handle_t,
    ze_image_handle_t,
    ze_image_handle_t,
    const ze_image_region_t*,
    const ze_image_region_t*,
    ze_event_handle_t,
    uint32_t,
    ze_event_handle_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeCommandListAppendImageCopyToMemory 
typedef ze_result_t (ZE_APICALL *ze_pfnCommandListAppendImageCopyToMemory_t)(
    ze_command_list_handle_t,
    void*,
    ze_image_handle_t,
    const ze_image_region_t*,
    ze_event_handle_t,
    uint32_t,
    ze_event_handle_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeCommandListAppendImageCopyFromMemory 
typedef ze_result_t (ZE_APICALL *ze_pfnCommandListAppendImageCopyFromMemory_t)(
    ze_command_list_handle_t,
    ze_image_handle_t,
    const void*,
    const ze_image_region_t*,
    ze_event_handle_t,
    uint32_t,
    ze_event_handle_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeCommandListAppendMemoryPrefetch 
typedef ze_result_t (ZE_APICALL *ze_pfnCommandListAppendMemoryPrefetch_t)(
    ze_command_list_handle_t,
    const void*,
    size_t
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeCommandListAppendMemAdvise 
typedef ze_result_t (ZE_APICALL *ze_pfnCommandListAppendMemAdvise_t)(
    ze_command_list_handle_t,
    ze_device_handle_t,
    const void*,
    size_t,
    ze_memory_advice_t
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeCommandListAppendSignalEvent 
typedef ze_result_t (ZE_APICALL *ze_pfnCommandListAppendSignalEvent_t)(
    ze_command_list_handle_t,
    ze_event_handle_t
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeCommandListAppendWaitOnEvents 
typedef ze_result_t (ZE_APICALL *ze_pfnCommandListAppendWaitOnEvents_t)(
    ze_command_list_handle_t,
    uint32_t,
    ze_event_handle_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeCommandListAppendEventReset 
typedef ze_result_t (ZE_APICALL *ze_pfnCommandListAppendEventReset_t)(
    ze_command_list_handle_t,
    ze_event_handle_t
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeCommandListAppendQueryKernelTimestamps 
typedef ze_result_t (ZE_APICALL *ze_pfnCommandListAppendQueryKernelTimestamps_t)(
    ze_command_list_handle_t,
    uint32_t,
    ze_event_handle_t*,
    void*,
    const size_t*,
    ze_event_handle_t,
    uint32_t,
    ze_event_handle_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeCommandListAppendLaunchKernel 
typedef ze_result_t (ZE_APICALL *ze_pfnCommandListAppendLaunchKernel_t)(
    ze_command_list_handle_t,
    ze_kernel_handle_t,
    const ze_group_count_t*,
    ze_event_handle_t,
    uint32_t,
    ze_event_handle_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeCommandListAppendLaunchCooperativeKernel 
typedef ze_result_t (ZE_APICALL *ze_pfnCommandListAppendLaunchCooperativeKernel_t)(
    ze_command_list_handle_t,
    ze_kernel_handle_t,
    const ze_group_count_t*,
    ze_event_handle_t,
    uint32_t,
    ze_event_handle_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeCommandListAppendLaunchKernelIndirect 
typedef ze_result_t (ZE_APICALL *ze_pfnCommandListAppendLaunchKernelIndirect_t)(
    ze_command_list_handle_t,
    ze_kernel_handle_t,
    const ze_group_count_t*,
    ze_event_handle_t,
    uint32_t,
    ze_event_handle_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeCommandListAppendLaunchMultipleKernelsIndirect 
typedef ze_result_t (ZE_APICALL *ze_pfnCommandListAppendLaunchMultipleKernelsIndirect_t)(
    ze_command_list_handle_t,
    uint32_t,
    ze_kernel_handle_t*,
    const uint32_t*,
    const ze_group_count_t*,
    ze_event_handle_t,
    uint32_t,
    ze_event_handle_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeCommandListAppendImageCopyToMemoryExt 
typedef ze_result_t (ZE_APICALL *ze_pfnCommandListAppendImageCopyToMemoryExt_t)(
    ze_command_list_handle_t,
    void*,
    ze_image_handle_t,
    const ze_image_region_t*,
    uint32_t,
    uint32_t,
    ze_event_handle_t,
    uint32_t,
    ze_event_handle_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeCommandListAppendImageCopyFromMemoryExt 
typedef ze_result_t (ZE_APICALL *ze_pfnCommandListAppendImageCopyFromMemoryExt_t)(
    ze_command_list_handle_t,
    ze_image_handle_t,
    const void*,
    const ze_image_region_t*,
    uint32_t,
    uint32_t,
    ze_event_handle_t,
    uint32_t,
    ze_event_handle_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of CommandList functions pointers
typedef struct _ze_command_list_dditable_t
{
    ze_pfnCommandListCreate_t                                   pfnCreate;
    ze_pfnCommandListCreateImmediate_t                          pfnCreateImmediate;
    ze_pfnCommandListDestroy_t                                  pfnDestroy;
    ze_pfnCommandListClose_t                                    pfnClose;
    ze_pfnCommandListReset_t                                    pfnReset;
    ze_pfnCommandListAppendWriteGlobalTimestamp_t               pfnAppendWriteGlobalTimestamp;
    ze_pfnCommandListAppendBarrier_t                            pfnAppendBarrier;
    ze_pfnCommandListAppendMemoryRangesBarrier_t                pfnAppendMemoryRangesBarrier;
    ze_pfnCommandListAppendMemoryCopy_t                         pfnAppendMemoryCopy;
    ze_pfnCommandListAppendMemoryFill_t                         pfnAppendMemoryFill;
    ze_pfnCommandListAppendMemoryCopyRegion_t                   pfnAppendMemoryCopyRegion;
    ze_pfnCommandListAppendMemoryCopyFromContext_t              pfnAppendMemoryCopyFromContext;
    ze_pfnCommandListAppendImageCopy_t                          pfnAppendImageCopy;
    ze_pfnCommandListAppendImageCopyRegion_t                    pfnAppendImageCopyRegion;
    ze_pfnCommandListAppendImageCopyToMemory_t                  pfnAppendImageCopyToMemory;
    ze_pfnCommandListAppendImageCopyFromMemory_t                pfnAppendImageCopyFromMemory;
    ze_pfnCommandListAppendMemoryPrefetch_t                     pfnAppendMemoryPrefetch;
    ze_pfnCommandListAppendMemAdvise_t                          pfnAppendMemAdvise;
    ze_pfnCommandListAppendSignalEvent_t                        pfnAppendSignalEvent;
    ze_pfnCommandListAppendWaitOnEvents_t                       pfnAppendWaitOnEvents;
    ze_pfnCommandListAppendEventReset_t                         pfnAppendEventReset;
    ze_pfnCommandListAppendQueryKernelTimestamps_t              pfnAppendQueryKernelTimestamps;
    ze_pfnCommandListAppendLaunchKernel_t                       pfnAppendLaunchKernel;
    ze_pfnCommandListAppendLaunchCooperativeKernel_t            pfnAppendLaunchCooperativeKernel;
    ze_pfnCommandListAppendLaunchKernelIndirect_t               pfnAppendLaunchKernelIndirect;
    ze_pfnCommandListAppendLaunchMultipleKernelsIndirect_t      pfnAppendLaunchMultipleKernelsIndirect;
    ze_pfnCommandListAppendImageCopyToMemoryExt_t               pfnAppendImageCopyToMemoryExt;
    ze_pfnCommandListAppendImageCopyFromMemoryExt_t             pfnAppendImageCopyFromMemoryExt;
} ze_command_list_dditable_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's CommandList table
///        with current process' addresses
///
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_VERSION
ZE_DLLEXPORT ze_result_t ZE_APICALL
zeGetCommandListProcAddrTable(
    ze_api_version_t version,                       ///< [in] API version requested
    ze_command_list_dditable_t* pDdiTable           ///< [in,out] pointer to table of DDI function pointers
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeGetCommandListProcAddrTable
typedef ze_result_t (ZE_APICALL *ze_pfnGetCommandListProcAddrTable_t)(
    ze_api_version_t,
    ze_command_list_dditable_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeImageGetProperties 
typedef ze_result_t (ZE_APICALL *ze_pfnImageGetProperties_t)(
    ze_device_handle_t,
    const ze_image_desc_t*,
    ze_image_properties_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeImageCreate 
typedef ze_result_t (ZE_APICALL *ze_pfnImageCreate_t)(
    ze_context_handle_t,
    ze_device_handle_t,
    const ze_image_desc_t*,
    ze_image_handle_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeImageDestroy 
typedef ze_result_t (ZE_APICALL *ze_pfnImageDestroy_t)(
    ze_image_handle_t
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeImageGetAllocPropertiesExt 
typedef ze_result_t (ZE_APICALL *ze_pfnImageGetAllocPropertiesExt_t)(
    ze_context_handle_t,
    ze_image_handle_t,
    ze_image_allocation_ext_properties_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of Image functions pointers
typedef struct _ze_image_dditable_t
{
    ze_pfnImageGetProperties_t                                  pfnGetProperties;
    ze_pfnImageCreate_t                                         pfnCreate;
    ze_pfnImageDestroy_t                                        pfnDestroy;
    ze_pfnImageGetAllocPropertiesExt_t                          pfnGetAllocPropertiesExt;
} ze_image_dditable_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Image table
///        with current process' addresses
///
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_VERSION
ZE_DLLEXPORT ze_result_t ZE_APICALL
zeGetImageProcAddrTable(
    ze_api_version_t version,                       ///< [in] API version requested
    ze_image_dditable_t* pDdiTable                  ///< [in,out] pointer to table of DDI function pointers
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeGetImageProcAddrTable
typedef ze_result_t (ZE_APICALL *ze_pfnGetImageProcAddrTable_t)(
    ze_api_version_t,
    ze_image_dditable_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeImageGetMemoryPropertiesExp 
typedef ze_result_t (ZE_APICALL *ze_pfnImageGetMemoryPropertiesExp_t)(
    ze_image_handle_t,
    ze_image_memory_properties_exp_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeImageViewCreateExp 
typedef ze_result_t (ZE_APICALL *ze_pfnImageViewCreateExp_t)(
    ze_context_handle_t,
    ze_device_handle_t,
    const ze_image_desc_t*,
    ze_image_handle_t,
    ze_image_handle_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of ImageExp functions pointers
typedef struct _ze_image_exp_dditable_t
{
    ze_pfnImageGetMemoryPropertiesExp_t                         pfnGetMemoryPropertiesExp;
    ze_pfnImageViewCreateExp_t                                  pfnViewCreateExp;
} ze_image_exp_dditable_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's ImageExp table
///        with current process' addresses
///
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_VERSION
ZE_DLLEXPORT ze_result_t ZE_APICALL
zeGetImageExpProcAddrTable(
    ze_api_version_t version,                       ///< [in] API version requested
    ze_image_exp_dditable_t* pDdiTable              ///< [in,out] pointer to table of DDI function pointers
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeGetImageExpProcAddrTable
typedef ze_result_t (ZE_APICALL *ze_pfnGetImageExpProcAddrTable_t)(
    ze_api_version_t,
    ze_image_exp_dditable_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeFenceCreate 
typedef ze_result_t (ZE_APICALL *ze_pfnFenceCreate_t)(
    ze_command_queue_handle_t,
    const ze_fence_desc_t*,
    ze_fence_handle_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeFenceDestroy 
typedef ze_result_t (ZE_APICALL *ze_pfnFenceDestroy_t)(
    ze_fence_handle_t
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeFenceHostSynchronize 
typedef ze_result_t (ZE_APICALL *ze_pfnFenceHostSynchronize_t)(
    ze_fence_handle_t,
    uint64_t
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeFenceQueryStatus 
typedef ze_result_t (ZE_APICALL *ze_pfnFenceQueryStatus_t)(
    ze_fence_handle_t
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeFenceReset 
typedef ze_result_t (ZE_APICALL *ze_pfnFenceReset_t)(
    ze_fence_handle_t
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of Fence functions pointers
typedef struct _ze_fence_dditable_t
{
    ze_pfnFenceCreate_t                                         pfnCreate;
    ze_pfnFenceDestroy_t                                        pfnDestroy;
    ze_pfnFenceHostSynchronize_t                                pfnHostSynchronize;
    ze_pfnFenceQueryStatus_t                                    pfnQueryStatus;
    ze_pfnFenceReset_t                                          pfnReset;
} ze_fence_dditable_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Fence table
///        with current process' addresses
///
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_VERSION
ZE_DLLEXPORT ze_result_t ZE_APICALL
zeGetFenceProcAddrTable(
    ze_api_version_t version,                       ///< [in] API version requested
    ze_fence_dditable_t* pDdiTable                  ///< [in,out] pointer to table of DDI function pointers
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeGetFenceProcAddrTable
typedef ze_result_t (ZE_APICALL *ze_pfnGetFenceProcAddrTable_t)(
    ze_api_version_t,
    ze_fence_dditable_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeEventPoolCreate 
typedef ze_result_t (ZE_APICALL *ze_pfnEventPoolCreate_t)(
    ze_context_handle_t,
    const ze_event_pool_desc_t*,
    uint32_t,
    ze_device_handle_t*,
    ze_event_pool_handle_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeEventPoolDestroy 
typedef ze_result_t (ZE_APICALL *ze_pfnEventPoolDestroy_t)(
    ze_event_pool_handle_t
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeEventPoolGetIpcHandle 
typedef ze_result_t (ZE_APICALL *ze_pfnEventPoolGetIpcHandle_t)(
    ze_event_pool_handle_t,
    ze_ipc_event_pool_handle_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeEventPoolOpenIpcHandle 
typedef ze_result_t (ZE_APICALL *ze_pfnEventPoolOpenIpcHandle_t)(
    ze_context_handle_t,
    ze_ipc_event_pool_handle_t,
    ze_event_pool_handle_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeEventPoolCloseIpcHandle 
typedef ze_result_t (ZE_APICALL *ze_pfnEventPoolCloseIpcHandle_t)(
    ze_event_pool_handle_t
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of EventPool functions pointers
typedef struct _ze_event_pool_dditable_t
{
    ze_pfnEventPoolCreate_t                                     pfnCreate;
    ze_pfnEventPoolDestroy_t                                    pfnDestroy;
    ze_pfnEventPoolGetIpcHandle_t                               pfnGetIpcHandle;
    ze_pfnEventPoolOpenIpcHandle_t                              pfnOpenIpcHandle;
    ze_pfnEventPoolCloseIpcHandle_t                             pfnCloseIpcHandle;
} ze_event_pool_dditable_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's EventPool table
///        with current process' addresses
///
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_VERSION
ZE_DLLEXPORT ze_result_t ZE_APICALL
zeGetEventPoolProcAddrTable(
    ze_api_version_t version,                       ///< [in] API version requested
    ze_event_pool_dditable_t* pDdiTable             ///< [in,out] pointer to table of DDI function pointers
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeGetEventPoolProcAddrTable
typedef ze_result_t (ZE_APICALL *ze_pfnGetEventPoolProcAddrTable_t)(
    ze_api_version_t,
    ze_event_pool_dditable_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeEventCreate 
typedef ze_result_t (ZE_APICALL *ze_pfnEventCreate_t)(
    ze_event_pool_handle_t,
    const ze_event_desc_t*,
    ze_event_handle_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeEventDestroy 
typedef ze_result_t (ZE_APICALL *ze_pfnEventDestroy_t)(
    ze_event_handle_t
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeEventHostSignal 
typedef ze_result_t (ZE_APICALL *ze_pfnEventHostSignal_t)(
    ze_event_handle_t
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeEventHostSynchronize 
typedef ze_result_t (ZE_APICALL *ze_pfnEventHostSynchronize_t)(
    ze_event_handle_t,
    uint64_t
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeEventQueryStatus 
typedef ze_result_t (ZE_APICALL *ze_pfnEventQueryStatus_t)(
    ze_event_handle_t
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeEventHostReset 
typedef ze_result_t (ZE_APICALL *ze_pfnEventHostReset_t)(
    ze_event_handle_t
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeEventQueryKernelTimestamp 
typedef ze_result_t (ZE_APICALL *ze_pfnEventQueryKernelTimestamp_t)(
    ze_event_handle_t,
    ze_kernel_timestamp_result_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of Event functions pointers
typedef struct _ze_event_dditable_t
{
    ze_pfnEventCreate_t                                         pfnCreate;
    ze_pfnEventDestroy_t                                        pfnDestroy;
    ze_pfnEventHostSignal_t                                     pfnHostSignal;
    ze_pfnEventHostSynchronize_t                                pfnHostSynchronize;
    ze_pfnEventQueryStatus_t                                    pfnQueryStatus;
    ze_pfnEventHostReset_t                                      pfnHostReset;
    ze_pfnEventQueryKernelTimestamp_t                           pfnQueryKernelTimestamp;
} ze_event_dditable_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Event table
///        with current process' addresses
///
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_VERSION
ZE_DLLEXPORT ze_result_t ZE_APICALL
zeGetEventProcAddrTable(
    ze_api_version_t version,                       ///< [in] API version requested
    ze_event_dditable_t* pDdiTable                  ///< [in,out] pointer to table of DDI function pointers
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeGetEventProcAddrTable
typedef ze_result_t (ZE_APICALL *ze_pfnGetEventProcAddrTable_t)(
    ze_api_version_t,
    ze_event_dditable_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeEventQueryTimestampsExp 
typedef ze_result_t (ZE_APICALL *ze_pfnEventQueryTimestampsExp_t)(
    ze_event_handle_t,
    ze_device_handle_t,
    uint32_t*,
    ze_kernel_timestamp_result_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of EventExp functions pointers
typedef struct _ze_event_exp_dditable_t
{
    ze_pfnEventQueryTimestampsExp_t                             pfnQueryTimestampsExp;
} ze_event_exp_dditable_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's EventExp table
///        with current process' addresses
///
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_VERSION
ZE_DLLEXPORT ze_result_t ZE_APICALL
zeGetEventExpProcAddrTable(
    ze_api_version_t version,                       ///< [in] API version requested
    ze_event_exp_dditable_t* pDdiTable              ///< [in,out] pointer to table of DDI function pointers
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeGetEventExpProcAddrTable
typedef ze_result_t (ZE_APICALL *ze_pfnGetEventExpProcAddrTable_t)(
    ze_api_version_t,
    ze_event_exp_dditable_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeModuleCreate 
typedef ze_result_t (ZE_APICALL *ze_pfnModuleCreate_t)(
    ze_context_handle_t,
    ze_device_handle_t,
    const ze_module_desc_t*,
    ze_module_handle_t*,
    ze_module_build_log_handle_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeModuleDestroy 
typedef ze_result_t (ZE_APICALL *ze_pfnModuleDestroy_t)(
    ze_module_handle_t
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeModuleDynamicLink 
typedef ze_result_t (ZE_APICALL *ze_pfnModuleDynamicLink_t)(
    uint32_t,
    ze_module_handle_t*,
    ze_module_build_log_handle_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeModuleGetNativeBinary 
typedef ze_result_t (ZE_APICALL *ze_pfnModuleGetNativeBinary_t)(
    ze_module_handle_t,
    size_t*,
    uint8_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeModuleGetGlobalPointer 
typedef ze_result_t (ZE_APICALL *ze_pfnModuleGetGlobalPointer_t)(
    ze_module_handle_t,
    const char*,
    size_t*,
    void**
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeModuleGetKernelNames 
typedef ze_result_t (ZE_APICALL *ze_pfnModuleGetKernelNames_t)(
    ze_module_handle_t,
    uint32_t*,
    const char**
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeModuleGetProperties 
typedef ze_result_t (ZE_APICALL *ze_pfnModuleGetProperties_t)(
    ze_module_handle_t,
    ze_module_properties_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeModuleGetFunctionPointer 
typedef ze_result_t (ZE_APICALL *ze_pfnModuleGetFunctionPointer_t)(
    ze_module_handle_t,
    const char*,
    void**
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeModuleInspectLinkageExt 
typedef ze_result_t (ZE_APICALL *ze_pfnModuleInspectLinkageExt_t)(
    ze_linkage_inspection_ext_desc_t*,
    uint32_t,
    ze_module_handle_t*,
    ze_module_build_log_handle_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of Module functions pointers
typedef struct _ze_module_dditable_t
{
    ze_pfnModuleCreate_t                                        pfnCreate;
    ze_pfnModuleDestroy_t                                       pfnDestroy;
    ze_pfnModuleDynamicLink_t                                   pfnDynamicLink;
    ze_pfnModuleGetNativeBinary_t                               pfnGetNativeBinary;
    ze_pfnModuleGetGlobalPointer_t                              pfnGetGlobalPointer;
    ze_pfnModuleGetKernelNames_t                                pfnGetKernelNames;
    ze_pfnModuleGetProperties_t                                 pfnGetProperties;
    ze_pfnModuleGetFunctionPointer_t                            pfnGetFunctionPointer;
    ze_pfnModuleInspectLinkageExt_t                             pfnInspectLinkageExt;
} ze_module_dditable_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Module table
///        with current process' addresses
///
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_VERSION
ZE_DLLEXPORT ze_result_t ZE_APICALL
zeGetModuleProcAddrTable(
    ze_api_version_t version,                       ///< [in] API version requested
    ze_module_dditable_t* pDdiTable                 ///< [in,out] pointer to table of DDI function pointers
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeGetModuleProcAddrTable
typedef ze_result_t (ZE_APICALL *ze_pfnGetModuleProcAddrTable_t)(
    ze_api_version_t,
    ze_module_dditable_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeModuleBuildLogDestroy 
typedef ze_result_t (ZE_APICALL *ze_pfnModuleBuildLogDestroy_t)(
    ze_module_build_log_handle_t
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeModuleBuildLogGetString 
typedef ze_result_t (ZE_APICALL *ze_pfnModuleBuildLogGetString_t)(
    ze_module_build_log_handle_t,
    size_t*,
    char*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of ModuleBuildLog functions pointers
typedef struct _ze_module_build_log_dditable_t
{
    ze_pfnModuleBuildLogDestroy_t                               pfnDestroy;
    ze_pfnModuleBuildLogGetString_t                             pfnGetString;
} ze_module_build_log_dditable_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's ModuleBuildLog table
///        with current process' addresses
///
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_VERSION
ZE_DLLEXPORT ze_result_t ZE_APICALL
zeGetModuleBuildLogProcAddrTable(
    ze_api_version_t version,                       ///< [in] API version requested
    ze_module_build_log_dditable_t* pDdiTable       ///< [in,out] pointer to table of DDI function pointers
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeGetModuleBuildLogProcAddrTable
typedef ze_result_t (ZE_APICALL *ze_pfnGetModuleBuildLogProcAddrTable_t)(
    ze_api_version_t,
    ze_module_build_log_dditable_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeKernelCreate 
typedef ze_result_t (ZE_APICALL *ze_pfnKernelCreate_t)(
    ze_module_handle_t,
    const ze_kernel_desc_t*,
    ze_kernel_handle_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeKernelDestroy 
typedef ze_result_t (ZE_APICALL *ze_pfnKernelDestroy_t)(
    ze_kernel_handle_t
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeKernelSetCacheConfig 
typedef ze_result_t (ZE_APICALL *ze_pfnKernelSetCacheConfig_t)(
    ze_kernel_handle_t,
    ze_cache_config_flags_t
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeKernelSetGroupSize 
typedef ze_result_t (ZE_APICALL *ze_pfnKernelSetGroupSize_t)(
    ze_kernel_handle_t,
    uint32_t,
    uint32_t,
    uint32_t
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeKernelSuggestGroupSize 
typedef ze_result_t (ZE_APICALL *ze_pfnKernelSuggestGroupSize_t)(
    ze_kernel_handle_t,
    uint32_t,
    uint32_t,
    uint32_t,
    uint32_t*,
    uint32_t*,
    uint32_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeKernelSuggestMaxCooperativeGroupCount 
typedef ze_result_t (ZE_APICALL *ze_pfnKernelSuggestMaxCooperativeGroupCount_t)(
    ze_kernel_handle_t,
    uint32_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeKernelSetArgumentValue 
typedef ze_result_t (ZE_APICALL *ze_pfnKernelSetArgumentValue_t)(
    ze_kernel_handle_t,
    uint32_t,
    size_t,
    const void*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeKernelSetIndirectAccess 
typedef ze_result_t (ZE_APICALL *ze_pfnKernelSetIndirectAccess_t)(
    ze_kernel_handle_t,
    ze_kernel_indirect_access_flags_t
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeKernelGetIndirectAccess 
typedef ze_result_t (ZE_APICALL *ze_pfnKernelGetIndirectAccess_t)(
    ze_kernel_handle_t,
    ze_kernel_indirect_access_flags_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeKernelGetSourceAttributes 
typedef ze_result_t (ZE_APICALL *ze_pfnKernelGetSourceAttributes_t)(
    ze_kernel_handle_t,
    uint32_t*,
    char**
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeKernelGetProperties 
typedef ze_result_t (ZE_APICALL *ze_pfnKernelGetProperties_t)(
    ze_kernel_handle_t,
    ze_kernel_properties_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeKernelGetName 
typedef ze_result_t (ZE_APICALL *ze_pfnKernelGetName_t)(
    ze_kernel_handle_t,
    size_t*,
    char*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of Kernel functions pointers
typedef struct _ze_kernel_dditable_t
{
    ze_pfnKernelCreate_t                                        pfnCreate;
    ze_pfnKernelDestroy_t                                       pfnDestroy;
    ze_pfnKernelSetCacheConfig_t                                pfnSetCacheConfig;
    ze_pfnKernelSetGroupSize_t                                  pfnSetGroupSize;
    ze_pfnKernelSuggestGroupSize_t                              pfnSuggestGroupSize;
    ze_pfnKernelSuggestMaxCooperativeGroupCount_t               pfnSuggestMaxCooperativeGroupCount;
    ze_pfnKernelSetArgumentValue_t                              pfnSetArgumentValue;
    ze_pfnKernelSetIndirectAccess_t                             pfnSetIndirectAccess;
    ze_pfnKernelGetIndirectAccess_t                             pfnGetIndirectAccess;
    ze_pfnKernelGetSourceAttributes_t                           pfnGetSourceAttributes;
    ze_pfnKernelGetProperties_t                                 pfnGetProperties;
    ze_pfnKernelGetName_t                                       pfnGetName;
} ze_kernel_dditable_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Kernel table
///        with current process' addresses
///
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_VERSION
ZE_DLLEXPORT ze_result_t ZE_APICALL
zeGetKernelProcAddrTable(
    ze_api_version_t version,                       ///< [in] API version requested
    ze_kernel_dditable_t* pDdiTable                 ///< [in,out] pointer to table of DDI function pointers
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeGetKernelProcAddrTable
typedef ze_result_t (ZE_APICALL *ze_pfnGetKernelProcAddrTable_t)(
    ze_api_version_t,
    ze_kernel_dditable_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeKernelSetGlobalOffsetExp 
typedef ze_result_t (ZE_APICALL *ze_pfnKernelSetGlobalOffsetExp_t)(
    ze_kernel_handle_t,
    uint32_t,
    uint32_t,
    uint32_t
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeKernelSchedulingHintExp 
typedef ze_result_t (ZE_APICALL *ze_pfnKernelSchedulingHintExp_t)(
    ze_kernel_handle_t,
    ze_scheduling_hint_exp_desc_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of KernelExp functions pointers
typedef struct _ze_kernel_exp_dditable_t
{
    ze_pfnKernelSetGlobalOffsetExp_t                            pfnSetGlobalOffsetExp;
    ze_pfnKernelSchedulingHintExp_t                             pfnSchedulingHintExp;
} ze_kernel_exp_dditable_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's KernelExp table
///        with current process' addresses
///
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_VERSION
ZE_DLLEXPORT ze_result_t ZE_APICALL
zeGetKernelExpProcAddrTable(
    ze_api_version_t version,                       ///< [in] API version requested
    ze_kernel_exp_dditable_t* pDdiTable             ///< [in,out] pointer to table of DDI function pointers
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeGetKernelExpProcAddrTable
typedef ze_result_t (ZE_APICALL *ze_pfnGetKernelExpProcAddrTable_t)(
    ze_api_version_t,
    ze_kernel_exp_dditable_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeSamplerCreate 
typedef ze_result_t (ZE_APICALL *ze_pfnSamplerCreate_t)(
    ze_context_handle_t,
    ze_device_handle_t,
    const ze_sampler_desc_t*,
    ze_sampler_handle_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeSamplerDestroy 
typedef ze_result_t (ZE_APICALL *ze_pfnSamplerDestroy_t)(
    ze_sampler_handle_t
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of Sampler functions pointers
typedef struct _ze_sampler_dditable_t
{
    ze_pfnSamplerCreate_t                                       pfnCreate;
    ze_pfnSamplerDestroy_t                                      pfnDestroy;
} ze_sampler_dditable_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Sampler table
///        with current process' addresses
///
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_VERSION
ZE_DLLEXPORT ze_result_t ZE_APICALL
zeGetSamplerProcAddrTable(
    ze_api_version_t version,                       ///< [in] API version requested
    ze_sampler_dditable_t* pDdiTable                ///< [in,out] pointer to table of DDI function pointers
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeGetSamplerProcAddrTable
typedef ze_result_t (ZE_APICALL *ze_pfnGetSamplerProcAddrTable_t)(
    ze_api_version_t,
    ze_sampler_dditable_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zePhysicalMemCreate 
typedef ze_result_t (ZE_APICALL *ze_pfnPhysicalMemCreate_t)(
    ze_context_handle_t,
    ze_device_handle_t,
    ze_physical_mem_desc_t*,
    ze_physical_mem_handle_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zePhysicalMemDestroy 
typedef ze_result_t (ZE_APICALL *ze_pfnPhysicalMemDestroy_t)(
    ze_context_handle_t,
    ze_physical_mem_handle_t
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of PhysicalMem functions pointers
typedef struct _ze_physical_mem_dditable_t
{
    ze_pfnPhysicalMemCreate_t                                   pfnCreate;
    ze_pfnPhysicalMemDestroy_t                                  pfnDestroy;
} ze_physical_mem_dditable_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's PhysicalMem table
///        with current process' addresses
///
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_VERSION
ZE_DLLEXPORT ze_result_t ZE_APICALL
zeGetPhysicalMemProcAddrTable(
    ze_api_version_t version,                       ///< [in] API version requested
    ze_physical_mem_dditable_t* pDdiTable           ///< [in,out] pointer to table of DDI function pointers
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeGetPhysicalMemProcAddrTable
typedef ze_result_t (ZE_APICALL *ze_pfnGetPhysicalMemProcAddrTable_t)(
    ze_api_version_t,
    ze_physical_mem_dditable_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeMemAllocShared 
typedef ze_result_t (ZE_APICALL *ze_pfnMemAllocShared_t)(
    ze_context_handle_t,
    const ze_device_mem_alloc_desc_t*,
    const ze_host_mem_alloc_desc_t*,
    size_t,
    size_t,
    ze_device_handle_t,
    void**
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeMemAllocDevice 
typedef ze_result_t (ZE_APICALL *ze_pfnMemAllocDevice_t)(
    ze_context_handle_t,
    const ze_device_mem_alloc_desc_t*,
    size_t,
    size_t,
    ze_device_handle_t,
    void**
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeMemAllocHost 
typedef ze_result_t (ZE_APICALL *ze_pfnMemAllocHost_t)(
    ze_context_handle_t,
    const ze_host_mem_alloc_desc_t*,
    size_t,
    size_t,
    void**
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeMemFree 
typedef ze_result_t (ZE_APICALL *ze_pfnMemFree_t)(
    ze_context_handle_t,
    void*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeMemGetAllocProperties 
typedef ze_result_t (ZE_APICALL *ze_pfnMemGetAllocProperties_t)(
    ze_context_handle_t,
    const void*,
    ze_memory_allocation_properties_t*,
    ze_device_handle_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeMemGetAddressRange 
typedef ze_result_t (ZE_APICALL *ze_pfnMemGetAddressRange_t)(
    ze_context_handle_t,
    const void*,
    void**,
    size_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeMemGetIpcHandle 
typedef ze_result_t (ZE_APICALL *ze_pfnMemGetIpcHandle_t)(
    ze_context_handle_t,
    const void*,
    ze_ipc_mem_handle_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeMemOpenIpcHandle 
typedef ze_result_t (ZE_APICALL *ze_pfnMemOpenIpcHandle_t)(
    ze_context_handle_t,
    ze_device_handle_t,
    ze_ipc_mem_handle_t,
    ze_ipc_memory_flags_t,
    void**
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeMemCloseIpcHandle 
typedef ze_result_t (ZE_APICALL *ze_pfnMemCloseIpcHandle_t)(
    ze_context_handle_t,
    const void*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeMemFreeExt 
typedef ze_result_t (ZE_APICALL *ze_pfnMemFreeExt_t)(
    ze_context_handle_t,
    const ze_memory_free_ext_desc_t*,
    void*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of Mem functions pointers
typedef struct _ze_mem_dditable_t
{
    ze_pfnMemAllocShared_t                                      pfnAllocShared;
    ze_pfnMemAllocDevice_t                                      pfnAllocDevice;
    ze_pfnMemAllocHost_t                                        pfnAllocHost;
    ze_pfnMemFree_t                                             pfnFree;
    ze_pfnMemGetAllocProperties_t                               pfnGetAllocProperties;
    ze_pfnMemGetAddressRange_t                                  pfnGetAddressRange;
    ze_pfnMemGetIpcHandle_t                                     pfnGetIpcHandle;
    ze_pfnMemOpenIpcHandle_t                                    pfnOpenIpcHandle;
    ze_pfnMemCloseIpcHandle_t                                   pfnCloseIpcHandle;
    ze_pfnMemFreeExt_t                                          pfnFreeExt;
} ze_mem_dditable_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Mem table
///        with current process' addresses
///
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_VERSION
ZE_DLLEXPORT ze_result_t ZE_APICALL
zeGetMemProcAddrTable(
    ze_api_version_t version,                       ///< [in] API version requested
    ze_mem_dditable_t* pDdiTable                    ///< [in,out] pointer to table of DDI function pointers
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeGetMemProcAddrTable
typedef ze_result_t (ZE_APICALL *ze_pfnGetMemProcAddrTable_t)(
    ze_api_version_t,
    ze_mem_dditable_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeVirtualMemReserve 
typedef ze_result_t (ZE_APICALL *ze_pfnVirtualMemReserve_t)(
    ze_context_handle_t,
    const void*,
    size_t,
    void**
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeVirtualMemFree 
typedef ze_result_t (ZE_APICALL *ze_pfnVirtualMemFree_t)(
    ze_context_handle_t,
    const void*,
    size_t
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeVirtualMemQueryPageSize 
typedef ze_result_t (ZE_APICALL *ze_pfnVirtualMemQueryPageSize_t)(
    ze_context_handle_t,
    ze_device_handle_t,
    size_t,
    size_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeVirtualMemMap 
typedef ze_result_t (ZE_APICALL *ze_pfnVirtualMemMap_t)(
    ze_context_handle_t,
    const void*,
    size_t,
    ze_physical_mem_handle_t,
    size_t,
    ze_memory_access_attribute_t
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeVirtualMemUnmap 
typedef ze_result_t (ZE_APICALL *ze_pfnVirtualMemUnmap_t)(
    ze_context_handle_t,
    const void*,
    size_t
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeVirtualMemSetAccessAttribute 
typedef ze_result_t (ZE_APICALL *ze_pfnVirtualMemSetAccessAttribute_t)(
    ze_context_handle_t,
    const void*,
    size_t,
    ze_memory_access_attribute_t
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeVirtualMemGetAccessAttribute 
typedef ze_result_t (ZE_APICALL *ze_pfnVirtualMemGetAccessAttribute_t)(
    ze_context_handle_t,
    const void*,
    size_t,
    ze_memory_access_attribute_t*,
    size_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of VirtualMem functions pointers
typedef struct _ze_virtual_mem_dditable_t
{
    ze_pfnVirtualMemReserve_t                                   pfnReserve;
    ze_pfnVirtualMemFree_t                                      pfnFree;
    ze_pfnVirtualMemQueryPageSize_t                             pfnQueryPageSize;
    ze_pfnVirtualMemMap_t                                       pfnMap;
    ze_pfnVirtualMemUnmap_t                                     pfnUnmap;
    ze_pfnVirtualMemSetAccessAttribute_t                        pfnSetAccessAttribute;
    ze_pfnVirtualMemGetAccessAttribute_t                        pfnGetAccessAttribute;
} ze_virtual_mem_dditable_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's VirtualMem table
///        with current process' addresses
///
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_VERSION
ZE_DLLEXPORT ze_result_t ZE_APICALL
zeGetVirtualMemProcAddrTable(
    ze_api_version_t version,                       ///< [in] API version requested
    ze_virtual_mem_dditable_t* pDdiTable            ///< [in,out] pointer to table of DDI function pointers
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zeGetVirtualMemProcAddrTable
typedef ze_result_t (ZE_APICALL *ze_pfnGetVirtualMemProcAddrTable_t)(
    ze_api_version_t,
    ze_virtual_mem_dditable_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Container for all DDI tables
typedef struct _ze_dditable_t
{
    ze_global_dditable_t                Global;
    ze_driver_dditable_t                Driver;
    ze_device_dditable_t                Device;
    ze_context_dditable_t               Context;
    ze_command_queue_dditable_t         CommandQueue;
    ze_command_list_dditable_t          CommandList;
    ze_image_dditable_t                 Image;
    ze_image_exp_dditable_t             ImageExp;
    ze_fence_dditable_t                 Fence;
    ze_event_pool_dditable_t            EventPool;
    ze_event_dditable_t                 Event;
    ze_event_exp_dditable_t             EventExp;
    ze_module_dditable_t                Module;
    ze_module_build_log_dditable_t      ModuleBuildLog;
    ze_kernel_dditable_t                Kernel;
    ze_kernel_exp_dditable_t            KernelExp;
    ze_sampler_dditable_t               Sampler;
    ze_physical_mem_dditable_t          PhysicalMem;
    ze_mem_dditable_t                   Mem;
    ze_virtual_mem_dditable_t           VirtualMem;
} ze_dditable_t;

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // _ZE_DDI_H