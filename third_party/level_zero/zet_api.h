/*
 *
 * Copyright (C) 2019-2021 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 * @file zet_api.h
 * @version v1.11-r1.11.8
 *
 */
#ifndef _ZET_API_H
#define _ZET_API_H
#if defined(__cplusplus)
#pragma once
#endif

// 'core' API headers
#include "ze_api.h"

#if defined(__cplusplus)
extern "C" {
#endif

// Intel 'oneAPI' Level-Zero Tool API common types
#if !defined(__GNUC__)
#pragma region common
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Handle to a driver instance
typedef ze_driver_handle_t zet_driver_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of device object
typedef ze_device_handle_t zet_device_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of context object
typedef ze_context_handle_t zet_context_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of command list object
typedef ze_command_list_handle_t zet_command_list_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of module object
typedef ze_module_handle_t zet_module_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of function object
typedef ze_kernel_handle_t zet_kernel_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of metric group's object
typedef struct _zet_metric_group_handle_t *zet_metric_group_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of metric's object
typedef struct _zet_metric_handle_t *zet_metric_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of metric streamer's object
typedef struct _zet_metric_streamer_handle_t *zet_metric_streamer_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of metric query pool's object
typedef struct _zet_metric_query_pool_handle_t *zet_metric_query_pool_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of metric query's object
typedef struct _zet_metric_query_handle_t *zet_metric_query_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of tracer object
typedef struct _zet_tracer_exp_handle_t *zet_tracer_exp_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Debug session handle
typedef struct _zet_debug_session_handle_t *zet_debug_session_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Defines structure types
typedef enum _zet_structure_type_t
{
    ZET_STRUCTURE_TYPE_METRIC_GROUP_PROPERTIES = 0x1,                       ///< ::zet_metric_group_properties_t
    ZET_STRUCTURE_TYPE_METRIC_PROPERTIES = 0x2,                             ///< ::zet_metric_properties_t
    ZET_STRUCTURE_TYPE_METRIC_STREAMER_DESC = 0x3,                          ///< ::zet_metric_streamer_desc_t
    ZET_STRUCTURE_TYPE_METRIC_QUERY_POOL_DESC = 0x4,                        ///< ::zet_metric_query_pool_desc_t
    ZET_STRUCTURE_TYPE_PROFILE_PROPERTIES = 0x5,                            ///< ::zet_profile_properties_t
    ZET_STRUCTURE_TYPE_DEVICE_DEBUG_PROPERTIES = 0x6,                       ///< ::zet_device_debug_properties_t
    ZET_STRUCTURE_TYPE_DEBUG_MEMORY_SPACE_DESC = 0x7,                       ///< ::zet_debug_memory_space_desc_t
    ZET_STRUCTURE_TYPE_DEBUG_REGSET_PROPERTIES = 0x8,                       ///< ::zet_debug_regset_properties_t
    ZET_STRUCTURE_TYPE_GLOBAL_METRICS_TIMESTAMPS_EXP_PROPERTIES = 0x9,      ///< ::zet_metric_global_timestamps_resolution_exp_t. Deprecated, use
                                                                            ///< ::ZET_STRUCTURE_TYPE_METRIC_GLOBAL_TIMESTAMPS_RESOLUTION_EXP.
    ZET_STRUCTURE_TYPE_METRIC_GLOBAL_TIMESTAMPS_RESOLUTION_EXP = 0x9,       ///< ::zet_metric_global_timestamps_resolution_exp_t
    ZET_STRUCTURE_TYPE_TRACER_EXP_DESC = 0x00010001,                        ///< ::zet_tracer_exp_desc_t
    ZET_STRUCTURE_TYPE_METRICS_CALCULATE_EXP_DESC = 0x00010002,             ///< ::zet_metric_calculate_exp_desc_t. Deprecated, use
                                                                            ///< ::ZET_STRUCTURE_TYPE_METRIC_CALCULATE_EXP_DESC.
    ZET_STRUCTURE_TYPE_METRIC_CALCULATE_EXP_DESC = 0x00010002,              ///< ::zet_metric_calculate_exp_desc_t
    ZET_STRUCTURE_TYPE_METRIC_PROGRAMMABLE_EXP_PROPERTIES = 0x00010003,     ///< ::zet_metric_programmable_exp_properties_t
    ZET_STRUCTURE_TYPE_METRIC_PROGRAMMABLE_PARAM_INFO_EXP = 0x00010004,     ///< ::zet_metric_programmable_param_info_exp_t
    ZET_STRUCTURE_TYPE_METRIC_PROGRAMMABLE_PARAM_VALUE_INFO_EXP = 0x00010005,   ///< ::zet_metric_programmable_param_value_info_exp_t
    ZET_STRUCTURE_TYPE_METRIC_GROUP_TYPE_EXP = 0x00010006,                  ///< ::zet_metric_group_type_exp_t
    ZET_STRUCTURE_TYPE_EXPORT_DMA_EXP_PROPERTIES = 0x00010007,              ///< ::zet_export_dma_buf_exp_properties_t
    ZET_STRUCTURE_TYPE_METRIC_TRACER_EXP_DESC = 0x00010008,                 ///< ::zet_metric_tracer_exp_desc_t
    ZET_STRUCTURE_TYPE_FORCE_UINT32 = 0x7fffffff

} zet_structure_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Base for all properties types
typedef struct _zet_base_properties_t
{
    zet_structure_type_t stype;                                             ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).

} zet_base_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Base for all descriptor types
typedef struct _zet_base_desc_t
{
    zet_structure_type_t stype;                                             ///< [in] type of this structure
    const void* pNext;                                                      ///< [in][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).

} zet_base_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported value types
typedef enum _zet_value_type_t
{
    ZET_VALUE_TYPE_UINT32 = 0,                                              ///< 32-bit unsigned-integer
    ZET_VALUE_TYPE_UINT64 = 1,                                              ///< 64-bit unsigned-integer
    ZET_VALUE_TYPE_FLOAT32 = 2,                                             ///< 32-bit floating-point
    ZET_VALUE_TYPE_FLOAT64 = 3,                                             ///< 64-bit floating-point
    ZET_VALUE_TYPE_BOOL8 = 4,                                               ///< 8-bit boolean
    ZET_VALUE_TYPE_STRING = 5,                                              ///< C string
    ZET_VALUE_TYPE_UINT8 = 6,                                               ///< 8-bit unsigned-integer
    ZET_VALUE_TYPE_UINT16 = 7,                                              ///< 16-bit unsigned-integer
    ZET_VALUE_TYPE_FORCE_UINT32 = 0x7fffffff

} zet_value_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Union of values
typedef union _zet_value_t
{
    uint32_t ui32;                                                          ///< [out] 32-bit unsigned-integer
    uint64_t ui64;                                                          ///< [out] 64-bit unsigned-integer
    float fp32;                                                             ///< [out] 32-bit floating-point
    double fp64;                                                            ///< [out] 64-bit floating-point
    ze_bool_t b8;                                                           ///< [out] 8-bit boolean

} zet_value_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Typed value
typedef struct _zet_typed_value_t
{
    zet_value_type_t type;                                                  ///< [out] type of value
    zet_value_t value;                                                      ///< [out] value

} zet_typed_value_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Enables driver instrumentation and dependencies for device metrics

///////////////////////////////////////////////////////////////////////////////
/// @brief Enables driver instrumentation and dependencies for program
///        instrumentation

///////////////////////////////////////////////////////////////////////////////
/// @brief Enables driver instrumentation and dependencies for program debugging

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zet_base_properties_t
typedef struct _zet_base_properties_t zet_base_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zet_base_desc_t
typedef struct _zet_base_desc_t zet_base_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zet_typed_value_t
typedef struct _zet_typed_value_t zet_typed_value_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zet_device_debug_properties_t
typedef struct _zet_device_debug_properties_t zet_device_debug_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zet_debug_config_t
typedef struct _zet_debug_config_t zet_debug_config_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zet_debug_event_info_detached_t
typedef struct _zet_debug_event_info_detached_t zet_debug_event_info_detached_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zet_debug_event_info_module_t
typedef struct _zet_debug_event_info_module_t zet_debug_event_info_module_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zet_debug_event_info_thread_stopped_t
typedef struct _zet_debug_event_info_thread_stopped_t zet_debug_event_info_thread_stopped_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zet_debug_event_info_page_fault_t
typedef struct _zet_debug_event_info_page_fault_t zet_debug_event_info_page_fault_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zet_debug_event_t
typedef struct _zet_debug_event_t zet_debug_event_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zet_debug_memory_space_desc_t
typedef struct _zet_debug_memory_space_desc_t zet_debug_memory_space_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zet_debug_regset_properties_t
typedef struct _zet_debug_regset_properties_t zet_debug_regset_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zet_metric_group_properties_t
typedef struct _zet_metric_group_properties_t zet_metric_group_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zet_metric_properties_t
typedef struct _zet_metric_properties_t zet_metric_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zet_metric_streamer_desc_t
typedef struct _zet_metric_streamer_desc_t zet_metric_streamer_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zet_metric_query_pool_desc_t
typedef struct _zet_metric_query_pool_desc_t zet_metric_query_pool_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zet_profile_properties_t
typedef struct _zet_profile_properties_t zet_profile_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zet_profile_free_register_token_t
typedef struct _zet_profile_free_register_token_t zet_profile_free_register_token_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zet_profile_register_sequence_t
typedef struct _zet_profile_register_sequence_t zet_profile_register_sequence_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zet_tracer_exp_desc_t
typedef struct _zet_tracer_exp_desc_t zet_tracer_exp_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zet_metric_tracer_exp_desc_t
typedef struct _zet_metric_tracer_exp_desc_t zet_metric_tracer_exp_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zet_metric_entry_exp_t
typedef struct _zet_metric_entry_exp_t zet_metric_entry_exp_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zet_metric_group_type_exp_t
typedef struct _zet_metric_group_type_exp_t zet_metric_group_type_exp_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zet_export_dma_buf_exp_properties_t
typedef struct _zet_export_dma_buf_exp_properties_t zet_export_dma_buf_exp_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zet_metric_global_timestamps_resolution_exp_t
typedef struct _zet_metric_global_timestamps_resolution_exp_t zet_metric_global_timestamps_resolution_exp_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zet_metric_calculate_exp_desc_t
typedef struct _zet_metric_calculate_exp_desc_t zet_metric_calculate_exp_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zet_metric_programmable_exp_properties_t
typedef struct _zet_metric_programmable_exp_properties_t zet_metric_programmable_exp_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zet_value_uint64_range_exp_t
typedef struct _zet_value_uint64_range_exp_t zet_value_uint64_range_exp_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zet_value_fp64_range_exp_t
typedef struct _zet_value_fp64_range_exp_t zet_value_fp64_range_exp_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zet_metric_programmable_param_info_exp_t
typedef struct _zet_metric_programmable_param_info_exp_t zet_metric_programmable_param_info_exp_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zet_metric_programmable_param_value_info_exp_t
typedef struct _zet_metric_programmable_param_value_info_exp_t zet_metric_programmable_param_value_info_exp_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zet_metric_programmable_param_value_exp_t
typedef struct _zet_metric_programmable_param_value_exp_t zet_metric_programmable_param_value_exp_t;


#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Tool APIs for Device
#if !defined(__GNUC__)
#pragma region device
#endif
#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Tool APIs for Context
#if !defined(__GNUC__)
#pragma region context
#endif
#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Tool APIs for Command List
#if !defined(__GNUC__)
#pragma region cmdlist
#endif
#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Tool APIs for Module
#if !defined(__GNUC__)
#pragma region module
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Supported module debug info formats.
typedef enum _zet_module_debug_info_format_t
{
    ZET_MODULE_DEBUG_INFO_FORMAT_ELF_DWARF = 0,                             ///< Format is ELF/DWARF
    ZET_MODULE_DEBUG_INFO_FORMAT_FORCE_UINT32 = 0x7fffffff

} zet_module_debug_info_format_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieve debug info from module.
/// 
/// @details
///     - The caller can pass nullptr for pDebugInfo when querying only for
///       size.
///     - The implementation will copy the native binary into a buffer supplied
///       by the caller.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hModule`
///     - ::ZE_RESULT_ERROR_INVALID_ENUMERATION
///         + `::ZET_MODULE_DEBUG_INFO_FORMAT_ELF_DWARF < format`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pSize`
ZE_APIEXPORT ze_result_t ZE_APICALL
zetModuleGetDebugInfo(
    zet_module_handle_t hModule,                                            ///< [in] handle of the module
    zet_module_debug_info_format_t format,                                  ///< [in] debug info format requested
    size_t* pSize,                                                          ///< [in,out] size of debug info in bytes
    uint8_t* pDebugInfo                                                     ///< [in,out][optional] byte pointer to debug info
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Tool APIs for Program Debug
#if !defined(__GNUC__)
#pragma region debug
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Supported device debug property flags
typedef uint32_t zet_device_debug_property_flags_t;
typedef enum _zet_device_debug_property_flag_t
{
    ZET_DEVICE_DEBUG_PROPERTY_FLAG_ATTACH = ZE_BIT(0),                      ///< the device supports attaching for debug
    ZET_DEVICE_DEBUG_PROPERTY_FLAG_FORCE_UINT32 = 0x7fffffff

} zet_device_debug_property_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Device debug properties queried using ::zetDeviceGetDebugProperties.
typedef struct _zet_device_debug_properties_t
{
    zet_structure_type_t stype;                                             ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    zet_device_debug_property_flags_t flags;                                ///< [out] returns 0 (none) or a valid combination of
                                                                            ///< ::zet_device_debug_property_flag_t

} zet_device_debug_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves debug properties of the device.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pDebugProperties`
ZE_APIEXPORT ze_result_t ZE_APICALL
zetDeviceGetDebugProperties(
    zet_device_handle_t hDevice,                                            ///< [in] device handle
    zet_device_debug_properties_t* pDebugProperties                         ///< [in,out] query result for debug properties
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Debug configuration provided to ::zetDebugAttach
typedef struct _zet_debug_config_t
{
    uint32_t pid;                                                           ///< [in] the host process identifier

} zet_debug_config_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Attach to a device.
/// 
/// @details
///     - The device must be enabled for debug; see
///       ::zesSchedulerSetComputeUnitDebugMode.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == config`
///         + `nullptr == phDebug`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + attaching to this device is not supported
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + caller does not have sufficient permissions
///     - ::ZE_RESULT_ERROR_NOT_AVAILABLE
///         + a debugger is already attached
ZE_APIEXPORT ze_result_t ZE_APICALL
zetDebugAttach(
    zet_device_handle_t hDevice,                                            ///< [in] device handle
    const zet_debug_config_t* config,                                       ///< [in] the debug configuration
    zet_debug_session_handle_t* phDebug                                     ///< [out] debug session handle
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Close a debug session.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDebug`
ZE_APIEXPORT ze_result_t ZE_APICALL
zetDebugDetach(
    zet_debug_session_handle_t hDebug                                       ///< [in][release] debug session handle
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported debug event flags.
typedef uint32_t zet_debug_event_flags_t;
typedef enum _zet_debug_event_flag_t
{
    ZET_DEBUG_EVENT_FLAG_NEED_ACK = ZE_BIT(0),                              ///< The event needs to be acknowledged by calling
                                                                            ///< ::zetDebugAcknowledgeEvent.
    ZET_DEBUG_EVENT_FLAG_FORCE_UINT32 = 0x7fffffff

} zet_debug_event_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported debug event types.
typedef enum _zet_debug_event_type_t
{
    ZET_DEBUG_EVENT_TYPE_INVALID = 0,                                       ///< The event is invalid
    ZET_DEBUG_EVENT_TYPE_DETACHED = 1,                                      ///< The tool was detached
    ZET_DEBUG_EVENT_TYPE_PROCESS_ENTRY = 2,                                 ///< The debuggee process created command queues on the device
    ZET_DEBUG_EVENT_TYPE_PROCESS_EXIT = 3,                                  ///< The debuggee process destroyed all command queues on the device
    ZET_DEBUG_EVENT_TYPE_MODULE_LOAD = 4,                                   ///< An in-memory module was loaded onto the device
    ZET_DEBUG_EVENT_TYPE_MODULE_UNLOAD = 5,                                 ///< An in-memory module is about to get unloaded from the device
    ZET_DEBUG_EVENT_TYPE_THREAD_STOPPED = 6,                                ///< The thread stopped due to a device exception
    ZET_DEBUG_EVENT_TYPE_THREAD_UNAVAILABLE = 7,                            ///< The thread is not available to be stopped
    ZET_DEBUG_EVENT_TYPE_PAGE_FAULT = 8,                                    ///< A page request could not be completed on the device
    ZET_DEBUG_EVENT_TYPE_FORCE_UINT32 = 0x7fffffff

} zet_debug_event_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported debug detach reasons.
typedef enum _zet_debug_detach_reason_t
{
    ZET_DEBUG_DETACH_REASON_INVALID = 0,                                    ///< The detach reason is not valid
    ZET_DEBUG_DETACH_REASON_HOST_EXIT = 1,                                  ///< The host process exited
    ZET_DEBUG_DETACH_REASON_FORCE_UINT32 = 0x7fffffff

} zet_debug_detach_reason_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Event information for ::ZET_DEBUG_EVENT_TYPE_DETACHED
typedef struct _zet_debug_event_info_detached_t
{
    zet_debug_detach_reason_t reason;                                       ///< [out] the detach reason

} zet_debug_event_info_detached_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Event information for ::ZET_DEBUG_EVENT_TYPE_MODULE_LOAD and
///        ::ZET_DEBUG_EVENT_TYPE_MODULE_UNLOAD
typedef struct _zet_debug_event_info_module_t
{
    zet_module_debug_info_format_t format;                                  ///< [out] the module format
    uint64_t moduleBegin;                                                   ///< [out] the begin address of the in-memory module (inclusive)
    uint64_t moduleEnd;                                                     ///< [out] the end address of the in-memory module (exclusive)
    uint64_t load;                                                          ///< [out] the load address of the module on the device

} zet_debug_event_info_module_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Event information for ::ZET_DEBUG_EVENT_TYPE_THREAD_STOPPED and
///        ::ZET_DEBUG_EVENT_TYPE_THREAD_UNAVAILABLE
typedef struct _zet_debug_event_info_thread_stopped_t
{
    ze_device_thread_t thread;                                              ///< [out] the stopped/unavailable thread

} zet_debug_event_info_thread_stopped_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Page fault reasons.
typedef enum _zet_debug_page_fault_reason_t
{
    ZET_DEBUG_PAGE_FAULT_REASON_INVALID = 0,                                ///< The page fault reason is not valid
    ZET_DEBUG_PAGE_FAULT_REASON_MAPPING_ERROR = 1,                          ///< The address is not mapped
    ZET_DEBUG_PAGE_FAULT_REASON_PERMISSION_ERROR = 2,                       ///< Invalid access permissions
    ZET_DEBUG_PAGE_FAULT_REASON_FORCE_UINT32 = 0x7fffffff

} zet_debug_page_fault_reason_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Event information for ::ZET_DEBUG_EVENT_TYPE_PAGE_FAULT
typedef struct _zet_debug_event_info_page_fault_t
{
    uint64_t address;                                                       ///< [out] the faulting address
    uint64_t mask;                                                          ///< [out] the alignment mask
    zet_debug_page_fault_reason_t reason;                                   ///< [out] the page fault reason

} zet_debug_event_info_page_fault_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Event type-specific information
typedef union _zet_debug_event_info_t
{
    zet_debug_event_info_detached_t detached;                               ///< [out] type == ::ZET_DEBUG_EVENT_TYPE_DETACHED
    zet_debug_event_info_module_t module;                                   ///< [out] type == ::ZET_DEBUG_EVENT_TYPE_MODULE_LOAD or
                                                                            ///< ::ZET_DEBUG_EVENT_TYPE_MODULE_UNLOAD
    zet_debug_event_info_thread_stopped_t thread;                           ///< [out] type == ::ZET_DEBUG_EVENT_TYPE_THREAD_STOPPED or
                                                                            ///< ::ZET_DEBUG_EVENT_TYPE_THREAD_UNAVAILABLE
    zet_debug_event_info_page_fault_t page_fault;                           ///< [out] type == ::ZET_DEBUG_EVENT_TYPE_PAGE_FAULT

} zet_debug_event_info_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief A debug event on the device.
typedef struct _zet_debug_event_t
{
    zet_debug_event_type_t type;                                            ///< [out] the event type
    zet_debug_event_flags_t flags;                                          ///< [out] returns 0 (none) or a combination of ::zet_debug_event_flag_t
    zet_debug_event_info_t info;                                            ///< [out] event type specific information

} zet_debug_event_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Read the topmost debug event.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDebug`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == event`
///     - ::ZE_RESULT_NOT_READY
///         + the timeout expired
ZE_APIEXPORT ze_result_t ZE_APICALL
zetDebugReadEvent(
    zet_debug_session_handle_t hDebug,                                      ///< [in] debug session handle
    uint64_t timeout,                                                       ///< [in] if non-zero, then indicates the maximum time (in milliseconds) to
                                                                            ///< yield before returning ::ZE_RESULT_SUCCESS or ::ZE_RESULT_NOT_READY;
                                                                            ///< if zero, then immediately returns the status of the event;
                                                                            ///< if `UINT64_MAX`, then function will not return until complete or
                                                                            ///< device is lost.
                                                                            ///< Due to external dependencies, timeout may be rounded to the closest
                                                                            ///< value allowed by the accuracy of those dependencies.
    zet_debug_event_t* event                                                ///< [in,out] a pointer to a ::zet_debug_event_t.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Acknowledge a debug event.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDebug`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == event`
ZE_APIEXPORT ze_result_t ZE_APICALL
zetDebugAcknowledgeEvent(
    zet_debug_session_handle_t hDebug,                                      ///< [in] debug session handle
    const zet_debug_event_t* event                                          ///< [in] a pointer to a ::zet_debug_event_t.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Interrupt device threads.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDebug`
///     - ::ZE_RESULT_ERROR_NOT_AVAILABLE
///         + the thread is already stopped or unavailable
ZE_APIEXPORT ze_result_t ZE_APICALL
zetDebugInterrupt(
    zet_debug_session_handle_t hDebug,                                      ///< [in] debug session handle
    ze_device_thread_t thread                                               ///< [in] the thread to interrupt
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Resume device threads.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDebug`
///     - ::ZE_RESULT_ERROR_NOT_AVAILABLE
///         + the thread is already running or unavailable
ZE_APIEXPORT ze_result_t ZE_APICALL
zetDebugResume(
    zet_debug_session_handle_t hDebug,                                      ///< [in] debug session handle
    ze_device_thread_t thread                                               ///< [in] the thread to resume
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported device memory space types.
typedef enum _zet_debug_memory_space_type_t
{
    ZET_DEBUG_MEMORY_SPACE_TYPE_DEFAULT = 0,                                ///< default memory space (attribute may be omitted)
    ZET_DEBUG_MEMORY_SPACE_TYPE_SLM = 1,                                    ///< shared local memory space (GPU-only)
    ZET_DEBUG_MEMORY_SPACE_TYPE_ELF = 2,                                    ///< ELF file memory space
    ZET_DEBUG_MEMORY_SPACE_TYPE_FORCE_UINT32 = 0x7fffffff

} zet_debug_memory_space_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Device memory space descriptor
typedef struct _zet_debug_memory_space_desc_t
{
    zet_structure_type_t stype;                                             ///< [in] type of this structure
    const void* pNext;                                                      ///< [in][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    zet_debug_memory_space_type_t type;                                     ///< [in] type of memory space
    uint64_t address;                                                       ///< [in] the virtual address within the memory space

} zet_debug_memory_space_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Read memory.
/// 
/// @details
///     - The thread identifier 'all' can be used for accessing the default
///       memory space, e.g. for setting breakpoints.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDebug`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == desc`
///         + `nullptr == buffer`
///     - ::ZE_RESULT_ERROR_INVALID_ENUMERATION
///         + `::ZET_DEBUG_MEMORY_SPACE_TYPE_ELF < desc->type`
///     - ::ZE_RESULT_ERROR_NOT_AVAILABLE
///         + the thread is running or unavailable
///         + the memory cannot be accessed from the supplied thread
ZE_APIEXPORT ze_result_t ZE_APICALL
zetDebugReadMemory(
    zet_debug_session_handle_t hDebug,                                      ///< [in] debug session handle
    ze_device_thread_t thread,                                              ///< [in] the thread identifier.
    const zet_debug_memory_space_desc_t* desc,                              ///< [in] memory space descriptor
    size_t size,                                                            ///< [in] the number of bytes to read
    void* buffer                                                            ///< [in,out] a buffer to hold a copy of the memory
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Write memory.
/// 
/// @details
///     - The thread identifier 'all' can be used for accessing the default
///       memory space, e.g. for setting breakpoints.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDebug`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == desc`
///         + `nullptr == buffer`
///     - ::ZE_RESULT_ERROR_INVALID_ENUMERATION
///         + `::ZET_DEBUG_MEMORY_SPACE_TYPE_ELF < desc->type`
///     - ::ZE_RESULT_ERROR_NOT_AVAILABLE
///         + the thread is running or unavailable
///         + the memory cannot be accessed from the supplied thread
ZE_APIEXPORT ze_result_t ZE_APICALL
zetDebugWriteMemory(
    zet_debug_session_handle_t hDebug,                                      ///< [in] debug session handle
    ze_device_thread_t thread,                                              ///< [in] the thread identifier.
    const zet_debug_memory_space_desc_t* desc,                              ///< [in] memory space descriptor
    size_t size,                                                            ///< [in] the number of bytes to write
    const void* buffer                                                      ///< [in] a buffer holding the pattern to write
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported general register set flags.
typedef uint32_t zet_debug_regset_flags_t;
typedef enum _zet_debug_regset_flag_t
{
    ZET_DEBUG_REGSET_FLAG_READABLE = ZE_BIT(0),                             ///< register set is readable
    ZET_DEBUG_REGSET_FLAG_WRITEABLE = ZE_BIT(1),                            ///< register set is writeable
    ZET_DEBUG_REGSET_FLAG_FORCE_UINT32 = 0x7fffffff

} zet_debug_regset_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Device register set properties queried using
///        ::zetDebugGetRegisterSetProperties.
typedef struct _zet_debug_regset_properties_t
{
    zet_structure_type_t stype;                                             ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    uint32_t type;                                                          ///< [out] device-specific register set type
    uint32_t version;                                                       ///< [out] device-specific version of this register set
    zet_debug_regset_flags_t generalFlags;                                  ///< [out] general register set flags
    uint32_t deviceFlags;                                                   ///< [out] device-specific register set flags
    uint32_t count;                                                         ///< [out] number of registers in the set
    uint32_t bitSize;                                                       ///< [out] the size of a register in bits
    uint32_t byteSize;                                                      ///< [out] the size required for reading or writing a register in bytes

} zet_debug_regset_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves debug register set properties.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pCount`
ZE_APIEXPORT ze_result_t ZE_APICALL
zetDebugGetRegisterSetProperties(
    zet_device_handle_t hDevice,                                            ///< [in] device handle
    uint32_t* pCount,                                                       ///< [in,out] pointer to the number of register set properties.
                                                                            ///< if count is zero, then the driver shall update the value with the
                                                                            ///< total number of register set properties available.
                                                                            ///< if count is greater than the number of register set properties
                                                                            ///< available, then the driver shall update the value with the correct
                                                                            ///< number of registry set properties available.
    zet_debug_regset_properties_t* pRegisterSetProperties                   ///< [in,out][optional][range(0, *pCount)] array of query results for
                                                                            ///< register set properties.
                                                                            ///< if count is less than the number of register set properties available,
                                                                            ///< then driver shall only retrieve that number of register set properties.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves debug register set properties for a given thread.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDebug`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pCount`
///     - ::ZE_RESULT_ERROR_NOT_AVAILABLE
///         + the thread is running or unavailable
///     - ::ZE_RESULT_ERROR_INVALID_ARGUMENT
///         + the thread argument specifies more than one or a non-existant thread
ZE_APIEXPORT ze_result_t ZE_APICALL
zetDebugGetThreadRegisterSetProperties(
    zet_debug_session_handle_t hDebug,                                      ///< [in] debug session handle
    ze_device_thread_t thread,                                              ///< [in] the thread identifier specifying a single stopped thread
    uint32_t* pCount,                                                       ///< [in,out] pointer to the number of register set properties.
                                                                            ///< if count is zero, then the driver shall update the value with the
                                                                            ///< total number of register set properties available.
                                                                            ///< if count is greater than the number of register set properties
                                                                            ///< available, then the driver shall update the value with the correct
                                                                            ///< number of registry set properties available.
    zet_debug_regset_properties_t* pRegisterSetProperties                   ///< [in,out][optional][range(0, *pCount)] array of query results for
                                                                            ///< register set properties.
                                                                            ///< if count is less than the number of register set properties available,
                                                                            ///< then driver shall only retrieve that number of register set properties.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Read register state.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDebug`
///     - ::ZE_RESULT_ERROR_NOT_AVAILABLE
///         + the thread is running or unavailable
ZE_APIEXPORT ze_result_t ZE_APICALL
zetDebugReadRegisters(
    zet_debug_session_handle_t hDebug,                                      ///< [in] debug session handle
    ze_device_thread_t thread,                                              ///< [in] the thread identifier
    uint32_t type,                                                          ///< [in] register set type
    uint32_t start,                                                         ///< [in] the starting offset into the register state area; must be less
                                                                            ///< than the `count` member of ::zet_debug_regset_properties_t for the
                                                                            ///< type
    uint32_t count,                                                         ///< [in] the number of registers to read; start+count must be less than or
                                                                            ///< equal to the `count` member of ::zet_debug_register_group_properties_t
                                                                            ///< for the type
    void* pRegisterValues                                                   ///< [in,out][optional][range(0, count)] buffer of register values
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Write register state.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDebug`
///     - ::ZE_RESULT_ERROR_NOT_AVAILABLE
///         + the thread is running or unavailable
ZE_APIEXPORT ze_result_t ZE_APICALL
zetDebugWriteRegisters(
    zet_debug_session_handle_t hDebug,                                      ///< [in] debug session handle
    ze_device_thread_t thread,                                              ///< [in] the thread identifier
    uint32_t type,                                                          ///< [in] register set type
    uint32_t start,                                                         ///< [in] the starting offset into the register state area; must be less
                                                                            ///< than the `count` member of ::zet_debug_regset_properties_t for the
                                                                            ///< type
    uint32_t count,                                                         ///< [in] the number of registers to write; start+count must be less than
                                                                            ///< or equal to the `count` member of
                                                                            ///< ::zet_debug_register_group_properties_t for the type
    void* pRegisterValues                                                   ///< [in,out][optional][range(0, count)] buffer of register values
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Tool APIs for Metric
#if !defined(__GNUC__)
#pragma region metric
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves metric group for a device.
/// 
/// @details
///     - The application may call this function from simultaneous threads.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pCount`
ZE_APIEXPORT ze_result_t ZE_APICALL
zetMetricGroupGet(
    zet_device_handle_t hDevice,                                            ///< [in] handle of the device
    uint32_t* pCount,                                                       ///< [in,out] pointer to the number of metric groups.
                                                                            ///< if count is zero, then the driver shall update the value with the
                                                                            ///< total number of metric groups available.
                                                                            ///< if count is greater than the number of metric groups available, then
                                                                            ///< the driver shall update the value with the correct number of metric
                                                                            ///< groups available.
    zet_metric_group_handle_t* phMetricGroups                               ///< [in,out][optional][range(0, *pCount)] array of handle of metric groups.
                                                                            ///< if count is less than the number of metric groups available, then
                                                                            ///< driver shall only retrieve that number of metric groups.
    );

///////////////////////////////////////////////////////////////////////////////
#ifndef ZET_MAX_METRIC_GROUP_NAME
/// @brief Maximum metric group name string size
#define ZET_MAX_METRIC_GROUP_NAME  256
#endif // ZET_MAX_METRIC_GROUP_NAME

///////////////////////////////////////////////////////////////////////////////
#ifndef ZET_MAX_METRIC_GROUP_DESCRIPTION
/// @brief Maximum metric group description string size
#define ZET_MAX_METRIC_GROUP_DESCRIPTION  256
#endif // ZET_MAX_METRIC_GROUP_DESCRIPTION

///////////////////////////////////////////////////////////////////////////////
/// @brief Metric group sampling type
typedef uint32_t zet_metric_group_sampling_type_flags_t;
typedef enum _zet_metric_group_sampling_type_flag_t
{
    ZET_METRIC_GROUP_SAMPLING_TYPE_FLAG_EVENT_BASED = ZE_BIT(0),            ///< Event based sampling
    ZET_METRIC_GROUP_SAMPLING_TYPE_FLAG_TIME_BASED = ZE_BIT(1),             ///< Time based sampling
    ZET_METRIC_GROUP_SAMPLING_TYPE_FLAG_EXP_TRACER_BASED = ZE_BIT(2),       ///< Experimental Tracer based sampling
    ZET_METRIC_GROUP_SAMPLING_TYPE_FLAG_FORCE_UINT32 = 0x7fffffff

} zet_metric_group_sampling_type_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Metric group properties queried using ::zetMetricGroupGetProperties
typedef struct _zet_metric_group_properties_t
{
    zet_structure_type_t stype;                                             ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    char name[ZET_MAX_METRIC_GROUP_NAME];                                   ///< [out] metric group name
    char description[ZET_MAX_METRIC_GROUP_DESCRIPTION];                     ///< [out] metric group description
    zet_metric_group_sampling_type_flags_t samplingType;                    ///< [out] metric group sampling type.
                                                                            ///< returns a combination of ::zet_metric_group_sampling_type_flag_t.
    uint32_t domain;                                                        ///< [out] metric group domain number. Cannot use multiple, simultaneous
                                                                            ///< metric groups from the same domain.
    uint32_t metricCount;                                                   ///< [out] metric count belonging to this group

} zet_metric_group_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves attributes of a metric group.
/// 
/// @details
///     - The application may call this function from simultaneous threads.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hMetricGroup`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pProperties`
ZE_APIEXPORT ze_result_t ZE_APICALL
zetMetricGroupGetProperties(
    zet_metric_group_handle_t hMetricGroup,                                 ///< [in] handle of the metric group
    zet_metric_group_properties_t* pProperties                              ///< [in,out] metric group properties
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Metric types
typedef enum _zet_metric_type_t
{
    ZET_METRIC_TYPE_DURATION = 0,                                           ///< Metric type: duration
    ZET_METRIC_TYPE_EVENT = 1,                                              ///< Metric type: event
    ZET_METRIC_TYPE_EVENT_WITH_RANGE = 2,                                   ///< Metric type: event with range
    ZET_METRIC_TYPE_THROUGHPUT = 3,                                         ///< Metric type: throughput
    ZET_METRIC_TYPE_TIMESTAMP = 4,                                          ///< Metric type: timestamp
    ZET_METRIC_TYPE_FLAG = 5,                                               ///< Metric type: flag
    ZET_METRIC_TYPE_RATIO = 6,                                              ///< Metric type: ratio
    ZET_METRIC_TYPE_RAW = 7,                                                ///< Metric type: raw
    ZET_METRIC_TYPE_EVENT_EXP_TIMESTAMP = 0x7ffffff9,                       ///< Metric type: event with only timestamp and value has no meaning
    ZET_METRIC_TYPE_EVENT_EXP_START = 0x7ffffffa,                           ///< Metric type: the first event of a start/end event pair
    ZET_METRIC_TYPE_EVENT_EXP_END = 0x7ffffffb,                             ///< Metric type: the second event of a start/end event pair
    ZET_METRIC_TYPE_EVENT_EXP_MONOTONIC_WRAPS_VALUE = 0x7ffffffc,           ///< Metric type: value of the event is a monotonically increasing value
                                                                            ///< that can wrap around
    ZET_METRIC_TYPE_EXP_EXPORT_DMA_BUF = 0x7ffffffd,                        ///< Metric which exports linux dma_buf, which could be imported/mapped to
                                                                            ///< the host process
    ZET_METRIC_TYPE_IP_EXP = 0x7ffffffe,                                    ///< Metric type: instruction pointer. Deprecated, use
                                                                            ///< ::ZET_METRIC_TYPE_IP.
    ZET_METRIC_TYPE_IP = 0x7ffffffe,                                        ///< Metric type: instruction pointer
    ZET_METRIC_TYPE_FORCE_UINT32 = 0x7fffffff

} zet_metric_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Metric group calculation type
typedef enum _zet_metric_group_calculation_type_t
{
    ZET_METRIC_GROUP_CALCULATION_TYPE_METRIC_VALUES = 0,                    ///< Calculated metric values from raw data.
    ZET_METRIC_GROUP_CALCULATION_TYPE_MAX_METRIC_VALUES = 1,                ///< Maximum metric values.
    ZET_METRIC_GROUP_CALCULATION_TYPE_FORCE_UINT32 = 0x7fffffff

} zet_metric_group_calculation_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Calculates metric values from raw data.
/// 
/// @details
///     - The application may call this function from simultaneous threads.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hMetricGroup`
///     - ::ZE_RESULT_ERROR_INVALID_ENUMERATION
///         + `::ZET_METRIC_GROUP_CALCULATION_TYPE_MAX_METRIC_VALUES < type`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pRawData`
///         + `nullptr == pMetricValueCount`
ZE_APIEXPORT ze_result_t ZE_APICALL
zetMetricGroupCalculateMetricValues(
    zet_metric_group_handle_t hMetricGroup,                                 ///< [in] handle of the metric group
    zet_metric_group_calculation_type_t type,                               ///< [in] calculation type to be applied on raw data
    size_t rawDataSize,                                                     ///< [in] size in bytes of raw data buffer
    const uint8_t* pRawData,                                                ///< [in][range(0, rawDataSize)] buffer of raw data to calculate
    uint32_t* pMetricValueCount,                                            ///< [in,out] pointer to number of metric values calculated.
                                                                            ///< if count is zero, then the driver shall update the value with the
                                                                            ///< total number of metric values to be calculated.
                                                                            ///< if count is greater than the number available in the raw data buffer,
                                                                            ///< then the driver shall update the value with the actual number of
                                                                            ///< metric values to be calculated.
    zet_typed_value_t* pMetricValues                                        ///< [in,out][optional][range(0, *pMetricValueCount)] buffer of calculated metrics.
                                                                            ///< if count is less than the number available in the raw data buffer,
                                                                            ///< then driver shall only calculate that number of metric values.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves metric from a metric group.
/// 
/// @details
///     - The application may call this function from simultaneous threads.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hMetricGroup`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pCount`
ZE_APIEXPORT ze_result_t ZE_APICALL
zetMetricGet(
    zet_metric_group_handle_t hMetricGroup,                                 ///< [in] handle of the metric group
    uint32_t* pCount,                                                       ///< [in,out] pointer to the number of metrics.
                                                                            ///< if count is zero, then the driver shall update the value with the
                                                                            ///< total number of metrics available.
                                                                            ///< if count is greater than the number of metrics available, then the
                                                                            ///< driver shall update the value with the correct number of metrics available.
    zet_metric_handle_t* phMetrics                                          ///< [in,out][optional][range(0, *pCount)] array of handle of metrics.
                                                                            ///< if count is less than the number of metrics available, then driver
                                                                            ///< shall only retrieve that number of metrics.
    );

///////////////////////////////////////////////////////////////////////////////
#ifndef ZET_MAX_METRIC_NAME
/// @brief Maximum metric name string size
#define ZET_MAX_METRIC_NAME  256
#endif // ZET_MAX_METRIC_NAME

///////////////////////////////////////////////////////////////////////////////
#ifndef ZET_MAX_METRIC_DESCRIPTION
/// @brief Maximum metric description string size
#define ZET_MAX_METRIC_DESCRIPTION  256
#endif // ZET_MAX_METRIC_DESCRIPTION

///////////////////////////////////////////////////////////////////////////////
#ifndef ZET_MAX_METRIC_COMPONENT
/// @brief Maximum metric component string size
#define ZET_MAX_METRIC_COMPONENT  256
#endif // ZET_MAX_METRIC_COMPONENT

///////////////////////////////////////////////////////////////////////////////
#ifndef ZET_MAX_METRIC_RESULT_UNITS
/// @brief Maximum metric result units string size
#define ZET_MAX_METRIC_RESULT_UNITS  256
#endif // ZET_MAX_METRIC_RESULT_UNITS

///////////////////////////////////////////////////////////////////////////////
/// @brief Metric properties queried using ::zetMetricGetProperties
typedef struct _zet_metric_properties_t
{
    zet_structure_type_t stype;                                             ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    char name[ZET_MAX_METRIC_NAME];                                         ///< [out] metric name
    char description[ZET_MAX_METRIC_DESCRIPTION];                           ///< [out] metric description
    char component[ZET_MAX_METRIC_COMPONENT];                               ///< [out] metric component
    uint32_t tierNumber;                                                    ///< [out] number of tier
    zet_metric_type_t metricType;                                           ///< [out] metric type
    zet_value_type_t resultType;                                            ///< [out] metric result type
    char resultUnits[ZET_MAX_METRIC_RESULT_UNITS];                          ///< [out] metric result units

} zet_metric_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves attributes of a metric.
/// 
/// @details
///     - The application may call this function from simultaneous threads.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hMetric`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pProperties`
ZE_APIEXPORT ze_result_t ZE_APICALL
zetMetricGetProperties(
    zet_metric_handle_t hMetric,                                            ///< [in] handle of the metric
    zet_metric_properties_t* pProperties                                    ///< [in,out] metric properties
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Activates metric groups.
/// 
/// @details
///     - Immediately reconfigures the device to activate only those metric
///       groups provided.
///     - Any metric groups previously activated but not provided will be
///       deactivated.
///     - Deactivating metric groups that are still in-use will result in
///       undefined behavior.
///     - All metric groups must have different domains, see
///       ::zet_metric_group_properties_t.
///     - The application must **not** call this function from simultaneous
///       threads with the same device handle.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hContext`
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_SIZE
///         + `(nullptr == phMetricGroups) && (0 < count)`
///     - ::ZE_RESULT_ERROR_INVALID_ARGUMENT
///         + Multiple metric groups share the same domain
ZE_APIEXPORT ze_result_t ZE_APICALL
zetContextActivateMetricGroups(
    zet_context_handle_t hContext,                                          ///< [in] handle of the context object
    zet_device_handle_t hDevice,                                            ///< [in] handle of the device
    uint32_t count,                                                         ///< [in] metric group count to activate; must be 0 if `nullptr ==
                                                                            ///< phMetricGroups`
    zet_metric_group_handle_t* phMetricGroups                               ///< [in][optional][range(0, count)] handles of the metric groups to activate.
                                                                            ///< nullptr deactivates all previously used metric groups.
                                                                            ///< all metrics groups must come from a different domains.
                                                                            ///< metric query and metric stream must use activated metric groups.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Metric streamer descriptor
typedef struct _zet_metric_streamer_desc_t
{
    zet_structure_type_t stype;                                             ///< [in] type of this structure
    const void* pNext;                                                      ///< [in][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    uint32_t notifyEveryNReports;                                           ///< [in,out] number of collected reports after which notification event
                                                                            ///< will be signaled. If the requested value is not supported exactly,
                                                                            ///< then the driver may use a value that is the closest supported
                                                                            ///< approximation and shall update this member during ::zetMetricStreamerOpen.
    uint32_t samplingPeriod;                                                ///< [in,out] streamer sampling period in nanoseconds. If the requested
                                                                            ///< value is not supported exactly, then the driver may use a value that
                                                                            ///< is the closest supported approximation and shall update this member
                                                                            ///< during ::zetMetricStreamerOpen.

} zet_metric_streamer_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Opens metric streamer for a device.
/// 
/// @details
///     - The notification event must have been created from an event pool that
///       was created using ::ZE_EVENT_POOL_FLAG_HOST_VISIBLE flag.
///     - The duration of the signal event created from an event pool that was
///       created using ::ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP flag is undefined.
///       However, for consistency and orthogonality the event will report
///       correctly as signaled when used by other event API functionality.
///     - The application must **not** call this function from simultaneous
///       threads with the same device handle.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hContext`
///         + `nullptr == hDevice`
///         + `nullptr == hMetricGroup`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == desc`
///         + `nullptr == phMetricStreamer`
///     - ::ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT
ZE_APIEXPORT ze_result_t ZE_APICALL
zetMetricStreamerOpen(
    zet_context_handle_t hContext,                                          ///< [in] handle of the context object
    zet_device_handle_t hDevice,                                            ///< [in] handle of the device
    zet_metric_group_handle_t hMetricGroup,                                 ///< [in] handle of the metric group
    zet_metric_streamer_desc_t* desc,                                       ///< [in,out] metric streamer descriptor
    ze_event_handle_t hNotificationEvent,                                   ///< [in][optional] event used for report availability notification
    zet_metric_streamer_handle_t* phMetricStreamer                          ///< [out] handle of metric streamer
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Append metric streamer marker into a command list.
/// 
/// @details
///     - The application must ensure the metric streamer is accessible by the
///       device on which the command list was created.
///     - The application must ensure the command list and metric streamer were
///       created on the same context.
///     - The application must **not** call this function from simultaneous
///       threads with the same command list handle.
///     - Allow to associate metric stream time based metrics with executed
///       workload.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hCommandList`
///         + `nullptr == hMetricStreamer`
ZE_APIEXPORT ze_result_t ZE_APICALL
zetCommandListAppendMetricStreamerMarker(
    zet_command_list_handle_t hCommandList,                                 ///< [in] handle of the command list
    zet_metric_streamer_handle_t hMetricStreamer,                           ///< [in] handle of the metric streamer
    uint32_t value                                                          ///< [in] streamer marker value
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Closes metric streamer.
/// 
/// @details
///     - The application must **not** call this function from simultaneous
///       threads with the same metric streamer handle.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hMetricStreamer`
ZE_APIEXPORT ze_result_t ZE_APICALL
zetMetricStreamerClose(
    zet_metric_streamer_handle_t hMetricStreamer                            ///< [in][release] handle of the metric streamer
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Reads data from metric streamer.
/// 
/// @details
///     - The application may call this function from simultaneous threads.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hMetricStreamer`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pRawDataSize`
///     - ::ZE_RESULT_WARNING_DROPPED_DATA
///         + Metric streamer data may have been dropped. Reduce sampling period.
ZE_APIEXPORT ze_result_t ZE_APICALL
zetMetricStreamerReadData(
    zet_metric_streamer_handle_t hMetricStreamer,                           ///< [in] handle of the metric streamer
    uint32_t maxReportCount,                                                ///< [in] the maximum number of reports the application wants to receive.
                                                                            ///< if `UINT32_MAX`, then function will retrieve all reports available
    size_t* pRawDataSize,                                                   ///< [in,out] pointer to size in bytes of raw data requested to read.
                                                                            ///< if size is zero, then the driver will update the value with the total
                                                                            ///< size in bytes needed for all reports available.
                                                                            ///< if size is non-zero, then driver will only retrieve the number of
                                                                            ///< reports that fit into the buffer.
                                                                            ///< if size is larger than size needed for all reports, then driver will
                                                                            ///< update the value with the actual size needed.
    uint8_t* pRawData                                                       ///< [in,out][optional][range(0, *pRawDataSize)] buffer containing streamer
                                                                            ///< reports in raw format
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Metric query pool types
typedef enum _zet_metric_query_pool_type_t
{
    ZET_METRIC_QUERY_POOL_TYPE_PERFORMANCE = 0,                             ///< Performance metric query pool.
    ZET_METRIC_QUERY_POOL_TYPE_EXECUTION = 1,                               ///< Skips workload execution between begin/end calls.
    ZET_METRIC_QUERY_POOL_TYPE_FORCE_UINT32 = 0x7fffffff

} zet_metric_query_pool_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Metric query pool description
typedef struct _zet_metric_query_pool_desc_t
{
    zet_structure_type_t stype;                                             ///< [in] type of this structure
    const void* pNext;                                                      ///< [in][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    zet_metric_query_pool_type_t type;                                      ///< [in] Query pool type.
    uint32_t count;                                                         ///< [in] Internal slots count within query pool object.

} zet_metric_query_pool_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Creates a pool of metric queries on the context.
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function must be thread-safe.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hContext`
///         + `nullptr == hDevice`
///         + `nullptr == hMetricGroup`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == desc`
///         + `nullptr == phMetricQueryPool`
///     - ::ZE_RESULT_ERROR_INVALID_ENUMERATION
///         + `::ZET_METRIC_QUERY_POOL_TYPE_EXECUTION < desc->type`
ZE_APIEXPORT ze_result_t ZE_APICALL
zetMetricQueryPoolCreate(
    zet_context_handle_t hContext,                                          ///< [in] handle of the context object
    zet_device_handle_t hDevice,                                            ///< [in] handle of the device
    zet_metric_group_handle_t hMetricGroup,                                 ///< [in] metric group associated with the query object.
    const zet_metric_query_pool_desc_t* desc,                               ///< [in] metric query pool descriptor
    zet_metric_query_pool_handle_t* phMetricQueryPool                       ///< [out] handle of metric query pool
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Deletes a query pool object.
/// 
/// @details
///     - The application must destroy all query handles created from the pool
///       before destroying the pool itself.
///     - The application must ensure the device is not currently referencing
///       the any query within the pool before it is deleted.
///     - The application must **not** call this function from simultaneous
///       threads with the same query pool handle.
///     - The implementation of this function must be thread-safe.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hMetricQueryPool`
///     - ::ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE
ZE_APIEXPORT ze_result_t ZE_APICALL
zetMetricQueryPoolDestroy(
    zet_metric_query_pool_handle_t hMetricQueryPool                         ///< [in][release] handle of the metric query pool
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Creates metric query from the pool.
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function must be thread-safe.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hMetricQueryPool`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == phMetricQuery`
ZE_APIEXPORT ze_result_t ZE_APICALL
zetMetricQueryCreate(
    zet_metric_query_pool_handle_t hMetricQueryPool,                        ///< [in] handle of the metric query pool
    uint32_t index,                                                         ///< [in] index of the query within the pool
    zet_metric_query_handle_t* phMetricQuery                                ///< [out] handle of metric query
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Deletes a metric query object.
/// 
/// @details
///     - The application must ensure the device is not currently referencing
///       the query before it is deleted.
///     - The application must **not** call this function from simultaneous
///       threads with the same query handle.
///     - The implementation of this function must be thread-safe.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hMetricQuery`
///     - ::ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE
ZE_APIEXPORT ze_result_t ZE_APICALL
zetMetricQueryDestroy(
    zet_metric_query_handle_t hMetricQuery                                  ///< [in][release] handle of metric query
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Resets a metric query object back to initial state.
/// 
/// @details
///     - The application must ensure the device is not currently referencing
///       the query before it is reset
///     - The application must **not** call this function from simultaneous
///       threads with the same query handle.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hMetricQuery`
ZE_APIEXPORT ze_result_t ZE_APICALL
zetMetricQueryReset(
    zet_metric_query_handle_t hMetricQuery                                  ///< [in] handle of metric query
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Appends metric query begin into a command list.
/// 
/// @details
///     - The application must ensure the metric query is accessible by the
///       device on which the command list was created.
///     - The application must ensure the command list and metric query were
///       created on the same context.
///     - This command blocks all following commands from beginning until the
///       execution of the query completes.
///     - The application must **not** call this function from simultaneous
///       threads with the same command list handle.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hCommandList`
///         + `nullptr == hMetricQuery`
ZE_APIEXPORT ze_result_t ZE_APICALL
zetCommandListAppendMetricQueryBegin(
    zet_command_list_handle_t hCommandList,                                 ///< [in] handle of the command list
    zet_metric_query_handle_t hMetricQuery                                  ///< [in] handle of the metric query
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Appends metric query end into a command list.
/// 
/// @details
///     - The application must ensure the metric query and events are accessible
///       by the device on which the command list was created.
///     - The application must ensure the command list, events and metric query
///       were created on the same context.
///     - The duration of the signal event created from an event pool that was
///       created using ::ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP flag is undefined.
///       However, for consistency and orthogonality the event will report
///       correctly as signaled when used by other event API functionality.
///     - If numWaitEvents is zero, then all previous commands are completed
///       prior to the execution of the query.
///     - If numWaitEvents is non-zero, then all phWaitEvents must be signaled
///       prior to the execution of the query.
///     - This command blocks all following commands from beginning until the
///       execution of the query completes.
///     - The application must **not** call this function from simultaneous
///       threads with the same command list handle.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hCommandList`
///         + `nullptr == hMetricQuery`
///     - ::ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT
///     - ::ZE_RESULT_ERROR_INVALID_SIZE
///         + `(nullptr == phWaitEvents) && (0 < numWaitEvents)`
ZE_APIEXPORT ze_result_t ZE_APICALL
zetCommandListAppendMetricQueryEnd(
    zet_command_list_handle_t hCommandList,                                 ///< [in] handle of the command list
    zet_metric_query_handle_t hMetricQuery,                                 ///< [in] handle of the metric query
    ze_event_handle_t hSignalEvent,                                         ///< [in][optional] handle of the event to signal on completion
    uint32_t numWaitEvents,                                                 ///< [in] must be zero
    ze_event_handle_t* phWaitEvents                                         ///< [in][mbz] must be nullptr
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Appends metric query commands to flush all caches.
/// 
/// @details
///     - The application must **not** call this function from simultaneous
///       threads with the same command list handle.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hCommandList`
ZE_APIEXPORT ze_result_t ZE_APICALL
zetCommandListAppendMetricMemoryBarrier(
    zet_command_list_handle_t hCommandList                                  ///< [in] handle of the command list
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves raw data for a given metric query.
/// 
/// @details
///     - The application may call this function from simultaneous threads.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hMetricQuery`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pRawDataSize`
ZE_APIEXPORT ze_result_t ZE_APICALL
zetMetricQueryGetData(
    zet_metric_query_handle_t hMetricQuery,                                 ///< [in] handle of the metric query
    size_t* pRawDataSize,                                                   ///< [in,out] pointer to size in bytes of raw data requested to read.
                                                                            ///< if size is zero, then the driver will update the value with the total
                                                                            ///< size in bytes needed for all reports available.
                                                                            ///< if size is non-zero, then driver will only retrieve the number of
                                                                            ///< reports that fit into the buffer.
                                                                            ///< if size is larger than size needed for all reports, then driver will
                                                                            ///< update the value with the actual size needed.
    uint8_t* pRawData                                                       ///< [in,out][optional][range(0, *pRawDataSize)] buffer containing query
                                                                            ///< reports in raw format
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Tool APIs for Program Instrumentation (PIN)
#if !defined(__GNUC__)
#pragma region pin
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Supportted profile features
typedef uint32_t zet_profile_flags_t;
typedef enum _zet_profile_flag_t
{
    ZET_PROFILE_FLAG_REGISTER_REALLOCATION = ZE_BIT(0),                     ///< request the compiler attempt to minimize register usage as much as
                                                                            ///< possible to allow for instrumentation
    ZET_PROFILE_FLAG_FREE_REGISTER_INFO = ZE_BIT(1),                        ///< request the compiler generate free register info
    ZET_PROFILE_FLAG_FORCE_UINT32 = 0x7fffffff

} zet_profile_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Profiling meta-data for instrumentation
typedef struct _zet_profile_properties_t
{
    zet_structure_type_t stype;                                             ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    zet_profile_flags_t flags;                                              ///< [out] indicates which flags were enabled during compilation.
                                                                            ///< returns 0 (none) or a combination of ::zet_profile_flag_t
    uint32_t numTokens;                                                     ///< [out] number of tokens immediately following this structure

} zet_profile_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported profile token types
typedef enum _zet_profile_token_type_t
{
    ZET_PROFILE_TOKEN_TYPE_FREE_REGISTER = 0,                               ///< GRF info
    ZET_PROFILE_TOKEN_TYPE_FORCE_UINT32 = 0x7fffffff

} zet_profile_token_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Profile free register token detailing unused registers in the current
///        function
typedef struct _zet_profile_free_register_token_t
{
    zet_profile_token_type_t type;                                          ///< [out] type of token
    uint32_t size;                                                          ///< [out] total size of the token, in bytes
    uint32_t count;                                                         ///< [out] number of register sequences immediately following this
                                                                            ///< structure

} zet_profile_free_register_token_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Profile register sequence detailing consecutive bytes, all of which
///        are unused
typedef struct _zet_profile_register_sequence_t
{
    uint32_t start;                                                         ///< [out] starting byte in the register table, representing the start of
                                                                            ///< unused bytes in the current function
    uint32_t count;                                                         ///< [out] number of consecutive bytes in the sequence, starting from start

} zet_profile_register_sequence_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieve profiling information generated for the kernel.
/// 
/// @details
///     - Module must be created using the following build option:
///         + "-zet-profile-flags <n>" - enable generation of profile
///           information
///         + "<n>" must be a combination of ::zet_profile_flag_t, in hex
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hKernel`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pProfileProperties`
ZE_APIEXPORT ze_result_t ZE_APICALL
zetKernelGetProfileInfo(
    zet_kernel_handle_t hKernel,                                            ///< [in] handle to kernel
    zet_profile_properties_t* pProfileProperties                            ///< [out] pointer to profile properties
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Tool Experimental Extension APIs for API Tracing
#if !defined(__GNUC__)
#pragma region tracing
#endif
///////////////////////////////////////////////////////////////////////////////
#ifndef ZET_API_TRACING_EXP_NAME
/// @brief API Tracing Experimental Extension Name
#define ZET_API_TRACING_EXP_NAME  "ZET_experimental_api_tracing"
#endif // ZET_API_TRACING_EXP_NAME

///////////////////////////////////////////////////////////////////////////////
/// @brief API Tracing Experimental Extension Version(s)
typedef enum _zet_api_tracing_exp_version_t
{
    ZET_API_TRACING_EXP_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ),              ///< version 1.0
    ZET_API_TRACING_EXP_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ),          ///< latest known version
    ZET_API_TRACING_EXP_VERSION_FORCE_UINT32 = 0x7fffffff

} zet_api_tracing_exp_version_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Alias the existing callbacks definition for 'core' callbacks
typedef ze_callbacks_t zet_core_callbacks_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Tracer descriptor
typedef struct _zet_tracer_exp_desc_t
{
    zet_structure_type_t stype;                                             ///< [in] type of this structure
    const void* pNext;                                                      ///< [in][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    void* pUserData;                                                        ///< [in] pointer passed to every tracer's callbacks

} zet_tracer_exp_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Creates a tracer on the context.
/// 
/// @details
///     - The application must only use the tracer for the context which was
///       provided during creation.
///     - The tracer is created in the disabled state.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function must be thread-safe.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hContext`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == desc`
///         + `nullptr == desc->pUserData`
///         + `nullptr == phTracer`
ZE_APIEXPORT ze_result_t ZE_APICALL
zetTracerExpCreate(
    zet_context_handle_t hContext,                                          ///< [in] handle of the context object
    const zet_tracer_exp_desc_t* desc,                                      ///< [in] pointer to tracer descriptor
    zet_tracer_exp_handle_t* phTracer                                       ///< [out] pointer to handle of tracer object created
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
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hTracer`
///     - ::ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE
ZE_APIEXPORT ze_result_t ZE_APICALL
zetTracerExpDestroy(
    zet_tracer_exp_handle_t hTracer                                         ///< [in][release] handle of tracer object to destroy
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
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hTracer`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pCoreCbs`
ZE_APIEXPORT ze_result_t ZE_APICALL
zetTracerExpSetPrologues(
    zet_tracer_exp_handle_t hTracer,                                        ///< [in] handle of the tracer
    zet_core_callbacks_t* pCoreCbs                                          ///< [in] pointer to table of 'core' callback function pointers
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
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hTracer`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pCoreCbs`
ZE_APIEXPORT ze_result_t ZE_APICALL
zetTracerExpSetEpilogues(
    zet_tracer_exp_handle_t hTracer,                                        ///< [in] handle of the tracer
    zet_core_callbacks_t* pCoreCbs                                          ///< [in] pointer to table of 'core' callback function pointers
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
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hTracer`
ZE_APIEXPORT ze_result_t ZE_APICALL
zetTracerExpSetEnabled(
    zet_tracer_exp_handle_t hTracer,                                        ///< [in] handle of the tracer
    ze_bool_t enable                                                        ///< [in] enable the tracer if true; disable if false
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Tool Experimental Extension to get Concurrent Metric Groups
#if !defined(__GNUC__)
#pragma region concurrentMetricGroup
#endif
///////////////////////////////////////////////////////////////////////////////
#ifndef ZET_CONCURRENT_METRIC_GROUPS_EXP_NAME
/// @brief Concurrent Metric Groups Experimental Extension Name
#define ZET_CONCURRENT_METRIC_GROUPS_EXP_NAME  "ZET_experimental_concurrent_metric_groups"
#endif // ZET_CONCURRENT_METRIC_GROUPS_EXP_NAME

///////////////////////////////////////////////////////////////////////////////
/// @brief Concurrent Metric Groups Experimental Extension Version(s)
typedef enum _zet_concurrent_metric_groups_exp_version_t
{
    ZET_CONCURRENT_METRIC_GROUPS_EXP_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ), ///< version 1.0
    ZET_CONCURRENT_METRIC_GROUPS_EXP_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ), ///< latest known version
    ZET_CONCURRENT_METRIC_GROUPS_EXP_VERSION_FORCE_UINT32 = 0x7fffffff

} zet_concurrent_metric_groups_exp_version_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Get sets of metric groups which could be collected concurrently.
/// 
/// @details
///     - Re-arrange the input metric groups to provide sets of concurrent
///       metric groups.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///         + `nullptr == phMetricGroups`
ZE_APIEXPORT ze_result_t ZE_APICALL
zetDeviceGetConcurrentMetricGroupsExp(
    zet_device_handle_t hDevice,                                            ///< [in] handle of the device
    uint32_t metricGroupCount,                                              ///< [in] metric group count
    zet_metric_group_handle_t * phMetricGroups,                             ///< [in,out] metrics groups to be re-arranged to be sets of concurrent
                                                                            ///< groups
    uint32_t * pMetricGroupsCountPerConcurrentGroup,                        ///< [in,out][optional][*pConcurrentGroupCount] count of metric groups per
                                                                            ///< concurrent group.
    uint32_t * pConcurrentGroupCount                                        ///< [out] number of concurrent groups.
                                                                            ///< The value of this parameter could be used to determine the number of
                                                                            ///< replays necessary.
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Tool Experimental Extension for Metrics Tracer
#if !defined(__GNUC__)
#pragma region metricTracer
#endif
///////////////////////////////////////////////////////////////////////////////
#ifndef ZET_METRICS_TRACER_EXP_NAME
/// @brief Metric Tracer Experimental Extension Name
#define ZET_METRICS_TRACER_EXP_NAME  "ZET_experimental_metric_tracer"
#endif // ZET_METRICS_TRACER_EXP_NAME

///////////////////////////////////////////////////////////////////////////////
/// @brief Metric Tracer Experimental Extension Version(s)
typedef enum _zet_metric_tracer_exp_version_t
{
    ZET_METRIC_TRACER_EXP_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ),            ///< version 1.0
    ZET_METRIC_TRACER_EXP_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ),        ///< latest known version
    ZET_METRIC_TRACER_EXP_VERSION_FORCE_UINT32 = 0x7fffffff

} zet_metric_tracer_exp_version_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of metric tracer's object
typedef struct _zet_metric_tracer_exp_handle_t *zet_metric_tracer_exp_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of metric decoder's object
typedef struct _zet_metric_decoder_exp_handle_t *zet_metric_decoder_exp_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Metric tracer descriptor
typedef struct _zet_metric_tracer_exp_desc_t
{
    zet_structure_type_t stype;                                             ///< [in] type of this structure
    const void* pNext;                                                      ///< [in][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    uint32_t notifyEveryNBytes;                                             ///< [in,out] number of collected bytes after which notification event will
                                                                            ///< be signaled. If the requested value is not supported exactly, then the
                                                                            ///< driver may use a value that is the closest supported approximation and
                                                                            ///< shall update this member during ::zetMetricTracerCreateExp.

} zet_metric_tracer_exp_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Decoded metric entry
typedef struct _zet_metric_entry_exp_t
{
    zet_value_t value;                                                      ///< [out] value of the decodable metric entry or event. Number is
                                                                            ///< meaningful based on the metric type.
    uint64_t timeStamp;                                                     ///< [out] timestamp at which the event happened.
    uint32_t metricIndex;                                                   ///< [out] index to the decodable metric handle in the input array
                                                                            ///< (phMetric) in ::zetMetricTracerDecodeExp().
    ze_bool_t onSubdevice;                                                  ///< [out] True if the event occurred on a sub-device; false means the
                                                                            ///< device on which the metric tracer was opened does not have
                                                                            ///< sub-devices.
    uint32_t subdeviceId;                                                   ///< [out] If onSubdevice is true, this gives the ID of the sub-device.

} zet_metric_entry_exp_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Create a metric tracer for a device.
/// 
/// @details
///     - The notification event must have been created from an event pool that
///       was created using ::ZE_EVENT_POOL_FLAG_HOST_VISIBLE flag.
///     - The duration of the signal event created from an event pool that was
///       created using ::ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP flag is undefined.
///       However, for consistency and orthogonality the event will report
///       correctly as signaled when used by other event API functionality.
///     - The application must **not** call this function from simultaneous
///       threads with the same device handle.
///     - The metric tracer is created in disabled state
///     - Metric groups must support sampling type
///       ZET_METRIC_SAMPLING_TYPE_EXP_FLAG_TRACER_BASED
///     - All metric groups must be first activated
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hContext`
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == phMetricGroups`
///         + `nullptr == desc`
///         + `nullptr == phMetricTracer`
///     - ::ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT
ZE_APIEXPORT ze_result_t ZE_APICALL
zetMetricTracerCreateExp(
    zet_context_handle_t hContext,                                          ///< [in] handle of the context object
    zet_device_handle_t hDevice,                                            ///< [in] handle of the device
    uint32_t metricGroupCount,                                              ///< [in] metric group count
    zet_metric_group_handle_t* phMetricGroups,                              ///< [in][range(0, metricGroupCount )] handles of the metric groups to
                                                                            ///< trace
    zet_metric_tracer_exp_desc_t* desc,                                     ///< [in,out] metric tracer descriptor
    ze_event_handle_t hNotificationEvent,                                   ///< [in][optional] event used for report availability notification. Note:
                                                                            ///< If buffer is not drained when the event it flagged, there is a risk of
                                                                            ///< HW event buffer being overrun
    zet_metric_tracer_exp_handle_t* phMetricTracer                          ///< [out] handle of the metric tracer
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Destroy a metric tracer.
/// 
/// @details
///     - The application must **not** call this function from simultaneous
///       threads with the same metric tracer handle.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hMetricTracer`
ZE_APIEXPORT ze_result_t ZE_APICALL
zetMetricTracerDestroyExp(
    zet_metric_tracer_exp_handle_t hMetricTracer                            ///< [in] handle of the metric tracer
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Start events collection
/// 
/// @details
///     - Driver implementations must make this API call have as minimal
///       overhead as possible, to allow applications start/stop event
///       collection at any point during execution
///     - The application must **not** call this function from simultaneous
///       threads with the same metric tracer handle.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hMetricTracer`
ZE_APIEXPORT ze_result_t ZE_APICALL
zetMetricTracerEnableExp(
    zet_metric_tracer_exp_handle_t hMetricTracer,                           ///< [in] handle of the metric tracer
    ze_bool_t synchronous                                                   ///< [in] request synchronous behavior. Confirmation of successful
                                                                            ///< asynchronous operation is done by calling ::zetMetricTracerReadDataExp()
                                                                            ///< and checking the return status: ::ZE_RESULT_NOT_READY will be returned
                                                                            ///< when the tracer is inactive. ::ZE_RESULT_SUCCESS will be returned 
                                                                            ///< when the tracer is active.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Stop events collection
/// 
/// @details
///     - Driver implementations must make this API call have as minimal
///       overhead as possible, to allow applications start/stop event
///       collection at any point during execution
///     - The application must **not** call this function from simultaneous
///       threads with the same metric tracer handle.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hMetricTracer`
ZE_APIEXPORT ze_result_t ZE_APICALL
zetMetricTracerDisableExp(
    zet_metric_tracer_exp_handle_t hMetricTracer,                           ///< [in] handle of the metric tracer
    ze_bool_t synchronous                                                   ///< [in] request synchronous behavior. Confirmation of successful
                                                                            ///< asynchronous operation is done by calling ::zetMetricTracerReadDataExp()
                                                                            ///< and checking the return status: ::ZE_RESULT_SUCCESS will be returned
                                                                            ///< when the tracer is active or when it is inactive but still has data. 
                                                                            ///< ::ZE_RESULT_NOT_READY will be returned when the tracer is inactive and
                                                                            ///< has no more data to be retrieved.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Read data from the metric tracer
/// 
/// @details
///     - The application must **not** call this function from simultaneous
///       threads with the same metric tracer handle.
///     - Data can be retrieved after tracer is disabled. When buffers are
///       drained ::ZE_RESULT_NOT_READY will be returned
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hMetricTracer`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pRawDataSize`
///     - ::ZE_RESULT_WARNING_DROPPED_DATA
///         + Metric tracer data may have been dropped.
///     - ::ZE_RESULT_NOT_READY
///         + Metric tracer is disabled and no data is available to read.
ZE_APIEXPORT ze_result_t ZE_APICALL
zetMetricTracerReadDataExp(
    zet_metric_tracer_exp_handle_t hMetricTracer,                           ///< [in] handle of the metric tracer
    size_t* pRawDataSize,                                                   ///< [in,out] pointer to size in bytes of raw data requested to read.
                                                                            ///< if size is zero, then the driver will update the value with the total
                                                                            ///< size in bytes needed for all data available.
                                                                            ///< if size is non-zero, then driver will only retrieve that amount of
                                                                            ///< data. 
                                                                            ///< if size is larger than size needed for all data, then driver will
                                                                            ///< update the value with the actual size needed.
    uint8_t* pRawData                                                       ///< [in,out][optional][range(0, *pRawDataSize)] buffer containing tracer
                                                                            ///< data in raw format
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Create a metric decoder for a given metric tracer.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hMetricTracer`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == phMetricDecoder`
ZE_APIEXPORT ze_result_t ZE_APICALL
zetMetricDecoderCreateExp(
    zet_metric_tracer_exp_handle_t hMetricTracer,                           ///< [in] handle of the metric tracer
    zet_metric_decoder_exp_handle_t* phMetricDecoder                        ///< [out] handle of the metric decoder object
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Destroy a metric decoder.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == phMetricDecoder`
ZE_APIEXPORT ze_result_t ZE_APICALL
zetMetricDecoderDestroyExp(
    zet_metric_decoder_exp_handle_t phMetricDecoder                         ///< [in] handle of the metric decoder object
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Return the list of the decodable metrics from the decoder.
/// 
/// @details
///     - The decodable metrics handles returned by this API are defined by the
///       metric groups in the tracer on which the decoder was created.
///     - The decodable metrics handles returned by this API are only valid to
///       decode metrics raw data with ::zetMetricTracerDecodeExp(). Decodable
///       metric handles are not valid to compare with metrics handles included
///       in metric groups.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hMetricDecoder`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pCount`
///         + `nullptr == phMetrics`
ZE_APIEXPORT ze_result_t ZE_APICALL
zetMetricDecoderGetDecodableMetricsExp(
    zet_metric_decoder_exp_handle_t hMetricDecoder,                         ///< [in] handle of the metric decoder object
    uint32_t* pCount,                                                       ///< [in,out] pointer to number of decodable metric in the hMetricDecoder
                                                                            ///< handle. If count is zero, then the driver shall 
                                                                            ///< update the value with the total number of decodable metrics available
                                                                            ///< in the decoder. if count is greater than zero 
                                                                            ///< but less than the total number of decodable metrics available in the
                                                                            ///< decoder, then only that number will be returned. 
                                                                            ///< if count is greater than the number of decodable metrics available in
                                                                            ///< the decoder, then the driver shall update the 
                                                                            ///< value with the actual number of decodable metrics available. 
    zet_metric_handle_t* phMetrics                                          ///< [in,out] [range(0, *pCount)] array of handles of decodable metrics in
                                                                            ///< the hMetricDecoder handle provided.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Decode raw events collected from a tracer.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == phMetricDecoder`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pRawDataSize`
///         + `nullptr == phMetrics`
///         + `nullptr == pSetCount`
///         + `nullptr == pMetricEntriesCount`
ZE_APIEXPORT ze_result_t ZE_APICALL
zetMetricTracerDecodeExp(
    zet_metric_decoder_exp_handle_t phMetricDecoder,                        ///< [in] handle of the metric decoder object
    size_t* pRawDataSize,                                                   ///< [in,out] size in bytes of raw data buffer. If pMetricEntriesCount is
                                                                            ///< greater than zero but less than total number of 
                                                                            ///< decodable metrics available in the raw data buffer, then driver shall
                                                                            ///< update this value with actual number of raw 
                                                                            ///< data bytes processed.
    uint8_t* pRawData,                                                      ///< [in,out][optional][range(0, *pRawDataSize)] buffer containing tracer
                                                                            ///< data in raw format
    uint32_t metricsCount,                                                  ///< [in] number of decodable metrics in the tracer for which the
                                                                            ///< hMetricDecoder handle was provided. See 
                                                                            ///< ::zetMetricDecoderGetDecodableMetricsExp(). If metricCount is greater
                                                                            ///< than zero but less than the number decodable 
                                                                            ///< metrics available in the raw data buffer, then driver shall only
                                                                            ///< decode those.
    zet_metric_handle_t* phMetrics,                                         ///< [in] [range(0, metricsCount)] array of handles of decodable metrics in
                                                                            ///< the decoder for which the hMetricDecoder handle was 
                                                                            ///< provided. Metrics handles are expected to be for decodable metrics,
                                                                            ///< see ::zetMetricDecoderGetDecodableMetrics() 
    uint32_t* pSetCount,                                                    ///< [in,out] pointer to number of metric sets. If count is zero, then the
                                                                            ///< driver shall update the value with the total
                                                                            ///< number of metric sets to be decoded. If count is greater than the
                                                                            ///< number available in the raw data buffer, then the
                                                                            ///< driver shall update the value with the actual number of metric sets to
                                                                            ///< be decoded. There is a 1:1 relation between
                                                                            ///< the number of sets and sub-devices returned in the decoded entries.
    uint32_t* pMetricEntriesCountPerSet,                                    ///< [in,out][optional][range(0, *pSetCount)] buffer of metric entries
                                                                            ///< counts per metric set, one value per set.
    uint32_t* pMetricEntriesCount,                                          ///< [in,out]  pointer to the total number of metric entries decoded, for
                                                                            ///< all metric sets. If count is zero, then the
                                                                            ///< driver shall update the value with the total number of metric entries
                                                                            ///< to be decoded. If count is greater than zero
                                                                            ///< but less than the total number of metric entries available in the raw
                                                                            ///< data, then user provided number will be decoded.
                                                                            ///< If count is greater than the number available in the raw data buffer,
                                                                            ///< then the driver shall update the value with
                                                                            ///< the actual number of decodable metric entries decoded. If set to null,
                                                                            ///< then driver will only update the value of
                                                                            ///< pSetCount.
    zet_metric_entry_exp_t* pMetricEntries                                  ///< [in,out][optional][range(0, *pMetricEntriesCount)] buffer containing
                                                                            ///< decoded metric entries
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Tool Experimental Extension for Metrics/Metric Groups which export Memory
#if !defined(__GNUC__)
#pragma region metricExportMemory
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Metric group type
typedef uint32_t zet_metric_group_type_exp_flags_t;
typedef enum _zet_metric_group_type_exp_flag_t
{
    ZET_METRIC_GROUP_TYPE_EXP_FLAG_EXPORT_DMA_BUF = ZE_BIT(0),              ///< Metric group and metrics exports memory using linux dma-buf, which
                                                                            ///< could be imported/mapped to the host process. Properties of the
                                                                            ///< dma_buf could be queried using ::zet_export_dma_buf_exp_properties_t.
    ZET_METRIC_GROUP_TYPE_EXP_FLAG_USER_CREATED = ZE_BIT(1),                ///< Metric group created using ::zetMetricGroupCreateExp
    ZET_METRIC_GROUP_TYPE_EXP_FLAG_OTHER = ZE_BIT(2),                       ///< Metric group which has a collection of metrics
    ZET_METRIC_GROUP_TYPE_EXP_FLAG_FORCE_UINT32 = 0x7fffffff

} zet_metric_group_type_exp_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Query the metric group type using `pNext` of
///        ::zet_metric_group_properties_t
typedef struct _zet_metric_group_type_exp_t
{
    zet_structure_type_t stype;                                             ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    zet_metric_group_type_exp_flags_t type;                                 ///< [out] metric group type.
                                                                            ///< returns a combination of ::zet_metric_group_type_exp_flags_t.

} zet_metric_group_type_exp_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported dma_buf properties queried using `pNext` of
///        ::zet_metric_group_properties_t or ::zet_metric_properties_t
typedef struct _zet_export_dma_buf_exp_properties_t
{
    zet_structure_type_t stype;                                             ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    int fd;                                                                 ///< [out] the file descriptor handle that could be used to import the
                                                                            ///< memory by the host process.
    size_t size;                                                            ///< [out] size in bytes of the dma_buf

} zet_export_dma_buf_exp_properties_t;

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Tool Experimental Extension for Calculating Multiple Metrics
#if !defined(__GNUC__)
#pragma region multiMetricValues
#endif
///////////////////////////////////////////////////////////////////////////////
#ifndef ZET_MULTI_METRICS_EXP_NAME
/// @brief Calculating Multiple Metrics Experimental Extension Name
#define ZET_MULTI_METRICS_EXP_NAME  "ZET_experimental_calculate_multiple_metrics"
#endif // ZET_MULTI_METRICS_EXP_NAME

///////////////////////////////////////////////////////////////////////////////
/// @brief Calculating Multiple Metrics Experimental Extension Version(s)
typedef enum _ze_calculate_multiple_metrics_exp_version_t
{
    ZE_CALCULATE_MULTIPLE_METRICS_EXP_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ),///< version 1.0
    ZE_CALCULATE_MULTIPLE_METRICS_EXP_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ),///< latest known version
    ZE_CALCULATE_MULTIPLE_METRICS_EXP_VERSION_FORCE_UINT32 = 0x7fffffff

} ze_calculate_multiple_metrics_exp_version_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Calculate one or more sets of metric values from raw data.
/// 
/// @details
///     - This function is similar to ::zetMetricGroupCalculateMetricValues
///       except it may calculate more than one set of metric values from a
///       single data buffer.  There may be one set of metric values for each
///       sub-device, for example.
///     - Each set of metric values may consist of a different number of metric
///       values, returned as the metric value count.
///     - All metric values are calculated into a single buffer; use the metric
///       counts to determine which metric values belong to which set.
///     - The application may call this function from simultaneous threads.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hMetricGroup`
///     - ::ZE_RESULT_ERROR_INVALID_ENUMERATION
///         + `::ZET_METRIC_GROUP_CALCULATION_TYPE_MAX_METRIC_VALUES < type`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pRawData`
///         + `nullptr == pSetCount`
///         + `nullptr == pTotalMetricValueCount`
ZE_APIEXPORT ze_result_t ZE_APICALL
zetMetricGroupCalculateMultipleMetricValuesExp(
    zet_metric_group_handle_t hMetricGroup,                                 ///< [in] handle of the metric group
    zet_metric_group_calculation_type_t type,                               ///< [in] calculation type to be applied on raw data
    size_t rawDataSize,                                                     ///< [in] size in bytes of raw data buffer
    const uint8_t* pRawData,                                                ///< [in][range(0, rawDataSize)] buffer of raw data to calculate
    uint32_t* pSetCount,                                                    ///< [in,out] pointer to number of metric sets.
                                                                            ///< if count is zero, then the driver shall update the value with the
                                                                            ///< total number of metric sets to be calculated.
                                                                            ///< if count is greater than the number available in the raw data buffer,
                                                                            ///< then the driver shall update the value with the actual number of
                                                                            ///< metric sets to be calculated.
    uint32_t* pTotalMetricValueCount,                                       ///< [in,out] pointer to number of the total number of metric values
                                                                            ///< calculated, for all metric sets.
                                                                            ///< if count is zero, then the driver shall update the value with the
                                                                            ///< total number of metric values to be calculated.
                                                                            ///< if count is greater than the number available in the raw data buffer,
                                                                            ///< then the driver shall update the value with the actual number of
                                                                            ///< metric values to be calculated.
    uint32_t* pMetricCounts,                                                ///< [in,out][optional][range(0, *pSetCount)] buffer of metric counts per
                                                                            ///< metric set.
    zet_typed_value_t* pMetricValues                                        ///< [in,out][optional][range(0, *pTotalMetricValueCount)] buffer of
                                                                            ///< calculated metrics.
                                                                            ///< if count is less than the number available in the raw data buffer,
                                                                            ///< then driver shall only calculate that number of metric values.
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Tool Experimental Extension for Global Metric Timestamps
#if !defined(__GNUC__)
#pragma region GlobalTimestamps
#endif
///////////////////////////////////////////////////////////////////////////////
#ifndef ZET_GLOBAL_METRICS_TIMESTAMPS_EXP_NAME
/// @brief Global Metric Timestamps Experimental Extension Name
#define ZET_GLOBAL_METRICS_TIMESTAMPS_EXP_NAME  "ZET_experimental_global_metric_timestamps"
#endif // ZET_GLOBAL_METRICS_TIMESTAMPS_EXP_NAME

///////////////////////////////////////////////////////////////////////////////
/// @brief Global Metric Timestamps Experimental Extension Version(s)
typedef enum _ze_metric_global_timestamps_exp_version_t
{
    ZE_METRIC_GLOBAL_TIMESTAMPS_EXP_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ),  ///< version 1.0
    ZE_METRIC_GLOBAL_TIMESTAMPS_EXP_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ),  ///< latest known version
    ZE_METRIC_GLOBAL_TIMESTAMPS_EXP_VERSION_FORCE_UINT32 = 0x7fffffff

} ze_metric_global_timestamps_exp_version_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Metric timestamps resolution
/// 
/// @details
///     - This structure may be returned from ::zetMetricGroupGetProperties via
///       the `pNext` member of ::zet_metric_group_properties_t.
///     - Used for mapping metric timestamps to other timers.
typedef struct _zet_metric_global_timestamps_resolution_exp_t
{
    zet_structure_type_t stype;                                             ///< [in] type of this structure
    const void* pNext;                                                      ///< [in][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    uint64_t timerResolution;                                               ///< [out] Returns the resolution of metrics timer (used for timestamps) in
                                                                            ///< cycles/sec.
    uint64_t timestampValidBits;                                            ///< [out] Returns the number of valid bits in the timestamp value.

} zet_metric_global_timestamps_resolution_exp_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Returns metric timestamps synchronized with global device timestamps,
///        optionally synchronized with host
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - By default, the global and metrics timestamps are synchronized to the
///       device.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hMetricGroup`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == globalTimestamp`
///         + `nullptr == metricTimestamp`
ZE_APIEXPORT ze_result_t ZE_APICALL
zetMetricGroupGetGlobalTimestampsExp(
    zet_metric_group_handle_t hMetricGroup,                                 ///< [in] handle of the metric group
    ze_bool_t synchronizedWithHost,                                         ///< [in] Returns the timestamps synchronized to the host or the device.
    uint64_t* globalTimestamp,                                              ///< [out] Device timestamp.
    uint64_t* metricTimestamp                                               ///< [out] Metric timestamp.
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Tool Experimental Extension for Exporting Metrics Data
#if !defined(__GNUC__)
#pragma region metricExportData
#endif
///////////////////////////////////////////////////////////////////////////////
#ifndef ZET_EXPORT_METRICS_DATA_EXP_NAME
/// @brief Exporting Metrics Data Experimental Extension Name
#define ZET_EXPORT_METRICS_DATA_EXP_NAME  "ZET_experimental_metric_export_data"
#endif // ZET_EXPORT_METRICS_DATA_EXP_NAME

///////////////////////////////////////////////////////////////////////////////
/// @brief Exporting Metrics Data Experimental Extension Version(s)
typedef enum _zet_export_metric_data_exp_version_t
{
    ZET_EXPORT_METRIC_DATA_EXP_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ),       ///< version 1.0
    ZET_EXPORT_METRIC_DATA_EXP_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ),   ///< latest known version
    ZET_EXPORT_METRIC_DATA_EXP_VERSION_FORCE_UINT32 = 0x7fffffff

} zet_export_metric_data_exp_version_t;

///////////////////////////////////////////////////////////////////////////////
#ifndef ZET_MAX_METRIC_EXPORT_DATA_ELEMENT_NAME_EXP
/// @brief Maximum count of characters in export data element name
#define ZET_MAX_METRIC_EXPORT_DATA_ELEMENT_NAME_EXP  256
#endif // ZET_MAX_METRIC_EXPORT_DATA_ELEMENT_NAME_EXP

///////////////////////////////////////////////////////////////////////////////
#ifndef ZET_MAX_METRIC_EXPORT_DATA_ELEMENT_DESCRIPTION_EXP
/// @brief Maximum export data element description string size
#define ZET_MAX_METRIC_EXPORT_DATA_ELEMENT_DESCRIPTION_EXP  256
#endif // ZET_MAX_METRIC_EXPORT_DATA_ELEMENT_DESCRIPTION_EXP

///////////////////////////////////////////////////////////////////////////////
/// @brief Metrics calculation descriptor
typedef struct _zet_metric_calculate_exp_desc_t
{
    zet_structure_type_t stype;                                             ///< [in] type of this structure
    const void* pNext;                                                      ///< [in][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    uint32_t rawReportSkipCount;                                            ///< [in] number of reports to skip during calculation

} zet_metric_calculate_exp_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Export Metrics Data for system independent calculation.
/// 
/// @details
///     - This function exports raw data and necessary information to perform
///       metrics calculation of collected data in a different system than where
///       data was collected, which may or may not have accelerators.
///     - Implementations can choose to describe the data arrangement of the
///       exported data, using any mechanism which allows users to read and
///       process them.
///     - The application may call this function from simultaneous threads.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hMetricGroup`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pRawData`
///         + `nullptr == pExportDataSize`
ZE_APIEXPORT ze_result_t ZE_APICALL
zetMetricGroupGetExportDataExp(
    zet_metric_group_handle_t hMetricGroup,                                 ///< [in] handle of the metric group
    const uint8_t* pRawData,                                                ///< [in] buffer of raw data
    size_t rawDataSize,                                                     ///< [in] size in bytes of raw data buffer
    size_t* pExportDataSize,                                                ///< [in,out] size in bytes of export data buffer
                                                                            ///< if size is zero, then the driver shall update the value with the
                                                                            ///< number of bytes necessary to store the exported data.
                                                                            ///< if size is greater than required, then the driver shall update the
                                                                            ///< value with the actual number of bytes necessary to store the exported data.
    uint8_t * pExportData                                                   ///< [in,out][optional][range(0, *pExportDataSize)] buffer of exported data.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Calculate one or more sets of metric values from exported raw data.
/// 
/// @details
///     - Calculate metrics values using exported data returned by
///       ::zetMetricGroupGetExportDataExp.
///     - This function is similar to
///       ::zetMetricGroupCalculateMultipleMetricValuesExp except it would
///       calculate from exported metric data.
///     - This function could be used to calculate metrics on a system different
///       from where the metric raw data was collected.
///     - The application may call this function from simultaneous threads.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDriver`
///     - ::ZE_RESULT_ERROR_INVALID_ENUMERATION
///         + `::ZET_METRIC_GROUP_CALCULATION_TYPE_MAX_METRIC_VALUES < type`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pExportData`
///         + `nullptr == pCalculateDescriptor`
///         + `nullptr == pSetCount`
///         + `nullptr == pTotalMetricValueCount`
ZE_APIEXPORT ze_result_t ZE_APICALL
zetMetricGroupCalculateMetricExportDataExp(
    ze_driver_handle_t hDriver,                                             ///< [in] handle of the driver instance
    zet_metric_group_calculation_type_t type,                               ///< [in] calculation type to be applied on raw data
    size_t exportDataSize,                                                  ///< [in] size in bytes of exported data buffer
    const uint8_t* pExportData,                                             ///< [in][range(0, exportDataSize)] buffer of exported data to calculate
    zet_metric_calculate_exp_desc_t* pCalculateDescriptor,                  ///< [in] descriptor specifying calculation specific parameters
    uint32_t* pSetCount,                                                    ///< [in,out] pointer to number of metric sets.
                                                                            ///< if count is zero, then the driver shall update the value with the
                                                                            ///< total number of metric sets to be calculated.
                                                                            ///< if count is greater than the number available in the raw data buffer,
                                                                            ///< then the driver shall update the value with the actual number of
                                                                            ///< metric sets to be calculated.
    uint32_t* pTotalMetricValueCount,                                       ///< [in,out] pointer to number of the total number of metric values
                                                                            ///< calculated, for all metric sets.
                                                                            ///< if count is zero, then the driver shall update the value with the
                                                                            ///< total number of metric values to be calculated.
                                                                            ///< if count is greater than the number available in the raw data buffer,
                                                                            ///< then the driver shall update the value with the actual number of
                                                                            ///< metric values to be calculated.
    uint32_t* pMetricCounts,                                                ///< [in,out][optional][range(0, *pSetCount)] buffer of metric counts per
                                                                            ///< metric set.
    zet_typed_value_t* pMetricValues                                        ///< [in,out][optional][range(0, *pTotalMetricValueCount)] buffer of
                                                                            ///< calculated metrics.
                                                                            ///< if count is less than the number available in the raw data buffer,
                                                                            ///< then driver shall only calculate that number of metric values.
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Tool Experimental Extension for Programmable Metrics
#if !defined(__GNUC__)
#pragma region metricProgrammable
#endif
///////////////////////////////////////////////////////////////////////////////
#ifndef ZET_PROGRAMMABLE_METRICS_EXP_NAME
/// @brief Programmable Metrics Experimental Extension Name
#define ZET_PROGRAMMABLE_METRICS_EXP_NAME  "ZET_experimental_programmable_metrics"
#endif // ZET_PROGRAMMABLE_METRICS_EXP_NAME

///////////////////////////////////////////////////////////////////////////////
/// @brief Programmable Metrics Experimental Extension Version(s)
typedef enum _zet_metric_programmable_exp_version_t
{
    ZET_METRIC_PROGRAMMABLE_EXP_VERSION_1_1 = ZE_MAKE_VERSION( 1, 1 ),      ///< version 1.1
    ZET_METRIC_PROGRAMMABLE_EXP_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 1 ),  ///< latest known version
    ZET_METRIC_PROGRAMMABLE_EXP_VERSION_FORCE_UINT32 = 0x7fffffff

} zet_metric_programmable_exp_version_t;

///////////////////////////////////////////////////////////////////////////////
#ifndef ZET_MAX_PROGRAMMABLE_METRICS_ELEMENT_NAME_EXP
/// @brief Maximum count of characters in export data element name
#define ZET_MAX_PROGRAMMABLE_METRICS_ELEMENT_NAME_EXP  256
#endif // ZET_MAX_PROGRAMMABLE_METRICS_ELEMENT_NAME_EXP

///////////////////////////////////////////////////////////////////////////////
#ifndef ZET_MAX_PROGRAMMABLE_METRICS_ELEMENT_DESCRIPTION_EXP
/// @brief Maximum export data element description string size
#define ZET_MAX_PROGRAMMABLE_METRICS_ELEMENT_DESCRIPTION_EXP  256
#endif // ZET_MAX_PROGRAMMABLE_METRICS_ELEMENT_DESCRIPTION_EXP

///////////////////////////////////////////////////////////////////////////////
#ifndef ZET_MAX_METRIC_PROGRAMMABLE_NAME_EXP
/// @brief Maximum metric programmable name string size
#define ZET_MAX_METRIC_PROGRAMMABLE_NAME_EXP  128
#endif // ZET_MAX_METRIC_PROGRAMMABLE_NAME_EXP

///////////////////////////////////////////////////////////////////////////////
#ifndef ZET_MAX_METRIC_PROGRAMMABLE_DESCRIPTION_EXP
/// @brief Maximum metric programmable description string size
#define ZET_MAX_METRIC_PROGRAMMABLE_DESCRIPTION_EXP  128
#endif // ZET_MAX_METRIC_PROGRAMMABLE_DESCRIPTION_EXP

///////////////////////////////////////////////////////////////////////////////
#ifndef ZET_MAX_METRIC_PROGRAMMABLE_COMPONENT_EXP
/// @brief Maximum metric programmable component string size
#define ZET_MAX_METRIC_PROGRAMMABLE_COMPONENT_EXP  128
#endif // ZET_MAX_METRIC_PROGRAMMABLE_COMPONENT_EXP

///////////////////////////////////////////////////////////////////////////////
#ifndef ZET_MAX_METRIC_PROGRAMMABLE_PARAMETER_NAME_EXP
/// @brief Maximum metric programmable parameter string size
#define ZET_MAX_METRIC_PROGRAMMABLE_PARAMETER_NAME_EXP  128
#endif // ZET_MAX_METRIC_PROGRAMMABLE_PARAMETER_NAME_EXP

///////////////////////////////////////////////////////////////////////////////
#ifndef ZET_MAX_METRIC_PROGRAMMABLE_VALUE_DESCRIPTION_EXP
/// @brief Maximum value for programmable value description
#define ZET_MAX_METRIC_PROGRAMMABLE_VALUE_DESCRIPTION_EXP  128
#endif // ZET_MAX_METRIC_PROGRAMMABLE_VALUE_DESCRIPTION_EXP

///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_MAX_METRIC_GROUP_NAME_PREFIX
/// @brief Maximum value metric group name prefix
#define ZE_MAX_METRIC_GROUP_NAME_PREFIX  64
#endif // ZE_MAX_METRIC_GROUP_NAME_PREFIX

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of metric programmable's object
typedef struct _zet_metric_programmable_exp_handle_t *zet_metric_programmable_exp_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Metric Programmable properties queried using
///        ::zetMetricProgrammableGetPropertiesExp
typedef struct _zet_metric_programmable_exp_properties_t
{
    zet_structure_type_t stype;                                             ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    char name[ZET_MAX_METRIC_PROGRAMMABLE_NAME_EXP];                        ///< [out] metric programmable name
    char description[ZET_MAX_METRIC_PROGRAMMABLE_DESCRIPTION_EXP];          ///< [out] metric programmable description
    char component[ZET_MAX_METRIC_PROGRAMMABLE_COMPONENT_EXP];              ///< [out] metric programmable component
    uint32_t tierNumber;                                                    ///< [out] tier number
    uint32_t domain;                                                        ///< [out] metric domain number.
    uint32_t parameterCount;                                                ///< [out] number of parameters in the programmable
    zet_metric_group_sampling_type_flags_t samplingType;                    ///< [out] metric sampling type.
                                                                            ///< returns a combination of ::zet_metric_group_sampling_type_flag_t.
    uint32_t sourceId;                                                      ///< [out] unique metric source identifier(within platform)to identify the
                                                                            ///< HW block where the metric is collected.

} zet_metric_programmable_exp_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Metric Programmable Parameter types
typedef enum _zet_metric_programmable_param_type_exp_t
{
    ZET_METRIC_PROGRAMMABLE_PARAM_TYPE_EXP_DISAGGREGATION = 0,              ///< Metric is disaggregated.
    ZET_METRIC_PROGRAMMABLE_PARAM_TYPE_EXP_LATENCY = 1,                     ///< Metric for latency measurement.
    ZET_METRIC_PROGRAMMABLE_PARAM_TYPE_EXP_NORMALIZATION_UTILIZATION = 2,   ///< Produces normalization in percent using raw_metric * 100 / cycles / HW
                                                                            ///< instance_count.
    ZET_METRIC_PROGRAMMABLE_PARAM_TYPE_EXP_NORMALIZATION_AVERAGE = 3,       ///< Produces normalization using raw_metric / HW instance_count.
    ZET_METRIC_PROGRAMMABLE_PARAM_TYPE_EXP_NORMALIZATION_RATE = 4,          ///< Produces normalization average using raw_metric / timestamp.
    ZET_METRIC_PROGRAMMABLE_PARAM_TYPE_EXP_NORMALIZATION_BYTES = 5,         ///< Produces normalization average using raw_metric * n bytes.
    ZET_METRIC_PROGRAMMABLE_PARAM_TYPE_EXP_FORCE_UINT32 = 0x7fffffff

} zet_metric_programmable_param_type_exp_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported value info types
typedef enum _zet_value_info_type_exp_t
{
    ZET_VALUE_INFO_TYPE_EXP_UINT32 = 0,                                     ///< 32-bit unsigned-integer
    ZET_VALUE_INFO_TYPE_EXP_UINT64 = 1,                                     ///< 64-bit unsigned-integer
    ZET_VALUE_INFO_TYPE_EXP_FLOAT32 = 2,                                    ///< 32-bit floating-point
    ZET_VALUE_INFO_TYPE_EXP_FLOAT64 = 3,                                    ///< 64-bit floating-point
    ZET_VALUE_INFO_TYPE_EXP_BOOL8 = 4,                                      ///< 8-bit boolean
    ZET_VALUE_INFO_TYPE_EXP_UINT8 = 5,                                      ///< 8-bit unsigned-integer
    ZET_VALUE_INFO_TYPE_EXP_UINT16 = 6,                                     ///< 16-bit unsigned-integer
    ZET_VALUE_INFO_TYPE_EXP_UINT64_RANGE = 7,                               ///< 64-bit unsigned-integer range (minimum and maximum)
    ZET_VALUE_INFO_TYPE_EXP_FLOAT64_RANGE = 8,                              ///< 64-bit floating point range (minimum and maximum)
    ZET_VALUE_INFO_TYPE_EXP_FORCE_UINT32 = 0x7fffffff

} zet_value_info_type_exp_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Value info of type uint64_t range
typedef struct _zet_value_uint64_range_exp_t
{
    uint64_t ui64Min;                                                       ///< [out] minimum value of the range
    uint64_t ui64Max;                                                       ///< [out] maximum value of the range

} zet_value_uint64_range_exp_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Value info of type float64 range
typedef struct _zet_value_fp64_range_exp_t
{
    double fp64Min;                                                         ///< [out] minimum value of the range
    double fp64Max;                                                         ///< [out] maximum value of the range

} zet_value_fp64_range_exp_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Union of value information
typedef union _zet_value_info_exp_t
{
    uint32_t ui32;                                                          ///< [out] 32-bit unsigned-integer
    uint64_t ui64;                                                          ///< [out] 64-bit unsigned-integer
    float fp32;                                                             ///< [out] 32-bit floating-point
    double fp64;                                                            ///< [out] 64-bit floating-point
    ze_bool_t b8;                                                           ///< [out] 8-bit boolean
    uint8_t ui8;                                                            ///< [out] 8-bit unsigned integer
    uint16_t ui16;                                                          ///< [out] 16-bit unsigned integer
    zet_value_uint64_range_exp_t ui64Range;                                 ///< [out] minimum and maximum value of the range
    zet_value_fp64_range_exp_t fp64Range;                                   ///< [out] minimum and maximum value of the range

} zet_value_info_exp_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Metric Programmable parameter information
typedef struct _zet_metric_programmable_param_info_exp_t
{
    zet_structure_type_t stype;                                             ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    zet_metric_programmable_param_type_exp_t type;                          ///< [out] programmable parameter type
    char name[ZET_MAX_METRIC_PROGRAMMABLE_PARAMETER_NAME_EXP];              ///< [out] metric programmable parameter name
    zet_value_info_type_exp_t valueInfoType;                                ///< [out] value info type
    zet_value_t defaultValue;                                               ///< [out] default value for the parameter
    uint32_t valueInfoCount;                                                ///< [out] count of ::zet_metric_programmable_param_value_info_exp_t

} zet_metric_programmable_param_info_exp_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Metric Programmable parameter value information
typedef struct _zet_metric_programmable_param_value_info_exp_t
{
    zet_structure_type_t stype;                                             ///< [in] type of this structure
    void* pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains stype and pNext).
    zet_value_info_exp_t valueInfo;                                         ///< [out] information about the parameter value
    char description[ZET_MAX_METRIC_PROGRAMMABLE_VALUE_DESCRIPTION_EXP];    ///< [out] description about the value

} zet_metric_programmable_param_value_info_exp_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Metric Programmable parameter value
typedef struct _zet_metric_programmable_param_value_exp_t
{
    zet_value_t value;                                                      ///< [in] parameter value

} zet_metric_programmable_param_value_exp_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Query and get the available metric programmable handles.
/// 
/// @details
///     - Query the available programmable handles using *pCount = 0.
///     - Returns all programmable metric handles available in the device.
///     - The application may call this function from simultaneous threads.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pCount`
ZE_APIEXPORT ze_result_t ZE_APICALL
zetMetricProgrammableGetExp(
    zet_device_handle_t hDevice,                                            ///< [in] handle of the device
    uint32_t* pCount,                                                       ///< [in,out] pointer to the number of metric programmable handles.
                                                                            ///< if count is zero, then the driver shall update the value with the
                                                                            ///< total number of metric programmable handles available.
                                                                            ///< if count is greater than the number of metric programmable handles
                                                                            ///< available, then the driver shall update the value with the correct
                                                                            ///< number of metric programmable handles available.
    zet_metric_programmable_exp_handle_t* phMetricProgrammables             ///< [in,out][optional][range(0, *pCount)] array of handle of metric programmables.
                                                                            ///< if count is less than the number of metric programmables available,
                                                                            ///< then driver shall only retrieve that number of metric programmables.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get the properties of the metric programmable.
/// 
/// @details
///     - Returns the properties of the metric programmable.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hMetricProgrammable`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pProperties`
ZE_APIEXPORT ze_result_t ZE_APICALL
zetMetricProgrammableGetPropertiesExp(
    zet_metric_programmable_exp_handle_t hMetricProgrammable,               ///< [in] handle of the metric programmable
    zet_metric_programmable_exp_properties_t* pProperties                   ///< [in,out] properties of the metric programmable
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get the information about the parameters of the metric programmable.
/// 
/// @details
///     - Returns information about the parameters of the metric programmable
///       handle.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hMetricProgrammable`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pParameterCount`
///         + `nullptr == pParameterInfo`
ZE_APIEXPORT ze_result_t ZE_APICALL
zetMetricProgrammableGetParamInfoExp(
    zet_metric_programmable_exp_handle_t hMetricProgrammable,               ///< [in] handle of the metric programmable
    uint32_t* pParameterCount,                                              ///< [in,out] count of the parameters to retrieve parameter info.
                                                                            ///< if value pParameterCount is greater than count of parameters
                                                                            ///< available, then pParameterCount will be updated with count of
                                                                            ///< parameters available.
                                                                            ///< The count of parameters available can be queried using ::zetMetricProgrammableGetPropertiesExp.
    zet_metric_programmable_param_info_exp_t* pParameterInfo                ///< [in,out][range(1, *pParameterCount)] array of parameter info.
                                                                            ///< if parameterCount is less than the number of parameters available,
                                                                            ///< then driver shall only retrieve that number of parameter info.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get the information about the parameter value of the metric
///        programmable.
/// 
/// @details
///     - Returns the value-information about the parameter at the specific
///       ordinal of the metric programmable handle.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hMetricProgrammable`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pValueInfoCount`
///         + `nullptr == pValueInfo`
ZE_APIEXPORT ze_result_t ZE_APICALL
zetMetricProgrammableGetParamValueInfoExp(
    zet_metric_programmable_exp_handle_t hMetricProgrammable,               ///< [in] handle of the metric programmable
    uint32_t parameterOrdinal,                                              ///< [in] ordinal of the parameter in the metric programmable
    uint32_t* pValueInfoCount,                                              ///< [in,out] count of parameter value information to retrieve.
                                                                            ///< if value at pValueInfoCount is greater than count of value info
                                                                            ///< available, then pValueInfoCount will be updated with count of value
                                                                            ///< info available.
                                                                            ///< The count of parameter value info available can be queried using ::zetMetricProgrammableGetParamInfoExp.
    zet_metric_programmable_param_value_info_exp_t* pValueInfo              ///< [in,out][range(1, *pValueInfoCount)] array of parameter value info.
                                                                            ///< if pValueInfoCount is less than the number of value info available,
                                                                            ///< then driver shall only retrieve that number of value info.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Create metric handles by applying parameter values on the metric
///        programmable handle.
/// 
/// @details
///     - Multiple parameter values could be used to prepare a metric.
///     - If parameterCount = 0, the default value of the metric programmable
///       would be used for all parameters.
///     - The implementation can post-fix a C string to the metric name and
///       description, based on the parameter values chosen.
///     - ::zetMetricProgrammableGetParamInfoExp() returns a list of parameters
///       in a defined order.
///     - Therefore, the list of values passed in to the API should respect the
///       same order such that the desired parameter is set with expected value
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hMetricProgrammable`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pParameterValues`
///         + `nullptr == pName`
///         + `nullptr == pDescription`
///         + `nullptr == pMetricHandleCount`
ZE_APIEXPORT ze_result_t ZE_APICALL
zetMetricCreateFromProgrammableExp2(
    zet_metric_programmable_exp_handle_t hMetricProgrammable,               ///< [in] handle of the metric programmable
    uint32_t parameterCount,                                                ///< [in] Count of parameters to set.
    zet_metric_programmable_param_value_exp_t* pParameterValues,            ///< [in] list of parameter values to be set.
    const char* pName,                                                      ///< [in] pointer to metric name to be used. Must point to a
                                                                            ///< null-terminated character array no longer than ::ZET_MAX_METRIC_NAME.
    const char* pDescription,                                               ///< [in] pointer to metric description to be used. Must point to a
                                                                            ///< null-terminated character array no longer than
                                                                            ///< ::ZET_MAX_METRIC_DESCRIPTION.
    uint32_t* pMetricHandleCount,                                           ///< [in,out] Pointer to the number of metric handles.
                                                                            ///< if count is zero, then the driver shall update the value with the
                                                                            ///< number of metric handles available for this programmable.
                                                                            ///< if count is greater than the number of metric handles available, then
                                                                            ///< the driver shall update the value with the correct number of metric
                                                                            ///< handles available.
    zet_metric_handle_t* phMetricHandles                                    ///< [in,out][optional][range(0,*pMetricHandleCount)] array of handle of metrics.
                                                                            ///< if count is less than the number of metrics available, then driver
                                                                            ///< shall only retrieve that number of metric handles.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Create metric handles by applying parameter values on the metric
///        programmable handle.
/// 
/// @details
///     - This API is deprecated. Please use
///       ::zetMetricCreateFromProgrammableExp2()
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hMetricProgrammable`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pParameterValues`
///         + `nullptr == pName`
///         + `nullptr == pDescription`
///         + `nullptr == pMetricHandleCount`
ZE_APIEXPORT ze_result_t ZE_APICALL
zetMetricCreateFromProgrammableExp(
    zet_metric_programmable_exp_handle_t hMetricProgrammable,               ///< [in] handle of the metric programmable
    zet_metric_programmable_param_value_exp_t* pParameterValues,            ///< [in] list of parameter values to be set.
    uint32_t parameterCount,                                                ///< [in] Count of parameters to set.
    const char* pName,                                                      ///< [in] pointer to metric name to be used. Must point to a
                                                                            ///< null-terminated character array no longer than ::ZET_MAX_METRIC_NAME.
    const char* pDescription,                                               ///< [in] pointer to metric description to be used. Must point to a
                                                                            ///< null-terminated character array no longer than
                                                                            ///< ::ZET_MAX_METRIC_DESCRIPTION.
    uint32_t* pMetricHandleCount,                                           ///< [in,out] Pointer to the number of metric handles.
                                                                            ///< if count is zero, then the driver shall update the value with the
                                                                            ///< number of metric handles available for this programmable.
                                                                            ///< if count is greater than the number of metric handles available, then
                                                                            ///< the driver shall update the value with the correct number of metric
                                                                            ///< handles available.
    zet_metric_handle_t* phMetricHandles                                    ///< [in,out][optional][range(0,*pMetricHandleCount)] array of handle of metrics.
                                                                            ///< if count is less than the number of metrics available, then driver
                                                                            ///< shall only retrieve that number of metric handles.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Create multiple metric group handles from metric handles.
/// 
/// @details
///     - Creates multiple metric groups from metrics which were created using
///       ::zetMetricCreateFromProgrammableExp2().
///     - Metrics whose Hardware resources do not overlap are added to same
///       metric group.
///     - The metric groups created using this API are managed by the
///       application and cannot be retrieved using ::zetMetricGroupGet().
///     - The created metric groups are ready for activation and collection.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///         + `nullptr == phMetrics`
///     - ::ZE_RESULT_ERROR_INVALID_ARGUMENT
///         + metricGroupCount is lesser than the number of metric group handles that could be created.
ZE_APIEXPORT ze_result_t ZE_APICALL
zetDeviceCreateMetricGroupsFromMetricsExp(
    zet_device_handle_t hDevice,                                            ///< [in] handle of the device.
    uint32_t metricCount,                                                   ///< [in] number of metric handles.
    zet_metric_handle_t * phMetrics,                                        ///< [in] metric handles to be added to the metric groups.
    const char * pMetricGroupNamePrefix,                                    ///< [in] prefix to the name created for the metric groups. Must point to a
                                                                            ///< null-terminated character array no longer than
                                                                            ///< ZEX_MAX_METRIC_GROUP_NAME_PREFIX.
    const char * pDescription,                                              ///< [in] pointer to description of the metric groups. Must point to a
                                                                            ///< null-terminated character array no longer than
                                                                            ///< ::ZET_MAX_METRIC_GROUP_DESCRIPTION.
    uint32_t * pMetricGroupCount,                                           ///< [in,out] pointer to the number of metric group handles to be created.
                                                                            ///< if pMetricGroupCount is zero, then the driver shall update the value
                                                                            ///< with the maximum possible number of metric group handles that could be created.
                                                                            ///< if pMetricGroupCount is greater than the number of metric group
                                                                            ///< handles that could be created, then the driver shall update the value
                                                                            ///< with the correct number of metric group handles generated.
                                                                            ///< if pMetricGroupCount is lesser than the number of metric group handles
                                                                            ///< that could be created, then ::ZE_RESULT_ERROR_INVALID_ARGUMENT is returned.
    zet_metric_group_handle_t* phMetricGroup                                ///< [in,out][optional][range(0, *pMetricGroupCount)] array of handle of
                                                                            ///< metric group handles.
                                                                            ///< Created Metric group handles.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Create metric group handle.
/// 
/// @details
///     - This API is deprecated. Please use
///       ::zetCreateMetricGroupsFromMetricsExp() 
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pName`
///         + `nullptr == pDescription`
///         + `nullptr == phMetricGroup`
///     - ::ZE_RESULT_ERROR_INVALID_ENUMERATION
///         + `0x7 < samplingType`
ZE_APIEXPORT ze_result_t ZE_APICALL
zetMetricGroupCreateExp(
    zet_device_handle_t hDevice,                                            ///< [in] handle of the device
    const char* pName,                                                      ///< [in] pointer to metric group name. Must point to a null-terminated
                                                                            ///< character array no longer than ::ZET_MAX_METRIC_GROUP_NAME.
    const char* pDescription,                                               ///< [in] pointer to metric group description. Must point to a
                                                                            ///< null-terminated character array no longer than
                                                                            ///< ::ZET_MAX_METRIC_GROUP_DESCRIPTION.
    zet_metric_group_sampling_type_flags_t samplingType,                    ///< [in] Sampling type for the metric group.
    zet_metric_group_handle_t* phMetricGroup                                ///< [in,out] Created Metric group handle
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Add a metric handle to the metric group handle created using
///        ::zetMetricGroupCreateExp.
/// 
/// @details
///     - Reasons for failing to add the metric could be queried using
///       pErrorString
///     - Multiple additions of same metric would add the metric only once to
///       the hMetricGroup
///     - Metric handles from multiple domains may be used in a single metric
///       group.
///     - Metric handles from different sourceIds (refer
///       ::zet_metric_programmable_exp_properties_t) are not allowed in a
///       single metric group.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hMetricGroup`
///         + `nullptr == hMetric`
///     - ::ZE_RESULT_ERROR_INVALID_ARGUMENT
///         + If a Metric handle from a pre-defined metric group is requested to be added.
///     - ::ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE
///         + If the metric group is currently activated.
ZE_APIEXPORT ze_result_t ZE_APICALL
zetMetricGroupAddMetricExp(
    zet_metric_group_handle_t hMetricGroup,                                 ///< [in] Handle of the metric group
    zet_metric_handle_t hMetric,                                            ///< [in] Metric to be added to the group.
    size_t * pErrorStringSize,                                              ///< [in,out][optional] Size of the error string to query, if an error was
                                                                            ///< reported during adding the metric handle.
                                                                            ///< if *pErrorStringSize is zero, then the driver shall update the value
                                                                            ///< with the size of the error string in bytes.
    char* pErrorString                                                      ///< [in,out][optional][range(0, *pErrorStringSize)] Error string.
                                                                            ///< if *pErrorStringSize is less than the length of the error string
                                                                            ///< available, then driver shall only retrieve that length of error string.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Remove a metric from the metric group handle created using
///        ::zetMetricGroupCreateExp.
/// 
/// @details
///     - Remove an already added metric handle from the metric group.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hMetricGroup`
///         + `nullptr == hMetric`
///     - ::ZE_RESULT_ERROR_INVALID_ARGUMENT
///         + If trying to remove a metric not previously added to the metric group
///         + If the input metric group is a pre-defined metric group
///     - ::ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE
///         + If the metric group is currently activated
ZE_APIEXPORT ze_result_t ZE_APICALL
zetMetricGroupRemoveMetricExp(
    zet_metric_group_handle_t hMetricGroup,                                 ///< [in] Handle of the metric group
    zet_metric_handle_t hMetric                                             ///< [in] Metric handle to be removed from the metric group.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Closes a created metric group using ::zetMetricGroupCreateExp, so that
///        it can be activated.
/// 
/// @details
///     - Finalizes the ::zetMetricGroupAddMetricExp and
///       ::zetMetricGroupRemoveMetricExp operations on the metric group.
///     - This is a necessary step before activation of the created metric
///       group.
///     - Add / Remove of metrics is possible after ::zetMetricGroupCloseExp.
///       However, a call to ::zetMetricGroupCloseExp is necessary after
///       modifying the metric group.
///     - Implementations could choose to add new metrics to the group during
///       ::zetMetricGroupCloseExp, which are related and might add value to the
///       metrics already added by the application
///     - Applications can query the list of metrics in the metric group using
///       ::zetMetricGet
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hMetricGroup`
///     - ::ZE_RESULT_ERROR_INVALID_ARGUMENT
///         + If the input metric group is a pre-defined metric group
///     - ::ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE
///         + If the metric group is currently activated
ZE_APIEXPORT ze_result_t ZE_APICALL
zetMetricGroupCloseExp(
    zet_metric_group_handle_t hMetricGroup                                  ///< [in] Handle of the metric group
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Destroy a metric group created using ::zetMetricGroupCreateExp.
/// 
/// @details
///     - Metric handles created using ::zetMetricCreateFromProgrammableExp2 and
///       are part of the metricGroup are not destroyed.
///     - It is necessary to call ::zetMetricDestroyExp for each of the metric
///       handles (created from ::zetMetricCreateFromProgrammableExp2) to
///       destroy them.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hMetricGroup`
///     - ::ZE_RESULT_ERROR_INVALID_ARGUMENT
///         + If trying to destroy a pre-defined metric group
///     - ::ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE
///         + If trying to destroy an activated metric group
ZE_APIEXPORT ze_result_t ZE_APICALL
zetMetricGroupDestroyExp(
    zet_metric_group_handle_t hMetricGroup                                  ///< [in] Handle of the metric group to destroy
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Destroy a metric created using ::zetMetricCreateFromProgrammableExp2.
/// 
/// @details
///     - If a metric is added to a metric group, the metric has to be removed
///       using ::zetMetricGroupRemoveMetricExp before it can be destroyed.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hMetric`
///     - ::ZE_RESULT_ERROR_INVALID_ARGUMENT
///         + If trying to destroy a metric from pre-defined metric group
///     - ::ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE
///         + If trying to destroy a metric currently added to a metric group
ZE_APIEXPORT ze_result_t ZE_APICALL
zetMetricDestroyExp(
    zet_metric_handle_t hMetric                                             ///< [in] Handle of the metric to destroy
    );

#if !defined(__GNUC__)
#pragma endregion
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // _ZET_API_H