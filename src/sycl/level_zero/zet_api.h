/*
 *
 * Copyright (C) 2019-2021 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 * @file zet_api.h
 * @version v1.3-r1.3.7
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
    ZET_STRUCTURE_TYPE_METRIC_GROUP_PROPERTIES = 0x1,   ///< ::zet_metric_group_properties_t
    ZET_STRUCTURE_TYPE_METRIC_PROPERTIES = 0x2,     ///< ::zet_metric_properties_t
    ZET_STRUCTURE_TYPE_METRIC_STREAMER_DESC = 0x3,  ///< ::zet_metric_streamer_desc_t
    ZET_STRUCTURE_TYPE_METRIC_QUERY_POOL_DESC = 0x4,///< ::zet_metric_query_pool_desc_t
    ZET_STRUCTURE_TYPE_PROFILE_PROPERTIES = 0x5,    ///< ::zet_profile_properties_t
    ZET_STRUCTURE_TYPE_DEVICE_DEBUG_PROPERTIES = 0x6,   ///< ::zet_device_debug_properties_t
    ZET_STRUCTURE_TYPE_DEBUG_MEMORY_SPACE_DESC = 0x7,   ///< ::zet_debug_memory_space_desc_t
    ZET_STRUCTURE_TYPE_DEBUG_REGSET_PROPERTIES = 0x8,   ///< ::zet_debug_regset_properties_t
    ZET_STRUCTURE_TYPE_TRACER_EXP_DESC = 0x00010001,///< ::zet_tracer_exp_desc_t
    ZET_STRUCTURE_TYPE_FORCE_UINT32 = 0x7fffffff

} zet_structure_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Base for all properties types
typedef struct _zet_base_properties_t
{
    zet_structure_type_t stype;                     ///< [in] type of this structure
    void* pNext;                                    ///< [in,out][optional] pointer to extension-specific structure

} zet_base_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Base for all descriptor types
typedef struct _zet_base_desc_t
{
    zet_structure_type_t stype;                     ///< [in] type of this structure
    const void* pNext;                              ///< [in][optional] pointer to extension-specific structure

} zet_base_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported value types
typedef enum _zet_value_type_t
{
    ZET_VALUE_TYPE_UINT32 = 0,                      ///< 32-bit unsigned-integer
    ZET_VALUE_TYPE_UINT64 = 1,                      ///< 64-bit unsigned-integer
    ZET_VALUE_TYPE_FLOAT32 = 2,                     ///< 32-bit floating-point
    ZET_VALUE_TYPE_FLOAT64 = 3,                     ///< 64-bit floating-point
    ZET_VALUE_TYPE_BOOL8 = 4,                       ///< 8-bit boolean
    ZET_VALUE_TYPE_FORCE_UINT32 = 0x7fffffff

} zet_value_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Union of values
typedef union _zet_value_t
{
    uint32_t ui32;                                  ///< [out] 32-bit unsigned-integer
    uint64_t ui64;                                  ///< [out] 32-bit unsigned-integer
    float fp32;                                     ///< [out] 32-bit floating-point
    double fp64;                                    ///< [out] 64-bit floating-point
    ze_bool_t b8;                                   ///< [out] 8-bit boolean

} zet_value_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Typed value
typedef struct _zet_typed_value_t
{
    zet_value_type_t type;                          ///< [out] type of value
    zet_value_t value;                              ///< [out] value

} zet_typed_value_t;

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
    ZET_MODULE_DEBUG_INFO_FORMAT_ELF_DWARF = 0,     ///< Format is ELF/DWARF
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
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hModule`
///     - ::ZE_RESULT_ERROR_INVALID_ENUMERATION
///         + `::ZET_MODULE_DEBUG_INFO_FORMAT_ELF_DWARF < format`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pSize`
ZE_APIEXPORT ze_result_t ZE_APICALL
zetModuleGetDebugInfo(
    zet_module_handle_t hModule,                    ///< [in] handle of the module
    zet_module_debug_info_format_t format,          ///< [in] debug info format requested
    size_t* pSize,                                  ///< [in,out] size of debug info in bytes
    uint8_t* pDebugInfo                             ///< [in,out][optional] byte pointer to debug info
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
    ZET_DEVICE_DEBUG_PROPERTY_FLAG_ATTACH = ZE_BIT(0),  ///< the device supports attaching for debug
    ZET_DEVICE_DEBUG_PROPERTY_FLAG_FORCE_UINT32 = 0x7fffffff

} zet_device_debug_property_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Device debug properties queried using ::zetDeviceGetDebugProperties.
typedef struct _zet_device_debug_properties_t
{
    zet_structure_type_t stype;                     ///< [in] type of this structure
    void* pNext;                                    ///< [in,out][optional] pointer to extension-specific structure
    zet_device_debug_property_flags_t flags;        ///< [out] returns 0 (none) or a valid combination of
                                                    ///< ::zet_device_debug_property_flag_t

} zet_device_debug_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves debug properties of the device.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pDebugProperties`
ZE_APIEXPORT ze_result_t ZE_APICALL
zetDeviceGetDebugProperties(
    zet_device_handle_t hDevice,                    ///< [in] device handle
    zet_device_debug_properties_t* pDebugProperties ///< [in,out] query result for debug properties
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Debug configuration provided to ::zetDebugAttach
typedef struct _zet_debug_config_t
{
    uint32_t pid;                                   ///< [in] the host process identifier

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
    zet_device_handle_t hDevice,                    ///< [in] device handle
    const zet_debug_config_t* config,               ///< [in] the debug configuration
    zet_debug_session_handle_t* phDebug             ///< [out] debug session handle
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Close a debug session.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDebug`
ZE_APIEXPORT ze_result_t ZE_APICALL
zetDebugDetach(
    zet_debug_session_handle_t hDebug               ///< [in][release] debug session handle
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported debug event flags.
typedef uint32_t zet_debug_event_flags_t;
typedef enum _zet_debug_event_flag_t
{
    ZET_DEBUG_EVENT_FLAG_NEED_ACK = ZE_BIT(0),      ///< The event needs to be acknowledged by calling
                                                    ///< ::zetDebugAcknowledgeEvent.
    ZET_DEBUG_EVENT_FLAG_FORCE_UINT32 = 0x7fffffff

} zet_debug_event_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported debug event types.
typedef enum _zet_debug_event_type_t
{
    ZET_DEBUG_EVENT_TYPE_INVALID = 0,               ///< The event is invalid
    ZET_DEBUG_EVENT_TYPE_DETACHED = 1,              ///< The tool was detached
    ZET_DEBUG_EVENT_TYPE_PROCESS_ENTRY = 2,         ///< The debuggee process created command queues on the device
    ZET_DEBUG_EVENT_TYPE_PROCESS_EXIT = 3,          ///< The debuggee process destroyed all command queues on the device
    ZET_DEBUG_EVENT_TYPE_MODULE_LOAD = 4,           ///< An in-memory module was loaded onto the device
    ZET_DEBUG_EVENT_TYPE_MODULE_UNLOAD = 5,         ///< An in-memory module is about to get unloaded from the device
    ZET_DEBUG_EVENT_TYPE_THREAD_STOPPED = 6,        ///< The thread stopped due to a device exception
    ZET_DEBUG_EVENT_TYPE_THREAD_UNAVAILABLE = 7,    ///< The thread is not available to be stopped
    ZET_DEBUG_EVENT_TYPE_PAGE_FAULT = 8,            ///< A page request could not be completed on the device
    ZET_DEBUG_EVENT_TYPE_FORCE_UINT32 = 0x7fffffff

} zet_debug_event_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported debug detach reasons.
typedef enum _zet_debug_detach_reason_t
{
    ZET_DEBUG_DETACH_REASON_INVALID = 0,            ///< The detach reason is not valid
    ZET_DEBUG_DETACH_REASON_HOST_EXIT = 1,          ///< The host process exited
    ZET_DEBUG_DETACH_REASON_FORCE_UINT32 = 0x7fffffff

} zet_debug_detach_reason_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Event information for ::ZET_DEBUG_EVENT_TYPE_DETACHED
typedef struct _zet_debug_event_info_detached_t
{
    zet_debug_detach_reason_t reason;               ///< [out] the detach reason

} zet_debug_event_info_detached_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Event information for ::ZET_DEBUG_EVENT_TYPE_MODULE_LOAD and
///        ::ZET_DEBUG_EVENT_TYPE_MODULE_UNLOAD
typedef struct _zet_debug_event_info_module_t
{
    zet_module_debug_info_format_t format;          ///< [out] the module format
    uint64_t moduleBegin;                           ///< [out] the begin address of the in-memory module (inclusive)
    uint64_t moduleEnd;                             ///< [out] the end address of the in-memory module (exclusive)
    uint64_t load;                                  ///< [out] the load address of the module on the device

} zet_debug_event_info_module_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Event information for ::ZET_DEBUG_EVENT_TYPE_THREAD_STOPPED and
///        ::ZET_DEBUG_EVENT_TYPE_THREAD_UNAVAILABLE
typedef struct _zet_debug_event_info_thread_stopped_t
{
    ze_device_thread_t thread;                      ///< [out] the stopped/unavailable thread

} zet_debug_event_info_thread_stopped_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Page fault reasons.
typedef enum _zet_debug_page_fault_reason_t
{
    ZET_DEBUG_PAGE_FAULT_REASON_INVALID = 0,        ///< The page fault reason is not valid
    ZET_DEBUG_PAGE_FAULT_REASON_MAPPING_ERROR = 1,  ///< The address is not mapped
    ZET_DEBUG_PAGE_FAULT_REASON_PERMISSION_ERROR = 2,   ///< Invalid access permissions
    ZET_DEBUG_PAGE_FAULT_REASON_FORCE_UINT32 = 0x7fffffff

} zet_debug_page_fault_reason_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Event information for ::ZET_DEBUG_EVENT_TYPE_PAGE_FAULT
typedef struct _zet_debug_event_info_page_fault_t
{
    uint64_t address;                               ///< [out] the faulting address
    uint64_t mask;                                  ///< [out] the alignment mask
    zet_debug_page_fault_reason_t reason;           ///< [out] the page fault reason

} zet_debug_event_info_page_fault_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Event type-specific information
typedef union _zet_debug_event_info_t
{
    zet_debug_event_info_detached_t detached;       ///< [out] type == ::ZET_DEBUG_EVENT_TYPE_DETACHED
    zet_debug_event_info_module_t module;           ///< [out] type == ::ZET_DEBUG_EVENT_TYPE_MODULE_LOAD or
                                                    ///< ::ZET_DEBUG_EVENT_TYPE_MODULE_UNLOAD
    zet_debug_event_info_thread_stopped_t thread;   ///< [out] type == ::ZET_DEBUG_EVENT_TYPE_THREAD_STOPPED or
                                                    ///< ::ZET_DEBUG_EVENT_TYPE_THREAD_UNAVAILABLE
    zet_debug_event_info_page_fault_t page_fault;   ///< [out] type == ::ZET_DEBUG_EVENT_TYPE_PAGE_FAULT

} zet_debug_event_info_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief A debug event on the device.
typedef struct _zet_debug_event_t
{
    zet_debug_event_type_t type;                    ///< [out] the event type
    zet_debug_event_flags_t flags;                  ///< [out] returns 0 (none) or a combination of ::zet_debug_event_flag_t
    zet_debug_event_info_t info;                    ///< [out] event type specific information

} zet_debug_event_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Read the topmost debug event.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDebug`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == event`
///     - ::ZE_RESULT_NOT_READY
///         + the timeout expired
ZE_APIEXPORT ze_result_t ZE_APICALL
zetDebugReadEvent(
    zet_debug_session_handle_t hDebug,              ///< [in] debug session handle
    uint64_t timeout,                               ///< [in] if non-zero, then indicates the maximum time (in milliseconds) to
                                                    ///< yield before returning ::ZE_RESULT_SUCCESS or ::ZE_RESULT_NOT_READY;
                                                    ///< if zero, then immediately returns the status of the event;
                                                    ///< if UINT64_MAX, then function will not return until complete or device
                                                    ///< is lost.
                                                    ///< Due to external dependencies, timeout may be rounded to the closest
                                                    ///< value allowed by the accuracy of those dependencies.
    zet_debug_event_t* event                        ///< [in,out] a pointer to a ::zet_debug_event_t.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Acknowledge a debug event.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDebug`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == event`
ZE_APIEXPORT ze_result_t ZE_APICALL
zetDebugAcknowledgeEvent(
    zet_debug_session_handle_t hDebug,              ///< [in] debug session handle
    const zet_debug_event_t* event                  ///< [in] a pointer to a ::zet_debug_event_t.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Interrupt device threads.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDebug`
///     - ::ZE_RESULT_ERROR_NOT_AVAILABLE
///         + the thread is already stopped or unavailable
ZE_APIEXPORT ze_result_t ZE_APICALL
zetDebugInterrupt(
    zet_debug_session_handle_t hDebug,              ///< [in] debug session handle
    ze_device_thread_t thread                       ///< [in] the thread to interrupt
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Resume device threads.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDebug`
///     - ::ZE_RESULT_ERROR_NOT_AVAILABLE
///         + the thread is already running or unavailable
ZE_APIEXPORT ze_result_t ZE_APICALL
zetDebugResume(
    zet_debug_session_handle_t hDebug,              ///< [in] debug session handle
    ze_device_thread_t thread                       ///< [in] the thread to resume
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported device memory space types.
typedef enum _zet_debug_memory_space_type_t
{
    ZET_DEBUG_MEMORY_SPACE_TYPE_DEFAULT = 0,        ///< default memory space (attribute may be omitted)
    ZET_DEBUG_MEMORY_SPACE_TYPE_SLM = 1,            ///< shared local memory space (GPU-only)
    ZET_DEBUG_MEMORY_SPACE_TYPE_FORCE_UINT32 = 0x7fffffff

} zet_debug_memory_space_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Device memory space descriptor
typedef struct _zet_debug_memory_space_desc_t
{
    zet_structure_type_t stype;                     ///< [in] type of this structure
    const void* pNext;                              ///< [in][optional] pointer to extension-specific structure
    zet_debug_memory_space_type_t type;             ///< [in] type of memory space
    uint64_t address;                               ///< [in] the virtual address within the memory space

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
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDebug`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == desc`
///         + `nullptr == buffer`
///     - ::ZE_RESULT_ERROR_INVALID_ENUMERATION
///         + `::ZET_DEBUG_MEMORY_SPACE_TYPE_SLM < desc->type`
///     - ::ZE_RESULT_ERROR_NOT_AVAILABLE
///         + the thread is running or unavailable
///         + the memory cannot be accessed from the supplied thread
ZE_APIEXPORT ze_result_t ZE_APICALL
zetDebugReadMemory(
    zet_debug_session_handle_t hDebug,              ///< [in] debug session handle
    ze_device_thread_t thread,                      ///< [in] the thread identifier.
    const zet_debug_memory_space_desc_t* desc,      ///< [in] memory space descriptor
    size_t size,                                    ///< [in] the number of bytes to read
    void* buffer                                    ///< [in,out] a buffer to hold a copy of the memory
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
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDebug`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == desc`
///         + `nullptr == buffer`
///     - ::ZE_RESULT_ERROR_INVALID_ENUMERATION
///         + `::ZET_DEBUG_MEMORY_SPACE_TYPE_SLM < desc->type`
///     - ::ZE_RESULT_ERROR_NOT_AVAILABLE
///         + the thread is running or unavailable
///         + the memory cannot be accessed from the supplied thread
ZE_APIEXPORT ze_result_t ZE_APICALL
zetDebugWriteMemory(
    zet_debug_session_handle_t hDebug,              ///< [in] debug session handle
    ze_device_thread_t thread,                      ///< [in] the thread identifier.
    const zet_debug_memory_space_desc_t* desc,      ///< [in] memory space descriptor
    size_t size,                                    ///< [in] the number of bytes to write
    const void* buffer                              ///< [in] a buffer holding the pattern to write
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported general register set flags.
typedef uint32_t zet_debug_regset_flags_t;
typedef enum _zet_debug_regset_flag_t
{
    ZET_DEBUG_REGSET_FLAG_READABLE = ZE_BIT(0),     ///< register set is readable
    ZET_DEBUG_REGSET_FLAG_WRITEABLE = ZE_BIT(1),    ///< register set is writeable
    ZET_DEBUG_REGSET_FLAG_FORCE_UINT32 = 0x7fffffff

} zet_debug_regset_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Device register set properties queried using
///        ::zetDebugGetRegisterSetProperties.
typedef struct _zet_debug_regset_properties_t
{
    zet_structure_type_t stype;                     ///< [in] type of this structure
    void* pNext;                                    ///< [in,out][optional] pointer to extension-specific structure
    uint32_t type;                                  ///< [out] device-specific register set type
    uint32_t version;                               ///< [out] device-specific version of this register set
    zet_debug_regset_flags_t generalFlags;          ///< [out] general register set flags
    uint32_t deviceFlags;                           ///< [out] device-specific register set flags
    uint32_t count;                                 ///< [out] number of registers in the set
    uint32_t bitSize;                               ///< [out] the size of a register in bits
    uint32_t byteSize;                              ///< [out] the size required for reading or writing a register in bytes

} zet_debug_regset_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves debug register set properties.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pCount`
ZE_APIEXPORT ze_result_t ZE_APICALL
zetDebugGetRegisterSetProperties(
    zet_device_handle_t hDevice,                    ///< [in] device handle
    uint32_t* pCount,                               ///< [in,out] pointer to the number of register set properties.
                                                    ///< if count is zero, then the driver shall update the value with the
                                                    ///< total number of register set properties available.
                                                    ///< if count is greater than the number of register set properties
                                                    ///< available, then the driver shall update the value with the correct
                                                    ///< number of registry set properties available.
    zet_debug_regset_properties_t* pRegisterSetProperties   ///< [in,out][optional][range(0, *pCount)] array of query results for
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
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDebug`
///     - ::ZE_RESULT_ERROR_NOT_AVAILABLE
///         + the thread is running or unavailable
ZE_APIEXPORT ze_result_t ZE_APICALL
zetDebugReadRegisters(
    zet_debug_session_handle_t hDebug,              ///< [in] debug session handle
    ze_device_thread_t thread,                      ///< [in] the thread identifier
    uint32_t type,                                  ///< [in] register set type
    uint32_t start,                                 ///< [in] the starting offset into the register state area; must be less
                                                    ///< than ::zet_debug_regset_properties_t.count for the type
    uint32_t count,                                 ///< [in] the number of registers to read; start+count must be <=
                                                    ///< zet_debug_register_group_properties_t.count for the type
    void* pRegisterValues                           ///< [in,out][optional][range(0, count)] buffer of register values
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Write register state.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDebug`
///     - ::ZE_RESULT_ERROR_NOT_AVAILABLE
///         + the thread is running or unavailable
ZE_APIEXPORT ze_result_t ZE_APICALL
zetDebugWriteRegisters(
    zet_debug_session_handle_t hDebug,              ///< [in] debug session handle
    ze_device_thread_t thread,                      ///< [in] the thread identifier
    uint32_t type,                                  ///< [in] register set type
    uint32_t start,                                 ///< [in] the starting offset into the register state area; must be less
                                                    ///< than ::zet_debug_regset_properties_t.count for the type
    uint32_t count,                                 ///< [in] the number of registers to write; start+count must be <=
                                                    ///< zet_debug_register_group_properties_t.count for the type
    void* pRegisterValues                           ///< [in,out][optional][range(0, count)] buffer of register values
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
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pCount`
ZE_APIEXPORT ze_result_t ZE_APICALL
zetMetricGroupGet(
    zet_device_handle_t hDevice,                    ///< [in] handle of the device
    uint32_t* pCount,                               ///< [in,out] pointer to the number of metric groups.
                                                    ///< if count is zero, then the driver shall update the value with the
                                                    ///< total number of metric groups available.
                                                    ///< if count is greater than the number of metric groups available, then
                                                    ///< the driver shall update the value with the correct number of metric
                                                    ///< groups available.
    zet_metric_group_handle_t* phMetricGroups       ///< [in,out][optional][range(0, *pCount)] array of handle of metric groups.
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
    ZET_METRIC_GROUP_SAMPLING_TYPE_FLAG_EVENT_BASED = ZE_BIT(0),///< Event based sampling
    ZET_METRIC_GROUP_SAMPLING_TYPE_FLAG_TIME_BASED = ZE_BIT(1), ///< Time based sampling
    ZET_METRIC_GROUP_SAMPLING_TYPE_FLAG_FORCE_UINT32 = 0x7fffffff

} zet_metric_group_sampling_type_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Metric group properties queried using ::zetMetricGroupGetProperties
typedef struct _zet_metric_group_properties_t
{
    zet_structure_type_t stype;                     ///< [in] type of this structure
    void* pNext;                                    ///< [in,out][optional] pointer to extension-specific structure
    char name[ZET_MAX_METRIC_GROUP_NAME];           ///< [out] metric group name
    char description[ZET_MAX_METRIC_GROUP_DESCRIPTION]; ///< [out] metric group description
    zet_metric_group_sampling_type_flags_t samplingType;///< [out] metric group sampling type.
                                                    ///< returns a combination of ::zet_metric_group_sampling_type_flag_t.
    uint32_t domain;                                ///< [out] metric group domain number. Cannot use multiple, simultaneous
                                                    ///< metric groups from the same domain.
    uint32_t metricCount;                           ///< [out] metric count belonging to this group

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
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hMetricGroup`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pProperties`
ZE_APIEXPORT ze_result_t ZE_APICALL
zetMetricGroupGetProperties(
    zet_metric_group_handle_t hMetricGroup,         ///< [in] handle of the metric group
    zet_metric_group_properties_t* pProperties      ///< [in,out] metric group properties
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Metric types
typedef enum _zet_metric_type_t
{
    ZET_METRIC_TYPE_DURATION = 0,                   ///< Metric type: duration
    ZET_METRIC_TYPE_EVENT = 1,                      ///< Metric type: event
    ZET_METRIC_TYPE_EVENT_WITH_RANGE = 2,           ///< Metric type: event with range
    ZET_METRIC_TYPE_THROUGHPUT = 3,                 ///< Metric type: throughput
    ZET_METRIC_TYPE_TIMESTAMP = 4,                  ///< Metric type: timestamp
    ZET_METRIC_TYPE_FLAG = 5,                       ///< Metric type: flag
    ZET_METRIC_TYPE_RATIO = 6,                      ///< Metric type: ratio
    ZET_METRIC_TYPE_RAW = 7,                        ///< Metric type: raw
    ZET_METRIC_TYPE_IP_EXP = 0x7ffffffe,            ///< Metric type: instruction pointer
    ZET_METRIC_TYPE_FORCE_UINT32 = 0x7fffffff

} zet_metric_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Metric group calculation type
typedef enum _zet_metric_group_calculation_type_t
{
    ZET_METRIC_GROUP_CALCULATION_TYPE_METRIC_VALUES = 0,///< Calculated metric values from raw data.
    ZET_METRIC_GROUP_CALCULATION_TYPE_MAX_METRIC_VALUES = 1,///< Maximum metric values.
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
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hMetricGroup`
///     - ::ZE_RESULT_ERROR_INVALID_ENUMERATION
///         + `::ZET_METRIC_GROUP_CALCULATION_TYPE_MAX_METRIC_VALUES < type`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pRawData`
///         + `nullptr == pMetricValueCount`
ZE_APIEXPORT ze_result_t ZE_APICALL
zetMetricGroupCalculateMetricValues(
    zet_metric_group_handle_t hMetricGroup,         ///< [in] handle of the metric group
    zet_metric_group_calculation_type_t type,       ///< [in] calculation type to be applied on raw data
    size_t rawDataSize,                             ///< [in] size in bytes of raw data buffer
    const uint8_t* pRawData,                        ///< [in][range(0, rawDataSize)] buffer of raw data to calculate
    uint32_t* pMetricValueCount,                    ///< [in,out] pointer to number of metric values calculated.
                                                    ///< if count is zero, then the driver shall update the value with the
                                                    ///< total number of metric values to be calculated.
                                                    ///< if count is greater than the number available in the raw data buffer,
                                                    ///< then the driver shall update the value with the actual number of
                                                    ///< metric values to be calculated.
    zet_typed_value_t* pMetricValues                ///< [in,out][optional][range(0, *pMetricValueCount)] buffer of calculated metrics.
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
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hMetricGroup`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pCount`
ZE_APIEXPORT ze_result_t ZE_APICALL
zetMetricGet(
    zet_metric_group_handle_t hMetricGroup,         ///< [in] handle of the metric group
    uint32_t* pCount,                               ///< [in,out] pointer to the number of metrics.
                                                    ///< if count is zero, then the driver shall update the value with the
                                                    ///< total number of metrics available.
                                                    ///< if count is greater than the number of metrics available, then the
                                                    ///< driver shall update the value with the correct number of metrics available.
    zet_metric_handle_t* phMetrics                  ///< [in,out][optional][range(0, *pCount)] array of handle of metrics.
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
    zet_structure_type_t stype;                     ///< [in] type of this structure
    void* pNext;                                    ///< [in,out][optional] pointer to extension-specific structure
    char name[ZET_MAX_METRIC_NAME];                 ///< [out] metric name
    char description[ZET_MAX_METRIC_DESCRIPTION];   ///< [out] metric description
    char component[ZET_MAX_METRIC_COMPONENT];       ///< [out] metric component
    uint32_t tierNumber;                            ///< [out] number of tier
    zet_metric_type_t metricType;                   ///< [out] metric type
    zet_value_type_t resultType;                    ///< [out] metric result type
    char resultUnits[ZET_MAX_METRIC_RESULT_UNITS];  ///< [out] metric result units

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
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hMetric`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pProperties`
ZE_APIEXPORT ze_result_t ZE_APICALL
zetMetricGetProperties(
    zet_metric_handle_t hMetric,                    ///< [in] handle of the metric
    zet_metric_properties_t* pProperties            ///< [in,out] metric properties
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
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hContext`
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_SIZE
///         + `(nullptr == phMetricGroups) && (0 < count)`
///     - ::ZE_RESULT_ERROR_INVALID_ARGUMENT
///         + Multiple metric groups share the same domain
ZE_APIEXPORT ze_result_t ZE_APICALL
zetContextActivateMetricGroups(
    zet_context_handle_t hContext,                  ///< [in] handle of the context object
    zet_device_handle_t hDevice,                    ///< [in] handle of the device
    uint32_t count,                                 ///< [in] metric group count to activate; must be 0 if `nullptr ==
                                                    ///< phMetricGroups`
    zet_metric_group_handle_t* phMetricGroups       ///< [in][optional][range(0, count)] handles of the metric groups to activate.
                                                    ///< nullptr deactivates all previously used metric groups.
                                                    ///< all metrics groups must come from a different domains.
                                                    ///< metric query and metric stream must use activated metric groups.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Metric streamer descriptor
typedef struct _zet_metric_streamer_desc_t
{
    zet_structure_type_t stype;                     ///< [in] type of this structure
    const void* pNext;                              ///< [in][optional] pointer to extension-specific structure
    uint32_t notifyEveryNReports;                   ///< [in,out] number of collected reports after which notification event
                                                    ///< will be signalled
    uint32_t samplingPeriod;                        ///< [in,out] streamer sampling period in nanoseconds

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
    zet_context_handle_t hContext,                  ///< [in] handle of the context object
    zet_device_handle_t hDevice,                    ///< [in] handle of the device
    zet_metric_group_handle_t hMetricGroup,         ///< [in] handle of the metric group
    zet_metric_streamer_desc_t* desc,               ///< [in,out] metric streamer descriptor
    ze_event_handle_t hNotificationEvent,           ///< [in][optional] event used for report availability notification
    zet_metric_streamer_handle_t* phMetricStreamer  ///< [out] handle of metric streamer
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
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hCommandList`
///         + `nullptr == hMetricStreamer`
ZE_APIEXPORT ze_result_t ZE_APICALL
zetCommandListAppendMetricStreamerMarker(
    zet_command_list_handle_t hCommandList,         ///< [in] handle of the command list
    zet_metric_streamer_handle_t hMetricStreamer,   ///< [in] handle of the metric streamer
    uint32_t value                                  ///< [in] streamer marker value
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
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hMetricStreamer`
ZE_APIEXPORT ze_result_t ZE_APICALL
zetMetricStreamerClose(
    zet_metric_streamer_handle_t hMetricStreamer    ///< [in][release] handle of the metric streamer
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
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hMetricStreamer`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pRawDataSize`
ZE_APIEXPORT ze_result_t ZE_APICALL
zetMetricStreamerReadData(
    zet_metric_streamer_handle_t hMetricStreamer,   ///< [in] handle of the metric streamer
    uint32_t maxReportCount,                        ///< [in] the maximum number of reports the application wants to receive.
                                                    ///< if UINT32_MAX, then function will retrieve all reports available
    size_t* pRawDataSize,                           ///< [in,out] pointer to size in bytes of raw data requested to read.
                                                    ///< if size is zero, then the driver will update the value with the total
                                                    ///< size in bytes needed for all reports available.
                                                    ///< if size is non-zero, then driver will only retrieve the number of
                                                    ///< reports that fit into the buffer.
                                                    ///< if size is larger than size needed for all reports, then driver will
                                                    ///< update the value with the actual size needed.
    uint8_t* pRawData                               ///< [in,out][optional][range(0, *pRawDataSize)] buffer containing streamer
                                                    ///< reports in raw format
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Metric query pool types
typedef enum _zet_metric_query_pool_type_t
{
    ZET_METRIC_QUERY_POOL_TYPE_PERFORMANCE = 0,     ///< Performance metric query pool.
    ZET_METRIC_QUERY_POOL_TYPE_EXECUTION = 1,       ///< Skips workload execution between begin/end calls.
    ZET_METRIC_QUERY_POOL_TYPE_FORCE_UINT32 = 0x7fffffff

} zet_metric_query_pool_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Metric query pool description
typedef struct _zet_metric_query_pool_desc_t
{
    zet_structure_type_t stype;                     ///< [in] type of this structure
    const void* pNext;                              ///< [in][optional] pointer to extension-specific structure
    zet_metric_query_pool_type_t type;              ///< [in] Query pool type.
    uint32_t count;                                 ///< [in] Internal slots count within query pool object.

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
    zet_context_handle_t hContext,                  ///< [in] handle of the context object
    zet_device_handle_t hDevice,                    ///< [in] handle of the device
    zet_metric_group_handle_t hMetricGroup,         ///< [in] metric group associated with the query object.
    const zet_metric_query_pool_desc_t* desc,       ///< [in] metric query pool descriptor
    zet_metric_query_pool_handle_t* phMetricQueryPool   ///< [out] handle of metric query pool
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
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hMetricQueryPool`
///     - ::ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE
ZE_APIEXPORT ze_result_t ZE_APICALL
zetMetricQueryPoolDestroy(
    zet_metric_query_pool_handle_t hMetricQueryPool ///< [in][release] handle of the metric query pool
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
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hMetricQueryPool`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == phMetricQuery`
ZE_APIEXPORT ze_result_t ZE_APICALL
zetMetricQueryCreate(
    zet_metric_query_pool_handle_t hMetricQueryPool,///< [in] handle of the metric query pool
    uint32_t index,                                 ///< [in] index of the query within the pool
    zet_metric_query_handle_t* phMetricQuery        ///< [out] handle of metric query
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
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hMetricQuery`
///     - ::ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE
ZE_APIEXPORT ze_result_t ZE_APICALL
zetMetricQueryDestroy(
    zet_metric_query_handle_t hMetricQuery          ///< [in][release] handle of metric query
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Resets a metric query object back to inital state.
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
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hMetricQuery`
ZE_APIEXPORT ze_result_t ZE_APICALL
zetMetricQueryReset(
    zet_metric_query_handle_t hMetricQuery          ///< [in] handle of metric query
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
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hCommandList`
///         + `nullptr == hMetricQuery`
ZE_APIEXPORT ze_result_t ZE_APICALL
zetCommandListAppendMetricQueryBegin(
    zet_command_list_handle_t hCommandList,         ///< [in] handle of the command list
    zet_metric_query_handle_t hMetricQuery          ///< [in] handle of the metric query
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
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hCommandList`
///         + `nullptr == hMetricQuery`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == phWaitEvents`
///     - ::ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT
///     - ::ZE_RESULT_ERROR_INVALID_SIZE
///         + `(nullptr == phWaitEvents) && (0 < numWaitEvents)`
ZE_APIEXPORT ze_result_t ZE_APICALL
zetCommandListAppendMetricQueryEnd(
    zet_command_list_handle_t hCommandList,         ///< [in] handle of the command list
    zet_metric_query_handle_t hMetricQuery,         ///< [in] handle of the metric query
    ze_event_handle_t hSignalEvent,                 ///< [in][optional] handle of the event to signal on completion
    uint32_t numWaitEvents,                         ///< [in] must be zero
    ze_event_handle_t* phWaitEvents                 ///< [in][mbz] must be nullptr
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
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hCommandList`
ZE_APIEXPORT ze_result_t ZE_APICALL
zetCommandListAppendMetricMemoryBarrier(
    zet_command_list_handle_t hCommandList          ///< [in] handle of the command list
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
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hMetricQuery`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pRawDataSize`
ZE_APIEXPORT ze_result_t ZE_APICALL
zetMetricQueryGetData(
    zet_metric_query_handle_t hMetricQuery,         ///< [in] handle of the metric query
    size_t* pRawDataSize,                           ///< [in,out] pointer to size in bytes of raw data requested to read.
                                                    ///< if size is zero, then the driver will update the value with the total
                                                    ///< size in bytes needed for all reports available.
                                                    ///< if size is non-zero, then driver will only retrieve the number of
                                                    ///< reports that fit into the buffer.
                                                    ///< if size is larger than size needed for all reports, then driver will
                                                    ///< update the value with the actual size needed.
    uint8_t* pRawData                               ///< [in,out][optional][range(0, *pRawDataSize)] buffer containing query
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
    ZET_PROFILE_FLAG_REGISTER_REALLOCATION = ZE_BIT(0), ///< request the compiler attempt to minimize register usage as much as
                                                    ///< possible to allow for instrumentation
    ZET_PROFILE_FLAG_FREE_REGISTER_INFO = ZE_BIT(1),///< request the compiler generate free register info
    ZET_PROFILE_FLAG_FORCE_UINT32 = 0x7fffffff

} zet_profile_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Profiling meta-data for instrumentation
typedef struct _zet_profile_properties_t
{
    zet_structure_type_t stype;                     ///< [in] type of this structure
    void* pNext;                                    ///< [in,out][optional] pointer to extension-specific structure
    zet_profile_flags_t flags;                      ///< [out] indicates which flags were enabled during compilation.
                                                    ///< returns 0 (none) or a combination of ::zet_profile_flag_t
    uint32_t numTokens;                             ///< [out] number of tokens immediately following this structure

} zet_profile_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported profile token types
typedef enum _zet_profile_token_type_t
{
    ZET_PROFILE_TOKEN_TYPE_FREE_REGISTER = 0,       ///< GRF info
    ZET_PROFILE_TOKEN_TYPE_FORCE_UINT32 = 0x7fffffff

} zet_profile_token_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Profile free register token detailing unused registers in the current
///        function
typedef struct _zet_profile_free_register_token_t
{
    zet_profile_token_type_t type;                  ///< [out] type of token
    uint32_t size;                                  ///< [out] total size of the token, in bytes
    uint32_t count;                                 ///< [out] number of register sequences immediately following this
                                                    ///< structure

} zet_profile_free_register_token_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Profile register sequence detailing consecutive bytes, all of which
///        are unused
typedef struct _zet_profile_register_sequence_t
{
    uint32_t start;                                 ///< [out] starting byte in the register table, representing the start of
                                                    ///< unused bytes in the current function
    uint32_t count;                                 ///< [out] number of consecutive bytes in the sequence, starting from start

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
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hKernel`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pProfileProperties`
ZE_APIEXPORT ze_result_t ZE_APICALL
zetKernelGetProfileInfo(
    zet_kernel_handle_t hKernel,                    ///< [in] handle to kernel
    zet_profile_properties_t* pProfileProperties    ///< [out] pointer to profile properties
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
    ZET_API_TRACING_EXP_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ),  ///< version 1.0
    ZET_API_TRACING_EXP_VERSION_CURRENT = ZE_MAKE_VERSION( 1, 0 ),  ///< latest known version
    ZET_API_TRACING_EXP_VERSION_FORCE_UINT32 = 0x7fffffff

} zet_api_tracing_exp_version_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Alias the existing callbacks definition for 'core' callbacks
typedef ze_callbacks_t zet_core_callbacks_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Tracer descriptor
typedef struct _zet_tracer_exp_desc_t
{
    zet_structure_type_t stype;                     ///< [in] type of this structure
    const void* pNext;                              ///< [in][optional] pointer to extension-specific structure
    void* pUserData;                                ///< [in] pointer passed to every tracer's callbacks

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
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hContext`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == desc`
///         + `nullptr == desc->pUserData`
///         + `nullptr == phTracer`
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
ZE_APIEXPORT ze_result_t ZE_APICALL
zetTracerExpCreate(
    zet_context_handle_t hContext,                  ///< [in] handle of the context object
    const zet_tracer_exp_desc_t* desc,              ///< [in] pointer to tracer descriptor
    zet_tracer_exp_handle_t* phTracer               ///< [out] pointer to handle of tracer object created
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
zetTracerExpDestroy(
    zet_tracer_exp_handle_t hTracer                 ///< [in][release] handle of tracer object to destroy
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
zetTracerExpSetPrologues(
    zet_tracer_exp_handle_t hTracer,                ///< [in] handle of the tracer
    zet_core_callbacks_t* pCoreCbs                  ///< [in] pointer to table of 'core' callback function pointers
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
zetTracerExpSetEpilogues(
    zet_tracer_exp_handle_t hTracer,                ///< [in] handle of the tracer
    zet_core_callbacks_t* pCoreCbs                  ///< [in] pointer to table of 'core' callback function pointers
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
zetTracerExpSetEnabled(
    zet_tracer_exp_handle_t hTracer,                ///< [in] handle of the tracer
    ze_bool_t enable                                ///< [in] enable the tracer if true; disable if false
    );

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
    zet_metric_group_handle_t hMetricGroup,         ///< [in] handle of the metric group
    zet_metric_group_calculation_type_t type,       ///< [in] calculation type to be applied on raw data
    size_t rawDataSize,                             ///< [in] size in bytes of raw data buffer
    const uint8_t* pRawData,                        ///< [in][range(0, rawDataSize)] buffer of raw data to calculate
    uint32_t* pSetCount,                            ///< [in,out] pointer to number of metric sets.
                                                    ///< if count is zero, then the driver shall update the value with the
                                                    ///< total number of metric sets to be calculated.
                                                    ///< if count is greater than the number available in the raw data buffer,
                                                    ///< then the driver shall update the value with the actual number of
                                                    ///< metric sets to be calculated.
    uint32_t* pTotalMetricValueCount,               ///< [in,out] pointer to number of the total number of metric values
                                                    ///< calculated, for all metric sets.
                                                    ///< if count is zero, then the driver shall update the value with the
                                                    ///< total number of metric values to be calculated.
                                                    ///< if count is greater than the number available in the raw data buffer,
                                                    ///< then the driver shall update the value with the actual number of
                                                    ///< metric values to be calculated.
    uint32_t* pMetricCounts,                        ///< [in,out][optional][range(0, *pSetCount)] buffer of metric counts per
                                                    ///< metric set.
    zet_typed_value_t* pMetricValues                ///< [in,out][optional][range(0, *pTotalMetricValueCount)] buffer of
                                                    ///< calculated metrics.
                                                    ///< if count is less than the number available in the raw data buffer,
                                                    ///< then driver shall only calculate that number of metric values.
    );

#if !defined(__GNUC__)
#pragma endregion
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // _ZET_API_H