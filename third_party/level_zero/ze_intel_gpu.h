/*
 * Copyright (C) 2020-2024 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 */

#ifndef _ZE_INTEL_GPU_H
#define _ZE_INTEL_GPU_H

#include "ze_stypes.h"
#include "ze_api.h"

#if defined(__cplusplus)
#pragma once
extern "C" {
#endif

#include <stdint.h>

#define ZE_INTEL_GPU_VERSION_MAJOR 0
#define ZE_INTEL_GPU_VERSION_MINOR 1

///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_INTEL_DEVICE_MODULE_DP_PROPERTIES_EXP_NAME
/// @brief Module DP properties driver extension name
#define ZE_INTEL_DEVICE_MODULE_DP_PROPERTIES_EXP_NAME "ZE_intel_experimental_device_module_dp_properties"
#endif // ZE_INTEL_DEVICE_MODULE_DP_PROPERTIES_EXP_NAME

///////////////////////////////////////////////////////////////////////////////
/// @brief Module DP properties driver extension Version(s)
typedef enum _ze_intel_device_module_dp_properties_exp_version_t {
    ZE_INTEL_DEVICE_MODULE_DP_PROPERTIES_EXP_VERSION_1_0 = ZE_MAKE_VERSION(1, 0),     ///< version 1.0
    ZE_INTEL_DEVICE_MODULE_DP_PROPERTIES_EXP_VERSION_CURRENT = ZE_MAKE_VERSION(1, 0), ///< latest known version
    ZE_INTEL_DEVICE_MODULE_DP_PROPERTIES_EXP_VERSION_FORCE_UINT32 = 0x7fffffff

} ze_intel_device_module_dp_properties_exp_version_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported Dot Product flags
typedef uint32_t ze_intel_device_module_dp_exp_flags_t;
typedef enum _ze_intel_device_module_dp_exp_flag_t {
    ZE_INTEL_DEVICE_MODULE_EXP_FLAG_DP4A = ZE_BIT(0), ///< Supports DP4A operation
    ZE_INTEL_DEVICE_MODULE_EXP_FLAG_DPAS = ZE_BIT(1), ///< Supports DPAS operation
    ZE_INTEL_DEVICE_MODULE_EXP_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_intel_device_module_dp_exp_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Device Module dot product properties queried using
///        ::zeDeviceGetModuleProperties
///
/// @details
///     - This structure may be passed to ::zeDeviceGetModuleProperties, via
///       `pNext` member of ::ze_device_module_properties_t.
/// @brief Device module dot product properties
typedef struct _ze_intel_device_module_dp_exp_properties_t {
    ze_structure_type_t stype = ZE_STRUCTURE_INTEL_DEVICE_MODULE_DP_EXP_PROPERTIES; ///< [in] type of this structure
    void *pNext;                                                                    ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                                    ///< structure (i.e. contains sType and pNext).
    ze_intel_device_module_dp_exp_flags_t flags;                                    ///< [out] 0 (none) or a valid combination of ::ze_intel_device_module_dp_flag_t
} ze_intel_device_module_dp_exp_properties_t;

#ifndef ZE_INTEL_COMMAND_LIST_MEMORY_SYNC
/// @brief wait on memory extension name
#define ZE_INTEL_COMMAND_LIST_MEMORY_SYNC "ZE_intel_experimental_command_list_memory_sync"
#endif // ZE_INTEL_COMMAND_LIST_MEMORY_SYNC

///////////////////////////////////////////////////////////////////////////////
/// @brief Cmd List memory sync extension Version(s)
typedef enum _ze_intel_command_list_memory_sync_exp_version_t {
    ZE_INTEL_COMMAND_LIST_MEMORY_SYNC_EXP_VERSION_1_0 = ZE_MAKE_VERSION(1, 0),     ///< version 1.0
    ZE_INTEL_COMMAND_LIST_MEMORY_SYNC_EXP_VERSION_CURRENT = ZE_MAKE_VERSION(1, 0), ///< latest known version
    ZE_INTEL_COMMAND_LIST_MEMORY_SYNC_EXP_VERSION_FORCE_UINT32 = 0x7fffffff
} ze_intel_command_list_memory_sync_exp_version_t;

#ifndef ZE_INTEL_STRUCTURE_TYPE_DEVICE_COMMAND_LIST_WAIT_ON_MEMORY_DATA_SIZE_EXP_DESC
/// @brief stype for _ze_intel_device_command_list_wait_on_memory_data_size_exp_desc_t
#endif

///////////////////////////////////////////////////////////////////////////////
/// @brief Extended descriptor for cmd list memory sync
///
/// @details
///     - Implementation must support ::ZE_intel_experimental_command_list_memory_sync extension
///     - May be passed to ze_device_properties_t through pNext.
typedef struct _ze_intel_device_command_list_wait_on_memory_data_size_exp_desc_t {
    ze_structure_type_t stype;                   ///< [in] type of this structure
    const void *pNext;                           ///< [in][optional] must be null or a pointer to an extension-specific
                                                 ///< structure (i.e. contains stype and pNext).
    uint32_t cmdListWaitOnMemoryDataSizeInBytes; /// <out> Defines supported data size for zexCommandListAppendWaitOnMemory[64] API
} ze_intel_device_command_list_wait_on_memory_data_size_exp_desc_t;

#ifndef ZEX_INTEL_EVENT_SYNC_MODE_EXP_NAME
/// @brief Event sync mode extension name
#define ZEX_INTEL_EVENT_SYNC_MODE_EXP_NAME "ZEX_intel_experimental_event_sync_mode"
#endif // ZE_INTEL_EVENT_SYNC_MODE_EXP_NAME

///////////////////////////////////////////////////////////////////////////////
/// @brief Event sync mode extension Version(s)
typedef enum _zex_intel_event_sync_mode_exp_version_t {
    ZEX_INTEL_EVENT_SYNC_MODE_EXP_VERSION_1_0 = ZE_MAKE_VERSION(1, 0),     ///< version 1.0
    ZEX_INTEL_EVENT_SYNC_MODE_EXP_VERSION_CURRENT = ZE_MAKE_VERSION(1, 0), ///< latest known version
    ZEX_INTEL_EVENT_SYNC_MODE_EXP_VERSION_FORCE_UINT32 = 0x7fffffff
} zex_intel_event_sync_mode_exp_version_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported event sync mode flags
typedef uint32_t zex_intel_event_sync_mode_exp_flags_t;
typedef enum _zex_intel_event_sync_mode_exp_flag_t {
    ZEX_INTEL_EVENT_SYNC_MODE_EXP_FLAG_LOW_POWER_WAIT = ZE_BIT(0),          ///< Low power host synchronization mode, for better CPU utilization
    ZEX_INTEL_EVENT_SYNC_MODE_EXP_FLAG_SIGNAL_INTERRUPT = ZE_BIT(1),        ///< Generate interrupt when Event is signalled on Device
    ZEX_INTEL_EVENT_SYNC_MODE_EXP_FLAG_EXTERNAL_INTERRUPT_WAIT = ZE_BIT(2), ///< Host synchronization APIs wait for external interrupt. Can be used only for Events created via zexCounterBasedEventCreate
    ZEX_INTEL_EVENT_SYNC_MODE_EXP_EXP_FLAG_FORCE_UINT32 = 0x7fffffff

} zex_intel_event_sync_mode_exp_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Extended descriptor for event sync mode
///
/// @details
///     - Implementation must support ::ZEX_intel_experimental_event_sync_mode extension
///     - May be passed to ze_event_desc_t through pNext.
typedef struct _zex_intel_event_sync_mode_exp_desc_t {
    ze_structure_type_t stype;                           ///< [in] type of this structure
    const void *pNext;                                   ///< [in][optional] must be null or a pointer to an extension-specific
                                                         ///< structure (i.e. contains stype and pNext).
    zex_intel_event_sync_mode_exp_flags_t syncModeFlags; /// <in> valid combination of ::ze_intel_event_sync_mode_exp_flag_t
    uint32_t externalInterruptId;                        /// <in> External interrupt id. Used only when ZEX_INTEL_EVENT_SYNC_MODE_EXP_FLAG_EXTERNAL_INTERRUPT_WAIT flag is set
} zex_intel_event_sync_mode_exp_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zex_intel_queue_allocate_msix_hint_exp_desc_t
typedef struct _zex_intel_queue_allocate_msix_hint_exp_desc_t zex_intel_queue_allocate_msix_hint_exp_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Command queue descriptor for allocating unique msix. This structure may be
/// passed as pNext member of ::ze_command_queue_desc_t.

typedef struct _zex_intel_queue_allocate_msix_hint_exp_desc_t {
    ze_structure_type_t stype; ///< [in] type of this structure
    const void *pNext;         ///< [in][optional] must be null or a pointer to an extension-specific
                               ///< structure (i.e. contains stype and pNext).
    ze_bool_t uniqueMsix;      ///< [in] If set, try to allocate unique msix for command queue.
                               ///< If not set, driver will follow default behaviour. It may share msix for signaling completion with other queues.
                               ///< Number of unique msixes may be limited. On unsuccessful allocation, queue or immediate cmd list creation API fallbacks to default behaviour.

} zex_intel_queue_allocate_msix_hint_exp_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Command queue descriptor for enabling copy operations offload. This structure may be
/// passed as pNext member of ::ze_command_queue_desc_t.

typedef struct _zex_intel_queue_copy_operations_offload_hint_exp_desc_t {
    ze_structure_type_t stype;    ///< [in] type of this structure
    const void *pNext;            ///< [in][optional] must be null or a pointer to an extension-specific
                                  ///< structure (i.e. contains stype and pNext).
    ze_bool_t copyOffloadEnabled; ///< [in] If set, try to offload copy operations to different engines. Applicable only for compute queues.
                                  ///< This is only a hint. Driver may ignore it per append call, based on platform capabilities or internal heuristics.
                                  ///< If not set, driver will follow default behaviour. Copy operations will be submitted to same engine as compute operations.

} zex_intel_queue_copy_operations_offload_hint_exp_desc_t;

#ifndef ZEX_INTEL_QUEUE_COPY_OPERATIONS_OFFLOAD_HINT_EXP_NAME
/// @brief Queue copy operations offload hint extension name
#define ZEX_INTEL_QUEUE_COPY_OPERATIONS_OFFLOAD_HINT_EXP_NAME "ZEX_intel_experimental_queue_copy_operations_offload_hint"
#endif // ZEX_INTEL_QUEUE_COPY_OPERATIONS_OFFLOAD_HINT_EXP_NAME

///////////////////////////////////////////////////////////////////////////////
/// @brief Queue copy operations offload hint extension version(s)
typedef enum _zex_intel_queue_copy_operations_offload_hint_exp_version_t {
    ZEX_INTEL_QUEUE_COPY_OPERATIONS_OFFLOAD_HINT_EXP_VERSION_1_0 = ZE_MAKE_VERSION(1, 0),     ///< version 1.0
    ZEX_INTEL_QUEUE_COPY_OPERATIONS_OFFLOAD_HINT_EXP_VERSION_CURRENT = ZE_MAKE_VERSION(1, 0), ///< latest known version
    ZEX_INTEL_QUEUE_COPY_OPERATIONS_OFFLOAD_HINT_EXP_VERSION_FORCE_UINT32 = 0x7fffffff
} zex_intel_queue_copy_operations_offload_hint_exp_version_t;

#ifndef ZE_INTEL_GET_DRIVER_VERSION_STRING_EXP_NAME
/// @brief Extension name for query to read the Intel Level Zero Driver Version String
#define ZE_INTEL_GET_DRIVER_VERSION_STRING_EXP_NAME "ZE_intel_get_driver_version_string"
#endif // ZE_INTEL_GET_DRIVER_VERSION_STRING_EXP_NAME

///////////////////////////////////////////////////////////////////////////////
/// @brief Query to read the Intel Level Zero Driver Version String extension version(s)
typedef enum _ze_intel_get_driver_version_string_exp_version_t {
    ZE_INTEL_GET_DRIVER_VERSION_STRING_EXP_VERSION_1_0 = ZE_MAKE_VERSION(1, 0),     ///< version 1.0
    ZE_INTEL_GET_DRIVER_VERSION_STRING_EXP_VERSION_CURRENT = ZE_MAKE_VERSION(1, 0), ///< latest known version
    ZE_INTEL_GET_DRIVER_VERSION_STRING_EXP_VERSION_FORCE_UINT32 = 0x7fffffff
} ze_intel_get_driver_version_string_exp_version_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported 2D Block Array flags
typedef uint32_t ze_intel_device_block_array_exp_flags_t;
typedef enum _ze_intel_device_block_array_exp_flag_t {
    ZE_INTEL_DEVICE_EXP_FLAG_2D_BLOCK_STORE = ZE_BIT(0), ///< Supports store operation
    ZE_INTEL_DEVICE_EXP_FLAG_2D_BLOCK_LOAD = ZE_BIT(1),  ///< Supports load operation
    ZE_INTEL_DEVICE_EXP_FLAG_2D_BLOCK_FORCE_UINT32 = 0x7fffffff

} ze_intel_device_block_array_exp_flag_t;

///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_INTEL_DEVICE_BLOCK_ARRAY_EXP_NAME
/// @brief Device 2D block array properties driver extension name
#define ZE_INTEL_DEVICE_BLOCK_ARRAY_EXP_NAME "ZE_intel_experimental_device_block_array_properties"
#endif // ZE_INTEL_DEVICE_BLOCK_ARRAY_EXP_NAME

/// @brief Device 2D block array properties queried using
///        ::zeDeviceGetProperties
///
/// @details
///     - This structure may be passed to ::zeDeviceGetProperties, via
///       `pNext` member of ::ze_device_properties_t.
/// @brief Device 2D block array properties

typedef struct _ze_intel_device_block_array_exp_properties_t {
    ze_structure_type_t stype = ZE_INTEL_DEVICE_BLOCK_ARRAY_EXP_PROPERTIES; ///< [in] type of this structure
    void *pNext;                                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                            ///< structure (i.e. contains sType and pNext).
    ze_intel_device_block_array_exp_flags_t flags;                          ///< [out] 0 (none) or a valid combination of ::ze_intel_device_block_array_exp_flag_t
} ze_intel_device_block_array_exp_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Device 2D block array properties driver extension versions
typedef enum _ze_intel_device_block_array_exp_properties_version_t {
    ZE_INTEL_DEVICE_BLOCK_ARRAY_EXP_PROPERTIES_EXP_VERSION_1_0 = ZE_MAKE_VERSION(1, 0), ///< version 1.0
    ZE_INTEL_DEVICE_BLOCK_ARRAY_EXP_PROPERTIES_VERSION_CURRENT = ZE_MAKE_VERSION(1, 0), ///< latest known version
    ZE_INTEL_DEVICE_BLOCK_ARRAY_EXP_PROPERTIES_VERSION_FORCE_UINT32 = 0x7fffffff

} ze_intel_device_block_array_exp_properties_version_t;

/// @brief Query to read the Intel Level Zero Driver Version String
///
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
///     - The Driver Version String will be in the format:
///     - Major.Minor.Patch+Optional per semver guidelines https://semver.org/#spec-item-10
/// @returns
///     - ::ZE_RESULT_SUCCESS
ze_result_t ZE_APICALL
zeIntelGetDriverVersionString(
    ze_driver_handle_t hDriver, ///< [in] Driver handle whose version is being read.
    char *pDriverVersion,       ///< [in,out] pointer to driver version string.
    size_t *pVersionSize);      ///< [in,out] pointer to the size of the driver version string.
                                ///< if size is zero, then the size of the version string is returned.

/// @brief Get Kernel Program Binary
///
/// @details
///     - A valid kernel handle must be created with zeKernelCreate.
///     - Returns Intel Graphics Assembly (GEN ISA) format binary program data for kernel handle.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// @returns
///     - ::ZE_RESULT_SUCCESS

#ifndef ZE_INTEL_KERNEL_GET_PROGRAM_BINARY_EXP_NAME
/// @brief Get Kernel Program Binary experimental name
#define ZE_INTEL_KERNEL_GET_PROGRAM_BINARY_EXP_NAME "ZE_intel_experimental_kernel_get_program_binary"
#endif // ZE_INTEL_KERNEL_GET_PROGRAM_BINARY_EXP_NAME

///////////////////////////////////////////////////////////////////////////////
/// @brief Intel Kernel Get Binary Extension Version(s)
typedef enum _ze_intel_kernel_get_binary_exp_version_t {
    ZE_INTEL_KERNEL_GET_PROGRAM_BINARY_EXP_VERSION_1_0 = ZE_MAKE_VERSION(1, 0),     ///< version 1.0
    ZE_INTEL_KERNEL_GET_PROGRAM_BINARY_EXP_VERSION_CURRENT = ZE_MAKE_VERSION(1, 0), ///< latest known version
    ZE_INTEL_KERNEL_GET_PROGRAM_BINARY_EXP_VERSION_FORCE_UINT32 = 0x7fffffff

} ze_intel_kernel_get_binary_exp_version_t;

ze_result_t ZE_APICALL
zeIntelKernelGetBinaryExp(
    ze_kernel_handle_t hKernel, ///< [in] Kernel handle
    size_t *pSize,              ///< [in, out] pointer to variable with size of GEN ISA binary
    char *pKernelBinary         ///< [in,out] pointer to storage area for GEN ISA binary function
);

///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_INTEL_EXTERNAL_SEMAPHORE_EXP_NAME
/// @brief External semaphore extension name
#define ZE_INTEL_EXTERNAL_SEMAPHORE_EXP_NAME "ZE_intel_experimental_external_semaphore"
#endif // ZE_INTEL_EXTERNAL_SEMAPHORE_EXP_NAME

typedef enum _ze_intel_external_semaphore_exp_version_t {
    ZE_EXTERNAL_SEMAPHORE_EXP_VERSION_1_0 = ZE_MAKE_VERSION(1, 0),     ///< version 1.0
    ZE_EXTERNAL_SEMAPHORE_EXP_VERSION_CURRENT = ZE_MAKE_VERSION(1, 0), ///< latest known version
    ZE_EXTERNAL_SEMAPHORE_EXP_VERSION_FORCE_UINT32 = 0x7fffffff
} ze_intel_external_semaphore_exp_version_t;

typedef enum _ze_intel_external_semaphore_exp_flags_t {
    ZE_EXTERNAL_SEMAPHORE_EXP_FLAGS_OPAQUE_FD,
    ZE_EXTERNAL_SEMAPHORE_EXP_FLAGS_OPAQUE_WIN32,
    ZE_EXTERNAL_SEMAPHORE_EXP_FLAGS_OPAQUE_WIN32_KMT,
    ZE_EXTERNAL_SEMAPHORE_EXP_FLAGS_D3D12_FENCE,
    ZE_EXTERNAL_SEMAPHORE_EXP_FLAGS_D3D11_FENCE,
    ZE_EXTERNAL_SEMAPHORE_EXP_FLAGS_KEYED_MUTEX,
    ZE_EXTERNAL_SEMAPHORE_EXP_FLAGS_KEYED_MUTEX_KMT,
    ZE_EXTERNAL_SEMAPHORE_EXP_FLAGS_TIMELINE_SEMAPHORE_FD,
    ZE_EXTERNAL_SEMAPHORE_EXP_FLAGS_TIMELINE_SEMAPHORE_WIN32
} ze_intel_external_semaphore_exp_flags_t;

typedef struct _ze_intel_external_semaphore_exp_desc_t {
    ze_structure_type_t stype;                     ///< [in] type of this structure
    const void *pNext;                             ///< [in] must be null or a pointer to an extension-specific
                                                   ///< structure (i.e. contains stype and pNext). When importing
                                                   ///< a semaphore, this can be a pointer to either ze_intel_external_semaphore_win32_exp_desc_t
                                                   ///< or ze_intel_external_semaphore_fd_exp_desc_t.
    ze_intel_external_semaphore_exp_flags_t flags; ///< [in] External semaphore flags. Must be 0 or a valid combination of ::ze_intel_external_semaphore_exp_flags_t
} ze_intel_external_semaphore_exp_desc_t;

typedef struct _ze_intel_external_semaphore_win32_exp_desc_t {
    ze_structure_type_t stype; ///< [in] type of this structure
    const void *pNext;         ///< [in] must be null or a pointer to an extension-specific
                               ///< structure (i.e. contains stype and pNext).
    void *handle;              ///< [in] Win32 HANDLE for semaphore
    const char *name;          ///< [in] Name of the semaphore. Must be valid NULL terminated string.
} ze_intel_external_semaphore_win32_exp_desc_t;

typedef struct _ze_intel_external_semaphore_fd_exp_desc_t {
    ze_structure_type_t stype; ///< [in] type of this structure
    const void *pNext;         ///< [in] must be null or a pointer to an extension-specific
                               ///< structure (i.e. contains stype and pNext).
    int fd;                    ///< [in] File descriptor for semaphore. Must be valid file descriptor.
} ze_intel_external_semaphore_desc_fd_exp_desc_t;

typedef struct _ze_intel_external_semaphore_signal_params_exp_t {
    ze_structure_type_t stype; ///< [in] type of this structure
    const void *pNext;         ///< [in] must be null or a pointer to an extension-specific
                               ///< structure (i.e. contains stype and pNext).
    uint64_t value;            /// [in] [optional] Value to signal the semaphore with
} ze_intel_external_semaphore_signal_params_exp_t;

typedef struct _ze_intel_external_semaphore_wait_params_exp_t {
    ze_structure_type_t stype; ///< [in] type of this structure
    const void *pNext;         ///< [in] must be null or a pointer to an extension-specific
                               ///< structure (i.e. contains stype and pNext).
    uint64_t value;            /// [in] [optional] Value to wait on the semaphore for
} ze_intel_external_semaphore_wait_params_exp_t;

typedef struct _ze_intel_external_semaphore_exp_handle_t *ze_intel_external_semaphore_exp_handle_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zeIntelDeviceImportExternalSemaphoreExp(
    ze_device_handle_t device,                                   ///< [in] handle of the device
    const ze_intel_external_semaphore_exp_desc_t *semaphoreDesc, ///< [in] pointer to external semaphore descriptor
    ze_intel_external_semaphore_exp_handle_t *phSemaphore        ///< [out] pointer to handle of the external semaphore
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeIntelCommandListAppendWaitExternalSemaphoresExp(
    ze_command_list_handle_t hCmdList,                            ///< [in] handle of the command list
    unsigned int numExternalSemaphores,                           ///< [in] number of external semaphores
    const ze_intel_external_semaphore_exp_handle_t *phSemaphores, ///< [in] pointer to array of external semaphore handles
    const ze_intel_external_semaphore_wait_params_exp_t *params,  ///< [in] pointer to array of wait parameters
    ze_event_handle_t hSignalEvent,                               ///< [in][optional] handle of the event to signal on completion
    uint32_t numWaitEvents,                                       ///< [in][optional] number of events to wait on before continuing
    ze_event_handle_t *phWaitEvents                               ///< [in][optional][range(0, numWaitEvents)] handles of the events to wait on before continuing
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeIntelCommandListAppendSignalExternalSemaphoresExp(
    ze_command_list_handle_t hCmdList,                             ///< [in] handle of the command list
    size_t numExternalSemaphores,                                  ///< [in] number of external semaphores
    const ze_intel_external_semaphore_exp_handle_t *phSemaphores,  ///< [in] pointer to array of external semaphore handles
    const ze_intel_external_semaphore_signal_params_exp_t *params, ///< [in] pointer to array of signal parameters
    ze_event_handle_t hSignalEvent,                                ///< [in][optional] handle of the event to signal on completion
    uint32_t numWaitEvents,                                        ///< [in][optional] number of events to wait on before continuing
    ze_event_handle_t *phWaitEvents                                ///< [in][optional][range(0, numWaitEvents)] handles of the events to wait on before continuing
);

ZE_APIEXPORT ze_result_t ZE_APICALL
zeIntelDeviceReleaseExternalSemaphoreExp(
    ze_intel_external_semaphore_exp_handle_t hSemaphore ///< [in] handle of the external semaphore
);

#if defined(__cplusplus)
} // extern "C"
#endif

#endif
