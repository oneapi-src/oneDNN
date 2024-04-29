/*
 *
 * Copyright (C) 2019-2021 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 * @file zes_api.h
 * @version v1.3-r1.3.7
 *
 */
#ifndef _ZES_API_H
#define _ZES_API_H
#if defined(__cplusplus)
#pragma once
#endif

// 'core' API headers
#include "ze_api.h"

#if defined(__cplusplus)
extern "C" {
#endif

// Intel 'oneAPI' Level-Zero Sysman API common types
#if !defined(__GNUC__)
#pragma region common
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Handle to a driver instance
typedef ze_driver_handle_t zes_driver_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of device object
typedef ze_device_handle_t zes_device_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle for a Sysman device scheduler queue
typedef struct _zes_sched_handle_t *zes_sched_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle for a Sysman device performance factors
typedef struct _zes_perf_handle_t *zes_perf_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle for a Sysman device power domain
typedef struct _zes_pwr_handle_t *zes_pwr_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle for a Sysman device frequency domain
typedef struct _zes_freq_handle_t *zes_freq_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle for a Sysman device engine group
typedef struct _zes_engine_handle_t *zes_engine_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle for a Sysman device standby control
typedef struct _zes_standby_handle_t *zes_standby_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle for a Sysman device firmware
typedef struct _zes_firmware_handle_t *zes_firmware_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle for a Sysman device memory module
typedef struct _zes_mem_handle_t *zes_mem_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle for a Sysman fabric port
typedef struct _zes_fabric_port_handle_t *zes_fabric_port_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle for a Sysman device temperature sensor
typedef struct _zes_temp_handle_t *zes_temp_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle for a Sysman device power supply
typedef struct _zes_psu_handle_t *zes_psu_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle for a Sysman device fan
typedef struct _zes_fan_handle_t *zes_fan_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle for a Sysman device LED
typedef struct _zes_led_handle_t *zes_led_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle for a Sysman device RAS error set
typedef struct _zes_ras_handle_t *zes_ras_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle for a Sysman device diagnostics test suite
typedef struct _zes_diag_handle_t *zes_diag_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Defines structure types
typedef enum _zes_structure_type_t
{
    ZES_STRUCTURE_TYPE_DEVICE_PROPERTIES = 0x1,     ///< ::zes_device_properties_t
    ZES_STRUCTURE_TYPE_PCI_PROPERTIES = 0x2,        ///< ::zes_pci_properties_t
    ZES_STRUCTURE_TYPE_PCI_BAR_PROPERTIES = 0x3,    ///< ::zes_pci_bar_properties_t
    ZES_STRUCTURE_TYPE_DIAG_PROPERTIES = 0x4,       ///< ::zes_diag_properties_t
    ZES_STRUCTURE_TYPE_ENGINE_PROPERTIES = 0x5,     ///< ::zes_engine_properties_t
    ZES_STRUCTURE_TYPE_FABRIC_PORT_PROPERTIES = 0x6,///< ::zes_fabric_port_properties_t
    ZES_STRUCTURE_TYPE_FAN_PROPERTIES = 0x7,        ///< ::zes_fan_properties_t
    ZES_STRUCTURE_TYPE_FIRMWARE_PROPERTIES = 0x8,   ///< ::zes_firmware_properties_t
    ZES_STRUCTURE_TYPE_FREQ_PROPERTIES = 0x9,       ///< ::zes_freq_properties_t
    ZES_STRUCTURE_TYPE_LED_PROPERTIES = 0xa,        ///< ::zes_led_properties_t
    ZES_STRUCTURE_TYPE_MEM_PROPERTIES = 0xb,        ///< ::zes_mem_properties_t
    ZES_STRUCTURE_TYPE_PERF_PROPERTIES = 0xc,       ///< ::zes_perf_properties_t
    ZES_STRUCTURE_TYPE_POWER_PROPERTIES = 0xd,      ///< ::zes_power_properties_t
    ZES_STRUCTURE_TYPE_PSU_PROPERTIES = 0xe,        ///< ::zes_psu_properties_t
    ZES_STRUCTURE_TYPE_RAS_PROPERTIES = 0xf,        ///< ::zes_ras_properties_t
    ZES_STRUCTURE_TYPE_SCHED_PROPERTIES = 0x10,     ///< ::zes_sched_properties_t
    ZES_STRUCTURE_TYPE_SCHED_TIMEOUT_PROPERTIES = 0x11, ///< ::zes_sched_timeout_properties_t
    ZES_STRUCTURE_TYPE_SCHED_TIMESLICE_PROPERTIES = 0x12,   ///< ::zes_sched_timeslice_properties_t
    ZES_STRUCTURE_TYPE_STANDBY_PROPERTIES = 0x13,   ///< ::zes_standby_properties_t
    ZES_STRUCTURE_TYPE_TEMP_PROPERTIES = 0x14,      ///< ::zes_temp_properties_t
    ZES_STRUCTURE_TYPE_DEVICE_STATE = 0x15,         ///< ::zes_device_state_t
    ZES_STRUCTURE_TYPE_PROCESS_STATE = 0x16,        ///< ::zes_process_state_t
    ZES_STRUCTURE_TYPE_PCI_STATE = 0x17,            ///< ::zes_pci_state_t
    ZES_STRUCTURE_TYPE_FABRIC_PORT_CONFIG = 0x18,   ///< ::zes_fabric_port_config_t
    ZES_STRUCTURE_TYPE_FABRIC_PORT_STATE = 0x19,    ///< ::zes_fabric_port_state_t
    ZES_STRUCTURE_TYPE_FAN_CONFIG = 0x1a,           ///< ::zes_fan_config_t
    ZES_STRUCTURE_TYPE_FREQ_STATE = 0x1b,           ///< ::zes_freq_state_t
    ZES_STRUCTURE_TYPE_OC_CAPABILITIES = 0x1c,      ///< ::zes_oc_capabilities_t
    ZES_STRUCTURE_TYPE_LED_STATE = 0x1d,            ///< ::zes_led_state_t
    ZES_STRUCTURE_TYPE_MEM_STATE = 0x1e,            ///< ::zes_mem_state_t
    ZES_STRUCTURE_TYPE_PSU_STATE = 0x1f,            ///< ::zes_psu_state_t
    ZES_STRUCTURE_TYPE_BASE_STATE = 0x20,           ///< ::zes_base_state_t
    ZES_STRUCTURE_TYPE_RAS_CONFIG = 0x21,           ///< ::zes_ras_config_t
    ZES_STRUCTURE_TYPE_RAS_STATE = 0x22,            ///< ::zes_ras_state_t
    ZES_STRUCTURE_TYPE_TEMP_CONFIG = 0x23,          ///< ::zes_temp_config_t
    ZES_STRUCTURE_TYPE_PCI_BAR_PROPERTIES_1_2 = 0x24,   ///< ::zes_pci_bar_properties_1_2_t
    ZES_STRUCTURE_TYPE_FORCE_UINT32 = 0x7fffffff

} zes_structure_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Base for all properties types
typedef struct _zes_base_properties_t
{
    zes_structure_type_t stype;                     ///< [in] type of this structure
    void* pNext;                                    ///< [in,out][optional] pointer to extension-specific structure

} zes_base_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Base for all descriptor types
typedef struct _zes_base_desc_t
{
    zes_structure_type_t stype;                     ///< [in] type of this structure
    const void* pNext;                              ///< [in][optional] pointer to extension-specific structure

} zes_base_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Base for all state types
typedef struct _zes_base_state_t
{
    zes_structure_type_t stype;                     ///< [in] type of this structure
    const void* pNext;                              ///< [in][optional] pointer to extension-specific structure

} zes_base_state_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Base for all config types
typedef struct _zes_base_config_t
{
    zes_structure_type_t stype;                     ///< [in] type of this structure
    const void* pNext;                              ///< [in][optional] pointer to extension-specific structure

} zes_base_config_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Base for all capability types
typedef struct _zes_base_capability_t
{
    zes_structure_type_t stype;                     ///< [in] type of this structure
    const void* pNext;                              ///< [in][optional] pointer to extension-specific structure

} zes_base_capability_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_base_properties_t
typedef struct _zes_base_properties_t zes_base_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_base_desc_t
typedef struct _zes_base_desc_t zes_base_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_base_state_t
typedef struct _zes_base_state_t zes_base_state_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_base_config_t
typedef struct _zes_base_config_t zes_base_config_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_base_capability_t
typedef struct _zes_base_capability_t zes_base_capability_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_device_state_t
typedef struct _zes_device_state_t zes_device_state_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_device_properties_t
typedef struct _zes_device_properties_t zes_device_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_process_state_t
typedef struct _zes_process_state_t zes_process_state_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_pci_address_t
typedef struct _zes_pci_address_t zes_pci_address_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_pci_speed_t
typedef struct _zes_pci_speed_t zes_pci_speed_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_pci_properties_t
typedef struct _zes_pci_properties_t zes_pci_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_pci_state_t
typedef struct _zes_pci_state_t zes_pci_state_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_pci_bar_properties_t
typedef struct _zes_pci_bar_properties_t zes_pci_bar_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_pci_bar_properties_1_2_t
typedef struct _zes_pci_bar_properties_1_2_t zes_pci_bar_properties_1_2_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_pci_stats_t
typedef struct _zes_pci_stats_t zes_pci_stats_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_diag_test_t
typedef struct _zes_diag_test_t zes_diag_test_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_diag_properties_t
typedef struct _zes_diag_properties_t zes_diag_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_engine_properties_t
typedef struct _zes_engine_properties_t zes_engine_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_engine_stats_t
typedef struct _zes_engine_stats_t zes_engine_stats_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_fabric_port_id_t
typedef struct _zes_fabric_port_id_t zes_fabric_port_id_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_fabric_port_speed_t
typedef struct _zes_fabric_port_speed_t zes_fabric_port_speed_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_fabric_port_properties_t
typedef struct _zes_fabric_port_properties_t zes_fabric_port_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_fabric_link_type_t
typedef struct _zes_fabric_link_type_t zes_fabric_link_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_fabric_port_config_t
typedef struct _zes_fabric_port_config_t zes_fabric_port_config_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_fabric_port_state_t
typedef struct _zes_fabric_port_state_t zes_fabric_port_state_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_fabric_port_throughput_t
typedef struct _zes_fabric_port_throughput_t zes_fabric_port_throughput_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_fan_speed_t
typedef struct _zes_fan_speed_t zes_fan_speed_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_fan_temp_speed_t
typedef struct _zes_fan_temp_speed_t zes_fan_temp_speed_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_fan_speed_table_t
typedef struct _zes_fan_speed_table_t zes_fan_speed_table_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_fan_properties_t
typedef struct _zes_fan_properties_t zes_fan_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_fan_config_t
typedef struct _zes_fan_config_t zes_fan_config_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_firmware_properties_t
typedef struct _zes_firmware_properties_t zes_firmware_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_freq_properties_t
typedef struct _zes_freq_properties_t zes_freq_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_freq_range_t
typedef struct _zes_freq_range_t zes_freq_range_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_freq_state_t
typedef struct _zes_freq_state_t zes_freq_state_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_freq_throttle_time_t
typedef struct _zes_freq_throttle_time_t zes_freq_throttle_time_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_oc_capabilities_t
typedef struct _zes_oc_capabilities_t zes_oc_capabilities_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_led_properties_t
typedef struct _zes_led_properties_t zes_led_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_led_color_t
typedef struct _zes_led_color_t zes_led_color_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_led_state_t
typedef struct _zes_led_state_t zes_led_state_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_mem_properties_t
typedef struct _zes_mem_properties_t zes_mem_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_mem_state_t
typedef struct _zes_mem_state_t zes_mem_state_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_mem_bandwidth_t
typedef struct _zes_mem_bandwidth_t zes_mem_bandwidth_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_perf_properties_t
typedef struct _zes_perf_properties_t zes_perf_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_power_properties_t
typedef struct _zes_power_properties_t zes_power_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_power_energy_counter_t
typedef struct _zes_power_energy_counter_t zes_power_energy_counter_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_power_sustained_limit_t
typedef struct _zes_power_sustained_limit_t zes_power_sustained_limit_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_power_burst_limit_t
typedef struct _zes_power_burst_limit_t zes_power_burst_limit_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_power_peak_limit_t
typedef struct _zes_power_peak_limit_t zes_power_peak_limit_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_energy_threshold_t
typedef struct _zes_energy_threshold_t zes_energy_threshold_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_psu_properties_t
typedef struct _zes_psu_properties_t zes_psu_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_psu_state_t
typedef struct _zes_psu_state_t zes_psu_state_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_ras_properties_t
typedef struct _zes_ras_properties_t zes_ras_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_ras_state_t
typedef struct _zes_ras_state_t zes_ras_state_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_ras_config_t
typedef struct _zes_ras_config_t zes_ras_config_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_sched_properties_t
typedef struct _zes_sched_properties_t zes_sched_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_sched_timeout_properties_t
typedef struct _zes_sched_timeout_properties_t zes_sched_timeout_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_sched_timeslice_properties_t
typedef struct _zes_sched_timeslice_properties_t zes_sched_timeslice_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_standby_properties_t
typedef struct _zes_standby_properties_t zes_standby_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_temp_properties_t
typedef struct _zes_temp_properties_t zes_temp_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_temp_threshold_t
typedef struct _zes_temp_threshold_t zes_temp_threshold_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zes_temp_config_t
typedef struct _zes_temp_config_t zes_temp_config_t;


#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Tool APIs for System Resource Management (Sysman) - Device management
#if !defined(__GNUC__)
#pragma region device
#endif
///////////////////////////////////////////////////////////////////////////////
#ifndef ZES_STRING_PROPERTY_SIZE
/// @brief Maximum number of characters in string properties.
#define ZES_STRING_PROPERTY_SIZE  64
#endif // ZES_STRING_PROPERTY_SIZE

///////////////////////////////////////////////////////////////////////////////
/// @brief Types of accelerator engines
typedef uint32_t zes_engine_type_flags_t;
typedef enum _zes_engine_type_flag_t
{
    ZES_ENGINE_TYPE_FLAG_OTHER = ZE_BIT(0),         ///< Undefined types of accelerators.
    ZES_ENGINE_TYPE_FLAG_COMPUTE = ZE_BIT(1),       ///< Engines that process compute kernels only (no 3D content).
    ZES_ENGINE_TYPE_FLAG_3D = ZE_BIT(2),            ///< Engines that process 3D content only (no compute kernels).
    ZES_ENGINE_TYPE_FLAG_MEDIA = ZE_BIT(3),         ///< Engines that process media workloads.
    ZES_ENGINE_TYPE_FLAG_DMA = ZE_BIT(4),           ///< Engines that copy blocks of data.
    ZES_ENGINE_TYPE_FLAG_RENDER = ZE_BIT(5),        ///< Engines that can process both 3D content and compute kernels.
    ZES_ENGINE_TYPE_FLAG_FORCE_UINT32 = 0x7fffffff

} zes_engine_type_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Device repair status
typedef enum _zes_repair_status_t
{
    ZES_REPAIR_STATUS_UNSUPPORTED = 0,              ///< The device does not support in-field repairs.
    ZES_REPAIR_STATUS_NOT_PERFORMED = 1,            ///< The device has never been repaired.
    ZES_REPAIR_STATUS_PERFORMED = 2,                ///< The device has been repaired.
    ZES_REPAIR_STATUS_FORCE_UINT32 = 0x7fffffff

} zes_repair_status_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Device reset reasons
typedef uint32_t zes_reset_reason_flags_t;
typedef enum _zes_reset_reason_flag_t
{
    ZES_RESET_REASON_FLAG_WEDGED = ZE_BIT(0),       ///< The device needs to be reset because one or more parts of the hardware
                                                    ///< is wedged
    ZES_RESET_REASON_FLAG_REPAIR = ZE_BIT(1),       ///< The device needs to be reset in order to complete in-field repairs
    ZES_RESET_REASON_FLAG_FORCE_UINT32 = 0x7fffffff

} zes_reset_reason_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Device state
typedef struct _zes_device_state_t
{
    zes_structure_type_t stype;                     ///< [in] type of this structure
    const void* pNext;                              ///< [in][optional] pointer to extension-specific structure
    zes_reset_reason_flags_t reset;                 ///< [out] Indicates if the device needs to be reset and for what reasons.
                                                    ///< returns 0 (none) or combination of ::zes_reset_reason_flag_t
    zes_repair_status_t repaired;                   ///< [out] Indicates if the device has been repaired

} zes_device_state_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Device properties
typedef struct _zes_device_properties_t
{
    zes_structure_type_t stype;                     ///< [in] type of this structure
    void* pNext;                                    ///< [in,out][optional] pointer to extension-specific structure
    ze_device_properties_t core;                    ///< [out] Core device properties
    uint32_t numSubdevices;                         ///< [out] Number of sub-devices. A value of 0 indicates that this device
                                                    ///< doesn't have sub-devices.
    char serialNumber[ZES_STRING_PROPERTY_SIZE];    ///< [out] Manufacturing serial number (NULL terminated string value). Will
                                                    ///< be set to the string "unkown" if this cannot be determined for the
                                                    ///< device.
    char boardNumber[ZES_STRING_PROPERTY_SIZE];     ///< [out] Manufacturing board number (NULL terminated string value). Will
                                                    ///< be set to the string "unkown" if this cannot be determined for the
                                                    ///< device.
    char brandName[ZES_STRING_PROPERTY_SIZE];       ///< [out] Brand name of the device (NULL terminated string value). Will be
                                                    ///< set to the string "unkown" if this cannot be determined for the
                                                    ///< device.
    char modelName[ZES_STRING_PROPERTY_SIZE];       ///< [out] Model name of the device (NULL terminated string value). Will be
                                                    ///< set to the string "unkown" if this cannot be determined for the
                                                    ///< device.
    char vendorName[ZES_STRING_PROPERTY_SIZE];      ///< [out] Vendor name of the device (NULL terminated string value). Will
                                                    ///< be set to the string "unkown" if this cannot be determined for the
                                                    ///< device.
    char driverVersion[ZES_STRING_PROPERTY_SIZE];   ///< [out] Installed driver version (NULL terminated string value). Will be
                                                    ///< set to the string "unkown" if this cannot be determined for the
                                                    ///< device.

} zes_device_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Get properties about the device
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pProperties`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesDeviceGetProperties(
    zes_device_handle_t hDevice,                    ///< [in] Sysman handle of the device.
    zes_device_properties_t* pProperties            ///< [in,out] Structure that will contain information about the device.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get information about the state of the device - if a reset is
///        required, reasons for the reset and if the device has been repaired
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pState`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesDeviceGetState(
    zes_device_handle_t hDevice,                    ///< [in] Sysman handle of the device.
    zes_device_state_t* pState                      ///< [in,out] Structure that will contain information about the device.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Reset device
/// 
/// @details
///     - Performs a PCI bus reset of the device. This will result in all
///       current device state being lost.
///     - All applications using the device should be stopped before calling
///       this function.
///     - If the force argument is specified, all applications using the device
///       will be forcibly killed.
///     - The function will block until the device has restarted or a timeout
///       occurred waiting for the reset to complete.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to perform this operation.
///     - ::ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE - "Reset cannot be performed because applications are using this device."
///     - ::ZE_RESULT_ERROR_UNKNOWN - "There were problems unloading the device driver, performing a bus reset or reloading the device driver."
ZE_APIEXPORT ze_result_t ZE_APICALL
zesDeviceReset(
    zes_device_handle_t hDevice,                    ///< [in] Sysman handle for the device
    ze_bool_t force                                 ///< [in] If set to true, all applications that are currently using the
                                                    ///< device will be forcibly killed.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Contains information about a process that has an open connection with
///        this device
/// 
/// @details
///     - The application can use the process ID to query the OS for the owner
///       and the path to the executable.
typedef struct _zes_process_state_t
{
    zes_structure_type_t stype;                     ///< [in] type of this structure
    const void* pNext;                              ///< [in][optional] pointer to extension-specific structure
    uint32_t processId;                             ///< [out] Host OS process ID.
    uint64_t memSize;                               ///< [out] Device memory size in bytes allocated by this process (may not
                                                    ///< necessarily be resident on the device at the time of reading).
    uint64_t sharedSize;                            ///< [out] The size of shared device memory mapped into this process (may
                                                    ///< not necessarily be resident on the device at the time of reading).
    zes_engine_type_flags_t engines;                ///< [out] Bitfield of accelerator engine types being used by this process.

} zes_process_state_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Get information about host processes using the device
/// 
/// @details
///     - The number of processes connected to the device is dynamic. This means
///       that between a call to determine the value of pCount and the
///       subsequent call, the number of processes may have increased or
///       decreased. It is recommended that a large array be passed in so as to
///       avoid receiving the error ::ZE_RESULT_ERROR_INVALID_SIZE. Also, always
///       check the returned value in pCount since it may be less than the
///       earlier call to get the required array size.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pCount`
///     - ::ZE_RESULT_ERROR_INVALID_SIZE
///         + The provided value of pCount is not big enough to store information about all the processes currently attached to the device.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesDeviceProcessesGetState(
    zes_device_handle_t hDevice,                    ///< [in] Sysman handle for the device
    uint32_t* pCount,                               ///< [in,out] pointer to the number of processes.
                                                    ///< if count is zero, then the driver shall update the value with the
                                                    ///< total number of processes currently attached to the device.
                                                    ///< if count is greater than the number of processes currently attached to
                                                    ///< the device, then the driver shall update the value with the correct
                                                    ///< number of processes.
    zes_process_state_t* pProcesses                 ///< [in,out][optional][range(0, *pCount)] array of process information.
                                                    ///< if count is less than the number of processes currently attached to
                                                    ///< the device, then the driver shall only retrieve information about that
                                                    ///< number of processes. In this case, the return code will ::ZE_RESULT_ERROR_INVALID_SIZE.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief PCI address
typedef struct _zes_pci_address_t
{
    uint32_t domain;                                ///< [out] BDF domain
    uint32_t bus;                                   ///< [out] BDF bus
    uint32_t device;                                ///< [out] BDF device
    uint32_t function;                              ///< [out] BDF function

} zes_pci_address_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief PCI speed
typedef struct _zes_pci_speed_t
{
    int32_t gen;                                    ///< [out] The link generation. A value of -1 means that this property is
                                                    ///< unknown.
    int32_t width;                                  ///< [out] The number of lanes. A value of -1 means that this property is
                                                    ///< unknown.
    int64_t maxBandwidth;                           ///< [out] The maximum bandwidth in bytes/sec (sum of all lanes). A value
                                                    ///< of -1 means that this property is unknown.

} zes_pci_speed_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Static PCI properties
typedef struct _zes_pci_properties_t
{
    zes_structure_type_t stype;                     ///< [in] type of this structure
    void* pNext;                                    ///< [in,out][optional] pointer to extension-specific structure
    zes_pci_address_t address;                      ///< [out] The BDF address
    zes_pci_speed_t maxSpeed;                       ///< [out] Fastest port configuration supported by the device (sum of all
                                                    ///< lanes)
    ze_bool_t haveBandwidthCounters;                ///< [out] Indicates if ::zes_pci_stats_t.rxCounter and
                                                    ///< ::zes_pci_stats_t.txCounter will have valid values
    ze_bool_t havePacketCounters;                   ///< [out] Indicates if ::zes_pci_stats_t.packetCounter will have valid
                                                    ///< values
    ze_bool_t haveReplayCounters;                   ///< [out] Indicates if ::zes_pci_stats_t.replayCounter will have valid
                                                    ///< values

} zes_pci_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief PCI link status
typedef enum _zes_pci_link_status_t
{
    ZES_PCI_LINK_STATUS_UNKNOWN = 0,                ///< The link status could not be determined
    ZES_PCI_LINK_STATUS_GOOD = 1,                   ///< The link is up and operating as expected
    ZES_PCI_LINK_STATUS_QUALITY_ISSUES = 2,         ///< The link is up but has quality and/or bandwidth degradation
    ZES_PCI_LINK_STATUS_STABILITY_ISSUES = 3,       ///< The link has stability issues and preventing workloads making forward
                                                    ///< progress
    ZES_PCI_LINK_STATUS_FORCE_UINT32 = 0x7fffffff

} zes_pci_link_status_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief PCI link quality degradation reasons
typedef uint32_t zes_pci_link_qual_issue_flags_t;
typedef enum _zes_pci_link_qual_issue_flag_t
{
    ZES_PCI_LINK_QUAL_ISSUE_FLAG_REPLAYS = ZE_BIT(0),   ///< A significant number of replays are occurring
    ZES_PCI_LINK_QUAL_ISSUE_FLAG_SPEED = ZE_BIT(1), ///< There is a degradation in the maximum bandwidth of the link
    ZES_PCI_LINK_QUAL_ISSUE_FLAG_FORCE_UINT32 = 0x7fffffff

} zes_pci_link_qual_issue_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief PCI link stability issues
typedef uint32_t zes_pci_link_stab_issue_flags_t;
typedef enum _zes_pci_link_stab_issue_flag_t
{
    ZES_PCI_LINK_STAB_ISSUE_FLAG_RETRAINING = ZE_BIT(0),///< Link retraining has occurred to deal with quality issues
    ZES_PCI_LINK_STAB_ISSUE_FLAG_FORCE_UINT32 = 0x7fffffff

} zes_pci_link_stab_issue_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Dynamic PCI state
typedef struct _zes_pci_state_t
{
    zes_structure_type_t stype;                     ///< [in] type of this structure
    const void* pNext;                              ///< [in][optional] pointer to extension-specific structure
    zes_pci_link_status_t status;                   ///< [out] The current status of the port
    zes_pci_link_qual_issue_flags_t qualityIssues;  ///< [out] If status is ::ZES_PCI_LINK_STATUS_QUALITY_ISSUES, 
                                                    ///< then this gives a combination of ::zes_pci_link_qual_issue_flag_t for
                                                    ///< quality issues that have been detected;
                                                    ///< otherwise, 0 indicates there are no quality issues with the link at
                                                    ///< this time."
    zes_pci_link_stab_issue_flags_t stabilityIssues;///< [out] If status is ::ZES_PCI_LINK_STATUS_STABILITY_ISSUES, 
                                                    ///< then this gives a combination of ::zes_pci_link_stab_issue_flag_t for
                                                    ///< reasons for the connection instability;
                                                    ///< otherwise, 0 indicates there are no connection stability issues at
                                                    ///< this time."
    zes_pci_speed_t speed;                          ///< [out] The current port configure speed

} zes_pci_state_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief PCI bar types
typedef enum _zes_pci_bar_type_t
{
    ZES_PCI_BAR_TYPE_MMIO = 0,                      ///< MMIO registers
    ZES_PCI_BAR_TYPE_ROM = 1,                       ///< ROM aperture
    ZES_PCI_BAR_TYPE_MEM = 2,                       ///< Device memory
    ZES_PCI_BAR_TYPE_FORCE_UINT32 = 0x7fffffff

} zes_pci_bar_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Properties of a pci bar
typedef struct _zes_pci_bar_properties_t
{
    zes_structure_type_t stype;                     ///< [in] type of this structure
    void* pNext;                                    ///< [in,out][optional] pointer to extension-specific structure
    zes_pci_bar_type_t type;                        ///< [out] The type of bar
    uint32_t index;                                 ///< [out] The index of the bar
    uint64_t base;                                  ///< [out] Base address of the bar.
    uint64_t size;                                  ///< [out] Size of the bar.

} zes_pci_bar_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Properties of a pci bar, including the resizable bar.
typedef struct _zes_pci_bar_properties_1_2_t
{
    zes_structure_type_t stype;                     ///< [in] type of this structure
    void* pNext;                                    ///< [in,out][optional] pointer to extension-specific structure
    zes_pci_bar_type_t type;                        ///< [out] The type of bar
    uint32_t index;                                 ///< [out] The index of the bar
    uint64_t base;                                  ///< [out] Base address of the bar.
    uint64_t size;                                  ///< [out] Size of the bar.
    ze_bool_t resizableBarSupported;                ///< [out] Support for Resizable Bar on this device.
    ze_bool_t resizableBarEnabled;                  ///< [out] Resizable Bar enabled on this device

} zes_pci_bar_properties_1_2_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief PCI stats counters
/// 
/// @details
///     - Percent replays is calculated by taking two snapshots (s1, s2) and
///       using the equation: %replay = 10^6 * (s2.replayCounter -
///       s1.replayCounter) / (s2.maxBandwidth * (s2.timestamp - s1.timestamp))
///     - Percent throughput is calculated by taking two snapshots (s1, s2) and
///       using the equation: %bw = 10^6 * ((s2.rxCounter - s1.rxCounter) +
///       (s2.txCounter - s1.txCounter)) / (s2.maxBandwidth * (s2.timestamp -
///       s1.timestamp))
typedef struct _zes_pci_stats_t
{
    uint64_t timestamp;                             ///< [out] Monotonic timestamp counter in microseconds when the measurement
                                                    ///< was made.
                                                    ///< This timestamp should only be used to calculate delta time between
                                                    ///< snapshots of this structure.
                                                    ///< Never take the delta of this timestamp with the timestamp from a
                                                    ///< different structure since they are not guaranteed to have the same base.
                                                    ///< The absolute value of the timestamp is only valid during within the
                                                    ///< application and may be different on the next execution.
    uint64_t replayCounter;                         ///< [out] Monotonic counter for the number of replay packets (sum of all
                                                    ///< lanes). Will always be 0 if ::zes_pci_properties_t.haveReplayCounters
                                                    ///< is FALSE.
    uint64_t packetCounter;                         ///< [out] Monotonic counter for the number of packets (sum of all lanes).
                                                    ///< Will always be 0 if ::zes_pci_properties_t.havePacketCounters is
                                                    ///< FALSE.
    uint64_t rxCounter;                             ///< [out] Monotonic counter for the number of bytes received (sum of all
                                                    ///< lanes). Will always be 0 if
                                                    ///< ::zes_pci_properties_t.haveBandwidthCounters is FALSE.
    uint64_t txCounter;                             ///< [out] Monotonic counter for the number of bytes transmitted (including
                                                    ///< replays) (sum of all lanes). Will always be 0 if
                                                    ///< ::zes_pci_properties_t.haveBandwidthCounters is FALSE.
    zes_pci_speed_t speed;                          ///< [out] The current speed of the link (sum of all lanes)

} zes_pci_stats_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Get PCI properties - address, max speed
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pProperties`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesDevicePciGetProperties(
    zes_device_handle_t hDevice,                    ///< [in] Sysman handle of the device.
    zes_pci_properties_t* pProperties               ///< [in,out] Will contain the PCI properties.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get current PCI state - current speed
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pState`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesDevicePciGetState(
    zes_device_handle_t hDevice,                    ///< [in] Sysman handle of the device.
    zes_pci_state_t* pState                         ///< [in,out] Will contain the PCI properties.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get information about each configured bar
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
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
zesDevicePciGetBars(
    zes_device_handle_t hDevice,                    ///< [in] Sysman handle of the device.
    uint32_t* pCount,                               ///< [in,out] pointer to the number of PCI bars.
                                                    ///< if count is zero, then the driver shall update the value with the
                                                    ///< total number of PCI bars that are setup.
                                                    ///< if count is greater than the number of PCI bars that are setup, then
                                                    ///< the driver shall update the value with the correct number of PCI bars.
    zes_pci_bar_properties_t* pProperties           ///< [in,out][optional][range(0, *pCount)] array of information about setup
                                                    ///< PCI bars.
                                                    ///< if count is less than the number of PCI bars that are setup, then the
                                                    ///< driver shall only retrieve information about that number of PCI bars.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get PCI stats - bandwidth, number of packets, number of replays
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pStats`
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to query this telemetry.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesDevicePciGetStats(
    zes_device_handle_t hDevice,                    ///< [in] Sysman handle of the device.
    zes_pci_stats_t* pStats                         ///< [in,out] Will contain a snapshot of the latest stats.
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Tool APIs for System Resource Management (Sysman) - Firmware management
#if !defined(__GNUC__)
#pragma region diagnostics
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Diagnostic results
typedef enum _zes_diag_result_t
{
    ZES_DIAG_RESULT_NO_ERRORS = 0,                  ///< Diagnostic completed without finding errors to repair
    ZES_DIAG_RESULT_ABORT = 1,                      ///< Diagnostic had problems running tests
    ZES_DIAG_RESULT_FAIL_CANT_REPAIR = 2,           ///< Diagnostic had problems setting up repairs
    ZES_DIAG_RESULT_REBOOT_FOR_REPAIR = 3,          ///< Diagnostics found errors, setup for repair and reboot is required to
                                                    ///< complete the process
    ZES_DIAG_RESULT_FORCE_UINT32 = 0x7fffffff

} zes_diag_result_t;

///////////////////////////////////////////////////////////////////////////////
#ifndef ZES_DIAG_FIRST_TEST_INDEX
/// @brief Diagnostic test index to use for the very first test.
#define ZES_DIAG_FIRST_TEST_INDEX  0x0
#endif // ZES_DIAG_FIRST_TEST_INDEX

///////////////////////////////////////////////////////////////////////////////
#ifndef ZES_DIAG_LAST_TEST_INDEX
/// @brief Diagnostic test index to use for the very last test.
#define ZES_DIAG_LAST_TEST_INDEX  0xFFFFFFFF
#endif // ZES_DIAG_LAST_TEST_INDEX

///////////////////////////////////////////////////////////////////////////////
/// @brief Diagnostic test
typedef struct _zes_diag_test_t
{
    uint32_t index;                                 ///< [out] Index of the test
    char name[ZES_STRING_PROPERTY_SIZE];            ///< [out] Name of the test

} zes_diag_test_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Diagnostics test suite properties
typedef struct _zes_diag_properties_t
{
    zes_structure_type_t stype;                     ///< [in] type of this structure
    void* pNext;                                    ///< [in,out][optional] pointer to extension-specific structure
    ze_bool_t onSubdevice;                          ///< [out] True if the resource is located on a sub-device; false means
                                                    ///< that the resource is on the device of the calling Sysman handle
    uint32_t subdeviceId;                           ///< [out] If onSubdevice is true, this gives the ID of the sub-device
    char name[ZES_STRING_PROPERTY_SIZE];            ///< [out] Name of the diagnostics test suite
    ze_bool_t haveTests;                            ///< [out] Indicates if this test suite has individual tests which can be
                                                    ///< run separately (use the function ::zesDiagnosticsGetTests() to get the
                                                    ///< list of these tests)

} zes_diag_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Get handle of diagnostics test suites
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
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
zesDeviceEnumDiagnosticTestSuites(
    zes_device_handle_t hDevice,                    ///< [in] Sysman handle of the device.
    uint32_t* pCount,                               ///< [in,out] pointer to the number of components of this type.
                                                    ///< if count is zero, then the driver shall update the value with the
                                                    ///< total number of components of this type that are available.
                                                    ///< if count is greater than the number of components of this type that
                                                    ///< are available, then the driver shall update the value with the correct
                                                    ///< number of components.
    zes_diag_handle_t* phDiagnostics                ///< [in,out][optional][range(0, *pCount)] array of handle of components of
                                                    ///< this type.
                                                    ///< if count is less than the number of components of this type that are
                                                    ///< available, then the driver shall only retrieve that number of
                                                    ///< component handles.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get properties of a diagnostics test suite
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDiagnostics`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pProperties`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesDiagnosticsGetProperties(
    zes_diag_handle_t hDiagnostics,                 ///< [in] Handle for the component.
    zes_diag_properties_t* pProperties              ///< [in,out] Structure describing the properties of a diagnostics test
                                                    ///< suite
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get individual tests that can be run separately. Not all test suites
///        permit running individual tests - check
///        ::zes_diag_properties_t.haveTests
/// 
/// @details
///     - The list of available tests is returned in order of increasing test
///       index ::zes_diag_test_t.index.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDiagnostics`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pCount`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesDiagnosticsGetTests(
    zes_diag_handle_t hDiagnostics,                 ///< [in] Handle for the component.
    uint32_t* pCount,                               ///< [in,out] pointer to the number of tests.
                                                    ///< if count is zero, then the driver shall update the value with the
                                                    ///< total number of tests that are available.
                                                    ///< if count is greater than the number of tests that are available, then
                                                    ///< the driver shall update the value with the correct number of tests.
    zes_diag_test_t* pTests                         ///< [in,out][optional][range(0, *pCount)] array of information about
                                                    ///< individual tests sorted by increasing value of ::zes_diag_test_t.index.
                                                    ///< if count is less than the number of tests that are available, then the
                                                    ///< driver shall only retrieve that number of tests.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Run a diagnostics test suite, either all tests or a subset of tests.
/// 
/// @details
///     - WARNING: Running diagnostics may destroy current device state
///       information. Gracefully close any running workloads before initiating.
///     - To run all tests in a test suite, set start =
///       ::ZES_DIAG_FIRST_TEST_INDEX and end = ::ZES_DIAG_LAST_TEST_INDEX.
///     - If the test suite permits running individual tests,
///       ::zes_diag_properties_t.haveTests will be true. In this case, the
///       function ::zesDiagnosticsGetTests() can be called to get the list of
///       tests and corresponding indices that can be supplied to the arguments
///       start and end in this function.
///     - This function will block until the diagnostics have completed and
///       force reset based on result
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDiagnostics`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pResult`
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to perform diagnostics.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesDiagnosticsRunTests(
    zes_diag_handle_t hDiagnostics,                 ///< [in] Handle for the component.
    uint32_t startIndex,                            ///< [in] The index of the first test to run. Set to
                                                    ///< ::ZES_DIAG_FIRST_TEST_INDEX to start from the beginning.
    uint32_t endIndex,                              ///< [in] The index of the last test to run. Set to
                                                    ///< ::ZES_DIAG_LAST_TEST_INDEX to complete all tests after the start test.
    zes_diag_result_t* pResult                      ///< [in,out] The result of the diagnostics
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Tool APIs for System Resource Management (Sysman) - Engine groups
#if !defined(__GNUC__)
#pragma region engine
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Accelerator engine groups
typedef enum _zes_engine_group_t
{
    ZES_ENGINE_GROUP_ALL = 0,                       ///< Access information about all engines combined.
    ZES_ENGINE_GROUP_COMPUTE_ALL = 1,               ///< Access information about all compute engines combined. Compute engines
                                                    ///< can only process compute kernels (no 3D content).
    ZES_ENGINE_GROUP_MEDIA_ALL = 2,                 ///< Access information about all media engines combined.
    ZES_ENGINE_GROUP_COPY_ALL = 3,                  ///< Access information about all copy (blitter) engines combined.
    ZES_ENGINE_GROUP_COMPUTE_SINGLE = 4,            ///< Access information about a single compute engine - this is an engine
                                                    ///< that can process compute kernels. Note that single engines may share
                                                    ///< the same underlying accelerator resources as other engines so activity
                                                    ///< of such an engine may not be indicative of the underlying resource
                                                    ///< utilization - use ::ZES_ENGINE_GROUP_3D_RENDER_COMPUTE_ALL for that.
    ZES_ENGINE_GROUP_RENDER_SINGLE = 5,             ///< Access information about a single render engine - this is an engine
                                                    ///< that can process both 3D content and compute kernels. Note that single
                                                    ///< engines may share the same underlying accelerator resources as other
                                                    ///< engines so activity of such an engine may not be indicative of the
                                                    ///< underlying resource utilization - use
                                                    ///< ::ZES_ENGINE_GROUP_3D_RENDER_COMPUTE_ALL for that.
    ZES_ENGINE_GROUP_MEDIA_DECODE_SINGLE = 6,       ///< Access information about a single media decode engine. Note that
                                                    ///< single engines may share the same underlying accelerator resources as
                                                    ///< other engines so activity of such an engine may not be indicative of
                                                    ///< the underlying resource utilization - use ::ZES_ENGINE_GROUP_MEDIA_ALL
                                                    ///< for that.
    ZES_ENGINE_GROUP_MEDIA_ENCODE_SINGLE = 7,       ///< Access information about a single media encode engine. Note that
                                                    ///< single engines may share the same underlying accelerator resources as
                                                    ///< other engines so activity of such an engine may not be indicative of
                                                    ///< the underlying resource utilization - use ::ZES_ENGINE_GROUP_MEDIA_ALL
                                                    ///< for that.
    ZES_ENGINE_GROUP_COPY_SINGLE = 8,               ///< Access information about a single media encode engine. Note that
                                                    ///< single engines may share the same underlying accelerator resources as
                                                    ///< other engines so activity of such an engine may not be indicative of
                                                    ///< the underlying resource utilization - use ::ZES_ENGINE_GROUP_COPY_ALL
                                                    ///< for that.
    ZES_ENGINE_GROUP_MEDIA_ENHANCEMENT_SINGLE = 9,  ///< Access information about a single media enhancement engine. Note that
                                                    ///< single engines may share the same underlying accelerator resources as
                                                    ///< other engines so activity of such an engine may not be indicative of
                                                    ///< the underlying resource utilization - use ::ZES_ENGINE_GROUP_MEDIA_ALL
                                                    ///< for that.
    ZES_ENGINE_GROUP_3D_SINGLE = 10,                ///< Access information about a single 3D engine - this is an engine that
                                                    ///< can process 3D content only. Note that single engines may share the
                                                    ///< same underlying accelerator resources as other engines so activity of
                                                    ///< such an engine may not be indicative of the underlying resource
                                                    ///< utilization - use ::ZES_ENGINE_GROUP_3D_RENDER_COMPUTE_ALL for that.
    ZES_ENGINE_GROUP_3D_RENDER_COMPUTE_ALL = 11,    ///< Access information about all 3D/render/compute engines combined.
    ZES_ENGINE_GROUP_RENDER_ALL = 12,               ///< Access information about all render engines combined. Render engines
                                                    ///< are those than process both 3D content and compute kernels.
    ZES_ENGINE_GROUP_3D_ALL = 13,                   ///< Access information about all 3D engines combined. 3D engines can
                                                    ///< process 3D content only (no compute kernels).
    ZES_ENGINE_GROUP_FORCE_UINT32 = 0x7fffffff

} zes_engine_group_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Engine group properties
typedef struct _zes_engine_properties_t
{
    zes_structure_type_t stype;                     ///< [in] type of this structure
    void* pNext;                                    ///< [in,out][optional] pointer to extension-specific structure
    zes_engine_group_t type;                        ///< [out] The engine group
    ze_bool_t onSubdevice;                          ///< [out] True if this resource is located on a sub-device; false means
                                                    ///< that the resource is on the device of the calling Sysman handle
    uint32_t subdeviceId;                           ///< [out] If onSubdevice is true, this gives the ID of the sub-device

} zes_engine_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Engine activity counters
/// 
/// @details
///     - Percent utilization is calculated by taking two snapshots (s1, s2) and
///       using the equation: %util = (s2.activeTime - s1.activeTime) /
///       (s2.timestamp - s1.timestamp)
typedef struct _zes_engine_stats_t
{
    uint64_t activeTime;                            ///< [out] Monotonic counter for time in microseconds that this resource is
                                                    ///< actively running workloads.
    uint64_t timestamp;                             ///< [out] Monotonic timestamp counter in microseconds when activeTime
                                                    ///< counter was sampled.
                                                    ///< This timestamp should only be used to calculate delta time between
                                                    ///< snapshots of this structure.
                                                    ///< Never take the delta of this timestamp with the timestamp from a
                                                    ///< different structure since they are not guaranteed to have the same base.
                                                    ///< The absolute value of the timestamp is only valid during within the
                                                    ///< application and may be different on the next execution.

} zes_engine_stats_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Get handle of engine groups
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
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
zesDeviceEnumEngineGroups(
    zes_device_handle_t hDevice,                    ///< [in] Sysman handle of the device.
    uint32_t* pCount,                               ///< [in,out] pointer to the number of components of this type.
                                                    ///< if count is zero, then the driver shall update the value with the
                                                    ///< total number of components of this type that are available.
                                                    ///< if count is greater than the number of components of this type that
                                                    ///< are available, then the driver shall update the value with the correct
                                                    ///< number of components.
    zes_engine_handle_t* phEngine                   ///< [in,out][optional][range(0, *pCount)] array of handle of components of
                                                    ///< this type.
                                                    ///< if count is less than the number of components of this type that are
                                                    ///< available, then the driver shall only retrieve that number of
                                                    ///< component handles.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get engine group properties
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hEngine`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pProperties`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesEngineGetProperties(
    zes_engine_handle_t hEngine,                    ///< [in] Handle for the component.
    zes_engine_properties_t* pProperties            ///< [in,out] The properties for the specified engine group.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get the activity stats for an engine group
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hEngine`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pStats`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesEngineGetActivity(
    zes_engine_handle_t hEngine,                    ///< [in] Handle for the component.
    zes_engine_stats_t* pStats                      ///< [in,out] Will contain a snapshot of the engine group activity
                                                    ///< counters.
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Tool APIs for System Resource Management (Sysman) - Event management
#if !defined(__GNUC__)
#pragma region events
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Event types
typedef uint32_t zes_event_type_flags_t;
typedef enum _zes_event_type_flag_t
{
    ZES_EVENT_TYPE_FLAG_DEVICE_DETACH = ZE_BIT(0),  ///< Event is triggered when the device is no longer available (due to a
                                                    ///< reset or being disabled).
    ZES_EVENT_TYPE_FLAG_DEVICE_ATTACH = ZE_BIT(1),  ///< Event is triggered after the device is available again.
    ZES_EVENT_TYPE_FLAG_DEVICE_SLEEP_STATE_ENTER = ZE_BIT(2),   ///< Event is triggered when the driver is about to put the device into a
                                                    ///< deep sleep state
    ZES_EVENT_TYPE_FLAG_DEVICE_SLEEP_STATE_EXIT = ZE_BIT(3),///< Event is triggered when the driver is waking the device up from a deep
                                                    ///< sleep state
    ZES_EVENT_TYPE_FLAG_FREQ_THROTTLED = ZE_BIT(4), ///< Event is triggered when the frequency starts being throttled
    ZES_EVENT_TYPE_FLAG_ENERGY_THRESHOLD_CROSSED = ZE_BIT(5),   ///< Event is triggered when the energy consumption threshold is reached
                                                    ///< (use ::zesPowerSetEnergyThreshold() to configure).
    ZES_EVENT_TYPE_FLAG_TEMP_CRITICAL = ZE_BIT(6),  ///< Event is triggered when the critical temperature is reached (use
                                                    ///< ::zesTemperatureSetConfig() to configure - disabled by default).
    ZES_EVENT_TYPE_FLAG_TEMP_THRESHOLD1 = ZE_BIT(7),///< Event is triggered when the temperature crosses threshold 1 (use
                                                    ///< ::zesTemperatureSetConfig() to configure - disabled by default).
    ZES_EVENT_TYPE_FLAG_TEMP_THRESHOLD2 = ZE_BIT(8),///< Event is triggered when the temperature crosses threshold 2 (use
                                                    ///< ::zesTemperatureSetConfig() to configure - disabled by default).
    ZES_EVENT_TYPE_FLAG_MEM_HEALTH = ZE_BIT(9),     ///< Event is triggered when the health of device memory changes.
    ZES_EVENT_TYPE_FLAG_FABRIC_PORT_HEALTH = ZE_BIT(10),///< Event is triggered when the health of fabric ports change.
    ZES_EVENT_TYPE_FLAG_PCI_LINK_HEALTH = ZE_BIT(11),   ///< Event is triggered when the health of the PCI link changes.
    ZES_EVENT_TYPE_FLAG_RAS_CORRECTABLE_ERRORS = ZE_BIT(12),///< Event is triggered when accelerator RAS correctable errors cross
                                                    ///< thresholds (use ::zesRasSetConfig() to configure - disabled by
                                                    ///< default).
    ZES_EVENT_TYPE_FLAG_RAS_UNCORRECTABLE_ERRORS = ZE_BIT(13),  ///< Event is triggered when accelerator RAS uncorrectable errors cross
                                                    ///< thresholds (use ::zesRasSetConfig() to configure - disabled by
                                                    ///< default).
    ZES_EVENT_TYPE_FLAG_DEVICE_RESET_REQUIRED = ZE_BIT(14), ///< Event is triggered when the device needs to be reset (use
                                                    ///< ::zesDeviceGetState() to determine the reasons for the reset).
    ZES_EVENT_TYPE_FLAG_FORCE_UINT32 = 0x7fffffff

} zes_event_type_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Specify the list of events to listen to for a given device
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_ENUMERATION
///         + `0x7fff < events`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesDeviceEventRegister(
    zes_device_handle_t hDevice,                    ///< [in] The device handle.
    zes_event_type_flags_t events                   ///< [in] List of events to listen to.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Wait for events to be received from a one or more devices.
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDriver`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == phDevices`
///         + `nullptr == pNumDeviceEvents`
///         + `nullptr == pEvents`
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to listen to events.
///     - ::ZE_RESULT_ERROR_INVALID_ARGUMENT
///         + One or more of the supplied device handles belongs to a different driver.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesDriverEventListen(
    ze_driver_handle_t hDriver,                     ///< [in] handle of the driver instance
    uint32_t timeout,                               ///< [in] if non-zero, then indicates the maximum time (in milliseconds) to
                                                    ///< yield before returning ::ZE_RESULT_SUCCESS or ::ZE_RESULT_NOT_READY;
                                                    ///< if zero, then will check status and return immediately;
                                                    ///< if UINT32_MAX, then function will not return until events arrive.
    uint32_t count,                                 ///< [in] Number of device handles in phDevices.
    zes_device_handle_t* phDevices,                 ///< [in][range(0, count)] Device handles to listen to for events. Only
                                                    ///< devices from the provided driver handle can be specified in this list.
    uint32_t* pNumDeviceEvents,                     ///< [in,out] Will contain the actual number of devices in phDevices that
                                                    ///< generated events. If non-zero, check pEvents to determine the devices
                                                    ///< and events that were received.
    zes_event_type_flags_t* pEvents                 ///< [in,out] An array that will continue the list of events for each
                                                    ///< device listened in phDevices.
                                                    ///< This array must be at least as big as count.
                                                    ///< For every device handle in phDevices, this will provide the events
                                                    ///< that occurred for that device at the same position in this array. If
                                                    ///< no event was received for a given device, the corresponding array
                                                    ///< entry will be zero.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Wait for events to be received from a one or more devices.
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDriver`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == phDevices`
///         + `nullptr == pNumDeviceEvents`
///         + `nullptr == pEvents`
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to listen to events.
///     - ::ZE_RESULT_ERROR_INVALID_ARGUMENT
///         + One or more of the supplied device handles belongs to a different driver.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesDriverEventListenEx(
    ze_driver_handle_t hDriver,                     ///< [in] handle of the driver instance
    uint64_t timeout,                               ///< [in] if non-zero, then indicates the maximum time (in milliseconds) to
                                                    ///< yield before returning ::ZE_RESULT_SUCCESS or ::ZE_RESULT_NOT_READY;
                                                    ///< if zero, then will check status and return immediately;
                                                    ///< if UINT64_MAX, then function will not return until events arrive.
    uint32_t count,                                 ///< [in] Number of device handles in phDevices.
    zes_device_handle_t* phDevices,                 ///< [in][range(0, count)] Device handles to listen to for events. Only
                                                    ///< devices from the provided driver handle can be specified in this list.
    uint32_t* pNumDeviceEvents,                     ///< [in,out] Will contain the actual number of devices in phDevices that
                                                    ///< generated events. If non-zero, check pEvents to determine the devices
                                                    ///< and events that were received.
    zes_event_type_flags_t* pEvents                 ///< [in,out] An array that will continue the list of events for each
                                                    ///< device listened in phDevices.
                                                    ///< This array must be at least as big as count.
                                                    ///< For every device handle in phDevices, this will provide the events
                                                    ///< that occurred for that device at the same position in this array. If
                                                    ///< no event was received for a given device, the corresponding array
                                                    ///< entry will be zero.
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Tool APIs for System Resource Management (Sysman) - Firmware management
#if !defined(__GNUC__)
#pragma region fabric
#endif
///////////////////////////////////////////////////////////////////////////////
#ifndef ZES_MAX_FABRIC_PORT_MODEL_SIZE
/// @brief Maximum Fabric port model string size
#define ZES_MAX_FABRIC_PORT_MODEL_SIZE  256
#endif // ZES_MAX_FABRIC_PORT_MODEL_SIZE

///////////////////////////////////////////////////////////////////////////////
#ifndef ZES_MAX_FABRIC_LINK_TYPE_SIZE
/// @brief Maximum size of the buffer that will return information about link
///        types
#define ZES_MAX_FABRIC_LINK_TYPE_SIZE  256
#endif // ZES_MAX_FABRIC_LINK_TYPE_SIZE

///////////////////////////////////////////////////////////////////////////////
/// @brief Fabric port status
typedef enum _zes_fabric_port_status_t
{
    ZES_FABRIC_PORT_STATUS_UNKNOWN = 0,             ///< The port status cannot be determined
    ZES_FABRIC_PORT_STATUS_HEALTHY = 1,             ///< The port is up and operating as expected
    ZES_FABRIC_PORT_STATUS_DEGRADED = 2,            ///< The port is up but has quality and/or speed degradation
    ZES_FABRIC_PORT_STATUS_FAILED = 3,              ///< Port connection instabilities are preventing workloads making forward
                                                    ///< progress
    ZES_FABRIC_PORT_STATUS_DISABLED = 4,            ///< The port is configured down
    ZES_FABRIC_PORT_STATUS_FORCE_UINT32 = 0x7fffffff

} zes_fabric_port_status_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Fabric port quality degradation reasons
typedef uint32_t zes_fabric_port_qual_issue_flags_t;
typedef enum _zes_fabric_port_qual_issue_flag_t
{
    ZES_FABRIC_PORT_QUAL_ISSUE_FLAG_LINK_ERRORS = ZE_BIT(0),///< Excessive link errors are occurring
    ZES_FABRIC_PORT_QUAL_ISSUE_FLAG_SPEED = ZE_BIT(1),  ///< There is a degradation in the bitrate and/or width of the link
    ZES_FABRIC_PORT_QUAL_ISSUE_FLAG_FORCE_UINT32 = 0x7fffffff

} zes_fabric_port_qual_issue_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Fabric port failure reasons
typedef uint32_t zes_fabric_port_failure_flags_t;
typedef enum _zes_fabric_port_failure_flag_t
{
    ZES_FABRIC_PORT_FAILURE_FLAG_FAILED = ZE_BIT(0),///< A previously operating link has failed. Hardware will automatically
                                                    ///< retrain this port. This state will persist until either the physical
                                                    ///< connection is removed or the link trains successfully.
    ZES_FABRIC_PORT_FAILURE_FLAG_TRAINING_TIMEOUT = ZE_BIT(1),  ///< A connection has not been established within an expected time.
                                                    ///< Hardware will continue to attempt port training. This status will
                                                    ///< persist until either the physical connection is removed or the link
                                                    ///< successfully trains.
    ZES_FABRIC_PORT_FAILURE_FLAG_FLAPPING = ZE_BIT(2),  ///< Port has excessively trained and then transitioned down for some
                                                    ///< period of time. Driver will allow port to continue to train, but will
                                                    ///< not enable the port for use until the port has been disabled and
                                                    ///< subsequently re-enabled using ::zesFabricPortSetConfig().
    ZES_FABRIC_PORT_FAILURE_FLAG_FORCE_UINT32 = 0x7fffffff

} zes_fabric_port_failure_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Unique identifier for a fabric port
/// 
/// @details
///     - This not a universal identifier. The identified is garanteed to be
///       unique for the current hardware configuration of the system. Changes
///       in the hardware may result in a different identifier for a given port.
///     - The main purpose of this identifier to build up an instantaneous
///       topology map of system connectivity. An application should enumerate
///       all fabric ports and match ::zes_fabric_port_state_t.remotePortId to
///       ::zes_fabric_port_properties_t.portId.
typedef struct _zes_fabric_port_id_t
{
    uint32_t fabricId;                              ///< [out] Unique identifier for the fabric end-point
    uint32_t attachId;                              ///< [out] Unique identifier for the device attachment point
    uint8_t portNumber;                             ///< [out] The logical port number (this is typically marked somewhere on
                                                    ///< the physical device)

} zes_fabric_port_id_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Fabric port speed in one direction
typedef struct _zes_fabric_port_speed_t
{
    int64_t bitRate;                                ///< [out] Bits/sec that the link is operating at. A value of -1 means that
                                                    ///< this property is unknown.
    int32_t width;                                  ///< [out] The number of lanes. A value of -1 means that this property is
                                                    ///< unknown.

} zes_fabric_port_speed_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Fabric port properties
typedef struct _zes_fabric_port_properties_t
{
    zes_structure_type_t stype;                     ///< [in] type of this structure
    void* pNext;                                    ///< [in,out][optional] pointer to extension-specific structure
    char model[ZES_MAX_FABRIC_PORT_MODEL_SIZE];     ///< [out] Description of port technology. Will be set to the string
                                                    ///< "unkown" if this cannot be determined for this port.
    ze_bool_t onSubdevice;                          ///< [out] True if the port is located on a sub-device; false means that
                                                    ///< the port is on the device of the calling Sysman handle
    uint32_t subdeviceId;                           ///< [out] If onSubdevice is true, this gives the ID of the sub-device
    zes_fabric_port_id_t portId;                    ///< [out] The unique port identifier
    zes_fabric_port_speed_t maxRxSpeed;             ///< [out] Maximum speed supported by the receive side of the port (sum of
                                                    ///< all lanes)
    zes_fabric_port_speed_t maxTxSpeed;             ///< [out] Maximum speed supported by the transmit side of the port (sum of
                                                    ///< all lanes)

} zes_fabric_port_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Provides information about the fabric link attached to a port
typedef struct _zes_fabric_link_type_t
{
    char desc[ZES_MAX_FABRIC_LINK_TYPE_SIZE];       ///< [out] This provides a static textural description of the physic
                                                    ///< attachment type. Will be set to the string "unkown" if this cannot be
                                                    ///< determined for this port.

} zes_fabric_link_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Fabric port configuration
typedef struct _zes_fabric_port_config_t
{
    zes_structure_type_t stype;                     ///< [in] type of this structure
    const void* pNext;                              ///< [in][optional] pointer to extension-specific structure
    ze_bool_t enabled;                              ///< [in,out] Port is configured up/down
    ze_bool_t beaconing;                            ///< [in,out] Beaconing is configured on/off

} zes_fabric_port_config_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Fabric port state
typedef struct _zes_fabric_port_state_t
{
    zes_structure_type_t stype;                     ///< [in] type of this structure
    const void* pNext;                              ///< [in][optional] pointer to extension-specific structure
    zes_fabric_port_status_t status;                ///< [out] The current status of the port
    zes_fabric_port_qual_issue_flags_t qualityIssues;   ///< [out] If status is ::ZES_FABRIC_PORT_STATUS_DEGRADED, 
                                                    ///< then this gives a combination of ::zes_fabric_port_qual_issue_flag_t
                                                    ///< for quality issues that have been detected;
                                                    ///< otherwise, 0 indicates there are no quality issues with the link at
                                                    ///< this time.
    zes_fabric_port_failure_flags_t failureReasons; ///< [out] If status is ::ZES_FABRIC_PORT_STATUS_FAILED,
                                                    ///< then this gives a combination of ::zes_fabric_port_failure_flag_t for
                                                    ///< reasons for the connection instability;
                                                    ///< otherwise, 0 indicates there are no connection stability issues at
                                                    ///< this time.
    zes_fabric_port_id_t remotePortId;              ///< [out] The unique port identifier for the remote connection point if
                                                    ///< status is ::ZES_FABRIC_PORT_STATUS_HEALTHY,
                                                    ///< ::ZES_FABRIC_PORT_STATUS_DEGRADED or ::ZES_FABRIC_PORT_STATUS_FAILED
    zes_fabric_port_speed_t rxSpeed;                ///< [out] Current maximum receive speed (sum of all lanes)
    zes_fabric_port_speed_t txSpeed;                ///< [out] Current maximum transmit speed (sum of all lanes)

} zes_fabric_port_state_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Fabric port throughput.
typedef struct _zes_fabric_port_throughput_t
{
    uint64_t timestamp;                             ///< [out] Monotonic timestamp counter in microseconds when the measurement
                                                    ///< was made.
                                                    ///< This timestamp should only be used to calculate delta time between
                                                    ///< snapshots of this structure.
                                                    ///< Never take the delta of this timestamp with the timestamp from a
                                                    ///< different structure since they are not guaranteed to have the same base.
                                                    ///< The absolute value of the timestamp is only valid during within the
                                                    ///< application and may be different on the next execution.
    uint64_t rxCounter;                             ///< [out] Monotonic counter for the number of bytes received (sum of all
                                                    ///< lanes). This includes all protocol overhead, not only the GPU traffic.
    uint64_t txCounter;                             ///< [out] Monotonic counter for the number of bytes transmitted (sum of
                                                    ///< all lanes). This includes all protocol overhead, not only the GPU
                                                    ///< traffic.

} zes_fabric_port_throughput_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Get handle of Fabric ports in a device
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
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
zesDeviceEnumFabricPorts(
    zes_device_handle_t hDevice,                    ///< [in] Sysman handle of the device.
    uint32_t* pCount,                               ///< [in,out] pointer to the number of components of this type.
                                                    ///< if count is zero, then the driver shall update the value with the
                                                    ///< total number of components of this type that are available.
                                                    ///< if count is greater than the number of components of this type that
                                                    ///< are available, then the driver shall update the value with the correct
                                                    ///< number of components.
    zes_fabric_port_handle_t* phPort                ///< [in,out][optional][range(0, *pCount)] array of handle of components of
                                                    ///< this type.
                                                    ///< if count is less than the number of components of this type that are
                                                    ///< available, then the driver shall only retrieve that number of
                                                    ///< component handles.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get Fabric port properties
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hPort`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pProperties`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesFabricPortGetProperties(
    zes_fabric_port_handle_t hPort,                 ///< [in] Handle for the component.
    zes_fabric_port_properties_t* pProperties       ///< [in,out] Will contain properties of the Fabric Port.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get Fabric port link type
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hPort`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pLinkType`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesFabricPortGetLinkType(
    zes_fabric_port_handle_t hPort,                 ///< [in] Handle for the component.
    zes_fabric_link_type_t* pLinkType               ///< [in,out] Will contain details about the link attached to the Fabric
                                                    ///< port.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get Fabric port configuration
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hPort`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pConfig`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesFabricPortGetConfig(
    zes_fabric_port_handle_t hPort,                 ///< [in] Handle for the component.
    zes_fabric_port_config_t* pConfig               ///< [in,out] Will contain configuration of the Fabric Port.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Set Fabric port configuration
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hPort`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pConfig`
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to make these modifications.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesFabricPortSetConfig(
    zes_fabric_port_handle_t hPort,                 ///< [in] Handle for the component.
    const zes_fabric_port_config_t* pConfig         ///< [in] Contains new configuration of the Fabric Port.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get Fabric port state - status (health/degraded/failed/disabled),
///        reasons for link degradation or instability, current rx/tx speed
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hPort`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pState`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesFabricPortGetState(
    zes_fabric_port_handle_t hPort,                 ///< [in] Handle for the component.
    zes_fabric_port_state_t* pState                 ///< [in,out] Will contain the current state of the Fabric Port
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get Fabric port throughput
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hPort`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pThroughput`
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to query this telemetry.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesFabricPortGetThroughput(
    zes_fabric_port_handle_t hPort,                 ///< [in] Handle for the component.
    zes_fabric_port_throughput_t* pThroughput       ///< [in,out] Will contain the Fabric port throughput counters.
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Tool APIs for System Resource Management (Sysman) - Firmware management
#if !defined(__GNUC__)
#pragma region fan
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Fan resource speed mode
typedef enum _zes_fan_speed_mode_t
{
    ZES_FAN_SPEED_MODE_DEFAULT = 0,                 ///< The fan speed is operating using the hardware default settings
    ZES_FAN_SPEED_MODE_FIXED = 1,                   ///< The fan speed is currently set to a fixed value
    ZES_FAN_SPEED_MODE_TABLE = 2,                   ///< The fan speed is currently controlled dynamically by hardware based on
                                                    ///< a temp/speed table
    ZES_FAN_SPEED_MODE_FORCE_UINT32 = 0x7fffffff

} zes_fan_speed_mode_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Fan speed units
typedef enum _zes_fan_speed_units_t
{
    ZES_FAN_SPEED_UNITS_RPM = 0,                    ///< The fan speed is in units of revolutions per minute (rpm)
    ZES_FAN_SPEED_UNITS_PERCENT = 1,                ///< The fan speed is a percentage of the maximum speed of the fan
    ZES_FAN_SPEED_UNITS_FORCE_UINT32 = 0x7fffffff

} zes_fan_speed_units_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Fan speed
typedef struct _zes_fan_speed_t
{
    int32_t speed;                                  ///< [in,out] The speed of the fan. On output, a value of -1 indicates that
                                                    ///< there is no fixed fan speed setting.
    zes_fan_speed_units_t units;                    ///< [in,out] The units that the fan speed is expressed in. On output, if
                                                    ///< fan speed is -1 then units should be ignored.

} zes_fan_speed_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Fan temperature/speed pair
typedef struct _zes_fan_temp_speed_t
{
    uint32_t temperature;                           ///< [in,out] Temperature in degrees Celsius.
    zes_fan_speed_t speed;                          ///< [in,out] The speed of the fan

} zes_fan_temp_speed_t;

///////////////////////////////////////////////////////////////////////////////
#ifndef ZES_FAN_TEMP_SPEED_PAIR_COUNT
/// @brief Maximum number of fan temperature/speed pairs in the fan speed table.
#define ZES_FAN_TEMP_SPEED_PAIR_COUNT  32
#endif // ZES_FAN_TEMP_SPEED_PAIR_COUNT

///////////////////////////////////////////////////////////////////////////////
/// @brief Fan speed table
typedef struct _zes_fan_speed_table_t
{
    int32_t numPoints;                              ///< [in,out] The number of valid points in the fan speed table. 0 means
                                                    ///< that there is no fan speed table configured. -1 means that a fan speed
                                                    ///< table is not supported by the hardware.
    zes_fan_temp_speed_t table[ZES_FAN_TEMP_SPEED_PAIR_COUNT];  ///< [in,out] Array of temperature/fan speed pairs. The table is ordered
                                                    ///< based on temperature from lowest to highest.

} zes_fan_speed_table_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Fan properties
typedef struct _zes_fan_properties_t
{
    zes_structure_type_t stype;                     ///< [in] type of this structure
    void* pNext;                                    ///< [in,out][optional] pointer to extension-specific structure
    ze_bool_t onSubdevice;                          ///< [out] True if the resource is located on a sub-device; false means
                                                    ///< that the resource is on the device of the calling Sysman handle
    uint32_t subdeviceId;                           ///< [out] If onSubdevice is true, this gives the ID of the sub-device
    ze_bool_t canControl;                           ///< [out] Indicates if software can control the fan speed assuming the
                                                    ///< user has permissions
    uint32_t supportedModes;                        ///< [out] Bitfield of supported fan configuration modes
                                                    ///< (1<<::zes_fan_speed_mode_t)
    uint32_t supportedUnits;                        ///< [out] Bitfield of supported fan speed units
                                                    ///< (1<<::zes_fan_speed_units_t)
    int32_t maxRPM;                                 ///< [out] The maximum RPM of the fan. A value of -1 means that this
                                                    ///< property is unknown. 
    int32_t maxPoints;                              ///< [out] The maximum number of points in the fan temp/speed table. A
                                                    ///< value of -1 means that this fan doesn't support providing a temp/speed
                                                    ///< table.

} zes_fan_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Fan configuration
typedef struct _zes_fan_config_t
{
    zes_structure_type_t stype;                     ///< [in] type of this structure
    const void* pNext;                              ///< [in][optional] pointer to extension-specific structure
    zes_fan_speed_mode_t mode;                      ///< [in,out] The fan speed mode (fixed, temp-speed table)
    zes_fan_speed_t speedFixed;                     ///< [in,out] The current fixed fan speed setting
    zes_fan_speed_table_t speedTable;               ///< [out] A table containing temperature/speed pairs

} zes_fan_config_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Get handle of fans
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
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
zesDeviceEnumFans(
    zes_device_handle_t hDevice,                    ///< [in] Sysman handle of the device.
    uint32_t* pCount,                               ///< [in,out] pointer to the number of components of this type.
                                                    ///< if count is zero, then the driver shall update the value with the
                                                    ///< total number of components of this type that are available.
                                                    ///< if count is greater than the number of components of this type that
                                                    ///< are available, then the driver shall update the value with the correct
                                                    ///< number of components.
    zes_fan_handle_t* phFan                         ///< [in,out][optional][range(0, *pCount)] array of handle of components of
                                                    ///< this type.
                                                    ///< if count is less than the number of components of this type that are
                                                    ///< available, then the driver shall only retrieve that number of
                                                    ///< component handles.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get fan properties
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hFan`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pProperties`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesFanGetProperties(
    zes_fan_handle_t hFan,                          ///< [in] Handle for the component.
    zes_fan_properties_t* pProperties               ///< [in,out] Will contain the properties of the fan.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get fan configurations and the current fan speed mode (default, fixed,
///        temp-speed table)
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hFan`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pConfig`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesFanGetConfig(
    zes_fan_handle_t hFan,                          ///< [in] Handle for the component.
    zes_fan_config_t* pConfig                       ///< [in,out] Will contain the current configuration of the fan.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Configure the fan to run with hardware factory settings (set mode to
///        ::ZES_FAN_SPEED_MODE_DEFAULT)
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hFan`
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to make these modifications.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesFanSetDefaultMode(
    zes_fan_handle_t hFan                           ///< [in] Handle for the component.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Configure the fan to rotate at a fixed speed (set mode to
///        ::ZES_FAN_SPEED_MODE_FIXED)
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hFan`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == speed`
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to make these modifications.
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + Fixing the fan speed not supported by the hardware or the fan speed units are not supported. See ::zes_fan_properties_t.supportedModes and ::zes_fan_properties_t.supportedUnits.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesFanSetFixedSpeedMode(
    zes_fan_handle_t hFan,                          ///< [in] Handle for the component.
    const zes_fan_speed_t* speed                    ///< [in] The fixed fan speed setting
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Configure the fan to adjust speed based on a temperature/speed table
///        (set mode to ::ZES_FAN_SPEED_MODE_TABLE)
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hFan`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == speedTable`
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to make these modifications.
///     - ::ZE_RESULT_ERROR_INVALID_ARGUMENT
///         + The temperature/speed pairs in the array are not sorted on temperature from lowest to highest.
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + Fan speed table not supported by the hardware or the fan speed units are not supported. See ::zes_fan_properties_t.supportedModes and ::zes_fan_properties_t.supportedUnits.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesFanSetSpeedTableMode(
    zes_fan_handle_t hFan,                          ///< [in] Handle for the component.
    const zes_fan_speed_table_t* speedTable         ///< [in] A table containing temperature/speed pairs.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get current state of a fan - current mode and speed
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hFan`
///     - ::ZE_RESULT_ERROR_INVALID_ENUMERATION
///         + `::ZES_FAN_SPEED_UNITS_PERCENT < units`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pSpeed`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + The requested fan speed units are not supported. See ::zes_fan_properties_t.supportedUnits.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesFanGetState(
    zes_fan_handle_t hFan,                          ///< [in] Handle for the component.
    zes_fan_speed_units_t units,                    ///< [in] The units in which the fan speed should be returned.
    int32_t* pSpeed                                 ///< [in,out] Will contain the current speed of the fan in the units
                                                    ///< requested. A value of -1 indicates that the fan speed cannot be
                                                    ///< measured.
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Tool APIs for System Resource Management (Sysman) - Firmware management
#if !defined(__GNUC__)
#pragma region firmware
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Firmware properties
typedef struct _zes_firmware_properties_t
{
    zes_structure_type_t stype;                     ///< [in] type of this structure
    void* pNext;                                    ///< [in,out][optional] pointer to extension-specific structure
    ze_bool_t onSubdevice;                          ///< [out] True if the resource is located on a sub-device; false means
                                                    ///< that the resource is on the device of the calling Sysman handle
    uint32_t subdeviceId;                           ///< [out] If onSubdevice is true, this gives the ID of the sub-device
    ze_bool_t canControl;                           ///< [out] Indicates if software can flash the firmware assuming the user
                                                    ///< has permissions
    char name[ZES_STRING_PROPERTY_SIZE];            ///< [out] NULL terminated string value. The string "unknown" will be
                                                    ///< returned if this property cannot be determined.
    char version[ZES_STRING_PROPERTY_SIZE];         ///< [out] NULL terminated string value. The string "unknown" will be
                                                    ///< returned if this property cannot be determined.

} zes_firmware_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Get handle of firmwares
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
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
zesDeviceEnumFirmwares(
    zes_device_handle_t hDevice,                    ///< [in] Sysman handle of the device.
    uint32_t* pCount,                               ///< [in,out] pointer to the number of components of this type.
                                                    ///< if count is zero, then the driver shall update the value with the
                                                    ///< total number of components of this type that are available.
                                                    ///< if count is greater than the number of components of this type that
                                                    ///< are available, then the driver shall update the value with the correct
                                                    ///< number of components.
    zes_firmware_handle_t* phFirmware               ///< [in,out][optional][range(0, *pCount)] array of handle of components of
                                                    ///< this type.
                                                    ///< if count is less than the number of components of this type that are
                                                    ///< available, then the driver shall only retrieve that number of
                                                    ///< component handles.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get firmware properties
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hFirmware`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pProperties`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesFirmwareGetProperties(
    zes_firmware_handle_t hFirmware,                ///< [in] Handle for the component.
    zes_firmware_properties_t* pProperties          ///< [in,out] Pointer to an array that will hold the properties of the
                                                    ///< firmware
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Flash a new firmware image
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hFirmware`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pImage`
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to perform this operation.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesFirmwareFlash(
    zes_firmware_handle_t hFirmware,                ///< [in] Handle for the component.
    void* pImage,                                   ///< [in] Image of the new firmware to flash.
    uint32_t size                                   ///< [in] Size of the flash image.
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Tool APIs for System Resource Management (Sysman) - Frequency domains
#if !defined(__GNUC__)
#pragma region frequency
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Frequency domains.
typedef enum _zes_freq_domain_t
{
    ZES_FREQ_DOMAIN_GPU = 0,                        ///< GPU Core Domain.
    ZES_FREQ_DOMAIN_MEMORY = 1,                     ///< Local Memory Domain.
    ZES_FREQ_DOMAIN_FORCE_UINT32 = 0x7fffffff

} zes_freq_domain_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Frequency properties
/// 
/// @details
///     - Indicates if this frequency domain can be overclocked (if true,
///       functions such as ::zesFrequencyOcSetFrequencyTarget() are supported).
///     - The min/max hardware frequencies are specified for non-overclock
///       configurations. For overclock configurations, use
///       ::zesFrequencyOcGetFrequencyTarget() to determine the maximum
///       frequency that can be requested.
typedef struct _zes_freq_properties_t
{
    zes_structure_type_t stype;                     ///< [in] type of this structure
    void* pNext;                                    ///< [in,out][optional] pointer to extension-specific structure
    zes_freq_domain_t type;                         ///< [out] The hardware block that this frequency domain controls (GPU,
                                                    ///< memory, ...)
    ze_bool_t onSubdevice;                          ///< [out] True if this resource is located on a sub-device; false means
                                                    ///< that the resource is on the device of the calling Sysman handle
    uint32_t subdeviceId;                           ///< [out] If onSubdevice is true, this gives the ID of the sub-device
    ze_bool_t canControl;                           ///< [out] Indicates if software can control the frequency of this domain
                                                    ///< assuming the user has permissions
    ze_bool_t isThrottleEventSupported;             ///< [out] Indicates if software can register to receive event
                                                    ///< ::ZES_EVENT_TYPE_FLAG_FREQ_THROTTLED
    double min;                                     ///< [out] The minimum hardware clock frequency in units of MHz.
    double max;                                     ///< [out] The maximum non-overclock hardware clock frequency in units of
                                                    ///< MHz.

} zes_freq_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Frequency range between which the hardware can operate. The limits can
///        be above or below the hardware limits - the hardware will clamp
///        appropriately.
typedef struct _zes_freq_range_t
{
    double min;                                     ///< [in,out] The min frequency in MHz below which hardware frequency
                                                    ///< management will not request frequencies. On input, setting to 0 will
                                                    ///< permit the frequency to go down to the hardware minimum. On output, a
                                                    ///< negative value indicates that no external minimum frequency limit is
                                                    ///< in effect.
    double max;                                     ///< [in,out] The max frequency in MHz above which hardware frequency
                                                    ///< management will not request frequencies. On input, setting to 0 or a
                                                    ///< very big number will permit the frequency to go all the way up to the
                                                    ///< hardware maximum. On output, a negative number indicates that no
                                                    ///< external maximum frequency limit is in effect.

} zes_freq_range_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Frequency throttle reasons
typedef uint32_t zes_freq_throttle_reason_flags_t;
typedef enum _zes_freq_throttle_reason_flag_t
{
    ZES_FREQ_THROTTLE_REASON_FLAG_AVE_PWR_CAP = ZE_BIT(0),  ///< frequency throttled due to average power excursion (PL1)
    ZES_FREQ_THROTTLE_REASON_FLAG_BURST_PWR_CAP = ZE_BIT(1),///< frequency throttled due to burst power excursion (PL2)
    ZES_FREQ_THROTTLE_REASON_FLAG_CURRENT_LIMIT = ZE_BIT(2),///< frequency throttled due to current excursion (PL4)
    ZES_FREQ_THROTTLE_REASON_FLAG_THERMAL_LIMIT = ZE_BIT(3),///< frequency throttled due to thermal excursion (T > TjMax)
    ZES_FREQ_THROTTLE_REASON_FLAG_PSU_ALERT = ZE_BIT(4),///< frequency throttled due to power supply assertion
    ZES_FREQ_THROTTLE_REASON_FLAG_SW_RANGE = ZE_BIT(5), ///< frequency throttled due to software supplied frequency range
    ZES_FREQ_THROTTLE_REASON_FLAG_HW_RANGE = ZE_BIT(6), ///< frequency throttled due to a sub block that has a lower frequency
                                                    ///< range when it receives clocks
    ZES_FREQ_THROTTLE_REASON_FLAG_FORCE_UINT32 = 0x7fffffff

} zes_freq_throttle_reason_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Frequency state
typedef struct _zes_freq_state_t
{
    zes_structure_type_t stype;                     ///< [in] type of this structure
    const void* pNext;                              ///< [in][optional] pointer to extension-specific structure
    double currentVoltage;                          ///< [out] Current voltage in Volts. A negative value indicates that this
                                                    ///< property is not known.
    double request;                                 ///< [out] The current frequency request in MHz. A negative value indicates
                                                    ///< that this property is not known.
    double tdp;                                     ///< [out] The maximum frequency in MHz supported under the current TDP
                                                    ///< conditions. This fluctuates dynamically based on the power and thermal
                                                    ///< limits of the part. A negative value indicates that this property is
                                                    ///< not known.
    double efficient;                               ///< [out] The efficient minimum frequency in MHz. A negative value
                                                    ///< indicates that this property is not known.
    double actual;                                  ///< [out] The resolved frequency in MHz. A negative value indicates that
                                                    ///< this property is not known.
    zes_freq_throttle_reason_flags_t throttleReasons;   ///< [out] The reasons that the frequency is being limited by the hardware.
                                                    ///< Returns 0 (frequency not throttled) or a combination of ::zes_freq_throttle_reason_flag_t.

} zes_freq_state_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Frequency throttle time snapshot
/// 
/// @details
///     - Percent time throttled is calculated by taking two snapshots (s1, s2)
///       and using the equation: %throttled = (s2.throttleTime -
///       s1.throttleTime) / (s2.timestamp - s1.timestamp)
typedef struct _zes_freq_throttle_time_t
{
    uint64_t throttleTime;                          ///< [out] The monotonic counter of time in microseconds that the frequency
                                                    ///< has been limited by the hardware.
    uint64_t timestamp;                             ///< [out] Microsecond timestamp when throttleTime was captured.
                                                    ///< This timestamp should only be used to calculate delta time between
                                                    ///< snapshots of this structure.
                                                    ///< Never take the delta of this timestamp with the timestamp from a
                                                    ///< different structure since they are not guaranteed to have the same base.
                                                    ///< The absolute value of the timestamp is only valid during within the
                                                    ///< application and may be different on the next execution.

} zes_freq_throttle_time_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Overclocking modes
typedef enum _zes_oc_mode_t
{
    ZES_OC_MODE_OFF = 0,                            ///< Overclocking if off - hardware is running using factory default
                                                    ///< voltages/frequencies.
    ZES_OC_MODE_OVERRIDE = 1,                       ///< Overclock override mode - In this mode, a fixed user-supplied voltage
                                                    ///< is applied independent of the frequency request. The maximum permitted
                                                    ///< frequency can also be increased. This mode disables INTERPOLATIVE and
                                                    ///< FIXED modes.
    ZES_OC_MODE_INTERPOLATIVE = 2,                  ///< Overclock interpolative mode - In this mode, the voltage/frequency
                                                    ///< curve can be extended with a new voltage/frequency point that will be
                                                    ///< interpolated. The existing voltage/frequency points can also be offset
                                                    ///< (up or down) by a fixed voltage. This mode disables FIXED and OVERRIDE
                                                    ///< modes.
    ZES_OC_MODE_FIXED = 3,                          ///< Overclocking fixed Mode - In this mode, hardware will disable most
                                                    ///< frequency throttling and lock the frequency and voltage at the
                                                    ///< specified overclock values. This mode disables OVERRIDE and
                                                    ///< INTERPOLATIVE modes. This mode can damage the part, most of the
                                                    ///< protections are disabled on this mode.
    ZES_OC_MODE_FORCE_UINT32 = 0x7fffffff

} zes_oc_mode_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Overclocking properties
/// 
/// @details
///     - Provides all the overclocking capabilities and properties supported by
///       the device for the frequency domain.
typedef struct _zes_oc_capabilities_t
{
    zes_structure_type_t stype;                     ///< [in] type of this structure
    const void* pNext;                              ///< [in][optional] pointer to extension-specific structure
    ze_bool_t isOcSupported;                        ///< [out] Indicates if any overclocking features are supported on this
                                                    ///< frequency domain.
    double maxFactoryDefaultFrequency;              ///< [out] Factory default non-overclock maximum frequency in Mhz.
    double maxFactoryDefaultVoltage;                ///< [out] Factory default voltage used for the non-overclock maximum
                                                    ///< frequency in MHz.
    double maxOcFrequency;                          ///< [out] Maximum hardware overclocking frequency limit in Mhz.
    double minOcVoltageOffset;                      ///< [out] The minimum voltage offset that can be applied to the
                                                    ///< voltage/frequency curve. Note that this number can be negative.
    double maxOcVoltageOffset;                      ///< [out] The maximum voltage offset that can be applied to the
                                                    ///< voltage/frequency curve.
    double maxOcVoltage;                            ///< [out] The maximum overclock voltage that hardware supports.
    ze_bool_t isTjMaxSupported;                     ///< [out] Indicates if the maximum temperature limit (TjMax) can be
                                                    ///< changed for this frequency domain.
    ze_bool_t isIccMaxSupported;                    ///< [out] Indicates if the maximum current (IccMax) can be changed for
                                                    ///< this frequency domain.
    ze_bool_t isHighVoltModeCapable;                ///< [out] Indicates if this frequency domains supports a feature to set
                                                    ///< very high voltages.
    ze_bool_t isHighVoltModeEnabled;                ///< [out] Indicates if very high voltages are permitted on this frequency
                                                    ///< domain.
    ze_bool_t isExtendedModeSupported;              ///< [out] Indicates if the extended overclocking features are supported.
                                                    ///< If this is supported, increments are on 1 Mhz basis.
    ze_bool_t isFixedModeSupported;                 ///< [out] Indicates if the fixed mode is supported. In this mode, hardware
                                                    ///< will disable most frequency throttling and lock the frequency and
                                                    ///< voltage at the specified overclock values.

} zes_oc_capabilities_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Get handle of frequency domains
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
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
zesDeviceEnumFrequencyDomains(
    zes_device_handle_t hDevice,                    ///< [in] Sysman handle of the device.
    uint32_t* pCount,                               ///< [in,out] pointer to the number of components of this type.
                                                    ///< if count is zero, then the driver shall update the value with the
                                                    ///< total number of components of this type that are available.
                                                    ///< if count is greater than the number of components of this type that
                                                    ///< are available, then the driver shall update the value with the correct
                                                    ///< number of components.
    zes_freq_handle_t* phFrequency                  ///< [in,out][optional][range(0, *pCount)] array of handle of components of
                                                    ///< this type.
                                                    ///< if count is less than the number of components of this type that are
                                                    ///< available, then the driver shall only retrieve that number of
                                                    ///< component handles.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get frequency properties - available frequencies
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hFrequency`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pProperties`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesFrequencyGetProperties(
    zes_freq_handle_t hFrequency,                   ///< [in] Handle for the component.
    zes_freq_properties_t* pProperties              ///< [in,out] The frequency properties for the specified domain.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get available non-overclocked hardware clock frequencies for the
///        frequency domain
/// 
/// @details
///     - The list of available frequencies is returned in order of slowest to
///       fastest.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hFrequency`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pCount`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesFrequencyGetAvailableClocks(
    zes_freq_handle_t hFrequency,                   ///< [in] Sysman handle of the device.
    uint32_t* pCount,                               ///< [in,out] pointer to the number of frequencies.
                                                    ///< if count is zero, then the driver shall update the value with the
                                                    ///< total number of frequencies that are available.
                                                    ///< if count is greater than the number of frequencies that are available,
                                                    ///< then the driver shall update the value with the correct number of frequencies.
    double* phFrequency                             ///< [in,out][optional][range(0, *pCount)] array of frequencies in units of
                                                    ///< MHz and sorted from slowest to fastest.
                                                    ///< if count is less than the number of frequencies that are available,
                                                    ///< then the driver shall only retrieve that number of frequencies.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get current frequency limits
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hFrequency`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pLimits`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesFrequencyGetRange(
    zes_freq_handle_t hFrequency,                   ///< [in] Handle for the component.
    zes_freq_range_t* pLimits                       ///< [in,out] The range between which the hardware can operate for the
                                                    ///< specified domain.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Set frequency range between which the hardware can operate.
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hFrequency`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pLimits`
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to make these modifications.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesFrequencySetRange(
    zes_freq_handle_t hFrequency,                   ///< [in] Handle for the component.
    const zes_freq_range_t* pLimits                 ///< [in] The limits between which the hardware can operate for the
                                                    ///< specified domain.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get current frequency state - frequency request, actual frequency, TDP
///        limits
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hFrequency`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pState`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesFrequencyGetState(
    zes_freq_handle_t hFrequency,                   ///< [in] Handle for the component.
    zes_freq_state_t* pState                        ///< [in,out] Frequency state for the specified domain.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get frequency throttle time
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hFrequency`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pThrottleTime`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesFrequencyGetThrottleTime(
    zes_freq_handle_t hFrequency,                   ///< [in] Handle for the component.
    zes_freq_throttle_time_t* pThrottleTime         ///< [in,out] Will contain a snapshot of the throttle time counters for the
                                                    ///< specified domain.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get the overclocking capabilities.
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hFrequency`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pOcCapabilities`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesFrequencyOcGetCapabilities(
    zes_freq_handle_t hFrequency,                   ///< [in] Handle for the component.
    zes_oc_capabilities_t* pOcCapabilities          ///< [in,out] Pointer to the capabilities structure
                                                    ///< ::zes_oc_capabilities_t.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get the current overclocking frequency target, if extended moded is
///        supported, will returned in 1 Mhz granularity.
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hFrequency`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pCurrentOcFrequency`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + Overclocking is not supported on this frequency domain (::zes_oc_capabilities_t.isOcSupported)
///         + The specified voltage and/or frequency overclock settings exceed the hardware values (see ::zes_oc_capabilities_t.maxOcFrequency, ::zes_oc_capabilities_t.maxOcVoltage, ::zes_oc_capabilities_t.minOcVoltageOffset, ::zes_oc_capabilities_t.maxOcVoltageOffset).
///         + Requested voltage overclock is very high but ::zes_oc_capabilities_t.isHighVoltModeEnabled is not enabled for the device.
///     - ::ZE_RESULT_ERROR_NOT_AVAILABLE
///         + Overclocking feature is locked on this frequency domain
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to make these modifications.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesFrequencyOcGetFrequencyTarget(
    zes_freq_handle_t hFrequency,                   ///< [in] Handle for the component.
    double* pCurrentOcFrequency                     ///< [out] Overclocking Frequency in MHz, if extended moded is supported,
                                                    ///< will returned in 1 Mhz granularity, else, in multiples of 50 Mhz. This
                                                    ///< cannot be greater than ::zes_oc_capabilities_t.maxOcFrequency.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Set the current overclocking frequency target, if extended moded is
///        supported, can be set in 1 Mhz granularity.
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hFrequency`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + Overclocking is not supported on this frequency domain (::zes_oc_capabilities_t.isOcSupported)
///         + The specified voltage and/or frequency overclock settings exceed the hardware values (see ::zes_oc_capabilities_t.maxOcFrequency, ::zes_oc_capabilities_t.maxOcVoltage, ::zes_oc_capabilities_t.minOcVoltageOffset, ::zes_oc_capabilities_t.maxOcVoltageOffset).
///         + Requested voltage overclock is very high but ::zes_oc_capabilities_t.isHighVoltModeEnabled is not enabled for the device.
///     - ::ZE_RESULT_ERROR_NOT_AVAILABLE
///         + Overclocking feature is locked on this frequency domain
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to make these modifications.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesFrequencyOcSetFrequencyTarget(
    zes_freq_handle_t hFrequency,                   ///< [in] Handle for the component.
    double CurrentOcFrequency                       ///< [in] Overclocking Frequency in MHz, if extended moded is supported, it
                                                    ///< could be set in 1 Mhz granularity, else, in multiples of 50 Mhz. This
                                                    ///< cannot be greater than ::zes_oc_capabilities_t.maxOcFrequency.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get the current overclocking voltage settings.
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hFrequency`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pCurrentVoltageTarget`
///         + `nullptr == pCurrentVoltageOffset`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + Overclocking is not supported on this frequency domain (::zes_oc_capabilities_t.isOcSupported)
///         + The specified voltage and/or frequency overclock settings exceed the hardware values (see ::zes_oc_capabilities_t.maxOcFrequency, ::zes_oc_capabilities_t.maxOcVoltage, ::zes_oc_capabilities_t.minOcVoltageOffset, ::zes_oc_capabilities_t.maxOcVoltageOffset).
///         + Requested voltage overclock is very high but ::zes_oc_capabilities_t.isHighVoltModeEnabled is not enabled for the device.
///     - ::ZE_RESULT_ERROR_NOT_AVAILABLE
///         + Overclocking feature is locked on this frequency domain
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to make these modifications.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesFrequencyOcGetVoltageTarget(
    zes_freq_handle_t hFrequency,                   ///< [in] Handle for the component.
    double* pCurrentVoltageTarget,                  ///< [out] Overclock voltage in Volts. This cannot be greater than
                                                    ///< ::zes_oc_capabilities_t.maxOcVoltage.
    double* pCurrentVoltageOffset                   ///< [out] This voltage offset is applied to all points on the
                                                    ///< voltage/frequency curve, include the new overclock voltageTarget. It
                                                    ///< can be in the range (::zes_oc_capabilities_t.minOcVoltageOffset,
                                                    ///< ::zes_oc_capabilities_t.maxOcVoltageOffset).
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Set the current overclocking voltage settings.
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hFrequency`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + Overclocking is not supported on this frequency domain (::zes_oc_capabilities_t.isOcSupported)
///         + The specified voltage and/or frequency overclock settings exceed the hardware values (see ::zes_oc_capabilities_t.maxOcFrequency, ::zes_oc_capabilities_t.maxOcVoltage, ::zes_oc_capabilities_t.minOcVoltageOffset, ::zes_oc_capabilities_t.maxOcVoltageOffset).
///         + Requested voltage overclock is very high but ::zes_oc_capabilities_t.isHighVoltModeEnabled is not enabled for the device.
///     - ::ZE_RESULT_ERROR_NOT_AVAILABLE
///         + Overclocking feature is locked on this frequency domain
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to make these modifications.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesFrequencyOcSetVoltageTarget(
    zes_freq_handle_t hFrequency,                   ///< [in] Handle for the component.
    double CurrentVoltageTarget,                    ///< [in] Overclock voltage in Volts. This cannot be greater than
                                                    ///< ::zes_oc_capabilities_t.maxOcVoltage.
    double CurrentVoltageOffset                     ///< [in] This voltage offset is applied to all points on the
                                                    ///< voltage/frequency curve, include the new overclock voltageTarget. It
                                                    ///< can be in the range (::zes_oc_capabilities_t.minOcVoltageOffset,
                                                    ///< ::zes_oc_capabilities_t.maxOcVoltageOffset).
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Set the current overclocking mode.
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hFrequency`
///     - ::ZE_RESULT_ERROR_INVALID_ENUMERATION
///         + `::ZES_OC_MODE_FIXED < CurrentOcMode`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + Overclocking is not supported on this frequency domain (::zes_oc_capabilities_t.isOcSupported)
///         + The specified voltage and/or frequency overclock settings exceed the hardware values (see ::zes_oc_capabilities_t.maxOcFrequency, ::zes_oc_capabilities_t.maxOcVoltage, ::zes_oc_capabilities_t.minOcVoltageOffset, ::zes_oc_capabilities_t.maxOcVoltageOffset).
///         + Requested voltage overclock is very high but ::zes_oc_capabilities_t.isHighVoltModeEnabled is not enabled for the device.
///     - ::ZE_RESULT_ERROR_NOT_AVAILABLE
///         + Overclocking feature is locked on this frequency domain
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to make these modifications.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesFrequencyOcSetMode(
    zes_freq_handle_t hFrequency,                   ///< [in] Handle for the component.
    zes_oc_mode_t CurrentOcMode                     ///< [in] Current Overclocking Mode ::zes_oc_mode_t.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get the current overclocking mode.
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hFrequency`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pCurrentOcMode`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + Overclocking is not supported on this frequency domain (::zes_oc_capabilities_t.isOcSupported)
///         + The specified voltage and/or frequency overclock settings exceed the hardware values (see ::zes_oc_capabilities_t.maxOcFrequency, ::zes_oc_capabilities_t.maxOcVoltage, ::zes_oc_capabilities_t.minOcVoltageOffset, ::zes_oc_capabilities_t.maxOcVoltageOffset).
///         + Requested voltage overclock is very high but ::zes_oc_capabilities_t.isHighVoltModeEnabled is not enabled for the device.
///     - ::ZE_RESULT_ERROR_NOT_AVAILABLE
///         + Overclocking feature is locked on this frequency domain
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to make these modifications.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesFrequencyOcGetMode(
    zes_freq_handle_t hFrequency,                   ///< [in] Handle for the component.
    zes_oc_mode_t* pCurrentOcMode                   ///< [out] Current Overclocking Mode ::zes_oc_mode_t.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get the maximum current limit setting.
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hFrequency`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pOcIccMax`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + Overclocking is not supported on this frequency domain (::zes_oc_capabilities_t.isOcSupported)
///         + Capability ::zes_oc_capabilities_t.isIccMaxSupported is false for this frequency domain
ZE_APIEXPORT ze_result_t ZE_APICALL
zesFrequencyOcGetIccMax(
    zes_freq_handle_t hFrequency,                   ///< [in] Handle for the component.
    double* pOcIccMax                               ///< [in,out] Will contain the maximum current limit in Amperes on
                                                    ///< successful return.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Change the maximum current limit setting.
/// 
/// @details
///     - Setting ocIccMax to 0.0 will return the value to the factory default.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hFrequency`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + Overclocking is not supported on this frequency domain (::zes_oc_capabilities_t.isOcSupported)
///         + Capability ::zes_oc_capabilities_t.isIccMaxSupported is false for this frequency domain
///     - ::ZE_RESULT_ERROR_NOT_AVAILABLE
///         + Overclocking feature is locked on this frequency domain
///     - ::ZE_RESULT_ERROR_INVALID_ARGUMENT
///         + The specified current limit is too low or too high
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to make these modifications.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesFrequencyOcSetIccMax(
    zes_freq_handle_t hFrequency,                   ///< [in] Handle for the component.
    double ocIccMax                                 ///< [in] The new maximum current limit in Amperes.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get the maximum temperature limit setting.
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hFrequency`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pOcTjMax`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + Overclocking is not supported on this frequency domain (::zes_oc_capabilities_t.isOcSupported)
ZE_APIEXPORT ze_result_t ZE_APICALL
zesFrequencyOcGetTjMax(
    zes_freq_handle_t hFrequency,                   ///< [in] Handle for the component.
    double* pOcTjMax                                ///< [in,out] Will contain the maximum temperature limit in degrees Celsius
                                                    ///< on successful return.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Change the maximum temperature limit setting.
/// 
/// @details
///     - Setting ocTjMax to 0.0 will return the value to the factory default.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hFrequency`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + Overclocking is not supported on this frequency domain (::zes_oc_capabilities_t.isOcSupported)
///         + Capability ::zes_oc_capabilities_t.isTjMaxSupported is false for this frequency domain
///     - ::ZE_RESULT_ERROR_NOT_AVAILABLE
///         + Overclocking feature is locked on this frequency domain
///     - ::ZE_RESULT_ERROR_INVALID_ARGUMENT
///         + The specified temperature limit is too high
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to make these modifications.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesFrequencyOcSetTjMax(
    zes_freq_handle_t hFrequency,                   ///< [in] Handle for the component.
    double ocTjMax                                  ///< [in] The new maximum temperature limit in degrees Celsius.
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Tool APIs for System Resource Management (Sysman) - Firmware management
#if !defined(__GNUC__)
#pragma region led
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief LED properties
typedef struct _zes_led_properties_t
{
    zes_structure_type_t stype;                     ///< [in] type of this structure
    void* pNext;                                    ///< [in,out][optional] pointer to extension-specific structure
    ze_bool_t onSubdevice;                          ///< [out] True if the resource is located on a sub-device; false means
                                                    ///< that the resource is on the device of the calling Sysman handle
    uint32_t subdeviceId;                           ///< [out] If onSubdevice is true, this gives the ID of the sub-device
    ze_bool_t canControl;                           ///< [out] Indicates if software can control the LED assuming the user has
                                                    ///< permissions
    ze_bool_t haveRGB;                              ///< [out] Indicates if the LED is RGB capable

} zes_led_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief LED color
typedef struct _zes_led_color_t
{
    double red;                                     ///< [in,out][range(0.0, 1.0)] The LED red value. On output, a value less
                                                    ///< than 0.0 indicates that the color is not known.
    double green;                                   ///< [in,out][range(0.0, 1.0)] The LED green value. On output, a value less
                                                    ///< than 0.0 indicates that the color is not known.
    double blue;                                    ///< [in,out][range(0.0, 1.0)] The LED blue value. On output, a value less
                                                    ///< than 0.0 indicates that the color is not known.

} zes_led_color_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief LED state
typedef struct _zes_led_state_t
{
    zes_structure_type_t stype;                     ///< [in] type of this structure
    const void* pNext;                              ///< [in][optional] pointer to extension-specific structure
    ze_bool_t isOn;                                 ///< [out] Indicates if the LED is on or off
    zes_led_color_t color;                          ///< [out] Color of the LED

} zes_led_state_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Get handle of LEDs
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
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
zesDeviceEnumLeds(
    zes_device_handle_t hDevice,                    ///< [in] Sysman handle of the device.
    uint32_t* pCount,                               ///< [in,out] pointer to the number of components of this type.
                                                    ///< if count is zero, then the driver shall update the value with the
                                                    ///< total number of components of this type that are available.
                                                    ///< if count is greater than the number of components of this type that
                                                    ///< are available, then the driver shall update the value with the correct
                                                    ///< number of components.
    zes_led_handle_t* phLed                         ///< [in,out][optional][range(0, *pCount)] array of handle of components of
                                                    ///< this type.
                                                    ///< if count is less than the number of components of this type that are
                                                    ///< available, then the driver shall only retrieve that number of
                                                    ///< component handles.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get LED properties
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hLed`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pProperties`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesLedGetProperties(
    zes_led_handle_t hLed,                          ///< [in] Handle for the component.
    zes_led_properties_t* pProperties               ///< [in,out] Will contain the properties of the LED.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get current state of a LED - on/off, color
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hLed`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pState`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesLedGetState(
    zes_led_handle_t hLed,                          ///< [in] Handle for the component.
    zes_led_state_t* pState                         ///< [in,out] Will contain the current state of the LED.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Turn the LED on/off
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hLed`
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to make these modifications.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesLedSetState(
    zes_led_handle_t hLed,                          ///< [in] Handle for the component.
    ze_bool_t enable                                ///< [in] Set to TRUE to turn the LED on, FALSE to turn off.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Set the color of the LED
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hLed`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pColor`
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to make these modifications.
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + This LED doesn't not support color changes. See ::zes_led_properties_t.haveRGB.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesLedSetColor(
    zes_led_handle_t hLed,                          ///< [in] Handle for the component.
    const zes_led_color_t* pColor                   ///< [in] New color of the LED.
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Tool APIs for System Resource Management (Sysman) - Memory management
#if !defined(__GNUC__)
#pragma region memory
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Memory module types
typedef enum _zes_mem_type_t
{
    ZES_MEM_TYPE_HBM = 0,                           ///< HBM memory
    ZES_MEM_TYPE_DDR = 1,                           ///< DDR memory
    ZES_MEM_TYPE_DDR3 = 2,                          ///< DDR3 memory
    ZES_MEM_TYPE_DDR4 = 3,                          ///< DDR4 memory
    ZES_MEM_TYPE_DDR5 = 4,                          ///< DDR5 memory
    ZES_MEM_TYPE_LPDDR = 5,                         ///< LPDDR memory
    ZES_MEM_TYPE_LPDDR3 = 6,                        ///< LPDDR3 memory
    ZES_MEM_TYPE_LPDDR4 = 7,                        ///< LPDDR4 memory
    ZES_MEM_TYPE_LPDDR5 = 8,                        ///< LPDDR5 memory
    ZES_MEM_TYPE_SRAM = 9,                          ///< SRAM memory
    ZES_MEM_TYPE_L1 = 10,                           ///< L1 cache
    ZES_MEM_TYPE_L3 = 11,                           ///< L3 cache
    ZES_MEM_TYPE_GRF = 12,                          ///< Execution unit register file
    ZES_MEM_TYPE_SLM = 13,                          ///< Execution unit shared local memory
    ZES_MEM_TYPE_GDDR4 = 14,                        ///< GDDR4 memory
    ZES_MEM_TYPE_GDDR5 = 15,                        ///< GDDR5 memory
    ZES_MEM_TYPE_GDDR5X = 16,                       ///< GDDR5X memory
    ZES_MEM_TYPE_GDDR6 = 17,                        ///< GDDR6 memory
    ZES_MEM_TYPE_GDDR6X = 18,                       ///< GDDR6X memory
    ZES_MEM_TYPE_GDDR7 = 19,                        ///< GDDR7 memory
    ZES_MEM_TYPE_FORCE_UINT32 = 0x7fffffff

} zes_mem_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Memory module location
typedef enum _zes_mem_loc_t
{
    ZES_MEM_LOC_SYSTEM = 0,                         ///< System memory
    ZES_MEM_LOC_DEVICE = 1,                         ///< On board local device memory
    ZES_MEM_LOC_FORCE_UINT32 = 0x7fffffff

} zes_mem_loc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Memory health
typedef enum _zes_mem_health_t
{
    ZES_MEM_HEALTH_UNKNOWN = 0,                     ///< The memory health cannot be determined.
    ZES_MEM_HEALTH_OK = 1,                          ///< All memory channels are healthy.
    ZES_MEM_HEALTH_DEGRADED = 2,                    ///< Excessive correctable errors have been detected on one or more
                                                    ///< channels. Device should be reset.
    ZES_MEM_HEALTH_CRITICAL = 3,                    ///< Operating with reduced memory to cover banks with too many
                                                    ///< uncorrectable errors.
    ZES_MEM_HEALTH_REPLACE = 4,                     ///< Device should be replaced due to excessive uncorrectable errors.
    ZES_MEM_HEALTH_FORCE_UINT32 = 0x7fffffff

} zes_mem_health_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Memory properties
typedef struct _zes_mem_properties_t
{
    zes_structure_type_t stype;                     ///< [in] type of this structure
    void* pNext;                                    ///< [in,out][optional] pointer to extension-specific structure
    zes_mem_type_t type;                            ///< [out] The memory type
    ze_bool_t onSubdevice;                          ///< [out] True if this resource is located on a sub-device; false means
                                                    ///< that the resource is on the device of the calling Sysman handle
    uint32_t subdeviceId;                           ///< [out] If onSubdevice is true, this gives the ID of the sub-device
    zes_mem_loc_t location;                         ///< [out] Location of this memory (system, device)
    uint64_t physicalSize;                          ///< [out] Physical memory size in bytes. A value of 0 indicates that this
                                                    ///< property is not known. However, a call to ::zesMemoryGetState() will
                                                    ///< correctly return the total size of usable memory.
    int32_t busWidth;                               ///< [out] Width of the memory bus. A value of -1 means that this property
                                                    ///< is unknown.
    int32_t numChannels;                            ///< [out] The number of memory channels. A value of -1 means that this
                                                    ///< property is unknown.

} zes_mem_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Memory state - health, allocated
/// 
/// @details
///     - Percent allocation is given by 100 * (size - free / size.
///     - Percent free is given by 100 * free / size.
typedef struct _zes_mem_state_t
{
    zes_structure_type_t stype;                     ///< [in] type of this structure
    const void* pNext;                              ///< [in][optional] pointer to extension-specific structure
    zes_mem_health_t health;                        ///< [out] Indicates the health of the memory
    uint64_t free;                                  ///< [out] The free memory in bytes
    uint64_t size;                                  ///< [out] The total allocatable memory in bytes (can be less than
                                                    ///< ::zes_mem_properties_t.physicalSize)

} zes_mem_state_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Memory bandwidth
/// 
/// @details
///     - Percent bandwidth is calculated by taking two snapshots (s1, s2) and
///       using the equation: %bw = 10^6 * ((s2.readCounter - s1.readCounter) +
///       (s2.writeCounter - s1.writeCounter)) / (s2.maxBandwidth *
///       (s2.timestamp - s1.timestamp))
typedef struct _zes_mem_bandwidth_t
{
    uint64_t readCounter;                           ///< [out] Total bytes read from memory
    uint64_t writeCounter;                          ///< [out] Total bytes written to memory
    uint64_t maxBandwidth;                          ///< [out] Current maximum bandwidth in units of bytes/sec
    uint64_t timestamp;                             ///< [out] The timestamp when these measurements were sampled.
                                                    ///< This timestamp should only be used to calculate delta time between
                                                    ///< snapshots of this structure.
                                                    ///< Never take the delta of this timestamp with the timestamp from a
                                                    ///< different structure since they are not guaranteed to have the same base.
                                                    ///< The absolute value of the timestamp is only valid during within the
                                                    ///< application and may be different on the next execution.

} zes_mem_bandwidth_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Get handle of memory modules
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
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
zesDeviceEnumMemoryModules(
    zes_device_handle_t hDevice,                    ///< [in] Sysman handle of the device.
    uint32_t* pCount,                               ///< [in,out] pointer to the number of components of this type.
                                                    ///< if count is zero, then the driver shall update the value with the
                                                    ///< total number of components of this type that are available.
                                                    ///< if count is greater than the number of components of this type that
                                                    ///< are available, then the driver shall update the value with the correct
                                                    ///< number of components.
    zes_mem_handle_t* phMemory                      ///< [in,out][optional][range(0, *pCount)] array of handle of components of
                                                    ///< this type.
                                                    ///< if count is less than the number of components of this type that are
                                                    ///< available, then the driver shall only retrieve that number of
                                                    ///< component handles.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get memory properties
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hMemory`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pProperties`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesMemoryGetProperties(
    zes_mem_handle_t hMemory,                       ///< [in] Handle for the component.
    zes_mem_properties_t* pProperties               ///< [in,out] Will contain memory properties.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get memory state - health, allocated
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hMemory`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pState`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesMemoryGetState(
    zes_mem_handle_t hMemory,                       ///< [in] Handle for the component.
    zes_mem_state_t* pState                         ///< [in,out] Will contain the current health and allocated memory.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get memory bandwidth
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hMemory`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pBandwidth`
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to query this telemetry.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesMemoryGetBandwidth(
    zes_mem_handle_t hMemory,                       ///< [in] Handle for the component.
    zes_mem_bandwidth_t* pBandwidth                 ///< [in,out] Will contain the current health, free memory, total memory
                                                    ///< size.
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Tool APIs for System Resource Management (Sysman) - Performance factor
#if !defined(__GNUC__)
#pragma region performance
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Static information about a Performance Factor domain
typedef struct _zes_perf_properties_t
{
    zes_structure_type_t stype;                     ///< [in] type of this structure
    void* pNext;                                    ///< [in,out][optional] pointer to extension-specific structure
    ze_bool_t onSubdevice;                          ///< [out] True if this Performance Factor affects accelerators located on
                                                    ///< a sub-device
    uint32_t subdeviceId;                           ///< [out] If onSubdevice is true, this gives the ID of the sub-device
    zes_engine_type_flags_t engines;                ///< [out] Bitfield of accelerator engine types that are affected by this
                                                    ///< Performance Factor.

} zes_perf_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Get handles to accelerator domains whose performance can be optimized
///        via a Performance Factor
/// 
/// @details
///     - A Performance Factor should be tuned for each workload.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
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
zesDeviceEnumPerformanceFactorDomains(
    zes_device_handle_t hDevice,                    ///< [in] Sysman handle of the device.
    uint32_t* pCount,                               ///< [in,out] pointer to the number of components of this type.
                                                    ///< if count is zero, then the driver shall update the value with the
                                                    ///< total number of components of this type that are available.
                                                    ///< if count is greater than the number of components of this type that
                                                    ///< are available, then the driver shall update the value with the correct
                                                    ///< number of components.
    zes_perf_handle_t* phPerf                       ///< [in,out][optional][range(0, *pCount)] array of handle of components of
                                                    ///< this type.
                                                    ///< if count is less than the number of components of this type that are
                                                    ///< available, then the driver shall only retrieve that number of
                                                    ///< component handles.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get properties about a Performance Factor domain
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hPerf`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pProperties`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesPerformanceFactorGetProperties(
    zes_perf_handle_t hPerf,                        ///< [in] Handle for the Performance Factor domain.
    zes_perf_properties_t* pProperties              ///< [in,out] Will contain information about the specified Performance
                                                    ///< Factor domain.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get current Performance Factor for a given domain
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hPerf`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pFactor`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesPerformanceFactorGetConfig(
    zes_perf_handle_t hPerf,                        ///< [in] Handle for the Performance Factor domain.
    double* pFactor                                 ///< [in,out] Will contain the actual Performance Factor being used by the
                                                    ///< hardware (may not be the same as the requested Performance Factor).
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Change the performance factor for a domain
/// 
/// @details
///     - The Performance Factor is a number between 0 and 100.
///     - A Performance Factor is a hint to the hardware. Depending on the
///       hardware, the request may not be granted. Follow up this function with
///       a call to ::zesPerformanceFactorGetConfig() to determine the actual
///       factor being used by the hardware.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hPerf`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesPerformanceFactorSetConfig(
    zes_perf_handle_t hPerf,                        ///< [in] Handle for the Performance Factor domain.
    double factor                                   ///< [in] The new Performance Factor.
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Tool APIs for System Resource Management (Sysman) - Scheduler management
#if !defined(__GNUC__)
#pragma region power
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Properties related to device power settings
typedef struct _zes_power_properties_t
{
    zes_structure_type_t stype;                     ///< [in] type of this structure
    void* pNext;                                    ///< [in,out][optional] pointer to extension-specific structure
    ze_bool_t onSubdevice;                          ///< [out] True if this resource is located on a sub-device; false means
                                                    ///< that the resource is on the device of the calling Sysman handle
    uint32_t subdeviceId;                           ///< [out] If onSubdevice is true, this gives the ID of the sub-device
    ze_bool_t canControl;                           ///< [out] Software can change the power limits of this domain assuming the
                                                    ///< user has permissions.
    ze_bool_t isEnergyThresholdSupported;           ///< [out] Indicates if this power domain supports the energy threshold
                                                    ///< event (::ZES_EVENT_TYPE_FLAG_ENERGY_THRESHOLD_CROSSED).
    int32_t defaultLimit;                           ///< [out] The factory default TDP power limit of the part in milliwatts. A
                                                    ///< value of -1 means that this is not known.
    int32_t minLimit;                               ///< [out] The minimum power limit in milliwatts that can be requested.
    int32_t maxLimit;                               ///< [out] The maximum power limit in milliwatts that can be requested.

} zes_power_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Energy counter snapshot
/// 
/// @details
///     - Average power is calculated by taking two snapshots (s1, s2) and using
///       the equation: PowerWatts = (s2.energy - s1.energy) / (s2.timestamp -
///       s1.timestamp)
typedef struct _zes_power_energy_counter_t
{
    uint64_t energy;                                ///< [out] The monotonic energy counter in microjoules.
    uint64_t timestamp;                             ///< [out] Microsecond timestamp when energy was captured.
                                                    ///< This timestamp should only be used to calculate delta time between
                                                    ///< snapshots of this structure.
                                                    ///< Never take the delta of this timestamp with the timestamp from a
                                                    ///< different structure since they are not guaranteed to have the same base.
                                                    ///< The absolute value of the timestamp is only valid during within the
                                                    ///< application and may be different on the next execution.

} zes_power_energy_counter_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Sustained power limits
/// 
/// @details
///     - The power controller (Punit) will throttle the operating frequency if
///       the power averaged over a window (typically seconds) exceeds this
///       limit.
typedef struct _zes_power_sustained_limit_t
{
    ze_bool_t enabled;                              ///< [in,out] indicates if the limit is enabled (true) or ignored (false)
    int32_t power;                                  ///< [in,out] power limit in milliwatts
    int32_t interval;                               ///< [in,out] power averaging window (Tau) in milliseconds

} zes_power_sustained_limit_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Burst power limit
/// 
/// @details
///     - The power controller (Punit) will throttle the operating frequency of
///       the device if the power averaged over a few milliseconds exceeds a
///       limit known as PL2. Typically PL2 > PL1 so that it permits the
///       frequency to burst higher for short periods than would be otherwise
///       permitted by PL1.
typedef struct _zes_power_burst_limit_t
{
    ze_bool_t enabled;                              ///< [in,out] indicates if the limit is enabled (true) or ignored (false)
    int32_t power;                                  ///< [in,out] power limit in milliwatts

} zes_power_burst_limit_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Peak power limit
/// 
/// @details
///     - The power controller (Punit) will reactively/proactively throttle the
///       operating frequency of the device when the instantaneous/100usec power
///       exceeds this limit. The limit is known as PL4 or Psys. It expresses
///       the maximum power that can be drawn from the power supply.
///     - If this power limit is removed or set too high, the power supply will
///       generate an interrupt when it detects an overcurrent condition and the
///       power controller will throttle the device frequencies down to min. It
///       is thus better to tune the PL4 value in order to avoid such
///       excursions.
typedef struct _zes_power_peak_limit_t
{
    int32_t powerAC;                                ///< [in,out] power limit in milliwatts for the AC power source.
    int32_t powerDC;                                ///< [in,out] power limit in milliwatts for the DC power source. On input,
                                                    ///< this is ignored if the product does not have a battery. On output,
                                                    ///< this will be -1 if the product does not have a battery.

} zes_power_peak_limit_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Energy threshold
/// 
/// @details
///     - .
typedef struct _zes_energy_threshold_t
{
    ze_bool_t enable;                               ///< [in,out] Indicates if the energy threshold is enabled.
    double threshold;                               ///< [in,out] The energy threshold in Joules. Will be 0.0 if no threshold
                                                    ///< has been set.
    uint32_t processId;                             ///< [in,out] The host process ID that set the energy threshold. Will be
                                                    ///< 0xFFFFFFFF if no threshold has been set.

} zes_energy_threshold_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Get handle of power domains
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
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
zesDeviceEnumPowerDomains(
    zes_device_handle_t hDevice,                    ///< [in] Sysman handle of the device.
    uint32_t* pCount,                               ///< [in,out] pointer to the number of components of this type.
                                                    ///< if count is zero, then the driver shall update the value with the
                                                    ///< total number of components of this type that are available.
                                                    ///< if count is greater than the number of components of this type that
                                                    ///< are available, then the driver shall update the value with the correct
                                                    ///< number of components.
    zes_pwr_handle_t* phPower                       ///< [in,out][optional][range(0, *pCount)] array of handle of components of
                                                    ///< this type.
                                                    ///< if count is less than the number of components of this type that are
                                                    ///< available, then the driver shall only retrieve that number of
                                                    ///< component handles.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get handle of the PCIe card-level power
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == phPower`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + The device does not provide access to card level power controls or telemetry. An invalid power domain handle will be returned in phPower.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesDeviceGetCardPowerDomain(
    zes_device_handle_t hDevice,                    ///< [in] Sysman handle of the device.
    zes_pwr_handle_t* phPower                       ///< [in,out] power domain handle for the entire PCIe card.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get properties related to a power domain
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hPower`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pProperties`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesPowerGetProperties(
    zes_pwr_handle_t hPower,                        ///< [in] Handle for the component.
    zes_power_properties_t* pProperties             ///< [in,out] Structure that will contain property data.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get energy counter
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hPower`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pEnergy`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesPowerGetEnergyCounter(
    zes_pwr_handle_t hPower,                        ///< [in] Handle for the component.
    zes_power_energy_counter_t* pEnergy             ///< [in,out] Will contain the latest snapshot of the energy counter and
                                                    ///< timestamp when the last counter value was measured.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get power limits
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hPower`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesPowerGetLimits(
    zes_pwr_handle_t hPower,                        ///< [in] Handle for the component.
    zes_power_sustained_limit_t* pSustained,        ///< [in,out][optional] The sustained power limit. If this is null, the
                                                    ///< current sustained power limits will not be returned.
    zes_power_burst_limit_t* pBurst,                ///< [in,out][optional] The burst power limit. If this is null, the current
                                                    ///< peak power limits will not be returned.
    zes_power_peak_limit_t* pPeak                   ///< [in,out][optional] The peak power limit. If this is null, the peak
                                                    ///< power limits will not be returned.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Set power limits
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hPower`
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to make these modifications.
///     - ::ZE_RESULT_ERROR_NOT_AVAILABLE
///         + The device is in use, meaning that the GPU is under Over clocking, applying power limits under overclocking is not supported.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesPowerSetLimits(
    zes_pwr_handle_t hPower,                        ///< [in] Handle for the component.
    const zes_power_sustained_limit_t* pSustained,  ///< [in][optional] The sustained power limit. If this is null, no changes
                                                    ///< will be made to the sustained power limits.
    const zes_power_burst_limit_t* pBurst,          ///< [in][optional] The burst power limit. If this is null, no changes will
                                                    ///< be made to the burst power limits.
    const zes_power_peak_limit_t* pPeak             ///< [in][optional] The peak power limit. If this is null, no changes will
                                                    ///< be made to the peak power limits.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get energy threshold
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hPower`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pThreshold`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + Energy threshold not supported on this power domain (check ::zes_power_properties_t.isEnergyThresholdSupported).
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to request this feature.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesPowerGetEnergyThreshold(
    zes_pwr_handle_t hPower,                        ///< [in] Handle for the component.
    zes_energy_threshold_t* pThreshold              ///< [in,out] Returns information about the energy threshold setting -
                                                    ///< enabled/energy threshold/process ID.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Set energy threshold
/// 
/// @details
///     - An event ::ZES_EVENT_TYPE_FLAG_ENERGY_THRESHOLD_CROSSED will be
///       generated when the delta energy consumed starting from this call
///       exceeds the specified threshold. Use the function
///       ::zesDeviceEventRegister() to start receiving the event.
///     - Only one running process can control the energy threshold at a given
///       time. If another process attempts to change the energy threshold, the
///       error ::ZE_RESULT_ERROR_NOT_AVAILABLE will be returned. The function
///       ::zesPowerGetEnergyThreshold() to determine the process ID currently
///       controlling this setting.
///     - Calling this function will remove any pending energy thresholds and
///       start counting from the time of this call.
///     - Once the energy threshold has been reached and the event generated,
///       the threshold is automatically removed. It is up to the application to
///       request a new threshold.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hPower`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + Energy threshold not supported on this power domain (check ::zes_power_properties_t.isEnergyThresholdSupported).
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to request this feature.
///     - ::ZE_RESULT_ERROR_NOT_AVAILABLE
///         + Another running process has set the energy threshold.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesPowerSetEnergyThreshold(
    zes_pwr_handle_t hPower,                        ///< [in] Handle for the component.
    double threshold                                ///< [in] The energy threshold to be set in joules.
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Tool APIs for System Resource Management (Sysman) - Firmware management
#if !defined(__GNUC__)
#pragma region psu
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief PSU voltage status
typedef enum _zes_psu_voltage_status_t
{
    ZES_PSU_VOLTAGE_STATUS_UNKNOWN = 0,             ///< The status of the power supply voltage controllers cannot be
                                                    ///< determined
    ZES_PSU_VOLTAGE_STATUS_NORMAL = 1,              ///< No unusual voltages have been detected
    ZES_PSU_VOLTAGE_STATUS_OVER = 2,                ///< Over-voltage has occurred
    ZES_PSU_VOLTAGE_STATUS_UNDER = 3,               ///< Under-voltage has occurred
    ZES_PSU_VOLTAGE_STATUS_FORCE_UINT32 = 0x7fffffff

} zes_psu_voltage_status_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Static properties of the power supply
typedef struct _zes_psu_properties_t
{
    zes_structure_type_t stype;                     ///< [in] type of this structure
    void* pNext;                                    ///< [in,out][optional] pointer to extension-specific structure
    ze_bool_t onSubdevice;                          ///< [out] True if the resource is located on a sub-device; false means
                                                    ///< that the resource is on the device of the calling Sysman handle
    uint32_t subdeviceId;                           ///< [out] If onSubdevice is true, this gives the ID of the sub-device
    ze_bool_t haveFan;                              ///< [out] True if the power supply has a fan
    int32_t ampLimit;                               ///< [out] The maximum electrical current in milliamperes that can be
                                                    ///< drawn. A value of -1 indicates that this property cannot be
                                                    ///< determined.

} zes_psu_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Dynamic state of the power supply
typedef struct _zes_psu_state_t
{
    zes_structure_type_t stype;                     ///< [in] type of this structure
    const void* pNext;                              ///< [in][optional] pointer to extension-specific structure
    zes_psu_voltage_status_t voltStatus;            ///< [out] The current PSU voltage status
    ze_bool_t fanFailed;                            ///< [out] Indicates if the fan has failed
    int32_t temperature;                            ///< [out] Read the current heatsink temperature in degrees Celsius. A
                                                    ///< value of -1 indicates that this property cannot be determined.
    int32_t current;                                ///< [out] The amps being drawn in milliamperes. A value of -1 indicates
                                                    ///< that this property cannot be determined.

} zes_psu_state_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Get handle of power supplies
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
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
zesDeviceEnumPsus(
    zes_device_handle_t hDevice,                    ///< [in] Sysman handle of the device.
    uint32_t* pCount,                               ///< [in,out] pointer to the number of components of this type.
                                                    ///< if count is zero, then the driver shall update the value with the
                                                    ///< total number of components of this type that are available.
                                                    ///< if count is greater than the number of components of this type that
                                                    ///< are available, then the driver shall update the value with the correct
                                                    ///< number of components.
    zes_psu_handle_t* phPsu                         ///< [in,out][optional][range(0, *pCount)] array of handle of components of
                                                    ///< this type.
                                                    ///< if count is less than the number of components of this type that are
                                                    ///< available, then the driver shall only retrieve that number of
                                                    ///< component handles.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get power supply properties
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hPsu`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pProperties`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesPsuGetProperties(
    zes_psu_handle_t hPsu,                          ///< [in] Handle for the component.
    zes_psu_properties_t* pProperties               ///< [in,out] Will contain the properties of the power supply.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get current power supply state
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hPsu`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pState`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesPsuGetState(
    zes_psu_handle_t hPsu,                          ///< [in] Handle for the component.
    zes_psu_state_t* pState                         ///< [in,out] Will contain the current state of the power supply.
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Tool APIs for System Resource Management (Sysman) - Firmware management
#if !defined(__GNUC__)
#pragma region ras
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief RAS error type
typedef enum _zes_ras_error_type_t
{
    ZES_RAS_ERROR_TYPE_CORRECTABLE = 0,             ///< Errors were corrected by hardware
    ZES_RAS_ERROR_TYPE_UNCORRECTABLE = 1,           ///< Error were not corrected
    ZES_RAS_ERROR_TYPE_FORCE_UINT32 = 0x7fffffff

} zes_ras_error_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief RAS error categories
typedef enum _zes_ras_error_cat_t
{
    ZES_RAS_ERROR_CAT_RESET = 0,                    ///< The number of accelerator engine resets attempted by the driver
    ZES_RAS_ERROR_CAT_PROGRAMMING_ERRORS = 1,       ///< The number of hardware exceptions generated by the way workloads have
                                                    ///< programmed the hardware
    ZES_RAS_ERROR_CAT_DRIVER_ERRORS = 2,            ///< The number of low level driver communication errors have occurred
    ZES_RAS_ERROR_CAT_COMPUTE_ERRORS = 3,           ///< The number of errors that have occurred in the compute accelerator
                                                    ///< hardware
    ZES_RAS_ERROR_CAT_NON_COMPUTE_ERRORS = 4,       ///< The number of errors that have occurred in the fixed-function
                                                    ///< accelerator hardware
    ZES_RAS_ERROR_CAT_CACHE_ERRORS = 5,             ///< The number of errors that have occurred in caches (L1/L3/register
                                                    ///< file/shared local memory/sampler)
    ZES_RAS_ERROR_CAT_DISPLAY_ERRORS = 6,           ///< The number of errors that have occurred in the display
    ZES_RAS_ERROR_CAT_FORCE_UINT32 = 0x7fffffff

} zes_ras_error_cat_t;

///////////////////////////////////////////////////////////////////////////////
#ifndef ZES_MAX_RAS_ERROR_CATEGORY_COUNT
/// @brief The maximum number of categories
#define ZES_MAX_RAS_ERROR_CATEGORY_COUNT  7
#endif // ZES_MAX_RAS_ERROR_CATEGORY_COUNT

///////////////////////////////////////////////////////////////////////////////
/// @brief RAS properties
typedef struct _zes_ras_properties_t
{
    zes_structure_type_t stype;                     ///< [in] type of this structure
    void* pNext;                                    ///< [in,out][optional] pointer to extension-specific structure
    zes_ras_error_type_t type;                      ///< [out] The type of RAS error
    ze_bool_t onSubdevice;                          ///< [out] True if the resource is located on a sub-device; false means
                                                    ///< that the resource is on the device of the calling Sysman handle
    uint32_t subdeviceId;                           ///< [out] If onSubdevice is true, this gives the ID of the sub-device

} zes_ras_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief RAS error details
typedef struct _zes_ras_state_t
{
    zes_structure_type_t stype;                     ///< [in] type of this structure
    const void* pNext;                              ///< [in][optional] pointer to extension-specific structure
    uint64_t category[ZES_MAX_RAS_ERROR_CATEGORY_COUNT];///< [in][out] Breakdown of error by category

} zes_ras_state_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief RAS error configuration - thresholds used for triggering RAS events
///        (::ZES_EVENT_TYPE_FLAG_RAS_CORRECTABLE_ERRORS,
///        ::ZES_EVENT_TYPE_FLAG_RAS_UNCORRECTABLE_ERRORS)
/// 
/// @details
///     - The driver maintains a total counter which is updated every time a
///       hardware block covered by the corresponding RAS error set notifies
///       that an error has occurred. When this total count goes above the
///       totalThreshold specified below, a RAS event is triggered.
///     - The driver also maintains a counter for each category of RAS error
///       (see ::zes_ras_state_t for a breakdown). Each time a hardware block of
///       that category notifies that an error has occurred, that corresponding
///       category counter is updated. When it goes above the threshold
///       specified in detailedThresholds, a RAS event is triggered.
typedef struct _zes_ras_config_t
{
    zes_structure_type_t stype;                     ///< [in] type of this structure
    const void* pNext;                              ///< [in][optional] pointer to extension-specific structure
    uint64_t totalThreshold;                        ///< [in,out] If the total RAS errors exceeds this threshold, the event
                                                    ///< will be triggered. A value of 0ULL disables triggering the event based
                                                    ///< on the total counter.
    zes_ras_state_t detailedThresholds;             ///< [in,out] If the RAS errors for each category exceed the threshold for
                                                    ///< that category, the event will be triggered. A value of 0ULL will
                                                    ///< disable an event being triggered for that category.

} zes_ras_config_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Get handle of all RAS error sets on a device
/// 
/// @details
///     - A RAS error set is a collection of RAS error counters of a given type
///       (correctable/uncorrectable) from hardware blocks contained within a
///       sub-device or within the device.
///     - A device without sub-devices will typically return two handles, one
///       for correctable errors sets and one for uncorrectable error sets.
///     - A device with sub-devices will return RAS error sets for each
///       sub-device and possibly RAS error sets for hardware blocks outside the
///       sub-devices.
///     - If the function completes successfully but pCount is set to 0, RAS
///       features are not available/enabled on this device.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
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
zesDeviceEnumRasErrorSets(
    zes_device_handle_t hDevice,                    ///< [in] Sysman handle of the device.
    uint32_t* pCount,                               ///< [in,out] pointer to the number of components of this type.
                                                    ///< if count is zero, then the driver shall update the value with the
                                                    ///< total number of components of this type that are available.
                                                    ///< if count is greater than the number of components of this type that
                                                    ///< are available, then the driver shall update the value with the correct
                                                    ///< number of components.
    zes_ras_handle_t* phRas                         ///< [in,out][optional][range(0, *pCount)] array of handle of components of
                                                    ///< this type.
                                                    ///< if count is less than the number of components of this type that are
                                                    ///< available, then the driver shall only retrieve that number of
                                                    ///< component handles.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get RAS properties of a given RAS error set - this enables discovery
///        of the type of RAS error set (correctable/uncorrectable) and if
///        located on a sub-device
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hRas`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pProperties`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesRasGetProperties(
    zes_ras_handle_t hRas,                          ///< [in] Handle for the component.
    zes_ras_properties_t* pProperties               ///< [in,out] Structure describing RAS properties
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get RAS error thresholds that control when RAS events are generated
/// 
/// @details
///     - The driver maintains counters for all RAS error sets and error
///       categories. Events are generated when errors occur. The configuration
///       enables setting thresholds to limit when events are sent.
///     - When a particular RAS correctable error counter exceeds the configured
///       threshold, the event ::ZES_EVENT_TYPE_FLAG_RAS_CORRECTABLE_ERRORS will
///       be triggered.
///     - When a particular RAS uncorrectable error counter exceeds the
///       configured threshold, the event
///       ::ZES_EVENT_TYPE_FLAG_RAS_UNCORRECTABLE_ERRORS will be triggered.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hRas`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pConfig`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesRasGetConfig(
    zes_ras_handle_t hRas,                          ///< [in] Handle for the component.
    zes_ras_config_t* pConfig                       ///< [in,out] Will be populed with the current RAS configuration -
                                                    ///< thresholds used to trigger events
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Set RAS error thresholds that control when RAS events are generated
/// 
/// @details
///     - The driver maintains counters for all RAS error sets and error
///       categories. Events are generated when errors occur. The configuration
///       enables setting thresholds to limit when events are sent.
///     - When a particular RAS correctable error counter exceeds the specified
///       threshold, the event ::ZES_EVENT_TYPE_FLAG_RAS_CORRECTABLE_ERRORS will
///       be generated.
///     - When a particular RAS uncorrectable error counter exceeds the
///       specified threshold, the event
///       ::ZES_EVENT_TYPE_FLAG_RAS_UNCORRECTABLE_ERRORS will be generated.
///     - Call ::zesRasGetState() and set the clear flag to true to restart
///       event generation once counters have exceeded thresholds.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hRas`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pConfig`
///     - ::ZE_RESULT_ERROR_NOT_AVAILABLE
///         + Another running process is controlling these settings.
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + Don't have permissions to set thresholds.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesRasSetConfig(
    zes_ras_handle_t hRas,                          ///< [in] Handle for the component.
    const zes_ras_config_t* pConfig                 ///< [in] Change the RAS configuration - thresholds used to trigger events
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get the current value of RAS error counters for a particular error set
/// 
/// @details
///     - Clearing errors will affect other threads/applications - the counter
///       values will start from zero.
///     - Clearing errors requires write permissions.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hRas`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pState`
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + Don't have permissions to clear error counters.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesRasGetState(
    zes_ras_handle_t hRas,                          ///< [in] Handle for the component.
    ze_bool_t clear,                                ///< [in] Set to 1 to clear the counters of this type
    zes_ras_state_t* pState                         ///< [in,out] Breakdown of where errors have occurred
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Tool APIs for System Resource Management (Sysman) - Scheduler management
#if !defined(__GNUC__)
#pragma region scheduler
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Scheduler mode
typedef enum _zes_sched_mode_t
{
    ZES_SCHED_MODE_TIMEOUT = 0,                     ///< Multiple applications or contexts are submitting work to the hardware.
                                                    ///< When higher priority work arrives, the scheduler attempts to pause the
                                                    ///< current executing work within some timeout interval, then submits the
                                                    ///< other work.
    ZES_SCHED_MODE_TIMESLICE = 1,                   ///< The scheduler attempts to fairly timeslice hardware execution time
                                                    ///< between multiple contexts submitting work to the hardware
                                                    ///< concurrently.
    ZES_SCHED_MODE_EXCLUSIVE = 2,                   ///< Any application or context can run indefinitely on the hardware
                                                    ///< without being preempted or terminated. All pending work for other
                                                    ///< contexts must wait until the running context completes with no further
                                                    ///< submitted work.
    ZES_SCHED_MODE_COMPUTE_UNIT_DEBUG = 3,          ///< This is a special mode that must ben enabled when debugging an
                                                    ///< application that uses this device e.g. using the Level0 Debug API. It
                                                    ///< has the effect of disabling any timeouts on workload execution time
                                                    ///< and will change workload scheduling to ensure debug accuracy.
    ZES_SCHED_MODE_FORCE_UINT32 = 0x7fffffff

} zes_sched_mode_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Properties related to scheduler component
typedef struct _zes_sched_properties_t
{
    zes_structure_type_t stype;                     ///< [in] type of this structure
    void* pNext;                                    ///< [in,out][optional] pointer to extension-specific structure
    ze_bool_t onSubdevice;                          ///< [out] True if this resource is located on a sub-device; false means
                                                    ///< that the resource is on the device of the calling Sysman handle
    uint32_t subdeviceId;                           ///< [out] If onSubdevice is true, this gives the ID of the sub-device
    ze_bool_t canControl;                           ///< [out] Software can change the scheduler component configuration
                                                    ///< assuming the user has permissions.
    zes_engine_type_flags_t engines;                ///< [out] Bitfield of accelerator engine types that are managed by this
                                                    ///< scheduler component. Note that there can be more than one scheduler
                                                    ///< component for the same type of accelerator engine.
    uint32_t supportedModes;                        ///< [out] Bitfield of scheduler modes that can be configured for this
                                                    ///< scheduler component (bitfield of 1<<::zes_sched_mode_t).

} zes_sched_properties_t;

///////////////////////////////////////////////////////////////////////////////
#ifndef ZES_SCHED_WATCHDOG_DISABLE
/// @brief Disable forward progress guard timeout.
#define ZES_SCHED_WATCHDOG_DISABLE  (~(0ULL))
#endif // ZES_SCHED_WATCHDOG_DISABLE

///////////////////////////////////////////////////////////////////////////////
/// @brief Configuration for timeout scheduler mode (::ZES_SCHED_MODE_TIMEOUT)
typedef struct _zes_sched_timeout_properties_t
{
    zes_structure_type_t stype;                     ///< [in] type of this structure
    void* pNext;                                    ///< [in,out][optional] pointer to extension-specific structure
    uint64_t watchdogTimeout;                       ///< [in,out] The maximum time in microseconds that the scheduler will wait
                                                    ///< for a batch of work submitted to a hardware engine to complete or to
                                                    ///< be preempted so as to run another context.
                                                    ///< If this time is exceeded, the hardware engine is reset and the context terminated.
                                                    ///< If set to ::ZES_SCHED_WATCHDOG_DISABLE, a running workload can run as
                                                    ///< long as it wants without being terminated, but preemption attempts to
                                                    ///< run other contexts are permitted but not enforced.

} zes_sched_timeout_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Configuration for timeslice scheduler mode
///        (::ZES_SCHED_MODE_TIMESLICE)
typedef struct _zes_sched_timeslice_properties_t
{
    zes_structure_type_t stype;                     ///< [in] type of this structure
    void* pNext;                                    ///< [in,out][optional] pointer to extension-specific structure
    uint64_t interval;                              ///< [in,out] The average interval in microseconds that a submission for a
                                                    ///< context will run on a hardware engine before being preempted out to
                                                    ///< run a pending submission for another context.
    uint64_t yieldTimeout;                          ///< [in,out] The maximum time in microseconds that the scheduler will wait
                                                    ///< to preempt a workload running on an engine before deciding to reset
                                                    ///< the hardware engine and terminating the associated context.

} zes_sched_timeslice_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Returns handles to scheduler components.
/// 
/// @details
///     - Each scheduler component manages the distribution of work across one
///       or more accelerator engines.
///     - If an application wishes to change the scheduler behavior for all
///       accelerator engines of a specific type (e.g. compute), it should
///       select all the handles where the structure member
///       ::zes_sched_properties_t.engines contains that type.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
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
zesDeviceEnumSchedulers(
    zes_device_handle_t hDevice,                    ///< [in] Sysman handle of the device.
    uint32_t* pCount,                               ///< [in,out] pointer to the number of components of this type.
                                                    ///< if count is zero, then the driver shall update the value with the
                                                    ///< total number of components of this type that are available.
                                                    ///< if count is greater than the number of components of this type that
                                                    ///< are available, then the driver shall update the value with the correct
                                                    ///< number of components.
    zes_sched_handle_t* phScheduler                 ///< [in,out][optional][range(0, *pCount)] array of handle of components of
                                                    ///< this type.
                                                    ///< if count is less than the number of components of this type that are
                                                    ///< available, then the driver shall only retrieve that number of
                                                    ///< component handles.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get properties related to a scheduler component
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hScheduler`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pProperties`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesSchedulerGetProperties(
    zes_sched_handle_t hScheduler,                  ///< [in] Handle for the component.
    zes_sched_properties_t* pProperties             ///< [in,out] Structure that will contain property data.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get current scheduling mode in effect on a scheduler component.
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hScheduler`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pMode`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + This scheduler component does not support scheduler modes.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesSchedulerGetCurrentMode(
    zes_sched_handle_t hScheduler,                  ///< [in] Sysman handle for the component.
    zes_sched_mode_t* pMode                         ///< [in,out] Will contain the current scheduler mode.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get scheduler config for mode ::ZES_SCHED_MODE_TIMEOUT
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hScheduler`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pConfig`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + This scheduler component does not support scheduler modes.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesSchedulerGetTimeoutModeProperties(
    zes_sched_handle_t hScheduler,                  ///< [in] Sysman handle for the component.
    ze_bool_t getDefaults,                          ///< [in] If TRUE, the driver will return the system default properties for
                                                    ///< this mode, otherwise it will return the current properties.
    zes_sched_timeout_properties_t* pConfig         ///< [in,out] Will contain the current parameters for this mode.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get scheduler config for mode ::ZES_SCHED_MODE_TIMESLICE
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hScheduler`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pConfig`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + This scheduler component does not support scheduler modes.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesSchedulerGetTimesliceModeProperties(
    zes_sched_handle_t hScheduler,                  ///< [in] Sysman handle for the component.
    ze_bool_t getDefaults,                          ///< [in] If TRUE, the driver will return the system default properties for
                                                    ///< this mode, otherwise it will return the current properties.
    zes_sched_timeslice_properties_t* pConfig       ///< [in,out] Will contain the current parameters for this mode.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Change scheduler mode to ::ZES_SCHED_MODE_TIMEOUT or update scheduler
///        mode parameters if already running in this mode.
/// 
/// @details
///     - This mode is optimized for multiple applications or contexts
///       submitting work to the hardware. When higher priority work arrives,
///       the scheduler attempts to pause the current executing work within some
///       timeout interval, then submits the other work.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hScheduler`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pProperties`
///         + `nullptr == pNeedReload`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + This scheduler component does not support scheduler modes.
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to make this modification.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesSchedulerSetTimeoutMode(
    zes_sched_handle_t hScheduler,                  ///< [in] Sysman handle for the component.
    zes_sched_timeout_properties_t* pProperties,    ///< [in] The properties to use when configurating this mode.
    ze_bool_t* pNeedReload                          ///< [in,out] Will be set to TRUE if a device driver reload is needed to
                                                    ///< apply the new scheduler mode.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Change scheduler mode to ::ZES_SCHED_MODE_TIMESLICE or update
///        scheduler mode parameters if already running in this mode.
/// 
/// @details
///     - This mode is optimized to provide fair sharing of hardware execution
///       time between multiple contexts submitting work to the hardware
///       concurrently.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hScheduler`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pProperties`
///         + `nullptr == pNeedReload`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + This scheduler component does not support scheduler modes.
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to make this modification.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesSchedulerSetTimesliceMode(
    zes_sched_handle_t hScheduler,                  ///< [in] Sysman handle for the component.
    zes_sched_timeslice_properties_t* pProperties,  ///< [in] The properties to use when configurating this mode.
    ze_bool_t* pNeedReload                          ///< [in,out] Will be set to TRUE if a device driver reload is needed to
                                                    ///< apply the new scheduler mode.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Change scheduler mode to ::ZES_SCHED_MODE_EXCLUSIVE
/// 
/// @details
///     - This mode is optimized for single application/context use-cases. It
///       permits a context to run indefinitely on the hardware without being
///       preempted or terminated. All pending work for other contexts must wait
///       until the running context completes with no further submitted work.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hScheduler`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pNeedReload`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + This scheduler component does not support scheduler modes.
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to make this modification.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesSchedulerSetExclusiveMode(
    zes_sched_handle_t hScheduler,                  ///< [in] Sysman handle for the component.
    ze_bool_t* pNeedReload                          ///< [in,out] Will be set to TRUE if a device driver reload is needed to
                                                    ///< apply the new scheduler mode.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Change scheduler mode to ::ZES_SCHED_MODE_COMPUTE_UNIT_DEBUG
/// 
/// @details
///     - This is a special mode that must ben enabled when debugging an
///       application that uses this device e.g. using the Level0 Debug API.
///     - It ensures that only one command queue can execute work on the
///       hardware at a given time. Work is permitted to run as long as needed
///       without enforcing any scheduler fairness policies.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hScheduler`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pNeedReload`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + This scheduler component does not support scheduler modes.
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to make this modification.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesSchedulerSetComputeUnitDebugMode(
    zes_sched_handle_t hScheduler,                  ///< [in] Sysman handle for the component.
    ze_bool_t* pNeedReload                          ///< [in,out] Will be set to TRUE if a device driver reload is needed to
                                                    ///< apply the new scheduler mode.
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Tool APIs for System Resource Management (Sysman) - Standby domains
#if !defined(__GNUC__)
#pragma region standby
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Standby hardware components
typedef enum _zes_standby_type_t
{
    ZES_STANDBY_TYPE_GLOBAL = 0,                    ///< Control the overall standby policy of the device/sub-device
    ZES_STANDBY_TYPE_FORCE_UINT32 = 0x7fffffff

} zes_standby_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Standby hardware component properties
typedef struct _zes_standby_properties_t
{
    zes_structure_type_t stype;                     ///< [in] type of this structure
    void* pNext;                                    ///< [in,out][optional] pointer to extension-specific structure
    zes_standby_type_t type;                        ///< [out] Which standby hardware component this controls
    ze_bool_t onSubdevice;                          ///< [out] True if the resource is located on a sub-device; false means
                                                    ///< that the resource is on the device of the calling Sysman handle
    uint32_t subdeviceId;                           ///< [out] If onSubdevice is true, this gives the ID of the sub-device

} zes_standby_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Standby promotion modes
typedef enum _zes_standby_promo_mode_t
{
    ZES_STANDBY_PROMO_MODE_DEFAULT = 0,             ///< Best compromise between performance and energy savings.
    ZES_STANDBY_PROMO_MODE_NEVER = 1,               ///< The device/component will never shutdown. This can improve performance
                                                    ///< but uses more energy.
    ZES_STANDBY_PROMO_MODE_FORCE_UINT32 = 0x7fffffff

} zes_standby_promo_mode_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Get handle of standby controls
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
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
zesDeviceEnumStandbyDomains(
    zes_device_handle_t hDevice,                    ///< [in] Sysman handle of the device.
    uint32_t* pCount,                               ///< [in,out] pointer to the number of components of this type.
                                                    ///< if count is zero, then the driver shall update the value with the
                                                    ///< total number of components of this type that are available.
                                                    ///< if count is greater than the number of components of this type that
                                                    ///< are available, then the driver shall update the value with the correct
                                                    ///< number of components.
    zes_standby_handle_t* phStandby                 ///< [in,out][optional][range(0, *pCount)] array of handle of components of
                                                    ///< this type.
                                                    ///< if count is less than the number of components of this type that are
                                                    ///< available, then the driver shall only retrieve that number of
                                                    ///< component handles.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get standby hardware component properties
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hStandby`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pProperties`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesStandbyGetProperties(
    zes_standby_handle_t hStandby,                  ///< [in] Handle for the component.
    zes_standby_properties_t* pProperties           ///< [in,out] Will contain the standby hardware properties.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get the current standby promotion mode
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hStandby`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pMode`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesStandbyGetMode(
    zes_standby_handle_t hStandby,                  ///< [in] Handle for the component.
    zes_standby_promo_mode_t* pMode                 ///< [in,out] Will contain the current standby mode.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Set standby promotion mode
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hStandby`
///     - ::ZE_RESULT_ERROR_INVALID_ENUMERATION
///         + `::ZES_STANDBY_PROMO_MODE_NEVER < mode`
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to make these modifications.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesStandbySetMode(
    zes_standby_handle_t hStandby,                  ///< [in] Handle for the component.
    zes_standby_promo_mode_t mode                   ///< [in] New standby mode.
    );

#if !defined(__GNUC__)
#pragma endregion
#endif
// Intel 'oneAPI' Level-Zero Tool APIs for System Resource Management (Sysman) - Firmware management
#if !defined(__GNUC__)
#pragma region temperature
#endif
///////////////////////////////////////////////////////////////////////////////
/// @brief Temperature sensors
typedef enum _zes_temp_sensors_t
{
    ZES_TEMP_SENSORS_GLOBAL = 0,                    ///< The maximum temperature across all device sensors
    ZES_TEMP_SENSORS_GPU = 1,                       ///< The maximum temperature across all sensors in the GPU
    ZES_TEMP_SENSORS_MEMORY = 2,                    ///< The maximum temperature across all sensors in the local memory
    ZES_TEMP_SENSORS_GLOBAL_MIN = 3,                ///< The minimum temperature across all device sensors
    ZES_TEMP_SENSORS_GPU_MIN = 4,                   ///< The minimum temperature across all sensors in the GPU
    ZES_TEMP_SENSORS_MEMORY_MIN = 5,                ///< The minimum temperature across all sensors in the local device memory
    ZES_TEMP_SENSORS_FORCE_UINT32 = 0x7fffffff

} zes_temp_sensors_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Temperature sensor properties
typedef struct _zes_temp_properties_t
{
    zes_structure_type_t stype;                     ///< [in] type of this structure
    void* pNext;                                    ///< [in,out][optional] pointer to extension-specific structure
    zes_temp_sensors_t type;                        ///< [out] Which part of the device the temperature sensor measures
    ze_bool_t onSubdevice;                          ///< [out] True if the resource is located on a sub-device; false means
                                                    ///< that the resource is on the device of the calling Sysman handle
    uint32_t subdeviceId;                           ///< [out] If onSubdevice is true, this gives the ID of the sub-device
    double maxTemperature;                          ///< [out] Will contain the maximum temperature for the specific device in
                                                    ///< degrees Celsius.
    ze_bool_t isCriticalTempSupported;              ///< [out] Indicates if the critical temperature event
                                                    ///< ::ZES_EVENT_TYPE_FLAG_TEMP_CRITICAL is supported
    ze_bool_t isThreshold1Supported;                ///< [out] Indicates if the temperature threshold 1 event
                                                    ///< ::ZES_EVENT_TYPE_FLAG_TEMP_THRESHOLD1 is supported
    ze_bool_t isThreshold2Supported;                ///< [out] Indicates if the temperature threshold 2 event
                                                    ///< ::ZES_EVENT_TYPE_FLAG_TEMP_THRESHOLD2 is supported

} zes_temp_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Temperature sensor threshold
typedef struct _zes_temp_threshold_t
{
    ze_bool_t enableLowToHigh;                      ///< [in,out] Trigger an event when the temperature crosses from below the
                                                    ///< threshold to above.
    ze_bool_t enableHighToLow;                      ///< [in,out] Trigger an event when the temperature crosses from above the
                                                    ///< threshold to below.
    double threshold;                               ///< [in,out] The threshold in degrees Celsius.

} zes_temp_threshold_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Temperature configuration - which events should be triggered and the
///        trigger conditions.
typedef struct _zes_temp_config_t
{
    zes_structure_type_t stype;                     ///< [in] type of this structure
    const void* pNext;                              ///< [in][optional] pointer to extension-specific structure
    ze_bool_t enableCritical;                       ///< [in,out] Indicates if event ::ZES_EVENT_TYPE_FLAG_TEMP_CRITICAL should
                                                    ///< be triggered by the driver.
    zes_temp_threshold_t threshold1;                ///< [in,out] Configuration controlling if and when event
                                                    ///< ::ZES_EVENT_TYPE_FLAG_TEMP_THRESHOLD1 should be triggered by the
                                                    ///< driver.
    zes_temp_threshold_t threshold2;                ///< [in,out] Configuration controlling if and when event
                                                    ///< ::ZES_EVENT_TYPE_FLAG_TEMP_THRESHOLD2 should be triggered by the
                                                    ///< driver.

} zes_temp_config_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Get handle of temperature sensors
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
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
zesDeviceEnumTemperatureSensors(
    zes_device_handle_t hDevice,                    ///< [in] Sysman handle of the device.
    uint32_t* pCount,                               ///< [in,out] pointer to the number of components of this type.
                                                    ///< if count is zero, then the driver shall update the value with the
                                                    ///< total number of components of this type that are available.
                                                    ///< if count is greater than the number of components of this type that
                                                    ///< are available, then the driver shall update the value with the correct
                                                    ///< number of components.
    zes_temp_handle_t* phTemperature                ///< [in,out][optional][range(0, *pCount)] array of handle of components of
                                                    ///< this type.
                                                    ///< if count is less than the number of components of this type that are
                                                    ///< available, then the driver shall only retrieve that number of
                                                    ///< component handles.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get temperature sensor properties
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hTemperature`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pProperties`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesTemperatureGetProperties(
    zes_temp_handle_t hTemperature,                 ///< [in] Handle for the component.
    zes_temp_properties_t* pProperties              ///< [in,out] Will contain the temperature sensor properties.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get temperature configuration for this sensor - which events are
///        triggered and the trigger conditions
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hTemperature`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pConfig`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + Temperature thresholds are not supported on this temperature sensor. Generally this is only supported for temperature sensor ::ZES_TEMP_SENSORS_GLOBAL
///         + One or both of the thresholds is not supported - check ::zes_temp_properties_t.isThreshold1Supported and ::zes_temp_properties_t.isThreshold2Supported
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to request this feature.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesTemperatureGetConfig(
    zes_temp_handle_t hTemperature,                 ///< [in] Handle for the component.
    zes_temp_config_t* pConfig                      ///< [in,out] Returns current configuration.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Set temperature configuration for this sensor - indicates which events
///        are triggered and the trigger conditions
/// 
/// @details
///     - Events ::ZES_EVENT_TYPE_FLAG_TEMP_CRITICAL will be triggered when
///       temperature reaches the critical range. Use the function
///       ::zesDeviceEventRegister() to start receiving this event.
///     - Events ::ZES_EVENT_TYPE_FLAG_TEMP_THRESHOLD1 and
///       ::ZES_EVENT_TYPE_FLAG_TEMP_THRESHOLD2 will be generated when
///       temperature cross the thresholds set using this function. Use the
///       function ::zesDeviceEventRegister() to start receiving these events.
///     - Only one running process can set the temperature configuration at a
///       time. If another process attempts to change the configuration, the
///       error ::ZE_RESULT_ERROR_NOT_AVAILABLE will be returned. The function
///       ::zesTemperatureGetConfig() will return the process ID currently
///       controlling these settings.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hTemperature`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pConfig`
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///         + Temperature thresholds are not supported on this temperature sensor. Generally they are only supported for temperature sensor ::ZES_TEMP_SENSORS_GLOBAL
///         + Enabling the critical temperature event is not supported - check ::zes_temp_properties_t.isCriticalTempSupported
///         + One or both of the thresholds is not supported - check ::zes_temp_properties_t.isThreshold1Supported and ::zes_temp_properties_t.isThreshold2Supported
///     - ::ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS
///         + User does not have permissions to request this feature.
///     - ::ZE_RESULT_ERROR_NOT_AVAILABLE
///         + Another running process is controlling these settings.
///     - ::ZE_RESULT_ERROR_INVALID_ARGUMENT
///         + One or both the thresholds is above TjMax (see ::zesFrequencyOcGetTjMax()). Temperature thresholds must be below this value.
ZE_APIEXPORT ze_result_t ZE_APICALL
zesTemperatureSetConfig(
    zes_temp_handle_t hTemperature,                 ///< [in] Handle for the component.
    const zes_temp_config_t* pConfig                ///< [in] New configuration.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Get the temperature from a specified sensor
/// 
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// 
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hTemperature`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pTemperature`
ZE_APIEXPORT ze_result_t ZE_APICALL
zesTemperatureGetState(
    zes_temp_handle_t hTemperature,                 ///< [in] Handle for the component.
    double* pTemperature                            ///< [in,out] Will contain the temperature read from the specified sensor
                                                    ///< in degrees Celsius.
    );

#if !defined(__GNUC__)
#pragma endregion
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // _ZES_API_H