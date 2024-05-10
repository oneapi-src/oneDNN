/*
 *
 * Copyright (C) 2019-2021 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 * @file zes_ddi.h
 * @version v1.3-r1.3.7
 *
 */
#ifndef _ZES_DDI_H
#define _ZES_DDI_H
#if defined(__cplusplus)
#pragma once
#endif
#include "zes_api.h"

#if defined(__cplusplus)
extern "C" {
#endif

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesDriverEventListen 
typedef ze_result_t (ZE_APICALL *zes_pfnDriverEventListen_t)(
    ze_driver_handle_t,
    uint32_t,
    uint32_t,
    zes_device_handle_t*,
    uint32_t*,
    zes_event_type_flags_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesDriverEventListenEx 
typedef ze_result_t (ZE_APICALL *zes_pfnDriverEventListenEx_t)(
    ze_driver_handle_t,
    uint64_t,
    uint32_t,
    zes_device_handle_t*,
    uint32_t*,
    zes_event_type_flags_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of Driver functions pointers
typedef struct _zes_driver_dditable_t
{
    zes_pfnDriverEventListen_t                                  pfnEventListen;
    zes_pfnDriverEventListenEx_t                                pfnEventListenEx;
} zes_driver_dditable_t;

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
zesGetDriverProcAddrTable(
    ze_api_version_t version,                       ///< [in] API version requested
    zes_driver_dditable_t* pDdiTable                ///< [in,out] pointer to table of DDI function pointers
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesGetDriverProcAddrTable
typedef ze_result_t (ZE_APICALL *zes_pfnGetDriverProcAddrTable_t)(
    ze_api_version_t,
    zes_driver_dditable_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesDeviceGetProperties 
typedef ze_result_t (ZE_APICALL *zes_pfnDeviceGetProperties_t)(
    zes_device_handle_t,
    zes_device_properties_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesDeviceGetState 
typedef ze_result_t (ZE_APICALL *zes_pfnDeviceGetState_t)(
    zes_device_handle_t,
    zes_device_state_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesDeviceReset 
typedef ze_result_t (ZE_APICALL *zes_pfnDeviceReset_t)(
    zes_device_handle_t,
    ze_bool_t
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesDeviceProcessesGetState 
typedef ze_result_t (ZE_APICALL *zes_pfnDeviceProcessesGetState_t)(
    zes_device_handle_t,
    uint32_t*,
    zes_process_state_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesDevicePciGetProperties 
typedef ze_result_t (ZE_APICALL *zes_pfnDevicePciGetProperties_t)(
    zes_device_handle_t,
    zes_pci_properties_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesDevicePciGetState 
typedef ze_result_t (ZE_APICALL *zes_pfnDevicePciGetState_t)(
    zes_device_handle_t,
    zes_pci_state_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesDevicePciGetBars 
typedef ze_result_t (ZE_APICALL *zes_pfnDevicePciGetBars_t)(
    zes_device_handle_t,
    uint32_t*,
    zes_pci_bar_properties_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesDevicePciGetStats 
typedef ze_result_t (ZE_APICALL *zes_pfnDevicePciGetStats_t)(
    zes_device_handle_t,
    zes_pci_stats_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesDeviceEnumDiagnosticTestSuites 
typedef ze_result_t (ZE_APICALL *zes_pfnDeviceEnumDiagnosticTestSuites_t)(
    zes_device_handle_t,
    uint32_t*,
    zes_diag_handle_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesDeviceEnumEngineGroups 
typedef ze_result_t (ZE_APICALL *zes_pfnDeviceEnumEngineGroups_t)(
    zes_device_handle_t,
    uint32_t*,
    zes_engine_handle_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesDeviceEventRegister 
typedef ze_result_t (ZE_APICALL *zes_pfnDeviceEventRegister_t)(
    zes_device_handle_t,
    zes_event_type_flags_t
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesDeviceEnumFabricPorts 
typedef ze_result_t (ZE_APICALL *zes_pfnDeviceEnumFabricPorts_t)(
    zes_device_handle_t,
    uint32_t*,
    zes_fabric_port_handle_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesDeviceEnumFans 
typedef ze_result_t (ZE_APICALL *zes_pfnDeviceEnumFans_t)(
    zes_device_handle_t,
    uint32_t*,
    zes_fan_handle_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesDeviceEnumFirmwares 
typedef ze_result_t (ZE_APICALL *zes_pfnDeviceEnumFirmwares_t)(
    zes_device_handle_t,
    uint32_t*,
    zes_firmware_handle_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesDeviceEnumFrequencyDomains 
typedef ze_result_t (ZE_APICALL *zes_pfnDeviceEnumFrequencyDomains_t)(
    zes_device_handle_t,
    uint32_t*,
    zes_freq_handle_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesDeviceEnumLeds 
typedef ze_result_t (ZE_APICALL *zes_pfnDeviceEnumLeds_t)(
    zes_device_handle_t,
    uint32_t*,
    zes_led_handle_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesDeviceEnumMemoryModules 
typedef ze_result_t (ZE_APICALL *zes_pfnDeviceEnumMemoryModules_t)(
    zes_device_handle_t,
    uint32_t*,
    zes_mem_handle_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesDeviceEnumPerformanceFactorDomains 
typedef ze_result_t (ZE_APICALL *zes_pfnDeviceEnumPerformanceFactorDomains_t)(
    zes_device_handle_t,
    uint32_t*,
    zes_perf_handle_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesDeviceEnumPowerDomains 
typedef ze_result_t (ZE_APICALL *zes_pfnDeviceEnumPowerDomains_t)(
    zes_device_handle_t,
    uint32_t*,
    zes_pwr_handle_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesDeviceGetCardPowerDomain 
typedef ze_result_t (ZE_APICALL *zes_pfnDeviceGetCardPowerDomain_t)(
    zes_device_handle_t,
    zes_pwr_handle_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesDeviceEnumPsus 
typedef ze_result_t (ZE_APICALL *zes_pfnDeviceEnumPsus_t)(
    zes_device_handle_t,
    uint32_t*,
    zes_psu_handle_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesDeviceEnumRasErrorSets 
typedef ze_result_t (ZE_APICALL *zes_pfnDeviceEnumRasErrorSets_t)(
    zes_device_handle_t,
    uint32_t*,
    zes_ras_handle_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesDeviceEnumSchedulers 
typedef ze_result_t (ZE_APICALL *zes_pfnDeviceEnumSchedulers_t)(
    zes_device_handle_t,
    uint32_t*,
    zes_sched_handle_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesDeviceEnumStandbyDomains 
typedef ze_result_t (ZE_APICALL *zes_pfnDeviceEnumStandbyDomains_t)(
    zes_device_handle_t,
    uint32_t*,
    zes_standby_handle_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesDeviceEnumTemperatureSensors 
typedef ze_result_t (ZE_APICALL *zes_pfnDeviceEnumTemperatureSensors_t)(
    zes_device_handle_t,
    uint32_t*,
    zes_temp_handle_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of Device functions pointers
typedef struct _zes_device_dditable_t
{
    zes_pfnDeviceGetProperties_t                                pfnGetProperties;
    zes_pfnDeviceGetState_t                                     pfnGetState;
    zes_pfnDeviceReset_t                                        pfnReset;
    zes_pfnDeviceProcessesGetState_t                            pfnProcessesGetState;
    zes_pfnDevicePciGetProperties_t                             pfnPciGetProperties;
    zes_pfnDevicePciGetState_t                                  pfnPciGetState;
    zes_pfnDevicePciGetBars_t                                   pfnPciGetBars;
    zes_pfnDevicePciGetStats_t                                  pfnPciGetStats;
    zes_pfnDeviceEnumDiagnosticTestSuites_t                     pfnEnumDiagnosticTestSuites;
    zes_pfnDeviceEnumEngineGroups_t                             pfnEnumEngineGroups;
    zes_pfnDeviceEventRegister_t                                pfnEventRegister;
    zes_pfnDeviceEnumFabricPorts_t                              pfnEnumFabricPorts;
    zes_pfnDeviceEnumFans_t                                     pfnEnumFans;
    zes_pfnDeviceEnumFirmwares_t                                pfnEnumFirmwares;
    zes_pfnDeviceEnumFrequencyDomains_t                         pfnEnumFrequencyDomains;
    zes_pfnDeviceEnumLeds_t                                     pfnEnumLeds;
    zes_pfnDeviceEnumMemoryModules_t                            pfnEnumMemoryModules;
    zes_pfnDeviceEnumPerformanceFactorDomains_t                 pfnEnumPerformanceFactorDomains;
    zes_pfnDeviceEnumPowerDomains_t                             pfnEnumPowerDomains;
    zes_pfnDeviceGetCardPowerDomain_t                           pfnGetCardPowerDomain;
    zes_pfnDeviceEnumPsus_t                                     pfnEnumPsus;
    zes_pfnDeviceEnumRasErrorSets_t                             pfnEnumRasErrorSets;
    zes_pfnDeviceEnumSchedulers_t                               pfnEnumSchedulers;
    zes_pfnDeviceEnumStandbyDomains_t                           pfnEnumStandbyDomains;
    zes_pfnDeviceEnumTemperatureSensors_t                       pfnEnumTemperatureSensors;
} zes_device_dditable_t;

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
zesGetDeviceProcAddrTable(
    ze_api_version_t version,                       ///< [in] API version requested
    zes_device_dditable_t* pDdiTable                ///< [in,out] pointer to table of DDI function pointers
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesGetDeviceProcAddrTable
typedef ze_result_t (ZE_APICALL *zes_pfnGetDeviceProcAddrTable_t)(
    ze_api_version_t,
    zes_device_dditable_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesSchedulerGetProperties 
typedef ze_result_t (ZE_APICALL *zes_pfnSchedulerGetProperties_t)(
    zes_sched_handle_t,
    zes_sched_properties_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesSchedulerGetCurrentMode 
typedef ze_result_t (ZE_APICALL *zes_pfnSchedulerGetCurrentMode_t)(
    zes_sched_handle_t,
    zes_sched_mode_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesSchedulerGetTimeoutModeProperties 
typedef ze_result_t (ZE_APICALL *zes_pfnSchedulerGetTimeoutModeProperties_t)(
    zes_sched_handle_t,
    ze_bool_t,
    zes_sched_timeout_properties_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesSchedulerGetTimesliceModeProperties 
typedef ze_result_t (ZE_APICALL *zes_pfnSchedulerGetTimesliceModeProperties_t)(
    zes_sched_handle_t,
    ze_bool_t,
    zes_sched_timeslice_properties_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesSchedulerSetTimeoutMode 
typedef ze_result_t (ZE_APICALL *zes_pfnSchedulerSetTimeoutMode_t)(
    zes_sched_handle_t,
    zes_sched_timeout_properties_t*,
    ze_bool_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesSchedulerSetTimesliceMode 
typedef ze_result_t (ZE_APICALL *zes_pfnSchedulerSetTimesliceMode_t)(
    zes_sched_handle_t,
    zes_sched_timeslice_properties_t*,
    ze_bool_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesSchedulerSetExclusiveMode 
typedef ze_result_t (ZE_APICALL *zes_pfnSchedulerSetExclusiveMode_t)(
    zes_sched_handle_t,
    ze_bool_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesSchedulerSetComputeUnitDebugMode 
typedef ze_result_t (ZE_APICALL *zes_pfnSchedulerSetComputeUnitDebugMode_t)(
    zes_sched_handle_t,
    ze_bool_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of Scheduler functions pointers
typedef struct _zes_scheduler_dditable_t
{
    zes_pfnSchedulerGetProperties_t                             pfnGetProperties;
    zes_pfnSchedulerGetCurrentMode_t                            pfnGetCurrentMode;
    zes_pfnSchedulerGetTimeoutModeProperties_t                  pfnGetTimeoutModeProperties;
    zes_pfnSchedulerGetTimesliceModeProperties_t                pfnGetTimesliceModeProperties;
    zes_pfnSchedulerSetTimeoutMode_t                            pfnSetTimeoutMode;
    zes_pfnSchedulerSetTimesliceMode_t                          pfnSetTimesliceMode;
    zes_pfnSchedulerSetExclusiveMode_t                          pfnSetExclusiveMode;
    zes_pfnSchedulerSetComputeUnitDebugMode_t                   pfnSetComputeUnitDebugMode;
} zes_scheduler_dditable_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Scheduler table
///        with current process' addresses
///
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_VERSION
ZE_DLLEXPORT ze_result_t ZE_APICALL
zesGetSchedulerProcAddrTable(
    ze_api_version_t version,                       ///< [in] API version requested
    zes_scheduler_dditable_t* pDdiTable             ///< [in,out] pointer to table of DDI function pointers
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesGetSchedulerProcAddrTable
typedef ze_result_t (ZE_APICALL *zes_pfnGetSchedulerProcAddrTable_t)(
    ze_api_version_t,
    zes_scheduler_dditable_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesPerformanceFactorGetProperties 
typedef ze_result_t (ZE_APICALL *zes_pfnPerformanceFactorGetProperties_t)(
    zes_perf_handle_t,
    zes_perf_properties_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesPerformanceFactorGetConfig 
typedef ze_result_t (ZE_APICALL *zes_pfnPerformanceFactorGetConfig_t)(
    zes_perf_handle_t,
    double*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesPerformanceFactorSetConfig 
typedef ze_result_t (ZE_APICALL *zes_pfnPerformanceFactorSetConfig_t)(
    zes_perf_handle_t,
    double
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of PerformanceFactor functions pointers
typedef struct _zes_performance_factor_dditable_t
{
    zes_pfnPerformanceFactorGetProperties_t                     pfnGetProperties;
    zes_pfnPerformanceFactorGetConfig_t                         pfnGetConfig;
    zes_pfnPerformanceFactorSetConfig_t                         pfnSetConfig;
} zes_performance_factor_dditable_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's PerformanceFactor table
///        with current process' addresses
///
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_VERSION
ZE_DLLEXPORT ze_result_t ZE_APICALL
zesGetPerformanceFactorProcAddrTable(
    ze_api_version_t version,                       ///< [in] API version requested
    zes_performance_factor_dditable_t* pDdiTable    ///< [in,out] pointer to table of DDI function pointers
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesGetPerformanceFactorProcAddrTable
typedef ze_result_t (ZE_APICALL *zes_pfnGetPerformanceFactorProcAddrTable_t)(
    ze_api_version_t,
    zes_performance_factor_dditable_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesPowerGetProperties 
typedef ze_result_t (ZE_APICALL *zes_pfnPowerGetProperties_t)(
    zes_pwr_handle_t,
    zes_power_properties_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesPowerGetEnergyCounter 
typedef ze_result_t (ZE_APICALL *zes_pfnPowerGetEnergyCounter_t)(
    zes_pwr_handle_t,
    zes_power_energy_counter_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesPowerGetLimits 
typedef ze_result_t (ZE_APICALL *zes_pfnPowerGetLimits_t)(
    zes_pwr_handle_t,
    zes_power_sustained_limit_t*,
    zes_power_burst_limit_t*,
    zes_power_peak_limit_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesPowerSetLimits 
typedef ze_result_t (ZE_APICALL *zes_pfnPowerSetLimits_t)(
    zes_pwr_handle_t,
    const zes_power_sustained_limit_t*,
    const zes_power_burst_limit_t*,
    const zes_power_peak_limit_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesPowerGetEnergyThreshold 
typedef ze_result_t (ZE_APICALL *zes_pfnPowerGetEnergyThreshold_t)(
    zes_pwr_handle_t,
    zes_energy_threshold_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesPowerSetEnergyThreshold 
typedef ze_result_t (ZE_APICALL *zes_pfnPowerSetEnergyThreshold_t)(
    zes_pwr_handle_t,
    double
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of Power functions pointers
typedef struct _zes_power_dditable_t
{
    zes_pfnPowerGetProperties_t                                 pfnGetProperties;
    zes_pfnPowerGetEnergyCounter_t                              pfnGetEnergyCounter;
    zes_pfnPowerGetLimits_t                                     pfnGetLimits;
    zes_pfnPowerSetLimits_t                                     pfnSetLimits;
    zes_pfnPowerGetEnergyThreshold_t                            pfnGetEnergyThreshold;
    zes_pfnPowerSetEnergyThreshold_t                            pfnSetEnergyThreshold;
} zes_power_dditable_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Power table
///        with current process' addresses
///
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_VERSION
ZE_DLLEXPORT ze_result_t ZE_APICALL
zesGetPowerProcAddrTable(
    ze_api_version_t version,                       ///< [in] API version requested
    zes_power_dditable_t* pDdiTable                 ///< [in,out] pointer to table of DDI function pointers
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesGetPowerProcAddrTable
typedef ze_result_t (ZE_APICALL *zes_pfnGetPowerProcAddrTable_t)(
    ze_api_version_t,
    zes_power_dditable_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesFrequencyGetProperties 
typedef ze_result_t (ZE_APICALL *zes_pfnFrequencyGetProperties_t)(
    zes_freq_handle_t,
    zes_freq_properties_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesFrequencyGetAvailableClocks 
typedef ze_result_t (ZE_APICALL *zes_pfnFrequencyGetAvailableClocks_t)(
    zes_freq_handle_t,
    uint32_t*,
    double*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesFrequencyGetRange 
typedef ze_result_t (ZE_APICALL *zes_pfnFrequencyGetRange_t)(
    zes_freq_handle_t,
    zes_freq_range_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesFrequencySetRange 
typedef ze_result_t (ZE_APICALL *zes_pfnFrequencySetRange_t)(
    zes_freq_handle_t,
    const zes_freq_range_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesFrequencyGetState 
typedef ze_result_t (ZE_APICALL *zes_pfnFrequencyGetState_t)(
    zes_freq_handle_t,
    zes_freq_state_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesFrequencyGetThrottleTime 
typedef ze_result_t (ZE_APICALL *zes_pfnFrequencyGetThrottleTime_t)(
    zes_freq_handle_t,
    zes_freq_throttle_time_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesFrequencyOcGetCapabilities 
typedef ze_result_t (ZE_APICALL *zes_pfnFrequencyOcGetCapabilities_t)(
    zes_freq_handle_t,
    zes_oc_capabilities_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesFrequencyOcGetFrequencyTarget 
typedef ze_result_t (ZE_APICALL *zes_pfnFrequencyOcGetFrequencyTarget_t)(
    zes_freq_handle_t,
    double*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesFrequencyOcSetFrequencyTarget 
typedef ze_result_t (ZE_APICALL *zes_pfnFrequencyOcSetFrequencyTarget_t)(
    zes_freq_handle_t,
    double
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesFrequencyOcGetVoltageTarget 
typedef ze_result_t (ZE_APICALL *zes_pfnFrequencyOcGetVoltageTarget_t)(
    zes_freq_handle_t,
    double*,
    double*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesFrequencyOcSetVoltageTarget 
typedef ze_result_t (ZE_APICALL *zes_pfnFrequencyOcSetVoltageTarget_t)(
    zes_freq_handle_t,
    double,
    double
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesFrequencyOcSetMode 
typedef ze_result_t (ZE_APICALL *zes_pfnFrequencyOcSetMode_t)(
    zes_freq_handle_t,
    zes_oc_mode_t
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesFrequencyOcGetMode 
typedef ze_result_t (ZE_APICALL *zes_pfnFrequencyOcGetMode_t)(
    zes_freq_handle_t,
    zes_oc_mode_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesFrequencyOcGetIccMax 
typedef ze_result_t (ZE_APICALL *zes_pfnFrequencyOcGetIccMax_t)(
    zes_freq_handle_t,
    double*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesFrequencyOcSetIccMax 
typedef ze_result_t (ZE_APICALL *zes_pfnFrequencyOcSetIccMax_t)(
    zes_freq_handle_t,
    double
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesFrequencyOcGetTjMax 
typedef ze_result_t (ZE_APICALL *zes_pfnFrequencyOcGetTjMax_t)(
    zes_freq_handle_t,
    double*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesFrequencyOcSetTjMax 
typedef ze_result_t (ZE_APICALL *zes_pfnFrequencyOcSetTjMax_t)(
    zes_freq_handle_t,
    double
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of Frequency functions pointers
typedef struct _zes_frequency_dditable_t
{
    zes_pfnFrequencyGetProperties_t                             pfnGetProperties;
    zes_pfnFrequencyGetAvailableClocks_t                        pfnGetAvailableClocks;
    zes_pfnFrequencyGetRange_t                                  pfnGetRange;
    zes_pfnFrequencySetRange_t                                  pfnSetRange;
    zes_pfnFrequencyGetState_t                                  pfnGetState;
    zes_pfnFrequencyGetThrottleTime_t                           pfnGetThrottleTime;
    zes_pfnFrequencyOcGetCapabilities_t                         pfnOcGetCapabilities;
    zes_pfnFrequencyOcGetFrequencyTarget_t                      pfnOcGetFrequencyTarget;
    zes_pfnFrequencyOcSetFrequencyTarget_t                      pfnOcSetFrequencyTarget;
    zes_pfnFrequencyOcGetVoltageTarget_t                        pfnOcGetVoltageTarget;
    zes_pfnFrequencyOcSetVoltageTarget_t                        pfnOcSetVoltageTarget;
    zes_pfnFrequencyOcSetMode_t                                 pfnOcSetMode;
    zes_pfnFrequencyOcGetMode_t                                 pfnOcGetMode;
    zes_pfnFrequencyOcGetIccMax_t                               pfnOcGetIccMax;
    zes_pfnFrequencyOcSetIccMax_t                               pfnOcSetIccMax;
    zes_pfnFrequencyOcGetTjMax_t                                pfnOcGetTjMax;
    zes_pfnFrequencyOcSetTjMax_t                                pfnOcSetTjMax;
} zes_frequency_dditable_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Frequency table
///        with current process' addresses
///
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_VERSION
ZE_DLLEXPORT ze_result_t ZE_APICALL
zesGetFrequencyProcAddrTable(
    ze_api_version_t version,                       ///< [in] API version requested
    zes_frequency_dditable_t* pDdiTable             ///< [in,out] pointer to table of DDI function pointers
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesGetFrequencyProcAddrTable
typedef ze_result_t (ZE_APICALL *zes_pfnGetFrequencyProcAddrTable_t)(
    ze_api_version_t,
    zes_frequency_dditable_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesEngineGetProperties 
typedef ze_result_t (ZE_APICALL *zes_pfnEngineGetProperties_t)(
    zes_engine_handle_t,
    zes_engine_properties_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesEngineGetActivity 
typedef ze_result_t (ZE_APICALL *zes_pfnEngineGetActivity_t)(
    zes_engine_handle_t,
    zes_engine_stats_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of Engine functions pointers
typedef struct _zes_engine_dditable_t
{
    zes_pfnEngineGetProperties_t                                pfnGetProperties;
    zes_pfnEngineGetActivity_t                                  pfnGetActivity;
} zes_engine_dditable_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Engine table
///        with current process' addresses
///
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_VERSION
ZE_DLLEXPORT ze_result_t ZE_APICALL
zesGetEngineProcAddrTable(
    ze_api_version_t version,                       ///< [in] API version requested
    zes_engine_dditable_t* pDdiTable                ///< [in,out] pointer to table of DDI function pointers
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesGetEngineProcAddrTable
typedef ze_result_t (ZE_APICALL *zes_pfnGetEngineProcAddrTable_t)(
    ze_api_version_t,
    zes_engine_dditable_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesStandbyGetProperties 
typedef ze_result_t (ZE_APICALL *zes_pfnStandbyGetProperties_t)(
    zes_standby_handle_t,
    zes_standby_properties_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesStandbyGetMode 
typedef ze_result_t (ZE_APICALL *zes_pfnStandbyGetMode_t)(
    zes_standby_handle_t,
    zes_standby_promo_mode_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesStandbySetMode 
typedef ze_result_t (ZE_APICALL *zes_pfnStandbySetMode_t)(
    zes_standby_handle_t,
    zes_standby_promo_mode_t
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of Standby functions pointers
typedef struct _zes_standby_dditable_t
{
    zes_pfnStandbyGetProperties_t                               pfnGetProperties;
    zes_pfnStandbyGetMode_t                                     pfnGetMode;
    zes_pfnStandbySetMode_t                                     pfnSetMode;
} zes_standby_dditable_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Standby table
///        with current process' addresses
///
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_VERSION
ZE_DLLEXPORT ze_result_t ZE_APICALL
zesGetStandbyProcAddrTable(
    ze_api_version_t version,                       ///< [in] API version requested
    zes_standby_dditable_t* pDdiTable               ///< [in,out] pointer to table of DDI function pointers
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesGetStandbyProcAddrTable
typedef ze_result_t (ZE_APICALL *zes_pfnGetStandbyProcAddrTable_t)(
    ze_api_version_t,
    zes_standby_dditable_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesFirmwareGetProperties 
typedef ze_result_t (ZE_APICALL *zes_pfnFirmwareGetProperties_t)(
    zes_firmware_handle_t,
    zes_firmware_properties_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesFirmwareFlash 
typedef ze_result_t (ZE_APICALL *zes_pfnFirmwareFlash_t)(
    zes_firmware_handle_t,
    void*,
    uint32_t
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of Firmware functions pointers
typedef struct _zes_firmware_dditable_t
{
    zes_pfnFirmwareGetProperties_t                              pfnGetProperties;
    zes_pfnFirmwareFlash_t                                      pfnFlash;
} zes_firmware_dditable_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Firmware table
///        with current process' addresses
///
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_VERSION
ZE_DLLEXPORT ze_result_t ZE_APICALL
zesGetFirmwareProcAddrTable(
    ze_api_version_t version,                       ///< [in] API version requested
    zes_firmware_dditable_t* pDdiTable              ///< [in,out] pointer to table of DDI function pointers
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesGetFirmwareProcAddrTable
typedef ze_result_t (ZE_APICALL *zes_pfnGetFirmwareProcAddrTable_t)(
    ze_api_version_t,
    zes_firmware_dditable_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesMemoryGetProperties 
typedef ze_result_t (ZE_APICALL *zes_pfnMemoryGetProperties_t)(
    zes_mem_handle_t,
    zes_mem_properties_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesMemoryGetState 
typedef ze_result_t (ZE_APICALL *zes_pfnMemoryGetState_t)(
    zes_mem_handle_t,
    zes_mem_state_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesMemoryGetBandwidth 
typedef ze_result_t (ZE_APICALL *zes_pfnMemoryGetBandwidth_t)(
    zes_mem_handle_t,
    zes_mem_bandwidth_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of Memory functions pointers
typedef struct _zes_memory_dditable_t
{
    zes_pfnMemoryGetProperties_t                                pfnGetProperties;
    zes_pfnMemoryGetState_t                                     pfnGetState;
    zes_pfnMemoryGetBandwidth_t                                 pfnGetBandwidth;
} zes_memory_dditable_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Memory table
///        with current process' addresses
///
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_VERSION
ZE_DLLEXPORT ze_result_t ZE_APICALL
zesGetMemoryProcAddrTable(
    ze_api_version_t version,                       ///< [in] API version requested
    zes_memory_dditable_t* pDdiTable                ///< [in,out] pointer to table of DDI function pointers
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesGetMemoryProcAddrTable
typedef ze_result_t (ZE_APICALL *zes_pfnGetMemoryProcAddrTable_t)(
    ze_api_version_t,
    zes_memory_dditable_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesFabricPortGetProperties 
typedef ze_result_t (ZE_APICALL *zes_pfnFabricPortGetProperties_t)(
    zes_fabric_port_handle_t,
    zes_fabric_port_properties_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesFabricPortGetLinkType 
typedef ze_result_t (ZE_APICALL *zes_pfnFabricPortGetLinkType_t)(
    zes_fabric_port_handle_t,
    zes_fabric_link_type_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesFabricPortGetConfig 
typedef ze_result_t (ZE_APICALL *zes_pfnFabricPortGetConfig_t)(
    zes_fabric_port_handle_t,
    zes_fabric_port_config_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesFabricPortSetConfig 
typedef ze_result_t (ZE_APICALL *zes_pfnFabricPortSetConfig_t)(
    zes_fabric_port_handle_t,
    const zes_fabric_port_config_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesFabricPortGetState 
typedef ze_result_t (ZE_APICALL *zes_pfnFabricPortGetState_t)(
    zes_fabric_port_handle_t,
    zes_fabric_port_state_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesFabricPortGetThroughput 
typedef ze_result_t (ZE_APICALL *zes_pfnFabricPortGetThroughput_t)(
    zes_fabric_port_handle_t,
    zes_fabric_port_throughput_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of FabricPort functions pointers
typedef struct _zes_fabric_port_dditable_t
{
    zes_pfnFabricPortGetProperties_t                            pfnGetProperties;
    zes_pfnFabricPortGetLinkType_t                              pfnGetLinkType;
    zes_pfnFabricPortGetConfig_t                                pfnGetConfig;
    zes_pfnFabricPortSetConfig_t                                pfnSetConfig;
    zes_pfnFabricPortGetState_t                                 pfnGetState;
    zes_pfnFabricPortGetThroughput_t                            pfnGetThroughput;
} zes_fabric_port_dditable_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's FabricPort table
///        with current process' addresses
///
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_VERSION
ZE_DLLEXPORT ze_result_t ZE_APICALL
zesGetFabricPortProcAddrTable(
    ze_api_version_t version,                       ///< [in] API version requested
    zes_fabric_port_dditable_t* pDdiTable           ///< [in,out] pointer to table of DDI function pointers
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesGetFabricPortProcAddrTable
typedef ze_result_t (ZE_APICALL *zes_pfnGetFabricPortProcAddrTable_t)(
    ze_api_version_t,
    zes_fabric_port_dditable_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesTemperatureGetProperties 
typedef ze_result_t (ZE_APICALL *zes_pfnTemperatureGetProperties_t)(
    zes_temp_handle_t,
    zes_temp_properties_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesTemperatureGetConfig 
typedef ze_result_t (ZE_APICALL *zes_pfnTemperatureGetConfig_t)(
    zes_temp_handle_t,
    zes_temp_config_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesTemperatureSetConfig 
typedef ze_result_t (ZE_APICALL *zes_pfnTemperatureSetConfig_t)(
    zes_temp_handle_t,
    const zes_temp_config_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesTemperatureGetState 
typedef ze_result_t (ZE_APICALL *zes_pfnTemperatureGetState_t)(
    zes_temp_handle_t,
    double*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of Temperature functions pointers
typedef struct _zes_temperature_dditable_t
{
    zes_pfnTemperatureGetProperties_t                           pfnGetProperties;
    zes_pfnTemperatureGetConfig_t                               pfnGetConfig;
    zes_pfnTemperatureSetConfig_t                               pfnSetConfig;
    zes_pfnTemperatureGetState_t                                pfnGetState;
} zes_temperature_dditable_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Temperature table
///        with current process' addresses
///
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_VERSION
ZE_DLLEXPORT ze_result_t ZE_APICALL
zesGetTemperatureProcAddrTable(
    ze_api_version_t version,                       ///< [in] API version requested
    zes_temperature_dditable_t* pDdiTable           ///< [in,out] pointer to table of DDI function pointers
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesGetTemperatureProcAddrTable
typedef ze_result_t (ZE_APICALL *zes_pfnGetTemperatureProcAddrTable_t)(
    ze_api_version_t,
    zes_temperature_dditable_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesPsuGetProperties 
typedef ze_result_t (ZE_APICALL *zes_pfnPsuGetProperties_t)(
    zes_psu_handle_t,
    zes_psu_properties_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesPsuGetState 
typedef ze_result_t (ZE_APICALL *zes_pfnPsuGetState_t)(
    zes_psu_handle_t,
    zes_psu_state_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of Psu functions pointers
typedef struct _zes_psu_dditable_t
{
    zes_pfnPsuGetProperties_t                                   pfnGetProperties;
    zes_pfnPsuGetState_t                                        pfnGetState;
} zes_psu_dditable_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Psu table
///        with current process' addresses
///
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_VERSION
ZE_DLLEXPORT ze_result_t ZE_APICALL
zesGetPsuProcAddrTable(
    ze_api_version_t version,                       ///< [in] API version requested
    zes_psu_dditable_t* pDdiTable                   ///< [in,out] pointer to table of DDI function pointers
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesGetPsuProcAddrTable
typedef ze_result_t (ZE_APICALL *zes_pfnGetPsuProcAddrTable_t)(
    ze_api_version_t,
    zes_psu_dditable_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesFanGetProperties 
typedef ze_result_t (ZE_APICALL *zes_pfnFanGetProperties_t)(
    zes_fan_handle_t,
    zes_fan_properties_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesFanGetConfig 
typedef ze_result_t (ZE_APICALL *zes_pfnFanGetConfig_t)(
    zes_fan_handle_t,
    zes_fan_config_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesFanSetDefaultMode 
typedef ze_result_t (ZE_APICALL *zes_pfnFanSetDefaultMode_t)(
    zes_fan_handle_t
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesFanSetFixedSpeedMode 
typedef ze_result_t (ZE_APICALL *zes_pfnFanSetFixedSpeedMode_t)(
    zes_fan_handle_t,
    const zes_fan_speed_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesFanSetSpeedTableMode 
typedef ze_result_t (ZE_APICALL *zes_pfnFanSetSpeedTableMode_t)(
    zes_fan_handle_t,
    const zes_fan_speed_table_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesFanGetState 
typedef ze_result_t (ZE_APICALL *zes_pfnFanGetState_t)(
    zes_fan_handle_t,
    zes_fan_speed_units_t,
    int32_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of Fan functions pointers
typedef struct _zes_fan_dditable_t
{
    zes_pfnFanGetProperties_t                                   pfnGetProperties;
    zes_pfnFanGetConfig_t                                       pfnGetConfig;
    zes_pfnFanSetDefaultMode_t                                  pfnSetDefaultMode;
    zes_pfnFanSetFixedSpeedMode_t                               pfnSetFixedSpeedMode;
    zes_pfnFanSetSpeedTableMode_t                               pfnSetSpeedTableMode;
    zes_pfnFanGetState_t                                        pfnGetState;
} zes_fan_dditable_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Fan table
///        with current process' addresses
///
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_VERSION
ZE_DLLEXPORT ze_result_t ZE_APICALL
zesGetFanProcAddrTable(
    ze_api_version_t version,                       ///< [in] API version requested
    zes_fan_dditable_t* pDdiTable                   ///< [in,out] pointer to table of DDI function pointers
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesGetFanProcAddrTable
typedef ze_result_t (ZE_APICALL *zes_pfnGetFanProcAddrTable_t)(
    ze_api_version_t,
    zes_fan_dditable_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesLedGetProperties 
typedef ze_result_t (ZE_APICALL *zes_pfnLedGetProperties_t)(
    zes_led_handle_t,
    zes_led_properties_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesLedGetState 
typedef ze_result_t (ZE_APICALL *zes_pfnLedGetState_t)(
    zes_led_handle_t,
    zes_led_state_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesLedSetState 
typedef ze_result_t (ZE_APICALL *zes_pfnLedSetState_t)(
    zes_led_handle_t,
    ze_bool_t
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesLedSetColor 
typedef ze_result_t (ZE_APICALL *zes_pfnLedSetColor_t)(
    zes_led_handle_t,
    const zes_led_color_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of Led functions pointers
typedef struct _zes_led_dditable_t
{
    zes_pfnLedGetProperties_t                                   pfnGetProperties;
    zes_pfnLedGetState_t                                        pfnGetState;
    zes_pfnLedSetState_t                                        pfnSetState;
    zes_pfnLedSetColor_t                                        pfnSetColor;
} zes_led_dditable_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Led table
///        with current process' addresses
///
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_VERSION
ZE_DLLEXPORT ze_result_t ZE_APICALL
zesGetLedProcAddrTable(
    ze_api_version_t version,                       ///< [in] API version requested
    zes_led_dditable_t* pDdiTable                   ///< [in,out] pointer to table of DDI function pointers
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesGetLedProcAddrTable
typedef ze_result_t (ZE_APICALL *zes_pfnGetLedProcAddrTable_t)(
    ze_api_version_t,
    zes_led_dditable_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesRasGetProperties 
typedef ze_result_t (ZE_APICALL *zes_pfnRasGetProperties_t)(
    zes_ras_handle_t,
    zes_ras_properties_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesRasGetConfig 
typedef ze_result_t (ZE_APICALL *zes_pfnRasGetConfig_t)(
    zes_ras_handle_t,
    zes_ras_config_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesRasSetConfig 
typedef ze_result_t (ZE_APICALL *zes_pfnRasSetConfig_t)(
    zes_ras_handle_t,
    const zes_ras_config_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesRasGetState 
typedef ze_result_t (ZE_APICALL *zes_pfnRasGetState_t)(
    zes_ras_handle_t,
    ze_bool_t,
    zes_ras_state_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of Ras functions pointers
typedef struct _zes_ras_dditable_t
{
    zes_pfnRasGetProperties_t                                   pfnGetProperties;
    zes_pfnRasGetConfig_t                                       pfnGetConfig;
    zes_pfnRasSetConfig_t                                       pfnSetConfig;
    zes_pfnRasGetState_t                                        pfnGetState;
} zes_ras_dditable_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Ras table
///        with current process' addresses
///
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_VERSION
ZE_DLLEXPORT ze_result_t ZE_APICALL
zesGetRasProcAddrTable(
    ze_api_version_t version,                       ///< [in] API version requested
    zes_ras_dditable_t* pDdiTable                   ///< [in,out] pointer to table of DDI function pointers
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesGetRasProcAddrTable
typedef ze_result_t (ZE_APICALL *zes_pfnGetRasProcAddrTable_t)(
    ze_api_version_t,
    zes_ras_dditable_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesDiagnosticsGetProperties 
typedef ze_result_t (ZE_APICALL *zes_pfnDiagnosticsGetProperties_t)(
    zes_diag_handle_t,
    zes_diag_properties_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesDiagnosticsGetTests 
typedef ze_result_t (ZE_APICALL *zes_pfnDiagnosticsGetTests_t)(
    zes_diag_handle_t,
    uint32_t*,
    zes_diag_test_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesDiagnosticsRunTests 
typedef ze_result_t (ZE_APICALL *zes_pfnDiagnosticsRunTests_t)(
    zes_diag_handle_t,
    uint32_t,
    uint32_t,
    zes_diag_result_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of Diagnostics functions pointers
typedef struct _zes_diagnostics_dditable_t
{
    zes_pfnDiagnosticsGetProperties_t                           pfnGetProperties;
    zes_pfnDiagnosticsGetTests_t                                pfnGetTests;
    zes_pfnDiagnosticsRunTests_t                                pfnRunTests;
} zes_diagnostics_dditable_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Exported function for filling application's Diagnostics table
///        with current process' addresses
///
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_VERSION
ZE_DLLEXPORT ze_result_t ZE_APICALL
zesGetDiagnosticsProcAddrTable(
    ze_api_version_t version,                       ///< [in] API version requested
    zes_diagnostics_dditable_t* pDdiTable           ///< [in,out] pointer to table of DDI function pointers
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Function-pointer for zesGetDiagnosticsProcAddrTable
typedef ze_result_t (ZE_APICALL *zes_pfnGetDiagnosticsProcAddrTable_t)(
    ze_api_version_t,
    zes_diagnostics_dditable_t*
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Container for all DDI tables
typedef struct _zes_dditable_t
{
    zes_driver_dditable_t               Driver;
    zes_device_dditable_t               Device;
    zes_scheduler_dditable_t            Scheduler;
    zes_performance_factor_dditable_t   PerformanceFactor;
    zes_power_dditable_t                Power;
    zes_frequency_dditable_t            Frequency;
    zes_engine_dditable_t               Engine;
    zes_standby_dditable_t              Standby;
    zes_firmware_dditable_t             Firmware;
    zes_memory_dditable_t               Memory;
    zes_fabric_port_dditable_t          FabricPort;
    zes_temperature_dditable_t          Temperature;
    zes_psu_dditable_t                  Psu;
    zes_fan_dditable_t                  Fan;
    zes_led_dditable_t                  Led;
    zes_ras_dditable_t                  Ras;
    zes_diagnostics_dditable_t          Diagnostics;
} zes_dditable_t;

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // _ZES_DDI_H